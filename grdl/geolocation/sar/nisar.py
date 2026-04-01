# -*- coding: utf-8 -*-
"""
NISAR Geolocation - Coordinate transforms for NISAR RSLC and GSLC products.

Provides pixel-to-geographic and geographic-to-pixel transforms for NISAR
imagery.

- **RSLC** products use a 3-D geolocation grid sampled at multiple heights.
  We select the middle height layer and build ``RectBivariateSpline``
  interpolators (same approach as ``Sentinel1SLCGeolocation``).

- **GSLC** products are already geocoded and described by a regular
  coordinate grid with an EPSG code, so we delegate to
  ``AffineGeolocation``.

Dependencies
------------
scipy (for RSLC)
rasterio, pyproj (for GSLC, via ``AffineGeolocation``)

Author
------
Duane Smalley, PhD
duane.d.smalley@gmail.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-25

Modified
--------
2026-03-10
"""

# Standard library
from typing import Optional, Tuple, Union, Any, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.exceptions import DependencyError

try:
    from scipy.interpolate import RectBivariateSpline, LinearNDInterpolator
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# GRDL internal
from grdl.geolocation.base import Geolocation

if TYPE_CHECKING:
    from grdl.IO.sar.nisar import NISARReader
    from grdl.IO.models.nisar import NISARMetadata


class NISARGeolocation(Geolocation):
    """Geolocation for NISAR RSLC products using annotation grid interpolation.

    The NISAR RSLC geolocation grid is 3-D: ``(heights, azimuth, range)``
    mapping pixel coordinates to geographic coordinates at multiple terrain
    heights.  This class selects the middle height layer and builds smooth
    2-D interpolators for forward and inverse transforms.

    For GSLC products, use ``NISARGeolocation.from_reader()`` which returns
    an ``AffineGeolocation`` instead.

    Parameters
    ----------
    metadata : NISARMetadata
        Typed NISAR metadata with a populated ``geolocation_grid`` and
        ``swath_parameters``.
    dem_path : str or Path, optional
        Path to DEM data for terrain-corrected heights.
    geoid_path : str or Path, optional
        Path to geoid correction file.

    Raises
    ------
    ImportError
        If scipy is not installed.
    ValueError
        If the geolocation grid or swath parameters are missing.
    """

    def __init__(
        self,
        metadata: 'NISARMetadata',
        dem_path: Optional[Union[str, Any]] = None,
        geoid_path: Optional[Union[str, Any]] = None,
        interpolation: int = 3,
    ) -> None:
        if not _HAS_SCIPY:
            raise DependencyError(
                "NISAR RSLC geolocation requires scipy. "
                "Install with: conda install -c conda-forge scipy"
            )

        geo_grid = metadata.geolocation_grid
        sp = metadata.swath_parameters
        if geo_grid is None:
            raise ValueError(
                "NISARMetadata.geolocation_grid is required for "
                "RSLC geolocation."
            )
        if (geo_grid.coordinate_x is None or geo_grid.coordinate_y is None
                or geo_grid.zero_doppler_time is None
                or geo_grid.slant_range is None):
            raise ValueError(
                "NISAR geolocation grid missing required arrays "
                "(coordinate_x, coordinate_y, zero_doppler_time, "
                "slant_range)."
            )

        self._build_interpolators(metadata)
        shape = (metadata.rows, metadata.cols)
        super().__init__(shape, crs='WGS84', dem_path=dem_path,
                         geoid_path=geoid_path,
                         interpolation=interpolation)

    def _build_interpolators(self, metadata: 'NISARMetadata') -> None:
        """Build forward and inverse interpolators from the geolocation grid.

        Selects the middle height layer from the 3-D grid and builds
        bilinear splines in (row, col) space.
        """
        geo_grid = metadata.geolocation_grid
        sp = metadata.swath_parameters

        # Select middle height layer
        coord_x = geo_grid.coordinate_x  # (heights, azimuth, range)
        coord_y = geo_grid.coordinate_y
        if coord_x.ndim == 3:
            mid_h = coord_x.shape[0] // 2
            lon_grid = coord_x[mid_h]  # (azimuth, range)
            lat_grid = coord_y[mid_h]
        elif coord_x.ndim == 2:
            lon_grid = coord_x
            lat_grid = coord_y
        else:
            raise ValueError(
                f"Unexpected coordinate_x ndim={coord_x.ndim}, expected 2 or 3."
            )

        # Grid axes in image coordinates
        # The geolocation grid is sampled at specific azimuth times and
        # slant ranges.  Convert to row/col using swath parameters.
        grid_az_time = geo_grid.zero_doppler_time  # (n_az,)
        grid_slant_range = geo_grid.slant_range     # (n_rg,)

        # Swath parameters define the full image pixel grid
        if (sp is not None
                and sp.zero_doppler_time is not None
                and sp.slant_range is not None):
            # Map grid azimuth times to row indices by interpolation
            img_az_time = sp.zero_doppler_time  # (rows,)
            img_slant_range = sp.slant_range     # (cols,)

            # Convert grid sample positions to row/col by linear interp
            row_vals = np.interp(grid_az_time, img_az_time,
                                 np.arange(len(img_az_time)))
            col_vals = np.interp(grid_slant_range, img_slant_range,
                                 np.arange(len(img_slant_range)))
        elif (sp is not None
              and sp.zero_doppler_time_spacing is not None
              and sp.slant_range_spacing is not None):
            # Fallback: uniform spacing from origin
            dt = sp.zero_doppler_time_spacing
            dr = sp.slant_range_spacing
            t0 = grid_az_time[0]
            r0 = grid_slant_range[0]
            row_vals = (grid_az_time - t0) / dt if dt != 0 else grid_az_time
            col_vals = (grid_slant_range - r0) / dr if dr != 0 else grid_slant_range
        else:
            # Last resort: assume grid axes map directly to rows/cols
            row_vals = np.linspace(0, metadata.rows - 1, len(grid_az_time))
            col_vals = np.linspace(0, metadata.cols - 1, len(grid_slant_range))

        # Forward: (row, col) → (lat, lon)
        self._lat_spline = RectBivariateSpline(
            row_vals, col_vals, lat_grid, kx=1, ky=1,
        )
        self._lon_spline = RectBivariateSpline(
            row_vals, col_vals, lon_grid, kx=1, ky=1,
        )

        # Inverse: (lat, lon) → (row, col)
        n_az, n_rg = lat_grid.shape
        flat_lats = lat_grid.ravel()
        flat_lons = lon_grid.ravel()
        flat_rows = np.repeat(row_vals, n_rg)
        flat_cols = np.tile(col_vals, n_az)

        valid = ~(np.isnan(flat_lats) | np.isnan(flat_lons))
        geo_points = np.column_stack([flat_lats[valid], flat_lons[valid]])
        self._row_interp = LinearNDInterpolator(
            geo_points, flat_rows[valid], fill_value=np.nan,
        )
        self._col_interp = LinearNDInterpolator(
            geo_points, flat_cols[valid], fill_value=np.nan,
        )

    def _image_to_latlon_array(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform pixel arrays to geographic arrays."""
        lats = self._lat_spline.ev(rows, cols)
        lons = self._lon_spline.ev(rows, cols)
        if np.ndim(height) > 0:
            heights = np.asarray(height, dtype=np.float64)
        else:
            heights = np.full_like(lats, float(height))
        return lats, lons, heights

    def _latlon_to_image_array(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform geographic arrays to pixel arrays."""
        query = np.column_stack([lats, lons])
        rows = self._row_interp(query)
        cols = self._col_interp(query)
        return rows, cols

    @classmethod
    def from_reader(
        cls,
        reader: 'NISARReader',
        dem_path: Optional[Union[str, Any]] = None,
        geoid_path: Optional[Union[str, Any]] = None,
    ) -> 'Geolocation':
        """Create geolocation from a NISAR reader.

        For RSLC products, returns a ``NISARGeolocation`` using grid
        interpolation.  For GSLC products, returns an ``AffineGeolocation``
        since the data is already geocoded.

        Parameters
        ----------
        reader : NISARReader
            Open reader with metadata loaded.
        dem_path : str or Path, optional
            Path to DEM data.
        geoid_path : str or Path, optional
            Path to geoid correction file.

        Returns
        -------
        Geolocation
            ``NISARGeolocation`` for RSLC, ``AffineGeolocation`` for GSLC.
        """
        meta = reader.metadata
        if meta.product_type == 'GSLC':
            return cls._build_gslc_geolocation(meta, dem_path, geoid_path)
        return cls(meta, dem_path=dem_path, geoid_path=geoid_path)

    @staticmethod
    def _build_gslc_geolocation(
        metadata: 'NISARMetadata',
        dem_path: Optional[Union[str, Any]],
        geoid_path: Optional[Union[str, Any]],
    ) -> 'Geolocation':
        """Build an AffineGeolocation for a GSLC product."""
        from rasterio.transform import Affine
        from grdl.geolocation.eo.affine import AffineGeolocation

        gp = metadata.grid_parameters
        if gp is None:
            raise ValueError(
                "NISAR GSLC metadata missing grid_parameters."
            )
        if (gp.x_coordinates is None or gp.y_coordinates is None
                or gp.epsg is None):
            raise ValueError(
                "NISAR GSLC grid_parameters missing x_coordinates, "
                "y_coordinates, or epsg."
            )

        dx = gp.x_coordinate_spacing or (
            gp.x_coordinates[1] - gp.x_coordinates[0]
            if len(gp.x_coordinates) > 1 else 1.0
        )
        dy = gp.y_coordinate_spacing or (
            gp.y_coordinates[1] - gp.y_coordinates[0]
            if len(gp.y_coordinates) > 1 else -1.0
        )

        transform = Affine(
            dx, 0.0, float(gp.x_coordinates[0]),
            0.0, dy, float(gp.y_coordinates[0]),
        )
        crs = f'EPSG:{gp.epsg}'
        shape = (metadata.rows, metadata.cols)
        return AffineGeolocation(
            transform, shape, crs,
            dem_path=dem_path, geoid_path=geoid_path,
        )
