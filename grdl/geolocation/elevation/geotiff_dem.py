# -*- coding: utf-8 -*-
"""
GeoTIFF DEM Elevation Model - Terrain elevation lookup from a single GeoTIFF.

Reads a single GeoTIFF DEM file using rasterio/GDAL. Supports both
geographic (EPSG:4326) and projected coordinate reference systems. When the
DEM uses a projected CRS, lat/lon queries are automatically reprojected
using pyproj before sampling.

Dependencies
------------
rasterio
pyproj

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-11

Modified
--------
2026-03-20
"""

# Standard library
import logging
from pathlib import Path
from typing import Optional

# Third-party
import numpy as np

# GRDL internal
from grdl.exceptions import DependencyError
from grdl.geolocation.elevation._backend import require_elevation_backend
from grdl.geolocation.elevation.base import ElevationModel

logger = logging.getLogger(__name__)


class GeoTIFFDEM(ElevationModel):
    """Elevation model that reads a single GeoTIFF DEM file.

    Opens a GeoTIFF DEM with rasterio and provides vectorized elevation
    queries. If the DEM uses a projected CRS (e.g., UTM), lat/lon inputs
    are reprojected automatically using pyproj.

    The dataset is held open for the lifetime of the object to avoid
    repeated file open/close overhead. Call ``close()`` or use as a
    context manager to release file handles.

    Parameters
    ----------
    dem_path : str or Path
        Path to the GeoTIFF DEM file. Must exist and be a file.
    geoid_path : str or Path, optional
        Path to geoid model file for MSL-to-HAE correction.
    interpolation : int, default=3
        Spline interpolation order for DEM sampling:

        - ``1`` — bilinear (C0, fast, derivative discontinuities at cell edges)
        - ``3`` — bicubic (C1, smooth derivatives, recommended for ortho)
        - ``5`` — quintic (C2, very smooth, slower)

        Higher orders produce smoother height fields, eliminating
        the kinks at DEM cell boundaries that cause visible
        distortion in sub-meter orthorectified imagery.

    Raises
    ------
    FileNotFoundError
        If ``dem_path`` does not exist.
    ImportError
        If rasterio is not installed.

    Examples
    --------
    >>> from grdl.geolocation.elevation.geotiff_dem import GeoTIFFDEM
    >>> with GeoTIFFDEM('/data/srtm_34_04.tif') as elev:
    ...     height = elev.get_elevation(34.05, -118.25)
    ...     print(f"Elevation: {height:.1f} m")
    """

    def __init__(
        self,
        dem_path: str,
        geoid_path: Optional[str] = None,
        interpolation: int = 3,
    ) -> None:
        """Initialize GeoTIFF DEM elevation model.

        Parameters
        ----------
        dem_path : str or Path
            Path to the GeoTIFF DEM file.
        geoid_path : str or Path, optional
            Path to geoid model file for MSL-to-HAE correction.
        interpolation : int, default=3
            Spline interpolation order (1=bilinear, 3=bicubic, 5=quintic).

        Raises
        ------
        FileNotFoundError
            If ``dem_path`` does not exist.
        ImportError
            If rasterio is not installed.
        """
        require_elevation_backend()

        import rasterio

        dem_path = Path(dem_path)
        if not dem_path.exists():
            raise FileNotFoundError(
                f"GeoTIFF DEM file does not exist: {dem_path}"
            )

        super().__init__(dem_path=str(dem_path), geoid_path=geoid_path)

        # Open dataset and extract spatial reference information
        self._dataset = rasterio.open(str(dem_path))
        logger.info("Loaded DEM %s", dem_path.name)
        self._transform = self._dataset.transform
        self._inv_transform = ~self._transform
        self._crs = self._dataset.crs
        self._nrows = self._dataset.height
        self._ncols = self._dataset.width
        self._nodata = self._dataset.nodata
        self._interp_order = interpolation

        # Build CRS transformer if the DEM is not in geographic coordinates
        self._transformer = None
        if self._crs is not None and not self._crs.is_geographic:
            try:
                import pyproj
                self._transformer = pyproj.Transformer.from_crs(
                    'EPSG:4326', self._crs, always_xy=True
                )
            except ImportError:
                raise DependencyError(
                    "GeoTIFFDEM with projected CRS requires pyproj. "
                    "Install with: pip install pyproj"
                )

    def close(self) -> None:
        """Close the underlying rasterio dataset and release file handles."""
        ds = getattr(self, '_dataset', None)
        if ds is not None and not ds.closed:
            ds.close()

    def __enter__(self) -> 'GeoTIFFDEM':
        """Support context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close dataset on context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Close dataset on garbage collection."""
        self.close()

    def _get_elevation_array(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> np.ndarray:
        """Look up terrain elevation from the GeoTIFF DEM.

        Parameters
        ----------
        lats : np.ndarray
            Latitudes in degrees North. Shape ``(N,)``.
        lons : np.ndarray
            Longitudes in degrees East. Shape ``(N,)``.

        Returns
        -------
        np.ndarray
            Elevation values in meters (MSL). Shape ``(N,)``. NaN for
            points outside DEM coverage.
        """
        n = lats.shape[0]
        heights = np.full(n, np.nan, dtype=np.float64)

        # Transform to DEM CRS if needed
        if self._transformer is not None:
            x_coords, y_coords = self._transformer.transform(lons, lats)
        else:
            # Geographic CRS: x=lon, y=lat
            x_coords = lons.copy()
            y_coords = lats.copy()

        # Inverse affine: map coordinates -> pixel coordinates
        px_cols, px_rows = self._inv_transform * (x_coords, y_coords)
        px_cols = np.asarray(px_cols, dtype=np.float64)
        px_rows = np.asarray(px_rows, dtype=np.float64)

        # Margin needed around each query point for interpolation kernel.
        # Bilinear needs +1, bicubic needs +2, quintic needs +3.
        margin = max(1, (self._interp_order + 1) // 2)

        # Bounds check: query points must have enough neighbors
        valid = (
            (px_rows >= margin) & (px_rows < self._nrows - margin)
            & (px_cols >= margin) & (px_cols < self._ncols - margin)
        )

        if not np.any(valid):
            return heights

        v_rows = px_rows[valid]
        v_cols = px_cols[valid]

        # Windowed read: bounding box of needed pixels + margin
        from rasterio.windows import Window

        r_min = int(np.floor(v_rows.min())) - margin
        r_max = int(np.ceil(v_rows.max())) + margin
        c_min = int(np.floor(v_cols.min())) - margin
        c_max = int(np.ceil(v_cols.max())) + margin
        r_min = max(0, r_min)
        c_min = max(0, c_min)
        r_max = min(self._nrows - 1, r_max)
        c_max = min(self._ncols - 1, c_max)

        window = Window(
            col_off=c_min, row_off=r_min,
            width=c_max - c_min + 1, height=r_max - r_min + 1,
        )
        data = self._dataset.read(1, window=window).astype(np.float64)

        # Replace nodata with NaN before interpolation
        if self._nodata is not None:
            nodata_mask = np.isclose(data, float(self._nodata), atol=0.5)
            data[nodata_mask] = np.nan

        # Local coordinates within the window
        local_rows = v_rows - r_min
        local_cols = v_cols - c_min

        if self._interp_order == 1:
            # Fast bilinear path (no scipy dependency for simple case)
            row_f = np.floor(local_rows).astype(np.intp)
            col_f = np.floor(local_cols).astype(np.intp)
            dr = local_rows - row_f
            dc = local_cols - col_f

            z00 = data[row_f, col_f]
            z01 = data[row_f, col_f + 1]
            z10 = data[row_f + 1, col_f]
            z11 = data[row_f + 1, col_f + 1]

            sampled = (z00 * (1 - dc) * (1 - dr)
                       + z01 * dc * (1 - dr)
                       + z10 * (1 - dc) * dr
                       + z11 * dc * dr)
        else:
            # Higher-order spline via scipy.ndimage.map_coordinates.
            # Produces C1 (cubic) or C2 (quintic) continuous surfaces
            # that eliminate derivative discontinuities at DEM cell edges.
            from scipy.ndimage import map_coordinates

            # map_coordinates doesn't handle NaN; fill with nearest
            # valid value before interpolation, then re-mask.
            nan_mask = np.isnan(data)
            if np.any(nan_mask):
                from scipy.ndimage import distance_transform_edt
                _, nearest_idx = distance_transform_edt(
                    nan_mask, return_distances=True, return_indices=True,
                )
                data_filled = data.copy()
                data_filled[nan_mask] = data[
                    nearest_idx[0][nan_mask], nearest_idx[1][nan_mask]
                ]
            else:
                data_filled = data

            coords = np.vstack([local_rows, local_cols])
            sampled = map_coordinates(
                data_filled, coords,
                order=self._interp_order, mode='nearest',
            )

            # Re-apply NaN where any input was nodata (check nearest cells)
            if np.any(nan_mask):
                row_nn = np.clip(
                    np.round(local_rows).astype(np.intp),
                    0, data.shape[0] - 1,
                )
                col_nn = np.clip(
                    np.round(local_cols).astype(np.intp),
                    0, data.shape[1] - 1,
                )
                sampled[nan_mask[row_nn, col_nn]] = np.nan

        heights[valid] = sampled
        return heights
