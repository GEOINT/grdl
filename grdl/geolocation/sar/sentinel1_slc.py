# -*- coding: utf-8 -*-
"""
Sentinel-1 SLC Geolocation - Coordinate transforms via annotation grid.

Provides pixel-to-geographic and geographic-to-pixel transforms for
Sentinel-1 IW SLC imagery using the annotation geolocation grid.  The
grid is regular in image coordinates (line, pixel), so forward transforms
use ``RectBivariateSpline`` for smooth, fast interpolation.  The inverse
uses ``LinearNDInterpolator`` (Delaunay triangulation) since the grid in
geographic space is warped by SAR geometry.

Dependencies
------------
scipy

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
2026-02-16

Modified
--------
2026-02-16
"""

# Standard library
from typing import List, Optional, Tuple, Union, Any, TYPE_CHECKING

# Third-party
import numpy as np

try:
    from scipy.interpolate import RectBivariateSpline, LinearNDInterpolator
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# GRDL internal
from grdl.geolocation.base import Geolocation

if TYPE_CHECKING:
    from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader
    from grdl.IO.models.sentinel1_slc import (
        Sentinel1SLCMetadata, S1SLCGeoGridPoint,
    )


class Sentinel1SLCGeolocation(Geolocation):
    """Geolocation for Sentinel-1 IW SLC using annotation grid interpolation.

    The Sentinel-1 annotation XML provides a regular rectangular grid of
    tie points mapping ``(line, pixel)`` to ``(latitude, longitude, height)``.
    This class builds smooth 2D interpolators over that grid for both
    forward and inverse coordinate transforms.

    Parameters
    ----------
    metadata : Sentinel1SLCMetadata
        Typed metadata from a ``Sentinel1SLCReader``.  Must contain a
        populated ``geolocation_grid`` with at least 4 points.
    dem_path : str or Path, optional
        Path to DEM data for terrain-corrected heights.
    geoid_path : str or Path, optional
        Path to geoid correction file.

    Raises
    ------
    ImportError
        If scipy is not installed.
    ValueError
        If the geolocation grid is missing or has fewer than 4 points.

    Examples
    --------
    >>> from grdl.IO.sar import Sentinel1SLCReader
    >>> from grdl.geolocation import Sentinel1SLCGeolocation
    >>> with Sentinel1SLCReader('product.SAFE', swath='IW1') as reader:
    ...     geo = Sentinel1SLCGeolocation.from_reader(reader)
    ...     lat, lon, h = geo.image_to_latlon(5000, 10000)
    ...     row, col = geo.latlon_to_image(lat, lon)
    """

    def __init__(
        self,
        metadata: 'Sentinel1SLCMetadata',
        dem_path: Optional[Union[str, Any]] = None,
        geoid_path: Optional[Union[str, Any]] = None,
    ) -> None:
        if not _HAS_SCIPY:
            raise ImportError(
                "Sentinel-1 SLC geolocation requires scipy. "
                "Install with: conda install -c conda-forge scipy"
            )

        grid = metadata.geolocation_grid
        if grid is None or len(grid) < 4:
            raise ValueError(
                "Sentinel1SLCMetadata.geolocation_grid must contain "
                "at least 4 points for interpolation."
            )

        self._build_interpolators(grid)

        shape = (metadata.rows, metadata.cols)
        super().__init__(shape, crs='WGS84', dem_path=dem_path,
                         geoid_path=geoid_path)

    def _build_interpolators(
        self,
        grid: 'List[S1SLCGeoGridPoint]',
    ) -> None:
        """Build forward and inverse interpolators from the grid.

        Parameters
        ----------
        grid : List[S1SLCGeoGridPoint]
            Geolocation tie points from the annotation XML.
        """
        # Extract unique sorted line and pixel values
        line_set = sorted(set(pt.line for pt in grid))
        pixel_set = sorted(set(pt.pixel for pt in grid))
        n_lines = len(line_set)
        n_pixels = len(pixel_set)

        line_vals = np.array(line_set, dtype=np.float64)
        pixel_vals = np.array(pixel_set, dtype=np.float64)

        # Build lookup for grid point by (line, pixel)
        grid_lookup = {(pt.line, pt.pixel): pt for pt in grid}

        # Reshape into 2D grids: lat_grid[i, j], lon_grid[i, j]
        lat_grid = np.empty((n_lines, n_pixels), dtype=np.float64)
        lon_grid = np.empty((n_lines, n_pixels), dtype=np.float64)
        height_grid = np.empty((n_lines, n_pixels), dtype=np.float64)

        for i, line in enumerate(line_set):
            for j, pixel in enumerate(pixel_set):
                pt = grid_lookup.get((line, pixel))
                if pt is None:
                    lat_grid[i, j] = np.nan
                    lon_grid[i, j] = np.nan
                    height_grid[i, j] = np.nan
                else:
                    lat_grid[i, j] = pt.latitude
                    lon_grid[i, j] = pt.longitude
                    height_grid[i, j] = pt.height

        # Forward: (row, col) → (lat, lon, height)
        # RectBivariateSpline for regular grids (kx=ky=1 for bilinear)
        self._lat_spline = RectBivariateSpline(
            line_vals, pixel_vals, lat_grid, kx=1, ky=1,
        )
        self._lon_spline = RectBivariateSpline(
            line_vals, pixel_vals, lon_grid, kx=1, ky=1,
        )
        self._height_spline = RectBivariateSpline(
            line_vals, pixel_vals, height_grid, kx=1, ky=1,
        )

        # Inverse: (lat, lon) → (row, col)
        # Flatten grids for LinearNDInterpolator
        flat_lats = lat_grid.ravel()
        flat_lons = lon_grid.ravel()
        flat_rows = np.repeat(line_vals, n_pixels)
        flat_cols = np.tile(pixel_vals, n_lines)

        # Filter out NaN points
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
        height: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform pixel arrays to geographic arrays.

        Parameters
        ----------
        rows : np.ndarray
            Row (line) coordinates, 1D float64.
        cols : np.ndarray
            Column (pixel) coordinates, 1D float64.
        height : float
            Not used (heights come from the grid).

        Returns
        -------
        lats, lons, heights : np.ndarray
            Geographic coordinates in WGS84.
        """
        lats = self._lat_spline.ev(rows, cols)
        lons = self._lon_spline.ev(rows, cols)
        heights = self._height_spline.ev(rows, cols)
        return lats, lons, heights

    def _latlon_to_image_array(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform geographic arrays to pixel arrays.

        Parameters
        ----------
        lats : np.ndarray
            Latitudes in degrees, 1D float64.
        lons : np.ndarray
            Longitudes in degrees, 1D float64.
        height : float
            Not used.

        Returns
        -------
        rows, cols : np.ndarray
            Pixel coordinates. NaN for points outside the grid convex hull.
        """
        query = np.column_stack([lats, lons])
        rows = self._row_interp(query)
        cols = self._col_interp(query)
        return rows, cols

    @classmethod
    def from_reader(
        cls,
        reader: 'Sentinel1SLCReader',
        dem_path: Optional[Union[str, Any]] = None,
        geoid_path: Optional[Union[str, Any]] = None,
    ) -> 'Sentinel1SLCGeolocation':
        """Create geolocation from a Sentinel-1 SLC reader.

        Parameters
        ----------
        reader : Sentinel1SLCReader
            Open reader with metadata loaded.
        dem_path : str or Path, optional
            Path to DEM data.
        geoid_path : str or Path, optional
            Path to geoid correction file.

        Returns
        -------
        Sentinel1SLCGeolocation
            Configured geolocation instance.
        """
        return cls(reader.metadata, dem_path=dem_path,
                   geoid_path=geoid_path)
