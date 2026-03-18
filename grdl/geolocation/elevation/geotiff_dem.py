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
duane.d.smalley@gmail.com

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
2026-03-10
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
    ) -> None:
        """Initialize GeoTIFF DEM elevation model.

        Parameters
        ----------
        dem_path : str or Path
            Path to the GeoTIFF DEM file.
        geoid_path : str or Path, optional
            Path to geoid model file for MSL-to-HAE correction.

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

        # Bilinear interpolation between the 4 surrounding DEM pixels.
        # Floor to get the top-left pixel of the interpolation cell.
        col_f = np.floor(px_cols).astype(np.intp)
        row_f = np.floor(px_rows).astype(np.intp)

        # Fractional offsets within the cell [0, 1)
        dc = px_cols - col_f
        dr = px_rows - row_f

        # Bounds check (need the +1 neighbor to exist)
        valid = (
            (row_f >= 0) & (row_f < self._nrows - 1)
            & (col_f >= 0) & (col_f < self._ncols - 1)
        )

        if not np.any(valid):
            return heights

        v_row = row_f[valid]
        v_col = col_f[valid]
        v_dr = dr[valid]
        v_dc = dc[valid]

        # Windowed read: bounding box of needed pixels (+1 for interp)
        from rasterio.windows import Window

        r_min, r_max = int(v_row.min()), int(v_row.max()) + 1
        c_min, c_max = int(v_col.min()), int(v_col.max()) + 1
        window = Window(
            col_off=c_min, row_off=r_min,
            width=c_max - c_min + 1, height=r_max - r_min + 1,
        )
        data = self._dataset.read(1, window=window).astype(np.float64)

        # Local pixel indices within the window
        lr = v_row - r_min
        lc = v_col - c_min

        # Four corners of each interpolation cell
        z00 = data[lr, lc]
        z01 = data[lr, lc + 1]
        z10 = data[lr + 1, lc]
        z11 = data[lr + 1, lc + 1]

        # Mask nodata in any corner → NaN for that point
        if self._nodata is not None:
            nd = float(self._nodata)
            nodata_mask = (
                np.isclose(z00, nd, atol=0.5)
                | np.isclose(z01, nd, atol=0.5)
                | np.isclose(z10, nd, atol=0.5)
                | np.isclose(z11, nd, atol=0.5)
            )
            z00[nodata_mask] = np.nan

        # Bilinear interpolation
        sampled = (z00 * (1 - v_dc) * (1 - v_dr)
                   + z01 * v_dc * (1 - v_dr)
                   + z10 * (1 - v_dc) * v_dr
                   + z11 * v_dc * v_dr)

        heights[valid] = sampled
        return heights
