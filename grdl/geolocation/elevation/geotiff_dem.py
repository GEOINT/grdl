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
2026-02-11
"""

# Standard library
from pathlib import Path
from typing import Optional

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.elevation._backend import require_elevation_backend
from grdl.geolocation.elevation.base import ElevationModel


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
                raise ImportError(
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

        # Nearest-neighbor sampling
        col_idx = np.round(px_cols).astype(np.intp)
        row_idx = np.round(px_rows).astype(np.intp)

        # Bounds check
        valid = (
            (row_idx >= 0) & (row_idx < self._nrows)
            & (col_idx >= 0) & (col_idx < self._ncols)
        )

        if not np.any(valid):
            return heights

        valid_rows = row_idx[valid]
        valid_cols = col_idx[valid]

        # Read elevation values from the first band
        data = self._dataset.read(1)
        sampled = data[valid_rows, valid_cols].astype(np.float64)

        # Mask nodata values
        if self._nodata is not None:
            nodata_mask = np.isclose(sampled, float(self._nodata), atol=0.5)
            sampled[nodata_mask] = np.nan

        heights[valid] = sampled
        return heights
