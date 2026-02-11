# -*- coding: utf-8 -*-
"""
DTED Elevation Model - Terrain elevation lookup from DTED tiles.

Reads Digital Terrain Elevation Data (DTED) tiles organized in the standard
directory structure using rasterio/GDAL. Supports DTED Level 0, 1, and 2
(.dt0, .dt1, .dt2). Tiles are indexed by floor(lat)/floor(lon) and opened
on demand for efficient batch queries.

DTED directory structure::

    dted_root/
        e116/
            n34.dt2
            n35.dt2
        w074/
            n40.dt1
            n41.dt1

Dependencies
------------
rasterio

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
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.elevation._backend import require_elevation_backend
from grdl.geolocation.elevation.base import ElevationModel

# DTED file extensions in order of preference (highest resolution first)
_DTED_EXTENSIONS = ('.dt2', '.dt1', '.dt0')


def _parse_dted_tile_key(filepath: Path) -> Optional[Tuple[int, int]]:
    """Extract (lon_floor, lat_floor) from a DTED file path.

    DTED naming convention:
    - Directory: ``e116`` or ``w074`` (hemisphere + 3-digit longitude)
    - File: ``n34.dt2`` or ``s12.dt1`` (hemisphere + 2-digit latitude)

    Parameters
    ----------
    filepath : Path
        Path to a DTED tile file.

    Returns
    -------
    Tuple[int, int] or None
        ``(lon_floor, lat_floor)`` integer degrees, or None if the path
        does not match DTED naming conventions.
    """
    try:
        lat_stem = filepath.stem.lower()
        lon_dir = filepath.parent.name.lower()

        # Parse latitude from filename
        if lat_stem.startswith('n'):
            lat = int(lat_stem[1:])
        elif lat_stem.startswith('s'):
            lat = -int(lat_stem[1:])
        else:
            return None

        # Parse longitude from directory name
        if lon_dir.startswith('e'):
            lon = int(lon_dir[1:])
        elif lon_dir.startswith('w'):
            lon = -int(lon_dir[1:])
        else:
            return None

        return (lon, lat)
    except (ValueError, IndexError):
        return None


class DTEDElevation(ElevationModel):
    """Elevation model that reads DTED tiles using rasterio/GDAL.

    Scans a directory tree for DTED tiles (.dt0, .dt1, .dt2) at
    construction time and builds a spatial index mapping integer
    (lon, lat) tile keys to file paths. Queries are grouped by tile
    for efficient batch reads.

    Parameters
    ----------
    dem_path : str or Path
        Root directory containing DTED tiles. Must exist and be a
        directory. Tiles are discovered recursively.
    geoid_path : str or Path, optional
        Path to geoid model file for MSL-to-HAE correction.

    Raises
    ------
    FileNotFoundError
        If ``dem_path`` does not exist.
    ValueError
        If ``dem_path`` is not a directory.
    ImportError
        If rasterio is not installed.

    Examples
    --------
    >>> from grdl.geolocation.elevation.dted import DTEDElevation
    >>> elev = DTEDElevation('/data/dted')
    >>> elev.get_elevation(34.05, -118.25)
    432.0
    >>> import numpy as np
    >>> lats = np.array([34.0, 34.5, 35.0])
    >>> lons = np.array([-118.0, -117.5, -117.0])
    >>> elev.get_elevation(lats, lons)
    array([...])
    """

    def __init__(
        self,
        dem_path: str,
        geoid_path: Optional[str] = None,
    ) -> None:
        """Initialize DTED elevation model.

        Parameters
        ----------
        dem_path : str or Path
            Root directory containing DTED tiles.
        geoid_path : str or Path, optional
            Path to geoid model file for MSL-to-HAE correction.

        Raises
        ------
        FileNotFoundError
            If ``dem_path`` does not exist.
        ValueError
            If ``dem_path`` is not a directory.
        ImportError
            If rasterio is not installed.
        """
        require_elevation_backend()

        dem_path = Path(dem_path)
        if not dem_path.exists():
            raise FileNotFoundError(
                f"DTED directory does not exist: {dem_path}"
            )
        if not dem_path.is_dir():
            raise ValueError(
                f"DTED dem_path must be a directory, got file: {dem_path}"
            )

        super().__init__(dem_path=str(dem_path), geoid_path=geoid_path)

        # Build spatial index: (lon_floor, lat_floor) -> file path
        # When multiple resolutions exist for the same tile, prefer
        # higher resolution (dt2 > dt1 > dt0).
        self._tile_index: Dict[Tuple[int, int], Path] = {}
        self._scan_tiles(dem_path)

    def _scan_tiles(self, root: Path) -> None:
        """Recursively scan directory for DTED tiles and build index.

        Parameters
        ----------
        root : Path
            Root directory to scan.
        """
        # Collect all DTED files
        tiles_by_key: Dict[Tuple[int, int], Dict[str, Path]] = {}

        for ext in _DTED_EXTENSIONS:
            for filepath in root.rglob(f'*{ext}'):
                key = _parse_dted_tile_key(filepath)
                if key is not None:
                    if key not in tiles_by_key:
                        tiles_by_key[key] = {}
                    tiles_by_key[key][ext] = filepath

        # Select highest resolution for each tile
        for key, ext_map in tiles_by_key.items():
            for ext in _DTED_EXTENSIONS:
                if ext in ext_map:
                    self._tile_index[key] = ext_map[ext]
                    break

    @property
    def tile_count(self) -> int:
        """Number of DTED tiles indexed.

        Returns
        -------
        int
            Count of discovered tiles.
        """
        return len(self._tile_index)

    @property
    def coverage_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Geographic bounding box of all indexed tiles.

        Returns
        -------
        Tuple[float, float, float, float] or None
            ``(min_lon, min_lat, max_lon, max_lat)`` in degrees, or None
            if no tiles are indexed.
        """
        if not self._tile_index:
            return None
        lons = [k[0] for k in self._tile_index]
        lats = [k[1] for k in self._tile_index]
        return (
            float(min(lons)),
            float(min(lats)),
            float(max(lons)) + 1.0,
            float(max(lats)) + 1.0,
        )

    def _get_elevation_array(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> np.ndarray:
        """Look up terrain elevation from DTED tiles.

        Points are grouped by tile key for efficient batch reads.
        Points outside tile coverage receive NaN.

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
            points outside DTED coverage.
        """
        import rasterio

        n = lats.shape[0]
        heights = np.full(n, np.nan, dtype=np.float64)

        # Compute tile keys for all points
        lat_floors = np.floor(lats).astype(np.int64)
        lon_floors = np.floor(lons).astype(np.int64)

        # Group point indices by tile key
        tile_groups: Dict[Tuple[int, int], list] = {}
        for i in range(n):
            key = (int(lon_floors[i]), int(lat_floors[i]))
            if key in self._tile_index:
                if key not in tile_groups:
                    tile_groups[key] = []
                tile_groups[key].append(i)

        # Process each tile batch
        for key, indices in tile_groups.items():
            tile_path = self._tile_index[key]
            idx_arr = np.array(indices, dtype=np.intp)
            batch_lats = lats[idx_arr]
            batch_lons = lons[idx_arr]

            try:
                with rasterio.open(str(tile_path)) as ds:
                    transform = ds.transform
                    data = ds.read(1)
                    nodata = ds.nodata

                    # Inverse affine: geographic -> pixel coordinates
                    inv_transform = ~transform
                    px_cols, px_rows = inv_transform * (batch_lons, batch_lats)
                    px_cols = np.asarray(px_cols, dtype=np.float64)
                    px_rows = np.asarray(px_rows, dtype=np.float64)

                    # Nearest-neighbor sampling
                    col_idx = np.round(px_cols).astype(np.intp)
                    row_idx = np.round(px_rows).astype(np.intp)

                    # Bounds check
                    nrows, ncols = data.shape
                    valid = (
                        (row_idx >= 0) & (row_idx < nrows)
                        & (col_idx >= 0) & (col_idx < ncols)
                    )

                    valid_rows = row_idx[valid]
                    valid_cols = col_idx[valid]
                    sampled = data[valid_rows, valid_cols].astype(np.float64)

                    # Mask nodata values
                    if nodata is not None:
                        nodata_mask = np.isclose(
                            sampled, float(nodata), atol=0.5
                        )
                        sampled[nodata_mask] = np.nan

                    # Write valid results
                    valid_indices = idx_arr[valid]
                    heights[valid_indices] = sampled

            except (rasterio.errors.RasterioIOError, OSError):
                # Tile is unreadable; leave NaN for these points
                continue

        return heights
