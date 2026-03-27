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
2026-02-11

Modified
--------
2026-03-27  Add configurable interpolation (bilinear/bicubic/quintic),
            cross-tile boundary stitching, and nodata void handling.
2026-02-11
"""

# Standard library
import logging
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.elevation._backend import require_elevation_backend
from grdl.geolocation.elevation.base import ElevationModel

logger = logging.getLogger(__name__)

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
    ValueError
        If ``dem_path`` is not a directory.
    ImportError
        If rasterio is not installed.

    Notes
    -----
    Adjacent tiles are loaded automatically when interpolation kernels
    extend past tile edges, preventing discontinuities at 1-degree
    boundaries. Loaded tile data is cached for the lifetime of the
    object; call ``clear_cache()`` to release memory.

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
        interpolation: int = 3,
    ) -> None:
        """Initialize DTED elevation model.

        Parameters
        ----------
        dem_path : str or Path
            Root directory containing DTED tiles.
        geoid_path : str or Path, optional
            Path to geoid model file for MSL-to-HAE correction.
        interpolation : int, default=3
            Spline interpolation order (1=bilinear, 3=bicubic, 5=quintic).

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

        self._interp_order = interpolation

        # Build spatial index: (lon_floor, lat_floor) -> file path
        # When multiple resolutions exist for the same tile, prefer
        # higher resolution (dt2 > dt1 > dt0).
        self._tile_index: Dict[Tuple[int, int], Path] = {}
        self._scan_tiles(dem_path)

        # Tile data cache: key -> (data_float64, transform) or None
        self._tile_cache: Dict[
            Tuple[int, int], Optional[Tuple[np.ndarray, object]]
        ] = {}

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

    def clear_cache(self) -> None:
        """Release cached tile data to free memory."""
        self._tile_cache.clear()

    def _read_tile_cached(
        self, key: Tuple[int, int]
    ) -> Optional[Tuple[np.ndarray, object]]:
        """Load a DTED tile with caching.

        Parameters
        ----------
        key : Tuple[int, int]
            ``(lon_floor, lat_floor)`` tile key.

        Returns
        -------
        Tuple[np.ndarray, object] or None
            ``(data, transform)`` where data is float64 with nodata
            replaced by NaN, or None if the tile is unavailable.
        """
        if key in self._tile_cache:
            return self._tile_cache[key]

        if key not in self._tile_index:
            self._tile_cache[key] = None
            return None

        import rasterio

        try:
            with rasterio.open(str(self._tile_index[key])) as ds:
                data = ds.read(1).astype(np.float64)
                transform = ds.transform
                nodata = ds.nodata

            # Replace nodata with NaN so interpolation kernels skip voids
            if nodata is not None:
                nodata_mask = np.isclose(data, float(nodata), atol=0.5)
                data[nodata_mask] = np.nan

            result = (data, transform)
            self._tile_cache[key] = result
            return result

        except (rasterio.errors.RasterioIOError, OSError):
            self._tile_cache[key] = None
            return None

    def _build_padded_tile(
        self,
        key: Tuple[int, int],
        data: np.ndarray,
        margin: int,
    ) -> np.ndarray:
        """Build padded tile array with data from adjacent tiles.

        DTED tiles share boundary posts (edge-inclusive grids). This
        method loads adjacent tiles and extracts the margin posts just
        beyond each shared boundary to create a seamless interpolation
        neighborhood.

        Parameters
        ----------
        key : Tuple[int, int]
            ``(lon_floor, lat_floor)`` of the primary tile.
        data : np.ndarray
            Primary tile data. Shape ``(nrows, ncols)``.
        margin : int
            Number of posts to pad on each side.

        Returns
        -------
        np.ndarray
            Padded array of shape ``(nrows + 2*margin, ncols + 2*margin)``.
            Regions without adjacent tile data remain NaN.
        """
        nrows, ncols = data.shape
        padded = np.full(
            (nrows + 2 * margin, ncols + 2 * margin),
            np.nan, dtype=np.float64,
        )
        padded[margin:margin + nrows, margin:margin + ncols] = data

        lon_key, lat_key = key

        # Cardinal neighbors.
        # DTED loaded via rasterio/GDAL uses north-to-south row ordering:
        #   row 0 = northern boundary (lat + 1)
        #   row nrows-1 = southern boundary (lat)
        # Boundary posts are shared between adjacent tiles, so we skip
        # the shared row/column (index 0 or nrows-1) and take the
        # `margin` posts just beyond it.
        cardinal = [
            # North neighbor: its southern rows just above our row 0
            ((lon_key, lat_key + 1),
             (slice(0, margin), slice(margin, margin + ncols)),
             lambda nd: nd[-margin - 1:-1, :ncols]),
            # South neighbor: its northern rows just below our last row
            ((lon_key, lat_key - 1),
             (slice(margin + nrows, margin + nrows + margin),
              slice(margin, margin + ncols)),
             lambda nd: nd[1:1 + margin, :ncols]),
            # West neighbor: its eastern columns just left of our col 0
            ((lon_key - 1, lat_key),
             (slice(margin, margin + nrows), slice(0, margin)),
             lambda nd: nd[:nrows, -margin - 1:-1]),
            # East neighbor: its western columns just right of our last col
            ((lon_key + 1, lat_key),
             (slice(margin, margin + nrows),
              slice(margin + ncols, margin + ncols + margin)),
             lambda nd: nd[:nrows, 1:1 + margin]),
        ]

        for nkey, (pad_row_sl, pad_col_sl), extract_fn in cardinal:
            ntile = self._read_tile_cached(nkey)
            if ntile is None:
                continue
            ndata = ntile[0]
            try:
                patch = extract_fn(ndata)
                target_shape = padded[pad_row_sl, pad_col_sl].shape
                if patch.shape == target_shape:
                    padded[pad_row_sl, pad_col_sl] = patch
            except (IndexError, ValueError):
                continue

        # Corner neighbors (NE, NW, SE, SW)
        corners = [
            ((lon_key + 1, lat_key + 1),
             (slice(0, margin), slice(margin + ncols, margin + ncols + margin)),
             lambda nd: nd[-margin - 1:-1, 1:1 + margin]),
            ((lon_key - 1, lat_key + 1),
             (slice(0, margin), slice(0, margin)),
             lambda nd: nd[-margin - 1:-1, -margin - 1:-1]),
            ((lon_key + 1, lat_key - 1),
             (slice(margin + nrows, margin + nrows + margin),
              slice(margin + ncols, margin + ncols + margin)),
             lambda nd: nd[1:1 + margin, 1:1 + margin]),
            ((lon_key - 1, lat_key - 1),
             (slice(margin + nrows, margin + nrows + margin),
              slice(0, margin)),
             lambda nd: nd[1:1 + margin, -margin - 1:-1]),
        ]

        for ckey, (pad_row_sl, pad_col_sl), extract_fn in corners:
            ctile = self._read_tile_cached(ckey)
            if ctile is None:
                continue
            cdata = ctile[0]
            try:
                patch = extract_fn(cdata)
                target_shape = padded[pad_row_sl, pad_col_sl].shape
                if patch.shape == target_shape:
                    padded[pad_row_sl, pad_col_sl] = patch
            except (IndexError, ValueError):
                continue

        return padded

    def _get_elevation_array(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> np.ndarray:
        """Look up terrain elevation from DTED tiles.

        Points are grouped by tile key for efficient batch reads.
        When interpolation kernels extend past tile edges, adjacent
        tiles are loaded and stitched to prevent boundary
        discontinuities. Points outside tile coverage receive NaN.

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
        n = lats.shape[0]
        heights = np.full(n, np.nan, dtype=np.float64)
        margin = max(1, (self._interp_order + 1) // 2)

        # Compute tile keys for all points
        lat_floors = np.floor(lats).astype(np.int64)
        lon_floors = np.floor(lons).astype(np.int64)

        # Group point indices by tile key
        tile_groups: Dict[Tuple[int, int], list] = {}
        missing_keys: set = set()
        for i in range(n):
            key = (int(lon_floors[i]), int(lat_floors[i]))
            if key in self._tile_index:
                tile_groups.setdefault(key, []).append(i)
            else:
                missing_keys.add(key)
        for mk in missing_keys:
            logger.warning("No DTED tile for (%d, %d)", mk[0], mk[1])

        # Process each tile batch
        for key, indices in tile_groups.items():
            idx_arr = np.array(indices, dtype=np.intp)
            batch_lats = lats[idx_arr]
            batch_lons = lons[idx_arr]

            tile = self._read_tile_cached(key)
            if tile is None:
                continue
            data, transform = tile
            nrows, ncols = data.shape

            # Fractional pixel coordinates via inverse affine
            inv_transform = ~transform
            px_cols, px_rows = inv_transform * (batch_lons, batch_lats)
            px_cols = np.asarray(px_cols, dtype=np.float64)
            px_rows = np.asarray(px_rows, dtype=np.float64)

            # Pad with adjacent tiles if any point needs cross-tile data
            needs_padding = (
                np.any(px_rows < margin)
                or np.any(px_rows >= nrows - margin)
                or np.any(px_cols < margin)
                or np.any(px_cols >= ncols - margin)
            )
            if needs_padding:
                data = self._build_padded_tile(key, data, margin)
                px_rows = px_rows + margin
                px_cols = px_cols + margin
                nrows, ncols = data.shape

            # Bounds check with margin for interpolation kernel
            valid = (
                (px_rows >= margin) & (px_rows < nrows - margin)
                & (px_cols >= margin) & (px_cols < ncols - margin)
            )

            if not np.any(valid):
                continue

            v_rows = px_rows[valid]
            v_cols = px_cols[valid]

            if self._interp_order == 1:
                # Bilinear interpolation (no scipy dependency)
                row_f = np.floor(v_rows).astype(np.intp)
                col_f = np.floor(v_cols).astype(np.intp)
                dr = v_rows - row_f
                dc = v_cols - col_f

                z00 = data[row_f, col_f]
                z01 = data[row_f, col_f + 1]
                z10 = data[row_f + 1, col_f]
                z11 = data[row_f + 1, col_f + 1]

                sampled = (
                    z00 * (1.0 - dc) * (1.0 - dr)
                    + z01 * dc * (1.0 - dr)
                    + z10 * (1.0 - dc) * dr
                    + z11 * dc * dr
                )
            else:
                # Higher-order spline via scipy.ndimage.map_coordinates.
                # Produces C1 (cubic) or C2 (quintic) continuous surfaces
                # that eliminate derivative discontinuities at DEM cell
                # edges.
                from scipy.ndimage import map_coordinates

                # map_coordinates doesn't handle NaN; fill voids with
                # nearest valid value before interpolation, then re-mask.
                nan_mask = np.isnan(data)
                if np.any(nan_mask):
                    from scipy.ndimage import distance_transform_edt
                    _, nearest_idx = distance_transform_edt(
                        nan_mask,
                        return_distances=True,
                        return_indices=True,
                    )
                    data_filled = data.copy()
                    data_filled[nan_mask] = data[
                        nearest_idx[0][nan_mask],
                        nearest_idx[1][nan_mask],
                    ]
                else:
                    data_filled = data

                coords = np.vstack([v_rows, v_cols])
                sampled = map_coordinates(
                    data_filled, coords,
                    order=self._interp_order, mode='nearest',
                )

                # Re-apply NaN where the nearest DEM cell was void
                if np.any(nan_mask):
                    row_nn = np.clip(
                        np.round(v_rows).astype(np.intp),
                        0, data.shape[0] - 1,
                    )
                    col_nn = np.clip(
                        np.round(v_cols).astype(np.intp),
                        0, data.shape[1] - 1,
                    )
                    sampled[nan_mask[row_nn, col_nn]] = np.nan

            heights[idx_arr[valid]] = sampled

        return heights
