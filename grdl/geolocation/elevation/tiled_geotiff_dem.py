# -*- coding: utf-8 -*-
"""
Tiled GeoTIFF DEM - Seamless terrain elevation from multiple GeoTIFF tiles.

Indexes a directory of GeoTIFF DEM tiles (FABDEM, SRTM, Copernicus, etc.)
and presents them as a single elevation model with seamless cross-tile
interpolation.  Tiles are opened on demand with an LRU cache and queried
via windowed reads for memory efficiency.

Tile naming conventions automatically parsed (no rasterio I/O at scan time):

- FABDEM:     ``N43E004_FABDEM_V1-2.tif``
- SRTM:       ``N43E004.tif`` or ``N43E004.hgt``
- Copernicus: ``Copernicus_DSM_10_N43_00_E004_00.tif``
- Generic:    any ``N/SxxE/Wyyy*.tif`` prefix

Files that do not match any naming convention are opened with rasterio
to extract bounds.  Projected-CRS tiles are skipped with a warning.

Cross-tile boundary interpolation is handled transparently: when a
query point's interpolation kernel (bilinear 1px, bicubic 2px, quintic
3px margin) extends beyond the primary tile, the overflow pixels are
read from adjacent tiles.

Dependencies
------------
rasterio
scipy (for bicubic / quintic interpolation)

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
2026-03-21

Modified
--------
2026-03-21
"""

# Standard library
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.elevation._backend import require_elevation_backend
from grdl.geolocation.elevation.base import ElevationModel

logger = logging.getLogger(__name__)


class _TileMeta(NamedTuple):
    """Lightweight metadata for a single DEM tile."""
    path: Path
    transform: object       # rasterio.Affine
    inv_transform: object   # ~transform
    nrows: int
    ncols: int
    nodata: Optional[float]


# Regex for common tile naming: N43E004, S27W060, etc.
_TILE_RE = re.compile(r'([NS])(\d{1,2})([EW])(\d{1,3})')


def _parse_tile_key_from_name(filepath: Path) -> Optional[Tuple[int, int]]:
    """Extract (lon_floor, lat_floor) from tile filename.

    Parameters
    ----------
    filepath : Path
        Path to a GeoTIFF tile.

    Returns
    -------
    Tuple[int, int] or None
        ``(lon_floor, lat_floor)`` in integer degrees, or None if the
        filename does not match any known convention.
    """
    stem = filepath.stem.upper()
    m = _TILE_RE.search(stem)
    if m:
        lat = int(m.group(2)) * (-1 if m.group(1) == 'S' else 1)
        lon = int(m.group(4)) * (-1 if m.group(3) == 'W' else 1)
        return (lon, lat)
    return None


class TiledGeoTIFFDEM(ElevationModel):
    """Seamless elevation model across multiple GeoTIFF DEM tiles.

    Scans a directory tree for GeoTIFF tiles, builds a spatial index
    mapping integer ``(lon_floor, lat_floor)`` keys to file paths, and
    provides vectorized elevation queries that transparently span tile
    boundaries.

    Interpolation kernels (bilinear, bicubic, quintic) that extend
    beyond a tile edge are filled from adjacent tiles when available.
    Missing adjacent tiles produce NaN-filled regions that are handled
    by nearest-neighbor extrapolation before interpolation.

    Parameters
    ----------
    dem_path : str or Path
        Root directory containing GeoTIFF DEM tiles.  Searched
        recursively for ``.tif`` and ``.tiff`` files.
    geoid_path : str or Path, optional
        Path to geoid model file for MSL-to-HAE correction.
    interpolation : int, default=3
        Spline interpolation order:

        - ``1`` -- bilinear (C0, fast)
        - ``3`` -- bicubic (C1, smooth, recommended)
        - ``5`` -- quintic (C2, very smooth, slower)
    max_open_tiles : int, default=8
        Maximum number of tile file handles to keep open
        simultaneously (LRU eviction).

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
    >>> from grdl.geolocation.elevation.tiled_geotiff_dem import TiledGeoTIFFDEM
    >>> elev = TiledGeoTIFFDEM('/data/FABDEM/')
    >>> elev.tile_count
    4
    >>> elev.get_elevation(43.78, 4.98)
    127.3
    >>> import numpy as np
    >>> lats = np.array([43.5, 44.1])
    >>> lons = np.array([4.8, 5.2])
    >>> elev.get_elevation(lats, lons)
    array([85.2, 312.7])
    """

    def __init__(
        self,
        dem_path: str,
        geoid_path: Optional[str] = None,
        interpolation: int = 3,
        max_open_tiles: int = 8,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        require_elevation_backend()

        dem_path = Path(dem_path)
        if not dem_path.exists():
            raise FileNotFoundError(
                f"DEM directory does not exist: {dem_path}"
            )
        if not dem_path.is_dir():
            raise ValueError(
                f"dem_path must be a directory, got file: {dem_path}"
            )

        super().__init__(dem_path=str(dem_path), geoid_path=geoid_path)

        self._interp_order = interpolation
        self._margin = max(1, (interpolation + 1) // 2)
        self._max_open = max_open_tiles

        # Spatial index: (lon_floor, lat_floor) -> _TileMeta
        self._tile_index: Dict[Tuple[int, int], _TileMeta] = {}

        # LRU cache of open rasterio datasets
        self._open_tiles: OrderedDict = OrderedDict()

        # Optional bbox filter: only index tiles whose
        # (lon_floor, lat_floor) cell intersects the
        # scene footprint. Massive speed-up + memory
        # saving when the DEM tree has thousands of tiles
        # but the scene only sits on 1-4 of them.
        self._bbox = bbox

        self._scan_tiles(dem_path)
        if bbox is not None:
            logger.info(
                "TiledGeoTIFFDEM: indexed %d tile(s) "
                "intersecting bbox %s from %s",
                len(self._tile_index), bbox, dem_path,
            )
        else:
            logger.info(
                "TiledGeoTIFFDEM: indexed %d tiles from %s",
                len(self._tile_index), dem_path,
            )

    def _scan_tiles(self, root: Path) -> None:
        """Recursively scan directory for GeoTIFF tiles and build index.

        Two-pass approach for speed: first parse tile keys from filenames
        (no I/O), then fall back to rasterio for unmatched files.

        Parameters
        ----------
        root : Path
            Root directory to scan.
        """
        import rasterio

        tif_files: List[Path] = []
        for ext in ('*.tif', '*.tiff'):
            tif_files.extend(root.rglob(ext))

        # Pass 1: parse keys from filenames. The bbox
        # filter is applied here — when set, tiles whose
        # (lon_floor, lat_floor) cell doesn't intersect
        # the scene footprint are dropped without ever
        # being opened, saving thousands of GDAL handle
        # opens against a large FABDEM/Copernicus tree.
        unmatched: List[Path] = []
        for fp in tif_files:
            key = _parse_tile_key_from_name(fp)
            if key is not None:
                if not self._key_intersects_bbox(key):
                    continue
                self._index_tile_from_name(key, fp)
            else:
                unmatched.append(fp)

        # Pass 2: open unmatched files with rasterio to read bounds
        for fp in unmatched:
            try:
                with rasterio.open(str(fp)) as ds:
                    if ds.crs is not None and not ds.crs.is_geographic:
                        logger.warning(
                            "Skipping projected-CRS tile: %s", fp.name)
                        continue
                    bounds = ds.bounds
                    lon_floor = int(np.floor(bounds.left))
                    lat_floor = int(np.floor(bounds.bottom))
                    key = (lon_floor, lat_floor)
                    # Honour the bbox filter for the
                    # bounds-discovered branch as well.
                    if not self._key_intersects_bbox(key):
                        continue
                    if key not in self._tile_index:
                        meta = _TileMeta(
                            path=fp,
                            transform=ds.transform,
                            inv_transform=~ds.transform,
                            nrows=ds.height,
                            ncols=ds.width,
                            nodata=ds.nodata,
                        )
                        self._tile_index[key] = meta
            except Exception as e:
                logger.debug("Failed to probe %s: %s", fp.name, e)

    def _key_intersects_bbox(
        self, key: Tuple[int, int],
    ) -> bool:
        """True if the 1°×1° cell at *key* intersects bbox.

        Returns ``True`` when no bbox is configured. The
        bbox layout matches
        :meth:`grdl.geolocation.base.Geolocation.get_footprint`'s
        ``bounds`` field — ``(min_lon, min_lat, max_lon,
        max_lat)``. The cell at *key* occupies
        ``[lon_floor, lon_floor+1) × [lat_floor, lat_floor+1)``;
        we keep it whenever the bbox spans any portion of
        it, with a single-cell halo so cross-tile spline
        kernels at the scene edge still find a neighbour.
        """
        if self._bbox is None:
            return True
        min_lon, min_lat, max_lon, max_lat = self._bbox
        lon_floor, lat_floor = key
        return (
            lon_floor + 1 >= min_lon - 1
            and lon_floor <= max_lon + 1
            and lat_floor + 1 >= min_lat - 1
            and lat_floor <= max_lat + 1
        )

    def _index_tile_from_name(
        self, key: Tuple[int, int], filepath: Path
    ) -> None:
        """Index a tile whose key was parsed from its filename.

        Opens the file briefly to read transform and dimensions.

        Parameters
        ----------
        key : Tuple[int, int]
            ``(lon_floor, lat_floor)`` tile key.
        filepath : Path
            Path to the GeoTIFF file.
        """
        if key in self._tile_index:
            return  # already indexed

        import rasterio
        try:
            with rasterio.open(str(filepath)) as ds:
                if ds.crs is not None and not ds.crs.is_geographic:
                    logger.warning(
                        "Skipping projected-CRS tile: %s", filepath.name)
                    return
                meta = _TileMeta(
                    path=filepath,
                    transform=ds.transform,
                    inv_transform=~ds.transform,
                    nrows=ds.height,
                    ncols=ds.width,
                    nodata=ds.nodata,
                )
                self._tile_index[key] = meta
        except Exception as e:
            logger.debug("Failed to index %s: %s", filepath.name, e)

    @property
    def tile_count(self) -> int:
        """Number of indexed tiles.

        Returns
        -------
        int
        """
        return len(self._tile_index)

    @property
    def coverage_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Geographic bounding box of all indexed tiles.

        Returns
        -------
        Tuple[float, float, float, float] or None
            ``(min_lon, min_lat, max_lon, max_lat)`` in degrees, or
            None if no tiles are indexed.
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

    def _open_tile(self, key: Tuple[int, int]) -> object:
        """Open a tile dataset with LRU caching.

        Parameters
        ----------
        key : Tuple[int, int]
            Tile key.

        Returns
        -------
        rasterio.DatasetReader
            Open dataset.

        Raises
        ------
        KeyError
            If the tile key is not in the index.
        """
        import rasterio

        if key in self._open_tiles:
            # Move to end (most recently used)
            self._open_tiles.move_to_end(key)
            return self._open_tiles[key]

        meta = self._tile_index[key]
        ds = rasterio.open(str(meta.path))
        self._open_tiles[key] = ds

        # Evict oldest if over limit
        while len(self._open_tiles) > self._max_open:
            evict_key, evict_ds = self._open_tiles.popitem(last=False)
            try:
                evict_ds.close()
            except Exception:
                pass

        return ds

    def _read_tile_window(
        self,
        key: Tuple[int, int],
        r_min: int,
        r_max: int,
        c_min: int,
        c_max: int,
    ) -> Optional[np.ndarray]:
        """Read a window from a single tile, clamped to valid bounds.

        Parameters
        ----------
        key : Tuple[int, int]
            Tile key.
        r_min, r_max : int
            Row range (inclusive).
        c_min, c_max : int
            Column range (inclusive).

        Returns
        -------
        np.ndarray or None
            2D array of elevation values, or None if the tile is
            unavailable or the window is entirely out of bounds.
        """
        if key not in self._tile_index:
            return None

        meta = self._tile_index[key]
        nrows, ncols = meta.nrows, meta.ncols

        # Clamp to valid tile bounds
        cr_min = max(0, r_min)
        cr_max = min(nrows - 1, r_max)
        cc_min = max(0, c_min)
        cc_max = min(ncols - 1, c_max)

        if cr_min > cr_max or cc_min > cc_max:
            return None

        from rasterio.windows import Window
        try:
            ds = self._open_tile(key)
            window = Window(
                col_off=cc_min, row_off=cr_min,
                width=cc_max - cc_min + 1,
                height=cr_max - cr_min + 1,
            )
            data = ds.read(1, window=window).astype(np.float64)

            # Replace nodata with NaN
            if meta.nodata is not None:
                nodata_mask = np.isclose(data, float(meta.nodata), atol=0.5)
                data[nodata_mask] = np.nan

            return data
        except Exception:
            return None

    def _read_padded_window(
        self,
        key: Tuple[int, int],
        r_min: int,
        r_max: int,
        c_min: int,
        c_max: int,
    ) -> np.ndarray:
        """Read a window that may span tile boundaries.

        Assembles data from the primary tile and up to 8 adjacent tiles
        (N, S, E, W, NE, NW, SE, SW) to fill the requested window.

        Row direction: row 0 is the north edge (north-up convention),
        so ``r_min < 0`` extends into the tile to the **north**
        (lat_floor + 1), and ``r_max >= nrows`` extends **south**
        (lat_floor - 1).

        Parameters
        ----------
        key : Tuple[int, int]
            Primary tile key ``(lon_floor, lat_floor)``.
        r_min, r_max : int
            Row range in the primary tile's pixel coordinate system
            (may be negative or exceed tile dimensions).
        c_min, c_max : int
            Column range (may be negative or exceed tile dimensions).

        Returns
        -------
        np.ndarray
            2D array of shape ``(r_max - r_min + 1, c_max - c_min + 1)``.
            NaN where no data is available.
        """
        meta = self._tile_index[key]
        nrows, ncols = meta.nrows, meta.ncols
        out_h = r_max - r_min + 1
        out_w = c_max - c_min + 1
        result = np.full((out_h, out_w), np.nan, dtype=np.float64)

        lon0, lat0 = key

        # Determine which of the 9 regions (primary + 8 neighbors) are needed.
        # Each region is defined by its tile key and the row/col mapping.
        #
        # Tile coordinate mapping (north-up):
        #   primary row r -> northern adj row (nrows_adj + r)  when r < 0
        #   primary row r -> southern adj row (r - nrows)      when r >= nrows
        #   primary col c -> western adj col  (ncols_adj + c)  when c < 0
        #   primary col c -> eastern adj col  (c - ncols)      when c >= ncols

        # Define the 9 potential regions
        regions = []

        # row/col splits
        r_splits = [(r_min, min(r_max, nrows - 1))]
        if r_min < 0:
            r_splits.insert(0, (r_min, -1))
            r_splits[1] = (0, r_splits[1][1])
        if r_max >= nrows:
            if r_splits[-1][1] >= nrows:
                r_splits[-1] = (r_splits[-1][0], nrows - 1)
            r_splits.append((nrows, r_max))

        c_splits = [(c_min, min(c_max, ncols - 1))]
        if c_min < 0:
            c_splits.insert(0, (c_min, -1))
            c_splits[1] = (0, c_splits[1][1])
        if c_max >= ncols:
            if c_splits[-1][1] >= ncols:
                c_splits[-1] = (c_splits[-1][0], ncols - 1)
            c_splits.append((ncols, c_max))

        for rs_min, rs_max in r_splits:
            for cs_min, cs_max in c_splits:
                if rs_min > rs_max or cs_min > cs_max:
                    continue

                # Determine which tile this sub-region maps to
                adj_lon, adj_lat = lon0, lat0
                adj_rs_min, adj_rs_max = rs_min, rs_max
                adj_cs_min, adj_cs_max = cs_min, cs_max

                if rs_min < 0:
                    # North tile (row 0 = north edge, so negative rows = north)
                    adj_lat = lat0 + 1
                    adj_key_check = (adj_lon, adj_lat)
                    if adj_key_check in self._tile_index:
                        adj_nrows = self._tile_index[adj_key_check].nrows
                    else:
                        adj_nrows = nrows  # assume same size
                    adj_rs_min = adj_nrows + rs_min
                    adj_rs_max = adj_nrows + rs_max
                elif rs_min >= nrows:
                    # South tile
                    adj_lat = lat0 - 1
                    adj_rs_min = rs_min - nrows
                    adj_rs_max = rs_max - nrows

                if cs_min < 0:
                    # West tile
                    adj_lon = lon0 - 1
                    adj_key_check = (adj_lon, adj_lat)
                    if adj_key_check in self._tile_index:
                        adj_ncols = self._tile_index[adj_key_check].ncols
                    else:
                        adj_ncols = ncols
                    adj_cs_min = adj_ncols + cs_min
                    adj_cs_max = adj_ncols + cs_max
                elif cs_min >= ncols:
                    # East tile
                    adj_lon = lon0 + 1
                    adj_cs_min = cs_min - ncols
                    adj_cs_max = cs_max - ncols

                adj_key = (adj_lon, adj_lat)

                data = self._read_tile_window(
                    adj_key, adj_rs_min, adj_rs_max, adj_cs_min, adj_cs_max)
                if data is not None:
                    # Place into result at the correct position
                    out_r0 = rs_min - r_min
                    out_r1 = rs_max - r_min + 1
                    out_c0 = cs_min - c_min
                    out_c1 = cs_max - c_min + 1
                    # Data shape may be smaller if clamped at adj tile edge
                    dr = min(data.shape[0], out_r1 - out_r0)
                    dc = min(data.shape[1], out_c1 - out_c0)
                    result[out_r0:out_r0 + dr, out_c0:out_c0 + dc] = (
                        data[:dr, :dc])

        return result

    def _get_elevation_array(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> np.ndarray:
        """Look up terrain elevation across multiple GeoTIFF tiles.

        Points are grouped by tile key.  Each group is queried with a
        single padded windowed read that handles cross-tile boundaries.

        Parameters
        ----------
        lats : np.ndarray
            Latitudes in degrees North. Shape ``(N,)``.
        lons : np.ndarray
            Longitudes in degrees East. Shape ``(N,)``.

        Returns
        -------
        np.ndarray
            Elevation values in meters. Shape ``(N,)``. NaN for points
            outside tile coverage.
        """
        n = lats.shape[0]
        heights = np.full(n, np.nan, dtype=np.float64)

        # Compute tile keys
        lat_floors = np.floor(lats).astype(np.int64)
        lon_floors = np.floor(lons).astype(np.int64)

        # Group point indices by tile key
        tile_groups: Dict[Tuple[int, int], List[int]] = {}
        for i in range(n):
            key = (int(lon_floors[i]), int(lat_floors[i]))
            if key in self._tile_index:
                if key not in tile_groups:
                    tile_groups[key] = []
                tile_groups[key].append(i)

        margin = self._margin

        # Process each tile group
        for key, indices in tile_groups.items():
            meta = self._tile_index[key]
            idx_arr = np.array(indices, dtype=np.intp)
            batch_lats = lats[idx_arr]
            batch_lons = lons[idx_arr]

            # Compute pixel coordinates in this tile
            px_cols, px_rows = meta.inv_transform * (batch_lons, batch_lats)
            px_cols = np.asarray(px_cols, dtype=np.float64)
            px_rows = np.asarray(px_rows, dtype=np.float64)

            # Bounding window with margin for interpolation kernel
            r_min = int(np.floor(px_rows.min())) - margin
            r_max = int(np.ceil(px_rows.max())) + margin
            c_min = int(np.floor(px_cols.min())) - margin
            c_max = int(np.ceil(px_cols.max())) + margin

            # Read padded window (handles cross-tile boundaries)
            data = self._read_padded_window(key, r_min, r_max, c_min, c_max)

            # Local pixel coordinates within the window
            local_rows = px_rows - r_min
            local_cols = px_cols - c_min

            # Bounds check: points must be within the window
            valid = (
                (local_rows >= 0)
                & (local_rows < data.shape[0])
                & (local_cols >= 0)
                & (local_cols < data.shape[1])
            )

            if not np.any(valid):
                continue

            v_rows = local_rows[valid]
            v_cols = local_cols[valid]

            if self._interp_order == 1:
                # Bilinear fast path
                row_f = np.floor(v_rows).astype(np.intp)
                col_f = np.floor(v_cols).astype(np.intp)
                dr = v_rows - row_f
                dc = v_cols - col_f

                # Clamp to avoid out-of-bounds
                row_f1 = np.clip(row_f + 1, 0, data.shape[0] - 1)
                col_f1 = np.clip(col_f + 1, 0, data.shape[1] - 1)
                row_f = np.clip(row_f, 0, data.shape[0] - 1)
                col_f = np.clip(col_f, 0, data.shape[1] - 1)

                z00 = data[row_f, col_f]
                z01 = data[row_f, col_f1]
                z10 = data[row_f1, col_f]
                z11 = data[row_f1, col_f1]

                sampled = (z00 * (1 - dc) * (1 - dr)
                           + z01 * dc * (1 - dr)
                           + z10 * (1 - dc) * dr
                           + z11 * dc * dr)
            else:
                # Higher-order spline via scipy.ndimage.map_coordinates
                from scipy.ndimage import map_coordinates

                # Fill NaN with nearest valid value before interpolation
                nan_mask = np.isnan(data)
                if np.any(nan_mask):
                    from scipy.ndimage import distance_transform_edt
                    _, nearest_idx = distance_transform_edt(
                        nan_mask, return_distances=True,
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

                # Re-apply NaN where input was nodata
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

            valid_indices = idx_arr[valid]
            heights[valid_indices] = sampled

        return heights

    def close(self) -> None:
        """Close all open tile datasets and release file handles."""
        for ds in self._open_tiles.values():
            try:
                ds.close()
            except Exception:
                pass
        self._open_tiles.clear()

    def __enter__(self) -> 'TiledGeoTIFFDEM':
        """Support context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close all tile datasets on context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Close all tile datasets on garbage collection."""
        self.close()
