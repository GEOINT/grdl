# -*- coding: utf-8 -*-
"""
TiledGeoDTED - Bbox-driven DTED elevation model with LRU file handles.

Indexes a DTED archive without recursive ``rglob`` and queries tiles via
windowed rasterio reads.  Built to replace :class:`DTEDElevation` on the
hot path of multi-worker SAR pipelines where the legacy class's
behaviors are pain points:

- A six-pattern ``rglob`` across the whole archive at construction
  (``.dt0/.dt1/.dt2`` × upper/lower case) becomes the dominant cost
  when workers each construct the model.
- The whole-tile ``_tile_cache`` carries float64 raster bytes into
  every pickle payload sent to worker processes.
- The bbox is known at the call site but cannot be passed through.

This class fixes all three:

- **Discovery is bbox-driven.**  Given a scene bbox, enumerate the
  integer ``(lon_floor, lat_floor)`` cells it overlaps (plus a 1-cell
  halo for cross-tile interpolation) and probe each candidate DTED
  path directly.  ``Path.exists()`` ×  ~12-per-cell is cheap.  No
  ``rglob`` is ever invoked when a bbox is supplied.
- **The cache stores file metadata, not raster bytes.**  Tiles open
  once at index time to capture transform/dims/nodata, then close.
  Pickle payload is just paths + a few floats per tile.
- **An LRU of open rasterio datasets** keeps file handles warm without
  unbounded growth, mirroring :class:`TiledGeoTIFFDEM`.
- **Cross-tile interpolation** uses padded windowed reads, identical
  in spirit to :class:`TiledGeoTIFFDEM`.

The bilinear sampler hot path uses
:func:`grdl.geolocation.elevation._numba_dted.dted_bilinear_fast` when
numba is available; bicubic / quintic stay on ``scipy.ndimage.map_coordinates``
because spline filtering is already hand-tuned C.

Dependencies
------------
rasterio
scipy (for bicubic / quintic interpolation)

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-05-26

Modified
--------
2026-06-08  Share DTED discovery with DTEDElevation: nested archive
            layouts and OS-independent, cached case-insensitive probing
            for both the bbox and no-bbox paths.
2026-05-26
"""

# Standard library
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.elevation._backend import require_elevation_backend
from grdl.geolocation.elevation.base import ElevationModel
from grdl.geolocation.elevation.dted import (
    _CaseInsensitiveDirCache,
    _discover_dted_tiles,
    _find_dted_file,
)

logger = logging.getLogger(__name__)


class _DTEDTileMeta(NamedTuple):
    """Lightweight metadata for a single DTED tile."""
    path: Path
    transform: object       # rasterio.Affine
    inv_transform: object   # ~transform
    nrows: int
    ncols: int
    nodata: Optional[float]


def _hemi_lon(lon: int) -> str:
    """Hemisphere prefix for a longitude floor."""
    return 'w' if lon < 0 else 'e'


def _hemi_lat(lat: int) -> str:
    """Hemisphere prefix for a latitude floor."""
    return 's' if lat < 0 else 'n'


class TiledGeoDTED(ElevationModel):
    """Seamless elevation model across DTED tiles, bbox-aware.

    Parameters
    ----------
    dem_path : str or Path
        Root directory containing DTED tiles laid out as
        ``<root>/{e,w}<lon3>/{n,s}<lat2>.dt{0,1,2}``.  Both lower- and
        upper-case spellings are accepted.
    geoid_path : str or Path, optional
        Geoid model for MSL → HAE correction.
    interpolation : int, default=3
        Spline order for sampling.  ``1`` = bilinear (numba-accelerated
        when available), ``3`` = bicubic, ``5`` = quintic.
    max_open_tiles : int, default=8
        LRU cap on open rasterio datasets.
    bbox : tuple of (min_lon, min_lat, max_lon, max_lat), optional
        Scene bounding box.  When supplied, only tile cells intersecting
        the bbox (plus a 1-cell halo) are indexed.  This is the fast
        path: no directory walk happens at all.

    Notes
    -----
    The bbox halo is intentional: cross-tile interpolation needs the
    neighbouring tile when the kernel footprint of an edge query
    crosses a 1° boundary.

    Examples
    --------
    >>> elev = TiledGeoDTED('/data/dted', bbox=(116.0, 34.0, 117.5, 35.5))
    >>> elev.tile_count
    4
    >>> elev.get_elevation(34.5, 116.5)
    432.0
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
                f"DTED directory does not exist: {dem_path}"
            )
        if not dem_path.is_dir():
            raise ValueError(
                f"dem_path must be a directory, got file: {dem_path}"
            )

        super().__init__(dem_path=str(dem_path), geoid_path=geoid_path)

        self._interp_order = interpolation
        self._margin = max(1, (interpolation + 1) // 2)
        self._max_open = max_open_tiles
        self._bbox = bbox

        # path-only index; rasterio is opened lazily through the LRU
        self._tile_index: Dict[Tuple[int, int], _DTEDTileMeta] = {}
        self._open_tiles: "OrderedDict[Tuple[int, int], object]" = OrderedDict()

        if bbox is not None:
            self._index_by_bbox(dem_path, bbox)
            logger.info(
                "TiledGeoDTED: indexed %d tile(s) intersecting bbox %s from %s",
                len(self._tile_index), bbox, dem_path,
            )
        else:
            self._index_by_scan(dem_path)
            logger.info(
                "TiledGeoDTED: indexed %d tile(s) from %s "
                "(no bbox; single-level scan)",
                len(self._tile_index), dem_path,
            )

    # ── Indexing ─────────────────────────────────────────────────────

    def _index_by_bbox(
        self,
        root: Path,
        bbox: Tuple[float, float, float, float],
    ) -> None:
        """Index DTED cells overlapping ``bbox`` (with 1-cell halo).

        Discovery is shared with :class:`DTEDElevation`: a cached,
        case-insensitive resolver (:func:`_find_dted_file`) probes both
        the standard layout and nested archive layouts
        (``<root>/dted/dted{2,1,0}/<lon>/<lat>.dt?``) in resolution order
        (``.dt2 → .dt1 → .dt0``), first hit wins. No ``rglob``; each
        directory is listed at most once across the whole sweep, so the
        probe is fast and OS-independent.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        lon_lo = int(np.floor(min_lon)) - 1
        lon_hi = int(np.floor(max_lon)) + 1
        lat_lo = int(np.floor(min_lat)) - 1
        lat_hi = int(np.floor(max_lat)) + 1
        dir_cache = _CaseInsensitiveDirCache()
        for lon in range(lon_lo, lon_hi + 1):
            for lat in range(lat_lo, lat_hi + 1):
                found = _find_dted_file(root, lon, lat, dir_cache)
                if found is not None:
                    self._index_tile(lon, lat, found)

    def _index_by_scan(self, root: Path) -> None:
        """Bounded scan of both the standard and nested archive layouts.

        Shares :func:`_discover_dted_tiles` with :class:`DTEDElevation`'s
        non-bbox path: a one-level, OS-independent scan of
        ``<root>/<lon>/<lat>.dt?`` and
        ``<root>/dted/dted{0,1,2}/<lon>/<lat>.dt?``, highest resolution
        per cell. No recursive descent into deeper subdirectories; this
        is the documented slow path for callers that do not pass a bbox.
        """
        for (lon, lat), fpath in _discover_dted_tiles(root).items():
            self._index_tile(lon, lat, fpath)

    def _index_tile(self, lon: int, lat: int, filepath: Path) -> None:
        """Open a tile once to capture transform + dims, then close."""
        key = (int(lon), int(lat))
        if key in self._tile_index:
            return
        import rasterio
        try:
            with rasterio.open(str(filepath)) as ds:
                meta = _DTEDTileMeta(
                    path=filepath,
                    transform=ds.transform,
                    inv_transform=~ds.transform,
                    nrows=ds.height,
                    ncols=ds.width,
                    nodata=ds.nodata,
                )
            self._tile_index[key] = meta
        except Exception as exc:  # pragma: no cover - depends on disk state
            logger.debug("Failed to index %s: %s", filepath, exc)

    # ── Read-side helpers ────────────────────────────────────────────

    @property
    def tile_count(self) -> int:
        """Number of indexed DTED tiles."""
        return len(self._tile_index)

    @property
    def coverage_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Geographic bounding box of all indexed tiles."""
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
        """LRU-cached rasterio dataset for a tile key."""
        import rasterio

        if key in self._open_tiles:
            self._open_tiles.move_to_end(key)
            return self._open_tiles[key]
        meta = self._tile_index[key]
        ds = rasterio.open(str(meta.path))
        self._open_tiles[key] = ds
        while len(self._open_tiles) > self._max_open:
            _evict_key, evict_ds = self._open_tiles.popitem(last=False)
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
        """Read a window from one tile, clamped to its valid extent."""
        if key not in self._tile_index:
            return None
        meta = self._tile_index[key]
        nrows, ncols = meta.nrows, meta.ncols
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
            if meta.nodata is not None:
                data[np.isclose(data, float(meta.nodata), atol=0.5)] = np.nan
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
        """Read a window that may extend across 1° tile boundaries.

        Identical region-decomposition to
        :meth:`TiledGeoTIFFDEM._read_padded_window`.  DTED is also
        north-up with ``row 0`` at the northern edge, so the
        ``r < 0 → north``, ``r >= nrows → south`` convention applies.
        """
        meta = self._tile_index[key]
        nrows, ncols = meta.nrows, meta.ncols
        out_h = r_max - r_min + 1
        out_w = c_max - c_min + 1
        result = np.full((out_h, out_w), np.nan, dtype=np.float64)
        lon0, lat0 = key

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
                adj_lon, adj_lat = lon0, lat0
                adj_rs_min, adj_rs_max = rs_min, rs_max
                adj_cs_min, adj_cs_max = cs_min, cs_max

                if rs_min < 0:
                    adj_lat = lat0 + 1
                    adj_key_check = (adj_lon, adj_lat)
                    adj_nrows = (
                        self._tile_index[adj_key_check].nrows
                        if adj_key_check in self._tile_index else nrows
                    )
                    adj_rs_min = adj_nrows + rs_min
                    adj_rs_max = adj_nrows + rs_max
                elif rs_min >= nrows:
                    adj_lat = lat0 - 1
                    adj_rs_min = rs_min - nrows
                    adj_rs_max = rs_max - nrows

                if cs_min < 0:
                    adj_lon = lon0 - 1
                    adj_key_check = (adj_lon, adj_lat)
                    adj_ncols = (
                        self._tile_index[adj_key_check].ncols
                        if adj_key_check in self._tile_index else ncols
                    )
                    adj_cs_min = adj_ncols + cs_min
                    adj_cs_max = adj_ncols + cs_max
                elif cs_min >= ncols:
                    adj_lon = lon0 + 1
                    adj_cs_min = cs_min - ncols
                    adj_cs_max = cs_max - ncols

                adj_key = (adj_lon, adj_lat)
                data = self._read_tile_window(
                    adj_key, adj_rs_min, adj_rs_max, adj_cs_min, adj_cs_max,
                )
                if data is None:
                    continue
                out_r0 = rs_min - r_min
                out_r1 = rs_max - r_min + 1
                out_c0 = cs_min - c_min
                out_c1 = cs_max - c_min + 1
                dr = min(data.shape[0], out_r1 - out_r0)
                dc = min(data.shape[1], out_c1 - out_c0)
                result[out_r0:out_r0 + dr, out_c0:out_c0 + dc] = data[:dr, :dc]

        return result

    # ── Public sampling ──────────────────────────────────────────────

    def _get_elevation_array(
        self, lats: np.ndarray, lons: np.ndarray,
    ) -> np.ndarray:
        """Vectorised DTED sample.

        Groups points by tile key, reads one padded window per tile,
        interpolates with bilinear (numba), bicubic, or quintic.
        Points outside every indexed tile return NaN.

        The grouping uses ``np.lexsort`` on
        ``(lon_floor, lat_floor)`` rather than a Python dict-of-lists so
        the per-point partition is O(n log n) in vectorised NumPy rather
        than O(n) in interpreted Python.  For attention_graber tile
        chunks of 256-512 points × 12 workers × 10 R/Rdot iterations the
        savings stack up.
        """
        n = lats.shape[0]
        heights = np.full(n, np.nan, dtype=np.float64)
        if n == 0:
            return heights

        lat_floors = np.floor(lats).astype(np.int64)
        lon_floors = np.floor(lons).astype(np.int64)

        order = np.lexsort((lat_floors, lon_floors))
        sorted_lon = lon_floors[order]
        sorted_lat = lat_floors[order]

        # Group boundaries: where the tile key changes.
        change = np.empty(n, dtype=bool)
        change[0] = True
        change[1:] = (sorted_lon[1:] != sorted_lon[:-1]) | (
            sorted_lat[1:] != sorted_lat[:-1]
        )
        starts = np.flatnonzero(change)
        ends = np.concatenate([starts[1:], [n]])

        for s, e in zip(starts, ends):
            key = (int(sorted_lon[s]), int(sorted_lat[s]))
            if key not in self._tile_index:
                continue
            idx = order[s:e]
            self._sample_tile_group(key, idx, lats, lons, heights)

        return heights

    def _sample_tile_group(
        self,
        key: Tuple[int, int],
        idx: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        heights_out: np.ndarray,
    ) -> None:
        """Sample all points whose tile key is ``key``."""
        meta = self._tile_index[key]
        batch_lats = lats[idx]
        batch_lons = lons[idx]

        px_cols, px_rows = meta.inv_transform * (batch_lons, batch_lats)
        px_cols = np.asarray(px_cols, dtype=np.float64)
        px_rows = np.asarray(px_rows, dtype=np.float64)

        margin = self._margin
        r_min = int(np.floor(px_rows.min())) - margin
        r_max = int(np.ceil(px_rows.max())) + margin
        c_min = int(np.floor(px_cols.min())) - margin
        c_max = int(np.ceil(px_cols.max())) + margin

        data = self._read_padded_window(key, r_min, r_max, c_min, c_max)
        local_rows = px_rows - r_min
        local_cols = px_cols - c_min

        valid = (
            (local_rows >= 0)
            & (local_rows < data.shape[0])
            & (local_cols >= 0)
            & (local_cols < data.shape[1])
        )
        if not np.any(valid):
            return
        v_rows = local_rows[valid]
        v_cols = local_cols[valid]

        if self._interp_order == 1:
            sampled = self._bilinear_sample(data, v_rows, v_cols)
        else:
            sampled = self._spline_sample(data, v_rows, v_cols)

        heights_out[idx[valid]] = sampled

    def _bilinear_sample(
        self,
        data: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> np.ndarray:
        """Numba-accelerated bilinear sample with numpy fallback."""
        try:
            from grdl.geolocation.elevation._numba_dted import (
                dted_bilinear_fast,
            )
            fast = dted_bilinear_fast(data, rows, cols)
            if fast is not None:
                return fast
        except ImportError:
            pass

        row_f = np.floor(rows).astype(np.intp)
        col_f = np.floor(cols).astype(np.intp)
        dr = rows - row_f
        dc = cols - col_f
        row_f1 = np.clip(row_f + 1, 0, data.shape[0] - 1)
        col_f1 = np.clip(col_f + 1, 0, data.shape[1] - 1)
        row_f = np.clip(row_f, 0, data.shape[0] - 1)
        col_f = np.clip(col_f, 0, data.shape[1] - 1)
        z00 = data[row_f, col_f]
        z01 = data[row_f, col_f1]
        z10 = data[row_f1, col_f]
        z11 = data[row_f1, col_f1]
        return (
            z00 * (1.0 - dc) * (1.0 - dr)
            + z01 * dc * (1.0 - dr)
            + z10 * (1.0 - dc) * dr
            + z11 * dc * dr
        )

    def _spline_sample(
        self,
        data: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> np.ndarray:
        """Bicubic / quintic via ``scipy.ndimage.map_coordinates``.

        NaN voids are nearest-neighbour-filled before the spline call
        so the kernel does not propagate the void over a 4- or 6-pixel
        footprint, then re-masked afterwards.
        """
        from scipy.ndimage import map_coordinates

        nan_mask = np.isnan(data)
        if np.any(nan_mask):
            from scipy.ndimage import distance_transform_edt
            _, nearest_idx = distance_transform_edt(
                nan_mask, return_distances=True, return_indices=True,
            )
            data_filled = data.copy()
            data_filled[nan_mask] = data[
                nearest_idx[0][nan_mask],
                nearest_idx[1][nan_mask],
            ]
        else:
            data_filled = data

        coords = np.vstack([rows, cols])
        sampled = map_coordinates(
            data_filled, coords, order=self._interp_order, mode='nearest',
        )
        if np.any(nan_mask):
            row_nn = np.clip(
                np.round(rows).astype(np.intp), 0, data.shape[0] - 1,
            )
            col_nn = np.clip(
                np.round(cols).astype(np.intp), 0, data.shape[1] - 1,
            )
            sampled[nan_mask[row_nn, col_nn]] = np.nan
        return sampled

    # ── Pickle / lifecycle ───────────────────────────────────────────

    def __getstate__(self) -> dict:
        """Drop open file handles before pickling.

        ``rasterio.DatasetReader`` wraps a GDAL handle that cannot
        survive ``pickle``.  Workers reopen tiles on demand from the
        path-only index.
        """
        state = self.__dict__.copy()
        cache = state.get("_open_tiles")
        if cache is not None:
            for ds in list(cache.values()):
                try:
                    ds.close()
                except Exception:
                    pass
            state["_open_tiles"] = OrderedDict()
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if not isinstance(self.__dict__.get("_open_tiles"), OrderedDict):
            self._open_tiles = OrderedDict()

    def close(self) -> None:
        """Close all open tile datasets.

        Defensive against construction failures: ``__del__`` may run
        on a partially-initialised instance where ``_open_tiles`` was
        never assigned (e.g., bad ``dem_path``).
        """
        cache = self.__dict__.get("_open_tiles")
        if cache is None:
            return
        for ds in cache.values():
            try:
                ds.close()
            except Exception:
                pass
        cache.clear()

    def __enter__(self) -> "TiledGeoDTED":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


__all__ = ["TiledGeoDTED"]
