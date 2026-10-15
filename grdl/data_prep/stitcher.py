# -*- coding: utf-8 -*-
"""
Stitcher - Assemble per-tile results into a full-image array.

Places tile results (arrays or dicts of arrays whose last two axes are
spatial) into a full-scene output and resolves overlap regions with a
selectable method. The default is a weighted average using pyramid
(distance-to-edge) weights.

Author
------
Jason Fritz
43161141+stryder-vtx@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-10-07

Modified
--------
2026-10-07
"""

# Standard library
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.data_prep.base import ChipRegion
from grdl.data_prep.tiler import HaloTile

METHODS = ('weighted', 'mean', 'first', 'last', 'max', 'min')

TileData = Union[np.ndarray, Dict[str, np.ndarray]]
Sink = Callable[[str, Tuple[int, ...], np.dtype], np.ndarray]


def memmap_sink(directory: Union[str, Path]) -> Sink:
    """Create an output sink backed by ``np.memmap`` files.

    Parameters
    ----------
    directory : str or Path
        Directory for the ``<key>.dat`` files; created if missing.

    Returns
    -------
    Callable
        ``sink(key, shape, dtype) -> np.ndarray`` for ``Stitcher``.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)

    def _sink(key: str, shape: Tuple[int, ...], dtype: np.dtype):
        return np.memmap(
            root / f'{key}.dat', dtype=dtype, mode='w+', shape=shape,
        )

    return _sink


def _in_memory_sink(
    key: str, shape: Tuple[int, ...], dtype: np.dtype,
) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


class Stitcher:
    """Combine tile results into one full-image output.

    Parameters
    ----------
    nrows : int
        Output rows.
    ncols : int
        Output columns.
    method : str
        Overlap resolution: ``'weighted'`` (default; pyramid-weighted
        average), ``'mean'``, ``'first'``, ``'last'``, ``'max'`` or
        ``'min'``. ``'max'``/``'min'`` require real data.
    feather : int, optional
        For ``'weighted'``, cap the weight ramp at this many pixels from
        each tile edge. ``None`` uses full pyramid weights.
    sink : callable, optional
        ``sink(key, shape, dtype)`` allocating each output array, e.g.
        :func:`memmap_sink`. Defaults to in-memory arrays.

    Notes
    -----
    Tile data must have shape ``(..., rows, cols)`` matching its region;
    leading axes (channels, matrix indices) are preserved. A dict of such
    arrays is stitched per key. Averaged methods output at least float32.

    Examples
    --------
    >>> st = Stitcher(100, 100)
    >>> st.add(ChipRegion(0, 0, 60, 100), a)
    >>> st.add(ChipRegion(40, 0, 100, 100), b)
    >>> full = st.finalize()
    """

    def __init__(
        self,
        nrows: int,
        ncols: int,
        method: str = 'weighted',
        feather: Optional[int] = None,
        sink: Optional[Sink] = None,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if nrows <= 0 or ncols <= 0:
            raise ValueError("nrows and ncols must be positive")
        if feather is not None and feather < 1:
            raise ValueError("feather must be >= 1")
        self._shape = (nrows, ncols)
        self._method = method
        self._feather = feather
        self._sink = sink or _in_memory_sink
        self._out: Dict[Optional[str], np.ndarray] = {}
        self._is_dict: Optional[bool] = None
        self._weights: Optional[np.ndarray] = None
        self._counts: Optional[np.ndarray] = None
        self._covered: Optional[np.ndarray] = None
        self._finalized = False

    def add(self, region: ChipRegion, data: TileData) -> None:
        """Place one tile result.

        Parameters
        ----------
        region : ChipRegion
            Image location of ``data``.
        data : np.ndarray or dict of np.ndarray
            Tile result with trailing shape ``(rows, cols)`` of ``region``.

        Raises
        ------
        ValueError
            If the region is out of bounds or the data shape or dtype is
            inconsistent with earlier tiles.
        """
        if self._finalized:
            raise RuntimeError("Stitcher already finalized")
        r0, c0, r1, c1 = region
        if r0 < 0 or c0 < 0 or r1 > self._shape[0] or c1 > self._shape[1] \
                or r1 <= r0 or c1 <= c0:
            raise ValueError(
                f"region {tuple(region)} outside output {self._shape}"
            )
        is_dict = isinstance(data, dict)
        if self._is_dict is None:
            self._is_dict = is_dict
            self._alloc_state()
        elif self._is_dict != is_dict:
            raise ValueError("Tiles must all be arrays or all be dicts")

        items = data.items() if is_dict else [(None, data)]
        if is_dict and set(data) != set(self._out) and self._out:
            raise ValueError("Dict tiles must have identical keys")

        h, w = r1 - r0, c1 - c0
        weight = self._tile_weight(h, w)
        for key, arr in items:
            arr = np.asarray(arr)
            if arr.ndim < 2 or arr.shape[-2:] != (h, w):
                raise ValueError(
                    f"data{'[' + key + ']' if key else ''} shape "
                    f"{arr.shape} does not match region size ({h}, {w})"
                )
            out = self._out.get(key)
            if out is None:
                out = self._allocate(key, arr)
            elif arr.shape[:-2] != out.shape[:-2]:
                raise ValueError(
                    f"Leading shape {arr.shape[:-2]} differs from "
                    f"{out.shape[:-2]} for key {key!r}"
                )
            self._place(out, arr, r0, r1, c0, c1, weight)
        self._mark(r0, r1, c0, c1, weight)

    def add_halo_tile(self, tile: HaloTile, data: TileData) -> None:
        """Crop a processed ``tile.read`` block to its core and place it."""
        rs, cs = tile.crop
        if isinstance(data, dict):
            cropped = {k: np.asarray(v)[..., rs, cs] for k, v in data.items()}
        else:
            cropped = np.asarray(data)[..., rs, cs]
        self.add(tile.core, cropped)

    def finalize(self, allow_gaps: bool = False) -> TileData:
        """Normalize and return the stitched output.

        Parameters
        ----------
        allow_gaps : bool
            If False (default), raise when any pixel was never covered.

        Returns
        -------
        np.ndarray or dict of np.ndarray
        """
        if self._is_dict is None:
            raise ValueError("No tiles were added")
        if self._finalized:
            raise RuntimeError("Stitcher already finalized")
        if self._weights is not None:
            covered = self._weights > 0
        elif self._counts is not None:
            covered = self._counts > 0
        else:
            covered = self._covered
        if not allow_gaps and not bool(covered.all()):
            raise ValueError(
                f"{int((~covered).sum())} output pixels not covered by any tile"
            )
        if self._method == 'weighted':
            safe = np.where(self._weights > 0, self._weights, 1.0)
            for out in self._out.values():
                out /= safe
        elif self._method == 'mean':
            safe = np.where(self._counts > 0, self._counts, 1)
            for out in self._out.values():
                out /= safe
        self._finalized = True
        if self._is_dict:
            return dict(self._out)
        return self._out[None]

    # ------------------------------------------------------------------

    def _alloc_state(self) -> None:
        if self._method == 'weighted':
            self._weights = np.zeros(self._shape, dtype=np.float32)
        elif self._method == 'mean':
            self._counts = np.zeros(self._shape, dtype=np.float32)
        else:
            self._covered = np.zeros(self._shape, dtype=bool)

    def _allocate(self, key: Optional[str], arr: np.ndarray) -> np.ndarray:
        dtype = arr.dtype
        if self._method in ('max', 'min') and np.iscomplexobj(arr):
            raise ValueError(f"method {self._method!r} requires real data")
        if self._method in ('weighted', 'mean'):
            dtype = np.result_type(dtype, np.float32)
        shape = arr.shape[:-2] + self._shape
        out = self._sink(
            'out' if key is None else str(key), shape, np.dtype(dtype),
        )
        self._out[key] = out
        return out

    def _tile_weight(self, h: int, w: int) -> Optional[np.ndarray]:
        if self._method != 'weighted':
            return None

        def ramp(n: int) -> np.ndarray:
            d = np.minimum(np.arange(1, n + 1), np.arange(n, 0, -1))
            if self._feather is not None:
                d = np.minimum(d, self._feather)
            return d.astype(np.float32)

        return np.outer(ramp(h), ramp(w))

    def _place(self, out, arr, r0, r1, c0, c1, weight) -> None:
        sl = (Ellipsis, slice(r0, r1), slice(c0, c1))
        m = self._method
        if m == 'weighted':
            out[sl] += arr * weight
        elif m == 'mean':
            out[sl] += arr
        elif m == 'last':
            out[sl] = arr
        elif m == 'first':
            fresh = ~self._covered[r0:r1, c0:c1]
            region = out[sl]
            region[..., fresh] = arr[..., fresh]
            out[sl] = region
        else:
            cov = self._covered[r0:r1, c0:c1]
            region = out[sl]
            pick = np.maximum if m == 'max' else np.minimum
            region = np.where(cov, pick(region, arr), arr)
            out[sl] = region

    def _mark(self, r0, r1, c0, c1, weight) -> None:
        if self._method == 'weighted':
            self._weights[r0:r1, c0:c1] += weight
        elif self._method == 'mean':
            self._counts[r0:r1, c0:c1] += 1
        else:
            self._covered[r0:r1, c0:c1] = True
