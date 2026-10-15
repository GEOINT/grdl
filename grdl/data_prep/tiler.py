# -*- coding: utf-8 -*-
"""
Tiler - Compute overlapping tile regions within a bounded image.

Provides stride-based tile grid computation for systematic image coverage.
Returns ``ChipRegion`` named tuples containing clipped index bounds,
decoupling tile layout planning from pixel data access.

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
2026-02-06

Modified
--------
2026-06-07
"""

# Standard library
from typing import List, NamedTuple, Optional, Tuple, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.data_prep.base import ChipBase, ChipRegion, _normalize_pair


class HaloTile(NamedTuple):
    """A tile core plus the padded region needed to process it.

    Read ``read``, process it, then keep only ``crop`` of the result;
    the cropped block covers exactly ``core``.

    Attributes
    ----------
    core : ChipRegion
        Pixels this tile is responsible for (a partition cell).
    read : ChipRegion
        ``core`` grown by the halo and clipped to the image bounds.
    """

    core: ChipRegion
    read: ChipRegion

    @property
    def crop(self) -> Tuple[slice, slice]:
        """Row and column slices selecting ``core`` inside a ``read`` block."""
        return (
            slice(self.core.row_start - self.read.row_start,
                  self.core.row_end - self.read.row_start),
            slice(self.core.col_start - self.read.col_start,
                  self.core.col_end - self.read.col_start),
        )


def total_halo(*halos: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Sum per-step halos of a processing chain.

    Parameters
    ----------
    *halos : int or Tuple[int, int]
        Per-step ``(row, col)`` halos; an ``int`` applies to both axes.

    Returns
    -------
    Tuple[int, int]
        Combined ``(row_halo, col_halo)``.
    """
    rows = cols = 0
    for h in halos:
        r, c = _normalize_halo(h)
        rows += r
        cols += c
    return rows, cols


def _normalize_halo(halo: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(halo, (int, np.integer)):
        pair = (int(halo), int(halo))
    else:
        pair = tuple(int(h) for h in halo)
        if len(pair) != 2:
            raise ValueError(f"halo must be an int or a pair, got {halo!r}")
    if pair[0] < 0 or pair[1] < 0:
        raise ValueError(f"halo must be non-negative, got {halo!r}")
    return pair


class Tiler(ChipBase):
    """Compute overlapping tile regions with configurable stride.

    Tiles are laid out row-major from top-left to bottom-right. When
    stride is less than tile size, tiles overlap. Edge tiles snap
    inward to maintain the full tile size.

    Parameters
    ----------
    nrows : int
        Number of rows in the image.
    ncols : int
        Number of columns in the image.
    tile_size : int or Tuple[int, int]
        ``(tile_rows, tile_cols)``. If ``int``, square tiles.
    stride : int or Tuple[int, int], optional
        ``(stride_rows, stride_cols)``. Defaults to ``tile_size``
        (no overlap). Must not exceed ``tile_size`` in either dimension.

    Raises
    ------
    TypeError
        If ``nrows``, ``ncols``, ``tile_size``, or ``stride`` has wrong type.
    ValueError
        If ``nrows`` or ``ncols`` is not positive, ``tile_size`` or
        ``stride`` has non-positive elements, or ``stride`` exceeds
        ``tile_size`` in any dimension.

    Examples
    --------
    >>> from grdl.data_prep import Tiler
    >>> tiler = Tiler(nrows=100, ncols=100, tile_size=32, stride=16)
    >>> regions = tiler.tile_positions()
    >>> len(regions)
    25
    >>> regions[0]
    ChipRegion(row_start=0, col_start=0, row_end=32, col_end=32)
    """

    def __init__(
        self,
        nrows: int,
        ncols: int,
        tile_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> None:
        super().__init__(nrows, ncols)
        self._tile_size = _normalize_pair(tile_size, 'tile_size')
        if stride is None:
            self._stride = self._tile_size
        else:
            self._stride = _normalize_pair(stride, 'stride')
        if self._stride[0] > self._tile_size[0] or \
                self._stride[1] > self._tile_size[1]:
            raise ValueError(
                f"stride {self._stride} must not exceed "
                f"tile_size {self._tile_size}"
            )

    @property
    def tile_size(self) -> Tuple[int, int]:
        """The ``(tile_rows, tile_cols)`` dimensions.

        Returns
        -------
        Tuple[int, int]
        """
        return self._tile_size

    @property
    def stride(self) -> Tuple[int, int]:
        """The ``(stride_rows, stride_cols)`` step sizes.

        Returns
        -------
        Tuple[int, int]
        """
        return self._stride

    def tile_positions(self) -> List[ChipRegion]:
        """Compute tile regions covering the image.

        Tiles are placed at regular stride intervals, row-major from
        top-left to bottom-right. Edge tiles snap inward to maintain
        the full ``tile_size``, which may increase overlap at the
        boundary.

        Returns
        -------
        List[ChipRegion]
            All tile regions in row-major order.

        Examples
        --------
        >>> tiler = Tiler(nrows=100, ncols=100, tile_size=64)
        >>> regions = tiler.tile_positions()
        >>> len(regions)
        4
        >>> regions[-1]
        ChipRegion(row_start=36, col_start=36, row_end=100, col_end=100)
        """
        tr, tc = self._tile_size
        sr, sc = self._stride

        # Compute number of tiles needed to cover each dimension
        if self._nrows <= tr:
            n_row_tiles = 1
        else:
            n_row_tiles = 1 + int(np.ceil((self._nrows - tr) / sr))

        if self._ncols <= tc:
            n_col_tiles = 1
        else:
            n_col_tiles = 1 + int(np.ceil((self._ncols - tc) / sc))

        row_starts = np.arange(n_row_tiles) * sr
        col_starts = np.arange(n_col_tiles) * sc

        rr, cc = np.meshgrid(row_starts, col_starts, indexing='ij')
        r_origins = rr.ravel()
        c_origins = cc.ravel()

        return [
            self._snap_region(int(r), int(c), tr, tc)
            for r, c in zip(r_origins, c_origins)
        ]

    def partition_positions(self) -> List[ChipRegion]:
        """Compute a strict non-overlapping partition of the image.

        Unlike :meth:`tile_positions` (which snaps edge tiles inward to keep a
        constant size, creating overlap), this covers every pixel exactly
        once: edge tiles are clipped to the image bounds and are therefore
        smaller than ``tile_size`` where the dimension is not an exact
        multiple. ``stride`` is ignored. Use this for per-pixel aggregation
        (statistics, histograms, masks, reductions), where overlap would
        double-count pixels.

        Returns
        -------
        List[ChipRegion]
            Non-overlapping tile regions in row-major order whose union is
            the full image.

        Examples
        --------
        >>> from grdl.data_prep import Tiler
        >>> tiler = Tiler(nrows=100, ncols=100, tile_size=64)
        >>> regions = tiler.partition_positions()
        >>> len(regions)
        4
        >>> regions[-1]
        ChipRegion(row_start=64, col_start=64, row_end=100, col_end=100)
        """
        tr, tc = self._tile_size
        regions: List[ChipRegion] = []
        for r0 in range(0, self._nrows, tr):
            r1 = min(r0 + tr, self._nrows)
            for c0 in range(0, self._ncols, tc):
                c1 = min(c0 + tc, self._ncols)
                regions.append(ChipRegion(r0, c0, r1, c1))
        return regions

    def halo_tiles(
        self,
        halo: Union[int, Tuple[int, int]],
        overlapping: bool = False,
    ) -> List[HaloTile]:
        """Attach a halo to every tile cell.

        By default cores form a partition (no overlap, every pixel once).
        With ``overlapping=True`` cores follow :meth:`tile_positions`, so
        neighbors overlap and results need blending, e.g. with
        :class:`~grdl.data_prep.Stitcher`. Each ``read`` region extends
        the core by ``halo`` pixels, clipped at the image edge so edge
        behavior matches a full-image run.

        Parameters
        ----------
        halo : int or Tuple[int, int]
            ``(row_halo, col_halo)`` in pixels.
        overlapping : bool
            Use the stride-based overlapping layout instead of a partition.

        Returns
        -------
        List[HaloTile]
            Tiles in row-major order.
        """
        hr, hc = _normalize_halo(halo)
        cores = self.tile_positions() if overlapping \
            else self.partition_positions()
        tiles = []
        for core in cores:
            read = ChipRegion(
                max(0, core.row_start - hr),
                max(0, core.col_start - hc),
                min(self._nrows, core.row_end + hr),
                min(self._ncols, core.col_end + hc),
            )
            tiles.append(HaloTile(core, read))
        return tiles

    def __repr__(self) -> str:
        return (
            f"Tiler(nrows={self._nrows}, ncols={self._ncols}, "
            f"tile_size={self._tile_size}, stride={self._stride})"
        )
