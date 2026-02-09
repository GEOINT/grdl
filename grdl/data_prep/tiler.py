# -*- coding: utf-8 -*-
"""
Tiler - Compute overlapping tile regions within a bounded image.

Provides stride-based tile grid computation for systematic image coverage.
Returns ``ChipRegion`` named tuples containing clipped index bounds,
decoupling tile layout planning from pixel data access.

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
2026-02-06

Modified
--------
2026-02-09
"""

# Standard library
from typing import List, Optional, Tuple, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.data_prep.base import ChipBase, ChipRegion, _normalize_pair


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
            self._snap_region(
                int(r_origins[i]), int(c_origins[i]), tr, tc
            )
            for i in range(len(r_origins))
        ]

    def __repr__(self) -> str:
        return (
            f"Tiler(nrows={self._nrows}, ncols={self._ncols}, "
            f"tile_size={self._tile_size}, stride={self._stride})"
        )
