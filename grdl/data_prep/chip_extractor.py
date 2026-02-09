# -*- coding: utf-8 -*-
"""
Chip Extractor - Compute chip regions within a bounded image.

Provides point-centered chip computation and whole-image chunking. Returns
``ChipRegion`` named tuples containing clipped index bounds, decoupling
chip planning from pixel data access.

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
from typing import List, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.data_prep.base import ChipBase, ChipRegion


def _is_scalar(val: object) -> bool:
    """Check if a value is scalar (not array-like)."""
    if isinstance(val, np.ndarray):
        return val.ndim == 0
    return isinstance(val, (int, float, np.integer, np.floating))


def _validate_positive_int(value: int, name: str) -> None:
    """Validate that a value is a positive integer.

    Parameters
    ----------
    value : int
        Value to validate.
    name : str
        Parameter name for error messages.

    Raises
    ------
    TypeError
        If ``value`` is not ``int``.
    ValueError
        If ``value`` is not positive.
    """
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


class ChipExtractor(ChipBase):
    """Compute chip regions for point-centered extraction and whole-image chunking.

    A planning utility that computes where chips fall within an image,
    returning clipped index bounds as ``ChipRegion`` instances. Does not
    handle pixel data.

    Parameters
    ----------
    nrows : int
        Number of rows in the image.
    ncols : int
        Number of columns in the image.

    Raises
    ------
    TypeError
        If ``nrows`` or ``ncols`` is not ``int``.
    ValueError
        If ``nrows`` or ``ncols`` is not positive.

    Examples
    --------
    >>> from grdl.data_prep import ChipExtractor
    >>> ext = ChipExtractor(nrows=100, ncols=200)
    >>> region = ext.chip_at_point(50, 100, row_width=32, col_width=32)
    >>> region
    ChipRegion(row_start=34, col_start=84, row_end=66, col_end=116)
    >>> image[region.row_start:region.row_end, region.col_start:region.col_end]
    """

    def __init__(self, nrows: int, ncols: int) -> None:
        super().__init__(nrows, ncols)

    def chip_at_point(
        self,
        row: Union[int, float, list, np.ndarray],
        col: Union[int, float, list, np.ndarray],
        row_width: int,
        col_width: int,
    ) -> Union[ChipRegion, List[ChipRegion]]:
        """Compute chip region(s) centered at the given point(s).

        The chip is centered at the given point. When centering would
        place the chip partially outside the image, the window snaps
        inward to maintain the full requested dimensions. The chip is
        only smaller than requested when the image itself is smaller
        than the chip dimensions.

        Accepts scalar or array inputs. Returns a single ``ChipRegion``
        for scalar inputs, or a list for array/list inputs.

        Parameters
        ----------
        row : int, float, list, or np.ndarray
            Row coordinate(s) of chip center(s). Rounded to nearest int.
        col : int, float, list, or np.ndarray
            Column coordinate(s) of chip center(s). Rounded to nearest int.
        row_width : int
            Number of rows in the chip (before clipping).
        col_width : int
            Number of columns in the chip (before clipping).

        Returns
        -------
        ChipRegion or List[ChipRegion]
            Clipped chip region(s). Single ``ChipRegion`` when inputs are
            scalar; list when inputs are array-like.

        Raises
        ------
        TypeError
            If ``row_width`` or ``col_width`` is not ``int``.
        ValueError
            If ``row_width`` or ``col_width`` is not positive, or if any
            center point is outside image bounds
            ``[0, nrows)`` x ``[0, ncols)``.

        Examples
        --------
        Single point:

        >>> ext = ChipExtractor(nrows=100, ncols=100)
        >>> ext.chip_at_point(50, 50, row_width=20, col_width=20)
        ChipRegion(row_start=40, col_start=40, row_end=60, col_end=60)

        Near edge (snapped to maintain full size):

        >>> ext.chip_at_point(5, 5, row_width=20, col_width=20)
        ChipRegion(row_start=0, col_start=0, row_end=20, col_end=20)

        Multiple points:

        >>> ext.chip_at_point([50, 5], [50, 5], row_width=20, col_width=20)
        [ChipRegion(row_start=40, ...), ChipRegion(row_start=0, ...)]
        """
        _validate_positive_int(row_width, 'row_width')
        _validate_positive_int(col_width, 'col_width')

        scalar = _is_scalar(row) and _is_scalar(col)

        rows_arr = np.round(np.asarray(row, dtype=np.float64)).astype(np.int64)
        cols_arr = np.round(np.asarray(col, dtype=np.float64)).astype(np.int64)

        # Ensure 1D arrays for uniform processing
        rows_arr = np.atleast_1d(rows_arr)
        cols_arr = np.atleast_1d(cols_arr)

        # Validate all points are within image bounds
        if np.any(rows_arr < 0) or np.any(rows_arr >= self._nrows):
            raise ValueError(
                f"row values must be in [0, {self._nrows}), "
                f"got range [{rows_arr.min()}, {rows_arr.max()}]"
            )
        if np.any(cols_arr < 0) or np.any(cols_arr >= self._ncols):
            raise ValueError(
                f"col values must be in [0, {self._ncols}), "
                f"got range [{cols_arr.min()}, {cols_arr.max()}]"
            )

        # Compute initial top-left corners from center points
        half_r = row_width // 2
        half_c = col_width // 2
        r_starts = rows_arr - half_r
        c_starts = cols_arr - half_c

        # Snap each region to fit inside image bounds
        regions = [
            self._snap_region(
                int(r_starts[i]), int(c_starts[i]), row_width, col_width
            )
            for i in range(len(rows_arr))
        ]

        if scalar:
            return regions[0]
        return regions

    def chip_positions(
        self,
        row_width: int,
        col_width: int,
    ) -> List[ChipRegion]:
        """Partition the image into chip regions of uniform size.

        Chips are laid out row-major from top-left to bottom-right. Edge
        chips snap inward to maintain the full requested dimensions,
        which may cause overlap with adjacent chips at the boundary.
        When the image is smaller than the requested chip size, a single
        chip covering the full image is returned.

        Parameters
        ----------
        row_width : int
            Number of rows per chip.
        col_width : int
            Number of columns per chip.

        Returns
        -------
        List[ChipRegion]
            All chip regions in row-major order.

        Raises
        ------
        TypeError
            If ``row_width`` or ``col_width`` is not ``int``.
        ValueError
            If ``row_width`` or ``col_width`` is not positive.

        Examples
        --------
        >>> ext = ChipExtractor(nrows=100, ncols=100)
        >>> regions = ext.chip_positions(row_width=50, col_width=50)
        >>> len(regions)
        4
        >>> regions[0]
        ChipRegion(row_start=0, col_start=0, row_end=50, col_end=50)
        """
        _validate_positive_int(row_width, 'row_width')
        _validate_positive_int(col_width, 'col_width')

        row_starts = np.arange(0, self._nrows, row_width)
        col_starts = np.arange(0, self._ncols, col_width)

        rr, cc = np.meshgrid(row_starts, col_starts, indexing='ij')
        r_origins = rr.ravel()
        c_origins = cc.ravel()

        return [
            self._snap_region(
                int(r_origins[i]), int(c_origins[i]), row_width, col_width
            )
            for i in range(len(r_origins))
        ]
