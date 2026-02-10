# -*- coding: utf-8 -*-
"""
Data Preparation Base - ABC and shared types for chip/tile index computation.

Defines the ``ChipBase`` abstract base class that provides image dimension
management and coordinate clipping, along with the ``ChipRegion`` named tuple
used as the standardized return type for all chip and tile position queries.

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
from abc import ABC
from typing import NamedTuple, Tuple, Union

# Third-party
import numpy as np


class ChipRegion(NamedTuple):
    """Rectangular region within an image, defined by clipped pixel bounds.

    All indices are in image coordinates and guaranteed within valid image
    bounds. Use directly for numpy slicing::

        chip = image[region.row_start:region.row_end,
                     region.col_start:region.col_end]

    Attributes
    ----------
    row_start : int
        First row (inclusive). Always ``>= 0``.
    col_start : int
        First column (inclusive). Always ``>= 0``.
    row_end : int
        Last row (exclusive). Always ``<= nrows``.
    col_end : int
        Last column (exclusive). Always ``<= ncols``.
    """

    row_start: int
    col_start: int
    row_end: int
    col_end: int


class ChipBase(ABC):
    """Base class for chip and tile index computation.

    Manages image dimensions and provides coordinate snapping. Subclasses
    implement specific chipping strategies (point-centered, tiled grids)
    that return ``ChipRegion`` instances.

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
    """

    def __init__(self, nrows: int, ncols: int) -> None:
        if not isinstance(nrows, int) or not isinstance(ncols, int):
            raise TypeError(
                f"nrows and ncols must be int, got "
                f"nrows={type(nrows).__name__}, ncols={type(ncols).__name__}"
            )
        if nrows <= 0 or ncols <= 0:
            raise ValueError(
                f"nrows and ncols must be positive, got "
                f"nrows={nrows}, ncols={ncols}"
            )
        self._nrows = nrows
        self._ncols = ncols

    @property
    def nrows(self) -> int:
        """Number of image rows.

        Returns
        -------
        int
        """
        return self._nrows

    @property
    def ncols(self) -> int:
        """Number of image columns.

        Returns
        -------
        int
        """
        return self._ncols

    @property
    def shape(self) -> Tuple[int, int]:
        """Image dimensions as ``(nrows, ncols)``.

        Returns
        -------
        Tuple[int, int]
        """
        return (self._nrows, self._ncols)

    def _snap_region(
        self,
        row_start: int,
        col_start: int,
        row_width: int,
        col_width: int,
    ) -> ChipRegion:
        """Snap a region to fit entirely within image bounds.

        Slides the window inward to maintain the requested dimensions.
        If the requested width exceeds the image dimension, the region
        is clamped to the full image extent in that dimension.

        Parameters
        ----------
        row_start : int
            Requested first row (may be negative or overshoot).
        col_start : int
            Requested first column (may be negative or overshoot).
        row_width : int
            Requested number of rows.
        col_width : int
            Requested number of columns.

        Returns
        -------
        ChipRegion
            Region snapped inside ``[0, nrows]`` x ``[0, ncols]``
            with dimensions preserved when possible.

        Examples
        --------
        500-row image, chip of 100 near the top edge:

        >>> base._snap_region(-25, 0, 100, 100)
        ChipRegion(row_start=0, col_start=0, row_end=100, col_end=100)

        500-row image, chip of 100 near the bottom edge:

        >>> base._snap_region(425, 0, 100, 100)
        ChipRegion(row_start=400, col_start=0, row_end=500, col_end=100)
        """
        # Clamp dimensions to image size
        rw = min(row_width, self._nrows)
        cw = min(col_width, self._ncols)

        # Snap start positions to keep region inside bounds
        rs = max(0, min(row_start, self._nrows - rw))
        cs = max(0, min(col_start, self._ncols - cw))

        return ChipRegion(rs, cs, rs + rw, cs + cw)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nrows={self._nrows}, ncols={self._ncols})"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalize_pair(
    value: Union[int, Tuple[int, int]], name: str
) -> Tuple[int, int]:
    """Convert an int or (int, int) tuple to a validated (int, int) pair.

    Parameters
    ----------
    value : int or Tuple[int, int]
        Scalar or pair value. If scalar, both elements are set equal.
    name : str
        Parameter name for error messages.

    Returns
    -------
    Tuple[int, int]
        Validated (rows, cols) pair.

    Raises
    ------
    TypeError
        If value is not int or tuple of two ints.
    ValueError
        If any element is not positive.
    """
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return (value, value)
    if isinstance(value, tuple) and len(value) == 2:
        r, c = value
        if not isinstance(r, int) or not isinstance(c, int):
            raise TypeError(f"{name} tuple elements must be int")
        if r <= 0 or c <= 0:
            raise ValueError(
                f"{name} elements must be positive, got ({r}, {c})"
            )
        return (r, c)
    raise TypeError(
        f"{name} must be int or Tuple[int, int], got {type(value).__name__}"
    )
