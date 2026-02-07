# -*- coding: utf-8 -*-
"""
Chip Extractor - Extract image chips at specified locations.

Provides chip extraction centered at point coordinates or from bounding
boxes of polygon regions. Handles edge cases where chips extend beyond
image bounds by padding. Supports both 2D and 3D imagery arrays.

Author
------
Steven Siebert

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
2026-02-06
"""

# Standard library
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.data_prep.base import _normalize_pair, _validate_image


class ChipExtractor:
    """Extract image chips at specified locations.

    Chips are rectangular sub-images extracted from a larger image. Point
    extraction centers chips at given coordinates. Polygon extraction uses
    the bounding box of each polygon and also produces a binary mask
    indicating which pixels fall inside the polygon.

    Parameters
    ----------
    chip_size : int or Tuple[int, int]
        (chip_rows, chip_cols). If int, square chips.
    pad_mode : str
        numpy pad mode for edge chips. Default ``'constant'``.
    pad_value : float
        Value for constant padding. Default ``0.0``.

    Raises
    ------
    TypeError
        If chip_size is not int or Tuple[int, int].
    ValueError
        If chip_size has non-positive elements.

    Examples
    --------
    >>> import numpy as np
    >>> from grdl.data_prep import ChipExtractor
    >>> image = np.random.rand(100, 100)
    >>> extractor = ChipExtractor(chip_size=32)
    >>> points = np.array([[50, 50], [10, 10]])
    >>> chips = extractor.extract_at_points(image, points)
    >>> chips.shape
    (2, 32, 32)
    """

    def __init__(
        self,
        chip_size: Union[int, Tuple[int, int]],
        pad_mode: str = 'constant',
        pad_value: float = 0.0,
    ) -> None:
        self._chip_size = _normalize_pair(chip_size, 'chip_size')
        self._pad_mode = pad_mode
        self._pad_value = pad_value

    @property
    def chip_size(self) -> Tuple[int, int]:
        """The (chip_rows, chip_cols) dimensions.

        Returns
        -------
        Tuple[int, int]
            Chip dimensions.
        """
        return self._chip_size

    def _extract_chip(
        self, image: np.ndarray, row_start: int, col_start: int,
        chip_rows: int, chip_cols: int
    ) -> np.ndarray:
        """Extract a single chip with padding for out-of-bounds regions.

        Parameters
        ----------
        image : np.ndarray
            2D or 3D source image.
        row_start : int
            Top-left row of the chip (may be negative).
        col_start : int
            Top-left col of the chip (may be negative).
        chip_rows : int
            Height of the chip.
        chip_cols : int
            Width of the chip.

        Returns
        -------
        np.ndarray
            Extracted chip of shape ``(chip_rows, chip_cols)`` for 2D or
            ``(bands, chip_rows, chip_cols)`` for 3D.
        """
        is_3d = image.ndim == 3
        if is_3d:
            bands, img_rows, img_cols = image.shape
        else:
            img_rows, img_cols = image.shape

        # Compute the overlap between the chip window and the image
        src_row_start = max(0, row_start)
        src_col_start = max(0, col_start)
        src_row_end = min(img_rows, row_start + chip_rows)
        src_col_end = min(img_cols, col_start + chip_cols)

        # Offsets into the chip array
        dst_row_start = src_row_start - row_start
        dst_col_start = src_col_start - col_start
        dst_row_end = dst_row_start + (src_row_end - src_row_start)
        dst_col_end = dst_col_start + (src_col_end - src_col_start)

        # Handle fully out-of-bounds chips
        if src_row_start >= src_row_end or src_col_start >= src_col_end:
            if is_3d:
                chip = np.full(
                    (bands, chip_rows, chip_cols),
                    self._pad_value, dtype=image.dtype
                )
            else:
                chip = np.full(
                    (chip_rows, chip_cols),
                    self._pad_value, dtype=image.dtype
                )
            return chip

        # Allocate chip and fill with pad value
        if is_3d:
            chip = np.full(
                (bands, chip_rows, chip_cols),
                self._pad_value, dtype=image.dtype
            )
            chip[:, dst_row_start:dst_row_end,
                 dst_col_start:dst_col_end] = (
                image[:, src_row_start:src_row_end,
                      src_col_start:src_col_end]
            )
        else:
            chip = np.full(
                (chip_rows, chip_cols),
                self._pad_value, dtype=image.dtype
            )
            chip[dst_row_start:dst_row_end,
                 dst_col_start:dst_col_end] = (
                image[src_row_start:src_row_end,
                      src_col_start:src_col_end]
            )

        return chip

    def extract_at_points(
        self, image: np.ndarray, points: np.ndarray
    ) -> np.ndarray:
        """Extract chips centered at given (row, col) points.

        Each chip is centered at the specified point. If a chip extends
        beyond image boundaries, it is padded.

        Parameters
        ----------
        image : np.ndarray
            2D ``(rows, cols)`` or 3D ``(bands, rows, cols)`` source image.
        points : np.ndarray
            ``(N, 2)`` array of ``(row, col)`` center coordinates. Values
            are rounded to the nearest integer.

        Returns
        -------
        np.ndarray
            For 2D input: ``(N, chip_rows, chip_cols)`` array.
            For 3D input: ``(N, bands, chip_rows, chip_cols)`` array.

        Raises
        ------
        TypeError
            If image is not a numpy ndarray or points is not a numpy
            ndarray.
        ValueError
            If image is not 2D or 3D, or points is not shape ``(N, 2)``.
        """
        _validate_image(image)
        if not isinstance(points, np.ndarray):
            raise TypeError(
                f"points must be np.ndarray, got {type(points).__name__}"
            )
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                f"points must have shape (N, 2), got {points.shape}"
            )

        n_points = points.shape[0]
        cr, cc = self._chip_size
        is_3d = image.ndim == 3

        # Compute top-left corners from center points
        half_r = cr // 2
        half_c = cc // 2
        centers = np.round(points).astype(np.int64)
        row_starts = centers[:, 0] - half_r
        col_starts = centers[:, 1] - half_c

        if is_3d:
            bands = image.shape[0]
            chips = np.empty(
                (n_points, bands, cr, cc), dtype=image.dtype
            )
        else:
            chips = np.empty((n_points, cr, cc), dtype=image.dtype)

        for i in range(n_points):
            chips[i] = self._extract_chip(
                image, int(row_starts[i]), int(col_starts[i]), cr, cc
            )

        return chips

    def extract_at_polygons(
        self,
        image: np.ndarray,
        polygons: List[np.ndarray],
        labels: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Extract chips from bounding boxes of polygon regions.

        For each polygon, computes the bounding box, extracts a chip of
        that size (padded if needed), and creates a binary mask indicating
        which pixels fall inside the polygon.

        Parameters
        ----------
        image : np.ndarray
            2D ``(rows, cols)`` or 3D ``(bands, rows, cols)`` source image.
        polygons : List[np.ndarray]
            List of ``(M, 2)`` arrays of ``(row, col)`` polygon vertices.
            Each polygon must have at least 3 vertices.
        labels : np.ndarray, optional
            ``(N,)`` array of labels for each polygon. If provided, must
            have the same length as ``polygons``.

        Returns
        -------
        List[Dict[str, Any]]
            Each dict contains:

            - ``'chip'``: np.ndarray -- extracted chip image.
            - ``'mask'``: np.ndarray -- binary mask (bool), same spatial
              shape as chip. True where pixels are inside the polygon.
            - ``'bbox'``: Tuple[int, int, int, int] -- bounding box as
              ``(row_min, col_min, row_max, col_max)``.
            - ``'label'``: value from labels array (only if labels
              provided).

        Raises
        ------
        TypeError
            If image is not a numpy ndarray.
        ValueError
            If image is not 2D or 3D, polygons is empty, any polygon
            has fewer than 3 vertices, or labels length does not match
            polygons length.
        """
        _validate_image(image)
        if not polygons:
            raise ValueError("polygons list must not be empty")
        if labels is not None and len(labels) != len(polygons):
            raise ValueError(
                f"labels length ({len(labels)}) must match polygons "
                f"length ({len(polygons)})"
            )

        results: List[Dict[str, Any]] = []

        for idx, polygon in enumerate(polygons):
            if not isinstance(polygon, np.ndarray):
                raise TypeError(
                    f"polygon at index {idx} must be np.ndarray, "
                    f"got {type(polygon).__name__}"
                )
            if polygon.ndim != 2 or polygon.shape[1] != 2:
                raise ValueError(
                    f"polygon at index {idx} must have shape (M, 2), "
                    f"got {polygon.shape}"
                )
            if polygon.shape[0] < 3:
                raise ValueError(
                    f"polygon at index {idx} must have at least 3 vertices, "
                    f"got {polygon.shape[0]}"
                )

            # Compute bounding box
            row_min = int(np.floor(polygon[:, 0].min()))
            col_min = int(np.floor(polygon[:, 1].min()))
            row_max = int(np.ceil(polygon[:, 0].max()))
            col_max = int(np.ceil(polygon[:, 1].max()))

            chip_rows = row_max - row_min
            chip_cols = col_max - col_min

            # Ensure minimum chip size of 1x1
            chip_rows = max(1, chip_rows)
            chip_cols = max(1, chip_cols)

            # Extract chip at bounding box location
            chip = self._extract_chip(
                image, row_min, col_min, chip_rows, chip_cols
            )

            # Create polygon mask using ray-casting algorithm (vectorized)
            mask = self._polygon_mask(
                polygon, row_min, col_min, chip_rows, chip_cols
            )

            entry: Dict[str, Any] = {
                'chip': chip,
                'mask': mask,
                'bbox': (row_min, col_min, row_max, col_max),
            }
            if labels is not None:
                entry['label'] = labels[idx]

            results.append(entry)

        return results

    @staticmethod
    def _polygon_mask(
        polygon: np.ndarray,
        row_offset: int,
        col_offset: int,
        mask_rows: int,
        mask_cols: int,
    ) -> np.ndarray:
        """Create a binary mask for a polygon using vectorized ray casting.

        Uses the even-odd rule: a point is inside the polygon if a ray
        cast from the point crosses an odd number of polygon edges.

        Parameters
        ----------
        polygon : np.ndarray
            ``(M, 2)`` array of ``(row, col)`` polygon vertices.
        row_offset : int
            Row offset of the mask origin in image coordinates.
        col_offset : int
            Column offset of the mask origin in image coordinates.
        mask_rows : int
            Number of rows in the mask.
        mask_cols : int
            Number of columns in the mask.

        Returns
        -------
        np.ndarray
            Boolean mask of shape ``(mask_rows, mask_cols)``.
        """
        # Translate polygon to mask-local coordinates (pixel centers at +0.5)
        verts = polygon.astype(np.float64)
        local_rows = verts[:, 0] - row_offset
        local_cols = verts[:, 1] - col_offset

        n_verts = len(local_rows)

        # Create grid of pixel center coordinates
        grid_r, grid_c = np.mgrid[0:mask_rows, 0:mask_cols]
        grid_r = grid_r.astype(np.float64) + 0.5
        grid_c = grid_c.astype(np.float64) + 0.5
        # Flatten for vectorized processing
        pr = grid_r.ravel()  # (mask_rows * mask_cols,)
        pc = grid_c.ravel()

        inside = np.zeros(pr.shape[0], dtype=bool)

        # Ray casting: for each edge, determine which points are crossed
        for i in range(n_verts):
            j = (i + 1) % n_verts
            ri, ci = local_rows[i], local_cols[i]
            rj, cj = local_rows[j], local_cols[j]

            # Check if the test point's row is between the edge endpoints
            cond = ((ri <= pr) & (pr < rj)) | ((rj <= pr) & (pr < ri))

            # Compute the column of the edge at the test row
            if abs(rj - ri) > 1e-12:
                col_intersect = ci + (pr - ri) * (cj - ci) / (rj - ri)
                crossing = cond & (pc < col_intersect)
                inside ^= crossing

        return inside.reshape(mask_rows, mask_cols)

    def __repr__(self) -> str:
        return (
            f"ChipExtractor(chip_size={self._chip_size}, "
            f"pad_mode='{self._pad_mode}', pad_value={self._pad_value})"
        )
