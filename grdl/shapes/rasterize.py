# -*- coding: utf-8 -*-
"""
Boolean mask rasterization for projected shape polygons.

Converts an ``(N, 2)`` pixel-space vertex list into a boolean mask of a
target image shape. Uses scikit-image scan conversion for interior fill
(O(filled-pixels) via horizontal sweep) and for perimeter drawing.

Outline thickness greater than one pixel is implemented via
``scipy.ndimage.binary_dilation`` with a disk structuring element. All
coordinate handling is in ``(row, col)`` to match the library convention.

Dependencies
------------
numpy
scipy
scikit-image

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
2026-04-18

Modified
--------
2026-04-18
"""

# Standard library
from typing import Tuple

# Third-party
import numpy as np
from scipy.ndimage import binary_dilation
from skimage.draw import polygon as _sk_polygon
from skimage.draw import polygon_perimeter as _sk_polygon_perimeter
from skimage.draw import line as _sk_line


def rasterize_polygon(
    pixels: np.ndarray,
    image_shape: Tuple[int, int],
    fill: bool = True,
    outline: bool = False,
    outline_thickness: int = 1,
    closed: bool = True,
) -> np.ndarray:
    """Rasterize a pixel-space polygon into a boolean mask.

    Parameters
    ----------
    pixels : np.ndarray
        Shape ``(N, 2)`` vertex list ``[row, col]``. Need not be
        explicitly closed; for closed shapes the last vertex is
        implicitly connected to the first.
    image_shape : Tuple[int, int]
        ``(rows, cols)`` of the output mask.
    fill : bool
        Whether to fill the interior. Mutually meaningful only for
        closed shapes; requires ``len(pixels) >= 3``.
    outline : bool
        Whether to include the perimeter. For open shapes (``closed=False``)
        the result is always a polyline, regardless of this flag's value.
    outline_thickness : int
        Outline thickness in pixels. 1 = native skimage perimeter,
        greater values dilate with a disk kernel.
    closed : bool
        When False, treat ``pixels`` as an open polyline and skip the
        wrap edge.

    Returns
    -------
    np.ndarray[bool]
        Shape ``image_shape``. ``True`` where the shape lies.
    """
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"pixels must be (N, 2); got {pixels.shape}")
    if outline_thickness < 1:
        raise ValueError(
            f"outline_thickness must be >= 1; got {outline_thickness}"
        )

    rows_total, cols_total = image_shape
    mask = np.zeros((rows_total, cols_total), dtype=bool)

    if not _bbox_overlaps_image(pixels, rows_total, cols_total):
        # Polygon lies entirely outside the image: nothing to draw.
        # Return the empty mask -- no skimage calls (polygon_perimeter
        # raises IndexError when its Sutherland-Hodgman clip is empty).
        return mask

    rows = pixels[:, 0]
    cols = pixels[:, 1]

    if fill and closed and len(pixels) >= 3:
        rr, cc = _sk_polygon(rows, cols, shape=image_shape)
        mask[rr, cc] = True

    draw_outline = outline or not closed
    if draw_outline and len(pixels) >= 2:
        if closed and len(pixels) >= 3:
            try:
                rr, cc = _sk_polygon_perimeter(
                    rows, cols, shape=image_shape, clip=True,
                )
                mask[rr, cc] = True
            except IndexError:
                # Sutherland-Hodgman returned no surviving polygon
                # despite the bbox overlap check (shape grazes the
                # boundary). Safe to ignore the perimeter pass.
                pass
        else:
            # Open polyline: draw each segment explicitly.
            rr_int = np.rint(rows).astype(np.int64)
            cc_int = np.rint(cols).astype(np.int64)
            for i in range(len(pixels) - 1):
                line_rr, line_cc = _sk_line(
                    rr_int[i], cc_int[i], rr_int[i + 1], cc_int[i + 1],
                )
                valid = (
                    (line_rr >= 0) & (line_rr < rows_total)
                    & (line_cc >= 0) & (line_cc < cols_total)
                )
                mask[line_rr[valid], line_cc[valid]] = True

        if outline_thickness > 1:
            radius = outline_thickness
            struct = _disk(radius)
            mask = binary_dilation(mask, structure=struct)

    return mask


def _disk(radius: int) -> np.ndarray:
    """Return a disk-shaped boolean structuring element of given radius."""
    r = int(max(1, radius))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x ** 2 + y ** 2) <= r ** 2


def _bbox_overlaps_image(
    pixels: np.ndarray,
    rows_total: int,
    cols_total: int,
) -> bool:
    """True when the polygon bbox intersects the image rectangle."""
    finite = np.isfinite(pixels).all(axis=1)
    if not finite.any():
        return False
    pts = pixels[finite]
    row_min = pts[:, 0].min()
    row_max = pts[:, 0].max()
    col_min = pts[:, 1].min()
    col_max = pts[:, 1].max()
    if row_max < 0 or row_min > rows_total - 1:
        return False
    if col_max < 0 or col_min > cols_total - 1:
        return False
    return True
