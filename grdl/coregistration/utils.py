# -*- coding: utf-8 -*-
"""
Co-Registration Utilities - Quality metrics and helper functions.

Provides helper functions for computing registration quality metrics,
estimating overlap between images, and validating control point sets.

Dependencies
------------
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
2026-02-06

Modified
--------
2026-02-06
"""

# Standard library
from typing import Optional, Tuple

# Third-party
import numpy as np
from scipy.ndimage import map_coordinates


def compute_residuals(
    fixed_points: np.ndarray,
    moving_points: np.ndarray,
    transform_matrix: np.ndarray,
) -> np.ndarray:
    """Compute per-point residuals after applying a transform.

    Applies the transform to the moving points and computes the Euclidean
    distance to the corresponding fixed points.

    Parameters
    ----------
    fixed_points : np.ndarray
        Target points in fixed image. Shape (N, 2), columns are (row, col).
    moving_points : np.ndarray
        Source points in moving image. Shape (N, 2), columns are (row, col).
    transform_matrix : np.ndarray
        Affine (2, 3) or projective (3, 3) transform matrix.

    Returns
    -------
    np.ndarray
        Per-point Euclidean residuals in pixels. Shape (N,).
    """
    transformed = apply_transform_to_points(moving_points, transform_matrix)
    diff = fixed_points - transformed
    return np.sqrt(np.sum(diff ** 2, axis=1))


def compute_rms(residuals: np.ndarray) -> float:
    """Compute root mean square of residual errors.

    Parameters
    ----------
    residuals : np.ndarray
        Per-point residuals. Shape (N,).

    Returns
    -------
    float
        RMS residual error in pixels.
    """
    return float(np.sqrt(np.mean(residuals ** 2)))


def apply_transform_to_points(
    points: np.ndarray,
    transform_matrix: np.ndarray,
) -> np.ndarray:
    """Apply a spatial transform to a set of 2D points.

    Parameters
    ----------
    points : np.ndarray
        Points to transform. Shape (N, 2), columns are (row, col).
    transform_matrix : np.ndarray
        Affine (2, 3) or projective (3, 3) transform matrix.

    Returns
    -------
    np.ndarray
        Transformed points. Shape (N, 2), columns are (row, col).
    """
    n = points.shape[0]

    if transform_matrix.shape == (2, 3):
        # Affine: [A | t] @ [x; 1]
        ones = np.ones((n, 1))
        pts_h = np.hstack([points, ones])  # (N, 3)
        result = pts_h @ transform_matrix.T  # (N, 2)
        return result

    elif transform_matrix.shape == (3, 3):
        # Projective: H @ [x; 1], then divide by w
        ones = np.ones((n, 1))
        pts_h = np.hstack([points, ones])  # (N, 3)
        result_h = pts_h @ transform_matrix.T  # (N, 3)
        w = result_h[:, 2:3]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        return result_h[:, :2] / w

    else:
        raise ValueError(
            f"Transform matrix must be (2, 3) or (3, 3), got {transform_matrix.shape}"
        )


def warp_image(
    image: np.ndarray,
    transform_matrix: np.ndarray,
    output_shape: Optional[Tuple[int, int]] = None,
    order: int = 1,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Warp an image using a spatial transform via inverse mapping.

    Uses scipy's map_coordinates for interpolation. The transform maps
    moving→fixed coordinates, so we invert it to compute the inverse
    mapping (fixed→moving) for each output pixel.

    Parameters
    ----------
    image : np.ndarray
        Input image. Shape (rows, cols) or (rows, cols, bands).
    transform_matrix : np.ndarray
        Forward transform (moving→fixed). Shape (2, 3) or (3, 3).
    output_shape : Optional[Tuple[int, int]]
        Output (rows, cols). If None, uses input image shape.
    order : int
        Interpolation order: 0=nearest, 1=bilinear, 3=bicubic.
    fill_value : float
        Value for pixels outside the input image bounds.

    Returns
    -------
    np.ndarray
        Warped image. Shape (output_rows, output_cols) or
        (output_rows, output_cols, bands).
    """
    is_multiband = image.ndim == 3
    if output_shape is None:
        output_shape = image.shape[:2]

    out_rows, out_cols = output_shape

    # Compute inverse transform
    if transform_matrix.shape == (2, 3):
        # Convert to 3x3 for inversion
        full = np.eye(3)
        full[:2, :] = transform_matrix
        inv_matrix = np.linalg.inv(full)
    else:
        inv_matrix = np.linalg.inv(transform_matrix)

    # Create output coordinate grid
    row_coords, col_coords = np.mgrid[0:out_rows, 0:out_cols]
    ones = np.ones_like(row_coords)
    coords_h = np.stack([row_coords, col_coords, ones], axis=0)  # (3, R, C)
    coords_flat = coords_h.reshape(3, -1)  # (3, R*C)

    # Apply inverse transform to get source coordinates
    src_flat = inv_matrix @ coords_flat  # (3, R*C)

    if transform_matrix.shape == (3, 3):
        w = src_flat[2:3, :]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        src_flat = src_flat[:2, :] / w
    else:
        src_flat = src_flat[:2, :]

    src_rows = src_flat[0, :].reshape(out_rows, out_cols)
    src_cols = src_flat[1, :].reshape(out_rows, out_cols)

    if is_multiband:
        num_bands = image.shape[2]
        result = np.full(
            (out_rows, out_cols, num_bands), fill_value, dtype=image.dtype
        )
        for b in range(num_bands):
            result[:, :, b] = map_coordinates(
                image[:, :, b],
                [src_rows, src_cols],
                order=order,
                mode='constant',
                cval=fill_value,
            )
    else:
        result = map_coordinates(
            image,
            [src_rows, src_cols],
            order=order,
            mode='constant',
            cval=fill_value,
        )

    return result


def estimate_overlap_fraction(
    fixed_shape: Tuple[int, int],
    moving_shape: Tuple[int, int],
    transform_matrix: np.ndarray,
) -> float:
    """Estimate the fractional overlap between fixed and moving images.

    Transforms the corners of the moving image to the fixed coordinate
    frame and computes the area of intersection relative to the fixed
    image area.

    Parameters
    ----------
    fixed_shape : Tuple[int, int]
        (rows, cols) of the fixed image.
    moving_shape : Tuple[int, int]
        (rows, cols) of the moving image.
    transform_matrix : np.ndarray
        Transform matrix (moving → fixed). Shape (2, 3) or (3, 3).

    Returns
    -------
    float
        Overlap fraction (0.0 to 1.0) relative to the fixed image.
    """
    mr, mc = moving_shape
    corners = np.array([
        [0, 0],
        [0, mc - 1],
        [mr - 1, mc - 1],
        [mr - 1, 0],
    ], dtype=np.float64)

    transformed = apply_transform_to_points(corners, transform_matrix)

    fr, fc = fixed_shape
    # Clip to fixed image bounds
    clipped_rows = np.clip(transformed[:, 0], 0, fr - 1)
    clipped_cols = np.clip(transformed[:, 1], 0, fc - 1)

    if clipped_rows.max() <= clipped_rows.min() or clipped_cols.max() <= clipped_cols.min():
        return 0.0

    overlap_area = (clipped_rows.max() - clipped_rows.min()) * (
        clipped_cols.max() - clipped_cols.min()
    )
    fixed_area = fr * fc

    return float(np.clip(overlap_area / fixed_area, 0.0, 1.0))
