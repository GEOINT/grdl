# -*- coding: utf-8 -*-
"""
Affine Co-Registration - Least-squares affine transform estimation.

Estimates a 2D affine transform (translation, rotation, scale, shear) from
point correspondences using least-squares fitting. Suitable when the
geometric distortion between images is well-modeled by an affine
transformation (6 degrees of freedom).

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
from typing import Any, Optional, Tuple

# Third-party
import numpy as np

# GRDL internal
from grdl.coregistration.base import CoRegistration, RegistrationResult
from grdl.coregistration.utils import (
    compute_residuals,
    compute_rms,
    warp_image,
)
from grdl.image_processing.versioning import processor_version


@processor_version('0.1.0')
class AffineCoRegistration(CoRegistration):
    """Affine co-registration from control point correspondences.

    Estimates an affine transform (2x3 matrix) that maps moving image
    control points to fixed image control points using ordinary
    least-squares. Requires a minimum of 3 non-collinear point pairs.

    Parameters
    ----------
    control_points_fixed : np.ndarray
        Control points in the fixed image. Shape (N, 2), columns are
        (row, col). N >= 3.
    control_points_moving : np.ndarray
        Corresponding control points in the moving image. Shape (N, 2),
        columns are (row, col). Must have same N as fixed.
    interpolation_order : int
        Interpolation order for ``apply``: 0=nearest, 1=bilinear,
        3=bicubic. Default is 1.

    Raises
    ------
    ValueError
        If fewer than 3 point pairs are provided or shapes do not match.

    Examples
    --------
    >>> import numpy as np
    >>> fixed_pts = np.array([[10, 10], [10, 90], [90, 50]])
    >>> moving_pts = np.array([[12, 11], [12, 91], [92, 51]])
    >>> coreg = AffineCoRegistration(fixed_pts, moving_pts)
    >>> result = coreg.estimate(fixed_img, moving_img)
    >>> aligned = coreg.apply(moving_img, result)
    """

    def __init__(
        self,
        control_points_fixed: np.ndarray,
        control_points_moving: np.ndarray,
        interpolation_order: int = 1,
    ) -> None:
        if control_points_fixed.shape[0] < 3:
            raise ValueError(
                f"Affine registration requires at least 3 control points, "
                f"got {control_points_fixed.shape[0]}"
            )
        if control_points_fixed.shape != control_points_moving.shape:
            raise ValueError(
                f"Control point arrays must have the same shape. "
                f"Fixed: {control_points_fixed.shape}, "
                f"Moving: {control_points_moving.shape}"
            )
        self._cp_fixed = np.asarray(control_points_fixed, dtype=np.float64)
        self._cp_moving = np.asarray(control_points_moving, dtype=np.float64)
        self._interpolation_order = interpolation_order

    def estimate(
        self,
        fixed: np.ndarray,
        moving: np.ndarray,
        fixed_geo: Optional[Any] = None,
        moving_geo: Optional[Any] = None,
    ) -> RegistrationResult:
        """Estimate affine transform using least-squares on control points.

        The images are not used directly -- the transform is computed
        entirely from the control points provided at construction time.
        The image parameters are accepted for interface compatibility.

        Parameters
        ----------
        fixed : np.ndarray
            Reference image (unused in estimation, kept for ABC compliance).
        moving : np.ndarray
            Moving image (unused in estimation).
        fixed_geo : Optional[Geolocation]
            Geolocation for fixed image (unused).
        moving_geo : Optional[Geolocation]
            Geolocation for moving image (unused).

        Returns
        -------
        RegistrationResult
            Affine transform (2x3 matrix) and quality metrics.
        """
        n = self._cp_moving.shape[0]

        # Build system: for each point, [row_m, col_m, 1] @ A^T = [row_f, col_f]
        # A is (2, 3), flattened to 6 unknowns solved via least-squares.
        ones = np.ones((n, 1))
        src = np.hstack([self._cp_moving, ones])  # (N, 3)

        # Solve for row and col transforms independently
        # src @ a_row = fixed_rows, src @ a_col = fixed_cols
        a_row, _, _, _ = np.linalg.lstsq(src, self._cp_fixed[:, 0], rcond=None)
        a_col, _, _, _ = np.linalg.lstsq(src, self._cp_fixed[:, 1], rcond=None)

        transform_matrix = np.vstack([a_row, a_col])  # (2, 3)

        residuals = compute_residuals(
            self._cp_fixed, self._cp_moving, transform_matrix
        )
        rms = compute_rms(residuals)

        return RegistrationResult(
            transform_matrix=transform_matrix,
            residual_rms=rms,
            num_matches=n,
            inlier_ratio=1.0,
            metadata={
                'method': 'affine_least_squares',
                'interpolation_order': self._interpolation_order,
                'max_residual': float(np.max(residuals)),
            },
        )

    def apply(
        self,
        moving: np.ndarray,
        result: RegistrationResult,
        output_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Warp moving image using the estimated affine transform.

        Parameters
        ----------
        moving : np.ndarray
            Image to warp. Shape (rows, cols) or (rows, cols, bands).
        result : RegistrationResult
            Registration result from ``estimate``.
        output_shape : Optional[Tuple[int, int]]
            Output (rows, cols). If None, uses moving image shape.

        Returns
        -------
        np.ndarray
            Warped image aligned to the fixed image coordinate frame.
        """
        return warp_image(
            moving,
            result.transform_matrix,
            output_shape=output_shape,
            order=self._interpolation_order,
        )
