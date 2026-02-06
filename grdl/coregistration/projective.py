# -*- coding: utf-8 -*-
"""
Projective Co-Registration - Homography estimation from control points.

Estimates a projective (homography) transform from point correspondences
using Direct Linear Transform (DLT). Suitable when perspective distortion
is present between images (8 degrees of freedom).

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
class ProjectiveCoRegistration(CoRegistration):
    """Projective (homography) co-registration from control points.

    Estimates a 3x3 homography matrix using the Direct Linear Transform
    (DLT) algorithm. Requires a minimum of 4 non-collinear point pairs.

    Parameters
    ----------
    control_points_fixed : np.ndarray
        Control points in the fixed image. Shape (N, 2), columns are
        (row, col). N >= 4.
    control_points_moving : np.ndarray
        Corresponding control points in the moving image. Shape (N, 2).
    interpolation_order : int
        Interpolation order for ``apply``: 0=nearest, 1=bilinear,
        3=bicubic. Default is 1.

    Raises
    ------
    ValueError
        If fewer than 4 point pairs are provided or shapes do not match.

    Examples
    --------
    >>> import numpy as np
    >>> fixed_pts = np.array([[10, 10], [10, 90], [90, 10], [90, 90]])
    >>> moving_pts = np.array([[12, 11], [11, 92], [91, 12], [92, 91]])
    >>> coreg = ProjectiveCoRegistration(fixed_pts, moving_pts)
    >>> result = coreg.estimate(fixed_img, moving_img)
    >>> aligned = coreg.apply(moving_img, result)
    """

    def __init__(
        self,
        control_points_fixed: np.ndarray,
        control_points_moving: np.ndarray,
        interpolation_order: int = 1,
    ) -> None:
        if control_points_fixed.shape[0] < 4:
            raise ValueError(
                f"Projective registration requires at least 4 control points, "
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
        """Estimate homography using Direct Linear Transform (DLT).

        Parameters
        ----------
        fixed : np.ndarray
            Reference image (unused, kept for ABC compliance).
        moving : np.ndarray
            Moving image (unused).
        fixed_geo : Optional[Geolocation]
            Geolocation for fixed image (unused).
        moving_geo : Optional[Geolocation]
            Geolocation for moving image (unused).

        Returns
        -------
        RegistrationResult
            Projective transform (3x3 homography) and quality metrics.
        """
        n = self._cp_moving.shape[0]

        # Normalize points for numerical stability
        src_norm, T_src = self._normalize_points(self._cp_moving)
        dst_norm, T_dst = self._normalize_points(self._cp_fixed)

        # Build DLT system: Ah = 0
        A = np.zeros((2 * n, 9))
        for i in range(n):
            r_s, c_s = src_norm[i]
            r_d, c_d = dst_norm[i]
            A[2 * i] = [
                r_s, c_s, 1, 0, 0, 0,
                -r_d * r_s, -r_d * c_s, -r_d,
            ]
            A[2 * i + 1] = [
                0, 0, 0, r_s, c_s, 1,
                -c_d * r_s, -c_d * c_s, -c_d,
            ]

        # Solve via SVD
        _, _, Vt = np.linalg.svd(A)
        H_norm = Vt[-1].reshape(3, 3)

        # Denormalize
        H = np.linalg.inv(T_dst) @ H_norm @ T_src

        # Normalize so H[2, 2] = 1
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]

        residuals = compute_residuals(self._cp_fixed, self._cp_moving, H)
        rms = compute_rms(residuals)

        return RegistrationResult(
            transform_matrix=H,
            residual_rms=rms,
            num_matches=n,
            inlier_ratio=1.0,
            metadata={
                'method': 'projective_dlt',
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
        """Warp moving image using the estimated homography.

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

    @staticmethod
    def _normalize_points(
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize points for numerical stability in DLT.

        Translates centroid to origin and scales so mean distance from
        origin is sqrt(2).

        Parameters
        ----------
        points : np.ndarray
            Points to normalize. Shape (N, 2).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (normalized_points, normalization_matrix) where the matrix
            is 3x3 and can be used to denormalize results.
        """
        centroid = np.mean(points, axis=0)
        shifted = points - centroid
        mean_dist = np.mean(np.sqrt(np.sum(shifted ** 2, axis=1)))

        if mean_dist < 1e-12:
            scale = 1.0
        else:
            scale = np.sqrt(2.0) / mean_dist

        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1],
        ])

        ones = np.ones((points.shape[0], 1))
        pts_h = np.hstack([points, ones])
        normalized = (T @ pts_h.T).T[:, :2]

        return normalized, T
