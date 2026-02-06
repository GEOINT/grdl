# -*- coding: utf-8 -*-
"""
Feature-Match Co-Registration - Automated feature-based image registration.

Provides automated co-registration by detecting and matching keypoint
features (ORB, SIFT) between a fixed and moving image, then estimating
a spatial transform using RANSAC for robust outlier rejection.

Dependencies
------------
opencv-python-headless

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

try:
    import cv2
except ImportError:
    raise ImportError(
        "FeatureMatchCoRegistration requires opencv-python-headless. "
        "Install with: pip install opencv-python-headless>=4.5"
    )

# GRDL internal
from grdl.coregistration.base import CoRegistration, RegistrationResult
from grdl.coregistration.utils import (
    compute_residuals,
    compute_rms,
    warp_image,
)
from grdl.image_processing.versioning import processor_version


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an image to uint8 for OpenCV feature detection.

    Handles float, complex, and multi-band images by extracting
    magnitude and normalizing to 0-255 range.

    Parameters
    ----------
    image : np.ndarray
        Input image. Shape (rows, cols) or (rows, cols, bands).

    Returns
    -------
    np.ndarray
        Single-channel uint8 image suitable for feature detection.
    """
    if image.ndim == 3:
        # Use first band
        img = image[:, :, 0]
    else:
        img = image

    # Handle complex imagery (SAR)
    if np.iscomplexobj(img):
        img = np.abs(img)

    img = img.astype(np.float64)

    # Normalize to 0-255
    vmin, vmax = np.nanmin(img), np.nanmax(img)
    if vmax - vmin > 0:
        img = (img - vmin) / (vmax - vmin) * 255.0
    else:
        img = np.zeros_like(img)

    return img.astype(np.uint8)


@processor_version('0.1.0')
class FeatureMatchCoRegistration(CoRegistration):
    """Automated feature-based co-registration using OpenCV.

    Detects keypoint features in both images, matches them, and estimates
    either an affine or projective transform using RANSAC for robust
    outlier rejection.

    Parameters
    ----------
    method : str
        Feature detection method. One of 'orb' or 'sift'. Default 'orb'.
    max_features : int
        Maximum number of features to detect. Default 5000.
    transform_type : str
        Transform model to fit. One of 'affine' or 'homography'.
        Default 'affine'.
    ransac_threshold : float
        RANSAC inlier distance threshold in pixels. Default 5.0.
    match_ratio : float
        Lowe's ratio test threshold for filtering matches. Default 0.75.
    interpolation_order : int
        Interpolation order for ``apply``: 0=nearest, 1=bilinear,
        3=bicubic. Default is 1.

    Raises
    ------
    ValueError
        If method or transform_type is not recognized.

    Examples
    --------
    >>> coreg = FeatureMatchCoRegistration(method='orb', max_features=2000)
    >>> result = coreg.estimate(fixed_image, moving_image)
    >>> aligned = coreg.apply(moving_image, result)
    """

    _VALID_METHODS = ('orb', 'sift')
    _VALID_TRANSFORMS = ('affine', 'homography')

    def __init__(
        self,
        method: str = 'orb',
        max_features: int = 5000,
        transform_type: str = 'affine',
        ransac_threshold: float = 5.0,
        match_ratio: float = 0.75,
        interpolation_order: int = 1,
    ) -> None:
        method = method.lower()
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"Unknown feature method '{method}'. "
                f"Choose from: {self._VALID_METHODS}"
            )
        transform_type = transform_type.lower()
        if transform_type not in self._VALID_TRANSFORMS:
            raise ValueError(
                f"Unknown transform type '{transform_type}'. "
                f"Choose from: {self._VALID_TRANSFORMS}"
            )

        self._method = method
        self._max_features = max_features
        self._transform_type = transform_type
        self._ransac_threshold = ransac_threshold
        self._match_ratio = match_ratio
        self._interpolation_order = interpolation_order

    def estimate(
        self,
        fixed: np.ndarray,
        moving: np.ndarray,
        fixed_geo: Optional[Any] = None,
        moving_geo: Optional[Any] = None,
    ) -> RegistrationResult:
        """Detect features, match, and estimate transform with RANSAC.

        Parameters
        ----------
        fixed : np.ndarray
            Reference image. Shape (rows, cols) or (rows, cols, bands).
        moving : np.ndarray
            Image to register. Same shape conventions as fixed.
        fixed_geo : Optional[Geolocation]
            Geolocation for fixed image (unused).
        moving_geo : Optional[Geolocation]
            Geolocation for moving image (unused).

        Returns
        -------
        RegistrationResult
            Estimated transform and quality metrics.

        Raises
        ------
        RuntimeError
            If insufficient matches are found for estimation.
        """
        fixed_u8 = _to_uint8(fixed)
        moving_u8 = _to_uint8(moving)

        # Detect features
        detector = self._create_detector()
        kp_fixed, desc_fixed = detector.detectAndCompute(fixed_u8, None)
        kp_moving, desc_moving = detector.detectAndCompute(moving_u8, None)

        if desc_fixed is None or desc_moving is None:
            raise RuntimeError(
                "Feature detection failed: no descriptors found in one or "
                "both images."
            )

        # Match features
        good_matches = self._match_features(desc_fixed, desc_moving)

        min_matches = 4 if self._transform_type == 'homography' else 3
        if len(good_matches) < min_matches:
            raise RuntimeError(
                f"Insufficient matches ({len(good_matches)}) for "
                f"{self._transform_type} estimation (need >= {min_matches})."
            )

        # Extract matched point coordinates
        # OpenCV keypoints use (x, y) = (col, row), we need (row, col)
        pts_fixed = np.array(
            [kp_fixed[m.queryIdx].pt for m in good_matches], dtype=np.float64
        )
        pts_moving = np.array(
            [kp_moving[m.trainIdx].pt for m in good_matches], dtype=np.float64
        )

        # Convert (x, y) → (row, col)
        pts_fixed_rc = pts_fixed[:, ::-1]
        pts_moving_rc = pts_moving[:, ::-1]

        # Estimate transform with RANSAC
        # OpenCV functions use (x, y) convention
        if self._transform_type == 'homography':
            H, mask = cv2.findHomography(
                pts_moving, pts_fixed,
                cv2.RANSAC, self._ransac_threshold,
            )
            if H is None:
                raise RuntimeError(
                    "Homography estimation failed (RANSAC could not find "
                    "a valid model)."
                )
            # Convert from OpenCV (x,y) convention to (row,col)
            # H_xy maps (x_m, y_m) -> (x_f, y_f) where x=col, y=row
            # We need H_rc mapping (r_m, c_m) -> (r_f, c_f)
            # swap: P = [[0,1,0],[1,0,0],[0,0,1]]
            P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
            transform_matrix = P @ H @ P
        else:
            M, mask = cv2.estimateAffinePartial2D(
                pts_moving, pts_fixed,
                method=cv2.RANSAC,
                ransacReprojThreshold=self._ransac_threshold,
            )
            if M is None:
                raise RuntimeError(
                    "Affine estimation failed (RANSAC could not find "
                    "a valid model)."
                )
            # Convert (x,y) affine to (row,col) affine
            # M_xy is 2x3: [a b tx; c d ty] in (x,y) space
            # For (row,col): swap rows/cols of the linear part and translation
            P2x2 = np.array([[0, 1], [1, 0]], dtype=np.float64)
            linear = M[:, :2]  # 2x2
            trans = M[:, 2:3]  # 2x1
            transform_matrix = np.hstack([
                P2x2 @ linear @ P2x2,
                P2x2 @ trans,
            ])

        if mask is not None:
            inlier_mask = mask.ravel().astype(bool)
        else:
            inlier_mask = np.ones(len(good_matches), dtype=bool)

        num_inliers = int(np.sum(inlier_mask))
        inlier_ratio = num_inliers / len(good_matches) if good_matches else 0.0

        # Compute residuals on inliers
        inlier_fixed = pts_fixed_rc[inlier_mask]
        inlier_moving = pts_moving_rc[inlier_mask]
        residuals = compute_residuals(
            inlier_fixed, inlier_moving, transform_matrix
        )
        rms = compute_rms(residuals)

        return RegistrationResult(
            transform_matrix=transform_matrix,
            residual_rms=rms,
            num_matches=num_inliers,
            inlier_ratio=inlier_ratio,
            metadata={
                'method': f'feature_match_{self._method}',
                'transform_type': self._transform_type,
                'total_matches': len(good_matches),
                'num_inliers': num_inliers,
                'ransac_threshold': self._ransac_threshold,
                'max_features': self._max_features,
                'max_residual': float(np.max(residuals)) if len(residuals) > 0 else 0.0,
            },
        )

    def apply(
        self,
        moving: np.ndarray,
        result: RegistrationResult,
        output_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Warp moving image using the estimated transform.

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

    def _create_detector(self) -> cv2.Feature2D:
        """Create the feature detector based on configuration.

        Returns
        -------
        cv2.Feature2D
            OpenCV feature detector instance.
        """
        if self._method == 'orb':
            return cv2.ORB_create(nfeatures=self._max_features)
        elif self._method == 'sift':
            return cv2.SIFT_create(nfeatures=self._max_features)
        else:
            raise ValueError(f"Unknown method: {self._method}")

    def _match_features(
        self,
        desc_fixed: np.ndarray,
        desc_moving: np.ndarray,
    ) -> list:
        """Match descriptors using brute-force matching with ratio test.

        Parameters
        ----------
        desc_fixed : np.ndarray
            Descriptors from fixed image.
        desc_moving : np.ndarray
            Descriptors from moving image.

        Returns
        -------
        list
            Good matches after Lowe's ratio test.
        """
        if self._method == 'orb':
            norm_type = cv2.NORM_HAMMING
        else:
            norm_type = cv2.NORM_L2

        bf = cv2.BFMatcher(norm_type)
        raw_matches = bf.knnMatch(desc_fixed, desc_moving, k=2)

        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self._match_ratio * n.distance:
                    good_matches.append(m)

        return good_matches
