# -*- coding: utf-8 -*-
"""
Co-Registration Base Classes - Abstract interfaces for image co-registration.

Defines the ``CoRegistration`` ABC and the ``RegistrationResult`` data class
that all co-registration algorithms must produce. Co-registration estimates a
spatial transform that maps pixel coordinates from a moving image to the
coordinate frame of a fixed (reference) image.

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
2026-02-11
"""

# Standard library
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

# Third-party
import numpy as np

# GRDL internal
from grdl.coregistration.utils import apply_transform_to_points


class RegistrationResult:
    """Result of an image co-registration estimation.

    Contains the estimated spatial transform and quality metrics produced by
    a ``CoRegistration.estimate()`` call. The transform maps pixel coordinates
    from the moving image to the fixed image coordinate frame.

    The transform can be applied to:

    - **Raster data**: via ``CoRegistration.apply()`` (warps image pixels).
    - **Vector data**: via ``transform_points()`` on this object, or the
      ``transform_geometry`` / ``transform_detection_set`` bridge functions
      in ``grdl.transforms`` (transforms detection coordinates without
      raster interpolation).

    Parameters
    ----------
    transform_matrix : np.ndarray
        Transformation matrix. Shape is (2, 3) for affine transforms or
        (3, 3) for projective (homography) transforms. Maps moving image
        pixel coordinates (row, col) to fixed image coordinates.
    residual_rms : float
        Root mean square residual error in pixels after applying the
        estimated transform to the matched points.
    num_matches : int
        Number of point correspondences used to estimate the transform.
    inlier_ratio : float
        Fraction of initial matches classified as inliers (0.0 to 1.0).
        Only meaningful for RANSAC-based methods; set to 1.0 for
        least-squares methods without outlier rejection.
    metadata : Dict[str, Any]
        Algorithm-specific metadata (e.g., feature descriptor type,
        RANSAC threshold, number of iterations).

    Attributes
    ----------
    transform_matrix : np.ndarray
    residual_rms : float
    num_matches : int
    inlier_ratio : float
    metadata : Dict[str, Any]
    """

    def __init__(
        self,
        transform_matrix: np.ndarray,
        residual_rms: float,
        num_matches: int,
        inlier_ratio: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.transform_matrix = transform_matrix
        self.residual_rms = residual_rms
        self.num_matches = num_matches
        self.inlier_ratio = inlier_ratio
        self.metadata = metadata or {}

    @property
    def is_affine(self) -> bool:
        """Whether this is an affine (2x3) or projective (3x3) transform.

        Returns
        -------
        bool
            True if transform_matrix has shape (2, 3).
        """
        return self.transform_matrix.shape == (2, 3)

    @property
    def is_projective(self) -> bool:
        """Whether this is a projective (homography) transform.

        Returns
        -------
        bool
            True if transform_matrix has shape (3, 3).
        """
        return self.transform_matrix.shape == (3, 3)

    @property
    def inverse_transform_matrix(self) -> np.ndarray:
        """Inverse of the transform matrix (fixed -> moving).

        For affine (2, 3) transforms, the matrix is expanded to (3, 3)
        before inversion. The result is always (3, 3).

        Returns
        -------
        np.ndarray
            Inverse transform matrix, shape (3, 3).

        Raises
        ------
        np.linalg.LinAlgError
            If the transform matrix is singular.
        """
        if self.transform_matrix.shape == (2, 3):
            full = np.eye(3, dtype=np.float64)
            full[:2, :] = self.transform_matrix
            return np.linalg.inv(full)
        return np.linalg.inv(self.transform_matrix)

    def transform_points(
        self,
        points: np.ndarray,
        inverse: bool = False,
    ) -> np.ndarray:
        """Transform a set of 2D points using this registration result.

        Convenience wrapper around ``apply_transform_to_points`` that
        uses the stored transform matrix (or its inverse).

        Parameters
        ----------
        points : np.ndarray
            Points to transform. Shape (N, 2), columns are (row, col).
        inverse : bool
            If False (default), apply the forward transform
            (moving -> fixed). If True, apply the inverse transform
            (fixed -> moving).

        Returns
        -------
        np.ndarray
            Transformed points. Shape (N, 2), columns are (row, col).
        """
        matrix = (
            self.inverse_transform_matrix if inverse
            else self.transform_matrix
        )
        return apply_transform_to_points(points, matrix)

    def __repr__(self) -> str:
        shape = self.transform_matrix.shape
        kind = 'affine' if self.is_affine else 'projective'
        return (
            f"RegistrationResult({kind}, "
            f"rms={self.residual_rms:.4f}px, "
            f"matches={self.num_matches}, "
            f"inliers={self.inlier_ratio:.1%})"
        )


class CoRegistration(ABC):
    """Abstract base class for image co-registration algorithms.

    Co-registration aligns a moving image to a fixed (reference) image by
    estimating a spatial transform. The two-step interface separates
    estimation (``estimate``) from application (``apply``), allowing the
    same transform to be applied to multiple images or bands.

    Subclasses must implement ``estimate`` and ``apply``. Geolocation
    objects are optional and can be used by algorithms that benefit from
    geographic prior knowledge (e.g., initial alignment guess).
    """

    @abstractmethod
    def estimate(
        self,
        fixed: np.ndarray,
        moving: np.ndarray,
        fixed_geo: Optional[Any] = None,
        moving_geo: Optional[Any] = None,
    ) -> RegistrationResult:
        """Estimate the transform that aligns moving to fixed.

        Parameters
        ----------
        fixed : np.ndarray
            Reference image. Shape (rows, cols) for single-band or
            (rows, cols, bands) for multi-band. The moving image will
            be aligned to this coordinate frame.
        moving : np.ndarray
            Image to be registered. Same shape conventions as fixed.
        fixed_geo : Optional[Geolocation]
            Geolocation for the fixed image. Used by algorithms that
            leverage geographic priors for initial alignment.
        moving_geo : Optional[Geolocation]
            Geolocation for the moving image.

        Returns
        -------
        RegistrationResult
            Estimated transform and quality metrics.

        Raises
        ------
        ValueError
            If images have incompatible shapes or insufficient overlap
            for registration.
        RuntimeError
            If the algorithm fails to converge or finds insufficient
            matches.
        """
        ...

    @abstractmethod
    def apply(
        self,
        moving: np.ndarray,
        result: RegistrationResult,
        output_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Warp the moving image using an estimated transform.

        Parameters
        ----------
        moving : np.ndarray
            Image to warp. Shape (rows, cols) or (rows, cols, bands).
        result : RegistrationResult
            Registration result from a previous ``estimate`` call.
        output_shape : Optional[Tuple[int, int]]
            Output (rows, cols). If None, uses the moving image shape.

        Returns
        -------
        np.ndarray
            Warped image aligned to the fixed image coordinate frame.
            Same number of bands as input; pixels outside the valid
            region are set to 0.
        """
        ...
