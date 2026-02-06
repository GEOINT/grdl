# -*- coding: utf-8 -*-
"""
Tests for the co-registration module.

Tests co-registration base classes, affine and projective implementations,
and utility functions using synthetic image data with known transforms.

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

import numpy as np
import pytest

from grdl.coregistration.base import CoRegistration, RegistrationResult
from grdl.coregistration.affine import AffineCoRegistration
from grdl.coregistration.projective import ProjectiveCoRegistration
from grdl.coregistration.utils import (
    apply_transform_to_points,
    compute_residuals,
    compute_rms,
    estimate_overlap_fraction,
    warp_image,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_image():
    """Create a 100x100 synthetic image with distinct features."""
    rng = np.random.default_rng(42)
    img = rng.random((100, 100)).astype(np.float64)
    # Add some structure
    img[20:30, 20:30] = 1.0
    img[60:80, 40:60] = 0.0
    return img


@pytest.fixture
def identity_affine():
    """Identity affine transform (2x3)."""
    return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)


@pytest.fixture
def translation_affine():
    """Translation-only affine: shift by (5, 3) pixels."""
    return np.array([[1, 0, 5], [0, 1, 3]], dtype=np.float64)


@pytest.fixture
def identity_homography():
    """Identity homography (3x3)."""
    return np.eye(3, dtype=np.float64)


# ---------------------------------------------------------------------------
# RegistrationResult Tests
# ---------------------------------------------------------------------------

class TestRegistrationResult:

    def test_affine_result(self):
        mat = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        result = RegistrationResult(mat, 0.5, 10, 0.95)
        assert result.is_affine
        assert not result.is_projective
        assert result.residual_rms == 0.5
        assert result.num_matches == 10
        assert result.inlier_ratio == 0.95
        assert result.metadata == {}

    def test_projective_result(self):
        mat = np.eye(3, dtype=np.float64)
        result = RegistrationResult(mat, 1.0, 20, 0.8, {'key': 'val'})
        assert not result.is_affine
        assert result.is_projective
        assert result.metadata == {'key': 'val'}

    def test_repr(self):
        mat = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        result = RegistrationResult(mat, 0.1234, 5, 0.9)
        text = repr(result)
        assert 'affine' in text
        assert '0.1234' in text


# ---------------------------------------------------------------------------
# Utility Function Tests
# ---------------------------------------------------------------------------

class TestApplyTransformToPoints:

    def test_identity_affine(self, identity_affine):
        pts = np.array([[10, 20], [30, 40]], dtype=np.float64)
        result = apply_transform_to_points(pts, identity_affine)
        np.testing.assert_allclose(result, pts)

    def test_translation_affine(self, translation_affine):
        pts = np.array([[0, 0], [10, 10]], dtype=np.float64)
        result = apply_transform_to_points(pts, translation_affine)
        expected = np.array([[5, 3], [15, 13]], dtype=np.float64)
        np.testing.assert_allclose(result, expected)

    def test_identity_homography(self, identity_homography):
        pts = np.array([[10, 20], [30, 40]], dtype=np.float64)
        result = apply_transform_to_points(pts, identity_homography)
        np.testing.assert_allclose(result, pts)

    def test_invalid_shape_raises(self):
        pts = np.array([[1, 2]], dtype=np.float64)
        bad_mat = np.eye(4)
        with pytest.raises(ValueError, match="must be"):
            apply_transform_to_points(pts, bad_mat)


class TestComputeResiduals:

    def test_zero_residuals_identity(self, identity_affine):
        pts = np.array([[10, 20], [30, 40]], dtype=np.float64)
        residuals = compute_residuals(pts, pts, identity_affine)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)

    def test_known_residuals(self, identity_affine):
        fixed = np.array([[10, 10], [20, 20]], dtype=np.float64)
        moving = np.array([[13, 14], [20, 20]], dtype=np.float64)
        residuals = compute_residuals(fixed, moving, identity_affine)
        # First point: dist = sqrt((10-13)^2 + (10-14)^2) = 5.0
        assert pytest.approx(residuals[0], abs=1e-10) == 5.0
        # Second point: dist = 0
        assert pytest.approx(residuals[1], abs=1e-10) == 0.0


class TestComputeRms:

    def test_zero_residuals(self):
        assert compute_rms(np.array([0.0, 0.0, 0.0])) == 0.0

    def test_known_rms(self):
        residuals = np.array([3.0, 4.0])
        expected = np.sqrt((9 + 16) / 2)
        assert pytest.approx(compute_rms(residuals)) == expected


class TestWarpImage:

    def test_identity_warp_preserves_image(self, synthetic_image, identity_affine):
        result = warp_image(synthetic_image, identity_affine, order=1)
        # Interior should match (edges may have interpolation artifacts)
        np.testing.assert_allclose(
            result[5:-5, 5:-5], synthetic_image[5:-5, 5:-5], atol=1e-10
        )

    def test_identity_homography_preserves_image(
        self, synthetic_image, identity_homography
    ):
        result = warp_image(synthetic_image, identity_homography, order=1)
        np.testing.assert_allclose(
            result[5:-5, 5:-5], synthetic_image[5:-5, 5:-5], atol=1e-10
        )

    def test_multiband_warp(self, identity_affine):
        img = np.random.default_rng(0).random((50, 50, 3))
        result = warp_image(img, identity_affine, order=1)
        assert result.shape == img.shape
        np.testing.assert_allclose(
            result[5:-5, 5:-5, :], img[5:-5, 5:-5, :], atol=1e-10
        )

    def test_custom_output_shape(self, synthetic_image, identity_affine):
        result = warp_image(
            synthetic_image, identity_affine, output_shape=(200, 200)
        )
        assert result.shape == (200, 200)


class TestEstimateOverlapFraction:

    def test_identity_full_overlap(self, identity_affine):
        full = np.eye(3)
        full[:2, :] = identity_affine
        overlap = estimate_overlap_fraction((100, 100), (100, 100), identity_affine)
        assert pytest.approx(overlap, abs=0.05) == 1.0

    def test_half_overlap_translation(self):
        # Shift moving image 50 pixels right (col direction)
        T = np.array([[1, 0, 0], [0, 1, 50]], dtype=np.float64)
        overlap = estimate_overlap_fraction((100, 100), (100, 100), T)
        assert 0.4 < overlap < 0.6


# ---------------------------------------------------------------------------
# AffineCoRegistration Tests
# ---------------------------------------------------------------------------

class TestAffineCoRegistration:

    def test_recover_identity(self, synthetic_image):
        """Registration with identical points should produce identity."""
        pts = np.array([
            [10, 10], [10, 80], [80, 10], [80, 80], [50, 50]
        ], dtype=np.float64)
        coreg = AffineCoRegistration(pts, pts)
        result = coreg.estimate(synthetic_image, synthetic_image)
        # Transform should be close to identity
        expected = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        np.testing.assert_allclose(result.transform_matrix, expected, atol=1e-10)
        assert result.residual_rms < 1e-10

    def test_recover_translation(self, synthetic_image):
        """Registration should recover a known translation."""
        fixed_pts = np.array([
            [10, 10], [10, 80], [80, 10], [80, 80]
        ], dtype=np.float64)
        shift = np.array([5.0, 3.0])
        moving_pts = fixed_pts - shift  # moving is shifted back
        coreg = AffineCoRegistration(fixed_pts, moving_pts)
        result = coreg.estimate(synthetic_image, synthetic_image)
        # Translation components (last column)
        np.testing.assert_allclose(
            result.transform_matrix[:, 2], shift, atol=1e-10
        )
        assert result.residual_rms < 1e-10

    def test_apply_produces_correct_shape(self, synthetic_image):
        pts = np.array([
            [10, 10], [10, 80], [80, 10]
        ], dtype=np.float64)
        coreg = AffineCoRegistration(pts, pts)
        result = coreg.estimate(synthetic_image, synthetic_image)
        warped = coreg.apply(synthetic_image, result)
        assert warped.shape == synthetic_image.shape

    def test_apply_custom_output_shape(self, synthetic_image):
        pts = np.array([
            [10, 10], [10, 80], [80, 10]
        ], dtype=np.float64)
        coreg = AffineCoRegistration(pts, pts)
        result = coreg.estimate(synthetic_image, synthetic_image)
        warped = coreg.apply(synthetic_image, result, output_shape=(200, 200))
        assert warped.shape == (200, 200)

    def test_too_few_points_raises(self):
        pts = np.array([[10, 10], [20, 20]], dtype=np.float64)
        with pytest.raises(ValueError, match="at least 3"):
            AffineCoRegistration(pts, pts)

    def test_mismatched_shapes_raises(self):
        pts3 = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float64)
        pts4 = np.array([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=np.float64)
        with pytest.raises(ValueError, match="same shape"):
            AffineCoRegistration(pts3, pts4)

    def test_result_metadata(self, synthetic_image):
        pts = np.array([
            [10, 10], [10, 80], [80, 10]
        ], dtype=np.float64)
        coreg = AffineCoRegistration(pts, pts)
        result = coreg.estimate(synthetic_image, synthetic_image)
        assert result.metadata['method'] == 'affine_least_squares'
        assert result.num_matches == 3
        assert result.inlier_ratio == 1.0
        assert result.is_affine

    def test_has_processor_version(self):
        assert hasattr(AffineCoRegistration, '__processor_version__')
        assert AffineCoRegistration.__processor_version__ == '0.1.0'


# ---------------------------------------------------------------------------
# ProjectiveCoRegistration Tests
# ---------------------------------------------------------------------------

class TestProjectiveCoRegistration:

    def test_recover_identity(self, synthetic_image):
        """Registration with identical points should approximate identity."""
        pts = np.array([
            [10, 10], [10, 80], [80, 10], [80, 80], [50, 50]
        ], dtype=np.float64)
        coreg = ProjectiveCoRegistration(pts, pts)
        result = coreg.estimate(synthetic_image, synthetic_image)
        # Should be close to identity homography
        expected = np.eye(3, dtype=np.float64)
        np.testing.assert_allclose(result.transform_matrix, expected, atol=1e-6)
        assert result.residual_rms < 1e-6

    def test_recover_translation(self, synthetic_image):
        """Homography should recover a pure translation."""
        fixed_pts = np.array([
            [10, 10], [10, 80], [80, 10], [80, 80]
        ], dtype=np.float64)
        shift = np.array([7.0, 4.0])
        moving_pts = fixed_pts - shift
        coreg = ProjectiveCoRegistration(fixed_pts, moving_pts)
        result = coreg.estimate(synthetic_image, synthetic_image)
        # H should have translation in last column
        np.testing.assert_allclose(
            result.transform_matrix[:2, 2],
            shift,
            atol=1e-6,
        )
        assert result.residual_rms < 1e-6

    def test_apply_produces_correct_shape(self, synthetic_image):
        pts = np.array([
            [10, 10], [10, 80], [80, 10], [80, 80]
        ], dtype=np.float64)
        coreg = ProjectiveCoRegistration(pts, pts)
        result = coreg.estimate(synthetic_image, synthetic_image)
        warped = coreg.apply(synthetic_image, result)
        assert warped.shape == synthetic_image.shape

    def test_too_few_points_raises(self):
        pts = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float64)
        with pytest.raises(ValueError, match="at least 4"):
            ProjectiveCoRegistration(pts, pts)

    def test_mismatched_shapes_raises(self):
        pts4 = np.array(
            [[10, 10], [20, 20], [30, 30], [40, 40]], dtype=np.float64
        )
        pts5 = np.array(
            [[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]], dtype=np.float64
        )
        with pytest.raises(ValueError, match="same shape"):
            ProjectiveCoRegistration(pts4, pts5)

    def test_result_metadata(self, synthetic_image):
        pts = np.array([
            [10, 10], [10, 80], [80, 10], [80, 80]
        ], dtype=np.float64)
        coreg = ProjectiveCoRegistration(pts, pts)
        result = coreg.estimate(synthetic_image, synthetic_image)
        assert result.metadata['method'] == 'projective_dlt'
        assert result.is_projective

    def test_has_processor_version(self):
        assert hasattr(ProjectiveCoRegistration, '__processor_version__')
        assert ProjectiveCoRegistration.__processor_version__ == '0.1.0'


# ---------------------------------------------------------------------------
# ABC Contract Tests
# ---------------------------------------------------------------------------

class TestCoRegistrationABC:

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            CoRegistration()

    def test_affine_is_coregistration(self):
        pts = np.array([
            [10, 10], [10, 80], [80, 10]
        ], dtype=np.float64)
        coreg = AffineCoRegistration(pts, pts)
        assert isinstance(coreg, CoRegistration)

    def test_projective_is_coregistration(self):
        pts = np.array([
            [10, 10], [10, 80], [80, 10], [80, 80]
        ], dtype=np.float64)
        coreg = ProjectiveCoRegistration(pts, pts)
        assert isinstance(coreg, CoRegistration)
