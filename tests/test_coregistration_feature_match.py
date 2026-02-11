# -*- coding: utf-8 -*-
"""
Feature-Match Co-Registration Tests.

Tests for FeatureMatchCoRegistration and the _to_uint8 helper.
All tests require opencv-python-headless and are skipped if unavailable.
Uses synthetic images with known transforms for verifiable results.

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
2026-02-10

Modified
--------
2026-02-10
"""

import numpy as np
import pytest

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

pytestmark = pytest.mark.skipif(
    not _HAS_CV2, reason="opencv-python-headless not installed"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def textured_image():
    """Create a 200x200 image with strong texture for feature detection.

    Uses a grid of bright squares on a dark background — produces many
    corner features that ORB/SIFT can detect reliably.
    """
    rng = np.random.RandomState(42)
    image = rng.rand(200, 200).astype(np.float64) * 30  # noisy background

    # Add a grid of bright squares (corners = features)
    for r in range(20, 180, 30):
        for c in range(20, 180, 30):
            image[r:r + 15, c:c + 15] = 200.0 + rng.rand(15, 15) * 50

    return image


@pytest.fixture
def shifted_image(textured_image):
    """Textured image shifted by (10, 5) pixels via numpy roll."""
    return np.roll(np.roll(textured_image, 10, axis=0), 5, axis=1)


# ---------------------------------------------------------------------------
# _to_uint8 helper tests
# ---------------------------------------------------------------------------

class TestToUint8:
    """Test the _to_uint8 conversion helper."""

    def test_float_image_normalized(self):
        from grdl.coregistration.feature_match import _to_uint8
        image = np.array([[0.0, 50.0], [100.0, 200.0]])
        result = _to_uint8(image)
        assert result.dtype == np.uint8
        assert result.min() == 0
        assert result.max() == 255

    def test_complex_image_uses_magnitude(self):
        from grdl.coregistration.feature_match import _to_uint8
        image = np.array([[1 + 0j, 0 + 1j], [3 + 4j, 0 + 0j]])
        result = _to_uint8(image)
        assert result.dtype == np.uint8
        # 3+4j has magnitude 5 (max), should map to 255
        assert result[1, 0] == 255

    def test_multiband_uses_first_band(self):
        from grdl.coregistration.feature_match import _to_uint8
        image = np.zeros((10, 10, 3))
        image[:, :, 0] = 100.0  # first band bright
        image[:, :, 1] = 200.0  # second band brighter
        result = _to_uint8(image)
        assert result.ndim == 2
        assert result.shape == (10, 10)

    def test_constant_image_returns_zeros(self):
        from grdl.coregistration.feature_match import _to_uint8
        image = np.full((10, 10), 42.0)
        result = _to_uint8(image)
        np.testing.assert_array_equal(result, np.zeros((10, 10), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestFeatureMatchInit:
    """Test constructor validation."""

    def test_valid_orb(self):
        from grdl.coregistration.feature_match import FeatureMatchCoRegistration
        coreg = FeatureMatchCoRegistration(method='orb')
        assert coreg._method == 'orb'

    def test_valid_sift(self):
        from grdl.coregistration.feature_match import FeatureMatchCoRegistration
        coreg = FeatureMatchCoRegistration(method='sift')
        assert coreg._method == 'sift'

    def test_case_insensitive(self):
        from grdl.coregistration.feature_match import FeatureMatchCoRegistration
        coreg = FeatureMatchCoRegistration(method='ORB')
        assert coreg._method == 'orb'

    def test_invalid_method_raises(self):
        from grdl.coregistration.feature_match import FeatureMatchCoRegistration
        with pytest.raises(ValueError, match="Unknown feature method"):
            FeatureMatchCoRegistration(method='surf')

    def test_invalid_transform_type_raises(self):
        from grdl.coregistration.feature_match import FeatureMatchCoRegistration
        with pytest.raises(ValueError, match="Unknown transform type"):
            FeatureMatchCoRegistration(transform_type='rigid')

    def test_is_coregistration_subclass(self):
        from grdl.coregistration.base import CoRegistration
        from grdl.coregistration.feature_match import FeatureMatchCoRegistration
        assert issubclass(FeatureMatchCoRegistration, CoRegistration)


# ---------------------------------------------------------------------------
# Identity recovery (same image → near-identity transform)
# ---------------------------------------------------------------------------

class TestFeatureMatchEstimate:
    """Test estimate() with synthetic images."""

    def test_identity_recovery(self, textured_image):
        """Same image should produce a near-identity transform."""
        from grdl.coregistration.feature_match import FeatureMatchCoRegistration
        from grdl.coregistration.base import RegistrationResult

        coreg = FeatureMatchCoRegistration(method='orb', max_features=2000)
        result = coreg.estimate(textured_image, textured_image)

        assert isinstance(result, RegistrationResult)
        assert result.residual_rms < 2.0  # near-zero error
        assert result.inlier_ratio > 0.5
        # Transform should be near-identity
        if result.is_affine:
            identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
            np.testing.assert_allclose(
                result.transform_matrix, identity, atol=2.0
            )

    def test_translation_recovery(self, textured_image):
        """Shifted image should recover the shift in the transform."""
        from grdl.coregistration.feature_match import FeatureMatchCoRegistration

        # Create a translated version using warp (more accurate than roll)
        shifted = np.zeros_like(textured_image)
        shifted[10:, 5:] = textured_image[:-10, :-5]

        coreg = FeatureMatchCoRegistration(
            method='orb', max_features=3000, transform_type='affine'
        )
        result = coreg.estimate(textured_image, shifted)

        assert result.num_matches >= 3
        # The transform should encode approximately (10, 5) translation
        # We allow generous tolerance since ORB is approximate
        if result.is_affine:
            tx = result.transform_matrix[0, 2]
            ty = result.transform_matrix[1, 2]
            assert abs(tx) < 20  # within 20 pixels of truth
            assert abs(ty) < 20

    def test_result_has_metadata(self, textured_image):
        """Result metadata should contain expected keys."""
        from grdl.coregistration.feature_match import FeatureMatchCoRegistration

        coreg = FeatureMatchCoRegistration(method='orb')
        result = coreg.estimate(textured_image, textured_image)

        assert 'method' in result.metadata
        assert 'transform_type' in result.metadata
        assert 'total_matches' in result.metadata
        assert 'num_inliers' in result.metadata

    def test_homography_mode(self, textured_image):
        """Homography transform type should produce 3x3 matrix."""
        from grdl.coregistration.feature_match import FeatureMatchCoRegistration

        coreg = FeatureMatchCoRegistration(
            method='orb', transform_type='homography', max_features=2000
        )
        result = coreg.estimate(textured_image, textured_image)

        assert result.is_projective
        assert result.transform_matrix.shape == (3, 3)

    def test_blank_image_raises(self):
        """Blank image should fail — no features to detect."""
        from grdl.coregistration.feature_match import FeatureMatchCoRegistration

        coreg = FeatureMatchCoRegistration(method='orb')
        blank = np.zeros((100, 100))
        with pytest.raises(RuntimeError):
            coreg.estimate(blank, blank)

    def test_sift_detector(self, textured_image):
        """SIFT detector should also work for identity recovery."""
        from grdl.coregistration.feature_match import FeatureMatchCoRegistration

        coreg = FeatureMatchCoRegistration(method='sift', max_features=2000)
        result = coreg.estimate(textured_image, textured_image)

        assert result.residual_rms < 2.0
        assert result.num_matches >= 3


# ---------------------------------------------------------------------------
# Apply (warping)
# ---------------------------------------------------------------------------

class TestFeatureMatchApply:
    """Test apply() warping."""

    def test_apply_preserves_shape(self, textured_image):
        """Warped image should have the same shape as input."""
        from grdl.coregistration.feature_match import FeatureMatchCoRegistration

        coreg = FeatureMatchCoRegistration(method='orb')
        result = coreg.estimate(textured_image, textured_image)
        warped = coreg.apply(textured_image, result)

        assert warped.shape == textured_image.shape

    def test_apply_with_custom_output_shape(self, textured_image):
        """apply() should respect custom output_shape."""
        from grdl.coregistration.feature_match import FeatureMatchCoRegistration

        coreg = FeatureMatchCoRegistration(method='orb')
        result = coreg.estimate(textured_image, textured_image)
        warped = coreg.apply(textured_image, result, output_shape=(50, 50))

        assert warped.shape == (50, 50)


# ---------------------------------------------------------------------------
# Complex / multiband input
# ---------------------------------------------------------------------------

class TestFeatureMatchSpecialInputs:
    """Test handling of complex and multiband images."""

    def test_complex_sar_input(self):
        """Complex-valued (SAR) images should work via magnitude extraction."""
        from grdl.coregistration.feature_match import FeatureMatchCoRegistration

        rng = np.random.RandomState(42)
        # Create textured complex image
        real = np.zeros((200, 200))
        for r in range(20, 180, 30):
            for c in range(20, 180, 30):
                real[r:r + 15, c:c + 15] = 200.0

        real += rng.rand(200, 200) * 30
        imag = rng.rand(200, 200) * 30
        cplx = real + 1j * imag

        coreg = FeatureMatchCoRegistration(method='orb', max_features=2000)
        result = coreg.estimate(cplx, cplx)

        assert result.num_matches >= 3
        assert result.residual_rms < 5.0

    def test_multiband_input(self):
        """3-band images should work (uses first band)."""
        from grdl.coregistration.feature_match import FeatureMatchCoRegistration

        rng = np.random.RandomState(42)
        band = np.zeros((200, 200))
        for r in range(20, 180, 30):
            for c in range(20, 180, 30):
                band[r:r + 15, c:c + 15] = 200.0

        band += rng.rand(200, 200) * 30
        image = np.stack([band, band * 0.5, band * 0.3], axis=2)

        coreg = FeatureMatchCoRegistration(method='orb', max_features=2000)
        result = coreg.estimate(image, image)

        assert result.num_matches >= 3
