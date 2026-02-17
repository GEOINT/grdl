# -*- coding: utf-8 -*-
"""
Tests for dual-pol H/Alpha eigenvalue decomposition.

All tests use synthetic data — no real SAR imagery required.

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
2026-02-16

Modified
--------
2026-02-16
"""

# Third-party
import numpy as np
import pytest

# GRDL
from grdl.image_processing.decomposition.dual_pol_halpha import DualPolHAlpha


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def halpha():
    """Default DualPolHAlpha instance."""
    return DualPolHAlpha(window_size=3)


@pytest.fixture
def copol_only():
    """Co-pol dominant: strong VV, zero VH.

    A pure co-pol return with no cross-pol should produce low entropy
    and alpha near 0 (surface-like scattering).
    """
    rng = np.random.default_rng(42)
    shape = (64, 64)
    s_co = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    s_cross = np.zeros(shape, dtype=np.complex64)
    return s_co, s_cross


@pytest.fixture
def equal_power():
    """Equal power in both channels — maximally random.

    When co-pol and cross-pol have the same power and are uncorrelated,
    entropy should approach 1.0.
    """
    rng = np.random.default_rng(123)
    shape = (128, 128)
    s_co = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    s_cross = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    return s_co, s_cross


# ===================================================================
# Component names
# ===================================================================

class TestComponentNames:

    def test_names(self, halpha):
        assert halpha.component_names == ('entropy', 'alpha', 'anisotropy', 'span')

    def test_repr(self, halpha):
        assert 'DualPolHAlpha' in repr(halpha)
        assert 'window_size=3' in repr(halpha)


# ===================================================================
# Decompose — output structure
# ===================================================================

class TestDecomposeStructure:

    def test_output_keys(self, halpha, copol_only):
        s_co, s_cross = copol_only
        result = halpha.decompose(s_co, s_cross)
        assert set(result.keys()) == {'entropy', 'alpha', 'anisotropy', 'span'}

    def test_output_shapes(self, halpha, copol_only):
        s_co, s_cross = copol_only
        result = halpha.decompose(s_co, s_cross)
        for key in result:
            assert result[key].shape == s_co.shape

    def test_output_real(self, halpha, copol_only):
        """All outputs must be real-valued."""
        s_co, s_cross = copol_only
        result = halpha.decompose(s_co, s_cross)
        for key in result:
            assert not np.iscomplexobj(result[key])


# ===================================================================
# Decompose — physical correctness
# ===================================================================

class TestDecomposePhysics:

    def test_copol_only_low_entropy(self, halpha, copol_only):
        """Pure co-pol (no cross-pol) should have low entropy."""
        s_co, s_cross = copol_only
        result = halpha.decompose(s_co, s_cross)
        # With zero cross-pol, one eigenvalue dominates → H ≈ 0
        mean_h = np.mean(result['entropy'])
        assert mean_h < 0.1

    def test_copol_only_low_alpha(self, halpha, copol_only):
        """Pure co-pol (no cross-pol) should have alpha near 0."""
        s_co, s_cross = copol_only
        result = halpha.decompose(s_co, s_cross)
        mean_alpha = np.mean(result['alpha'])
        # Surface scattering → alpha close to 0 degrees
        assert mean_alpha < 10.0

    def test_equal_power_high_entropy(self):
        """Equal uncorrelated channels should have high entropy."""
        rng = np.random.default_rng(99)
        shape = (256, 256)
        s_co = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
        s_cross = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
        halpha = DualPolHAlpha(window_size=7)
        result = halpha.decompose(s_co, s_cross)
        mean_h = np.mean(result['entropy'])
        # Uncorrelated equal-power → H close to 1
        assert mean_h > 0.7

    def test_entropy_bounded(self, halpha, equal_power):
        """Entropy must be in [0, 1]."""
        s_co, s_cross = equal_power
        result = halpha.decompose(s_co, s_cross)
        assert np.all(result['entropy'] >= 0.0)
        assert np.all(result['entropy'] <= 1.0)

    def test_alpha_bounded(self, halpha, equal_power):
        """Alpha must be in [0, 90] degrees."""
        s_co, s_cross = equal_power
        result = halpha.decompose(s_co, s_cross)
        assert np.all(result['alpha'] >= 0.0)
        assert np.all(result['alpha'] <= 90.0)

    def test_anisotropy_bounded(self, halpha, equal_power):
        """Anisotropy must be in [0, 1]."""
        s_co, s_cross = equal_power
        result = halpha.decompose(s_co, s_cross)
        assert np.all(result['anisotropy'] >= 0.0)
        assert np.all(result['anisotropy'] <= 1.0)

    def test_span_nonnegative(self, halpha, equal_power):
        """Span must be non-negative."""
        s_co, s_cross = equal_power
        result = halpha.decompose(s_co, s_cross)
        assert np.all(result['span'] >= 0.0)

    def test_span_equals_total_power(self, halpha, copol_only):
        """Span should approximate the spatially averaged total power."""
        s_co, s_cross = copol_only
        result = halpha.decompose(s_co, s_cross)
        from scipy.ndimage import uniform_filter
        expected = uniform_filter(np.abs(s_co) ** 2, size=3)
        np.testing.assert_allclose(result['span'], expected, rtol=1e-5)

    def test_copol_only_high_anisotropy(self, halpha, copol_only):
        """Pure co-pol should have high anisotropy (single mechanism)."""
        s_co, s_cross = copol_only
        result = halpha.decompose(s_co, s_cross)
        mean_a = np.mean(result['anisotropy'])
        assert mean_a > 0.9


# ===================================================================
# Window size
# ===================================================================

class TestWindowSize:

    def test_default_window(self):
        h = DualPolHAlpha()
        assert h.window_size == 7

    def test_custom_window(self):
        h = DualPolHAlpha(window_size=11)
        assert h.window_size == 11

    def test_larger_window_smoother(self, copol_only):
        """Larger window should produce smoother output."""
        s_co, s_cross = copol_only
        h3 = DualPolHAlpha(window_size=3)
        h11 = DualPolHAlpha(window_size=11)
        r3 = h3.decompose(s_co, s_cross)
        r11 = h11.decompose(s_co, s_cross)
        # Standard deviation of span should be lower with larger window
        assert np.std(r11['span']) < np.std(r3['span'])


# ===================================================================
# Validation
# ===================================================================

class TestValidation:

    def test_not_complex_raises(self, halpha):
        real = np.ones((10, 10), dtype=np.float32)
        with pytest.raises(TypeError, match="complex"):
            halpha.decompose(real, real)

    def test_not_ndarray_raises(self, halpha):
        with pytest.raises(TypeError, match="ndarray"):
            halpha.decompose([[1+0j]], np.ones((1, 1), dtype=np.complex64))

    def test_not_2d_raises(self, halpha):
        arr3d = np.ones((2, 3, 4), dtype=np.complex64)
        arr2d = np.ones((2, 3), dtype=np.complex64)
        with pytest.raises(ValueError, match="2D"):
            halpha.decompose(arr3d, arr2d)

    def test_shape_mismatch_raises(self, halpha):
        a = np.ones((10, 10), dtype=np.complex64)
        b = np.ones((10, 20), dtype=np.complex64)
        with pytest.raises(ValueError, match="Shape mismatch"):
            halpha.decompose(a, b)


# ===================================================================
# to_rgb
# ===================================================================

class TestToRgb:

    def test_output_shape(self, halpha, copol_only):
        s_co, s_cross = copol_only
        result = halpha.decompose(s_co, s_cross)
        rgb = halpha.to_rgb(result)
        assert rgb.shape == (s_co.shape[0], s_co.shape[1], 3)

    def test_output_dtype(self, halpha, copol_only):
        s_co, s_cross = copol_only
        result = halpha.decompose(s_co, s_cross)
        rgb = halpha.to_rgb(result)
        assert rgb.dtype == np.float32

    def test_output_range(self, halpha, copol_only):
        s_co, s_cross = copol_only
        result = halpha.decompose(s_co, s_cross)
        rgb = halpha.to_rgb(result)
        assert np.all(rgb >= 0.0)
        assert np.all(rgb <= 1.0)

    def test_missing_key_raises(self, halpha):
        with pytest.raises(ValueError, match="Missing"):
            halpha.to_rgb({'entropy': np.zeros((3, 3))})
