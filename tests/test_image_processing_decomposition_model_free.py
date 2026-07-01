# -*- coding: utf-8 -*-
"""
Tests for Model-Free Decompositions (MF3CF / MF4CF).

Tests the ModelFree3C and ModelFree4C decompositions against known
physical properties and internal consistency.

Author
------
Jason Fritz, PhD
43161141+stryder-vtx@users.noreply.github.com
"""

import numpy as np
import pytest

from grdl.image_processing.decomposition import (
    ModelFree3C,
    ModelFree4C,
    CoherencyMatrix,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def quad_pol_random():
    """Random quad-pol data (100x100)."""
    rng = np.random.default_rng(42)
    N = 100
    shh = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))).astype(np.complex64)
    shv = (0.3 * (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))).astype(np.complex64)
    svh = shv.copy()
    svv = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))).astype(np.complex64)
    return shh, shv, svh, svv


@pytest.fixture
def t3_precomputed(quad_pol_random):
    """Pre-computed T3 matrix from random data."""
    shh, shv, svh, svv = quad_pol_random
    channels = np.stack([shh, shv, svh, svv], axis=0)
    return CoherencyMatrix(window_size=3).compute(channels)


@pytest.fixture
def mf3():
    return ModelFree3C(window_size=3)


@pytest.fixture
def mf4():
    return ModelFree4C(window_size=3)


# ======================================================================
# MF3CF Tests
# ======================================================================

class TestModelFree3C:
    """Tests for MF3CF decomposition."""

    def test_component_names(self, mf3):
        assert mf3.component_names == (
            'surface', 'double_bounce', 'volume', 'span', 'theta_fp'
        )

    def test_repr(self, mf3):
        assert 'ModelFree3C' in repr(mf3)
        assert 'window_size=3' in repr(mf3)

    def test_decompose_keys(self, mf3, quad_pol_random):
        result = mf3.decompose(*quad_pol_random)
        expected = {'surface', 'double_bounce', 'volume', 'span', 'theta_fp'}
        assert set(result.keys()) == expected

    def test_decompose_shapes(self, mf3, quad_pol_random):
        shh = quad_pol_random[0]
        result = mf3.decompose(*quad_pol_random)
        for key in result:
            assert result[key].shape == shh.shape

    def test_decompose_real(self, mf3, quad_pol_random):
        result = mf3.decompose(*quad_pol_random)
        for key in result:
            assert np.isrealobj(result[key])

    def test_power_conservation(self, mf3, quad_pol_random):
        """Ps + Pd + Pv == Span (power conservation)."""
        result = mf3.decompose(*quad_pol_random)
        total = result['surface'] + result['double_bounce'] + result['volume']
        mask = np.isfinite(total)
        np.testing.assert_allclose(
            total[mask], result['span'][mask], atol=1e-5
        )

    def test_powers_nonnegative(self, mf3, quad_pol_random):
        result = mf3.decompose(*quad_pol_random)
        for key in ('surface', 'double_bounce', 'volume'):
            arr = result[key]
            assert np.all(arr[np.isfinite(arr)] >= 0.0)

    def test_theta_fp_range(self, mf3, quad_pol_random):
        """θ_FP should be in [-45, 45] degrees."""
        result = mf3.decompose(*quad_pol_random)
        theta = result['theta_fp']
        valid = theta[np.isfinite(theta)]
        assert np.all(valid >= -45.0 - 1e-10)
        assert np.all(valid <= 45.0 + 1e-10)

    def test_decompose_from_t3(self, mf3, quad_pol_random, t3_precomputed):
        """decompose_from_t3() matches decompose()."""
        result_channels = mf3.decompose(*quad_pol_random)
        result_t3 = mf3.decompose_from_t3(t3_precomputed)
        for key in ('surface', 'double_bounce', 'volume', 'span'):
            np.testing.assert_allclose(
                result_channels[key], result_t3[key], atol=1e-10
            )

    def test_decompose_from_t3_validation(self, mf3):
        """Invalid T3 shape raises ValueError."""
        with pytest.raises(ValueError, match="Expected t3 shape"):
            mf3.decompose_from_t3(np.zeros((4, 4, 10, 10)))

    def test_to_rgb_shape(self, mf3, quad_pol_random):
        result = mf3.decompose(*quad_pol_random)
        rgb, meta = mf3.to_rgb(result)
        assert rgb.shape == (3, 100, 100)
        assert rgb.dtype == np.float32

    def test_to_rgb_range(self, mf3, quad_pol_random):
        result = mf3.decompose(*quad_pol_random)
        rgb, _ = mf3.to_rgb(result)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_to_rgb_missing_key(self, mf3):
        with pytest.raises(ValueError, match="Missing component keys"):
            mf3.to_rgb({'surface': np.zeros((10, 10))})

    def test_surface_dominant_negative_theta(self, mf3):
        """Surface-dominant pixel should have θ_FP < 0."""
        # Create a pixel with strong surface scattering:
        # T11 >> T22 + T33 → θ_FP < 0
        N = 20
        rng = np.random.default_rng(99)
        # Surface: strong HH+VV, weak HV → T11 dominant
        shh = (5.0 + 0.1 * rng.standard_normal((N, N)) +
               1j * 0.1 * rng.standard_normal((N, N))).astype(np.complex64)
        svv = shh.copy()  # Similar to shh for surface
        shv = (0.01 * (rng.standard_normal((N, N)) +
               1j * rng.standard_normal((N, N)))).astype(np.complex64)
        svh = shv.copy()
        result = mf3.decompose(shh, shv, svh, svv)
        # Central pixels should have θ_FP < 0
        center = slice(5, 15)
        mean_theta = np.nanmean(result['theta_fp'][center, center])
        assert mean_theta < 0, f"Expected θ_FP < 0 for surface, got {mean_theta:.1f}°"


# ======================================================================
# MF4CF Tests
# ======================================================================

class TestModelFree4C:
    """Tests for MF4CF decomposition."""

    def test_component_names(self, mf4):
        assert mf4.component_names == (
            'surface', 'double_bounce', 'volume', 'helix',
            'span', 'theta_fp', 'tau_fp'
        )

    def test_repr(self, mf4):
        assert 'ModelFree4C' in repr(mf4)

    def test_decompose_keys(self, mf4, quad_pol_random):
        result = mf4.decompose(*quad_pol_random)
        expected = {
            'surface', 'double_bounce', 'volume', 'helix',
            'span', 'theta_fp', 'tau_fp'
        }
        assert set(result.keys()) == expected

    def test_power_conservation(self, mf4, quad_pol_random):
        """Ps + Pd + Pv + Pc == Span."""
        result = mf4.decompose(*quad_pol_random)
        total = (result['surface'] + result['double_bounce']
                 + result['volume'] + result['helix'])
        mask = np.isfinite(total)
        np.testing.assert_allclose(
            total[mask], result['span'][mask], atol=1e-5
        )

    def test_powers_nonnegative(self, mf4, quad_pol_random):
        result = mf4.decompose(*quad_pol_random)
        for key in ('surface', 'double_bounce', 'volume', 'helix'):
            arr = result[key]
            assert np.all(arr[np.isfinite(arr)] >= 0.0)

    def test_tau_fp_range(self, mf4, quad_pol_random):
        """τ_FP should be in [-45, 45] degrees."""
        result = mf4.decompose(*quad_pol_random)
        tau = result['tau_fp']
        valid = tau[np.isfinite(tau)]
        assert np.all(valid >= -45.0 - 1e-10)
        assert np.all(valid <= 45.0 + 1e-10)

    def test_helix_less_than_span(self, mf4, quad_pol_random):
        """Helix power should be less than total span."""
        result = mf4.decompose(*quad_pol_random)
        mask = np.isfinite(result['helix'])
        assert np.all(result['helix'][mask] <= result['span'][mask] + 1e-10)

    def test_decompose_from_t3(self, mf4, quad_pol_random, t3_precomputed):
        """decompose_from_t3() matches decompose()."""
        result_channels = mf4.decompose(*quad_pol_random)
        result_t3 = mf4.decompose_from_t3(t3_precomputed)
        for key in ('surface', 'double_bounce', 'volume', 'helix', 'span'):
            np.testing.assert_allclose(
                result_channels[key], result_t3[key], atol=1e-10
            )

    def test_mf4c_helix_zero_for_symmetric(self):
        """Reciprocal data with no asymmetry → near-zero helix."""
        N = 30
        rng = np.random.default_rng(77)
        # Symmetric scattering: svh = shv, real-valued → Im(T23) ≈ 0
        shh = (rng.standard_normal((N, N))).astype(np.complex64)
        svv = (rng.standard_normal((N, N))).astype(np.complex64)
        shv = np.zeros((N, N), dtype=np.complex64)
        svh = shv.copy()
        mf4 = ModelFree4C(window_size=3)
        result = mf4.decompose(shh, shv, svh, svv)
        # With zero cross-pol, T23 should be ~0 → τ_FP ≈ 0 → Pc ≈ 0
        mask = np.isfinite(result['helix'])
        mean_helix = np.nanmean(result['helix'])
        mean_span = np.nanmean(result['span'])
        ratio = mean_helix / mean_span if mean_span > 0 else 0
        assert ratio < 0.05, f"Expected near-zero helix ratio, got {ratio:.3f}"

    def test_to_rgb_shape(self, mf4, quad_pol_random):
        result = mf4.decompose(*quad_pol_random)
        rgb, meta = mf4.to_rgb(result)
        assert rgb.shape == (3, 100, 100)
        assert rgb.dtype == np.float32

    def test_decompose_from_t3_validation(self, mf4):
        with pytest.raises(ValueError, match="Expected t3 shape"):
            mf4.decompose_from_t3(np.zeros((2, 2, 10, 10)))


# ======================================================================
# Consistency: MF3CF vs MF4CF
# ======================================================================

class TestMF3CvsMF4C:
    """Cross-checks between 3- and 4-component decompositions."""

    def test_theta_fp_matches(self, mf3, mf4, quad_pol_random):
        """Both should compute the same θ_FP."""
        r3 = mf3.decompose(*quad_pol_random)
        r4 = mf4.decompose(*quad_pol_random)
        np.testing.assert_allclose(r3['theta_fp'], r4['theta_fp'], atol=1e-10)

    def test_span_matches(self, mf3, mf4, quad_pol_random):
        """Both should compute the same span."""
        r3 = mf3.decompose(*quad_pol_random)
        r4 = mf4.decompose(*quad_pol_random)
        np.testing.assert_allclose(r3['span'], r4['span'], atol=1e-10)

    def test_volume_matches(self, mf3, mf4, quad_pol_random):
        """Volume (depolarized) component should be the same."""
        r3 = mf3.decompose(*quad_pol_random)
        r4 = mf4.decompose(*quad_pol_random)
        np.testing.assert_allclose(r3['volume'], r4['volume'], atol=1e-10)

    def test_mf4c_helix_redistributes(self, mf3, mf4, quad_pol_random):
        """MF4CF helix comes from the polarized Ps+Pd budget."""
        r3 = mf3.decompose(*quad_pol_random)
        r4 = mf4.decompose(*quad_pol_random)
        # Ps3 + Pd3 == Ps4 + Pd4 + Pc4
        polarized_3 = r3['surface'] + r3['double_bounce']
        polarized_4 = r4['surface'] + r4['double_bounce'] + r4['helix']
        mask = np.isfinite(polarized_3) & np.isfinite(polarized_4)
        np.testing.assert_allclose(
            polarized_3[mask], polarized_4[mask], atol=1e-5
        )
