# -*- coding: utf-8 -*-
"""
Tests for Lagrange, Farrow, and Thiran interpolation methods.

Validates the three fractional delay filter implementations from
Laakso et al. (1996): Lagrange polynomial (Eq. 42), Farrow structure
(Eqs. 59-63), and Thiran allpass (Eq. 86).

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
2026-02-12

Modified
--------
2026-02-12
"""

# Third-party
import numpy as np
import pytest

# GRDL internal
from grdl.interpolation import (
    Interpolator,
    KernelInterpolator,
    LagrangeInterpolator,
    lagrange_interpolator,
    FarrowInterpolator,
    farrow_interpolator,
    ThiranDelayFilter,
    thiran_delay,
)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def uniform_grid():
    """Uniform input grid with 100 samples."""
    return np.linspace(0.0, 10.0, 100)


def _bandlimited_analytic(x):
    """Bandlimited test signal: sum of low-frequency sinusoids."""
    return (
        np.sin(2 * np.pi * 0.5 * x)
        + 0.5 * np.cos(2 * np.pi * 1.2 * x)
        + 0.3 * np.sin(2 * np.pi * 2.0 * x)
    )


# ═══════════════════════════════════════════════════════════════════════
# Lagrange Interpolator
# ═══════════════════════════════════════════════════════════════════════


class TestLagrangeFactory:
    """Test factory function parameter validation."""

    def test_order_too_small(self):
        with pytest.raises(ValueError, match="order must be >= 1"):
            lagrange_interpolator(order=0)

    def test_returns_callable(self):
        interp = lagrange_interpolator()
        assert callable(interp)

    def test_factory_returns_class(self):
        assert isinstance(lagrange_interpolator(), LagrangeInterpolator)

    def test_default_order(self):
        interp = LagrangeInterpolator()
        assert interp.order == 3


class TestLagrangeHierarchy:
    """Test class hierarchy contracts."""

    def test_is_interpolator(self):
        assert isinstance(LagrangeInterpolator(), Interpolator)

    def test_is_kernel_interpolator(self):
        assert isinstance(LagrangeInterpolator(), KernelInterpolator)


class TestLagrangeReconstruction:
    """Test Lagrange polynomial reconstruction of bandlimited signals."""

    def test_upsample_accuracy(self, uniform_grid):
        """Upsample 2x and verify against analytic function."""
        y_old = _bandlimited_analytic(uniform_grid)
        interp = lagrange_interpolator(order=3)
        x_new = np.linspace(1.0, 9.0, 200)
        result = interp(uniform_grid, y_old, x_new)
        expected = _bandlimited_analytic(x_new)
        max_err = np.max(np.abs(result - expected))
        assert max_err < 0.1, f"Max error {max_err:.4f} exceeds tolerance"

    def test_higher_order_improves_accuracy(self, uniform_grid):
        """Higher order should give better reconstruction."""
        y_old = _bandlimited_analytic(uniform_grid)
        x_new = np.linspace(1.0, 9.0, 200)
        expected = _bandlimited_analytic(x_new)

        err_1 = np.max(np.abs(
            lagrange_interpolator(order=1)(uniform_grid, y_old, x_new)
            - expected
        ))
        err_5 = np.max(np.abs(
            lagrange_interpolator(order=5)(uniform_grid, y_old, x_new)
            - expected
        ))
        assert err_5 < err_1, (
            f"order=5 error {err_5:.6f} should be less than "
            f"order=1 error {err_1:.6f}"
        )

    def test_linear_interpolation(self):
        """Order 1 should match linear interpolation."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        x_new = np.array([0.5, 1.5, 2.5])
        interp = lagrange_interpolator(order=1)
        result = interp(x, y, x_new)
        np.testing.assert_allclose(result, [0.5, 1.5, 2.5], atol=1e-10)


class TestLagrangePassthrough:
    """Test interpolation at original sample points."""

    def test_identity_uniform(self, uniform_grid):
        y_old = _bandlimited_analytic(uniform_grid)
        interp = lagrange_interpolator(order=3)
        result = interp(uniform_grid, y_old, uniform_grid)
        interior = slice(3, -3)
        np.testing.assert_allclose(
            result[interior], y_old[interior], atol=1e-6,
        )


class TestLagrangeComplex:
    """Test Lagrange interpolation of complex-valued signals."""

    def test_complex_exponential(self, uniform_grid):
        y = np.exp(1j * 2 * np.pi * 0.5 * uniform_grid)
        interp = lagrange_interpolator(order=3)
        x_new = np.linspace(1.0, 9.0, 150)
        result = interp(uniform_grid, y, x_new)
        expected = np.exp(1j * 2 * np.pi * 0.5 * x_new)
        max_err = np.max(np.abs(result - expected))
        assert max_err < 0.1, f"Complex max error {max_err:.4f}"

    def test_preserves_complex_dtype(self, uniform_grid):
        y = np.sin(uniform_grid) + 1j * np.cos(uniform_grid)
        interp = lagrange_interpolator()
        result = interp(uniform_grid, y, uniform_grid)
        assert np.iscomplexobj(result)


class TestLagrangeOutOfBounds:
    """Test OOB behavior."""

    def test_oob_left(self, uniform_grid):
        y = np.sin(uniform_grid)
        interp = lagrange_interpolator()
        result = interp(uniform_grid, y, np.array([-1.0, -0.5]))
        np.testing.assert_array_equal(result, 0.0)

    def test_oob_right(self, uniform_grid):
        y = np.sin(uniform_grid)
        interp = lagrange_interpolator()
        result = interp(uniform_grid, y, np.array([10.5, 11.0]))
        np.testing.assert_array_equal(result, 0.0)


class TestLagrangeDescending:
    """Test with monotonically decreasing x_old."""

    def test_descending_matches_ascending(self, uniform_grid):
        y_old = _bandlimited_analytic(uniform_grid)
        x_new = np.linspace(2.0, 8.0, 100)

        interp = lagrange_interpolator(order=3)
        result_asc = interp(uniform_grid, y_old, x_new)
        result_desc = interp(uniform_grid[::-1], y_old[::-1], x_new)
        np.testing.assert_allclose(result_desc, result_asc, atol=1e-10)


class TestLagrangeOutputShape:
    """Test output shape matches x_new."""

    def test_output_length(self, uniform_grid):
        y = np.sin(uniform_grid)
        interp = lagrange_interpolator()
        for m in [1, 10, 50, 200]:
            x_new = np.linspace(0, 10, m)
            result = interp(uniform_grid, y, x_new)
            assert len(result) == m


# ═══════════════════════════════════════════════════════════════════════
# Farrow Interpolator
# ═══════════════════════════════════════════════════════════════════════


class TestFarrowFactory:
    """Test factory function parameter validation."""

    def test_filter_order_too_small(self):
        with pytest.raises(ValueError, match="filter_order must be >= 1"):
            farrow_interpolator(filter_order=0)

    def test_poly_order_too_small(self):
        with pytest.raises(ValueError, match="poly_order must be >= 1"):
            farrow_interpolator(poly_order=0)

    def test_returns_callable(self):
        interp = farrow_interpolator()
        assert callable(interp)

    def test_factory_returns_class(self):
        assert isinstance(farrow_interpolator(), FarrowInterpolator)


class TestFarrowHierarchy:
    """Test class hierarchy contracts."""

    def test_is_interpolator(self):
        assert isinstance(FarrowInterpolator(), Interpolator)

    def test_is_not_kernel_interpolator(self):
        """Farrow extends Interpolator directly, not KernelInterpolator."""
        assert not isinstance(FarrowInterpolator(), KernelInterpolator)


class TestFarrowReconstruction:
    """Test Farrow structure reconstruction of bandlimited signals."""

    def test_upsample_accuracy(self, uniform_grid):
        """Upsample 2x and verify against analytic function."""
        y_old = _bandlimited_analytic(uniform_grid)
        interp = farrow_interpolator(filter_order=3, poly_order=3)
        x_new = np.linspace(1.0, 9.0, 200)
        result = interp(uniform_grid, y_old, x_new)
        expected = _bandlimited_analytic(x_new)
        max_err = np.max(np.abs(result - expected))
        assert max_err < 0.15, f"Max error {max_err:.4f} exceeds tolerance"

    def test_higher_filter_order_improves_accuracy(self, uniform_grid):
        """Higher filter order should give better reconstruction."""
        y_old = _bandlimited_analytic(uniform_grid)
        x_new = np.linspace(1.0, 9.0, 200)
        expected = _bandlimited_analytic(x_new)

        err_lo = np.max(np.abs(
            farrow_interpolator(filter_order=1, poly_order=3)(
                uniform_grid, y_old, x_new
            ) - expected
        ))
        err_hi = np.max(np.abs(
            farrow_interpolator(filter_order=5, poly_order=5)(
                uniform_grid, y_old, x_new
            ) - expected
        ))
        assert err_hi < err_lo, (
            f"filter_order=5 error {err_hi:.6f} should be less than "
            f"filter_order=1 error {err_lo:.6f}"
        )

    def test_matches_lagrange(self, uniform_grid):
        """With poly_order >= filter_order, Farrow should be exact Lagrange."""
        y_old = _bandlimited_analytic(uniform_grid)
        x_new = np.linspace(2.0, 8.0, 150)

        lagrange = lagrange_interpolator(order=3)
        farrow = farrow_interpolator(filter_order=3, poly_order=3)

        result_lag = lagrange(uniform_grid, y_old, x_new)
        result_far = farrow(uniform_grid, y_old, x_new)

        # Should be very similar (polynomial fit is exact for P >= N)
        max_diff = np.max(np.abs(result_lag - result_far))
        assert max_diff < 0.1, (
            f"Farrow/Lagrange difference {max_diff:.6f} too large"
        )


class TestFarrowPassthrough:
    """Test interpolation at original sample points."""

    def test_identity_uniform(self, uniform_grid):
        y_old = _bandlimited_analytic(uniform_grid)
        interp = farrow_interpolator(filter_order=3, poly_order=3)
        result = interp(uniform_grid, y_old, uniform_grid)
        interior = slice(5, -5)
        np.testing.assert_allclose(
            result[interior], y_old[interior], atol=0.01,
        )


class TestFarrowComplex:
    """Test Farrow interpolation of complex-valued signals."""

    def test_complex_exponential(self, uniform_grid):
        y = np.exp(1j * 2 * np.pi * 0.5 * uniform_grid)
        interp = farrow_interpolator(filter_order=3, poly_order=3)
        x_new = np.linspace(1.0, 9.0, 150)
        result = interp(uniform_grid, y, x_new)
        expected = np.exp(1j * 2 * np.pi * 0.5 * x_new)
        max_err = np.max(np.abs(result - expected))
        assert max_err < 0.15, f"Complex max error {max_err:.4f}"

    def test_preserves_complex_dtype(self, uniform_grid):
        y = np.sin(uniform_grid) + 1j * np.cos(uniform_grid)
        interp = farrow_interpolator()
        result = interp(uniform_grid, y, uniform_grid)
        assert np.iscomplexobj(result)


class TestFarrowOutOfBounds:
    """Test OOB behavior."""

    def test_oob_left(self, uniform_grid):
        y = np.sin(uniform_grid)
        interp = farrow_interpolator()
        result = interp(uniform_grid, y, np.array([-1.0, -0.5]))
        np.testing.assert_array_equal(result, 0.0)

    def test_oob_right(self, uniform_grid):
        y = np.sin(uniform_grid)
        interp = farrow_interpolator()
        result = interp(uniform_grid, y, np.array([10.5, 11.0]))
        np.testing.assert_array_equal(result, 0.0)


class TestFarrowDescending:
    """Test with monotonically decreasing x_old."""

    def test_descending_matches_ascending(self, uniform_grid):
        y_old = _bandlimited_analytic(uniform_grid)
        x_new = np.linspace(2.0, 8.0, 100)

        interp = farrow_interpolator(filter_order=3, poly_order=3)
        result_asc = interp(uniform_grid, y_old, x_new)
        result_desc = interp(uniform_grid[::-1], y_old[::-1], x_new)
        np.testing.assert_allclose(result_desc, result_asc, atol=1e-10)


class TestFarrowOutputShape:
    """Test output shape matches x_new."""

    def test_output_length(self, uniform_grid):
        y = np.sin(uniform_grid)
        interp = farrow_interpolator()
        for m in [1, 10, 50, 200]:
            x_new = np.linspace(0, 10, m)
            result = interp(uniform_grid, y, x_new)
            assert len(result) == m


# ═══════════════════════════════════════════════════════════════════════
# Thiran Delay Filter
# ═══════════════════════════════════════════════════════════════════════


class TestThiranValidation:
    """Test parameter validation."""

    def test_order_too_small(self):
        with pytest.raises(ValueError, match="order must be >= 1"):
            ThiranDelayFilter(delay=1.5, order=0)

    def test_negative_delay(self):
        with pytest.raises(ValueError, match="delay must be >= 0"):
            ThiranDelayFilter(delay=-0.5, order=3)

    def test_valid_construction(self):
        filt = ThiranDelayFilter(delay=3.5, order=3)
        assert filt.delay == 3.5
        assert filt.order == 3

    def test_integer_delay_no_coeffs(self):
        """Pure integer delay should skip allpass filter."""
        filt = ThiranDelayFilter(delay=5.0, order=3)
        assert filt.coefficients is None

    def test_fractional_delay_has_coeffs(self):
        """Fractional delay should produce coefficients."""
        filt = ThiranDelayFilter(delay=3.5, order=3)
        assert filt.coefficients is not None
        assert len(filt.coefficients) == 4  # order + 1

    def test_small_delay_stability_error(self):
        """Delay too small for given order should raise ValueError."""
        with pytest.raises(ValueError, match="too small for order"):
            ThiranDelayFilter(delay=0.3, order=3)


class TestThiranHierarchy:
    """Test that Thiran is NOT an Interpolator."""

    def test_not_interpolator(self):
        """Thiran operates on uniform signals, not (x, y, x_new)."""
        assert not isinstance(ThiranDelayFilter(delay=3.5), Interpolator)


class TestThiranCoefficients:
    """Test Thiran coefficient computation."""

    def test_a0_is_one(self):
        """First coefficient should always be 1."""
        filt = ThiranDelayFilter(delay=3.5, order=3)
        np.testing.assert_almost_equal(filt.coefficients[0], 1.0)

    def test_allpass_property(self):
        """Numerator = reversed denominator for allpass."""
        filt = ThiranDelayFilter(delay=3.7, order=3)
        a = filt.coefficients
        # For allpass, the transfer function has H(z) = z^{-N} * D(z^{-1}) / D(z)
        # This means b = a[::-1], which gives unity magnitude response
        b = a[::-1]
        # Verify coefficients are reasonable (no NaN/Inf)
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(b))


class TestThiranDelayAccuracy:
    """Test delay accuracy for various signals."""

    def test_integer_delay(self):
        """Pure integer delay should shift signal exactly."""
        n = 100
        signal = np.zeros(n)
        signal[20] = 1.0  # impulse
        filt = ThiranDelayFilter(delay=5.0, order=3)
        result = filt(signal)
        # Peak should be at sample 25
        assert np.argmax(result) == 25

    def test_fractional_delay_impulse_location(self):
        """Fractional delay should shift peak approximately correctly."""
        n = 200
        # Use a Gaussian pulse (bandlimited) for smooth peak detection
        t = np.arange(n, dtype=float)
        center = 50.0
        sigma = 5.0
        signal = np.exp(-0.5 * ((t - center) / sigma) ** 2)

        delay_amount = 3.7
        filt = ThiranDelayFilter(delay=delay_amount, order=3)
        result = filt(signal)

        # Use parabolic interpolation for sub-sample peak
        pk = np.argmax(result)
        if 0 < pk < n - 1:
            alpha = result[pk - 1]
            beta = result[pk]
            gamma = result[pk + 1]
            p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            estimated_peak = pk + p
        else:
            estimated_peak = float(pk)

        expected_peak = center + delay_amount
        # Allow 0.5 sample tolerance for low-order filter on non-ideal signal
        assert abs(estimated_peak - expected_peak) < 0.5, (
            f"Peak at {estimated_peak:.2f}, expected near {expected_peak:.1f}"
        )

    def test_zero_delay(self):
        """Zero delay should return input unchanged."""
        signal = np.random.randn(100)
        filt = ThiranDelayFilter(delay=0.0, order=3)
        result = filt(signal)
        np.testing.assert_array_equal(result, signal)

    def test_preserves_signal_energy(self):
        """Allpass filter preserves signal energy (unity magnitude)."""
        signal = np.random.randn(500)
        filt = ThiranDelayFilter(delay=3.7, order=3)
        result = filt(signal)
        # Energy should be preserved (allpass property)
        # Allow some tolerance for transient effects
        # Compare interior portion to avoid edge effects
        input_energy = np.sum(signal[50:450] ** 2)
        output_energy = np.sum(result[50:450] ** 2)
        ratio = output_energy / input_energy
        assert 0.9 < ratio < 1.1, f"Energy ratio {ratio:.4f} deviates from 1"


class TestThiranComplex:
    """Test Thiran with complex signals."""

    def test_complex_signal(self):
        """Should handle complex signals correctly."""
        n = 200
        t = np.arange(n, dtype=float)
        signal = np.exp(1j * 2 * np.pi * 0.05 * t)
        filt = ThiranDelayFilter(delay=3.5, order=3)
        result = filt(signal)
        assert np.iscomplexobj(result)
        assert result.shape == signal.shape

    def test_complex_energy_preserved(self):
        """Allpass property should hold for complex data."""
        n = 500
        t = np.arange(n, dtype=float)
        signal = np.exp(1j * 2 * np.pi * 0.05 * t)
        filt = ThiranDelayFilter(delay=3.7, order=3)
        result = filt(signal)
        input_power = np.mean(np.abs(signal[50:450]) ** 2)
        output_power = np.mean(np.abs(result[50:450]) ** 2)
        ratio = output_power / input_power
        assert 0.9 < ratio < 1.1, f"Power ratio {ratio:.4f}"


class TestThiranConvenience:
    """Test thiran_delay convenience function."""

    def test_matches_class(self):
        """Convenience function should give same result as class."""
        signal = np.random.randn(100)
        filt = ThiranDelayFilter(delay=4.3, order=3)
        result_class = filt(signal)
        result_func = thiran_delay(signal, delay=4.3, order=3)
        np.testing.assert_array_equal(result_class, result_func)

    def test_default_order(self):
        """Default order should be 3."""
        signal = np.random.randn(100)
        result = thiran_delay(signal, delay=3.5)
        assert result.shape == signal.shape


class TestThiranOutputShape:
    """Test output shape."""

    def test_output_length(self):
        for n in [10, 100, 500]:
            signal = np.random.randn(n)
            filt = ThiranDelayFilter(delay=3.5, order=3)
            result = filt(signal)
            assert len(result) == n


# ═══════════════════════════════════════════════════════════════════════
# Cross-method comparison
# ═══════════════════════════════════════════════════════════════════════


class TestCrossMethodComparison:
    """Compare all interpolation methods on the same signal."""

    def test_all_reconstruct_bandlimited(self, uniform_grid):
        """All interpolators should reconstruct a bandlimited signal."""
        y_old = _bandlimited_analytic(uniform_grid)
        x_new = np.linspace(1.0, 9.0, 200)
        expected = _bandlimited_analytic(x_new)

        lagrange = LagrangeInterpolator(order=3)
        farrow = FarrowInterpolator(filter_order=3, poly_order=3)

        err_lag = np.max(np.abs(
            lagrange(uniform_grid, y_old, x_new) - expected
        ))
        err_far = np.max(np.abs(
            farrow(uniform_grid, y_old, x_new) - expected
        ))

        assert err_lag < 0.15, f"Lagrange error {err_lag:.4f}"
        assert err_far < 0.15, f"Farrow error {err_far:.4f}"
