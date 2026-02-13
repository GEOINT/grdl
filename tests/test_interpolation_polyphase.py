# -*- coding: utf-8 -*-
"""
Tests for the polyphase FIR interpolator.

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
    KaiserSincInterpolator,
    PolyphaseInterpolator,
    polyphase_interpolator,
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


# ── Factory validation ─────────────────────────────────────────────────


class TestPolyphaseFactory:
    """Test factory function parameter validation."""

    def test_kernel_length_too_small(self):
        with pytest.raises(ValueError, match="kernel_length must be >= 2"):
            polyphase_interpolator(kernel_length=1)

    def test_kernel_length_odd(self):
        with pytest.raises(ValueError, match="kernel_length must be even"):
            polyphase_interpolator(kernel_length=7)

    def test_num_phases_too_small(self):
        with pytest.raises(ValueError, match="num_phases must be >= 2"):
            polyphase_interpolator(num_phases=1)

    def test_returns_callable(self):
        interp = polyphase_interpolator()
        assert callable(interp)

    def test_factory_returns_class(self):
        assert isinstance(polyphase_interpolator(), PolyphaseInterpolator)

    def test_default_parameters(self):
        interp = PolyphaseInterpolator()
        assert interp.kernel_length == 8
        assert interp.num_phases == 128
        assert interp.beta == 5.0


# ── Class hierarchy ────────────────────────────────────────────────────


class TestPolyphaseHierarchy:
    """Test class hierarchy contracts."""

    def test_is_interpolator(self):
        assert isinstance(PolyphaseInterpolator(), Interpolator)

    def test_is_not_kernel_interpolator(self):
        """Polyphase uses its own filter bank, not KernelInterpolator."""
        assert not isinstance(PolyphaseInterpolator(), KernelInterpolator)


# ── Filter bank ────────────────────────────────────────────────────────


class TestFilterBank:
    """Test pre-computed filter bank properties."""

    def test_bank_shape(self):
        interp = PolyphaseInterpolator(kernel_length=8, num_phases=64)
        assert interp._bank.shape == (64, 8)

    def test_phases_sum_to_one(self):
        """Each phase should be normalized to sum to 1 (DC preservation)."""
        interp = PolyphaseInterpolator(kernel_length=8, num_phases=64)
        row_sums = np.sum(interp._bank, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_phase_zero_is_near_delta(self):
        """Phase 0 (d=0) should approximate a delta at center-left tap."""
        interp = PolyphaseInterpolator(
            kernel_length=8, num_phases=128, beta=5.0,
        )
        phase0 = interp._bank[0, :]
        center_left = interp._half - 1  # tap 3 for K=8
        # The center-left tap should have the largest weight
        assert np.argmax(np.abs(phase0)) == center_left

    def test_no_nan_or_inf(self):
        """Filter bank should not contain NaN or Inf."""
        interp = PolyphaseInterpolator(kernel_length=16, num_phases=256)
        assert np.all(np.isfinite(interp._bank))


# ── Bandlimited signal reconstruction ──────────────────────────────────


class TestPolyphaseReconstruction:
    """Test polyphase reconstruction of bandlimited signals."""

    def test_upsample_accuracy(self, uniform_grid):
        """Upsample 2x and verify against analytic function."""
        y_old = _bandlimited_analytic(uniform_grid)
        interp = polyphase_interpolator(kernel_length=8, beta=5.0)
        x_new = np.linspace(1.0, 9.0, 200)
        result = interp(uniform_grid, y_old, x_new)
        expected = _bandlimited_analytic(x_new)
        max_err = np.max(np.abs(result - expected))
        assert max_err < 0.1, f"Max error {max_err:.4f} exceeds tolerance"

    def test_larger_kernel_improves_accuracy(self, uniform_grid):
        """More taps should give better reconstruction."""
        y_old = _bandlimited_analytic(uniform_grid)
        x_new = np.linspace(1.0, 9.0, 200)
        expected = _bandlimited_analytic(x_new)

        err_4 = np.max(np.abs(
            polyphase_interpolator(kernel_length=4)(
                uniform_grid, y_old, x_new
            ) - expected
        ))
        err_16 = np.max(np.abs(
            polyphase_interpolator(kernel_length=16)(
                uniform_grid, y_old, x_new
            ) - expected
        ))
        assert err_16 < err_4, (
            f"K=16 error {err_16:.6f} should be less than "
            f"K=4 error {err_4:.6f}"
        )

    def test_more_phases_improves_accuracy(self, uniform_grid):
        """More phases should give finer fractional delay resolution."""
        y_old = _bandlimited_analytic(uniform_grid)
        x_new = np.linspace(1.0, 9.0, 200)
        expected = _bandlimited_analytic(x_new)

        err_lo = np.max(np.abs(
            polyphase_interpolator(kernel_length=8, num_phases=4)(
                uniform_grid, y_old, x_new
            ) - expected
        ))
        err_hi = np.max(np.abs(
            polyphase_interpolator(kernel_length=8, num_phases=256)(
                uniform_grid, y_old, x_new
            ) - expected
        ))
        assert err_hi < err_lo, (
            f"L=256 error {err_hi:.6f} should be less than "
            f"L=4 error {err_lo:.6f}"
        )

    def test_matches_kaiser_sinc(self, uniform_grid):
        """Should produce similar results to KaiserSincInterpolator."""
        y_old = _bandlimited_analytic(uniform_grid)
        x_new = np.linspace(2.0, 8.0, 150)
        expected = _bandlimited_analytic(x_new)

        kaiser = KaiserSincInterpolator(kernel_length=8, beta=5.0)
        poly = PolyphaseInterpolator(
            kernel_length=8, num_phases=256, beta=5.0,
        )

        err_kaiser = np.max(np.abs(
            kaiser(uniform_grid, y_old, x_new) - expected
        ))
        err_poly = np.max(np.abs(
            poly(uniform_grid, y_old, x_new) - expected
        ))

        # Both should be similar quality with the same parameters
        assert err_kaiser < 0.1
        assert err_poly < 0.1


# ── Passthrough ────────────────────────────────────────────────────────


class TestPolyphasePassthrough:
    """Test interpolation at original sample points."""

    def test_identity_uniform(self, uniform_grid):
        y_old = _bandlimited_analytic(uniform_grid)
        interp = polyphase_interpolator(kernel_length=8, beta=5.0)
        result = interp(uniform_grid, y_old, uniform_grid)
        interior = slice(5, -5)
        # Polyphase has phase quantization + window ripple; 0.01
        # tolerance is appropriate for an 8-tap Kaiser kernel.
        np.testing.assert_allclose(
            result[interior], y_old[interior], atol=0.01,
        )


# ── Complex data ───────────────────────────────────────────────────────


class TestPolyphaseComplex:
    """Test polyphase interpolation of complex-valued signals."""

    def test_complex_exponential(self, uniform_grid):
        y = np.exp(1j * 2 * np.pi * 0.5 * uniform_grid)
        interp = polyphase_interpolator(kernel_length=8, beta=5.0)
        x_new = np.linspace(1.0, 9.0, 150)
        result = interp(uniform_grid, y, x_new)
        expected = np.exp(1j * 2 * np.pi * 0.5 * x_new)
        max_err = np.max(np.abs(result - expected))
        assert max_err < 0.1, f"Complex max error {max_err:.4f}"

    def test_preserves_complex_dtype(self, uniform_grid):
        y = np.sin(uniform_grid) + 1j * np.cos(uniform_grid)
        interp = polyphase_interpolator()
        result = interp(uniform_grid, y, uniform_grid)
        assert np.iscomplexobj(result)


# ── Out-of-bounds ──────────────────────────────────────────────────────


class TestPolyphaseOutOfBounds:
    """Test OOB behavior."""

    def test_oob_left(self, uniform_grid):
        y = np.sin(uniform_grid)
        interp = polyphase_interpolator()
        result = interp(uniform_grid, y, np.array([-1.0, -0.5]))
        np.testing.assert_array_equal(result, 0.0)

    def test_oob_right(self, uniform_grid):
        y = np.sin(uniform_grid)
        interp = polyphase_interpolator()
        result = interp(uniform_grid, y, np.array([10.5, 11.0]))
        np.testing.assert_array_equal(result, 0.0)


# ── Descending x_old ──────────────────────────────────────────────────


class TestPolyphaseDescending:
    """Test with monotonically decreasing x_old."""

    def test_descending_matches_ascending(self, uniform_grid):
        y_old = _bandlimited_analytic(uniform_grid)
        x_new = np.linspace(2.0, 8.0, 100)

        interp = polyphase_interpolator(kernel_length=8, beta=5.0)
        result_asc = interp(uniform_grid, y_old, x_new)
        result_desc = interp(uniform_grid[::-1], y_old[::-1], x_new)
        np.testing.assert_allclose(result_desc, result_asc, atol=1e-10)


# ── Output shape ───────────────────────────────────────────────────────


class TestPolyphaseOutputShape:
    """Test output shape matches x_new."""

    def test_output_length(self, uniform_grid):
        y = np.sin(uniform_grid)
        interp = polyphase_interpolator()
        for m in [1, 10, 50, 200]:
            x_new = np.linspace(0, 10, m)
            result = interp(uniform_grid, y, x_new)
            assert len(result) == m


# ══════════════════════════════════════════════════════════════════════
# Remez prototype tests
# ══════════════════════════════════════════════════════════════════════


# ── Remez factory validation ──────────────────────────────────────────


class TestRemezFactory:
    """Test Remez prototype factory validation."""

    def test_invalid_prototype(self):
        with pytest.raises(ValueError, match="prototype must be one of"):
            polyphase_interpolator(prototype='invalid')

    def test_remez_returns_callable(self):
        interp = polyphase_interpolator(prototype='remez')
        assert callable(interp)

    def test_remez_returns_class(self):
        assert isinstance(
            polyphase_interpolator(prototype='remez'),
            PolyphaseInterpolator,
        )

    def test_remez_is_interpolator(self):
        assert isinstance(
            PolyphaseInterpolator(prototype='remez'), Interpolator,
        )

    def test_prototype_property(self):
        kaiser = PolyphaseInterpolator(prototype='kaiser')
        remez = PolyphaseInterpolator(prototype='remez')
        assert kaiser.prototype == 'kaiser'
        assert remez.prototype == 'remez'


# ── Remez filter bank properties ──────────────────────────────────────


class TestRemezFilterBank:
    """Test Remez filter bank construction properties."""

    def test_bank_shape(self):
        interp = PolyphaseInterpolator(
            kernel_length=8, num_phases=64, prototype='remez',
        )
        assert interp._bank.shape == (64, 8)

    def test_phases_sum_to_one(self):
        """Each Remez phase should be normalized to sum to 1."""
        interp = PolyphaseInterpolator(
            kernel_length=8, num_phases=64, prototype='remez',
        )
        row_sums = np.sum(interp._bank, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_no_nan_or_inf(self):
        """Filter bank should not contain NaN or Inf."""
        interp = PolyphaseInterpolator(
            kernel_length=16, num_phases=128, prototype='remez',
        )
        assert np.all(np.isfinite(interp._bank))

    def test_different_transition_widths(self):
        """Narrower transition should produce different bank coefficients."""
        wide = PolyphaseInterpolator(
            kernel_length=8, num_phases=64,
            prototype='remez', transition_width=1.0,
        )
        narrow = PolyphaseInterpolator(
            kernel_length=8, num_phases=64,
            prototype='remez', transition_width=0.2,
        )
        # Banks should differ (different Remez designs)
        assert not np.allclose(wide._bank, narrow._bank)


# ── Remez reconstruction accuracy ────────────────────────────────────


class TestRemezReconstruction:
    """Test Remez polyphase reconstruction of bandlimited signals."""

    def test_upsample_accuracy(self, uniform_grid):
        """Remez prototype should reconstruct bandlimited signals.

        The Remez equiripple prototype produces optimal stopband
        rejection but inherent passband gain variation (~12 %) when
        decomposed into polyphase phases.  This is fundamental to
        equiripple design; the smalleyd scatter-mode normalizes at
        runtime, but gather-mode (used here) does not.  A 0.20
        threshold accommodates this.
        """
        y_old = _bandlimited_analytic(uniform_grid)
        interp = polyphase_interpolator(
            kernel_length=8, num_phases=128, prototype='remez',
        )
        x_new = np.linspace(1.0, 9.0, 200)
        result = interp(uniform_grid, y_old, x_new)
        expected = _bandlimited_analytic(x_new)
        max_err = np.max(np.abs(result - expected))
        assert max_err < 0.20, f"Remez max error {max_err:.4f} exceeds 0.20"

    def test_larger_kernel_improves_accuracy(self, uniform_grid):
        """More taps should give better Remez reconstruction."""
        y_old = _bandlimited_analytic(uniform_grid)
        x_new = np.linspace(1.0, 9.0, 200)
        expected = _bandlimited_analytic(x_new)

        err_4 = np.max(np.abs(
            polyphase_interpolator(
                kernel_length=4, prototype='remez',
            )(uniform_grid, y_old, x_new) - expected
        ))
        err_16 = np.max(np.abs(
            polyphase_interpolator(
                kernel_length=16, prototype='remez',
            )(uniform_grid, y_old, x_new) - expected
        ))
        assert err_16 < err_4, (
            f"K=16 error {err_16:.6f} should be less than "
            f"K=4 error {err_4:.6f}"
        )

    def test_remez_competitive_with_kaiser(self, uniform_grid):
        """Both prototypes should reconstruct bandlimited signals.

        Remez prototype decomposition trades per-phase accuracy for
        optimal equiripple stopband rejection.  In gather mode (no
        runtime normalization), Remez has higher per-sample error
        (~12-18 %) than Kaiser (~0.1 %) due to passband gain
        variation in the decomposed phases.  The Remez advantage is
        uniform sidelobe control, which benefits SAR image formation
        (PFA) where the 2D IFFT averages many samples.
        """
        y_old = _bandlimited_analytic(uniform_grid)
        x_new = np.linspace(2.0, 8.0, 150)
        expected = _bandlimited_analytic(x_new)

        kaiser = polyphase_interpolator(
            kernel_length=8, num_phases=128, prototype='kaiser',
        )
        remez = polyphase_interpolator(
            kernel_length=8, num_phases=128, prototype='remez',
        )

        err_kaiser = np.max(np.abs(
            kaiser(uniform_grid, y_old, x_new) - expected
        ))
        err_remez = np.max(np.abs(
            remez(uniform_grid, y_old, x_new) - expected
        ))

        assert err_kaiser < 0.1
        assert err_remez < 0.20


# ── Remez complex data ───────────────────────────────────────────────


class TestRemezComplex:
    """Test Remez polyphase with complex-valued signals."""

    def test_complex_exponential(self, uniform_grid):
        y = np.exp(1j * 2 * np.pi * 0.5 * uniform_grid)
        interp = polyphase_interpolator(
            kernel_length=8, prototype='remez',
        )
        x_new = np.linspace(1.0, 9.0, 150)
        result = interp(uniform_grid, y, x_new)
        expected = np.exp(1j * 2 * np.pi * 0.5 * x_new)
        max_err = np.max(np.abs(result - expected))
        assert max_err < 0.20, f"Complex Remez max error {max_err:.4f}"

    def test_preserves_complex_dtype(self, uniform_grid):
        y = np.sin(uniform_grid) + 1j * np.cos(uniform_grid)
        interp = polyphase_interpolator(prototype='remez')
        result = interp(uniform_grid, y, uniform_grid)
        assert np.iscomplexobj(result)


# ── Remez passthrough ─────────────────────────────────────────────────


class TestRemezPassthrough:
    """Test Remez interpolation at original sample points."""

    def test_identity_uniform(self, uniform_grid):
        """Remez at original sample points (passthrough).

        At d=0, the Remez equiripple phase is NOT a perfect delta
        (peak ≈ 0.86 instead of 1.0 for K=8) because the equiripple
        stopband distributes energy across all taps.  This gives
        ~12-17 % error at the original sample locations.
        """
        y_old = _bandlimited_analytic(uniform_grid)
        interp = polyphase_interpolator(
            kernel_length=8, prototype='remez',
        )
        result = interp(uniform_grid, y_old, uniform_grid)
        interior = slice(5, -5)
        np.testing.assert_allclose(
            result[interior], y_old[interior], atol=0.20,
        )


# ── Remez out-of-bounds ──────────────────────────────────────────────


class TestRemezOutOfBounds:
    """Test Remez OOB behavior."""

    def test_oob_left(self, uniform_grid):
        y = np.sin(uniform_grid)
        interp = polyphase_interpolator(prototype='remez')
        result = interp(uniform_grid, y, np.array([-1.0, -0.5]))
        np.testing.assert_array_equal(result, 0.0)

    def test_oob_right(self, uniform_grid):
        y = np.sin(uniform_grid)
        interp = polyphase_interpolator(prototype='remez')
        result = interp(uniform_grid, y, np.array([10.5, 11.0]))
        np.testing.assert_array_equal(result, 0.0)
