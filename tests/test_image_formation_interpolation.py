# -*- coding: utf-8 -*-
"""
Tests for windowed sinc interpolation.

Verifies bandwidth-preserving interpolation for PFA k-space resampling.

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

import numpy as np
import pytest

from grdl.interpolation import windowed_sinc_interpolator


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def uniform_grid():
    """Uniform input grid with 100 samples."""
    return np.linspace(0.0, 10.0, 100)


@pytest.fixture
def bandlimited_signal(uniform_grid):
    """Bandlimited signal: sum of sinusoids well below Nyquist."""
    x = uniform_grid
    # Nyquist = 1 / (2 * dx) ~ 5 cycles/unit
    # Use frequencies well below Nyquist
    return (
        np.sin(2 * np.pi * 0.5 * x)
        + 0.5 * np.cos(2 * np.pi * 1.2 * x)
        + 0.3 * np.sin(2 * np.pi * 2.0 * x)
    )


def _bandlimited_analytic(x):
    """Analytic form of the bandlimited test signal."""
    return (
        np.sin(2 * np.pi * 0.5 * x)
        + 0.5 * np.cos(2 * np.pi * 1.2 * x)
        + 0.3 * np.sin(2 * np.pi * 2.0 * x)
    )


# ── Factory validation ──────────────────────────────────────────────────


class TestFactoryValidation:
    """Test factory function parameter validation."""

    def test_kernel_length_too_small(self):
        with pytest.raises(ValueError, match="kernel_length must be even"):
            windowed_sinc_interpolator(kernel_length=1)

    def test_kernel_length_odd(self):
        with pytest.raises(ValueError, match="kernel_length must be even"):
            windowed_sinc_interpolator(kernel_length=7)

    def test_returns_callable(self):
        interp = windowed_sinc_interpolator()
        assert callable(interp)

    def test_default_params(self):
        """Default kernel_length=8, beta=5.0 — should not raise."""
        interp = windowed_sinc_interpolator()
        x = np.linspace(0, 1, 20)
        y = np.sin(x)
        result = interp(x, y, x)
        assert result.shape == x.shape


# ── Bandlimited signal reconstruction ───────────────────────────────────


class TestBandlimitedReconstruction:
    """Test that windowed sinc reconstructs bandlimited signals."""

    def test_upsample_accuracy(self, uniform_grid, bandlimited_signal):
        """Upsample 2x and verify against analytic function."""
        interp = windowed_sinc_interpolator(kernel_length=8, beta=5.0)
        # Upsample: query at 2x finer grid, interior only
        x_new = np.linspace(1.0, 9.0, 200)
        result = interp(uniform_grid, bandlimited_signal, x_new)
        expected = _bandlimited_analytic(x_new)
        # Should be very close for a bandlimited signal
        max_err = np.max(np.abs(result - expected))
        assert max_err < 0.05, f"Max error {max_err:.4f} exceeds tolerance"

    def test_larger_kernel_improves_accuracy(
        self, uniform_grid, bandlimited_signal,
    ):
        """Larger kernel should give better reconstruction."""
        x_new = np.linspace(1.0, 9.0, 200)
        expected = _bandlimited_analytic(x_new)

        interp_4 = windowed_sinc_interpolator(kernel_length=4, beta=5.0)
        interp_16 = windowed_sinc_interpolator(kernel_length=16, beta=5.0)

        err_4 = np.max(np.abs(interp_4(uniform_grid, bandlimited_signal, x_new) - expected))
        err_16 = np.max(np.abs(interp_16(uniform_grid, bandlimited_signal, x_new) - expected))

        assert err_16 < err_4, (
            f"16-tap error {err_16:.6f} should be less than "
            f"4-tap error {err_4:.6f}"
        )


# ── Passthrough ─────────────────────────────────────────────────────────


class TestPassthrough:
    """Test interpolation at original sample points."""

    def test_identity_uniform(self, uniform_grid, bandlimited_signal):
        """When x_new == x_old, output should closely match input."""
        interp = windowed_sinc_interpolator(kernel_length=8, beta=5.0)
        result = interp(uniform_grid, bandlimited_signal, uniform_grid)
        # Interior points should be very close
        # Edge points may differ due to kernel truncation
        interior = slice(4, -4)
        np.testing.assert_allclose(
            result[interior],
            bandlimited_signal[interior],
            atol=1e-6,
        )

    def test_identity_preserves_dtype_float(self, uniform_grid):
        """Float input should produce float output."""
        y = np.sin(uniform_grid).astype(np.float32)
        interp = windowed_sinc_interpolator()
        result = interp(uniform_grid, y, uniform_grid)
        assert np.issubdtype(result.dtype, np.floating)


# ── Complex data ────────────────────────────────────────────────────────


class TestComplexData:
    """Test interpolation of complex-valued signals."""

    def test_complex_signal(self, uniform_grid):
        """Complex signal should be interpolated correctly."""
        y = np.exp(1j * 2 * np.pi * 0.5 * uniform_grid)
        interp = windowed_sinc_interpolator(kernel_length=8, beta=5.0)
        x_new = np.linspace(1.0, 9.0, 150)
        result = interp(uniform_grid, y, x_new)
        expected = np.exp(1j * 2 * np.pi * 0.5 * x_new)
        max_err = np.max(np.abs(result - expected))
        assert max_err < 0.05, f"Complex max error {max_err:.4f}"

    def test_complex_preserves_phase(self, uniform_grid):
        """Phase of single-frequency complex exponential should be preserved."""
        freq = 1.0
        y = np.exp(1j * 2 * np.pi * freq * uniform_grid)
        interp = windowed_sinc_interpolator(kernel_length=16, beta=6.0)
        x_new = np.linspace(2.0, 8.0, 100)
        result = interp(uniform_grid, y, x_new)
        expected_phase = 2 * np.pi * freq * x_new
        result_phase = np.angle(result)
        # Unwrap for comparison
        phase_err = np.abs(np.exp(1j * result_phase) - np.exp(1j * expected_phase))
        assert np.max(phase_err) < 0.1


# ── Non-uniform input grid ──────────────────────────────────────────────


class TestNonUniformGrid:
    """Test interpolation with non-uniformly spaced input."""

    def test_non_uniform_input(self):
        """Mildly non-uniform x_old (perturbed uniform, realistic for PFA)."""
        rng = np.random.default_rng(42)
        # Perturbed uniform grid: ~10% jitter on spacing
        dx = 0.1
        x_uniform = np.arange(0, 10, dx)
        jitter = rng.uniform(-0.03, 0.03, len(x_uniform))
        jitter[0] = 0.0  # keep endpoints fixed
        jitter[-1] = 0.0
        x_old = x_uniform + jitter

        y_old = np.sin(2 * np.pi * 0.3 * x_old)
        x_new = np.linspace(1.0, 9.0, 80)
        expected = np.sin(2 * np.pi * 0.3 * x_new)

        interp = windowed_sinc_interpolator(kernel_length=8, beta=5.0)
        result = interp(x_old, y_old, x_new)
        max_err = np.max(np.abs(result - expected))
        # Sinc interpolation assumes approximately uniform sampling;
        # ±30% jitter is aggressive. PFA grids have < 5% variation.
        assert max_err < 0.15, f"Non-uniform max error {max_err:.4f}"


# ── Out-of-bounds fill ──────────────────────────────────────────────────


class TestOutOfBounds:
    """Test behavior for points outside the input range."""

    def test_oob_left(self, uniform_grid, bandlimited_signal):
        """Points left of x_old range should be zero."""
        interp = windowed_sinc_interpolator()
        x_new = np.array([-1.0, -0.5])
        result = interp(uniform_grid, bandlimited_signal, x_new)
        np.testing.assert_array_equal(result, 0.0)

    def test_oob_right(self, uniform_grid, bandlimited_signal):
        """Points right of x_old range should be zero."""
        interp = windowed_sinc_interpolator()
        x_new = np.array([10.5, 11.0])
        result = interp(uniform_grid, bandlimited_signal, x_new)
        np.testing.assert_array_equal(result, 0.0)

    def test_mixed_inbound_oob(self, uniform_grid, bandlimited_signal):
        """Mix of in-bounds and out-of-bounds points."""
        interp = windowed_sinc_interpolator()
        x_new = np.array([-1.0, 5.0, 11.0])
        result = interp(uniform_grid, bandlimited_signal, x_new)
        assert result[0] == 0.0
        assert result[2] == 0.0
        assert result[1] != 0.0  # in-bounds should be non-zero


# ── PFA interface compatibility ─────────────────────────────────────────


class TestPFAInterface:
    """Test compatibility with PolarFormatAlgorithm interpolator interface."""

    def test_signature_matches(self):
        """Callable should accept (x_old, y_old, x_new) and return ndarray."""
        interp = windowed_sinc_interpolator()
        x_old = np.linspace(0, 1, 50)
        y_old = np.sin(x_old) + 1j * np.cos(x_old)
        x_new = np.linspace(0.1, 0.9, 30)
        result = interp(x_old, y_old, x_new)
        assert isinstance(result, np.ndarray)
        assert result.shape == (30,)
        assert np.iscomplexobj(result)

    def test_output_length_matches_x_new(self):
        """Output length should always equal len(x_new)."""
        interp = windowed_sinc_interpolator()
        x_old = np.linspace(0, 10, 100)
        y_old = np.ones(100)
        for m in [1, 10, 50, 200]:
            x_new = np.linspace(0, 10, m)
            result = interp(x_old, y_old, x_new)
            assert len(result) == m, f"Expected {m}, got {len(result)}"
