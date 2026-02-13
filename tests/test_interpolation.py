# -*- coding: utf-8 -*-
"""
Tests for general-purpose interpolation functions.

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

from grdl.interpolation import (
    Interpolator,
    KernelInterpolator,
    LanczosInterpolator,
    KaiserSincInterpolator,
    lanczos_interpolator,
    windowed_sinc_interpolator,
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


# ── Factory validation ──────────────────────────────────────────────────


class TestFactoryValidation:
    """Test factory function parameter validation."""

    def test_a_too_small(self):
        with pytest.raises(ValueError, match="a must be >= 1"):
            lanczos_interpolator(a=0)

    def test_returns_callable(self):
        interp = lanczos_interpolator()
        assert callable(interp)

    def test_a_equals_1(self):
        """a=1 should work (2-tap kernel, equivalent to linear)."""
        interp = lanczos_interpolator(a=1)
        x = np.linspace(0, 1, 20)
        y = np.sin(x)
        result = interp(x, y, x)
        assert result.shape == x.shape


# ── Bandlimited signal reconstruction ───────────────────────────────────


class TestBandlimitedReconstruction:
    """Test that Lanczos reconstructs bandlimited signals."""

    def test_upsample_accuracy(self, uniform_grid):
        """Upsample 2x and verify against analytic function."""
        y_old = _bandlimited_analytic(uniform_grid)
        interp = lanczos_interpolator(a=3)
        x_new = np.linspace(1.0, 9.0, 200)
        result = interp(uniform_grid, y_old, x_new)
        expected = _bandlimited_analytic(x_new)
        max_err = np.max(np.abs(result - expected))
        # Lanczos-3 uses 6 taps; tolerance is looser than 8-tap kernels
        assert max_err < 0.07, f"Max error {max_err:.4f} exceeds tolerance"

    def test_larger_a_improves_accuracy(self, uniform_grid):
        """More lobes should give better reconstruction."""
        y_old = _bandlimited_analytic(uniform_grid)
        x_new = np.linspace(1.0, 9.0, 200)
        expected = _bandlimited_analytic(x_new)

        err_2 = np.max(np.abs(
            lanczos_interpolator(a=2)(uniform_grid, y_old, x_new) - expected
        ))
        err_5 = np.max(np.abs(
            lanczos_interpolator(a=5)(uniform_grid, y_old, x_new) - expected
        ))

        assert err_5 < err_2, (
            f"a=5 error {err_5:.6f} should be less than "
            f"a=2 error {err_2:.6f}"
        )


# ── Passthrough ─────────────────────────────────────────────────────────


class TestPassthrough:
    """Test interpolation at original sample points."""

    def test_identity_uniform(self, uniform_grid):
        """When x_new == x_old, output should closely match input."""
        y_old = _bandlimited_analytic(uniform_grid)
        interp = lanczos_interpolator(a=3)
        result = interp(uniform_grid, y_old, uniform_grid)
        # Interior points should be very close; edges may truncate
        interior = slice(3, -3)
        np.testing.assert_allclose(
            result[interior], y_old[interior], atol=1e-6,
        )


# ── Complex data ────────────────────────────────────────────────────────


class TestComplexData:
    """Test interpolation of complex-valued signals."""

    def test_complex_exponential(self, uniform_grid):
        """Complex exponential should be reconstructed accurately."""
        y = np.exp(1j * 2 * np.pi * 0.5 * uniform_grid)
        interp = lanczos_interpolator(a=3)
        x_new = np.linspace(1.0, 9.0, 150)
        result = interp(uniform_grid, y, x_new)
        expected = np.exp(1j * 2 * np.pi * 0.5 * x_new)
        max_err = np.max(np.abs(result - expected))
        assert max_err < 0.05, f"Complex max error {max_err:.4f}"

    def test_preserves_complex_dtype(self, uniform_grid):
        """Complex input should produce complex output."""
        y = np.sin(uniform_grid) + 1j * np.cos(uniform_grid)
        interp = lanczos_interpolator()
        result = interp(uniform_grid, y, uniform_grid)
        assert np.iscomplexobj(result)


# ── Out-of-bounds ───────────────────────────────────────────────────────


class TestOutOfBounds:
    """Test behavior for points outside the input range."""

    def test_oob_left(self, uniform_grid):
        y = np.sin(uniform_grid)
        interp = lanczos_interpolator()
        result = interp(uniform_grid, y, np.array([-1.0, -0.5]))
        np.testing.assert_array_equal(result, 0.0)

    def test_oob_right(self, uniform_grid):
        y = np.sin(uniform_grid)
        interp = lanczos_interpolator()
        result = interp(uniform_grid, y, np.array([10.5, 11.0]))
        np.testing.assert_array_equal(result, 0.0)

    def test_mixed(self, uniform_grid):
        y = np.ones_like(uniform_grid)
        interp = lanczos_interpolator()
        result = interp(uniform_grid, y, np.array([-1.0, 5.0, 11.0]))
        assert result[0] == 0.0
        assert result[2] == 0.0
        assert result[1] != 0.0


# ── Output shape ────────────────────────────────────────────────────────


class TestOutputShape:
    """Test output matches x_new length."""

    def test_output_length(self, uniform_grid):
        y = np.sin(uniform_grid)
        interp = lanczos_interpolator()
        for m in [1, 10, 50, 200]:
            x_new = np.linspace(0, 10, m)
            result = interp(uniform_grid, y, x_new)
            assert len(result) == m


# ── ABC and class hierarchy ─────────────────────────────────────────────


class TestABC:
    """Test abstract base class contracts."""

    def test_cannot_instantiate_interpolator(self):
        with pytest.raises(TypeError):
            Interpolator()

    def test_cannot_instantiate_kernel_interpolator(self):
        with pytest.raises(TypeError):
            KernelInterpolator(kernel_length=4)

    def test_lanczos_is_interpolator(self):
        assert isinstance(LanczosInterpolator(), Interpolator)

    def test_lanczos_is_kernel_interpolator(self):
        assert isinstance(LanczosInterpolator(), KernelInterpolator)

    def test_kaiser_sinc_is_interpolator(self):
        assert isinstance(KaiserSincInterpolator(), Interpolator)

    def test_kaiser_sinc_is_kernel_interpolator(self):
        assert isinstance(KaiserSincInterpolator(), KernelInterpolator)

    def test_factory_returns_class_instance(self):
        assert isinstance(lanczos_interpolator(), LanczosInterpolator)
        assert isinstance(windowed_sinc_interpolator(), KaiserSincInterpolator)


# ── Descending x_old ───────────────────────────────────────────────────


class TestDescendingInput:
    """Test interpolation with monotonically decreasing x_old."""

    def test_descending_lanczos(self, uniform_grid):
        """Descending x_old should produce same result as ascending."""
        y_old = _bandlimited_analytic(uniform_grid)
        x_new = np.linspace(2.0, 8.0, 100)
        expected = _bandlimited_analytic(x_new)

        interp = lanczos_interpolator(a=3)
        result_asc = interp(uniform_grid, y_old, x_new)
        result_desc = interp(uniform_grid[::-1], y_old[::-1], x_new)
        np.testing.assert_allclose(result_desc, result_asc, atol=1e-12)

    def test_descending_kaiser(self, uniform_grid):
        """Descending x_old should produce same result as ascending."""
        y_old = _bandlimited_analytic(uniform_grid)
        x_new = np.linspace(2.0, 8.0, 100)

        interp = windowed_sinc_interpolator(kernel_length=8, beta=5.0)
        result_asc = interp(uniform_grid, y_old, x_new)
        result_desc = interp(uniform_grid[::-1], y_old[::-1], x_new)
        np.testing.assert_allclose(result_desc, result_asc, atol=1e-12)

    def test_descending_complex(self, uniform_grid):
        """Descending x_old with complex data."""
        y = np.exp(1j * 2 * np.pi * 0.5 * uniform_grid)
        x_new = np.linspace(2.0, 8.0, 80)

        interp = windowed_sinc_interpolator(kernel_length=8, beta=5.0)
        result_asc = interp(uniform_grid, y, x_new)
        result_desc = interp(uniform_grid[::-1], y[::-1], x_new)
        np.testing.assert_allclose(result_desc, result_asc, atol=1e-12)


# ── Cross-interpolator comparison ───────────────────────────────────────


class TestCrossComparison:
    """Compare Lanczos and Kaiser sinc on the same signal."""

    def test_both_reconstruct_bandlimited(self, uniform_grid):
        """Both interpolators should reconstruct a bandlimited signal."""
        y_old = _bandlimited_analytic(uniform_grid)
        x_new = np.linspace(1.0, 9.0, 200)
        expected = _bandlimited_analytic(x_new)

        lanczos = LanczosInterpolator(a=4)  # 8 taps
        kaiser = KaiserSincInterpolator(kernel_length=8, beta=5.0)  # 8 taps

        err_lanczos = np.max(np.abs(lanczos(uniform_grid, y_old, x_new) - expected))
        err_kaiser = np.max(np.abs(kaiser(uniform_grid, y_old, x_new) - expected))

        # Both should be reasonable
        assert err_lanczos < 0.1
        assert err_kaiser < 0.1
