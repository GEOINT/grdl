# -*- coding: utf-8 -*-
"""
Spatial Filter Tests - Tests for mean, median, Gaussian, min, max, std dev, Lee, and phase gradient filters.

Tests correctness, shape preservation, parameter validation, bandwise
dispatch, GPU compatibility flags, and cross-filter ordering properties
using synthetic test data from conftest.py fixtures.

Dependencies
------------
pytest

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
2026-02-11

Modified
--------
2026-02-11
"""

import pytest
import numpy as np

from grdl.exceptions import ValidationError
from grdl.image_processing.filters import (
    ComplexLeeFilter,
    GaussianFilter,
    LeeFilter,
    MaxFilter,
    MeanFilter,
    MedianFilter,
    MinFilter,
    PhaseGradientFilter,
    StdDevFilter,
)
from grdl.image_processing.filters._validation import (
    validate_kernel_size,
    validate_mode,
)


# ---------------------------------------------------------------------------
# Validation helper tests
# ---------------------------------------------------------------------------

class TestValidation:
    """Test shared validation helpers."""

    def test_valid_kernel_sizes(self):
        """Test that valid odd kernel sizes pass."""
        for ks in (3, 5, 7, 9, 11, 21, 51, 101):
            validate_kernel_size(ks)

    def test_even_kernel_raises(self):
        """Test that even kernel size raises ValidationError."""
        with pytest.raises(ValidationError, match="odd"):
            validate_kernel_size(4)

    def test_kernel_too_small_raises(self):
        """Test that kernel size < 3 raises ValidationError."""
        with pytest.raises(ValidationError, match=">= 3"):
            validate_kernel_size(1)

    def test_kernel_not_int_raises(self):
        """Test that non-integer kernel size raises ValidationError."""
        with pytest.raises(ValidationError, match="integer"):
            validate_kernel_size(3.0)

    def test_valid_modes(self):
        """Test that valid modes pass."""
        for mode in ('reflect', 'constant', 'nearest', 'wrap'):
            validate_mode(mode)

    def test_invalid_mode_raises(self):
        """Test that invalid mode raises ValidationError."""
        with pytest.raises(ValidationError, match="mode"):
            validate_mode('mirror')


# ---------------------------------------------------------------------------
# MeanFilter tests
# ---------------------------------------------------------------------------

class TestMeanFilter:
    """Test MeanFilter correctness and parameters."""

    def test_constant_image_unchanged(self, flat_image):
        """Constant image is a fixed point of mean filtering."""
        f = MeanFilter(kernel_size=5)
        result = f.apply(flat_image)
        np.testing.assert_allclose(result, flat_image, atol=1e-6)

    def test_single_peak_spreads(self, single_peak_image):
        """Single bright pixel spreads to value/(k*k) in center region."""
        f = MeanFilter(kernel_size=5)
        result = f.apply(single_peak_image)
        # Peak at [25,25] with value 100, kernel 5x5 -> 100/25 = 4.0 at center
        assert abs(result[25, 25] - 4.0) < 0.01

    def test_preserves_shape_2d(self, random_image):
        """Output shape matches input for 2D images."""
        f = MeanFilter(kernel_size=3)
        result = f.apply(random_image)
        assert result.shape == random_image.shape

    def test_bandwise_3d(self):
        """3D (bands, rows, cols) arrays are processed per-band."""
        image_3d = np.random.rand(3, 20, 20) * 100
        f = MeanFilter(kernel_size=3)
        result = f.apply(image_3d)
        assert result.shape == (3, 20, 20)

    def test_output_dtype_float(self, random_image):
        """Output is floating-point."""
        f = MeanFilter(kernel_size=3)
        result = f.apply(random_image)
        assert np.issubdtype(result.dtype, np.floating)

    def test_even_kernel_raises(self):
        """Even kernel size raises ValidationError."""
        with pytest.raises(ValidationError):
            MeanFilter(kernel_size=4)

    def test_kernel_too_small_raises(self):
        """Kernel size < 3 raises ValidationError."""
        with pytest.raises(ValidationError):
            MeanFilter(kernel_size=1)

    def test_runtime_param_override(self, random_image):
        """Runtime kernel_size override works."""
        f = MeanFilter(kernel_size=3)
        result_3 = f.apply(random_image)
        result_5 = f.apply(random_image, kernel_size=5)
        # Different kernel sizes should produce different results
        assert not np.allclose(result_3, result_5)

    def test_gpu_compatible_false(self):
        """MeanFilter uses scipy, so not GPU-compatible."""
        assert MeanFilter.__gpu_compatible__ is False

    def test_mode_options(self, random_image):
        """All boundary modes produce valid output."""
        for mode in ('reflect', 'constant', 'nearest', 'wrap'):
            f = MeanFilter(kernel_size=3, mode=mode)
            result = f.apply(random_image)
            assert result.shape == random_image.shape
            assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# GaussianFilter tests
# ---------------------------------------------------------------------------

class TestGaussianFilter:
    """Test GaussianFilter correctness and parameters."""

    def test_constant_image_unchanged(self, flat_image):
        """Constant image is a fixed point of Gaussian filtering."""
        f = GaussianFilter(sigma=2.0)
        result = f.apply(flat_image)
        np.testing.assert_allclose(result, flat_image, atol=1e-6)

    def test_larger_sigma_more_smoothing(self, random_image):
        """Larger sigma produces more smoothing (lower variance in output)."""
        f1 = GaussianFilter(sigma=1.0)
        f2 = GaussianFilter(sigma=5.0)
        r1 = f1.apply(random_image)
        r2 = f2.apply(random_image)
        assert np.var(r2) < np.var(r1)

    def test_preserves_shape_2d(self, random_image):
        """Output shape matches input for 2D images."""
        f = GaussianFilter(sigma=1.0)
        result = f.apply(random_image)
        assert result.shape == random_image.shape

    def test_impulse_response_is_bell_shaped(self):
        """Impulse response peaks at center and decays outward."""
        image = np.zeros((51, 51))
        image[25, 25] = 1000.0
        f = GaussianFilter(sigma=3.0)
        result = f.apply(image)
        # Center should be the maximum
        assert result[25, 25] == np.max(result)
        # Value should decrease away from center
        assert result[25, 25] > result[25, 28]
        assert result[25, 28] > result[25, 31]

    def test_gpu_compatible_false(self):
        """GaussianFilter uses scipy, so not GPU-compatible."""
        assert GaussianFilter.__gpu_compatible__ is False


# ---------------------------------------------------------------------------
# MedianFilter tests
# ---------------------------------------------------------------------------

class TestMedianFilter:
    """Test MedianFilter correctness and parameters."""

    def test_constant_image_unchanged(self, flat_image):
        """Constant image is a fixed point of median filtering."""
        f = MedianFilter(kernel_size=5)
        result = f.apply(flat_image)
        np.testing.assert_allclose(result, flat_image)

    def test_removes_salt_and_pepper(self, salt_pepper_image):
        """Median filter reduces salt-and-pepper noise."""
        base_value = 100.0
        f = MedianFilter(kernel_size=3)
        result = f.apply(salt_pepper_image)
        # MAE from base should be lower after filtering
        mae_before = np.mean(np.abs(salt_pepper_image - base_value))
        mae_after = np.mean(np.abs(result - base_value))
        assert mae_after < mae_before

    def test_preserves_shape_2d(self, random_image):
        """Output shape matches input for 2D images."""
        f = MedianFilter(kernel_size=3)
        result = f.apply(random_image)
        assert result.shape == random_image.shape

    def test_even_kernel_raises(self):
        """Even kernel size raises ValidationError."""
        with pytest.raises(ValidationError):
            MedianFilter(kernel_size=4)

    def test_gpu_compatible_false(self):
        """MedianFilter uses scipy, so not GPU-compatible."""
        assert MedianFilter.__gpu_compatible__ is False


# ---------------------------------------------------------------------------
# MinFilter tests
# ---------------------------------------------------------------------------

class TestMinFilter:
    """Test MinFilter correctness and parameters."""

    def test_constant_image_unchanged(self, flat_image):
        """Constant image is a fixed point of min filtering."""
        f = MinFilter(kernel_size=3)
        result = f.apply(flat_image)
        np.testing.assert_allclose(result, flat_image)

    def test_min_leq_original(self, random_image):
        """Min-filtered output is <= original at every pixel."""
        f = MinFilter(kernel_size=3)
        result = f.apply(random_image)
        assert np.all(result <= random_image + 1e-10)

    def test_preserves_shape_2d(self, random_image):
        """Output shape matches input for 2D images."""
        f = MinFilter(kernel_size=3)
        result = f.apply(random_image)
        assert result.shape == random_image.shape

    def test_gpu_compatible_false(self):
        """MinFilter uses scipy, so not GPU-compatible."""
        assert MinFilter.__gpu_compatible__ is False


# ---------------------------------------------------------------------------
# MaxFilter tests
# ---------------------------------------------------------------------------

class TestMaxFilter:
    """Test MaxFilter correctness and parameters."""

    def test_constant_image_unchanged(self, flat_image):
        """Constant image is a fixed point of max filtering."""
        f = MaxFilter(kernel_size=3)
        result = f.apply(flat_image)
        np.testing.assert_allclose(result, flat_image)

    def test_max_geq_original(self, random_image):
        """Max-filtered output is >= original at every pixel."""
        f = MaxFilter(kernel_size=3)
        result = f.apply(random_image)
        assert np.all(result >= random_image - 1e-10)

    def test_preserves_shape_2d(self, random_image):
        """Output shape matches input for 2D images."""
        f = MaxFilter(kernel_size=3)
        result = f.apply(random_image)
        assert result.shape == random_image.shape

    def test_gpu_compatible_false(self):
        """MaxFilter uses scipy, so not GPU-compatible."""
        assert MaxFilter.__gpu_compatible__ is False


# ---------------------------------------------------------------------------
# StdDevFilter tests
# ---------------------------------------------------------------------------

class TestStdDevFilter:
    """Test StdDevFilter correctness and parameters."""

    def test_constant_image_zero_std(self, flat_image):
        """Std dev of a constant image is zero everywhere."""
        f = StdDevFilter(kernel_size=5)
        result = f.apply(flat_image)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_nonnegative_output(self, random_image):
        """Standard deviation is always non-negative."""
        f = StdDevFilter(kernel_size=5)
        result = f.apply(random_image)
        assert np.all(result >= 0.0)

    def test_boundary_has_nonzero_std(self, step_edge_image):
        """Step edge produces non-zero std at the boundary."""
        f = StdDevFilter(kernel_size=5)
        result = f.apply(step_edge_image)
        # Near the edge at column 40, std should be non-zero
        assert result[20, 40] > 0.0
        # Far from edge, std should be near zero
        assert result[20, 10] < 1.0

    def test_preserves_shape_2d(self, random_image):
        """Output shape matches input for 2D images."""
        f = StdDevFilter(kernel_size=3)
        result = f.apply(random_image)
        assert result.shape == random_image.shape

    def test_output_dtype_float64(self, random_image):
        """Output is float64 for numerical stability."""
        f = StdDevFilter(kernel_size=3)
        result = f.apply(random_image)
        assert result.dtype == np.float64

    def test_even_kernel_raises(self):
        """Even kernel size raises ValidationError."""
        with pytest.raises(ValidationError):
            StdDevFilter(kernel_size=4)

    def test_gpu_compatible_false(self):
        """StdDevFilter uses scipy, so not GPU-compatible."""
        assert StdDevFilter.__gpu_compatible__ is False


# ---------------------------------------------------------------------------
# LeeFilter tests
# ---------------------------------------------------------------------------

class TestLeeFilter:
    """Test LeeFilter correctness and parameters."""

    def test_constant_image_unchanged(self, flat_image):
        """Constant image is a fixed point of Lee filtering."""
        f = LeeFilter(kernel_size=7, noise_variance=1.0)
        result = f.apply(flat_image)
        np.testing.assert_allclose(result, flat_image, atol=1e-10)

    def test_reduces_variance(self):
        """Lee filter reduces variance of noisy data."""
        rng = np.random.RandomState(42)
        clean = np.full((100, 100), 100.0)
        noisy = clean + rng.randn(100, 100) * 20.0
        f = LeeFilter(kernel_size=7)
        result = f.apply(noisy)
        assert np.var(result) < np.var(noisy)

    def test_preserves_edges(self, step_edge_image):
        """Lee filter preserves the step edge contrast."""
        f = LeeFilter(kernel_size=5, noise_variance=1.0)
        result = f.apply(step_edge_image)
        # Well inside each region, values should be close to original
        left_mean = np.mean(result[10:30, 5:15])
        right_mean = np.mean(result[10:30, 55:75])
        # Edge contrast should be mostly preserved
        assert abs(right_mean - left_mean) > 150.0

    def test_auto_noise_estimation(self):
        """Auto noise estimation (noise_variance=0) works."""
        rng = np.random.RandomState(42)
        noisy = np.full((50, 50), 100.0) + rng.randn(50, 50) * 10.0
        f = LeeFilter(kernel_size=7, noise_variance=0.0)
        result = f.apply(noisy)
        assert result.shape == noisy.shape
        assert np.var(result) < np.var(noisy)

    def test_preserves_shape_2d(self, random_image):
        """Output shape matches input for 2D images."""
        f = LeeFilter(kernel_size=7)
        result = f.apply(random_image)
        assert result.shape == random_image.shape

    def test_bandwise_3d(self):
        """3D (bands, rows, cols) arrays are processed per-band."""
        image_3d = np.random.rand(3, 30, 30) * 100
        f = LeeFilter(kernel_size=5)
        result = f.apply(image_3d)
        assert result.shape == (3, 30, 30)

    def test_gpu_compatible_true(self):
        """LeeFilter is pure numpy, so GPU-compatible."""
        assert LeeFilter.__gpu_compatible__ is True

    def test_even_kernel_raises(self):
        """Even kernel size raises ValidationError."""
        with pytest.raises(ValidationError):
            LeeFilter(kernel_size=4)


# ---------------------------------------------------------------------------
# ComplexLeeFilter tests
# ---------------------------------------------------------------------------

class TestComplexLeeFilter:
    """Test ComplexLeeFilter correctness and parameters."""

    def test_rejects_real_input(self):
        """Real-valued input raises ValidationError."""
        f = ComplexLeeFilter(kernel_size=7)
        with pytest.raises(ValidationError, match="complex"):
            f.apply(np.ones((20, 20), dtype=np.float64))

    def test_constant_complex_unchanged(self):
        """Constant complex image is a fixed point."""
        z = np.full((30, 30), 3.0 + 4.0j)
        f = ComplexLeeFilter(kernel_size=7, noise_variance=1.0)
        result = f.apply(z)
        np.testing.assert_allclose(result, z, atol=1e-10)

    def test_output_is_complex(self):
        """Output preserves complex dtype."""
        rng = np.random.RandomState(42)
        z = rng.randn(30, 30) + 1j * rng.randn(30, 30)
        f = ComplexLeeFilter(kernel_size=5)
        result = f.apply(z)
        assert np.iscomplexobj(result)

    def test_preserves_shape_2d(self):
        """Output shape matches input."""
        z = np.ones((25, 35), dtype=np.complex128)
        f = ComplexLeeFilter(kernel_size=3)
        result = f.apply(z)
        assert result.shape == z.shape

    def test_reduces_intensity_variance(self):
        """Despeckled intensity has lower variance than noisy input."""
        rng = np.random.RandomState(42)
        # Constant amplitude with multiplicative speckle
        phase = rng.uniform(-np.pi, np.pi, (100, 100))
        amplitude = 10.0 + rng.randn(100, 100) * 3.0
        z = amplitude * np.exp(1j * phase)
        f = ComplexLeeFilter(kernel_size=7)
        result = f.apply(z)
        assert np.var(np.abs(result)) < np.var(np.abs(z))

    def test_preserves_phase_on_constant_amplitude(self):
        """Phase is preserved when amplitude is constant (no variance)."""
        rows, cols = 40, 40
        phase = np.linspace(0, 2 * np.pi, rows * cols).reshape(rows, cols)
        z = 10.0 * np.exp(1j * phase)
        f = ComplexLeeFilter(kernel_size=5, noise_variance=1.0)
        result = f.apply(z)
        # With constant amplitude, local_var_I ≈ 0 → weight ≈ 0 → output = local_mean
        # Phase should still be close in interior
        np.testing.assert_allclose(
            np.angle(result[10:-10, 10:-10]),
            np.angle(z[10:-10, 10:-10]),
            atol=0.3,
        )

    def test_bandwise_3d(self):
        """3D (bands, rows, cols) complex arrays are processed per-band."""
        z = np.ones((3, 20, 20), dtype=np.complex128)
        f = ComplexLeeFilter(kernel_size=3)
        result = f.apply(z)
        assert result.shape == (3, 20, 20)
        assert np.iscomplexobj(result)

    def test_auto_noise_estimation(self):
        """Auto noise estimation (noise_variance=0) works."""
        rng = np.random.RandomState(42)
        z = rng.randn(50, 50) + 1j * rng.randn(50, 50)
        f = ComplexLeeFilter(kernel_size=7, noise_variance=0.0)
        result = f.apply(z)
        assert result.shape == z.shape
        assert np.iscomplexobj(result)

    def test_even_kernel_raises(self):
        """Even kernel size raises ValidationError."""
        with pytest.raises(ValidationError):
            ComplexLeeFilter(kernel_size=4)

    def test_gpu_compatible_false(self):
        """ComplexLeeFilter uses scipy, so not GPU-compatible."""
        assert ComplexLeeFilter.__gpu_compatible__ is False


# ---------------------------------------------------------------------------
# PhaseGradientFilter tests
# ---------------------------------------------------------------------------

class TestPhaseGradientFilter:
    """Test PhaseGradientFilter correctness and parameters."""

    def test_rejects_real_input(self):
        """Real-valued input raises ValidationError."""
        f = PhaseGradientFilter(kernel_size=3)
        with pytest.raises(ValidationError, match="complex"):
            f.apply(np.ones((20, 20), dtype=np.float64))

    def test_constant_phase_zero_gradient(self):
        """Constant complex image has zero phase gradient everywhere."""
        # All pixels have the same phase -> gradient is zero
        z = np.full((30, 30), 1.0 + 2.0j)
        f = PhaseGradientFilter(kernel_size=3, direction='magnitude')
        result = f.apply(z)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_linear_phase_ramp_row(self):
        """Linear phase ramp along rows produces known row gradient."""
        rows, cols = 50, 50
        # Phase increases by 0.1 rad per row
        phase_rate = 0.1
        row_idx = np.arange(rows)[:, None] * np.ones((1, cols))
        z = np.exp(1j * phase_rate * row_idx)
        f = PhaseGradientFilter(kernel_size=3, direction='row')
        result = f.apply(z)
        # Interior pixels (away from boundaries) should be ~ phase_rate
        interior = result[5:-5, 5:-5]
        np.testing.assert_allclose(interior, phase_rate, atol=1e-6)

    def test_linear_phase_ramp_col(self):
        """Linear phase ramp along columns produces known col gradient."""
        rows, cols = 50, 50
        phase_rate = 0.15
        col_idx = np.ones((rows, 1)) * np.arange(cols)[None, :]
        z = np.exp(1j * phase_rate * col_idx)
        f = PhaseGradientFilter(kernel_size=3, direction='col')
        result = f.apply(z)
        interior = result[5:-5, 5:-5]
        np.testing.assert_allclose(interior, phase_rate, atol=1e-6)

    def test_magnitude_nonnegative(self):
        """Gradient magnitude is always non-negative."""
        rng = np.random.RandomState(42)
        z = (rng.randn(40, 40) + 1j * rng.randn(40, 40))
        f = PhaseGradientFilter(kernel_size=5, direction='magnitude')
        result = f.apply(z)
        assert np.all(result >= 0.0)

    def test_preserves_shape_2d(self):
        """Output shape matches input for 2D complex images."""
        z = np.ones((25, 35), dtype=np.complex128)
        f = PhaseGradientFilter(kernel_size=3)
        result = f.apply(z)
        assert result.shape == z.shape

    def test_output_dtype_float(self):
        """Output is real-valued floating-point."""
        z = np.ones((20, 20), dtype=np.complex128)
        f = PhaseGradientFilter(kernel_size=3)
        result = f.apply(z)
        assert np.issubdtype(result.dtype, np.floating)

    def test_bandwise_3d(self):
        """3D (bands, rows, cols) complex arrays are processed per-band."""
        z = np.ones((3, 20, 20), dtype=np.complex128)
        f = PhaseGradientFilter(kernel_size=3)
        result = f.apply(z)
        assert result.shape == (3, 20, 20)

    def test_invalid_direction_raises(self):
        """Invalid direction raises ValidationError."""
        with pytest.raises(ValidationError, match="direction"):
            PhaseGradientFilter(direction='diagonal')

    def test_even_kernel_raises(self):
        """Even kernel size raises ValidationError."""
        with pytest.raises(ValidationError):
            PhaseGradientFilter(kernel_size=4)

    def test_all_directions_produce_output(self):
        """All three direction options produce valid output."""
        rng = np.random.RandomState(99)
        z = rng.randn(30, 30) + 1j * rng.randn(30, 30)
        for direction in ('magnitude', 'row', 'col'):
            f = PhaseGradientFilter(kernel_size=5, direction=direction)
            result = f.apply(z)
            assert result.shape == z.shape
            assert np.all(np.isfinite(result))

    def test_gpu_compatible_false(self):
        """PhaseGradientFilter uses scipy, so not GPU-compatible."""
        assert PhaseGradientFilter.__gpu_compatible__ is False


# ---------------------------------------------------------------------------
# Cross-filter ordering tests
# ---------------------------------------------------------------------------

class TestFilterOrdering:
    """Test cross-filter ordering properties."""

    def test_min_leq_mean_leq_max(self, random_image):
        """Min <= Mean <= Max at every pixel (same kernel)."""
        ks = 5
        min_result = MinFilter(kernel_size=ks).apply(random_image)
        mean_result = MeanFilter(kernel_size=ks).apply(random_image)
        max_result = MaxFilter(kernel_size=ks).apply(random_image)
        assert np.all(min_result <= mean_result + 1e-6)
        assert np.all(mean_result <= max_result + 1e-6)


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """Test filters compose correctly in a Pipeline."""

    def test_mean_then_percentile_stretch(self, random_image):
        """MeanFilter -> PercentileStretch pipeline works."""
        from grdl.image_processing import Pipeline, PercentileStretch

        pipe = Pipeline([MeanFilter(kernel_size=3), PercentileStretch()])
        result = pipe.apply(random_image)
        assert result.shape == random_image.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0
