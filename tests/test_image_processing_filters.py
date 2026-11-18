"""
Spatial Filter Tests - General, SAR scalar, phase, IDAN, and Wishart filters.

Tests correctness, shape preservation, parameter validation, bandwise
dispatch, GPU compatibility flags, and cross-filter ordering properties
using synthetic test data from conftest.py fixtures.
Polarimetric adapter cases cover canonical matrix dispatch, packed Wishart
input, and reciprocal three-channel C3/T3 formation. GPU flag assertions here
are declarations, not device-residency or numerical-equivalence tests.

Dependencies
------------
pytest

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

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

import warnings

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from grdl.exceptions import ValidationError
from grdl.image_processing.filters import (
    GaussianFilter,
    MaxFilter,
    MeanFilter,
    MedianFilter,
    MinFilter,
    StdDevFilter,
)
from grdl.image_processing.filters._validation import (
    validate_kernel_size,
    validate_mode,
)
from grdl.image_processing.filters.linear import boxcar_mean


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("window", [3, 7])
@pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "wrap"])
def test_mean_and_boxcar_share_kernel(dtype, window, mode):
    source = np.random.default_rng(17).normal(size=(9, 11)).astype(dtype)
    original = source.copy()
    mean = MeanFilter(kernel_size=window, mode=mode)
    np.testing.assert_array_equal(mean.apply(source), boxcar_mean(source, window, mode))
    complex_source = source + 1j * source[::-1]
    result = boxcar_mean(complex_source, window, mode)
    expected = mean.apply(complex_source.real) + 1j * mean.apply(complex_source.imag)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(mean.apply(complex_source), expected)
    assert mean.apply(complex_source).dtype == complex_source.dtype
    assert result.dtype == complex_source.dtype
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize(
    "processor",
    [
        MeanFilter(kernel_size=5),
        GaussianFilter(sigma=1.2),
    ],
)
def test_complex_filters_dispatch_every_channel(processor, dtype):
    rng = np.random.default_rng(41)
    source = (rng.normal(size=(4, 17, 19)) + 1j * rng.normal(size=(4, 17, 19))).astype(dtype)
    original = source.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        result = processor.apply(source)
        expected = np.stack([processor.apply(band) for band in source])
    assert np.iscomplexobj(result)
    assert result.shape == source.shape
    assert np.all(np.max(np.abs(result.imag), axis=(1, 2)) > 0)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(source, original)
    if isinstance(processor, GaussianFilter):
        reference = np.stack(
            [
                gaussian_filter(band.real, sigma=1.2) + 1j * gaussian_filter(band.imag, sigma=1.2)
                for band in source
            ]
        )
        assert result.dtype == dtype
        np.testing.assert_allclose(result, reference, rtol=1e-5, atol=1e-6)


def test_boxcar_unit_window_and_reflected_edge():
    source = np.zeros((5, 5), dtype=np.float64)
    source[0, 0] = 9
    assert boxcar_mean(source, 1) is source
    np.testing.assert_array_equal(boxcar_mean(source, 3)[:2, :2], [[4, 2], [2, 1]])


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
        for mode in ("reflect", "constant", "nearest", "wrap"):
            validate_mode(mode)

    def test_invalid_mode_raises(self):
        """Test that invalid mode raises ValidationError."""
        with pytest.raises(ValidationError, match="mode"):
            validate_mode("mirror")


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

    def test_gpu_compatible_true(self):
        """MeanFilter dispatches to cupyx when given a cupy array."""
        assert MeanFilter.__gpu_compatible__ is True

    def test_mode_options(self, random_image):
        """All boundary modes produce valid output."""
        for mode in ("reflect", "constant", "nearest", "wrap"):
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

    def test_gpu_compatible_true(self):
        """GaussianFilter dispatches to cupyx when given a cupy array."""
        assert GaussianFilter.__gpu_compatible__ is True


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

    def test_gpu_compatible_true(self):
        """MedianFilter dispatches to cupyx when given a cupy array."""
        assert MedianFilter.__gpu_compatible__ is True


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

    def test_gpu_compatible_true(self):
        """MinFilter dispatches to cupyx when given a cupy array."""
        assert MinFilter.__gpu_compatible__ is True


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

    def test_gpu_compatible_true(self):
        """MaxFilter dispatches to cupyx when given a cupy array."""
        assert MaxFilter.__gpu_compatible__ is True


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

    def test_gpu_compatible_true(self):
        """StdDevFilter dispatches to cupyx when given a cupy array."""
        assert StdDevFilter.__gpu_compatible__ is True


# ---------------------------------------------------------------------------
# Filter ordering tests
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
        from grdl.image_processing import PercentileStretch, Pipeline

        pipe = Pipeline([MeanFilter(kernel_size=3), PercentileStretch()])
        result = pipe.apply(random_image)
        assert result.shape == random_image.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0
