# -*- coding: utf-8 -*-
"""
Tests for ImageJ/Fiji ported components.

Verifies algorithmic correctness, edge cases, and parameter validation
for all six ported ImageJ/Fiji image processing components.

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


# ============================================================================
# RollingBallBackground Tests
# ============================================================================

class TestRollingBallBackground:
    """Tests for Rolling Ball Background Subtraction."""

    def test_import(self):
        from grdl.imagej import RollingBallBackground
        assert RollingBallBackground is not None

    def test_version_attribute(self):
        from grdl.imagej import RollingBallBackground
        assert RollingBallBackground.__imagej_version__ == '1.54j'
        assert RollingBallBackground.__imagej_source__ == (
            'ij/plugin/filter/BackgroundSubtracter.java'
        )
        assert RollingBallBackground.__processor_version__ == '1.54j'

    def test_flat_image_low_variation(self):
        """A flat image should produce uniform output (constant everywhere)."""
        from grdl.imagej import RollingBallBackground
        rb = RollingBallBackground(radius=10)
        flat = np.full((50, 50), 100.0)
        result = rb.apply(flat)
        assert result.shape == (50, 50)
        # Output should be spatially uniform (all same value)
        assert result.std() < 0.01

    def test_removes_gradient(self):
        """A linear gradient should be mostly removed."""
        from grdl.imagej import RollingBallBackground
        rb = RollingBallBackground(radius=20)
        rows, cols = 100, 100
        gradient = np.tile(np.linspace(0, 200, cols), (rows, 1))
        result = rb.apply(gradient)
        # Result should have much less variation than input
        assert result.std() < gradient.std() * 0.5

    def test_preserves_small_features(self):
        """Small bright features on dark background should be preserved."""
        from grdl.imagej import RollingBallBackground
        rb = RollingBallBackground(radius=30)
        image = np.zeros((100, 100))
        # Add a small bright spot
        image[45:55, 45:55] = 200.0
        result = rb.apply(image)
        # The bright spot should still be present
        assert result[50, 50] > 50.0

    def test_create_background(self):
        """create_background=True should return the background estimate."""
        from grdl.imagej import RollingBallBackground
        rb = RollingBallBackground(radius=20, create_background=True)
        rows, cols = 80, 80
        gradient = np.tile(np.linspace(50, 200, cols), (rows, 1))
        bg = rb.apply(gradient)
        assert bg.shape == (rows, cols)
        # Background should track the gradient
        assert bg.mean() > 50.0

    def test_light_background(self):
        """light_background inverts the rolling direction."""
        from grdl.imagej import RollingBallBackground
        rb = RollingBallBackground(radius=20, light_background=True)
        image = np.full((50, 50), 200.0)
        image[20:30, 20:30] = 50.0  # Dark spot on light background
        result = rb.apply(image)
        assert result.shape == (50, 50)

    def test_rejects_non_2d(self):
        from grdl.imagej import RollingBallBackground
        rb = RollingBallBackground()
        with pytest.raises(ValueError, match="2D"):
            rb.apply(np.zeros((3, 10, 10)))

    def test_output_dtype(self):
        from grdl.imagej import RollingBallBackground
        rb = RollingBallBackground(radius=5)
        result = rb.apply(np.ones((20, 20), dtype=np.uint8) * 128)
        assert result.dtype == np.float64

    def test_nonnegative_output(self):
        """Result should be non-negative (clipped at zero)."""
        from grdl.imagej import RollingBallBackground
        rb = RollingBallBackground(radius=10)
        image = np.random.RandomState(42).rand(50, 50) * 255
        result = rb.apply(image)
        assert np.all(result >= 0.0)

    def test_small_radius(self):
        """Radius=1 should still work without error."""
        from grdl.imagej import RollingBallBackground
        rb = RollingBallBackground(radius=1)
        result = rb.apply(np.random.RandomState(0).rand(30, 30) * 100)
        assert result.shape == (30, 30)


# ============================================================================
# CLAHE Tests
# ============================================================================

class TestCLAHE:
    """Tests for Contrast Limited Adaptive Histogram Equalization."""

    def test_import(self):
        from grdl.imagej import CLAHE
        assert CLAHE is not None

    def test_version_attribute(self):
        from grdl.imagej import CLAHE
        assert CLAHE.__imagej_version__ == '0.5.0'
        assert CLAHE.__processor_version__ == '0.5.0'

    def test_output_range(self):
        """Output should be in [0, 1]."""
        from grdl.imagej import CLAHE
        clahe = CLAHE(block_size=15, n_bins=64, max_slope=3.0)
        image = np.random.RandomState(42).rand(50, 50) * 255
        result = clahe.apply(image)
        assert result.min() >= -0.01  # Allow tiny numerical error
        assert result.max() <= 1.01

    def test_flat_image_stays_flat(self):
        """A constant image should map to all zeros (no variation)."""
        from grdl.imagej import CLAHE
        clahe = CLAHE(block_size=15, n_bins=64)
        flat = np.full((40, 40), 100.0)
        result = clahe.apply(flat)
        assert np.allclose(result, 0.0, atol=0.01)

    def test_enhances_contrast(self):
        """CLAHE should increase contrast in a low-contrast image."""
        from grdl.imagej import CLAHE
        clahe = CLAHE(block_size=20, n_bins=64, max_slope=3.0)
        # Low contrast image: values between 100 and 110
        rng = np.random.RandomState(42)
        image = rng.rand(60, 60) * 10 + 100
        result = clahe.apply(image)
        # Normalized result should span more of [0, 1] than input
        input_range = (image.max() - image.min()) / 255
        output_range = result.max() - result.min()
        assert output_range > input_range

    def test_output_shape(self):
        from grdl.imagej import CLAHE
        clahe = CLAHE(block_size=15)
        image = np.random.RandomState(0).rand(70, 90)
        result = clahe.apply(image)
        assert result.shape == (70, 90)

    def test_rejects_non_2d(self):
        from grdl.imagej import CLAHE
        clahe = CLAHE()
        with pytest.raises(ValueError, match="2D"):
            clahe.apply(np.zeros((3, 10, 10)))

    def test_invalid_params(self):
        from grdl.imagej import CLAHE
        with pytest.raises(ValueError):
            CLAHE(block_size=1)
        with pytest.raises(ValueError):
            CLAHE(n_bins=1)
        with pytest.raises(ValueError):
            CLAHE(max_slope=0.5)

    def test_max_slope_1_is_standard_ahe(self):
        """max_slope=1.0 should produce standard AHE (no clipping)."""
        from grdl.imagej import CLAHE
        clahe = CLAHE(block_size=20, max_slope=1.0)
        image = np.random.RandomState(7).rand(40, 40) * 200
        result = clahe.apply(image)
        assert result.shape == (40, 40)


# ============================================================================
# AutoLocalThreshold Tests
# ============================================================================

class TestAutoLocalThreshold:
    """Tests for Auto Local Threshold."""

    def test_import(self):
        from grdl.imagej import AutoLocalThreshold
        assert AutoLocalThreshold is not None

    def test_version_attribute(self):
        from grdl.imagej import AutoLocalThreshold
        assert AutoLocalThreshold.__imagej_version__ == '1.10.1'
        assert AutoLocalThreshold.__processor_version__ == '1.10.1'

    def test_all_methods_run(self):
        """Every method should execute without error."""
        from grdl.imagej import AutoLocalThreshold
        rng = np.random.RandomState(42)
        image = rng.rand(50, 50) * 255

        methods = [
            'bernsen', 'mean', 'median', 'midgrey',
            'niblack', 'sauvola', 'phansalkar', 'contrast',
        ]
        for method in methods:
            alt = AutoLocalThreshold(method=method, radius=7)
            result = alt.apply(image)
            assert result.shape == (50, 50), f"Failed for {method}"
            # Output should be binary
            unique = np.unique(result)
            assert all(v in [0.0, 1.0] for v in unique), (
                f"Non-binary output for {method}: {unique}"
            )

    def test_binary_output(self):
        """Output must only contain 0.0 and 1.0."""
        from grdl.imagej import AutoLocalThreshold
        alt = AutoLocalThreshold(method='sauvola', radius=10)
        image = np.random.RandomState(0).rand(40, 40) * 200
        result = alt.apply(image)
        unique = set(np.unique(result))
        assert unique.issubset({0.0, 1.0})

    def test_niblack_dark_image_mostly_background(self):
        """Very dark image with negative k should be mostly background."""
        from grdl.imagej import AutoLocalThreshold
        alt = AutoLocalThreshold(method='niblack', radius=10, k=-0.2)
        dark = np.full((40, 40), 10.0)
        result = alt.apply(dark)
        # Uniform image → all same threshold → all same class
        assert result.sum() == 0.0 or result.sum() == result.size

    def test_sauvola_params(self):
        """Sauvola with different k and r values should produce different results."""
        from grdl.imagej import AutoLocalThreshold
        rng = np.random.RandomState(5)
        image = rng.rand(50, 50) * 200 + 20

        alt1 = AutoLocalThreshold(method='sauvola', radius=10, k=0.2, r=128)
        alt2 = AutoLocalThreshold(method='sauvola', radius=10, k=0.8, r=128)
        r1 = alt1.apply(image)
        r2 = alt2.apply(image)
        # Different k should give different results
        assert not np.array_equal(r1, r2)

    def test_rejects_non_2d(self):
        from grdl.imagej import AutoLocalThreshold
        alt = AutoLocalThreshold()
        with pytest.raises(ValueError, match="2D"):
            alt.apply(np.zeros((3, 10, 10)))

    def test_invalid_method(self):
        from grdl.imagej import AutoLocalThreshold
        with pytest.raises(ValueError, match="Unknown method"):
            AutoLocalThreshold(method='nonexistent')

    def test_invalid_radius(self):
        from grdl.imagej import AutoLocalThreshold
        with pytest.raises(ValueError, match="radius"):
            AutoLocalThreshold(radius=0)

    def test_bernsen_low_contrast_region(self):
        """Bernsen should classify low-contrast regions as background."""
        from grdl.imagej import AutoLocalThreshold
        alt = AutoLocalThreshold(
            method='bernsen', radius=10, contrast_threshold=15
        )
        # Uniform region with contrast < threshold
        image = np.full((40, 40), 100.0)
        image += np.random.RandomState(0).rand(40, 40) * 5  # contrast ~5
        result = alt.apply(image)
        # Low contrast → mostly background
        assert result.mean() < 0.3

    def test_phansalkar_runs(self):
        """Phansalkar with default p, q parameters."""
        from grdl.imagej import AutoLocalThreshold
        alt = AutoLocalThreshold(
            method='phansalkar', radius=10, k=0.25, p=2.0, q=10.0, r=128.0
        )
        image = np.random.RandomState(3).rand(40, 40) * 200
        result = alt.apply(image)
        assert result.shape == (40, 40)


# ============================================================================
# UnsharpMask Tests
# ============================================================================

class TestUnsharpMask:
    """Tests for Unsharp Mask."""

    def test_import(self):
        from grdl.imagej import UnsharpMask
        assert UnsharpMask is not None

    def test_version_attribute(self):
        from grdl.imagej import UnsharpMask
        assert UnsharpMask.__imagej_version__ == '1.54j'
        assert UnsharpMask.__processor_version__ == '1.54j'

    def test_weight_zero_is_identity(self):
        """weight=0 should return the original image."""
        from grdl.imagej import UnsharpMask
        usm = UnsharpMask(sigma=2.0, weight=0.0)
        image = np.random.RandomState(42).rand(50, 50) * 200
        result = usm.apply(image)
        np.testing.assert_allclose(result, image, atol=1e-10)

    def test_increases_edge_contrast(self):
        """Sharpening should increase the gradient at edges."""
        from grdl.imagej import UnsharpMask
        usm = UnsharpMask(sigma=2.0, weight=0.6)
        # Step edge
        image = np.zeros((50, 100))
        image[:, 50:] = 200.0
        result = usm.apply(image)
        # Gradient at edge should be steeper in sharpened image
        orig_grad = np.abs(np.diff(image[25, :]))
        sharp_grad = np.abs(np.diff(result[25, :]))
        assert sharp_grad.max() >= orig_grad.max()

    def test_flat_image_unchanged(self):
        """A flat image has no edges to sharpen."""
        from grdl.imagej import UnsharpMask
        usm = UnsharpMask(sigma=2.0, weight=0.6)
        flat = np.full((40, 40), 128.0)
        result = usm.apply(flat)
        np.testing.assert_allclose(result, flat, atol=0.01)

    def test_output_shape_and_dtype(self):
        from grdl.imagej import UnsharpMask
        usm = UnsharpMask()
        result = usm.apply(np.ones((30, 40), dtype=np.uint8) * 100)
        assert result.shape == (30, 40)
        assert result.dtype == np.float64

    def test_rejects_non_2d(self):
        from grdl.imagej import UnsharpMask
        usm = UnsharpMask()
        with pytest.raises(ValueError, match="2D"):
            usm.apply(np.zeros((3, 10, 10)))

    def test_invalid_sigma(self):
        from grdl.imagej import UnsharpMask
        with pytest.raises(ValueError, match="sigma"):
            UnsharpMask(sigma=0)

    def test_invalid_weight(self):
        from grdl.imagej import UnsharpMask
        with pytest.raises(ValueError, match="weight"):
            UnsharpMask(weight=-1)

    def test_higher_weight_stronger_sharpening(self):
        """Higher weight should produce stronger sharpening effect."""
        from grdl.imagej import UnsharpMask
        image = np.zeros((50, 100))
        image[:, 50:] = 200.0

        usm_weak = UnsharpMask(sigma=2.0, weight=0.3)
        usm_strong = UnsharpMask(sigma=2.0, weight=1.5)
        r_weak = usm_weak.apply(image)
        r_strong = usm_strong.apply(image)

        # Stronger sharpening → larger overshoot at edges
        assert np.abs(np.diff(r_strong[25, :])).max() > np.abs(np.diff(r_weak[25, :])).max()


# ============================================================================
# FFTBandpassFilter Tests
# ============================================================================

class TestFFTBandpassFilter:
    """Tests for FFT Bandpass Filter."""

    def test_import(self):
        from grdl.imagej import FFTBandpassFilter
        assert FFTBandpassFilter is not None

    def test_version_attribute(self):
        from grdl.imagej import FFTBandpassFilter
        assert FFTBandpassFilter.__imagej_version__ == '1.54j'
        assert FFTBandpassFilter.__processor_version__ == '1.54j'

    def test_removes_dc_gradient(self):
        """Large-structure filter should remove smooth gradients."""
        from grdl.imagej import FFTBandpassFilter
        bp = FFTBandpassFilter(filter_large=20, filter_small=0, autoscale=False)
        rows, cols = 64, 64
        gradient = np.tile(np.linspace(0, 200, cols), (rows, 1))
        result = bp.apply(gradient)
        # Gradient (large structure) should be removed
        assert result.std() < gradient.std() * 0.5

    def test_preserves_mid_frequency(self):
        """Mid-frequency sinusoid should pass through bandpass filter."""
        from grdl.imagej import FFTBandpassFilter
        bp = FFTBandpassFilter(filter_large=100, filter_small=2)
        rows, cols = 128, 128
        x = np.arange(cols) / cols
        # Period ~16 pixels (mid-frequency)
        sine = np.sin(2 * np.pi * 8 * x)
        image = np.tile(sine, (rows, 1)) * 100
        result = bp.apply(image)
        # Mid-frequency should survive
        assert result.std() > 10.0

    def test_output_shape(self):
        from grdl.imagej import FFTBandpassFilter
        bp = FFTBandpassFilter()
        image = np.random.RandomState(0).rand(60, 80)
        result = bp.apply(image)
        assert result.shape == (60, 80)

    def test_autoscale_matches_stats(self):
        """With autoscale=True, output should have similar stats to input."""
        from grdl.imagej import FFTBandpassFilter
        bp = FFTBandpassFilter(filter_large=30, filter_small=3, autoscale=True)
        image = np.random.RandomState(42).rand(64, 64) * 200 + 50
        result = bp.apply(image)
        # Mean should be approximately preserved
        assert abs(result.mean() - image.mean()) < image.std() * 0.5

    def test_no_autoscale(self):
        from grdl.imagej import FFTBandpassFilter
        bp = FFTBandpassFilter(filter_large=20, filter_small=3, autoscale=False)
        image = np.random.RandomState(0).rand(64, 64) * 100
        result = bp.apply(image)
        assert result.shape == (64, 64)

    def test_stripe_suppression_horizontal(self):
        """Horizontal stripe removal."""
        from grdl.imagej import FFTBandpassFilter
        bp = FFTBandpassFilter(
            filter_large=0, filter_small=0,
            suppress_stripes='horizontal', stripe_tolerance=5.0,
            autoscale=False,
        )
        rows, cols = 64, 64
        image = np.zeros((rows, cols))
        # Add horizontal stripes
        for r in range(0, rows, 4):
            image[r, :] = 100.0
        result = bp.apply(image)
        # Measure stripe strength: variance of row means
        orig_stripe = np.var(image.mean(axis=1))
        result_stripe = np.var(result.mean(axis=1))
        assert result_stripe < orig_stripe

    def test_stripe_suppression_vertical(self):
        """Vertical stripe removal."""
        from grdl.imagej import FFTBandpassFilter
        bp = FFTBandpassFilter(
            filter_large=0, filter_small=0,
            suppress_stripes='vertical', stripe_tolerance=5.0,
            autoscale=False,
        )
        rows, cols = 64, 64
        image = np.zeros((rows, cols))
        for c in range(0, cols, 4):
            image[:, c] = 100.0
        result = bp.apply(image)
        # Measure stripe strength: variance of column means
        orig_stripe = np.var(image.mean(axis=0))
        result_stripe = np.var(result.mean(axis=0))
        assert result_stripe < orig_stripe

    def test_rejects_non_2d(self):
        from grdl.imagej import FFTBandpassFilter
        bp = FFTBandpassFilter()
        with pytest.raises(ValueError, match="2D"):
            bp.apply(np.zeros((3, 10, 10)))

    def test_invalid_params(self):
        from grdl.imagej import FFTBandpassFilter
        with pytest.raises(ValueError):
            FFTBandpassFilter(filter_large=-1)
        with pytest.raises(ValueError):
            FFTBandpassFilter(filter_small=-1)
        with pytest.raises(ValueError):
            FFTBandpassFilter(suppress_stripes='diagonal')


# ============================================================================
# ZProjection Tests
# ============================================================================

class TestZProjection:
    """Tests for Z-Projection."""

    def test_import(self):
        from grdl.imagej import ZProjection
        assert ZProjection is not None

    def test_version_attribute(self):
        from grdl.imagej import ZProjection
        assert ZProjection.__imagej_version__ == '1.54j'
        assert ZProjection.__processor_version__ == '1.54j'

    def test_average_projection(self):
        from grdl.imagej import ZProjection
        zp = ZProjection(method='average')
        stack = np.array([[[1, 2], [3, 4]],
                          [[5, 6], [7, 8]],
                          [[3, 3], [3, 3]]], dtype=np.float64)
        result = zp.apply(stack)
        expected = np.array([[3, 11 / 3], [13 / 3, 5]])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_max_projection(self):
        from grdl.imagej import ZProjection
        zp = ZProjection(method='max')
        stack = np.array([[[1, 5], [3, 2]],
                          [[4, 2], [6, 8]]], dtype=np.float64)
        result = zp.apply(stack)
        expected = np.array([[4, 5], [6, 8]])
        np.testing.assert_array_equal(result, expected)

    def test_min_projection(self):
        from grdl.imagej import ZProjection
        zp = ZProjection(method='min')
        stack = np.array([[[1, 5], [3, 2]],
                          [[4, 2], [6, 8]]], dtype=np.float64)
        result = zp.apply(stack)
        expected = np.array([[1, 2], [3, 2]])
        np.testing.assert_array_equal(result, expected)

    def test_sum_projection(self):
        from grdl.imagej import ZProjection
        zp = ZProjection(method='sum')
        stack = np.ones((5, 10, 10), dtype=np.float64) * 3.0
        result = zp.apply(stack)
        np.testing.assert_allclose(result, 15.0)

    def test_std_projection(self):
        from grdl.imagej import ZProjection
        zp = ZProjection(method='std')
        stack = np.array([[[0], [0]],
                          [[10], [10]]], dtype=np.float64)
        result = zp.apply(stack)
        # std of [0, 10] = 5.0
        np.testing.assert_allclose(result, 5.0, atol=1e-10)

    def test_median_projection(self):
        from grdl.imagej import ZProjection
        zp = ZProjection(method='median')
        stack = np.array([[[1], [3]],
                          [[5], [1]],
                          [[3], [5]]], dtype=np.float64)
        result = zp.apply(stack)
        expected = np.array([[3], [3]])
        np.testing.assert_array_equal(result, expected)

    def test_slice_range(self):
        """start_slice and stop_slice should select subset of stack."""
        from grdl.imagej import ZProjection
        zp = ZProjection(method='sum', start_slice=1, stop_slice=3)
        stack = np.arange(40, dtype=np.float64).reshape(4, 2, 5)
        result = zp.apply(stack)
        expected = stack[1:3].sum(axis=0)
        np.testing.assert_array_equal(result, expected)

    def test_output_shape_2d(self):
        from grdl.imagej import ZProjection
        zp = ZProjection()
        stack = np.random.RandomState(0).rand(10, 50, 60)
        result = zp.apply(stack)
        assert result.shape == (50, 60)
        assert result.ndim == 2

    def test_rejects_non_3d(self):
        from grdl.imagej import ZProjection
        zp = ZProjection()
        with pytest.raises(ValueError, match="3D"):
            zp.apply(np.zeros((10, 10)))

    def test_invalid_method(self):
        from grdl.imagej import ZProjection
        with pytest.raises(ValueError, match="Unknown method"):
            ZProjection(method='mode')

    def test_single_slice_stack(self):
        """Stack with one slice should return that slice."""
        from grdl.imagej import ZProjection
        zp = ZProjection(method='max')
        stack = np.random.RandomState(0).rand(1, 20, 30)
        result = zp.apply(stack)
        np.testing.assert_array_equal(result, stack[0])

    def test_all_methods_consistent_on_uniform(self):
        """All methods on a uniform stack should return the same value."""
        from grdl.imagej import ZProjection
        stack = np.full((5, 10, 10), 42.0)
        for method in ['average', 'max', 'min', 'median']:
            zp = ZProjection(method=method)
            result = zp.apply(stack)
            np.testing.assert_allclose(result, 42.0, atol=1e-10,
                                       err_msg=f"Failed for {method}")


# ============================================================================
# RankFilters Tests
# ============================================================================

class TestRankFilters:
    """Tests for Rank Filters."""

    def test_import_and_version(self):
        from grdl.imagej import RankFilters
        assert RankFilters.__imagej_version__ == '1.54j'
        assert RankFilters.__processor_version__ == '1.54j'

    def test_median_removes_salt_pepper(self):
        """Median filter should remove impulse noise."""
        from grdl.imagej import RankFilters
        rng = np.random.RandomState(42)
        image = np.full((50, 50), 100.0)
        # Add salt-and-pepper noise
        noise_mask = rng.rand(50, 50) < 0.05
        image[noise_mask] = 255.0
        noise_mask2 = rng.rand(50, 50) < 0.05
        image[noise_mask2] = 0.0

        mf = RankFilters(method='median', radius=1)
        result = mf.apply(image)
        # After filtering, should be closer to 100
        assert abs(result.mean() - 100) < abs(image.mean() - 100)

    def test_min_shrinks_bright(self):
        """Min filter should shrink bright regions."""
        from grdl.imagej import RankFilters
        image = np.zeros((30, 30))
        image[10:20, 10:20] = 200.0
        rf = RankFilters(method='min', radius=1)
        result = rf.apply(image)
        # Bright area should be smaller
        assert (result > 100).sum() < (image > 100).sum()

    def test_max_expands_bright(self):
        """Max filter should expand bright regions."""
        from grdl.imagej import RankFilters
        image = np.zeros((30, 30))
        image[10:20, 10:20] = 200.0
        rf = RankFilters(method='max', radius=1)
        result = rf.apply(image)
        assert (result > 100).sum() > (image > 100).sum()

    def test_variance_detects_edges(self):
        """Variance filter should be high at edges."""
        from grdl.imagej import RankFilters
        image = np.zeros((40, 40))
        image[:, 20:] = 200.0
        vf = RankFilters(method='variance', radius=2)
        result = vf.apply(image)
        # Variance should be high near the edge (column 20)
        assert result[:, 18:22].mean() > result[:, 0:5].mean()

    def test_despeckle_is_3x3_median(self):
        """Despeckle should be equivalent to median with radius=1."""
        from grdl.imagej import RankFilters
        image = np.random.RandomState(7).rand(30, 30) * 200
        d = RankFilters(method='despeckle')
        m = RankFilters(method='median', radius=1)
        np.testing.assert_array_equal(d.apply(image), m.apply(image))

    def test_all_methods_run(self):
        from grdl.imagej import RankFilters
        image = np.random.RandomState(0).rand(30, 30) * 255
        for method in ('median', 'min', 'max', 'mean', 'variance', 'despeckle'):
            rf = RankFilters(method=method, radius=2)
            result = rf.apply(image)
            assert result.shape == (30, 30), f"Failed for {method}"

    def test_rejects_non_2d(self):
        from grdl.imagej import RankFilters
        rf = RankFilters()
        with pytest.raises(ValueError, match="2D"):
            rf.apply(np.zeros((3, 10, 10)))

    def test_invalid_method(self):
        from grdl.imagej import RankFilters
        with pytest.raises(ValueError):
            RankFilters(method='percentile')


# ============================================================================
# MorphologicalFilter Tests
# ============================================================================

class TestMorphologicalFilter:
    """Tests for Binary and Grayscale Morphological Operations."""

    def test_import_and_version(self):
        from grdl.imagej import MorphologicalFilter
        assert MorphologicalFilter.__imagej_version__ == '1.54j'

    def test_erode_shrinks_binary(self):
        from grdl.imagej import MorphologicalFilter
        image = np.zeros((30, 30))
        image[10:20, 10:20] = 1.0
        mf = MorphologicalFilter(operation='erode', radius=1)
        result = mf.apply(image)
        assert result.sum() < image.sum()

    def test_dilate_expands_binary(self):
        from grdl.imagej import MorphologicalFilter
        image = np.zeros((30, 30))
        image[10:20, 10:20] = 1.0
        mf = MorphologicalFilter(operation='dilate', radius=1)
        result = mf.apply(image)
        assert result.sum() > image.sum()

    def test_open_removes_small_noise(self):
        """Opening should remove small isolated pixels."""
        from grdl.imagej import MorphologicalFilter
        image = np.zeros((30, 30))
        image[15, 15] = 1.0  # Single isolated pixel
        image[5:10, 5:10] = 1.0  # Large region
        mf = MorphologicalFilter(operation='open', radius=1)
        result = mf.apply(image)
        # Isolated pixel should be removed, large region mostly kept
        assert result[15, 15] == 0.0
        assert result[7, 7] == 1.0

    def test_close_fills_small_holes(self):
        """Closing should fill small holes."""
        from grdl.imagej import MorphologicalFilter
        image = np.ones((30, 30))
        image[15, 15] = 0.0  # Small hole
        mf = MorphologicalFilter(operation='close', radius=1)
        result = mf.apply(image)
        assert result[15, 15] == 1.0

    def test_tophat_extracts_small_bright(self):
        """Top-hat should extract small bright features."""
        from grdl.imagej import MorphologicalFilter
        image = np.full((40, 40), 50.0)
        image[18:22, 18:22] = 200.0  # Small bright feature
        mf = MorphologicalFilter(operation='tophat', radius=3)
        result = mf.apply(image)
        # The bright spot should be in the top-hat result
        assert result[20, 20] > result[0, 0]

    def test_gradient_detects_boundaries(self):
        """Gradient should highlight region boundaries."""
        from grdl.imagej import MorphologicalFilter
        image = np.zeros((30, 30))
        image[10:20, 10:20] = 1.0
        mf = MorphologicalFilter(operation='gradient', radius=1)
        result = mf.apply(image)
        # Interior and exterior should be ~0, boundary should be > 0
        assert result[15, 15] < 0.5  # Interior
        assert result[10, 15] > 0.5 or result[9, 15] > 0.5  # Boundary

    def test_all_operations_run(self):
        from grdl.imagej import MorphologicalFilter
        image = np.random.RandomState(0).rand(30, 30)
        for op in ('erode', 'dilate', 'open', 'close', 'tophat', 'blackhat', 'gradient'):
            mf = MorphologicalFilter(operation=op, radius=1)
            result = mf.apply(image)
            assert result.shape == (30, 30), f"Failed for {op}"

    def test_kernel_shapes(self):
        from grdl.imagej import MorphologicalFilter
        image = np.random.RandomState(0).rand(20, 20)
        for shape in ('square', 'cross', 'disk'):
            mf = MorphologicalFilter(operation='erode', kernel_shape=shape)
            result = mf.apply(image)
            assert result.shape == (20, 20)

    def test_rejects_non_2d(self):
        from grdl.imagej import MorphologicalFilter
        mf = MorphologicalFilter()
        with pytest.raises(ValueError, match="2D"):
            mf.apply(np.zeros((3, 10, 10)))

    def test_invalid_operation(self):
        from grdl.imagej import MorphologicalFilter
        with pytest.raises(ValueError):
            MorphologicalFilter(operation='skeletonize')


# ============================================================================
# EdgeDetector Tests
# ============================================================================

class TestEdgeDetector:
    """Tests for Edge Detection filters."""

    def test_import_and_version(self):
        from grdl.imagej import EdgeDetector
        assert EdgeDetector.__imagej_version__ == '1.54j'

    def test_sobel_detects_step_edge(self):
        """Sobel should produce strong response at a step edge."""
        from grdl.imagej import EdgeDetector
        image = np.zeros((40, 80))
        image[:, 40:] = 200.0
        ed = EdgeDetector(method='sobel')
        result = ed.apply(image)
        # Edge response should be high near column 40
        assert result[:, 38:42].mean() > result[:, 0:5].mean() * 5

    def test_all_methods_run(self):
        from grdl.imagej import EdgeDetector
        image = np.random.RandomState(0).rand(40, 40) * 200
        for method in ('sobel', 'prewitt', 'roberts', 'log', 'scharr'):
            ed = EdgeDetector(method=method)
            result = ed.apply(image)
            assert result.shape == (40, 40), f"Failed for {method}"
            assert np.all(result >= 0), f"Negative values for {method}"

    def test_flat_image_zero_edges(self):
        """Flat image should have near-zero edge response."""
        from grdl.imagej import EdgeDetector
        flat = np.full((30, 30), 128.0)
        ed = EdgeDetector(method='sobel')
        result = ed.apply(flat)
        assert result.max() < 0.01

    def test_log_sigma_affects_scale(self):
        """Larger LoG sigma should smooth over finer detail."""
        from grdl.imagej import EdgeDetector
        image = np.zeros((60, 60))
        image[:, 30:] = 200.0
        ed_fine = EdgeDetector(method='log', sigma=0.5)
        ed_coarse = EdgeDetector(method='log', sigma=3.0)
        r_fine = ed_fine.apply(image)
        r_coarse = ed_coarse.apply(image)
        # Coarser sigma → wider but lower edge response
        assert r_fine.max() > r_coarse.max()

    def test_nonnegative_output(self):
        """Edge magnitude is always non-negative."""
        from grdl.imagej import EdgeDetector
        image = np.random.RandomState(5).rand(30, 30) * 200 - 50
        for method in ('sobel', 'prewitt', 'roberts', 'scharr'):
            ed = EdgeDetector(method=method)
            result = ed.apply(image)
            assert np.all(result >= -1e-10), f"Negative for {method}"

    def test_rejects_non_2d(self):
        from grdl.imagej import EdgeDetector
        ed = EdgeDetector()
        with pytest.raises(ValueError, match="2D"):
            ed.apply(np.zeros((3, 10, 10)))

    def test_invalid_method(self):
        from grdl.imagej import EdgeDetector
        with pytest.raises(ValueError):
            EdgeDetector(method='canny')


# ============================================================================
# GammaCorrection Tests
# ============================================================================

class TestGammaCorrection:
    """Tests for Gamma Correction."""

    def test_import_and_version(self):
        from grdl.imagej import GammaCorrection
        assert GammaCorrection.__imagej_version__ == '1.54j'

    def test_gamma_1_is_identity(self):
        """Gamma=1.0 should return the original image."""
        from grdl.imagej import GammaCorrection
        gc = GammaCorrection(gamma=1.0)
        image = np.random.RandomState(42).rand(30, 30) * 200
        result = gc.apply(image)
        np.testing.assert_allclose(result, image, atol=1e-10)

    def test_gamma_less_than_1_brightens(self):
        """Gamma < 1 should increase mid-tone values."""
        from grdl.imagej import GammaCorrection
        gc = GammaCorrection(gamma=0.5)
        image = np.linspace(0, 255, 100).reshape(10, 10)
        result = gc.apply(image)
        # Mid-tones should be brighter (higher)
        mid_idx = 5 * 10 + 5
        assert result.ravel()[mid_idx] > image.ravel()[mid_idx]

    def test_gamma_greater_than_1_darkens(self):
        """Gamma > 1 should decrease mid-tone values."""
        from grdl.imagej import GammaCorrection
        gc = GammaCorrection(gamma=2.0)
        image = np.linspace(0, 255, 100).reshape(10, 10)
        result = gc.apply(image)
        mid_idx = 5 * 10 + 5
        assert result.ravel()[mid_idx] < image.ravel()[mid_idx]

    def test_preserves_range(self):
        """Output should have same min/max as input."""
        from grdl.imagej import GammaCorrection
        gc = GammaCorrection(gamma=0.4)
        image = np.random.RandomState(3).rand(30, 30) * 200 + 10
        result = gc.apply(image)
        np.testing.assert_allclose(result.min(), image.min(), atol=1e-10)
        np.testing.assert_allclose(result.max(), image.max(), atol=1e-10)

    def test_flat_image_unchanged(self):
        from grdl.imagej import GammaCorrection
        gc = GammaCorrection(gamma=0.5)
        flat = np.full((20, 20), 100.0)
        result = gc.apply(flat)
        np.testing.assert_allclose(result, 100.0, atol=1e-10)

    def test_rejects_non_2d(self):
        from grdl.imagej import GammaCorrection
        gc = GammaCorrection()
        with pytest.raises(ValueError, match="2D"):
            gc.apply(np.zeros((3, 10, 10)))

    def test_invalid_gamma(self):
        from grdl.imagej import GammaCorrection
        with pytest.raises(ValueError):
            GammaCorrection(gamma=0)
        with pytest.raises(ValueError):
            GammaCorrection(gamma=-1)


# ============================================================================
# FindMaxima Tests
# ============================================================================

class TestFindMaxima:
    """Tests for Find Maxima peak detection."""

    def test_import_and_version(self):
        from grdl.imagej import FindMaxima
        assert FindMaxima.__imagej_version__ == '1.54j'

    def test_detects_isolated_peak(self):
        """A single bright peak should be detected."""
        from grdl.imagej import FindMaxima
        image = np.zeros((50, 50))
        image[25, 25] = 100.0
        fm = FindMaxima(prominence=10.0)
        result = fm.apply(image)
        assert result[25, 25] == 1.0

    def test_rejects_low_prominence(self):
        """Peaks below prominence threshold should not be detected."""
        from grdl.imagej import FindMaxima
        image = np.full((30, 30), 100.0)
        image[15, 15] = 105.0  # Only 5 above background
        fm = FindMaxima(prominence=20.0)
        result = fm.apply(image)
        assert result.sum() == 0.0

    def test_multiple_peaks(self):
        """Multiple well-separated peaks should all be detected."""
        from grdl.imagej import FindMaxima
        image = np.zeros((50, 50))
        image[10, 10] = 100.0
        image[10, 40] = 100.0
        image[40, 10] = 100.0
        image[40, 40] = 100.0
        fm = FindMaxima(prominence=10.0)
        result = fm.apply(image)
        assert result.sum() >= 4.0

    def test_count_map_output(self):
        """count_map should label each peak region with a unique integer."""
        from grdl.imagej import FindMaxima
        image = np.zeros((50, 50))
        image[10, 10] = 100.0
        image[40, 40] = 100.0
        fm = FindMaxima(prominence=10.0, output='count_map')
        result = fm.apply(image)
        assert result.max() >= 2.0

    def test_find_peaks_method(self):
        """find_peaks() should return coordinate array."""
        from grdl.imagej import FindMaxima
        image = np.zeros((30, 30))
        image[15, 15] = 100.0
        fm = FindMaxima(prominence=10.0)
        coords = fm.find_peaks(image)
        assert coords.shape[1] == 2
        assert len(coords) >= 1

    def test_exclude_on_edges(self):
        """Edge maxima should be excluded when flag is set."""
        from grdl.imagej import FindMaxima
        image = np.zeros((30, 30))
        image[0, 15] = 100.0  # Edge peak
        image[15, 15] = 100.0  # Interior peak
        fm = FindMaxima(prominence=10.0, exclude_on_edges=True)
        result = fm.apply(image)
        assert result[0, 15] == 0.0
        assert result[15, 15] == 1.0

    def test_rejects_non_2d(self):
        from grdl.imagej import FindMaxima
        fm = FindMaxima()
        with pytest.raises(ValueError, match="2D"):
            fm.apply(np.zeros((3, 10, 10)))

    def test_no_peaks_returns_empty(self):
        """Flat image should return no peaks."""
        from grdl.imagej import FindMaxima
        fm = FindMaxima(prominence=10.0)
        coords = fm.find_peaks(np.full((20, 20), 50.0))
        assert coords.shape == (0, 2)


# ============================================================================
# StatisticalRegionMerging Tests
# ============================================================================

class TestStatisticalRegionMerging:
    """Tests for Statistical Region Merging segmentation."""

    def test_import_and_version(self):
        from grdl.imagej import StatisticalRegionMerging
        assert StatisticalRegionMerging.__imagej_version__ == '1.0'

    def test_uniform_image_single_region(self):
        """A uniform image should produce a single region."""
        from grdl.imagej import StatisticalRegionMerging
        srm = StatisticalRegionMerging(Q=25)
        flat = np.full((20, 20), 100.0)
        labels = srm.apply(flat)
        assert labels.max() == 1.0  # Single region

    def test_two_distinct_regions(self):
        """Two clearly separated intensity regions should be segmented."""
        from grdl.imagej import StatisticalRegionMerging
        srm = StatisticalRegionMerging(Q=10)
        image = np.zeros((20, 40))
        image[:, 20:] = 200.0
        labels = srm.apply(image)
        # Should have at least 2 regions
        assert labels.max() >= 2.0
        # Left and right halves should have different labels
        assert labels[10, 5] != labels[10, 35]

    def test_mean_output(self):
        """Mean output should replace pixels with region means."""
        from grdl.imagej import StatisticalRegionMerging
        srm = StatisticalRegionMerging(Q=10, output='mean')
        image = np.zeros((20, 40))
        image[:, 20:] = 200.0
        result = srm.apply(image)
        # Left half should be ~0, right half should be ~200
        assert abs(result[10, 5] - 0.0) < 50
        assert abs(result[10, 35] - 200.0) < 50

    def test_higher_q_more_regions(self):
        """Larger Q should produce more (finer) regions."""
        from grdl.imagej import StatisticalRegionMerging
        rng = np.random.RandomState(42)
        image = rng.rand(20, 20) * 100 + 50
        srm_coarse = StatisticalRegionMerging(Q=5)
        srm_fine = StatisticalRegionMerging(Q=200)
        labels_coarse = srm_coarse.apply(image)
        labels_fine = srm_fine.apply(image)
        assert labels_fine.max() >= labels_coarse.max()

    def test_output_shape(self):
        from grdl.imagej import StatisticalRegionMerging
        srm = StatisticalRegionMerging(Q=25)
        image = np.random.RandomState(0).rand(15, 25) * 200
        result = srm.apply(image)
        assert result.shape == (15, 25)

    def test_labels_are_positive_integers(self):
        from grdl.imagej import StatisticalRegionMerging
        srm = StatisticalRegionMerging(Q=25)
        image = np.random.RandomState(0).rand(15, 15) * 200
        labels = srm.apply(image)
        assert labels.min() >= 1.0

    def test_rejects_non_2d(self):
        from grdl.imagej import StatisticalRegionMerging
        srm = StatisticalRegionMerging()
        with pytest.raises(ValueError, match="2D"):
            srm.apply(np.zeros((3, 10, 10)))

    def test_invalid_q(self):
        from grdl.imagej import StatisticalRegionMerging
        with pytest.raises(ValueError):
            StatisticalRegionMerging(Q=0)
        with pytest.raises(ValueError):
            StatisticalRegionMerging(Q=-5)


# ============================================================================
# Module-level integration tests
# ============================================================================

ALL_CLASSES = None

def _get_all_classes():
    global ALL_CLASSES
    if ALL_CLASSES is None:
        from grdl.imagej import (
            RollingBallBackground, CLAHE, AutoLocalThreshold,
            UnsharpMask, FFTBandpassFilter, ZProjection,
            RankFilters, MorphologicalFilter, EdgeDetector,
            GammaCorrection, FindMaxima, StatisticalRegionMerging,
        )
        ALL_CLASSES = [
            RollingBallBackground, CLAHE, AutoLocalThreshold,
            UnsharpMask, FFTBandpassFilter, ZProjection,
            RankFilters, MorphologicalFilter, EdgeDetector,
            GammaCorrection, FindMaxima, StatisticalRegionMerging,
        ]
    return ALL_CLASSES


class TestModuleExports:
    """Verify all 12 components are importable and properly configured."""

    def test_all_exports(self):
        classes = _get_all_classes()
        assert len(classes) == 12
        assert all(cls is not None for cls in classes)

    def test_all_inherit_from_image_transform(self):
        from grdl.image_processing.base import ImageTransform
        for cls in _get_all_classes():
            assert issubclass(cls, ImageTransform), (
                f"{cls.__name__} is not a subclass of ImageTransform"
            )

    def test_all_have_imagej_version(self):
        for cls in _get_all_classes():
            assert hasattr(cls, '__imagej_version__'), (
                f"{cls.__name__} missing __imagej_version__"
            )
            assert hasattr(cls, '__imagej_source__'), (
                f"{cls.__name__} missing __imagej_source__"
            )

    def test_all_have_processor_version(self):
        for cls in _get_all_classes():
            assert hasattr(cls, '__processor_version__'), (
                f"{cls.__name__} missing __processor_version__"
            )
