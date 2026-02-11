# -*- coding: utf-8 -*-
"""
Normalizer Tests - Comprehensive tests for intensity normalization.

Tests all four normalization methods (minmax, zscore, percentile,
unit_norm), fit/transform semantics, edge cases, and input validation.

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

from grdl.data_prep import Normalizer


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestNormalizerInit:
    """Test constructor validation and defaults."""

    def test_default_method_is_minmax(self):
        norm = Normalizer()
        assert norm.method == 'minmax'

    def test_all_valid_methods(self):
        for method in ('minmax', 'zscore', 'percentile', 'unit_norm'):
            norm = Normalizer(method=method)
            assert norm.method == method

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be one of"):
            Normalizer(method='invalid')

    def test_percentile_low_out_of_range_raises(self):
        with pytest.raises(ValueError, match="percentile_low"):
            Normalizer(percentile_low=-1.0)

    def test_percentile_high_out_of_range_raises(self):
        with pytest.raises(ValueError, match="percentile_high"):
            Normalizer(percentile_high=101.0)

    def test_percentile_low_ge_high_raises(self):
        with pytest.raises(ValueError, match="must be less than"):
            Normalizer(percentile_low=50.0, percentile_high=50.0)

    def test_not_fitted_initially(self):
        norm = Normalizer()
        assert norm.is_fitted is False


# ---------------------------------------------------------------------------
# MinMax normalization
# ---------------------------------------------------------------------------

class TestNormalizerMinMax:
    """Test minmax normalization: values scaled to [0, 1]."""

    def test_known_values(self):
        norm = Normalizer(method='minmax')
        data = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        result = norm.normalize(data)
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(result, expected)

    def test_output_range(self):
        norm = Normalizer(method='minmax')
        rng = np.random.RandomState(42)
        data = rng.rand(100, 100) * 500 - 200  # range [-200, 300]
        result = norm.normalize(data)
        assert result.min() == pytest.approx(0.0, abs=1e-12)
        assert result.max() == pytest.approx(1.0, abs=1e-12)

    def test_constant_array_returns_zeros(self):
        """Constant array has zero range — should return zeros, not NaN."""
        norm = Normalizer(method='minmax')
        data = np.full((20, 20), 42.0)
        result = norm.normalize(data)
        np.testing.assert_array_equal(result, np.zeros_like(result))

    def test_preserves_shape_2d(self):
        norm = Normalizer(method='minmax')
        data = np.arange(100.0).reshape(10, 10)
        result = norm.normalize(data)
        assert result.shape == (10, 10)

    def test_preserves_shape_3d(self):
        norm = Normalizer(method='minmax')
        data = np.arange(60.0).reshape(3, 4, 5)
        result = norm.normalize(data)
        assert result.shape == (3, 4, 5)

    def test_negative_values(self):
        norm = Normalizer(method='minmax')
        data = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])
        result = norm.normalize(data)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_single_element_returns_zero(self):
        norm = Normalizer(method='minmax')
        data = np.array([7.0])
        result = norm.normalize(data)
        assert result[0] == 0.0

    def test_output_dtype_float64(self):
        norm = Normalizer(method='minmax')
        data = np.array([1, 2, 3], dtype=np.int32)
        result = norm.normalize(data)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# Z-score normalization
# ---------------------------------------------------------------------------

class TestNormalizerZScore:
    """Test zscore normalization: mean≈0, std≈1."""

    def test_known_statistics(self):
        norm = Normalizer(method='zscore')
        rng = np.random.RandomState(42)
        data = rng.randn(10000) * 5 + 100  # mean≈100, std≈5
        result = norm.normalize(data)
        assert abs(result.mean()) < 0.05
        assert abs(result.std() - 1.0) < 0.05

    def test_constant_array_returns_zeros(self):
        """Constant array has zero std — should return zeros, not NaN."""
        norm = Normalizer(method='zscore')
        data = np.full(50, 42.0)
        result = norm.normalize(data)
        np.testing.assert_array_equal(result, np.zeros(50))

    def test_already_standardized(self):
        """Data with mean~0, std~1 should produce mean~0, std~1 after zscore."""
        norm = Normalizer(method='zscore')
        rng = np.random.RandomState(7)
        data = rng.randn(10000)
        result = norm.normalize(data)
        assert abs(result.mean()) < 0.05
        assert abs(result.std() - 1.0) < 0.05


# ---------------------------------------------------------------------------
# Percentile normalization
# ---------------------------------------------------------------------------

class TestNormalizerPercentile:
    """Test percentile normalization: clips then scales to [0, 1]."""

    def test_clips_outliers(self):
        """Values outside percentile range should be clipped."""
        norm = Normalizer(method='percentile', percentile_low=10.0,
                          percentile_high=90.0)
        data = np.arange(1000, dtype=np.float64)
        result = norm.normalize(data)
        # Should be in [0, 1]
        assert result.min() >= -0.01
        assert result.max() <= 1.01

    def test_known_percentile_range(self):
        """Uniform data: p2=2, p98=98 should map 0-100 → ~[0,1]."""
        norm = Normalizer(method='percentile')
        data = np.arange(101, dtype=np.float64)  # 0 to 100
        result = norm.normalize(data)
        # Middle values should be properly scaled
        assert result[50] == pytest.approx(0.5, abs=0.05)

    def test_custom_percentile_bounds(self):
        """Custom bounds should use different clip range."""
        norm25_75 = Normalizer(method='percentile',
                               percentile_low=25.0, percentile_high=75.0)
        data = np.arange(100, dtype=np.float64)
        result = norm25_75.normalize(data)
        # Values at p25 should map to ~0, values at p75 to ~1
        assert result[25] == pytest.approx(0.0, abs=0.05)
        assert result[75] == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# Unit-norm normalization
# ---------------------------------------------------------------------------

class TestNormalizerUnitNorm:
    """Test unit_norm normalization: divides by L2 norm."""

    def test_known_values(self):
        norm = Normalizer(method='unit_norm')
        data = np.array([3.0, 4.0])  # L2 norm = 5
        result = norm.normalize(data)
        np.testing.assert_allclose(result, [0.6, 0.8])

    def test_output_has_unit_norm(self):
        norm = Normalizer(method='unit_norm')
        rng = np.random.RandomState(42)
        data = rng.rand(50) * 100
        result = norm.normalize(data)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-10)

    def test_zero_array_returns_zeros(self):
        """All-zero array has zero norm — should return zeros, not NaN."""
        norm = Normalizer(method='unit_norm')
        data = np.zeros(20)
        result = norm.normalize(data)
        np.testing.assert_array_equal(result, np.zeros(20))


# ---------------------------------------------------------------------------
# Fit / Transform semantics
# ---------------------------------------------------------------------------

class TestNormalizerFitTransform:
    """Test fit/transform workflow."""

    def test_fit_sets_is_fitted(self):
        norm = Normalizer(method='minmax')
        data = np.arange(10, dtype=np.float64)
        norm.fit(data)
        assert norm.is_fitted is True

    def test_transform_before_fit_raises(self):
        norm = Normalizer(method='minmax')
        with pytest.raises(RuntimeError, match="not been fitted"):
            norm.transform(np.array([1.0, 2.0]))

    def test_fit_transform_matches_normalize(self):
        """fit_transform should produce same result as normalize."""
        norm_stateless = Normalizer(method='minmax')
        norm_fitted = Normalizer(method='minmax')
        data = np.arange(100, dtype=np.float64)
        result_stateless = norm_stateless.normalize(data)
        result_fitted = norm_fitted.fit_transform(data)
        np.testing.assert_allclose(result_fitted, result_stateless)

    def test_fit_on_training_transform_on_test(self):
        """Transform on test data uses training statistics."""
        norm = Normalizer(method='minmax')
        train = np.array([0.0, 50.0, 100.0])
        test = np.array([25.0, 75.0, 125.0])
        norm.fit(train)
        result = norm.transform(test)
        # 25/100=0.25, 75/100=0.75, 125/100=1.25 (exceeds 1.0)
        np.testing.assert_allclose(result, [0.25, 0.75, 1.25])

    def test_fit_returns_self(self):
        norm = Normalizer(method='zscore')
        result = norm.fit(np.arange(10, dtype=np.float64))
        assert result is norm

    def test_zscore_fit_transform_uses_training_stats(self):
        """Z-score transform on test data uses training mean/std."""
        norm = Normalizer(method='zscore')
        train = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        norm.fit(train)
        # train mean=30, std≈14.14
        test = np.array([30.0])  # should map to ~0
        result = norm.transform(test)
        assert abs(result[0]) < 0.01

    def test_unit_norm_fit_transform_uses_training_norm(self):
        """Unit-norm transform on test data uses training L2 norm."""
        norm = Normalizer(method='unit_norm')
        train = np.array([3.0, 4.0])  # L2 norm = 5
        norm.fit(train)
        test = np.array([5.0, 0.0])
        result = norm.transform(test)
        np.testing.assert_allclose(result, [1.0, 0.0])


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestNormalizerInputValidation:
    """Test input validation on normalize/fit/transform."""

    def test_non_array_raises_type_error(self):
        norm = Normalizer()
        with pytest.raises(TypeError, match="np.ndarray"):
            norm.normalize([1, 2, 3])

    def test_list_raises_type_error_on_fit(self):
        norm = Normalizer()
        with pytest.raises(TypeError, match="np.ndarray"):
            norm.fit([1.0, 2.0])

    def test_list_raises_type_error_on_transform(self):
        norm = Normalizer()
        norm.fit(np.array([1.0, 2.0]))
        with pytest.raises(TypeError, match="np.ndarray"):
            norm.transform([1.0, 2.0])

    def test_integer_input_returns_float(self):
        norm = Normalizer(method='minmax')
        data = np.array([0, 50, 100], dtype=np.int32)
        result = norm.normalize(data)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])
