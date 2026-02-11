# -*- coding: utf-8 -*-
"""
Tests for grdl.image_processing.intensity — ToDecibels and PercentileStretch.

Author
------
Steven Siebert

Created
-------
2026-02-11
"""

import numpy as np
import pytest

from grdl.image_processing.intensity import ToDecibels, PercentileStretch


# ---------------------------------------------------------------------------
# ToDecibels
# ---------------------------------------------------------------------------

class TestToDecibels:
    def test_real_valued_input(self):
        to_db = ToDecibels()
        source = np.array([1.0, 10.0, 100.0])
        result = to_db.apply(source)
        np.testing.assert_array_almost_equal(result, [0.0, 20.0, 40.0], decimal=2)

    def test_complex_valued_input(self):
        to_db = ToDecibels()
        source = np.array([1 + 0j, 0 + 10j, 100 + 0j])
        result = to_db.apply(source)
        np.testing.assert_array_almost_equal(result, [0.0, 20.0, 40.0], decimal=2)

    def test_floor_clamps_low_values(self):
        to_db = ToDecibels(floor_db=-30.0)
        source = np.array([1e-10, 1.0])
        result = to_db.apply(source)
        assert result[0] == pytest.approx(-30.0)
        assert result[1] == pytest.approx(0.0, abs=0.1)

    def test_default_floor(self):
        to_db = ToDecibels()
        source = np.array([1e-10])
        result = to_db.apply(source)
        assert result[0] == pytest.approx(-60.0)

    def test_custom_floor(self):
        to_db = ToDecibels(floor_db=-20.0)
        source = np.array([1e-10])
        result = to_db.apply(source)
        assert result[0] == pytest.approx(-20.0)

    def test_preserves_shape_2d(self):
        to_db = ToDecibels()
        source = np.ones((5, 7))
        result = to_db.apply(source)
        assert result.shape == (5, 7)

    def test_preserves_shape_3d(self):
        to_db = ToDecibels()
        source = np.ones((3, 5, 7))
        result = to_db.apply(source)
        assert result.shape == (3, 5, 7)

    def test_output_dtype_float(self):
        to_db = ToDecibels()
        source = np.ones((4, 4), dtype=np.complex64)
        result = to_db.apply(source)
        assert np.issubdtype(result.dtype, np.floating)

    def test_zero_input_does_not_produce_nan(self):
        to_db = ToDecibels()
        source = np.array([0.0])
        result = to_db.apply(source)
        assert np.isfinite(result[0])

    def test_gpu_compatible_flag(self):
        assert ToDecibels.__gpu_compatible__ is True


# ---------------------------------------------------------------------------
# PercentileStretch
# ---------------------------------------------------------------------------

class TestPercentileStretch:
    def test_full_range_stretch(self):
        stretch = PercentileStretch(plow=0.0, phigh=100.0)
        source = np.array([0.0, 50.0, 100.0])
        result = stretch.apply(source)
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])

    def test_output_range(self):
        stretch = PercentileStretch(plow=2.0, phigh=98.0)
        rng = np.random.default_rng(42)
        source = rng.standard_normal((100, 100)).astype(np.float64)
        result = stretch.apply(source)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype_float32(self):
        stretch = PercentileStretch()
        source = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = stretch.apply(source)
        assert result.dtype == np.float32

    def test_constant_input_returns_zeros(self):
        stretch = PercentileStretch()
        source = np.ones((10, 10)) * 5.0
        result = stretch.apply(source)
        np.testing.assert_array_equal(result, np.zeros((10, 10), dtype=np.float32))

    def test_preserves_shape_2d(self):
        stretch = PercentileStretch()
        source = np.arange(100.0).reshape(10, 10)
        result = stretch.apply(source)
        assert result.shape == (10, 10)

    def test_preserves_shape_3d(self):
        stretch = PercentileStretch()
        source = np.arange(300.0).reshape(3, 10, 10)
        result = stretch.apply(source)
        assert result.shape == (3, 10, 10)

    def test_invalid_percentile_order_raises(self):
        with pytest.raises(ValueError, match="plow.*phigh"):
            PercentileStretch(plow=90.0, phigh=10.0)

    def test_equal_percentiles_raises(self):
        with pytest.raises(ValueError, match="plow.*phigh"):
            PercentileStretch(plow=50.0, phigh=50.0)

    def test_gpu_compatible_flag(self):
        assert PercentileStretch.__gpu_compatible__ is True

    def test_custom_percentiles(self):
        stretch = PercentileStretch(plow=10.0, phigh=90.0)
        source = np.arange(100.0)
        result = stretch.apply(source)
        # Values at p10 and p90 should map near 0 and 1
        assert result.min() >= 0.0
        assert result.max() <= 1.0
