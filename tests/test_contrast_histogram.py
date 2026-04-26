# -*- coding: utf-8 -*-
"""Tests for grdl.contrast.HistogramEqualization and CLAHE."""

import numpy as np
import pytest

from grdl.contrast import CLAHE, HistogramEqualization


class TestHistogramEqualization:
    def test_dtype_and_range(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((128, 128)).astype(np.float32)
        out = HistogramEqualization().apply(x)
        assert out.dtype == np.float32
        finite = out[np.isfinite(out)]
        assert finite.min() >= 0.0
        assert finite.max() <= 1.0

    def test_nan_preserved(self):
        x = np.array([1.0, 2.0, np.nan, 3.0], dtype=np.float64)
        out = HistogramEqualization(n_bins=16).apply(x)
        assert np.isnan(out[2])
        assert not np.isnan(out[0])

    def test_invalid_n_bins(self):
        with pytest.raises(ValueError):
            HistogramEqualization(n_bins=4)

    def test_constant_array(self):
        x = np.full((16, 16), 5.0, dtype=np.float32)
        out = HistogramEqualization().apply(x)
        # Constant input → CDF is degenerate but well-defined.
        assert out.dtype == np.float32


class TestCLAHE:
    def test_dtype_and_range(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((64, 64)).astype(np.float32)
        out = CLAHE(kernel_size=16, clip_limit=0.02).apply(x)
        assert out.dtype == np.float32
        assert 0.0 <= out.min() <= out.max() <= 1.0

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            CLAHE().apply(np.zeros((4, 4, 4), dtype=np.float32))

    def test_invalid_kernel_size(self):
        with pytest.raises(ValueError):
            CLAHE(kernel_size=2)

    def test_invalid_clip_limit(self):
        with pytest.raises(ValueError):
            CLAHE(clip_limit=-0.1)
        with pytest.raises(ValueError):
            CLAHE(clip_limit=1.1)
