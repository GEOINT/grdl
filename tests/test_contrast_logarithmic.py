# -*- coding: utf-8 -*-
"""Tests for grdl.contrast.LogStretch (sarpy Logarithmic port)."""

import numpy as np
import pytest

from grdl.contrast import LogStretch


class TestLogStretch:
    def test_dtype_and_range(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0.1, 1000.0, (32, 32)).astype(np.float32)
        out = LogStretch().apply(x)
        assert out.dtype == np.float32
        assert out.min() == pytest.approx(0.0, abs=1e-6)
        assert out.max() == pytest.approx(1.0, abs=1e-6)

    def test_zero_maps_to_zero(self):
        x = np.array([0.0, 1.0, 10.0, 100.0])
        out = LogStretch(min_value=1.0, max_value=100.0).apply(x)
        assert out[0] == pytest.approx(0.0)

    def test_nonfinite_saturates(self):
        x = np.array([np.nan, 1.0, 100.0], dtype=np.float64)
        out = LogStretch(min_value=1.0, max_value=100.0).apply(x)
        assert out[0] == pytest.approx(1.0)

    def test_matches_sarpy(self):
        sarpy_remap = pytest.importorskip("sarpy.visualization.remap")
        rng = np.random.default_rng(123)
        x = rng.uniform(0.5, 200.0, (16, 16))
        ours = LogStretch(min_value=0.5, max_value=200.0).apply(x)
        # sarpy's bit_depth=8 returns uint8 in [0, 255]; divide → [0, 1].
        sarpy_out = sarpy_remap.Logarithmic(
            bit_depth=8, min_value=0.5, max_value=200.0,
        ).raw_call(x) / 255.0
        np.testing.assert_allclose(ours, sarpy_out.astype(np.float32),
                                   atol=1e-5, rtol=1e-5)
