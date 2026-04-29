# -*- coding: utf-8 -*-
"""Tests for grdl.contrast.NRLStretch (sarpy NRL port)."""

import numpy as np
import pytest

from grdl.contrast import NRLStretch


@pytest.fixture
def sar_amp():
    rng = np.random.default_rng(7)
    return np.abs(rng.standard_normal((64, 64)) +
                  1j * rng.standard_normal((64, 64))).astype(np.float32) * 25.0


class TestNRLStretch:
    def test_dtype_and_range(self, sar_amp):
        out = NRLStretch().apply(sar_amp)
        assert out.dtype == np.float32
        assert 0.0 <= out.min() <= out.max() <= 1.0

    def test_invalid_knee(self):
        with pytest.raises(ValueError):
            NRLStretch(knee=0.0)
        with pytest.raises(ValueError):
            NRLStretch(knee=1.0)

    def test_invalid_percentile(self):
        with pytest.raises(ValueError):
            NRLStretch(percentile=0.0)
        with pytest.raises(ValueError):
            NRLStretch(percentile=100.0)

    def test_stats_kwarg_consistent_across_calls(self, sar_amp):
        remap = NRLStretch()
        stats = (
            float(np.min(sar_amp)),
            float(np.max(sar_amp)),
            float(np.percentile(sar_amp, 99.0)),
        )
        a = remap.apply(sar_amp, stats=stats)
        b = remap.apply(sar_amp, stats=stats)
        np.testing.assert_array_equal(a, b)

    def test_matches_sarpy(self, sar_amp):
        sarpy_remap = pytest.importorskip("sarpy.visualization.remap")
        ours = NRLStretch(knee=0.8, percentile=99.0).apply(sar_amp)
        sarpy_raw = sarpy_remap.NRL(
            bit_depth=8, knee=int(0.8 * 255), percentile=99.0,
        ).raw_call(sar_amp) / 255.0
        np.testing.assert_allclose(
            ours, sarpy_raw.astype(np.float32), atol=2e-3, rtol=1e-3,
        )
