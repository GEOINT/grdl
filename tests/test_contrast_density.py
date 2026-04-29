# -*- coding: utf-8 -*-
"""Tests for grdl.contrast density-family operators."""

import numpy as np
import pytest

from grdl.contrast import (
    Brighter,
    Darker,
    GDM,
    HighContrast,
    MangisDensity,
    PEDF,
)


@pytest.fixture
def sar_chip():
    """Small synthetic SAR-like complex chip with realistic dynamic range."""
    rng = np.random.default_rng(42)
    real = rng.standard_normal((64, 64))
    imag = rng.standard_normal((64, 64))
    return ((real + 1j * imag) * 50.0).astype(np.complex64)


class TestMangisDensity:
    def test_dtype_and_range(self, sar_chip):
        out = MangisDensity().apply(sar_chip)
        assert out.dtype == np.float32
        assert 0.0 <= out.min()
        assert out.max() <= 1.0

    def test_data_mean_carries_across_calls(self, sar_chip):
        remap = MangisDensity()
        mean = float(np.mean(np.abs(sar_chip)))
        a = remap.apply(sar_chip, data_mean=mean)
        b = remap.apply(sar_chip, data_mean=mean)
        np.testing.assert_array_equal(a, b)

    def test_matches_sarpy(self, sar_chip):
        sarpy_remap = pytest.importorskip("sarpy.visualization.remap")
        ours = MangisDensity().apply(sar_chip)
        sarpy_raw = sarpy_remap.Density(bit_depth=8).raw_call(sar_chip) / 255.0
        ours_clipped = np.clip(ours, 0.0, 1.0)
        sarpy_clipped = np.clip(sarpy_raw, 0.0, 1.0).astype(np.float32)
        np.testing.assert_allclose(ours_clipped, sarpy_clipped,
                                   atol=1e-4, rtol=1e-4)


class TestPresets:
    def test_brighter_uses_dmin60(self):
        b = Brighter()
        assert b.dmin == 60.0
        assert b.mmult == 40.0

    def test_darker_uses_dmin0(self):
        d = Darker()
        assert d.dmin == 0.0
        assert d.mmult == 40.0

    def test_high_contrast_uses_mmult4(self):
        h = HighContrast()
        assert h.dmin == 30.0
        assert h.mmult == 4.0

    def test_presets_produce_valid_output(self, sar_chip):
        for cls in (Brighter, Darker, HighContrast):
            out = cls().apply(sar_chip)
            assert out.dtype == np.float32
            assert 0.0 <= out.min() <= out.max() <= 1.0

    def test_presets_have_independent_tags(self):
        # Each preset is independently discoverable via its own tags.
        for cls in (Brighter, Darker, HighContrast, MangisDensity):
            assert hasattr(cls, '__processor_tags__')
            assert cls.__processor_tags__['description'] is not None


class TestPEDF:
    def test_dtype_and_range(self, sar_chip):
        out = PEDF().apply(sar_chip)
        assert out.dtype == np.float32
        assert 0.0 <= out.min() <= out.max() <= 1.0

    def test_top_compression(self):
        # PEDF compresses the upper half of Density's output by 0.5.
        # Density of 0.6 → PEDF: 0.5 * (0.6 + 0.5) = 0.55
        # Density of 0.4 → PEDF: 0.4 (unchanged, below threshold)
        density = MangisDensity()
        pedf = PEDF()
        rng = np.random.default_rng(0)
        chip = rng.standard_normal((32, 32)) * 30 + 0j
        d_out = density.apply(chip)
        p_out = pedf.apply(chip)
        # Where Density < 0.5, PEDF == Density.
        below = d_out < 0.5
        np.testing.assert_allclose(p_out[below], d_out[below], atol=1e-5)
        # Where Density >= 0.5, PEDF == 0.5 * (Density + 0.5).
        above = d_out >= 0.5
        if above.any():
            expected = 0.5 * (d_out[above] + 0.5)
            np.testing.assert_allclose(p_out[above], expected, atol=1e-5)


class TestGDM:
    def test_requires_graze_and_slope(self, sar_chip):
        out = GDM(graze_deg=30.0, slope_deg=0.0).apply(sar_chip)
        assert out.dtype == np.float32
        assert 0.0 <= out.min() <= out.max() <= 1.0

    def test_per_call_stats_override(self, sar_chip):
        gdm = GDM(graze_deg=30.0, slope_deg=0.0)
        amp = np.abs(sar_chip)
        mean = float(np.mean(amp))
        median = float(np.median(amp))
        a = gdm.apply(sar_chip, data_mean=mean, data_median=median)
        b = gdm.apply(sar_chip, data_mean=mean, data_median=median)
        np.testing.assert_array_equal(a, b)
