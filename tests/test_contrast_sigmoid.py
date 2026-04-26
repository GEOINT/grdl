# -*- coding: utf-8 -*-
"""Tests for grdl.contrast.SigmoidStretch."""

import numpy as np
import pytest

from grdl.contrast import SigmoidStretch


class TestSigmoidStretch:
    def test_endpoints_zero_and_one(self):
        x = np.array([0.0, 1.0], dtype=np.float32)
        out = SigmoidStretch().apply(x)
        np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-5)

    def test_monotonic(self):
        x = np.linspace(0.0, 1.0, 21, dtype=np.float32)
        out = SigmoidStretch().apply(x)
        assert np.all(np.diff(out) >= -1e-6)

    def test_dtype_and_range(self):
        x = np.linspace(-1.0, 2.0, 100, dtype=np.float32)
        out = SigmoidStretch().apply(x)
        assert out.dtype == np.float32
        assert 0.0 <= out.min() <= out.max() <= 1.0

    def test_invalid_slope(self):
        with pytest.raises(ValueError):
            SigmoidStretch(slope=0.0)
        with pytest.raises(ValueError):
            SigmoidStretch(slope=-1.0)

    def test_invalid_center(self):
        with pytest.raises(ValueError):
            SigmoidStretch(center=-0.1)
        with pytest.raises(ValueError):
            SigmoidStretch(center=1.1)

    def test_higher_slope_steeper_at_center(self):
        center = 0.5
        soft = SigmoidStretch(center=center, slope=2.0)
        sharp = SigmoidStretch(center=center, slope=20.0)
        x_lo = np.array([center - 0.1], dtype=np.float32)
        x_hi = np.array([center + 0.1], dtype=np.float32)
        slope_soft = (soft.apply(x_hi)[0] - soft.apply(x_lo)[0]) / 0.2
        slope_sharp = (sharp.apply(x_hi)[0] - sharp.apply(x_lo)[0]) / 0.2
        assert slope_sharp > slope_soft
