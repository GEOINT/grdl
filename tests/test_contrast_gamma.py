# -*- coding: utf-8 -*-
"""Tests for grdl.contrast.GammaCorrection."""

import numpy as np
import pytest

from grdl.contrast import GammaCorrection


class TestGammaCorrection:
    def test_identity_at_gamma_one(self):
        x = np.linspace(0.0, 1.0, 21, dtype=np.float32)
        out = GammaCorrection(gamma=1.0).apply(x)
        np.testing.assert_allclose(out, x, atol=1e-6)

    def test_brightens_when_gamma_above_one(self):
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        out = GammaCorrection(gamma=2.0).apply(x)
        # Power 1/2 → mid values brighten.
        assert out[1] > x[1]
        assert out[2] > x[2]
        assert out[3] > x[3]

    def test_darkens_when_gamma_below_one(self):
        x = np.array([0.25, 0.5, 0.75], dtype=np.float32)
        out = GammaCorrection(gamma=0.5).apply(x)
        for o, i in zip(out, x):
            assert o < i

    def test_invalid_gamma(self):
        with pytest.raises(ValueError):
            GammaCorrection(gamma=0.0)
        with pytest.raises(ValueError):
            GammaCorrection(gamma=-1.0)

    def test_clips_input_to_unit_range(self):
        x = np.array([-0.5, 0.5, 1.5])
        out = GammaCorrection(gamma=1.0).apply(x)
        assert 0.0 <= out.min() <= out.max() <= 1.0
