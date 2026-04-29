# -*- coding: utf-8 -*-
"""Tests for grdl.contrast.LinearStretch."""

import numpy as np
import pytest

from grdl.contrast import LinearStretch


class TestLinearStretch:
    def test_identity_on_unit_range(self):
        x = np.linspace(0.0, 1.0, 11, dtype=np.float32)
        out = LinearStretch(min_value=0.0, max_value=1.0).apply(x)
        np.testing.assert_allclose(out, x, atol=1e-6)

    def test_dtype_and_range(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-3.0, 7.0, (32, 32)).astype(np.float32)
        out = LinearStretch().apply(x)
        assert out.dtype == np.float32
        assert out.min() == pytest.approx(0.0, abs=1e-6)
        assert out.max() == pytest.approx(1.0, abs=1e-6)

    def test_nonfinite_saturates(self):
        x = np.array([np.nan, 0.0, 0.5, 1.0, np.inf], dtype=np.float64)
        out = LinearStretch(min_value=0.0, max_value=1.0).apply(x)
        # Non-finite samples → 1.0 (saturated white).
        assert out[0] == pytest.approx(1.0)
        assert out[-1] == pytest.approx(1.0)
        # Finite samples preserve their relative position.
        np.testing.assert_allclose(out[1:4], [0.0, 0.5, 1.0], atol=1e-6)

    def test_complex_input_uses_magnitude(self):
        x = np.array([0+0j, 3+4j, 6+8j])           # |.| = 0, 5, 10
        out = LinearStretch(min_value=0.0, max_value=10.0).apply(x)
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-6)

    def test_per_call_kwargs_override_instance(self):
        x = np.linspace(0.0, 100.0, 11, dtype=np.float32)
        stretch = LinearStretch(min_value=0.0, max_value=10.0)
        # Per-call override takes precedence over instance.
        out = stretch.apply(x, min_value=0.0, max_value=100.0)
        np.testing.assert_allclose(out, x / 100.0, atol=1e-6)

    def test_constant_array(self):
        x = np.full((8, 8), 7.0, dtype=np.float32)
        out = LinearStretch().apply(x)
        np.testing.assert_array_equal(out, np.zeros_like(x))
