# -*- coding: utf-8 -*-
"""Sanity check that re-exports from intensity.py are reachable."""

import numpy as np

from grdl.contrast import PercentileStretch as ContrastPercentile
from grdl.contrast import ToDecibels as ContrastToDecibels
from grdl.image_processing.intensity import PercentileStretch
from grdl.image_processing.intensity import ToDecibels


class TestReExports:
    def test_percentile_is_same_class(self):
        assert ContrastPercentile is PercentileStretch

    def test_to_decibels_is_same_class(self):
        assert ContrastToDecibels is ToDecibels

    def test_percentile_works_via_contrast_path(self):
        x = np.linspace(0.0, 1.0, 100).astype(np.float32)
        out = ContrastPercentile().apply(x)
        assert out.dtype == np.float32
        assert 0.0 <= out.min() <= out.max() <= 1.0

    def test_to_decibels_works_via_contrast_path(self):
        x = np.array([1.0, 10.0, 100.0])
        out = ContrastToDecibels().apply(x)
        # 20 * log10([1, 10, 100]) = [0, 20, 40]
        np.testing.assert_allclose(out, [0.0, 20.0, 40.0], atol=1e-6)
