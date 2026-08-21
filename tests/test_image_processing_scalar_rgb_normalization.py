# -*- coding: utf-8 -*-
"""Tests for percentile-driven scalar RGB normalization."""

import numpy as np

from grdl.image_processing.decomposition import (
    CompactPolDegreeOfPolarization,
    DegreeOfPolarization,
    DegreeOfPolarizationDP,
    DualPolRadarBuiltUpIndex,
    DualPolRadarSurfaceIndex,
)


def _assert_scalar_rgb_percentile_behavior(proc, key: str) -> None:
    arr = np.linspace(0.0, 1.0, 100, dtype=np.float64).reshape(10, 10)
    components = {key: arr}

    rgb_wide, _ = proc.to_rgb(
        components,
        percentile_low=0.0,
        percentile_high=100.0,
    )
    rgb_tight, _ = proc.to_rgb(
        components,
        percentile_low=20.0,
        percentile_high=80.0,
    )

    # Still grayscale (all channels equal)
    np.testing.assert_allclose(rgb_wide[0], rgb_wide[1])
    np.testing.assert_allclose(rgb_wide[1], rgb_wide[2])
    np.testing.assert_allclose(rgb_tight[0], rgb_tight[1])
    np.testing.assert_allclose(rgb_tight[1], rgb_tight[2])

    # Contrast settings should affect the output
    assert not np.allclose(rgb_wide, rgb_tight)

    # Output remains normalized
    assert np.all(rgb_wide >= 0.0) and np.all(rgb_wide <= 1.0)
    assert np.all(rgb_tight >= 0.0) and np.all(rgb_tight <= 1.0)


def test_full_pol_dop_to_rgb_uses_percentiles():
    _assert_scalar_rgb_percentile_behavior(DegreeOfPolarization(window_size=3), 'dop')


def test_dual_pol_dop_to_rgb_uses_percentiles():
    _assert_scalar_rgb_percentile_behavior(DegreeOfPolarizationDP(window_size=3), 'dop')


def test_compact_pol_dop_to_rgb_uses_percentiles():
    _assert_scalar_rgb_percentile_behavior(
        CompactPolDegreeOfPolarization(window_size=3), 'dop'
    )


def test_dprbi_to_rgb_uses_percentiles():
    _assert_scalar_rgb_percentile_behavior(
        DualPolRadarBuiltUpIndex(window_size=3), 'dprbi'
    )


def test_dprsi_to_rgb_uses_percentiles():
    _assert_scalar_rgb_percentile_behavior(
        DualPolRadarSurfaceIndex(window_size=3), 'dprsi'
    )
