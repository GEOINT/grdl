# -*- coding: utf-8 -*-
"""
Tests for the base-class DEM line-of-sight solve.

Exercises ``Geolocation._image_to_latlon_with_dem`` with a synthetic
oblique-projection geolocation and analytic DEMs: convergence accuracy,
cliff (no-root) handling, configurable tolerance/iterations, DEM
coverage gaps, and parameter validation.

Dependencies
------------
pytest

Author
------
Curtis Ruck
ruckc@users.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-10-06

Modified
--------
2026-10-06
"""

# Third-party
import numpy as np
import pytest

# GRDL internal
from grdl.geolocation.base import Geolocation, DEM_LOS_MAX_ITER, DEM_LOS_TOL

_TAN = np.tan(np.radians(40.0))  # degrees of "ground shift" per metre


class _ObliqueGeo(Geolocation):
    """Lon shifts with height: lon = lon0 + col*dx + h*k; lat = row*dy."""

    K = 1e-5  # degrees lon per metre of height

    def __init__(self):
        super().__init__(shape=(100, 100))
        self.n_calls = 0

    def _image_to_latlon_array(self, rows, cols, height=0.0):
        self.n_calls += 1
        h = np.broadcast_to(np.asarray(height, dtype=np.float64),
                            np.shape(rows)).copy()
        lats = 10.0 + np.asarray(rows, dtype=np.float64) * 1e-5
        lons = 20.0 + np.asarray(cols, dtype=np.float64) * 1e-5 + h * self.K
        return lats, lons, h

    def _latlon_to_image_array(self, lats, lons, height=0.0):
        return (np.asarray(lats) - 10.0) / 1e-5, (np.asarray(lons) - 20.0) / 1e-5


class _FuncDEM:
    """Analytic DEM: heights = fn(lats, lons)."""

    def __init__(self, fn):
        self.fn = fn

    def get_elevation(self, lats, lons):
        return np.asarray(self.fn(np.asarray(lats), np.asarray(lons)),
                          dtype=np.float64)


def _geo(fn):
    geo = _ObliqueGeo()
    geo.elevation = _FuncDEM(fn)
    return geo


def _px(n=5):
    return np.column_stack([np.arange(n) * 3.0, np.arange(n) * 7.0])


def _residual(geo, result):
    """|DEM(lat, lon) - h| at the returned positions."""
    dem = geo.elevation.get_elevation(result[:, 0], result[:, 1])
    return np.abs(dem - result[:, 2])


def test_defaults_tighter_than_before():
    assert DEM_LOS_TOL < 0.5
    assert DEM_LOS_MAX_ITER >= 5


def test_flat_dem_converges_exactly():
    geo = _geo(lambda la, lo: np.full_like(la, 123.0))
    out = geo.image_to_latlon(_px())
    assert np.allclose(out[:, 2], 123.0)
    assert np.allclose(out[:, 1], 20.0 + _px()[:, 1] * 1e-5 + 123.0 * geo.K)


def test_smooth_slope_tight_accuracy():
    # slope dDEM/dlon: ground moves K deg per metre; slope term s = 0.9
    # makes plain fixed-point converge slowly (error shrinks by 0.9/iter).
    s = 0.9 / _ObliqueGeo.K
    geo = _geo(lambda la, lo: 50.0 + s * (lo - 20.0))
    out = geo.image_to_latlon(_px(), tol=1e-3)
    assert np.all(np.isfinite(out))
    assert _residual(geo, out).max() < 1e-3


def test_steep_slope_that_diverges_for_fixed_point():
    # |d DEM / dh| = 1.5 > 1: fixed-point diverges, secant solves it.
    s = 1.5 / _ObliqueGeo.K
    geo = _geo(lambda la, lo: 10.0 + s * (lo - 20.0 - 0.001))
    out = geo.image_to_latlon(_px(), tol=1e-3)
    assert np.all(np.isfinite(out))
    assert _residual(geo, out).max() < 1e-3


def test_cliff_without_root_returns_nan():
    # DEM = 100 on the near side of X, 0 beyond.  h=0 -> 100, h=100 -> 0:
    # fixed-point cycles forever and no crossing exists.
    x_edge = 20.0 + 50.0 * _ObliqueGeo.K
    geo = _geo(lambda la, lo: np.where(lo < x_edge, 100.0, 0.0))
    pix = np.array([[0.0, 0.0], [0.0, 0.0]])
    out = geo.image_to_latlon(pix)
    assert np.all(np.isnan(out))


def test_mixed_batch_isolates_nonconverged_points():
    x_edge = 20.0 + 50.0 * _ObliqueGeo.K
    geo = _geo(lambda la, lo: np.where(lo < x_edge, 100.0, 0.0))
    # col 0: cliff cycle.  col 1000: far beyond edge -> DEM 0, converges.
    pix = np.array([[0.0, 0.0], [0.0, 1000.0]])
    out = geo.image_to_latlon(pix)
    assert np.all(np.isnan(out[0]))
    assert np.all(np.isfinite(out[1]))
    assert out[1, 2] == pytest.approx(0.0)


def test_max_iter_honored():
    s = 0.9 / _ObliqueGeo.K
    geo = _geo(lambda la, lo: 50.0 + s * (lo - 20.0))
    geo.n_calls = 0
    out = geo.image_to_latlon(_px(1), max_iter=1, tol=1e-6)
    # One evaluation cannot reach 1e-6 -> NaN, and only one solve call.
    assert np.all(np.isnan(out))
    assert geo.n_calls == 1


def test_tol_honored_loose_vs_tight():
    s = 0.5 / _ObliqueGeo.K
    geo = _geo(lambda la, lo: 50.0 + s * (lo - 20.0))
    loose = geo.image_to_latlon(_px(3), tol=5.0)
    tight = geo.image_to_latlon(_px(3), tol=1e-6)
    assert _residual(geo, tight).max() < 1e-6
    assert _residual(geo, loose).max() < 5.0


def test_scalar_call_accepts_params():
    geo = _geo(lambda la, lo: np.full_like(la, 7.0))
    out = geo.image_to_latlon(3, 4, tol=0.01, max_iter=10)
    assert out.shape == (3,)
    assert out[2] == pytest.approx(7.0)


def test_nan_dem_keeps_last_valid_height():
    # DEM is defined west of the edge, NaN east of it.  The first
    # evaluation lands in coverage (h=0 -> 300), which moves the ground
    # point east into the gap; the point must keep 300, not revert to 0.
    x_edge = 20.0 + 100.0 * _ObliqueGeo.K
    geo = _geo(lambda la, lo: np.where(lo < x_edge, 300.0, np.nan))
    out = geo.image_to_latlon(np.array([[0.0, 0.0]]))
    assert out[0, 2] == pytest.approx(300.0)


def test_nan_dem_everywhere_returns_initial_guess():
    geo = _geo(lambda la, lo: np.full_like(la, np.nan))
    out = geo.image_to_latlon(np.array([[0.0, 0.0]]), height=42.0)
    assert out[0, 2] == pytest.approx(42.0)


def test_no_elevation_unchanged_constant_height():
    geo = _ObliqueGeo()
    out = geo.image_to_latlon(np.array([[1.0, 2.0]]), height=10.0)
    assert out[0, 2] == pytest.approx(10.0)


@pytest.mark.parametrize("kwargs", [
    {"max_iter": 0}, {"max_iter": -3}, {"max_iter": 2.5},
    {"tol": 0.0}, {"tol": -1.0}, {"tol": float("nan")},
])
def test_invalid_params_raise(kwargs):
    geo = _geo(lambda la, lo: np.zeros_like(la))
    with pytest.raises(ValueError):
        geo.image_to_latlon(_px(2), **kwargs)


def test_invalid_params_raise_without_dem():
    with pytest.raises(ValueError):
        _ObliqueGeo().image_to_latlon(_px(2), tol=-1.0)
