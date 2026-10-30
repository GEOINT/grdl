# -*- coding: utf-8 -*-
"""
Real SRTM DTED2 regression tests for the DEM pixel-centre convention.

Decompresses a producer-made DTED Level 2 tile (``tests/data``) and checks
that ``DTEDElevation`` and ``TiledGeoDTED`` return exact post values and
bilinear midpoints read through GDAL's real DTED driver (issue #62).

Dependencies
------------
pytest
rasterio

Author
------
Curtis Ruck
curtis.ruck@gmail.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-10-27

Modified
--------
2026-10-27
"""

import gzip
import shutil
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from grdl.geolocation.elevation.dted import DTEDElevation
from grdl.geolocation.elevation.tiled_geotiff_dted import TiledGeoDTED

DATA = Path(__file__).parent / "data" / "n33_w082_1arc_v3.dt2.gz"
RES = 1.0 / 3600.0
BBOX = (-82.0, 33.0, -81.0, 34.0)


@pytest.fixture(scope="module")
def dted_root(tmp_path_factory):
    if not DATA.exists():
        pytest.skip("SRTM DTED2 test tile not present")
    with rasterio.Env():
        if "DTED" not in rasterio.Env().drivers():
            pytest.skip("GDAL DTED driver unavailable")
    root = tmp_path_factory.mktemp("srtm_dted2")
    (root / "w082").mkdir()
    with gzip.open(DATA, "rb") as src, open(root / "w082" / "n33.dt2", "wb") as dst:
        shutil.copyfileobj(src, dst)
    return root


@pytest.fixture(scope="module")
def posts(dted_root):
    with rasterio.open(dted_root / "w082" / "n33.dt2") as ds:
        return ds.read(1).astype(np.float64)


@pytest.fixture(scope="module", params=["dted", "tiled"])
def model(request, dted_root):
    if request.param == "dted":
        return DTEDElevation(str(dted_root), interpolation=1)
    return TiledGeoDTED(str(dted_root), interpolation=1, bbox=BBOX)


def test_tile_is_corner_origin(dted_root):
    """The real driver reports a corner-origin affine half a post out."""
    with rasterio.open(dted_root / "w082" / "n33.dt2") as ds:
        assert ds.shape == (3601, 3601)
        assert ds.bounds.left == pytest.approx(-82.0 - RES / 2)
        assert ds.bounds.top == pytest.approx(34.0 + RES / 2)


def test_posts_are_exact(model, posts):
    rng = np.random.default_rng(0)
    rows = rng.integers(5, 3590, 100)
    cols = rng.integers(5, 3590, 100)
    got = model.get_elevation(34.0 - RES * rows, -82.0 + RES * cols)
    np.testing.assert_allclose(got, posts[rows, cols], atol=1e-6)


def test_bilinear_midpoints(model, posts):
    rng = np.random.default_rng(1)
    rows = rng.integers(5, 3590, 100)
    cols = rng.integers(5, 3590, 100)
    fr, fc = rng.random(100), rng.random(100)
    truth = (
        posts[rows, cols] * (1 - fr) * (1 - fc)
        + posts[rows, cols + 1] * (1 - fr) * fc
        + posts[rows + 1, cols] * fr * (1 - fc)
        + posts[rows + 1, cols + 1] * fr * fc
    )
    got = model.get_elevation(34.0 - RES * (rows + fr), -82.0 + RES * (cols + fc))
    np.testing.assert_allclose(got, truth, atol=1e-6)


def test_outside_tile_is_nan(model):
    """Error path: a point far outside the tile yields NaN."""
    assert np.all(np.isnan(model.get_elevation(np.array([10.0]), np.array([10.0]))))
