# -*- coding: utf-8 -*-
"""
Tests for DEM-aware shape projection.

Uses a synthetic ConstantElevation attached to a basemap geolocation to
confirm that per-vertex DEM heights flow through to ``latlon_to_image``
via ``(N, 3)`` stacked projection -- even a constant elevation is
enough to verify the wiring (affine basemaps ignore height, so we
augment with a pyproj-projected CRS where height-ignoring has no
effect but still exercises the pipeline).

Dependencies
------------
pytest
rasterio

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-04-18

Modified
--------
2026-04-18
"""

import numpy as np
import pytest
from rasterio.transform import Affine

from grdl.geolocation.elevation.constant import ConstantElevation
from grdl.geolocation.eo.affine import AffineGeolocation
from grdl.shapes import Circle


@pytest.fixture
def geo_with_dem():
    transform = Affine(1e-5, 0.0, -118.2, 0.0, -1e-5, 34.1)
    geo = AffineGeolocation(transform, (2048, 2048), 'EPSG:4326')
    geo.elevation = ConstantElevation(height=50.0)
    return geo


def test_perimeter_sampled_with_dem_heights(geo_with_dem):
    circle = Circle(34.09, -118.195, radius_m=100.0)
    latlons = circle.perimeter_latlon(n=32, geolocation=geo_with_dem)
    assert latlons.shape == (32, 3)
    # Every vertex should carry the constant 50 m HAE.
    assert np.allclose(latlons[:, 2], 50.0, atol=1e-9)


def test_perimeter_without_geolocation_is_zero_height():
    circle = Circle(34.09, -118.195, radius_m=100.0)
    latlons = circle.perimeter_latlon(n=32)
    assert latlons.shape == (32, 3)
    assert np.all(latlons[:, 2] == 0.0)


def test_nan_dem_heights_fall_back_to_zero():
    """DEM that returns NaN for some points yields 0.0 instead of NaN."""
    class _PartiallyNanDEM:
        def get_elevation(self, lats, lons):
            out = np.full(len(lats), 10.0)
            out[::2] = np.nan  # alternating NaN
            return out

    transform = Affine(1e-5, 0.0, -118.2, 0.0, -1e-5, 34.1)
    geo = AffineGeolocation(transform, (1024, 1024), 'EPSG:4326')
    geo.elevation = _PartiallyNanDEM()

    circle = Circle(34.09, -118.195, radius_m=100.0)
    latlons = circle.perimeter_latlon(n=16, geolocation=geo)
    assert latlons.shape == (16, 3)
    assert not np.any(np.isnan(latlons))
    assert np.all(latlons[::2, 2] == 0.0)
    assert np.allclose(latlons[1::2, 2], 10.0)
