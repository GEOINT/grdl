# -*- coding: utf-8 -*-
"""
Tests for DEM pixel-centre sampling convention (issue #62).

A GDAL/rasterio affine is corner-origin (PixelIsArea): array index
(r, c) has its centre at (r + 0.5, c + 0.5) in affine pixel space.  The
DEM backends must subtract 0.5 after the inverse affine so that a DEM
whose posts lie on a known lon/lat grid is sampled at the correct place.

Dependencies
------------
pytest
rasterio
scipy

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
2026-10-06

Modified
--------
2026-10-06
"""

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
import rasterio.transform as rt
from scipy.ndimage import map_coordinates

from grdl.geolocation.elevation.dted import DTEDElevation
from grdl.geolocation.elevation.geotiff_dem import GeoTIFFDEM
from grdl.geolocation.elevation.tiled_geotiff_dem import TiledGeoTIFFDEM
from grdl.geolocation.elevation.tiled_geotiff_dted import TiledGeoDTED

N = 1201
RES = 1.0 / 1200.0
LAT = np.array([33.4712, 33.2523])
LON = np.array([-81.9733, -81.4312])


def _grid():
    """Posts on the 1/1200 deg grid nodes, as in the issue repro."""
    X, Y = np.meshgrid(-82 + RES * np.arange(N), 34 - RES * np.arange(N))
    return (250 + 200 * np.sin(X * 80) * np.cos(Y * 90)).astype("float32")


def _transform():
    # Pixel centres sit on the grid nodes -> corner origin is half a post out.
    return rt.from_origin(-82 - RES / 2, 34 + RES / 2, RES, RES)


def _write(path, Z):
    with rasterio.open(
        str(path), "w", driver="GTiff", width=N, height=N, count=1,
        dtype="float32", crs="EPSG:4326", transform=_transform(),
    ) as d:
        d.write(Z, 1)


def _expected(Z, order=1):
    c, r = (~_transform()) * (LON, LAT)
    return map_coordinates(Z.astype(float), [r - 0.5, c - 0.5], order=order)


@pytest.fixture(scope="module")
def Z():
    return _grid()


def test_reference_differs_from_uncorrected(Z):
    """Sanity: the half-pixel shift matters for this synthetic surface."""
    c, r = (~_transform()) * (LON, LAT)
    shifted = map_coordinates(Z.astype(float), [r, c], order=1)
    assert np.max(np.abs(shifted - _expected(Z))) > 1.0


def test_geotiff_dem_pixel_centre(tmp_path, Z):
    p = tmp_path / "dem.tif"
    _write(p, Z)
    got = GeoTIFFDEM(str(p), interpolation=1).get_elevation(LAT, LON)
    np.testing.assert_allclose(got, _expected(Z), atol=1e-3)


def test_geotiff_dem_bicubic_close_to_analytic_surface(tmp_path, Z):
    p = tmp_path / "dem.tif"
    _write(p, Z)
    got = GeoTIFFDEM(str(p), interpolation=3).get_elevation(LAT, LON)
    truth = 250 + 200 * np.sin(LON * 80) * np.cos(LAT * 90)
    np.testing.assert_allclose(got, truth, atol=0.05)


def test_geotiff_dem_node_values_exact(tmp_path, Z):
    """Querying exactly at a post returns that post's value."""
    p = tmp_path / "dem.tif"
    _write(p, Z)
    rows, cols = np.array([100, 600, 900]), np.array([200, 300, 1000])
    lats = 34 - RES * rows
    lons = -82 + RES * cols
    got = GeoTIFFDEM(str(p), interpolation=1).get_elevation(lats, lons)
    np.testing.assert_allclose(got, Z[rows, cols], atol=1e-3)


def test_tiled_geotiff_dem_pixel_centre(tmp_path, Z):
    _write(tmp_path / "N33W082.tif", Z)
    dem = TiledGeoTIFFDEM(
        str(tmp_path), interpolation=1, bbox=(-82.0, 33.0, -81.0, 34.0),
    )
    got = dem.get_elevation(LAT, LON)
    np.testing.assert_allclose(got, _expected(Z), atol=1e-3)


def _dted_archive(tmp_path, Z):
    (tmp_path / "w082").mkdir()
    _write(tmp_path / "w082" / "n33.dt2", Z)
    return tmp_path


def test_dted_elevation_pixel_centre(tmp_path, Z):
    root = _dted_archive(tmp_path, Z)
    got = DTEDElevation(str(root), interpolation=1).get_elevation(LAT, LON)
    np.testing.assert_allclose(got, _expected(Z), atol=1e-3)


def test_tiled_geodted_pixel_centre(tmp_path, Z):
    root = _dted_archive(tmp_path, Z)
    dem = TiledGeoDTED(
        str(root), interpolation=1, bbox=(-82.0, 33.0, -81.0, 34.0),
    )
    got = dem.get_elevation(LAT, LON)
    np.testing.assert_allclose(got, _expected(Z), atol=1e-3)


def test_geotiff_dem_outside_extent_is_nan(tmp_path, Z):
    """Error path: points beyond the DEM extent yield NaN, not data."""
    p = tmp_path / "dem.tif"
    _write(p, Z)
    got = GeoTIFFDEM(str(p), interpolation=1).get_elevation(
        np.array([10.0]), np.array([10.0]),
    )
    assert np.all(np.isnan(got))
