# -*- coding: utf-8 -*-
"""
Tests for DEM-integrated geolocation across all projection models.

Uses ConstantElevation as a synthetic DEM to verify that iterative
DEM refinement converges correctly for RPC, RSM, and R/Rdot projections.

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-03-17

Modified
--------
2026-03-17
"""

# Third-party
import numpy as np
import pytest

# GRDL
from grdl.IO.models.eo_nitf import RPCCoefficients, RSMCoefficients
from grdl.geolocation.eo.rpc import RPCGeolocation
from grdl.geolocation.eo.rsm import RSMGeolocation
from grdl.geolocation.elevation.constant import ConstantElevation


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def rpc_with_dem():
    """RPCGeolocation with a ConstantElevation DEM at 500 m."""
    line_num = np.zeros(20)
    line_num[0] = 0.001
    line_num[1] = 0.02
    line_num[2] = 0.98
    line_num[3] = -0.01  # stronger height term

    line_den = np.zeros(20)
    line_den[0] = 1.0

    samp_num = np.zeros(20)
    samp_num[0] = -0.002
    samp_num[1] = 0.97
    samp_num[2] = 0.03
    samp_num[3] = 0.008  # stronger height term

    samp_den = np.zeros(20)
    samp_den[0] = 1.0

    rpc = RPCCoefficients(
        line_off=5000.0, samp_off=5000.0,
        lat_off=38.0, long_off=-77.0, height_off=250.0,
        line_scale=5000.0, samp_scale=5000.0,
        lat_scale=0.5, long_scale=0.5, height_scale=500.0,
        line_num_coef=line_num, line_den_coef=line_den,
        samp_num_coef=samp_num, samp_den_coef=samp_den,
    )
    return rpc


@pytest.fixture
def rsm_with_dem():
    """RSMCoefficients for DEM testing."""
    # Linear RSM — order [1,1,1] → 8 terms
    row_num = np.zeros(8)
    row_num[0] = 0.0
    row_num[4] = 1.0   # x (lat)
    row_num[1] = -0.01  # z (height) — creates height dependency

    row_den = np.zeros(8)
    row_den[0] = 1.0

    col_num = np.zeros(8)
    col_num[0] = 0.0
    col_num[2] = 1.0   # y (lon)
    col_num[1] = 0.008  # z (height)

    col_den = np.zeros(8)
    col_den[0] = 1.0

    return RSMCoefficients(
        row_off=2048.0, col_off=2048.0,
        row_norm_sf=2048.0, col_norm_sf=2048.0,
        x_off=38.0, y_off=-77.0, z_off=250.0,
        x_norm_sf=0.5, y_norm_sf=0.5, z_norm_sf=500.0,
        row_num_powers=np.array([1, 1, 1]),
        row_den_powers=np.array([1, 1, 1]),
        col_num_powers=np.array([1, 1, 1]),
        col_den_powers=np.array([1, 1, 1]),
        row_num_coefs=row_num, row_den_coefs=row_den,
        col_num_coefs=col_num, col_den_coefs=col_den,
    )


# ── RPC + DEM ────────────────────────────────────────────────────────


class TestRPCWithDEM:
    """Tests for RPC geolocation with elevation model."""

    def test_dem_changes_result(self, rpc_with_dem):
        """Projecting with DEM should differ from height=0."""
        geo_no_dem = RPCGeolocation(rpc_with_dem, shape=(10000, 10000))
        lat0, lon0, h0 = geo_no_dem.image_to_latlon(5000, 5000, 0.0)

        lat500, lon500, h500 = geo_no_dem.image_to_latlon(5000, 5000, 500.0)

        # With height dependency, different HAE should give different coords
        assert lat0 != lat500 or lon0 != lon500

    def test_dem_constant_elevation(self, rpc_with_dem):
        """ConstantElevation DEM should project at the fixed height."""
        # Create geo with DEM at 500 m
        dem = ConstantElevation(height=500.0)
        geo = RPCGeolocation(rpc_with_dem, shape=(10000, 10000))
        geo.elevation = dem  # attach elevation model

        lat_dem, lon_dem, h_dem = geo.image_to_latlon(5000, 5000)

        # Should match projection at height=500
        lat_h, lon_h, h_h = RPCGeolocation(
            rpc_with_dem, shape=(10000, 10000)
        ).image_to_latlon(5000, 5000, 500.0)

        assert lat_dem == pytest.approx(lat_h, abs=0.01)
        assert lon_dem == pytest.approx(lon_h, abs=0.01)
        assert h_dem == pytest.approx(500.0, abs=1.0)

    def test_dem_round_trip(self, rpc_with_dem):
        """Forward with DEM → inverse at DEM height should round-trip."""
        dem = ConstantElevation(height=300.0)
        geo = RPCGeolocation(rpc_with_dem, shape=(10000, 10000))
        geo.elevation = dem

        lat, lon, h = geo.image_to_latlon(5000, 5000)
        row, col = geo.latlon_to_image(lat, lon, h)

        assert row == pytest.approx(5000.0, abs=0.5)
        assert col == pytest.approx(5000.0, abs=0.5)

    def test_dem_array_input(self, rpc_with_dem):
        """DEM integration should work with array inputs."""
        dem = ConstantElevation(height=200.0)
        geo = RPCGeolocation(rpc_with_dem, shape=(10000, 10000))
        geo.elevation = dem

        rows = np.array([4000.0, 5000.0, 6000.0])
        cols = np.array([4000.0, 5000.0, 6000.0])
        lats, lons, heights = geo.image_to_latlon(rows, cols)

        assert lats.shape == (3,)
        np.testing.assert_allclose(heights, 200.0, atol=1.0)


# ── RSM + DEM ────────────────────────────────────────────────────────


class TestRSMWithDEM:
    """Tests for RSM geolocation with elevation model."""

    def test_dem_constant_elevation(self, rsm_with_dem):
        """ConstantElevation DEM should project at the fixed height."""
        dem = ConstantElevation(height=400.0)
        geo = RSMGeolocation(rsm_with_dem, shape=(4096, 4096))
        geo.elevation = dem

        lat_dem, lon_dem, h_dem = geo.image_to_latlon(2048, 2048)

        lat_h, lon_h, _ = RSMGeolocation(
            rsm_with_dem, shape=(4096, 4096)
        ).image_to_latlon(2048, 2048, 400.0)

        assert lat_dem == pytest.approx(lat_h, abs=0.01)
        assert lon_dem == pytest.approx(lon_h, abs=0.01)
        assert h_dem == pytest.approx(400.0, abs=1.0)

    def test_dem_round_trip(self, rsm_with_dem):
        """Forward with DEM → inverse at DEM height should round-trip."""
        dem = ConstantElevation(height=250.0)
        geo = RSMGeolocation(rsm_with_dem, shape=(4096, 4096))
        geo.elevation = dem

        lat, lon, h = geo.image_to_latlon(2048, 2048)
        row, col = geo.latlon_to_image(lat, lon, h)

        assert row == pytest.approx(2048.0, abs=0.5)
        assert col == pytest.approx(2048.0, abs=0.5)


# ── Base class DEM dispatch ─────────────────────────────────────────


class TestBaseClassDEMDispatch:
    """Tests for the Geolocation base class DEM iterative refinement."""

    def test_no_dem_passthrough(self, rpc_with_dem):
        """Without DEM, _image_to_latlon_with_dem is just a passthrough."""
        geo = RPCGeolocation(rpc_with_dem, shape=(10000, 10000))
        assert geo.elevation is None

        lats, lons, heights = geo._image_to_latlon_with_dem(
            np.array([5000.0]), np.array([5000.0]), 100.0)

        # Should use height=100 directly
        lats2, lons2, h2 = geo._image_to_latlon_array(
            np.array([5000.0]), np.array([5000.0]), 100.0)

        np.testing.assert_allclose(lats, lats2)
        np.testing.assert_allclose(lons, lons2)

    def test_dem_iterates_to_convergence(self, rpc_with_dem):
        """With DEM, should converge to the DEM height."""
        geo = RPCGeolocation(rpc_with_dem, shape=(10000, 10000))
        geo.elevation = ConstantElevation(height=750.0)

        lats, lons, heights = geo._image_to_latlon_with_dem(
            np.array([5000.0]), np.array([5000.0]), 0.0)

        assert heights[0] == pytest.approx(750.0, abs=1.0)
