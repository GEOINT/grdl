# -*- coding: utf-8 -*-
"""
Tests for grdl.geolocation.coordinates — ECEF and ENU conversions.

Verifies round-trip accuracy of geodetic↔ECEF and geodetic↔ENU transforms
using known reference points.

Author
------
Duane Smalley, PhD
duane.d.smalley@gmail.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-03-08

Modified
--------
2026-03-22  Update to (N, M) stacked ndarray convention.
"""

import numpy as np
import pytest

from grdl.geolocation.coordinates import (
    WGS84_A,
    WGS84_B,
    geodetic_to_ecef,
    ecef_to_geodetic,
    geodetic_to_enu,
    enu_to_geodetic,
)


# ===================================================================
# geodetic_to_ecef / ecef_to_geodetic
# ===================================================================

class TestGeodeticECEF:
    """Tests for geodetic ↔ ECEF conversions."""

    def test_equator_prime_meridian(self):
        """Point on equator at prime meridian → X = semi-major axis."""
        ecef = geodetic_to_ecef(np.array([0.0, 0.0, 0.0]))
        assert abs(ecef[0] - WGS84_A) < 0.01
        assert abs(ecef[1]) < 0.01
        assert abs(ecef[2]) < 0.01

    def test_north_pole(self):
        """North pole → Z = semi-minor axis."""
        ecef = geodetic_to_ecef(np.array([90.0, 0.0, 0.0]))
        assert abs(ecef[0]) < 0.01
        assert abs(ecef[1]) < 0.01
        assert abs(ecef[2] - WGS84_B) < 0.01

    def test_south_pole(self):
        """South pole → Z = -semi-minor axis."""
        ecef = geodetic_to_ecef(np.array([-90.0, 0.0, 0.0]))
        assert abs(ecef[2] + WGS84_B) < 0.01

    def test_round_trip_scalar(self):
        """geodetic → ECEF → geodetic round-trip with scalar (3,) input."""
        lat0, lon0, h0 = 36.0, -75.5, 100.0
        ecef = geodetic_to_ecef(np.array([lat0, lon0, h0]))
        geo = ecef_to_geodetic(ecef)
        assert abs(geo[0] - lat0) < 1e-9
        assert abs(geo[1] - lon0) < 1e-9
        assert abs(geo[2] - h0) < 1e-3

    def test_round_trip_array(self):
        """Round-trip with array inputs at diverse locations."""
        lats = np.array([0.0, 45.0, -30.0, 89.9, -89.9])
        lons = np.array([0.0, 90.0, -120.0, 180.0, -45.0])
        heights = np.array([0.0, 500.0, 10000.0, 0.0, 5000.0])

        ecef = geodetic_to_ecef(np.column_stack([lats, lons, heights]))
        geo = ecef_to_geodetic(ecef)

        np.testing.assert_allclose(geo[:, 0], lats, atol=1e-9)
        np.testing.assert_allclose(geo[:, 1], lons, atol=1e-9)
        np.testing.assert_allclose(geo[:, 2], heights, atol=1e-3)

    def test_height_above_ellipsoid(self):
        """Non-zero height moves point radially outward."""
        h = 1000.0
        ecef0 = geodetic_to_ecef(np.array([0.0, 0.0, 0.0]))
        ecef1 = geodetic_to_ecef(np.array([0.0, 0.0, h]))
        # At equator/prime meridian, X increases by h
        assert abs((ecef1[0] - ecef0[0]) - h) < 0.01


# ===================================================================
# geodetic_to_enu / enu_to_geodetic
# ===================================================================

class TestGeodeticENU:
    """Tests for geodetic ↔ ENU conversions."""

    def test_reference_point_is_origin(self):
        """The reference point itself maps to ENU (0, 0, 0)."""
        ref_lat, ref_lon, ref_alt = 36.0, -75.5, 0.0
        ref = np.array([ref_lat, ref_lon, ref_alt])
        enu = geodetic_to_enu(
            np.array([ref_lat, ref_lon, ref_alt]), ref)
        assert abs(enu[0]) < 1e-6
        assert abs(enu[1]) < 1e-6
        assert abs(enu[2]) < 1e-6

    def test_point_north_of_reference(self):
        """A point slightly north has positive north, near-zero east."""
        ref_lat, ref_lon, ref_alt = 36.0, -75.5, 0.0
        ref = np.array([ref_lat, ref_lon, ref_alt])
        offset_deg = 0.01  # ~1.1 km
        enu = geodetic_to_enu(
            np.array([ref_lat + offset_deg, ref_lon, ref_alt]), ref)
        assert enu[1] > 1000  # should be ~1.1 km
        assert abs(enu[0]) < 1.0  # negligible east

    def test_point_east_of_reference(self):
        """A point slightly east has positive east, near-zero north."""
        ref_lat, ref_lon, ref_alt = 36.0, -75.5, 0.0
        ref = np.array([ref_lat, ref_lon, ref_alt])
        offset_deg = 0.01
        enu = geodetic_to_enu(
            np.array([ref_lat, ref_lon + offset_deg, ref_alt]), ref)
        assert enu[0] > 500  # positive east (cos(36) shortens it)
        assert abs(enu[1]) < 1.0

    def test_point_above_reference(self):
        """A point at higher altitude has positive up."""
        ref_lat, ref_lon, ref_alt = 36.0, -75.5, 0.0
        ref = np.array([ref_lat, ref_lon, ref_alt])
        enu = geodetic_to_enu(
            np.array([ref_lat, ref_lon, 1000.0]), ref)
        assert abs(enu[0]) < 1.0
        assert abs(enu[1]) < 1.0
        assert abs(enu[2] - 1000.0) < 0.1

    def test_round_trip_scalar(self):
        """geodetic → ENU → geodetic round-trip."""
        ref_lat, ref_lon, ref_alt = 40.0, -100.0, 500.0
        ref = np.array([ref_lat, ref_lon, ref_alt])
        target_lat, target_lon, target_h = 40.01, -99.99, 600.0

        enu = geodetic_to_enu(
            np.array([target_lat, target_lon, target_h]), ref)
        geo = enu_to_geodetic(enu, ref)

        assert abs(geo[0] - target_lat) < 1e-9
        assert abs(geo[1] - target_lon) < 1e-9
        assert abs(geo[2] - target_h) < 1e-3

    def test_round_trip_array(self):
        """Round-trip with multiple target points."""
        ref_lat, ref_lon, ref_alt = 36.0, -75.5, 0.0
        ref = np.array([ref_lat, ref_lon, ref_alt])
        lats = np.array([36.01, 35.99, 36.005, 36.0])
        lons = np.array([-75.5, -75.5, -75.49, -75.51])
        heights = np.array([0.0, 100.0, 50.0, 200.0])

        enu = geodetic_to_enu(
            np.column_stack([lats, lons, heights]), ref)
        geo = enu_to_geodetic(enu, ref)

        np.testing.assert_allclose(geo[:, 0], lats, atol=1e-9)
        np.testing.assert_allclose(geo[:, 1], lons, atol=1e-9)
        np.testing.assert_allclose(geo[:, 2], heights, atol=1e-3)

    def test_enu_orthogonality(self):
        """E, N, U directions are approximately orthogonal."""
        ref_lat, ref_lon, ref_alt = 36.0, -75.5, 0.0
        ref = np.array([ref_lat, ref_lon, ref_alt])
        delta = 0.001  # ~100 m

        # East direction
        enu_e = geodetic_to_enu(
            np.array([ref_lat, ref_lon + delta, ref_alt]), ref)
        vec_e = np.array([enu_e[0], enu_e[1], enu_e[2]])

        # North direction
        enu_n = geodetic_to_enu(
            np.array([ref_lat + delta, ref_lon, ref_alt]), ref)
        vec_n = np.array([enu_n[0], enu_n[1], enu_n[2]])

        # Dot product should be near zero
        dot = np.dot(vec_e, vec_n)
        norms = np.linalg.norm(vec_e) * np.linalg.norm(vec_n)
        cos_angle = dot / norms
        assert abs(cos_angle) < 0.01  # nearly orthogonal
