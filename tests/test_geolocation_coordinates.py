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
2026-03-08
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
        x, y, z = geodetic_to_ecef(
            np.array([0.0]), np.array([0.0]), np.array([0.0])
        )
        assert abs(float(x) - WGS84_A) < 0.01
        assert abs(float(y)) < 0.01
        assert abs(float(z)) < 0.01

    def test_north_pole(self):
        """North pole → Z = semi-minor axis."""
        x, y, z = geodetic_to_ecef(
            np.array([90.0]), np.array([0.0]), np.array([0.0])
        )
        assert abs(float(x)) < 0.01
        assert abs(float(y)) < 0.01
        assert abs(float(z) - WGS84_B) < 0.01

    def test_south_pole(self):
        """South pole → Z = -semi-minor axis."""
        x, y, z = geodetic_to_ecef(
            np.array([-90.0]), np.array([0.0]), np.array([0.0])
        )
        assert abs(float(z) + WGS84_B) < 0.01

    def test_round_trip_scalar(self):
        """geodetic → ECEF → geodetic round-trip with scalar-like arrays."""
        lat0, lon0, h0 = 36.0, -75.5, 100.0
        x, y, z = geodetic_to_ecef(
            np.array([lat0]), np.array([lon0]), np.array([h0])
        )
        lat1, lon1, h1 = ecef_to_geodetic(x, y, z)
        assert abs(float(lat1) - lat0) < 1e-9
        assert abs(float(lon1) - lon0) < 1e-9
        assert abs(float(h1) - h0) < 1e-3

    def test_round_trip_array(self):
        """Round-trip with array inputs at diverse locations."""
        lats = np.array([0.0, 45.0, -30.0, 89.9, -89.9])
        lons = np.array([0.0, 90.0, -120.0, 180.0, -45.0])
        heights = np.array([0.0, 500.0, 10000.0, 0.0, 5000.0])

        x, y, z = geodetic_to_ecef(lats, lons, heights)
        lats2, lons2, heights2 = ecef_to_geodetic(x, y, z)

        np.testing.assert_allclose(lats2, lats, atol=1e-9)
        np.testing.assert_allclose(lons2, lons, atol=1e-9)
        np.testing.assert_allclose(heights2, heights, atol=1e-3)

    def test_height_above_ellipsoid(self):
        """Non-zero height moves point radially outward."""
        h = 1000.0
        x0, y0, z0 = geodetic_to_ecef(
            np.array([0.0]), np.array([0.0]), np.array([0.0])
        )
        x1, y1, z1 = geodetic_to_ecef(
            np.array([0.0]), np.array([0.0]), np.array([h])
        )
        # At equator/prime meridian, X increases by h
        assert abs(float(x1 - x0) - h) < 0.01


# ===================================================================
# geodetic_to_enu / enu_to_geodetic
# ===================================================================

class TestGeodeticENU:
    """Tests for geodetic ↔ ENU conversions."""

    def test_reference_point_is_origin(self):
        """The reference point itself maps to ENU (0, 0, 0)."""
        ref_lat, ref_lon, ref_alt = 36.0, -75.5, 0.0
        e, n, u = geodetic_to_enu(
            np.array([ref_lat]), np.array([ref_lon]),
            np.array([ref_alt]),
            ref_lat, ref_lon, ref_alt,
        )
        assert abs(float(e)) < 1e-6
        assert abs(float(n)) < 1e-6
        assert abs(float(u)) < 1e-6

    def test_point_north_of_reference(self):
        """A point slightly north has positive north, near-zero east."""
        ref_lat, ref_lon, ref_alt = 36.0, -75.5, 0.0
        offset_deg = 0.01  # ~1.1 km
        e, n, u = geodetic_to_enu(
            np.array([ref_lat + offset_deg]), np.array([ref_lon]),
            np.array([ref_alt]),
            ref_lat, ref_lon, ref_alt,
        )
        assert float(n) > 1000  # should be ~1.1 km
        assert abs(float(e)) < 1.0  # negligible east

    def test_point_east_of_reference(self):
        """A point slightly east has positive east, near-zero north."""
        ref_lat, ref_lon, ref_alt = 36.0, -75.5, 0.0
        offset_deg = 0.01
        e, n, u = geodetic_to_enu(
            np.array([ref_lat]), np.array([ref_lon + offset_deg]),
            np.array([ref_alt]),
            ref_lat, ref_lon, ref_alt,
        )
        assert float(e) > 500  # positive east (cos(36°) shortens it)
        assert abs(float(n)) < 1.0

    def test_point_above_reference(self):
        """A point at higher altitude has positive up."""
        ref_lat, ref_lon, ref_alt = 36.0, -75.5, 0.0
        e, n, u = geodetic_to_enu(
            np.array([ref_lat]), np.array([ref_lon]),
            np.array([1000.0]),
            ref_lat, ref_lon, ref_alt,
        )
        assert abs(float(e)) < 1.0
        assert abs(float(n)) < 1.0
        assert abs(float(u) - 1000.0) < 0.1

    def test_round_trip_scalar(self):
        """geodetic → ENU → geodetic round-trip."""
        ref_lat, ref_lon, ref_alt = 40.0, -100.0, 500.0
        target_lat, target_lon, target_h = 40.01, -99.99, 600.0

        e, n, u = geodetic_to_enu(
            np.array([target_lat]), np.array([target_lon]),
            np.array([target_h]),
            ref_lat, ref_lon, ref_alt,
        )
        lat2, lon2, h2 = enu_to_geodetic(e, n, u, ref_lat, ref_lon, ref_alt)

        assert abs(float(lat2) - target_lat) < 1e-9
        assert abs(float(lon2) - target_lon) < 1e-9
        assert abs(float(h2) - target_h) < 1e-3

    def test_round_trip_array(self):
        """Round-trip with multiple target points."""
        ref_lat, ref_lon, ref_alt = 36.0, -75.5, 0.0
        lats = np.array([36.01, 35.99, 36.005, 36.0])
        lons = np.array([-75.5, -75.5, -75.49, -75.51])
        heights = np.array([0.0, 100.0, 50.0, 200.0])

        e, n, u = geodetic_to_enu(
            lats, lons, heights, ref_lat, ref_lon, ref_alt,
        )
        lats2, lons2, h2 = enu_to_geodetic(
            e, n, u, ref_lat, ref_lon, ref_alt,
        )

        np.testing.assert_allclose(lats2, lats, atol=1e-9)
        np.testing.assert_allclose(lons2, lons, atol=1e-9)
        np.testing.assert_allclose(h2, heights, atol=1e-3)

    def test_enu_orthogonality(self):
        """E, N, U directions are approximately orthogonal."""
        ref_lat, ref_lon, ref_alt = 36.0, -75.5, 0.0
        delta = 0.001  # ~100 m

        # East direction
        e_e, n_e, u_e = geodetic_to_enu(
            np.array([ref_lat]), np.array([ref_lon + delta]),
            np.array([ref_alt]),
            ref_lat, ref_lon, ref_alt,
        )
        vec_e = np.array([float(e_e), float(n_e), float(u_e)])

        # North direction
        e_n, n_n, u_n = geodetic_to_enu(
            np.array([ref_lat + delta]), np.array([ref_lon]),
            np.array([ref_alt]),
            ref_lat, ref_lon, ref_alt,
        )
        vec_n = np.array([float(e_n), float(n_n), float(u_n)])

        # Dot product should be near zero
        dot = np.dot(vec_e, vec_n)
        norms = np.linalg.norm(vec_e) * np.linalg.norm(vec_n)
        cos_angle = dot / norms
        assert abs(cos_angle) < 0.01  # nearly orthogonal
