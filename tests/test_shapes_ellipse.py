# -*- coding: utf-8 -*-
"""
Tests for grdl.shapes.Ellipse and GeodesicEllipse.

Covers:
- Circle parity: semi_major == semi_minor ~= Circle perimeter at small
  radius (tangent-plane valid regime).
- Covariance round-trip via Ellipse.from_covariance.
- Axis-aligned geometry (rotation_deg=0): major along north, minor
  along east.
- GeodesicEllipse with coincident foci reproduces a Circle.

Dependencies
------------
pytest
pyproj

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
from pyproj import Geod

from grdl.shapes import Circle, Ellipse, GeodesicEllipse


_GEOD = Geod(ellps='WGS84')


class TestEllipseParametric:
    def test_circle_equivalent_small_radius(self):
        """Ellipse with a==b ~= Circle for radii where tangent-plane is valid."""
        center = (34.05, -118.19)
        r = 500.0
        ellipse = Ellipse(*center, semi_major_m=r, semi_minor_m=r)
        circle = Circle(*center, radius_m=r)
        e_pts = ellipse._perimeter_latlon(256)
        # Every ellipse point must land within ~1 cm of radius R
        # (tangent-plane error at 500 m is microscopic).
        lon0 = np.full(len(e_pts), center[1])
        lat0 = np.full(len(e_pts), center[0])
        _, _, dist = _GEOD.inv(lon0, lat0, e_pts[:, 1], e_pts[:, 0])
        assert np.allclose(dist, r, atol=0.01)

    def test_covariance_roundtrip(self):
        """from_covariance(cov(e)) should reproduce the original ellipse."""
        original = Ellipse(
            34.05, -118.19,
            semi_major_m=200.0, semi_minor_m=100.0, rotation_deg=30.0,
        )
        cov = original.covariance
        rebuilt = Ellipse.from_covariance(34.05, -118.19, cov)
        assert abs(rebuilt.semi_major_m - 200.0) < 1e-6
        assert abs(rebuilt.semi_minor_m - 100.0) < 1e-6
        # rotation_deg has a 180 ambiguity -- check either matches
        rot_delta = abs(rebuilt.rotation_deg - 30.0) % 180.0
        assert min(rot_delta, 180.0 - rot_delta) < 1e-6

    def test_axis_alignment(self):
        """rotation=0: major axis along north, minor along east."""
        a, b = 300.0, 100.0
        ellipse = Ellipse(34.05, -118.19, a, b, rotation_deg=0.0)
        cov = ellipse.covariance
        # cov[east, east] == b**2, cov[north, north] == a**2
        assert abs(cov[0, 0] - b * b) < 1e-6
        assert abs(cov[1, 1] - a * a) < 1e-6
        assert abs(cov[0, 1]) < 1e-6


class TestGeodesicEllipse:
    def test_coincident_foci_is_circle(self):
        """When both foci coincide, locus reduces to a circle of radius sum/2."""
        center = (0.0, 0.0)
        s = 2_000.0
        ellipse = GeodesicEllipse(
            focus1_lat=center[0], focus1_lon=center[1],
            focus2_lat=center[0], focus2_lon=center[1],
            sum_distance_m=s,
        )
        pts = ellipse._perimeter_latlon(128)
        # Every point should be radius s/2 from the center.
        lon0 = np.full(len(pts), center[1])
        lat0 = np.full(len(pts), center[0])
        _, _, dist = _GEOD.inv(lon0, lat0, pts[:, 1], pts[:, 0])
        assert np.allclose(dist, s / 2.0, atol=1.0)

    def test_focal_sum_holds(self):
        """Every perimeter point satisfies d(f1, P) + d(f2, P) == sum."""
        f1 = (34.0, -118.2)
        f2 = (34.0, -118.1)  # ~9 km separation at lat 34
        s = 20_000.0
        ellipse = GeodesicEllipse(
            focus1_lat=f1[0], focus1_lon=f1[1],
            focus2_lat=f2[0], focus2_lon=f2[1],
            sum_distance_m=s,
        )
        pts = ellipse._perimeter_latlon(64)
        lats = pts[:, 0]
        lons = pts[:, 1]
        _, _, d1 = _GEOD.inv(
            np.full_like(lats, f1[1]), np.full_like(lats, f1[0]), lons, lats,
        )
        _, _, d2 = _GEOD.inv(
            np.full_like(lats, f2[1]), np.full_like(lats, f2[0]), lons, lats,
        )
        assert np.allclose(np.asarray(d1) + np.asarray(d2), s, atol=1.0)

    def test_rejects_sum_less_than_focal_distance(self):
        with pytest.raises(ValueError):
            GeodesicEllipse(
                focus1_lat=0.0, focus1_lon=0.0,
                focus2_lat=0.0, focus2_lon=1.0,  # ~111 km
                sum_distance_m=1_000.0,
            )
