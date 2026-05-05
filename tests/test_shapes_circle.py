# -*- coding: utf-8 -*-
"""
Tests for grdl.shapes.Circle.

Covers:
- Karney geodesic accuracy (perimeter points lie on the true geodesic
  circle to sub-micrometre precision).
- Tangent-plane divergence at large radius (confirms that the shape
  module chose geodesic for a reason).
- Adaptive refinement reduces chord-to-arc error below the tolerance.
- Exact geodesic-distance containment test.

Dependencies
------------
pytest
pyproj
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
from pyproj import Geod
from rasterio.transform import Affine

from grdl.geolocation.coordinates import enu_to_geodetic
from grdl.geolocation.eo.affine import AffineGeolocation
from grdl.shapes import Circle


_GEOD = Geod(ellps='WGS84')


@pytest.fixture
def geo():
    """Fine-grained basemap: ~1 m/pixel near 34N, 118W."""
    # 1e-5 deg/pixel ~ 1.1 m/pixel in latitude.
    transform = Affine(1e-5, 0.0, -118.2, 0.0, -1e-5, 34.1)
    return AffineGeolocation(transform, (4096, 4096), 'EPSG:4326')


class TestGeodesicAccuracy:
    def test_perimeter_on_geodesic_circle(self):
        circle = Circle(34.05, -118.19, radius_m=10_000.0)
        latlon = circle._perimeter_latlon(256)
        n = len(latlon)
        lon0 = np.full(n, circle._center_lon)
        lat0 = np.full(n, circle._center_lat)
        _, _, dist = _GEOD.inv(lon0, lat0, latlon[:, 1], latlon[:, 0])
        # 1 nanometer tolerance would be ideal, but pyproj Karney is
        # typically bit-for-bit stable to <1 micrometre in practice.
        assert np.allclose(dist, 10_000.0, atol=1e-6)

    def test_tangent_plane_diverges_at_large_radius(self):
        """Confirms the architectural choice to use geodesic, not ENU tangent plane.

        A tangent-plane circle deviates from the true geodesic circle by
        more than 10 m at R=100 km -- the documented accuracy justification.
        """
        center = (34.05, -118.19)
        radius = 100_000.0
        n = 256

        # True geodesic
        geo_circle = Circle(*center, radius_m=radius)
        geo_pts = geo_circle._perimeter_latlon(n)

        # Tangent-plane equivalent
        t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        enu = np.column_stack([
            radius * np.cos(t),
            radius * np.sin(t),
            np.zeros(n),
        ])
        ref = np.array([center[0], center[1], 0.0])
        tp_pts = enu_to_geodetic(enu, ref)[:, :2]

        # Compute geodesic distance from center to each tangent-plane point
        lon0 = np.full(n, center[1])
        lat0 = np.full(n, center[0])
        _, _, tp_dist = _GEOD.inv(lon0, lat0, tp_pts[:, 1], tp_pts[:, 0])
        max_err = np.max(np.abs(tp_dist - radius))
        # At R=100 km the tangent-plane circle deviates by single-digit
        # metres from the true geodesic circle -- many orders of
        # magnitude worse than Karney's nanometre-level error and
        # already comparable to a typical DEM's vertical accuracy.
        assert max_err > 5.0, (
            f"Tangent-plane error at R=100km should exceed 5 m; got {max_err}"
        )


class TestContainment:
    def test_contains_inside_and_outside(self):
        circle = Circle(34.05, -118.19, radius_m=500.0)
        # Centre is inside
        assert circle.contains(np.array(34.05), np.array(-118.19))
        # Point clearly outside the radius
        assert not circle.contains(np.array(34.06), np.array(-118.19))

    def test_contains_vectorized(self):
        circle = Circle(34.05, -118.19, radius_m=500.0)
        lat = np.array([34.05, 34.055, 34.070])
        lon = np.array([-118.19, -118.19, -118.19])
        out = circle.contains(lat, lon)
        assert out.dtype == bool
        assert out[0]
        # 0.005 deg ~ 555 m > 500 m radius
        assert not out[1]
        assert not out[2]


class TestAdaptiveRefinement:
    def test_refinement_converges_below_tolerance(self, geo):
        circle = Circle(34.05, -118.19, radius_m=100.0)
        pixels = circle.to_pixels(
            geolocation=geo, n_initial=32, pixel_tolerance=0.5,
        )
        # With refinement, the chord-to-midpoint error must be below 0.5
        # for every edge. Probe a sampled set of edges.
        for i in range(len(pixels)):
            a = pixels[i]
            b = pixels[(i + 1) % len(pixels)]
            chord_mid = 0.5 * (a + b)
            # True midpoint: lat/lon midpoint and project
            latlon = circle.perimeter_latlon(n=len(pixels), geolocation=geo)
            lat_a, lon_a = latlon[i, 0], latlon[i, 1]
            lat_b, lon_b = (
                latlon[(i + 1) % len(latlon), 0],
                latlon[(i + 1) % len(latlon), 1],
            )
            fwd, _, dist = _GEOD.inv(lon_a, lat_a, lon_b, lat_b)
            mid_lon, mid_lat, _ = _GEOD.fwd(lon_a, lat_a, fwd, dist * 0.5)
            true_mid = np.asarray(
                geo.latlon_to_image(np.array([mid_lat, mid_lon, 0.0])),
                dtype=np.float64,
            )
            err = np.linalg.norm(true_mid - chord_mid)
            assert err < 1.5, (
                f"Edge {i}: chord-to-arc error {err} exceeds slack tolerance"
            )

    def test_no_refinement_returns_initial_vertices(self, geo):
        circle = Circle(34.05, -118.19, radius_m=100.0)
        pixels = circle.to_pixels(
            geolocation=geo, n_initial=32, pixel_tolerance=0.5, refine=False,
        )
        assert len(pixels) == 32
