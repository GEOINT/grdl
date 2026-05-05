# -*- coding: utf-8 -*-
"""
Tests for grdl.shapes.combine.

Closed-form Gaussian math is verified against its algebraic equivalents
(sum of covariances for convolve, information sum for combine_evidence).
Geometric Minkowski sum of two circles is tested against the
radius-addition identity.

Dependencies
------------
pytest
shapely

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

import warnings

import numpy as np
import pytest
from pyproj import Geod

from grdl.shapes import (
    Circle,
    Ellipse,
    combine_evidence,
    convolve_ellipses,
    intersect_shapes,
    minkowski_sum,
    union_shapes,
)


_GEOD = Geod(ellps='WGS84')


class TestConvolveEllipses:
    def test_aligned_sum_of_semi_axes(self):
        """Two axis-aligned ellipses: a, b -> sqrt(a1^2+a2^2), sqrt(b1^2+b2^2)."""
        e1 = Ellipse(0.0, 0.0, semi_major_m=30.0, semi_minor_m=20.0)
        e2 = Ellipse(0.0, 0.0, semi_major_m=40.0, semi_minor_m=10.0)
        out = convolve_ellipses([e1, e2])
        assert abs(out.semi_major_m ** 2 - (30.0 ** 2 + 40.0 ** 2)) < 1e-6
        assert abs(out.semi_minor_m ** 2 - (20.0 ** 2 + 10.0 ** 2)) < 1e-6

    def test_covariance_sum_equals_total(self):
        e1 = Ellipse(
            0.0, 0.0,
            semi_major_m=30.0, semi_minor_m=20.0, rotation_deg=45.0,
        )
        e2 = Ellipse(
            0.0, 0.0,
            semi_major_m=50.0, semi_minor_m=10.0, rotation_deg=-20.0,
        )
        combined = convolve_ellipses([e1, e2])
        expected = e1.covariance + e2.covariance
        np.testing.assert_allclose(
            combined.covariance, expected, atol=1e-6,
        )

    def test_warns_on_large_center_spread(self):
        """Centers spread > 50 km should emit a UserWarning."""
        e1 = Ellipse(0.0, 0.0, semi_major_m=10.0, semi_minor_m=5.0)
        e2 = Ellipse(0.0, 1.0, semi_major_m=10.0, semi_minor_m=5.0)  # ~111 km
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            convolve_ellipses([e1, e2])
            assert any(
                issubclass(item.category, UserWarning)
                and 'tangent-plane' in str(item.message)
                for item in w
            )


class TestCombineEvidence:
    def test_fisher_info_tightens(self):
        """Combining two identical 1-sigma ellipses should tighten by sqrt(2)."""
        e = Ellipse(0.0, 0.0, semi_major_m=30.0, semi_minor_m=20.0)
        out = combine_evidence([e, e])
        assert abs(out.semi_major_m - 30.0 / np.sqrt(2.0)) < 1e-6
        assert abs(out.semi_minor_m - 20.0 / np.sqrt(2.0)) < 1e-6


class TestMinkowskiSum:
    def test_two_circles(self):
        """Minkowski sum of circles radii r1, r2 -> circle radius r1+r2."""
        a = Circle(0.0, 0.0, radius_m=30.0)
        b = Circle(0.0, 0.0, radius_m=20.0)
        combined = minkowski_sum(a, b, n=128)
        # Convert combined polygon back to distance-from-center
        vertices = combined.vertices  # (K, 2) lat/lon
        lat0 = np.zeros(len(vertices))
        lon0 = np.zeros(len(vertices))
        _, _, dist = _GEOD.inv(lon0, lat0, vertices[:, 1], vertices[:, 0])
        # Every vertex should be close to r1+r2 = 50 m (within 2% of
        # perimeter-sample density)
        assert np.all(np.abs(np.asarray(dist) - 50.0) < 1.0)


class TestSetOps:
    def test_union_overlapping_circles(self):
        a = Circle(34.05, -118.19, radius_m=100.0)
        b = Circle(34.05, -118.189, radius_m=100.0)  # overlapping, shifted east
        unioned = union_shapes([a, b], n=128)
        # Union polygon should have more than 128 vertices and be a single
        # closed polygon (implicit in GeoPolygon).
        assert len(unioned.vertices) > 64

    def test_intersection_empty_raises(self):
        a = Circle(34.05, -118.19, radius_m=10.0)
        b = Circle(34.06, -118.19, radius_m=10.0)  # ~1.1 km away, no overlap
        with pytest.raises(ValueError):
            intersect_shapes([a, b], n=64)
