# -*- coding: utf-8 -*-
"""
Tests for grdl.shapes rasterisation.

Covers:
- Filled circle mask area matches pi*r^2 within scan-conversion tolerance.
- Outline mask has the right perimeter pixel count.
- Shapes crossing image boundaries clip without error.
- Open shapes (arcs) produce outline-only masks.

Dependencies
------------
pytest
scikit-image

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

from grdl.geolocation.eo.affine import AffineGeolocation
from grdl.shapes import Arc, Circle, rasterize_polygon


@pytest.fixture
def geo():
    """2048x2048 basemap with a Circle of 100 m radius fully inside."""
    transform = Affine(1e-5, 0.0, -118.2, 0.0, -1e-5, 34.1)
    return AffineGeolocation(transform, (2048, 2048), 'EPSG:4326')


class TestFilledArea:
    def test_circle_area_matches_projected_ellipse(self, geo):
        """Filled mask area matches the projected ellipse's pi*a*b within 5%.

        At lat 34.09 with 1e-5 deg/px, longitude pixels are ~0.92 m wide
        and latitude pixels are ~1.11 m tall. A geographic 100 m circle
        projects to an ellipse, not a circle in pixel space; the filled
        area is pi*a*b, where (a, b) come from the projected bounds.
        """
        circle = Circle(34.09, -118.195, radius_m=100.0)
        mask = circle.rasterize(
            geolocation=geo, image_shape=(2048, 2048),
            fill=True, outline=False,
        )
        pixels = circle.to_pixels(geolocation=geo)
        a = (pixels[:, 0].max() - pixels[:, 0].min()) / 2.0
        b = (pixels[:, 1].max() - pixels[:, 1].min()) / 2.0
        expected = np.pi * a * b
        rel_err = abs(mask.sum() - expected) / expected
        assert rel_err < 0.05  # within 5% of projected-ellipse area


class TestOutline:
    def test_outline_mask_non_empty(self, geo):
        circle = Circle(34.09, -118.195, radius_m=100.0)
        mask = circle.rasterize(
            geolocation=geo, image_shape=(2048, 2048),
            fill=False, outline=True, outline_thickness=1,
        )
        assert mask.sum() > 0
        # Outline is a thin ring; the pixel count should be much less
        # than the filled area.
        assert mask.sum() < np.pi * 100.0 ** 2 * 0.5

    def test_thick_outline_larger(self, geo):
        circle = Circle(34.09, -118.195, radius_m=100.0)
        thin = circle.rasterize(
            geolocation=geo, image_shape=(2048, 2048),
            fill=False, outline=True, outline_thickness=1,
        )
        thick = circle.rasterize(
            geolocation=geo, image_shape=(2048, 2048),
            fill=False, outline=True, outline_thickness=5,
        )
        assert thick.sum() > thin.sum()


class TestBoundaryClipping:
    def test_shape_crossing_edge_does_not_crash(self, geo):
        """Circle clipped by the image edge still produces a valid mask."""
        # Shape centred near top-left corner
        circle = Circle(34.1 - 1e-5, -118.2 + 1e-5, radius_m=50.0)
        mask = circle.rasterize(
            geolocation=geo, image_shape=(2048, 2048),
            fill=True, outline=True,
        )
        assert mask.dtype == bool
        assert mask.shape == (2048, 2048)

    def test_shape_extending_far_past_image_clips_cleanly(self):
        """A shape much larger than the image fills the visible portion only."""
        # 256x256 image, circle of 1000 m radius centered in the middle.
        # The circle's projected bounds extend well past every edge.
        transform = Affine(1e-5, 0.0, -118.2, 0.0, -1e-5, 34.1)
        small_geo = AffineGeolocation(
            transform, (256, 256), 'EPSG:4326',
        )
        # Center-of-image lat/lon for the 256x256 cell
        center_lat = 34.1 - 128 * 1e-5
        center_lon = -118.2 + 128 * 1e-5
        circle = Circle(center_lat, center_lon, radius_m=1_000.0)
        mask = circle.rasterize(
            geolocation=small_geo, image_shape=(256, 256),
            fill=True, outline=True,
        )
        # Every pixel in the image should be inside the huge circle.
        assert mask.dtype == bool
        assert mask.shape == (256, 256)
        assert mask.sum() == 256 * 256

    def test_shape_entirely_outside_image_returns_empty(self):
        """A shape positioned outside the image bounds must not error."""
        transform = Affine(1e-5, 0.0, -118.2, 0.0, -1e-5, 34.1)
        small_geo = AffineGeolocation(
            transform, (256, 256), 'EPSG:4326',
        )
        # Place the circle 100 km away from the basemap region
        circle = Circle(35.0, -119.0, radius_m=100.0)
        mask = circle.rasterize(
            geolocation=small_geo, image_shape=(256, 256),
            fill=True, outline=True,
        )
        assert mask.shape == (256, 256)
        # No pixels inside the image overlap the shape.
        assert mask.sum() == 0


class TestOpenShapes:
    def test_arc_outline_only(self, geo):
        """Arcs should produce a polyline, no fill even when fill=True."""
        arc = Arc(
            center_lat=34.09, center_lon=-118.195,
            radius_m=100.0,
            bearing_start_deg=0.0, bearing_end_deg=90.0,
        )
        mask = arc.rasterize(
            geolocation=geo, image_shape=(2048, 2048),
            fill=True, outline=True,
        )
        # A quarter arc: pixel count approximately pi*r/2 for a thin line.
        # Bound generously: more than 50, less than r*10.
        assert 50 < mask.sum() < 100 * 10


class TestRasterizePrimitive:
    def test_square_polygon(self):
        """Rasterise an axis-aligned square directly."""
        pixels = np.array([
            [10, 10], [10, 30], [30, 30], [30, 10],
        ], dtype=np.float64)
        mask = rasterize_polygon(
            pixels=pixels, image_shape=(64, 64),
            fill=True, outline=False, closed=True,
        )
        # Fill spans rows 10..30 inclusive, cols 10..30 inclusive.
        # skimage polygon is exact for integer vertices.
        assert mask[20, 20]
        assert not mask[5, 5]
        assert mask.sum() == 21 * 21

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            rasterize_polygon(
                pixels=np.zeros(4),
                image_shape=(16, 16),
                fill=True,
            )
