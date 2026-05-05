# -*- coding: utf-8 -*-
"""
Tests for grdl.shapes.display.

Smoke tests for overlay and burn-in -- the rasterization math is
covered in test_shapes_rasterize.py; here we confirm the
matplotlib/RGB wiring does not mutate input and returns the expected
shapes and dtypes.

Dependencies
------------
pytest
matplotlib (optional -- overlay tests skip if missing)

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
from grdl.shapes import Circle, burn_shape, overlay_shape


@pytest.fixture
def geo():
    transform = Affine(1e-5, 0.0, -118.2, 0.0, -1e-5, 34.1)
    return AffineGeolocation(transform, (1024, 1024), 'EPSG:4326')


class TestOverlay:
    def test_overlay_returns_matplotlib_polygon(self, geo):
        pytest.importorskip('matplotlib')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots()
        ax.imshow(np.zeros((1024, 1024)))
        circle = Circle(34.09, -118.195, radius_m=100.0)
        patch = overlay_shape(ax, circle, geo, color='red')
        assert isinstance(patch, mpatches.Polygon)
        # patch vertices count reflects adaptive refinement
        assert len(patch.get_xy()) >= 128
        plt.close(fig)


class TestBurnIn:
    def test_burn_shape_does_not_mutate_input(self, geo):
        circle = Circle(34.09, -118.195, radius_m=100.0)
        image = np.zeros((1024, 1024), dtype=np.uint8)
        image_before = image.copy()
        out = burn_shape(image, circle, geo, color=(255, 0, 0), thickness=2)
        assert out.shape == (1024, 1024, 3)
        assert out.dtype == np.uint8
        np.testing.assert_array_equal(image, image_before)

    def test_burn_shape_paints_outline(self, geo):
        circle = Circle(34.09, -118.195, radius_m=100.0)
        image = np.zeros((1024, 1024), dtype=np.uint8)
        out = burn_shape(image, circle, geo, color=(255, 0, 0), thickness=1)
        # At least some pixels should be red
        red_mask = (out[..., 0] == 255) & (out[..., 1] == 0) & (out[..., 2] == 0)
        assert red_mask.sum() > 50

    def test_burn_shape_with_fill(self, geo):
        circle = Circle(34.09, -118.195, radius_m=100.0)
        image = np.full((1024, 1024, 3), 128, dtype=np.uint8)
        out = burn_shape(
            image, circle, geo,
            color=(255, 0, 0), thickness=1,
            fill=True, fill_alpha=0.5,
        )
        # Locate one pixel inside the projected polygon and check that
        # the red channel was blended toward 255 (somewhere in
        # (128, 255]) while green and blue were dragged toward 0.
        from grdl.shapes.rasterize import rasterize_polygon
        pixels = circle.to_pixels(geolocation=geo)
        fill_mask = rasterize_polygon(
            pixels=pixels, image_shape=(1024, 1024),
            fill=True, outline=False, closed=True,
        )
        # Sample one definitely-inside pixel.
        inside_idx = np.argwhere(fill_mask)[len(np.argwhere(fill_mask)) // 2]
        r_val = int(out[inside_idx[0], inside_idx[1], 0])
        g_val = int(out[inside_idx[0], inside_idx[1], 1])
        assert 128 < r_val <= 255
        assert 0 <= g_val < 128
