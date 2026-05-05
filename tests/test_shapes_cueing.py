# -*- coding: utf-8 -*-
"""
Tests for grdl.shapes.cueing and DetectionSet.filter_by_region.

Builds a synthetic image with two bright targets -- one inside a shape,
one outside -- and verifies:

- Without valid_mask, both are detected.
- With a shape-derived valid_mask, only the inside target survives.
- cued_detect produces the same result as the manual two-step.
- DetectionSet.filter_by_region(mode='centroid') drops outside
  detections from an unmasked run.

Dependencies
------------
pytest
rasterio
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

import numpy as np
import pytest
from rasterio.transform import Affine

from grdl.geolocation.eo.affine import AffineGeolocation
from grdl.image_processing.detection.cfar import CACFARDetector
from grdl.shapes import Circle, cued_detect


@pytest.fixture
def geo():
    """100 m/pixel basemap covering a small region."""
    transform = Affine(1e-3, 0.0, -118.2, 0.0, -1e-3, 34.1)
    return AffineGeolocation(transform, (256, 256), 'EPSG:4326')


def _make_image_with_two_targets():
    """256x256 image: bright square inside and outside an ROI circle."""
    img = np.full((256, 256), 10.0, dtype=np.float64)
    # Add noise so CFAR has something to measure.
    rng = np.random.default_rng(42)
    img += rng.normal(0.0, 1.0, img.shape)
    # Target A at (64, 64): bright square, 8x8, amp=200
    img[60:68, 60:68] = 200.0
    # Target B at (192, 192): same brightness
    img[188:196, 188:196] = 200.0
    return img


def _cfar():
    return CACFARDetector(
        guard_cells=4, training_cells=8, pfa=1e-4, min_pixels=2,
        assumption='gaussian',
    )


class TestValidMaskKwarg:
    def test_no_mask_detects_both(self, geo):
        img = _make_image_with_two_targets()
        detector = _cfar()
        detset = detector.detect(img, geolocation=geo)
        assert len(detset) >= 2

    def test_mask_suppresses_out_of_region(self, geo):
        img = _make_image_with_two_targets()
        detector = _cfar()
        # ROI centred on target A only (around pixel (64, 64)).
        # Image row 64 -> lat = 34.1 - 64 * 1e-3 = 34.036
        # Image col 64 -> lon = -118.2 + 64 * 1e-3 = -118.136
        circle = Circle(34.1 - 64 * 1e-3, -118.2 + 64 * 1e-3, radius_m=5_000.0)
        mask = circle.rasterize(
            geolocation=geo, image_shape=img.shape,
            fill=True, outline=False,
        )
        detset = detector.detect(img, geolocation=geo, valid_mask=mask)
        # Every detection should have its centroid inside the mask.
        for det in detset:
            if det.pixel_geometry is None:
                continue
            c = det.pixel_geometry.centroid
            r, col = int(c.y), int(c.x)
            assert mask[r, col], (
                f"Detection at (row={r}, col={col}) is outside the ROI"
            )


class TestCuedDetect:
    def test_matches_manual_two_step(self, geo):
        img = _make_image_with_two_targets()
        detector = _cfar()
        circle = Circle(34.1 - 64 * 1e-3, -118.2 + 64 * 1e-3, radius_m=5_000.0)
        via_cued = cued_detect(detector, img, circle, geo)
        mask = circle.rasterize(
            geolocation=geo, image_shape=img.shape, fill=True, outline=False,
        )
        manual = detector.detect(img, geolocation=geo, valid_mask=mask)
        assert len(via_cued) == len(manual)


class TestFilterByRegion:
    def test_centroid_mode(self, geo):
        img = _make_image_with_two_targets()
        detector = _cfar()
        detset = detector.detect(img, geolocation=geo)
        circle = Circle(34.1 - 64 * 1e-3, -118.2 + 64 * 1e-3, radius_m=5_000.0)
        filtered = detset.filter_by_region(
            shape=circle, geolocation=geo, mode='centroid',
        )
        assert len(filtered) <= len(detset)
        # At least one detection should survive (the inside target).
        assert len(filtered) >= 1
        # Metadata should record the filter mode
        assert filtered.metadata.get('filter_by_region') == 'centroid'
