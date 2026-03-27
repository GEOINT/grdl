# -*- coding: utf-8 -*-
"""
ChipGeolocation Tests - Coordinate adapter for image sub-regions.

Tests ChipGeolocation offset logic, round-trip accuracy, elevation
inheritance, and DEM handling delegation.

Dependencies
------------
pytest

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
2026-03-27

Modified
--------
2026-03-27
"""

import pytest
import numpy as np

from grdl.geolocation.base import Geolocation
from grdl.geolocation.chip import ChipGeolocation
from grdl.geolocation.elevation.constant import ConstantElevation


# ---------------------------------------------------------------------------
# Synthetic geolocation fixture
# ---------------------------------------------------------------------------

class _SyntheticGeo(Geolocation):
    """Simple affine geolocation for testing."""

    def __init__(self, shape=(100, 200)):
        super().__init__(shape=shape, crs='WGS84')
        self.origin_lat = -30.0
        self.origin_lon = 115.0
        self.pixel_size_lat = 0.001
        self.pixel_size_lon = 0.001

    def _image_to_latlon_array(self, rows, cols, height=0.0):
        lats = self.origin_lat - rows * self.pixel_size_lat
        lons = self.origin_lon + cols * self.pixel_size_lon
        heights = np.full_like(lats, height)
        return lats, lons, heights

    def _latlon_to_image_array(self, lats, lons, height=0.0):
        rows = (self.origin_lat - lats) / self.pixel_size_lat
        cols = (lons - self.origin_lon) / self.pixel_size_lon
        return rows, cols


@pytest.fixture
def full_geo():
    return _SyntheticGeo(shape=(100, 200))


@pytest.fixture
def chip_geo(full_geo):
    return ChipGeolocation(
        full_geo,
        row_offset=20,
        col_offset=50,
        shape=(30, 40),
    )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestChipGeolocationConstruction:

    def test_shape(self, chip_geo):
        assert chip_geo.shape == (30, 40)

    def test_crs_inherited(self, chip_geo):
        assert chip_geo.crs == 'WGS84'

    def test_elevation_inherited(self, full_geo):
        elev = ConstantElevation(height=500.0)
        full_geo.elevation = elev
        chip = ChipGeolocation(full_geo, 0, 0, shape=(10, 10))
        assert chip.elevation is elev

    def test_default_hae_delegated(self, full_geo, chip_geo):
        assert chip_geo.default_hae == full_geo.default_hae


# ---------------------------------------------------------------------------
# Coordinate transform tests
# ---------------------------------------------------------------------------

class TestChipGeolocationTransform:

    def test_image_to_latlon_offset(self, full_geo, chip_geo):
        """Chip pixel (0, 0) should map to full-image (20, 50)."""
        chip_result = chip_geo.image_to_latlon(0.0, 0.0)
        full_result = full_geo.image_to_latlon(20.0, 50.0)
        np.testing.assert_allclose(chip_result, full_result, atol=1e-12)

    def test_latlon_to_image_offset(self, full_geo, chip_geo):
        """Inverse should give chip-local coordinates."""
        # Get lat/lon of full image pixel (30, 60)
        latlon = full_geo.image_to_latlon(30.0, 60.0)
        lat, lon = float(latlon[0]), float(latlon[1])

        # In chip coords this should be (10, 10) since offset is (20, 50)
        chip_px = chip_geo.latlon_to_image(lat, lon)
        np.testing.assert_allclose(chip_px, [10.0, 10.0], atol=1e-6)

    def test_round_trip(self, chip_geo):
        """Round-trip through image_to_latlon and latlon_to_image."""
        row, col = 5.0, 10.0
        latlon = chip_geo.image_to_latlon(row, col)
        px = chip_geo.latlon_to_image(float(latlon[0]), float(latlon[1]))
        np.testing.assert_allclose(px, [row, col], atol=1e-6)

    def test_batch_transform(self, full_geo, chip_geo):
        """Batch (N, 2) input should match element-wise."""
        chip_pts = np.array([[0.0, 0.0], [5.0, 10.0], [15.0, 25.0]])
        full_pts = chip_pts + np.array([[20.0, 50.0]])

        chip_result = chip_geo.image_to_latlon(chip_pts)
        full_result = full_geo.image_to_latlon(full_pts)

        np.testing.assert_allclose(chip_result, full_result, atol=1e-12)

    def test_get_bounds(self, chip_geo):
        """get_bounds should return valid bounding box."""
        bounds = chip_geo.get_bounds()
        min_lon, min_lat, max_lon, max_lat = bounds
        assert min_lon < max_lon
        assert min_lat < max_lat
