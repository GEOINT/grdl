# -*- coding: utf-8 -*-
"""UTMGrid tests — construction, roundtrip, sub_grid, protocol."""

import pytest
import numpy as np

from grdl.image_processing.ortho.utm_grid import UTMGrid
from grdl.image_processing.ortho.ortho import OutputGridProtocol


class TestConstruction:
    """Test UTMGrid construction and validation."""

    def test_basic(self):
        grid = UTMGrid(
            zone=56, north=False,
            min_easting=300000, max_easting=310000,
            min_northing=7100000, max_northing=7110000,
            pixel_size=10.0,
        )
        assert grid.rows == 1000
        assert grid.cols == 1000
        assert grid.epsg == 32756
        assert grid.zone == 56
        assert grid.north is False

    def test_invalid_zone(self):
        with pytest.raises(ValueError, match="zone"):
            UTMGrid(zone=0, north=True,
                    min_easting=0, max_easting=100,
                    min_northing=0, max_northing=100,
                    pixel_size=1.0)

    def test_invalid_pixel_size(self):
        with pytest.raises(ValueError, match="pixel_size"):
            UTMGrid(zone=33, north=True,
                    min_easting=0, max_easting=100,
                    min_northing=0, max_northing=100,
                    pixel_size=-1.0)

    def test_invalid_bounds(self):
        with pytest.raises(ValueError, match="max_easting"):
            UTMGrid(zone=33, north=True,
                    min_easting=100, max_easting=0,
                    min_northing=0, max_northing=100,
                    pixel_size=1.0)

    def test_protocol_compliance(self):
        grid = UTMGrid(
            zone=33, north=True,
            min_easting=500000, max_easting=500100,
            min_northing=5500000, max_northing=5500100,
            pixel_size=1.0,
        )
        assert isinstance(grid, OutputGridProtocol)


class TestRoundtrip:
    """Test coordinate roundtrip: pixel → latlon → pixel."""

    def test_center_roundtrip(self):
        grid = UTMGrid(
            zone=33, north=True,
            min_easting=500000, max_easting=501000,
            min_northing=5500000, max_northing=5501000,
            pixel_size=1.0,
        )
        cr, cc = 500.0, 500.0
        lat, lon = grid.image_to_latlon(cr, cc)
        r2, c2 = grid.latlon_to_image(lat, lon)
        assert abs(r2 - cr) < 0.01
        assert abs(c2 - cc) < 0.01

    def test_corner_roundtrip(self):
        grid = UTMGrid(
            zone=33, north=True,
            min_easting=500000, max_easting=501000,
            min_northing=5500000, max_northing=5501000,
            pixel_size=10.0,
        )
        corners = [(0.0, 0.0), (0.0, 99.0), (99.0, 0.0), (99.0, 99.0)]
        for r, c in corners:
            lat, lon = grid.image_to_latlon(r, c)
            r2, c2 = grid.latlon_to_image(lat, lon)
            assert abs(r2 - r) < 0.01, f"Row error at ({r},{c})"
            assert abs(c2 - c) < 0.01, f"Col error at ({r},{c})"

    def test_array_input(self):
        grid = UTMGrid(
            zone=33, north=True,
            min_easting=500000, max_easting=501000,
            min_northing=5500000, max_northing=5501000,
            pixel_size=10.0,
        )
        rows = np.array([0.0, 25.0, 50.0, 75.0, 99.0])
        cols = np.array([0.0, 25.0, 50.0, 75.0, 99.0])
        lats, lons = grid.image_to_latlon(rows, cols)
        assert lats.shape == (5,)
        rows2, cols2 = grid.latlon_to_image(lats, lons)
        np.testing.assert_allclose(rows2, rows, atol=0.01)
        np.testing.assert_allclose(cols2, cols, atol=0.01)


class TestSubGrid:
    """Test sub_grid extraction."""

    def test_sub_grid_dims(self):
        grid = UTMGrid(
            zone=33, north=True,
            min_easting=500000, max_easting=501000,
            min_northing=5500000, max_northing=5501000,
            pixel_size=10.0,
        )
        sub = grid.sub_grid(10, 20, 50, 80)
        assert sub.rows == 40
        assert sub.cols == 60
        assert sub.zone == grid.zone
        assert sub.north == grid.north

    def test_sub_grid_latlon_matches(self):
        grid = UTMGrid(
            zone=33, north=True,
            min_easting=500000, max_easting=501000,
            min_northing=5500000, max_northing=5501000,
            pixel_size=10.0,
        )
        sub = grid.sub_grid(10, 20, 50, 80)
        lat1, lon1 = grid.image_to_latlon(25.0, 40.0)
        lat2, lon2 = sub.image_to_latlon(15.0, 20.0)
        assert abs(lat1 - lat2) < 1e-8
        assert abs(lon1 - lon2) < 1e-8
