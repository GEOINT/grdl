# -*- coding: utf-8 -*-
"""WebMercatorGrid tests — construction, roundtrip, sub_grid, protocol."""

import pytest
import numpy as np

from grdl.image_processing.ortho.web_mercator_grid import WebMercatorGrid
from grdl.image_processing.ortho.ortho import OutputGridProtocol


class TestConstruction:
    """Test WebMercatorGrid construction and validation."""

    def test_basic(self):
        grid = WebMercatorGrid(
            min_x=3270000, max_x=3280000,
            min_y=-3020000, max_y=-3010000,
            pixel_size=10.0,
        )
        assert grid.rows == 1000
        assert grid.cols == 1000

    def test_from_bounds_latlon(self):
        grid = WebMercatorGrid.from_bounds_latlon(
            min_lat=-26.1, max_lat=-26.0,
            min_lon=29.4, max_lon=29.5,
            pixel_size=10.0,
        )
        assert grid.rows > 0
        assert grid.cols > 0

    def test_invalid_pixel_size(self):
        with pytest.raises(ValueError, match="pixel_size"):
            WebMercatorGrid(
                min_x=0, max_x=100, min_y=0, max_y=100, pixel_size=0)

    def test_invalid_bounds(self):
        with pytest.raises(ValueError, match="max_x"):
            WebMercatorGrid(
                min_x=100, max_x=0, min_y=0, max_y=100, pixel_size=1.0)

    def test_protocol_compliance(self):
        grid = WebMercatorGrid(
            min_x=3270000, max_x=3270100,
            min_y=-3020000, max_y=-3019900,
            pixel_size=1.0,
        )
        assert isinstance(grid, OutputGridProtocol)


class TestRoundtrip:
    """Test coordinate roundtrip: pixel → latlon → pixel."""

    def test_center_roundtrip(self):
        grid = WebMercatorGrid(
            min_x=3270000, max_x=3280000,
            min_y=-3020000, max_y=-3010000,
            pixel_size=10.0,
        )
        cr, cc = 500.0, 500.0
        lat, lon = grid.image_to_latlon(cr, cc)
        r2, c2 = grid.latlon_to_image(lat, lon)
        assert abs(r2 - cr) < 0.01
        assert abs(c2 - cc) < 0.01

    def test_corner_roundtrip(self):
        grid = WebMercatorGrid(
            min_x=3270000, max_x=3280000,
            min_y=-3020000, max_y=-3010000,
            pixel_size=10.0,
        )
        corners = [(0.0, 0.0), (0.0, 999.0), (999.0, 0.0), (999.0, 999.0)]
        for r, c in corners:
            lat, lon = grid.image_to_latlon(r, c)
            r2, c2 = grid.latlon_to_image(lat, lon)
            assert abs(r2 - r) < 0.01, f"Row error at ({r},{c})"
            assert abs(c2 - c) < 0.01, f"Col error at ({r},{c})"

    def test_array_input(self):
        grid = WebMercatorGrid(
            min_x=3270000, max_x=3280000,
            min_y=-3020000, max_y=-3010000,
            pixel_size=10.0,
        )
        rows = np.array([0.0, 250.0, 500.0, 750.0, 999.0])
        cols = np.array([0.0, 250.0, 500.0, 750.0, 999.0])
        lats, lons = grid.image_to_latlon(rows, cols)
        assert lats.shape == (5,)
        rows2, cols2 = grid.latlon_to_image(lats, lons)
        np.testing.assert_allclose(rows2, rows, atol=0.01)
        np.testing.assert_allclose(cols2, cols, atol=0.01)


class TestSubGrid:
    """Test sub_grid extraction."""

    def test_sub_grid_dims(self):
        grid = WebMercatorGrid(
            min_x=3270000, max_x=3280000,
            min_y=-3020000, max_y=-3010000,
            pixel_size=10.0,
        )
        sub = grid.sub_grid(10, 20, 50, 80)
        assert sub.rows == 40
        assert sub.cols == 60

    def test_sub_grid_latlon_matches(self):
        grid = WebMercatorGrid(
            min_x=3270000, max_x=3280000,
            min_y=-3020000, max_y=-3010000,
            pixel_size=10.0,
        )
        sub = grid.sub_grid(10, 20, 50, 80)
        lat1, lon1 = grid.image_to_latlon(25.0, 40.0)
        lat2, lon2 = sub.image_to_latlon(15.0, 20.0)
        assert abs(lat1 - lat2) < 1e-8
        assert abs(lon1 - lon2) < 1e-8
