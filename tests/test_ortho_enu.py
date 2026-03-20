# -*- coding: utf-8 -*-
"""
Tests for ENUGrid and ENU orthorectification integration.

Verifies ENUGrid construction, coordinate transforms, sub_grid extraction,
and integration with Orthorectifier and OrthoBuilder.

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

from grdl.image_processing.ortho.enu_grid import ENUGrid
from grdl.image_processing.ortho.ortho import Orthorectifier, OutputGrid
from grdl.image_processing.ortho.ortho_builder import OrthoBuilder, OrthoResult
from grdl.geolocation.coordinates import geodetic_to_enu, enu_to_geodetic


# ===================================================================
# Helpers
# ===================================================================

class MockGeolocation:
    """Minimal geolocation for testing — identity-like lat/lon mapping.

    Maps pixel (row, col) to lat/lon via a simple affine:
        lat = center_lat + (nrows/2 - row) * pixel_deg
        lon = center_lon + (col - ncols/2) * pixel_deg
    """

    def __init__(
        self,
        nrows: int = 200,
        ncols: int = 200,
        center_lat: float = 36.0,
        center_lon: float = -75.5,
        pixel_deg: float = 0.001,
    ):
        self.nrows = nrows
        self.ncols = ncols
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.pixel_deg = pixel_deg
        self.shape = (nrows, ncols)

    def image_to_latlon(self, row, col, height=0.0):
        row = np.asarray(row, dtype=np.float64)
        col = np.asarray(col, dtype=np.float64)
        lat = self.center_lat + (self.nrows / 2.0 - row) * self.pixel_deg
        lon = self.center_lon + (col - self.ncols / 2.0) * self.pixel_deg
        h = np.full_like(lat, height if np.isscalar(height) else 0.0)
        return lat, lon, h

    def latlon_to_image(self, lat, lon, height=0.0):
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        row = self.nrows / 2.0 - (lat - self.center_lat) / self.pixel_deg
        col = (lon - self.center_lon) / self.pixel_deg + self.ncols / 2.0
        return row, col

    def get_bounds(self):
        """Return (min_lon, min_lat, max_lon, max_lat)."""
        half_rows = self.nrows / 2.0 * self.pixel_deg
        half_cols = self.ncols / 2.0 * self.pixel_deg
        return (
            self.center_lon - half_cols,
            self.center_lat - half_rows,
            self.center_lon + half_cols,
            self.center_lat + half_rows,
        )


# ===================================================================
# ENUGrid construction
# ===================================================================

class TestENUGridConstruction:
    """Tests for ENUGrid init and validation."""

    def test_basic_construction(self):
        """Grid with valid parameters creates correct rows/cols."""
        grid = ENUGrid(
            ref_lat=36.0, ref_lon=-75.5, ref_alt=0.0,
            min_east=-1000, max_east=1000,
            min_north=-1000, max_north=1000,
            pixel_size_east=1.0, pixel_size_north=1.0,
        )
        assert grid.rows == 2000
        assert grid.cols == 2000

    def test_non_square_grid(self):
        """Rectangular grid with different east/north extents."""
        grid = ENUGrid(
            ref_lat=36.0, ref_lon=-75.5, ref_alt=0.0,
            min_east=-500, max_east=500,
            min_north=-1000, max_north=1000,
            pixel_size_east=2.0, pixel_size_north=1.0,
        )
        assert grid.rows == 2000
        assert grid.cols == 500

    def test_invalid_bounds_east(self):
        """max_east <= min_east raises ValueError."""
        with pytest.raises(ValueError, match="max_east"):
            ENUGrid(
                ref_lat=0, ref_lon=0, ref_alt=0,
                min_east=100, max_east=-100,
                min_north=-100, max_north=100,
                pixel_size_east=1.0, pixel_size_north=1.0,
            )

    def test_invalid_bounds_north(self):
        """max_north <= min_north raises ValueError."""
        with pytest.raises(ValueError, match="max_north"):
            ENUGrid(
                ref_lat=0, ref_lon=0, ref_alt=0,
                min_east=-100, max_east=100,
                min_north=100, max_north=-100,
                pixel_size_east=1.0, pixel_size_north=1.0,
            )

    def test_invalid_pixel_size(self):
        """Non-positive pixel size raises ValueError."""
        with pytest.raises(ValueError, match="pixel_size_east"):
            ENUGrid(
                ref_lat=0, ref_lon=0, ref_alt=0,
                min_east=-100, max_east=100,
                min_north=-100, max_north=100,
                pixel_size_east=0.0, pixel_size_north=1.0,
            )

    def test_from_geolocation(self):
        """from_geolocation auto-computes reference and bounds."""
        geo = MockGeolocation()
        grid = ENUGrid.from_geolocation(geo, pixel_size_m=10.0)
        assert grid.rows > 0
        assert grid.cols > 0
        assert abs(grid.ref_lat - geo.center_lat) < 0.01
        assert abs(grid.ref_lon - geo.center_lon) < 0.01

    def test_from_geolocation_with_margin(self):
        """Margin expands the grid bounds."""
        geo = MockGeolocation()
        grid_no_margin = ENUGrid.from_geolocation(geo, pixel_size_m=10.0)
        grid_margin = ENUGrid.from_geolocation(
            geo, pixel_size_m=10.0, margin_m=500.0
        )
        # Margin grid should be larger
        assert grid_margin.rows > grid_no_margin.rows
        assert grid_margin.cols > grid_no_margin.cols

    def test_from_geolocation_custom_ref(self):
        """Explicit ref_lat/ref_lon overrides auto-computation."""
        geo = MockGeolocation()
        grid = ENUGrid.from_geolocation(
            geo, pixel_size_m=10.0, ref_lat=40.0, ref_lon=-80.0,
        )
        assert grid.ref_lat == 40.0
        assert grid.ref_lon == -80.0


# ===================================================================
# ENUGrid coordinate transforms
# ===================================================================

class TestENUGridTransforms:
    """Tests for image_to_latlon and latlon_to_image."""

    @pytest.fixture
    def grid(self):
        return ENUGrid(
            ref_lat=36.0, ref_lon=-75.5, ref_alt=0.0,
            min_east=-1000, max_east=1000,
            min_north=-1000, max_north=1000,
            pixel_size_east=1.0, pixel_size_north=1.0,
        )

    def test_center_pixel_near_reference(self, grid):
        """Center pixel should map near the reference lat/lon."""
        lat, lon = grid.image_to_latlon(grid.rows / 2.0, grid.cols / 2.0)
        assert abs(float(lat) - grid.ref_lat) < 0.001
        assert abs(float(lon) - grid.ref_lon) < 0.001

    def test_round_trip_pixel(self, grid):
        """image_to_latlon → latlon_to_image round-trip."""
        test_row, test_col = 500.0, 750.0
        lat, lon = grid.image_to_latlon(test_row, test_col)
        row2, col2 = grid.latlon_to_image(lat, lon)
        assert abs(float(row2) - test_row) < 0.01
        assert abs(float(col2) - test_col) < 0.01

    def test_round_trip_array(self, grid):
        """Round-trip with array inputs."""
        rows = np.array([0.0, 500.0, 999.0, 1500.0, 1999.0])
        cols = np.array([0.0, 250.0, 999.0, 1500.0, 1999.0])
        lats, lons = grid.image_to_latlon(rows, cols)
        rows2, cols2 = grid.latlon_to_image(lats, lons)
        np.testing.assert_allclose(rows2, rows, atol=0.01)
        np.testing.assert_allclose(cols2, cols, atol=0.01)

    def test_row_increases_south(self, grid):
        """Increasing row should decrease latitude (move south)."""
        lat0, _ = grid.image_to_latlon(0.0, 1000.0)
        lat1, _ = grid.image_to_latlon(1999.0, 1000.0)
        assert float(lat0) > float(lat1)

    def test_col_increases_east(self, grid):
        """Increasing column should increase longitude (move east)."""
        _, lon0 = grid.image_to_latlon(1000.0, 0.0)
        _, lon1 = grid.image_to_latlon(1000.0, 1999.0)
        assert float(lon1) > float(lon0)


# ===================================================================
# ENUGrid sub_grid
# ===================================================================

class TestENUGridSubGrid:
    """Tests for sub_grid extraction."""

    @pytest.fixture
    def grid(self):
        return ENUGrid(
            ref_lat=36.0, ref_lon=-75.5, ref_alt=0.0,
            min_east=-1000, max_east=1000,
            min_north=-1000, max_north=1000,
            pixel_size_east=2.0, pixel_size_north=2.0,
        )

    def test_sub_grid_dimensions(self, grid):
        """Sub-grid has correct rows and cols."""
        sub = grid.sub_grid(100, 200, 300, 400)
        assert sub.rows == 200
        assert sub.cols == 200

    def test_sub_grid_preserves_reference(self, grid):
        """Sub-grid keeps the same reference point."""
        sub = grid.sub_grid(0, 0, 100, 100)
        assert sub.ref_lat == grid.ref_lat
        assert sub.ref_lon == grid.ref_lon
        assert sub.ref_alt == grid.ref_alt

    def test_sub_grid_preserves_pixel_size(self, grid):
        """Sub-grid keeps the same pixel sizes."""
        sub = grid.sub_grid(0, 0, 100, 100)
        assert sub.pixel_size_east == grid.pixel_size_east
        assert sub.pixel_size_north == grid.pixel_size_north

    def test_sub_grid_coordinate_consistency(self, grid):
        """Sub-grid pixel (0,0) matches parent pixel at (row_start, col_start)."""
        row_start, col_start = 100, 200
        sub = grid.sub_grid(row_start, col_start, 300, 400)
        lat_parent, lon_parent = grid.image_to_latlon(
            float(row_start), float(col_start),
        )
        lat_sub, lon_sub = sub.image_to_latlon(0.0, 0.0)
        np.testing.assert_allclose(float(lat_sub), float(lat_parent), atol=1e-9)
        np.testing.assert_allclose(float(lon_sub), float(lon_parent), atol=1e-9)

    def test_sub_grid_invalid_negative_index(self, grid):
        with pytest.raises(ValueError, match="non-negative"):
            grid.sub_grid(-1, 0, 100, 100)

    def test_sub_grid_exceeds_bounds(self, grid):
        with pytest.raises(ValueError, match="exceed"):
            grid.sub_grid(0, 0, grid.rows + 1, grid.cols)

    def test_sub_grid_empty_region(self, grid):
        with pytest.raises(ValueError, match="Empty"):
            grid.sub_grid(100, 100, 100, 200)


# ===================================================================
# Orthorectifier with ENUGrid
# ===================================================================

class TestOrthorectifierENU:
    """Tests for Orthorectifier using an ENUGrid."""

    def test_compute_mapping_enu(self):
        """compute_mapping works with ENUGrid (duck typing)."""
        geo = MockGeolocation(nrows=100, ncols=100)
        grid = ENUGrid.from_geolocation(geo, pixel_size_m=50.0)
        ortho = Orthorectifier(
            geolocation=geo, output_grid=grid, interpolation='nearest',
        )
        src_rows, src_cols, valid = ortho.compute_mapping()
        assert src_rows.shape == (grid.rows, grid.cols)
        assert src_cols.shape == (grid.rows, grid.cols)
        assert valid.shape == (grid.rows, grid.cols)
        assert np.any(valid)  # at least some pixels should map

    def test_apply_enu(self):
        """apply() produces output with correct shape from ENUGrid."""
        geo = MockGeolocation(nrows=100, ncols=100)
        grid = ENUGrid.from_geolocation(geo, pixel_size_m=50.0)
        ortho = Orthorectifier(
            geolocation=geo, output_grid=grid, interpolation='nearest',
        )
        source = np.random.rand(100, 100).astype(np.float32)
        result = ortho.apply(source, nodata=0.0)
        assert result.shape == (grid.rows, grid.cols)
        assert np.any(result != 0.0)  # some pixels should be filled

    def test_metadata_enu(self):
        """get_output_geolocation_metadata returns ENU-specific keys."""
        geo = MockGeolocation(nrows=100, ncols=100)
        grid = ENUGrid.from_geolocation(geo, pixel_size_m=50.0)
        ortho = Orthorectifier(
            geolocation=geo, output_grid=grid, interpolation='nearest',
        )
        meta = ortho.get_output_geolocation_metadata()
        assert meta['crs'] == 'ENU'
        assert 'reference_point' in meta
        assert 'bounds_enu' in meta
        assert 'pixel_size_east' in meta
        assert 'pixel_size_north' in meta
        assert meta['reference_point'] == (
            grid.ref_lat, grid.ref_lon, grid.ref_alt,
        )

    def test_metadata_latlon_grid(self):
        """get_output_geolocation_metadata returns WGS84 for OutputGrid."""
        geo = MockGeolocation(nrows=100, ncols=100)
        grid = OutputGrid.from_geolocation(geo, 0.001, 0.001)
        ortho = Orthorectifier(
            geolocation=geo, output_grid=grid, interpolation='nearest',
        )
        meta = ortho.get_output_geolocation_metadata()
        assert meta['crs'] == 'WGS84'
        assert 'bounds' in meta


# ===================================================================
# OrthoBuilder with ENU
# ===================================================================

class TestOrthoBuilderENU:
    """Tests for OrthoBuilder.with_enu_grid()."""

    def test_pipeline_enu_basic(self):
        """Pipeline with .with_enu_grid() produces ENU result."""
        geo = MockGeolocation(nrows=100, ncols=100)
        source = np.random.rand(100, 100).astype(np.float32)
        result = (
            OrthoBuilder()
            .with_source_array(source)
            .with_geolocation(geo)
            .with_enu_grid(pixel_size_m=50.0)
            .with_interpolation('nearest')
            .run()
        )
        assert isinstance(result, OrthoResult)
        assert isinstance(result.output_grid, ENUGrid)
        assert result.is_enu is True

    def test_pipeline_enu_reference_point(self):
        """ENU result exposes reference point."""
        geo = MockGeolocation(nrows=100, ncols=100)
        source = np.random.rand(100, 100).astype(np.float32)
        result = (
            OrthoBuilder()
            .with_source_array(source)
            .with_geolocation(geo)
            .with_enu_grid(pixel_size_m=50.0)
            .with_interpolation('nearest')
            .run()
        )
        ref = result.enu_reference_point
        assert ref is not None
        assert abs(ref[0] - geo.center_lat) < 0.01
        assert abs(ref[1] - geo.center_lon) < 0.01

    def test_pipeline_enu_custom_ref(self):
        """Custom ref_lat/ref_lon passed through pipeline."""
        geo = MockGeolocation(nrows=100, ncols=100)
        source = np.random.rand(100, 100).astype(np.float32)
        result = (
            OrthoBuilder()
            .with_source_array(source)
            .with_geolocation(geo)
            .with_enu_grid(
                pixel_size_m=50.0, ref_lat=40.0, ref_lon=-80.0,
            )
            .with_interpolation('nearest')
            .run()
        )
        ref = result.enu_reference_point
        assert ref[0] == 40.0
        assert ref[1] == -80.0

    def test_pipeline_enu_pixel_size(self):
        """ENU result exposes pixel size in meters."""
        geo = MockGeolocation(nrows=100, ncols=100)
        source = np.random.rand(100, 100).astype(np.float32)
        result = (
            OrthoBuilder()
            .with_source_array(source)
            .with_geolocation(geo)
            .with_enu_grid(pixel_size_m=25.0)
            .with_interpolation('nearest')
            .run()
        )
        ps = result.pixel_size_meters
        assert ps is not None
        assert ps[0] == 25.0
        assert ps[1] == 25.0

    def test_pipeline_enu_bounds(self):
        """ENU result exposes bounds in meters."""
        geo = MockGeolocation(nrows=100, ncols=100)
        source = np.random.rand(100, 100).astype(np.float32)
        result = (
            OrthoBuilder()
            .with_source_array(source)
            .with_geolocation(geo)
            .with_enu_grid(pixel_size_m=50.0)
            .with_interpolation('nearest')
            .run()
        )
        bounds = result.bounds_meters
        assert bounds is not None
        # min_east < max_east, min_north < max_north
        assert bounds[0] < bounds[2]
        assert bounds[1] < bounds[3]

    def test_pipeline_latlon_not_enu(self):
        """Standard lat/lon pipeline result is not ENU."""
        geo = MockGeolocation(nrows=100, ncols=100)
        source = np.random.rand(100, 100).astype(np.float32)
        result = (
            OrthoBuilder()
            .with_source_array(source)
            .with_geolocation(geo)
            .with_resolution(0.001, 0.001)
            .with_interpolation('nearest')
            .run()
        )
        assert result.is_enu is False
        assert result.enu_reference_point is None
        assert result.pixel_size_meters is None
        assert result.bounds_meters is None

    def test_pipeline_enu_tiled(self):
        """Tiled ENU pipeline produces correct output."""
        geo = MockGeolocation(nrows=100, ncols=100)
        source = np.random.rand(100, 100).astype(np.float32)
        result = (
            OrthoBuilder()
            .with_source_array(source)
            .with_geolocation(geo)
            .with_enu_grid(pixel_size_m=50.0)
            .with_interpolation('nearest')
            .with_tile_size(128)
            .run()
        )
        assert isinstance(result, OrthoResult)
        assert isinstance(result.output_grid, ENUGrid)
        assert result.is_enu is True
        assert result.data.ndim == 2


# ===================================================================
# ENUGrid repr
# ===================================================================

def test_enu_grid_repr():
    """ENUGrid repr includes key info."""
    grid = ENUGrid(
        ref_lat=36.0, ref_lon=-75.5, ref_alt=0.0,
        min_east=-100, max_east=100,
        min_north=-200, max_north=200,
        pixel_size_east=1.0, pixel_size_north=1.0,
    )
    r = repr(grid)
    assert 'ENUGrid' in r
    assert '36.0000' in r
    assert '-75.5000' in r
