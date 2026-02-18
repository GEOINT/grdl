# -*- coding: utf-8 -*-
"""
Tiled Ortho Tests - Tests for OutputGrid.sub_grid, ROI, and tiled processing.

Dependencies
------------
pytest
scipy

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
2026-02-17

Modified
--------
2026-02-17
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from grdl.geolocation.base import Geolocation
from grdl.geolocation.elevation.constant import ConstantElevation
from grdl.image_processing.ortho.ortho import OutputGrid, Orthorectifier
from grdl.image_processing.ortho.ortho_pipeline import OrthoPipeline, OrthoResult


# ---------------------------------------------------------------------------
# Synthetic geolocation
# ---------------------------------------------------------------------------

class SyntheticAffineGeo(Geolocation):
    """Simple affine geolocation for tests."""

    def __init__(
        self,
        shape: Tuple[int, int] = (100, 200),
        origin_lat: float = -30.0,
        origin_lon: float = 115.0,
        pixel_size_lat: float = 0.01,
        pixel_size_lon: float = 0.005,
    ) -> None:
        super().__init__(shape, crs='WGS84')
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.pixel_size_lat = pixel_size_lat
        self.pixel_size_lon = pixel_size_lon

    def _image_to_latlon_array(
        self, rows, cols, height=0.0,
    ):
        lats = self.origin_lat - rows * self.pixel_size_lat
        lons = self.origin_lon + cols * self.pixel_size_lon
        heights = np.full_like(lats, height)
        return lats, lons, heights

    def _latlon_to_image_array(
        self, lats, lons, height=0.0,
    ):
        rows = (self.origin_lat - lats) / self.pixel_size_lat
        cols = (lons - self.origin_lon) / self.pixel_size_lon
        return rows, cols


class MockReader:
    """Minimal reader stub for tests."""

    def __init__(self, data, metadata=None):
        self._data = data
        self.metadata = metadata

    def get_shape(self):
        return self._data.shape

    def read_full(self, bands=None):
        if bands is not None and self._data.ndim == 3:
            return self._data[bands]
        return self._data

    def read_chip(self, row_start, row_end, col_start, col_end, bands=None):
        if self._data.ndim == 3:
            chip = self._data[:, row_start:row_end, col_start:col_end]
            if bands is not None:
                chip = chip[bands]
            return chip
        return self._data[row_start:row_end, col_start:col_end]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def geo():
    return SyntheticAffineGeo()


@pytest.fixture
def source_2d():
    row_idx, col_idx = np.meshgrid(
        np.arange(100), np.arange(200), indexing='ij',
    )
    return (row_idx * 1000 + col_idx).astype(np.float64)


@pytest.fixture
def source_3d():
    row_idx, col_idx = np.meshgrid(
        np.arange(100), np.arange(200), indexing='ij',
    )
    out = np.zeros((3, 100, 200), dtype=np.float64)
    for b in range(3):
        out[b] = row_idx * 1000 + col_idx + b * 100000
    return out


# ---------------------------------------------------------------------------
# Tests: OutputGrid.sub_grid
# ---------------------------------------------------------------------------

class TestSubGrid:

    def test_full_extent(self):
        """sub_grid covering entire grid returns equivalent grid."""
        grid = OutputGrid(-31.0, -30.0, 115.0, 116.0, 0.01, 0.01)
        sub = grid.sub_grid(0, 0, grid.rows, grid.cols)
        assert sub.rows == grid.rows
        assert sub.cols == grid.cols
        assert abs(sub.min_lat - grid.min_lat) < 1e-10
        assert abs(sub.max_lat - grid.max_lat) < 1e-10
        assert abs(sub.min_lon - grid.min_lon) < 1e-10
        assert abs(sub.max_lon - grid.max_lon) < 1e-10

    def test_upper_left_tile(self):
        """First tile has correct northern/western bounds."""
        grid = OutputGrid(-31.0, -30.0, 115.0, 116.0, 0.01, 0.01)
        sub = grid.sub_grid(0, 0, 50, 50)
        assert sub.rows == 50
        assert sub.cols == 50
        # Northern bound matches parent
        assert abs(sub.max_lat - grid.max_lat) < 1e-10
        # Western bound matches parent
        assert abs(sub.min_lon - grid.min_lon) < 1e-10

    def test_lower_right_tile(self):
        """Last tile has correct southern/eastern bounds."""
        grid = OutputGrid(-31.0, -30.0, 115.0, 116.0, 0.01, 0.01)
        sub = grid.sub_grid(50, 50, grid.rows, grid.cols)
        assert sub.rows == 50
        assert sub.cols == 50
        # Southern bound matches parent
        assert abs(sub.min_lat - grid.min_lat) < 1e-10
        # Eastern bound matches parent
        assert abs(sub.max_lon - grid.max_lon) < 1e-10

    def test_pixel_sizes_preserved(self):
        """Sub-grid pixel sizes match parent."""
        grid = OutputGrid(-31.0, -30.0, 115.0, 116.0, 0.01, 0.005)
        sub = grid.sub_grid(10, 20, 60, 80)
        assert sub.pixel_size_lat == grid.pixel_size_lat
        assert sub.pixel_size_lon == grid.pixel_size_lon

    def test_dimensions_match_tile(self):
        """Sub-grid rows/cols match tile extent."""
        grid = OutputGrid(-31.0, -30.0, 115.0, 116.0, 0.01, 0.01)
        sub = grid.sub_grid(10, 20, 40, 70)
        assert sub.rows == 30
        assert sub.cols == 50

    def test_negative_index_raises(self):
        grid = OutputGrid(-31.0, -30.0, 115.0, 116.0, 0.01, 0.01)
        with pytest.raises(ValueError, match="non-negative"):
            grid.sub_grid(-1, 0, 50, 50)

    def test_exceeding_bounds_raises(self):
        grid = OutputGrid(-31.0, -30.0, 115.0, 116.0, 0.01, 0.01)
        with pytest.raises(ValueError, match="exceed"):
            grid.sub_grid(0, 0, grid.rows + 1, grid.cols)

    def test_empty_region_raises(self):
        grid = OutputGrid(-31.0, -30.0, 115.0, 116.0, 0.01, 0.01)
        with pytest.raises(ValueError, match="Empty"):
            grid.sub_grid(50, 50, 50, 100)


# ---------------------------------------------------------------------------
# Tests: ROI pipeline
# ---------------------------------------------------------------------------

class TestPipelineROI:

    def test_roi_restricts_output_bounds(self, geo, source_2d):
        """ROI bounds should control the output grid."""
        result = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_resolution(0.01, 0.005)
            .with_roi(-30.5, -30.2, 115.2, 115.5)
            .with_interpolation('nearest')
            .run()
        )
        grid = result.output_grid
        assert abs(grid.min_lat - (-30.5)) < 1e-10
        assert abs(grid.max_lat - (-30.2)) < 1e-10
        assert abs(grid.min_lon - 115.2) < 1e-10
        assert abs(grid.max_lon - 115.5) < 1e-10

    def test_roi_smaller_than_footprint(self, geo, source_2d):
        """ROI grid should be smaller than full footprint grid."""
        result_full = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_resolution(0.01, 0.005)
            .with_interpolation('nearest')
            .run()
        )
        result_roi = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_resolution(0.01, 0.005)
            .with_roi(-30.5, -30.2, 115.2, 115.5)
            .with_interpolation('nearest')
            .run()
        )
        assert result_roi.output_grid.rows < result_full.output_grid.rows
        assert result_roi.output_grid.cols < result_full.output_grid.cols

    def test_roi_with_reader(self, geo, source_2d):
        """ROI should work with reader path."""
        reader = MockReader(source_2d)
        result = (
            OrthoPipeline()
            .with_reader(reader)
            .with_geolocation(geo)
            .with_resolution(0.01, 0.005)
            .with_roi(-30.5, -30.2, 115.2, 115.5)
            .with_interpolation('nearest')
            .run()
        )
        assert isinstance(result, OrthoResult)
        assert result.data.ndim == 2

    def test_roi_outside_footprint_all_nodata(self, geo, source_2d):
        """ROI outside source coverage → all nodata."""
        result = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_resolution(0.01, 0.005)
            .with_roi(10.0, 11.0, 50.0, 51.0)
            .with_interpolation('nearest')
            .with_nodata(-999.0)
            .run()
        )
        assert np.all(result.data == -999.0)

    def test_roi_with_elevation(self, geo, source_2d):
        """ROI + DEM should not error."""
        elev = ConstantElevation(height=100.0)
        result = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_elevation(elev)
            .with_resolution(0.01, 0.005)
            .with_roi(-30.5, -30.2, 115.2, 115.5)
            .with_interpolation('nearest')
            .run()
        )
        assert isinstance(result, OrthoResult)


# ---------------------------------------------------------------------------
# Tests: Tiled pipeline
# ---------------------------------------------------------------------------

class TestPipelineTiled:

    def test_tiled_matches_full(self, geo, source_2d):
        """Tiled result should match non-tiled result (nearest).

        Uses output resolution (0.009/0.004) different from source
        (0.01/0.005) so inverse mappings never land on exact
        half-integer pixel boundaries where nearest-neighbor rounding
        is ambiguous between the two arithmetic paths.
        """
        result_full = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_resolution(0.009, 0.004)
            .with_interpolation('nearest')
            .run()
        )
        result_tiled = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_resolution(0.009, 0.004)
            .with_interpolation('nearest')
            .with_tile_size(32)
            .run()
        )
        np.testing.assert_array_equal(result_full.data, result_tiled.data)

    def test_tiled_output_dimensions(self, geo, source_2d):
        """Tiled output shape should match full grid dimensions."""
        result = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_resolution(0.01, 0.005)
            .with_interpolation('nearest')
            .with_tile_size(32)
            .run()
        )
        grid = result.output_grid
        assert result.data.shape == (grid.rows, grid.cols)

    def test_tiled_with_reader(self, geo, source_2d):
        """Tiled path should work with reader."""
        reader = MockReader(source_2d)
        result = (
            OrthoPipeline()
            .with_reader(reader)
            .with_geolocation(geo)
            .with_resolution(0.01, 0.005)
            .with_interpolation('nearest')
            .with_tile_size(64)
            .run()
        )
        assert isinstance(result, OrthoResult)
        assert result.data.ndim == 2

    def test_tiled_larger_than_output(self, geo, source_2d):
        """Tile larger than output → one tile, matches full result."""
        result_full = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_resolution(0.01, 0.005)
            .with_interpolation('nearest')
            .run()
        )
        result_tiled = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_resolution(0.01, 0.005)
            .with_interpolation('nearest')
            .with_tile_size(10000)
            .run()
        )
        np.testing.assert_array_equal(result_full.data, result_tiled.data)

    def test_tiled_with_roi(self, geo, source_2d):
        """ROI + tiling should compose correctly."""
        result_roi = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_resolution(0.009, 0.004)
            .with_roi(-30.5, -30.2, 115.2, 115.5)
            .with_interpolation('nearest')
            .run()
        )
        result_roi_tiled = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_resolution(0.009, 0.004)
            .with_roi(-30.5, -30.2, 115.2, 115.5)
            .with_interpolation('nearest')
            .with_tile_size(16)
            .run()
        )
        np.testing.assert_array_equal(
            result_roi.data, result_roi_tiled.data,
        )

    def test_tiled_with_elevation(self, geo, source_2d):
        """Tiling + DEM should match non-tiled + DEM."""
        elev = ConstantElevation(height=0.0)
        result_full = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_elevation(elev)
            .with_resolution(0.009, 0.004)
            .with_interpolation('nearest')
            .run()
        )
        result_tiled = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_elevation(elev)
            .with_resolution(0.009, 0.004)
            .with_interpolation('nearest')
            .with_tile_size(32)
            .run()
        )
        np.testing.assert_array_equal(result_full.data, result_tiled.data)

    def test_tiled_multiband(self, geo, source_3d):
        """Tiling should work with multi-band source arrays."""
        result_full = (
            OrthoPipeline()
            .with_source_array(source_3d)
            .with_geolocation(geo)
            .with_resolution(0.009, 0.004)
            .with_interpolation('nearest')
            .run()
        )
        result_tiled = (
            OrthoPipeline()
            .with_source_array(source_3d)
            .with_geolocation(geo)
            .with_resolution(0.009, 0.004)
            .with_interpolation('nearest')
            .with_tile_size(32)
            .run()
        )
        assert result_tiled.data.ndim == 3
        assert result_tiled.data.shape[0] == 3
        np.testing.assert_array_equal(result_full.data, result_tiled.data)

    def test_tiled_nodata_fill(self, geo, source_2d):
        """Nodata value should be correctly applied in tiled path."""
        big_grid = OutputGrid(-32.0, -28.0, 113.0, 118.0, 0.01, 0.005)
        result = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_output_grid(big_grid)
            .with_interpolation('nearest')
            .with_nodata(-999.0)
            .with_tile_size(64)
            .run()
        )
        assert np.any(result.data == -999.0)
        assert np.any(result.data != -999.0)

    def test_builder_returns_self(self, geo):
        """with_roi and with_tile_size should return self."""
        p = OrthoPipeline()
        assert p.with_roi(0, 1, 0, 1) is p
        assert p.with_tile_size(256) is p


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
