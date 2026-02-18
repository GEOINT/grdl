# -*- coding: utf-8 -*-
"""
Ortho Pipeline Tests - Tests for OrthoPipeline and OrthoResult.

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
from typing import Dict, List, Optional, Tuple, Union

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
    """Simple affine geolocation for pipeline tests."""

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
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lats = self.origin_lat - rows * self.pixel_size_lat
        lons = self.origin_lon + cols * self.pixel_size_lon
        heights = np.full_like(lats, height)
        return lats, lons, heights

    def _latlon_to_image_array(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rows = (self.origin_lat - lats) / self.pixel_size_lat
        cols = (lons - self.origin_lon) / self.pixel_size_lon
        return rows, cols


# ---------------------------------------------------------------------------
# Mock ImageReader (no filesystem dependency)
# ---------------------------------------------------------------------------

class MockReader:
    """Minimal reader stub for pipeline tests.

    Mimics ImageReader's public interface without requiring a real file.
    """

    def __init__(
        self,
        data: np.ndarray,
        metadata: object = None,
    ) -> None:
        self._data = data
        self.metadata = metadata

    def get_shape(self) -> Tuple[int, ...]:
        return self._data.shape

    def read_full(
        self, bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        if bands is not None and self._data.ndim == 3:
            return self._data[bands]
        return self._data

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
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
    """100x200 float image encoding position."""
    row_idx, col_idx = np.meshgrid(
        np.arange(100), np.arange(200), indexing='ij',
    )
    return (row_idx * 1000 + col_idx).astype(np.float64)


@pytest.fixture
def source_3d():
    """3-band 100x200 image."""
    row_idx, col_idx = np.meshgrid(
        np.arange(100), np.arange(200), indexing='ij',
    )
    out = np.zeros((3, 100, 200), dtype=np.float64)
    for b in range(3):
        out[b] = row_idx * 1000 + col_idx + b * 100000
    return out


# ---------------------------------------------------------------------------
# Tests: OrthoPipeline — validation
# ---------------------------------------------------------------------------

class TestPipelineValidation:

    def test_missing_geolocation_raises(self, source_2d):
        """Pipeline should fail without geolocation."""
        pipeline = OrthoPipeline().with_source_array(source_2d)
        with pytest.raises(ValueError, match="Geolocation"):
            pipeline.run()

    def test_missing_reader_and_source_raises(self, geo):
        """Pipeline should fail without reader or source array."""
        pipeline = OrthoPipeline().with_geolocation(geo)
        with pytest.raises(ValueError, match="reader"):
            pipeline.run()

    def test_missing_resolution_without_reader_raises(self, geo, source_2d):
        """Auto-resolution needs a reader with metadata."""
        pipeline = (
            OrthoPipeline()
            .with_geolocation(geo)
            .with_source_array(source_2d)
        )
        with pytest.raises(ValueError, match="resolution"):
            pipeline.run()


# ---------------------------------------------------------------------------
# Tests: OrthoPipeline — source array path
# ---------------------------------------------------------------------------

class TestPipelineSourceArray:

    def test_explicit_resolution_2d(self, geo, source_2d):
        """Pipeline with source array and explicit resolution."""
        result = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_resolution(0.01, 0.005)
            .with_interpolation('nearest')
            .run()
        )
        assert isinstance(result, OrthoResult)
        assert result.data.ndim == 2
        assert result.data.shape[0] > 0
        assert result.data.shape[1] > 0

    def test_explicit_resolution_3d(self, geo, source_3d):
        """Pipeline with multi-band source array."""
        result = (
            OrthoPipeline()
            .with_source_array(source_3d)
            .with_geolocation(geo)
            .with_resolution(0.01, 0.005)
            .with_interpolation('nearest')
            .run()
        )
        assert result.data.ndim == 3
        assert result.data.shape[0] == 3

    def test_nodata_value(self, geo, source_2d):
        """Pipeline should honor nodata fill value."""
        # Make grid extend beyond source coverage
        big_grid = OutputGrid(-32.0, -28.0, 113.0, 118.0, 0.01, 0.005)
        result = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_output_grid(big_grid)
            .with_interpolation('nearest')
            .with_nodata(-999.0)
            .run()
        )
        assert np.any(result.data == -999.0)
        assert np.any(result.data != -999.0)

    def test_explicit_output_grid(self, geo, source_2d):
        """Pipeline with explicit OutputGrid (skips resolution)."""
        grid = OutputGrid.from_geolocation(geo, 0.01, 0.005)
        result = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_output_grid(grid)
            .with_interpolation('nearest')
            .run()
        )
        assert result.output_grid is grid
        assert result.data.shape == (grid.rows, grid.cols)


# ---------------------------------------------------------------------------
# Tests: OrthoPipeline — reader path
# ---------------------------------------------------------------------------

class TestPipelineReader:

    def test_reader_basic(self, geo, source_2d):
        """Pipeline with mock reader and explicit resolution."""
        reader = MockReader(source_2d)
        result = (
            OrthoPipeline()
            .with_reader(reader)
            .with_geolocation(geo)
            .with_resolution(0.01, 0.005)
            .with_interpolation('nearest')
            .run()
        )
        assert isinstance(result, OrthoResult)
        assert result.data.ndim == 2

    def test_reader_multiband(self, geo, source_3d):
        """Pipeline with multi-band reader."""
        reader = MockReader(source_3d)
        result = (
            OrthoPipeline()
            .with_reader(reader)
            .with_geolocation(geo)
            .with_resolution(0.01, 0.005)
            .with_interpolation('nearest')
            .run()
        )
        assert result.data.ndim == 3

    def test_reader_band_selection(self, geo, source_3d):
        """Pipeline with band selection via reader."""
        reader = MockReader(source_3d)
        result = (
            OrthoPipeline()
            .with_reader(reader)
            .with_geolocation(geo)
            .with_resolution(0.01, 0.005)
            .with_interpolation('nearest')
            .with_bands([0])
            .run()
        )
        # Single band selected
        assert result.data.ndim == 2 or result.data.shape[0] == 1


# ---------------------------------------------------------------------------
# Tests: OrthoPipeline — DEM elevation
# ---------------------------------------------------------------------------

class TestPipelineElevation:

    def test_constant_elevation_accepted(self, geo, source_2d):
        """Pipeline should accept ConstantElevation without error."""
        elev = ConstantElevation(height=500.0)
        result = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_elevation(elev)
            .with_resolution(0.01, 0.005)
            .with_interpolation('nearest')
            .run()
        )
        assert isinstance(result, OrthoResult)
        assert result.data.shape[0] > 0

    def test_elevation_vs_no_elevation(self, geo, source_2d):
        """With flat affine geo, constant elevation should not change result."""
        # For a simple affine geolocation that ignores height, DEM height
        # doesn't change the mapping. This test verifies the DEM plumbing
        # executes without error and doesn't corrupt the result.
        result_no_dem = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_resolution(0.01, 0.005)
            .with_interpolation('nearest')
            .run()
        )
        elev = ConstantElevation(height=0.0)
        result_dem = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_elevation(elev)
            .with_resolution(0.01, 0.005)
            .with_interpolation('nearest')
            .run()
        )
        np.testing.assert_array_equal(result_no_dem.data, result_dem.data)


# ---------------------------------------------------------------------------
# Tests: OrthoResult
# ---------------------------------------------------------------------------

class TestOrthoResult:

    def test_shape_property(self, geo, source_2d):
        """OrthoResult.shape should match data.shape."""
        result = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_resolution(0.01, 0.005)
            .with_interpolation('nearest')
            .run()
        )
        assert result.shape == result.data.shape

    def test_geolocation_metadata_keys(self, geo, source_2d):
        """OrthoResult should have standard geolocation metadata."""
        result = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_resolution(0.01, 0.005)
            .with_interpolation('nearest')
            .run()
        )
        meta = result.geolocation_metadata
        assert 'crs' in meta
        assert 'bounds' in meta
        assert 'transform' in meta
        assert 'rows' in meta
        assert 'cols' in meta
        assert meta['crs'] == 'WGS84'

    def test_orthorectifier_cached(self, geo, source_2d):
        """OrthoResult should hold the configured Orthorectifier."""
        result = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_resolution(0.01, 0.005)
            .with_interpolation('nearest')
            .run()
        )
        assert isinstance(result.orthorectifier, Orthorectifier)
        # Mapping should be computed
        assert result.orthorectifier._source_rows is not None


# ---------------------------------------------------------------------------
# Tests: Builder chaining
# ---------------------------------------------------------------------------

class TestBuilderChaining:

    def test_all_builder_methods_return_self(self, geo, source_2d):
        """Every with_*() method should return the pipeline for chaining."""
        p = OrthoPipeline()
        assert p.with_geolocation(geo) is p
        assert p.with_source_array(source_2d) is p
        assert p.with_resolution(0.01, 0.005) is p
        assert p.with_interpolation('nearest') is p
        assert p.with_nodata(-1.0) is p
        assert p.with_margin(0.01) is p
        assert p.with_scale_factor(2.0) is p

        reader = MockReader(source_2d)
        assert p.with_reader(reader) is p

        elev = ConstantElevation(0.0)
        assert p.with_elevation(elev) is p

        grid = OutputGrid.from_geolocation(geo, 0.01, 0.005)
        assert p.with_output_grid(grid) is p

    def test_margin_expands_grid(self, geo, source_2d):
        """Margin should produce a larger output grid."""
        result_no_margin = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_resolution(0.01, 0.005)
            .with_interpolation('nearest')
            .run()
        )
        result_margin = (
            OrthoPipeline()
            .with_source_array(source_2d)
            .with_geolocation(geo)
            .with_resolution(0.01, 0.005)
            .with_interpolation('nearest')
            .with_margin(0.1)
            .run()
        )
        assert result_margin.output_grid.rows > result_no_margin.output_grid.rows
        assert result_margin.output_grid.cols > result_no_margin.output_grid.cols


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
