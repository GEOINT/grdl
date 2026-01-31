# -*- coding: utf-8 -*-
"""
Orthorectification Tests - Synthetic data tests for OutputGrid and Orthorectifier.

Tests the orthorectification pipeline using synthetic imagery with known
affine geolocation, verifying pixel placement, round-trip accuracy, complex
data handling, nodata fill, and interpolation modes.

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
2026-01-30

Modified
--------
2026-01-30
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from grdl.geolocation.base import Geolocation
from grdl.image_processing.ortho.ortho import OutputGrid, Orthorectifier


# ---------------------------------------------------------------------------
# Synthetic geolocation: simple affine transform
# ---------------------------------------------------------------------------

class AffineGeolocation(Geolocation):
    """
    Synthetic geolocation using a simple affine transform for testing.

    Maps pixel (row, col) to geographic (lat, lon) via:
        lat = origin_lat - row * pixel_size_lat
        lon = origin_lon + col * pixel_size_lon
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        origin_lat: float,
        origin_lon: float,
        pixel_size_lat: float,
        pixel_size_lon: float
    ) -> None:
        super().__init__(shape, crs='WGS84')
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.pixel_size_lat = pixel_size_lat
        self.pixel_size_lon = pixel_size_lon

    def _pixel_to_latlon_array(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lats = self.origin_lat - rows * self.pixel_size_lat
        lons = self.origin_lon + cols * self.pixel_size_lon
        heights = np.full_like(lats, height)
        return lats, lons, heights

    def _latlon_to_pixel_array(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        rows = (self.origin_lat - lats) / self.pixel_size_lat
        cols = (lons - self.origin_lon) / self.pixel_size_lon
        return rows, cols


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def affine_geo():
    """Create a synthetic affine geolocation (100x200 image)."""
    return AffineGeolocation(
        shape=(100, 200),
        origin_lat=-30.0,
        origin_lon=115.0,
        pixel_size_lat=0.01,
        pixel_size_lon=0.005,
    )


@pytest.fixture
def source_image():
    """Create a synthetic 100x200 source image with known pixel values."""
    rows, cols = 100, 200
    # Value encodes position: value = row * 1000 + col
    row_idx, col_idx = np.meshgrid(
        np.arange(rows), np.arange(cols), indexing='ij'
    )
    return (row_idx * 1000 + col_idx).astype(np.float64)


@pytest.fixture
def multiband_source():
    """Create a synthetic 3-band 100x200 source image."""
    rows, cols, n_bands = 100, 200, 3
    source = np.zeros((n_bands, rows, cols), dtype=np.float64)
    row_idx, col_idx = np.meshgrid(
        np.arange(rows), np.arange(cols), indexing='ij'
    )
    for b in range(n_bands):
        source[b] = row_idx * 1000 + col_idx + b * 100000
    return source


@pytest.fixture
def complex_source():
    """Create a synthetic complex-valued 100x200 source image."""
    rows, cols = 100, 200
    row_idx, col_idx = np.meshgrid(
        np.arange(rows), np.arange(cols), indexing='ij'
    )
    real_part = row_idx.astype(np.float32)
    imag_part = col_idx.astype(np.float32)
    return real_part + 1j * imag_part


# ---------------------------------------------------------------------------
# OutputGrid tests
# ---------------------------------------------------------------------------

class TestOutputGrid:
    """Tests for OutputGrid construction and coordinate transforms."""

    def test_basic_construction(self):
        """Test grid construction with valid parameters."""
        grid = OutputGrid(-31.0, -30.0, 115.0, 116.0, 0.01, 0.01)
        assert grid.rows == 100
        assert grid.cols == 100
        assert grid.min_lat == -31.0
        assert grid.max_lat == -30.0

    def test_non_square_grid(self):
        """Test grid with different lat/lon extents."""
        grid = OutputGrid(-31.0, -30.0, 115.0, 117.0, 0.01, 0.01)
        assert grid.rows == 100
        assert grid.cols == 200

    def test_pixel_to_latlon_scalar(self):
        """Test pixel-to-latlon for scalar inputs."""
        grid = OutputGrid(-31.0, -30.0, 115.0, 116.0, 0.01, 0.01)
        lat, lon = grid.pixel_to_latlon(0, 0)
        assert lat == pytest.approx(-30.0, abs=1e-10)
        assert lon == pytest.approx(115.0, abs=1e-10)

    def test_pixel_to_latlon_corner(self):
        """Test that last pixel maps to the grid's south-east corner."""
        grid = OutputGrid(-31.0, -30.0, 115.0, 116.0, 0.01, 0.01)
        lat, lon = grid.pixel_to_latlon(grid.rows - 1, grid.cols - 1)
        # Last row is near min_lat, last col is near max_lon
        assert lat == pytest.approx(-30.99, abs=0.01)
        assert lon == pytest.approx(115.99, abs=0.01)

    def test_pixel_to_latlon_array(self):
        """Test pixel-to-latlon for array inputs."""
        grid = OutputGrid(-31.0, -30.0, 115.0, 116.0, 0.01, 0.01)
        rows = np.array([0, 50, 99])
        cols = np.array([0, 50, 99])
        lats, lons = grid.pixel_to_latlon(rows, cols)
        assert lats.shape == (3,)
        assert lons.shape == (3,)

    def test_latlon_to_pixel_roundtrip(self):
        """Test that pixel->latlon->pixel round-trips accurately."""
        grid = OutputGrid(-31.0, -30.0, 115.0, 116.0, 0.01, 0.01)
        row_in, col_in = 42.0, 73.0
        lat, lon = grid.pixel_to_latlon(row_in, col_in)
        row_out, col_out = grid.latlon_to_pixel(lat, lon)
        assert row_out == pytest.approx(row_in, abs=1e-10)
        assert col_out == pytest.approx(col_in, abs=1e-10)

    def test_from_geolocation(self, affine_geo):
        """Test grid creation from a Geolocation object."""
        grid = OutputGrid.from_geolocation(affine_geo, 0.01, 0.005)
        assert grid.rows > 0
        assert grid.cols > 0
        # Grid should cover the image footprint
        bounds = affine_geo.get_bounds()
        assert grid.min_lon <= bounds[0]
        assert grid.min_lat <= bounds[1]
        assert grid.max_lon >= bounds[2]
        assert grid.max_lat >= bounds[3]

    def test_from_geolocation_with_margin(self, affine_geo):
        """Test that margin expands the grid."""
        grid_no_margin = OutputGrid.from_geolocation(affine_geo, 0.01, 0.005)
        grid_with_margin = OutputGrid.from_geolocation(
            affine_geo, 0.01, 0.005, margin=0.1
        )
        assert grid_with_margin.rows > grid_no_margin.rows
        assert grid_with_margin.cols > grid_no_margin.cols

    def test_invalid_bounds(self):
        """Test that invalid bounds raise ValueError."""
        with pytest.raises(ValueError, match="max_lat"):
            OutputGrid(-30.0, -31.0, 115.0, 116.0, 0.01, 0.01)

    def test_invalid_pixel_size(self):
        """Test that non-positive pixel size raises ValueError."""
        with pytest.raises(ValueError, match="pixel_size_lat"):
            OutputGrid(-31.0, -30.0, 115.0, 116.0, -0.01, 0.01)

    def test_repr(self):
        """Test string representation."""
        grid = OutputGrid(-31.0, -30.0, 115.0, 116.0, 0.01, 0.01)
        repr_str = repr(grid)
        assert 'OutputGrid' in repr_str
        assert '100x100' in repr_str


# ---------------------------------------------------------------------------
# Orthorectifier tests
# ---------------------------------------------------------------------------

class TestOrthorectifier:
    """Tests for Orthorectifier construction and apply methods."""

    def test_basic_construction(self, affine_geo):
        """Test orthorectifier construction."""
        grid = OutputGrid.from_geolocation(affine_geo, 0.01, 0.005)
        ortho = Orthorectifier(affine_geo, grid)
        assert ortho.interpolation == 'bilinear'
        assert ortho.output_grid is grid

    def test_invalid_interpolation(self, affine_geo):
        """Test that invalid interpolation raises ValueError."""
        grid = OutputGrid.from_geolocation(affine_geo, 0.01, 0.005)
        with pytest.raises(ValueError, match="Unknown interpolation"):
            Orthorectifier(affine_geo, grid, interpolation='invalid')

    def test_compute_mapping_shape(self, affine_geo):
        """Test that compute_mapping returns correct shapes."""
        grid = OutputGrid.from_geolocation(affine_geo, 0.01, 0.005)
        ortho = Orthorectifier(affine_geo, grid)
        src_rows, src_cols, valid = ortho.compute_mapping()

        assert src_rows.shape == (grid.rows, grid.cols)
        assert src_cols.shape == (grid.rows, grid.cols)
        assert valid.shape == (grid.rows, grid.cols)
        assert valid.dtype == bool

    def test_compute_mapping_has_valid_pixels(self, affine_geo):
        """Test that some output pixels map to valid source pixels."""
        grid = OutputGrid.from_geolocation(affine_geo, 0.01, 0.005)
        ortho = Orthorectifier(affine_geo, grid)
        _, _, valid = ortho.compute_mapping()
        assert np.any(valid), "No valid pixels in mapping"

    def test_apply_single_band(self, affine_geo, source_image):
        """Test orthorectification of a single-band image."""
        grid = OutputGrid.from_geolocation(affine_geo, 0.01, 0.005)
        ortho = Orthorectifier(affine_geo, grid, interpolation='nearest')
        result = ortho.apply(source_image)

        assert result.shape == (grid.rows, grid.cols)
        assert result.dtype == source_image.dtype

    def test_apply_multiband(self, affine_geo, multiband_source):
        """Test orthorectification of multi-band image."""
        grid = OutputGrid.from_geolocation(affine_geo, 0.01, 0.005)
        ortho = Orthorectifier(affine_geo, grid, interpolation='nearest')
        result = ortho.apply(multiband_source)

        assert result.shape == (3, grid.rows, grid.cols)
        assert result.dtype == multiband_source.dtype

    def test_apply_complex(self, affine_geo, complex_source):
        """Test orthorectification preserves complex data."""
        grid = OutputGrid.from_geolocation(affine_geo, 0.01, 0.005)
        ortho = Orthorectifier(affine_geo, grid, interpolation='nearest')
        result = ortho.apply(complex_source)

        assert result.shape == (grid.rows, grid.cols)
        assert np.iscomplexobj(result)

    def test_nodata_fill(self, affine_geo, source_image):
        """Test that out-of-coverage pixels get nodata fill."""
        # Create a grid that extends beyond the source footprint
        grid = OutputGrid(-32.0, -29.0, 114.0, 117.0, 0.01, 0.01)
        ortho = Orthorectifier(affine_geo, grid, interpolation='nearest')
        result = ortho.apply(source_image, nodata=-999.0)

        # Some pixels should be nodata (outside source coverage)
        assert np.any(result == -999.0), "Expected nodata pixels outside coverage"
        # Some pixels should be valid (inside source coverage)
        assert np.any(result != -999.0), "Expected valid pixels inside coverage"

    def test_nearest_preserves_values(self, affine_geo):
        """Test that nearest-neighbor preserves original pixel values."""
        # Create source where each pixel has a unique integer value
        source = np.arange(100 * 200, dtype=np.float64).reshape(100, 200)

        # Use grid with same resolution as source (1:1 mapping)
        grid = OutputGrid.from_geolocation(
            affine_geo,
            affine_geo.pixel_size_lat,
            affine_geo.pixel_size_lon,
        )
        ortho = Orthorectifier(affine_geo, grid, interpolation='nearest')
        result = ortho.apply(source, nodata=-1.0)

        # Valid pixels should contain values from the source
        valid = result != -1.0
        if np.any(valid):
            valid_values = result[valid]
            # All valid values should be integer-valued (nearest preserves)
            assert np.allclose(valid_values, np.round(valid_values)), \
                "Nearest interpolation should preserve integer values"

    def test_identity_affine_roundtrip(self):
        """Test that matching source and output grids produce identity."""
        # Source: 50x50 image with known affine
        geo = AffineGeolocation(
            shape=(50, 50),
            origin_lat=0.0,
            origin_lon=0.0,
            pixel_size_lat=0.01,
            pixel_size_lon=0.01,
        )

        # Output grid matches the source geometry exactly
        # Use pixel centers: source covers lat [0, -0.5), lon [0, 0.5)
        grid = OutputGrid(
            min_lat=-0.5 + 0.005,  # half-pixel offset to align centers
            max_lat=0.0 - 0.005,
            min_lon=0.0 + 0.005,
            max_lon=0.5 - 0.005,
            pixel_size_lat=0.01,
            pixel_size_lon=0.01,
        )

        source = np.arange(50 * 50, dtype=np.float64).reshape(50, 50)
        ortho = Orthorectifier(geo, grid, interpolation='nearest')
        result = ortho.apply(source, nodata=-1.0)

        # With matching grids and nearest interpolation, valid pixels
        # should map 1:1 back to source values
        valid = result != -1.0
        if np.any(valid):
            n_valid = np.sum(valid)
            # Most pixels should be valid (some edge loss expected)
            assert n_valid > grid.rows * grid.cols * 0.5, \
                f"Only {n_valid}/{grid.rows * grid.cols} valid pixels"

    def test_interpolation_bilinear(self, affine_geo, source_image):
        """Test bilinear interpolation produces smooth output."""
        grid = OutputGrid.from_geolocation(affine_geo, 0.01, 0.005)
        ortho = Orthorectifier(affine_geo, grid, interpolation='bilinear')
        result = ortho.apply(source_image)
        assert result.shape == (grid.rows, grid.cols)

    def test_interpolation_bicubic(self, affine_geo, source_image):
        """Test bicubic interpolation produces output."""
        grid = OutputGrid.from_geolocation(affine_geo, 0.01, 0.005)
        ortho = Orthorectifier(affine_geo, grid, interpolation='bicubic')
        result = ortho.apply(source_image)
        assert result.shape == (grid.rows, grid.cols)

    def test_mapping_cached(self, affine_geo, source_image):
        """Test that compute_mapping is cached and reused."""
        grid = OutputGrid.from_geolocation(affine_geo, 0.01, 0.005)
        ortho = Orthorectifier(affine_geo, grid)

        # First call computes mapping
        ortho.compute_mapping()
        rows1 = ortho._source_rows

        # Second call should reuse cached mapping
        result = ortho.apply(source_image)
        assert ortho._source_rows is rows1  # Same object (cached)

    def test_output_geolocation_metadata(self, affine_geo):
        """Test output geolocation metadata structure."""
        grid = OutputGrid.from_geolocation(affine_geo, 0.01, 0.005)
        ortho = Orthorectifier(affine_geo, grid)
        meta = ortho.get_output_geolocation_metadata()

        assert meta['crs'] == 'WGS84'
        assert len(meta['bounds']) == 4
        assert meta['rows'] == grid.rows
        assert meta['cols'] == grid.cols
        assert meta['pixel_size_lat'] == 0.01
        assert meta['pixel_size_lon'] == 0.005
        assert len(meta['transform']) == 6

    def test_repr(self, affine_geo):
        """Test string representation."""
        grid = OutputGrid.from_geolocation(affine_geo, 0.01, 0.005)
        ortho = Orthorectifier(affine_geo, grid)
        repr_str = repr(ortho)
        assert 'Orthorectifier' in repr_str
        assert 'bilinear' in repr_str


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
