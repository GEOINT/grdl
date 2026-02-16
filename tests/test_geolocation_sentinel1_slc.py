# -*- coding: utf-8 -*-
"""
Tests for Sentinel-1 SLC geolocation.

All tests use synthetic grid data — no real SAFE products required.

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
2026-02-16

Modified
--------
2026-02-16
"""

# Standard library
from unittest.mock import MagicMock

# Third-party
import numpy as np
import pytest

# GRDL
from grdl.IO.models.sentinel1_slc import (
    Sentinel1SLCMetadata,
    S1SLCGeoGridPoint,
)
from grdl.geolocation.sar.sentinel1_slc import Sentinel1SLCGeolocation


# ===================================================================
# Synthetic grid builder
# ===================================================================

# Grid dimensions: 5 lines × 8 pixels
_N_LINES = 5
_N_PIXELS = 8
_LINE_SPACING = 100.0
_PIXEL_SPACING = 200.0
_TOTAL_LINES = int((_N_LINES - 1) * _LINE_SPACING)
_TOTAL_PIXELS = int((_N_PIXELS - 1) * _PIXEL_SPACING)

# Geographic extent: lat 30.0–31.0, lon 50.0–52.0 (linear ramp)
_LAT_MIN, _LAT_MAX = 30.0, 31.0
_LON_MIN, _LON_MAX = 50.0, 52.0
_HEIGHT = 100.0


def _build_synthetic_grid():
    """Build a 5×8 grid with linear lat/lon ramps.

    lat increases with line (row), lon increases with pixel (col).
    This makes forward/inverse transforms analytically predictable.
    """
    grid = []
    for i in range(_N_LINES):
        line = i * _LINE_SPACING
        lat = _LAT_MIN + (_LAT_MAX - _LAT_MIN) * i / (_N_LINES - 1)
        for j in range(_N_PIXELS):
            pixel = j * _PIXEL_SPACING
            lon = _LON_MIN + (_LON_MAX - _LON_MIN) * j / (_N_PIXELS - 1)
            grid.append(S1SLCGeoGridPoint(
                line=line, pixel=pixel,
                latitude=lat, longitude=lon,
                height=_HEIGHT,
            ))
    return grid


def _build_synthetic_metadata():
    """Build Sentinel1SLCMetadata with the synthetic grid."""
    return Sentinel1SLCMetadata(
        format='Sentinel-1_IW_SLC',
        rows=_TOTAL_LINES,
        cols=_TOTAL_PIXELS,
        dtype='complex64',
        geolocation_grid=_build_synthetic_grid(),
    )


# ===================================================================
# Forward transform tests
# ===================================================================

class TestForwardTransform:
    """Test pixel → latlon transforms."""

    def test_grid_point_exact(self):
        """Forward transform at a grid point returns exact lat/lon."""
        meta = _build_synthetic_metadata()
        geo = Sentinel1SLCGeolocation(meta)

        # Grid point at (line=200, pixel=600) → i=2, j=3
        lat, lon, h = geo.image_to_latlon(200.0, 600.0)
        expected_lat = _LAT_MIN + (_LAT_MAX - _LAT_MIN) * 2 / 4
        expected_lon = _LON_MIN + (_LON_MAX - _LON_MIN) * 3 / 7
        assert lat == pytest.approx(expected_lat, abs=1e-6)
        assert lon == pytest.approx(expected_lon, abs=1e-6)
        assert h == pytest.approx(_HEIGHT, abs=1e-3)

    def test_midpoint_interpolation(self):
        """Forward transform at cell center interpolates correctly."""
        meta = _build_synthetic_metadata()
        geo = Sentinel1SLCGeolocation(meta)

        # Midpoint of first cell: (50, 100)
        lat, lon, h = geo.image_to_latlon(50.0, 100.0)
        # Linear ramp → midpoint should be average of corners
        expected_lat = _LAT_MIN + (_LAT_MAX - _LAT_MIN) * 0.5 / 4
        expected_lon = _LON_MIN + (_LON_MAX - _LON_MIN) * 0.5 / 7
        assert lat == pytest.approx(expected_lat, abs=1e-4)
        assert lon == pytest.approx(expected_lon, abs=1e-4)

    def test_array_input(self):
        """Forward transform works with array inputs."""
        meta = _build_synthetic_metadata()
        geo = Sentinel1SLCGeolocation(meta)

        rows = np.array([0.0, 200.0, 400.0])
        cols = np.array([0.0, 600.0, 1400.0])
        lats, lons, heights = geo.image_to_latlon(rows, cols)

        assert lats.shape == (3,)
        assert lons.shape == (3,)
        assert lats[0] == pytest.approx(_LAT_MIN, abs=1e-6)
        assert lons[0] == pytest.approx(_LON_MIN, abs=1e-6)

    def test_scalar_returns_scalars(self):
        """Scalar input returns scalar floats."""
        meta = _build_synthetic_metadata()
        geo = Sentinel1SLCGeolocation(meta)

        result = geo.image_to_latlon(0.0, 0.0)
        assert isinstance(result, tuple)
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)
        assert isinstance(result[2], float)

    def test_corners(self):
        """Four corners of the grid produce expected lat/lon."""
        meta = _build_synthetic_metadata()
        geo = Sentinel1SLCGeolocation(meta)

        # Top-left: (0, 0) → (LAT_MIN, LON_MIN)
        lat, lon, _ = geo.image_to_latlon(0.0, 0.0)
        assert lat == pytest.approx(_LAT_MIN, abs=1e-6)
        assert lon == pytest.approx(_LON_MIN, abs=1e-6)

        # Bottom-right: (400, 1400) → (LAT_MAX, LON_MAX)
        lat, lon, _ = geo.image_to_latlon(400.0, 1400.0)
        assert lat == pytest.approx(_LAT_MAX, abs=1e-6)
        assert lon == pytest.approx(_LON_MAX, abs=1e-6)


# ===================================================================
# Inverse transform tests
# ===================================================================

class TestInverseTransform:
    """Test latlon → pixel transforms."""

    def test_grid_point_inverse(self):
        """Inverse at a known grid point returns exact pixel coords."""
        meta = _build_synthetic_metadata()
        geo = Sentinel1SLCGeolocation(meta)

        expected_lat = _LAT_MIN + (_LAT_MAX - _LAT_MIN) * 2 / 4
        expected_lon = _LON_MIN + (_LON_MAX - _LON_MIN) * 3 / 7
        row, col = geo.latlon_to_image(expected_lat, expected_lon)
        assert row == pytest.approx(200.0, abs=1.0)
        assert col == pytest.approx(600.0, abs=1.0)

    def test_round_trip(self):
        """Forward then inverse returns original pixel coordinates."""
        meta = _build_synthetic_metadata()
        geo = Sentinel1SLCGeolocation(meta)

        # Pick an interior point
        orig_row, orig_col = 150.0, 500.0
        lat, lon, _ = geo.image_to_latlon(orig_row, orig_col)
        row, col = geo.latlon_to_image(lat, lon)
        assert row == pytest.approx(orig_row, abs=2.0)
        assert col == pytest.approx(orig_col, abs=2.0)

    def test_inverse_array(self):
        """Inverse transform works with array inputs."""
        meta = _build_synthetic_metadata()
        geo = Sentinel1SLCGeolocation(meta)

        lats = np.array([30.5, 30.75])
        lons = np.array([51.0, 51.5])
        rows, cols = geo.latlon_to_image(lats, lons)
        assert rows.shape == (2,)
        assert cols.shape == (2,)

    def test_outside_convex_hull_returns_nan(self):
        """Points outside the grid area return NaN."""
        meta = _build_synthetic_metadata()
        geo = Sentinel1SLCGeolocation(meta)

        # Far outside the grid
        row, col = geo.latlon_to_image(10.0, 10.0)
        assert np.isnan(row)
        assert np.isnan(col)


# ===================================================================
# Factory method tests
# ===================================================================

class TestFromReader:
    """Test the from_reader factory method."""

    def test_from_reader(self):
        """from_reader constructs a working geolocation."""
        meta = _build_synthetic_metadata()

        # Mock a Sentinel1SLCReader
        mock_reader = MagicMock()
        mock_reader.metadata = meta

        geo = Sentinel1SLCGeolocation.from_reader(mock_reader)
        assert geo.shape == (_TOTAL_LINES, _TOTAL_PIXELS)

        lat, lon, _ = geo.image_to_latlon(0.0, 0.0)
        assert lat == pytest.approx(_LAT_MIN, abs=1e-6)


# ===================================================================
# Error handling tests
# ===================================================================

class TestErrors:
    """Test error conditions."""

    def test_empty_grid_raises(self):
        """ValueError for empty geolocation grid."""
        meta = Sentinel1SLCMetadata(
            format='Sentinel-1_IW_SLC',
            rows=100, cols=100, dtype='complex64',
            geolocation_grid=[],
        )
        with pytest.raises(ValueError, match="at least 4 points"):
            Sentinel1SLCGeolocation(meta)

    def test_none_grid_raises(self):
        """ValueError for None geolocation grid."""
        meta = Sentinel1SLCMetadata(
            format='Sentinel-1_IW_SLC',
            rows=100, cols=100, dtype='complex64',
            geolocation_grid=None,
        )
        with pytest.raises(ValueError, match="at least 4 points"):
            Sentinel1SLCGeolocation(meta)

    def test_too_few_points_raises(self):
        """ValueError for grid with fewer than 4 points."""
        grid = [
            S1SLCGeoGridPoint(line=0, pixel=0, latitude=30, longitude=50,
                              height=0),
            S1SLCGeoGridPoint(line=0, pixel=100, latitude=30, longitude=51,
                              height=0),
        ]
        meta = Sentinel1SLCMetadata(
            format='Sentinel-1_IW_SLC',
            rows=100, cols=100, dtype='complex64',
            geolocation_grid=grid,
        )
        with pytest.raises(ValueError, match="at least 4 points"):
            Sentinel1SLCGeolocation(meta)


# ===================================================================
# get_footprint / get_bounds tests
# ===================================================================

class TestFootprint:
    """Test inherited footprint methods."""

    def test_get_footprint(self):
        """get_footprint returns a valid polygon."""
        meta = _build_synthetic_metadata()
        geo = Sentinel1SLCGeolocation(meta)

        fp = geo.get_footprint()
        assert fp['type'] == 'Polygon'
        assert fp['bounds'] is not None
        min_lon, min_lat, max_lon, max_lat = fp['bounds']
        assert min_lat >= _LAT_MIN - 0.1
        assert max_lat <= _LAT_MAX + 0.1

    def test_get_bounds(self):
        """get_bounds returns a bounding box tuple."""
        meta = _build_synthetic_metadata()
        geo = Sentinel1SLCGeolocation(meta)

        bounds = geo.get_bounds()
        assert len(bounds) == 4
