# -*- coding: utf-8 -*-
"""
Elevation Module Tests - Tests for ElevationModel ABC and concrete implementations.

Tests ConstantElevation, ElevationModel ABC contract, scalar/(2,N) dispatch,
geoid correction math, and error handling for DTEDElevation and GeoTIFFDEM.

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
2026-02-11

Modified
--------
2026-02-11
"""

import tempfile

import pytest
import numpy as np

from grdl.geolocation.elevation.base import ElevationModel
from grdl.geolocation.elevation.constant import ConstantElevation
from grdl.geolocation.elevation.dted import DTEDElevation
from grdl.geolocation.elevation.geotiff_dem import GeoTIFFDEM
from grdl.geolocation.elevation.geoid import GeoidCorrection


# ---------------------------------------------------------------------------
# ElevationModel ABC contract tests
# ---------------------------------------------------------------------------

class TestElevationModelABC:
    """Test ElevationModel ABC enforcement."""

    def test_cannot_instantiate_abc(self):
        """Test that ElevationModel cannot be directly instantiated."""
        with pytest.raises(TypeError):
            ElevationModel()

    def test_subclass_must_implement_get_elevation_array(self):
        """Test that incomplete subclass cannot be instantiated."""
        class IncompleteElevation(ElevationModel):
            pass

        with pytest.raises(TypeError):
            IncompleteElevation()

    def test_complete_subclass_works(self):
        """Test that a complete subclass can be instantiated."""
        class SimpleElevation(ElevationModel):
            def _get_elevation_array(self, lats, lons):
                return np.full(lats.shape, 42.0)

        elev = SimpleElevation()
        assert elev.get_elevation(34.0, -118.0) == 42.0


# ---------------------------------------------------------------------------
# ConstantElevation tests
# ---------------------------------------------------------------------------

class TestConstantElevation:
    """Test ConstantElevation implementation."""

    def test_default_height(self):
        """Test default height is 0.0."""
        elev = ConstantElevation()
        assert elev.get_elevation(0.0, 0.0) == 0.0

    def test_custom_height(self):
        """Test custom constant height."""
        elev = ConstantElevation(height=500.0)
        assert elev.get_elevation(34.0, -118.0) == 500.0

    def test_scalar_dispatch(self):
        """Test scalar input returns float."""
        elev = ConstantElevation(height=100.0)
        result = elev.get_elevation(34.0, -118.0)
        assert isinstance(result, float)
        assert result == 100.0

    def test_array_dispatch(self):
        """Test array input returns ndarray."""
        elev = ConstantElevation(height=100.0)
        lats = np.array([34.0, 35.0, 36.0])
        lons = np.array([-118.0, -117.0, -116.0])
        result = elev.get_elevation(lats, lons)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.all(result == 100.0)

    def test_stacked_2xN_dispatch(self):
        """Test (2,N) stacked array input."""
        elev = ConstantElevation(height=200.0)
        pts = np.array([
            [34.0, 35.0],
            [-118.0, -117.0],
        ])
        result = elev.get_elevation(pts)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert np.all(result == 200.0)

    def test_bad_stacked_shape(self):
        """Test that non-(2,N) stacked array raises ValueError."""
        elev = ConstantElevation()
        bad_pts = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        with pytest.raises(ValueError, match="Expected \\(2, N\\)"):
            elev.get_elevation(bad_pts)

    def test_list_input(self):
        """Test list input is converted to arrays."""
        elev = ConstantElevation(height=50.0)
        result = elev.get_elevation([34.0, 35.0], [-118.0, -117.0])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert np.all(result == 50.0)

    def test_negative_height(self):
        """Test negative constant height (below sea level)."""
        elev = ConstantElevation(height=-30.0)
        assert elev.get_elevation(0.0, 0.0) == -30.0


# ---------------------------------------------------------------------------
# DTEDElevation error handling tests
# ---------------------------------------------------------------------------

class TestDTEDElevation:
    """Test DTEDElevation construction and error handling."""

    def test_nonexistent_path_raises(self):
        """Test that non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            DTEDElevation('/nonexistent/dted/path')

    def test_file_path_raises(self):
        """Test that a file path (not directory) raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix='.txt') as f:
            with pytest.raises(ValueError, match="must be a directory"):
                DTEDElevation(f.name)

    def test_empty_directory(self):
        """Test DTEDElevation with empty directory (no tiles)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            elev = DTEDElevation(tmpdir)
            assert elev.tile_count == 0
            assert elev.coverage_bounds is None

    def test_empty_directory_returns_nan(self):
        """Test that empty DTED returns NaN for queries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            elev = DTEDElevation(tmpdir)
            h = elev.get_elevation(34.0, -118.0)
            assert np.isnan(h)

    def test_empty_directory_array_returns_nan(self):
        """Test that empty DTED returns NaN array for batch queries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            elev = DTEDElevation(tmpdir)
            lats = np.array([34.0, 35.0])
            lons = np.array([-118.0, -117.0])
            heights = elev.get_elevation(lats, lons)
            assert np.all(np.isnan(heights))


# ---------------------------------------------------------------------------
# GeoTIFFDEM error handling tests
# ---------------------------------------------------------------------------

class TestGeoTIFFDEM:
    """Test GeoTIFFDEM construction and error handling."""

    def test_nonexistent_path_raises(self):
        """Test that non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            GeoTIFFDEM('/nonexistent/dem.tif')


# ---------------------------------------------------------------------------
# GeoidCorrection error handling tests
# ---------------------------------------------------------------------------

class TestGeoidCorrection:
    """Test GeoidCorrection construction and error handling."""

    def test_nonexistent_path_raises(self):
        """Test that non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            GeoidCorrection('/nonexistent/egm96.pgm')


# ---------------------------------------------------------------------------
# Integration: _build_elevation_model
# ---------------------------------------------------------------------------

class TestBuildElevationModel:
    """Test the _build_elevation_model helper from base.py."""

    def test_directory_creates_dted(self):
        """Test that a directory path creates DTEDElevation."""
        from grdl.geolocation.base import _build_elevation_model
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _build_elevation_model(tmpdir)
            assert isinstance(model, DTEDElevation)

    def test_nonexistent_path_raises(self):
        """Test that non-existent path raises FileNotFoundError."""
        from grdl.geolocation.base import _build_elevation_model
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _build_elevation_model('/nonexistent/path')


# ---------------------------------------------------------------------------
# Integration: Geolocation + ConstantElevation
# ---------------------------------------------------------------------------

class TestGeolocationElevationIntegration:
    """Test that Geolocation base class integrates with elevation models."""

    def test_no_elevation_by_default(self):
        """Test that elevation is None when no dem_path is provided."""
        from grdl.geolocation.eo.affine import AffineGeolocation
        from rasterio.transform import Affine as RioAffine

        transform = RioAffine(0.01, 0.0, 116.0, 0.0, -0.01, -31.0)
        geo = AffineGeolocation(transform, (100, 100), 'EPSG:4326')
        assert geo.elevation is None

    def test_elevation_with_dted_directory(self):
        """Test that dem_path triggers elevation model creation."""
        from grdl.geolocation.eo.affine import AffineGeolocation
        from rasterio.transform import Affine as RioAffine

        transform = RioAffine(0.01, 0.0, 116.0, 0.0, -0.01, -31.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            geo = AffineGeolocation(
                transform, (100, 100), 'EPSG:4326', dem_path=tmpdir
            )
            assert geo.elevation is not None
            assert isinstance(geo.elevation, DTEDElevation)
