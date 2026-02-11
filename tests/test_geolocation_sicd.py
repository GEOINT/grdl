# -*- coding: utf-8 -*-
"""
SICD Geolocation Tests - Coordinate transform tests with mock backends.

Tests SICDGeolocation using mocked sarpy projection functions to verify
argument shapes, return reshaping, dispatch paths, and factory construction.
No real SICD data required.

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

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import Optional

import numpy as np

from grdl.geolocation.sar.sicd import SICDGeolocation


# ---------------------------------------------------------------------------
# Minimal mock metadata
# ---------------------------------------------------------------------------

@dataclass
class MockRowCol:
    row: float = 0.0
    col: float = 0.0


@dataclass
class MockLatLonHAE:
    lat: float = 0.0
    lon: float = 0.0
    hae: float = 0.0


@dataclass
class MockSCP:
    ecf: Optional[object] = None
    llh: Optional[MockLatLonHAE] = None


@dataclass
class MockGeoData:
    scp: Optional[MockSCP] = None


@dataclass
class MockImageData:
    scp_pixel: Optional[MockRowCol] = None


@dataclass
class MockSICDMetadata:
    rows: int = 2048
    cols: int = 4096
    image_data: Optional[MockImageData] = None
    geo_data: Optional[MockGeoData] = None

    def get(self, key, default=None):
        return getattr(self, key, default)


def _make_metadata():
    """Build a minimal SICDMetadata for testing."""
    return MockSICDMetadata(
        rows=2048,
        cols=4096,
        image_data=MockImageData(scp_pixel=MockRowCol(1024, 2048)),
        geo_data=MockGeoData(scp=MockSCP(llh=MockLatLonHAE(34.05, -118.25, 100.0))),
    )


# ---------------------------------------------------------------------------
# Mock sarpy projection functions
# ---------------------------------------------------------------------------

def _mock_image_to_ground_geo(im_points, sicd_meta, ordering='latlong',
                               projection_type='HAE'):
    """Mock sarpy image_to_ground_geo: returns synthetic Nx3 [lat, lon, hae]."""
    n = im_points.shape[0]
    lats = 34.0 + im_points[:, 0] * 0.001
    lons = -118.0 + im_points[:, 1] * 0.001
    haes = np.full(n, 100.0)
    return np.column_stack([lats, lons, haes])


def _mock_ground_to_image_geo(coords, sicd_meta, ordering='latlong'):
    """Mock sarpy ground_to_image_geo: returns (Nx2 [row, col], delta, iters)."""
    n = coords.shape[0]
    rows = (coords[:, 0] - 34.0) / 0.001
    cols = (coords[:, 1] - (-118.0)) / 0.001
    image_points = np.column_stack([rows, cols])
    delta = np.zeros(n)
    iters = np.ones(n, dtype=int)
    return image_points, delta, iters


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def metadata():
    """Minimal SICDMetadata fixture."""
    return _make_metadata()


@pytest.fixture
def raw_meta():
    """Mock raw sarpy SICDType object."""
    return MagicMock(name='SICDType')


@pytest.fixture
def geo(metadata, raw_meta):
    """SICDGeolocation with mocked sarpy backend."""
    return SICDGeolocation(metadata, raw_meta, backend='sarpy')


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestConstruction:
    """Test SICDGeolocation construction and validation."""

    def test_creation(self, metadata, raw_meta):
        """Test basic construction with valid metadata."""
        geo = SICDGeolocation(metadata, raw_meta, backend='sarpy')
        assert geo.shape == (2048, 4096)
        assert geo.crs == 'WGS84'
        assert geo.backend == 'sarpy'

    def test_missing_image_data(self, raw_meta):
        """Test that missing image_data raises ValueError."""
        meta = MockSICDMetadata(
            rows=100, cols=100,
            image_data=None,
            geo_data=MockGeoData(scp=MockSCP()),
        )
        with pytest.raises(ValueError, match="image_data"):
            SICDGeolocation(meta, raw_meta, backend='sarpy')

    def test_missing_scp_pixel(self, raw_meta):
        """Test that missing scp_pixel raises ValueError."""
        meta = MockSICDMetadata(
            rows=100, cols=100,
            image_data=MockImageData(scp_pixel=None),
            geo_data=MockGeoData(scp=MockSCP()),
        )
        with pytest.raises(ValueError, match="scp_pixel"):
            SICDGeolocation(meta, raw_meta, backend='sarpy')

    def test_missing_geo_data(self, raw_meta):
        """Test that missing geo_data raises ValueError."""
        meta = MockSICDMetadata(
            rows=100, cols=100,
            image_data=MockImageData(scp_pixel=MockRowCol()),
            geo_data=None,
        )
        with pytest.raises(ValueError, match="geo_data"):
            SICDGeolocation(meta, raw_meta, backend='sarpy')

    def test_missing_scp(self, raw_meta):
        """Test that missing geo_data.scp raises ValueError."""
        meta = MockSICDMetadata(
            rows=100, cols=100,
            image_data=MockImageData(scp_pixel=MockRowCol()),
            geo_data=MockGeoData(scp=None),
        )
        with pytest.raises(ValueError, match="geo_data.scp"):
            SICDGeolocation(meta, raw_meta, backend='sarpy')


# ---------------------------------------------------------------------------
# Forward transform tests with mocked sarpy
# ---------------------------------------------------------------------------

class TestForwardTransform:
    """Test image_to_latlon dispatch paths with mocked projection."""

    @patch(
        'grdl.geolocation.sar.sicd.image_to_ground_geo',
        side_effect=_mock_image_to_ground_geo,
        create=True,
    )
    def test_scalar_forward(self, mock_proj, geo):
        """Test scalar image_to_latlon returns tuple of floats."""
        with patch(
            'sarpy.geometry.point_projection.image_to_ground_geo',
            side_effect=_mock_image_to_ground_geo,
        ):
            lat, lon, h = geo.image_to_latlon(500.0, 1000.0)
            assert isinstance(lat, float)
            assert isinstance(lon, float)
            assert isinstance(h, float)

    @patch(
        'sarpy.geometry.point_projection.image_to_ground_geo',
        side_effect=_mock_image_to_ground_geo,
    )
    def test_array_forward(self, mock_proj, geo):
        """Test array image_to_latlon returns arrays."""
        rows = np.array([0.0, 500.0, 1000.0])
        cols = np.array([0.0, 2000.0, 4000.0])
        lats, lons, heights = geo.image_to_latlon(rows, cols)
        assert lats.shape == (3,)
        assert lons.shape == (3,)
        assert heights.shape == (3,)

    @patch(
        'sarpy.geometry.point_projection.image_to_ground_geo',
        side_effect=_mock_image_to_ground_geo,
    )
    def test_stacked_2xN_forward(self, mock_proj, geo):
        """Test (2,N) stacked array image_to_latlon returns (3,N)."""
        pts = np.array([
            [0.0, 500.0, 1000.0],
            [0.0, 2000.0, 4000.0],
        ])
        result = geo.image_to_latlon(pts)
        assert result.shape == (3, 3)

    @patch(
        'sarpy.geometry.point_projection.image_to_ground_geo',
        side_effect=_mock_image_to_ground_geo,
    )
    def test_forward_argument_shape(self, mock_proj, geo):
        """Verify sarpy receives Nx2 array of [row, col]."""
        rows = np.array([100.0, 200.0])
        cols = np.array([300.0, 400.0])
        geo.image_to_latlon(rows, cols)

        call_args = mock_proj.call_args
        im_points = call_args[0][0]
        assert im_points.shape == (2, 2)
        np.testing.assert_array_equal(im_points[:, 0], rows)
        np.testing.assert_array_equal(im_points[:, 1], cols)


# ---------------------------------------------------------------------------
# Inverse transform tests with mocked sarpy
# ---------------------------------------------------------------------------

class TestInverseTransform:
    """Test latlon_to_image dispatch paths with mocked projection."""

    @patch(
        'sarpy.geometry.point_projection.ground_to_image_geo',
        side_effect=_mock_ground_to_image_geo,
    )
    def test_scalar_inverse(self, mock_proj, geo):
        """Test scalar latlon_to_image returns tuple of floats."""
        row, col = geo.latlon_to_image(34.05, -118.25)
        assert isinstance(row, float)
        assert isinstance(col, float)

    @patch(
        'sarpy.geometry.point_projection.ground_to_image_geo',
        side_effect=_mock_ground_to_image_geo,
    )
    def test_array_inverse(self, mock_proj, geo):
        """Test array latlon_to_image returns arrays."""
        lats = np.array([34.0, 34.5, 35.0])
        lons = np.array([-118.0, -117.5, -117.0])
        rows, cols = geo.latlon_to_image(lats, lons)
        assert rows.shape == (3,)
        assert cols.shape == (3,)

    @patch(
        'sarpy.geometry.point_projection.ground_to_image_geo',
        side_effect=_mock_ground_to_image_geo,
    )
    def test_stacked_2xN_inverse(self, mock_proj, geo):
        """Test (2,N) stacked array latlon_to_image returns (2,N)."""
        pts = np.array([
            [34.0, 34.5, 35.0],
            [-118.0, -117.5, -117.0],
        ])
        result = geo.latlon_to_image(pts)
        assert result.shape == (2, 3)

    @patch(
        'sarpy.geometry.point_projection.ground_to_image_geo',
        side_effect=_mock_ground_to_image_geo,
    )
    def test_inverse_argument_shape(self, mock_proj, geo):
        """Verify sarpy receives Nx3 array of [lat, lon, height]."""
        lats = np.array([34.0, 34.5])
        lons = np.array([-118.0, -117.5])
        geo.latlon_to_image(lats, lons)

        call_args = mock_proj.call_args
        coords = call_args[0][0]
        assert coords.shape == (2, 3)
        np.testing.assert_array_equal(coords[:, 0], lats)
        np.testing.assert_array_equal(coords[:, 1], lons)


# ---------------------------------------------------------------------------
# Round-trip tests with mocked sarpy
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Test forward-inverse round-trip with mocked projection."""

    @patch(
        'sarpy.geometry.point_projection.ground_to_image_geo',
        side_effect=_mock_ground_to_image_geo,
    )
    @patch(
        'sarpy.geometry.point_projection.image_to_ground_geo',
        side_effect=_mock_image_to_ground_geo,
    )
    def test_scalar_roundtrip(self, mock_fwd, mock_inv, geo):
        """Test scalar round-trip consistency."""
        row, col = 500.0, 1000.0
        lat, lon, _ = geo.image_to_latlon(row, col)
        row_back, col_back = geo.latlon_to_image(lat, lon)
        assert abs(row_back - row) < 0.01
        assert abs(col_back - col) < 0.01


# ---------------------------------------------------------------------------
# Sarkit backend tests
# ---------------------------------------------------------------------------

class TestSarkitBackend:
    """Test sarkit backend raises NotImplementedError."""

    def test_sarkit_forward_raises(self, metadata):
        """Sarkit backend forward projection raises NotImplementedError."""
        xml_tree = MagicMock(name='XMLTree')
        geo = SICDGeolocation(metadata, xml_tree, backend='sarkit')
        with pytest.raises(NotImplementedError, match="sarkit"):
            geo.image_to_latlon(500.0, 1000.0)

    def test_sarkit_inverse_raises(self, metadata):
        """Sarkit backend inverse projection raises NotImplementedError."""
        xml_tree = MagicMock(name='XMLTree')
        geo = SICDGeolocation(metadata, xml_tree, backend='sarkit')
        with pytest.raises(NotImplementedError, match="sarkit"):
            geo.latlon_to_image(34.05, -118.25)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestFromReader:
    """Test from_reader factory classmethod."""

    def test_from_sarpy_reader(self, metadata):
        """Test from_reader with sarpy backend."""
        sarpy_meta = MagicMock(name='SICDType')
        mock_reader = MagicMock()
        mock_reader.metadata = metadata
        mock_reader.backend = 'sarpy'
        mock_reader._sarpy_meta = sarpy_meta

        geo = SICDGeolocation.from_reader(mock_reader)
        assert geo.backend == 'sarpy'
        assert geo.shape == (2048, 4096)

    def test_from_sarkit_reader(self, metadata):
        """Test from_reader with sarkit backend."""
        xmltree = MagicMock(name='XMLTree')
        mock_reader = MagicMock()
        mock_reader.metadata = metadata
        mock_reader.backend = 'sarkit'
        mock_reader._xmltree = xmltree

        geo = SICDGeolocation.from_reader(mock_reader)
        assert geo.backend == 'sarkit'
        assert geo.shape == (2048, 4096)

    def test_from_reader_unsupported_backend(self, metadata):
        """Test from_reader with unsupported backend raises ValueError."""
        mock_reader = MagicMock()
        mock_reader.metadata = metadata
        mock_reader.backend = 'unknown'

        with pytest.raises(ValueError, match="Unsupported"):
            SICDGeolocation.from_reader(mock_reader)
