# -*- coding: utf-8 -*-
"""
Affine Geolocation Tests - Coordinate transform tests for geocoded rasters.

Tests AffineGeolocation with both geographic (EPSG:4326) and projected
(UTM EPSG:32755) coordinate reference systems. Verifies scalar, array,
and (2,N) dispatch paths, round-trip accuracy, footprint/bounds, and
factory construction.

Dependencies
------------
pytest
rasterio
pyproj

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
import numpy as np

from rasterio.transform import Affine

from grdl.geolocation.eo.affine import AffineGeolocation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def geo_geographic():
    """AffineGeolocation with EPSG:4326 (geographic, no reprojection)."""
    # 0.01 degree/pixel, origin at (lon=116.0, lat=-31.0)
    transform = Affine(0.01, 0.0, 116.0, 0.0, -0.01, -31.0)
    shape = (1000, 2000)
    return AffineGeolocation(transform, shape, 'EPSG:4326')


@pytest.fixture
def geo_utm():
    """AffineGeolocation with EPSG:32755 (UTM zone 55S, projected CRS)."""
    # 10m/pixel, origin at (500000 E, 6000000 N)
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0)
    shape = (1024, 2048)
    return AffineGeolocation(transform, shape, 'EPSG:32755')


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestConstruction:
    """Test AffineGeolocation construction and validation."""

    def test_geographic_creation(self, geo_geographic):
        """Test creation with geographic CRS."""
        assert geo_geographic.shape == (1000, 2000)
        assert geo_geographic.crs == 'WGS84'
        assert geo_geographic.native_crs == 'EPSG:4326'

    def test_utm_creation(self, geo_utm):
        """Test creation with projected CRS."""
        assert geo_utm.shape == (1024, 2048)
        assert geo_utm.crs == 'WGS84'
        assert geo_utm.native_crs == 'EPSG:32755'

    def test_bad_transform_type(self):
        """Test that non-Affine transform raises TypeError."""
        with pytest.raises(TypeError, match="rasterio.transform.Affine"):
            AffineGeolocation((1, 0, 0, 0, -1, 0), (100, 100), 'EPSG:4326')

    def test_from_reader(self):
        """Test from_reader factory with mock reader."""
        transform = Affine(0.01, 0.0, 116.0, 0.0, -0.01, -31.0)
        mock_reader = type('MockReader', (), {
            'metadata': {
                'rows': 500,
                'cols': 600,
                'transform': transform,
                'crs': 'EPSG:4326',
            }
        })()
        geo = AffineGeolocation.from_reader(mock_reader)
        assert geo.shape == (500, 600)
        assert geo.native_crs == 'EPSG:4326'

    def test_from_reader_missing_transform(self):
        """Test from_reader raises ValueError when transform is missing."""
        mock_reader = type('MockReader', (), {
            'metadata': {
                'rows': 500, 'cols': 600,
                'crs': 'EPSG:4326',
            }
        })()
        with pytest.raises(ValueError, match="affine transform"):
            AffineGeolocation.from_reader(mock_reader)

    def test_from_reader_missing_crs(self):
        """Test from_reader raises ValueError when CRS is missing."""
        transform = Affine(0.01, 0.0, 116.0, 0.0, -0.01, -31.0)
        mock_reader = type('MockReader', (), {
            'metadata': {
                'rows': 500, 'cols': 600,
                'transform': transform,
            }
        })()
        with pytest.raises(ValueError, match="CRS"):
            AffineGeolocation.from_reader(mock_reader)


# ---------------------------------------------------------------------------
# Forward transform tests (image_to_latlon)
# ---------------------------------------------------------------------------

class TestForwardTransform:
    """Test image_to_latlon with various input forms."""

    def test_scalar_geographic(self, geo_geographic):
        """Test scalar forward transform with geographic CRS."""
        lat, lon, h = geo_geographic.image_to_latlon(0, 0)
        assert isinstance(lat, float)
        assert isinstance(lon, float)
        # Origin pixel (0,0) -> (lon=116.0, lat=-31.0)
        assert abs(lon - 116.0) < 0.01
        assert abs(lat - (-31.0)) < 0.01

    def test_scalar_utm(self, geo_utm):
        """Test scalar forward transform with projected CRS."""
        lat, lon, h = geo_utm.image_to_latlon(0, 0)
        assert isinstance(lat, float)
        assert isinstance(lon, float)
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180

    def test_array_geographic(self, geo_geographic):
        """Test array forward transform with geographic CRS."""
        rows = np.array([0.0, 100.0, 500.0, 999.0])
        cols = np.array([0.0, 500.0, 1000.0, 1999.0])
        lats, lons, heights = geo_geographic.image_to_latlon(rows, cols)
        assert lats.shape == (4,)
        assert lons.shape == (4,)
        assert heights.shape == (4,)

    def test_stacked_2xN_geographic(self, geo_geographic):
        """Test (2,N) stacked array forward transform."""
        pts = np.array([
            [0.0, 100.0, 500.0],   # rows
            [0.0, 500.0, 1000.0],  # cols
        ])
        result = geo_geographic.image_to_latlon(pts)
        assert result.shape == (3, 3)  # (3, N): lats, lons, heights

    def test_stacked_2xN_utm(self, geo_utm):
        """Test (2,N) stacked array forward transform with projected CRS."""
        pts = np.array([
            [0.0, 512.0, 1023.0],
            [0.0, 1024.0, 2047.0],
        ])
        result = geo_utm.image_to_latlon(pts)
        assert result.shape == (3, 3)
        # All lats/lons should be finite
        assert np.all(np.isfinite(result))

    def test_bad_stacked_shape(self, geo_geographic):
        """Test that non-(2,N) stacked array raises ValueError."""
        bad_pts = np.array([
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
        ])
        with pytest.raises(ValueError, match="Expected \\(2, N\\)"):
            geo_geographic.image_to_latlon(bad_pts)

    def test_height_passthrough(self, geo_geographic):
        """Test that height parameter is passed through."""
        _, _, h = geo_geographic.image_to_latlon(500, 1000, height=42.0)
        assert abs(h - 42.0) < 1e-6


# ---------------------------------------------------------------------------
# Inverse transform tests (latlon_to_image)
# ---------------------------------------------------------------------------

class TestInverseTransform:
    """Test latlon_to_image with various input forms."""

    def test_scalar_geographic(self, geo_geographic):
        """Test scalar inverse transform with geographic CRS."""
        row, col = geo_geographic.latlon_to_image(-31.0, 116.0)
        assert isinstance(row, float)
        assert isinstance(col, float)
        assert abs(row) < 0.5
        assert abs(col) < 0.5

    def test_scalar_utm(self, geo_utm):
        """Test scalar inverse transform with projected CRS."""
        # Forward then inverse
        lat, lon, _ = geo_utm.image_to_latlon(512, 1024)
        row, col = geo_utm.latlon_to_image(lat, lon)
        assert abs(row - 512.0) < 0.01
        assert abs(col - 1024.0) < 0.01

    def test_array_geographic(self, geo_geographic):
        """Test array inverse transform."""
        lats = np.array([-31.0, -31.5, -32.0])
        lons = np.array([116.0, 118.0, 120.0])
        rows, cols = geo_geographic.latlon_to_image(lats, lons)
        assert rows.shape == (3,)
        assert cols.shape == (3,)

    def test_stacked_2xN(self, geo_geographic):
        """Test (2,N) stacked array inverse transform."""
        pts = np.array([
            [-31.0, -31.5, -32.0],   # lats
            [116.0, 118.0, 120.0],    # lons
        ])
        result = geo_geographic.latlon_to_image(pts)
        assert result.shape == (2, 3)  # (2, N): rows, cols


# ---------------------------------------------------------------------------
# Round-trip accuracy
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Test forward-inverse round-trip accuracy."""

    def test_scalar_roundtrip_geographic(self, geo_geographic):
        """Test scalar round-trip with geographic CRS (< 0.01 pixel error)."""
        row, col = 500.0, 1000.0
        lat, lon, _ = geo_geographic.image_to_latlon(row, col)
        row_back, col_back = geo_geographic.latlon_to_image(lat, lon)
        assert abs(row_back - row) < 0.01
        assert abs(col_back - col) < 0.01

    def test_scalar_roundtrip_utm(self, geo_utm):
        """Test scalar round-trip with projected CRS (< 0.01 pixel error)."""
        row, col = 512.0, 1024.0
        lat, lon, _ = geo_utm.image_to_latlon(row, col)
        row_back, col_back = geo_utm.latlon_to_image(lat, lon)
        assert abs(row_back - row) < 0.01
        assert abs(col_back - col) < 0.01

    def test_batch_roundtrip_geographic(self, geo_geographic):
        """Test batch round-trip with geographic CRS."""
        rows = np.array([0.0, 250.0, 500.0, 750.0, 999.0])
        cols = np.array([0.0, 500.0, 1000.0, 1500.0, 1999.0])
        lats, lons, _ = geo_geographic.image_to_latlon(rows, cols)
        rows_back, cols_back = geo_geographic.latlon_to_image(lats, lons)
        assert np.max(np.abs(rows_back - rows)) < 0.01
        assert np.max(np.abs(cols_back - cols)) < 0.01

    def test_stacked_roundtrip_utm(self, geo_utm):
        """Test (2,N) round-trip with projected CRS."""
        pts = np.array([
            [0.0, 256.0, 512.0, 768.0, 1023.0],
            [0.0, 512.0, 1024.0, 1536.0, 2047.0],
        ])
        geo_pts = geo_utm.image_to_latlon(pts)  # (3, N)
        latlon_pts = geo_pts[:2]  # (2, N) = [lats, lons]
        img_back = geo_utm.latlon_to_image(latlon_pts)  # (2, N)
        assert np.max(np.abs(img_back - pts)) < 0.01


# ---------------------------------------------------------------------------
# Footprint and bounds
# ---------------------------------------------------------------------------

class TestFootprintBounds:
    """Test get_footprint and get_bounds."""

    def test_footprint_geographic(self, geo_geographic):
        """Test footprint returns a valid polygon."""
        footprint = geo_geographic.get_footprint()
        assert footprint['type'] == 'Polygon'
        assert footprint['coordinates'] is not None
        assert footprint['bounds'] is not None

    def test_bounds_geographic(self, geo_geographic):
        """Test bounds returns valid (min_lon, min_lat, max_lon, max_lat)."""
        min_lon, min_lat, max_lon, max_lat = geo_geographic.get_bounds()
        assert min_lon < max_lon
        assert min_lat < max_lat

    def test_footprint_utm(self, geo_utm):
        """Test footprint with projected CRS."""
        footprint = geo_utm.get_footprint()
        assert footprint['type'] == 'Polygon'
        assert footprint['bounds'] is not None

    def test_bounds_utm(self, geo_utm):
        """Test bounds with projected CRS."""
        min_lon, min_lat, max_lon, max_lat = geo_utm.get_bounds()
        assert min_lon < max_lon
        assert min_lat < max_lat
        assert -90 <= min_lat <= 90
        assert -180 <= max_lon <= 180


# ---------------------------------------------------------------------------
# Consistency check: array vs scalar
# ---------------------------------------------------------------------------

class TestConsistency:
    """Verify that scalar and array paths produce identical results."""

    def test_forward_consistency(self, geo_geographic):
        """Scalar and array forward transforms produce same results."""
        rows = np.array([100.0, 200.0, 300.0])
        cols = np.array([400.0, 500.0, 600.0])
        lats_arr, lons_arr, h_arr = geo_geographic.image_to_latlon(rows, cols)

        for i in range(3):
            lat_s, lon_s, h_s = geo_geographic.image_to_latlon(
                float(rows[i]), float(cols[i])
            )
            assert abs(lats_arr[i] - lat_s) < 1e-10
            assert abs(lons_arr[i] - lon_s) < 1e-10

    def test_inverse_consistency(self, geo_geographic):
        """Scalar and array inverse transforms produce same results."""
        lats = np.array([-31.5, -32.0, -32.5])
        lons = np.array([117.0, 118.0, 119.0])
        rows_arr, cols_arr = geo_geographic.latlon_to_image(lats, lons)

        for i in range(3):
            row_s, col_s = geo_geographic.latlon_to_image(
                float(lats[i]), float(lons[i])
            )
            assert abs(rows_arr[i] - row_s) < 1e-10
            assert abs(cols_arr[i] - col_s) < 1e-10
