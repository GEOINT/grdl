# -*- coding: utf-8 -*-
"""Tests for grdl.geolocation.sar.sidd — SIDDGeolocation."""

import numpy as np
import pytest

from grdl.geolocation.sar.sidd import (
    SIDDGeolocation,
    _geodetic_to_ecef,
    _ecef_to_geodetic,
)


# ===================================================================
# WGS-84 coordinate conversion tests
# ===================================================================

class TestGeodeticECEF:
    """Test geodetic ↔ ECEF conversions against known values."""

    def test_origin_on_equator_prime_meridian(self):
        """Lat=0, lon=0, h=0 → X=a, Y=0, Z=0."""
        x, y, z = _geodetic_to_ecef(
            np.array([0.0]), np.array([0.0]), np.array([0.0])
        )
        assert abs(x[0] - 6378137.0) < 0.01
        assert abs(y[0]) < 0.01
        assert abs(z[0]) < 0.01

    def test_north_pole(self):
        """Lat=90, lon=0, h=0 → X≈0, Y=0, Z≈b."""
        x, y, z = _geodetic_to_ecef(
            np.array([90.0]), np.array([0.0]), np.array([0.0])
        )
        assert abs(x[0]) < 0.01
        assert abs(y[0]) < 0.01
        assert abs(z[0] - 6356752.314245179) < 0.01

    def test_roundtrip(self):
        """geodetic → ECEF → geodetic should be identity."""
        lats = np.array([34.05, -31.95, 0.0, 89.99])
        lons = np.array([-118.25, 115.86, 0.0, 45.0])
        heights = np.array([100.0, 50.0, 0.0, 3000.0])

        x, y, z = _geodetic_to_ecef(lats, lons, heights)
        lats2, lons2, heights2 = _ecef_to_geodetic(x, y, z)

        np.testing.assert_allclose(lats2, lats, atol=1e-8)
        np.testing.assert_allclose(lons2, lons, atol=1e-8)
        np.testing.assert_allclose(heights2, heights, atol=1e-3)

    def test_vectorized(self):
        """Both functions accept arrays and return matching shapes."""
        n = 100
        lats = np.random.uniform(-90, 90, n)
        lons = np.random.uniform(-180, 180, n)
        heights = np.random.uniform(0, 10000, n)

        x, y, z = _geodetic_to_ecef(lats, lons, heights)
        assert x.shape == (n,)

        lats2, lons2, h2 = _ecef_to_geodetic(x, y, z)
        np.testing.assert_allclose(lats2, lats, atol=1e-8)
        np.testing.assert_allclose(lons2, lons, atol=1e-8)
        np.testing.assert_allclose(h2, heights, atol=1e-3)


# ===================================================================
# Synthetic SIDD metadata fixtures
# ===================================================================

class _FakeXYZ:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)


class _FakeRowCol:
    def __init__(self, row, col):
        self.row, self.col = float(row), float(col)


class _FakeRefPoint:
    def __init__(self, ecef, point, name=None):
        self.ecef = ecef
        self.point = point
        self.name = name


class _FakeProductPlane:
    def __init__(self, row_unit_vector, col_unit_vector):
        self.row_unit_vector = row_unit_vector
        self.col_unit_vector = col_unit_vector


class _FakePlaneProjection:
    def __init__(self, reference_point, sample_spacing, product_plane,
                 time_coa_poly=None):
        self.reference_point = reference_point
        self.sample_spacing = sample_spacing
        self.product_plane = product_plane
        self.time_coa_poly = time_coa_poly


class _FakeMeasurement:
    def __init__(self, projection_type, plane_projection,
                 pixel_footprint=None, arp_flag=None, arp_poly=None):
        self.projection_type = projection_type
        self.plane_projection = plane_projection
        self.pixel_footprint = pixel_footprint
        self.arp_flag = arp_flag
        self.arp_poly = arp_poly


class _FakeMetadata:
    """Minimal SIDDMetadata mock for testing."""
    def __init__(self, rows, cols, measurement, geo_data=None):
        self.rows = rows
        self.cols = cols
        self.measurement = measurement
        self.geo_data = geo_data
        self.format = 'SIDD'
        self.dtype = 'uint8'
        self.bands = 1
        self.crs = None
        self.nodata = None
        self.extras = None


def _make_pgd_metadata(
    lat0: float = 34.05,
    lon0: float = -118.25,
    h0: float = 100.0,
    spacing_m: float = 1.0,
    rows: int = 1000,
    cols: int = 1000,
):
    """Build a synthetic PGD (PlaneProjection) metadata object.

    The reference point is placed at the center of the image.
    Row direction is local North, column direction is local East.
    """
    # Reference point ECEF
    x0, y0, z0 = _geodetic_to_ecef(
        np.array([lat0]), np.array([lon0]), np.array([h0])
    )
    ecef = _FakeXYZ(x0[0], y0[0], z0[0])
    r0, c0 = rows / 2.0, cols / 2.0
    point = _FakeRowCol(r0, c0)
    ref = _FakeRefPoint(ecef, point)

    # Build local ENU unit vectors at (lat0, lon0)
    lat_r = np.radians(lat0)
    lon_r = np.radians(lon0)
    sin_lat, cos_lat = np.sin(lat_r), np.cos(lat_r)
    sin_lon, cos_lon = np.sin(lon_r), np.cos(lon_r)

    # East  = (-sin_lon, cos_lon, 0)
    # North = (-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat)
    east = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])

    # Row direction = North, Col direction = East
    row_uv = _FakeXYZ(*north)
    col_uv = _FakeXYZ(*east)

    plane = _FakeProductPlane(row_uv, col_uv)
    spacing = _FakeRowCol(spacing_m, spacing_m)
    pp = _FakePlaneProjection(ref, spacing, plane)
    meas = _FakeMeasurement('PlaneProjection', pp)

    return _FakeMetadata(rows, cols, meas)


# ===================================================================
# PlaneProjection tests
# ===================================================================

class TestPlaneProjection:
    """Test PGD (PlaneProjection) forward and inverse."""

    def test_center_roundtrip(self):
        """Center pixel should round-trip to < 0.01 px error."""
        meta = _make_pgd_metadata()
        geo = SIDDGeolocation(meta)

        lat, lon, h = geo.image_to_latlon(500.0, 500.0)
        row, col = geo.latlon_to_image(lat, lon, h)

        assert abs(row - 500.0) < 0.01
        assert abs(col - 500.0) < 0.01

    def test_corner_roundtrip(self):
        """All four corners should round-trip with sub-pixel error."""
        meta = _make_pgd_metadata()
        geo = SIDDGeolocation(meta)

        corners = [(0.0, 0.0), (0.0, 999.0), (999.0, 0.0), (999.0, 999.0)]
        for r, c in corners:
            lat, lon, h = geo.image_to_latlon(r, c)
            r2, c2 = geo.latlon_to_image(lat, lon, h)
            assert abs(r2 - r) < 0.1, f"Row error at ({r},{c}): {abs(r2-r)}"
            assert abs(c2 - c) < 0.1, f"Col error at ({r},{c}): {abs(c2-c)}"

    def test_array_input(self):
        """Forward and inverse accept arrays."""
        meta = _make_pgd_metadata()
        geo = SIDDGeolocation(meta)

        rows = np.array([0.0, 250.0, 500.0, 750.0, 999.0])
        cols = np.array([0.0, 250.0, 500.0, 750.0, 999.0])

        lats, lons, heights = geo.image_to_latlon(rows, cols)
        assert lats.shape == (5,)

        rows2, cols2 = geo.latlon_to_image(lats, lons, heights)
        np.testing.assert_allclose(rows2, rows, atol=0.1)
        np.testing.assert_allclose(cols2, cols, atol=0.1)

    def test_center_returns_reference_latlon(self):
        """Center pixel should geolocate near the reference lat/lon."""
        meta = _make_pgd_metadata(lat0=34.05, lon0=-118.25)
        geo = SIDDGeolocation(meta)

        lat, lon, h = geo.image_to_latlon(500.0, 500.0)
        assert abs(lat - 34.05) < 0.01
        assert abs(lon - (-118.25)) < 0.01

    def test_footprint(self):
        """get_footprint should return valid bounds."""
        meta = _make_pgd_metadata()
        geo = SIDDGeolocation(meta)

        fp = geo.get_footprint()
        assert fp['type'] == 'Polygon'
        assert fp['bounds'] is not None
        min_lon, min_lat, max_lon, max_lat = fp['bounds']
        assert min_lat < max_lat
        assert min_lon < max_lon

    def test_different_spacing(self):
        """Varying sample spacing should scale the footprint."""
        meta_fine = _make_pgd_metadata(spacing_m=0.5)
        meta_coarse = _make_pgd_metadata(spacing_m=2.0)
        geo_fine = SIDDGeolocation(meta_fine)
        geo_coarse = SIDDGeolocation(meta_coarse)

        # Corner pixel should be further away for coarse spacing
        lat_f, lon_f, _ = geo_fine.image_to_latlon(0.0, 0.0)
        lat_c, lon_c, _ = geo_coarse.image_to_latlon(0.0, 0.0)

        # Distance from center should be ~4x larger for coarse
        d_fine = abs(lat_f - 34.05) + abs(lon_f - (-118.25))
        d_coarse = abs(lat_c - 34.05) + abs(lon_c - (-118.25))
        assert d_coarse > d_fine * 3.5

    def test_shape_attribute(self):
        """Shape should match metadata rows/cols."""
        meta = _make_pgd_metadata(rows=2000, cols=3000)
        geo = SIDDGeolocation(meta)
        assert geo.shape == (2000, 3000)


# ===================================================================
# GeographicProjection tests
# ===================================================================

class TestGeographicProjection:
    """Test GGD (GeographicProjection) forward and inverse."""

    def _make_ggd_metadata(self):
        """Build synthetic GGD metadata (1 arc-sec spacing)."""
        lat0, lon0, h0 = 34.05, -118.25, 100.0
        x0, y0, z0 = _geodetic_to_ecef(
            np.array([lat0]), np.array([lon0]), np.array([h0])
        )
        ecef = _FakeXYZ(x0[0], y0[0], z0[0])
        ref = _FakeRefPoint(ecef, _FakeRowCol(500.0, 500.0))
        # 1 arc-second spacing
        spacing = _FakeRowCol(1.0, 1.0)
        # GGD doesn't use product_plane, but _init_geographic reads
        # from plane_projection for the reference point + spacing
        pp = _FakePlaneProjection(ref, spacing, _FakeProductPlane(
            _FakeXYZ(0, 0, 1), _FakeXYZ(1, 0, 0)))
        meas = _FakeMeasurement('GeographicProjection', pp)
        return _FakeMetadata(1000, 1000, meas)

    def test_center_roundtrip(self):
        meta = self._make_ggd_metadata()
        geo = SIDDGeolocation(meta)

        lat, lon, h = geo.image_to_latlon(500.0, 500.0)
        row, col = geo.latlon_to_image(lat, lon, h)

        assert abs(row - 500.0) < 0.01
        assert abs(col - 500.0) < 0.01

    def test_center_returns_reference_latlon(self):
        meta = self._make_ggd_metadata()
        geo = SIDDGeolocation(meta)

        lat, lon, h = geo.image_to_latlon(500.0, 500.0)
        assert abs(lat - 34.05) < 0.01
        assert abs(lon - (-118.25)) < 0.01

    def test_latitude_decreases_with_row(self):
        """Moving down (increasing row) should decrease latitude."""
        meta = self._make_ggd_metadata()
        geo = SIDDGeolocation(meta)

        lat_top, _, _ = geo.image_to_latlon(0.0, 500.0)
        lat_bot, _, _ = geo.image_to_latlon(999.0, 500.0)
        assert lat_top > lat_bot

    def test_longitude_increases_with_col(self):
        """Moving right (increasing col) should increase longitude."""
        meta = self._make_ggd_metadata()
        geo = SIDDGeolocation(meta)

        _, lon_left, _ = geo.image_to_latlon(500.0, 0.0)
        _, lon_right, _ = geo.image_to_latlon(500.0, 999.0)
        assert lon_right > lon_left


# ===================================================================
# Validation / error tests
# ===================================================================

class TestValidation:
    """Test that SIDDGeolocation fails fast on bad metadata."""

    def test_no_measurement(self):
        meta = _FakeMetadata(100, 100, measurement=None)
        with pytest.raises(ValueError, match="measurement is required"):
            SIDDGeolocation(meta)

    def test_unsupported_projection(self):
        meas = _FakeMeasurement('PolynomialProjection', None)
        meta = _FakeMetadata(100, 100, meas)
        with pytest.raises(ValueError, match="Unsupported"):
            SIDDGeolocation(meta)

    def test_plane_missing_reference_point(self):
        pp = _FakePlaneProjection(None, _FakeRowCol(1, 1),
                                  _FakeProductPlane(
                                      _FakeXYZ(0, 0, 1),
                                      _FakeXYZ(1, 0, 0)))
        meas = _FakeMeasurement('PlaneProjection', pp)
        meta = _FakeMetadata(100, 100, meas)
        with pytest.raises(ValueError, match="reference_point"):
            SIDDGeolocation(meta)

    def test_plane_missing_sample_spacing(self):
        ref = _FakeRefPoint(_FakeXYZ(1e6, 0, 0), _FakeRowCol(50, 50))
        pp = _FakePlaneProjection(ref, None, _FakeProductPlane(
            _FakeXYZ(0, 0, 1), _FakeXYZ(1, 0, 0)))
        meas = _FakeMeasurement('PlaneProjection', pp)
        meta = _FakeMetadata(100, 100, meas)
        with pytest.raises(ValueError, match="sample_spacing"):
            SIDDGeolocation(meta)

    def test_from_reader_wrong_type(self):
        with pytest.raises(TypeError, match="Expected SIDDReader"):
            SIDDGeolocation.from_reader("not_a_reader")
