# -*- coding: utf-8 -*-
"""
Corner Geolocation Tests - Corner-coordinate fallback for EO NITF.

Tests CornerGeolocation: homography fit and round-trip accuracy,
stacked (N, 2) batch API, BLOCKA corner-string parsing (both legal
formats), IGEOLO parsing for all ICORDS forms (G/C/D/N/S/U including
MGRS), from_reader source priority (CSCRNA -> BLOCKA -> IGEOLO), and
create_geolocation factory integration.

Dependencies
------------
pytest
pyproj

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-06-09

Modified
--------
2026-06-09
"""

# Standard library
from types import SimpleNamespace

# Third-party
import numpy as np
import pytest
import pyproj

# GRDL internal
from grdl.geolocation import create_geolocation
from grdl.geolocation.eo.corner import (
    CornerGeolocation,
    _parse_blocka_corner,
    _parse_igeolo,
    _parse_igeolo_corner,
)
from grdl.IO.models.eo_nitf import BLOCKAMetadata, EONITFMetadata


# ---------------------------------------------------------------------------
# Fixtures and stubs
# ---------------------------------------------------------------------------

# Non-affine quadrilateral (UL, UR, LR, LL) to exercise the projective fit
CORNERS = np.array([
    [39.00, -77.20],   # UL
    [39.05, -77.00],   # UR
    [38.90, -76.98],   # LR
    [38.88, -77.21],   # LL
])

SHAPE = (1000, 2000)


class _StubReader:
    """Minimal reader stub: metadata attribute + get_shape()."""

    def __init__(self, metadata, shape=SHAPE):
        self.metadata = metadata
        self._shape = shape

    def get_shape(self):
        return self._shape


def _utm_expected(epsg, easting, northing):
    """Compute expected (lat, lon) for a UTM point via pyproj."""
    transformer = pyproj.Transformer.from_crs(
        f'EPSG:{epsg}', 'EPSG:4326', always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    return lat, lon


@pytest.fixture
def geo():
    """CornerGeolocation over the test quadrilateral."""
    return CornerGeolocation(CORNERS, SHAPE)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------

class TestConstruction:
    """Construction, validation, and attributes."""

    def test_attributes(self, geo):
        assert geo.shape == SHAPE
        assert geo.crs == 'WGS84'
        assert geo.accuracy_source == 'corners'
        np.testing.assert_allclose(geo.corners, CORNERS)

    def test_bad_corner_shape_raises(self):
        with pytest.raises(ValueError):
            CornerGeolocation(CORNERS[:3], SHAPE)

    def test_nonfinite_corners_raise(self):
        bad = CORNERS.copy()
        bad[1, 0] = np.nan
        with pytest.raises(ValueError):
            CornerGeolocation(bad, SHAPE)

    def test_degenerate_corners_raise(self):
        degenerate = np.tile([[39.0, -77.0]], (4, 1))
        with pytest.raises(ValueError):
            CornerGeolocation(degenerate, SHAPE)

    def test_tiny_shape_raises(self):
        with pytest.raises(ValueError):
            CornerGeolocation(CORNERS, (1, 2000))


# ---------------------------------------------------------------------------
# Homography forward / inverse
# ---------------------------------------------------------------------------

class TestHomography:
    """Forward projection, inverse, and round-trip accuracy."""

    def test_corner_pixels_map_to_corners(self, geo):
        rows, cols = SHAPE
        pixel_corners = np.array([
            [0.0, 0.0],
            [0.0, cols - 1.0],
            [rows - 1.0, cols - 1.0],
            [rows - 1.0, 0.0],
        ])
        result = geo.image_to_latlon(pixel_corners)
        np.testing.assert_allclose(result[:, :2], CORNERS, atol=1e-9)

    def test_corners_map_to_corner_pixels(self, geo):
        rows, cols = SHAPE
        expected = np.array([
            [0.0, 0.0],
            [0.0, cols - 1.0],
            [rows - 1.0, cols - 1.0],
            [rows - 1.0, 0.0],
        ])
        result = geo.latlon_to_image(CORNERS)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_interior_round_trip(self, geo):
        pixels = np.array([
            [123.4, 567.8],
            [500.0, 1000.0],
            [999.0, 0.5],
            [1.0, 1998.0],
            [750.25, 250.75],
        ])
        geo_pts = geo.image_to_latlon(pixels)
        back = geo.latlon_to_image(geo_pts[:, :2])
        np.testing.assert_allclose(back, pixels, atol=1e-6)

    def test_scalar_unpacking(self, geo):
        lat, lon, h = geo.image_to_latlon(0, 0)
        assert lat == pytest.approx(39.0)
        assert lon == pytest.approx(-77.2)
        row, col = geo.latlon_to_image(lat, lon)
        assert row == pytest.approx(0.0, abs=1e-6)
        assert col == pytest.approx(0.0, abs=1e-6)

    def test_batch_stacked_shapes(self, geo):
        pixels = np.column_stack([
            np.linspace(0, 999, 7), np.linspace(0, 1999, 7)])
        geo_pts = geo.image_to_latlon(pixels)
        assert geo_pts.shape == (7, 3)
        back = geo.latlon_to_image(geo_pts)  # (N, 3) input accepted
        assert back.shape == (7, 2)
        np.testing.assert_allclose(back, pixels, atol=1e-6)

    def test_height_passthrough(self, geo):
        result = geo.image_to_latlon(500, 1000, height=42.5)
        assert result[2] == pytest.approx(42.5)

    def test_constructor_height_is_default_hae(self):
        geo_h = CornerGeolocation(CORNERS, SHAPE, height=125.0)
        assert geo_h.default_hae == pytest.approx(125.0)
        result = geo_h.image_to_latlon(500, 1000)
        assert result[2] == pytest.approx(125.0)

    def test_footprint_bounds(self, geo):
        bounds = geo.get_bounds()
        min_lon, min_lat, max_lon, max_lat = bounds
        assert min_lat == pytest.approx(38.88, abs=0.01)
        assert max_lat == pytest.approx(39.05, abs=0.01)
        assert min_lon == pytest.approx(-77.21, abs=0.01)
        assert max_lon == pytest.approx(-76.98, abs=0.01)


# ---------------------------------------------------------------------------
# BLOCKA corner parsing
# ---------------------------------------------------------------------------

class TestBlockaParsing:
    """BLOCKA 21-character corner strings, both legal formats."""

    def test_dms_format(self):
        lat, lon = _parse_blocka_corner('383045.50N0770130.25W')
        assert lat == pytest.approx(38 + 30 / 60 + 45.50 / 3600)
        assert lon == pytest.approx(-(77 + 1 / 60 + 30.25 / 3600))

    def test_dms_southern_eastern(self):
        lat, lon = _parse_blocka_corner('335130.00S1511230.00E')
        assert lat == pytest.approx(-(33 + 51 / 60 + 30 / 3600))
        assert lon == pytest.approx(151 + 12 / 60 + 30 / 3600)

    def test_decimal_format(self):
        lat, lon = _parse_blocka_corner('+38.756083-077.020972')
        assert lat == pytest.approx(38.756083)
        assert lon == pytest.approx(-77.020972)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_blocka_corner('not a corner string!!')

    def test_from_reader_blocka_dms(self):
        blocka = BLOCKAMetadata(
            block_number=1,
            frfc_loc='390000.00N0771200.00W',  # UL
            frlc_loc='390000.00N0770000.00W',  # UR
            lrfc_loc='385400.00N0771200.00W',  # LL
            lrlc_loc='385400.00N0770000.00W',  # LR
        )
        meta = SimpleNamespace(blocka=blocka, igeolo=None, icords=None)
        geo = CornerGeolocation.from_reader(_StubReader(meta))
        assert geo.accuracy_source == 'BLOCKA'
        expected = np.array([
            [39.0, -77.2],   # UL = FRFC
            [39.0, -77.0],   # UR = FRLC
            [38.9, -77.0],   # LR = LRLC
            [38.9, -77.2],   # LL = LRFC
        ])
        np.testing.assert_allclose(geo.corners, expected, atol=1e-9)

    def test_from_reader_blocka_decimal(self):
        blocka = BLOCKAMetadata(
            block_number=1,
            frfc_loc='+39.000000-077.200000',
            frlc_loc='+39.000000-077.000000',
            lrfc_loc='+38.900000-077.200000',
            lrlc_loc='+38.900000-077.000000',
        )
        meta = SimpleNamespace(blocka=blocka)
        geo = CornerGeolocation.from_reader(_StubReader(meta))
        assert geo.accuracy_source == 'BLOCKA'
        np.testing.assert_allclose(
            geo.corners,
            [[39.0, -77.2], [39.0, -77.0], [38.9, -77.0], [38.9, -77.2]],
            atol=1e-9,
        )


# ---------------------------------------------------------------------------
# IGEOLO parsing
# ---------------------------------------------------------------------------

class TestIgeoloParsing:
    """IGEOLO 60-character field across all ICORDS forms."""

    def test_icords_g(self):
        igeolo = (
            '390000N0771200W'   # UL
            '390000N0770000W'   # UR
            '385400N0770000W'   # LR
            '385400N0771200W'   # LL
        )
        corners = _parse_igeolo(igeolo, 'G')
        expected = np.array([
            [39.0, -77.2], [39.0, -77.0], [38.9, -77.0], [38.9, -77.2]])
        np.testing.assert_allclose(corners, expected, atol=1e-9)

    def test_icords_c_same_as_g(self):
        igeolo = '390000N0771200W' * 4
        corners = _parse_igeolo(igeolo, 'C')
        np.testing.assert_allclose(corners, np.tile([39.0, -77.2], (4, 1)))

    def test_icords_d(self):
        igeolo = (
            '+39.000-077.200'
            '+39.000-077.000'
            '+38.900-077.000'
            '+38.900-077.200'
        )
        corners = _parse_igeolo(igeolo, 'D')
        expected = np.array([
            [39.0, -77.2], [39.0, -77.0], [38.9, -77.0], [38.9, -77.2]])
        np.testing.assert_allclose(corners, expected, atol=1e-9)

    def test_icords_n_utm_north(self):
        # Zone 18 north, rectangle near Washington DC
        utm = [
            (310000, 4320000),  # UL
            (330000, 4320000),  # UR
            (330000, 4300000),  # LR
            (310000, 4300000),  # LL
        ]
        igeolo = ''.join(f'18{e:06d}{n:07d}' for e, n in utm)
        corners = _parse_igeolo(igeolo, 'N')
        expected = np.array([_utm_expected(32618, e, n) for e, n in utm])
        np.testing.assert_allclose(corners, expected, atol=1e-9)
        # Coarse sanity: DC area
        assert np.all((corners[:, 0] > 38.5) & (corners[:, 0] < 39.5))
        assert np.all((corners[:, 1] > -78.5) & (corners[:, 1] < -76.5))

    def test_icords_s_utm_south(self):
        # Zone 56 south, rectangle near Sydney
        utm = [
            (330000, 6260000),  # UL
            (340000, 6260000),  # UR
            (340000, 6250000),  # LR
            (330000, 6250000),  # LL
        ]
        igeolo = ''.join(f'56{e:06d}{n:07d}' for e, n in utm)
        corners = _parse_igeolo(igeolo, 'S')
        expected = np.array([_utm_expected(32756, e, n) for e, n in utm])
        np.testing.assert_allclose(corners, expected, atol=1e-9)
        # Coarse sanity: Sydney area, southern hemisphere
        assert np.all(corners[:, 0] < -33.0)
        assert np.all(corners[:, 1] > 150.0)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            _parse_igeolo('390000N0771200W', 'G')

    def test_unsupported_icords_raises(self):
        igeolo = '390000N0771200W' * 4
        with pytest.raises(ValueError):
            _parse_igeolo(igeolo, 'X')

    def test_blank_icords_raises(self):
        igeolo = '390000N0771200W' * 4
        with pytest.raises(ValueError):
            _parse_igeolo(igeolo, ' ')

    def test_malformed_corner_raises(self):
        igeolo = '99zz00N0771200W' + '390000N0771200W' * 3
        with pytest.raises(ValueError):
            _parse_igeolo(igeolo, 'G')


class TestMgrsParsing:
    """ICORDS 'U' MGRS corners against known MGRS <-> lat/lon pairs."""

    def test_equator_prime_meridian(self):
        # Canonical pair: 31NAA6602100000 is (0 N, 0 E) to within meters
        lat, lon = _parse_igeolo_corner('31NAA6602100000', 'U')
        assert lat == pytest.approx(0.0, abs=1e-3)
        assert lon == pytest.approx(0.0, abs=1e-3)

    def test_honolulu_example(self):
        # Canonical MGRS example 4QFJ 12345 67890 (Honolulu area).
        # Decomposition: zone 4 set 1 -> 'F' = 600 km easting;
        # row 'J' (even zone, offset 5) -> 300 km, band Q min northing
        # 1,700 km -> +2,000 km cycle = 2,300 km.
        lat, lon = _parse_igeolo_corner('04QFJ1234567890', 'U')
        exp_lat, exp_lon = _utm_expected(32604, 612345, 2367890)
        assert lat == pytest.approx(exp_lat, abs=1e-9)
        assert lon == pytest.approx(exp_lon, abs=1e-9)
        # Coarse sanity: Oahu
        assert 21.0 < lat < 22.0
        assert -158.5 < lon < -157.0

    def test_sydney_southern_hemisphere(self):
        # 56HLH 34786 52080 -> UTM 56S (334786, 6252080), Sydney area
        lat, lon = _parse_igeolo_corner('56HLH3478652080', 'U')
        exp_lat, exp_lon = _utm_expected(32756, 334786, 6252080)
        assert lat == pytest.approx(exp_lat, abs=1e-9)
        assert lon == pytest.approx(exp_lon, abs=1e-9)
        # Coarse sanity: Sydney
        assert -34.5 < lat < -33.0
        assert 150.5 < lon < 152.0

    def test_illegal_column_letter_for_zone_raises(self):
        # Zone 31 uses column set A-H; 'J' is illegal there
        with pytest.raises(ValueError):
            _parse_igeolo_corner('31NJA6602100000', 'U')

    def test_malformed_mgrs_raises(self):
        with pytest.raises(ValueError):
            _parse_igeolo_corner('31NAI6602100000', 'U')  # 'I' excluded

    def test_full_igeolo_u_from_reader(self):
        # Four MGRS corners forming a small square near Sydney
        utm = [
            (330000, 6260000),  # UL -> 56HLH3000060000
            (340000, 6260000),  # UR
            (340000, 6250000),  # LR
            (330000, 6250000),  # LL
        ]
        # Build MGRS strings by hand: zone 56 set 1 -> 'L' = 300 km;
        # rows: 6260000 mod 2e6 = 260000 -> idx 2 (+5 even offset) ->
        # 'H'; 6250000 likewise 'H'.
        igeolo = (
            '56HLH3000060000'
            '56HLH4000060000'
            '56HLH4000050000'
            '56HLH3000050000'
        )
        meta = SimpleNamespace(igeolo=igeolo, icords='U')
        geo = CornerGeolocation.from_reader(_StubReader(meta))
        assert geo.accuracy_source == 'IGEOLO'
        expected = np.array([_utm_expected(32756, e, n) for e, n in utm])
        np.testing.assert_allclose(geo.corners, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# from_reader priority order
# ---------------------------------------------------------------------------

_IGEOLO_G = (
    '390000N0771200W'
    '390000N0770000W'
    '385400N0770000W'
    '385400N0771200W'
)

_BLOCKA_OK = BLOCKAMetadata(
    block_number=1,
    frfc_loc='+39.100000-077.300000',
    frlc_loc='+39.100000-077.100000',
    lrfc_loc='+39.000000-077.300000',
    lrlc_loc='+39.000000-077.100000',
)


class TestSourcePriority:
    """CSCRNA -> BLOCKA -> IGEOLO priority in from_reader."""

    def test_cscrna_wins(self):
        cscrna = SimpleNamespace(corners=CORNERS)
        meta = SimpleNamespace(
            cscrna=cscrna, blocka=_BLOCKA_OK,
            igeolo=_IGEOLO_G, icords='G',
        )
        geo = CornerGeolocation.from_reader(_StubReader(meta))
        assert geo.accuracy_source == 'CSCRNA'
        np.testing.assert_allclose(geo.corners, CORNERS)

    def test_blocka_beats_igeolo(self):
        meta = SimpleNamespace(
            blocka=_BLOCKA_OK, igeolo=_IGEOLO_G, icords='G')
        geo = CornerGeolocation.from_reader(_StubReader(meta))
        assert geo.accuracy_source == 'BLOCKA'
        assert geo.corners[0, 0] == pytest.approx(39.1)

    def test_igeolo_fallback(self):
        meta = SimpleNamespace(igeolo=_IGEOLO_G, icords='G')
        geo = CornerGeolocation.from_reader(_StubReader(meta))
        assert geo.accuracy_source == 'IGEOLO'
        assert geo.corners[0, 0] == pytest.approx(39.0)

    def test_malformed_blocka_falls_through_to_igeolo(self):
        bad_blocka = BLOCKAMetadata(
            block_number=1,
            frfc_loc='garbage but 21 chars',
            frlc_loc='garbage but 21 chars',
            lrfc_loc='garbage but 21 chars',
            lrlc_loc='garbage but 21 chars',
        )
        meta = SimpleNamespace(
            blocka=bad_blocka, igeolo=_IGEOLO_G, icords='G')
        geo = CornerGeolocation.from_reader(_StubReader(meta))
        assert geo.accuracy_source == 'IGEOLO'

    def test_incomplete_blocka_falls_through(self):
        partial = BLOCKAMetadata(
            block_number=1, frfc_loc='+39.000000-077.200000')
        meta = SimpleNamespace(
            blocka=partial, igeolo=_IGEOLO_G, icords='G')
        geo = CornerGeolocation.from_reader(_StubReader(meta))
        assert geo.accuracy_source == 'IGEOLO'

    def test_no_source_raises(self):
        meta = SimpleNamespace()
        with pytest.raises(ValueError):
            CornerGeolocation.from_reader(_StubReader(meta))

    def test_shape_from_reader(self):
        meta = SimpleNamespace(igeolo=_IGEOLO_G, icords='G')
        geo = CornerGeolocation.from_reader(
            _StubReader(meta, shape=(512, 768)))
        assert geo.shape == (512, 768)


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------

class TestFactory:
    """create_geolocation returns CornerGeolocation for corner-only NITF."""

    def test_factory_returns_corner_geolocation(self):
        meta = EONITFMetadata(
            format='NITF', rows=1000, cols=2000, dtype='uint16',
            bands=1, icords='G', igeolo=_IGEOLO_G,
        )
        geo = create_geolocation(_StubReader(meta))
        assert isinstance(geo, CornerGeolocation)
        assert geo.accuracy_source == 'IGEOLO'
        lat, lon, _ = geo.image_to_latlon(0, 0)
        assert lat == pytest.approx(39.0)
        assert lon == pytest.approx(-77.2)

    def test_factory_blocka_priority(self):
        meta = EONITFMetadata(
            format='NITF', rows=1000, cols=2000, dtype='uint16',
            bands=1, blocka=_BLOCKA_OK, icords='G', igeolo=_IGEOLO_G,
        )
        geo = create_geolocation(_StubReader(meta))
        assert isinstance(geo, CornerGeolocation)
        assert geo.accuracy_source == 'BLOCKA'

    def test_factory_no_corners_still_raises(self):
        meta = EONITFMetadata(
            format='NITF', rows=1000, cols=2000, dtype='uint16', bands=1,
        )
        with pytest.raises(TypeError):
            create_geolocation(_StubReader(meta))
