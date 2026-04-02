# -*- coding: utf-8 -*-
"""
Tests for grdl.geolocation.eo.rsm — RSM geolocation.

Uses synthetic RSM coefficients to verify forward/inverse projection
round-trips with geodetic ground domain.  Monomial ordering and
coordinate conventions follow STDI-0002 Vol 1 Appendix U.

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-03-17

Modified
--------
2026-04-01
"""

# Third-party
import numpy as np
import pytest

# GRDL
from grdl.IO.models.eo_nitf import (
    ICHIPBMetadata,
    RSMCoefficients,
    RSMIdentification,
    RSMSegmentGrid,
)
from grdl.IO.models.common import XYZ
from grdl.geolocation.eo.rsm import (
    RSMGeolocation,
    _build_monomial_exponents,
    _rsm_monomials,
    _rsm_evaluate,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def linear_rsm():
    """Simple linear RSM (order 1 in each axis) for testing.

    Monomial ordering per RSMPCA spec (x varies fastest):
      k=0, j=0: i=0,1 → (0,0,0), (1,0,0)  [indices 0, 1]
      k=0, j=1: i=0,1 → (0,1,0), (1,1,0)  [indices 2, 3]
      k=1, j=0: i=0,1 → (0,0,1), (1,0,1)  [indices 4, 5]
      k=1, j=1: i=0,1 → (0,1,1), (1,1,1)  [indices 6, 7]

    For Geodetic (GRNDD='G'): x=longitude, y=latitude, z=height.
    Row dominated by y (latitude), col dominated by x (longitude).
    Normalization offsets in radians per RSMIDA spec.
    """
    # 8 terms for max_powers [1, 1, 1]
    row_num = np.zeros(8)
    row_num[0] = 0.0   # constant
    row_num[2] = 1.0   # y (latitude) dominant — index 2 = (0,1,0)

    row_den = np.zeros(8)
    row_den[0] = 1.0   # constant denominator

    col_num = np.zeros(8)
    col_num[0] = 0.0
    col_num[1] = 1.0   # x (longitude) dominant — index 1 = (1,0,0)

    col_den = np.zeros(8)
    col_den[0] = 1.0

    return RSMCoefficients(
        row_off=2048.0,
        col_off=2048.0,
        row_norm_sf=2048.0,
        col_norm_sf=2048.0,
        # Normalization offsets in radians (per RSMIDA spec for Geodetic)
        x_off=np.deg2rad(-77.0),    # longitude center
        y_off=np.deg2rad(38.0),     # latitude center
        z_off=100.0,
        x_norm_sf=np.deg2rad(0.5),  # longitude half-range
        y_norm_sf=np.deg2rad(0.5),  # latitude half-range
        z_norm_sf=500.0,
        row_num_powers=np.array([1, 1, 1]),
        row_den_powers=np.array([1, 1, 1]),
        col_num_powers=np.array([1, 1, 1]),
        col_den_powers=np.array([1, 1, 1]),
        row_num_coefs=row_num,
        row_den_coefs=row_den,
        col_num_coefs=col_num,
        col_den_coefs=col_den,
    )


@pytest.fixture
def quadratic_rsm():
    """Quadratic RSM with cross-terms for round-trip testing.

    Monomial ordering for max_powers [2,2,1] (x varies fastest):
      k=0, j=0: i=0,1,2 → (0,0,0), (1,0,0), (2,0,0)   [0,1,2]
      k=0, j=1: i=0,1,2 → (0,1,0), (1,1,0), (2,1,0)   [3,4,5]
      k=0, j=2: i=0,1,2 → (0,2,0), (1,2,0), (2,2,0)   [6,7,8]
      k=1, j=0: i=0,1,2 → (0,0,1), (1,0,1), (2,0,1)   [9,10,11]
      k=1, j=1: i=0,1,2 → (0,1,1), (1,1,1), (2,1,1)   [12,13,14]
      k=1, j=2: i=0,1,2 → (0,2,1), (1,2,1), (2,2,1)   [15,16,17]
    """
    # 18 terms for max_powers [2, 2, 1]
    row_num = np.zeros(18)
    row_num[0] = 0.001    # constant (0,0,0)
    row_num[3] = 0.95     # y (latitude)  (0,1,0) — dominant
    row_num[1] = 0.03     # x (longitude) (1,0,0)
    row_num[9] = -0.001   # z (height)    (0,0,1)
    row_num[6] = 0.005    # y²            (0,2,0)

    row_den = np.zeros(18)
    row_den[0] = 1.0

    col_num = np.zeros(18)
    col_num[0] = -0.001
    col_num[3] = 0.04     # y (latitude)  (0,1,0)
    col_num[1] = 0.94     # x (longitude) (1,0,0) — dominant
    col_num[9] = 0.001    # z (height)    (0,0,1)

    col_den = np.zeros(18)
    col_den[0] = 1.0

    return RSMCoefficients(
        row_off=5000.0,
        col_off=5000.0,
        row_norm_sf=5000.0,
        col_norm_sf=5000.0,
        x_off=np.deg2rad(-77.0),
        y_off=np.deg2rad(38.0),
        z_off=200.0,
        x_norm_sf=np.deg2rad(0.5),
        y_norm_sf=np.deg2rad(0.5),
        z_norm_sf=500.0,
        row_num_powers=np.array([2, 2, 1]),
        row_den_powers=np.array([2, 2, 1]),
        col_num_powers=np.array([2, 2, 1]),
        col_den_powers=np.array([2, 2, 1]),
        row_num_coefs=row_num,
        row_den_coefs=row_den,
        col_num_coefs=col_num,
        col_den_coefs=col_den,
    )


# ── Monomial exponents ──────────────────────────────────────────────


class TestMonomialExponents:
    """Tests for RSM monomial exponent generation."""

    def test_order_111(self):
        exp = _build_monomial_exponents(np.array([1, 1, 1]))
        assert exp.shape == (8, 3)  # 2×2×2 = 8 terms

    def test_order_221(self):
        exp = _build_monomial_exponents(np.array([2, 2, 1]))
        assert exp.shape == (18, 3)  # 3×3×2 = 18 terms

    def test_order_000(self):
        exp = _build_monomial_exponents(np.array([0, 0, 0]))
        assert exp.shape == (1, 3)
        np.testing.assert_array_equal(exp[0], [0, 0, 0])

    def test_x_varies_fastest(self):
        """Verify coefficient ordering: x varies fastest per RSMPCA spec.

        For max_powers [2,1,1], the ordering should be:
        a_000, a_100, a_200, a_010, a_110, a_210,
        a_001, a_101, a_201, a_011, a_111, a_211
        """
        exp = _build_monomial_exponents(np.array([2, 1, 1]))
        # First three entries: x varies 0→1→2 with j=0, k=0
        np.testing.assert_array_equal(exp[0], [0, 0, 0])
        np.testing.assert_array_equal(exp[1], [1, 0, 0])
        np.testing.assert_array_equal(exp[2], [2, 0, 0])
        # Next three: x varies 0→1→2 with j=1, k=0
        np.testing.assert_array_equal(exp[3], [0, 1, 0])
        np.testing.assert_array_equal(exp[4], [1, 1, 0])
        np.testing.assert_array_equal(exp[5], [2, 1, 0])
        # Next three: x varies 0→1→2 with j=0, k=1
        np.testing.assert_array_equal(exp[6], [0, 0, 1])


# ── RSM evaluation ──────────────────────────────────────────────────


class TestRSMEvaluate:
    """Tests for direct RSM polynomial evaluation."""

    def test_center_maps_to_offset(self, linear_rsm):
        """RSM center coords → offset pixels."""
        rows, cols = _rsm_evaluate(
            np.array([38.0]), np.array([-77.0]), np.array([100.0]),
            linear_rsm, ground_domain_type='G',
        )
        assert rows[0] == pytest.approx(2048.0, abs=0.1)
        assert cols[0] == pytest.approx(2048.0, abs=0.1)

    def test_vectorized(self, linear_rsm):
        lats = np.array([37.5, 38.0, 38.5])
        lons = np.array([-77.5, -77.0, -76.5])
        heights = np.full(3, 100.0)
        rows, cols = _rsm_evaluate(
            lats, lons, heights, linear_rsm, 'G')
        assert rows.shape == (3,)


# ── RSMGeolocation class ────────────────────────────────────────────


class TestRSMGeolocation:
    """Tests for RSMGeolocation forward and inverse projection."""

    def test_inverse_center(self, linear_rsm):
        geo = RSMGeolocation(linear_rsm, shape=(4096, 4096))
        row, col = geo.latlon_to_image(38.0, -77.0, 100.0)
        assert row == pytest.approx(2048.0, abs=0.1)
        assert col == pytest.approx(2048.0, abs=0.1)

    def test_forward_center(self, linear_rsm):
        geo = RSMGeolocation(linear_rsm, shape=(4096, 4096))
        lat, lon, h = geo.image_to_latlon(2048.0, 2048.0, 100.0)
        assert lat == pytest.approx(38.0, abs=0.01)
        assert lon == pytest.approx(-77.0, abs=0.01)

    def test_round_trip(self, quadratic_rsm):
        """Ground → image → ground round-trip."""
        geo = RSMGeolocation(quadratic_rsm, shape=(10000, 10000))
        lat0, lon0 = 38.1, -76.9

        row, col = geo.latlon_to_image(lat0, lon0, 200.0)
        lat1, lon1, _ = geo.image_to_latlon(row, col, 200.0)

        assert lat1 == pytest.approx(lat0, abs=1e-5)
        assert lon1 == pytest.approx(lon0, abs=1e-5)

    def test_array_round_trip(self, quadratic_rsm):
        geo = RSMGeolocation(quadratic_rsm, shape=(10000, 10000))
        lats = np.array([37.8, 38.0, 38.2])
        lons = np.array([-77.2, -77.0, -76.8])
        heights = np.full(3, 200.0)

        inv_result = geo.latlon_to_image(
            np.column_stack([lats, lons, heights]))
        assert inv_result.shape == (3, 2)
        rows, cols = inv_result[:, 0], inv_result[:, 1]

        fwd_result = geo.image_to_latlon(
            np.column_stack([rows, cols]), height=200.0)
        assert fwd_result.shape == (3, 3)
        lats2, lons2 = fwd_result[:, 0], fwd_result[:, 1]

        np.testing.assert_allclose(lats2, lats, atol=1e-4)
        np.testing.assert_allclose(lons2, lons, atol=1e-4)

    def test_no_rsm_raises(self):
        with pytest.raises(ValueError, match="required"):
            RSMGeolocation(None, shape=(100, 100))

    def test_with_rsm_id(self, linear_rsm):
        rsm_id = RSMIdentification(
            image_id='TEST_IMAGE',
            ground_domain_type='G',
            ground_ref_point=XYZ(
                np.deg2rad(-77.0), np.deg2rad(38.0), 100.0),
        )
        geo = RSMGeolocation(linear_rsm, rsm_id=rsm_id, shape=(4096, 4096))
        assert geo._ground_domain == 'G'

    def test_lat_lon_sensitivity(self, linear_rsm):
        """Row tracks latitude, column tracks longitude."""
        geo = RSMGeolocation(linear_rsm, shape=(4096, 4096))
        # Vary latitude, hold longitude fixed
        row_a, col_a = geo.latlon_to_image(37.8, -77.0, 100.0)
        row_b, col_b = geo.latlon_to_image(38.2, -77.0, 100.0)
        # Row should change significantly
        assert abs(row_b - row_a) > 100
        # Col should stay approximately constant
        assert abs(col_b - col_a) < 10

        # Vary longitude, hold latitude fixed
        row_c, col_c = geo.latlon_to_image(38.0, -77.2, 100.0)
        row_d, col_d = geo.latlon_to_image(38.0, -76.8, 100.0)
        # Col should change significantly
        assert abs(col_d - col_c) > 100
        # Row should stay approximately constant
        assert abs(row_d - row_c) < 10


# ── Multi-segment RSM tests ────────────────────────────────────────


class TestMultiSegmentRSM:
    """Tests for multi-segment RSM geolocation."""

    @staticmethod
    def _make_segment(row_off, col_off, row_sf=1024.0, col_sf=1024.0):
        """Create a simple linear RSM segment centered at (row_off, col_off)."""
        row_num = np.zeros(8)
        row_num[2] = 1.0   # y dominant
        row_den = np.zeros(8)
        row_den[0] = 1.0
        col_num = np.zeros(8)
        col_num[1] = 1.0   # x dominant
        col_den = np.zeros(8)
        col_den[0] = 1.0
        return RSMCoefficients(
            row_off=row_off, col_off=col_off,
            row_norm_sf=row_sf, col_norm_sf=col_sf,
            x_off=np.deg2rad(-77.0), y_off=np.deg2rad(38.0),
            z_off=100.0,
            x_norm_sf=np.deg2rad(0.5), y_norm_sf=np.deg2rad(0.5),
            z_norm_sf=500.0,
            row_num_powers=np.array([1, 1, 1]),
            row_den_powers=np.array([1, 1, 1]),
            col_num_powers=np.array([1, 1, 1]),
            col_den_powers=np.array([1, 1, 1]),
            row_num_coefs=row_num, row_den_coefs=row_den,
            col_num_coefs=col_num, col_den_coefs=col_den,
            rsn=1, csn=1,
        )

    def test_single_segment_backward_compat(self):
        """Single segment with no grid behaves identically to before."""
        seg = self._make_segment(2048.0, 2048.0)
        geo = RSMGeolocation(seg, shape=(4096, 4096))
        lat, lon, h = geo.image_to_latlon(2048, 2048)
        assert lat == pytest.approx(38.0, abs=0.5)
        assert lon == pytest.approx(-77.0, abs=0.5)

    def test_multi_segment_construction(self):
        """Multi-segment RSMGeolocation stores segment bounds."""
        seg1 = self._make_segment(1024.0, 1024.0)
        seg1.rsn = 1
        seg1.csn = 1
        seg2 = self._make_segment(1024.0, 3072.0)
        seg2.rsn = 1
        seg2.csn = 2
        seg3 = self._make_segment(3072.0, 1024.0)
        seg3.rsn = 2
        seg3.csn = 1
        seg4 = self._make_segment(3072.0, 3072.0)
        seg4.rsn = 2
        seg4.csn = 2

        grid = RSMSegmentGrid(
            num_row_sections=2,
            num_col_sections=2,
            segments={
                (1, 1): seg1, (1, 2): seg2,
                (2, 1): seg3, (2, 2): seg4,
            },
        )
        geo = RSMGeolocation(
            seg1, rsm_segments=grid, shape=(4096, 4096))
        assert geo._segments is not None
        assert len(geo._segment_bounds) == 4

    def test_multi_segment_selects_correct_segment(self):
        """Segment selection picks the segment containing the pixel."""
        seg1 = self._make_segment(1024.0, 1024.0)
        seg1.rsn = 1
        seg1.csn = 1
        seg2 = self._make_segment(3072.0, 3072.0)
        seg2.rsn = 2
        seg2.csn = 2

        grid = RSMSegmentGrid(
            num_row_sections=2,
            num_col_sections=2,
            segments={(1, 1): seg1, (2, 2): seg2},
        )
        geo = RSMGeolocation(
            seg1, rsm_segments=grid, shape=(4096, 4096))

        # Pixel near (500, 500) should use segment (1,1)
        assignments = geo._select_segments_for_pixels(
            np.array([500.0]), np.array([500.0]))
        assert assignments[0] == (1, 1)

        # Pixel near (3500, 3500) should use segment (2,2)
        assignments = geo._select_segments_for_pixels(
            np.array([3500.0]), np.array([3500.0]))
        assert assignments[0] == (2, 2)


# ── ICHIPB Integration tests ───────────────────────────────────────


class TestICHIPBIntegration:
    """Tests for ICHIPB chip transform integration with RSM."""

    def test_ichipb_round_trip(self, linear_rsm):
        """ICHIPB offset is correctly applied and inverted."""
        ichipb = ICHIPBMetadata(
            xfrm_flag=2,
            fi_row_off=1000.0,
            fi_col_off=500.0,
            fi_row_scale=1.0,
            fi_col_scale=1.0,
        )
        geo_no_chip = RSMGeolocation(linear_rsm, shape=(4096, 4096))
        geo_with_chip = RSMGeolocation(
            linear_rsm, ichipb=ichipb, shape=(4096, 4096))

        # Forward: same ground point should give different chip coords
        lat, lon = 38.0, -77.0
        row_no, col_no = geo_no_chip.latlon_to_image(lat, lon, 100.0)
        row_ch, col_ch = geo_with_chip.latlon_to_image(lat, lon, 100.0)

        # Chip coords should be offset by (1000, 500)
        assert row_no - row_ch == pytest.approx(1000.0, abs=0.1)
        assert col_no - col_ch == pytest.approx(500.0, abs=0.1)

        # Inverse round-trip: chip pixel -> ground -> chip pixel
        lat_rt, lon_rt, _ = geo_with_chip.image_to_latlon(row_ch, col_ch)
        row_rt, col_rt = geo_with_chip.latlon_to_image(
            lat_rt, lon_rt, 100.0)
        assert row_rt == pytest.approx(float(row_ch), abs=0.01)
        assert col_rt == pytest.approx(float(col_ch), abs=0.01)
