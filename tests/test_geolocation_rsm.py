# -*- coding: utf-8 -*-
"""
Tests for grdl.geolocation.eo.rsm — RSM geolocation.

Uses synthetic RSM coefficients to verify forward/inverse projection
round-trips with geodetic ground domain.

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
2026-03-17
"""

# Third-party
import numpy as np
import pytest

# GRDL
from grdl.IO.models.eo_nitf import RSMCoefficients, RSMIdentification
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

    row ≈ lat, col ≈ lon in normalized coordinates.
    """
    # Max powers [1, 1, 1] → 8 terms
    # Ordering: (0,0,0)=1, (0,0,1)=z, (0,1,0)=y, (0,1,1)=yz,
    #           (1,0,0)=x, (1,0,1)=xz, (1,1,0)=xy, (1,1,1)=xyz
    row_num = np.zeros(8)
    row_num[0] = 0.0   # constant
    row_num[4] = 1.0   # x (lat) dominant — index 4 = (1,0,0)

    row_den = np.zeros(8)
    row_den[0] = 1.0

    col_num = np.zeros(8)
    col_num[0] = 0.0
    col_num[2] = 1.0   # y (lon) dominant — index 2 = (0,1,0)

    col_den = np.zeros(8)
    col_den[0] = 1.0

    return RSMCoefficients(
        row_off=2048.0,
        col_off=2048.0,
        row_norm_sf=2048.0,
        col_norm_sf=2048.0,
        x_off=38.0,
        y_off=-77.0,
        z_off=100.0,
        x_norm_sf=0.5,
        y_norm_sf=0.5,
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

    Monomial ordering for max_powers [2,2,1] is:
    idx 0: (0,0,0)=1, idx 1: (0,0,1)=z, idx 2: (0,1,0)=y,
    idx 6: (1,0,0)=x, idx 12: (2,0,0)=x².
    """
    # Max powers [2, 2, 1] → 18 terms
    row_num = np.zeros(18)
    row_num[0] = 0.001   # constant
    row_num[6] = 0.95    # x (lat)
    row_num[2] = 0.03    # y (lon)
    row_num[1] = -0.001  # z (height)
    row_num[12] = 0.005  # x²

    row_den = np.zeros(18)
    row_den[0] = 1.0

    col_num = np.zeros(18)
    col_num[0] = -0.001
    col_num[6] = 0.04    # x (lat)
    col_num[2] = 0.94    # y (lon)
    col_num[1] = 0.001   # z (height)

    col_den = np.zeros(18)
    col_den[0] = 1.0

    return RSMCoefficients(
        row_off=5000.0,
        col_off=5000.0,
        row_norm_sf=5000.0,
        col_norm_sf=5000.0,
        x_off=38.0,
        y_off=-77.0,
        z_off=200.0,
        x_norm_sf=0.5,
        y_norm_sf=0.5,
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

        rows, cols = geo.latlon_to_image(lats, lons, 200.0)
        lats2, lons2, _ = geo.image_to_latlon(rows, cols, 200.0)

        np.testing.assert_allclose(lats2, lats, atol=1e-4)
        np.testing.assert_allclose(lons2, lons, atol=1e-4)

    def test_no_rsm_raises(self):
        with pytest.raises(ValueError, match="required"):
            RSMGeolocation(None, shape=(100, 100))

    def test_with_rsm_id(self, linear_rsm):
        rsm_id = RSMIdentification(
            image_id='TEST_IMAGE',
            ground_domain_type='G',
            ground_ref_point=XYZ(38.0, -77.0, 100.0),
        )
        geo = RSMGeolocation(linear_rsm, rsm_id=rsm_id, shape=(4096, 4096))
        assert geo._ground_domain == 'G'
