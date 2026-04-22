# -*- coding: utf-8 -*-
"""
Tests for grdl.geolocation.eo.rpc — RPC geolocation.

Uses realistic synthetic RPC coefficients to verify forward/inverse
projection round-trips and monomial evaluation.

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
from grdl.IO.models.eo_nitf import RPCCoefficients
from grdl.geolocation.eo.rpc import RPCGeolocation, _rpc_monomials, _rpc_evaluate


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def simple_rpc():
    """Simple RPC with identity-like coefficients for testing.

    Constructs an RPC where line = lat (approximately) and
    samp = lon (approximately) via simple linear numerators.
    """
    # Normalization centered on (34.0, -118.0, 500 m)
    line_num = np.zeros(20)
    line_num[0] = 0.0   # constant
    line_num[2] = 1.0   # P (normalized latitude) coefficient

    line_den = np.zeros(20)
    line_den[0] = 1.0   # constant denominator

    samp_num = np.zeros(20)
    samp_num[0] = 0.0
    samp_num[1] = 1.0   # L (normalized longitude) coefficient

    samp_den = np.zeros(20)
    samp_den[0] = 1.0

    return RPCCoefficients(
        line_off=2048.0,
        samp_off=2048.0,
        lat_off=34.0,
        long_off=-118.0,
        height_off=500.0,
        line_scale=2048.0,
        samp_scale=2048.0,
        lat_scale=1.0,
        long_scale=1.0,
        height_scale=500.0,
        line_num_coef=line_num,
        line_den_coef=line_den,
        samp_num_coef=samp_num,
        samp_den_coef=samp_den,
    )


@pytest.fixture
def realistic_rpc():
    """More realistic RPC with cross-terms for round-trip testing."""
    np.random.seed(42)
    line_num = np.zeros(20)
    line_num[0] = 0.001
    line_num[1] = 0.02    # L
    line_num[2] = 0.98    # P (dominant)
    line_num[3] = -0.001  # H
    line_num[4] = 0.003   # LP

    line_den = np.zeros(20)
    line_den[0] = 1.0
    line_den[2] = 0.0001  # small P term

    samp_num = np.zeros(20)
    samp_num[0] = -0.002
    samp_num[1] = 0.97    # L (dominant)
    samp_num[2] = 0.03    # P
    samp_num[3] = 0.001   # H
    samp_num[4] = -0.002  # LP

    samp_den = np.zeros(20)
    samp_den[0] = 1.0
    samp_den[1] = 0.00005  # small L term

    return RPCCoefficients(
        line_off=5000.0,
        samp_off=5000.0,
        lat_off=38.0,
        long_off=-77.0,
        height_off=100.0,
        line_scale=5000.0,
        samp_scale=5000.0,
        lat_scale=0.5,
        long_scale=0.5,
        height_scale=500.0,
        line_num_coef=line_num,
        line_den_coef=line_den,
        samp_num_coef=samp_num,
        samp_den_coef=samp_den,
    )


# ── Monomial vector ─────────────────────────────────────────────────


class TestRPCMonomials:
    """Tests for the 20-term monomial vector."""

    def test_shape(self):
        p = np.array([0.0, 0.5])
        l = np.array([0.0, 0.3])
        h = np.array([0.0, 0.1])
        mono = _rpc_monomials(p, l, h)
        assert mono.shape == (2, 20)

    def test_origin(self):
        """At (0,0,0), only the constant term is 1."""
        mono = _rpc_monomials(
            np.array([0.0]), np.array([0.0]), np.array([0.0]))
        assert mono[0, 0] == 1.0
        np.testing.assert_allclose(mono[0, 1:], 0.0)

    def test_known_values(self):
        """Verify specific monomial terms."""
        p = np.array([2.0])
        l = np.array([3.0])
        h = np.array([5.0])
        mono = _rpc_monomials(p, l, h)
        assert mono[0, 0] == 1.0       # 1
        assert mono[0, 1] == 3.0       # L
        assert mono[0, 2] == 2.0       # P
        assert mono[0, 3] == 5.0       # H
        assert mono[0, 4] == 6.0       # L*P
        assert mono[0, 7] == 9.0       # L²
        assert mono[0, 10] == 30.0     # P*L*H
        assert mono[0, 19] == 125.0    # H³


# ── Direct evaluation (ground → image) ──────────────────────────────


class TestRPCEvaluate:
    """Tests for direct RPC polynomial evaluation."""

    def test_center_maps_to_offset(self, simple_rpc):
        """RPC center (lat_off, long_off, h_off) → (line_off, samp_off)."""
        rows, cols = _rpc_evaluate(
            np.array([34.0]), np.array([-118.0]), np.array([500.0]),
            simple_rpc,
        )
        assert rows[0] == pytest.approx(2048.0, abs=0.01)
        assert cols[0] == pytest.approx(2048.0, abs=0.01)

    def test_vectorized(self, simple_rpc):
        """Multiple points evaluated at once."""
        lats = np.array([33.5, 34.0, 34.5])
        lons = np.array([-118.5, -118.0, -117.5])
        heights = np.full(3, 500.0)
        rows, cols = _rpc_evaluate(lats, lons, heights, simple_rpc)
        assert rows.shape == (3,)
        assert cols.shape == (3,)


# ── RPCGeolocation class ────────────────────────────────────────────


class TestRPCGeolocation:
    """Tests for RPCGeolocation forward and inverse projection."""

    def test_inverse_center(self, simple_rpc):
        """latlon_to_image at center should return offset pixels."""
        geo = RPCGeolocation(simple_rpc, shape=(4096, 4096))
        row, col = geo.latlon_to_image(34.0, -118.0, 500.0)
        assert row == pytest.approx(2048.0, abs=0.1)
        assert col == pytest.approx(2048.0, abs=0.1)

    def test_forward_center(self, simple_rpc):
        """image_to_latlon at offset pixel should return center coords."""
        geo = RPCGeolocation(simple_rpc, shape=(4096, 4096))
        lat, lon, h = geo.image_to_latlon(2048.0, 2048.0, 500.0)
        assert lat == pytest.approx(34.0, abs=0.01)
        assert lon == pytest.approx(-118.0, abs=0.01)

    def test_round_trip_inverse_then_forward(self, realistic_rpc):
        """latlon_to_image → image_to_latlon should round-trip."""
        geo = RPCGeolocation(realistic_rpc, shape=(10000, 10000))
        lat0, lon0, h0 = 38.1, -76.9, 200.0

        # Ground → image
        row, col = geo.latlon_to_image(lat0, lon0, h0)

        # Image → ground
        lat1, lon1, h1 = geo.image_to_latlon(row, col, h0)

        assert lat1 == pytest.approx(lat0, abs=1e-6)
        assert lon1 == pytest.approx(lon0, abs=1e-6)

    def test_round_trip_forward_then_inverse(self, realistic_rpc):
        """image_to_latlon → latlon_to_image should round-trip."""
        geo = RPCGeolocation(realistic_rpc, shape=(10000, 10000))
        row0, col0 = 5000.0, 5000.0
        h0 = 100.0

        lat, lon, h = geo.image_to_latlon(row0, col0, h0)
        row1, col1 = geo.latlon_to_image(lat, lon, h0)

        assert row1 == pytest.approx(row0, abs=0.01)
        assert col1 == pytest.approx(col0, abs=0.01)

    def test_array_round_trip(self, realistic_rpc):
        """Array input round-trips correctly."""
        geo = RPCGeolocation(realistic_rpc, shape=(10000, 10000))
        lats = np.array([37.8, 38.0, 38.2])
        lons = np.array([-77.2, -77.0, -76.8])
        heights = np.full(3, 100.0)

        inv_result = geo.latlon_to_image(
            np.column_stack([lats, lons, heights]))
        assert inv_result.shape == (3, 2)
        rows, cols = inv_result[:, 0], inv_result[:, 1]

        fwd_result = geo.image_to_latlon(
            np.column_stack([rows, cols]), height=100.0)
        assert fwd_result.shape == (3, 3)
        lats2, lons2 = fwd_result[:, 0], fwd_result[:, 1]

        np.testing.assert_allclose(lats2, lats, atol=1e-5)
        np.testing.assert_allclose(lons2, lons, atol=1e-5)

    def test_no_rpc_raises(self):
        """Constructor with None RPC should raise."""
        with pytest.raises(ValueError, match="required"):
            RPCGeolocation(None, shape=(100, 100))

    def test_height_changes_position(self, realistic_rpc):
        """Different HAE should produce different ground coords."""
        geo = RPCGeolocation(realistic_rpc, shape=(10000, 10000))
        lat0, lon0, _ = geo.image_to_latlon(5000, 5000, 0.0)
        lat1, lon1, _ = geo.image_to_latlon(5000, 5000, 1000.0)
        assert lat0 != lat1 or lon0 != lon1


# ── ICHIPB Integration tests ───────────────────────────────────────


class TestRPCICHIPBIntegration:
    """Tests for ICHIPB chip transform integration with RPC."""

    def test_ichipb_offset_applied(self, simple_rpc):
        """ICHIPB offset shifts pixel coordinates."""
        from grdl.IO.models.eo_nitf import ICHIPBMetadata

        ichipb = ICHIPBMetadata(
            xfrm_flag=2,
            fi_row_off=500.0,
            fi_col_off=300.0,
            fi_row_scale=1.0,
            fi_col_scale=1.0,
        )
        geo_no_chip = RPCGeolocation(simple_rpc, shape=(10000, 10000))
        geo_with_chip = RPCGeolocation(
            simple_rpc, ichipb=ichipb, shape=(10000, 10000))

        lat, lon = simple_rpc.lat_off, simple_rpc.long_off
        row_no, col_no = geo_no_chip.latlon_to_image(lat, lon, 0.0)
        row_ch, col_ch = geo_with_chip.latlon_to_image(lat, lon, 0.0)

        # Chip coords should be shifted
        assert row_no - row_ch == pytest.approx(500.0, abs=0.1)
        assert col_no - col_ch == pytest.approx(300.0, abs=0.1)

    def test_ichipb_round_trip(self, realistic_rpc):
        """Round-trip with ICHIPB offset preserves pixel accuracy."""
        from grdl.IO.models.eo_nitf import ICHIPBMetadata

        ichipb = ICHIPBMetadata(
            xfrm_flag=2,
            fi_row_off=200.0,
            fi_col_off=100.0,
            fi_row_scale=1.0,
            fi_col_scale=1.0,
        )
        geo = RPCGeolocation(
            realistic_rpc, ichipb=ichipb, shape=(10000, 10000))

        chip_row, chip_col = 4000.0, 4000.0
        lat, lon, h = geo.image_to_latlon(chip_row, chip_col, 100.0)
        row_rt, col_rt = geo.latlon_to_image(lat, lon, 100.0)

        assert row_rt == pytest.approx(chip_row, abs=0.01)
        assert col_rt == pytest.approx(chip_col, abs=0.01)
