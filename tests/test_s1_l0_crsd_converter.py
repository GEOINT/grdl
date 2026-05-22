# -*- coding: utf-8 -*-
"""
Unit tests for the Sentinel-1 L0 → CRSD converter.

Tests cover the three core modules:
  - crsd_pvp_builder (quantization, PVP/PPP array construction)
  - crsd_metadata_builder (ECEF↔geodetic, reference geometry, XML build)
  - crsd_converter (channel IDs, swath mapping, time helpers)

All tests use synthetic data — no real imagery or network access required.

Author
------
James Fritz
jpfritz@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-05-22

Modified
--------
2026-05-22
"""

# Standard library
from datetime import datetime, timedelta

# Third-party
import numpy as np
import pytest

# GRDL internal — converters
from grdl.IO.sar.sentinel1_l0.crsd_converter import (
    _gps_seconds_to_utc,
    _gps_to_relative,
    _make_channel_id,
    _swath_name,
    _utc_to_relative,
)
from grdl.IO.sar.sentinel1_l0.crsd_metadata_builder import (
    BurstChannelInfo,
    CRSDMetadataBuilder,
    CRSDReferenceGeometryInfo,
    CRSDSceneInfo,
    _compute_reference_geometry,
    ecef_to_geodetic,
)
from grdl.IO.sar.sentinel1_l0.crsd_pvp_builder import (
    build_ppp_array,
    build_pvp_array,
    get_ppp_dtype,
    get_pvp_dtype,
    quantize_to_ci2,
)
from grdl.IO.sar.sentinel1_l0.constants import (
    GPS_LEAP_SECONDS,
    SPEED_OF_LIGHT,
)


# =====================================================================
# Fixtures — reusable synthetic data
# =====================================================================


def _make_channel_info(
    *,
    identifier: str = "043_000001_IW1",
    swath_name: str = "IW1",
    num_vectors: int = 10,
    num_samples: int = 100,
) -> BurstChannelInfo:
    """Minimal BurstChannelInfo for testing."""
    return BurstChannelInfo(
        identifier=identifier,
        swath_name=swath_name,
        polarization="VV",
        num_vectors=num_vectors,
        num_samples=num_samples,
        tx_time_first=5.0,
        tx_time_last=5.0 + (num_vectors - 1) * 0.000582,
        rcv_start_first=5.00058,
        rcv_start_last=5.00058 + (num_vectors - 1) * 0.000582,
        f0_ref=5.405e9,
        fs=64.345238e6,
        bw_inst=56.5e6,
        fx_freq0=5.376e9,
        fx_rate=779.038e9,
        fx_bw=56.5e6,
        tx_pulse_duration=52.0e-6,
        ref_vector_index=num_vectors // 2,
        tx_ref_pos=np.array([4.0e6, 3.0e6, 5.0e6]),
        tx_ref_vel=np.array([-1000.0, 7000.0, 500.0]),
    )


@pytest.fixture
def channel_info():
    """A basic BurstChannelInfo with 10 vectors × 100 samples."""
    return _make_channel_info()


@pytest.fixture
def leo_orbit():
    """Synthetic LEO orbit positions/velocities (N=10, ~7070 km altitude).

    Returns (positions, velocities) each of shape (10, 3).
    """
    n = 10
    # Roughly equatorial circular orbit at 700 km alt
    t = np.linspace(0, 0.01, n)
    r = 7.07e6  # metres
    omega = 7500.0 / r  # angular velocity (rad/s)
    theta = omega * t

    positions = np.column_stack([
        r * np.cos(theta),
        r * np.sin(theta),
        np.zeros(n),
    ])
    velocities = np.column_stack([
        -7500.0 * np.sin(theta),
        7500.0 * np.cos(theta),
        np.zeros(n),
    ])
    return positions, velocities


@pytest.fixture
def acf_vectors():
    """Synthetic antenna coordinate frame vectors (N=10).

    Returns (acx, acy) each of shape (10, 3).
    """
    n = 10
    acx = np.tile([0.0, 0.0, 1.0], (n, 1))
    acy = np.tile([0.0, 1.0, 0.0], (n, 1))
    return acx, acy


# =====================================================================
# Tests: crsd_converter — helper functions
# =====================================================================


class TestMakeChannelId:
    """Tests for _make_channel_id()."""

    def test_basic_format(self):
        result = _make_channel_id(43, 90466, "IW1")
        assert result == "043_090466_IW1"

    def test_zero_padded_orbit(self):
        result = _make_channel_id(1, 5, "IW3")
        assert result == "001_000005_IW3"

    def test_large_cycle(self):
        result = _make_channel_id(175, 999999, "IW2")
        assert result == "175_999999_IW2"


class TestSwathName:
    """Tests for _swath_name()."""

    def test_raw_swath_10(self):
        assert _swath_name(10) == "IW1"

    def test_raw_swath_11(self):
        assert _swath_name(11) == "IW2"

    def test_raw_swath_12(self):
        assert _swath_name(12) == "IW3"

    def test_logical_swath_1(self):
        assert _swath_name(1) == "IW1"

    def test_logical_swath_2(self):
        assert _swath_name(2) == "IW2"

    def test_logical_swath_3(self):
        assert _swath_name(3) == "IW3"

    def test_unknown_swath_fallback(self):
        assert _swath_name(99) == "SW99"


class TestTimeHelpers:
    """Tests for GPS/UTC time conversion helpers."""

    def test_gps_to_utc_known_epoch(self):
        # GPS epoch (1980-01-06) + GPS_LEAP_SECONDS offset = UTC
        gps_seconds = 0.0
        utc = _gps_seconds_to_utc(gps_seconds)
        expected = datetime(1980, 1, 6) - timedelta(seconds=GPS_LEAP_SECONDS)
        assert utc == expected

    def test_gps_to_utc_positive(self):
        gps_seconds = 86400.0  # +1 day from GPS epoch
        utc = _gps_seconds_to_utc(gps_seconds)
        expected = (
            datetime(1980, 1, 6) + timedelta(days=1)
            - timedelta(seconds=GPS_LEAP_SECONDS)
        )
        assert utc == expected

    def test_utc_to_relative(self):
        ref = datetime(2025, 12, 9, 15, 19, 25)
        utc_dt = datetime(2025, 12, 9, 15, 19, 30)
        assert _utc_to_relative(utc_dt, ref) == pytest.approx(5.0)

    def test_utc_to_relative_negative(self):
        ref = datetime(2025, 12, 9, 15, 19, 30)
        utc_dt = datetime(2025, 12, 9, 15, 19, 25)
        assert _utc_to_relative(utc_dt, ref) == pytest.approx(-5.0)

    def test_gps_to_relative_round_trip(self):
        ref_time = datetime(2025, 12, 9, 15, 19, 25)
        gps_seconds = 1449328783.0  # a realistic GPS timestamp
        utc = _gps_seconds_to_utc(gps_seconds)
        rel = _utc_to_relative(utc, ref_time)
        rel2 = _gps_to_relative(gps_seconds, ref_time)
        assert rel == pytest.approx(rel2)


# =====================================================================
# Tests: crsd_metadata_builder — geometry helpers
# =====================================================================


class TestEcefToGeodetic:
    """Tests for ecef_to_geodetic() — Bowring iterative method."""

    def test_equator_prime_meridian(self):
        """Point on equator at 0°N, 0°E, ~0m HAE."""
        x, y, z = 6378137.0, 0.0, 0.0  # WGS-84 semi-major
        lat, lon, hae = ecef_to_geodetic(x, y, z)
        assert lat == pytest.approx(0.0, abs=1e-10)
        assert lon == pytest.approx(0.0, abs=1e-10)
        assert hae == pytest.approx(0.0, abs=0.01)

    def test_north_pole(self):
        """North pole: 90°N, any lon, ~0m HAE."""
        # WGS-84 semi-minor = a * (1 - f)
        b = 6356752.314245
        lat, lon, hae = ecef_to_geodetic(0.0, 0.0, b)
        assert lat == pytest.approx(90.0, abs=1e-8)
        assert hae == pytest.approx(0.0, abs=1.0)  # Bowring ~0.3m error at pole

    def test_known_location(self):
        """Verify against a known ECEF ↔ geodetic conversion.

        Washington DC: ~38.8977°N, -77.0365°E, 0m HAE
        """
        lat_true, lon_true = 38.8977, -77.0365
        # Forward: geodetic → ECEF (approximate)
        lat_r = np.radians(lat_true)
        lon_r = np.radians(lon_true)
        a = 6378137.0
        e2 = 0.00669437999014
        n_val = a / np.sqrt(1.0 - e2 * np.sin(lat_r) ** 2)
        x = n_val * np.cos(lat_r) * np.cos(lon_r)
        y = n_val * np.cos(lat_r) * np.sin(lon_r)
        z = n_val * (1.0 - e2) * np.sin(lat_r)

        lat, lon, hae = ecef_to_geodetic(x, y, z)
        assert lat == pytest.approx(lat_true, abs=1e-6)
        assert lon == pytest.approx(lon_true, abs=1e-6)
        assert hae == pytest.approx(0.0, abs=1.0)

    def test_southern_hemisphere(self):
        """Point at -34°S, 151°E (near Sydney)."""
        lat_true, lon_true = -33.8688, 151.2093
        lat_r = np.radians(lat_true)
        lon_r = np.radians(lon_true)
        a = 6378137.0
        e2 = 0.00669437999014
        n_val = a / np.sqrt(1.0 - e2 * np.sin(lat_r) ** 2)
        x = n_val * np.cos(lat_r) * np.cos(lon_r)
        y = n_val * np.cos(lat_r) * np.sin(lon_r)
        z = n_val * (1.0 - e2) * np.sin(lat_r)

        lat, lon, hae = ecef_to_geodetic(x, y, z)
        assert lat == pytest.approx(lat_true, abs=1e-5)
        assert lon == pytest.approx(lon_true, abs=1e-5)


class TestComputeReferenceGeometry:
    """Tests for _compute_reference_geometry()."""

    def test_basic_geometry(self):
        """Satellite overhead with known geometry."""
        ref_pos = np.array([6378137.0, 0.0, 0.0])  # On equator
        plat_pos = np.array([7078137.0, 0.0, 0.0])  # 700km above
        plat_vel = np.array([0.0, 7500.0, 0.0])  # perpendicular

        result = _compute_reference_geometry(ref_pos, plat_pos, plat_vel)
        assert result["SlantRange"] == pytest.approx(700000.0, rel=1e-6)
        # Looking straight down → graze ≈ 90°, incidence ≈ 0°
        assert result["GrazeAngle"] == pytest.approx(90.0, abs=1.0)
        assert result["IncidenceAngle"] == pytest.approx(0.0, abs=1.0)
        # Velocity perpendicular to LOS → DCA ≈ 90°
        assert result["DopplerConeAngle"] == pytest.approx(90.0, abs=0.5)

    def test_side_looking(self):
        """Satellite with side-looking geometry (typical SAR)."""
        # Satellite at (0, 0, 7078137) looking at point on equator
        plat_pos = np.array([0.0, 0.0, 7078137.0])
        ref_pos = np.array([6378137.0, 0.0, 0.0])
        plat_vel = np.array([0.0, 7500.0, 0.0])

        result = _compute_reference_geometry(ref_pos, plat_pos, plat_vel)
        assert result["SlantRange"] > 0
        # This extreme geometry (90° from nadir) yields small/negative graze
        assert result["IncidenceAngle"] > 0

    def test_ground_range_positive(self):
        ref_pos = np.array([6378137.0, 0.0, 0.0])
        plat_pos = np.array([0.0, 0.0, 7078137.0])
        plat_vel = np.array([0.0, 7500.0, 0.0])

        result = _compute_reference_geometry(ref_pos, plat_pos, plat_vel)
        assert result["GroundRange"] > 0


# =====================================================================
# Tests: crsd_pvp_builder — quantize_to_ci2
# =====================================================================


class TestQuantizeToCi2:
    """Tests for quantize_to_ci2()."""

    def test_output_shape(self):
        signal = np.ones((5, 20), dtype=np.complex64) * (1 + 2j)
        ci2, amp_sf = quantize_to_ci2(signal)
        assert ci2.shape == (5, 20)
        assert amp_sf.shape == (5,)

    def test_output_dtype(self):
        signal = np.ones((3, 10), dtype=np.complex64)
        ci2, amp_sf = quantize_to_ci2(signal)
        assert ci2.dtype.names == ("real", "imag")
        assert ci2["real"].dtype == np.int8
        assert ci2["imag"].dtype == np.int8
        assert amp_sf.dtype == np.float64

    def test_zero_signal(self):
        signal = np.zeros((2, 5), dtype=np.complex64)
        ci2, amp_sf = quantize_to_ci2(signal)
        assert np.all(ci2["real"] == 0)
        assert np.all(ci2["imag"] == 0)
        assert np.all(amp_sf == 1.0)

    def test_scale_factor_reconstruction(self):
        """Verify ci2 * amp_sf ≈ original signal."""
        rng = np.random.RandomState(42)
        real = rng.randn(4, 50).astype(np.float32) * 100.0
        imag = rng.randn(4, 50).astype(np.float32) * 100.0
        signal = real + 1j * imag

        ci2, amp_sf = quantize_to_ci2(signal)
        reconstructed_real = ci2["real"].astype(np.float64) * amp_sf[:, None]
        reconstructed_imag = ci2["imag"].astype(np.float64) * amp_sf[:, None]

        # Quantization error should be small relative to signal range
        max_err_real = np.max(np.abs(reconstructed_real - real))
        max_err_imag = np.max(np.abs(reconstructed_imag - imag))
        # Max error bounded by amp_sf * 0.5 (rounding) ≈ max_val/127*0.5
        for row in range(4):
            max_val = max(np.max(np.abs(real[row])), np.max(np.abs(imag[row])))
            bound = max_val / 127.0 * 0.5 + 1e-10
            row_err = max(
                np.max(np.abs(reconstructed_real[row] - real[row])),
                np.max(np.abs(reconstructed_imag[row] - imag[row])),
            )
            assert row_err < bound * 2.0  # Allow 1 LSB tolerance

    def test_range_clipping(self):
        """Values should be clipped to [-128, 127]."""
        signal = np.ones((1, 10), dtype=np.complex64) * (127 + 127j)
        ci2, amp_sf = quantize_to_ci2(signal)
        assert np.all(ci2["real"] >= -128)
        assert np.all(ci2["real"] <= 127)
        assert np.all(ci2["imag"] >= -128)
        assert np.all(ci2["imag"] <= 127)

    def test_per_vector_scaling(self):
        """Each row gets its own scale factor."""
        signal = np.zeros((3, 10), dtype=np.complex64)
        signal[0] = 10 + 0j
        signal[1] = 100 + 0j
        signal[2] = 1000 + 0j
        ci2, amp_sf = quantize_to_ci2(signal)

        # Scale factors should differ per row
        assert amp_sf[0] < amp_sf[1] < amp_sf[2]
        # All ci2 real values at max (127) since uniform signal
        assert np.all(ci2["real"] == 127)

    def test_negative_values(self):
        """Negative values are preserved through quantization."""
        signal = np.array(
            [[-50 - 50j, 50 + 50j, -100 + 0j]],
            dtype=np.complex64,
        )
        ci2, amp_sf = quantize_to_ci2(signal)
        # First sample should have negative real & imag
        assert ci2["real"][0, 0] < 0
        assert ci2["imag"][0, 0] < 0
        # Second sample should be positive
        assert ci2["real"][0, 1] > 0
        assert ci2["imag"][0, 1] > 0


# =====================================================================
# Tests: crsd_pvp_builder — PVP/PPP array construction
# =====================================================================


class TestBuildPvpArray:
    """Tests for build_pvp_array()."""

    @pytest.fixture
    def xml_and_channel(self, channel_info, leo_orbit, acf_vectors):
        """Build a minimal XML tree and return (xmltree, channel, args)."""
        scene = CRSDSceneInfo(
            iarp_ecf=np.array([6378137.0, 0.0, 0.0]),
            iarp_llh=(0.0, 0.0, 0.0),
            uiax=np.array([0.0, 0.0, 1.0]),
            uiay=np.array([0.0, 1.0, 0.0]),
        )
        ref_geom = CRSDReferenceGeometryInfo(
            ref_pos_ecf=scene.iarp_ecf,
            ref_pos_iac=(0.0, 0.0),
            cod_time=5.003,
            dwell_time=0.006,
            platform_pos=leo_orbit[0][5],
            platform_vel=leo_orbit[1][5],
        )
        builder = CRSDMetadataBuilder(
            product_name="TEST_PRODUCT",
            collection_ref_time=datetime(2025, 12, 9, 15, 19, 25),
            sensor_name="S1A",
            channels=[channel_info],
            scene=scene,
            ref_geometry=ref_geom,
        )
        xmltree = builder.build()
        positions, velocities = leo_orbit
        acx, acy = acf_vectors
        rcv_times = np.linspace(5.0, 5.006, 10)
        amp_sf = np.ones(10) * 0.15
        return xmltree, channel_info, rcv_times, positions, velocities, acx, acy, amp_sf

    def test_output_shape_and_dtype(self, xml_and_channel):
        xmltree, ch, rcv_times, pos, vel, acx, acy, amp_sf = xml_and_channel
        pvp_dtype = get_pvp_dtype(xmltree)
        pvp = build_pvp_array(ch, pvp_dtype, rcv_times, pos, vel, acx, acy, amp_sf)
        assert pvp.shape == (10,)
        assert pvp.dtype == pvp_dtype

    def test_rcv_start_decomposition(self, xml_and_channel):
        """RcvStart should decompose to integer + fractional seconds."""
        xmltree, ch, rcv_times, pos, vel, acx, acy, amp_sf = xml_and_channel
        pvp_dtype = get_pvp_dtype(xmltree)
        pvp = build_pvp_array(ch, pvp_dtype, rcv_times, pos, vel, acx, acy, amp_sf)

        reconstructed = pvp["RcvStart"]["Int"].astype(float) + pvp["RcvStart"]["Frac"]
        np.testing.assert_allclose(reconstructed, rcv_times, atol=1e-15)

    def test_rcv_positions_stored(self, xml_and_channel):
        """Positions should round-trip through the PVP array."""
        xmltree, ch, rcv_times, pos, vel, acx, acy, amp_sf = xml_and_channel
        pvp_dtype = get_pvp_dtype(xmltree)
        pvp = build_pvp_array(ch, pvp_dtype, rcv_times, pos, vel, acx, acy, amp_sf)

        stored = np.column_stack([
            pvp["RcvPos"][:, 0],
            pvp["RcvPos"][:, 1],
            pvp["RcvPos"][:, 2],
        ])
        np.testing.assert_allclose(stored, pos, rtol=1e-12)

    def test_velocity_stored(self, xml_and_channel):
        xmltree, ch, rcv_times, pos, vel, acx, acy, amp_sf = xml_and_channel
        pvp_dtype = get_pvp_dtype(xmltree)
        pvp = build_pvp_array(ch, pvp_dtype, rcv_times, pos, vel, acx, acy, amp_sf)

        stored = np.column_stack([
            pvp["RcvVel"][:, 0],
            pvp["RcvVel"][:, 1],
            pvp["RcvVel"][:, 2],
        ])
        np.testing.assert_allclose(stored, vel, rtol=1e-12)

    def test_frequency_extents(self, xml_and_channel):
        xmltree, ch, rcv_times, pos, vel, acx, acy, amp_sf = xml_and_channel
        pvp_dtype = get_pvp_dtype(xmltree)
        pvp = build_pvp_array(ch, pvp_dtype, rcv_times, pos, vel, acx, acy, amp_sf)

        expected_frcv1 = ch.f0_ref - ch.bw_inst / 2.0
        expected_frcv2 = ch.f0_ref + ch.bw_inst / 2.0
        assert pvp["FRCV1"][0] == pytest.approx(expected_frcv1)
        assert pvp["FRCV2"][0] == pytest.approx(expected_frcv2)

    def test_compensation_fields_zero(self, xml_and_channel):
        """Compensation fields should all be zero (raw data)."""
        xmltree, ch, rcv_times, pos, vel, acx, acy, amp_sf = xml_and_channel
        pvp_dtype = get_pvp_dtype(xmltree)
        pvp = build_pvp_array(ch, pvp_dtype, rcv_times, pos, vel, acx, acy, amp_sf)

        assert np.all(pvp["RefPhi0"]["Int"] == 0)
        assert np.all(pvp["RefPhi0"]["Frac"] == 0.0)
        assert np.all(pvp["DFIC0"] == 0.0)
        assert np.all(pvp["FICRate"] == 0.0)
        assert np.all(pvp["DGRGC"] == 0.0)

    def test_signal_flag_valid(self, xml_and_channel):
        xmltree, ch, rcv_times, pos, vel, acx, acy, amp_sf = xml_and_channel
        pvp_dtype = get_pvp_dtype(xmltree)
        pvp = build_pvp_array(ch, pvp_dtype, rcv_times, pos, vel, acx, acy, amp_sf)

        assert np.all(pvp["SIGNAL"] == 1)

    def test_amp_sf_stored(self, xml_and_channel):
        xmltree, ch, rcv_times, pos, vel, acx, acy, amp_sf = xml_and_channel
        pvp_dtype = get_pvp_dtype(xmltree)
        pvp = build_pvp_array(ch, pvp_dtype, rcv_times, pos, vel, acx, acy, amp_sf)

        np.testing.assert_allclose(pvp["AmpSF"], amp_sf)

    def test_tx_pulse_index_sequential(self, xml_and_channel):
        xmltree, ch, rcv_times, pos, vel, acx, acy, amp_sf = xml_and_channel
        pvp_dtype = get_pvp_dtype(xmltree)
        pvp = build_pvp_array(ch, pvp_dtype, rcv_times, pos, vel, acx, acy, amp_sf)

        np.testing.assert_array_equal(pvp["TxPulseIndex"], np.arange(10))


class TestBuildPppArray:
    """Tests for build_ppp_array()."""

    @pytest.fixture
    def xml_and_channel(self, channel_info, leo_orbit, acf_vectors):
        """Build a minimal XML tree and return components."""
        scene = CRSDSceneInfo(
            iarp_ecf=np.array([6378137.0, 0.0, 0.0]),
            iarp_llh=(0.0, 0.0, 0.0),
            uiax=np.array([0.0, 0.0, 1.0]),
            uiay=np.array([0.0, 1.0, 0.0]),
        )
        ref_geom = CRSDReferenceGeometryInfo(
            ref_pos_ecf=scene.iarp_ecf,
            ref_pos_iac=(0.0, 0.0),
            cod_time=5.003,
            dwell_time=0.006,
            platform_pos=leo_orbit[0][5],
            platform_vel=leo_orbit[1][5],
        )
        builder = CRSDMetadataBuilder(
            product_name="TEST_PRODUCT",
            collection_ref_time=datetime(2025, 12, 9, 15, 19, 25),
            sensor_name="S1A",
            channels=[channel_info],
            scene=scene,
            ref_geometry=ref_geom,
        )
        xmltree = builder.build()
        positions, velocities = leo_orbit
        acx, acy = acf_vectors
        tx_times = np.linspace(5.0, 5.006, 10)
        tx_rad_int = np.ones(10) * 2500.0
        return xmltree, channel_info, tx_times, positions, velocities, acx, acy, tx_rad_int

    def test_output_shape_and_dtype(self, xml_and_channel):
        xmltree, ch, tx_times, pos, vel, acx, acy, tx_rad = xml_and_channel
        ppp_dtype = get_ppp_dtype(xmltree)
        ppp = build_ppp_array(ch, ppp_dtype, tx_times, pos, vel, acx, acy, tx_rad)
        assert ppp.shape == (10,)
        assert ppp.dtype == ppp_dtype

    def test_tx_time_decomposition(self, xml_and_channel):
        """TxTime should decompose to integer + fractional seconds."""
        xmltree, ch, tx_times, pos, vel, acx, acy, tx_rad = xml_and_channel
        ppp_dtype = get_ppp_dtype(xmltree)
        ppp = build_ppp_array(ch, ppp_dtype, tx_times, pos, vel, acx, acy, tx_rad)

        reconstructed = ppp["TxTime"]["Int"].astype(float) + ppp["TxTime"]["Frac"]
        np.testing.assert_allclose(reconstructed, tx_times, atol=1e-15)

    def test_chirp_frequency_extents(self, xml_and_channel):
        xmltree, ch, tx_times, pos, vel, acx, acy, tx_rad = xml_and_channel
        ppp_dtype = get_ppp_dtype(xmltree)
        ppp = build_ppp_array(ch, ppp_dtype, tx_times, pos, vel, acx, acy, tx_rad)

        assert ppp["FX1"][0] == pytest.approx(ch.fx_freq0)
        assert ppp["FX2"][0] == pytest.approx(ch.fx_freq0 + ch.fx_bw)

    def test_pulse_duration(self, xml_and_channel):
        xmltree, ch, tx_times, pos, vel, acx, acy, tx_rad = xml_and_channel
        ppp_dtype = get_ppp_dtype(xmltree)
        ppp = build_ppp_array(ch, ppp_dtype, tx_times, pos, vel, acx, acy, tx_rad)

        assert np.all(ppp["TXmt"] == pytest.approx(ch.tx_pulse_duration))

    def test_chirp_params(self, xml_and_channel):
        xmltree, ch, tx_times, pos, vel, acx, acy, tx_rad = xml_and_channel
        ppp_dtype = get_ppp_dtype(xmltree)
        ppp = build_ppp_array(ch, ppp_dtype, tx_times, pos, vel, acx, acy, tx_rad)

        assert np.all(ppp["FxFreq0"] == pytest.approx(ch.fx_freq0))
        assert np.all(ppp["FxRate"] == pytest.approx(ch.fx_rate))

    def test_phase_compensation_zero(self, xml_and_channel):
        xmltree, ch, tx_times, pos, vel, acx, acy, tx_rad = xml_and_channel
        ppp_dtype = get_ppp_dtype(xmltree)
        ppp = build_ppp_array(ch, ppp_dtype, tx_times, pos, vel, acx, acy, tx_rad)

        assert np.all(ppp["PhiX0"]["Int"] == 0)
        assert np.all(ppp["PhiX0"]["Frac"] == 0.0)

    def test_positions_stored(self, xml_and_channel):
        xmltree, ch, tx_times, pos, vel, acx, acy, tx_rad = xml_and_channel
        ppp_dtype = get_ppp_dtype(xmltree)
        ppp = build_ppp_array(ch, ppp_dtype, tx_times, pos, vel, acx, acy, tx_rad)

        stored = np.column_stack([
            ppp["TxPos"][:, 0],
            ppp["TxPos"][:, 1],
            ppp["TxPos"][:, 2],
        ])
        np.testing.assert_allclose(stored, pos, rtol=1e-12)


# =====================================================================
# Tests: crsd_metadata_builder — XML builder
# =====================================================================


class TestCRSDMetadataBuilder:
    """Tests for CRSDMetadataBuilder.build()."""

    @pytest.fixture
    def builder(self, channel_info, leo_orbit):
        """Create a metadata builder with minimal inputs."""
        scene = CRSDSceneInfo(
            iarp_ecf=np.array([6378137.0, 0.0, 0.0]),
            iarp_llh=(0.0, 0.0, 0.0),
            uiax=np.array([0.0, 0.0, 1.0]),
            uiay=np.array([0.0, 1.0, 0.0]),
            image_area_x1y1=(-50000.0, -50000.0),
            image_area_x2y2=(50000.0, 50000.0),
            corner_coords=[
                (1.0, 1.0), (-1.0, 1.0),
                (-1.0, -1.0), (1.0, -1.0),
            ],
        )
        ref_geom = CRSDReferenceGeometryInfo(
            ref_pos_ecf=scene.iarp_ecf,
            ref_pos_iac=(0.0, 0.0),
            cod_time=5.003,
            dwell_time=0.006,
            platform_pos=leo_orbit[0][5],
            platform_vel=leo_orbit[1][5],
        )
        return CRSDMetadataBuilder(
            product_name="TEST_S1A_IW_RAW",
            collection_ref_time=datetime(2025, 12, 9, 15, 19, 25),
            sensor_name="S1A",
            channels=[channel_info],
            scene=scene,
            ref_geometry=ref_geom,
        )

    def test_build_returns_etree(self, builder):
        from lxml import etree
        xmltree = builder.build()
        assert isinstance(xmltree, etree._ElementTree)

    def test_root_element_is_crsdsar(self, builder):
        xmltree = builder.build()
        root = xmltree.getroot()
        local_name = root.tag.split("}")[-1] if "}" in root.tag else root.tag
        assert local_name == "CRSDsar"

    def test_namespace(self, builder):
        xmltree = builder.build()
        root = xmltree.getroot()
        assert "crsd" in root.tag.lower() or "1.0" in root.tag

    def test_product_name(self, builder):
        xmltree = builder.build()
        root = xmltree.getroot()
        product_name = root.findtext("ProductInfo/ProductName")
        assert product_name == "TEST_S1A_IW_RAW"

    def test_collection_ref_time(self, builder):
        xmltree = builder.build()
        root = xmltree.getroot()
        crt = root.findtext("Global/CollectionRefTime")
        assert "2025-12-09" in crt
        assert "15:19:25" in crt

    def test_sensor_name(self, builder):
        xmltree = builder.build()
        root = xmltree.getroot()
        sensor = root.findtext("TransmitInfo/SensorName")
        assert sensor == "S1A"

    def test_mode_type(self, builder):
        xmltree = builder.build()
        root = xmltree.getroot()
        mode = root.findtext(".//ModeType")
        assert mode == "STRIPMAP"

    def test_collect_type(self, builder):
        xmltree = builder.build()
        root = xmltree.getroot()
        collect_type = root.findtext(".//CollectType")
        assert collect_type == "MONOSTATIC"

    def test_channel_count_in_data(self, builder):
        xmltree = builder.build()
        root = xmltree.getroot()
        channels = root.findall("Data/Receive/Channel")
        assert len(channels) == 1  # single channel

    def test_channel_dimensions(self, builder, channel_info):
        xmltree = builder.build()
        root = xmltree.getroot()
        ch = root.find("Data/Receive/Channel")
        nv = int(ch.findtext("NumVectors"))
        ns_ = int(ch.findtext("NumSamples"))
        assert nv == channel_info.num_vectors
        assert ns_ == channel_info.num_samples

    def test_iarp_position(self, builder):
        xmltree = builder.build()
        root = xmltree.getroot()
        ecf = root.find(".//IARP/ECF")
        x = float(ecf.findtext("X"))
        assert x == pytest.approx(6378137.0, rel=1e-6)

    def test_reference_geometry_present(self, builder):
        xmltree = builder.build()
        root = xmltree.getroot()
        ref_geom = root.find("ReferenceGeometry")
        assert ref_geom is not None

    def test_pvp_section_present(self, builder):
        xmltree = builder.build()
        root = xmltree.getroot()
        pvp = root.find("PVP")
        assert pvp is not None

    def test_ppp_section_present(self, builder):
        xmltree = builder.build()
        root = xmltree.getroot()
        ppp = root.find("PPP")
        assert ppp is not None

    def test_multi_channel(self, leo_orbit):
        """Builder with 3 channels (IW1, IW2, IW3)."""
        channels = [
            _make_channel_info(
                identifier=f"043_000001_{sw}",
                swath_name=sw,
                num_vectors=100,
                num_samples=200,
            )
            for sw in ["IW1", "IW2", "IW3"]
        ]
        scene = CRSDSceneInfo(
            iarp_ecf=np.array([6378137.0, 0.0, 0.0]),
            iarp_llh=(0.0, 0.0, 0.0),
            uiax=np.array([0.0, 0.0, 1.0]),
            uiay=np.array([0.0, 1.0, 0.0]),
        )
        ref_geom = CRSDReferenceGeometryInfo(
            ref_pos_ecf=scene.iarp_ecf,
            ref_pos_iac=(0.0, 0.0),
            cod_time=5.0,
            dwell_time=1.0,
            platform_pos=leo_orbit[0][5],
            platform_vel=leo_orbit[1][5],
        )
        builder = CRSDMetadataBuilder(
            product_name="TEST_MULTI",
            collection_ref_time=datetime(2025, 12, 9, 15, 19, 25),
            sensor_name="S1A",
            channels=channels,
            scene=scene,
            ref_geometry=ref_geom,
        )
        xmltree = builder.build()
        root = xmltree.getroot()
        data_channels = root.findall("Data/Receive/Channel")
        assert len(data_channels) == 3


# =====================================================================
# Tests: sarkit round-trip (write → read)
# =====================================================================


class TestSarkitRoundTrip:
    """Write a minimal CRSD with sarkit, read it back, verify."""

    @pytest.fixture
    def crsd_components(self, channel_info, leo_orbit, acf_vectors, tmp_path):
        """Build all components for a minimal CRSD file."""
        scene = CRSDSceneInfo(
            iarp_ecf=np.array([6378137.0, 0.0, 0.0]),
            iarp_llh=(0.0, 0.0, 0.0),
            uiax=np.array([0.0, 0.0, 1.0]),
            uiay=np.array([0.0, 1.0, 0.0]),
            image_area_x1y1=(-50000.0, -50000.0),
            image_area_x2y2=(50000.0, 50000.0),
            corner_coords=[
                (1.0, 1.0), (-1.0, 1.0),
                (-1.0, -1.0), (1.0, -1.0),
            ],
        )
        ref_geom = CRSDReferenceGeometryInfo(
            ref_pos_ecf=scene.iarp_ecf,
            ref_pos_iac=(0.0, 0.0),
            cod_time=5.003,
            dwell_time=0.006,
            platform_pos=leo_orbit[0][5],
            platform_vel=leo_orbit[1][5],
        )
        builder = CRSDMetadataBuilder(
            product_name="TEST_ROUNDTRIP",
            collection_ref_time=datetime(2025, 12, 9, 15, 19, 25),
            sensor_name="S1A",
            channels=[channel_info],
            scene=scene,
            ref_geometry=ref_geom,
        )
        xmltree = builder.build()

        # Build PVP/PPP
        positions, velocities = leo_orbit
        acx, acy = acf_vectors
        rcv_times = np.linspace(5.0, 5.006, 10)
        amp_sf = np.ones(10) * 0.15
        tx_rad_int = np.ones(10) * 2500.0

        pvp_dtype = get_pvp_dtype(xmltree)
        ppp_dtype = get_ppp_dtype(xmltree)

        pvp = build_pvp_array(
            channel_info, pvp_dtype, rcv_times,
            positions, velocities, acx, acy, amp_sf,
        )
        ppp = build_ppp_array(
            channel_info, ppp_dtype, rcv_times,
            positions, velocities, acx, acy, tx_rad_int,
        )

        # Build signal
        rng = np.random.RandomState(42)
        signal = (
            rng.randn(10, 100).astype(np.float32)
            + 1j * rng.randn(10, 100).astype(np.float32)
        ) * 50.0
        ci2, _ = quantize_to_ci2(signal)

        return tmp_path, xmltree, channel_info, pvp, ppp, ci2

    def test_write_and_read_back(self, crsd_components):
        """Write a CRSD file with sarkit, read it back, check dims."""
        import sarkit.crsd

        tmp_path, xmltree, ch, pvp, ppp, ci2 = crsd_components
        out_file = tmp_path / "test.crsd"

        metadata = sarkit.crsd.Metadata(xmltree=xmltree)
        with open(out_file, "wb") as f:
            writer = sarkit.crsd.Writer(f, metadata)
            writer.write_signal(ch.identifier, ci2)
            writer.write_pvp(ch.identifier, pvp)
            # Find the TxSequence ID from XML
            root = xmltree.getroot()
            tx_seqs = root.findall("Data/Transmit/TxSequence")
            if tx_seqs:
                seq_id = tx_seqs[0].findtext("TxId")
                writer.write_ppp(seq_id, ppp)
            # Write support array (NumCols=30 hardcoded in builder)
            sa_dtype = np.dtype([("Amp", "<f4"), ("Phase", "<f4")])
            sa = np.zeros((1, 30), dtype=sa_dtype)
            sa["Amp"] = 1.0
            sa["Phase"] = 0.0
            writer.write_support_array("flat_fx_response", sa)
            writer.done()

        # Read back
        assert out_file.exists()
        with open(out_file, "rb") as f:
            reader = sarkit.crsd.Reader(f)
            sig_back = reader.read_signal(ch.identifier)
            pvp_back = reader.read_pvps(ch.identifier)
            reader.done()

        assert sig_back.shape == ci2.shape
        assert pvp_back.shape == pvp.shape

    def test_signal_data_round_trip(self, crsd_components):
        """Signal CI2 data should be bit-identical after write/read."""
        import sarkit.crsd

        tmp_path, xmltree, ch, pvp, ppp, ci2 = crsd_components
        out_file = tmp_path / "test_signal.crsd"

        metadata = sarkit.crsd.Metadata(xmltree=xmltree)
        with open(out_file, "wb") as f:
            writer = sarkit.crsd.Writer(f, metadata)
            writer.write_signal(ch.identifier, ci2)
            writer.write_pvp(ch.identifier, pvp)
            root = xmltree.getroot()
            tx_seqs = root.findall("Data/Transmit/TxSequence")
            if tx_seqs:
                seq_id = tx_seqs[0].findtext("TxId")
                writer.write_ppp(seq_id, ppp)
            sa_dtype = np.dtype([("Amp", "<f4"), ("Phase", "<f4")])
            sa = np.zeros((1, 30), dtype=sa_dtype)
            sa["Amp"] = 1.0
            writer.write_support_array("flat_fx_response", sa)
            writer.done()

        with open(out_file, "rb") as f:
            reader = sarkit.crsd.Reader(f)
            sig_back = reader.read_signal(ch.identifier)
            reader.done()

        np.testing.assert_array_equal(sig_back["real"], ci2["real"])
        np.testing.assert_array_equal(sig_back["imag"], ci2["imag"])

    def test_pvp_round_trip(self, crsd_components):
        """PVP fields should be preserved after write/read."""
        import sarkit.crsd

        tmp_path, xmltree, ch, pvp, ppp, ci2 = crsd_components
        out_file = tmp_path / "test_pvp.crsd"

        metadata = sarkit.crsd.Metadata(xmltree=xmltree)
        with open(out_file, "wb") as f:
            writer = sarkit.crsd.Writer(f, metadata)
            writer.write_signal(ch.identifier, ci2)
            writer.write_pvp(ch.identifier, pvp)
            root = xmltree.getroot()
            tx_seqs = root.findall("Data/Transmit/TxSequence")
            if tx_seqs:
                seq_id = tx_seqs[0].findtext("TxId")
                writer.write_ppp(seq_id, ppp)
            sa_dtype = np.dtype([("Amp", "<f4"), ("Phase", "<f4")])
            sa = np.zeros((1, 30), dtype=sa_dtype)
            sa["Amp"] = 1.0
            writer.write_support_array("flat_fx_response", sa)
            writer.done()

        with open(out_file, "rb") as f:
            reader = sarkit.crsd.Reader(f)
            pvp_back = reader.read_pvps(ch.identifier)
            reader.done()

        # Check key fields are preserved
        np.testing.assert_allclose(
            pvp_back["RcvStart"]["Int"],
            pvp["RcvStart"]["Int"],
        )
        np.testing.assert_allclose(
            pvp_back["RcvStart"]["Frac"],
            pvp["RcvStart"]["Frac"],
            atol=1e-15,
        )
        np.testing.assert_allclose(
            pvp_back["RcvPos"][:, 0],
            pvp["RcvPos"][:, 0],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            pvp_back["AmpSF"],
            pvp["AmpSF"],
            rtol=1e-12,
        )


# =====================================================================
# Tests: BurstChannelInfo dataclass
# =====================================================================


class TestBurstChannelInfo:
    """Tests for BurstChannelInfo construction and defaults."""

    def test_basic_construction(self, channel_info):
        assert channel_info.identifier == "043_000001_IW1"
        assert channel_info.swath_name == "IW1"
        assert channel_info.polarization == "VV"
        assert channel_info.num_vectors == 10
        assert channel_info.num_samples == 100

    def test_default_iac(self, channel_info):
        assert channel_info.rcv_ref_pos_iac == (0.0, 0.0)

    def test_frequency_params(self, channel_info):
        assert channel_info.f0_ref == pytest.approx(5.405e9)
        assert channel_info.fs == pytest.approx(64.345238e6)
        assert channel_info.fx_rate == pytest.approx(779.038e9)

    def test_time_ordering(self, channel_info):
        assert channel_info.tx_time_first < channel_info.tx_time_last
        assert channel_info.rcv_start_first < channel_info.rcv_start_last
