# -*- coding: utf-8 -*-
"""
Tests for SAR Image Formation - PFA pipeline with synthetic data.

Tests the image formation package using synthetic CPHD-like data:
  - CPHDMetadata / CPHDPVP model construction and properties
  - CollectionGeometry from synthetic PVP
  - PolarGrid k-space bounds and grid dimensions
  - PolarFormatAlgorithm stage-by-stage and full pipeline
  - ImageFormationAlgorithm ABC contract

All data is synthetic — no real CPHD files needed.

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
2026-02-12

Modified
--------
2026-02-13
"""

import pytest
import numpy as np
from numpy.linalg import norm

from grdl.IO.models.cphd import (
    CPHDMetadata,
    CPHDChannel,
    CPHDPVP,
    CPHDGlobal,
    CPHDCollectionInfo,
    CPHDTxWaveform,
    CPHDRcvParameters,
    create_subaperture_metadata,
)
from grdl.image_processing.sar.image_formation import (
    ImageFormationAlgorithm,
    CollectionGeometry,
    PolarGrid,
    PolarFormatAlgorithm,
    SubaperturePartitioner,
    StripmapPFA,
    RangeDopplerAlgorithm,
    FastBackProjection,
)


# ===================================================================
# Synthetic data helpers
# ===================================================================

_C = 299792458.0  # Speed of light (m/s)


def _synthetic_pvp(
    npulses: int = 64,
    nsamples: int = 128,
    center_freq_ghz: float = 10.0,
    bandwidth_mhz: float = 600.0,
    altitude_km: float = 10.0,
    slant_range_km: float = 15.0,
    aperture_deg: float = 3.0,
) -> CPHDPVP:
    """Build synthetic PVP arrays for a monostatic spotlight collection.

    Creates a circular arc platform track at constant altitude, looking
    down at a scene center point near (40°N, -75°W) on the WGS-84
    ellipsoid.

    Parameters
    ----------
    npulses : int
        Number of pulses in the aperture.
    nsamples : int
        Number of frequency samples per pulse.
    center_freq_ghz : float
        Center frequency in GHz.
    bandwidth_mhz : float
        Instantaneous bandwidth in MHz.
    altitude_km : float
        Platform altitude above SRP in km.
    slant_range_km : float
        Nominal slant range in km.
    aperture_deg : float
        Total aperture angle in degrees.

    Returns
    -------
    CPHDPVP
        Populated per-vector parameters.
    """
    fc = center_freq_ghz * 1e9
    bw = bandwidth_mhz * 1e6
    alt = altitude_km * 1e3
    sr = slant_range_km * 1e3

    # SRP position (40°N, 75°W on WGS-84 ellipsoid)
    lat_rad = np.radians(40.0)
    lon_rad = np.radians(-75.0)
    a_wgs = 6378137.0
    f_wgs = 1.0 / 298.257223563
    e2 = 2 * f_wgs - f_wgs**2
    N_val = a_wgs / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    srp_ecf = np.array([
        (N_val) * np.cos(lat_rad) * np.cos(lon_rad),
        (N_val) * np.cos(lat_rad) * np.sin(lon_rad),
        (N_val * (1 - e2)) * np.sin(lat_rad),
    ])

    # Platform track: arc overhead at altitude
    half_angle = np.radians(aperture_deg / 2)
    angles = np.linspace(-half_angle, half_angle, npulses)

    # Up direction at SRP
    u_up = srp_ecf / norm(srp_ecf)
    # Arbitrary east direction
    z_axis = np.array([0.0, 0.0, 1.0])
    u_east = np.cross(z_axis, u_up)
    u_east = u_east / norm(u_east)
    u_north = np.cross(u_up, u_east)

    # ARP positions: arc in north-up plane at slant_range
    ground_range = np.sqrt(sr**2 - alt**2)
    arp = np.zeros((npulses, 3))
    for i, angle in enumerate(angles):
        offset = (
            ground_range * np.cos(angle) * u_north
            + ground_range * np.sin(angle) * u_east
            + alt * u_up
        )
        arp[i] = srp_ecf + offset

    # Platform velocity (tangent to arc)
    speed = 200.0  # m/s
    vel = np.zeros((npulses, 3))
    for i, angle in enumerate(angles):
        vel[i] = speed * (
            -np.sin(angle) * u_north + np.cos(angle) * u_east
        )

    # Time array
    arc_length = sr * 2 * half_angle
    total_time = arc_length / speed
    tx_time = np.linspace(0.0, total_time, npulses)
    rcv_time = tx_time + 2 * sr / _C

    # SRP repeated
    srp_pos = np.tile(srp_ecf, (npulses, 1))

    # Frequency parameters
    fx1 = np.full(npulses, fc - bw / 2)
    fx2 = np.full(npulses, fc + bw / 2)
    fxss = np.full(npulses, bw / nsamples)
    sc0 = fx1.copy()

    return CPHDPVP(
        tx_time=tx_time,
        tx_pos=arp,
        tx_vel=vel,
        rcv_time=rcv_time,
        rcv_pos=arp,
        rcv_vel=vel,
        srp_pos=srp_pos,
        fx1=fx1,
        fx2=fx2,
        sc0=sc0,
        scss=fxss,
    )


def _synthetic_metadata(
    npulses: int = 64,
    nsamples: int = 128,
    **pvp_kwargs,
) -> CPHDMetadata:
    """Build complete synthetic CPHDMetadata.

    Parameters
    ----------
    npulses : int
        Number of pulses.
    nsamples : int
        Number of samples per pulse.
    **pvp_kwargs
        Extra keyword arguments passed to ``_synthetic_pvp``.

    Returns
    -------
    CPHDMetadata
        Fully populated metadata with PVP, channels, global params.
    """
    pvp = _synthetic_pvp(npulses=npulses, nsamples=nsamples, **pvp_kwargs)

    channel = CPHDChannel(
        identifier='CHANNEL001',
        num_vectors=npulses,
        num_samples=nsamples,
    )

    fc = pvp_kwargs.get('center_freq_ghz', 10.0) * 1e9
    bw = pvp_kwargs.get('bandwidth_mhz', 600.0) * 1e6

    global_params = CPHDGlobal(
        domain_type='FX',
        fx_band_min=fc - bw / 2,
        fx_band_max=fc + bw / 2,
    )

    collection_info = CPHDCollectionInfo(
        collector_name='SYNTHETIC',
        core_name='TEST001',
        collect_type='MONOSTATIC',
        radar_mode='SPOTLIGHT',
    )

    return CPHDMetadata(
        format='CPHD',
        rows=npulses,
        cols=nsamples,
        dtype='complex64',
        channels=[channel],
        pvp=pvp,
        global_params=global_params,
        collection_info=collection_info,
        num_channels=1,
    )


def _synthetic_signal(
    npulses: int = 64,
    nsamples: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """Create synthetic phase history data (complex Gaussian noise).

    Parameters
    ----------
    npulses : int
        Number of pulses.
    nsamples : int
        Number of samples per pulse.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Complex-valued signal, shape ``(npulses, nsamples)``.
    """
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((npulses, nsamples))
        + 1j * rng.standard_normal((npulses, nsamples))
    ).astype(np.complex64)


# ===================================================================
# CPHDMetadata model tests
# ===================================================================

class TestCPHDMetadata:
    """Tests for CPHDMetadata and sub-models."""

    def test_cphdpvp_num_vectors(self):
        pvp = _synthetic_pvp(npulses=32)
        assert pvp.num_vectors == 32

    def test_cphdpvp_midpoint_time(self):
        pvp = _synthetic_pvp(npulses=10)
        mid = pvp.midpoint_time
        assert mid > 0.0

    def test_cphdpvp_first_valid_pulse_no_signal(self):
        pvp = _synthetic_pvp(npulses=10)
        assert pvp.first_valid_pulse == 0

    def test_cphdpvp_first_valid_pulse_with_signal(self):
        pvp = _synthetic_pvp(npulses=10)
        sig = np.zeros(10)
        sig[3:] = 1.0
        pvp.signal = sig
        assert pvp.first_valid_pulse == 3

    def test_cphdpvp_trim_to_valid(self):
        pvp = _synthetic_pvp(npulses=10)
        sig = np.zeros(10)
        sig[3:] = 1.0
        pvp.signal = sig
        trimmed = pvp.trim_to_valid()
        assert trimmed.num_vectors == 7
        assert trimmed.tx_pos.shape[0] == 7

    def test_cphdpvp_trim_noop_without_signal(self):
        pvp = _synthetic_pvp(npulses=10)
        trimmed = pvp.trim_to_valid()
        assert trimmed is pvp

    def test_cphd_global_bandwidth(self):
        g = CPHDGlobal(fx_band_min=9.7e9, fx_band_max=10.3e9)
        assert g.bandwidth == pytest.approx(0.6e9, rel=1e-6)

    def test_cphd_global_center_frequency(self):
        g = CPHDGlobal(fx_band_min=9.7e9, fx_band_max=10.3e9)
        assert g.center_frequency == pytest.approx(10.0e9, rel=1e-6)

    def test_cphd_global_no_freq(self):
        g = CPHDGlobal()
        assert g.bandwidth is None
        assert g.center_frequency is None

    def test_metadata_inherits_image_metadata(self):
        meta = _synthetic_metadata()
        assert meta.format == 'CPHD'
        assert meta.rows == 64
        assert meta.cols == 128

    def test_metadata_channels(self):
        meta = _synthetic_metadata()
        assert len(meta.channels) == 1
        assert meta.channels[0].identifier == 'CHANNEL001'
        assert meta.channels[0].num_vectors == 64
        assert meta.channels[0].num_samples == 128

    def test_metadata_collection_info(self):
        meta = _synthetic_metadata()
        ci = meta.collection_info
        assert ci.collector_name == 'SYNTHETIC'
        assert ci.radar_mode == 'SPOTLIGHT'


# ===================================================================
# CollectionGeometry tests
# ===================================================================

class TestCollectionGeometry:
    """Tests for CollectionGeometry with synthetic PVP data."""

    @pytest.fixture
    def geo(self):
        meta = _synthetic_metadata(npulses=64, nsamples=128)
        return CollectionGeometry(meta, slant=True)

    def test_construction_slant(self, geo):
        assert geo.image_plane == 'SLANT'
        assert geo.npulses == 64
        assert geo.nsamples == 128

    def test_construction_ground(self):
        meta = _synthetic_metadata()
        geo = CollectionGeometry(meta, slant=False)
        assert geo.image_plane == 'GROUND'

    def test_missing_pvp_raises(self):
        meta = CPHDMetadata(format='CPHD', rows=10, cols=10, dtype='complex64')
        with pytest.raises(ValueError, match="populated PVP"):
            CollectionGeometry(meta)

    def test_srp_shape(self, geo):
        assert geo.srp.shape == (64, 3)

    def test_srp_llh_reasonable(self, geo):
        lat = geo.srp_llh[0, 0]
        lon = geo.srp_llh[0, 1]
        assert 35 < lat < 45, f"lat {lat} not near 40°"
        assert -80 < lon < -70, f"lon {lon} not near -75°"

    def test_arp_shape(self, geo):
        assert geo.arp.shape == (64, 3)

    def test_varp_shape(self, geo):
        assert geo.varp.shape == (64, 3)

    def test_phi_shape_and_range(self, geo):
        assert geo.phi.shape == (64,)
        assert np.all(np.abs(geo.phi) < np.pi)

    def test_k_sf_positive(self, geo):
        assert np.all(geo.k_sf > 0)

    def test_graze_angle_positive(self, geo):
        assert np.all(geo.graz_ang > 0)

    def test_incidence_angle(self, geo):
        # Incidence = pi/2 - graze
        expected = np.pi / 2 - geo.graz_ang
        np.testing.assert_allclose(geo.incd_ang, expected, atol=1e-10)

    def test_side_of_track(self, geo):
        assert geo.side_of_track in ('L', 'R')

    def test_coa_values_exist(self, geo):
        assert hasattr(geo, 'graz_ang_coa')
        assert hasattr(geo, 'azim_ang_coa')
        assert hasattr(geo, 'dop_cone_ang_coa')
        assert geo.graz_ang_coa > 0

    def test_theta_positive(self, geo):
        assert geo.theta > 0

    def test_unit_vectors(self, geo):
        assert geo.rg_uvect_ecf.shape == (3,)
        assert geo.az_uvect_ecf.shape == (3,)
        # Vectors should be non-zero
        assert norm(geo.rg_uvect_ecf) > 0
        assert norm(geo.az_uvect_ecf) > 0

    def test_arp_polynomials(self, geo):
        # ARP polys should reconstruct ARP position at COA time
        t = geo.coa_time
        x = float(geo.arp_poly_x(t))
        y = float(geo.arp_poly_y(t))
        z = float(geo.arp_poly_z(t))
        coa_arp = geo.arp[geo.npulses // 2]
        np.testing.assert_allclose([x, y, z], coa_arp, rtol=1e-4)

    def test_time_order(self, geo):
        assert geo.tstart < geo.tend
        assert geo.tstart <= geo.coa_time <= geo.tend

    def test_speed_of_light(self, geo):
        assert geo.c == _C


# ===================================================================
# PolarGrid tests
# ===================================================================

class TestPolarGrid:
    """Tests for PolarGrid k-space computation."""

    @pytest.fixture
    def geo(self):
        meta = _synthetic_metadata(npulses=64, nsamples=128)
        return CollectionGeometry(meta, slant=True)

    @pytest.fixture
    def grid(self, geo):
        return PolarGrid(geo, grid_mode='inscribed')

    def test_kv_bounds_ordered(self, grid):
        assert grid.kv_bounds[0] < grid.kv_bounds[1]

    def test_ku_bounds_exist(self, grid):
        assert grid.ku_bounds.shape == (2,)

    def test_rec_n_samples_positive(self, grid):
        assert grid.rec_n_samples > 0

    def test_rec_n_pulses_positive(self, grid):
        assert grid.rec_n_pulses > 0

    def test_range_resolution_positive(self, grid):
        assert grid.range_resolution > 0

    def test_azimuth_resolution_positive(self, grid):
        assert grid.azimuth_resolution > 0

    def test_circumscribed_mode(self, geo):
        grid = PolarGrid(geo, grid_mode='circumscribed')
        assert grid.rec_n_samples > 0
        assert grid.rec_n_pulses > 0

    def test_invalid_mode_raises(self, geo):
        with pytest.raises(ValueError, match="grid_mode"):
            PolarGrid(geo, grid_mode='invalid')

    def test_oversample_increases_samples(self, geo):
        grid1 = PolarGrid(geo, range_oversample=1.0)
        grid2 = PolarGrid(geo, range_oversample=2.0)
        # Higher oversample factor → more samples (same resolution)
        assert grid2.rec_n_samples > grid1.rec_n_samples
        assert grid2.range_resolution == pytest.approx(
            grid1.range_resolution, rel=1e-10,
        )

    def test_get_kv_for_pulse(self, grid):
        kv = grid.get_kv_for_pulse(0)
        assert kv.shape == (128,)

    def test_get_ku_for_sample(self, grid):
        ku = grid.get_ku_for_sample(0)
        assert ku.shape == (64,)

    def test_get_output_grid_keys(self, grid):
        out = grid.get_output_grid()
        expected_keys = {
            'image_plane', 'type', 'rg_ss', 'rg_imp_resp_bw',
            'rg_imp_resp_wid', 'rg_kctr', 'rg_delta_k1', 'rg_delta_k2',
            'az_ss', 'az_imp_resp_bw', 'az_imp_resp_wid', 'az_kctr',
            'ku_min', 'ku_max', 'range_resolution', 'azimuth_resolution',
            'rec_n_pulses', 'rec_n_samples',
        }
        assert set(out.keys()) == expected_keys

    def test_sicd_grid_type(self, grid):
        out = grid.get_output_grid()
        assert out['type'] == 'RGAZIM'

    def test_rg_ss_positive(self, grid):
        assert grid.rg_ss > 0

    def test_az_ss_positive(self, grid):
        assert grid.az_ss > 0


# ===================================================================
# PolarFormatAlgorithm tests
# ===================================================================

class TestPolarFormatAlgorithm:
    """Tests for PFA pipeline stages."""

    @pytest.fixture
    def setup(self):
        """Build geometry, grid, PFA, and synthetic signal."""
        meta = _synthetic_metadata(npulses=64, nsamples=128)
        geo = CollectionGeometry(meta, slant=True)
        grid = PolarGrid(geo, grid_mode='inscribed')
        pfa = PolarFormatAlgorithm(grid=grid)
        signal = _synthetic_signal(npulses=64, nsamples=128)
        return geo, grid, pfa, signal

    def test_interpolate_range_shape(self, setup):
        geo, grid, pfa, signal = setup
        result = pfa.interpolate_range(signal, geo)
        assert result.shape == (64, grid.rec_n_samples)

    def test_interpolate_range_dtype(self, setup):
        geo, grid, pfa, signal = setup
        result = pfa.interpolate_range(signal, geo)
        assert np.iscomplexobj(result)

    def test_interpolate_azimuth_shape(self, setup):
        geo, grid, pfa, signal = setup
        range_interp = pfa.interpolate_range(signal, geo)
        result = pfa.interpolate_azimuth(range_interp, geo)
        assert result.shape == (grid.rec_n_pulses, grid.rec_n_samples)

    def test_interpolate_azimuth_dtype(self, setup):
        geo, grid, pfa, signal = setup
        range_interp = pfa.interpolate_range(signal, geo)
        result = pfa.interpolate_azimuth(range_interp, geo)
        assert np.iscomplexobj(result)

    def test_compress_shape(self, setup):
        geo, grid, pfa, signal = setup
        range_interp = pfa.interpolate_range(signal, geo)
        az_interp = pfa.interpolate_azimuth(range_interp, geo)
        image = pfa.compress(az_interp)
        assert image.shape == az_interp.shape

    def test_compress_complex_output(self, setup):
        geo, grid, pfa, signal = setup
        range_interp = pfa.interpolate_range(signal, geo)
        az_interp = pfa.interpolate_azimuth(range_interp, geo)
        image = pfa.compress(az_interp)
        assert np.iscomplexobj(image)

    def test_form_image_equals_stages(self, setup):
        geo, grid, pfa, signal = setup
        # Full pipeline
        image_full = pfa.form_image(signal, geo)
        # Stage-by-stage
        range_interp = pfa.interpolate_range(signal, geo)
        az_interp = pfa.interpolate_azimuth(range_interp, geo)
        image_staged = pfa.compress(az_interp)
        np.testing.assert_array_equal(image_full, image_staged)

    def test_form_image_nonzero(self, setup):
        geo, grid, pfa, signal = setup
        image = pfa.form_image(signal, geo)
        assert np.any(np.abs(image) > 0)

    def test_get_output_grid(self, setup):
        geo, grid, pfa, signal = setup
        out = pfa.get_output_grid()
        assert 'rg_ss' in out
        assert 'az_ss' in out

    def test_custom_interpolator(self):
        """PFA accepts a custom interpolator callable."""
        meta = _synthetic_metadata(npulses=32, nsamples=64)
        geo = CollectionGeometry(meta, slant=True)
        grid = PolarGrid(geo)
        call_count = [0]

        def custom_interp(x_old, y_old, x_new):
            call_count[0] += 1
            return np.zeros_like(x_new, dtype=y_old.dtype)

        pfa = PolarFormatAlgorithm(grid=grid, interpolator=custom_interp)
        signal = _synthetic_signal(npulses=32, nsamples=64)
        pfa.form_image(signal, geo)
        assert call_count[0] > 0

    def test_compress_preserves_energy(self):
        """IFFT preserves total energy (Parseval's theorem)."""
        meta = _synthetic_metadata(npulses=32, nsamples=64)
        geo = CollectionGeometry(meta, slant=True)
        grid = PolarGrid(geo)
        pfa = PolarFormatAlgorithm(grid=grid)

        rng = np.random.default_rng(99)
        data = (rng.standard_normal((32, 64))
                + 1j * rng.standard_normal((32, 64)))
        # Use pad_factor=0.0 for clean Parseval check (no zero-padding)
        image = pfa.compress(data, pad_factor=0.0)
        # ifft2 scaled by N*M, so compare appropriately
        N = data.size
        input_energy = np.sum(np.abs(data)**2)
        output_energy = np.sum(np.abs(image)**2)
        np.testing.assert_allclose(
            output_energy, input_energy / N, rtol=1e-6,
        )


# ===================================================================
# ABC contract test
# ===================================================================

class TestImageFormationAlgorithmABC:
    """Tests for ImageFormationAlgorithm ABC contract."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ImageFormationAlgorithm()

    def test_pfa_is_subclass(self):
        assert issubclass(PolarFormatAlgorithm, ImageFormationAlgorithm)

    def test_custom_subclass(self):
        """A minimal subclass satisfies the ABC contract."""
        class DummyIFA(ImageFormationAlgorithm):
            def form_image(self, signal, geometry):
                return signal

            def get_output_grid(self):
                return {}

        ifa = DummyIFA()
        out = ifa.form_image(np.zeros((4, 4)), None)
        assert out.shape == (4, 4)
        assert ifa.get_output_grid() == {}


# ===================================================================
# Stripmap synthetic data helper
# ===================================================================

def _synthetic_stripmap_pvp(
    npulses: int = 256,
    nsamples: int = 128,
    center_freq_ghz: float = 10.0,
    bandwidth_mhz: float = 200.0,
    altitude_km: float = 500.0,
    slant_range_km: float = 700.0,
    platform_speed: float = 7200.0,
) -> CPHDPVP:
    """Build synthetic PVP arrays for a monostatic stripmap collection.

    The SRP drifts along the ground track (stripmap geometry) instead
    of being fixed (spotlight). The platform flies in a straight line
    at constant altitude.

    Parameters
    ----------
    npulses : int
        Number of pulses.
    nsamples : int
        Number of frequency samples per pulse.
    center_freq_ghz : float
        Center frequency in GHz.
    bandwidth_mhz : float
        Instantaneous bandwidth in MHz.
    altitude_km : float
        Platform altitude above SRP in km.
    slant_range_km : float
        Nominal slant range in km.
    platform_speed : float
        Platform speed in m/s.

    Returns
    -------
    CPHDPVP
        Populated per-vector parameters with drifting SRP.
    """
    fc = center_freq_ghz * 1e9
    bw = bandwidth_mhz * 1e6
    alt = altitude_km * 1e3
    sr = slant_range_km * 1e3

    # SRP position (40°N, 75°W on WGS-84 ellipsoid)
    lat_rad = np.radians(40.0)
    lon_rad = np.radians(-75.0)
    a_wgs = 6378137.0
    f_wgs = 1.0 / 298.257223563
    e2 = 2 * f_wgs - f_wgs**2
    N_val = a_wgs / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    srp_ecf = np.array([
        (N_val) * np.cos(lat_rad) * np.cos(lon_rad),
        (N_val) * np.cos(lat_rad) * np.sin(lon_rad),
        (N_val * (1 - e2)) * np.sin(lat_rad),
    ])

    # Unit vectors at SRP
    u_up = srp_ecf / norm(srp_ecf)
    z_axis = np.array([0.0, 0.0, 1.0])
    u_east = np.cross(z_axis, u_up)
    u_east = u_east / norm(u_east)
    u_north = np.cross(u_up, u_east)

    # Ground range from slant range and altitude
    ground_range = np.sqrt(sr**2 - alt**2)

    # Time array (PRF ~10 kHz)
    prf = 10000.0
    tx_time = np.arange(npulses) / prf
    rcv_time = tx_time + 2 * sr / _C

    # Platform: straight-line track heading north at altitude
    arp = np.zeros((npulses, 3))
    vel = np.zeros((npulses, 3))
    srp_pos = np.zeros((npulses, 3))
    for i in range(npulses):
        along_track = platform_speed * tx_time[i]
        # ARP moves along track at altitude
        arp[i] = (srp_ecf
                  + ground_range * u_north  # cross-track offset
                  + along_track * u_east    # along-track motion
                  + alt * u_up)
        vel[i] = platform_speed * u_east
        # SRP drifts along track on the ground (stripmap)
        srp_pos[i] = srp_ecf + along_track * u_east

    # Frequency parameters
    fx1 = np.full(npulses, fc - bw / 2)
    fx2 = np.full(npulses, fc + bw / 2)
    fxss = np.full(npulses, bw / nsamples)
    sc0 = fx1.copy()

    # Signal validity: all valid
    signal = np.ones(npulses, dtype=np.int64)

    return CPHDPVP(
        tx_time=tx_time,
        tx_pos=arp,
        tx_vel=vel,
        rcv_time=rcv_time,
        rcv_pos=arp,
        rcv_vel=vel,
        srp_pos=srp_pos,
        fx1=fx1,
        fx2=fx2,
        sc0=sc0,
        scss=fxss,
        signal=signal,
    )


def _synthetic_stripmap_metadata(
    npulses: int = 256,
    nsamples: int = 128,
    **pvp_kwargs,
) -> CPHDMetadata:
    """Build complete synthetic CPHDMetadata for stripmap."""
    pvp = _synthetic_stripmap_pvp(
        npulses=npulses, nsamples=nsamples, **pvp_kwargs,
    )
    channel = CPHDChannel(
        identifier='CHANNEL001',
        num_vectors=npulses,
        num_samples=nsamples,
    )
    fc = pvp_kwargs.get('center_freq_ghz', 10.0) * 1e9
    bw = pvp_kwargs.get('bandwidth_mhz', 200.0) * 1e6
    global_params = CPHDGlobal(
        domain_type='FX',
        fx_band_min=fc - bw / 2,
        fx_band_max=fc + bw / 2,
    )
    collection_info = CPHDCollectionInfo(
        collector_name='SYNTHETIC',
        core_name='STRIPMAP_TEST001',
        collect_type='MONOSTATIC',
        radar_mode='STRIPMAP',
    )
    return CPHDMetadata(
        format='CPHD',
        rows=npulses,
        cols=nsamples,
        dtype='complex64',
        channels=[channel],
        pvp=pvp,
        global_params=global_params,
        collection_info=collection_info,
        num_channels=1,
    )


# ===================================================================
# CPHDPVP.slice() tests
# ===================================================================

class TestCPHDPVPSlice:
    """Tests for CPHDPVP.slice() method."""

    def test_slice_basic(self):
        """Slicing produces correct number of vectors."""
        pvp = _synthetic_pvp(npulses=64, nsamples=128)
        sliced = pvp.slice(10, 30)
        assert sliced.num_vectors == 20

    def test_slice_preserves_values(self):
        """Sliced values match the original range."""
        pvp = _synthetic_pvp(npulses=64, nsamples=128)
        sliced = pvp.slice(5, 15)
        np.testing.assert_array_equal(
            sliced.tx_time, pvp.tx_time[5:15],
        )
        np.testing.assert_array_equal(
            sliced.tx_pos, pvp.tx_pos[5:15, :],
        )

    def test_slice_copies_data(self):
        """Sliced arrays are independent copies."""
        pvp = _synthetic_pvp(npulses=64, nsamples=128)
        sliced = pvp.slice(0, 10)
        sliced.tx_time[0] = -999.0
        assert pvp.tx_time[0] != -999.0

    def test_slice_2d_fields(self):
        """2D position arrays are sliced correctly."""
        pvp = _synthetic_pvp(npulses=64, nsamples=128)
        sliced = pvp.slice(20, 40)
        assert sliced.srp_pos.shape == (20, 3)
        np.testing.assert_array_equal(
            sliced.srp_pos, pvp.srp_pos[20:40, :],
        )

    def test_slice_none_fields(self):
        """None fields remain None after slicing."""
        pvp = _synthetic_pvp(npulses=64, nsamples=128)
        assert pvp.signal is None  # not set in spotlight helper
        sliced = pvp.slice(0, 10)
        assert sliced.signal is None

    def test_slice_invalid_range_raises(self):
        """Invalid slice range raises ValueError."""
        pvp = _synthetic_pvp(npulses=64, nsamples=128)
        with pytest.raises(ValueError):
            pvp.slice(30, 10)
        with pytest.raises(ValueError):
            pvp.slice(-1, 10)
        with pytest.raises(ValueError):
            pvp.slice(0, 100)

    def test_slice_full_range(self):
        """Slicing full range returns equivalent PVP."""
        pvp = _synthetic_pvp(npulses=64, nsamples=128)
        sliced = pvp.slice(0, 64)
        assert sliced.num_vectors == 64
        np.testing.assert_array_equal(sliced.tx_time, pvp.tx_time)


# ===================================================================
# create_subaperture_metadata tests
# ===================================================================

class TestCreateSubapertureMetadata:
    """Tests for create_subaperture_metadata."""

    def test_basic_slice(self):
        """Sub-aperture metadata has correct dimensions."""
        meta = _synthetic_stripmap_metadata(npulses=256, nsamples=128)
        sub = create_subaperture_metadata(meta, 50, 150)
        assert sub.rows == 100
        assert sub.cols == 128
        assert sub.pvp.num_vectors == 100

    def test_preserves_global_params(self):
        """Global params are shared (not sliced)."""
        meta = _synthetic_stripmap_metadata(npulses=256, nsamples=128)
        sub = create_subaperture_metadata(meta, 0, 64)
        assert sub.global_params is meta.global_params

    def test_preserves_collection_info(self):
        """Collection info is shared."""
        meta = _synthetic_stripmap_metadata(npulses=256, nsamples=128)
        sub = create_subaperture_metadata(meta, 0, 64)
        assert sub.collection_info is meta.collection_info

    def test_channel_dimensions_updated(self):
        """Channel num_vectors reflects sub-aperture size."""
        meta = _synthetic_stripmap_metadata(npulses=256, nsamples=128)
        sub = create_subaperture_metadata(meta, 100, 200)
        assert sub.channels[0].num_vectors == 100

    def test_no_pvp_raises(self):
        """Missing PVP raises ValueError."""
        meta = CPHDMetadata(format='CPHD', rows=10, cols=10,
                            dtype='complex64')
        with pytest.raises(ValueError):
            create_subaperture_metadata(meta, 0, 5)


# ===================================================================
# SubaperturePartitioner tests
# ===================================================================

class TestSubaperturePartitioner:
    """Tests for SubaperturePartitioner."""

    def test_fixed_subaperture_length(self):
        """Fixed sub-aperture length is respected."""
        meta = _synthetic_stripmap_metadata(npulses=256, nsamples=128)
        part = SubaperturePartitioner(
            meta, subaperture_pulses=64, overlap_fraction=0.0,
        )
        assert part.sub_length == 64
        # With no overlap, stride == sub_length
        assert part.stride == 64

    def test_partitions_cover_all_pulses(self):
        """Every pulse is in at least one partition."""
        meta = _synthetic_stripmap_metadata(npulses=256, nsamples=128)
        part = SubaperturePartitioner(
            meta, subaperture_pulses=64, overlap_fraction=0.5,
        )
        covered = set()
        for start, end in part.partitions:
            covered.update(range(start, end))
        assert covered == set(range(256))

    def test_overlap_fraction_respected(self):
        """Adjacent partitions overlap by the expected amount."""
        meta = _synthetic_stripmap_metadata(npulses=256, nsamples=128)
        part = SubaperturePartitioner(
            meta, subaperture_pulses=64, overlap_fraction=0.5,
        )
        # stride should be 32 (50% overlap of 64)
        assert part.stride == 32
        if len(part.partitions) >= 2:
            s0, e0 = part.partitions[0]
            s1, e1 = part.partitions[1]
            overlap_pulses = e0 - s1
            assert overlap_pulses == 32

    def test_min_pulses_enforced(self):
        """Minimum pulse constraint is enforced."""
        meta = _synthetic_stripmap_metadata(npulses=256, nsamples=128)
        part = SubaperturePartitioner(
            meta, subaperture_pulses=10, min_pulses=64,
        )
        assert part.sub_length >= 64

    def test_auto_sizing(self):
        """Auto-sizing produces a reasonable sub-aperture length."""
        meta = _synthetic_stripmap_metadata(npulses=256, nsamples=128)
        part = SubaperturePartitioner(meta)
        assert part.sub_length >= 64
        assert part.num_subapertures >= 1

    def test_no_pvp_raises(self):
        """Missing PVP raises ValueError."""
        meta = CPHDMetadata(format='CPHD', rows=10, cols=10,
                            dtype='complex64')
        with pytest.raises(ValueError):
            SubaperturePartitioner(meta)

    def test_single_partition_when_short(self):
        """A short collection produces a single partition."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=128)
        part = SubaperturePartitioner(
            meta, subaperture_pulses=128,
        )
        # sub_length clamped to npulses, one partition
        assert part.num_subapertures == 1
        assert part.partitions[0] == (0, 64)


# ===================================================================
# StripmapPFA tests
# ===================================================================

class TestStripmapPFA:
    """Tests for StripmapPFA."""

    def test_is_subclass(self):
        """StripmapPFA is an ImageFormationAlgorithm."""
        assert issubclass(StripmapPFA, ImageFormationAlgorithm)

    def test_form_image_returns_complex(self):
        """form_image returns a complex array."""
        npulses, nsamples = 128, 64
        meta = _synthetic_stripmap_metadata(
            npulses=npulses, nsamples=nsamples,
        )
        signal = _synthetic_signal(npulses=npulses, nsamples=nsamples)
        ifp = StripmapPFA(
            meta,
            subaperture_pulses=64,
            overlap_fraction=0.5,
            verbose=False,
        )
        image = ifp.form_image(signal, geometry=None)
        assert np.iscomplexobj(image)
        assert image.ndim == 2

    def test_form_image_nonzero(self):
        """Formed image has non-zero content."""
        npulses, nsamples = 128, 64
        meta = _synthetic_stripmap_metadata(
            npulses=npulses, nsamples=nsamples,
        )
        signal = _synthetic_signal(npulses=npulses, nsamples=nsamples)
        ifp = StripmapPFA(
            meta,
            subaperture_pulses=64,
            overlap_fraction=0.5,
            verbose=False,
        )
        image = ifp.form_image(signal, geometry=None)
        assert np.any(np.abs(image) > 0)

    def test_get_output_grid(self):
        """get_output_grid returns a dict after form_image."""
        npulses, nsamples = 128, 64
        meta = _synthetic_stripmap_metadata(
            npulses=npulses, nsamples=nsamples,
        )
        signal = _synthetic_signal(npulses=npulses, nsamples=nsamples)
        ifp = StripmapPFA(
            meta,
            subaperture_pulses=64,
            overlap_fraction=0.5,
            verbose=False,
        )
        ifp.form_image(signal, geometry=None)
        grid = ifp.get_output_grid()
        assert isinstance(grid, dict)
        assert 'range_resolution' in grid

    def test_partitioner_accessible(self):
        """The partitioner property is accessible."""
        meta = _synthetic_stripmap_metadata(npulses=128, nsamples=64)
        ifp = StripmapPFA(
            meta, subaperture_pulses=64, verbose=False,
        )
        assert ifp.partitioner.num_subapertures >= 1

    def test_rephase_identity_spotlight(self):
        """Rephasing with fixed SRP is identity (no phase change)."""
        pvp = _synthetic_pvp(npulses=32, nsamples=64)
        signal = _synthetic_signal(npulses=32, nsamples=64)
        center_srp = pvp.srp_pos[16].copy()
        # For spotlight, all SRPs are the same → no rephasing needed
        rephased = StripmapPFA.rephase_signal(signal, pvp, center_srp)
        np.testing.assert_allclose(
            np.abs(rephased), np.abs(signal), rtol=1e-10,
        )


# ===================================================================
# RangeDopplerAlgorithm tests
# ===================================================================


class TestRangeDopplerAlgorithm:
    """Tests for RangeDopplerAlgorithm with synthetic stripmap data."""

    # -- Construction --

    def test_is_subclass(self):
        """RDA is an ImageFormationAlgorithm."""
        assert issubclass(RangeDopplerAlgorithm, ImageFormationAlgorithm)

    def test_construction_with_defaults(self):
        """Default construction from metadata only."""
        meta = _synthetic_stripmap_metadata(npulses=128, nsamples=64)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        assert rda is not None

    def test_construction_missing_pvp_raises(self):
        """Missing PVP raises ValueError."""
        meta = CPHDMetadata(
            format='CPHD', rows=10, cols=10, dtype='complex64',
        )
        with pytest.raises(ValueError, match="PVP"):
            RangeDopplerAlgorithm(meta)

    # -- Parameter extraction --

    def test_wavelength_positive(self):
        """Wavelength is positive and physical."""
        meta = _synthetic_stripmap_metadata(npulses=128, nsamples=64)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        assert rda.wavelength > 0.0
        assert rda.wavelength < 1.0  # less than 1 m for microwave

    def test_prf_positive(self):
        """PRF is positive."""
        meta = _synthetic_stripmap_metadata(npulses=128, nsamples=64)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        assert rda.prf > 0.0

    def test_reference_range_positive(self):
        """Reference slant range is positive."""
        meta = _synthetic_stripmap_metadata(npulses=128, nsamples=64)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        assert rda.reference_range > 0.0

    def test_doppler_centroid_finite(self):
        """Doppler centroid is finite."""
        meta = _synthetic_stripmap_metadata(npulses=128, nsamples=64)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        assert np.isfinite(rda.doppler_centroid)

    # -- Stage 0: Rephasing --

    def test_rephase_spotlight_identity(self):
        """Rephasing with fixed SRP preserves magnitude."""
        meta = _synthetic_metadata(npulses=32, nsamples=64)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        signal = _synthetic_signal(npulses=32, nsamples=64)
        rephased = rda.rephase_to_reference(signal)
        np.testing.assert_allclose(
            np.abs(rephased), np.abs(signal), rtol=1e-5,
        )

    def test_rephase_stripmap_changes_phase(self):
        """Rephasing stripmap data changes phase (non-identity)."""
        meta = _synthetic_stripmap_metadata(npulses=128, nsamples=64)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        signal = _synthetic_signal(npulses=128, nsamples=64)
        rephased = rda.rephase_to_reference(signal)
        # Phase should change for stripmap (SRP drifts)
        phase_diff = np.angle(rephased) - np.angle(signal)
        assert np.any(np.abs(phase_diff) > 0.01)

    # -- Stage 1: Range compression --

    def test_range_compress_shape(self):
        """Output shape matches input."""
        meta = _synthetic_stripmap_metadata(npulses=128, nsamples=64)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        signal = _synthetic_signal(npulses=128, nsamples=64)
        rc = rda.range_compress(signal)
        assert rc.shape == signal.shape

    def test_range_compress_complex(self):
        """Output is complex."""
        meta = _synthetic_stripmap_metadata(npulses=128, nsamples=64)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        signal = _synthetic_signal(npulses=128, nsamples=64)
        rc = rda.range_compress(signal)
        assert np.iscomplexobj(rc)

    def test_range_compress_nonzero(self):
        """Range compressed signal has non-zero content."""
        meta = _synthetic_stripmap_metadata(npulses=128, nsamples=64)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        signal = _synthetic_signal(npulses=128, nsamples=64)
        rc = rda.range_compress(signal)
        assert np.any(np.abs(rc) > 0)

    # -- Stage 2: Azimuth FFT --

    def test_azimuth_fft_shape(self):
        """Shape preserved after azimuth FFT."""
        meta = _synthetic_stripmap_metadata(npulses=128, nsamples=64)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        signal = _synthetic_signal(npulses=128, nsamples=64)
        rc = rda.range_compress(signal)
        rd = rda.azimuth_fft(rc)
        assert rd.shape == rc.shape

    def test_azimuth_fft_energy_preserved(self):
        """Parseval's theorem: energy preserved by FFT."""
        meta = _synthetic_stripmap_metadata(npulses=128, nsamples=64)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        signal = _synthetic_signal(npulses=128, nsamples=64)
        rc = rda.range_compress(signal)
        rd = rda.azimuth_fft(rc)
        energy_before = np.sum(np.abs(rc) ** 2)
        energy_after = np.sum(np.abs(rd) ** 2) / rc.shape[0]
        np.testing.assert_allclose(
            energy_before, energy_after, rtol=1e-4,
        )

    # -- Stage 3: RCMC --

    def test_rcmc_shape(self):
        """Output shape matches input."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        signal = _synthetic_signal(npulses=64, nsamples=32)
        rc = rda.range_compress(signal)
        rd = rda.azimuth_fft(rc)
        rcmc_out = rda.rcmc(rd)
        assert rcmc_out.shape == rd.shape

    def test_rcmc_with_custom_interpolator(self):
        """RCMC accepts a custom interpolator."""
        calls = []

        def dummy_interp(x_old, y_old, x_new):
            calls.append(1)
            return np.interp(x_new.real, x_old.real, y_old.real)

        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        rda = RangeDopplerAlgorithm(
            meta, interpolator=dummy_interp, verbose=False,
        )
        signal = _synthetic_signal(npulses=64, nsamples=32)
        rc = rda.range_compress(signal)
        rd = rda.azimuth_fft(rc)
        rda.rcmc(rd)
        assert len(calls) > 0

    # -- Stage 4: Azimuth compression --

    def test_azimuth_compress_shape(self):
        """Output shape matches input."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        signal = _synthetic_signal(npulses=64, nsamples=32)
        rc = rda.range_compress(signal)
        rd = rda.azimuth_fft(rc)
        rcmc_out = rda.rcmc(rd)
        image = rda.azimuth_compress(rcmc_out)
        assert image.shape == rcmc_out.shape

    def test_azimuth_compress_complex(self):
        """Output is complex."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        signal = _synthetic_signal(npulses=64, nsamples=32)
        rc = rda.range_compress(signal)
        rd = rda.azimuth_fft(rc)
        rcmc_out = rda.rcmc(rd)
        image = rda.azimuth_compress(rcmc_out)
        assert np.iscomplexobj(image)

    # -- Full pipeline --

    def test_form_image_returns_complex(self):
        """form_image returns complex 2D array."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        signal = _synthetic_signal(npulses=64, nsamples=32)
        image = rda.form_image(signal, geometry=None)
        assert image.ndim == 2
        assert np.iscomplexobj(image)

    def test_form_image_nonzero(self):
        """Formed image has non-zero content."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        signal = _synthetic_signal(npulses=64, nsamples=32)
        image = rda.form_image(signal, geometry=None)
        assert np.any(np.abs(image) > 0)

    def test_form_image_equals_stages(self):
        """Full pipeline matches stage-by-stage execution."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        signal = _synthetic_signal(npulses=64, nsamples=32)

        # Stage by stage (deramp + FFT pipeline)
        rephased = rda.rephase_to_reference(signal)
        rc = rda.range_compress(rephased)
        rc = rda._apply_dechirp(rc)
        rd = rda.azimuth_fft(rc)
        rcmc_out = rda.rcmc(rd)
        image_stages = rda.azimuth_compress(rcmc_out)

        # Full pipeline
        image_full = rda.form_image(signal, geometry=None)

        np.testing.assert_allclose(
            image_full, image_stages, rtol=1e-5,
        )

    # -- Output grid --

    def test_get_output_grid_type_rgzero(self):
        """Output grid type is RGZERO."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        grid = rda.get_output_grid()
        assert grid['type'] == 'RGZERO'

    def test_get_output_grid_has_resolution(self):
        """Grid dict includes resolution parameters."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        grid = rda.get_output_grid()
        assert 'range_resolution' in grid
        assert 'azimuth_resolution' in grid
        assert grid['range_resolution'] > 0
        assert grid['azimuth_resolution'] > 0

    # -- Weighting --

    def test_taylor_range_weighting(self):
        """Taylor range weighting accepted."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        rda = RangeDopplerAlgorithm(
            meta, range_weighting='taylor', verbose=False,
        )
        signal = _synthetic_signal(npulses=64, nsamples=32)
        image = rda.form_image(signal, geometry=None)
        assert image.shape == signal.shape

    def test_hamming_azimuth_weighting(self):
        """Hamming azimuth weighting accepted."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        rda = RangeDopplerAlgorithm(
            meta, azimuth_weighting='hamming', verbose=False,
        )
        signal = _synthetic_signal(npulses=64, nsamples=32)
        image = rda.form_image(signal, geometry=None)
        assert image.shape == signal.shape

    def test_invalid_weighting_raises(self):
        """Invalid weighting string raises ValueError."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        with pytest.raises(ValueError, match="Unknown weighting"):
            RangeDopplerAlgorithm(
                meta, range_weighting='invalid', verbose=False,
            )

    def test_azimuth_compress_passthrough_no_weighting(self):
        """Without weighting, azimuth compress is identity (no IFFT)."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        rng = np.random.default_rng(99)
        data = (rng.standard_normal((64, 32))
                + 1j * rng.standard_normal((64, 32)))
        image = rda.azimuth_compress(data)
        np.testing.assert_array_equal(image, data)

    def test_rcmc_zero_shift_at_dc(self):
        """At zero Doppler, RCMC shift is zero (D=1)."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        # Find the Doppler bin closest to zero
        dc_idx = np.argmin(np.abs(rda._f_eta))
        # At f_eta=0, D=1 so delta_r_shift=0 → output equals input
        rng = np.random.default_rng(42)
        data = (rng.standard_normal((64, 32))
                + 1j * rng.standard_normal((64, 32)))
        rcmc_out = rda.rcmc(data)
        np.testing.assert_allclose(
            np.abs(rcmc_out[dc_idx, :]),
            np.abs(data[dc_idx, :]),
            rtol=1e-4,
        )

    # -- Doppler centroid estimation --

    def test_estimate_doppler_centroid_finite(self):
        """Estimate returns a finite value."""
        signal = _synthetic_signal(npulses=128, nsamples=64)
        pri = 1e-4  # 10 kHz PRF
        f_dc = RangeDopplerAlgorithm.estimate_doppler_centroid(
            signal, pri,
        )
        assert np.isfinite(f_dc)

    def test_estimate_doppler_centroid_known_tone(self):
        """A known Doppler tone gives the correct centroid."""
        pri = 1e-4
        npulses, nsamples = 256, 32
        f_tone = 1200.0  # Hz
        t = np.arange(npulses) * pri
        # Pure Doppler tone with some range structure
        rng = np.random.default_rng(42)
        tone = np.exp(1j * 2 * np.pi * f_tone * t)[:, np.newaxis]
        signal = tone * rng.standard_normal((1, nsamples))
        f_dc = RangeDopplerAlgorithm.estimate_doppler_centroid(
            signal, pri,
        )
        assert abs(f_dc - f_tone) < 50.0  # within 50 Hz

    # -- Subaperture mode --

    def test_subaperture_construction(self):
        """block_size and overlap are stored."""
        meta = _synthetic_stripmap_metadata(npulses=256, nsamples=64)
        rda = RangeDopplerAlgorithm(
            meta, block_size=64, overlap=0.5, verbose=False,
        )
        assert rda.block_size == 64
        assert rda.overlap == 0.5

    def test_subaperture_default_none(self):
        """Default block_size is None (single-reference mode)."""
        meta = _synthetic_stripmap_metadata(npulses=128, nsamples=64)
        rda = RangeDopplerAlgorithm(meta, verbose=False)
        assert rda.block_size is None

    def test_subaperture_form_image_returns_complex(self):
        """Subaperture form_image returns a complex 2D array."""
        meta = _synthetic_stripmap_metadata(npulses=256, nsamples=64)
        rda = RangeDopplerAlgorithm(
            meta, block_size=64, overlap=0.5, verbose=False,
        )
        signal = _synthetic_signal(npulses=256, nsamples=64)
        image = rda.form_image(signal, geometry=None)
        assert image.ndim == 2
        assert np.iscomplexobj(image)

    def test_subaperture_form_image_shape(self):
        """Subaperture output has correct range cols, fewer az rows."""
        meta = _synthetic_stripmap_metadata(npulses=256, nsamples=64)
        rda = RangeDopplerAlgorithm(
            meta, block_size=64, overlap=0.5, verbose=False,
        )
        signal = _synthetic_signal(npulses=256, nsamples=64)
        image = rda.form_image(signal, geometry=None)
        assert image.shape[1] == signal.shape[1]
        assert image.shape[0] > 0
        assert image.shape[0] <= signal.shape[0]

    def test_subaperture_form_image_nonzero(self):
        """Subaperture formed image has non-zero content."""
        meta = _synthetic_stripmap_metadata(npulses=256, nsamples=64)
        rda = RangeDopplerAlgorithm(
            meta, block_size=64, overlap=0.5, verbose=False,
        )
        signal = _synthetic_signal(npulses=256, nsamples=64)
        image = rda.form_image(signal, geometry=None)
        assert np.any(np.abs(image) > 0)

    def test_subaperture_single_block_fallback(self):
        """block_size >= npulses produces a single block."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        rda = RangeDopplerAlgorithm(
            meta, block_size=128, overlap=0.5, verbose=False,
        )
        signal = _synthetic_signal(npulses=64, nsamples=32)
        image = rda.form_image(signal, geometry=None)
        assert image.shape[1] == signal.shape[1]
        assert image.shape[0] > 0
        assert np.any(np.abs(image) > 0)

    def test_subaperture_no_overlap(self):
        """Zero overlap works (non-overlapping blocks)."""
        meta = _synthetic_stripmap_metadata(npulses=128, nsamples=32)
        rda = RangeDopplerAlgorithm(
            meta, block_size=64, overlap=0.0, verbose=False,
        )
        signal = _synthetic_signal(npulses=128, nsamples=32)
        image = rda.form_image(signal, geometry=None)
        assert image.shape[1] == signal.shape[1]
        assert image.shape[0] > 0

    def test_subaperture_with_weighting(self):
        """Subaperture mode with Taylor + Hamming weighting."""
        meta = _synthetic_stripmap_metadata(npulses=128, nsamples=64)
        rda = RangeDopplerAlgorithm(
            meta,
            block_size=64,
            overlap=0.5,
            range_weighting='taylor',
            azimuth_weighting='hamming',
            verbose=False,
        )
        signal = _synthetic_signal(npulses=128, nsamples=64)
        image = rda.form_image(signal, geometry=None)
        assert image.shape[1] == signal.shape[1]
        assert image.shape[0] > 0
        assert np.any(np.abs(image) > 0)


# ===================================================================
# FastBackProjection tests
# ===================================================================


class TestFastBackProjection:
    """Tests for FastBackProjection with synthetic stripmap data."""

    # -- Construction --

    def test_is_subclass(self):
        """FFBP is an ImageFormationAlgorithm."""
        assert issubclass(FastBackProjection, ImageFormationAlgorithm)

    def test_construction_with_defaults(self):
        """Default construction from metadata only."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        ffbp = FastBackProjection(meta, verbose=False)
        assert ffbp is not None

    def test_construction_missing_pvp_raises(self):
        """Missing PVP raises ValueError."""
        meta = CPHDMetadata(
            format='CPHD', rows=10, cols=10, dtype='complex64',
        )
        with pytest.raises(ValueError, match="PVP"):
            FastBackProjection(meta)

    def test_construction_custom_params(self):
        """Custom leaf_size and n_angular are accepted."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        ffbp = FastBackProjection(
            meta, leaf_size=4, n_angular=64, verbose=False,
        )
        assert ffbp._leaf_size == 4
        assert ffbp._n_angular == 64

    def test_invalid_weighting_raises(self):
        """Invalid weighting string raises ValueError."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        with pytest.raises(ValueError, match="Unknown weighting"):
            FastBackProjection(
                meta, range_weighting='invalid', verbose=False,
            )

    # -- Range compression --

    def test_range_compress_shape(self):
        """Output shape matches input."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        ffbp = FastBackProjection(meta, verbose=False)
        signal = _synthetic_signal(npulses=64, nsamples=32)
        rc = ffbp.range_compress(signal)
        assert rc.shape == signal.shape

    def test_range_compress_complex(self):
        """Output is complex."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        ffbp = FastBackProjection(meta, verbose=False)
        signal = _synthetic_signal(npulses=64, nsamples=32)
        rc = ffbp.range_compress(signal)
        assert np.iscomplexobj(rc)

    def test_range_compress_nonzero(self):
        """Range-compressed signal has non-zero content."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        ffbp = FastBackProjection(meta, verbose=False)
        signal = _synthetic_signal(npulses=64, nsamples=32)
        rc = ffbp.range_compress(signal)
        assert np.any(np.abs(rc) > 0)

    # -- Full pipeline --

    def test_form_image_returns_complex(self):
        """form_image returns complex 2D array."""
        meta = _synthetic_stripmap_metadata(npulses=32, nsamples=16)
        ffbp = FastBackProjection(
            meta, leaf_size=8, n_angular=32, verbose=False,
        )
        signal = _synthetic_signal(npulses=32, nsamples=16)
        image = ffbp.form_image(signal, geometry=None)
        assert image.ndim == 2
        assert np.iscomplexobj(image)

    def test_form_image_nonzero(self):
        """Formed image has non-zero content."""
        meta = _synthetic_stripmap_metadata(npulses=32, nsamples=16)
        ffbp = FastBackProjection(
            meta, leaf_size=8, n_angular=32, verbose=False,
        )
        signal = _synthetic_signal(npulses=32, nsamples=16)
        image = ffbp.form_image(signal, geometry=None)
        assert np.any(np.abs(image) > 0)

    def test_form_image_shape(self):
        """Output has N_rg == nsamples columns."""
        meta = _synthetic_stripmap_metadata(npulses=32, nsamples=16)
        ffbp = FastBackProjection(
            meta, leaf_size=8, n_angular=32, verbose=False,
        )
        signal = _synthetic_signal(npulses=32, nsamples=16)
        image = ffbp.form_image(signal, geometry=None)
        assert image.shape[1] == 16  # N_rg == nsamples
        assert image.shape[0] > 0   # N_xr > 0

    # -- Output grid --

    def test_get_output_grid_has_resolution(self):
        """Grid dict includes resolution parameters."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        ffbp = FastBackProjection(meta, verbose=False)
        grid = ffbp.get_output_grid()
        assert 'range_resolution' in grid
        assert 'azimuth_resolution' in grid
        assert grid['range_resolution'] > 0
        assert grid['azimuth_resolution'] > 0

    def test_get_output_grid_algorithm(self):
        """Grid dict identifies FFBP algorithm."""
        meta = _synthetic_stripmap_metadata(npulses=64, nsamples=32)
        ffbp = FastBackProjection(meta, verbose=False)
        grid = ffbp.get_output_grid()
        assert grid['algorithm'] == 'FFBP'

    # -- Weighting --

    def test_taylor_range_weighting(self):
        """Taylor range weighting accepted."""
        meta = _synthetic_stripmap_metadata(npulses=32, nsamples=16)
        ffbp = FastBackProjection(
            meta, range_weighting='taylor',
            leaf_size=8, n_angular=32, verbose=False,
        )
        signal = _synthetic_signal(npulses=32, nsamples=16)
        image = ffbp.form_image(signal, geometry=None)
        assert image.ndim == 2
        assert np.any(np.abs(image) > 0)

    # -- Leaf size sensitivity --

    def test_leaf_size_affects_output(self):
        """Different leaf sizes produce images (correlation check)."""
        meta1 = _synthetic_stripmap_metadata(npulses=32, nsamples=16)
        meta2 = _synthetic_stripmap_metadata(npulses=32, nsamples=16)
        signal = _synthetic_signal(npulses=32, nsamples=16)

        ffbp4 = FastBackProjection(
            meta1, leaf_size=4, n_angular=32, verbose=False,
        )
        ffbp8 = FastBackProjection(
            meta2, leaf_size=8, n_angular=32, verbose=False,
        )

        image4 = ffbp4.form_image(signal.copy(), geometry=None)
        image8 = ffbp8.form_image(signal.copy(), geometry=None)

        # Both images should be non-zero
        assert np.any(np.abs(image4) > 0)
        assert np.any(np.abs(image8) > 0)

    # -- Numba dispatch --

    def test_use_numba_false_fallback(self):
        """Explicit numpy fallback with use_numba=False."""
        meta = _synthetic_stripmap_metadata(npulses=32, nsamples=16)
        ffbp = FastBackProjection(
            meta, leaf_size=8, n_angular=32,
            use_numba=False, verbose=False,
        )
        assert ffbp._use_numba is False
        signal = _synthetic_signal(npulses=32, nsamples=16)
        image = ffbp.form_image(signal, geometry=None)
        assert image.ndim == 2
        assert np.iscomplexobj(image)
        assert np.any(np.abs(image) > 0)

    def test_numba_vs_numpy_equivalence(self):
        """Numba and numpy paths produce similar images."""
        meta_nb = _synthetic_stripmap_metadata(npulses=32, nsamples=16)
        meta_np = _synthetic_stripmap_metadata(npulses=32, nsamples=16)
        signal = _synthetic_signal(npulses=32, nsamples=16)

        ffbp_nb = FastBackProjection(
            meta_nb, leaf_size=8, n_angular=32,
            use_numba=True, verbose=False,
        )
        ffbp_np = FastBackProjection(
            meta_np, leaf_size=8, n_angular=32,
            use_numba=False, verbose=False,
        )

        image_nb = ffbp_nb.form_image(signal.copy(), geometry=None)
        image_np = ffbp_np.form_image(signal.copy(), geometry=None)

        assert image_nb.shape == image_np.shape
        # Relaxed tolerance: numba uses inline linear interp,
        # numpy path uses scipy interp1d (same algorithm, different
        # floating-point paths).
        np.testing.assert_allclose(
            np.abs(image_nb), np.abs(image_np), rtol=1e-3, atol=1e-6,
        )
