# -*- coding: utf-8 -*-
"""
Smoke tests for the typed CPHD 1.1.0 metadata model.

Verifies dataclass construction, the legacy convenience projections
(``meta.tx_waveform``, ``meta.dwell.cod_time_poly``,
``meta.antenna_pattern``), and the ``create_subaperture_metadata``
factory carrying every section through unchanged.

Author
------
Duane Smalley, PhD

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-05-05
"""

# Standard library
# (none)

# Third-party
import numpy as np
import pytest

# GRDL internal
from grdl.IO.models.cphd import (
    CPHDAntCoordFrame,
    CPHDAntenna,
    CPHDAntEB,
    CPHDAntGainPhasePoly,
    CPHDAntPatternFull,
    CPHDAntPhaseCenter,
    CPHDChannel,
    CPHDChannelParameters,
    CPHDChannelSection,
    CPHDCODTime,
    CPHDCollectionInfo,
    CPHDData,
    CPHDDwell,
    CPHDDwellPolynomial,
    CPHDDwellTime,
    CPHDErrorParameters,
    CPHDGeoInfo,
    CPHDGlobal,
    CPHDIonoParameters,
    CPHDMatchCollection,
    CPHDMatchInfo,
    CPHDMatchType,
    CPHDMetadata,
    CPHDMonoRadarSensorError,
    CPHDMonostaticError,
    CPHDMonostaticGeometry,
    CPHDParameter,
    CPHDPolarization,
    CPHDPosVelErr,
    CPHDProductInfo,
    CPHDPVP,
    CPHDReferenceGeometry,
    CPHDSceneCoordinates,
    CPHDSupportArrays,
    CPHDTimeline,
    CPHDTropoParameters,
    CPHDTxRcv,
    CPHDTxWaveform,
    CPHDRcvParameters,
    create_subaperture_metadata,
)


# ===================================================================
# Fixture: a fully-populated CPHDMetadata
# ===================================================================

def _full_metadata(n_pulses: int = 16, n_samples: int = 32) -> CPHDMetadata:
    """Build a CPHDMetadata with every spec section populated."""
    rng = np.random.default_rng(0)

    # PVP arrays — all required + a sample of optional
    pvp = CPHDPVP(
        tx_time=np.linspace(0.0, 1.0, n_pulses),
        rcv_time=np.linspace(0.001, 1.001, n_pulses),
        tx_pos=rng.standard_normal((n_pulses, 3)) * 1e6,
        tx_vel=rng.standard_normal((n_pulses, 3)) * 1e3,
        rcv_pos=rng.standard_normal((n_pulses, 3)) * 1e6,
        rcv_vel=rng.standard_normal((n_pulses, 3)) * 1e3,
        srp_pos=rng.standard_normal((n_pulses, 3)) * 1e3,
        fx1=np.full(n_pulses, 9e9),
        fx2=np.full(n_pulses, 10e9),
        toa1=np.zeros(n_pulses),
        toa2=np.full(n_pulses, 1e-5),
        td_tropo_srp=np.full(n_pulses, 5e-9),
        sc0=np.zeros(n_pulses),
        scss=np.full(n_pulses, 1e6),
        a_fdop=np.zeros(n_pulses),
        a_frr1=np.zeros(n_pulses),
        a_frr2=np.zeros(n_pulses),
        signal=np.ones(n_pulses, dtype=np.int64),
        amp_sf=np.ones(n_pulses),
        td_iono_srp=np.full(n_pulses, 1e-10),
        added_pvps={'CustomA': np.linspace(2.0, 3.0, n_pulses)},
    )

    channels = [CPHDChannel(
        identifier='Ch1',
        num_vectors=n_pulses,
        num_samples=n_samples,
        signal_array_byte_offset=0,
        pvp_array_byte_offset=1024,
        compressed_signal_size=None,
    )]

    data = CPHDData(
        signal_array_format='CF8',
        num_bytes_pvp=240,
        num_cphd_channels=1,
        signal_compression_id=None,
        channels=channels,
        num_support_arrays=0,
        support_arrays=[],
    )

    channel_section = CPHDChannelSection(
        ref_ch_id='Ch1',
        fx_fixed_cphd=True,
        toa_fixed_cphd=True,
        srp_fixed_cphd=True,
        parameters=[CPHDChannelParameters(
            identifier='Ch1',
            ref_vector_index=0,
            fx_fixed=True,
            toa_fixed=True,
            srp_fixed=True,
            polarization=CPHDPolarization(tx_pol='V', rcv_pol='V'),
            fx_c=9.5e9,
            fx_bw=1e9,
            toa_saved=1e-5,
        )],
    )

    tx_rcv = CPHDTxRcv(
        num_tx_wfs=1,
        tx_waveforms=[CPHDTxWaveform(
            identifier='WF1',
            pulse_length=10e-6,
            rf_bandwidth=1e9,
            freq_center=9.5e9,
            lfm_rate=1e14,
            polarization='V',
            power=1e3,
        )],
        num_rcvs=1,
        rcv_parameters=[CPHDRcvParameters(
            identifier='Rcv1',
            window_length=20e-6,
            sample_rate=2e9,
            if_filter_bw=1e9,
            freq_center=9.5e9,
            polarization='V',
            path_gain=10.0,
        )],
    )

    antenna = CPHDAntenna(
        num_acfs=1,
        num_apcs=1,
        num_ant_pats=1,
        ant_coord_frames=[CPHDAntCoordFrame(
            identifier='ACF1',
            x_axis_poly=np.array([[1.0, 0.0, 0.0]]),
            y_axis_poly=np.array([[0.0, 1.0, 0.0]]),
        )],
        ant_phase_centers=[CPHDAntPhaseCenter(
            identifier='APC1',
            acf_id='ACF1',
            apc_xyz=np.zeros(3),
        )],
        ant_patterns=[CPHDAntPatternFull(
            identifier='AP1',
            freq_zero=9.5e9,
            gain_zero=30.0,
            eb=CPHDAntEB(
                dcx_poly=np.array([0.0]),
                dcy_poly=np.array([0.0]),
            ),
            array=CPHDAntGainPhasePoly(
                gain_poly=np.zeros((1, 1)),
                phase_poly=np.zeros((1, 1)),
            ),
            element=CPHDAntGainPhasePoly(
                gain_poly=np.zeros((1, 1)),
                phase_poly=np.zeros((1, 1)),
            ),
        )],
    )

    dwell = CPHDDwell(
        num_cod_times=1,
        cod_times=[CPHDCODTime(
            identifier='COD1',
            cod_time_poly=np.zeros((2, 2)),
        )],
        num_dwell_times=1,
        dwell_times=[CPHDDwellTime(
            identifier='DT1',
            dwell_time_poly=np.full((2, 2), 1e-3),
        )],
    )

    reference_geometry = CPHDReferenceGeometry(
        ref_time=0.5,
        srp_ecf=np.array([6.378e6, 0.0, 0.0]),
        srp_iac=np.zeros(3),
        srp_llh=np.array([0.0, 0.0, 0.0]),
        srp_cod_time=0.5,
        srp_dwell_time=1e-3,
        side_of_track='R',
        graze_angle_deg=30.0,
        azimuth_angle_deg=120.0,
        twist_angle_deg=0.0,
        slope_angle_deg=30.0,
        layover_angle_deg=180.0,
        monostatic=CPHDMonostaticGeometry(
            arp_pos=np.array([7e6, 0.0, 0.0]),
            arp_vel=np.array([0.0, 7e3, 0.0]),
            side_of_track='R',
            slant_range=6.5e5,
            ground_range=5.7e5,
            doppler_cone_angle=85.0,
            graze_angle=30.0,
            incidence_angle=60.0,
            azimuth_angle=120.0,
            twist_angle=0.0,
            slope_angle=30.0,
            layover_angle=180.0,
        ),
    )

    error_parameters = CPHDErrorParameters(
        monostatic=CPHDMonostaticError(
            pos_vel_err=CPHDPosVelErr(
                frame='ECF',
                p1=1.0, p2=1.0, p3=1.0,
                v1=0.1, v2=0.1, v3=0.1,
            ),
            radar_sensor=CPHDMonoRadarSensorError(range_bias=0.5),
        ),
    )

    return CPHDMetadata(
        format='CPHD',
        rows=n_pulses,
        cols=n_samples,
        dtype='complex64',
        collection_info=CPHDCollectionInfo(
            collector_name='TestSensor',
            illuminator_name=None,
            core_name='Run00001',
            collect_type='MONOSTATIC',
            radar_mode='SPOTLIGHT',
            radar_mode_id='SPOT01',
            classification='UNCLASSIFIED',
            release_info='UNRESTRICTED',
            country_code='US',
            parameters=[CPHDParameter(name='Origin', value='unit-test')],
        ),
        global_params=CPHDGlobal(
            domain_type='FX',
            phase_sgn=-1,
            timeline=CPHDTimeline(
                collection_start='2025-01-01T00:00:00Z',
                tx_time1=0.0,
                tx_time2=1.0,
            ),
            fx_band_min=9e9,
            fx_band_max=10e9,
            toa_swath_min=0.0,
            toa_swath_max=1e-5,
            tropo_parameters=CPHDTropoParameters(n0=300.0, ref_height='IARP'),
            iono_parameters=CPHDIonoParameters(tecv=10.0, f2_height=350e3),
        ),
        scene_coordinates=CPHDSceneCoordinates(earth_model='WGS_84'),
        data=data,
        channel_section=channel_section,
        pvp=pvp,
        dwell=dwell,
        reference_geometry=reference_geometry,
        support_arrays=CPHDSupportArrays(),
        antenna=antenna,
        tx_rcv=tx_rcv,
        error_parameters=error_parameters,
        product_info=CPHDProductInfo(profile='Test'),
        geo_info=[CPHDGeoInfo(name='AOI')],
        match_info=CPHDMatchInfo(
            num_match_types=1,
            match_types=[CPHDMatchType(
                index=1,
                type_id='MT1',
                num_match_collections=0,
                match_collections=[CPHDMatchCollection(
                    index=1, core_name='OtherRun',
                )],
            )],
        ),
        channels=channels,
        num_channels=1,
        tx_waveform=tx_rcv.tx_waveforms[0],
        rcv_parameters=tx_rcv.rcv_parameters[0],
    )


# ===================================================================
# Tests
# ===================================================================

def test_pvp_required_fields_constructed() -> None:
    """All required PVP arrays — including TDTropoSRP — are present."""
    meta = _full_metadata()
    pvp = meta.pvp
    for name in (
        'tx_time', 'tx_pos', 'tx_vel', 'rcv_time', 'rcv_pos', 'rcv_vel',
        'srp_pos', 'fx1', 'fx2', 'toa1', 'toa2', 'td_tropo_srp',
        'sc0', 'scss', 'a_fdop', 'a_frr1', 'a_frr2',
    ):
        assert getattr(pvp, name) is not None, f'{name} is unexpectedly None'


def test_pvp_optional_fields_constructed() -> None:
    """Optional PVP arrays we set are propagated."""
    meta = _full_metadata()
    pvp = meta.pvp
    assert pvp.signal is not None
    assert pvp.amp_sf is not None
    assert pvp.td_iono_srp is not None
    assert 'CustomA' in pvp.added_pvps


def test_pvp_slice_handles_new_fields() -> None:
    """``CPHDPVP.slice`` round-trips every populated field."""
    meta = _full_metadata(n_pulses=16)
    sub = meta.pvp.slice(2, 7)
    assert sub.num_vectors == 5
    assert sub.tx_pos.shape == (5, 3)
    assert sub.td_tropo_srp.shape == (5,)
    assert sub.signal.shape == (5,)
    assert sub.added_pvps['CustomA'].shape == (5,)


def test_pvp_trim_to_valid_uses_signal() -> None:
    """``trim_to_valid`` skips leading invalid pulses (signal == 0)."""
    pvp = CPHDPVP(
        tx_time=np.arange(10.0),
        signal=np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64),
    )
    trimmed = pvp.trim_to_valid()
    assert trimmed.num_vectors == 8
    assert np.allclose(trimmed.tx_time, np.arange(2.0, 10.0))


def test_legacy_tx_waveform_projection() -> None:
    """``meta.tx_waveform`` mirrors the first ``tx_rcv.tx_waveforms``."""
    meta = _full_metadata()
    assert meta.tx_waveform is meta.tx_rcv.tx_waveforms[0]
    assert meta.tx_waveform.lfm_rate == 1e14
    assert meta.tx_waveform.identifier == 'WF1'


def test_legacy_rcv_parameters_projection() -> None:
    """``meta.rcv_parameters`` mirrors the first ``tx_rcv.rcv_parameters``."""
    meta = _full_metadata()
    assert meta.rcv_parameters is meta.tx_rcv.rcv_parameters[0]
    assert meta.rcv_parameters.window_length == 20e-6


def test_dwell_legacy_poly_properties() -> None:
    """``dwell.cod_time_poly`` proxies to the first list entry."""
    meta = _full_metadata()
    assert meta.dwell.cod_time_poly is meta.dwell.cod_times[0].cod_time_poly
    assert (
        meta.dwell.dwell_time_poly is meta.dwell.dwell_times[0].dwell_time_poly
    )


def test_cphd_dwell_polynomial_alias() -> None:
    """``CPHDDwellPolynomial`` is preserved as an importable alias."""
    assert CPHDDwellPolynomial is CPHDDwell


def test_global_bandwidth_and_center_freq_properties() -> None:
    """``CPHDGlobal.bandwidth`` and ``.center_frequency`` derive from FX band."""
    meta = _full_metadata()
    assert meta.global_params.bandwidth == 1e9
    assert meta.global_params.center_frequency == 9.5e9


def test_collection_info_extended_fields() -> None:
    """New CollectionID fields (release_info, country_code, parameters) round-trip."""
    meta = _full_metadata()
    assert meta.collection_info.release_info == 'UNRESTRICTED'
    assert meta.collection_info.country_code == 'US'
    assert len(meta.collection_info.parameters) == 1
    assert meta.collection_info.parameters[0].name == 'Origin'


def test_data_section_has_signal_format() -> None:
    """``meta.data`` exposes signal layout fields."""
    meta = _full_metadata()
    assert meta.data.signal_array_format == 'CF8'
    assert meta.data.num_bytes_pvp == 240
    assert meta.data.channels[0].pvp_array_byte_offset == 1024


def test_channel_section_parameters_populated() -> None:
    """``channel_section.parameters`` carries the full ChannelParametersType."""
    meta = _full_metadata()
    cp = meta.channel_section.parameters[0]
    assert cp.fx_c == 9.5e9
    assert cp.polarization.tx_pol == 'V'


def test_reference_geometry_monostatic_full() -> None:
    """Monostatic geometry exposes ARPPos, slant range, etc."""
    meta = _full_metadata()
    mono = meta.reference_geometry.monostatic
    assert mono is not None
    assert mono.slant_range == 6.5e5
    assert mono.doppler_cone_angle == 85.0


def test_create_subaperture_metadata_threads_all_sections() -> None:
    """Subaperture factory preserves every spec section."""
    meta = _full_metadata(n_pulses=20)
    sub = create_subaperture_metadata(meta, start_pulse=4, end_pulse=12)

    assert sub.rows == 8
    assert sub.cols == meta.cols
    assert sub.pvp.num_vectors == 8
    assert sub.pvp.td_tropo_srp.shape == (8,)
    assert sub.channels[0].num_vectors == 8
    # Sections passed through unchanged
    assert sub.collection_info is meta.collection_info
    assert sub.global_params is meta.global_params
    assert sub.scene_coordinates is meta.scene_coordinates
    assert sub.channel_section is meta.channel_section
    assert sub.dwell is meta.dwell
    assert sub.reference_geometry is meta.reference_geometry
    assert sub.support_arrays is meta.support_arrays
    assert sub.antenna is meta.antenna
    assert sub.tx_rcv is meta.tx_rcv
    assert sub.error_parameters is meta.error_parameters
    assert sub.product_info is meta.product_info
    assert sub.match_info is meta.match_info
    # Data is rebuilt with sliced channel sizes
    assert sub.data.signal_array_format == meta.data.signal_array_format
    assert sub.data.channels[0].num_vectors == 8


def test_create_subaperture_requires_pvp() -> None:
    """Factory raises if PVP arrays are missing."""
    meta = CPHDMetadata(format='CPHD', rows=0, cols=0, dtype='complex64')
    with pytest.raises(ValueError):
        create_subaperture_metadata(meta, 0, 1)


def test_pvp_slice_validates_indices() -> None:
    """Slice rejects invalid index ranges."""
    pvp = CPHDPVP(tx_time=np.arange(10.0))
    with pytest.raises(ValueError):
        pvp.slice(5, 5)
    with pytest.raises(ValueError):
        pvp.slice(-1, 3)
    with pytest.raises(ValueError):
        pvp.slice(0, 11)
