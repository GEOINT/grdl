# -*- coding: utf-8 -*-
"""
Tests for the CPHD → SICDMetadata builder.

Confirms that ``build_sicd_metadata`` populates every SICD section
sarpy validates and that the resulting metadata round-trips through
``SICDWriter._sicd_metadata_to_sarpy`` without raising.

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
    CPHDChannel,
    CPHDChannelParameters,
    CPHDChannelSection,
    CPHDCollectionInfo,
    CPHDGlobal,
    CPHDMetadata,
    CPHDPolarization,
    CPHDPVP,
    CPHDRcvParameters,
    CPHDTimeline,
    CPHDTxRcv,
    CPHDTxWaveform,
)
from grdl.IO.sar.cphd_to_sicd import build_sicd_metadata
from grdl.image_processing.sar.image_formation import (
    CollectionGeometry,
    PolarFormatAlgorithm,
    PolarGrid,
)


# ===================================================================
# Synthetic CPHD fixture
# ===================================================================

_C = 299792458.0


def _synthetic_pvp(npulses: int = 32, nsamples: int = 64) -> CPHDPVP:
    """Build a synthetic PVP block: spotlight arc over a fixed SRP."""
    srp_lat, srp_lon, srp_alt = 0.0, 0.0, 0.0
    R_e = 6378137.0
    srp_ecf = np.array([
        (R_e + srp_alt) * np.cos(np.radians(srp_lat)) * np.cos(np.radians(srp_lon)),
        (R_e + srp_alt) * np.cos(np.radians(srp_lat)) * np.sin(np.radians(srp_lon)),
        (R_e + srp_alt) * np.sin(np.radians(srp_lat)),
    ])

    sr = 50_000.0
    alt = 30_000.0
    half_angle = np.radians(2.0)
    angles = np.linspace(-half_angle, half_angle, npulses)

    u_up = srp_ecf / np.linalg.norm(srp_ecf)
    z_axis = np.array([0.0, 0.0, 1.0])
    u_east = np.cross(z_axis, u_up)
    u_east /= np.linalg.norm(u_east)
    u_north = np.cross(u_up, u_east)

    ground_range = np.sqrt(sr**2 - alt**2)
    arp = np.zeros((npulses, 3))
    for i, ang in enumerate(angles):
        arp[i] = (
            srp_ecf
            + ground_range * np.cos(ang) * u_north
            + ground_range * np.sin(ang) * u_east
            + alt * u_up
        )

    speed = 200.0
    vel = np.zeros((npulses, 3))
    for i, ang in enumerate(angles):
        vel[i] = speed * (-np.sin(ang) * u_north + np.cos(ang) * u_east)

    arc_length = sr * 2.0 * half_angle
    total_time = arc_length / speed
    tx_time = np.linspace(0.0, total_time, npulses)
    rcv_time = tx_time + 2.0 * sr / _C

    fc = 10.0e9
    bw = 600.0e6
    fx1 = np.full(npulses, fc - bw / 2.0)
    fx2 = np.full(npulses, fc + bw / 2.0)
    fxss = np.full(npulses, bw / nsamples)
    sc0 = fx1.copy()

    return CPHDPVP(
        tx_time=tx_time,
        tx_pos=arp,
        tx_vel=vel,
        rcv_time=rcv_time,
        rcv_pos=arp,
        rcv_vel=vel,
        srp_pos=np.tile(srp_ecf, (npulses, 1)),
        fx1=fx1,
        fx2=fx2,
        sc0=sc0,
        scss=fxss,
    )


def _synthetic_cphd(npulses: int = 32, nsamples: int = 64) -> CPHDMetadata:
    """Build a fully-populated synthetic CPHDMetadata."""
    pvp = _synthetic_pvp(npulses, nsamples)
    fc = 10.0e9
    bw = 600.0e6

    return CPHDMetadata(
        format='CPHD',
        rows=npulses,
        cols=nsamples,
        dtype='complex64',
        collection_info=CPHDCollectionInfo(
            collector_name='SynSat-1',
            core_name='RUN-0001',
            collect_type='MONOSTATIC',
            radar_mode='SPOTLIGHT',
            radar_mode_id='SPOT01',
            classification='UNCLASSIFIED',
            release_info='UNRESTRICTED',
            country_code='US',
        ),
        global_params=CPHDGlobal(
            domain_type='FX',
            phase_sgn=-1,
            timeline=CPHDTimeline(
                collection_start='2025-04-15T12:00:00.000000Z',
                tx_time1=0.0,
                tx_time2=float(pvp.tx_time[-1]),
            ),
            fx_band_min=fc - bw / 2.0,
            fx_band_max=fc + bw / 2.0,
        ),
        channel_section=CPHDChannelSection(
            ref_ch_id='Ch1',
            parameters=[CPHDChannelParameters(
                identifier='Ch1',
                ref_vector_index=0,
                fx_c=fc,
                fx_bw=bw,
                polarization=CPHDPolarization(tx_pol='V', rcv_pol='V'),
            )],
        ),
        tx_rcv=CPHDTxRcv(
            tx_waveforms=[CPHDTxWaveform(
                identifier='WF1',
                pulse_length=10.0e-6,
                rf_bandwidth=bw,
                freq_center=fc,
                lfm_rate=bw / 10.0e-6,
                polarization='V',
            )],
            rcv_parameters=[CPHDRcvParameters(
                identifier='Rcv1',
                window_length=20.0e-6,
                sample_rate=2.0 * bw,
                if_filter_bw=bw,
                freq_center=fc,
                polarization='V',
            )],
        ),
        pvp=pvp,
        channels=[CPHDChannel(
            identifier='Ch1',
            num_vectors=npulses,
            num_samples=nsamples,
        )],
        num_channels=1,
        tx_waveform=None,
        rcv_parameters=None,
    )


@pytest.fixture(scope='module')
def cphd_meta() -> CPHDMetadata:
    return _synthetic_cphd()


@pytest.fixture(scope='module')
def geometry(cphd_meta: CPHDMetadata) -> CollectionGeometry:
    return CollectionGeometry(cphd_meta, slant=True)


@pytest.fixture(scope='module')
def pfa(geometry: CollectionGeometry) -> PolarFormatAlgorithm:
    grid = PolarGrid(geometry)
    return PolarFormatAlgorithm(grid=grid, phase_sgn=-1)


# ===================================================================
# Builder coverage
# ===================================================================

def test_collection_info_populated(cphd_meta, geometry) -> None:
    sicd = build_sicd_metadata(
        cphd_meta, geometry, image_shape=(64, 64),
        image_form_algo='PFA',
    )
    ci = sicd.collection_info
    assert ci is not None
    assert ci.collector_name == 'SynSat-1'
    assert ci.core_name == 'RUN-0001'
    assert ci.classification == 'UNCLASSIFIED'
    assert ci.collect_type == 'MONOSTATIC'
    assert ci.radar_mode is not None
    assert ci.radar_mode.mode_type == 'SPOTLIGHT'


def test_image_data_populated(cphd_meta, geometry) -> None:
    sicd = build_sicd_metadata(
        cphd_meta, geometry, image_shape=(80, 100),
    )
    idata = sicd.image_data
    assert idata.num_rows == 80
    assert idata.num_cols == 100
    assert idata.pixel_type == 'RE32F_IM32F'
    assert idata.scp_pixel.row == 40
    assert idata.scp_pixel.col == 50
    assert idata.full_image.num_rows == 80


def test_geo_data_populated(cphd_meta, geometry) -> None:
    sicd = build_sicd_metadata(cphd_meta, geometry, image_shape=(64, 64))
    gd = sicd.geo_data
    assert gd is not None
    assert gd.earth_model == 'WGS_84'
    assert gd.scp is not None
    assert gd.scp.ecf is not None
    assert gd.scp.llh is not None
    assert gd.image_corners is not None
    assert len(gd.image_corners) == 4


def test_grid_populated(cphd_meta, geometry, pfa) -> None:
    sicd = build_sicd_metadata(
        cphd_meta, geometry, image_shape=(64, 64),
        grid_params=pfa.get_output_grid(),
    )
    grid = sicd.grid
    assert grid is not None
    assert grid.image_plane in ('SLANT', 'GROUND')
    assert grid.type in ('RGAZIM', 'RGZERO', 'XRGYCR', 'XCTYAT', 'PLANE')
    assert grid.row is not None
    assert grid.row.uvect_ecf is not None
    assert grid.row.ss is not None and grid.row.ss > 0
    assert grid.row.imp_resp_bw is not None
    assert grid.col is not None
    assert grid.col.uvect_ecf is not None
    assert grid.col.ss is not None and grid.col.ss > 0


def test_timeline_populated(cphd_meta, geometry) -> None:
    sicd = build_sicd_metadata(cphd_meta, geometry, image_shape=(64, 64))
    tl = sicd.timeline
    assert tl is not None
    assert tl.collect_start.startswith('2025-04-15')
    assert tl.collect_duration is not None
    assert tl.collect_duration > 0


def test_position_arp_poly(cphd_meta, geometry) -> None:
    sicd = build_sicd_metadata(cphd_meta, geometry, image_shape=(64, 64))
    pos = sicd.position
    assert pos is not None
    assert pos.arp_poly is not None
    assert pos.arp_poly.x is not None
    assert pos.arp_poly.x.coefs is not None
    # 5th-order fit -> 6 coefficients per axis
    assert len(pos.arp_poly.x.coefs) == 6
    assert len(pos.arp_poly.y.coefs) == 6
    assert len(pos.arp_poly.z.coefs) == 6


def test_scpcoa_populated(cphd_meta, geometry) -> None:
    sicd = build_sicd_metadata(cphd_meta, geometry, image_shape=(64, 64))
    sc = sicd.scpcoa
    assert sc is not None
    assert sc.scp_time is not None
    assert sc.arp_pos is not None
    assert sc.arp_vel is not None
    assert sc.side_of_track in ('L', 'R')
    assert sc.slant_range > 0
    assert 0.0 <= sc.graze_ang <= 90.0
    assert 0.0 <= sc.incidence_ang <= 90.0
    assert -180.0 <= sc.azim_ang <= 360.0
    assert sc.arp_acc is not None  # numerical differentiation succeeded


def test_radar_collection_populated(cphd_meta, geometry) -> None:
    sicd = build_sicd_metadata(cphd_meta, geometry, image_shape=(64, 64))
    rc = sicd.radar_collection
    assert rc is not None
    assert rc.tx_frequency is not None
    assert rc.tx_frequency.min < rc.tx_frequency.max
    assert rc.waveform is not None and len(rc.waveform) == 1
    assert rc.waveform[0].tx_pulse_length == 10.0e-6
    assert rc.tx_polarization == 'V'
    assert rc.rcv_channels is not None and len(rc.rcv_channels) == 1
    assert rc.rcv_channels[0].tx_rcv_polarization == 'V:V'


def test_image_formation_populated(cphd_meta, geometry) -> None:
    sicd = build_sicd_metadata(
        cphd_meta, geometry, image_shape=(64, 64),
        image_form_algo='PFA',
    )
    imf = sicd.image_formation
    assert imf is not None
    assert imf.image_form_algo == 'PFA'
    assert imf.t_start_proc is not None
    assert imf.t_end_proc is not None
    assert imf.t_end_proc > imf.t_start_proc
    assert imf.tx_frequency_proc is not None
    assert imf.rcv_chan_proc is not None
    assert imf.rcv_chan_proc.num_chan_proc == 1


def test_invalid_algo_rejected(cphd_meta, geometry) -> None:
    with pytest.raises(ValueError, match='image_form_algo'):
        build_sicd_metadata(
            cphd_meta, geometry, image_shape=(64, 64),
            image_form_algo='RDA',  # not a SICD enum value
        )


def test_pfa_to_sicd_metadata_method(cphd_meta, geometry, pfa) -> None:
    """PolarFormatAlgorithm.to_sicd_metadata wires through to the builder."""
    sicd = pfa.to_sicd_metadata(cphd_meta, geometry, image_shape=(64, 64))
    assert sicd.collection_info is not None
    assert sicd.image_formation.image_form_algo == 'PFA'
    assert sicd.grid.row.ss is not None
    assert sicd.grid.col.ss is not None


# ===================================================================
# Sarpy round-trip — does sarpy accept what we built?
# ===================================================================

def test_sarpy_roundtrip(cphd_meta, geometry, pfa) -> None:
    """Resulting SICDMetadata converts to sarpy SICDType without raising."""
    from grdl.IO.sar.sicd_writer import _sicd_metadata_to_sarpy

    sicd_meta = pfa.to_sicd_metadata(cphd_meta, geometry, image_shape=(64, 64))

    sicd_type = _sicd_metadata_to_sarpy(sicd_meta)
    # Sarpy populates derived fields and runs internal validation when
    # asked. ``derive`` triggers cross-section consistency checks that
    # would fail loudly on missing required fields.
    sicd_type.derive()

    # Spot-check that key sections survived the conversion
    assert sicd_type.CollectionInfo is not None
    assert sicd_type.CollectionInfo.CollectorName == 'SynSat-1'
    assert sicd_type.ImageData is not None
    assert sicd_type.ImageData.NumRows == 64
    assert sicd_type.ImageData.NumCols == 64
    assert sicd_type.GeoData is not None
    assert sicd_type.GeoData.SCP is not None
    assert sicd_type.Grid is not None
    assert sicd_type.Grid.Row is not None
    assert sicd_type.SCPCOA is not None
    assert sicd_type.RadarCollection is not None
    assert sicd_type.ImageFormation is not None
    assert sicd_type.ImageFormation.ImageFormAlgo == 'PFA'
    assert sicd_type.Position is not None
    assert sicd_type.Position.ARPPoly is not None
