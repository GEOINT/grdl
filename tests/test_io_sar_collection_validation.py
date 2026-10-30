# -*- coding: utf-8 -*-
"""
Tests for grdl.IO.sar.collection_validation.

Author
------
Jason Fritz
43161141+stryder-vtx@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-10-07

Modified
--------
2026-10-07
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from grdl.IO.models.base import ChannelMetadata
from grdl.IO.models.common import LatLonHAE, XYZ, XYZPoly
from grdl.IO.models.sicd import (
    SICDAntParam, SICDAntenna, SICDCollectionInfo, SICDDirParam, SICDGeoData,
    SICDGrid, SICDImageFormation, SICDMetadata, SICDPosition,
    SICDRadarCollection, SICDRcvChanProc, SICDRcvChannel, SICDSCP,
    SICDSCPCOA, SICDTimeline,
)
from grdl.IO.sar.collection_validation import (
    CollectionValidationError, ReciprocityError, check_reciprocity,
    check_reciprocity_reader, classify_polarizations, parse_tx_rcv,
    sicd_polarization, validate_collection, validate_sicd_collection,
)

QUAD = ['HH', 'HV', 'VH', 'VV']


def _meta(tx_rcv=None, antenna_tx=None, **over):
    kw = dict(
        format='SICD', rows=100, cols=200, dtype='complex64',
        collection_info=SICDCollectionInfo(
            collector_name='SAT-1', core_name='CORE1'),
        timeline=SICDTimeline(collect_start='2026-01-01T00:00:00.000000Z'),
        geo_data=SICDGeoData(scp=SICDSCP(
            ecf=XYZ(1e6, 2e6, 3e6), llh=LatLonHAE(10.0, 20.0, 0.0))),
        grid=SICDGrid(row=SICDDirParam(ss=0.5), col=SICDDirParam(ss=0.4)),
        scpcoa=SICDSCPCOA(side_of_track='R', graze_ang=30.0,
                          doppler_cone_ang=90.0),
    )
    if tx_rcv is not None:
        kw['radar_collection'] = SICDRadarCollection(
            rcv_channels=[SICDRcvChannel(tx_rcv_polarization=tx_rcv, index=1)])
    if antenna_tx is not None:
        # Everything get_antenna_polarization requires.
        channels = (kw.get('radar_collection') or SICDRadarCollection()).rcv_channels
        kw['radar_collection'] = SICDRadarCollection(
            tx_polarization=antenna_tx, rcv_channels=channels)
        kw['timeline'] = SICDTimeline(
            collect_start='2026-01-01T00:00:00.000000Z', collect_duration=1.0)
        kw['position'] = SICDPosition(arp_poly=XYZPoly())
        kw['antenna'] = SICDAntenna(tx=SICDAntParam())
        kw['image_formation'] = SICDImageFormation(
            rcv_chan_proc=SICDRcvChanProc(chan_indices=[1]))
    kw.update(over)
    return SICDMetadata(**kw)


def _quad_metas():
    return [_meta(tx_rcv=f'{p[0]}:{p[1]}') for p in QUAD]


def _codes(report, level=None):
    return {i.code for i in report.issues if level in (None, i.level)}


class TestClassify:

    @pytest.mark.parametrize('pols,mode', [
        (QUAD, 'quad'), (['HH', 'HV', 'VV'], 'triple'),
        (['VV', 'VH'], 'dual'), (['HH'], 'single'),
        (['RH', 'RV'], 'compact'), (['UNKNOWN'], 'other'), ([], 'other'),
        (['hh', 'hv', 'vh', 'vv'], 'quad'),
    ])
    def test_modes(self, pols, mode):
        assert classify_polarizations(pols) == mode


class TestSicdPolarization:

    @pytest.mark.parametrize('text,expected', [
        ('H:V', ('H', 'V')), ('v:v', ('V', 'V')), ('HV', ('H', 'V')),
        ('RHC:H', ('R', 'H')), ('LHC:LHC', ('L', 'L')), ('X:Y', ('X', 'Y')),
        ('OTHER', None), ('SEQUENCE', None), ('UNSPECIFIED:OTHER', None),
        ('', None), (None, None),
    ])
    def test_parse_tx_rcv(self, text, expected):
        assert parse_tx_rcv(text) == expected

    def test_single_rcv_channel(self):
        pol = sicd_polarization(_meta(tx_rcv='H:V'))
        assert (pol.tx, pol.rcv, pol.label) == ('H', 'V', 'HV')
        assert 'rcv_channels' in pol.source

    def test_processed_polarization_wins(self):
        meta = _meta(
            tx_rcv='H:H',
            image_formation=SICDImageFormation(tx_rcv_polarization_proc='V:V'))
        pol = sicd_polarization(meta)
        assert pol.label == 'VV' and 'proc' in pol.source

    def test_processed_polarization_list_of_one(self):
        meta = _meta(image_formation=SICDImageFormation(
            tx_rcv_polarization_proc=['V:H']))
        assert sicd_polarization(meta).label == 'VH'

    def test_chan_indices_select_channel(self):
        meta = _meta(
            radar_collection=SICDRadarCollection(rcv_channels=[
                SICDRcvChannel(tx_rcv_polarization='H:H', index=1),
                SICDRcvChannel(tx_rcv_polarization='H:V', index=2),
            ]),
            image_formation=SICDImageFormation(
                rcv_chan_proc=SICDRcvChanProc(chan_indices=[2])))
        assert sicd_polarization(meta).label == 'HV'

    def test_multiple_channels_without_indices_is_ambiguous(self):
        meta = _meta(radar_collection=SICDRadarCollection(rcv_channels=[
            SICDRcvChannel(tx_rcv_polarization='H:H', index=1),
            SICDRcvChannel(tx_rcv_polarization='V:V', index=2),
        ]))
        assert sicd_polarization(meta) is None

    def test_compact_pol_label(self):
        assert sicd_polarization(_meta(tx_rcv='RHC:H')).label == 'RH'

    def test_unrecognized_value(self):
        assert sicd_polarization(_meta(tx_rcv='OTHER')) is None
        assert sicd_polarization(_meta()) is None

    def test_labels_taken_from_metadata_when_omitted(self):
        rep = validate_sicd_collection(_quad_metas())
        assert rep.polarizations == QUAD and rep.is_quad_pol and not rep.issues

    def test_unresolvable_channel_is_unknown(self):
        rep = validate_sicd_collection([_meta(tx_rcv='H:H'), _meta()])
        assert rep.polarizations == ['HH', 'UNKNOWN']
        assert 'polarization_not_in_metadata' in _codes(rep, 'warning')

    def test_label_contradicting_metadata_is_error(self):
        metas = _quad_metas()
        labels = ['HH', 'VH', 'HV', 'VV']
        rep = validate_sicd_collection(metas, labels)
        assert 'polarization_label_mismatch' in _codes(rep, 'error')
        assert any('HV' in i.message for i in rep.errors)

    def test_unverifiable_label_is_warning(self):
        rep = validate_sicd_collection([_meta(), _meta()], ['HH', 'VV'])
        assert rep.ok
        assert 'polarization_not_in_metadata' in _codes(rep, 'warning')

    def test_proc_and_channel_disagreement_warns(self):
        meta = _meta(
            tx_rcv='H:H',
            image_formation=SICDImageFormation(tx_rcv_polarization_proc='V:V'))
        rep = validate_sicd_collection([meta], ['VV'])
        assert 'polarization_fields_disagree' in _codes(rep, 'warning')


class TestAntennaPolarization:
    """Labels are also verified with get_antenna_polarization."""

    def test_consistent_quad_has_no_issues(self):
        metas = [_meta(tx_rcv=f'{p[0]}:{p[1]}', antenna_tx=p[0]) for p in QUAD]
        rep = validate_sicd_collection(metas, QUAD)
        assert rep.ok and not rep.issues

    def test_label_checked_when_only_antenna_resolves(self):
        # Two receive channels and no ChanIndex: sicd_polarization cannot
        # choose, get_antenna_polarization defaults to the first channel.
        channels = [SICDRcvChannel(tx_rcv_polarization='H:H', index=1),
                    SICDRcvChannel(tx_rcv_polarization='H:V', index=2)]
        meta = _meta(antenna_tx='H',
                     radar_collection=SICDRadarCollection(
                         tx_polarization='H', rcv_channels=channels),
                     image_formation=SICDImageFormation(
                         rcv_chan_proc=SICDRcvChanProc(chan_indices=None)))
        assert sicd_polarization(meta) is None
        assert 'polarization_not_in_metadata' not in _codes(
            validate_sicd_collection([meta], ['HH']))
        rep = validate_sicd_collection([meta], ['VV'])
        assert 'polarization_label_mismatch' in _codes(rep, 'error')
        assert any('get_antenna_polarization says HH' in i.message
                   for i in rep.errors)

    def test_labels_derived_from_antenna_when_omitted(self):
        channels = [SICDRcvChannel(tx_rcv_polarization='H:V', index=1),
                    SICDRcvChannel(tx_rcv_polarization='V:V', index=2)]
        meta = _meta(antenna_tx='H',
                     radar_collection=SICDRadarCollection(
                         tx_polarization='H', rcv_channels=channels),
                     image_formation=SICDImageFormation(
                         rcv_chan_proc=SICDRcvChanProc(chan_indices=None)))
        assert validate_sicd_collection([meta]).polarizations == ['HV']

    def test_rotated_platform_accepts_both_labels(self):
        # X:X in the metadata is HH once get_antenna_polarization maps it.
        meta = _meta(tx_rcv='X:X', antenna_tx='X')
        for label in ('XX', 'HH'):
            rep = validate_sicd_collection([meta], [label])
            assert rep.ok and not rep.issues
        rep = validate_sicd_collection([meta], ['VV'])
        assert 'polarization_label_mismatch' in _codes(rep, 'error')

    def test_tx_polarization_disagreement_warns(self):
        meta = _meta(tx_rcv='H:H', antenna_tx='V')
        rep = validate_sicd_collection([meta], ['HH'])
        assert rep.ok
        assert 'antenna_polarization_disagrees' in _codes(rep, 'warning')

    def test_sequence_transmit_leaves_metadata_label(self):
        meta = _meta(tx_rcv='H:V', antenna_tx='SEQUENCE')
        rep = validate_sicd_collection([meta], ['HV'])
        assert rep.ok and not rep.issues

    def test_missing_antenna_falls_back_to_metadata(self):
        rep = validate_sicd_collection([_meta(tx_rcv='H:V')], ['HV'])
        assert rep.ok and not rep.issues


class TestValidateSicd:

    def test_matching_quad_ok(self):
        rep = validate_sicd_collection(_quad_metas(), QUAD)
        assert rep.ok and rep.is_quad_pol and not rep.issues

    def test_dimension_mismatch(self):
        metas = [_meta(), _meta(rows=101), _meta(), _meta()]
        assert 'dimension_mismatch' in _codes(
            validate_sicd_collection(metas, QUAD), 'error')

    def test_collector_mismatch(self):
        other = _meta(collection_info=SICDCollectionInfo(
            collector_name='SAT-2', core_name='CORE1'))
        rep = validate_sicd_collection([_meta(), other], ['HH', 'VV'])
        assert 'collector_mismatch' in _codes(rep, 'error')

    def test_time_mismatch(self):
        later = _meta(timeline=SICDTimeline(
            collect_start='2026-01-01T00:00:30Z'))
        rep = validate_sicd_collection([_meta(), later], ['HH', 'VV'])
        assert 'time_mismatch' in _codes(rep, 'error')

    def test_time_within_tolerance(self):
        near = _meta(timeline=SICDTimeline(
            collect_start='2026-01-01T00:00:00.5Z'))
        rep = validate_sicd_collection([_meta(), near], ['HH', 'VV'])
        assert rep.ok

    def test_scp_mismatch(self):
        far = _meta(geo_data=SICDGeoData(scp=SICDSCP(
            ecf=XYZ(1e6 + 500, 2e6, 3e6))))
        rep = validate_sicd_collection([_meta(), far], ['HH', 'VV'])
        assert 'scp_mismatch' in _codes(rep, 'error')

    def test_spacing_and_angle_mismatch(self):
        bad = _meta(
            grid=SICDGrid(row=SICDDirParam(ss=0.6), col=SICDDirParam(ss=0.4)),
            scpcoa=SICDSCPCOA(side_of_track='R', graze_ang=35.0,
                              doppler_cone_ang=90.0))
        errs = _codes(validate_sicd_collection(
            [_meta(), bad], ['HH', 'VV']), 'error')
        assert {'row_spacing_mismatch', 'graze_ang_mismatch'} <= errs

    def test_side_of_track_mismatch(self):
        left = _meta(scpcoa=SICDSCPCOA(side_of_track='L', graze_ang=30.0,
                                       doppler_cone_ang=90.0))
        rep = validate_sicd_collection([_meta(), left], ['HH', 'VV'])
        assert 'side_of_track_mismatch' in _codes(rep, 'error')

    def test_core_name_difference_is_warning(self):
        other = _meta(collection_info=SICDCollectionInfo(
            collector_name='SAT-1', core_name='CORE2'))
        rep = validate_sicd_collection([_meta(), other], ['HH', 'VV'])
        assert rep.ok and 'core_name_differs' in _codes(rep, 'warning')

    def test_duplicate_and_unknown_pol(self):
        rep = validate_sicd_collection(
            [_meta(), _meta()], ['HH', 'HH'])
        assert 'duplicate_polarization' in _codes(rep, 'error')
        rep = validate_sicd_collection([_meta(), _meta()], ['HH', 'UNKNOWN'])
        assert 'unknown_polarization' in _codes(rep, 'warning')

    def test_missing_metadata_warns_not_errors(self):
        bare = [_meta(timeline=None, geo_data=None, scpcoa=None, grid=None,
                      collection_info=None) for _ in range(2)]
        rep = validate_sicd_collection(bare, ['HH', 'VV'])
        assert rep.ok and rep.warnings

    def test_raise_if_invalid(self):
        rep = validate_sicd_collection(
            [_meta(), _meta(rows=5)], ['HH', 'VV'])
        with pytest.raises(CollectionValidationError, match='dimension'):
            rep.raise_if_invalid()


def _reader(cube, labels, per_file=None, axis_order='CYX'):
    chans = [ChannelMetadata(index=i, name=p, polarization=p)
             for i, p in enumerate(labels)]
    meta = SimpleNamespace(
        rows=cube.shape[-2], cols=cube.shape[-1], channel_metadata=chans,
        axis_order=axis_order, per_file_metadata=per_file)
    return SimpleNamespace(
        metadata=meta,
        read_chip=lambda r0, r1, c0, c1: cube[:, r0:r1, c0:c1])


class TestValidateReader:

    def test_collection_reader_uses_cross_file_checks(self):
        cube = np.zeros((2, 4, 4), np.complex64)
        rdr = _reader(cube, ['HH', 'VV'], per_file=[_meta(), _meta(rows=9)])
        assert 'dimension_mismatch' in _codes(validate_collection(rdr))

    def test_single_file_ok(self):
        rdr = _reader(np.zeros((4, 4, 4), np.complex64), QUAD)
        rep = validate_collection(rdr)
        assert rep.ok and rep.is_quad_pol

    def test_axis_order_error(self):
        rdr = _reader(np.zeros((2, 4, 4), np.complex64), ['HH', 'VV'],
                      axis_order='YX')
        assert 'axis_order' in _codes(validate_collection(rdr), 'error')


def _speckle(shape, seed):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape))


class TestReciprocity:

    def test_reciprocal_passes(self):
        hv = _speckle((64, 64), 0)
        noise = 0.05 * _speckle((64, 64), 1)
        res = check_reciprocity(hv, hv + noise)
        assert res.applicable and res.passed
        assert res.correlation > 0.95 and abs(res.phase_deg) < 5

    def test_independent_fails_with_warning(self):
        with pytest.warns(UserWarning, match='reciprocity'):
            res = check_reciprocity(_speckle((64, 64), 0),
                                    _speckle((64, 64), 1))
        assert not res.passed

    def test_strict_raises(self):
        with pytest.raises(ReciprocityError):
            check_reciprocity(_speckle((32, 32), 0), _speckle((32, 32), 1),
                              strict=True)

    def test_phase_offset_fails(self):
        hv = _speckle((64, 64), 0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = check_reciprocity(hv, hv * np.exp(1j * np.pi / 2))
        assert not res.passed and abs(abs(res.phase_deg) - 90) < 1

    def test_amplitude_imbalance_fails(self):
        hv = _speckle((64, 64), 0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = check_reciprocity(hv, 3.0 * hv)
        assert not res.passed and res.amplitude_ratio_db < -5

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            check_reciprocity(np.ones((2, 2)), np.ones((3, 3)))

    def test_all_zero(self):
        with pytest.warns(UserWarning, match='no non-zero pixels'):
            res = check_reciprocity(np.zeros((4, 4)), np.zeros((4, 4)))
        assert not res.passed and res.n_pixels == 0

    def test_all_zero_strict_raises(self):
        with pytest.raises(ReciprocityError, match='no non-zero pixels'):
            check_reciprocity(np.zeros((4, 4)), np.zeros((4, 4)), strict=True)

    def test_zero_padding_does_not_change_statistics(self):
        hv = _speckle((16, 16), 2)
        expected = check_reciprocity(hv, hv)
        padded = np.pad(hv, 8)
        actual = check_reciprocity(padded, padded)
        assert actual.passed and actual.n_pixels == expected.n_pixels
        assert actual.correlation == pytest.approx(expected.correlation)
        assert actual.phase_deg == pytest.approx(expected.phase_deg)
        assert actual.amplitude_ratio_db == pytest.approx(expected.amplitude_ratio_db)

    @pytest.mark.parametrize('zero_channel', ['HV', 'VH'])
    def test_one_zero_channel_fails(self, zero_channel):
        hv, vh = np.ones((4, 4)), np.ones((4, 4))
        if zero_channel == 'HV':
            hv[:] = 0
        else:
            vh[:] = 0
        with pytest.raises(ReciprocityError, match='correlation'):
            check_reciprocity(hv, vh, strict=True)

    @pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize('channel', ['HV', 'VH'])
    def test_nonfinite_input_is_rejected(self, value, channel):
        hv, vh = np.ones((4, 4)), np.ones((4, 4))
        (hv if channel == 'HV' else vh)[0, 0] = value
        with pytest.raises(ValueError, match='finite'):
            check_reciprocity(hv, vh)

    def test_reader_quad(self):
        hv = _speckle((600, 700), 0)
        cube = np.stack([hv * 0, hv, hv, hv * 0]).astype(np.complex64)
        res = check_reciprocity_reader(_reader(cube, QUAD))
        assert res.applicable and res.passed

    def test_reader_dual_not_applicable(self):
        cube = np.zeros((2, 8, 8), np.complex64)
        res = check_reciprocity_reader(_reader(cube, ['HH', 'HV']))
        assert not res.applicable and res.passed
