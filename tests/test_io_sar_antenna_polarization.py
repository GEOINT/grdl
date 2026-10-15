# -*- coding: utf-8 -*-
"""
Tests for grdl.IO.sar.get_antenna_polarization.

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

from unittest.mock import patch

import pytest

from grdl.IO.models.common import XYZ, XYZPoly
from grdl.IO.models.sicd import (
    SICDAntParam, SICDAntenna, SICDGeoData, SICDImageFormation,
    SICDMetadata, SICDPosition, SICDRadarCollection, SICDRcvChanProc,
    SICDRcvChannel, SICDSCP, SICDTimeline,
)
from grdl.IO.sar.get_antenna_polarization import (
    get_antenna_polarization, get_antenna_polarization_from_file,
)


def _meta(rcv_pols, chan_indices, proc=None, tx='H'):
    return SICDMetadata(
        format='SICD', rows=10, cols=10, dtype='complex64',
        timeline=SICDTimeline(collect_duration=1.0),
        position=SICDPosition(arp_poly=XYZPoly()),
        geo_data=SICDGeoData(scp=SICDSCP(ecf=XYZ(1.0, 2.0, 3.0))),
        antenna=SICDAntenna(tx=SICDAntParam()),
        radar_collection=SICDRadarCollection(
            tx_polarization=tx,
            rcv_channels=[
                SICDRcvChannel(tx_rcv_polarization=p, index=i + 1)
                for i, p in enumerate(rcv_pols)
            ],
        ),
        image_formation=SICDImageFormation(
            rcv_chan_proc=SICDRcvChanProc(chan_indices=chan_indices),
            tx_rcv_polarization_proc=proc,
        ),
    )


class _StubReader:
    def __init__(self, meta):
        self.metadata = meta

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _from_file(meta, **kwargs):
    with patch('grdl.IO.sar.sicd.SICDReader', lambda path: _StubReader(meta)):
        return get_antenna_polarization_from_file('scene.nitf', **kwargs)


class TestFromFile:

    def test_matches_get_antenna_polarization(self):
        meta = _meta(['H:V'], [1])
        pol, angles, _ = _from_file(meta)
        assert (pol, angles) == get_antenna_polarization(meta)
        assert pol == 'HV'

    @pytest.mark.parametrize('rx', ['H', 'V'])
    def test_no_rotation_for_linear_rx(self, rx):
        assert _from_file(_meta([f'H:{rx}'], [1]))[2] is None

    def test_rotation_for_x_rx(self):
        pol, angles, rotation = _from_file(_meta(['H:X'], [1]))
        assert rotation == angles['rx'] - 0.0

    def test_rotation_for_y_rx(self):
        pol, angles, rotation = _from_file(_meta(['H:Y'], [1]))
        assert rotation == angles['rx'] - 90.0

    def test_processed_polarization_preferred(self):
        meta = _meta(['H:H'], [1], proc='H:Y')
        _, angles, rotation = _from_file(meta)
        assert rotation == angles['rx'] - 90.0

    def test_multi_channel_returns_list(self):
        meta = _meta(['H:H', 'H:X', 'H:Y'], [1, 2, 3])
        pol, angles, rotation = _from_file(meta)
        assert len(pol) == len(angles['rx']) == len(rotation) == 3
        assert rotation[0] is None
        assert rotation[1] == angles['rx'][1]
        assert rotation[2] == angles['rx'][2] - 90.0

    def test_bare_two_letter_value(self):
        assert _from_file(_meta(['HX'], [1]))[2] is not None

    def test_options_forwarded(self):
        # X at 0 deg is within tolerance of H, so the mapped label is 'HH'
        pol, _, _ = _from_file(_meta(['H:X'], [1]), ang_tol=5.0)
        assert pol == 'HH'

    def test_missing_metadata_raises(self):
        meta = _meta(['H:H'], [1])
        meta.antenna = None
        with pytest.raises(ValueError):
            _from_file(meta)
