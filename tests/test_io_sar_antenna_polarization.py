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

from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pytest

from grdl.geolocation.ecf_sez import ecf_to_sez
from grdl.IO.models.common import XYZ, Poly1D, XYZPoly
from grdl.IO.models.sicd import (
    SICDSCP,
    SICDAntenna,
    SICDAntParam,
    SICDGeoData,
    SICDImageFormation,
    SICDMetadata,
    SICDPosition,
    SICDRadarCollection,
    SICDRcvChannel,
    SICDRcvChanProc,
    SICDTimeline,
)
from grdl.IO.sar.get_antenna_polarization import (
    _eval_vector_poly,
    _extract_antenna_polynomials,
    _get_scp_position,
    get_antenna_polarization,
    get_antenna_polarization_from_file,
)


def _meta(rcv_pols, chan_indices, proc=None, tx="H"):
    return SICDMetadata(
        format="SICD",
        rows=10,
        cols=10,
        dtype="complex64",
        timeline=SICDTimeline(collect_duration=1.0),
        position=SICDPosition(
            arp_poly=XYZPoly(
                x=Poly1D(coefs=[7000000.0]),
                y=Poly1D(coefs=[0.0]),
                z=Poly1D(coefs=[0.0]),
            )
        ),
        geo_data=SICDGeoData(scp=SICDSCP(ecf=XYZ(6378137.0, 0.0, 0.0))),
        antenna=SICDAntenna(
            tx=SICDAntParam(
                x_axis_poly=XYZPoly(
                    x=Poly1D(coefs=[1.0]),
                    y=Poly1D(coefs=[0.0]),
                    z=Poly1D(coefs=[0.0]),
                ),
                y_axis_poly=XYZPoly(
                    x=Poly1D(coefs=[0.0]),
                    y=Poly1D(coefs=[1.0]),
                    z=Poly1D(coefs=[0.0]),
                ),
            )
        ),
        radar_collection=SICDRadarCollection(
            tx_polarization=tx,
            rcv_channels=[
                SICDRcvChannel(tx_rcv_polarization=p, index=i + 1) for i, p in enumerate(rcv_pols)
            ],
        ),
        image_formation=SICDImageFormation(
            rcv_chan_proc=SICDRcvChanProc(chan_indices=chan_indices),
            tx_rcv_polarization_proc=proc,
        ),
    )


@pytest.mark.parametrize("vectors", [[1.0, 2.0, 3.0], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
def test_ecf_to_sez_equatorial_rotation(vectors):
    source = np.asarray(vectors)
    expected = np.stack([-source[..., 2], source[..., 1], source[..., 0]], axis=-1)
    np.testing.assert_array_equal(ecf_to_sez(source, 0.0, 0.0), expected.flatten())


@pytest.mark.parametrize("shape", [(2,), (3, 1), (1, 1, 3)])
def test_ecf_to_sez_rejects_invalid_vector_shapes(shape):
    with pytest.raises(ValueError, match="shape|1D or 2D"):
        ecf_to_sez(np.zeros(shape), 0.0, 0.0)


def test_antenna_polynomial_coefficients_are_ascending():
    meta = _meta(["H:H"], [1])
    meta.antenna.tx.x_axis_poly = XYZPoly(
        x=Poly1D(coefs=[1.0, 2.0]),
        y=Poly1D(coefs=[3.0, 4.0]),
        z=Poly1D(coefs=[5.0, 6.0]),
    )
    x_coefs, _ = _extract_antenna_polynomials(meta)
    np.testing.assert_array_equal(_eval_vector_poly(x_coefs, 0.25), [1.5, 4.0, 6.5])


def test_scene_position_accepts_dictionary_polynomials():
    meta = {"position": {"grp_poly": {"x": [1.0, 2.0], "y": [3.0], "z": [4.0]}}}
    np.testing.assert_array_equal(_get_scp_position(meta, 0.25), [1.5, 3.0, 4.0])


@pytest.mark.parametrize("proc, indices", [(None, [1]), ("H:Y", [1]), (None, [1, 2])])
def test_dictionary_metadata_matches_dataclass(proc, indices):
    meta = _meta(["H:H", "H:V"], indices, proc=proc)
    assert get_antenna_polarization(asdict(meta)) == get_antenna_polarization(meta)


class _StubReader:
    def __init__(self, meta):
        self.metadata = meta

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _from_file(meta, **kwargs):
    with patch("grdl.IO.sar.sicd.SICDReader", lambda path: _StubReader(meta)):
        return get_antenna_polarization_from_file("scene.nitf", **kwargs)


class TestFromFile:

    def test_matches_get_antenna_polarization(self):
        meta = _meta(["H:V"], [1])
        pol, angles, _ = _from_file(meta)
        assert (pol, angles) == get_antenna_polarization(meta)
        assert pol == "HV"

    @pytest.mark.parametrize("rx", ["H", "V"])
    def test_no_rotation_for_linear_rx(self, rx):
        assert _from_file(_meta([f"H:{rx}"], [1]))[2] is None

    def test_rotation_for_x_rx(self):
        pol, angles, rotation = _from_file(_meta(["H:X"], [1]))
        assert rotation == angles["rx"] - 0.0

    def test_rotation_for_y_rx(self):
        pol, angles, rotation = _from_file(_meta(["H:Y"], [1]))
        assert rotation == angles["rx"] - 90.0

    def test_processed_polarization_preferred(self):
        meta = _meta(["H:H"], [1], proc="H:Y")
        _, angles, rotation = _from_file(meta)
        assert rotation == angles["rx"] - 90.0

    def test_multi_channel_returns_list(self):
        meta = _meta(["H:H", "H:X", "H:Y"], [1, 2, 3])
        pol, angles, rotation = _from_file(meta)
        assert len(pol) == len(angles["rx"]) == len(rotation) == 3
        assert rotation[0] is None
        assert rotation[1] == angles["rx"][1]
        assert rotation[2] == angles["rx"][2] - 90.0

    def test_bare_two_letter_value(self):
        assert _from_file(_meta(["HX"], [1]))[2] is not None

    def test_options_forwarded(self):
        # X at 0 deg is within tolerance of H, so the mapped label is 'HH'
        pol, _, _ = _from_file(_meta(["H:X"], [1]), ang_tol=5.0)
        assert pol == "HH"

    def test_missing_metadata_raises(self):
        meta = _meta(["H:H"], [1])
        meta.antenna = None
        with pytest.raises(ValueError):
            _from_file(meta)
