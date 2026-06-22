# -*- coding: utf-8 -*-
"""
Tests for the RSM error-model TRE parsers (RSMPIA/DCA/ECA/APA + B).

Builds synthetic fixed-width CEDATA strings field-by-field and
synthetic ``xml:TRE``-style documents for every TRE, then exercises
both entry points of each parser, the malformed-input -> None
contract, the RSMECA never-raise contract, summarize_accuracy
numerics, and the B-variant ``variant`` flag.

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-06-09

Modified
--------
2026-06-09
"""

# Standard library
from xml.etree import ElementTree as ET

# Third-party
import numpy as np
import pytest

# GRDL internal
from grdl.IO.eo._tre_rsm_error import (
    parse_rsmapa,
    parse_rsmapa_cedata,
    parse_rsmapb,
    parse_rsmapb_cedata,
    parse_rsmdca,
    parse_rsmdca_cedata,
    parse_rsmdcb,
    parse_rsmdcb_cedata,
    parse_rsmeca,
    parse_rsmeca_cedata,
    parse_rsmecb,
    parse_rsmecb_cedata,
    parse_rsmpia,
    parse_rsmpia_cedata,
    summarize_accuracy,
)
from grdl.IO.models.rsm_error import (
    RSMAPAMetadata,
    RSMDCAMetadata,
    RSMECAMetadata,
    RSMPIAMetadata,
)


# ===================================================================
# Fixed-width field builders
# ===================================================================


def f21(value: float) -> str:
    """21-byte signed real, e.g. '+1.50000000000000E+00'."""
    s = f'{value:+1.14E}'
    assert len(s) == 21
    return s


IDENTITY_9 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

IID_A = 'IMG_A'.ljust(80)
EDITION_A = 'ED_001'.ljust(40)
TID_A = 'TRIG_9'.ljust(40)


def _xml(s: str) -> ET.Element:
    return ET.fromstring(s)


def _field(name: str, value) -> str:
    return f'<field name="{name}" value="{value}"/>'


def _repeated(name: str, group_bodies) -> str:
    groups = ''.join(
        f'<group index="{i}">{body}</group>'
        for i, body in enumerate(group_bodies)
    )
    return (f'<repeated name="{name}" number="{len(group_bodies)}">'
            f'{groups}</repeated>')


# -------------------------------------------------------------------
# RSMPIA fixtures
# -------------------------------------------------------------------

PIA_ROW = [100.0, 1.5, -2.0, 0.25, 0.0, 0.001, 0.0, -0.002, 0.0, 0.0]
PIA_COL = [200.0, -1.0, 3.0, 0.5, 0.01, 0.0, 0.0, 0.0, 0.003, 0.0]


def build_rsmpia_cedata() -> str:
    s = IID_A + EDITION_A
    s += ''.join(f21(v) for v in PIA_ROW)
    s += ''.join(f21(v) for v in PIA_COL)
    s += '002' + '003' + '006'
    s += f21(2048.0) + f21(4096.0)
    assert len(s) == 591
    return s


def build_rsmpia_xml() -> ET.Element:
    terms = ('0', 'X', 'Y', 'Z', 'XX', 'XY', 'XZ', 'YY', 'YZ', 'ZZ')
    body = _field('IID', 'IMG_A') + _field('EDITION', 'ED_001')
    body += ''.join(
        _field(f'R{t}', v) for t, v in zip(terms, PIA_ROW))
    body += ''.join(
        _field(f'C{t}', v) for t, v in zip(terms, PIA_COL))
    body += (_field('RNIS', 2) + _field('CNIS', 3) + _field('TNIS', 6)
             + _field('RSSIZ', 2048.0) + _field('CSSIZ', 4096.0))
    return _xml(f'<tre name="RSMPIA">{body}</tre>')


# -------------------------------------------------------------------
# RSMDCA fixtures -- 1 image, 3 params (GXO/GYO/GZO), diag(4, 9, 16)
# -------------------------------------------------------------------

DCA_UPPER = [4.0, 0.0, 0.0, 9.0, 0.0, 16.0]   # (0,0)(0,1)(0,2)(1,1)(1,2)(2,2)


def _indices_cedata(ground_first3=('01', '02', '03')) -> str:
    """36 x 2-byte index fields: rows and cols blank, ground 1..3."""
    s = '  ' * 10            # IRO..IRZZ
    s += '  ' * 10           # IC0..ICZZ
    s += ''.join(ground_first3)   # GXO GYO GZO
    s += '  ' * 13           # GXR..GZZ
    return s


def _local_frame_cedata() -> str:
    s = f21(1000.0) + f21(2000.0) + f21(3000.0)
    s += ''.join(f21(v) for v in IDENTITY_9)
    return s


def build_rsmdca_cedata() -> str:
    s = IID_A + EDITION_A + TID_A
    s += '03' + '001' + '00003'
    s += IID_A + '03'                  # per-image: IID, NPARI
    s += _local_frame_cedata()
    s += _indices_cedata()
    s += ''.join(f21(v) for v in DCA_UPPER)
    return s


def build_rsmdca_xml() -> ET.Element:
    body = (_field('IID', 'IMG_A') + _field('EDITION', 'ED_001')
            + _field('TID', 'TRIG_9')
            + _field('NPAR', 3) + _field('NIMGE', 1)
            + _field('NPART', 3))
    body += _repeated('IMAGE', [
        _field('IID', 'IMG_A') + _field('NPARI', 3),
    ])
    body += (_field('XUOL', 1000.0) + _field('YUOL', 2000.0)
             + _field('ZUOL', 3000.0))
    uv_names = ('XUXL', 'XUYL', 'XUZL', 'YUXL', 'YUYL', 'YUZL',
                'ZUXL', 'ZUYL', 'ZUZL')
    body += ''.join(
        _field(n, v) for n, v in zip(uv_names, IDENTITY_9))
    # GDAL spellings: IRO (letter O), IC0 (zero), GXO/GYO/GZO
    body += _field('GXO', 1) + _field('GYO', 2) + _field('GZO', 3)
    body += _repeated('DERCOV', [
        _field('DERCOV', v) for v in DCA_UPPER
    ])
    return _xml(f'<tre name="RSMDCA">{body}</tre>')


# -------------------------------------------------------------------
# RSMECA fixtures -- INCLIC=Y (one group of 3), INCLUC=Y
# -------------------------------------------------------------------

ECA_MAP_IDENTITY = IDENTITY_9


def build_rsmeca_cedata() -> str:
    s = IID_A + EDITION_A + TID_A + 'Y' + 'Y'
    s += '03' + '03' + '01' + '20260609'   # NPAR NPARO IGN CVDATE
    s += _local_frame_cedata()
    s += _indices_cedata()
    # one subgroup: NUMOPG=3, upper triangle, TCDF, NCSEG=2, 2 segs
    s += '03'
    s += ''.join(f21(v) for v in DCA_UPPER)
    s += '1' + '2'
    s += f21(0.9) + f21(10.0) + f21(0.1) + f21(100.0)
    # MAP: 3x3 identity, row-major
    s += ''.join(f21(v) for v in ECA_MAP_IDENTITY)
    # unmodeled error
    s += f21(0.25) + f21(0.0) + f21(0.36)
    s += '2' + f21(0.8) + f21(5.0) + f21(0.2) + f21(50.0)
    s += '2' + f21(0.7) + f21(4.0) + f21(0.3) + f21(40.0)
    return s


def build_rsmeca_xml() -> ET.Element:
    body = (_field('IID', 'IMG_A') + _field('EDITION', 'ED_001')
            + _field('TID', 'TRIG_9')
            + _field('INCLIC', 'Y') + _field('INCLUC', 'Y')
            + _field('NPAR', 3) + _field('NPARO', 3)
            + _field('IGN', 1) + _field('CVDATE', '20260609'))
    body += (_field('XUOL', 1000.0) + _field('YUOL', 2000.0)
             + _field('ZUOL', 3000.0))
    uv_names = ('XUXL', 'XUYL', 'XUZL', 'YUXL', 'YUYL', 'YUZL',
                'ZUXL', 'ZUYL', 'ZUZL')
    body += ''.join(
        _field(n, v) for n, v in zip(uv_names, IDENTITY_9))
    body += _field('GXO', 1) + _field('GYO', 2) + _field('GZO', 3)
    # one IG group with nested ERRCVG and correlation segments
    ig_body = _field('NUMOPG', 3)
    ig_body += _repeated('EG', [_field('ERRCVG', v) for v in DCA_UPPER])
    ig_body += _field('TCDF', 1) + _field('NCSEG', 2)
    ig_body += _repeated('CORSEG', [
        _field('CORSEG', 0.9) + _field('TAUSEG', 10.0),
        _field('CORSEG', 0.1) + _field('TAUSEG', 100.0),
    ])
    body += _repeated('IG', [ig_body])
    body += _repeated('MAP', [
        _field('MAP', v) for v in ECA_MAP_IDENTITY
    ])
    body += (_field('URR', 0.25) + _field('URC', 0.0)
             + _field('UCC', 0.36))
    return _xml(f'<tre name="RSMECA">{body}</tre>')


# -------------------------------------------------------------------
# RSMAPA fixtures -- NPAR=2, GXO/GYO active
# -------------------------------------------------------------------

APA_VALUES = [1.5, -2.5]


def build_rsmapa_cedata() -> str:
    s = IID_A + EDITION_A + TID_A + '02'
    s += _local_frame_cedata()
    s += _indices_cedata(('01', '02', '  '))
    s += ''.join(f21(v) for v in APA_VALUES)
    return s


def build_rsmapa_xml() -> ET.Element:
    body = (_field('IID', 'IMG_A') + _field('EDITION', 'ED_001')
            + _field('TID', 'TRIG_9') + _field('NPAR', 2))
    body += (_field('XUOL', 1000.0) + _field('YUOL', 2000.0)
             + _field('ZUOL', 3000.0))
    uv_names = ('XUXL', 'XUYL', 'XUZL', 'YUXL', 'YUYL', 'YUZL',
                'ZUXL', 'ZUYL', 'ZUZL')
    body += ''.join(
        _field(n, v) for n, v in zip(uv_names, IDENTITY_9))
    body += _field('GXO', 1) + _field('GYO', 2)
    body += _repeated('PAR', [
        _field('PARVAL', v) for v in APA_VALUES
    ])
    return _xml(f'<tre name="RSMAPA">{body}</tre>')


# -------------------------------------------------------------------
# B-variant fixtures
# -------------------------------------------------------------------


def build_rsmdcb_cedata() -> str:
    s = IID_A + EDITION_A + TID_A
    s += '04' + '002'                          # NROWCB, NIMGE
    s += 'IMG_A'.ljust(80) + '04'
    s += 'IMG_B'.ljust(80) + '03'
    s += 'N'                                   # INCAPD
    return s


def build_rsmdcb_xml() -> ET.Element:
    body = (_field('IID', 'IMG_A') + _field('EDITION', 'ED_001')
            + _field('TID', 'TRIG_9')
            + _field('NROWCB', 4) + _field('NIMGE', 2))
    body += _repeated('IMGE', [
        _field('IIDI', 'IMG_A') + _field('NCOLCB', 4),
        _field('IIDI', 'IMG_B') + _field('NCOLCB', 3),
    ])
    body += _field('INCAPD', 'N')
    return _xml(f'<tre name="RSMDCB">{body}</tre>')


def build_rsmecb_cedata_inclic() -> str:
    s = IID_A + EDITION_A + TID_A + 'Y' + 'N'
    s += '04' + '02' + '20260609' + '03'   # NPARO IGN CVDATE NPAR
    s += 'G'                               # APTYP (unparsed remainder)
    return s


def build_rsmecb_cedata_unmodeled() -> str:
    s = IID_A + EDITION_A + TID_A + 'N' + 'Y'
    s += f21(0.25) + f21(0.0) + f21(0.36)
    s += '2' + f21(0.8) + f21(5.0) + f21(0.2) + f21(50.0)
    s += '2' + f21(0.7) + f21(4.0) + f21(0.3) + f21(40.0)
    return s


def build_rsmecb_xml() -> ET.Element:
    body = (_field('IID', 'IMG_A') + _field('EDITION', 'ED_001')
            + _field('TID', 'TRIG_9')
            + _field('INCLIC', 'Y') + _field('INCLUC', 'N')
            + _field('NPARO', 4) + _field('IGN', 2)
            + _field('CVDATE', '20260609') + _field('NPAR', 3))
    return _xml(f'<tre name="RSMECB">{body}</tre>')


def build_rsmapb_cedata() -> str:
    s = IID_A + EDITION_A + TID_A + '02'
    s += 'G' + 'R'                              # APTYP, LOCTYP
    s += ''.join(f21(v) for v in
                 [1.0, 1.0, 1.0, 0.0, 0.0, 0.0])  # NSFX..NOFFZ
    s += _local_frame_cedata()
    s += 'N'                                    # APBASE
    s += _field_free_parvals()
    return s


def _field_free_parvals() -> str:
    return ''.join(f21(v) for v in APA_VALUES)


def build_rsmapb_xml() -> ET.Element:
    body = (_field('IID', 'IMG_A') + _field('EDITION', 'ED_001')
            + _field('TID', 'TRIG_9') + _field('NPAR', 2)
            + _field('APTYP', 'G') + _field('LOCTYP', 'R'))
    body += (_field('XUOL', 1000.0) + _field('YUOL', 2000.0)
             + _field('ZUOL', 3000.0))
    uv_names = ('XUXL', 'XUYL', 'XUZL', 'YUXL', 'YUYL', 'YUZL',
                'ZUXL', 'ZUYL', 'ZUZL')
    body += ''.join(
        _field(n, v) for n, v in zip(uv_names, IDENTITY_9))
    body += _repeated('PAR', [
        _field('PARVAL', v) for v in APA_VALUES
    ])
    return _xml(f'<tre name="RSMAPB">{body}</tre>')


# ===================================================================
# RSMPIA
# ===================================================================


class TestRSMPIA:

    def test_cedata_expected(self):
        meta = parse_rsmpia_cedata(build_rsmpia_cedata())
        assert isinstance(meta, RSMPIAMetadata)
        assert meta.image_id == 'IMG_A'
        assert meta.edition == 'ED_001'
        np.testing.assert_allclose(meta.row_poly, PIA_ROW)
        np.testing.assert_allclose(meta.col_poly, PIA_COL)
        assert meta.row_poly.dtype == np.float64
        assert meta.num_row_sections == 2
        assert meta.num_col_sections == 3
        assert meta.num_total_sections == 6
        assert meta.row_section_size == pytest.approx(2048.0)
        assert meta.col_section_size == pytest.approx(4096.0)

    def test_cedata_too_short(self):
        assert parse_rsmpia_cedata('X' * 100) is None

    def test_cedata_garbage_floats(self):
        assert parse_rsmpia_cedata('X' * 591) is None

    def test_cedata_none(self):
        assert parse_rsmpia_cedata(None) is None

    def test_xml_expected(self):
        meta = parse_rsmpia(build_rsmpia_xml())
        assert isinstance(meta, RSMPIAMetadata)
        assert meta.image_id == 'IMG_A'
        np.testing.assert_allclose(meta.row_poly, PIA_ROW)
        np.testing.assert_allclose(meta.col_poly, PIA_COL)
        assert meta.num_total_sections == 6

    def test_xml_wrong_name(self):
        assert parse_rsmpia(_xml('<tre name="RSMIDA"/>')) is None

    def test_xml_missing_required(self):
        node = _xml(f'<tre name="RSMPIA">{_field("IID", "X")}</tre>')
        assert parse_rsmpia(node) is None


# ===================================================================
# RSMDCA
# ===================================================================


class TestRSMDCA:

    def test_cedata_expected(self):
        meta = parse_rsmdca_cedata(build_rsmdca_cedata())
        assert isinstance(meta, RSMDCAMetadata)
        assert meta.variant == 'A'
        assert meta.image_id == 'IMG_A'
        assert meta.edition == 'ED_001'
        assert meta.trigger_id == 'TRIG_9'
        assert meta.npar == 3
        assert meta.nimge == 1
        assert meta.npar_total == 3
        assert meta.image_ids == ['IMG_A']
        np.testing.assert_array_equal(meta.image_npars, [3])
        assert meta.covariance.shape == (3, 3)
        assert meta.covariance.dtype == np.float64
        np.testing.assert_allclose(
            meta.covariance, np.diag([4.0, 9.0, 16.0]))
        np.testing.assert_allclose(meta.covariance, meta.covariance.T)
        np.testing.assert_allclose(meta.local_origin,
                                   [1000.0, 2000.0, 3000.0])
        np.testing.assert_allclose(meta.local_unit_vectors, np.eye(3))
        np.testing.assert_array_equal(
            meta.ground_param_indices[:3], [1, 2, 3])
        assert meta.sigma_x == pytest.approx(2.0)
        assert meta.sigma_y == pytest.approx(3.0)
        assert meta.sigma_z == pytest.approx(4.0)
        assert meta.raw == build_rsmdca_cedata()

    def test_cedata_too_short(self):
        assert parse_rsmdca_cedata('X' * 200) is None

    def test_cedata_truncated_covariance(self):
        full = build_rsmdca_cedata()
        assert parse_rsmdca_cedata(full[:-30]) is None

    def test_xml_expected(self):
        meta = parse_rsmdca(build_rsmdca_xml())
        assert isinstance(meta, RSMDCAMetadata)
        assert meta.npar_total == 3
        assert meta.image_ids == ['IMG_A']
        np.testing.assert_allclose(
            meta.covariance, np.diag([4.0, 9.0, 16.0]))
        assert meta.sigma_x == pytest.approx(2.0)
        assert meta.sigma_y == pytest.approx(3.0)
        assert meta.sigma_z == pytest.approx(4.0)
        assert meta.raw is not None

    def test_xml_wrong_name(self):
        assert parse_rsmdca(_xml('<tre name="RSMECA"/>')) is None

    def test_xml_missing_dercov(self):
        body = (_field('IID', 'A') + _field('EDITION', 'E')
                + _field('TID', 'T') + _field('NPAR', 3)
                + _field('NIMGE', 1) + _field('NPART', 3))
        assert parse_rsmdca(_xml(f'<tre name="RSMDCA">{body}</tre>')) is None


# ===================================================================
# RSMECA
# ===================================================================


class TestRSMECA:

    def test_cedata_expected(self):
        meta = parse_rsmeca_cedata(build_rsmeca_cedata())
        assert isinstance(meta, RSMECAMetadata)
        assert meta.variant == 'A'
        assert meta.image_id == 'IMG_A'
        assert meta.inclic is True
        assert meta.incluc is True
        assert meta.npar == 3
        assert meta.nparo == 3
        assert meta.num_groups == 1
        assert meta.cov_date == '20260609'
        assert len(meta.group_covariances) == 1
        np.testing.assert_allclose(
            meta.group_covariances[0], np.diag([4.0, 9.0, 16.0]))
        np.testing.assert_allclose(
            meta.covariance, np.diag([4.0, 9.0, 16.0]))
        np.testing.assert_allclose(meta.mapping_matrix, np.eye(3))
        assert meta.unmodeled_rr == pytest.approx(0.25)
        assert meta.unmodeled_rc == pytest.approx(0.0)
        assert meta.unmodeled_cc == pytest.approx(0.36)
        assert meta.sigma_x == pytest.approx(2.0)
        assert meta.sigma_y == pytest.approx(3.0)
        assert meta.sigma_z == pytest.approx(4.0)
        assert meta.raw == build_rsmeca_cedata()

    def test_cedata_too_short_is_none(self):
        assert parse_rsmeca_cedata('x' * 100) is None
        assert parse_rsmeca_cedata(None) is None

    def test_cedata_structural_surprise_never_raises(self):
        # Truncate mid-payload: header survives, deep fields do not.
        full = build_rsmeca_cedata()
        truncated = full[:200]
        meta = parse_rsmeca_cedata(truncated)
        assert isinstance(meta, RSMECAMetadata)
        assert meta.image_id == 'IMG_A'
        assert meta.inclic is True
        assert meta.covariance is None
        assert meta.sigma_x is None
        assert meta.raw == truncated

    def test_cedata_garbage_payload_never_raises(self):
        s = IID_A + EDITION_A + TID_A + 'Y' + 'N' + 'Z' * 50
        meta = parse_rsmeca_cedata(s)
        assert isinstance(meta, RSMECAMetadata)
        assert meta.image_id == 'IMG_A'
        assert meta.npar is None
        assert meta.raw == s

    def test_xml_expected(self):
        meta = parse_rsmeca(build_rsmeca_xml())
        assert isinstance(meta, RSMECAMetadata)
        assert meta.inclic is True
        assert meta.incluc is True
        assert meta.npar == 3
        assert meta.nparo == 3
        np.testing.assert_allclose(
            meta.covariance, np.diag([4.0, 9.0, 16.0]))
        np.testing.assert_allclose(meta.mapping_matrix, np.eye(3))
        assert meta.unmodeled_rr == pytest.approx(0.25)
        assert meta.sigma_x == pytest.approx(2.0)
        assert meta.sigma_z == pytest.approx(4.0)

    def test_xml_wrong_name(self):
        assert parse_rsmeca(_xml('<tre name="RSMPCA"/>')) is None

    def test_xml_sparse_never_raises(self):
        node = _xml(f'<tre name="RSMECA">{_field("IID", "ONLY")}</tre>')
        meta = parse_rsmeca(node)
        assert isinstance(meta, RSMECAMetadata)
        assert meta.image_id == 'ONLY'
        assert meta.covariance is None


# ===================================================================
# RSMAPA
# ===================================================================


class TestRSMAPA:

    def test_cedata_expected(self):
        meta = parse_rsmapa_cedata(build_rsmapa_cedata())
        assert isinstance(meta, RSMAPAMetadata)
        assert meta.variant == 'A'
        assert meta.image_id == 'IMG_A'
        assert meta.trigger_id == 'TRIG_9'
        assert meta.npar == 2
        np.testing.assert_allclose(meta.param_values, APA_VALUES)
        assert meta.param_values.dtype == np.float64
        assert meta.param_indices.shape == (36,)
        # GXO and GYO are entries 20 and 21 of the canonical order
        assert meta.param_indices[20] == 1
        assert meta.param_indices[21] == 2
        assert meta.param_indices.sum() == 3
        np.testing.assert_allclose(meta.local_unit_vectors, np.eye(3))

    def test_cedata_too_short(self):
        assert parse_rsmapa_cedata('X' * 100) is None

    def test_cedata_garbage(self):
        assert parse_rsmapa_cedata('X' * 600) is None

    def test_xml_expected(self):
        meta = parse_rsmapa(build_rsmapa_xml())
        assert isinstance(meta, RSMAPAMetadata)
        assert meta.npar == 2
        np.testing.assert_allclose(meta.param_values, APA_VALUES)
        assert meta.param_indices[20] == 1
        assert meta.param_indices[21] == 2

    def test_xml_wrong_name(self):
        assert parse_rsmapa(_xml('<tre name="RSMAPB"/>')) is None

    def test_xml_parval_count_mismatch(self):
        body = (_field('IID', 'A') + _field('NPAR', 3)
                + _repeated('PAR', [_field('PARVAL', 1.0)]))
        assert parse_rsmapa(_xml(f'<tre name="RSMAPA">{body}</tre>')) is None


# ===================================================================
# B-variants
# ===================================================================


class TestRSMDCB:

    def test_cedata_expected(self):
        meta = parse_rsmdcb_cedata(build_rsmdcb_cedata())
        assert isinstance(meta, RSMDCAMetadata)
        assert meta.variant == 'B'
        assert meta.image_id == 'IMG_A'
        assert meta.edition == 'ED_001'
        assert meta.trigger_id == 'TRIG_9'
        assert meta.npar == 4              # NROWCB
        assert meta.nimge == 2
        assert meta.image_ids == ['IMG_A', 'IMG_B']
        np.testing.assert_array_equal(meta.image_npars, [4, 3])
        assert meta.covariance is None     # B keeps blocks in raw only
        assert meta.raw == build_rsmdcb_cedata()

    def test_cedata_too_short(self):
        assert parse_rsmdcb_cedata('X' * 50) is None

    def test_xml_expected(self):
        meta = parse_rsmdcb(build_rsmdcb_xml())
        assert isinstance(meta, RSMDCAMetadata)
        assert meta.variant == 'B'
        assert meta.npar == 4
        assert meta.nimge == 2
        assert meta.image_ids == ['IMG_A', 'IMG_B']

    def test_xml_wrong_name(self):
        assert parse_rsmdcb(_xml('<tre name="RSMDCA"/>')) is None


class TestRSMECB:

    def test_cedata_inclic(self):
        meta = parse_rsmecb_cedata(build_rsmecb_cedata_inclic())
        assert isinstance(meta, RSMECAMetadata)
        assert meta.variant == 'B'
        assert meta.image_id == 'IMG_A'
        assert meta.inclic is True
        assert meta.incluc is False
        # B-variant count order: NPARO, IGN, CVDATE, NPAR
        assert meta.nparo == 4
        assert meta.num_groups == 2
        assert meta.cov_date == '20260609'
        assert meta.npar == 3
        assert meta.raw == build_rsmecb_cedata_inclic()

    def test_cedata_unmodeled_only(self):
        meta = parse_rsmecb_cedata(build_rsmecb_cedata_unmodeled())
        assert isinstance(meta, RSMECAMetadata)
        assert meta.variant == 'B'
        assert meta.inclic is False
        assert meta.incluc is True
        assert meta.unmodeled_rr == pytest.approx(0.25)
        assert meta.unmodeled_cc == pytest.approx(0.36)

    def test_cedata_too_short(self):
        assert parse_rsmecb_cedata('X' * 50) is None

    def test_xml_expected(self):
        meta = parse_rsmecb(build_rsmecb_xml())
        assert isinstance(meta, RSMECAMetadata)
        assert meta.variant == 'B'
        assert meta.inclic is True
        assert meta.nparo == 4
        assert meta.num_groups == 2
        assert meta.npar == 3

    def test_xml_wrong_name(self):
        assert parse_rsmecb(_xml('<tre name="RSMECA"/>')) is None


class TestRSMAPB:

    def test_cedata_expected(self):
        meta = parse_rsmapb_cedata(build_rsmapb_cedata())
        assert isinstance(meta, RSMAPAMetadata)
        assert meta.variant == 'B'
        assert meta.image_id == 'IMG_A'
        assert meta.npar == 2
        np.testing.assert_allclose(meta.param_values, APA_VALUES)
        np.testing.assert_allclose(meta.local_origin,
                                   [1000.0, 2000.0, 3000.0])
        np.testing.assert_allclose(meta.local_unit_vectors, np.eye(3))
        # B-variant basis does not map to the canonical 36 indices
        assert meta.param_indices.sum() == 0
        assert meta.raw == build_rsmapb_cedata()

    def test_cedata_too_short(self):
        assert parse_rsmapb_cedata('X' * 50) is None

    def test_xml_expected(self):
        meta = parse_rsmapb(build_rsmapb_xml())
        assert isinstance(meta, RSMAPAMetadata)
        assert meta.variant == 'B'
        assert meta.npar == 2
        np.testing.assert_allclose(meta.param_values, APA_VALUES)
        np.testing.assert_allclose(meta.local_origin,
                                   [1000.0, 2000.0, 3000.0])

    def test_xml_wrong_name(self):
        assert parse_rsmapb(_xml('<tre name="RSMAPA"/>')) is None


# ===================================================================
# summarize_accuracy
# ===================================================================


class TestSummarizeAccuracy:

    def test_circular_case(self):
        # sigma_x == sigma_y -> CE90 = 2.1460 * sigma
        dca = RSMDCAMetadata(sigma_x=2.0, sigma_y=2.0, sigma_z=4.0)
        result = summarize_accuracy(dca, None)
        assert result is not None
        ce90, le90 = result
        assert ce90 == pytest.approx(2.1460 * 2.0)
        assert le90 == pytest.approx(1.6449 * 4.0)

    def test_degenerate_case(self):
        # sigma_min = 0 -> CE90 = 1.6449 * sigma_max
        dca = RSMDCAMetadata(sigma_x=3.0, sigma_y=0.0, sigma_z=1.0)
        ce90, le90 = summarize_accuracy(dca, None)
        assert ce90 == pytest.approx(1.6449 * 3.0)
        assert le90 == pytest.approx(1.6449 * 1.0)

    def test_interpolated_case(self):
        # r = 0.5 -> CE90 = sigma_max * (1.6449 + 0.5011 * 0.5)
        dca = RSMDCAMetadata(sigma_x=1.0, sigma_y=2.0, sigma_z=3.0)
        ce90, le90 = summarize_accuracy(dca, None)
        expected = 2.0 * (1.6449 + (2.1460 - 1.6449) * 0.5)
        assert ce90 == pytest.approx(expected)

    def test_dca_preferred_over_eca(self):
        dca = RSMDCAMetadata(sigma_x=1.0, sigma_y=1.0, sigma_z=1.0)
        eca = RSMECAMetadata(sigma_x=10.0, sigma_y=10.0, sigma_z=10.0)
        ce90, le90 = summarize_accuracy(dca, eca)
        assert ce90 == pytest.approx(2.1460)
        assert le90 == pytest.approx(1.6449)

    def test_eca_fallback(self):
        dca = RSMDCAMetadata()        # no sigmas
        eca = RSMECAMetadata(sigma_x=2.0, sigma_y=2.0, sigma_z=2.0)
        ce90, le90 = summarize_accuracy(dca, eca)
        assert ce90 == pytest.approx(2.1460 * 2.0)
        assert le90 == pytest.approx(1.6449 * 2.0)

    def test_no_data(self):
        assert summarize_accuracy(None, None) is None
        assert summarize_accuracy(
            RSMDCAMetadata(), RSMECAMetadata()) is None

    def test_partial_sigmas_rejected(self):
        dca = RSMDCAMetadata(sigma_x=1.0, sigma_y=1.0)  # no sigma_z
        assert summarize_accuracy(dca, None) is None

    def test_end_to_end_from_parsed(self):
        dca = parse_rsmdca_cedata(build_rsmdca_cedata())
        eca = parse_rsmeca_cedata(build_rsmeca_cedata())
        result = summarize_accuracy(dca, eca)
        assert result is not None
        ce90, le90 = result
        # sigma_x=2, sigma_y=3 -> r=2/3
        expected_ce = 3.0 * (1.6449 + (2.1460 - 1.6449) * (2.0 / 3.0))
        assert ce90 == pytest.approx(expected_ce)
        assert le90 == pytest.approx(1.6449 * 4.0)
