# -*- coding: utf-8 -*-
"""
Parsers for the RSM error-model TREs (STDI-0002 Vol 1 Appendix U).

Covers RSMPIA (polynomial identification), RSMDCA (direct error
covariance), RSMECA (indirect error covariance), RSMAPA (adjustable
parameters), and the B-variants RSMDCB / RSMECB / RSMAPB.

Each TRE has two entry points:

* ``parse_<name>(node)`` -- takes an ``xml.etree.ElementTree.Element``
  from GDAL's ``xml:TRE`` metadata domain
  (``<tre name="RSMPIA"><field name="IID" value="..."/>...``).
* ``parse_<name>_cedata(s)`` -- takes the raw fixed-width CEDATA
  string, parsed with a byte-offset cursor in the style of
  ``grdl.IO.eo.nitf._parse_rsmpca_tre``.

All parsers return ``None`` on failure -- they never raise.  RSMECA
in particular is highly variable; its parsers degrade gracefully,
returning a partially populated dataclass with the identification
header and the ``raw`` payload retained whenever the deeper structure
surprises them.

:func:`summarize_accuracy` converts parsed ground-space one-sigma
uncertainties into (CE90, LE90) meters.

Field layouts follow STDI-0002 Vol 1 Appendix U as encoded in GDAL's
``nitf_spec.xml`` (which annotates the B-variants against
STDI-0002-1-v5.0).  21-byte real fields are of the form
``+1.50000000000000E+00``.

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
from typing import List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.eo._tre_xml import (
    _first_field,
    _optional_float,
    _optional_int,
    _optional_str,
    _required_float,
    _required_int,
)
from grdl.IO.models.rsm_error import (
    RSMAPAMetadata,
    RSMDCAMetadata,
    RSMECAMetadata,
    RSMPIAMetadata,
)


# ===================================================================
# Shared helpers
# ===================================================================


#: 10 low-order polynomial term suffixes, in TRE storage order.
_POLY_TERMS = ('0', 'X', 'Y', 'Z', 'XX', 'XY', 'XZ', 'YY', 'YZ', 'ZZ')

#: XML field-name candidates for the 36 adjustable-parameter indices.
#: GDAL spells the row constant ``IRO`` (letter O) but the column
#: constant ``IC0`` (zero); both spellings are accepted everywhere.
_ROW_IDX_NAMES: Tuple[Tuple[str, ...], ...] = (
    ('IRO', 'IR0'), ('IRX',), ('IRY',), ('IRZ',), ('IRXX',),
    ('IRXY',), ('IRXZ',), ('IRYY',), ('IRYZ',), ('IRZZ',),
)
_COL_IDX_NAMES: Tuple[Tuple[str, ...], ...] = (
    ('IC0', 'ICO'), ('ICX',), ('ICY',), ('ICZ',), ('ICXX',),
    ('ICXY',), ('ICXZ',), ('ICYY',), ('ICYZ',), ('ICZZ',),
)
_GND_IDX_NAMES: Tuple[Tuple[str, ...], ...] = (
    ('GXO', 'GX0'), ('GYO', 'GY0'), ('GZO', 'GZ0'),
    ('GXR',), ('GYR',), ('GZR',), ('GS',),
    ('GXX',), ('GXY',), ('GXZ',),
    ('GYX',), ('GYY',), ('GYZ',),
    ('GZX',), ('GZY',), ('GZZ',),
)

_UV_NAMES = (
    'XUXL', 'XUYL', 'XUZL',
    'YUXL', 'YUYL', 'YUZL',
    'ZUXL', 'ZUYL', 'ZUZL',
)

# CE90 approximation endpoints (see summarize_accuracy docstring).
_K_CE90_CIRCULAR = 2.1460   # r = sigma_min / sigma_max = 1
_K_CE90_LINEAR = 1.6449     # r = 0 (and the one-dimensional 90% factor)
_K_LE90 = 1.6449


class _Cursor:
    """Byte-offset cursor over a fixed-width CEDATA string.

    Raises ``IndexError`` when a read would run past the end of the
    string and ``ValueError`` on numeric conversion failure; callers
    catch both and return ``None`` (or a partial dataclass).
    """

    def __init__(self, s: str) -> None:
        self.s = s
        self.pos = 0

    def take(self, n: int) -> str:
        if self.pos + n > len(self.s):
            raise IndexError(
                f"CEDATA truncated at offset {self.pos} (need {n} bytes)")
        out = self.s[self.pos:self.pos + n]
        self.pos += n
        return out

    def str_(self, n: int) -> str:
        return self.take(n).strip()

    def float_(self, n: int = 21) -> float:
        return float(self.take(n).strip())

    def opt_float(self, n: int = 21) -> Optional[float]:
        raw = self.take(n).strip()
        return float(raw) if raw else None

    def int_(self, n: int) -> int:
        return int(self.take(n).strip())

    def idx_(self, n: int = 2) -> int:
        """Read a parameter-index field; blank means 'not adjusted'."""
        raw = self.take(n).strip()
        return int(raw) if raw else 0


def _grouped_values(node: ET.Element, *names: str) -> List[str]:
    """Ordered values of grouped fields matching any candidate name.

    Like ``_repeated_field_values`` but accepts multiple candidate
    names and also matches on the ``longname`` attribute -- GDAL's
    spec defines some loop fields (e.g. RSMDCA ``DERCOV``, RSMECA
    ``ERRCVG``/``MAP``) with an empty ``name`` and the identifier in
    ``longname``.
    """
    targets = set(names)
    values: List[str] = []
    for group in node.iter('group'):
        for child in group:
            if child.tag != 'field':
                continue
            if (child.get('name') in targets
                    or child.get('longname') in targets):
                v = child.get('value', '')
                values.append(v.strip() if v else '')
    return values


def _idx_array_xml(
    node: ET.Element, name_groups: Sequence[Tuple[str, ...]],
) -> np.ndarray:
    """Read a block of adjustable-parameter index fields from XML."""
    out = np.zeros(len(name_groups), dtype=np.int64)
    for i, names in enumerate(name_groups):
        v = _optional_int(node, *names)
        out[i] = v if v is not None and v > 0 else 0
    return out


def _local_frame_xml(
    node: ET.Element,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Read XUOL/YUOL/ZUOL and the 9 unit-vector fields from XML."""
    origin_vals = [_optional_float(node, n)
                   for n in ('XUOL', 'YUOL', 'ZUOL')]
    origin = None
    if all(v is not None for v in origin_vals):
        origin = np.array(origin_vals, dtype=np.float64)

    uv_vals = [_optional_float(node, n) for n in _UV_NAMES]
    unit_vectors = None
    if all(v is not None for v in uv_vals):
        unit_vectors = np.array(
            uv_vals, dtype=np.float64).reshape(3, 3)
    return origin, unit_vectors


def _symmetric_from_upper(
    values: Sequence[float], n: int,
) -> Optional[np.ndarray]:
    """Build a symmetric ``(n, n)`` matrix from its upper triangle.

    ``values`` is the row-major upper triangle including the
    diagonal: ``(0,0), (0,1), ..., (0,n-1), (1,1), ..., (n-1,n-1)``.
    """
    if n <= 0 or len(values) != n * (n + 1) // 2:
        return None
    mat = np.zeros((n, n), dtype=np.float64)
    iu = np.triu_indices(n)
    mat[iu] = np.asarray(values, dtype=np.float64)
    mat[(iu[1], iu[0])] = mat[iu]
    return mat


def _derive_dca_sigmas(meta: RSMDCAMetadata) -> None:
    """Fill sigma_x/y/z from the covariance diagonal when derivable.

    The DERCOV matrix spans the concatenated parameter blocks of all
    NIMGE images (in the order of the per-image list).  The
    GXO/GYO/GZO indices are 1-based positions within *this* image's
    block; the block offset is the sum of NPARI over the preceding
    images.  When the IID of this TRE does not match any listed
    image and more than one image is present, no sigmas are derived.
    """
    cov = meta.covariance
    if cov is None or cov.size == 0:
        return

    offset = 0
    if meta.image_ids:
        target = (meta.image_id or '').strip()
        match = None
        for i, iid in enumerate(meta.image_ids):
            if iid.strip() == target:
                match = i
                break
        if match is None:
            if len(meta.image_ids) == 1:
                match = 0
            else:
                return
        offset = int(np.sum(meta.image_npars[:match]))

    n = cov.shape[0]
    sigmas: List[Optional[float]] = []
    for k in range(3):  # GXO, GYO, GZO
        g = int(meta.ground_param_indices[k])
        if g <= 0 or offset + g - 1 >= n:
            sigmas.append(None)
            continue
        var = float(cov[offset + g - 1, offset + g - 1])
        sigmas.append(float(np.sqrt(var)) if var >= 0.0 else None)
    meta.sigma_x, meta.sigma_y, meta.sigma_z = sigmas


def _derive_eca_sigmas(meta: RSMECAMetadata) -> None:
    """Fill sigma_x/y/z from the adjusted-space covariance.

    The original-space covariance ``C`` (NPARO x NPARO) is mapped to
    the active adjustable parameters via ``C_adj = M C M^T`` with
    ``M`` the MAP matrix (NPAR x NPARO).  The GXO/GYO/GZO indices
    are 1-based positions in the active parameter vector.  When the
    mapping matrix is unavailable but NPAR == NPARO, ``C`` is used
    directly (identity mapping).
    """
    cov = meta.covariance
    if cov is None or cov.size == 0:
        return
    n_orig = cov.shape[0]

    m = meta.mapping_matrix
    if m is not None and m.ndim == 2 and m.shape[1] == n_orig:
        c_adj = m @ cov @ m.T
    elif m is None and meta.npar == n_orig:
        c_adj = cov
    else:
        return

    n = c_adj.shape[0]
    sigmas: List[Optional[float]] = []
    for k in range(3):  # GXO, GYO, GZO
        g = int(meta.ground_param_indices[k])
        if g <= 0 or g - 1 >= n:
            sigmas.append(None)
            continue
        var = float(c_adj[g - 1, g - 1])
        sigmas.append(float(np.sqrt(var)) if var >= 0.0 else None)
    meta.sigma_x, meta.sigma_y, meta.sigma_z = sigmas


# ===================================================================
# RSMPIA -- Polynomial Identification (App U Table 3)
# ===================================================================


def parse_rsmpia(node: ET.Element) -> Optional[RSMPIAMetadata]:
    """Parse an ``<tre name="RSMPIA">`` element.

    The 20 low-order polynomial fields (R0..RZZ, C0..CZZ) are
    required; the section counts and sizes are optional.

    Returns
    -------
    RSMPIAMetadata or None
        ``None`` when the node is not RSMPIA or required fields are
        missing or malformed.
    """
    if node.get('name') != 'RSMPIA':
        return None

    try:
        row_poly = np.array(
            [_required_float(node, f'R{t}') for t in _POLY_TERMS],
            dtype=np.float64)
        col_poly = np.array(
            [_required_float(node, f'C{t}') for t in _POLY_TERMS],
            dtype=np.float64)

        return RSMPIAMetadata(
            image_id=_optional_str(node, 'IID'),
            edition=_optional_str(node, 'EDITION'),
            row_poly=row_poly,
            col_poly=col_poly,
            num_row_sections=_optional_int(node, 'RNIS'),
            num_col_sections=_optional_int(node, 'CNIS'),
            num_total_sections=_optional_int(node, 'TNIS'),
            row_section_size=_optional_float(node, 'RSSIZ'),
            col_section_size=_optional_float(node, 'CSSIZ'),
        )
    except (ValueError, TypeError):
        return None


def parse_rsmpia_cedata(s: str) -> Optional[RSMPIAMetadata]:
    """Parse a raw RSMPIA CEDATA string (fixed length, 591 bytes).

    Layout::

        IID(80) EDITION(40)
        R0..RZZ (10 x 21)  C0..CZZ (10 x 21)
        RNIS(3) CNIS(3) TNIS(3) RSSIZ(21) CSSIZ(21)

    Returns
    -------
    RSMPIAMetadata or None
        ``None`` when the string is shorter than 591 bytes or any
        polynomial field fails to parse as a real.
    """
    try:
        if s is None or len(s) < 591:
            return None
        c = _Cursor(s)
        image_id = c.str_(80)
        edition = c.str_(40)
        row_poly = np.array(
            [c.float_() for _ in range(10)], dtype=np.float64)
        col_poly = np.array(
            [c.float_() for _ in range(10)], dtype=np.float64)
        rnis = c.str_(3)
        cnis = c.str_(3)
        tnis = c.str_(3)
        rssiz = c.opt_float()
        cssiz = c.opt_float()

        return RSMPIAMetadata(
            image_id=image_id or None,
            edition=edition or None,
            row_poly=row_poly,
            col_poly=col_poly,
            num_row_sections=int(rnis) if rnis else None,
            num_col_sections=int(cnis) if cnis else None,
            num_total_sections=int(tnis) if tnis else None,
            row_section_size=rssiz,
            col_section_size=cssiz,
        )
    except (ValueError, TypeError, IndexError):
        return None


# ===================================================================
# RSMDCA -- Direct Error Covariance (App U Table 9)
# ===================================================================


def parse_rsmdca(node: ET.Element) -> Optional[RSMDCAMetadata]:
    """Parse an ``<tre name="RSMDCA">`` element.

    Per-image entries (IID / NPARI) are read from ``<repeated>``
    groups; the DERCOV upper triangle is collected from grouped
    fields named (or long-named) ``DERCOV`` and must contain exactly
    ``NPART * (NPART + 1) / 2`` values.

    Returns
    -------
    RSMDCAMetadata or None
        ``None`` on any structural mismatch.
    """
    if node.get('name') != 'RSMDCA':
        return None

    try:
        npar = _required_int(node, 'NPAR')
        nimge = _required_int(node, 'NIMGE')
        npart = _required_int(node, 'NPART')
        if npart <= 0:
            return None

        image_ids = _grouped_values(node, 'IID', 'IIDI')
        npari_vals = _grouped_values(node, 'NPARI')
        image_npars = np.array(
            [int(v) for v in npari_vals if v], dtype=np.int64)

        origin, unit_vectors = _local_frame_xml(node)
        row_idx = _idx_array_xml(node, _ROW_IDX_NAMES)
        col_idx = _idx_array_xml(node, _COL_IDX_NAMES)
        gnd_idx = _idx_array_xml(node, _GND_IDX_NAMES)

        dercov = [float(v) for v in _grouped_values(node, 'DERCOV') if v]
        cov = _symmetric_from_upper(dercov, npart)
        if cov is None:
            return None

        meta = RSMDCAMetadata(
            image_id=_optional_str(node, 'IID'),
            edition=_optional_str(node, 'EDITION'),
            trigger_id=_optional_str(node, 'TID'),
            npar=npar,
            nimge=nimge,
            npar_total=npart,
            image_ids=image_ids[:nimge],
            image_npars=image_npars[:nimge],
            local_origin=origin,
            local_unit_vectors=unit_vectors,
            row_param_indices=row_idx,
            col_param_indices=col_idx,
            ground_param_indices=gnd_idx,
            covariance=cov,
            raw=ET.tostring(node, encoding='unicode'),
        )
        _derive_dca_sigmas(meta)
        return meta
    except (ValueError, TypeError):
        return None


def parse_rsmdca_cedata(s: str) -> Optional[RSMDCAMetadata]:
    """Parse a raw RSMDCA CEDATA string.

    Layout::

        IID(80) EDITION(40) TID(40) NPAR(2) NIMGE(3) NPART(5)
        [IID(80) NPARI(2)] x NIMGE
        XUOL YUOL ZUOL (3 x 21)
        XUXL XUYL XUZL YUXL YUYL YUZL ZUXL ZUYL ZUZL (9 x 21)
        IRO..IRZZ (10 x 2) IC0..ICZZ (10 x 2) GXO..GZZ (16 x 2)
        DERCOV(21) x NPART*(NPART+1)/2   [row-major upper triangle]

    Returns
    -------
    RSMDCAMetadata or None
        ``None`` on truncation or numeric failure.
    """
    try:
        if s is None or len(s) < 597:
            return None
        c = _Cursor(s)
        image_id = c.str_(80)
        edition = c.str_(40)
        trigger_id = c.str_(40)
        npar = c.int_(2)
        nimge = c.int_(3)
        npart = c.int_(5)
        if nimge <= 0 or npart <= 0:
            return None

        image_ids: List[str] = []
        npari: List[int] = []
        for _ in range(nimge):
            image_ids.append(c.str_(80))
            npari.append(c.int_(2))

        origin = np.array(
            [c.float_() for _ in range(3)], dtype=np.float64)
        unit_vectors = np.array(
            [c.float_() for _ in range(9)],
            dtype=np.float64).reshape(3, 3)

        row_idx = np.array(
            [c.idx_() for _ in range(10)], dtype=np.int64)
        col_idx = np.array(
            [c.idx_() for _ in range(10)], dtype=np.int64)
        gnd_idx = np.array(
            [c.idx_() for _ in range(16)], dtype=np.int64)

        n_upper = npart * (npart + 1) // 2
        dercov = [c.float_() for _ in range(n_upper)]
        cov = _symmetric_from_upper(dercov, npart)
        if cov is None:
            return None

        meta = RSMDCAMetadata(
            image_id=image_id or None,
            edition=edition or None,
            trigger_id=trigger_id or None,
            npar=npar,
            nimge=nimge,
            npar_total=npart,
            image_ids=image_ids,
            image_npars=np.array(npari, dtype=np.int64),
            local_origin=origin,
            local_unit_vectors=unit_vectors,
            row_param_indices=row_idx,
            col_param_indices=col_idx,
            ground_param_indices=gnd_idx,
            covariance=cov,
            raw=s,
        )
        _derive_dca_sigmas(meta)
        return meta
    except (ValueError, TypeError, IndexError):
        return None


# ===================================================================
# RSMECA -- Indirect Error Covariance (App U Table 8)
# ===================================================================


def parse_rsmeca(node: ET.Element) -> Optional[RSMECAMetadata]:
    """Parse an ``<tre name="RSMECA">`` element -- defensively.

    The identification header (IID/EDITION/TID/INCLIC/INCLUC) is
    always populated.  The variable INCLIC payload (groups, MAP) and
    INCLUC payload (URR/URC/UCC) are parsed when their structure is
    consistent; any surprise leaves the corresponding fields unset
    while the dataclass (with ``raw``) is still returned.  Never
    raises.

    Returns
    -------
    RSMECAMetadata or None
        ``None`` only when the node is not an RSMECA TRE.
    """
    if node.get('name') != 'RSMECA':
        return None

    meta = RSMECAMetadata(raw=ET.tostring(node, encoding='unicode'))
    try:
        meta.image_id = _optional_str(node, 'IID')
        meta.edition = _optional_str(node, 'EDITION')
        meta.trigger_id = _optional_str(node, 'TID')
        inclic_s = _first_field(node, 'INCLIC')
        incluc_s = _first_field(node, 'INCLUC')
        meta.inclic = (inclic_s.upper() == 'Y') if inclic_s else None
        meta.incluc = (incluc_s.upper() == 'Y') if incluc_s else None
    except Exception:
        return meta

    try:
        if meta.inclic:
            meta.npar = _optional_int(node, 'NPAR')
            meta.nparo = _optional_int(node, 'NPARO')
            meta.num_groups = _optional_int(node, 'IGN')
            meta.cov_date = _optional_str(node, 'CVDATE')
            meta.local_origin, meta.local_unit_vectors = (
                _local_frame_xml(node))
            meta.row_param_indices = _idx_array_xml(node, _ROW_IDX_NAMES)
            meta.col_param_indices = _idx_array_xml(node, _COL_IDX_NAMES)
            meta.ground_param_indices = _idx_array_xml(node, _GND_IDX_NAMES)

            numopg = [int(v) for v in _grouped_values(node, 'NUMOPG') if v]
            errcvg = [float(v) for v in _grouped_values(node, 'ERRCVG') if v]
            cursor = 0
            groups: List[np.ndarray] = []
            for n_g in numopg:
                n_vals = n_g * (n_g + 1) // 2
                block = _symmetric_from_upper(
                    errcvg[cursor:cursor + n_vals], n_g)
                if block is None:
                    groups = []
                    break
                groups.append(block)
                cursor += n_vals
            meta.group_covariances = groups

            if groups and meta.nparo is not None:
                if sum(g.shape[0] for g in groups) == meta.nparo:
                    cov = np.zeros(
                        (meta.nparo, meta.nparo), dtype=np.float64)
                    off = 0
                    for g in groups:
                        n_g = g.shape[0]
                        cov[off:off + n_g, off:off + n_g] = g
                        off += n_g
                    meta.covariance = cov

            map_vals = [float(v) for v in _grouped_values(node, 'MAP') if v]
            if (meta.npar is not None and meta.nparo is not None
                    and len(map_vals) == meta.npar * meta.nparo):
                meta.mapping_matrix = np.array(
                    map_vals, dtype=np.float64,
                ).reshape(meta.npar, meta.nparo)

            _derive_eca_sigmas(meta)
    except Exception:
        pass

    try:
        if meta.incluc:
            meta.unmodeled_rr = _optional_float(node, 'URR')
            meta.unmodeled_rc = _optional_float(node, 'URC')
            meta.unmodeled_cc = _optional_float(node, 'UCC')
    except Exception:
        pass

    return meta


def parse_rsmeca_cedata(s: str) -> Optional[RSMECAMetadata]:
    """Parse a raw RSMECA CEDATA string -- defensively.

    Layout::

        IID(80) EDITION(40) TID(40) INCLIC(1) INCLUC(1)
        [if INCLIC=Y]
            NPAR(2) NPARO(2) IGN(2) CVDATE(8)
            XUOL YUOL ZUOL (3 x 21)  unit vectors (9 x 21)
            IRO..IRZZ IC0..ICZZ GXO..GZZ (36 x 2)
            [per group, IGN times]
                NUMOPG(2)
                ERRCVG(21) x NUMOPG*(NUMOPG+1)/2
                TCDF(1) NCSEG(1) [CORSEG(21) TAUSEG(21)] x NCSEG
            MAP(21) x NPAR*NPARO   [row-major NPAR x NPARO]
        [if INCLUC=Y]
            URR(21) URC(21) UCC(21)
            UNCSR(1) [UCORSR(21) UTAUSR(21)] x UNCSR
            UNCSC(1) [UCORSC(21) UTAUSC(21)] x UNCSC

    Correlation-segment data is consumed positionally but not stored
    (it remains available in ``raw``).  On structural surprise the
    already-populated fields and ``raw`` are returned -- never raises.

    Returns
    -------
    RSMECAMetadata or None
        ``None`` only when the string cannot hold the 162-byte
        identification header.
    """
    if s is None or len(s) < 162:
        return None

    c = _Cursor(s)
    meta = RSMECAMetadata(raw=s)
    image_id = c.str_(80)
    edition = c.str_(40)
    trigger_id = c.str_(40)
    meta.image_id = image_id or None
    meta.edition = edition or None
    meta.trigger_id = trigger_id or None
    meta.inclic = c.take(1).upper() == 'Y'
    meta.incluc = c.take(1).upper() == 'Y'

    try:
        if meta.inclic:
            meta.npar = c.int_(2)
            meta.nparo = c.int_(2)
            meta.num_groups = c.int_(2)
            meta.cov_date = c.str_(8) or None
            meta.local_origin = np.array(
                [c.float_() for _ in range(3)], dtype=np.float64)
            meta.local_unit_vectors = np.array(
                [c.float_() for _ in range(9)],
                dtype=np.float64).reshape(3, 3)
            meta.row_param_indices = np.array(
                [c.idx_() for _ in range(10)], dtype=np.int64)
            meta.col_param_indices = np.array(
                [c.idx_() for _ in range(10)], dtype=np.int64)
            meta.ground_param_indices = np.array(
                [c.idx_() for _ in range(16)], dtype=np.int64)

            groups: List[np.ndarray] = []
            for _ in range(meta.num_groups):
                n_g = c.int_(2)
                n_vals = n_g * (n_g + 1) // 2
                block_vals = [c.float_() for _ in range(n_vals)]
                block = _symmetric_from_upper(block_vals, n_g)
                if block is None:
                    raise ValueError("bad ERRCVG block")
                groups.append(block)
                c.take(1)              # TCDF
                ncseg = c.int_(1)
                for _ in range(ncseg):
                    c.take(21)         # CORSEG
                    c.take(21)         # TAUSEG
            meta.group_covariances = groups

            if sum(g.shape[0] for g in groups) == meta.nparo:
                cov = np.zeros(
                    (meta.nparo, meta.nparo), dtype=np.float64)
                off = 0
                for g in groups:
                    n_g = g.shape[0]
                    cov[off:off + n_g, off:off + n_g] = g
                    off += n_g
                meta.covariance = cov

            map_vals = [c.float_()
                        for _ in range(meta.npar * meta.nparo)]
            meta.mapping_matrix = np.array(
                map_vals, dtype=np.float64,
            ).reshape(meta.npar, meta.nparo)

            _derive_eca_sigmas(meta)

        if meta.incluc:
            meta.unmodeled_rr = c.float_()
            meta.unmodeled_rc = c.float_()
            meta.unmodeled_cc = c.float_()
            uncsr = c.int_(1)
            for _ in range(uncsr):
                c.take(42)             # UCORSR + UTAUSR
            uncsc = c.int_(1)
            for _ in range(uncsc):
                c.take(42)             # UCORSC + UTAUSC
    except Exception:
        pass

    return meta


# ===================================================================
# RSMAPA -- Adjustable Parameters (App U Table 10)
# ===================================================================


def parse_rsmapa(node: ET.Element) -> Optional[RSMAPAMetadata]:
    """Parse an ``<tre name="RSMAPA">`` element.

    Parameter values are collected from grouped ``PARVAL`` fields
    and must number exactly NPAR.

    Returns
    -------
    RSMAPAMetadata or None
        ``None`` on structural mismatch.
    """
    if node.get('name') != 'RSMAPA':
        return None

    try:
        npar = _required_int(node, 'NPAR')
        parvals = [float(v) for v in _grouped_values(node, 'PARVAL') if v]
        if len(parvals) != npar:
            return None

        origin, unit_vectors = _local_frame_xml(node)
        param_indices = np.concatenate([
            _idx_array_xml(node, _ROW_IDX_NAMES),
            _idx_array_xml(node, _COL_IDX_NAMES),
            _idx_array_xml(node, _GND_IDX_NAMES),
        ])

        return RSMAPAMetadata(
            image_id=_optional_str(node, 'IID'),
            edition=_optional_str(node, 'EDITION'),
            trigger_id=_optional_str(node, 'TID'),
            npar=npar,
            local_origin=origin,
            local_unit_vectors=unit_vectors,
            param_indices=param_indices,
            param_values=np.array(parvals, dtype=np.float64),
            raw=ET.tostring(node, encoding='unicode'),
        )
    except (ValueError, TypeError):
        return None


def parse_rsmapa_cedata(s: str) -> Optional[RSMAPAMetadata]:
    """Parse a raw RSMAPA CEDATA string.

    Layout (486 + 21*NPAR bytes)::

        IID(80) EDITION(40) TID(40) NPAR(2)
        XUOL YUOL ZUOL (3 x 21)  unit vectors (9 x 21)
        IRO..IRZZ IC0..ICZZ GXO..GZZ (36 x 2)
        PARVAL(21) x NPAR

    Returns
    -------
    RSMAPAMetadata or None
        ``None`` on truncation or numeric failure.
    """
    try:
        if s is None or len(s) < 507:
            return None
        c = _Cursor(s)
        image_id = c.str_(80)
        edition = c.str_(40)
        trigger_id = c.str_(40)
        npar = c.int_(2)
        if npar <= 0:
            return None

        origin = np.array(
            [c.float_() for _ in range(3)], dtype=np.float64)
        unit_vectors = np.array(
            [c.float_() for _ in range(9)],
            dtype=np.float64).reshape(3, 3)
        param_indices = np.array(
            [c.idx_() for _ in range(36)], dtype=np.int64)
        param_values = np.array(
            [c.float_() for _ in range(npar)], dtype=np.float64)

        return RSMAPAMetadata(
            image_id=image_id or None,
            edition=edition or None,
            trigger_id=trigger_id or None,
            npar=npar,
            local_origin=origin,
            local_unit_vectors=unit_vectors,
            param_indices=param_indices,
            param_values=param_values,
            raw=s,
        )
    except (ValueError, TypeError, IndexError):
        return None


# ===================================================================
# RSMDCB -- Direct Error Covariance, B variant (v5.0 App U Table 6)
# ===================================================================


def parse_rsmdcb(node: ET.Element) -> Optional[RSMDCAMetadata]:
    """Parse an ``<tre name="RSMDCB">`` element (identification depth).

    Populates ``image_id``, ``edition``, ``trigger_id``, ``npar``
    (from NROWCB), ``nimge``, and the per-image ``image_ids`` /
    ``image_npars`` (IIDI / NCOLCB).  The cross-covariance blocks
    and APTYP-dependent parameter definitions are retained in
    ``raw`` only.  ``variant`` is ``'B'``.

    Returns
    -------
    RSMDCAMetadata or None
        ``None`` only when the node is not an RSMDCB TRE.
    """
    if node.get('name') != 'RSMDCB':
        return None

    meta = RSMDCAMetadata(
        variant='B', raw=ET.tostring(node, encoding='unicode'))
    try:
        meta.image_id = _optional_str(node, 'IID')
        meta.edition = _optional_str(node, 'EDITION')
        meta.trigger_id = _optional_str(node, 'TID')
        meta.npar = _optional_int(node, 'NROWCB')
        meta.nimge = _optional_int(node, 'NIMGE')
        meta.image_ids = _grouped_values(node, 'IIDI')
        ncolcb = _grouped_values(node, 'NCOLCB')
        meta.image_npars = np.array(
            [int(v) for v in ncolcb if v], dtype=np.int64)
    except Exception:
        pass
    return meta


def parse_rsmdcb_cedata(s: str) -> Optional[RSMDCAMetadata]:
    """Parse a raw RSMDCB CEDATA string (identification depth).

    Parsed prefix::

        IID(80) EDITION(40) TID(40) NROWCB(2) NIMGE(3)
        [IIDI(80) NCOLCB(2)] x NIMGE
        INCAPD(1) ...

    Everything from INCAPD onward (adjustable-parameter definition
    and cross-covariance blocks) is retained in ``raw`` only.
    ``variant`` is ``'B'``.

    Returns
    -------
    RSMDCAMetadata or None
        ``None`` when the string cannot hold the 165-byte fixed
        prefix.
    """
    if s is None or len(s) < 165:
        return None

    c = _Cursor(s)
    meta = RSMDCAMetadata(variant='B', raw=s)
    meta.image_id = c.str_(80) or None
    meta.edition = c.str_(40) or None
    meta.trigger_id = c.str_(40) or None
    try:
        meta.npar = c.int_(2)        # NROWCB
        meta.nimge = c.int_(3)
        image_ids: List[str] = []
        ncolcb: List[int] = []
        for _ in range(meta.nimge):
            image_ids.append(c.str_(80))
            ncolcb.append(c.int_(2))
        meta.image_ids = image_ids
        meta.image_npars = np.array(ncolcb, dtype=np.int64)
    except Exception:
        pass
    return meta


# ===================================================================
# RSMECB -- Indirect Error Covariance, B variant (v5.0 App U Table 10)
# ===================================================================


def parse_rsmecb(node: ET.Element) -> Optional[RSMECAMetadata]:
    """Parse an ``<tre name="RSMECB">`` element (identification depth).

    Populates the identification header, INCLIC/INCLUC, the counts
    NPARO/IGN/NPAR with CVDATE, and the unmodeled URR/URC/UCC when
    present.  The APTYP-dependent parameter basis, group covariance
    blocks, and MAP are retained in ``raw`` only.  ``variant`` is
    ``'B'``.

    Returns
    -------
    RSMECAMetadata or None
        ``None`` only when the node is not an RSMECB TRE.
    """
    if node.get('name') != 'RSMECB':
        return None

    meta = RSMECAMetadata(
        variant='B', raw=ET.tostring(node, encoding='unicode'))
    try:
        meta.image_id = _optional_str(node, 'IID')
        meta.edition = _optional_str(node, 'EDITION')
        meta.trigger_id = _optional_str(node, 'TID')
        inclic_s = _first_field(node, 'INCLIC')
        incluc_s = _first_field(node, 'INCLUC')
        meta.inclic = (inclic_s.upper() == 'Y') if inclic_s else None
        meta.incluc = (incluc_s.upper() == 'Y') if incluc_s else None
        meta.nparo = _optional_int(node, 'NPARO')
        meta.num_groups = _optional_int(node, 'IGN')
        meta.cov_date = _optional_str(node, 'CVDATE')
        meta.npar = _optional_int(node, 'NPAR')
        meta.unmodeled_rr = _optional_float(node, 'URR')
        meta.unmodeled_rc = _optional_float(node, 'URC')
        meta.unmodeled_cc = _optional_float(node, 'UCC')
    except Exception:
        pass
    return meta


def parse_rsmecb_cedata(s: str) -> Optional[RSMECAMetadata]:
    """Parse a raw RSMECB CEDATA string (identification depth).

    Parsed prefix::

        IID(80) EDITION(40) TID(40) INCLIC(1) INCLUC(1)
        [if INCLIC=Y]  NPARO(2) IGN(2) CVDATE(8) NPAR(2) ...
        [if INCLIC=N and INCLUC=Y]  URR(21) URC(21) UCC(21) ...

    Note the B-variant reorders the counts relative to RSMECA
    (NPARO, IGN, CVDATE, NPAR).  When INCLIC='Y' the structure
    beyond NPAR is APTYP/APBASE-dependent and is retained in ``raw``
    only -- the unmodeled fields are then *not* read because their
    offset is not fixed.  ``variant`` is ``'B'``.

    Returns
    -------
    RSMECAMetadata or None
        ``None`` when the string cannot hold the 162-byte
        identification header.
    """
    if s is None or len(s) < 162:
        return None

    c = _Cursor(s)
    meta = RSMECAMetadata(variant='B', raw=s)
    meta.image_id = c.str_(80) or None
    meta.edition = c.str_(40) or None
    meta.trigger_id = c.str_(40) or None
    meta.inclic = c.take(1).upper() == 'Y'
    meta.incluc = c.take(1).upper() == 'Y'
    try:
        if meta.inclic:
            meta.nparo = c.int_(2)
            meta.num_groups = c.int_(2)
            meta.cov_date = c.str_(8) or None
            meta.npar = c.int_(2)
        elif meta.incluc:
            meta.unmodeled_rr = c.float_()
            meta.unmodeled_rc = c.float_()
            meta.unmodeled_cc = c.float_()
    except Exception:
        pass
    return meta


# ===================================================================
# RSMAPB -- Adjustable Parameters, B variant (v5.0 App U Table 8)
# ===================================================================


def parse_rsmapb(node: ET.Element) -> Optional[RSMAPAMetadata]:
    """Parse an ``<tre name="RSMAPB">`` element.

    Populates the identification header, NPAR, the local frame when
    the LOCTYP='R' fields are present, and ``param_values`` from
    grouped ``PARVAL`` fields (must number NPAR when NPAR is known).
    ``param_indices`` is left unpopulated -- the B-variant parameter
    set is APTYP-dependent and does not map onto the canonical
    36-element basis.  ``variant`` is ``'B'``.

    Returns
    -------
    RSMAPAMetadata or None
        ``None`` only when the node is not an RSMAPB TRE.
    """
    if node.get('name') != 'RSMAPB':
        return None

    meta = RSMAPAMetadata(
        variant='B', raw=ET.tostring(node, encoding='unicode'))
    try:
        meta.image_id = _optional_str(node, 'IID')
        meta.edition = _optional_str(node, 'EDITION')
        meta.trigger_id = _optional_str(node, 'TID')
        meta.npar = _optional_int(node, 'NPAR')
        meta.local_origin, meta.local_unit_vectors = (
            _local_frame_xml(node))
        parvals = [float(v) for v in _grouped_values(node, 'PARVAL') if v]
        if parvals and (meta.npar is None or len(parvals) == meta.npar):
            meta.param_values = np.array(parvals, dtype=np.float64)
    except Exception:
        pass
    return meta


def parse_rsmapb_cedata(s: str) -> Optional[RSMAPAMetadata]:
    """Parse a raw RSMAPB CEDATA string.

    Parsed structure::

        IID(80) EDITION(40) TID(40) NPAR(2) APTYP(1) LOCTYP(1)
        NSFX NSFY NSFZ NOFFX NOFFY NOFFZ (6 x 21)
        [if LOCTYP=R]  XUOL YUOL ZUOL + unit vectors (12 x 21)
        APBASE(1) [APTYP/APBASE-dependent definitions...]
        PARVAL(21) x NPAR

    ``param_values`` is read from the *trailing* ``NPAR x 21`` bytes
    -- PARVAL is always the final block of RSMAPB -- which assumes
    the CEDATA string carries no trailing padding.  The
    APTYP-dependent middle section is retained in ``raw`` only, and
    ``param_indices`` is not populated.  ``variant`` is ``'B'``.

    Returns
    -------
    RSMAPAMetadata or None
        ``None`` when the string cannot hold the 162-byte fixed
        prefix through NPAR.
    """
    if s is None or len(s) < 162:
        return None

    c = _Cursor(s)
    meta = RSMAPAMetadata(variant='B', raw=s)
    meta.image_id = c.str_(80) or None
    meta.edition = c.str_(40) or None
    meta.trigger_id = c.str_(40) or None
    try:
        meta.npar = c.int_(2)
        c.take(1)                      # APTYP
        loctyp = c.take(1).strip().upper()
        for _ in range(6):             # NSFX..NOFFZ
            c.take(21)
        if loctyp == 'R':
            meta.local_origin = np.array(
                [c.float_() for _ in range(3)], dtype=np.float64)
            meta.local_unit_vectors = np.array(
                [c.float_() for _ in range(9)],
                dtype=np.float64).reshape(3, 3)
    except Exception:
        return meta

    try:
        if meta.npar and meta.npar > 0:
            tail_len = 21 * meta.npar
            if len(s) >= c.pos + tail_len:
                tail = s[len(s) - tail_len:]
                meta.param_values = np.array(
                    [float(tail[i * 21:(i + 1) * 21].strip())
                     for i in range(meta.npar)],
                    dtype=np.float64)
    except Exception:
        pass
    return meta


# ===================================================================
# Accuracy summary
# ===================================================================


def summarize_accuracy(
    dca: Optional[RSMDCAMetadata],
    eca: Optional[RSMECAMetadata],
) -> Optional[Tuple[float, float]]:
    """Derive (CE90, LE90) in meters from RSM error covariance data.

    Uses the ground-space one-sigma uncertainties (``sigma_x``,
    ``sigma_y``, ``sigma_z``, meters in the local frame) derived
    during parsing.  The direct covariance (``dca``) is preferred;
    the indirect covariance (``eca``) is the fallback.  The first
    source providing all three sigmas wins.

    Formulas
    --------
    LE90 (linear error, 90%) is the one-dimensional Gaussian
    90th-percentile factor::

        LE90 = 1.6449 * sigma_z

    CE90 (circular error, 90%) for a bivariate Gaussian has no
    closed form; this uses the standard eccentricity-ratio
    interpolation tabulated in MIL-HDBK-850 / Greenwalt & Shultz,
    "Principles of Error Theory and Cartographic Applications"
    (ACIC TR-96), linearized between its exact endpoints in
    ``r = sigma_min / sigma_max``::

        CE90 = sigma_max * (1.6449 + (2.1460 - 1.6449) * r)

    which reproduces the circular case ``CE90 = 2.1460 * sigma``
    at ``r = 1`` and the degenerate one-dimensional case
    ``CE90 = 1.6449 * sigma_max`` at ``r = 0``.

    Parameters
    ----------
    dca : RSMDCAMetadata, optional
        Parsed direct error covariance (RSMDCA/RSMDCB).
    eca : RSMECAMetadata, optional
        Parsed indirect error covariance (RSMECA/RSMECB).

    Returns
    -------
    tuple of (float, float) or None
        ``(ce90_meters, le90_meters)``, or ``None`` when neither
        input provides all three ground-space sigmas.
    """
    for src in (dca, eca):
        if src is None:
            continue
        sx, sy, sz = src.sigma_x, src.sigma_y, src.sigma_z
        if sx is None or sy is None or sz is None:
            continue
        s_max = max(abs(sx), abs(sy))
        s_min = min(abs(sx), abs(sy))
        if s_max <= 0.0:
            ce90 = 0.0
        else:
            r = s_min / s_max
            ce90 = s_max * (
                _K_CE90_LINEAR
                + (_K_CE90_CIRCULAR - _K_CE90_LINEAR) * r
            )
        le90 = _K_LE90 * abs(sz)
        return (float(ce90), float(le90))
    return None
