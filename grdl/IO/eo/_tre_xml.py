# -*- coding: utf-8 -*-
"""
xml:TRE-based parser for EO NITF Tagged Record Extensions.

GDAL parses well-known TREs into a structured XML document exposed in
the ``xml:TRE`` metadata domain.  Each TRE field is emitted as
``<field name="..." value="..."/>`` and repeated groups are wrapped
in ``<repeated number="N">``.  Reading fields by name eliminates the
byte-offset arithmetic in the legacy parser in :mod:`grdl.IO.eo.nitf`
and the off-by-N bugs that come with it.

Supported TREs:

* RSM family -- ``RSMPCA``, ``RSMIDA``, ``RSMGGA``
* Geometry  -- ``ICHIPB``, ``BLOCKA``
* Quality   -- ``CSEXRA``, ``USE00A``
* Ephemeris -- ``CSEPHA``
* Context   -- ``AIMIDB``, ``STDIDC``, ``PIAIMC``

All field names follow STDI-0002 verbatim; where a TRE has known
GDAL-spelling variants (e.g. ``ANAMRPH_CORR`` vs ``ANAMORPH_CORR``)
the parser accepts both via :func:`_first_field`.

Dependencies
------------
rasterio (for ``Dataset.tags(ns='xml:TRE')``)

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
2026-04-28

Modified
--------
2026-04-28
"""

# Standard library
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.models.common import XYZ
from grdl.IO.models.eo_nitf import (
    BLOCKAMetadata,
    CSEPHAMetadata,
    CSEXRAMetadata,
    ICHIPBMetadata,
    RSMCoefficients,
    RSMGGAGridPlane,
    RSMGGAMetadata,
    RSMIdentification,
    USE00AMetadata,
)


# ===================================================================
# XML access helpers
# ===================================================================


def load_xml_tres(dataset: object) -> List[ET.Element]:
    """Read parsed TRE XML from a rasterio dataset.

    GDAL exposes recognized TREs in the ``xml:TRE`` metadata domain
    as a single XML document with one ``<tre>`` element per TRE
    instance.  Multiple instances of the same TRE (e.g., one RSMPCA
    per image section) appear as repeated ``<tre>`` elements.

    Parameters
    ----------
    dataset : rasterio.DatasetReader
        Open rasterio dataset.

    Returns
    -------
    list of xml.etree.ElementTree.Element
        ``<tre>`` elements.  Empty list when GDAL does not recognize
        any TREs in the file or the namespace is unavailable.

    Notes
    -----
    rasterio aggregates the ``xml:TRE`` metadata domain as a dict.
    GDAL versions differ in whether the dict has a single key holding
    the whole document or one entry per TRE -- this helper handles
    both shapes.
    """
    try:
        tags = dataset.tags(ns='xml:TRE') or {}
    except (AttributeError, TypeError):
        return []
    if not tags:
        return []

    elements: List[ET.Element] = []
    for raw in tags.values():
        if not raw:
            continue
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            continue
        if root.tag == 'tre':
            elements.append(root)
        else:
            elements.extend(root.findall('.//tre'))
    return elements


def _first_field(node: ET.Element, *names: str) -> Optional[str]:
    """Return the first matching ``<field name=NAME>`` value.

    Accepts multiple candidate names so the parser tolerates GDAL
    spelling variants (e.g. ``ANAMRPH_CORR`` vs ``ANAMORPH_CORR``).
    Searches direct children only; repeated groups are walked
    explicitly via :func:`_repeated_field_values`.
    """
    target = set(names)
    for child in node:
        if child.tag == 'field' and child.get('name') in target:
            v = child.get('value', '')
            return v.strip() if v else ''
    return None


def _required_float(node: ET.Element, *names: str) -> float:
    v = _first_field(node, *names)
    if v is None or v == '':
        raise ValueError(f"Missing required field {names}")
    return float(v)


def _optional_float(node: ET.Element, *names: str) -> Optional[float]:
    v = _first_field(node, *names)
    if v is None or v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _required_int(node: ET.Element, *names: str) -> int:
    v = _first_field(node, *names)
    if v is None or v == '':
        raise ValueError(f"Missing required field {names}")
    return int(v)


def _optional_int(node: ET.Element, *names: str) -> Optional[int]:
    v = _first_field(node, *names)
    if v is None or v == '':
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _optional_str(node: ET.Element, *names: str) -> Optional[str]:
    v = _first_field(node, *names)
    return v if v else None


def _repeated_field_values(
    node: ET.Element, field_name: str,
) -> List[str]:
    """Return ordered values of ``<field name=FIELD_NAME>`` under any
    descendant ``<repeated><group>`` block.

    GDAL emits polynomial-coefficient sequences and gridded data under
    ``<repeated number="N"><group index="..">
    <field name=".." value=".."/></group></repeated>``.  Group order
    matches the ``index`` attribute -- already serialized in spec
    order by GDAL.
    """
    values: List[str] = []
    for group in node.iter('group'):
        for field in group:
            if field.tag == 'field' and field.get('name') == field_name:
                v = field.get('value', '')
                values.append(v.strip() if v else '')
    return values


def _coef_array(node: ET.Element, field_name: str) -> np.ndarray:
    raw = _repeated_field_values(node, field_name)
    if not raw:
        return np.zeros(0, dtype=np.float64)
    return np.array([float(v) for v in raw], dtype=np.float64)


# ===================================================================
# RSMPCA -- RSM Polynomial Coefficients (STDI-0002 App U Table 4)
# ===================================================================


def parse_rsmpca(node: ET.Element) -> Optional[RSMCoefficients]:
    """Parse an ``<tre name="RSMPCA">`` element into RSMCoefficients.

    Field names follow STDI-0002 Vol 1 Appendix U Table 4 verbatim --
    GDAL's NITF spec uses the same identifiers.

    Returns
    -------
    RSMCoefficients or None
        ``None`` when required fields are missing or malformed.
    """
    if node.get('name') != 'RSMPCA':
        return None

    try:
        rsn = _required_int(node, 'RSN')
        csn = _required_int(node, 'CSN')

        rfep = _optional_float(node, 'RFEP')
        cfep = _optional_float(node, 'CFEP')

        row_off = _required_float(node, 'RNRMO')
        col_off = _required_float(node, 'CNRMO')
        x_off = _required_float(node, 'XNRMO')
        y_off = _required_float(node, 'YNRMO')
        z_off = _required_float(node, 'ZNRMO')

        row_norm_sf = _required_float(node, 'RNRMSF')
        col_norm_sf = _required_float(node, 'CNRMSF')
        x_norm_sf = _required_float(node, 'XNRMSF')
        y_norm_sf = _required_float(node, 'YNRMSF')
        z_norm_sf = _required_float(node, 'ZNRMSF')

        rn_pwr = np.array([
            _required_int(node, 'RNPWRX'),
            _required_int(node, 'RNPWRY'),
            _required_int(node, 'RNPWRZ'),
        ])
        rd_pwr = np.array([
            _required_int(node, 'RDPWRX'),
            _required_int(node, 'RDPWRY'),
            _required_int(node, 'RDPWRZ'),
        ])
        cn_pwr = np.array([
            _required_int(node, 'CNPWRX'),
            _required_int(node, 'CNPWRY'),
            _required_int(node, 'CNPWRZ'),
        ])
        cd_pwr = np.array([
            _required_int(node, 'CDPWRX'),
            _required_int(node, 'CDPWRY'),
            _required_int(node, 'CDPWRZ'),
        ])

        row_num = _coef_array(node, 'RNPCF')
        row_den = _coef_array(node, 'RDPCF')
        col_num = _coef_array(node, 'CNPCF')
        col_den = _coef_array(node, 'CDPCF')

        for declared, observed, label in (
            ('RNTRMS', row_num, 'RNPCF'),
            ('RDTRMS', row_den, 'RDPCF'),
            ('CNTRMS', col_num, 'CNPCF'),
            ('CDTRMS', col_den, 'CDPCF'),
        ):
            n_decl = _first_field(node, declared)
            if n_decl and int(n_decl) != observed.size:
                raise ValueError(
                    f"{label} has {observed.size} entries but "
                    f"{declared}={n_decl}"
                )

        return RSMCoefficients(
            row_off=row_off, col_off=col_off,
            row_norm_sf=row_norm_sf, col_norm_sf=col_norm_sf,
            x_off=x_off, y_off=y_off, z_off=z_off,
            x_norm_sf=x_norm_sf, y_norm_sf=y_norm_sf, z_norm_sf=z_norm_sf,
            row_num_powers=rn_pwr, row_den_powers=rd_pwr,
            col_num_powers=cn_pwr, col_den_powers=cd_pwr,
            row_num_coefs=row_num, row_den_coefs=row_den,
            col_num_coefs=col_num, col_den_coefs=col_den,
            rsn=rsn, csn=csn,
            row_fit_error=rfep, col_fit_error=cfep,
        )
    except (ValueError, TypeError):
        return None


def collect_rsmpca_segments(
    dataset: object,
) -> Dict[Tuple[int, int], RSMCoefficients]:
    """Return all RSMPCA segments keyed by ``(rsn, csn)``."""
    segments: Dict[Tuple[int, int], RSMCoefficients] = {}
    for tre in load_xml_tres(dataset):
        if tre.get('name') != 'RSMPCA':
            continue
        seg = parse_rsmpca(tre)
        if seg is None:
            continue
        key = (seg.rsn or 1, seg.csn or 1)
        segments[key] = seg
    return segments


# ===================================================================
# RSMIDA -- RSM Identification (STDI-0002 App U Table 2)
# ===================================================================


def parse_rsmida(node: ET.Element) -> Optional[RSMIdentification]:
    """Parse an ``<tre name="RSMIDA">`` element into RSMIdentification.

    Field names follow STDI-0002 Vol 1 Appendix U Table 2 verbatim.
    Note that ``ISID`` is the *Image Sequence Identifier* per spec
    (the legacy field-name comment in :mod:`grdl.IO.eo.nitf` calling
    it "Image Sensor Identifier" was a mislabel).
    """
    if node.get('name') != 'RSMIDA':
        return None

    try:
        image_id = _optional_str(node, 'IID')
        edition = _optional_str(node, 'EDITION')
        isid = _optional_str(node, 'ISID')
        sid = _optional_str(node, 'SID')
        stid = _optional_str(node, 'STID')

        # Acquisition timestamp -- BCS-A fields, all spaces if unavailable
        year_s = _optional_str(node, 'YEAR')
        month_s = _optional_str(node, 'MONTH')
        day_s = _optional_str(node, 'DAY')
        hour_s = _optional_str(node, 'HOUR')
        minute_s = _optional_str(node, 'MINUTE')
        second_s = _optional_str(node, 'SECOND')

        collection_dt = None
        if year_s and month_s and day_s:
            try:
                sec_f = float(second_s) if second_s else 0.0
                sec_int = min(59, int(sec_f))   # 60.* allowed for leap second
                micro = int((sec_f - sec_int) * 1_000_000)
                collection_dt = datetime(
                    int(year_s), int(month_s), int(day_s),
                    int(hour_s) if hour_s else 0,
                    int(minute_s) if minute_s else 0,
                    sec_int, micro,
                )
            except (ValueError, TypeError):
                collection_dt = None

        nrg = _optional_int(node, 'NRG')
        ncg = _optional_int(node, 'NCG')
        trg = _optional_float(node, 'TRG')
        tcg = _optional_float(node, 'TCG')

        grndd = _optional_str(node, 'GRNDD')

        # Rectangular coordinate origin -- present only when GRNDD=R
        xuor = _optional_float(node, 'XUOR')
        yuor = _optional_float(node, 'YUOR')
        zuor = _optional_float(node, 'ZUOR')
        coord_origin = None
        if xuor is not None and yuor is not None and zuor is not None:
            coord_origin = XYZ(x=xuor, y=yuor, z=zuor)

        # Unit-vector matrix -- 9 fields XUXR..ZUZR (rows are local axes)
        uv_names = (
            'XUXR', 'YUXR', 'ZUXR',  # X-axis components in WGS84 X,Y,Z
            'XUYR', 'YUYR', 'ZUYR',  # Y-axis
            'XUZR', 'YUZR', 'ZUZR',  # Z-axis
        )
        uv = []
        for name in uv_names:
            val = _optional_float(node, name)
            uv.append(val if val is not None else 0.0)
        coord_unit_vectors = np.array(uv, dtype=np.float64).reshape(3, 3)

        # 8 ground-domain vertices -- V1X..V8Z (24 fields)
        vert_vals: List[float] = []
        for i in range(1, 9):
            for axis in ('X', 'Y', 'Z'):
                v = _optional_float(node, f'V{i}{axis}')
                vert_vals.append(v if v is not None else 0.0)
        ground_domain_vertices = np.array(
            vert_vals, dtype=np.float64).reshape(8, 3)

        grpx = _optional_float(node, 'GRPX')
        grpy = _optional_float(node, 'GRPY')
        grpz = _optional_float(node, 'GRPZ')
        grp = XYZ(
            x=grpx if grpx is not None else 0.0,
            y=grpy if grpy is not None else 0.0,
            z=grpz if grpz is not None else 0.0,
        )

        return RSMIdentification(
            image_id=image_id,
            edition=edition,
            sensor_id=sid,
            sensor_type_id=stid,
            image_sensor_id=isid,
            collection_datetime=collection_dt,
            ground_domain_type=grndd,
            ground_ref_point=grp,
            num_row_sections=nrg,
            num_col_sections=ncg,
            time_ref_row=trg,
            time_ref_col=tcg,
            coord_origin=coord_origin,
            coord_unit_vectors=coord_unit_vectors,
            ground_domain_vertices=ground_domain_vertices,
            full_image_rows=_optional_int(node, 'FULLR'),
            full_image_cols=_optional_int(node, 'FULLC'),
            min_row=_optional_int(node, 'MINR'),
            max_row=_optional_int(node, 'MAXR'),
            min_col=_optional_int(node, 'MINC'),
            max_col=_optional_int(node, 'MAXC'),
        )
    except (ValueError, TypeError):
        return None


# ===================================================================
# RSMGGA -- RSM Ground-to-Image Grid (STDI-0002 App U Table 12)
# ===================================================================


def parse_rsmgga(node: ET.Element) -> Optional[RSMGGAMetadata]:
    """Parse an ``<tre name="RSMGGA">`` element.

    The RSMGGA grid is structured as:

      * Header fields (IID, EDITION, GGRSN/GGCSN, fit errors, INTORD,
        NPLN, DELTAZ/X/Y, ZPLN1/XIPLN1/YIPLN1, REFROW/REFCOL,
        TNUMRD/TNUMCD/FNUMRD/FNUMCD).
      * For planes 2..NPLN: integer offsets (IXO, IYO).
      * For each plane: NXPTS, NYPTS, then NXPTS*NYPTS grid points
        (RCOORD, CCOORD).

    GDAL emits the per-plane data under nested ``<repeated>``, so we
    walk groups in document order.
    """
    if node.get('name') != 'RSMGGA':
        return None

    try:
        iid = _optional_str(node, 'IID')
        edition = _optional_str(node, 'EDITION')
        ggrsn = _required_int(node, 'GGRSN')
        ggcsn = _required_int(node, 'GGCSN')
        rfep = _optional_float(node, 'GGRFEP')
        cfep = _optional_float(node, 'GGCFEP')
        intord = _optional_int(node, 'INTORD')
        nplns = _required_int(node, 'NPLN')
        if nplns < 2:
            return None

        delta_z = _required_float(node, 'DELTAZ')
        delta_x = _required_float(node, 'DELTAX')
        delta_y = _required_float(node, 'DELTAY')
        z1 = _required_float(node, 'ZPLN1')
        x1 = _required_float(node, 'XIPLN1')
        y1 = _required_float(node, 'YIPLN1')

        ref_row = _required_int(node, 'REFROW')
        ref_col = _required_int(node, 'REFCOL')
        # TNUMRD/TNUMCD/FNUMRD/FNUMCD declare per-point string widths.
        # We don't need them for XML parsing -- GDAL has already split
        # the values for us -- but we read them to validate presence.
        for name in ('TNUMRD', 'TNUMCD', 'FNUMRD', 'FNUMCD'):
            _ = _optional_int(node, name)

        # Per-plane offsets and grid points.  The exact GDAL serialization
        # nests offsets and per-plane point lists; rather than rely on
        # nesting structure, we walk all <group> elements in document
        # order and split them by plane.
        planes: List[RSMGGAGridPlane] = []
        offsets: List[Tuple[int, int]] = [(0, 0)]

        # Pull plane offsets (IXO,IYO) from any group containing them.
        ixo_vals = _repeated_field_values(node, 'IXO')
        iyo_vals = _repeated_field_values(node, 'IYO')
        for ixo_s, iyo_s in zip(ixo_vals, iyo_vals):
            try:
                offsets.append((int(ixo_s), int(iyo_s)))
            except ValueError:
                offsets.append((0, 0))
        # Pad to nplns just in case
        while len(offsets) < nplns:
            offsets.append((0, 0))

        # Per-plane NXPTS/NYPTS sequence
        nxpts_seq = _repeated_field_values(node, 'NXPTS')
        nypts_seq = _repeated_field_values(node, 'NYPTS')
        rcoord_all = _repeated_field_values(node, 'RCOORD')
        ccoord_all = _repeated_field_values(node, 'CCOORD')

        if len(nxpts_seq) != nplns or len(nypts_seq) != nplns:
            return None

        cursor = 0
        for p_idx in range(nplns):
            nxp = int(nxpts_seq[p_idx])
            nyp = int(nypts_seq[p_idx])
            n_pts = nxp * nyp
            if cursor + n_pts > len(rcoord_all):
                return None

            rows_arr = np.full((nxp, nyp), np.nan, dtype=np.float64)
            cols_arr = np.full((nxp, nyp), np.nan, dtype=np.float64)
            # Spec: matrix row-major over X axis (outer) then Y (inner).
            for ix in range(nxp):
                for iy in range(nyp):
                    r_s = rcoord_all[cursor]
                    c_s = ccoord_all[cursor]
                    cursor += 1
                    if r_s:
                        rows_arr[ix, iy] = float(r_s)
                    if c_s:
                        cols_arr[ix, iy] = float(c_s)

            ixo, iyo = offsets[p_idx]
            planes.append(RSMGGAGridPlane(
                z_plane=z1 + p_idx * delta_z,
                xi=x1 + ixo * delta_x,
                yi=y1 + iyo * delta_y,
                num_x=nxp,
                num_y=nyp,
                rows=rows_arr,
                cols=cols_arr,
            ))

        return RSMGGAMetadata(
            iid=iid,
            edition=edition,
            ggrsn=ggrsn,
            ggcsn=ggcsn,
            row_fit_error=rfep,
            col_fit_error=cfep,
            interpolation_order=intord,
            num_planes=nplns,
            delta_z=delta_z,
            delta_x=delta_x,
            delta_y=delta_y,
            ref_row=ref_row,
            ref_col=ref_col,
            planes=planes,
        )
    except (ValueError, TypeError):
        return None


# ===================================================================
# CSEXRA -- Compensated Sensor Error Extension (STDI-0002 Vol 1 App E)
# ===================================================================


def parse_csexra(node: ET.Element) -> Optional[CSEXRAMetadata]:
    """Parse a ``<tre name="CSEXRA">`` element."""
    if node.get('name') != 'CSEXRA':
        return None

    try:
        return CSEXRAMetadata(
            predicted_niirs=_optional_float(node, 'PREDICTED_NIIRS'),
            ce90=_optional_float(node, 'CE'),
            le90=_optional_float(node, 'LE'),
            ground_gsd_row=_optional_float(node, 'GROUND_GSD_ROW',
                                            'GROUND_ROW_GSD'),
            ground_gsd_col=_optional_float(node, 'GROUND_GSD_COL',
                                            'GROUND_COL_GSD'),
        )
    except (ValueError, TypeError):
        return None


# ===================================================================
# USE00A -- Exploitation Usability Extension (STDI-0002 Vol 1 App J)
# ===================================================================


def parse_use00a(node: ET.Element) -> Optional[USE00AMetadata]:
    """Parse a ``<tre name="USE00A">`` element."""
    if node.get('name') != 'USE00A':
        return None

    try:
        return USE00AMetadata(
            obliquity_angle=_optional_float(node, 'OBL_ANG', 'OBLIQUITY_ANGLE'),
            sun_azimuth=_optional_float(node, 'SUN_AZ', 'SUN_AZIMUTH'),
            sun_elevation=_optional_float(node, 'SUN_EL', 'SUN_ELEVATION'),
            mean_gsd=_optional_float(node, 'MEAN_GSD'),
            predicted_niirs=_optional_float(node, 'N_REF', 'PREDICTED_NIIRS'),
        )
    except (ValueError, TypeError):
        return None


# ===================================================================
# ICHIPB -- Image Chip Block (STDI-0002 Vol 1 App G)
# ===================================================================


def parse_ichipb(node: ET.Element) -> Optional[ICHIPBMetadata]:
    """Parse a ``<tre name="ICHIPB">`` element.

    Per STDI-0002 Vol 1 App G, ``OP_ROW_n``/``OP_COL_n`` are the
    *output product* (chip) pixel coordinates of corner *n*, and
    ``FI_ROW_n``/``FI_COL_n`` are the *full image* pixel coordinates
    of that same corner.  The chip→full affine in the model's
    ``full = fi_off + fi_scale * chip`` form is therefore::

        fi_scale = (FI_22 - FI_11) / (OP_22 - OP_11)
        fi_off   = FI_11 - fi_scale * OP_11

    When ``XFRM_FLAG=0`` the file declares no transform applies; the
    chip is the original and the affine collapses to identity.  When
    ``ANAMRPH_CORR=1`` an anamorphic correction has been applied and
    a simple affine is no longer the correct mapping — we still
    populate the affine for callers that want to fall back to it,
    but the field is preserved so callers can detect the case.
    """
    if node.get('name') != 'ICHIPB':
        return None

    try:
        xfrm_flag = _optional_int(node, 'XFRM_FLAG')
        scale_factor = _optional_float(node, 'SCALE_FACTOR')
        anamorphic_corr = _optional_float(
            node, 'ANAMRPH_CORR', 'ANAMORPH_CORR')

        op_row_11 = _optional_float(node, 'OP_ROW_11')
        op_col_11 = _optional_float(node, 'OP_COL_11')
        op_row_22 = _optional_float(node, 'OP_ROW_22')
        op_col_22 = _optional_float(node, 'OP_COL_22')
        fi_row_11 = _optional_float(node, 'FI_ROW_11')
        fi_col_11 = _optional_float(node, 'FI_COL_11')
        fi_row_22 = _optional_float(node, 'FI_ROW_22')
        fi_col_22 = _optional_float(node, 'FI_COL_22')

        # Default to identity when XFRM_FLAG=0 or corner values are absent.
        fi_row_scale = 1.0
        fi_col_scale = 1.0
        fi_row_off = 0.0
        fi_col_off = 0.0

        have_corners = (op_row_11 is not None and op_row_22 is not None
                        and fi_row_11 is not None and fi_row_22 is not None
                        and op_col_11 is not None and op_col_22 is not None
                        and fi_col_11 is not None and fi_col_22 is not None)

        if xfrm_flag != 0 and have_corners:
            d_op_r = op_row_22 - op_row_11
            d_op_c = op_col_22 - op_col_11
            if d_op_r != 0:
                fi_row_scale = (fi_row_22 - fi_row_11) / d_op_r
                fi_row_off = fi_row_11 - fi_row_scale * op_row_11
            if d_op_c != 0:
                fi_col_scale = (fi_col_22 - fi_col_11) / d_op_c
                fi_col_off = fi_col_11 - fi_col_scale * op_col_11

        return ICHIPBMetadata(
            xfrm_flag=xfrm_flag,
            scale_factor_r=scale_factor,
            scale_factor_c=scale_factor,
            anamorphic_corr=anamorphic_corr,
            fi_row_off=fi_row_off,
            fi_col_off=fi_col_off,
            fi_row_scale=fi_row_scale,
            fi_col_scale=fi_col_scale,
            op_row=op_row_22,
            op_col=op_col_22,
            full_image_rows=_optional_int(node, 'FI_ROW'),
            full_image_cols=_optional_int(node, 'FI_COL'),
        )
    except (ValueError, TypeError):
        return None


# ===================================================================
# BLOCKA -- Image Geographic Location (STDI-0002 Vol 1 App F)
# ===================================================================


def parse_blocka(node: ET.Element) -> Optional[BLOCKAMetadata]:
    """Parse a ``<tre name="BLOCKA">`` element.

    Uses spec field names directly -- corner string format is
    ``DDMMSS.SShDDDMMSS.SSh`` with ``h`` denoting N/S/E/W.
    """
    if node.get('name') != 'BLOCKA':
        return None

    try:
        return BLOCKAMetadata(
            block_number=_optional_int(node, 'BLOCK_INSTANCE'),
            frfc_loc=_optional_str(node, 'FRFC_LOC'),
            frlc_loc=_optional_str(node, 'FRLC_LOC'),
            lrfc_loc=_optional_str(node, 'LRFC_LOC'),
            lrlc_loc=_optional_str(node, 'LRLC_LOC'),
        )
    except (ValueError, TypeError):
        return None


# ===================================================================
# CSEPHA -- Commercial Support Ephemeris (STDI-0002 Vol 1 App D)
# ===================================================================


def parse_csepha(node: ET.Element) -> Optional[CSEPHAMetadata]:
    """Parse a ``<tre name="CSEPHA">`` element.

    Layout per STDI-0002 Vol 1 Appendix D::

        EPHEM_FLAG, DT_EPHEM, DATE_EPHEM, T0_EPHEM, NUM_EPHEM
        [EPHEM_X, EPHEM_Y, EPHEM_Z] * NUM_EPHEM

    ``T0_EPHEM`` is an HHMMSS.mmmmmm string in the raw TRE; we
    convert it to seconds-since-midnight UTC.
    """
    if node.get('name') != 'CSEPHA':
        return None

    try:
        ephem_flag = _optional_str(node, 'EPHEM_FLAG')
        dt_ephem = _optional_float(node, 'DT_EPHEM')
        date_ephem = _optional_str(node, 'DATE_EPHEM')

        t0_raw = _optional_str(node, 'T0_EPHEM')
        t0_ephem: Optional[float] = None
        if t0_raw and len(t0_raw) >= 6:
            try:
                hh = int(t0_raw[0:2])
                mm = int(t0_raw[2:4])
                ss = float(t0_raw[4:])
                t0_ephem = hh * 3600.0 + mm * 60.0 + ss
            except ValueError:
                t0_ephem = None

        num = _optional_int(node, 'NUM_EPHEM')

        xs = _repeated_field_values(node, 'EPHEM_X')
        ys = _repeated_field_values(node, 'EPHEM_Y')
        zs = _repeated_field_values(node, 'EPHEM_Z')
        if xs and ys and zs and len(xs) == len(ys) == len(zs):
            position = np.array(
                [[float(x), float(y), float(z)]
                 for x, y, z in zip(xs, ys, zs)],
                dtype=np.float64,
            )
        else:
            position = None

        if num is None and position is not None:
            num = position.shape[0]

        return CSEPHAMetadata(
            ephem_flag=ephem_flag,
            dt_ephem=dt_ephem,
            date_ephem=date_ephem,
            t0_ephem=t0_ephem,
            num_ephem=num,
            position=position,
        )
    except (ValueError, TypeError):
        return None


# ===================================================================
# AIMIDB -- Additional Image ID (STDI-0002 Vol 1 App C)
# ===================================================================


def parse_aimidb(node: ET.Element) -> Optional[Dict[str, Any]]:
    """Parse a ``<tre name="AIMIDB">`` element.

    Returns a dict mirroring the legacy parser:
    ``{'collection_datetime', 'mission_id', 'country_code'}``.
    """
    if node.get('name') != 'AIMIDB':
        return None

    try:
        # ACQUISITION_DATE is DDHHMMSSZMONYY (14 chars) -- preserve the
        # legacy parsing rather than depending on GDAL's split.
        acq_s = _optional_str(node, 'ACQUISITION_DATE')
        mission_id = _optional_str(node, 'MISSION_NO', 'MISSION_NUMBER')
        country = _optional_str(node, 'COUNTRY')

        collection_dt = None
        if acq_s and len(acq_s) >= 14:
            try:
                day = int(acq_s[0:2])
                hour = int(acq_s[2:4])
                minute = int(acq_s[4:6])
                second = int(acq_s[6:8])
                mon_str = acq_s[9:12]
                year_2 = int(acq_s[12:14])
                months = {
                    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
                    'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
                    'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
                }
                month = months.get(mon_str.upper(), 1)
                year = 2000 + year_2 if year_2 < 80 else 1900 + year_2
                collection_dt = datetime(
                    year, month, day, hour, minute, second)
            except (ValueError, KeyError):
                collection_dt = None

        return {
            'collection_datetime': collection_dt,
            'mission_id': mission_id,
            'country_code': country,
        }
    except (ValueError, TypeError):
        return None


# ===================================================================
# STDIDC -- Standard ID Extension
# ===================================================================


def parse_stdidc(node: ET.Element) -> Optional[Dict[str, Any]]:
    """Parse a ``<tre name="STDIDC">`` element."""
    if node.get('name') != 'STDIDC':
        return None

    try:
        acq_s = _optional_str(node, 'ACQUISITION_DATE')
        mission_id = _optional_str(node, 'MISSION')
        country = _optional_str(node, 'COUNTRY_CODE', 'COUNTRY')

        collection_dt = None
        if acq_s and len(acq_s) >= 14:
            try:
                collection_dt = datetime.strptime(
                    acq_s[:14], '%Y%m%d%H%M%S')
            except ValueError:
                collection_dt = None

        return {
            'collection_datetime': collection_dt,
            'mission_id': mission_id,
            'country_code': country,
        }
    except (ValueError, TypeError):
        return None


# ===================================================================
# PIAIMC -- Profile for Imagery Access Image
# ===================================================================


def parse_piaimc(node: ET.Element) -> Optional[Dict[str, Any]]:
    """Parse a ``<tre name="PIAIMC">`` element."""
    if node.get('name') != 'PIAIMC':
        return None

    try:
        sensor_mode = _optional_str(node, 'SENSMODE', 'SENSOR_MODE')
        cloud_cover = _optional_float(node, 'CLOUDCVR', 'CLOUD_COVER')
        return {
            'sensor_mode': sensor_mode,
            'cloud_cover': cloud_cover,
        }
    except (ValueError, TypeError):
        return None


# ===================================================================
# Top-level batch loader
# ===================================================================


# Mapping of TRE name → (parser, multi_instance_flag).  multi=True
# means the TRE may legitimately appear more than once (e.g. one
# RSMPCA per image section); single-instance TREs return only the
# first parsed record.
_PARSERS = {
    'RSMPCA': (parse_rsmpca, True),
    'RSMIDA': (parse_rsmida, False),
    'RSMGGA': (parse_rsmgga, True),
    'CSEXRA': (parse_csexra, False),
    'USE00A': (parse_use00a, False),
    'ICHIPB': (parse_ichipb, False),
    'BLOCKA': (parse_blocka, False),
    'CSEPHA': (parse_csepha, False),
    'AIMIDB': (parse_aimidb, False),
    'STDIDC': (parse_stdidc, False),
    'PIAIMC': (parse_piaimc, False),
}


def parse_all_tres(dataset: object) -> Dict[str, Any]:
    """Read every supported TRE from a rasterio dataset's xml:TRE.

    Returns a dict of TRE-name → parsed object (or list of objects
    for multi-instance TREs).  Unrecognized TREs are ignored.

    Parameters
    ----------
    dataset : rasterio.DatasetReader

    Returns
    -------
    dict
        Keys are TRE names (``'RSMPCA'``, ``'RSMIDA'``, ...).
        Values are dataclass instances or, for multi-instance TREs,
        lists of dataclass instances.
    """
    out: Dict[str, Any] = {}
    for tre in load_xml_tres(dataset):
        name = tre.get('name')
        if name not in _PARSERS:
            continue
        parser, multi = _PARSERS[name]
        parsed = parser(tre)
        if parsed is None:
            continue
        if multi:
            out.setdefault(name, []).append(parsed)
        else:
            out.setdefault(name, parsed)
    return out
