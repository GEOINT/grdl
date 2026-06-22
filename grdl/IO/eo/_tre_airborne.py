# -*- coding: utf-8 -*-
"""
SENSRB / MENSRB / MENSRA / ACFTB airborne EO TRE parsers.

Two entry points per TRE:

* ``parse_<name>(node)`` -- takes an ``<tre>``
  :class:`xml.etree.ElementTree.Element` from GDAL's ``xml:TRE``
  metadata domain (see :mod:`grdl.IO.eo._tre_xml`).
* ``parse_<name>_cedata(s)`` -- takes the raw fixed-width CEDATA
  string.  Returns ``None`` on any parse failure, never raises.

SENSRB coverage is pragmatic (see
:class:`grdl.IO.models.eo_airborne.SENSRBMetadata`): the General
Data module, the mandatory reference time/pixel and sensor position
fields, and the Attitude Euler Angles module are promoted to typed
fields; all other modules are retained verbatim in ``raw``.

MENSRA exists in three fixed-width variants (CEL 155, 174, 185);
the CEDATA parser dispatches on string length.

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
from typing import Dict, Optional
from xml.etree import ElementTree as ET

# GRDL internal
from grdl.IO.eo._tre_xml import (
    _optional_float,
    _optional_int,
    _optional_str,
)
from grdl.IO.models.eo_airborne import (
    ACFTBMetadata,
    MENSRAMetadata,
    MENSRBMetadata,
    SENSRBMetadata,
)


def _collect_raw_fields(node: ET.Element) -> Dict[str, str]:
    """Collect every ``<field>`` name/value pair under ``node``.

    Walks all descendants (including repeated groups); when a field
    name repeats, the last occurrence wins.  Used to retain
    uncovered SENSRB module content.
    """
    raw: Dict[str, str] = {}
    for fld in node.iter('field'):
        name = fld.get('name')
        if name:
            raw[name] = (fld.get('value') or '').strip()
    return raw


# ===================================================================
# SENSRB -- General airborne sensor model (STDI-0002-1 App Z)
# ===================================================================


def parse_sensrb(node: ET.Element) -> Optional[SENSRBMetadata]:
    """Parse a ``<tre name="SENSRB">`` element (pragmatic subset).

    Covered modules: General Data, mandatory reference time/pixel,
    mandatory sensor position, and Attitude Euler Angles.  Every
    field GDAL emitted -- including uncovered modules such as Sensor
    Array Data, Image Formation Data, attitude vectors/quaternions,
    velocity, point sets, and uncertainty data -- is retained as
    name/value pairs in :attr:`SENSRBMetadata.raw`.

    Parameters
    ----------
    node : xml.etree.ElementTree.Element
        ``<tre>`` element from the ``xml:TRE`` domain.

    Returns
    -------
    SENSRBMetadata or None
        ``None`` when the node is not a SENSRB TRE or is malformed.
    """
    if node.get('name') != 'SENSRB':
        return None

    try:
        return SENSRBMetadata(
            sensor=_optional_str(node, 'SENSOR'),
            sensor_uri=_optional_str(node, 'SENSOR_URI'),
            platform=_optional_str(node, 'PLATFORM'),
            platform_uri=_optional_str(node, 'PLATFORM_URI'),
            operation_domain=_optional_str(node, 'OPERATION_DOMAIN'),
            content_level=_optional_int(node, 'CONTENT_LEVEL'),
            geodetic_system=_optional_str(node, 'GEODETIC_SYSTEM'),
            geodetic_type=_optional_str(node, 'GEODETIC_TYPE'),
            elevation_datum=_optional_str(node, 'ELEVATION_DATUM'),
            length_unit=_optional_str(node, 'LENGTH_UNIT'),
            angular_unit=_optional_str(node, 'ANGULAR_UNIT'),
            start_date=_optional_str(node, 'START_DATE'),
            start_time=_optional_float(node, 'START_TIME'),
            end_date=_optional_str(node, 'END_DATE'),
            end_time=_optional_float(node, 'END_TIME'),
            generation_count=_optional_int(node, 'GENERATION_COUNT'),
            generation_date=_optional_str(node, 'GENERATION_DATE'),
            generation_time=_optional_str(node, 'GENERATION_TIME'),
            reference_time=_optional_float(node, 'REFERENCE_TIME'),
            reference_row=_optional_float(node, 'REFERENCE_ROW'),
            reference_column=_optional_float(node, 'REFERENCE_COLUMN'),
            latitude_or_x=_optional_float(node, 'LATITUDE_OR_X'),
            longitude_or_y=_optional_float(node, 'LONGITUDE_OR_Y'),
            altitude_or_z=_optional_float(node, 'ALTITUDE_OR_Z'),
            sensor_x_offset=_optional_float(node, 'SENSOR_X_OFFSET'),
            sensor_y_offset=_optional_float(node, 'SENSOR_Y_OFFSET'),
            sensor_z_offset=_optional_float(node, 'SENSOR_Z_OFFSET'),
            sensor_angle_model=_optional_int(node, 'SENSOR_ANGLE_MODEL'),
            sensor_angle_1=_optional_float(node, 'SENSOR_ANGLE_1'),
            sensor_angle_2=_optional_float(node, 'SENSOR_ANGLE_2'),
            sensor_angle_3=_optional_float(node, 'SENSOR_ANGLE_3'),
            platform_relative=_optional_str(node, 'PLATFORM_RELATIVE'),
            platform_heading=_optional_float(node, 'PLATFORM_HEADING'),
            platform_pitch=_optional_float(node, 'PLATFORM_PITCH'),
            platform_roll=_optional_float(node, 'PLATFORM_ROLL'),
            raw=_collect_raw_fields(node),
        )
    except (ValueError, TypeError):
        return None


# Fixed module content widths (excluding the 1-char Y/N flag).
_SENSRB_ARRAY_DATA_LEN = 77          # DETECTION..CALIBRATED
_SENSRB_CALIBRATION_DATA_LEN = 121   # CALIBRATION_UNIT..CALIBRATION_DATE
_SENSRB_FORMATION_FIXED_LEN = 86     # METHOD..FIRST_PIXEL_COLUMN
_SENSRB_TRANSFORM_PARAM_LEN = 12


def parse_sensrb_cedata(s: str) -> Optional[SENSRBMetadata]:
    """Parse a raw SENSRB CEDATA string (pragmatic subset).

    Walks the modular layout in TRE order: General Data (parsed),
    Sensor Array Data / Sensor Calibration Data / Image Formation
    Data (skipped by fixed width, content retained in ``raw`` keyed
    by module flag name), the mandatory reference and position
    fields (parsed), and Attitude Euler Angles (parsed).  Everything
    after the Euler module -- attitude vectors/quaternions, velocity,
    point sets, time-stamped/pixel-referenced sets, uncertainty and
    additional parameters -- is retained unparsed in
    ``raw['_unparsed']``.

    Returns
    -------
    SENSRBMetadata or None
        ``None`` on any parse failure.
    """
    try:
        v = s
        pos = 0
        raw: Dict[str, str] = {}

        def read(n: int) -> str:
            nonlocal pos
            if pos + n > len(v):
                raise ValueError('truncated SENSRB CEDATA')
            out = v[pos:pos + n]
            pos += n
            return out

        def read_str(n: int) -> Optional[str]:
            value = read(n).strip()
            return value if value else None

        def read_opt_float(n: int) -> Optional[float]:
            value = read(n).strip()
            if not value or value == '-':
                return None
            return float(value)

        def read_opt_int(n: int) -> Optional[int]:
            value = read(n).strip()
            if not value:
                return None
            return int(value)

        meta = SENSRBMetadata(raw=raw)

        # --- Module: General Data -----------------------------------
        if read(1) == 'Y':
            meta.sensor = read_str(25)
            meta.sensor_uri = read_str(32)
            meta.platform = read_str(25)
            meta.platform_uri = read_str(32)
            meta.operation_domain = read_str(10)
            meta.content_level = read_opt_int(1)
            meta.geodetic_system = read_str(5)
            meta.geodetic_type = read_str(1)
            meta.elevation_datum = read_str(3)
            meta.length_unit = read_str(2)
            meta.angular_unit = read_str(3)
            meta.start_date = read_str(8)
            meta.start_time = read_opt_float(14)
            meta.end_date = read_str(8)
            meta.end_time = read_opt_float(14)
            meta.generation_count = read_opt_int(2)
            meta.generation_date = read_str(8)
            meta.generation_time = read_str(10)

        # --- Module: Sensor Array Data (retained, not parsed) -------
        if read(1) == 'Y':
            raw['SENSOR_ARRAY_DATA'] = read(_SENSRB_ARRAY_DATA_LEN)

        # --- Module: Sensor Calibration Data (retained) -------------
        if read(1) == 'Y':
            raw['SENSOR_CALIBRATION_DATA'] = read(
                _SENSRB_CALIBRATION_DATA_LEN)

        # --- Module: Image Formation Data (retained) ----------------
        if read(1) == 'Y':
            fixed = read(_SENSRB_FORMATION_FIXED_LEN)
            n_params_s = read(1)                  # TRANSFORM_PARAMS
            n_params = int(n_params_s)
            params = read(n_params * _SENSRB_TRANSFORM_PARAM_LEN)
            raw['IMAGE_FORMATION_DATA'] = fixed + n_params_s + params

        # --- Mandatory: reference time / pixel ----------------------
        meta.reference_time = read_opt_float(12)   # REFERENCE_TIME
        meta.reference_row = read_opt_float(8)     # REFERENCE_ROW
        meta.reference_column = read_opt_float(8)  # REFERENCE_COLUMN

        # --- Mandatory: sensor position ------------------------------
        meta.latitude_or_x = read_opt_float(11)    # LATITUDE_OR_X
        meta.longitude_or_y = read_opt_float(12)   # LONGITUDE_OR_Y
        meta.altitude_or_z = read_opt_float(11)    # ALTITUDE_OR_Z
        meta.sensor_x_offset = read_opt_float(8)   # SENSOR_X_OFFSET
        meta.sensor_y_offset = read_opt_float(8)   # SENSOR_Y_OFFSET
        meta.sensor_z_offset = read_opt_float(8)   # SENSOR_Z_OFFSET

        # --- Module: Attitude Euler Angles ---------------------------
        if read(1) == 'Y':
            meta.sensor_angle_model = read_opt_int(1)
            meta.sensor_angle_1 = read_opt_float(10)
            meta.sensor_angle_2 = read_opt_float(9)
            meta.sensor_angle_3 = read_opt_float(10)
            meta.platform_relative = read_str(1)
            meta.platform_heading = read_opt_float(9)
            meta.platform_pitch = read_opt_float(9)
            meta.platform_roll = read_opt_float(10)

        # --- Remaining modules retained unparsed ----------------------
        if pos < len(v):
            raw['_unparsed'] = v[pos:]

        return meta
    except (ValueError, TypeError, IndexError):
        return None


# ===================================================================
# MENSRB -- Airborne mensuration data (STDI-0002 ASDE)
# ===================================================================


def parse_mensrb(node: ET.Element) -> Optional[MENSRBMetadata]:
    """Parse a ``<tre name="MENSRB">`` element.

    Field names follow STDI-0002 / GDAL's NITF TRE spec verbatim.

    Returns
    -------
    MENSRBMetadata or None
        ``None`` when the node is not a MENSRB TRE or is malformed.
    """
    if node.get('name') != 'MENSRB':
        return None

    try:
        return MENSRBMetadata(
            acft_loc=_optional_str(node, 'ACFT_LOC'),
            acft_loc_accy=_optional_float(node, 'ACFT_LOC_ACCY'),
            acft_alt=_optional_float(node, 'ACFT_ALT'),
            rp_loc=_optional_str(node, 'RP_LOC'),
            rp_loc_accy=_optional_float(node, 'RP_LOC_ACCY'),
            rp_elv=_optional_float(node, 'RP_ELV'),
            of_pc_r=_optional_float(node, 'OF_PC_R'),
            of_pc_a=_optional_float(node, 'OF_PC_A'),
            cos_graze=_optional_float(node, 'COSGRZ'),
            range_crp=_optional_float(node, 'RGCRP'),
            rl_map=_optional_str(node, 'RLMAP'),
            rp_row=_optional_int(node, 'RP_ROW'),
            rp_col=_optional_int(node, 'RP_COL'),
            c_r_nc=_optional_float(node, 'C_R_NC'),
            c_r_ec=_optional_float(node, 'C_R_EC'),
            c_r_dc=_optional_float(node, 'C_R_DC'),
            c_az_nc=_optional_float(node, 'C_AZ_NC'),
            c_az_ec=_optional_float(node, 'C_AZ_EC'),
            c_az_dc=_optional_float(node, 'C_AZ_DC'),
            c_al_nc=_optional_float(node, 'C_AL_NC'),
            c_al_ec=_optional_float(node, 'C_AL_EC'),
            c_al_dc=_optional_float(node, 'C_AL_DC'),
            total_tiles_cols=_optional_int(node, 'TOTAL_TILES_COLS'),
            total_tiles_rows=_optional_int(node, 'TOTAL_TILES_ROWS'),
        )
    except (ValueError, TypeError):
        return None


def parse_mensrb_cedata(s: str) -> Optional[MENSRBMetadata]:
    """Parse a raw MENSRB CEDATA string.

    Fixed-width layout (205 bytes) per STDI-0002::

        ACFT_LOC(25) ACFT_LOC_ACCY(6) ACFT_ALT(6)
        RP_LOC(25) RP_LOC_ACCY(6) RP_ELV(6)
        OF_PC_R(7) OF_PC_A(7) COSGRZ(7) RGCRP(7) RLMAP(1)
        RP_ROW(5) RP_COL(5)
        C_R_NC(10) C_R_EC(10) C_R_DC(10)
        C_AZ_NC(9) C_AZ_EC(9) C_AZ_DC(9)
        C_AL_NC(9) C_AL_EC(9) C_AL_DC(9)
        TOTAL_TILES_COLS(3) TOTAL_TILES_ROWS(5)

    Returns
    -------
    MENSRBMetadata or None
        ``None`` on any parse failure.
    """
    try:
        v = s
        if len(v) < 205:
            return None
        pos = 0

        def read(n: int) -> str:
            nonlocal pos
            out = v[pos:pos + n]
            pos += n
            return out

        def read_str(n: int) -> Optional[str]:
            value = read(n).strip()
            return value if value else None

        def read_opt_float(n: int) -> Optional[float]:
            value = read(n).strip()
            if not value or value == '-':
                return None
            return float(value)

        def read_opt_int(n: int) -> Optional[int]:
            value = read(n).strip()
            if not value:
                return None
            return int(value)

        return MENSRBMetadata(
            acft_loc=read_str(25),
            acft_loc_accy=read_opt_float(6),
            acft_alt=read_opt_float(6),
            rp_loc=read_str(25),
            rp_loc_accy=read_opt_float(6),
            rp_elv=read_opt_float(6),
            of_pc_r=read_opt_float(7),
            of_pc_a=read_opt_float(7),
            cos_graze=read_opt_float(7),
            range_crp=read_opt_float(7),
            rl_map=read_str(1),
            rp_row=read_opt_int(5),
            rp_col=read_opt_int(5),
            c_r_nc=read_opt_float(10),
            c_r_ec=read_opt_float(10),
            c_r_dc=read_opt_float(10),
            c_az_nc=read_opt_float(9),
            c_az_ec=read_opt_float(9),
            c_az_dc=read_opt_float(9),
            c_al_nc=read_opt_float(9),
            c_al_ec=read_opt_float(9),
            c_al_dc=read_opt_float(9),
            total_tiles_cols=read_opt_int(3),
            total_tiles_rows=read_opt_int(5),
        )
    except (ValueError, TypeError, IndexError):
        return None


# ===================================================================
# MENSRA -- Airborne mensuration data, legacy (STDI-0002 ASDE)
# ===================================================================


def parse_mensra(node: ET.Element) -> Optional[MENSRAMetadata]:
    """Parse a ``<tre name="MENSRA">`` element.

    Accepts the field-name variants across MENSRA versions
    (``COSGRZ`` vs ``COSGZ``).

    Returns
    -------
    MENSRAMetadata or None
        ``None`` when the node is not a MENSRA TRE or is malformed.
    """
    if node.get('name') != 'MENSRA':
        return None

    try:
        return MENSRAMetadata(
            acft_loc=_optional_str(node, 'ACFT_LOC'),
            acft_alt=_optional_float(node, 'ACFT_ALT'),
            ccrp_loc=_optional_str(node, 'CCRP_LOC'),
            ccrp_alt=_optional_float(node, 'CCRP_ALT'),
            of_pc_r=_optional_float(node, 'OF_PC_R'),
            of_pc_a=_optional_float(node, 'OF_PC_A'),
            cos_graze=_optional_float(node, 'COSGRZ', 'COSGZ'),
            range_ccrp=_optional_float(node, 'RGCCRP'),
            rl_map=_optional_str(node, 'RLMAP'),
            ccrp_row=_optional_int(node, 'CCRP_ROW'),
            ccrp_col=_optional_int(node, 'CCRP_COL'),
            c_r_nc=_optional_float(node, 'C_R_NC'),
            c_r_ec=_optional_float(node, 'C_R_EC'),
            c_r_dc=_optional_float(node, 'C_R_DC'),
            c_az_nc=_optional_float(node, 'C_AZ_NC'),
            c_az_ec=_optional_float(node, 'C_AZ_EC'),
            c_az_dc=_optional_float(node, 'C_AZ_DC'),
            c_al_nc=_optional_float(node, 'C_AL_NC'),
            c_al_ec=_optional_float(node, 'C_AL_EC'),
            c_al_dc=_optional_float(node, 'C_AL_DC'),
            total_tiles_cols=_optional_int(node, 'TOTAL_TILES_COLS'),
            total_tiles_rows=_optional_int(node, 'TOTAL_TILES_ROWS'),
        )
    except (ValueError, TypeError):
        return None


def parse_mensra_cedata(s: str) -> Optional[MENSRAMetadata]:
    """Parse a raw MENSRA CEDATA string.

    MENSRA has three fixed-width variants, dispatched on the exact
    CEDATA length:

    * **155 bytes** -- CCRP block first, 21-char locations, 7-char
      direction cosines, no tile counts.
    * **174 bytes** -- aircraft block first, 21-char locations,
      9-char direction cosines, no tile counts.
    * **185 bytes** -- aircraft block first, 25-char locations,
      10/9-char direction cosines, plus ``TOTAL_TILES_COLS`` /
      ``TOTAL_TILES_ROWS``.

    Returns
    -------
    MENSRAMetadata or None
        ``None`` when the length matches no known variant or any
        field fails to parse.
    """
    try:
        v = s
        pos = 0

        def read(n: int) -> str:
            nonlocal pos
            out = v[pos:pos + n]
            pos += n
            return out

        def read_str(n: int) -> Optional[str]:
            value = read(n).strip()
            return value if value else None

        def read_opt_float(n: int) -> Optional[float]:
            value = read(n).strip()
            if not value or value == '-':
                return None
            return float(value)

        def read_opt_int(n: int) -> Optional[int]:
            value = read(n).strip()
            if not value:
                return None
            return int(value)

        meta = MENSRAMetadata()
        n = len(v)

        if n == 155:
            meta.ccrp_loc = read_str(21)
            meta.ccrp_alt = read_opt_float(6)
            meta.of_pc_r = read_opt_float(7)
            meta.of_pc_a = read_opt_float(7)
            meta.cos_graze = read_opt_float(7)
            meta.range_ccrp = read_opt_float(7)
            meta.rl_map = read_str(1)
            meta.ccrp_row = read_opt_int(5)
            meta.ccrp_col = read_opt_int(5)
            meta.acft_loc = read_str(21)
            meta.acft_alt = read_opt_float(5)
            cosine_widths = (7, 7, 7, 7, 7, 7, 7, 7, 7)
        elif n == 174:
            meta.acft_loc = read_str(21)
            meta.acft_alt = read_opt_float(6)
            meta.ccrp_loc = read_str(21)
            meta.ccrp_alt = read_opt_float(6)
            meta.of_pc_r = read_opt_float(7)
            meta.of_pc_a = read_opt_float(7)
            meta.cos_graze = read_opt_float(7)
            meta.range_ccrp = read_opt_float(7)
            meta.rl_map = read_str(1)
            meta.ccrp_row = read_opt_int(5)
            meta.ccrp_col = read_opt_int(5)
            cosine_widths = (9, 9, 9, 9, 9, 9, 9, 9, 9)
        elif n == 185:
            meta.acft_loc = read_str(25)
            meta.acft_alt = read_opt_float(6)
            meta.ccrp_loc = read_str(25)
            meta.ccrp_alt = read_opt_float(6)
            meta.of_pc_r = read_opt_float(7)
            meta.of_pc_a = read_opt_float(7)
            meta.cos_graze = read_opt_float(7)
            meta.range_ccrp = read_opt_float(7)
            meta.rl_map = read_str(1)
            meta.ccrp_row = read_opt_int(5)
            meta.ccrp_col = read_opt_int(5)
            cosine_widths = (10, 10, 10, 9, 9, 9, 9, 9, 9)
        else:
            return None

        (meta.c_r_nc, meta.c_r_ec, meta.c_r_dc,
         meta.c_az_nc, meta.c_az_ec, meta.c_az_dc,
         meta.c_al_nc, meta.c_al_ec, meta.c_al_dc) = [
            read_opt_float(w) for w in cosine_widths
        ]

        if n == 185:
            meta.total_tiles_cols = read_opt_int(3)
            meta.total_tiles_rows = read_opt_int(5)

        return meta
    except (ValueError, TypeError, IndexError):
        return None


# ===================================================================
# ACFTB -- Aircraft information (STDI-0002 ASDE)
# ===================================================================


def parse_acftb(node: ET.Element) -> Optional[ACFTBMetadata]:
    """Parse a ``<tre name="ACFTB">`` element.

    Field names follow STDI-0002 / GDAL's NITF TRE spec verbatim.

    Returns
    -------
    ACFTBMetadata or None
        ``None`` when the node is not an ACFTB TRE or is malformed.
    """
    if node.get('name') != 'ACFTB':
        return None

    try:
        return ACFTBMetadata(
            mission_id=_optional_str(node, 'AC_MSN_ID'),
            tail_number=_optional_str(node, 'AC_TAIL_NO'),
            takeoff_datetime=_optional_str(node, 'AC_TO'),
            sensor_id_type=_optional_str(node, 'SENSOR_ID_TYPE'),
            sensor_id=_optional_str(node, 'SENSOR_ID'),
            scene_source=_optional_str(node, 'SCENE_SOURCE'),
            scene_number=_optional_str(node, 'SCNUM'),
            processing_date=_optional_str(node, 'PDATE'),
            immediate_scene_host=_optional_str(node, 'IMHOSTNO'),
            immediate_scene_request_id=_optional_str(node, 'IMREQID'),
            mission_plan_mode=_optional_str(node, 'MPLAN'),
            entry_location=_optional_str(node, 'ENTLOC'),
            location_accuracy=_optional_float(node, 'LOC_ACCY'),
            entry_elevation=_optional_float(node, 'ENTELV'),
            elevation_unit=_optional_str(node, 'ELV_UNIT'),
            exit_location=_optional_str(node, 'EXITLOC'),
            exit_elevation=_optional_float(node, 'EXITELV'),
            true_map_angle=_optional_float(node, 'TMAP'),
            row_gsd=_optional_float(node, 'ROW_SPACING'),
            row_gsd_units=_optional_str(node, 'ROW_SPACING_UNITS'),
            col_gsd=_optional_float(node, 'COL_SPACING'),
            col_gsd_units=_optional_str(node, 'COL_SPACING_UNITS'),
            focal_length=_optional_float(node, 'FOCAL_LENGTH'),
            sensor_serial=_optional_str(node, 'SENSERIAL'),
            software_version=_optional_str(node, 'ABSWVER'),
            calibration_date=_optional_str(node, 'CAL_DATE'),
            patch_total=_optional_int(node, 'PATCH_TOT'),
            mti_total=_optional_int(node, 'MTI_TOT'),
        )
    except (ValueError, TypeError):
        return None


def parse_acftb_cedata(s: str) -> Optional[ACFTBMetadata]:
    """Parse a raw ACFTB CEDATA string.

    Fixed-width layout (207 bytes) per STDI-0002::

        AC_MSN_ID(20) AC_TAIL_NO(10) AC_TO(12)
        SENSOR_ID_TYPE(4) SENSOR_ID(6) SCENE_SOURCE(1) SCNUM(6)
        PDATE(8) IMHOSTNO(6) IMREQID(5) MPLAN(3)
        ENTLOC(25) LOC_ACCY(6) ENTELV(6) ELV_UNIT(1)
        EXITLOC(25) EXITELV(6) TMAP(7)
        ROW_SPACING(7) ROW_SPACING_UNITS(1)
        COL_SPACING(7) COL_SPACING_UNITS(1)
        FOCAL_LENGTH(6) SENSERIAL(6) ABSWVER(7) CAL_DATE(8)
        PATCH_TOT(4) MTI_TOT(3)

    Returns
    -------
    ACFTBMetadata or None
        ``None`` on any parse failure.
    """
    try:
        v = s
        if len(v) < 207:
            return None
        pos = 0

        def read(n: int) -> str:
            nonlocal pos
            out = v[pos:pos + n]
            pos += n
            return out

        def read_str(n: int) -> Optional[str]:
            value = read(n).strip()
            return value if value else None

        def read_opt_float(n: int) -> Optional[float]:
            value = read(n).strip()
            if not value or value == '-':
                return None
            return float(value)

        def read_opt_int(n: int) -> Optional[int]:
            value = read(n).strip()
            if not value:
                return None
            return int(value)

        return ACFTBMetadata(
            mission_id=read_str(20),
            tail_number=read_str(10),
            takeoff_datetime=read_str(12),
            sensor_id_type=read_str(4),
            sensor_id=read_str(6),
            scene_source=read_str(1),
            scene_number=read_str(6),
            processing_date=read_str(8),
            immediate_scene_host=read_str(6),
            immediate_scene_request_id=read_str(5),
            mission_plan_mode=read_str(3),
            entry_location=read_str(25),
            location_accuracy=read_opt_float(6),
            entry_elevation=read_opt_float(6),
            elevation_unit=read_str(1),
            exit_location=read_str(25),
            exit_elevation=read_opt_float(6),
            true_map_angle=read_opt_float(7),
            row_gsd=read_opt_float(7),
            row_gsd_units=read_str(1),
            col_gsd=read_opt_float(7),
            col_gsd_units=read_str(1),
            focal_length=read_opt_float(6),
            sensor_serial=read_str(6),
            software_version=read_str(7),
            calibration_date=read_str(8),
            patch_total=read_opt_int(4),
            mti_total=read_opt_int(3),
        )
    except (ValueError, TypeError, IndexError):
        return None
