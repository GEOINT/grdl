# -*- coding: utf-8 -*-
"""
Airborne EO TRE metadata -- SENSRB, MENSRB, MENSRA, ACFTB dataclasses.

Typed metadata models for the airborne support data extensions
defined in STDI-0002:

* ``SENSRB`` -- General airborne sensor model (modular TRE).
* ``MENSRB`` / ``MENSRA`` -- Airborne mensuration data (fixed-width
  reference-point and direction-cosine fields).
* ``ACFTB`` -- Aircraft information (mission, tail number, sensor
  identification, GSDs, scene location).

SENSRB coverage is pragmatic: the General Data module, the mandatory
reference time/pixel and sensor position fields, and the Attitude
Euler Angles module are promoted to typed fields.  All other SENSRB
modules (Sensor Array Data, Sensor Calibration Data, Image Formation
Data, Attitude Unit Vectors, Attitude Quaternion, Sensor Velocity,
Point Sets, Time Stamped Data, Pixel Referenced Data, Uncertainty
Data, Additional Parameters) are retained verbatim in :attr:`SENSRBMetadata.raw`.

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
from dataclasses import dataclass, field
from typing import Dict, Optional


# ===================================================================
# SENSRB -- General airborne sensor model (STDI-0002-1 App Z)
# ===================================================================


@dataclass
class SENSRBMetadata:
    """SENSRB TRE -- airborne sensor model (pragmatic subset).

    Covered SENSRB modules
    ----------------------
    * **General Data** (``GENERAL_DATA=Y`` block) -- sensor and
      platform identification, geodetic system, units, and
      collection timing.
    * **Reference time / pixel** (mandatory) -- ``REFERENCE_TIME``,
      ``REFERENCE_ROW``, ``REFERENCE_COLUMN``.
    * **Sensor position** (mandatory) -- ``LATITUDE_OR_X``,
      ``LONGITUDE_OR_Y``, ``ALTITUDE_OR_Z`` and the sensor offsets.
    * **Attitude Euler Angles** (``ATTITUDE_EULER_ANGLES=Y`` block) --
      sensor angles and platform heading/pitch/roll.

    All other modules (Sensor Array Data, Sensor Calibration Data,
    Image Formation Data, Attitude Unit Vectors, Attitude
    Quaternion, Sensor Velocity, Point Sets, Time Stamped Data,
    Pixel Referenced Data, Uncertainty Data, Additional Parameters)
    are not promoted to typed fields but are retained in
    :attr:`raw` -- as field name/value pairs when parsed from the
    ``xml:TRE`` domain, or as raw fixed-width substrings (keyed by
    module flag name, plus ``'_unparsed'`` for the tail) when parsed
    from CEDATA.

    Parameters
    ----------
    sensor : str, optional
        Sensor name (``SENSOR``).
    sensor_uri : str, optional
        Sensor URI (``SENSOR_URI``).
    platform : str, optional
        Platform name (``PLATFORM``).
    platform_uri : str, optional
        Platform URI (``PLATFORM_URI``).
    operation_domain : str, optional
        Operation domain, e.g. ``AIRBORNE`` (``OPERATION_DOMAIN``).
    content_level : int, optional
        Content level 0-9 (``CONTENT_LEVEL``).
    geodetic_system : str, optional
        Geodetic system, e.g. ``WGS84`` (``GEODETIC_SYSTEM``).
    geodetic_type : str, optional
        Geodetic coordinate type: ``G`` (geodetic) or ``C``
        (Cartesian) (``GEODETIC_TYPE``).
    elevation_datum : str, optional
        Elevation datum, e.g. ``HAE`` or ``MSL``
        (``ELEVATION_DATUM``).
    length_unit : str, optional
        Length unit system (``LENGTH_UNIT``).
    angular_unit : str, optional
        Angular unit (``ANGULAR_UNIT``).
    start_date : str, optional
        Collection start date ``YYYYMMDD`` (``START_DATE``).
    start_time : float, optional
        Collection start time, seconds in day (``START_TIME``).
    end_date : str, optional
        Collection end date ``YYYYMMDD`` (``END_DATE``).
    end_time : float, optional
        Collection end time, seconds in day (``END_TIME``).
    generation_count : int, optional
        Product generation count (``GENERATION_COUNT``).
    generation_date : str, optional
        Product generation date (``GENERATION_DATE``).
    generation_time : str, optional
        Product generation time (``GENERATION_TIME``).
    reference_time : float, optional
        Reference time (``REFERENCE_TIME``).
    reference_row : float, optional
        Reference image row (``REFERENCE_ROW``).
    reference_column : float, optional
        Reference image column (``REFERENCE_COLUMN``).
    latitude_or_x : float, optional
        Sensor latitude (deg) or ECEF X (``LATITUDE_OR_X``).
    longitude_or_y : float, optional
        Sensor longitude (deg) or ECEF Y (``LONGITUDE_OR_Y``).
    altitude_or_z : float, optional
        Sensor altitude or ECEF Z (``ALTITUDE_OR_Z``).
    sensor_x_offset : float, optional
        Sensor X offset from platform reference
        (``SENSOR_X_OFFSET``).
    sensor_y_offset : float, optional
        Sensor Y offset (``SENSOR_Y_OFFSET``).
    sensor_z_offset : float, optional
        Sensor Z offset (``SENSOR_Z_OFFSET``).
    sensor_angle_model : int, optional
        Sensor angle model selector (``SENSOR_ANGLE_MODEL``).
    sensor_angle_1 : float, optional
        First sensor rotation angle (``SENSOR_ANGLE_1``).
    sensor_angle_2 : float, optional
        Second sensor rotation angle (``SENSOR_ANGLE_2``).
    sensor_angle_3 : float, optional
        Third sensor rotation angle (``SENSOR_ANGLE_3``).
    platform_relative : str, optional
        ``Y`` when sensor angles are platform-relative
        (``PLATFORM_RELATIVE``).
    platform_heading : float, optional
        Platform heading angle (``PLATFORM_HEADING``).
    platform_pitch : float, optional
        Platform pitch angle (``PLATFORM_PITCH``).
    platform_roll : float, optional
        Platform roll angle (``PLATFORM_ROLL``).
    raw : dict of str to str
        Uncovered module content (see class docstring).
    """

    sensor: Optional[str] = None
    sensor_uri: Optional[str] = None
    platform: Optional[str] = None
    platform_uri: Optional[str] = None
    operation_domain: Optional[str] = None
    content_level: Optional[int] = None
    geodetic_system: Optional[str] = None
    geodetic_type: Optional[str] = None
    elevation_datum: Optional[str] = None
    length_unit: Optional[str] = None
    angular_unit: Optional[str] = None
    start_date: Optional[str] = None
    start_time: Optional[float] = None
    end_date: Optional[str] = None
    end_time: Optional[float] = None
    generation_count: Optional[int] = None
    generation_date: Optional[str] = None
    generation_time: Optional[str] = None
    reference_time: Optional[float] = None
    reference_row: Optional[float] = None
    reference_column: Optional[float] = None
    latitude_or_x: Optional[float] = None
    longitude_or_y: Optional[float] = None
    altitude_or_z: Optional[float] = None
    sensor_x_offset: Optional[float] = None
    sensor_y_offset: Optional[float] = None
    sensor_z_offset: Optional[float] = None
    sensor_angle_model: Optional[int] = None
    sensor_angle_1: Optional[float] = None
    sensor_angle_2: Optional[float] = None
    sensor_angle_3: Optional[float] = None
    platform_relative: Optional[str] = None
    platform_heading: Optional[float] = None
    platform_pitch: Optional[float] = None
    platform_roll: Optional[float] = None
    raw: Dict[str, str] = field(default_factory=dict)


# ===================================================================
# MENSRB -- Airborne mensuration data (STDI-0002 ASDE)
# ===================================================================


@dataclass
class MENSRBMetadata:
    """MENSRB TRE -- airborne mensuration data.

    Fixed-width mensuration fields per STDI-0002 (205-byte layout).
    Aircraft altitude (``ACFT_ALT``) and reference point elevation
    (``RP_ELV``) are in feet per spec.

    Parameters
    ----------
    acft_loc : str, optional
        Aircraft location string (``ACFT_LOC``).
    acft_loc_accy : float, optional
        Aircraft location accuracy, feet (``ACFT_LOC_ACCY``).
    acft_alt : float, optional
        Aircraft altitude, feet MSL (``ACFT_ALT``).
    rp_loc : str, optional
        Reference point location string (``RP_LOC``).
    rp_loc_accy : float, optional
        Reference point location accuracy, feet (``RP_LOC_ACCY``).
    rp_elv : float, optional
        Reference point elevation, feet (``RP_ELV``).
    of_pc_r : float, optional
        Range offset of patch center (``OF_PC_R``).
    of_pc_a : float, optional
        Azimuth offset of patch center (``OF_PC_A``).
    cos_graze : float, optional
        Cosine of grazing angle (``COSGRZ``).
    range_crp : float, optional
        Range to common reference point, feet (``RGCRP``).
    rl_map : str, optional
        Right/left map indicator (``RLMAP``).
    rp_row : int, optional
        Reference point row (``RP_ROW``).
    rp_col : int, optional
        Reference point column (``RP_COL``).
    c_r_nc : float, optional
        Range unit vector north component (``C_R_NC``).
    c_r_ec : float, optional
        Range unit vector east component (``C_R_EC``).
    c_r_dc : float, optional
        Range unit vector down component (``C_R_DC``).
    c_az_nc : float, optional
        Azimuth unit vector north component (``C_AZ_NC``).
    c_az_ec : float, optional
        Azimuth unit vector east component (``C_AZ_EC``).
    c_az_dc : float, optional
        Azimuth unit vector down component (``C_AZ_DC``).
    c_al_nc : float, optional
        Altitude unit vector north component (``C_AL_NC``).
    c_al_ec : float, optional
        Altitude unit vector east component (``C_AL_EC``).
    c_al_dc : float, optional
        Altitude unit vector down component (``C_AL_DC``).
    total_tiles_cols : int, optional
        Total tiles in column direction (``TOTAL_TILES_COLS``).
    total_tiles_rows : int, optional
        Total tiles in row direction (``TOTAL_TILES_ROWS``).
    """

    acft_loc: Optional[str] = None
    acft_loc_accy: Optional[float] = None
    acft_alt: Optional[float] = None
    rp_loc: Optional[str] = None
    rp_loc_accy: Optional[float] = None
    rp_elv: Optional[float] = None
    of_pc_r: Optional[float] = None
    of_pc_a: Optional[float] = None
    cos_graze: Optional[float] = None
    range_crp: Optional[float] = None
    rl_map: Optional[str] = None
    rp_row: Optional[int] = None
    rp_col: Optional[int] = None
    c_r_nc: Optional[float] = None
    c_r_ec: Optional[float] = None
    c_r_dc: Optional[float] = None
    c_az_nc: Optional[float] = None
    c_az_ec: Optional[float] = None
    c_az_dc: Optional[float] = None
    c_al_nc: Optional[float] = None
    c_al_ec: Optional[float] = None
    c_al_dc: Optional[float] = None
    total_tiles_cols: Optional[int] = None
    total_tiles_rows: Optional[int] = None


# ===================================================================
# MENSRA -- Airborne mensuration data, legacy (STDI-0002 ASDE)
# ===================================================================


@dataclass
class MENSRAMetadata:
    """MENSRA TRE -- legacy airborne mensuration data.

    MENSRA exists in three fixed-width variants (CEL 155, 174, and
    185 bytes) with the same conceptual content; fields absent from
    a given variant are ``None``.  ``CCRP`` is the central command
    reference point.

    Parameters
    ----------
    acft_loc : str, optional
        Aircraft location string (``ACFT_LOC``).
    acft_alt : float, optional
        Aircraft altitude, feet MSL (``ACFT_ALT``).
    ccrp_loc : str, optional
        CCRP location string (``CCRP_LOC``).
    ccrp_alt : float, optional
        CCRP altitude, feet (``CCRP_ALT``).
    of_pc_r : float, optional
        Range offset of patch center (``OF_PC_R``).
    of_pc_a : float, optional
        Azimuth offset of patch center (``OF_PC_A``).
    cos_graze : float, optional
        Cosine of grazing angle (``COSGRZ`` / ``COSGZ``).
    range_ccrp : float, optional
        Range to CCRP, feet (``RGCCRP``).
    rl_map : str, optional
        Right/left map indicator (``RLMAP``).
    ccrp_row : int, optional
        CCRP row (``CCRP_ROW``).
    ccrp_col : int, optional
        CCRP column (``CCRP_COL``).
    c_r_nc : float, optional
        Range unit vector north component (``C_R_NC``).
    c_r_ec : float, optional
        Range unit vector east component (``C_R_EC``).
    c_r_dc : float, optional
        Range unit vector down component (``C_R_DC``).
    c_az_nc : float, optional
        Azimuth unit vector north component (``C_AZ_NC``).
    c_az_ec : float, optional
        Azimuth unit vector east component (``C_AZ_EC``).
    c_az_dc : float, optional
        Azimuth unit vector down component (``C_AZ_DC``).
    c_al_nc : float, optional
        Altitude unit vector north component (``C_AL_NC``).
    c_al_ec : float, optional
        Altitude unit vector east component (``C_AL_EC``).
    c_al_dc : float, optional
        Altitude unit vector down component (``C_AL_DC``).
    total_tiles_cols : int, optional
        Total tiles in column direction (185-byte variant only).
    total_tiles_rows : int, optional
        Total tiles in row direction (185-byte variant only).
    """

    acft_loc: Optional[str] = None
    acft_alt: Optional[float] = None
    ccrp_loc: Optional[str] = None
    ccrp_alt: Optional[float] = None
    of_pc_r: Optional[float] = None
    of_pc_a: Optional[float] = None
    cos_graze: Optional[float] = None
    range_ccrp: Optional[float] = None
    rl_map: Optional[str] = None
    ccrp_row: Optional[int] = None
    ccrp_col: Optional[int] = None
    c_r_nc: Optional[float] = None
    c_r_ec: Optional[float] = None
    c_r_dc: Optional[float] = None
    c_az_nc: Optional[float] = None
    c_az_ec: Optional[float] = None
    c_az_dc: Optional[float] = None
    c_al_nc: Optional[float] = None
    c_al_ec: Optional[float] = None
    c_al_dc: Optional[float] = None
    total_tiles_cols: Optional[int] = None
    total_tiles_rows: Optional[int] = None


# ===================================================================
# ACFTB -- Aircraft information (STDI-0002 ASDE)
# ===================================================================


@dataclass
class ACFTBMetadata:
    """ACFTB TRE -- aircraft information.

    Fixed 207-byte layout per STDI-0002.

    Parameters
    ----------
    mission_id : str, optional
        Aircraft mission identification (``AC_MSN_ID``).
    tail_number : str, optional
        Aircraft tail number (``AC_TAIL_NO``).
    takeoff_datetime : str, optional
        Aircraft takeoff date/time ``YYYYMMDDhhmm`` (``AC_TO``).
    sensor_id_type : str, optional
        Sensor ID type (``SENSOR_ID_TYPE``).
    sensor_id : str, optional
        Sensor identification (``SENSOR_ID``).
    scene_source : str, optional
        Scene source indicator (``SCENE_SOURCE``).
    scene_number : str, optional
        Scene number (``SCNUM``).
    processing_date : str, optional
        Processing date ``YYYYMMDD`` (``PDATE``).
    immediate_scene_host : str, optional
        Immediate scene host (``IMHOSTNO``).
    immediate_scene_request_id : str, optional
        Immediate scene request ID (``IMREQID``).
    mission_plan_mode : str, optional
        Mission plan mode (``MPLAN``).
    entry_location : str, optional
        Entry location string (``ENTLOC``).
    location_accuracy : float, optional
        Location accuracy, feet (``LOC_ACCY``).
    entry_elevation : float, optional
        Entry elevation (``ENTELV``).
    elevation_unit : str, optional
        Elevation unit code, ``f`` (feet) or ``m`` (meters)
        (``ELV_UNIT``).
    exit_location : str, optional
        Exit location string (``EXITLOC``).
    exit_elevation : float, optional
        Exit elevation (``EXITELV``).
    true_map_angle : float, optional
        True map angle, degrees (``TMAP``).
    row_gsd : float, optional
        Row ground sample distance (``ROW_SPACING``).
    row_gsd_units : str, optional
        Row GSD unit code (``ROW_SPACING_UNITS``).
    col_gsd : float, optional
        Column ground sample distance (``COL_SPACING``).
    col_gsd_units : str, optional
        Column GSD unit code (``COL_SPACING_UNITS``).
    focal_length : float, optional
        Sensor focal length, centimeters (``FOCAL_LENGTH``).
    sensor_serial : str, optional
        Sensor vendor serial number (``SENSERIAL``).
    software_version : str, optional
        Airborne software version (``ABSWVER``).
    calibration_date : str, optional
        Sensor calibration date ``YYYYMMDD`` (``CAL_DATE``).
    patch_total : int, optional
        Total number of patches (``PATCH_TOT``).
    mti_total : int, optional
        Total number of MTI packets (``MTI_TOT``).
    """

    mission_id: Optional[str] = None
    tail_number: Optional[str] = None
    takeoff_datetime: Optional[str] = None
    sensor_id_type: Optional[str] = None
    sensor_id: Optional[str] = None
    scene_source: Optional[str] = None
    scene_number: Optional[str] = None
    processing_date: Optional[str] = None
    immediate_scene_host: Optional[str] = None
    immediate_scene_request_id: Optional[str] = None
    mission_plan_mode: Optional[str] = None
    entry_location: Optional[str] = None
    location_accuracy: Optional[float] = None
    entry_elevation: Optional[float] = None
    elevation_unit: Optional[str] = None
    exit_location: Optional[str] = None
    exit_elevation: Optional[float] = None
    true_map_angle: Optional[float] = None
    row_gsd: Optional[float] = None
    row_gsd_units: Optional[str] = None
    col_gsd: Optional[float] = None
    col_gsd_units: Optional[str] = None
    focal_length: Optional[float] = None
    sensor_serial: Optional[str] = None
    software_version: Optional[str] = None
    calibration_date: Optional[str] = None
    patch_total: Optional[int] = None
    mti_total: Optional[int] = None
