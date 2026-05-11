# -*- coding: utf-8 -*-
"""
STANAG 4607 (GMTI) Metadata - Typed dataclasses for moving-target indicator.

STANAG 4607 is the NATO standard for Ground Moving Target Indicator (GMTI)
radar data. A 4607 file is a stream of fixed-format packets, each carrying
one or more typed segments (mission, dwell, job definition, target reports,
free text, platform location, ...). Unlike imagery formats, STANAG 4607
carries **vector detections** rather than pixel grids: each dwell segment
holds a list of target reports with range, velocity, lat/lon and SNR per
detected mover.

The metadata classes below mirror the binary segment structure one-to-one,
giving a typed Python representation that the reader and writer in
``grdl.IO.gmti`` round-trip against. ``STANAG4607Metadata`` is the
top-level container; it inherits from ``ImageMetadata`` only for dict-like
access compatibility, with non-applicable raster fields left as zeros.

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
2026-04-29

Modified
--------
2026-04-29
"""

# Standard library
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

# GRDL internal
from grdl.IO.models.base import ImageMetadata


# ---------------------------------------------------------------------------
# Packet header
# ---------------------------------------------------------------------------


@dataclass
class PacketHeader:
    """STANAG 4607 packet header (32 bytes).

    Attributes
    ----------
    version_id : str
        Two-character ASCII edition identifier (``'20'`` for Ed 2,
        ``'30'`` for Ed 3, ``'40'`` for Ed 4).
    packet_size : int
        Total packet size in bytes including the 32-byte header.
    nationality : str
        Two-character ISO-3166 nationality code (e.g., ``'US'``).
    classification : int
        Security classification (1=TopSecret, 2=Secret, 3=Confidential,
        4=Restricted, 5=Unclassified).
    class_system : str
        Two-character classification system (e.g., ``'XN'`` for NATO).
    code : int
        Packet security code, 16-bit bit-encoded field.
    exercise_indicator : int
        0=Operation, 1=Exercise (real), 2=Exercise (simulated).
    platform_id : str
        Up to 10 ASCII characters identifying the platform.
    mission_id : int
        32-bit unsigned mission identifier.
    job_id : int
        32-bit unsigned job identifier.
    """

    version_id: str = '40'
    packet_size: int = 0
    nationality: str = 'XN'
    classification: int = 5
    class_system: str = 'XN'
    code: int = 0
    exercise_indicator: int = 0
    platform_id: str = ''
    mission_id: int = 0
    job_id: int = 0


# ---------------------------------------------------------------------------
# Mission segment (segment type 1)
# ---------------------------------------------------------------------------


@dataclass
class MissionSegment:
    """Mission segment (M1-M7).

    Identifies the mission, platform configuration, and reference time
    used by all dwells in the file.

    Attributes
    ----------
    mission_plan : str
        Mission plan name (M1, up to 12 ASCII chars).
    flight_plan : str
        Flight plan name (M2, up to 12 ASCII chars).
    platform_type : int
        Platform type enumeration (M3, 0-128).
    platform_configuration : str
        Platform configuration code (M4, up to 10 ASCII chars).
    reference_time_year : int
        Year of mission reference time (M5).
    reference_time_month : int
        Month of mission reference time (M6, 1-12).
    reference_time_day : int
        Day of mission reference time (M7, 1-31).
    """

    mission_plan: str = ''
    flight_plan: str = ''
    platform_type: int = 0
    platform_configuration: str = ''
    reference_time_year: int = 2000
    reference_time_month: int = 1
    reference_time_day: int = 1


# ---------------------------------------------------------------------------
# Job definition segment (segment type 5)
# ---------------------------------------------------------------------------


@dataclass
class JobDefinitionSegment:
    """Job definition segment (J1-J28+).

    Describes the radar job: sensor model, the geographic area being
    surveyed (four bounding corners), MDV, and processing options.
    Required at the start of every job; unlike dwells it has no
    existence mask — every field is mandatory.

    Attributes
    ----------
    job_id : int
        Unique job identifier (J1).
    sensor_id_type : int
        Sensor type code (J2).
    sensor_id_model : str
        Sensor model name (J3, 6 ASCII chars).
    target_filtering_flag : int
        Filtering applied to the data (J4).
    priority : int
        Job priority (J5, 0=lowest, 99=highest).
    bounding_a_lat : float
        Bounding area corner A latitude (J6, deg WGS84).
    bounding_a_lon : float
        Bounding area corner A longitude (J7, deg WGS84).
    bounding_b_lat : float
        Bounding area corner B latitude (J8).
    bounding_b_lon : float
        Bounding area corner B longitude (J9).
    bounding_c_lat : float
        Bounding area corner C latitude (J10).
    bounding_c_lon : float
        Bounding area corner C longitude (J11).
    bounding_d_lat : float
        Bounding area corner D latitude (J12).
    bounding_d_lon : float
        Bounding area corner D longitude (J13).
    radar_mode : int
        Radar operating mode (J14).
    nominal_revisit_interval : int
        Nominal revisit interval (J15, deciseconds).
    nominal_uncertainty_along_track : int
        Sensor track uncertainty along sensor track (J16, decimeters).
    nominal_uncertainty_cross_track : int
        Sensor track uncertainty cross sensor track (J17, decimeters).
    nominal_uncertainty_altitude : int
        Sensor altitude uncertainty (J18, decimeters).
    nominal_uncertainty_track_heading : int
        Sensor track heading uncertainty (J19, deg / 120).
    nominal_uncertainty_speed : int
        Sensor speed uncertainty (J20, mm/s).
    nominal_sensor_slant_range_std : int
        Sensor slant range standard deviation (J21, centimeters).
    nominal_sensor_cross_range_std : float
        Sensor cross range standard deviation (J22, deg).
    nominal_sensor_los_velocity_std : int
        Sensor line-of-sight velocity standard deviation (J23, cm/s).
    nominal_sensor_mdv : int
        Sensor minimum detectable velocity (J24, dm/s).
    nominal_sensor_detection_probability : int
        Probability of detection (J25, 0-100 percent).
    nominal_sensor_false_alarm_density : int
        False alarm density (J26, 1/km^2).
    terrain_elevation_model : int
        Terrain elevation model used (J27).
    geoid_model : int
        Geoid model used (J28).
    """

    job_id: int = 0
    sensor_id_type: int = 0
    sensor_id_model: str = ''
    target_filtering_flag: int = 0
    priority: int = 0
    bounding_a_lat: float = 0.0
    bounding_a_lon: float = 0.0
    bounding_b_lat: float = 0.0
    bounding_b_lon: float = 0.0
    bounding_c_lat: float = 0.0
    bounding_c_lon: float = 0.0
    bounding_d_lat: float = 0.0
    bounding_d_lon: float = 0.0
    radar_mode: int = 0
    nominal_revisit_interval: int = 0
    nominal_uncertainty_along_track: int = 0
    nominal_uncertainty_cross_track: int = 0
    nominal_uncertainty_altitude: int = 0
    nominal_uncertainty_track_heading: int = 0
    nominal_uncertainty_speed: int = 0
    nominal_sensor_slant_range_std: int = 0
    nominal_sensor_cross_range_std: float = 0.0
    nominal_sensor_los_velocity_std: int = 0
    nominal_sensor_mdv: int = 0
    nominal_sensor_detection_probability: int = 0
    nominal_sensor_false_alarm_density: int = 0
    terrain_elevation_model: int = 0
    geoid_model: int = 0


# ---------------------------------------------------------------------------
# Target report (a list of these is embedded in each dwell segment)
# ---------------------------------------------------------------------------


@dataclass
class TargetReport:
    """Single GMTI target report (T1-T19, embedded in a dwell).

    Each target report describes one detected moving target: location,
    radial velocity, classification, and per-axis uncertainties.

    Attributes
    ----------
    report_index : int
        MTI report index within the dwell (T1).
    target_lat : float
        Target latitude (T2, deg WGS84).
    target_lon : float
        Target longitude (T3, deg WGS84).
    target_height : int
        Target height above ellipsoid (T4, meters).
    target_velocity_los : int
        Target line-of-sight (radial) velocity (T5, cm/s).
    target_wrap_velocity : int
        Target wrap velocity (T6, cm/s, unsigned). 0 means no wrap.
    target_snr : int
        Target SNR (T7, dB).
    target_classification : int
        Target classification enumeration (T8).
    target_class_probability : int
        Probability of correct classification (T9, 0-100 percent).
    slant_range_std : int
        Target slant-range standard deviation (T10, centimeters).
    cross_range_std : int
        Target cross-range standard deviation (T11, deg / 65536).
    height_std : int
        Target height standard deviation (T12, meters).
    velocity_los_std : int
        Target line-of-sight velocity standard deviation (T13, cm/s).
    truth_tag_application : int
        Truth tag application identifier (T14, ground truth tag).
    truth_tag_entity : int
        Truth tag entity identifier (T15).
    target_rcs : int
        Target radar cross section (T16, dB).
    """

    report_index: Optional[int] = None
    target_lat: Optional[float] = None
    target_lon: Optional[float] = None
    target_delta_lat: Optional[int] = None
    target_delta_lon: Optional[int] = None
    target_height: Optional[int] = None
    target_velocity_los: Optional[int] = None
    target_wrap_velocity: Optional[int] = None
    target_snr: Optional[int] = None
    target_classification: Optional[int] = None
    target_class_probability: Optional[int] = None
    slant_range_std: Optional[int] = None
    cross_range_std: Optional[int] = None
    height_std: Optional[int] = None
    velocity_los_std: Optional[int] = None
    truth_tag_application: Optional[int] = None
    truth_tag_entity: Optional[int] = None
    target_rcs: Optional[int] = None


# ---------------------------------------------------------------------------
# Dwell segment (segment type 2) — the heart of STANAG 4607
# ---------------------------------------------------------------------------


@dataclass
class DwellSegment:
    """Dwell segment (D1-D32 + a list of target reports).

    A dwell describes a single radar dwell at a fixed sensor position
    and pointing, accompanied by every moving-target detection found
    in that dwell. ``target_reports`` may be empty if no movers were
    detected.

    Attributes
    ----------
    existence_mask : int
        64-bit bitmask of which optional fields are present (D1).
        Required fields are always present and not represented in the mask.
    revisit_index : int
        Revisit index within the job (D2).
    dwell_index : int
        Dwell index within the revisit (D3).
    last_dwell_of_revisit : int
        1 if this is the last dwell of the current revisit, else 0 (D4).
    target_report_count : int
        Number of target reports in the dwell (D5).
    dwell_time_ms : int
        Time of dwell in milliseconds since the mission reference time (D6).
    sensor_pos_lat : float
        Sensor latitude (D7, deg WGS84).
    sensor_pos_lon : float
        Sensor longitude (D8, deg WGS84).
    sensor_pos_alt : int
        Sensor altitude (D9, centimeters above WGS84 ellipsoid).
    scale_factor_lat : Optional[float]
        Latitude scale factor (D10, deg).
    scale_factor_lon : Optional[float]
        Longitude scale factor (D11, deg).
    sensor_track : Optional[float]
        Sensor track angle (D17, deg from True North).
    sensor_speed : Optional[int]
        Sensor speed (D18, mm/s).
    sensor_vertical_velocity : Optional[int]
        Sensor vertical velocity (D19, dm/s).
    platform_orientation_heading : Optional[float]
        Platform orientation heading (D20, deg).
    platform_orientation_pitch : Optional[float]
        Platform orientation pitch (D21, deg).
    platform_orientation_roll : Optional[float]
        Platform orientation roll (D22, deg).
    dwell_center_lat : Optional[float]
        Dwell area center latitude (D23, deg WGS84).
    dwell_center_lon : Optional[float]
        Dwell area center longitude (D24, deg WGS84).
    dwell_range_half_extent : Optional[float]
        Dwell area range half-extent (D25, kilometers).
    dwell_angle_half_extent : Optional[float]
        Dwell area dwell-angle half-extent (D26, deg).
    sensor_orientation_heading : Optional[float]
        Sensor orientation heading (D27, deg).
    sensor_orientation_pitch : Optional[float]
        Sensor orientation pitch (D28, deg).
    sensor_orientation_roll : Optional[float]
        Sensor orientation roll (D29, deg).
    mdv : Optional[int]
        Minimum detectable velocity (D32, dm/s).
    target_reports : List[TargetReport]
        Per-target detections in this dwell.
    """

    existence_mask: int = 0
    revisit_index: int = 0
    dwell_index: int = 0
    last_dwell_of_revisit: int = 0
    target_report_count: int = 0
    dwell_time_ms: int = 0
    sensor_pos_lat: float = 0.0
    sensor_pos_lon: float = 0.0
    sensor_pos_alt: int = 0
    scale_factor_lat: Optional[float] = None
    scale_factor_lon: Optional[float] = None
    sensor_pos_unc_along_track: Optional[int] = None
    sensor_pos_unc_cross_track: Optional[int] = None
    sensor_pos_unc_altitude: Optional[int] = None
    sensor_track: Optional[float] = None
    sensor_speed: Optional[int] = None
    sensor_vertical_velocity: Optional[int] = None
    sensor_track_uncertainty: Optional[int] = None
    sensor_speed_uncertainty: Optional[int] = None
    sensor_vertical_velocity_uncertainty: Optional[int] = None
    platform_orientation_heading: Optional[float] = None
    platform_orientation_pitch: Optional[float] = None
    platform_orientation_roll: Optional[float] = None
    dwell_center_lat: Optional[float] = None
    dwell_center_lon: Optional[float] = None
    dwell_range_half_extent: Optional[float] = None
    dwell_angle_half_extent: Optional[float] = None
    sensor_orientation_heading: Optional[float] = None
    sensor_orientation_pitch: Optional[float] = None
    sensor_orientation_roll: Optional[float] = None
    mdv: Optional[int] = None
    target_reports: List[TargetReport] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Free-text segment (segment type 6)
# ---------------------------------------------------------------------------


@dataclass
class FreeTextSegment:
    """Free-text segment (F1-F3).

    Operator free-text annotations attached to a packet.

    Attributes
    ----------
    originator : str
        Originator identifier (F1, 10 ASCII chars).
    recipient : str
        Recipient identifier (F2, 10 ASCII chars).
    text : str
        Free text body (F3, variable-length ASCII).
    """

    originator: str = ''
    recipient: str = ''
    text: str = ''


# ---------------------------------------------------------------------------
# Platform location segment (segment type 13)
# ---------------------------------------------------------------------------


@dataclass
class PlatformLocationSegment:
    """Platform location segment (P1-P7).

    Higher-resolution platform position/attitude updates (typically
    1 Hz, faster than dwell rate).

    Attributes
    ----------
    location_time_ms : int
        Time of location relative to mission reference time (P1, ms).
    platform_lat : float
        Platform latitude (P2, deg WGS84).
    platform_lon : float
        Platform longitude (P3, deg WGS84).
    platform_alt : int
        Platform altitude (P4, centimeters HAE).
    platform_track : float
        Platform true track angle (P5, deg).
    platform_speed : int
        Platform ground speed (P6, mm/s).
    platform_vertical_velocity : int
        Platform vertical velocity (P7, dm/s).
    """

    location_time_ms: int = 0
    platform_lat: float = 0.0
    platform_lon: float = 0.0
    platform_alt: int = 0
    platform_track: float = 0.0
    platform_speed: int = 0
    platform_vertical_velocity: int = 0


# ---------------------------------------------------------------------------
# Top-level packet container
# ---------------------------------------------------------------------------


# Union type covering every segment payload supported by the v1 codec.
Segment = Union[
    MissionSegment,
    DwellSegment,
    JobDefinitionSegment,
    FreeTextSegment,
    PlatformLocationSegment,
]


@dataclass
class Packet:
    """One STANAG 4607 packet: a header plus an ordered list of segments.

    Attributes
    ----------
    header : PacketHeader
        Fully populated 32-byte packet header.
    segments : List[Segment]
        Ordered segments contained in this packet.
    """

    header: PacketHeader = field(default_factory=PacketHeader)
    segments: List[Segment] = field(default_factory=list)


# ---------------------------------------------------------------------------
# File-level metadata container
# ---------------------------------------------------------------------------


@dataclass
class STANAG4607Metadata(ImageMetadata):
    """File-level metadata for a STANAG 4607 GMTI mission file.

    Inherits from ``ImageMetadata`` only for dict-like access compatibility
    with the rest of GRDL's IO ecosystem; the universal raster fields
    (``rows``, ``cols``, ``dtype``) are zero/placeholder because GMTI is
    not raster data. The real content lives in ``packets``.

    Attributes
    ----------
    edition : int
        STANAG 4607 edition number (2, 3, or 4).
    packets : List[Packet]
        Ordered packets read from or to be written to the file.
    """

    # Override the parent's required raster fields with placeholder defaults
    # so callers can construct ``STANAG4607Metadata()`` without supplying
    # raster shape arguments that don't apply to this format.
    format: str = 'stanag4607'
    rows: int = 0
    cols: int = 0
    dtype: str = 'uint8'

    edition: int = 4
    packets: List[Packet] = field(default_factory=list)

    # -- Convenience views -------------------------------------------------

    @property
    def missions(self) -> List[MissionSegment]:
        """All MissionSegment instances across all packets, in order."""
        return [s for p in self.packets for s in p.segments
                if isinstance(s, MissionSegment)]

    @property
    def dwells(self) -> List[DwellSegment]:
        """All DwellSegment instances across all packets, in order."""
        return [s for p in self.packets for s in p.segments
                if isinstance(s, DwellSegment)]

    @property
    def job_definitions(self) -> List[JobDefinitionSegment]:
        """All JobDefinitionSegment instances across all packets, in order."""
        return [s for p in self.packets for s in p.segments
                if isinstance(s, JobDefinitionSegment)]

    @property
    def free_texts(self) -> List[FreeTextSegment]:
        """All FreeTextSegment instances across all packets, in order."""
        return [s for p in self.packets for s in p.segments
                if isinstance(s, FreeTextSegment)]

    @property
    def platform_locations(self) -> List[PlatformLocationSegment]:
        """All PlatformLocationSegment instances across all packets."""
        return [s for p in self.packets for s in p.segments
                if isinstance(s, PlatformLocationSegment)]

    @property
    def num_packets(self) -> int:
        return len(self.packets)

    @property
    def num_dwells(self) -> int:
        return sum(1 for _ in self.dwells)

    @property
    def num_target_reports(self) -> int:
        return sum(len(d.target_reports) for d in self.dwells)

    @property
    def time_bounds(self) -> Optional[Tuple[int, int]]:
        """Return (min, max) dwell time in ms across all dwells, or None.

        Returns
        -------
        tuple of int, or None
            ``(t_min_ms, t_max_ms)`` relative to the mission reference
            time. ``None`` if no dwells are present.
        """
        times = [d.dwell_time_ms for d in self.dwells]
        if not times:
            return None
        return (min(times), max(times))

    @property
    def geographic_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Return WGS84 bounding box across all target reports.

        Returns
        -------
        tuple of float, or None
            ``(min_lon, min_lat, max_lon, max_lat)`` over all target
            reports. ``None`` if no targets are present.
        """
        lats: List[float] = []
        lons: List[float] = []
        for d in self.dwells:
            for t in d.target_reports:
                lats.append(t.target_lat)
                lons.append(t.target_lon)
        if not lats:
            return None
        return (min(lons), min(lats), max(lons), max(lats))
