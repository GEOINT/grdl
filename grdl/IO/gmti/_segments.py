# -*- coding: utf-8 -*-
"""
STANAG 4607 binary codec - per-segment serializers and parsers.

Encodes and decodes the wire format of every STANAG 4607 segment
supported by this module. Backed by ``struct`` from the standard
library; no third-party binary parsing dependency.

The codec is field-table driven: each segment type has a list of
``(attribute_name, type_code)`` tuples defining the byte layout. The
generic ``pack_field``/``unpack_field`` helpers consume that table and
the corresponding dataclass attributes.

Every field listed in a table is always written (no existence-mask
conditional). The ``existence_mask`` attribute of a ``DwellSegment``
is stored as-is; the writer fills it with all-1s for fields that are
present, and the reader returns it without interpretation. Real-world
files with selective masks may not round-trip with this v1 codec — the
plan documents this as a deferred concern until a sample file is
available for validation.

Type codes
----------
``I{8,16,32}``: signed two's-complement big-endian integers.
``B{8,16,32,64}`` / ``H{8,16,32}``: unsigned big-endian integers.
``S{8,16,32}``: signed integers (alias of ``I*``).
``F32``: IEEE-754 single-precision float, big-endian.
``BA{16,32}``: unsigned binary angle (longitude). Scale = ``360 / 2^N``
deg per LSB.
``SA{16,32}``: signed binary angle (latitude). Same scale, range
``[-180, +180)`` (latitudes use only ``[-90, +90]``).
``A<N>``: fixed-length ASCII string, space-padded on the right.

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
import struct
from typing import Any, Callable, Dict, List, Tuple

# GRDL internal
from grdl.IO.models.stanag4607 import (
    DwellSegment,
    FreeTextSegment,
    JobDefinitionSegment,
    MissionSegment,
    PacketHeader,
    PlatformLocationSegment,
    TargetReport,
)

# Segment type identifiers from the STANAG 4607 standard.
SEG_MISSION = 1
SEG_DWELL = 2
SEG_HRR = 3
SEG_JOB_DEFINITION = 5
SEG_FREE_TEXT = 6
SEG_PLATFORM_LOCATION = 13

PACKET_HEADER_SIZE = 32
SEGMENT_HEADER_SIZE = 5

# Existence-mask bit pattern used by the writer when all dwell optional
# fields are populated. Real-world files may use selective masks; the
# v1 codec writes everything and reads everything.
_DWELL_MASK_ALL = 0xFFFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# Binary-angle helpers
# ---------------------------------------------------------------------------


def _pack_ba32(deg: float) -> bytes:
    deg = float(deg) % 360.0
    raw = int(round(deg * (1 << 32) / 360.0)) & 0xFFFFFFFF
    return struct.pack('>I', raw)


def _unpack_ba32(buf: bytes) -> float:
    raw = struct.unpack('>I', buf)[0]
    return raw * (360.0 / (1 << 32))


def _pack_ba16(deg: float) -> bytes:
    deg = float(deg) % 360.0
    raw = int(round(deg * (1 << 16) / 360.0)) & 0xFFFF
    return struct.pack('>H', raw)


def _unpack_ba16(buf: bytes) -> float:
    raw = struct.unpack('>H', buf)[0]
    return raw * (360.0 / (1 << 16))


def _pack_sa32(deg: float) -> bytes:
    raw = int(round(float(deg) * (1 << 32) / 360.0))
    raw = max(-(1 << 31), min((1 << 31) - 1, raw))
    return struct.pack('>i', raw)


def _unpack_sa32(buf: bytes) -> float:
    raw = struct.unpack('>i', buf)[0]
    return raw * (360.0 / (1 << 32))


def _pack_sa16(deg: float) -> bytes:
    raw = int(round(float(deg) * (1 << 16) / 360.0))
    raw = max(-(1 << 15), min((1 << 15) - 1, raw))
    return struct.pack('>h', raw)


def _unpack_sa16(buf: bytes) -> float:
    raw = struct.unpack('>h', buf)[0]
    return raw * (360.0 / (1 << 16))


# ---------------------------------------------------------------------------
# Type codec registry
# ---------------------------------------------------------------------------


# Each entry is (size_bytes, pack_fn, unpack_fn).
_TYPE_CODECS: Dict[str, Tuple[int, Callable[[Any], bytes], Callable[[bytes], Any]]] = {
    'I8':   (1, lambda v: struct.pack('>b', int(v)),
             lambda b: struct.unpack('>b', b)[0]),
    'I16':  (2, lambda v: struct.pack('>h', int(v)),
             lambda b: struct.unpack('>h', b)[0]),
    'I32':  (4, lambda v: struct.pack('>i', int(v)),
             lambda b: struct.unpack('>i', b)[0]),
    'S8':   (1, lambda v: struct.pack('>b', int(v)),
             lambda b: struct.unpack('>b', b)[0]),
    'S16':  (2, lambda v: struct.pack('>h', int(v)),
             lambda b: struct.unpack('>h', b)[0]),
    'S32':  (4, lambda v: struct.pack('>i', int(v)),
             lambda b: struct.unpack('>i', b)[0]),
    'B8':   (1, lambda v: struct.pack('>B', int(v) & 0xFF),
             lambda b: struct.unpack('>B', b)[0]),
    'B16':  (2, lambda v: struct.pack('>H', int(v) & 0xFFFF),
             lambda b: struct.unpack('>H', b)[0]),
    'B32':  (4, lambda v: struct.pack('>I', int(v) & 0xFFFFFFFF),
             lambda b: struct.unpack('>I', b)[0]),
    'B64':  (8, lambda v: struct.pack('>Q', int(v) & 0xFFFFFFFFFFFFFFFF),
             lambda b: struct.unpack('>Q', b)[0]),
    'H8':   (1, lambda v: struct.pack('>B', int(v) & 0xFF),
             lambda b: struct.unpack('>B', b)[0]),
    'H16':  (2, lambda v: struct.pack('>H', int(v) & 0xFFFF),
             lambda b: struct.unpack('>H', b)[0]),
    'H32':  (4, lambda v: struct.pack('>I', int(v) & 0xFFFFFFFF),
             lambda b: struct.unpack('>I', b)[0]),
    'F32':  (4, lambda v: struct.pack('>f', float(v)),
             lambda b: struct.unpack('>f', b)[0]),
    'BA32': (4, _pack_ba32, _unpack_ba32),
    'BA16': (2, _pack_ba16, _unpack_ba16),
    'SA32': (4, _pack_sa32, _unpack_sa32),
    'SA16': (2, _pack_sa16, _unpack_sa16),
}


def _type_size(type_code: str) -> int:
    """Return the encoded byte size of a STANAG 4607 type code."""
    if type_code.startswith('A'):
        return int(type_code[1:])
    return _TYPE_CODECS[type_code][0]


def pack_field(value: Any, type_code: str) -> bytes:
    """Encode a Python value to its STANAG 4607 wire bytes.

    Parameters
    ----------
    value : Any
        Python value to encode. ``None`` is treated as zero / empty.
    type_code : str
        STANAG 4607 type code (e.g., ``'I32'``, ``'BA32'``, ``'A12'``).

    Returns
    -------
    bytes
        Wire bytes (length determined by the type code).
    """
    if type_code.startswith('A'):
        n = int(type_code[1:])
        if value is None:
            value = ''
        s = str(value).encode('ascii', errors='replace')[:n]
        return s.ljust(n, b' ')
    if value is None:
        value = 0
    return _TYPE_CODECS[type_code][1](value)


def unpack_field(buf: bytes, offset: int, type_code: str) -> Tuple[Any, int]:
    """Decode a STANAG 4607 field from a buffer at *offset*.

    Parameters
    ----------
    buf : bytes
        Full byte buffer being parsed.
    offset : int
        Byte offset into ``buf`` where the field starts.
    type_code : str
        STANAG 4607 type code.

    Returns
    -------
    tuple
        ``(value, new_offset)``. ``new_offset`` advances past the field.
    """
    size = _type_size(type_code)
    chunk = buf[offset:offset + size]
    if len(chunk) < size:
        raise ValueError(
            f"Buffer underrun: need {size} bytes for {type_code} at "
            f"offset {offset}, got {len(chunk)}"
        )
    if type_code.startswith('A'):
        s = chunk.decode('ascii', errors='replace').rstrip()
        return s, offset + size
    return _TYPE_CODECS[type_code][2](chunk), offset + size


# ---------------------------------------------------------------------------
# Field tables — one per segment type
# ---------------------------------------------------------------------------


_PACKET_HEADER_FIELDS: List[Tuple[str, str]] = [
    ('version_id', 'A2'),
    ('packet_size', 'I32'),
    ('nationality', 'A2'),
    ('classification', 'B8'),
    ('class_system', 'A2'),
    ('code', 'B16'),
    ('exercise_indicator', 'B8'),
    ('platform_id', 'A10'),
    ('mission_id', 'B32'),
    ('job_id', 'B32'),
]


_MISSION_FIELDS: List[Tuple[str, str]] = [
    ('mission_plan', 'A12'),
    ('flight_plan', 'A12'),
    ('platform_type', 'B8'),
    ('platform_configuration', 'A10'),
    ('reference_time_year', 'I16'),
    ('reference_time_month', 'B8'),
    ('reference_time_day', 'B8'),
]


_JOB_DEFINITION_FIELDS: List[Tuple[str, str]] = [
    ('job_id', 'B32'),
    ('sensor_id_type', 'B8'),
    ('sensor_id_model', 'A6'),
    ('target_filtering_flag', 'B8'),
    ('priority', 'B8'),
    ('bounding_a_lat', 'SA32'),
    ('bounding_a_lon', 'BA32'),
    ('bounding_b_lat', 'SA32'),
    ('bounding_b_lon', 'BA32'),
    ('bounding_c_lat', 'SA32'),
    ('bounding_c_lon', 'BA32'),
    ('bounding_d_lat', 'SA32'),
    ('bounding_d_lon', 'BA32'),
    ('radar_mode', 'B8'),
    ('nominal_revisit_interval', 'B16'),
    ('nominal_uncertainty_along_track', 'B16'),
    ('nominal_uncertainty_cross_track', 'B16'),
    ('nominal_uncertainty_altitude', 'B16'),
    ('nominal_uncertainty_track_heading', 'B8'),
    ('nominal_uncertainty_speed', 'B16'),
    ('nominal_sensor_slant_range_std', 'B16'),
    ('nominal_sensor_cross_range_std', 'F32'),
    ('nominal_sensor_los_velocity_std', 'B16'),
    ('nominal_sensor_mdv', 'B8'),
    ('nominal_sensor_detection_probability', 'B8'),
    ('nominal_sensor_false_alarm_density', 'B8'),
    ('terrain_elevation_model', 'B8'),
    ('geoid_model', 'B8'),
]


# Dwell-segment field table — required and optional fields concatenated.
# The codec writes all fields and sets the existence mask to all-1s; the
# reader consumes all fields. See module docstring on the v1 limitation.
_DWELL_FIELDS: List[Tuple[str, str]] = [
    ('existence_mask', 'B64'),
    ('revisit_index', 'I16'),
    ('dwell_index', 'I16'),
    ('last_dwell_of_revisit', 'B8'),
    ('target_report_count', 'I16'),
    ('dwell_time_ms', 'I32'),
    ('sensor_pos_lat', 'SA32'),
    ('sensor_pos_lon', 'BA32'),
    ('sensor_pos_alt', 'S32'),
    ('scale_factor_lat', 'SA32'),
    ('scale_factor_lon', 'BA32'),
    ('sensor_track', 'BA16'),
    ('sensor_speed', 'B32'),
    ('sensor_vertical_velocity', 'S8'),
    ('platform_orientation_heading', 'BA16'),
    ('platform_orientation_pitch', 'SA16'),
    ('platform_orientation_roll', 'SA16'),
    ('dwell_center_lat', 'SA32'),
    ('dwell_center_lon', 'BA32'),
    ('dwell_range_half_extent', 'B16'),
    ('dwell_angle_half_extent', 'BA16'),
    ('sensor_orientation_heading', 'BA16'),
    ('sensor_orientation_pitch', 'SA16'),
    ('sensor_orientation_roll', 'SA16'),
    ('mdv', 'B8'),
]


_TARGET_REPORT_FIELDS: List[Tuple[str, str]] = [
    ('report_index', 'I16'),
    ('target_lat', 'SA32'),
    ('target_lon', 'BA32'),
    ('target_height', 'S16'),
    ('target_velocity_los', 'S16'),
    ('target_wrap_velocity', 'B16'),
    ('target_snr', 'I8'),
    ('target_classification', 'B8'),
    ('target_class_probability', 'B8'),
    ('slant_range_std', 'B16'),
    ('cross_range_std', 'B16'),
    ('height_std', 'B8'),
    ('velocity_los_std', 'B16'),
    ('truth_tag_application', 'B8'),
    ('truth_tag_entity', 'B32'),
    ('target_rcs', 'I8'),
]


_FREE_TEXT_FIXED_FIELDS: List[Tuple[str, str]] = [
    ('originator', 'A10'),
    ('recipient', 'A10'),
    # text follows as variable-length ASCII to end of segment.
]


_PLATFORM_LOCATION_FIELDS: List[Tuple[str, str]] = [
    ('location_time_ms', 'I32'),
    ('platform_lat', 'SA32'),
    ('platform_lon', 'BA32'),
    ('platform_alt', 'S32'),
    ('platform_track', 'BA16'),
    ('platform_speed', 'B32'),
    ('platform_vertical_velocity', 'S8'),
]


# ---------------------------------------------------------------------------
# Generic field-table codec
# ---------------------------------------------------------------------------


def _serialize_table(obj: Any, fields: List[Tuple[str, str]]) -> bytes:
    """Serialize ``obj`` according to ``fields``, ``None`` → defaults."""
    parts: List[bytes] = []
    for name, type_code in fields:
        value = getattr(obj, name, None)
        parts.append(pack_field(value, type_code))
    return b''.join(parts)


def _parse_table(
    buf: bytes, offset: int, fields: List[Tuple[str, str]],
) -> Tuple[Dict[str, Any], int]:
    """Parse fields from ``buf`` at ``offset`` per the field table."""
    out: Dict[str, Any] = {}
    for name, type_code in fields:
        out[name], offset = unpack_field(buf, offset, type_code)
    return out, offset


# ---------------------------------------------------------------------------
# Packet header
# ---------------------------------------------------------------------------


def serialize_packet_header(header: PacketHeader) -> bytes:
    """Serialize a 32-byte packet header.

    Parameters
    ----------
    header : PacketHeader
        Header to serialize. ``header.packet_size`` must already be set
        to the total packet size in bytes (32 + sum of segments).

    Returns
    -------
    bytes
        Exactly 32 bytes.
    """
    raw = _serialize_table(header, _PACKET_HEADER_FIELDS)
    if len(raw) != PACKET_HEADER_SIZE:
        raise ValueError(
            f"Packet header serialization produced {len(raw)} bytes, "
            f"expected {PACKET_HEADER_SIZE}"
        )
    return raw


def parse_packet_header(buf: bytes, offset: int = 0) -> Tuple[PacketHeader, int]:
    """Parse a packet header from ``buf`` at ``offset``."""
    values, new_offset = _parse_table(buf, offset, _PACKET_HEADER_FIELDS)
    return PacketHeader(**values), new_offset


# ---------------------------------------------------------------------------
# Segment header
# ---------------------------------------------------------------------------


def serialize_segment_header(segment_type: int, segment_size: int) -> bytes:
    """Serialize the 5-byte segment header (type + size).

    Parameters
    ----------
    segment_type : int
        Segment type code (e.g., ``SEG_DWELL``).
    segment_size : int
        Total segment size in bytes including the header.
    """
    return struct.pack('>BI', segment_type, segment_size)


def parse_segment_header(buf: bytes, offset: int) -> Tuple[int, int, int]:
    """Parse the segment header.

    Returns
    -------
    tuple
        ``(segment_type, segment_size, new_offset)``. ``segment_size``
        includes the 5-byte header itself.
    """
    if offset + SEGMENT_HEADER_SIZE > len(buf):
        raise ValueError(
            f"Buffer underrun parsing segment header at offset {offset}"
        )
    seg_type, seg_size = struct.unpack(
        '>BI', buf[offset:offset + SEGMENT_HEADER_SIZE]
    )
    return seg_type, seg_size, offset + SEGMENT_HEADER_SIZE


# ---------------------------------------------------------------------------
# Segment-specific codecs
# ---------------------------------------------------------------------------


def serialize_mission(seg: MissionSegment) -> bytes:
    return _serialize_table(seg, _MISSION_FIELDS)


def parse_mission(buf: bytes, offset: int, end: int) -> MissionSegment:
    values, _ = _parse_table(buf, offset, _MISSION_FIELDS)
    return MissionSegment(**values)


def serialize_job_definition(seg: JobDefinitionSegment) -> bytes:
    return _serialize_table(seg, _JOB_DEFINITION_FIELDS)


def parse_job_definition(buf: bytes, offset: int, end: int) -> JobDefinitionSegment:
    values, _ = _parse_table(buf, offset, _JOB_DEFINITION_FIELDS)
    return JobDefinitionSegment(**values)


def serialize_target_report(target: TargetReport) -> bytes:
    return _serialize_table(target, _TARGET_REPORT_FIELDS)


def parse_target_report(buf: bytes, offset: int) -> Tuple[TargetReport, int]:
    values, new_offset = _parse_table(buf, offset, _TARGET_REPORT_FIELDS)
    return TargetReport(**values), new_offset


def serialize_dwell(seg: DwellSegment) -> bytes:
    """Serialize a dwell segment (header fields + target reports).

    The existence mask is forced to all-1s and the dwell-level
    ``target_report_count`` is set from ``len(seg.target_reports)`` to
    keep the wire format internally consistent regardless of how the
    user populated the dataclass.
    """
    seg.existence_mask = _DWELL_MASK_ALL
    seg.target_report_count = len(seg.target_reports)

    parts: List[bytes] = [_serialize_table(seg, _DWELL_FIELDS)]
    for target in seg.target_reports:
        parts.append(serialize_target_report(target))
    return b''.join(parts)


def parse_dwell(buf: bytes, offset: int, end: int) -> DwellSegment:
    """Parse a dwell segment including all embedded target reports."""
    values, offset = _parse_table(buf, offset, _DWELL_FIELDS)
    target_count = int(values.get('target_report_count', 0))
    target_reports: List[TargetReport] = []
    for _ in range(target_count):
        target, offset = parse_target_report(buf, offset)
        target_reports.append(target)
    values['target_reports'] = target_reports
    return DwellSegment(**values)


def serialize_free_text(seg: FreeTextSegment) -> bytes:
    """Serialize a free-text segment.

    Layout: 10 bytes originator + 10 bytes recipient + variable ASCII.
    """
    fixed = _serialize_table(seg, _FREE_TEXT_FIXED_FIELDS)
    text = (seg.text or '').encode('ascii', errors='replace')
    return fixed + text


def parse_free_text(buf: bytes, offset: int, end: int) -> FreeTextSegment:
    values, offset = _parse_table(buf, offset, _FREE_TEXT_FIXED_FIELDS)
    text_bytes = buf[offset:end]
    values['text'] = text_bytes.decode('ascii', errors='replace').rstrip()
    return FreeTextSegment(**values)


def serialize_platform_location(seg: PlatformLocationSegment) -> bytes:
    return _serialize_table(seg, _PLATFORM_LOCATION_FIELDS)


def parse_platform_location(
    buf: bytes, offset: int, end: int,
) -> PlatformLocationSegment:
    values, _ = _parse_table(buf, offset, _PLATFORM_LOCATION_FIELDS)
    return PlatformLocationSegment(**values)


# ---------------------------------------------------------------------------
# Segment dispatch
# ---------------------------------------------------------------------------


_SEGMENT_PARSERS = {
    SEG_MISSION: parse_mission,
    SEG_DWELL: parse_dwell,
    SEG_JOB_DEFINITION: parse_job_definition,
    SEG_FREE_TEXT: parse_free_text,
    SEG_PLATFORM_LOCATION: parse_platform_location,
}


_SEGMENT_TYPES = {
    MissionSegment: SEG_MISSION,
    DwellSegment: SEG_DWELL,
    JobDefinitionSegment: SEG_JOB_DEFINITION,
    FreeTextSegment: SEG_FREE_TEXT,
    PlatformLocationSegment: SEG_PLATFORM_LOCATION,
}


_SEGMENT_SERIALIZERS = {
    SEG_MISSION: serialize_mission,
    SEG_DWELL: serialize_dwell,
    SEG_JOB_DEFINITION: serialize_job_definition,
    SEG_FREE_TEXT: serialize_free_text,
    SEG_PLATFORM_LOCATION: serialize_platform_location,
}


def serialize_segment(seg: Any) -> bytes:
    """Serialize any supported segment, returning header + body bytes.

    Parameters
    ----------
    seg : Any
        A segment dataclass instance (Mission/Dwell/JobDefinition/...).

    Returns
    -------
    bytes
        ``segment_header (5 bytes) + body``. Total length matches the
        ``segment_size`` field in the header.
    """
    seg_type = _SEGMENT_TYPES.get(type(seg))
    if seg_type is None:
        raise ValueError(f"Unsupported segment type: {type(seg).__name__}")
    body = _SEGMENT_SERIALIZERS[seg_type](seg)
    total = SEGMENT_HEADER_SIZE + len(body)
    return serialize_segment_header(seg_type, total) + body


def parse_segment(buf: bytes, offset: int) -> Tuple[Any, int]:
    """Parse a single segment from ``buf`` starting at ``offset``.

    Returns
    -------
    tuple
        ``(segment, new_offset)``. ``segment`` is the typed dataclass
        for known types, or ``None`` for segments whose type is
        recognized but not yet supported (e.g., HRR).

    Raises
    ------
    ValueError
        If the segment header is truncated or the segment size is
        inconsistent with the remaining buffer.
    """
    seg_type, seg_size, body_offset = parse_segment_header(buf, offset)
    end = offset + seg_size
    if end > len(buf):
        raise ValueError(
            f"Segment of declared size {seg_size} at offset {offset} "
            f"extends past end of buffer ({len(buf)})"
        )

    parser = _SEGMENT_PARSERS.get(seg_type)
    if parser is None:
        # Unsupported segment type — skip its body but advance correctly.
        return None, end

    seg = parser(buf, body_offset, end)
    return seg, end
