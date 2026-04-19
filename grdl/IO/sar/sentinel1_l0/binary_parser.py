# -*- coding: utf-8 -*-
"""
Sentinel-1 L0 Binary File Parsers.

Parsers for Sentinel-1 Level 0 binary auxiliary files:

- Burst index files (``*-index.dat``)
- Packet annotation files (``*-annot.dat``)

These files provide fast burst seeking and per-packet metadata
without requiring parsing of the raw ISP packets.

File Locations
--------------
::

    measurement/
    ├── s1a-iw-raw-vv-....dat         # Raw ISP packets
    ├── s1a-iw-raw-vv-...-index.dat   # Burst index (this module)
    └── s1a-iw-raw-vv-...-annot.dat   # Packet annotations (this module)

References
----------
- Sentinel-1 Level-0 Product Format Specifications (S1PD.SP.00110)

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
2026-04-16

Modified
--------
2026-04-16
"""

# Standard library
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Burst Index Records
# =============================================================================


@dataclass(frozen=True)
class BurstIndexRecord:
    """Record from a Sentinel-1 Level 0 burst index file.

    Ref: [L0FMT] S1PD.SP.00110.ATSR, Table 3-9.

    File Structure (36 bytes per record, big-endian)
    ------------------------------------------------
    ::

        Bytes  0- 7: Reference time (float64, MJD 1950 days)
        Bytes  8-15: Delta time between ISPs (float64, ms)
        Bytes 16-19: ISP size (uint32, bytes)
        Bytes 20-23: Data units offset (uint32, ISP count)
        Bytes 24-31: Byte offset (uint64, from data start)
        Bytes 32-35: Flags (uint32): bit 0 = variable_size

    Parameters
    ----------
    burst_index : int
        Sequential index in index file.
    reference_time : float
        MJD 1950 at first ISP in block.
    duration : float
        Delta time between ISPs (milliseconds).
    isp_size : int, optional
        ISP size in bytes (all packets if not ``variable_size_flag``,
        else size of the first packet).
    start_packet : int, optional
        Data units offset from data start.
    byte_offset : int, optional
        Byte offset from data object start.
    variable_size_flag : bool, optional
        ``True`` if ISP sizes vary in block.
    swath_number : int, optional
        Swath number (from reserved flag bits, 0 if not encoded).
    """

    burst_index: int
    reference_time: float
    duration: float
    isp_size: int = 0
    start_packet: int = 0
    byte_offset: int = 0
    variable_size_flag: bool = False
    swath_number: int = 0

    @property
    def end_time(self) -> float:
        """End time (``reference_time + duration`` in days)."""
        return self.reference_time + self.duration


@dataclass(frozen=True)
class PacketAnnotationRecord:
    """Record from a Sentinel-1 Level 0 packet annotation file.

    Ref: [L0FMT] S1PD.SP.00110.ATSR, Table 3-8.

    File Structure (26 bytes per record, big-endian)
    ------------------------------------------------
    ::

        Bytes  0- 1: Sensing time days (uint16, MJD2000)
        Bytes  2- 5: Sensing time milliseconds (uint32)
        Bytes  6- 7: Sensing time microseconds (uint16)
        Bytes  8- 9: Downlink time days (uint16, MJD2000)
        Bytes 10-13: Downlink time milliseconds (uint32)
        Bytes 14-15: Downlink time microseconds (uint16)
        Bytes 16-17: ISP length (uint16, bytes from header)
        Bytes 18-19: Number of frames (uint16)
        Bytes 20-21: Number of missing frames (uint16)
        Byte  22:    CRC error flag (uint8, 0=ok 1=error)
        Byte  23:    Flags: bit 7=VCID present,
                     bits 1-6=VCID, bits 0 + byte 24 upper bits =
                     channel
        Byte  24:    Channel (2 bits: 01=C1, 10=X2)
        Byte  25:    Spare

    Parameters
    ----------
    record_index : int
        Sequential index in annotation file.
    coarse_time : int
        Sensing time days since MJD2000.
    fine_time : int
        Sensing time microseconds in ms.
    time_offset_ms : float
        Combined sensing time (ms in day + µs / 1000).
    apid : int
        Frames count (legacy name; not APID).
    packet_index : int
        Sequential packet count.
    packet_length : int
        ISP length from header (bytes).
    flags : int
        Raw flags byte.
    sensing_time_days, sensing_time_ms, sensing_time_us : int, optional
        Sensing time components (days, ms-in-day, µs).
    downlink_time_days, downlink_time_ms, downlink_time_us : int,
        optional
        Downlink time components.
    crc_flag : bool, optional
        CRC error detected.
    vcid_present : bool, optional
        VCID field is valid.
    vcid : int, optional
        Virtual Channel ID (6 bits).
    channel : int, optional
        Downlink channel (2 bits, 1 = C1, 2 = X2).
    missing_frames : int, optional
        Number of missing transfer frames.
    """

    record_index: int
    coarse_time: int
    fine_time: int
    time_offset_ms: float
    apid: int
    packet_index: int
    packet_length: int
    flags: int
    # --- Full Table 3-8 fields ---
    sensing_time_days: int = 0
    sensing_time_ms: int = 0
    sensing_time_us: int = 0
    downlink_time_days: int = 0
    downlink_time_ms: int = 0
    downlink_time_us: int = 0
    crc_flag: bool = False
    vcid_present: bool = False
    vcid: int = 0
    channel: int = 0
    missing_frames: int = 0


# =============================================================================
# Index File Parser
# =============================================================================

# Index record size in bytes.
INDEX_RECORD_SIZE = 36


def parse_burst_index_file(index_path: Path) -> List[BurstIndexRecord]:
    """Parse a Sentinel-1 Level 0 burst index file.

    The index file is a binary file with 36-byte records containing
    burst metadata and byte offsets for fast seeking.

    Parameters
    ----------
    index_path : Path
        Path to the ``*-index.dat`` file.

    Returns
    -------
    list of BurstIndexRecord
        Parsed burst index records.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file structure is invalid.
    """
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index file not found: {index_path}"
        )

    records: List[BurstIndexRecord] = []

    with open(index_path, "rb") as f:
        data = f.read()

    if len(data) < INDEX_RECORD_SIZE:
        logger.warning(f"Index file too small: {len(data)} bytes")
        return records

    num_records = len(data) // INDEX_RECORD_SIZE

    for i in range(num_records):
        offset = i * INDEX_RECORD_SIZE
        rec = data[offset:offset + INDEX_RECORD_SIZE]

        # Parse fields (big-endian).  Ref: [L0FMT] Table 3-9.
        ref_time = struct.unpack(">d", rec[0:8])[0]
        delta_time = struct.unpack(">d", rec[8:16])[0]
        isp_size = struct.unpack(">I", rec[16:20])[0]
        data_units_offset = struct.unpack(">I", rec[20:24])[0]
        byte_off = struct.unpack(">Q", rec[24:32])[0]
        flags_word = struct.unpack(">I", rec[32:36])[0]

        # Bit 0: variable size flag.
        var_size = bool(flags_word & 0x01)

        records.append(BurstIndexRecord(
            burst_index=i,
            reference_time=ref_time,
            duration=delta_time,
            isp_size=isp_size,
            start_packet=data_units_offset,
            byte_offset=byte_off,
            variable_size_flag=var_size,
            swath_number=(flags_word >> 1) & 0x7F,
        ))

    logger.debug(
        f"Parsed {len(records)} burst index records "
        f"from {index_path.name}"
    )
    return records


# =============================================================================
# Packet Annotation File Parser
# =============================================================================

# Annotation record size in bytes.
# Ref: [L0FMT] S1PD.SP.00110.ATSR, Table 3-8.
ANNOT_RECORD_SIZE = 26


def parse_packet_annotation_file(
    annot_path: Path,
    validate: bool = True,
) -> List[PacketAnnotationRecord]:
    """Parse a Sentinel-1 Level 0 packet annotation file.

    The annotation file (``*-annot.dat``) contains 26-byte records
    with per-packet metadata including timing, APID, and packet
    length.

    Parameters
    ----------
    annot_path : Path
        Path to the ``*-annot.dat`` file.
    validate : bool, optional
        Whether to validate marker values.  Default ``True``.

    Returns
    -------
    list of PacketAnnotationRecord
        Parsed annotation records.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file structure is invalid.
    """
    annot_path = Path(annot_path)
    if not annot_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {annot_path}"
        )

    records: List[PacketAnnotationRecord] = []

    with open(annot_path, "rb") as f:
        data = f.read()

    if len(data) % ANNOT_RECORD_SIZE != 0:
        raise ValueError(
            f"Annotation file size ({len(data)}) is not a "
            f"multiple of record size ({ANNOT_RECORD_SIZE})"
        )

    num_records = len(data) // ANNOT_RECORD_SIZE

    for i in range(num_records):
        offset = i * ANNOT_RECORD_SIZE
        rec = data[offset:offset + ANNOT_RECORD_SIZE]

        # Parse per Table 3-8 (big-endian).
        sens_days = struct.unpack(">H", rec[0:2])[0]
        sens_ms = struct.unpack(">I", rec[2:6])[0]
        sens_us = struct.unpack(">H", rec[6:8])[0]
        dl_days = struct.unpack(">H", rec[8:10])[0]
        dl_ms = struct.unpack(">I", rec[10:14])[0]
        dl_us = struct.unpack(">H", rec[14:16])[0]
        pkt_len = struct.unpack(">H", rec[16:18])[0]
        frames = struct.unpack(">H", rec[18:20])[0]
        missing = struct.unpack(">H", rec[20:22])[0]
        crc_byte = rec[22]
        flag_byte = rec[23]
        chan_byte = rec[24]
        # rec[25] = spare

        # Validate reserved field.
        if validate and i == 0 and pkt_len == 0:
            logger.warning(
                "First annotation record has zero packet "
                "length — file may be malformed"
            )

        # Decode flag byte.
        vcid_present = bool(flag_byte & 0x80)
        vcid_val = (flag_byte >> 1) & 0x3F
        channel_val = (
            ((flag_byte & 0x01) << 1)
            | ((chan_byte >> 7) & 0x01)
        )

        # Combined sensing time (ms-in-day + µs/1000).
        time_offset_ms = sens_ms + sens_us / 1000.0

        records.append(PacketAnnotationRecord(
            record_index=i,
            coarse_time=sens_days,
            fine_time=sens_us,
            time_offset_ms=time_offset_ms,
            apid=frames,
            packet_index=i,
            packet_length=pkt_len,
            flags=flag_byte,
            sensing_time_days=sens_days,
            sensing_time_ms=sens_ms,
            sensing_time_us=sens_us,
            downlink_time_days=dl_days,
            downlink_time_ms=dl_ms,
            downlink_time_us=dl_us,
            crc_flag=bool(crc_byte),
            vcid_present=vcid_present,
            vcid=vcid_val,
            channel=channel_val,
            missing_frames=missing,
        ))

    logger.debug(
        f"Parsed {len(records)} packet annotation records "
        f"from {annot_path.name}"
    )
    return records


# =============================================================================
# Convenience Functions
# =============================================================================


def get_burst_byte_offsets(index_path: Path) -> List[int]:
    """Get byte offsets for all bursts.

    Parameters
    ----------
    index_path : Path
        Path to the ``*-index.dat`` file.

    Returns
    -------
    list of int
        Byte offsets, one per burst record.
    """
    records = parse_burst_index_file(index_path)
    return [r.byte_offset for r in records]


def get_packet_times(annot_path: Path) -> List[float]:
    """Get time offsets for all packets.

    Parameters
    ----------
    annot_path : Path
        Path to the ``*-annot.dat`` file.

    Returns
    -------
    list of float
        Time offsets in milliseconds, one per packet.
    """
    records = parse_packet_annotation_file(annot_path)
    return [r.time_offset_ms for r in records]


def find_burst_for_time(
    records: List[BurstIndexRecord],
    gps_time: float,
) -> Optional[BurstIndexRecord]:
    """Find the burst containing a specific GPS time.

    Parameters
    ----------
    records : list of BurstIndexRecord
        Parsed burst index records.
    gps_time : float
        GPS time in seconds.

    Returns
    -------
    BurstIndexRecord or None
        The burst containing the time, or ``None`` if not found.
    """
    for record in records:
        if record.reference_time <= gps_time <= record.end_time:
            return record
    return None
