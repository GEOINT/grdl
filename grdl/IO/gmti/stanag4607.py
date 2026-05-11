# -*- coding: utf-8 -*-
"""
STANAG 4607 reader - parse a GMTI mission file into typed segments.

``STANAG4607Reader`` reads an entire 4607 file into memory (typical
files are MB-scale, not GB), parses the packet stream into a
``STANAG4607Metadata`` container of typed segments, and offers
iteration and conversion helpers.

Unlike GRDL's raster ``ImageReader`` subclasses, ``STANAG4607Reader``
is a standalone class: STANAG 4607 has no pixel grid, so the
``read_chip`` / ``get_shape`` contract does not apply. The reader
follows the same conventions otherwise (top-level imports, fail-fast
on missing or malformed inputs, context-manager support).

The bridge to the rest of GRDL is ``to_detection_set()``: each target
report becomes a single ``Detection`` populated with the ``gmti.*``
field domain plus ``physical.velocity_radial``.

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
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

# Third-party (optional — only needed for to_detection_set)
try:
    from shapely.geometry import Point as _ShapelyPoint
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False

# GRDL internal
from grdl.exceptions import DependencyError, ValidationError
from grdl.IO.gmti import _segments as _seg
from grdl.IO.models.stanag4607 import (
    DwellSegment,
    Packet,
    PacketHeader,
    STANAG4607Metadata,
    TargetReport,
)


# Editions supported by this v1 implementation. Reading any version
# byte not in this set raises ``ValidationError``.
_SUPPORTED_EDITIONS = {2, 3, 4}


def _version_id_to_edition(version_id: str) -> int:
    """Map the 2-char ASCII version field to an integer edition.

    The standard encodes the edition as decimal ASCII: ``'20'`` for
    Ed 2, ``'30'`` for Ed 3, ``'40'`` for Ed 4. Higher digits map to
    the integer division by 10 (so ``'45'`` → 4 still parses).
    """
    cleaned = (version_id or '').strip()
    if not cleaned or not cleaned[0].isdigit():
        raise ValidationError(
            f"Unrecognized STANAG 4607 version field: {version_id!r}"
        )
    return int(cleaned[0])


def _normalize_longitude(deg: float) -> float:
    """Wrap a longitude into ``[-180, 180]``.

    BA32 longitudes decode into ``[0, 360)``; this returns the signed
    convention used elsewhere in GRDL for geographic geometries.
    """
    deg = float(deg) % 360.0
    if deg > 180.0:
        deg -= 360.0
    return deg


class STANAG4607Reader:
    """Reader for STANAG 4607 (NATO GMTI) files.

    Loads the full file at construction, parses every packet and
    segment into typed dataclasses, and exposes them through the
    ``metadata`` attribute. Use ``to_detection_set()`` to convert
    target reports into a GRDL ``DetectionSet`` for downstream
    filtering, transforms, and GeoJSON export.

    Parameters
    ----------
    filepath : str or Path
        Path to a STANAG 4607 file (commonly ``.4607`` or ``.gmti``).

    Attributes
    ----------
    filepath : Path
        Resolved file path.
    metadata : STANAG4607Metadata
        Typed metadata containing the parsed packet/segment stream.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValidationError
        If the file is shorter than a packet header, declares an
        unsupported edition, or contains an inconsistent packet size.

    Examples
    --------
    >>> with STANAG4607Reader('mission.4607') as r:
    ...     print(r.num_dwells(), r.num_target_reports())
    ...     ds = r.to_detection_set()
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        with open(self.filepath, 'rb') as fh:
            buf = fh.read()

        self.metadata: STANAG4607Metadata = self._parse(buf)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(buf: bytes) -> STANAG4607Metadata:
        """Parse the full packet stream from a byte buffer."""
        if len(buf) < _seg.PACKET_HEADER_SIZE:
            raise ValidationError(
                f"File is too small to contain a STANAG 4607 packet "
                f"header ({len(buf)} bytes < {_seg.PACKET_HEADER_SIZE})"
            )

        packets: List[Packet] = []
        edition: Optional[int] = None
        offset = 0
        total = len(buf)

        while offset < total:
            if total - offset < _seg.PACKET_HEADER_SIZE:
                raise ValidationError(
                    f"Trailing {total - offset} bytes are too small for "
                    f"a packet header at offset {offset}"
                )
            header, body_offset = _seg.parse_packet_header(buf, offset)
            packet_edition = _version_id_to_edition(header.version_id)
            if packet_edition not in _SUPPORTED_EDITIONS:
                raise ValidationError(
                    f"Unsupported STANAG 4607 edition: {packet_edition} "
                    f"(version field {header.version_id!r} at offset "
                    f"{offset})"
                )
            if edition is None:
                edition = packet_edition

            packet_end = offset + int(header.packet_size)
            if packet_end > total or header.packet_size < _seg.PACKET_HEADER_SIZE:
                raise ValidationError(
                    f"Packet at offset {offset} declares size "
                    f"{header.packet_size}; only {total - offset} bytes "
                    f"remain"
                )

            segments = []
            seg_offset = body_offset
            while seg_offset < packet_end:
                segment, seg_offset = _seg.parse_segment(buf, seg_offset)
                if segment is not None:
                    segments.append(segment)

            packets.append(Packet(header=header, segments=segments))
            offset = packet_end

        return STANAG4607Metadata(
            edition=edition if edition is not None else 4,
            packets=packets,
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def edition(self) -> int:
        """STANAG 4607 edition detected from the first packet."""
        return self.metadata.edition

    def num_packets(self) -> int:
        """Total number of packets in the file."""
        return self.metadata.num_packets

    def num_dwells(self) -> int:
        """Total number of dwell segments across all packets."""
        return self.metadata.num_dwells

    def num_target_reports(self) -> int:
        """Total number of target reports across all dwells."""
        return self.metadata.num_target_reports

    def iter_packets(self) -> Iterator[Packet]:
        """Yield every packet in file order."""
        return iter(self.metadata.packets)

    def iter_dwells(self) -> Iterator[DwellSegment]:
        """Yield every dwell segment in file order."""
        return iter(self.metadata.dwells)

    def iter_target_reports(
        self,
    ) -> Iterator[Tuple[DwellSegment, TargetReport]]:
        """Yield ``(dwell, target)`` for every target across the file."""
        for dwell in self.metadata.dwells:
            for target in dwell.target_reports:
                yield dwell, target

    # ------------------------------------------------------------------
    # Bridge to GRDL Detection ecosystem
    # ------------------------------------------------------------------

    def to_detection_set(
        self,
        confidence_field: str = 'gmti.snr_db',
        snr_normalization: float = 40.0,
    ) -> 'DetectionSet':  # noqa: F821 — forward import to avoid hard dep
        """Convert all target reports to a GRDL ``DetectionSet``.

        Each target report becomes one ``Detection`` with:

        - ``pixel_geometry = None`` (GMTI has no pixel coordinates).
        - ``geo_geometry = shapely.Point(lon, lat)`` in WGS84.
        - ``properties`` populated from the ``gmti.*`` field domain
          plus ``physical.velocity_radial`` (m/s).
        - ``confidence`` derived from the field named in
          ``confidence_field``, normalized to ``[0, 1]`` by dividing
          the SNR (in dB) by ``snr_normalization``.

        Parameters
        ----------
        confidence_field : str
            Field name in each detection's properties to use as the
            confidence source. Default is ``'gmti.snr_db'``.
        snr_normalization : float
            SNR (dB) value mapped to confidence 1.0. Default 40 dB
            puts typical strong target SNRs near full confidence.

        Returns
        -------
        DetectionSet
            One ``Detection`` per target report.

        Raises
        ------
        DependencyError
            If shapely is not installed (required for geometry).
        """
        if not _HAS_SHAPELY:
            raise DependencyError(
                "to_detection_set() requires shapely. "
                "Install with: pip install grdl[detection]"
            )

        # Local import to avoid a hard dependency in the IO layer.
        from grdl.image_processing.detection.models import (
            Detection, DetectionSet,
        )

        # Helpers that tolerate the field being None (mask bit clear).
        def _opt_int(v):  # type: ignore[no-untyped-def]
            return None if v is None else int(v)

        def _opt_float(v):  # type: ignore[no-untyped-def]
            return None if v is None else float(v)

        detections: List[Detection] = []
        for dwell, target in self.iter_target_reports():
            # Prefer hi-res lat/lon (D32.2/D32.3); fall back to delta form
            # (D32.4/D32.5 + D24/D25 + D10/D11) if those are absent.
            if target.target_lat is not None and target.target_lon is not None:
                lat = float(target.target_lat)
                lon = _normalize_longitude(target.target_lon)
            elif (target.target_delta_lat is not None
                  and target.target_delta_lon is not None
                  and dwell.dwell_center_lat is not None
                  and dwell.dwell_center_lon is not None
                  and dwell.scale_factor_lat is not None
                  and dwell.scale_factor_lon is not None):
                lat = (float(target.target_delta_lat)
                       * float(dwell.scale_factor_lat)
                       + float(dwell.dwell_center_lat))
                lon = _normalize_longitude(
                    float(target.target_delta_lon)
                    * float(dwell.scale_factor_lon)
                    + float(dwell.dwell_center_lon)
                )
            else:
                # No usable position — skip this report rather than guess.
                continue
            point = _ShapelyPoint(lon, lat)

            v_los = _opt_float(target.target_velocity_los)
            properties = {
                'gmti.report_index': _opt_int(target.report_index),
                'gmti.dwell_index': int(dwell.dwell_index),
                'gmti.snr_db': _opt_float(target.target_snr),
                'gmti.target_classification':
                    _opt_int(target.target_classification),
                'gmti.target_class_probability':
                    _opt_int(target.target_class_probability),
                'gmti.target_height_m': _opt_int(target.target_height),
                'gmti.wrap_velocity_cmps':
                    _opt_int(target.target_wrap_velocity),
                'gmti.slant_range_std_cm': _opt_int(target.slant_range_std),
                'gmti.cross_range_std': _opt_int(target.cross_range_std),
                'gmti.height_std_m': _opt_int(target.height_std),
                'gmti.velocity_los_std_cmps':
                    _opt_int(target.velocity_los_std),
                'gmti.target_rcs_db': _opt_int(target.target_rcs),
                'gmti.dwell_time_ms': int(dwell.dwell_time_ms),
                'gmti.platform_lat': float(dwell.sensor_pos_lat),
                'gmti.platform_lon': _normalize_longitude(dwell.sensor_pos_lon),
                'gmti.platform_alt_cm': int(dwell.sensor_pos_alt),
                # Reuse existing physical.* domain for velocity.
                'physical.velocity_radial':
                    None if v_los is None else v_los / 100.0,
            }
            if dwell.mdv is not None:
                properties['gmti.mdv_dmps'] = int(dwell.mdv)

            confidence: Optional[float] = None
            if confidence_field in properties and snr_normalization > 0:
                value = properties[confidence_field]
                if value is not None:
                    confidence = max(
                        0.0, min(1.0, float(value) / snr_normalization),
                    )

            detections.append(Detection(
                pixel_geometry=None,
                properties=properties,
                confidence=confidence,
                geo_geometry=point,
            ))

        return DetectionSet(
            detections=detections,
            detector_name='STANAG4607Reader',
            detector_version='1.0.0',
            output_fields=(
                'gmti.report_index',
                'gmti.dwell_index',
                'gmti.snr_db',
                'gmti.target_classification',
                'physical.velocity_radial',
            ),
            metadata={
                'source_file': str(self.filepath),
                'edition': self.edition,
                'num_packets': self.num_packets(),
                'num_dwells': self.num_dwells(),
            },
        )

    # ------------------------------------------------------------------
    # Resource lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """No-op — the reader holds no open file handles after construction."""
        return None

    def __enter__(self) -> 'STANAG4607Reader':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def open_gmti(filepath: Union[str, Path]) -> STANAG4607Reader:
    """Open a GMTI file and return a ``STANAG4607Reader``.

    Mirrors ``open_sar``, ``open_eo``, etc. for symmetry; today only
    STANAG 4607 is supported, but additional GMTI formats can dispatch
    here in the future.

    Parameters
    ----------
    filepath : str or Path
        Path to the GMTI file.

    Returns
    -------
    STANAG4607Reader

    Examples
    --------
    >>> from grdl.IO.gmti import open_gmti
    >>> with open_gmti('mission.4607') as r:
    ...     ds = r.to_detection_set()
    """
    return STANAG4607Reader(filepath)
