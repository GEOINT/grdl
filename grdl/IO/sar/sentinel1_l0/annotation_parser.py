# -*- coding: utf-8 -*-
"""
Sentinel-1 L0 Annotation XML Parser.

Parses Sentinel-1 Level 0 annotation XML files to extract orbit,
attitude, radar parameters, and per-burst timing information.

Annotation File Structure
-------------------------
::

    <product>
        <generalAnnotation>
            <productInformation>      - Radar parameters
            <orbitList>               - Orbit state vectors
            <attitudeList>            - Attitude records
            <downlinkInformation>     - Downlink parameters
        </generalAnnotation>
        <swathTiming>                  - Swath and burst timing
        <geolocationGrid>              - Geolocation grid points
    </product>

References
----------
- Sentinel-1 Level-0 Product Format Specifications (S1PD.SP.00110)

Author
------
Duane Smalley, PhD
duane.d.smalley@gmail.com

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
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.models.sentinel1_l0 import (
    S1L0AttitudeRecord,
    S1L0BurstRecord,
    S1L0DownlinkInfo,
    S1L0GeolocationGrid,
    S1L0OrbitStateVector,
    S1L0RadarParameters,
    S1L0SwathParameters,
)
from grdl.IO.sar.sentinel1_l0.constants import (
    IW_MODE_PARAMS,
    SENTINEL1_CENTER_FREQUENCY_HZ,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Datetime parsing
# =============================================================================


def parse_annotation_datetime(time_str: str) -> datetime:
    """Parse a datetime string from an annotation XML element.

    Handles:

    - ``2023-05-01T06:00:01.123456Z``
    - ``2023-05-01T06:00:01.123456``
    - ``2023-05-01T06:00:01``

    Parameters
    ----------
    time_str : str
        Datetime string from XML.

    Returns
    -------
    datetime
        Parsed datetime.

    Raises
    ------
    ValueError
        If the format is not recognized.
    """
    if not time_str:
        raise ValueError("Empty datetime string")

    time_str = time_str.strip()
    if time_str.endswith("Z"):
        time_str = time_str[:-1]

    formats = [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unrecognized datetime format: {time_str}")


# =============================================================================
# Parse results container
# =============================================================================


@dataclass
class AnnotationData:
    """Container for data extracted from one annotation file.

    Parameters
    ----------
    source_file : Path
        Path to the parsed annotation file.
    polarization : str, optional
        Polarization derived from filename.
    orbit_state_vectors : list of S1L0OrbitStateVector, optional
        Parsed orbit state vectors.
    attitude_records : list of S1L0AttitudeRecord, optional
        Parsed attitude records.
    radar_parameters : S1L0RadarParameters, optional
        Radar signal parameters.
    swath_parameters : list of S1L0SwathParameters, optional
        Per-swath timing parameters.
    geolocation_grid : S1L0GeolocationGrid, optional
        Geolocation grid points.
    downlink_info : list of S1L0DownlinkInfo, optional
        Parsed downlink configurations.
    """

    source_file: Path
    polarization: str = ""
    orbit_state_vectors: List[S1L0OrbitStateVector] = field(
        default_factory=list
    )
    attitude_records: List[S1L0AttitudeRecord] = field(
        default_factory=list
    )
    radar_parameters: Optional[S1L0RadarParameters] = None
    swath_parameters: List[S1L0SwathParameters] = field(
        default_factory=list
    )
    geolocation_grid: Optional[S1L0GeolocationGrid] = None
    downlink_info: List[S1L0DownlinkInfo] = field(
        default_factory=list
    )


# =============================================================================
# Annotation parser
# =============================================================================


class AnnotationParser:
    """Parser for Sentinel-1 annotation XML files.

    Extracts orbit, attitude, radar parameters, and timing
    information from annotation XML files.  Stateless between
    ``parse()`` calls.

    Examples
    --------
    >>> parser = AnnotationParser()
    >>> data = parser.parse('/path/to/s1a-iw-raw-vv-...xml')
    >>> print(f'Found {len(data.orbit_state_vectors)} orbit vectors')
    """

    def __init__(self) -> None:
        self._tree: Optional[ET.ElementTree] = None
        self._root: Optional[ET.Element] = None

    def parse(self, annotation_path: Path) -> AnnotationData:
        """Parse an annotation XML file.

        Parameters
        ----------
        annotation_path : Path
            Path to annotation XML file.

        Returns
        -------
        AnnotationData
            Extracted annotation data.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        xml.etree.ElementTree.ParseError
            If the XML is malformed.
        """
        annotation_path = Path(annotation_path)
        if not annotation_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {annotation_path}"
            )

        self._tree = ET.parse(annotation_path)
        self._root = self._tree.getroot()

        polarization = self._extract_polarization(
            annotation_path.name
        )

        data = AnnotationData(
            source_file=annotation_path,
            polarization=polarization,
        )

        data.orbit_state_vectors = self._parse_orbit_list()
        data.attitude_records = self._parse_attitude_list()
        data.radar_parameters = self._parse_product_information()
        data.swath_parameters = self._parse_swath_timing(
            polarization
        )
        data.geolocation_grid = self._parse_geolocation_grid()
        data.downlink_info = self._parse_downlink_information()

        if data.radar_parameters and data.downlink_info:
            self._merge_downlink_to_radar(
                data.radar_parameters, data.downlink_info
            )

        logger.debug(f"Parsed annotation: {annotation_path.name}")
        return data

    @staticmethod
    def _extract_polarization(filename: str) -> str:
        """Extract polarization from annotation filename."""
        filename_lower = filename.lower()
        for pol in ["vv", "vh", "hh", "hv"]:
            if pol in filename_lower:
                return pol.upper()
        return "UNKNOWN"

    def _parse_orbit_list(self) -> List[S1L0OrbitStateVector]:
        """Parse orbit state vectors from ``<orbitList>``."""
        vectors: List[S1L0OrbitStateVector] = []
        orbit_list = self._root.find(".//orbitList")
        if orbit_list is None:
            return vectors

        for orbit in orbit_list.findall("orbit"):
            try:
                time_elem = orbit.find("time")
                if time_elem is None or not time_elem.text:
                    continue
                time = parse_annotation_datetime(time_elem.text)

                pos = orbit.find("position")
                if pos is None:
                    continue
                x = float(pos.find("x").text)
                y = float(pos.find("y").text)
                z = float(pos.find("z").text)

                vel = orbit.find("velocity")
                if vel is None:
                    continue
                vx = float(vel.find("x").text)
                vy = float(vel.find("y").text)
                vz = float(vel.find("z").text)

                vectors.append(S1L0OrbitStateVector(
                    time=time,
                    x=x, y=y, z=z,
                    vx=vx, vy=vy, vz=vz,
                ))
            except (ValueError, AttributeError) as e:
                warnings.warn(
                    f"Failed to parse orbit state vector: {e}"
                )

        vectors.sort(key=lambda v: v.time)
        return vectors

    def _parse_attitude_list(self) -> List[S1L0AttitudeRecord]:
        """Parse attitude records from ``<attitudeList>``."""
        records: List[S1L0AttitudeRecord] = []
        attitude_list = self._root.find(".//attitudeList")
        if attitude_list is None:
            return records

        for attitude in attitude_list.findall("attitude"):
            try:
                time_elem = attitude.find("time")
                if time_elem is None or not time_elem.text:
                    continue
                time = parse_annotation_datetime(time_elem.text)

                roll_elem = attitude.find("roll")
                pitch_elem = attitude.find("pitch")
                yaw_elem = attitude.find("yaw")
                roll = (
                    float(roll_elem.text)
                    if roll_elem is not None and roll_elem.text
                    else 0.0
                )
                pitch = (
                    float(pitch_elem.text)
                    if pitch_elem is not None and pitch_elem.text
                    else 0.0
                )
                yaw = (
                    float(yaw_elem.text)
                    if yaw_elem is not None and yaw_elem.text
                    else 0.0
                )

                q0 = q1 = q2 = q3 = None
                q0_elem = attitude.find("q0")
                if q0_elem is not None and q0_elem.text:
                    q0 = float(q0_elem.text)
                    q1 = float(attitude.find("q1").text)
                    q2 = float(attitude.find("q2").text)
                    q3 = float(attitude.find("q3").text)

                records.append(S1L0AttitudeRecord(
                    time=time,
                    roll=roll, pitch=pitch, yaw=yaw,
                    q0=q0, q1=q1, q2=q2, q3=q3,
                ))
            except (ValueError, AttributeError) as e:
                warnings.warn(
                    f"Failed to parse attitude record: {e}"
                )

        records.sort(key=lambda r: r.time)
        return records

    def _parse_product_information(
        self,
    ) -> Optional[S1L0RadarParameters]:
        """Parse radar parameters from ``<productInformation>``."""
        prod_info = self._root.find(".//productInformation")
        if prod_info is None:
            prod_info = self._root.find(
                ".//generalAnnotation/productInformation"
            )

        if prod_info is None:
            return S1L0RadarParameters(
                center_frequency_hz=SENTINEL1_CENTER_FREQUENCY_HZ,
                range_sampling_rate_hz=IW_MODE_PARAMS[
                    "range_sampling_rate_hz"
                ],
                pulse_repetition_frequency_hz=0.0,
            )

        try:
            freq_elem = prod_info.find("radarFrequency")
            center_freq = (
                float(freq_elem.text)
                if freq_elem is not None and freq_elem.text
                else SENTINEL1_CENTER_FREQUENCY_HZ
            )

            rate_elem = prod_info.find("rangeSamplingRate")
            sample_rate = (
                float(rate_elem.text)
                if rate_elem is not None and rate_elem.text
                else IW_MODE_PARAMS["range_sampling_rate_hz"]
            )

            steer_elem = prod_info.find("azimuthSteeringRate")
            steering_rate = (
                float(steer_elem.text)
                if steer_elem is not None and steer_elem.text
                else 0.0
            )

            pol_elem = prod_info.find("polarisation")
            polarization = (
                pol_elem.text if pol_elem is not None else ""
            )

            return S1L0RadarParameters(
                center_frequency_hz=center_freq,
                range_sampling_rate_hz=sample_rate,
                pulse_repetition_frequency_hz=0.0,
                azimuth_steering_rate_deg_per_s=steering_rate,
                polarization=polarization or "",
            )
        except (ValueError, AttributeError) as e:
            warnings.warn(
                f"Failed to parse product information: {e}"
            )
            return S1L0RadarParameters(
                center_frequency_hz=SENTINEL1_CENTER_FREQUENCY_HZ,
                range_sampling_rate_hz=IW_MODE_PARAMS[
                    "range_sampling_rate_hz"
                ],
                pulse_repetition_frequency_hz=0.0,
            )

    def _parse_downlink_information(
        self,
    ) -> List[S1L0DownlinkInfo]:
        """Parse downlink parameters from ``<downlinkInformation>``."""
        info_list: List[S1L0DownlinkInfo] = []
        downlink_info = self._root.find(".//downlinkInformation")
        if downlink_info is None:
            return info_list

        for downlink_values in downlink_info.findall(
            ".//downlinkValues"
        ):
            try:
                swath_elem = downlink_values.find("swath")
                swath = (
                    swath_elem.text
                    if swath_elem is not None and swath_elem.text
                    else ""
                )

                prf_elem = downlink_values.find("prf")
                prf = (
                    float(prf_elem.text)
                    if prf_elem is not None and prf_elem.text
                    else 0.0
                )
                pri = 1.0 / prf if prf > 0 else 0.0

                pulse_len_elem = downlink_values.find(
                    "txPulseLength"
                )
                pulse_length = (
                    float(pulse_len_elem.text)
                    if pulse_len_elem is not None
                    and pulse_len_elem.text
                    else 0.0
                )

                ramp_elem = downlink_values.find("txPulseRampRate")
                ramp_rate = (
                    float(ramp_elem.text)
                    if ramp_elem is not None and ramp_elem.text
                    else 0.0
                )

                rank_elem = downlink_values.find("rank")
                rank = (
                    int(rank_elem.text)
                    if rank_elem is not None and rank_elem.text
                    else 0
                )

                swst_elem = downlink_values.find("swst")
                swst = (
                    float(swst_elem.text)
                    if swst_elem is not None and swst_elem.text
                    else 0.0
                )

                swl_elem = downlink_values.find("swl")
                swl = (
                    int(float(swl_elem.text))
                    if swl_elem is not None and swl_elem.text
                    else 0
                )

                info_list.append(S1L0DownlinkInfo(
                    prf=prf,
                    pri=pri,
                    rank=rank,
                    swst=swst,
                    swl=swl,
                    swath=swath,
                    tx_pulse_ramp_rate=ramp_rate,
                    tx_pulse_length=pulse_length,
                ))
            except (ValueError, AttributeError) as e:
                warnings.warn(
                    f"Failed to parse downlink values: {e}"
                )

        return info_list

    def _parse_swath_timing(
        self, polarization: str,
    ) -> List[S1L0SwathParameters]:
        """Parse swath timing and burst list from ``<swathTiming>``."""
        swath_list: List[S1L0SwathParameters] = []
        swath_timing = self._root.find(".//swathTiming")
        if swath_timing is None:
            return swath_list

        swath_elem = swath_timing.find("swath")
        swath_id = (
            swath_elem.text
            if swath_elem is not None and swath_elem.text
            else "UNKNOWN"
        )

        swath = S1L0SwathParameters(
            swath_id=swath_id,
            polarization=polarization,
        )

        lines_elem = swath_timing.find("linesPerBurst")
        if lines_elem is not None and lines_elem.text:
            swath.lines_per_burst = int(lines_elem.text)

        samples_elem = swath_timing.find("samplesPerBurst")
        if samples_elem is not None and samples_elem.text:
            swath.samples_per_burst = int(samples_elem.text)

        az_interval_elem = swath_timing.find("azimuthTimeInterval")
        if az_interval_elem is not None and az_interval_elem.text:
            swath.azimuth_time_interval = float(
                az_interval_elem.text
            )

        burst_list = swath_timing.find("burstList")
        if burst_list is not None:
            count_attr = burst_list.get("count")
            if count_attr:
                swath.num_bursts = int(count_attr)

            for idx, burst in enumerate(
                burst_list.findall("burst")
            ):
                burst_rec = self._parse_burst_element(burst, idx)
                if burst_rec:
                    swath.bursts.append(burst_rec)

        swath_list.append(swath)
        return swath_list

    def _parse_burst_element(
        self, burst: ET.Element, index: int,
    ) -> Optional[S1L0BurstRecord]:
        """Parse a single ``<burst>`` element."""
        try:
            az_time_elem = burst.find("azimuthTime")
            az_time = (
                parse_annotation_datetime(az_time_elem.text)
                if az_time_elem is not None and az_time_elem.text
                else None
            )
            if az_time is None:
                return None

            sense_time_elem = burst.find("sensingTime")
            sense_time = (
                parse_annotation_datetime(sense_time_elem.text)
                if sense_time_elem is not None
                and sense_time_elem.text
                else None
            )

            offset_elem = burst.find("byteOffset")
            byte_offset = (
                int(offset_elem.text)
                if offset_elem is not None and offset_elem.text
                else 0
            )

            anx_elem = burst.find("azimuthAnxTime")
            anx_time = (
                float(anx_elem.text)
                if anx_elem is not None and anx_elem.text
                else 0.0
            )

            first_valid_elem = burst.find("firstValidSample")
            first_valid = 0
            if (
                first_valid_elem is not None
                and first_valid_elem.text
            ):
                values = first_valid_elem.text.split()
                if values:
                    first_valid = int(values[0])

            last_valid_elem = burst.find("lastValidSample")
            last_valid = 0
            if (
                last_valid_elem is not None
                and last_valid_elem.text
            ):
                values = last_valid_elem.text.split()
                if values:
                    last_valid = int(values[-1])

            return S1L0BurstRecord(
                burst_index=index,
                azimuth_time=az_time,
                sensing_time=sense_time,
                byte_offset=byte_offset,
                azimuth_anx_time=anx_time,
                first_valid_sample=first_valid,
                last_valid_sample=last_valid,
            )
        except (ValueError, AttributeError) as e:
            warnings.warn(f"Failed to parse burst {index}: {e}")
            return None

    def _parse_geolocation_grid(
        self,
    ) -> Optional[S1L0GeolocationGrid]:
        """Parse the geolocation grid from ``<geolocationGrid>``."""
        grid_elem = self._root.find(".//geolocationGrid")
        if grid_elem is None:
            return None

        grid_list = grid_elem.find("geolocationGridPointList")
        if grid_list is None:
            return None

        grid = S1L0GeolocationGrid()
        azimuth_times: List[datetime] = []
        slant_range_times: List[float] = []
        lats: List[float] = []
        lons: List[float] = []
        heights: List[float] = []

        for point in grid_list.findall("geolocationGridPoint"):
            try:
                az_elem = point.find("azimuthTime")
                if az_elem is not None and az_elem.text:
                    azimuth_times.append(
                        parse_annotation_datetime(az_elem.text)
                    )

                sr_elem = point.find("slantRangeTime")
                if sr_elem is not None and sr_elem.text:
                    slant_range_times.append(float(sr_elem.text))

                lat_elem = point.find("latitude")
                if lat_elem is not None and lat_elem.text:
                    lats.append(float(lat_elem.text))

                lon_elem = point.find("longitude")
                if lon_elem is not None and lon_elem.text:
                    lons.append(float(lon_elem.text))

                height_elem = point.find("height")
                if height_elem is not None and height_elem.text:
                    heights.append(float(height_elem.text))
            except (ValueError, AttributeError):
                continue

        if azimuth_times:
            grid.azimuth_times = azimuth_times
        if slant_range_times:
            grid.slant_range_times = slant_range_times
        if lats:
            grid.latitudes = np.array(lats)
        if lons:
            grid.longitudes = np.array(lons)
        if heights:
            grid.heights = np.array(heights)

        return grid

    @staticmethod
    def _merge_downlink_to_radar(
        radar: S1L0RadarParameters,
        downlink_list: List[S1L0DownlinkInfo],
    ) -> None:
        """Merge downlink info into radar parameters.

        Uses the first available downlink info to fill missing
        radar params.
        """
        if not downlink_list:
            return
        dl = downlink_list[0]
        if dl.prf and not radar.pulse_repetition_frequency_hz:
            radar.pulse_repetition_frequency_hz = dl.prf


# =============================================================================
# Convenience functions
# =============================================================================


def parse_annotation_file(
    annotation_path: Path,
) -> AnnotationData:
    """Parse an annotation XML file (convenience wrapper).

    Parameters
    ----------
    annotation_path : Path
        Path to annotation XML file.

    Returns
    -------
    AnnotationData
        Extracted annotation data.
    """
    parser = AnnotationParser()
    return parser.parse(annotation_path)


def parse_all_annotations(
    annotation_files: List[Path],
) -> Dict[str, AnnotationData]:
    """Parse multiple annotation files.

    Parameters
    ----------
    annotation_files : list of Path
        Paths to annotation XML files.

    Returns
    -------
    dict
        Mapping polarization → ``AnnotationData``.
    """
    parser = AnnotationParser()
    results: Dict[str, AnnotationData] = {}
    for path in annotation_files:
        try:
            data = parser.parse(path)
            results[data.polarization] = data
        except Exception as e:
            warnings.warn(f"Failed to parse {path.name}: {e}")
    return results


def merge_annotation_data(
    annotations: Dict[str, AnnotationData],
) -> AnnotationData:
    """Merge multiple annotation data objects into one container.

    Orbit and attitude data are shared across polarizations and are
    taken from the first file.  Swath parameters and downlink info
    are merged across all files.

    Parameters
    ----------
    annotations : dict
        Mapping polarization → ``AnnotationData``.

    Returns
    -------
    AnnotationData
        Merged annotation data.
    """
    if not annotations:
        return AnnotationData(source_file=Path("."))

    first_key = next(iter(annotations))
    first = annotations[first_key]

    all_downlinks: List[S1L0DownlinkInfo] = []
    seen_swaths: set = set()
    for data in annotations.values():
        for dl in data.downlink_info:
            if dl.swath and dl.swath not in seen_swaths:
                seen_swaths.add(dl.swath)
                all_downlinks.append(dl)
            elif not dl.swath:
                all_downlinks.append(dl)

    merged = AnnotationData(
        source_file=first.source_file,
        polarization=",".join(annotations.keys()),
        orbit_state_vectors=first.orbit_state_vectors,
        attitude_records=first.attitude_records,
        radar_parameters=first.radar_parameters,
        geolocation_grid=first.geolocation_grid,
        downlink_info=(
            all_downlinks
            if all_downlinks
            else first.downlink_info
        ),
    )

    for data in annotations.values():
        for sp in data.swath_parameters:
            existing = None
            for existing_sp in merged.swath_parameters:
                if existing_sp.swath_id == sp.swath_id:
                    existing = existing_sp
                    break
            if existing is None:
                merged.swath_parameters.append(sp)

    return merged
