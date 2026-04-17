# -*- coding: utf-8 -*-
"""
Sentinel-1 L0 Metadata - Typed metadata for Sentinel-1 Level 0 RAW products.

Nested dataclasses and enums representing the full annotation, orbit,
attitude, radar, and packet-level metadata contained in a Sentinel-1
Level 0 SAFE product.  Each swath+polarization combination within a
SAFE archive maps to one ``Sentinel1L0Metadata`` instance.

Data Sources
------------
- Annotation XML: ``annotation/s1[abcd]-*.xml``
- Manifest: ``manifest.safe``
- ISP Packet Headers: via the optional ``sentinel1decoder`` package

References
----------
- Sentinel-1 Level-0 Product Format Specifications (S1PD.SP.00110)
- Sentinel-1 SAR Space Packet Protocol Data Unit (S1-IF-ASD-PL-0007)

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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Tuple

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.models.base import ImageMetadata

# ``grdl.IO.sar.sentinel1_l0.constants`` is imported lazily inside
# the handful of methods that reference it.  Eager import here would
# trigger ``grdl.IO.sar.__init__`` during ``grdl.IO.models`` loading
# and produce a circular import via ``grdl.IO.base``.


# ===================================================================
# Enumerations
# ===================================================================


class Sentinel1Mission(Enum):
    """Sentinel-1 mission identifier.

    Reference: Sentinel-1 Product Specification Document.
    """

    S1A = "Sentinel-1A"
    S1B = "Sentinel-1B"
    S1C = "Sentinel-1C"
    S1D = "Sentinel-1D"

    @property
    def platform_name(self) -> str:
        """Platform name string (e.g., ``"Sentinel-1A"``)."""
        return self.value

    @property
    def sensor_name(self) -> str:
        """Sensor name string (e.g., ``"Sentinel-1A C-SAR"``)."""
        return f"{self.value} C-SAR"


class Sentinel1Mode(Enum):
    """Sentinel-1 acquisition modes.

    Reference: Sentinel-1 User Handbook Section 3.
    """

    IW = "IW"
    EW = "EW"
    SM = "SM"
    WV = "WV"

    @property
    def full_name(self) -> str:
        """Full mode name."""
        names = {
            "IW": "Interferometric Wide Swath",
            "EW": "Extra Wide Swath",
            "SM": "Stripmap",
            "WV": "Wave",
        }
        return names.get(self.value, self.value)

    @property
    def nominal_params(self) -> Dict[str, float]:
        """Nominal radar parameters for this mode."""
        from grdl.IO.sar.sentinel1_l0.constants import (
            EW_MODE_PARAMS,
            IW_MODE_PARAMS,
            SM_MODE_PARAMS,
        )
        params_map = {
            "IW": IW_MODE_PARAMS,
            "EW": EW_MODE_PARAMS,
            "SM": SM_MODE_PARAMS,
            "WV": IW_MODE_PARAMS,
        }
        return params_map.get(self.value, IW_MODE_PARAMS)


class Sentinel1Polarization(Enum):
    """Sentinel-1 polarization options."""

    HH = "HH"
    HV = "HV"
    VH = "VH"
    VV = "VV"

    @property
    def tx_pol(self) -> str:
        """Transmit polarization character."""
        return self.value[0]

    @property
    def rx_pol(self) -> str:
        """Receive polarization character."""
        return self.value[1]


class S1L0SwathID(Enum):
    """Sentinel-1 swath identifiers for IW, EW, SM, and WV modes."""

    IW1 = "IW1"
    IW2 = "IW2"
    IW3 = "IW3"
    EW1 = "EW1"
    EW2 = "EW2"
    EW3 = "EW3"
    EW4 = "EW4"
    EW5 = "EW5"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"
    S5 = "S5"
    S6 = "S6"
    WV1 = "WV1"
    WV2 = "WV2"


# ===================================================================
# Orbit state vector
# ===================================================================


@dataclass
class S1L0OrbitStateVector:
    """Orbit state vector at a specific time.

    Platform position and velocity in Earth-Centered Earth-Fixed
    (ECEF) coordinates at a specific UTC time.  Source: annotation
    XML ``<generalAnnotation><orbitList><orbit>``.

    Parameters
    ----------
    time : datetime
        UTC time of state vector.
    x, y, z : float
        ECEF position in metres.
    vx, vy, vz : float
        ECEF velocity in metres per second.
    """

    time: datetime
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float

    @property
    def position(self) -> np.ndarray:
        """Position as numpy array ``[X, Y, Z]`` in metres."""
        return np.array([self.x, self.y, self.z])

    @property
    def velocity(self) -> np.ndarray:
        """Velocity as numpy array ``[VX, VY, VZ]`` in m/s."""
        return np.array([self.vx, self.vy, self.vz])

    @property
    def speed(self) -> float:
        """Magnitude of velocity in m/s."""
        return float(np.linalg.norm(self.velocity))

    @property
    def altitude(self) -> float:
        """Height above WGS-84 ellipsoid in metres."""
        from grdl.geolocation.coordinates import ecef_to_geodetic
        result = ecef_to_geodetic(
            np.array([self.x, self.y, self.z])
        )
        return float(result[2])

    @property
    def geodetic(self) -> Tuple[float, float, float]:
        """Geodetic coordinates ``(lat_deg, lon_deg, alt_m)``."""
        from grdl.geolocation.coordinates import ecef_to_geodetic
        result = ecef_to_geodetic(
            np.array([self.x, self.y, self.z])
        )
        return (float(result[0]), float(result[1]), float(result[2]))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "time": self.time.isoformat(),
            "x": self.x, "y": self.y, "z": self.z,
            "vx": self.vx, "vy": self.vy, "vz": self.vz,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "S1L0OrbitStateVector":
        """Create from dictionary."""
        time = data["time"]
        if isinstance(time, str):
            time = datetime.fromisoformat(
                time.replace("Z", "+00:00")
            )
        return cls(
            time=time,
            x=data.get("x", data.get("position_x", 0)),
            y=data.get("y", data.get("position_y", 0)),
            z=data.get("z", data.get("position_z", 0)),
            vx=data.get("vx", data.get("velocity_x", 0)),
            vy=data.get("vy", data.get("velocity_y", 0)),
            vz=data.get("vz", data.get("velocity_z", 0)),
        )


# ===================================================================
# Attitude record
# ===================================================================


@dataclass
class S1L0AttitudeRecord:
    """Platform attitude at a specific time.

    Contains roll, pitch, yaw angles and optionally quaternion
    representation of platform orientation.  Source: annotation
    XML ``<generalAnnotation><attitudeList><attitude>``.

    Parameters
    ----------
    time : datetime
        UTC time of attitude measurement.
    roll, pitch, yaw : float
        Euler angles in degrees.
    q0, q1, q2, q3 : float, optional
        Quaternion components (scalar, i, j, k).
    """

    time: datetime
    roll: float
    pitch: float
    yaw: float
    q0: Optional[float] = None
    q1: Optional[float] = None
    q2: Optional[float] = None
    q3: Optional[float] = None

    @property
    def euler_angles_rad(self) -> Tuple[float, float, float]:
        """Roll, pitch, yaw in radians."""
        return (
            np.radians(self.roll),
            np.radians(self.pitch),
            np.radians(self.yaw),
        )

    @property
    def has_quaternion(self) -> bool:
        """Whether quaternion data is available."""
        return all(
            q is not None
            for q in [self.q0, self.q1, self.q2, self.q3]
        )

    @property
    def quaternion(self) -> Optional[np.ndarray]:
        """Quaternion as ``[q0, q1, q2, q3]`` if available."""
        if self.has_quaternion:
            return np.array(
                [self.q0, self.q1, self.q2, self.q3]
            )
        return None

    def to_rotation_matrix(self) -> np.ndarray:
        """Convert attitude to a 3x3 rotation matrix.

        Returns
        -------
        numpy.ndarray
            3x3 rotation matrix from body frame to ECEF.
        """
        if self.has_quaternion:
            return self._quaternion_to_matrix()
        return self._euler_to_matrix()

    def _quaternion_to_matrix(self) -> np.ndarray:
        q0, q1, q2, q3 = self.q0, self.q1, self.q2, self.q3
        return np.array([
            [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3),
             2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2),
             2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1),
             1 - 2 * (q1**2 + q2**2)],
        ])

    def _euler_to_matrix(self) -> np.ndarray:
        r, p, y = self.euler_angles_rad
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        # R = Rz * Ry * Rx
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr,
             cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr,
             sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "time": self.time.isoformat(),
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
        }
        if self.has_quaternion:
            result.update({
                "q0": self.q0, "q1": self.q1,
                "q2": self.q2, "q3": self.q3,
            })
        return result

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any],
    ) -> "S1L0AttitudeRecord":
        """Create from dictionary."""
        time = data["time"]
        if isinstance(time, str):
            time = datetime.fromisoformat(
                time.replace("Z", "+00:00")
            )
        return cls(
            time=time,
            roll=data["roll"],
            pitch=data["pitch"],
            yaw=data["yaw"],
            q0=data.get("q0"),
            q1=data.get("q1"),
            q2=data.get("q2"),
            q3=data.get("q3"),
        )


# ===================================================================
# Radar parameters
# ===================================================================


@dataclass
class S1L0RadarParameters:
    """Radar signal and system parameters.

    Source: annotation XML ``<generalAnnotation><productInformation>``
    and ``<generalAnnotation><downlinkInformation>``.

    Parameters
    ----------
    center_frequency_hz : float
        Radar center frequency in Hz.
    range_sampling_rate_hz : float
        A/D sampling rate in Hz.
    pulse_repetition_frequency_hz : float
        PRF in Hz.
    tx_pulse_length_s : float, optional
        Transmit pulse duration in seconds.
    tx_pulse_start_frequency_hz : float, optional
        Chirp start frequency offset in Hz.
    tx_pulse_ramp_rate_hz_per_s : float, optional
        Chirp rate (FM rate) in Hz/s.
    azimuth_steering_rate_deg_per_s : float, optional
        TOPS antenna steering rate in deg/s.
    rank : int, optional
        Range processing rank.
    polarization : str, optional
        Polarization mode string.
    """

    center_frequency_hz: float
    range_sampling_rate_hz: float
    pulse_repetition_frequency_hz: float
    tx_pulse_length_s: float = 0.0
    tx_pulse_start_frequency_hz: float = 0.0
    tx_pulse_ramp_rate_hz_per_s: float = 0.0
    azimuth_steering_rate_deg_per_s: float = 0.0
    rank: int = 0
    polarization: str = ""

    @property
    def wavelength_m(self) -> float:
        """Radar wavelength in metres."""
        from grdl.IO.sar.sentinel1_l0.constants import (
            SPEED_OF_LIGHT,
        )
        return SPEED_OF_LIGHT / self.center_frequency_hz

    @property
    def center_frequency_ghz(self) -> float:
        """Center frequency in GHz."""
        return self.center_frequency_hz / 1e9

    @property
    def chirp_bandwidth_hz(self) -> float:
        """Chirp bandwidth in Hz (|rate| × pulse_length)."""
        if (
            self.tx_pulse_ramp_rate_hz_per_s
            and self.tx_pulse_length_s
        ):
            return abs(
                self.tx_pulse_ramp_rate_hz_per_s
                * self.tx_pulse_length_s
            )
        return 0.0

    @property
    def chirp_bandwidth_mhz(self) -> float:
        """Chirp bandwidth in MHz."""
        return self.chirp_bandwidth_hz / 1e6

    @property
    def range_resolution_m(self) -> float:
        """Theoretical range resolution in metres."""
        from grdl.IO.sar.sentinel1_l0.constants import (
            SPEED_OF_LIGHT,
        )
        if self.chirp_bandwidth_hz > 0:
            return SPEED_OF_LIGHT / (2 * self.chirp_bandwidth_hz)
        return 0.0

    @property
    def pulse_repetition_interval_s(self) -> float:
        """Pulse repetition interval (PRI) in seconds."""
        if self.pulse_repetition_frequency_hz > 0:
            return 1.0 / self.pulse_repetition_frequency_hz
        return 0.0

    def get_frequency_bounds(self) -> Tuple[float, float]:
        """Frequency bounds ``(fx1, fx2)`` in Hz.

        Uses ``range_sampling_rate_hz`` to define the receive
        passband around ``center_frequency_hz``.
        """
        half_bw = self.range_sampling_rate_hz / 2.0
        fx1 = self.center_frequency_hz - half_bw
        fx2 = self.center_frequency_hz + half_bw
        return (fx1, fx2)

    def get_transmit_frequency_bounds(self) -> Tuple[float, float]:
        """Transmit frequency bounds ``(fx_min, fx_max)`` in Hz."""
        half_bw = self.chirp_bandwidth_hz / 2.0
        return (
            self.center_frequency_hz - half_bw,
            self.center_frequency_hz + half_bw,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "center_frequency_hz": self.center_frequency_hz,
            "range_sampling_rate_hz": self.range_sampling_rate_hz,
            "pulse_repetition_frequency_hz":
                self.pulse_repetition_frequency_hz,
            "tx_pulse_length_s": self.tx_pulse_length_s,
            "tx_pulse_start_frequency_hz":
                self.tx_pulse_start_frequency_hz,
            "tx_pulse_ramp_rate_hz_per_s":
                self.tx_pulse_ramp_rate_hz_per_s,
            "azimuth_steering_rate_deg_per_s":
                self.azimuth_steering_rate_deg_per_s,
            "rank": self.rank,
            "polarization": self.polarization,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any],
    ) -> "S1L0RadarParameters":
        """Create from dictionary."""
        return cls(**data)


# ===================================================================
# Downlink information
# ===================================================================


@dataclass
class S1L0DownlinkInfo:
    """Downlink timing and instrument configuration.

    Source: annotation XML or ISP packet headers.

    Parameters
    ----------
    prf : float
        Pulse repetition frequency in Hz.
    pri : float
        Pulse repetition interval in seconds.
    rank : int, optional
        Processing rank.
    swst : float, optional
        Sampling Window Start Time in seconds.
    swl : int, optional
        Sampling Window Length in samples.
    swath : str, optional
        Swath identifier (e.g. ``"IW1"``, ``"IW2"``).
    tx_pulse_ramp_rate : float, optional
        Chirp rate in Hz/s.
    tx_pulse_length : float, optional
        Transmit pulse length in seconds.
    tx_pulse_start_frequency : float, optional
        Chirp start frequency offset in Hz.
    range_decimation_code : int, optional
        Range decimation filter code (0-11).
    range_sampling_rate_hz : float, optional
        Actual sampling rate derived from decimation code.
    """

    prf: float
    pri: float
    rank: int = 0
    swst: float = 0.0
    swl: int = 0
    swath: str = ""
    tx_pulse_ramp_rate: float = 0.0
    tx_pulse_length: float = 0.0
    tx_pulse_start_frequency: float = 0.0
    range_decimation_code: int = -1
    range_sampling_rate_hz: float = 0.0


# ===================================================================
# Burst record
# ===================================================================


@dataclass
class S1L0BurstRecord:
    """Single burst timing and geometry information.

    Source: annotation XML ``<swathTiming><burstList><burst>``.

    Parameters
    ----------
    burst_index : int
        Index of burst within swath (0-indexed).
    azimuth_time : datetime
        Burst center azimuth time (UTC).
    sensing_time : datetime, optional
        Burst sensing start time (UTC).
    azimuth_anx_time : float, optional
        Azimuth time since ascending node.
    byte_offset : int, optional
        Offset in measurement file.
    first_valid_sample, last_valid_sample : int, optional
        Valid sample index range.
    lines_per_burst, samples_per_burst : int, optional
        Burst dimensions.
    """

    burst_index: int
    azimuth_time: datetime
    sensing_time: Optional[datetime] = None
    azimuth_anx_time: float = 0.0
    byte_offset: int = 0
    first_valid_sample: int = 0
    last_valid_sample: int = 0
    lines_per_burst: int = 0
    samples_per_burst: int = 0


# ===================================================================
# Swath parameters
# ===================================================================


@dataclass
class S1L0SwathParameters:
    """Per-swath parameters including burst list.

    Source: annotation XML ``<swathTiming>``.

    Parameters
    ----------
    swath_id : str
        Swath identifier (e.g., ``"IW1"``, ``"IW2"``, ``"IW3"``).
    polarization : str, optional
        Polarization for this swath.
    num_bursts, lines_per_burst, samples_per_burst : int, optional
        Burst count and nominal dimensions.
    azimuth_time_interval : float, optional
        Time between azimuth lines.
    range_sampling_rate : float, optional
        Range sampling rate for swath.
    slant_range_time : float, optional
        Slant range time to first sample.
    bursts : List[S1L0BurstRecord], optional
        List of per-burst records.
    """

    swath_id: str
    polarization: str = ""
    num_bursts: int = 0
    lines_per_burst: int = 0
    samples_per_burst: int = 0
    azimuth_time_interval: float = 0.0
    range_sampling_rate: float = 0.0
    slant_range_time: float = 0.0
    bursts: List[S1L0BurstRecord] = field(default_factory=list)


# ===================================================================
# Instrument timing
# ===================================================================


@dataclass
class S1L0InstrumentTiming:
    """Per-packet instrument timing from ISP headers.

    Source: decoded ISP packet headers.

    Parameters
    ----------
    packet_index : int
        Index in measurement file.
    coarse_time, fine_time : int, optional
        ISP Coarse Time (seconds since GPS epoch) and Fine Time
        (16-bit fraction of a second).
    pri_code, swst_code : int, optional
        Codes from packet header.
    swath_number : int, optional
        Raw swath number from header.
    polarization : int, optional
        Polarization code.
    ecc_number : int, optional
        ECC number for calibration.
    baq_mode : int, optional
        BAQ compression mode.
    num_quads : int, optional
        Number of quads (sample pairs).
    sensing_time : datetime, optional
        Computed sensing time (UTC).
    """

    packet_index: int
    coarse_time: int = 0
    fine_time: int = 0
    pri_code: Optional[int] = None
    swst_code: Optional[int] = None
    swath_number: Optional[int] = None
    polarization: Optional[int] = None
    ecc_number: Optional[int] = None
    baq_mode: Optional[int] = None
    num_quads: Optional[int] = None
    sensing_time: Optional[datetime] = None

    # Ref: GPS IS-GPS-200 Rev N, Section 20.3.3.
    GPS_EPOCH: ClassVar[datetime] = datetime(1980, 1, 6, 0, 0, 0)
    # Ref: ISP Section 3.2.1 (Datation Service), Table 3-2.
    FINE_TIME_DIVISOR: ClassVar[float] = 2 ** 16

    def compute_sensing_time(self) -> None:
        """Compute sensing time from coarse/fine time values."""
        if self.coarse_time == 0 and self.fine_time == 0:
            return
        fractional = self.fine_time / self.FINE_TIME_DIVISOR
        total_seconds = self.coarse_time + fractional
        self.sensing_time = (
            self.GPS_EPOCH + timedelta(seconds=total_seconds)
        )


# ===================================================================
# Geolocation grid
# ===================================================================


@dataclass
class S1L0GeolocationGrid:
    """Geolocation grid for range/azimuth to geographic mapping.

    Source: annotation XML ``<geolocationGrid>``.

    Parameters
    ----------
    azimuth_times : List[datetime]
        Azimuth times of grid points.
    slant_range_times : List[float]
        Slant range times of grid points.
    latitudes, longitudes, heights : numpy.ndarray, optional
        2D grids of geodetic coordinates.
    """

    azimuth_times: List[datetime] = field(default_factory=list)
    slant_range_times: List[float] = field(default_factory=list)
    latitudes: Optional[np.ndarray] = None
    longitudes: Optional[np.ndarray] = None
    heights: Optional[np.ndarray] = None


# ===================================================================
# Top-level metadata
# ===================================================================


@dataclass
class Sentinel1L0Metadata(ImageMetadata):
    """Complete typed metadata for a Sentinel-1 Level 0 product.

    Inherits universal fields (``format``, ``rows``, ``cols``,
    ``dtype``) from ``ImageMetadata`` and adds Sentinel-1 L0
    annotation, orbit, attitude, radar, and packet metadata as
    typed fields.

    Parameters
    ----------
    product_id : str
        Product identifier string from manifest.
    mission : Sentinel1Mission, optional
        Mission (S1A, S1B, S1C, S1D).
    mode : Sentinel1Mode, optional
        Acquisition mode (IW, EW, SM, WV).
    polarizations : List[str]
        List of polarization channels present.
    start_time, stop_time : datetime, optional
        Acquisition start and stop times (UTC).
    orbit_state_vectors : List[S1L0OrbitStateVector]
        Orbit state vectors from annotation or POE file.
    attitude_records : List[S1L0AttitudeRecord]
        Attitude records from annotation.
    radar_parameters : S1L0RadarParameters, optional
        Radar signal parameters.
    swath_parameters : List[S1L0SwathParameters]
        Per-swath parameters including burst lists.
    geolocation_grid : S1L0GeolocationGrid, optional
        Geolocation tie-point grid.
    instrument_timings : List[S1L0InstrumentTiming]
        Per-packet timing from ISP headers.
    downlink_info : List[S1L0DownlinkInfo]
        Per-swath downlink instrument configuration.

    Examples
    --------
    >>> from grdl.IO.sar import Sentinel1L0Reader
    >>> with Sentinel1L0Reader('product.SAFE') as reader:
    ...     meta = reader.metadata
    ...     print(meta.mission)                    # Sentinel1Mission.S1A
    ...     print(len(meta.orbit_state_vectors))   # 17
    ...     print(meta.radar_parameters.wavelength_m)  # 0.0555
    """

    product_id: str = ""
    mission: Optional[Sentinel1Mission] = None
    mode: Optional[Sentinel1Mode] = None
    polarizations: List[str] = field(default_factory=list)

    start_time: Optional[datetime] = None
    stop_time: Optional[datetime] = None

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
    instrument_timings: List[S1L0InstrumentTiming] = field(
        default_factory=list
    )
    downlink_info: List[S1L0DownlinkInfo] = field(
        default_factory=list
    )

    @property
    def num_orbit_state_vectors(self) -> int:
        """Number of orbit state vectors."""
        return len(self.orbit_state_vectors)

    @property
    def num_attitude_records(self) -> int:
        """Number of attitude records."""
        return len(self.attitude_records)

    @property
    def polarization(self) -> str:
        """Polarization code as a compact string.

        Returns
        -------
        str
            ``"DV"`` for dual VV+VH, ``"DH"`` for dual HH+HV,
            ``"SV"`` for single VV, ``"SH"`` for single HH, or
            a ``"+"``-joined string for other combinations.
        """
        if not self.polarizations:
            return ""
        pols = sorted(self.polarizations)
        if pols == ["VH", "VV"] or pols == ["VV", "VH"]:
            return "DV"
        if pols == ["HH", "HV"] or pols == ["HV", "HH"]:
            return "DH"
        if pols == ["VV"]:
            return "SV"
        if pols == ["HH"]:
            return "SH"
        if len(pols) == 1:
            return pols[0]
        return "+".join(pols)

    def get_reference_time(self) -> Optional[datetime]:
        """Reference time for relative-time calculations.

        Returns
        -------
        datetime or None
            Product start time truncated to whole seconds, or
            ``None`` if ``start_time`` is not populated.
        """
        if self.start_time is None:
            return None
        return self.start_time.replace(microsecond=0)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "Sentinel1L0Metadata:",
            f"  Product: {self.product_id}",
            f"  Mission: {self.mission}",
            f"  Mode: {self.mode}",
            f"  Polarizations: {self.polarizations}",
            f"  Time: {self.start_time} to {self.stop_time}",
            f"  Orbit vectors: {self.num_orbit_state_vectors}",
            f"  Attitude records: {self.num_attitude_records}",
            f"  Swaths: {len(self.swath_parameters)}",
        ]
        if self.radar_parameters:
            rp = self.radar_parameters
            lines.extend([
                f"  Center freq: {rp.center_frequency_ghz:.3f} GHz",
                f"  Bandwidth: {rp.chirp_bandwidth_mhz:.1f} MHz",
                f"  PRF: {rp.pulse_repetition_frequency_hz:.1f} Hz",
            ])
        return "\n".join(lines)
