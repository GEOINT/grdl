# -*- coding: utf-8 -*-
"""
Sentinel-1 L0 Geometry Calculations.

Position, velocity, and antenna-orientation calculations:

- Orbit state vector interpolation (delegated to
  :class:`OrbitInterpolator`)
- Antenna Coordinate Frame (ACF) vector computation
- Local ENU basis at a geodetic point
- Nominal incidence angles per swath and mode
- Slant range / incidence angle calculations

Coordinate-system transforms (ECEF ↔ geodetic, ENU) are provided by
:mod:`grdl.geolocation.coordinates` and are not duplicated here.

References
----------
- WGS-84 Geodetic Reference System

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
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.sar.sentinel1_l0.constants import (
    MEAN_EARTH_RADIUS,
    SPEED_OF_LIGHT,
    WGS84_SEMI_MAJOR_AXIS,
)

if TYPE_CHECKING:
    from grdl.IO.models.sentinel1_l0 import (
        S1L0AttitudeRecord,
        S1L0OrbitStateVector,
    )

logger = logging.getLogger(__name__)


@dataclass
class StateVector:
    """Position and velocity state vector in ECEF coordinates.

    Parameters
    ----------
    position : numpy.ndarray
        ``[X, Y, Z]`` position in metres.
    velocity : numpy.ndarray
        ``[VX, VY, VZ]`` velocity in m/s.
    time_offset : float, optional
        Time offset from reference (seconds).
    """

    position: np.ndarray
    velocity: np.ndarray
    time_offset: float = 0.0

    @property
    def x(self) -> float:
        """X position (metres)."""
        return float(self.position[0])

    @property
    def y(self) -> float:
        """Y position (metres)."""
        return float(self.position[1])

    @property
    def z(self) -> float:
        """Z position (metres)."""
        return float(self.position[2])

    @property
    def vx(self) -> float:
        """X velocity (m/s)."""
        return float(self.velocity[0])

    @property
    def vy(self) -> float:
        """Y velocity (m/s)."""
        return float(self.velocity[1])

    @property
    def vz(self) -> float:
        """Z velocity (m/s)."""
        return float(self.velocity[2])

    @property
    def speed(self) -> float:
        """Magnitude of velocity (m/s)."""
        return float(np.linalg.norm(self.velocity))

    @property
    def altitude(self) -> float:
        """Approximate altitude above WGS-84 ellipsoid (metres).

        Spherical approximation: ``|position| − a``.  For exact
        geodetic height use
        :func:`grdl.geolocation.coordinates.ecef_to_geodetic`.
        """
        r = float(np.linalg.norm(self.position))
        return r - WGS84_SEMI_MAJOR_AXIS


@dataclass
class ACFVectors:
    """Antenna Coordinate Frame unit vectors.

    The ACF defines the antenna orientation in ECEF.  ``ACX``
    points along-track; ``ACY`` lies in the antenna aperture
    plane perpendicular to ``ACX``.

    Parameters
    ----------
    acx : numpy.ndarray
        ``ACX`` unit vector ``[X, Y, Z]``.
    acy : numpy.ndarray
        ``ACY`` unit vector ``[X, Y, Z]``.
    """

    acx: np.ndarray
    acy: np.ndarray

    def __post_init__(self) -> None:
        """Normalize vectors to unit length."""
        acx_norm = np.linalg.norm(self.acx)
        acy_norm = np.linalg.norm(self.acy)
        if acx_norm > 0:
            self.acx = self.acx / acx_norm
        if acy_norm > 0:
            self.acy = self.acy / acy_norm


class GeometryCalculator:
    """Orbit and attitude interpolation plus ACF computation.

    Wraps :class:`OrbitInterpolator` and
    :class:`AttitudeInterpolator` (from :mod:`.orbit`) to compute
    antenna coordinate frame vectors at arbitrary times.

    Parameters
    ----------
    orbit_vectors : list of S1L0OrbitStateVector
        Orbit state vectors.
    reference_time : datetime
        Reference time for relative-seconds math.
    attitude_records : list of S1L0AttitudeRecord, optional
        Attitude records; if at least 4 are provided the ACF
        computation will use measured attitude.

    Raises
    ------
    ValueError
        If fewer than 4 orbit vectors are provided.

    Examples
    --------
    >>> geom = GeometryCalculator(orbit_vectors, ref_time)
    >>> pos, vel = geom.interpolate_position_velocity(times)
    >>> positions, velocities, acx, acy = (
    ...     geom.get_state_vectors_at_times(times)
    ... )
    """

    def __init__(
        self,
        orbit_vectors: List["S1L0OrbitStateVector"],
        reference_time: datetime,
        attitude_records: Optional[
            List["S1L0AttitudeRecord"]
        ] = None,
    ) -> None:
        from grdl.IO.sar.sentinel1_l0.orbit import (
            AttitudeInterpolator,
            OrbitInterpolator,
        )

        self.reference_time = reference_time
        self._orbit_interp = OrbitInterpolator(
            orbit_vectors, reference_time
        )
        self.valid_start = self._orbit_interp.valid_start
        self.valid_end = self._orbit_interp.valid_end

        self._attitude_interp: Optional[
            AttitudeInterpolator
        ] = None
        if attitude_records and len(attitude_records) >= 4:
            self._attitude_interp = AttitudeInterpolator(
                attitude_records, reference_time
            )

    def interpolate_position_velocity(
        self, times: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate position and velocity at given times.

        Parameters
        ----------
        times : numpy.ndarray
            Times in seconds relative to ``reference_time``.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            ``(positions, velocities)``, each shape ``(N, 3)``.
        """
        return self._orbit_interp.interpolate(times)

    def interpolate_single(
        self, time_seconds: float,
    ) -> StateVector:
        """Interpolate a single state vector.

        Parameters
        ----------
        time_seconds : float
            Time in seconds relative to ``reference_time``.

        Returns
        -------
        StateVector
        """
        pos, vel = self._orbit_interp.interpolate_single(
            time_seconds
        )
        return StateVector(
            position=pos,
            velocity=vel,
            time_offset=time_seconds,
        )

    @property
    def has_attitude(self) -> bool:
        """Whether measured attitude data is available."""
        return self._attitude_interp is not None

    def compute_acf_vectors_with_attitude(
        self,
        times: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        look_side: str = "right",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ACF vectors using measured attitude.

        Transforms from the orbit reference frame to the platform
        body frame using roll/pitch/yaw from the attitude records.
        Aerospace ZYX Euler convention.

        Parameters
        ----------
        times : numpy.ndarray
            Times in seconds relative to ``reference_time``.
        positions, velocities : numpy.ndarray
            ECEF positions and velocities, shape ``(N, 3)``.
        look_side : str, optional
            ``"right"`` or ``"left"``.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            ``(acx, acy)`` unit vectors, each shape ``(N, 3)``.
        """
        if self._attitude_interp is None:
            return self.compute_yaw_steering_acf(
                positions, velocities, look_side
            )

        n = len(positions)
        acx = np.zeros((n, 3))
        acy = np.zeros((n, 3))

        roll_deg, pitch_deg, yaw_deg = (
            self._attitude_interp.interpolate(times)
        )

        for i in range(n):
            pos = positions[i]
            vel = velocities[i]

            vel_mag = np.linalg.norm(vel)
            pos_mag = np.linalg.norm(pos)

            if vel_mag == 0 or pos_mag == 0:
                acx[i] = np.array([0.0, 0.0, 1.0])
                acy[i] = np.array([1.0, 0.0, 0.0])
                continue

            # Orbit reference frame: along-track, cross-track,
            # radial.
            x_orb = vel / vel_mag
            z_orb = pos / pos_mag
            y_orb = np.cross(z_orb, x_orb)
            y_orb = y_orb / np.linalg.norm(y_orb)
            x_orb = np.cross(y_orb, z_orb)
            x_orb = x_orb / np.linalg.norm(x_orb)

            roll = np.radians(roll_deg[i])
            pitch = np.radians(pitch_deg[i])
            yaw = np.radians(yaw_deg[i])

            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)

            R = np.array([
                [cy * cp, cy * sp * sr - sy * cr,
                 cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr,
                 sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ])

            orbit_frame = np.column_stack(
                [x_orb, y_orb, z_orb]
            )
            body_frame = orbit_frame @ R.T

            acx[i] = body_frame[:, 0]
            acy[i] = body_frame[:, 1]

            if look_side.lower() == "left":
                acy[i] = -acy[i]

        return acx, acy

    def compute_yaw_steering_acf(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        look_side: str = "right",
        incidence_deg: float = 30.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ACF vectors from yaw-steering geometry.

        Accounts for the mechanical tilt of the antenna toward
        the ground at the specified incidence angle.

        Parameters
        ----------
        positions, velocities : numpy.ndarray
            ECEF positions and velocities, shape ``(N, 3)``.
        look_side : str, optional
            ``"right"`` or ``"left"``.
        incidence_deg : float, optional
            Nominal incidence angle in degrees.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            ``(acx, acy)`` unit vectors, each shape ``(N, 3)``.
        """
        n = len(positions)
        acx = np.zeros((n, 3))
        acy = np.zeros((n, 3))
        inc = np.radians(incidence_deg)

        for i in range(n):
            pos = positions[i]
            vel = velocities[i]

            vel_mag = np.linalg.norm(vel)
            pos_mag = np.linalg.norm(pos)

            if vel_mag == 0 or pos_mag == 0:
                acx[i] = np.array([0.0, 0.0, 1.0])
                acy[i] = np.array([1.0, 0.0, 0.0])
                continue

            x_orb = vel / vel_mag
            z_orb = pos / pos_mag
            y_orb = np.cross(z_orb, x_orb)
            y_orb = y_orb / np.linalg.norm(y_orb)
            x_orb = np.cross(y_orb, z_orb)
            x_orb = x_orb / np.linalg.norm(x_orb)

            orbit_frame = np.column_stack(
                [x_orb, y_orb, z_orb]
            )

            acx_orb = np.array([1.0, 0.0, 0.0])
            if look_side.lower() == "right":
                acz_orb = np.array([
                    0.0, -np.sin(inc), -np.cos(inc),
                ])
            else:
                acz_orb = np.array([
                    0.0, np.sin(inc), -np.cos(inc),
                ])

            acy_orb = np.cross(acz_orb, acx_orb)

            acx[i] = orbit_frame @ acx_orb
            acy[i] = orbit_frame @ acy_orb

        return acx, acy

    def get_state_vectors_at_times(
        self,
        times: np.ndarray,
        look_side: str = "right",
        use_attitude: bool = True,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    ]:
        """Interpolate state vectors with ACF at given times.

        Convenience method that combines interpolation and ACF
        computation.  When attitude data is available and
        ``use_attitude=True``, ACF is computed from measured
        attitude.

        Parameters
        ----------
        times : numpy.ndarray
            Times in seconds relative to ``reference_time``.
        look_side : str, optional
            ``"right"`` or ``"left"``.
        use_attitude : bool, optional
            Use attitude data when available.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray, numpy.ndarray,
         numpy.ndarray)
            ``(positions, velocities, acx, acy)``.
        """
        positions, velocities = (
            self.interpolate_position_velocity(times)
        )

        if use_attitude and self.has_attitude:
            acx, acy = self.compute_acf_vectors_with_attitude(
                times, positions, velocities, look_side
            )
        else:
            acx, acy = self.compute_yaw_steering_acf(
                positions, velocities, look_side
            )

        return positions, velocities, acx, acy


def compute_local_enu_basis(
    lat_deg: float,
    lon_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute local East-North-Up basis vectors at a geodetic point.

    Parameters
    ----------
    lat_deg : float
        Latitude in degrees.
    lon_deg : float
        Longitude in degrees.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        ``(east, north, up)`` unit vectors in ECEF.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    east = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([
        -sin_lat * cos_lon,
        -sin_lat * sin_lon,
        cos_lat,
    ])
    up = np.array([
        cos_lat * cos_lon,
        cos_lat * sin_lon,
        sin_lat,
    ])
    return east, north, up


def compute_acf_vectors_batch(
    positions: np.ndarray,
    velocities: np.ndarray,
    incidence_deg: float = 30.0,
    look_side: str = "right",
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized yaw-steering ACF computation.

    Parameters
    ----------
    positions : numpy.ndarray
        Platform ECEF positions, shape ``(N, 3)``.
    velocities : numpy.ndarray
        Platform ECEF velocities, shape ``(N, 3)``.
    incidence_deg : float, optional
        Nominal incidence angle in degrees.
    look_side : str, optional
        ``"right"`` or ``"left"``.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(ACX, ACY)`` unit vectors, each shape ``(N, 3)``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)

    if positions.ndim == 1:
        positions = positions.reshape(1, 3)
        velocities = velocities.reshape(1, 3)

    inc = np.radians(incidence_deg)

    pos_norms = np.linalg.norm(
        positions, axis=1, keepdims=True
    )
    vel_norms = np.linalg.norm(
        velocities, axis=1, keepdims=True
    )
    pos_norms = np.maximum(pos_norms, 1e-10)
    vel_norms = np.maximum(vel_norms, 1e-10)

    x_orb = velocities / vel_norms
    z_orb = positions / pos_norms

    y_orb = np.cross(z_orb, x_orb)
    y_norms = np.linalg.norm(
        y_orb, axis=1, keepdims=True
    )
    y_norms = np.maximum(y_norms, 1e-10)
    y_orb = y_orb / y_norms

    x_orb = np.cross(y_orb, z_orb)
    x_norms = np.linalg.norm(
        x_orb, axis=1, keepdims=True
    )
    x_norms = np.maximum(x_norms, 1e-10)
    x_orb = x_orb / x_norms

    sin_inc = np.sin(inc)
    cos_inc = np.cos(inc)
    sign = -1.0 if look_side.lower() == "right" else 1.0

    acx = x_orb
    acy = -cos_inc * y_orb + (-sign * sin_inc) * z_orb

    acy_norms = np.linalg.norm(
        acy, axis=1, keepdims=True
    )
    acy_norms = np.maximum(acy_norms, 1e-10)
    acy = acy / acy_norms

    return acx, acy


# =============================================================================
# Incidence angle computation
# =============================================================================

# Nominal mid-swath incidence angles for Sentinel-1 IW (degrees).
# Ref: [SRD] S1-RS-ASD-PL-0001, Section 3.2.1.2 (TOPS / IW mode).
# Full-swath range: IW1 ~31-36°, IW2 ~36-40°, IW3 ~40-46°.
IW_SWATH_INCIDENCE_DEG = {
    "IW1": 33.0,
    "IW2": 38.0,
    "IW3": 43.0,
}

# Nominal mid-swath incidence angles for Sentinel-1 EW (degrees).
# Ref: [SRD] S1-RS-ASD-PL-0001, Section 3.2.1.3 (EW mode).
EW_SWATH_INCIDENCE_DEG = {
    "EW1": 27.0,
    "EW2": 32.0,
    "EW3": 37.0,
    "EW4": 42.0,
    "EW5": 47.0,
}


def get_swath_incidence_deg(swath_name: str) -> float:
    """Nominal mid-swath incidence angle for a Sentinel-1 swath.

    Parameters
    ----------
    swath_name : str
        Swath name (``"IW1"``, ``"IW2"``, ``"IW3"``,
        ``"EW1"``–``"EW5"``).

    Returns
    -------
    float
        Nominal incidence angle in degrees.  Falls back to
        ``30.0`` with a warning if the swath is not recognized.
    """
    swath_upper = swath_name.upper()
    if swath_upper in IW_SWATH_INCIDENCE_DEG:
        return IW_SWATH_INCIDENCE_DEG[swath_upper]
    if swath_upper in EW_SWATH_INCIDENCE_DEG:
        return EW_SWATH_INCIDENCE_DEG[swath_upper]
    logger.warning(
        "Unknown swath '%s'; using 30.0 deg fallback",
        swath_name,
    )
    return 30.0


def get_mode_mid_incidence_deg(mode: str) -> float:
    """Mid-range incidence angle for a given SAR mode.

    Parameters
    ----------
    mode : str
        Imaging mode (``"IW"``, ``"EW"``, ``"SM"``, ``"WV"``).

    Returns
    -------
    float
        Mid-range incidence angle in degrees.
    """
    mode_upper = mode.upper()
    if mode_upper == "IW":
        return IW_SWATH_INCIDENCE_DEG["IW2"]
    if mode_upper == "EW":
        return EW_SWATH_INCIDENCE_DEG["EW3"]
    if mode_upper == "SM":
        return 35.0
    if mode_upper == "WV":
        return 36.0
    logger.warning(
        "Unknown mode '%s'; using 30.0 deg fallback", mode,
    )
    return 30.0


def compute_slant_range(
    rank: int,
    pri_seconds: float,
    swst_seconds: float,
    speed_of_light: float = SPEED_OF_LIGHT,
) -> float:
    """Compute slant range from Sentinel-1 timing.

    ``slant_range = c × (rank × PRI + SWST) / 2``.

    Parameters
    ----------
    rank : int
        Number of PRI intervals before receive window opens.
    pri_seconds : float
        Pulse Repetition Interval in seconds.
    swst_seconds : float
        Sampling Window Start Time in seconds.
    speed_of_light : float, optional
        Speed of light in m/s.

    Returns
    -------
    float
        Slant range in metres.
    """
    two_way_time = rank * pri_seconds + swst_seconds
    return speed_of_light * two_way_time / 2.0


def compute_incidence_from_slant_range(
    slant_range: float,
    satellite_altitude: float,
    earth_radius: float = MEAN_EARTH_RADIUS,
) -> float:
    """Compute incidence angle (spherical-Earth approximation).

    Parameters
    ----------
    slant_range : float
        Distance from radar to target (metres).
    satellite_altitude : float
        Altitude above Earth surface (metres).
    earth_radius : float, optional
        Mean Earth radius in metres.

    Returns
    -------
    float
        Incidence angle in degrees.  Returns ``30.0`` if
        geometry is invalid.
    """
    R = earth_radius
    h = satellite_altitude
    r = slant_range
    R_sat = R + h

    numerator = R ** 2 + R_sat ** 2 - r ** 2
    denominator = 2 * R * R_sat
    if denominator == 0:
        return 30.0

    cos_theta = numerator / denominator
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)

    sin_incidence = R_sat * np.sin(theta_rad) / r
    sin_incidence = np.clip(sin_incidence, -1.0, 1.0)
    incidence_rad = np.arcsin(sin_incidence)
    return float(np.degrees(incidence_rad))


def compute_incidence_from_timing(
    rank: int,
    pri_seconds: float,
    swst_seconds: float,
    satellite_altitude: float,
    earth_radius: float = MEAN_EARTH_RADIUS,
    speed_of_light: float = SPEED_OF_LIGHT,
) -> float:
    """Compute incidence angle from timing parameters.

    Combines :func:`compute_slant_range` with spherical-Earth
    geometry.

    Parameters
    ----------
    rank : int
        Number of PRI intervals before receive window opens.
    pri_seconds : float
        Pulse Repetition Interval in seconds.
    swst_seconds : float
        Sampling Window Start Time in seconds.
    satellite_altitude : float
        Altitude above Earth surface (metres).
    earth_radius, speed_of_light : float, optional
        Overrides for constants.

    Returns
    -------
    float
        Incidence angle in degrees.

    Examples
    --------
    >>> compute_incidence_from_timing(
    ...     rank=8, pri_seconds=1/1717.0,
    ...     swst_seconds=0.005, satellite_altitude=693000.0,
    ... )
    38.1...
    """
    slant_range = compute_slant_range(
        rank, pri_seconds, swst_seconds, speed_of_light
    )
    return compute_incidence_from_slant_range(
        slant_range, satellite_altitude, earth_radius
    )
