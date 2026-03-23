# -*- coding: utf-8 -*-
"""
Coordinate Conversions - WGS-84 geodetic, ECEF, and ENU transforms.

Provides vectorized NumPy-based coordinate conversions between geodetic
(latitude, longitude, height), ECEF (Earth-Centered Earth-Fixed), and
ENU (East-North-Up) coordinate systems on the WGS-84 ellipsoid.

All functions accept stacked (N, 3) arrays and return stacked (N, 3)
arrays.  Scalar inputs (1D arrays of length 3) return 1D arrays of
length 3.

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
2026-03-08

Modified
--------
2026-03-22  Refactor to (N, M) stacked ndarray convention.
"""

from typing import Union

import numpy as np


# ===================================================================
# WGS-84 ellipsoid constants (NGA.STND.0025-1 Section 3.6)
# ===================================================================

WGS84_A = 6378137.0                            # semi-major axis (m)
WGS84_B = 6356752.314245179                     # semi-minor axis (m)
WGS84_F = 1.0 / 298.257223563                   # flattening
WGS84_E1_SQ = 2 * WGS84_F - WGS84_F ** 2       # first eccentricity squared
WGS84_E2_SQ = (WGS84_A ** 2 - WGS84_B ** 2) / WGS84_B ** 2  # second ecc sq


# ===================================================================
# Geodetic <-> ECEF
# ===================================================================

def geodetic_to_ecef(
    points: np.ndarray,
) -> np.ndarray:
    """Convert geodetic (lat, lon, height) to ECEF (X, Y, Z).

    Parameters
    ----------
    points : np.ndarray
        Geodetic coordinates.  Shape ``(N, 3)`` with columns
        ``[lat_deg, lon_deg, height_m]``, or shape ``(3,)`` for a
        single point.

    Returns
    -------
    np.ndarray
        ECEF coordinates in meters.  Shape ``(N, 3)`` with columns
        ``[X, Y, Z]``, or shape ``(3,)`` for scalar input.
    """
    pts = np.asarray(points, dtype=np.float64)
    scalar = pts.ndim == 1
    if scalar:
        pts = pts.reshape(1, -1)

    lats = pts[:, 0]
    lons = pts[:, 1]
    heights = pts[:, 2]

    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Radius of curvature in the prime vertical
    rc = WGS84_A / np.sqrt(1.0 - WGS84_E1_SQ * sin_lat ** 2)

    x = (rc + heights) * cos_lat * cos_lon
    y = (rc + heights) * cos_lat * sin_lon
    z = ((WGS84_B ** 2 / WGS84_A ** 2) * rc + heights) * sin_lat

    result = np.column_stack([x, y, z])
    return result[0] if scalar else result


def ecef_to_geodetic(
    points: np.ndarray,
    max_iter: int = 10,
    tol: float = 1e-12,
) -> np.ndarray:
    """Convert ECEF (X, Y, Z) to geodetic (lat, lon, height).

    Uses the iterative method from SIDD standard Section 3.7.

    Parameters
    ----------
    points : np.ndarray
        ECEF coordinates in meters.  Shape ``(N, 3)`` with columns
        ``[X, Y, Z]``, or shape ``(3,)`` for a single point.
    max_iter : int
        Maximum iterations for latitude convergence.
    tol : float
        Convergence tolerance on tan(latitude).

    Returns
    -------
    np.ndarray
        Geodetic coordinates.  Shape ``(N, 3)`` with columns
        ``[lat_deg, lon_deg, height_m]``, or shape ``(3,)`` for
        scalar input.
    """
    pts = np.asarray(points, dtype=np.float64)
    scalar = pts.ndim == 1
    if scalar:
        pts = pts.reshape(1, -1)

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    lon = np.arctan2(y, x)

    dxy = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(WGS84_A * z, WGS84_B * dxy)

    # Initial latitude estimate (Bowring's formula)
    tan_lat = (
        (z + WGS84_E2_SQ * WGS84_B * np.sin(theta) ** 3)
        / (dxy - WGS84_E1_SQ * WGS84_A * np.cos(theta) ** 3)
    )

    # Iterative refinement
    for _ in range(max_iter):
        tan_lat_prev = tan_lat
        theta_new = np.arctan2(WGS84_A * z, WGS84_B * dxy)
        tan_lat = (
            (z + WGS84_E2_SQ * WGS84_B * np.sin(theta_new) ** 3)
            / (dxy - WGS84_E1_SQ * WGS84_A * np.cos(theta_new) ** 3)
        )
        if np.all(np.abs(tan_lat - tan_lat_prev) < tol):
            break

    lat = np.arctan(tan_lat)

    # Height above ellipsoid
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    rc = WGS84_A / np.sqrt(1.0 - WGS84_E1_SQ * sin_lat ** 2)

    with np.errstate(invalid='ignore', divide='ignore'):
        h_equatorial = dxy / cos_lat - rc
        h_polar = np.abs(z) / np.where(
            np.abs(sin_lat) > 0, np.abs(sin_lat), 1.0
        ) - (WGS84_B ** 2 / WGS84_A ** 2) * rc
    height = np.where(np.abs(cos_lat) > 1e-10, h_equatorial, h_polar)

    result = np.column_stack([np.degrees(lat), np.degrees(lon), height])
    return result[0] if scalar else result


# ===================================================================
# Geodetic <-> ENU
# ===================================================================

def geodetic_to_enu(
    points: np.ndarray,
    ref: np.ndarray,
) -> np.ndarray:
    """Convert geodetic coordinates to ENU relative to a reference point.

    Parameters
    ----------
    points : np.ndarray
        Target geodetic coordinates.  Shape ``(N, 3)`` with columns
        ``[lat_deg, lon_deg, height_m]``, or shape ``(3,)`` for a
        single point.
    ref : np.ndarray
        Reference point ``[lat_deg, lon_deg, alt_m]``, shape ``(3,)``.

    Returns
    -------
    np.ndarray
        ENU displacements in meters.  Shape ``(N, 3)`` with columns
        ``[east, north, up]``, or shape ``(3,)`` for scalar input.
    """
    pts = np.asarray(points, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    scalar = pts.ndim == 1
    if scalar:
        pts = pts.reshape(1, -1)

    # Convert to ECEF
    target_ecef = geodetic_to_ecef(pts)
    ref_ecef = geodetic_to_ecef(ref)

    # Difference in ECEF
    dx = target_ecef[:, 0] - ref_ecef[0]
    dy = target_ecef[:, 1] - ref_ecef[1]
    dz = target_ecef[:, 2] - ref_ecef[2]

    # Rotation matrix from ECEF to ENU
    lat_r = np.radians(ref[0])
    lon_r = np.radians(ref[1])
    sin_lat = np.sin(lat_r)
    cos_lat = np.cos(lat_r)
    sin_lon = np.sin(lon_r)
    cos_lon = np.cos(lon_r)

    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    result = np.column_stack([east, north, up])
    return result[0] if scalar else result


def enu_to_geodetic(
    points: np.ndarray,
    ref: np.ndarray,
) -> np.ndarray:
    """Convert ENU coordinates to geodetic relative to a reference point.

    Parameters
    ----------
    points : np.ndarray
        ENU displacements in meters.  Shape ``(N, 3)`` with columns
        ``[east, north, up]``, or shape ``(3,)`` for a single point.
    ref : np.ndarray
        Reference point ``[lat_deg, lon_deg, alt_m]``, shape ``(3,)``.

    Returns
    -------
    np.ndarray
        Geodetic coordinates.  Shape ``(N, 3)`` with columns
        ``[lat_deg, lon_deg, height_m]``, or shape ``(3,)`` for
        scalar input.
    """
    pts = np.asarray(points, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    scalar = pts.ndim == 1
    if scalar:
        pts = pts.reshape(1, -1)

    east = pts[:, 0]
    north = pts[:, 1]
    up = pts[:, 2]

    # Inverse rotation: ENU -> ECEF difference
    lat_r = np.radians(ref[0])
    lon_r = np.radians(ref[1])
    sin_lat = np.sin(lat_r)
    cos_lat = np.cos(lat_r)
    sin_lon = np.sin(lon_r)
    cos_lon = np.cos(lon_r)

    dx = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
    dy = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
    dz = cos_lat * north + sin_lat * up

    # Add reference ECEF
    ref_ecef = geodetic_to_ecef(ref)

    ecef = np.column_stack([dx + ref_ecef[0], dy + ref_ecef[1],
                            dz + ref_ecef[2]])
    result = ecef_to_geodetic(ecef)
    return result[0] if scalar else result


# ===================================================================
# Ellipsoidal meters-per-degree
# ===================================================================

def meters_per_degree(lat: Union[float, np.ndarray]) -> np.ndarray:
    """WGS-84 ellipsoidal meters-per-degree at a given latitude.

    Uses the meridional (M) and prime-vertical (N) radii of curvature
    on the WGS-84 ellipsoid for exact conversion, replacing the common
    spherical approximation of 111320 m/deg.

    Parameters
    ----------
    lat : float or np.ndarray
        Geodetic latitude(s) in degrees.

    Returns
    -------
    np.ndarray
        Shape ``(2,)`` for scalar input: ``[meters_per_deg_lat,
        meters_per_deg_lon]``.  Shape ``(N, 2)`` for array input.
    """
    lat_arr = np.asarray(lat, dtype=np.float64)
    scalar = lat_arr.ndim == 0
    lat_arr = np.atleast_1d(lat_arr)

    lat_rad = np.radians(lat_arr)
    sin2 = np.sin(lat_rad) ** 2
    cos_lat = np.cos(lat_rad)

    # Meridional radius of curvature (north-south)
    M = WGS84_A * (1.0 - WGS84_E1_SQ) / (1.0 - WGS84_E1_SQ * sin2) ** 1.5
    # Prime vertical radius of curvature (east-west)
    N = WGS84_A / (1.0 - WGS84_E1_SQ * sin2) ** 0.5

    m_lat = M * np.pi / 180.0
    m_lon = N * cos_lat * np.pi / 180.0

    result = np.column_stack([m_lat, m_lon])
    return result[0] if scalar else result
