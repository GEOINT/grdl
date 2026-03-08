# -*- coding: utf-8 -*-
"""
Coordinate Conversions - WGS-84 geodetic, ECEF, and ENU transforms.

Provides vectorized NumPy-based coordinate conversions between geodetic
(latitude, longitude, height), ECEF (Earth-Centered Earth-Fixed), and
ENU (East-North-Up) coordinate systems on the WGS-84 ellipsoid.

All functions accept scalar or array inputs and operate entirely in
NumPy (no external geodesy libraries required).

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
2026-03-08
"""

from typing import Tuple

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
    lats: np.ndarray,
    lons: np.ndarray,
    heights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geodetic (lat, lon, height) to ECEF (X, Y, Z).

    Parameters
    ----------
    lats : np.ndarray
        Latitudes in degrees.
    lons : np.ndarray
        Longitudes in degrees.
    heights : np.ndarray
        Heights above WGS-84 ellipsoid in meters.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (X, Y, Z) ECEF coordinates in meters.
    """
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

    return x, y, z


def ecef_to_geodetic(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    max_iter: int = 10,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ECEF (X, Y, Z) to geodetic (lat, lon, height).

    Uses the iterative method from SIDD standard Section 3.7.

    Parameters
    ----------
    x, y, z : np.ndarray
        ECEF coordinates in meters.
    max_iter : int
        Maximum iterations for latitude convergence.
    tol : float
        Convergence tolerance on tan(latitude).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (lats, lons, heights) in degrees and meters.
    """
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

    return np.degrees(lat), np.degrees(lon), height


# ===================================================================
# Geodetic <-> ENU
# ===================================================================

def geodetic_to_enu(
    lats: np.ndarray,
    lons: np.ndarray,
    heights: np.ndarray,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geodetic coordinates to ENU relative to a reference point.

    Parameters
    ----------
    lats : np.ndarray
        Target latitudes in degrees.
    lons : np.ndarray
        Target longitudes in degrees.
    heights : np.ndarray
        Target heights above WGS-84 ellipsoid in meters.
    ref_lat : float
        Reference point latitude in degrees.
    ref_lon : float
        Reference point longitude in degrees.
    ref_alt : float
        Reference point altitude in meters HAE.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (east, north, up) in meters.
    """
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)

    # Convert to ECEF
    tx, ty, tz = geodetic_to_ecef(lats, lons, heights)
    rx, ry, rz = geodetic_to_ecef(
        np.float64(ref_lat), np.float64(ref_lon), np.float64(ref_alt),
    )

    # Difference in ECEF
    dx = tx - rx
    dy = ty - ry
    dz = tz - rz

    # Rotation matrix from ECEF to ENU
    lat_r = np.radians(ref_lat)
    lon_r = np.radians(ref_lon)
    sin_lat = np.sin(lat_r)
    cos_lat = np.cos(lat_r)
    sin_lon = np.sin(lon_r)
    cos_lon = np.cos(lon_r)

    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return east, north, up


def enu_to_geodetic(
    east: np.ndarray,
    north: np.ndarray,
    up: np.ndarray,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ENU coordinates to geodetic relative to a reference point.

    Parameters
    ----------
    east : np.ndarray
        East displacement in meters.
    north : np.ndarray
        North displacement in meters.
    up : np.ndarray
        Up displacement in meters.
    ref_lat : float
        Reference point latitude in degrees.
    ref_lon : float
        Reference point longitude in degrees.
    ref_alt : float
        Reference point altitude in meters HAE.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (lats, lons, heights) in degrees and meters.
    """
    east = np.asarray(east, dtype=np.float64)
    north = np.asarray(north, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    # Inverse rotation: ENU -> ECEF difference
    lat_r = np.radians(ref_lat)
    lon_r = np.radians(ref_lon)
    sin_lat = np.sin(lat_r)
    cos_lat = np.cos(lat_r)
    sin_lon = np.sin(lon_r)
    cos_lon = np.cos(lon_r)

    dx = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
    dy = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
    dz = cos_lat * north + sin_lat * up

    # Add reference ECEF
    rx, ry, rz = geodetic_to_ecef(
        np.float64(ref_lat), np.float64(ref_lon), np.float64(ref_alt),
    )

    return ecef_to_geodetic(dx + rx, dy + ry, dz + rz)
