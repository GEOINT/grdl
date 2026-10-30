# -*- coding: utf-8 -*-
"""
ECF (Earth-Centered-Fixed) to SEZ (East-South-Zenith) coordinate transformation functions.

Transforms Earth-Centered-Fixed (ECF) coordinates to East-South-Zenith (SEZ) coordinates
local reference frame. The transformation is baswed on the rotation matrix from
"Fundamentals of Astrodynamics" by Bate, Mueller, and White.

Dependencies:
- NumPy for array and matrix operations.

Author
------
Jason Fritz, PhD
43161141+stryder-vtx@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-10-16

References
----------
Bate, R. R., Mueller, D. D., & White, J. E. (1971). Fundamentals of Astrodynamics. Dover Publications.

"""

from typing import Union
import numpy as np


def ecf_to_sez(ecf_coords: Union[np.ndarray, list, tuple], lat_deg: float, lon_deg: float) -> np.ndarray:
    """
    Convert Earth-Centered-Fixed (ECF) coordinates to East-South-Zenith (SEZ) coordinates.

    Parameters
    ----------
    ecf_coords : Union[np.ndarray, list, tuple]
        ECF coordinates as a 3-element array, list, or tuple [x, y, z].
    lat_deg : float
        Latitude of the local reference frame in degrees.
    lon_deg : float
        Longitude of the local reference frame in degrees.

    Returns
    -------
    np.ndarray
        SEZ coordinates as a 3-element array [s, e, z].
    """

    # Convert input coordinates and angles to appropriate formats
    ecf_coords = np.asarray(ecf_coords).reshape(3, 1)
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    # Check input shape
    if ecf_coords.ndim == 1:
        if ecf_coords.shape[0] != 3:
            raise ValueError(f"ECF coordinates must have shape (3,), got {ecf_coords.shape}")
    elif ecf_coords.ndim == 2:
        if ecf_coords.shape[1] != 3:
            raise ValueError(f"Multiple vectors must have shape (N, 3), got {ecf_coords.shape}")
    else:
        raise ValueError(f"ECF coordinates must be 1D or 2D array, got {ecf_coords.ndim}D")

    # Precompute trigonometric functions of latitude and longitude
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # Rotation matrix from ECF to SEZ
    # 1st row corresponds to South, 2nd row to East, 3rd row to Zenith
    # Note: 1st row negative to match the SAR Toolbox conversion
    ecf2sez = np.array([
        [sin_lat * cos_lon, sin_lat * sin_lon,  -cos_lat],
        [-sin_lon,            cos_lon,           0      ],
        [ cos_lat * cos_lon,  cos_lat * sin_lon, sin_lat]
    ])

    # Apply rotation matrix to convert ECF to SEZ coordinates
    sez_coords = ecf_coords @ ecf2sez.T
    return sez_coords.flatten()
