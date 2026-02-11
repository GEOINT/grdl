# -*- coding: utf-8 -*-
"""
IO Models Common - Reusable primitive types for metadata dataclasses.

Provides building-block dataclasses shared across SICD, SIDD, and other
sensor-specific metadata: coordinate types (XYZ, LatLon, LatLonHAE,
RowCol) and polynomial types (Poly1D, Poly2D, XYZPoly).

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
2026-02-10

Modified
--------
2026-02-10
"""

# Standard library
from dataclasses import dataclass
from typing import Optional

# Third-party
import numpy as np


@dataclass
class XYZ:
    """ECF (Earth-Centered Fixed) coordinate in meters.

    Parameters
    ----------
    x : float
        X component (meters).
    y : float
        Y component (meters).
    z : float
        Z component (meters).
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class LatLon:
    """WGS-84 geographic point (2D).

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    """

    lat: float = 0.0
    lon: float = 0.0


@dataclass
class LatLonHAE:
    """WGS-84 geographic point with Height Above Ellipsoid.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    hae : float
        Height Above Ellipsoid in meters.
    """

    lat: float = 0.0
    lon: float = 0.0
    hae: float = 0.0


@dataclass
class RowCol:
    """Row/column pixel coordinate.

    Parameters
    ----------
    row : float
        Row (line) coordinate.
    col : float
        Column (sample) coordinate.
    """

    row: float = 0.0
    col: float = 0.0


@dataclass
class Poly1D:
    """1D polynomial defined by coefficient array.

    The polynomial value at point ``x`` is:
    ``sum(coefs[i] * x**i for i in range(len(coefs)))``

    Parameters
    ----------
    coefs : numpy.ndarray, optional
        1D array of polynomial coefficients, shape ``(order+1,)``.
    """

    coefs: Optional[np.ndarray] = None


@dataclass
class Poly2D:
    """2D polynomial defined by coefficient array.

    The polynomial value at point ``(x, y)`` is:
    ``sum(coefs[i,j] * x**i * y**j)``

    Parameters
    ----------
    coefs : numpy.ndarray, optional
        2D array of polynomial coefficients, shape
        ``(order_x+1, order_y+1)``.
    """

    coefs: Optional[np.ndarray] = None


@dataclass
class XYZPoly:
    """Position polynomial in three ECF components.

    Each component is a 1D polynomial in time.

    Parameters
    ----------
    x : Poly1D, optional
        X-component polynomial.
    y : Poly1D, optional
        Y-component polynomial.
    z : Poly1D, optional
        Z-component polynomial.
    """

    x: Optional[Poly1D] = None
    y: Optional[Poly1D] = None
    z: Optional[Poly1D] = None
