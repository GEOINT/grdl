# -*- coding: utf-8 -*-
"""
Circle shape defined on the WGS-84 ellipsoid.

A geographic circle centered at ``(lat, lon)`` with radius expressed in
metres along the ground. Perimeter generation uses Karney's geodesic
algorithm via :class:`pyproj.Geod`, giving numerical error on the order
of 10^-9 m (1 nanometer) -- far below any DEM or sensor-model floor.

``contains`` is exact: the geodesic distance from center to probe point
is compared directly to the radius.

Dependencies
------------
numpy
pyproj

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-04-18

Modified
--------
2026-04-18
"""

# Standard library
from typing import Tuple

# Third-party
import numpy as np
from pyproj import Geod

# GRDL internal
from grdl.shapes.base import GeographicShape


_GEOD = Geod(ellps='WGS84')


class Circle(GeographicShape):
    """A geographic circle on the WGS-84 ellipsoid.

    Parameters
    ----------
    center_lat : float
        Geodetic latitude of the center, degrees.
    center_lon : float
        Geodetic longitude of the center, degrees.
    radius_m : float
        Circle radius, metres along the ground (geodesic distance).

    Examples
    --------
    >>> c = Circle(34.05, -118.19, radius_m=500.0)
    >>> perim = c.perimeter_latlon(n=128)  # (128, 3) [lat, lon, hae]
    >>> mask = c.rasterize(geo, image_shape=(4096, 4096))
    """

    is_closed = True

    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        radius_m: float,
    ) -> None:
        if radius_m <= 0:
            raise ValueError(f"radius_m must be positive; got {radius_m}")
        if not (-90.0 <= center_lat <= 90.0):
            raise ValueError(
                f"center_lat must be in [-90, 90]; got {center_lat}"
            )
        self._center_lat = float(center_lat)
        self._center_lon = float(center_lon)
        self.radius_m = float(radius_m)

    # ----- GeographicShape contract -------------------------------------

    @property
    def center_latlon(self) -> Tuple[float, float]:
        return (self._center_lat, self._center_lon)

    def _perimeter_latlon(self, n: int) -> np.ndarray:
        if n < 3:
            raise ValueError(f"n must be >= 3; got {n}")
        bearings = np.linspace(0.0, 360.0, n, endpoint=False, dtype=np.float64)
        lon0 = np.full(n, self._center_lon, dtype=np.float64)
        lat0 = np.full(n, self._center_lat, dtype=np.float64)
        distances = np.full(n, self.radius_m, dtype=np.float64)
        # Geod.fwd returns (lon, lat, back_azimuth); all Karney-accurate.
        lons, lats, _ = _GEOD.fwd(lon0, lat0, bearings, distances)
        return np.column_stack([
            np.asarray(lats, dtype=np.float64),
            np.asarray(lons, dtype=np.float64),
        ])

    # ----- Exact containment override -----------------------------------

    def contains(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        n: int = 512,  # kept for ABC signature parity, unused here
    ) -> np.ndarray:
        """Exact geodesic-distance containment test.

        Ignores the ``n`` parameter: a true circle on the ellipsoid is
        tested analytically by comparing geodesic distance to radius.
        """
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        if lat.shape != lon.shape:
            lat, lon = np.broadcast_arrays(lat, lon)
        lat_flat = lat.ravel()
        lon_flat = lon.ravel()
        lat0 = np.full_like(lat_flat, self._center_lat)
        lon0 = np.full_like(lon_flat, self._center_lon)
        _, _, dist = _GEOD.inv(lon0, lat0, lon_flat, lat_flat)
        inside = np.asarray(dist, dtype=np.float64) <= self.radius_m
        return inside.reshape(lat.shape)

    # ----- Pretty repr --------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Circle(center=({self._center_lat:.6f}, {self._center_lon:.6f}), "
            f"radius_m={self.radius_m:.3f})"
        )
