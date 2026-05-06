# -*- coding: utf-8 -*-
"""
Geodesic arc: an open segment of a geographic circle.

An :class:`Arc` is a circular segment on the WGS-84 ellipsoid with a
bearing range ``[bearing_start_deg, bearing_end_deg]`` measured
clockwise from true north. It is explicitly open: rasterization draws
a polyline only (no fill), and the base-class ``is_closed`` flag is
``False``.

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


class Arc(GeographicShape):
    """A geodesic arc centered at ``(lat, lon)``.

    Parameters
    ----------
    center_lat, center_lon : float
        Arc center, degrees.
    radius_m : float
        Radius, metres along the ground.
    bearing_start_deg : float
        Starting bearing, degrees clockwise from true north.
    bearing_end_deg : float
        Ending bearing, degrees clockwise from true north. The arc
        sweeps clockwise from start to end. If ``end < start`` the
        sweep wraps through 360.
    """

    is_closed = False

    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        radius_m: float,
        bearing_start_deg: float,
        bearing_end_deg: float,
    ) -> None:
        if radius_m <= 0:
            raise ValueError(f"radius_m must be positive; got {radius_m}")
        self._center_lat = float(center_lat)
        self._center_lon = float(center_lon)
        self.radius_m = float(radius_m)
        self.bearing_start_deg = float(bearing_start_deg)
        self.bearing_end_deg = float(bearing_end_deg)

    # ----- GeographicShape contract -------------------------------------

    @property
    def center_latlon(self) -> Tuple[float, float]:
        return (self._center_lat, self._center_lon)

    def _perimeter_latlon(self, n: int) -> np.ndarray:
        if n < 2:
            raise ValueError(f"n must be >= 2; got {n}")
        sweep = (self.bearing_end_deg - self.bearing_start_deg) % 360.0
        if sweep == 0.0:
            sweep = 360.0  # treat equal start/end as full circle
        bearings = self.bearing_start_deg + np.linspace(
            0.0, sweep, n, endpoint=True, dtype=np.float64,
        )
        bearings = np.mod(bearings, 360.0)
        lon0 = np.full(n, self._center_lon, dtype=np.float64)
        lat0 = np.full(n, self._center_lat, dtype=np.float64)
        distances = np.full(n, self.radius_m, dtype=np.float64)
        lons, lats, _ = _GEOD.fwd(lon0, lat0, bearings, distances)
        return np.column_stack([
            np.asarray(lats, dtype=np.float64),
            np.asarray(lons, dtype=np.float64),
        ])

    def __repr__(self) -> str:
        return (
            f"Arc(center=({self._center_lat:.6f}, {self._center_lon:.6f}), "
            f"radius_m={self.radius_m:.3f}, "
            f"bearings=[{self.bearing_start_deg}, {self.bearing_end_deg}])"
        )
