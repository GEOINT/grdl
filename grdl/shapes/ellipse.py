# -*- coding: utf-8 -*-
"""
Ellipse shapes: tangent-plane parametric and true two-foci geodesic.

Two subclasses, two use cases:

- :class:`Ellipse` -- tangent-plane parametric ellipse in local ENU at
  the center. Standard convention for CEP / uncertainty ellipses.
  Closed form, fast, sub-millimetre accurate up to ~50 km radius.

- :class:`GeodesicEllipse` -- true locus of points whose geodesic-
  distance sum to two foci equals a constant. Slower (bisection per
  bearing) but exact on the WGS-84 ellipsoid at any scale.

Both subclasses expose their 2x2 covariance matrix (expressed in the
local ENU frame at ``center_latlon``) via :attr:`covariance`, which the
error-propagation routines in :mod:`grdl.shapes.combine` use directly.

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
from typing import Optional, Tuple

# Third-party
import numpy as np
from pyproj import Geod

# GRDL internal
from grdl.geolocation.coordinates import enu_to_geodetic
from grdl.shapes.base import GeographicShape


_GEOD = Geod(ellps='WGS84')


class Ellipse(GeographicShape):
    """Tangent-plane parametric ellipse -- standard CEP / uncertainty form.

    The ellipse is generated in local East-North coordinates at the
    center, then converted to geodetic via
    :func:`grdl.geolocation.coordinates.enu_to_geodetic`. Accuracy is
    sub-millimetre for radii up to ~10 km and grows quadratically as
    ``(R / R_earth)^2`` above that; for shapes larger than ~50 km prefer
    :class:`GeodesicEllipse` or build from a :class:`Circle` plus
    covariance.

    Parameters
    ----------
    center_lat, center_lon : float
        Geodetic center, degrees.
    semi_major_m : float
        Length of the major semi-axis in metres.
    semi_minor_m : float
        Length of the minor semi-axis in metres. Must be <= semi-major.
    rotation_deg : float
        Orientation of the major axis, degrees measured clockwise from
        true north (i.e., the standard navigation convention).
    sigma_scale : float
        Confidence multiplier applied on top of the stored 1-sigma
        semi-axes. Use 1.0 for 1-sigma, 2.4477 for 95% CEP, etc. When
        combining ellipses via :mod:`grdl.shapes.combine` the scale is
        removed so the underlying covariance math is unambiguous.
    """

    is_closed = True

    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        semi_major_m: float,
        semi_minor_m: float,
        rotation_deg: float = 0.0,
        sigma_scale: float = 1.0,
    ) -> None:
        if semi_major_m <= 0 or semi_minor_m <= 0:
            raise ValueError(
                f"semi-axes must be positive; got "
                f"major={semi_major_m}, minor={semi_minor_m}"
            )
        if semi_minor_m > semi_major_m:
            raise ValueError(
                "semi_minor_m must be <= semi_major_m "
                f"({semi_minor_m} > {semi_major_m})"
            )
        if sigma_scale <= 0:
            raise ValueError(f"sigma_scale must be positive; got {sigma_scale}")

        self._center_lat = float(center_lat)
        self._center_lon = float(center_lon)
        self.semi_major_m = float(semi_major_m)
        self.semi_minor_m = float(semi_minor_m)
        self.rotation_deg = float(rotation_deg)
        self.sigma_scale = float(sigma_scale)

    # ----- Factory from covariance -------------------------------------

    @classmethod
    def from_covariance(
        cls,
        center_lat: float,
        center_lon: float,
        covariance: np.ndarray,
        sigma_scale: float = 1.0,
    ) -> 'Ellipse':
        """Build an Ellipse from a 2x2 ENU covariance matrix.

        Parameters
        ----------
        covariance : array-like
            ``(2, 2)`` positive-definite matrix expressed in
            [east, north] order (metres^2).
        sigma_scale : float
            Confidence scale applied on output.

        Returns
        -------
        Ellipse
        """
        cov = np.asarray(covariance, dtype=np.float64)
        if cov.shape != (2, 2):
            raise ValueError(f"covariance must be (2, 2); got {cov.shape}")
        # Symmetrise for numerical safety
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # eigh returns ascending; flip so index 0 is the largest.
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        if np.any(eigvals <= 0):
            raise ValueError(
                f"covariance must be positive-definite; got eigenvalues "
                f"{eigvals}"
            )
        semi_major = float(np.sqrt(eigvals[0]))
        semi_minor = float(np.sqrt(eigvals[1]))
        major_vec_east, major_vec_north = eigvecs[:, 0]
        # Rotation: bearing clockwise from north of the major axis.
        bearing_rad = np.arctan2(major_vec_east, major_vec_north)
        rotation_deg = float(np.degrees(bearing_rad))
        return cls(
            center_lat=center_lat,
            center_lon=center_lon,
            semi_major_m=semi_major,
            semi_minor_m=semi_minor,
            rotation_deg=rotation_deg,
            sigma_scale=sigma_scale,
        )

    # ----- GeographicShape contract -------------------------------------

    @property
    def center_latlon(self) -> Tuple[float, float]:
        return (self._center_lat, self._center_lon)

    @property
    def covariance(self) -> np.ndarray:
        """Return the 2x2 ENU covariance matrix (sigma_scale factored out).

        Layout is ``[[var_east, cov_en], [cov_en, var_north]]`` in m^2.
        """
        a = self.semi_major_m
        b = self.semi_minor_m
        theta = np.radians(self.rotation_deg)
        # Rotation from (major, minor) -> (east, north) basis.
        # Major axis bearing `rotation_deg` clockwise from north => unit vec
        # (east, north) = (sin theta, cos theta).
        major_east = np.sin(theta)
        major_north = np.cos(theta)
        minor_east = np.cos(theta)
        minor_north = -np.sin(theta)
        R = np.array([
            [major_east, minor_east],
            [major_north, minor_north],
        ])
        D = np.diag([a * a, b * b])
        return R @ D @ R.T

    def _perimeter_latlon(self, n: int) -> np.ndarray:
        if n < 3:
            raise ValueError(f"n must be >= 3; got {n}")
        # Parameter theta sweeps 0..2pi around the ellipse.
        t = np.linspace(0.0, 2 * np.pi, n, endpoint=False, dtype=np.float64)
        rot = np.radians(self.rotation_deg)
        a = self.semi_major_m * self.sigma_scale
        b = self.semi_minor_m * self.sigma_scale
        # Ellipse in (major, minor) basis then rotate to (east, north).
        major = a * np.cos(t)
        minor = b * np.sin(t)
        # Major axis bearing = rotation_deg clockwise from north
        # so unit vector along major = (sin rot, cos rot) in (E, N).
        east = major * np.sin(rot) + minor * np.cos(rot)
        north = major * np.cos(rot) - minor * np.sin(rot)
        enu = np.column_stack([east, north, np.zeros_like(east)])
        ref = np.array(
            [self._center_lat, self._center_lon, 0.0], dtype=np.float64,
        )
        geodetic = enu_to_geodetic(enu, ref)
        return geodetic[:, :2]

    def __repr__(self) -> str:
        return (
            f"Ellipse(center=({self._center_lat:.6f}, "
            f"{self._center_lon:.6f}), "
            f"semi_major={self.semi_major_m:.3f}, "
            f"semi_minor={self.semi_minor_m:.3f}, "
            f"rotation_deg={self.rotation_deg:.3f}, "
            f"sigma_scale={self.sigma_scale:.3f})"
        )


class GeodesicEllipse(GeographicShape):
    """True two-foci geodesic ellipse on the WGS-84 ellipsoid.

    Defines the locus of points ``P`` where
    ``geod.inv(F1, P) + geod.inv(F2, P) == sum_distance_m``. Per-bearing
    bisection finds the radial distance that satisfies the constraint.
    Slower than :class:`Ellipse` but exact at any scale and geographic
    location.

    Parameters
    ----------
    focus1_lat, focus1_lon : float
        First focus, degrees.
    focus2_lat, focus2_lon : float
        Second focus, degrees.
    sum_distance_m : float
        Constant foci-distance sum, metres. Must exceed the geodesic
        distance between the foci (otherwise the locus is empty).
    """

    is_closed = True

    def __init__(
        self,
        focus1_lat: float,
        focus1_lon: float,
        focus2_lat: float,
        focus2_lon: float,
        sum_distance_m: float,
    ) -> None:
        focal_dist = _GEOD.inv(
            focus1_lon, focus1_lat, focus2_lon, focus2_lat,
        )[2]
        if sum_distance_m <= focal_dist:
            raise ValueError(
                "sum_distance_m must exceed inter-focal geodesic distance "
                f"({sum_distance_m} <= {focal_dist})"
            )
        self._f1 = (float(focus1_lat), float(focus1_lon))
        self._f2 = (float(focus2_lat), float(focus2_lon))
        self.sum_distance_m = float(sum_distance_m)
        self._focal_dist = float(focal_dist)

    # ----- GeographicShape contract -------------------------------------

    @property
    def center_latlon(self) -> Tuple[float, float]:
        # Geodesic midpoint of the two foci.
        lat1, lon1 = self._f1
        lat2, lon2 = self._f2
        fwd_az, _, dist = _GEOD.inv(lon1, lat1, lon2, lat2)
        mid_lon, mid_lat, _ = _GEOD.fwd(lon1, lat1, fwd_az, dist * 0.5)
        return (float(mid_lat), float(mid_lon))

    def _perimeter_latlon(self, n: int) -> np.ndarray:
        if n < 3:
            raise ValueError(f"n must be >= 3; got {n}")
        center_lat, center_lon = self.center_latlon

        bearings = np.linspace(
            0.0, 360.0, n, endpoint=False, dtype=np.float64,
        )

        # Upper bound on radial distance: (sum + focal)/2 is the
        # semi-major axis of the foci-sum ellipse.
        r_hi_default = 0.5 * (self.sum_distance_m + self._focal_dist) * 1.01
        r_hi_default = max(r_hi_default, 1.0)

        lats = np.empty(n, dtype=np.float64)
        lons = np.empty(n, dtype=np.float64)

        for i, bearing in enumerate(bearings):
            r = _bisect_radial_distance(
                center_lat=center_lat,
                center_lon=center_lon,
                bearing_deg=float(bearing),
                focus1=self._f1,
                focus2=self._f2,
                target_sum=self.sum_distance_m,
                r_hi=r_hi_default,
            )
            lon, lat, _ = _GEOD.fwd(center_lon, center_lat, bearing, r)
            lats[i] = lat
            lons[i] = lon

        return np.column_stack([lats, lons])

    def __repr__(self) -> str:
        return (
            f"GeodesicEllipse(focus1={self._f1}, focus2={self._f2}, "
            f"sum_distance_m={self.sum_distance_m:.3f})"
        )


def _bisect_radial_distance(
    center_lat: float,
    center_lon: float,
    bearing_deg: float,
    focus1: Tuple[float, float],
    focus2: Tuple[float, float],
    target_sum: float,
    r_hi: float,
    tol_m: float = 1e-3,
    max_iter: int = 64,
) -> float:
    """Bisection on radial distance to satisfy the two-foci sum constraint.

    Returns the distance ``r`` from center along ``bearing_deg`` such
    that ``d(f1, P) + d(f2, P) == target_sum`` within ``tol_m``.
    """
    f1_lat, f1_lon = focus1
    f2_lat, f2_lon = focus2

    def _sum_at(r: float) -> float:
        lon, lat, _ = _GEOD.fwd(center_lon, center_lat, bearing_deg, r)
        _, _, d1 = _GEOD.inv(f1_lon, f1_lat, lon, lat)
        _, _, d2 = _GEOD.inv(f2_lon, f2_lat, lon, lat)
        return d1 + d2

    lo = 0.0
    hi = r_hi
    sum_lo = _sum_at(lo)
    sum_hi = _sum_at(hi)

    # If the bracket doesn't straddle, grow hi (up to 2x a few times).
    for _ in range(8):
        if sum_lo <= target_sum <= sum_hi:
            break
        hi *= 2.0
        sum_hi = _sum_at(hi)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = _sum_at(mid)
        if abs(val - target_sum) < tol_m:
            return mid
        if val < target_sum:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
