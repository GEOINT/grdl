# -*- coding: utf-8 -*-
"""
Shape combination: Gaussian error propagation and set-valued operations.

Three flavours of combination, each useful for a different interpretation
of uncertainty:

- :func:`convolve_ellipses` -- independent additive Gaussian errors. For
  ``X = X_1 + ... + X_k`` with ``X_i ~ N(0, Sigma_i)``, the sum has
  covariance ``sum(Sigma_i)``. Equivalent to convolving probability
  density functions. Use for stacking independent error sources (pixel
  localisation + pointing + georegistration).

- :func:`combine_evidence` -- Fisher-information sum. For independent
  observations of the same quantity, ``Sigma^-1 = sum(Sigma_i^-1)``.
  Produces a tighter ellipse than any input. Use for combining
  multi-sensor estimates of the same target.

- :func:`minkowski_sum` -- set-valued Minkowski sum. Every point of A
  offset by every vector of B. Worst-case bound rather than
  probabilistic. Works on arbitrary shape pairs via shapely.

Set operations :func:`union_shapes` and :func:`intersect_shapes` wrap
``shapely.ops`` on densified lat/lon polygons.

All ellipse covariances are expressed in a local East-North tangent
frame at the combined shape's center. When inputs are defined at
different centers, each covariance is parallel-transported (rotated)
into the common frame; spreads beyond 50 km emit a UserWarning since
the tangent-plane approximation starts to matter at that scale.

Dependencies
------------
numpy
shapely

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
import warnings
from typing import Optional, Sequence, Tuple

# Third-party
import numpy as np
from pyproj import Geod

# GRDL internal
from grdl.shapes.base import GeographicShape
from grdl.shapes.ellipse import Ellipse
from grdl.shapes.polygon import GeoPolygon


_GEOD = Geod(ellps='WGS84')
_TANGENT_PLANE_WARN_M = 50_000.0


# ---------------------------------------------------------------------
# Ellipse combination
# ---------------------------------------------------------------------

def convolve_ellipses(
    ellipses: Sequence[Ellipse],
    center: Optional[Tuple[float, float]] = None,
    sigma_scale: float = 1.0,
) -> Ellipse:
    """Sum independent Gaussian error ellipses.

    For independent zero-mean Gaussian errors, the total covariance is
    the sum of the individual covariances. Each input covariance is
    expressed in ENU at ``center`` via parallel transport (a pure 2D
    rotation proportional to the bearing between centers).

    Parameters
    ----------
    ellipses : Sequence[Ellipse]
        Input ellipses. Each contributes a 1-sigma covariance (its
        ``sigma_scale`` is factored out before summation).
    center : Tuple[float, float], optional
        ``(lat_deg, lon_deg)`` for the combined frame. Defaults to
        the first ellipse's center.
    sigma_scale : float
        Confidence scale to apply to the output ellipse.

    Returns
    -------
    Ellipse
    """
    if len(ellipses) == 0:
        raise ValueError("convolve_ellipses requires at least one ellipse")

    center_lat, center_lon = center if center is not None else (
        ellipses[0].center_latlon
    )
    _warn_if_centers_spread(ellipses, center_lat, center_lon)

    total_cov = np.zeros((2, 2), dtype=np.float64)
    for e in ellipses:
        cov_local = _covariance_in_frame(e, center_lat, center_lon)
        total_cov = total_cov + cov_local

    return Ellipse.from_covariance(
        center_lat=center_lat,
        center_lon=center_lon,
        covariance=total_cov,
        sigma_scale=sigma_scale,
    )


def combine_evidence(
    ellipses: Sequence[Ellipse],
    center: Optional[Tuple[float, float]] = None,
    sigma_scale: float = 1.0,
) -> Ellipse:
    """Fisher-information combination of independent observations.

    ``Sigma_total^-1 = sum(Sigma_i^-1)``. Produces a tighter ellipse
    than any input. Use for combining multiple estimates of the same
    target location.

    Parameters are as in :func:`convolve_ellipses`.
    """
    if len(ellipses) == 0:
        raise ValueError("combine_evidence requires at least one ellipse")

    center_lat, center_lon = center if center is not None else (
        ellipses[0].center_latlon
    )
    _warn_if_centers_spread(ellipses, center_lat, center_lon)

    info_total = np.zeros((2, 2), dtype=np.float64)
    for e in ellipses:
        cov_local = _covariance_in_frame(e, center_lat, center_lon)
        info_total = info_total + np.linalg.inv(cov_local)

    cov_total = np.linalg.inv(info_total)
    return Ellipse.from_covariance(
        center_lat=center_lat,
        center_lon=center_lon,
        covariance=cov_total,
        sigma_scale=sigma_scale,
    )


def _covariance_in_frame(
    ellipse: Ellipse,
    center_lat: float,
    center_lon: float,
) -> np.ndarray:
    """Parallel-transport an ellipse's 1-sigma covariance into a target ENU.

    At sub-km separation the local ENU axes align to within microradians
    so rotation is negligible. At ~50 km the tangent-plane itself
    becomes a meaningful approximation (hence the warning) but the
    closed-form rotation still captures first-order coupling between the
    source and target frames.
    """
    cov = ellipse.covariance  # 1-sigma covariance at the source center
    if (
        abs(ellipse.center_latlon[0] - center_lat) < 1e-12
        and abs(ellipse.center_latlon[1] - center_lon) < 1e-12
    ):
        return cov

    # Bearing from target center to source center: the ENU axes rotate
    # by the difference in meridian convergence between the two points.
    # For tangent-plane work, the rotation angle equals the convergence
    # of longitude, approximated as (delta_lon) * sin(avg_lat).
    avg_lat = 0.5 * (center_lat + ellipse.center_latlon[0])
    delta_lon = ellipse.center_latlon[1] - center_lon
    rot_rad = np.radians(delta_lon) * np.sin(np.radians(avg_lat))
    c, s = np.cos(rot_rad), np.sin(rot_rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return R @ cov @ R.T


def _warn_if_centers_spread(
    ellipses: Sequence[Ellipse],
    center_lat: float,
    center_lon: float,
) -> None:
    max_dist = 0.0
    for e in ellipses:
        lat, lon = e.center_latlon
        _, _, d = _GEOD.inv(center_lon, center_lat, lon, lat)
        if d > max_dist:
            max_dist = float(d)
    if max_dist > _TANGENT_PLANE_WARN_M:
        warnings.warn(
            f"Combining ellipses whose centers span {max_dist:.0f} m "
            f"(> {_TANGENT_PLANE_WARN_M:.0f} m tangent-plane limit). "
            "Parallel-transport of covariance is approximate at this "
            "scale; consider moving all inputs to a common center first.",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------
# Geometric set operations (via shapely)
# ---------------------------------------------------------------------

def minkowski_sum(
    shape_a: GeographicShape,
    shape_b: GeographicShape,
    n: int = 256,
) -> GeoPolygon:
    """Minkowski sum of two geographic shapes.

    Densifies both perimeters to ``n`` samples, computes the set
    ``{a + b : a in A, b in B}`` in a local ENU frame at A's center,
    convex-hulls the result, and converts back to geodetic. For convex
    inputs this is exact up to sampling density. Non-convex inputs are
    approximated by their convex hull.
    """
    import shapely.geometry as _sg  # lazy
    from shapely.ops import unary_union  # noqa: F401 (documentation)
    from scipy.spatial import ConvexHull

    from grdl.geolocation.coordinates import (
        enu_to_geodetic,
        geodetic_to_enu,
    )

    center_lat, center_lon = shape_a.center_latlon
    ref = np.array([center_lat, center_lon, 0.0], dtype=np.float64)

    perim_a = shape_a._perimeter_latlon(n)
    perim_b = shape_b._perimeter_latlon(n)

    a_enu = geodetic_to_enu(
        np.column_stack([perim_a, np.zeros(len(perim_a))]),
        ref,
    )[:, :2]
    # Express B relative to its own center in ENU, not to A's center --
    # Minkowski sum is about the **vectors** in B, not its position.
    b_center = shape_b.center_latlon
    b_ref = np.array([b_center[0], b_center[1], 0.0], dtype=np.float64)
    b_enu = geodetic_to_enu(
        np.column_stack([perim_b, np.zeros(len(perim_b))]),
        b_ref,
    )[:, :2]

    # All pairwise sums, then convex hull to get the Minkowski envelope.
    sums = a_enu[:, None, :] + b_enu[None, :, :]
    pts = sums.reshape(-1, 2)

    if len(pts) >= 3:
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
        except Exception:
            hull_pts = pts
    else:
        hull_pts = pts

    hull3 = np.column_stack([hull_pts, np.zeros(len(hull_pts))])
    geodetic = enu_to_geodetic(hull3, ref)
    return GeoPolygon(vertices_latlon=geodetic[:, :2], edge_mode='geodesic')


def union_shapes(
    shapes: Sequence[GeographicShape],
    n: int = 256,
) -> GeoPolygon:
    """Geometric union of shapes via shapely.

    Each input is densified to ``n`` perimeter samples, promoted to a
    shapely Polygon in (lon, lat) space, and passed through
    ``unary_union``. The exterior ring of the result becomes a new
    :class:`GeoPolygon`.
    """
    import shapely.geometry as _sg
    from shapely.ops import unary_union

    polys = [_sg.Polygon(_latlon_to_lonlat(s._perimeter_latlon(n))) for s in shapes]
    merged = unary_union(polys)
    return _shapely_to_geopolygon(merged)


def intersect_shapes(
    shapes: Sequence[GeographicShape],
    n: int = 256,
) -> GeoPolygon:
    """Geometric intersection of shapes via shapely."""
    import shapely.geometry as _sg

    if len(shapes) == 0:
        raise ValueError("intersect_shapes requires at least one shape")

    polys = [_sg.Polygon(_latlon_to_lonlat(s._perimeter_latlon(n))) for s in shapes]
    result = polys[0]
    for poly in polys[1:]:
        result = result.intersection(poly)
        if result.is_empty:
            break
    return _shapely_to_geopolygon(result)


def _latlon_to_lonlat(latlon: np.ndarray) -> np.ndarray:
    """Shapely convention is (x, y) = (lon, lat)."""
    return np.column_stack([latlon[:, 1], latlon[:, 0]])


def _shapely_to_geopolygon(geom) -> GeoPolygon:
    import shapely.geometry as _sg

    if geom.is_empty:
        raise ValueError("shapely operation produced an empty geometry")
    if isinstance(geom, _sg.Polygon):
        xs, ys = geom.exterior.coords.xy
    elif isinstance(geom, _sg.MultiPolygon):
        # Return the largest component; callers that want them all should
        # iterate the input pairs manually.
        largest = max(geom.geoms, key=lambda g: g.area)
        xs, ys = largest.exterior.coords.xy
    else:
        raise TypeError(
            f"Unsupported geometry type {geom.geom_type!r} "
            "from shape combine"
        )
    latlon = np.column_stack([np.asarray(ys), np.asarray(xs)])
    return GeoPolygon(vertices_latlon=latlon, edge_mode='geodesic')
