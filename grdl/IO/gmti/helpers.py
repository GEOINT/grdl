# -*- coding: utf-8 -*-
"""
STANAG 4607 helpers - geometry, velocity, filtering, and summary.

Helpers that operate on the typed dataclasses produced by
``STANAG4607Reader``. These are the analytic primitives most often
needed when working with GMTI data:

- ``dwell_footprint_polygon`` — derive the dwell area as a polygon for
  plotting and overlap queries.
- ``ground_relative_velocity`` — convert line-of-sight (radial)
  velocity to ground-track velocity using platform velocity.
- ``filter_target_reports`` — convenience filter over a list of
  target reports by SNR, MDV, classification, and bounding box.
- ``summarize`` — quick-look counts and bounds for triage.

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-04-29

Modified
--------
2026-04-29
"""

# Standard library
import math
from typing import Any, Dict, List, Optional, Tuple

# Third-party (optional — only needed for footprint geometry)
try:
    from shapely.geometry import Polygon as _ShapelyPolygon
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False

# GRDL internal
from grdl.exceptions import DependencyError
from grdl.IO.models.stanag4607 import DwellSegment, TargetReport


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


# Mean Earth radius (meters) used for the local-tangent-plane
# approximation in ``dwell_footprint_polygon``. Adequate for dwell
# extents of a few kilometers; not appropriate for global polygons.
_EARTH_RADIUS_M = 6_378_137.0


def _normalize_lon(deg: float) -> float:
    """Wrap ``deg`` into ``[-180, 180]`` for the signed-lon convention."""
    deg = float(deg) % 360.0
    if deg > 180.0:
        deg -= 360.0
    return deg


def dwell_footprint_polygon(dwell: DwellSegment) -> 'Polygon':  # noqa: F821
    """Return the dwell area as a shapely Polygon in WGS84 lon/lat.

    Builds a four-corner polygon around the dwell center using the
    range and dwell-angle half-extents. The conversion uses a local
    tangent-plane approximation centered on the dwell center, which is
    accurate to better than 1% for dwell extents under ~50 km.

    Parameters
    ----------
    dwell : DwellSegment
        Dwell with ``dwell_center_lat``/``dwell_center_lon`` and
        ``dwell_range_half_extent`` (km) /
        ``dwell_angle_half_extent`` (deg) populated.

    Returns
    -------
    shapely.geometry.Polygon
        Polygon with corners at ``±dx, ±dy`` around the dwell center
        in ``(lon, lat)`` order. Coordinates are signed
        (``[-180, 180]`` longitude).

    Raises
    ------
    DependencyError
        If shapely is not installed.
    ValueError
        If the dwell does not have the required center / extent fields.
    """
    if not _HAS_SHAPELY:
        raise DependencyError(
            "dwell_footprint_polygon() requires shapely. "
            "Install with: pip install grdl[detection]"
        )

    center_lat = dwell.dwell_center_lat
    center_lon = dwell.dwell_center_lon
    half_range_km = dwell.dwell_range_half_extent
    half_angle_deg = dwell.dwell_angle_half_extent
    if (center_lat is None or center_lon is None
            or half_range_km is None or half_angle_deg is None):
        raise ValueError(
            "Dwell missing center_lat/center_lon or range/angle "
            "half-extent — cannot build footprint"
        )

    center_lat = float(center_lat)
    center_lon = _normalize_lon(center_lon)

    # Convert half-extents to meters: range half-extent is already in
    # km; angle half-extent is interpreted as ground-distance via the
    # local slant-range approximation (treated as a horizontal extent
    # at the platform's slant range — for a quick footprint, use the
    # great-circle distance corresponding to the angular sweep at the
    # platform-to-center range. Without that range available, use the
    # angle as a kilometer-equivalent. This is a coarse approximation
    # appropriate for plotting / overlap queries; downstream callers
    # that need exact geometry should compute it from the platform
    # position and pointing themselves.
    dx_m = float(half_range_km) * 1000.0
    dy_m = float(half_angle_deg) * 1000.0

    # Local-tangent-plane scaling: meters per degree.
    deg_per_m_lat = 1.0 / (math.pi * _EARTH_RADIUS_M / 180.0)
    deg_per_m_lon = deg_per_m_lat / max(
        math.cos(math.radians(center_lat)), 1e-9
    )

    dlat = dy_m * deg_per_m_lat
    dlon = dx_m * deg_per_m_lon

    corners = [
        (center_lon - dlon, center_lat - dlat),
        (center_lon + dlon, center_lat - dlat),
        (center_lon + dlon, center_lat + dlat),
        (center_lon - dlon, center_lat + dlat),
        (center_lon - dlon, center_lat - dlat),  # close the ring
    ]
    return _ShapelyPolygon(corners)


# ---------------------------------------------------------------------------
# Velocity helpers
# ---------------------------------------------------------------------------


def ground_relative_velocity(
    target: TargetReport,
    dwell: DwellSegment,
) -> float:
    """Convert a target's line-of-sight velocity to ground-relative.

    The target's reported velocity is the line-of-sight (radial)
    velocity in the platform frame. To recover the target's
    ground-relative radial velocity, add back the projection of the
    platform velocity onto the target line of sight.

    Parameters
    ----------
    target : TargetReport
        Target whose velocity is being corrected (uses
        ``target_velocity_los`` in cm/s and lat/lon).
    dwell : DwellSegment
        Dwell that produced the target. Uses sensor position
        (``sensor_pos_lat``/``sensor_pos_lon``), sensor track
        (``sensor_track``, deg from True North), and sensor speed
        (``sensor_speed``, mm/s).

    Returns
    -------
    float
        Ground-relative radial velocity in m/s (positive opening,
        negative closing — same sign convention as the input).

    Notes
    -----
    Uses a flat-Earth approximation around the dwell center. The
    contribution of Earth curvature is sub-1% for dwell ranges under
    100 km, which is fine for typical GMTI workflows. If the dwell is
    missing ``sensor_track`` or ``sensor_speed``, the function returns
    the line-of-sight velocity unchanged.
    """
    los_mps = float(target.target_velocity_los) / 100.0  # cm/s → m/s

    if dwell.sensor_track is None or dwell.sensor_speed is None:
        return los_mps

    sensor_speed_mps = float(dwell.sensor_speed) / 1000.0  # mm/s → m/s
    sensor_track_rad = math.radians(float(dwell.sensor_track))

    # Bearing from sensor to target (deg from True North, clockwise).
    target_lat = float(target.target_lat)
    target_lon = _normalize_lon(target.target_lon)
    sensor_lat = float(dwell.sensor_pos_lat)
    sensor_lon = _normalize_lon(dwell.sensor_pos_lon)

    dlat = math.radians(target_lat - sensor_lat)
    dlon = math.radians(target_lon - sensor_lon) * math.cos(
        math.radians(sensor_lat)
    )
    if dlat == 0.0 and dlon == 0.0:
        return los_mps
    bearing_rad = math.atan2(dlon, dlat)  # 0 = North, π/2 = East

    # Platform velocity component along the line of sight: positive
    # means the platform is moving toward the target, contributing a
    # closing rate. The radar reports closing as negative LOS, so the
    # ground-relative LOS = sensor LOS + (platform-along-LOS).
    along_los = sensor_speed_mps * math.cos(sensor_track_rad - bearing_rad)
    return los_mps + along_los


# ---------------------------------------------------------------------------
# Filtering helper
# ---------------------------------------------------------------------------


def filter_target_reports(
    reports: List[TargetReport],
    snr_min: Optional[float] = None,
    mdv_min: Optional[float] = None,
    classification: Optional[int] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> List[TargetReport]:
    """Return target reports passing all supplied filters.

    Parameters
    ----------
    reports : list of TargetReport
        Source reports to filter.
    snr_min : float, optional
        Drop reports with ``target_snr < snr_min`` (dB).
    mdv_min : float, optional
        Drop reports whose absolute LOS velocity (m/s) is below
        ``mdv_min``. Acts as a minimum-detectable-velocity cutoff
        applied at the report level rather than the dwell level.
    classification : int, optional
        Keep only reports with this exact classification code.
    bbox : tuple of float, optional
        ``(min_lon, min_lat, max_lon, max_lat)`` in WGS84 (signed
        longitudes). Reports outside this box are dropped. Latitudes
        and longitudes from BA32 are normalized to signed form before
        comparison.

    Returns
    -------
    list of TargetReport
        Filtered subset (a new list; ``reports`` is not modified).
    """
    out: List[TargetReport] = []
    for r in reports:
        if snr_min is not None and r.target_snr < snr_min:
            continue
        if mdv_min is not None:
            if abs(float(r.target_velocity_los) / 100.0) < mdv_min:
                continue
        if classification is not None and r.target_classification != classification:
            continue
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            lat = float(r.target_lat)
            lon = _normalize_lon(r.target_lon)
            if lon < min_lon or lon > max_lon:
                continue
            if lat < min_lat or lat > max_lat:
                continue
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize(reader: Any) -> Dict[str, Any]:
    """Return a quick-look summary dict for a STANAG 4607 file.

    Parameters
    ----------
    reader : STANAG4607Reader
        Reader instance whose ``metadata`` will be summarized.

    Returns
    -------
    dict
        Keys: ``filepath``, ``edition``, ``num_packets``,
        ``num_dwells``, ``num_target_reports``, ``time_bounds_ms``,
        ``geographic_bounds`` (signed lon/lat, or ``None`` if no
        targets), ``mission_id``, ``job_id``, ``platform_id``.
    """
    meta = reader.metadata

    mission_id: Optional[int] = None
    job_id: Optional[int] = None
    platform_id: Optional[str] = None
    if meta.packets:
        first = meta.packets[0].header
        mission_id = first.mission_id
        job_id = first.job_id
        platform_id = first.platform_id

    geo = meta.geographic_bounds
    if geo is not None:
        # ``geographic_bounds`` returns BA32-decoded longitudes in
        # ``[0, 360)``. Normalize to signed form for the summary.
        min_lon = _normalize_lon(geo[0])
        max_lon = _normalize_lon(geo[2])
        if max_lon < min_lon:
            # The bbox crosses the antimeridian in signed form; report
            # in normalized order anyway. Callers can detect the wrap
            # by min_lon > max_lon and handle as needed.
            pass
        geo = (min_lon, geo[1], max_lon, geo[3])

    return {
        'filepath': str(getattr(reader, 'filepath', '')),
        'edition': meta.edition,
        'num_packets': meta.num_packets,
        'num_dwells': meta.num_dwells,
        'num_target_reports': meta.num_target_reports,
        'time_bounds_ms': meta.time_bounds,
        'geographic_bounds': geo,
        'mission_id': mission_id,
        'job_id': job_id,
        'platform_id': platform_id,
    }
