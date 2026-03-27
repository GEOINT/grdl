# -*- coding: utf-8 -*-
"""
Output Resolution Computation - Estimate ortho grid pixel size from metadata.

Computes appropriate output grid pixel spacing in degrees from the native
pixel spacing of source imagery.  Dispatches by metadata type to handle
SICD, BIOMASS, Sentinel-1 SLC, NISAR, GeoTIFF, and other formats.  Used
by ``OrthoBuilder`` for automatic resolution selection.

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
2026-02-17

Modified
--------
2026-03-08
"""

# Standard library
import logging
from typing import Optional, Tuple, TYPE_CHECKING

# Third-party
import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata
    from grdl.geolocation.base import Geolocation


from grdl.geolocation.coordinates import meters_per_degree


def compute_output_resolution(
    metadata: 'ImageMetadata',
    geolocation: Optional['Geolocation'] = None,
    scale_factor: float = 1.0,
) -> Tuple[float, float]:
    """Compute output grid pixel spacing from imagery metadata.

    Dispatches to format-specific resolution computation based on the
    metadata type.  Returns pixel sizes in degrees suitable for
    ``OutputGrid`` construction.

    Parameters
    ----------
    metadata : ImageMetadata
        Source imagery metadata (``SICDMetadata``, ``BIOMASSMetadata``,
        dict-like GeoTIFF metadata, etc.).
    geolocation : Geolocation, optional
        Source geolocation for scene-center latitude lookup.  Required
        for SAR metadata where pixel spacings are in meters.
    scale_factor : float, default=1.0
        Multiplier for the computed resolution.  Values > 1.0 produce
        coarser output; values < 1.0 produce finer output.

    Returns
    -------
    Tuple[float, float]
        ``(pixel_size_lat, pixel_size_lon)`` in degrees.

    Raises
    ------
    ValueError
        If resolution cannot be determined from the metadata.
    """
    from grdl.IO.models.sicd import SICDMetadata

    if isinstance(metadata, SICDMetadata):
        logger.debug("Matched metadata type: SICDMetadata")
        return _resolution_from_sicd(metadata, geolocation, scale_factor)

    # Sentinel-1 SLC (typed dataclass, not dict-like)
    try:
        from grdl.IO.models.sentinel1_slc import Sentinel1SLCMetadata
        if isinstance(metadata, Sentinel1SLCMetadata):
            logger.debug("Matched metadata type: Sentinel1SLCMetadata")
            return _resolution_from_sentinel1_slc(
                metadata, geolocation, scale_factor,
            )
    except ImportError:
        pass

    # NISAR (typed dataclass)
    try:
        from grdl.IO.models.nisar import NISARMetadata
        if isinstance(metadata, NISARMetadata):
            logger.debug("Matched metadata type: NISARMetadata")
            return _resolution_from_nisar(
                metadata, geolocation, scale_factor,
            )
    except ImportError:
        pass

    # Dict-like metadata (BIOMASS, GeoTIFF, etc.)
    if hasattr(metadata, 'get'):
        if metadata.get('range_pixel_spacing') is not None:
            logger.debug("Matched metadata type: BIOMASS (dict-like)")
            return _resolution_from_biomass(metadata, geolocation, scale_factor)
        if getattr(metadata, 'transform', None) is not None:
            logger.debug("Matched metadata type: GeoTIFF/geocoded raster")
            return _resolution_from_geotiff(metadata, scale_factor)

    raise ValueError(
        f"Cannot determine output resolution from metadata type "
        f"{type(metadata).__name__}.  Provide pixel_size_lat and "
        f"pixel_size_lon explicitly."
    )


# ------------------------------------------------------------------
# Per-format strategies
# ------------------------------------------------------------------

def _resolution_from_sicd(
    metadata: 'ImageMetadata',
    geolocation: Optional['Geolocation'],
    scale_factor: float,
) -> Tuple[float, float]:
    """Compute resolution from SICD metadata.

    Uses ``grid.row.ss`` and ``grid.col.ss`` (sample spacings in meters).
    Prefers impulse response width (``imp_resp_wid``) when available
    because it represents actual resolution rather than (potentially
    oversampled) sample spacing.

    For SLANT-plane imagery the range-direction ground resolution is
    ``slant_spacing / sin(graze_angle)``.  The graze angle is taken
    from ``scpcoa.graze_ang``.

    Parameters
    ----------
    metadata : SICDMetadata
        Typed SICD metadata with ``grid``, ``scpcoa``, and ``geo_data``.
    geolocation : Geolocation, optional
        For scene-center latitude lookup.
    scale_factor : float
        Resolution multiplier.

    Returns
    -------
    Tuple[float, float]
        ``(pixel_size_lat, pixel_size_lon)`` in degrees.
    """
    if (metadata.grid is None or metadata.grid.row is None
            or metadata.grid.col is None):
        raise ValueError(
            "SICD metadata missing grid.row or grid.col -- "
            "cannot compute pixel spacing."
        )

    # Prefer impulse response width (actual resolution) over sample spacing
    row_m = metadata.grid.row.imp_resp_wid or metadata.grid.row.ss
    col_m = metadata.grid.col.imp_resp_wid or metadata.grid.col.ss

    if row_m is None or col_m is None:
        raise ValueError(
            "SICD metadata has no row/col sample spacing or impulse "
            "response width."
        )

    # For SLANT-plane data, project range direction to ground
    if (metadata.grid.image_plane == 'SLANT'
            and metadata.scpcoa is not None
            and metadata.scpcoa.graze_ang is not None):
        sin_graze = np.sin(np.radians(metadata.scpcoa.graze_ang))
        if sin_graze > 0.01:
            logger.debug(
                "SLANT-plane graze angle correction: %.2f deg",
                metadata.scpcoa.graze_ang,
            )
            row_m = row_m / sin_graze

    ground_m = max(row_m, col_m)
    center_lat = _get_center_latitude(metadata, geolocation)

    return _meters_to_degrees(ground_m * scale_factor, center_lat)


def _resolution_from_biomass(
    metadata: 'ImageMetadata',
    geolocation: Optional['Geolocation'],
    scale_factor: float,
) -> Tuple[float, float]:
    """Compute resolution from BIOMASS metadata.

    Uses ``range_pixel_spacing`` and ``azimuth_pixel_spacing`` keys
    from the dict-like BIOMASS metadata.

    Parameters
    ----------
    metadata : dict-like
        BIOMASS metadata with pixel spacing keys.
    geolocation : Geolocation, optional
        For scene-center latitude lookup.
    scale_factor : float
        Resolution multiplier.

    Returns
    -------
    Tuple[float, float]
        ``(pixel_size_lat, pixel_size_lon)`` in degrees.
    """
    range_m = metadata.get('range_pixel_spacing', 0)
    azimuth_m = metadata.get('azimuth_pixel_spacing', 0)

    if range_m <= 0 or azimuth_m <= 0:
        raise ValueError(
            "BIOMASS metadata missing range_pixel_spacing or "
            "azimuth_pixel_spacing."
        )

    ground_m = max(range_m, azimuth_m)
    center_lat = _get_center_latitude(metadata, geolocation)

    return _meters_to_degrees(ground_m * scale_factor, center_lat)


def _resolution_from_geotiff(
    metadata: 'ImageMetadata',
    scale_factor: float,
) -> Tuple[float, float]:
    """Compute resolution from GeoTIFF metadata.

    GeoTIFF resolution may already be in degrees (geographic CRS) or
    meters (projected CRS).

    Parameters
    ----------
    metadata : dict-like
        GeoTIFF metadata with ``resolution`` and optionally ``crs``.
    scale_factor : float
        Resolution multiplier.

    Returns
    -------
    Tuple[float, float]
        ``(pixel_size_lat, pixel_size_lon)`` in degrees.
    """
    resolution = metadata.pixel_resolution

    crs = str(getattr(metadata, 'crs', '') or '')
    if '4326' in crs or 'WGS' in crs.upper():
        # Already in degrees
        return (abs(resolution[0]) * scale_factor,
                abs(resolution[1]) * scale_factor)

    # Projected CRS -- resolution is in meters
    bounds = metadata.bounds
    if bounds is not None:
        center_lat = (bounds[1] + bounds[3]) / 2.0  # (minx, miny, maxx, maxy)
    else:
        center_lat = 0.0

    ground_m = max(abs(resolution[0]), abs(resolution[1]))
    return _meters_to_degrees(ground_m * scale_factor, center_lat)


def _resolution_from_sentinel1_slc(
    metadata: 'ImageMetadata',
    geolocation: Optional['Geolocation'],
    scale_factor: float,
) -> Tuple[float, float]:
    """Compute resolution from Sentinel-1 SLC metadata.

    Uses ``swath_info.range_pixel_spacing`` (slant range) and
    ``swath_info.azimuth_pixel_spacing``.  Projects range to ground
    using ``incidence_angle_mid``.

    Parameters
    ----------
    metadata : Sentinel1SLCMetadata
        Typed Sentinel-1 SLC metadata.
    geolocation : Geolocation, optional
        For scene-center latitude lookup.
    scale_factor : float
        Resolution multiplier.

    Returns
    -------
    Tuple[float, float]
        ``(pixel_size_lat, pixel_size_lon)`` in degrees.
    """
    si = getattr(metadata, 'swath_info', None)
    if si is None:
        raise ValueError(
            "Sentinel-1 SLC metadata missing swath_info."
        )
    range_m = getattr(si, 'range_pixel_spacing', None)
    azimuth_m = getattr(si, 'azimuth_pixel_spacing', None)
    if range_m is None or azimuth_m is None:
        raise ValueError(
            "Sentinel-1 SLC metadata missing range_pixel_spacing or "
            "azimuth_pixel_spacing in swath_info."
        )

    # Project slant range to ground range using mid-swath incidence angle
    inc = getattr(si, 'incidence_angle_mid', None)
    if inc is not None and inc > 0.1:
        sin_inc = np.sin(np.radians(inc))
        if sin_inc > 0.01:
            logger.debug(
                "SLANT-plane incidence angle correction: %.2f deg", inc,
            )
            range_m = range_m / sin_inc

    ground_m = max(range_m, azimuth_m)
    center_lat = _get_center_latitude(metadata, geolocation)
    return _meters_to_degrees(ground_m * scale_factor, center_lat)


def _resolution_from_nisar(
    metadata: 'ImageMetadata',
    geolocation: Optional['Geolocation'],
    scale_factor: float,
) -> Tuple[float, float]:
    """Compute resolution from NISAR metadata.

    For RSLC products, uses ``swath_parameters.scene_center_ground_range_spacing``
    (already ground-projected) and ``scene_center_along_track_spacing``.

    For GSLC products, uses ``grid_parameters.x_coordinate_spacing`` and
    ``y_coordinate_spacing``.  If the CRS is geographic (EPSG:4326) the
    spacings are already in degrees; otherwise they are in meters.

    Parameters
    ----------
    metadata : NISARMetadata
        Typed NISAR metadata.
    geolocation : Geolocation, optional
        For scene-center latitude lookup.
    scale_factor : float
        Resolution multiplier.

    Returns
    -------
    Tuple[float, float]
        ``(pixel_size_lat, pixel_size_lon)`` in degrees.
    """
    product_type = getattr(metadata, 'product_type', None)

    # RSLC: scene-center ground spacings (meters)
    sp = getattr(metadata, 'swath_parameters', None)
    if product_type == 'RSLC' and sp is not None:
        gr = getattr(sp, 'scene_center_ground_range_spacing', None)
        az = getattr(sp, 'scene_center_along_track_spacing', None)
        if gr is not None and az is not None and gr > 0 and az > 0:
            ground_m = max(gr, az)
            center_lat = _get_center_latitude(metadata, geolocation)
            return _meters_to_degrees(ground_m * scale_factor, center_lat)

    # GSLC: coordinate spacings (may be meters or degrees)
    gp = getattr(metadata, 'grid_parameters', None)
    if gp is not None:
        dx = getattr(gp, 'x_coordinate_spacing', None)
        dy = getattr(gp, 'y_coordinate_spacing', None)
        if dx is not None and dy is not None:
            epsg = getattr(gp, 'epsg', None)
            if epsg == 4326:
                # Already in degrees
                return (abs(dy) * scale_factor, abs(dx) * scale_factor)
            # Projected CRS: spacings are in meters
            ground_m = max(abs(dx), abs(dy))
            center_lat = _get_center_latitude(metadata, geolocation)
            return _meters_to_degrees(ground_m * scale_factor, center_lat)

    raise ValueError(
        "NISAR metadata missing resolution parameters "
        "(swath_parameters or grid_parameters)."
    )


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _get_center_latitude(
    metadata: 'ImageMetadata',
    geolocation: Optional['Geolocation'],
) -> float:
    """Get scene-center latitude for meters-to-degrees conversion.

    Tries SICD SCP, then geolocation image center, then falls back to
    the equator (0.0).
    """
    from grdl.IO.models.sicd import SICDMetadata

    if isinstance(metadata, SICDMetadata):
        if (metadata.geo_data is not None
                and metadata.geo_data.scp is not None
                and metadata.geo_data.scp.llh is not None):
            return metadata.geo_data.scp.llh.lat

    if geolocation is not None:
        try:
            center_row = geolocation.shape[0] // 2
            center_col = geolocation.shape[1] // 2
            lat = geolocation.image_to_latlon(center_row, center_col)[0]
            return lat
        except (ValueError, NotImplementedError):
            pass

    return 0.0


def _meters_to_degrees(
    spacing_m: float, center_lat: float
) -> Tuple[float, float]:
    """Convert a ground spacing in meters to ``(lat_deg, lon_deg)``.

    Uses WGS-84 ellipsoidal radii of curvature for accurate conversion
    at all latitudes.

    Parameters
    ----------
    spacing_m : float
        Ground spacing in meters.
    center_lat : float
        Scene-center latitude for latitude/longitude scaling.

    Returns
    -------
    Tuple[float, float]
        ``(pixel_size_lat, pixel_size_lon)`` in degrees.
    """
    m_lat, m_lon = meters_per_degree(center_lat)
    pixel_size_lat = spacing_m / m_lat
    if m_lon < m_lat * 0.01:
        m_lon = m_lat  # near-pole guard
    pixel_size_lon = spacing_m / m_lon
    return pixel_size_lat, pixel_size_lon
