# -*- coding: utf-8 -*-
"""
Output Resolution Computation - Estimate ortho grid pixel size from metadata.

Computes appropriate output grid pixel spacing in degrees from the native
pixel spacing of source imagery.  Dispatches by metadata type to handle
SICD, BIOMASS, GeoTIFF, and other formats.  Used by ``OrthoPipeline`` for
automatic resolution selection.

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
2026-02-17
"""

# Standard library
from typing import Optional, Tuple, TYPE_CHECKING

# Third-party
import numpy as np

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata
    from grdl.geolocation.base import Geolocation


# Earth constant for latitude-to-meters conversion
_METERS_PER_DEG_LAT = 111_320.0


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
        return _resolution_from_sicd(metadata, geolocation, scale_factor)

    # Dict-like metadata (BIOMASS, GeoTIFF, etc.)
    if hasattr(metadata, 'get'):
        if metadata.get('range_pixel_spacing') is not None:
            return _resolution_from_biomass(metadata, geolocation, scale_factor)
        if metadata.get('transform') is not None:
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
    resolution = metadata['resolution']

    crs = str(metadata.get('crs', ''))
    if '4326' in crs or 'WGS' in crs.upper():
        # Already in degrees
        return (abs(resolution[0]) * scale_factor,
                abs(resolution[1]) * scale_factor)

    # Projected CRS -- resolution is in meters
    bounds = metadata.get('bounds')
    if bounds is not None:
        center_lat = (bounds.bottom + bounds.top) / 2.0
    else:
        center_lat = 0.0

    ground_m = max(abs(resolution[0]), abs(resolution[1]))
    return _meters_to_degrees(ground_m * scale_factor, center_lat)


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
            lat, _, _ = geolocation.image_to_latlon(center_row, center_col)
            return lat
        except (ValueError, NotImplementedError):
            pass

    return 0.0


def _meters_to_degrees(
    spacing_m: float, center_lat: float
) -> Tuple[float, float]:
    """Convert a ground spacing in meters to ``(lat_deg, lon_deg)``.

    Parameters
    ----------
    spacing_m : float
        Ground spacing in meters.
    center_lat : float
        Scene-center latitude for longitude scaling.

    Returns
    -------
    Tuple[float, float]
        ``(pixel_size_lat, pixel_size_lon)`` in degrees.
    """
    pixel_size_lat = spacing_m / _METERS_PER_DEG_LAT
    cos_lat = np.cos(np.radians(center_lat))
    if cos_lat < 0.01:
        cos_lat = 1.0  # near-pole guard
    pixel_size_lon = spacing_m / (_METERS_PER_DEG_LAT * cos_lat)
    return pixel_size_lat, pixel_size_lon
