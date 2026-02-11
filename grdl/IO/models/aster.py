# -*- coding: utf-8 -*-
"""
ASTER Metadata - Typed metadata for ASTER satellite products.

Provides the ASTERMetadata dataclass for ASTER L1T (Registered Radiance
at Sensor) and ASTGTM (Global Digital Elevation Model) products. ASTER
is a joint NASA/METI instrument on the Terra satellite with VNIR, SWIR,
and TIR subsystems.

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
from typing import Dict, Optional, Tuple

# GRDL internal
from grdl.IO.models.base import ImageMetadata


@dataclass
class ASTERMetadata(ImageMetadata):
    """Typed metadata for ASTER (Advanced Spaceborne Thermal Emission
    and Reflection Radiometer) products.

    Extends ``ImageMetadata`` with ASTER-specific fields extracted from
    GeoTIFF tags and companion XML metadata. All sensor-specific fields
    default to ``None`` for forward compatibility.

    Attributes
    ----------
    acquisition_date : str, optional
        Acquisition date (``YYYY/MM/DD``).
    acquisition_time : str, optional
        Acquisition time (``HH:MM:SS``).
    entity_id : str, optional
        Unique ASTER scene identifier.
    local_granule_id : str, optional
        Filename / granule ID string.
    processing_level : str, optional
        Product level (``'L1T'``, ``'GDEM'``).
    orbit_direction : str, optional
        Orbit direction (``'AS'`` ascending, ``'DE'`` descending).
    wrs_path : int, optional
        Worldwide Reference System path (1--233).
    wrs_row : int, optional
        Worldwide Reference System row (1--244).
    scene_center_lat : float, optional
        Scene center latitude (decimal degrees).
    scene_center_lon : float, optional
        Scene center longitude (decimal degrees).
    corner_coords : dict, optional
        Scene corner coordinates as
        ``{'UL': (lat, lon), 'UR': ..., 'LR': ..., 'LL': ...}``.
    sun_azimuth : float, optional
        Solar azimuth angle at scene center (degrees, CW from north).
    sun_elevation : float, optional
        Solar elevation angle at scene center (degrees above horizon).
    cloud_cover : float, optional
        Percentage of scene obscured by clouds (0--100).
    vnir_available : bool, optional
        VNIR subsystem data present (bands 1--3, 15 m).
    swir_available : bool, optional
        SWIR subsystem data present (bands 4--9, 30 m).
        SWIR detector failed April 2008.
    tir_available : bool, optional
        TIR subsystem data present (bands 10--14, 90 m).
    correction_level : str, optional
        Terrain correction level for L1T products.
    """

    # Acquisition
    acquisition_date: Optional[str] = None
    acquisition_time: Optional[str] = None
    entity_id: Optional[str] = None
    local_granule_id: Optional[str] = None
    processing_level: Optional[str] = None

    # Orbital
    orbit_direction: Optional[str] = None
    wrs_path: Optional[int] = None
    wrs_row: Optional[int] = None

    # Geolocation
    scene_center_lat: Optional[float] = None
    scene_center_lon: Optional[float] = None
    corner_coords: Optional[Dict[str, Tuple[float, float]]] = None

    # Solar geometry
    sun_azimuth: Optional[float] = None
    sun_elevation: Optional[float] = None

    # Data quality
    cloud_cover: Optional[float] = None

    # Band availability
    vnir_available: Optional[bool] = None
    swir_available: Optional[bool] = None
    tir_available: Optional[bool] = None

    # Terrain correction (L1T)
    correction_level: Optional[str] = None
