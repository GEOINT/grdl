# -*- coding: utf-8 -*-
"""
Sentinel-2 Metadata - Typed metadata for Sentinel-2 MSI products.

Provides the Sentinel2Metadata dataclass for Sentinel-2A/2B/2C multispectral
products. Fields are extracted from filename parsing, GeoTIFF tags, and
optionally from SAFE archive XML metadata.

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
2026-02-11

Modified
--------
2026-02-11
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from grdl.IO.models.base import ImageMetadata


@dataclass
class Sentinel2Metadata(ImageMetadata):
    """Typed metadata for Sentinel-2 MSI products.

    Extends ImageMetadata with Sentinel-2-specific fields extracted from
    filename parsing, JPEG2000 tags, and SAFE archive metadata. Supports
    both Level-1C (TOA) and Level-2A (Surface Reflectance) products.

    Attributes
    ----------
    satellite : str, optional
        Satellite identifier ('S2A', 'S2B', 'S2C').
    processing_level : str, optional
        Processing level ('L1C', 'L2A').
    product_type : str, optional
        Product type identifier ('MSIL1C', 'MSIL2A').
    sensing_datetime : str, optional
        Sensing start time (ISO 8601: YYYY-MM-DDTHH:MM:SS).
    mgrs_tile_id : str, optional
        MGRS tile identifier (e.g., 'T10SEG').
    band_id : str, optional
        Band identifier ('B01'-'B12', 'B8A').
    resolution_tier : int, optional
        Spatial resolution in meters (10, 20, 60).
    baseline_processing : str, optional
        Processing baseline version (e.g., 'N0510').
    relative_orbit : int, optional
        Relative orbit number (0-143).
    product_discriminator : str, optional
        Product generation timestamp.
    utm_zone : int, optional
        UTM zone number (1-60).
    latitude_band : str, optional
        MGRS latitude band letter.
    wavelength_center : float, optional
        Central wavelength in nanometers.
    wavelength_range : Tuple[float, float], optional
        (min, max) wavelength in nanometers.
    orbit_direction : str, optional
        Orbit direction ('ASCENDING', 'DESCENDING').

    Examples
    --------
    >>> meta = reader.metadata  # Sentinel2Metadata
    >>> meta.satellite
    'S2A'
    >>> meta.processing_level
    'L2A'
    >>> meta.mgrs_tile_id
    'T10SEG'
    >>> meta.band_id
    'B04'
    >>> meta.resolution_tier
    10
    """

    # Mission identifiers
    satellite: Optional[str] = None
    processing_level: Optional[str] = None
    product_type: Optional[str] = None
    baseline_processing: Optional[str] = None

    # Temporal
    sensing_datetime: Optional[str] = None
    product_discriminator: Optional[str] = None

    # Spatial (MGRS tiling)
    mgrs_tile_id: Optional[str] = None
    utm_zone: Optional[int] = None
    latitude_band: Optional[str] = None

    # Band/spectral
    band_id: Optional[str] = None
    resolution_tier: Optional[int] = None
    wavelength_center: Optional[float] = None
    wavelength_range: Optional[Tuple[float, float]] = None

    # Orbital
    relative_orbit: Optional[int] = None
    orbit_direction: Optional[str] = None
