# -*- coding: utf-8 -*-
"""
BIOMASS Metadata - Typed metadata for ESA BIOMASS P-band SAR data.

Flat dataclass for BIOMASS L1 SCS products. Fields are extracted from
the BIOMASS XML annotation and measurement GeoTIFF files.

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
from typing import Any, Dict, List, Optional, Tuple

# GRDL internal
from grdl.IO.models.base import ImageMetadata


@dataclass
class BIOMASSMetadata(ImageMetadata):
    """Complete typed metadata for BIOMASS L1 SCS imagery.

    Contains all fields extracted from the BIOMASS annotation XML
    and measurement GeoTIFF files. Inherits from ``ImageMetadata``
    for universal fields and dict-like access.

    Parameters
    ----------
    mission : str, optional
        Mission name (``'BIOMASS'``).
    swath : str, optional
        Swath identifier.
    product_type : str, optional
        Product type (``'SCS'``).
    start_time : str, optional
        Acquisition start time (ISO 8601).
    stop_time : str, optional
        Acquisition stop time (ISO 8601).
    orbit_number : int, optional
        Absolute orbit number.
    orbit_pass : str, optional
        Orbit direction (``'ASCENDING'`` or ``'DESCENDING'``).
    polarizations : List[str], optional
        Available polarizations (e.g., ``['HH', 'HV', 'VH', 'VV']``).
    num_polarizations : int, optional
        Number of polarization channels.
    range_pixel_spacing : float, optional
        Range pixel spacing (meters).
    azimuth_pixel_spacing : float, optional
        Azimuth pixel spacing (meters).
    pixel_type : str, optional
        Pixel data type string (e.g., ``'32 bit Float'``).
    pixel_representation : str, optional
        Pixel representation (e.g., ``'Abs Phase'``).
    projection : str, optional
        Data projection (e.g., ``'Slant Range'``).
    nodata_value : float, optional
        No-data sentinel value.
    corner_coords : Dict[str, Tuple[float, float]], optional
        Corner coordinates as ``{name: (lat, lon)}`` pairs.
    prf : float, optional
        Pulse Repetition Frequency (Hz).
    gcps : List[Tuple[float, float, float, float, float]], optional
        Ground Control Points as
        ``[(x, y, z, row, col), ...]``.

    Examples
    --------
    >>> meta = reader.metadata  # BIOMASSMetadata
    >>> meta.mission
    'BIOMASS'
    >>> meta.polarizations
    ['HH', 'HV', 'VH', 'VV']
    >>> meta.range_pixel_spacing
    6.25
    """

    mission: Optional[str] = None
    swath: Optional[str] = None
    product_type: Optional[str] = None
    start_time: Optional[str] = None
    stop_time: Optional[str] = None
    orbit_number: Optional[int] = None
    orbit_pass: Optional[str] = None
    polarizations: Optional[List[str]] = None
    num_polarizations: Optional[int] = None
    range_pixel_spacing: Optional[float] = None
    azimuth_pixel_spacing: Optional[float] = None
    pixel_type: Optional[str] = None
    pixel_representation: Optional[str] = None
    projection: Optional[str] = None
    nodata_value: Optional[float] = None
    corner_coords: Optional[Dict[str, Tuple[float, float]]] = None
    prf: Optional[float] = None
    gcps: Optional[List[Tuple[float, float, float, float, float]]] = None
