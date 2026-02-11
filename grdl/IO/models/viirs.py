# -*- coding: utf-8 -*-
"""
VIIRS Metadata - Typed metadata for VIIRS satellite products.

Provides the VIIRSMetadata dataclass for VIIRS products from Suomi NPP,
NOAA-20, and NOAA-21. Covers nighttime lights (VNP46A1), vegetation
indices (VNP13A1), surface reflectance, and other gridded/swath products
stored in HDF5 format.

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
from typing import Dict, List, Optional, Tuple

# GRDL internal
from grdl.IO.models.base import ImageMetadata


@dataclass
class VIIRSMetadata(ImageMetadata):
    """Typed metadata for VIIRS (Visible Infrared Imaging Radiometer Suite).

    Extends ``ImageMetadata`` with VIIRS-specific fields extracted from
    HDF5 file-level and dataset-level attributes. All sensor-specific
    fields default to ``None`` for forward compatibility.

    Attributes
    ----------
    satellite_name : str, optional
        Satellite platform (``'Suomi NPP'``, ``'NOAA-20'``, ``'NOAA-21'``).
    instrument_name : str, optional
        Instrument name (always ``'VIIRS'``).
    processing_level : str, optional
        Product processing level (``'L1'``, ``'L2'``, ``'L3'``).
    product_short_name : str, optional
        Short product code (``'VNP46A1'``, ``'VNP13A1'``, etc.).
    product_long_name : str, optional
        Descriptive product name.
    collection_version : str, optional
        Data collection version.
    start_datetime : str, optional
        Acquisition start time (ISO 8601).
    end_datetime : str, optional
        Acquisition end time (ISO 8601).
    production_datetime : str, optional
        File production timestamp.
    day_night_flag : str, optional
        Day/night classification (``'Day'``, ``'Night'``, ``'Both'``).
    granule_id : str, optional
        Unique granule identifier.
    orbit_number : int, optional
        Orbit number for the acquisition.
    horizontal_tile_number : str, optional
        Tile grid column for gridded products.
    vertical_tile_number : str, optional
        Tile grid row for gridded products.
    spatial_resolution : int, optional
        Pixel resolution in meters (375, 750).
    projection : str, optional
        Map projection type.
    geospatial_bounds : tuple, optional
        Bounding box as ``(lat_min, lat_max, lon_min, lon_max)``.
    dataset_long_name : str, optional
        Full descriptive name of the active dataset.
    dataset_units : str, optional
        Physical units of the dataset.
    valid_range : tuple, optional
        Valid data range as ``(min, max)``.
    fill_value : float, optional
        No-data sentinel value.
    scale_factor : float, optional
        Multiplicative scale factor for calibration.
    add_offset : float, optional
        Additive offset for calibration.
    dataset_path : str, optional
        HDF5 internal path to the active dataset.
    """

    # Satellite / instrument
    satellite_name: Optional[str] = None
    instrument_name: Optional[str] = None
    processing_level: Optional[str] = None
    product_short_name: Optional[str] = None
    product_long_name: Optional[str] = None
    collection_version: Optional[str] = None

    # Temporal
    start_datetime: Optional[str] = None
    end_datetime: Optional[str] = None
    production_datetime: Optional[str] = None
    day_night_flag: Optional[str] = None

    # Spatial / orbit
    granule_id: Optional[str] = None
    orbit_number: Optional[int] = None
    horizontal_tile_number: Optional[str] = None
    vertical_tile_number: Optional[str] = None
    spatial_resolution: Optional[int] = None
    projection: Optional[str] = None
    geospatial_bounds: Optional[Tuple[float, float, float, float]] = None

    # Dataset-level calibration
    dataset_long_name: Optional[str] = None
    dataset_units: Optional[str] = None
    valid_range: Optional[Tuple[float, float]] = None
    fill_value: Optional[float] = None
    scale_factor: Optional[float] = None
    add_offset: Optional[float] = None
    dataset_path: Optional[str] = None
