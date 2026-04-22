# -*- coding: utf-8 -*-
"""
Sentinel-2 Metadata - Typed metadata for Sentinel-2 MSI products.

Nested dataclasses representing the product-level, tile-level, and
band-level metadata from Sentinel-2 Level-1C and Level-2A SAFE archives.
Sections map to the ESA Product Specification Document (PSD 14.9,
S2-PDGS-TAS-DI-PSD):

- ``S2ProductInfo``  — from ``MTD_MSIL{1C,2A}.xml`` General_Info
- ``S2QualityIndicators`` — from ``MTD_MSIL{1C,2A}.xml`` Quality_Indicators_Info
- ``S2TileGeocoding`` — from ``MTD_TL.xml`` Geometric_Info / Tile_Geocoding
- ``S2AngleGrid`` — from ``MTD_TL.xml`` Tile_Angles (sun + viewing)
- ``S2RadiometricInfo`` — from ``MTD_MSIL2A.xml`` General_Info (L2A only)
- ``S2SpectralInfo`` — per-band resolution / wavelength (PSD Table 4)

When reading standalone JP2 files without SAFE context, only filename-
derived fields and rasterio geotransform fields are populated.  When
reading from a SAFE archive, the reader parses the XML and populates
all available sections.

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
2026-02-11

Modified
--------
2026-03-27  Full PSD-based metadata model with nested dataclasses.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from grdl.IO.models.base import ImageMetadata


# ===================================================================
# Product-level info (from MTD_MSIL{1C,2A}.xml → General_Info)
# ===================================================================

@dataclass
class S2ProductInfo:
    """Product-level metadata from the SAFE manifest and product XML.

    Maps to ``General_Info/Product_Info`` in the product-level
    ``MTD_MSIL{1C,2A}.xml``.

    Parameters
    ----------
    product_type : str, optional
        Product type identifier (``'S2MSI1C'``, ``'S2MSI2A'``).
    processing_level : str, optional
        Processing level (``'Level-1C'``, ``'Level-2A'``).
    product_start_time : str, optional
        Datatake start time (ISO 8601 UTC).
    product_stop_time : str, optional
        Datatake stop time (ISO 8601 UTC).
    generation_time : str, optional
        Product generation time (ISO 8601 UTC).
    processing_baseline : str, optional
        Processing baseline version (e.g., ``'05.10'``).
    spacecraft_name : str, optional
        Spacecraft identifier (``'Sentinel-2A'``, ``'Sentinel-2B'``,
        ``'Sentinel-2C'``).
    datatake_type : str, optional
        Datatake type (``'INS-NOBS'``, ``'INS-DASC'``, etc.).
    datatake_sensing_start : str, optional
        Datatake sensing start time (ISO 8601 UTC).
    sensing_orbit_number : int, optional
        Absolute orbit number.
    sensing_orbit_direction : str, optional
        Orbit direction (``'ASCENDING'``, ``'DESCENDING'``).
    product_doi : str, optional
        Digital Object Identifier for the product.
    """

    product_type: Optional[str] = None
    processing_level: Optional[str] = None
    product_start_time: Optional[str] = None
    product_stop_time: Optional[str] = None
    generation_time: Optional[str] = None
    processing_baseline: Optional[str] = None
    spacecraft_name: Optional[str] = None
    datatake_type: Optional[str] = None
    datatake_sensing_start: Optional[str] = None
    sensing_orbit_number: Optional[int] = None
    sensing_orbit_direction: Optional[str] = None
    product_doi: Optional[str] = None


# ===================================================================
# Quality indicators (from MTD_MSIL{1C,2A}.xml → Quality_Indicators_Info)
# ===================================================================

@dataclass
class S2QualityIndicators:
    """Scene-level quality indicators.

    Maps to ``Quality_Indicators_Info`` in the product-level XML.
    Cloud coverage is always present; L2A adds scene classification
    percentages.

    Parameters
    ----------
    cloud_coverage_assessment : float, optional
        Cloud coverage percentage (0-100).
    degraded_msi_data_percentage : float, optional
        Percentage of degraded MSI data (0-100).
    nodata_pixel_percentage : float, optional
        Percentage of no-data pixels.
    saturated_defective_pixel_percentage : float, optional
        Percentage of saturated or defective pixels.
    dark_features_percentage : float, optional
        L2A scene classification: dark feature percentage.
    cloud_shadow_percentage : float, optional
        L2A scene classification: cloud shadow percentage.
    vegetation_percentage : float, optional
        L2A scene classification: vegetation percentage.
    not_vegetated_percentage : float, optional
        L2A scene classification: not-vegetated percentage.
    water_percentage : float, optional
        L2A scene classification: water percentage.
    unclassified_percentage : float, optional
        L2A scene classification: unclassified percentage.
    medium_proba_clouds_percentage : float, optional
        L2A scene classification: medium probability clouds.
    high_proba_clouds_percentage : float, optional
        L2A scene classification: high probability clouds.
    thin_cirrus_percentage : float, optional
        L2A scene classification: thin cirrus percentage.
    snow_ice_percentage : float, optional
        L2A scene classification: snow/ice percentage.
    """

    cloud_coverage_assessment: Optional[float] = None
    degraded_msi_data_percentage: Optional[float] = None
    nodata_pixel_percentage: Optional[float] = None
    saturated_defective_pixel_percentage: Optional[float] = None
    dark_features_percentage: Optional[float] = None
    cloud_shadow_percentage: Optional[float] = None
    vegetation_percentage: Optional[float] = None
    not_vegetated_percentage: Optional[float] = None
    water_percentage: Optional[float] = None
    unclassified_percentage: Optional[float] = None
    medium_proba_clouds_percentage: Optional[float] = None
    high_proba_clouds_percentage: Optional[float] = None
    thin_cirrus_percentage: Optional[float] = None
    snow_ice_percentage: Optional[float] = None


# ===================================================================
# Tile geocoding (from MTD_TL.xml → Geometric_Info / Tile_Geocoding)
# ===================================================================

@dataclass
class S2TileGeocoding:
    """Tile-level geocoding parameters.

    Maps to ``Geometric_Info/Tile_Geocoding`` in the granule-level
    ``MTD_TL.xml``.  Defines the spatial reference system, tile origin,
    and per-resolution pixel sizes.

    Parameters
    ----------
    horizontal_cs_name : str, optional
        Coordinate system name (e.g., ``'WGS 84 / UTM zone 35S'``).
    horizontal_cs_code : str, optional
        EPSG code string (e.g., ``'EPSG:32735'``).
    ulx : float, optional
        Upper-left X coordinate in CRS units (easting in meters).
    uly : float, optional
        Upper-left Y coordinate in CRS units (northing in meters).
    x_dim_10m : float, optional
        Pixel size in X (easting) for 10 m bands (meters).
    y_dim_10m : float, optional
        Pixel size in Y (northing) for 10 m bands (meters).
    nrows_10m : int, optional
        Number of rows for 10 m resolution.
    ncols_10m : int, optional
        Number of columns for 10 m resolution.
    x_dim_20m : float, optional
        Pixel size for 20 m bands (meters).
    y_dim_20m : float, optional
        Pixel size for 20 m bands (meters).
    nrows_20m : int, optional
        Number of rows for 20 m resolution.
    ncols_20m : int, optional
        Number of columns for 20 m resolution.
    x_dim_60m : float, optional
        Pixel size for 60 m bands (meters).
    y_dim_60m : float, optional
        Pixel size for 60 m bands (meters).
    nrows_60m : int, optional
        Number of rows for 60 m resolution.
    ncols_60m : int, optional
        Number of columns for 60 m resolution.
    """

    horizontal_cs_name: Optional[str] = None
    horizontal_cs_code: Optional[str] = None
    ulx: Optional[float] = None
    uly: Optional[float] = None
    x_dim_10m: Optional[float] = None
    y_dim_10m: Optional[float] = None
    nrows_10m: Optional[int] = None
    ncols_10m: Optional[int] = None
    x_dim_20m: Optional[float] = None
    y_dim_20m: Optional[float] = None
    nrows_20m: Optional[int] = None
    ncols_20m: Optional[int] = None
    x_dim_60m: Optional[float] = None
    y_dim_60m: Optional[float] = None
    nrows_60m: Optional[int] = None
    ncols_60m: Optional[int] = None


# ===================================================================
# Angle grids (from MTD_TL.xml → Tile_Angles)
# ===================================================================

@dataclass
class S2AngleGrid:
    """Sun or viewing angle grid at 5 km posting.

    Maps to ``Tile_Angles/Sun_Angles_Grid`` or
    ``Tile_Angles/Viewing_Incidence_Angles_Grids`` in the granule-level
    ``MTD_TL.xml``.

    Grids are sampled at 5000 m posting (~23x23 for a 10980x10980 tile).
    Values are in degrees.

    Parameters
    ----------
    band_id : int, optional
        Band index (0-12). ``None`` for sun angles.
    detector_id : int, optional
        Detector index. ``None`` for sun angles.
    zenith : np.ndarray, optional
        Zenith angle grid in degrees. Shape ``(nrows, ncols)``.
    azimuth : np.ndarray, optional
        Azimuth angle grid in degrees. Shape ``(nrows, ncols)``.
    col_step : float, optional
        Column step in meters (typically 5000).
    row_step : float, optional
        Row step in meters (typically 5000).
    """

    band_id: Optional[int] = None
    detector_id: Optional[int] = None
    zenith: Optional[np.ndarray] = None
    azimuth: Optional[np.ndarray] = None
    col_step: Optional[float] = None
    row_step: Optional[float] = None


@dataclass
class S2MeanAngles:
    """Scene-level mean sun and viewing angles.

    Parameters
    ----------
    sun_zenith : float, optional
        Mean solar zenith angle in degrees.
    sun_azimuth : float, optional
        Mean solar azimuth angle in degrees.
    viewing_zenith : float, optional
        Mean viewing zenith angle in degrees (across all bands).
    viewing_azimuth : float, optional
        Mean viewing azimuth angle in degrees (across all bands).
    """

    sun_zenith: Optional[float] = None
    sun_azimuth: Optional[float] = None
    viewing_zenith: Optional[float] = None
    viewing_azimuth: Optional[float] = None


# ===================================================================
# Radiometric info (from MTD_MSIL2A.xml → General_Info, L2A only)
# ===================================================================

@dataclass
class S2RadiometricInfo:
    """Radiometric calibration parameters.

    Maps to ``General_Info/Product_Image_Characteristics`` in the
    product-level XML.

    For L2A baseline >= 04.00, ``radio_add_offset`` provides per-band
    additive offsets applied before the quantification divisor:
    ``reflectance = (DN + offset) / quantification_value``.

    Parameters
    ----------
    quantification_value : float, optional
        DN-to-reflectance divisor (typically 10000 for BOA, TOA).
    radio_add_offset : Dict[str, int], optional
        Per-band additive offset. Keys are band IDs (``'B01'``..``'B12'``,
        ``'B8A'``).  Introduced in baseline N0400.
    u_sun : float, optional
        Sun-Earth distance correction factor.
    reflectance_conversion_factor : float, optional
        Reflectance conversion factor (U = 1/d^2).
    irradiance_values : Dict[str, float], optional
        Per-band solar irradiance (W/m^2/um). Keys are band IDs.
    """

    quantification_value: Optional[float] = None
    radio_add_offset: Optional[Dict[str, int]] = None
    u_sun: Optional[float] = None
    reflectance_conversion_factor: Optional[float] = None
    irradiance_values: Optional[Dict[str, float]] = None


# ===================================================================
# Spectral band info (PSD Table 4 — static per-band properties)
# ===================================================================

@dataclass
class S2SpectralBand:
    """Properties of a single Sentinel-2 spectral band.

    Parameters
    ----------
    band_id : str
        Band identifier (``'B01'``..``'B12'``, ``'B8A'``).
    resolution : int
        Spatial resolution in meters (10, 20, or 60).
    wavelength_center : float
        Central wavelength in nanometers.
    wavelength_min : float
        Minimum wavelength in nanometers.
    wavelength_max : float
        Maximum wavelength in nanometers.
    bandwidth : float
        Spectral bandwidth in nanometers.
    purpose : str
        Band purpose description.
    """

    band_id: str = ''
    resolution: int = 0
    wavelength_center: float = 0.0
    wavelength_min: float = 0.0
    wavelength_max: float = 0.0
    bandwidth: float = 0.0
    purpose: str = ''


# ===================================================================
# Product footprint
# ===================================================================

@dataclass
class S2Footprint:
    """Product geographic footprint.

    Maps to ``Product_Info/Product_Organisation/Granule_List`` and
    ``Product_Footprint`` in the product-level XML.

    Parameters
    ----------
    coordinates : List[Tuple[float, float]], optional
        Footprint polygon vertices as ``(lat, lon)`` tuples.
        First and last point are identical (closed ring).
    crs : str, optional
        Coordinate reference system (typically ``'EPSG:4326'``).
    """

    coordinates: Optional[List[Tuple[float, float]]] = None
    crs: Optional[str] = None


# ===================================================================
# Top-level Sentinel2Metadata
# ===================================================================

@dataclass
class Sentinel2Metadata(ImageMetadata):
    """Complete typed metadata for Sentinel-2 MSI products.

    Extends ``ImageMetadata`` with Sentinel-2-specific fields organized
    into typed nested dataclasses following the ESA PSD structure.
    Supports both Level-1C (TOA reflectance) and Level-2A (surface
    reflectance) products.

    **Geolocation fields** — ``transform`` and ``tile_geocoding`` are
    first-class typed attributes (not in ``extras``).

    When only a standalone JP2 file is provided (no SAFE context),
    filename-derived fields and the rasterio affine transform are
    populated.  When a SAFE archive is available, all XML-derived
    sections are populated.

    Attributes
    ----------
    satellite : str, optional
        Satellite short name (``'S2A'``, ``'S2B'``, ``'S2C'``).
    processing_level : str, optional
        Processing level (``'L1C'``, ``'L2A'``).
    sensing_datetime : str, optional
        Sensing start time (ISO 8601).
    mgrs_tile_id : str, optional
        MGRS tile identifier (e.g., ``'T10SEG'``).
    band_id : str, optional
        Band identifier for this file (``'B01'``..``'B12'``, ``'B8A'``).
    resolution_tier : int, optional
        Spatial resolution in meters (10, 20, 60).
    transform : object, optional
        Rasterio ``Affine`` transform for pixel-to-CRS mapping.
    bounds : tuple, optional
        Raster bounding box ``(minx, miny, maxx, maxy)`` in CRS units.
    pixel_resolution : tuple, optional
        Pixel dimensions ``(x_res, y_res)`` in CRS units.
    product_info : S2ProductInfo, optional
        Product-level metadata (spacecraft, orbit, processing).
    quality : S2QualityIndicators, optional
        Cloud cover and scene classification percentages.
    tile_geocoding : S2TileGeocoding, optional
        Tile CRS, origin, and per-resolution grid dimensions.
    sun_angles : S2AngleGrid, optional
        Solar zenith/azimuth grids (5 km posting).
    viewing_angles : List[S2AngleGrid], optional
        Per-band, per-detector viewing angle grids.
    mean_angles : S2MeanAngles, optional
        Scene-level mean sun and viewing angles.
    radiometric : S2RadiometricInfo, optional
        Quantification value, offsets, irradiance (L2A).
    spectral_bands : List[S2SpectralBand], optional
        Per-band spectral properties.
    footprint : S2Footprint, optional
        Product geographic footprint polygon.

    Filename-derived fields (always populated):

    product_type : str, optional
        Product type from filename (``'MSIL1C'``, ``'MSIL2A'``).
    baseline_processing : str, optional
        Processing baseline (e.g., ``'N0510'``).
    relative_orbit : int, optional
        Relative orbit number.
    product_discriminator : str, optional
        Product generation timestamp.
    utm_zone : int, optional
        UTM zone from MGRS tile.
    latitude_band : str, optional
        MGRS latitude band letter.
    wavelength_center : float, optional
        Central wavelength in nanometers (from band lookup).
    wavelength_range : Tuple[float, float], optional
        (min, max) wavelength in nanometers.
    orbit_direction : str, optional
        Orbit direction (``'ASCENDING'``, ``'DESCENDING'``).

    Examples
    --------
    >>> from grdl.IO.eo import Sentinel2Reader
    >>> with Sentinel2Reader('product.SAFE') as reader:
    ...     meta = reader.metadata
    ...     print(meta.satellite)           # 'S2A'
    ...     print(meta.tile_geocoding.horizontal_cs_code)  # 'EPSG:32735'
    ...     print(meta.quality.cloud_coverage_assessment)  # 12.3
    ...     print(meta.mean_angles.sun_zenith)  # 34.5
    ...     print(meta.transform)           # Affine(10.0, ...)
    """

    # -- Mission identifiers (filename-derived) -----------------------
    satellite: Optional[str] = None
    processing_level: Optional[str] = None
    product_type: Optional[str] = None
    baseline_processing: Optional[str] = None

    # -- Temporal (filename-derived) -----------------------------------
    sensing_datetime: Optional[str] = None
    product_discriminator: Optional[str] = None

    # -- Spatial: MGRS tiling (filename-derived) -----------------------
    mgrs_tile_id: Optional[str] = None
    utm_zone: Optional[int] = None
    latitude_band: Optional[str] = None

    # -- Band / spectral (filename-derived + lookup) -------------------
    band_id: Optional[str] = None
    resolution_tier: Optional[int] = None
    wavelength_center: Optional[float] = None
    wavelength_range: Optional[Tuple[float, float]] = None

    # -- Orbital (filename-derived) ------------------------------------
    relative_orbit: Optional[int] = None
    orbit_direction: Optional[str] = None

    # -- Geolocation (from rasterio / tile XML) — first-class fields ---
    transform: Optional[object] = None
    bounds: Optional[tuple] = None
    pixel_resolution: Optional[Tuple[float, float]] = None

    # -- Product-level XML sections ------------------------------------
    product_info: Optional[S2ProductInfo] = None
    quality: Optional[S2QualityIndicators] = None
    radiometric: Optional[S2RadiometricInfo] = None
    footprint: Optional[S2Footprint] = None

    # -- Tile / granule-level XML sections -----------------------------
    tile_geocoding: Optional[S2TileGeocoding] = None
    sun_angles: Optional[S2AngleGrid] = None
    viewing_angles: Optional[List[S2AngleGrid]] = field(default=None)
    mean_angles: Optional[S2MeanAngles] = None

    # -- Static spectral band table ------------------------------------
    spectral_bands: Optional[List[S2SpectralBand]] = field(default=None)
