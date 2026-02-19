# -*- coding: utf-8 -*-
"""
TerraSAR-X / TanDEM-X Metadata - Typed metadata for TSX/TDX SAR products.

Nested dataclasses representing the full annotation, geolocation grid,
and calibration metadata from TerraSAR-X and TanDEM-X products (SSC,
MGD, GEC, EEC).  Each reader instance populates one
``TerraSARMetadata`` with all parsed sections.

Author
------
Jason Fritz
jpfritz@zai.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-19

Modified
--------
2026-02-19
"""

# Standard library
from dataclasses import dataclass
from typing import List, Optional

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.models.base import ImageMetadata
from grdl.IO.models.common import XYZ, LatLonHAE


# ===================================================================
# Product-level info (from generalHeader + productInfo)
# ===================================================================

@dataclass
class TSXProductInfo:
    """TerraSAR-X / TanDEM-X product-level metadata.

    Parameters
    ----------
    mission : str, optional
        Mission name (``'TerraSAR-X'`` or ``'TanDEM-X'``).
    satellite : str, optional
        Satellite identifier (``'TSX-1'`` or ``'TDX-1'``).
    product_type : str, optional
        Product type (``'SSC'``, ``'MGD'``, ``'GEC'``, ``'EEC'``).
    imaging_mode : str, optional
        Imaging mode (``'SM'``, ``'HS'``, ``'SL'``, ``'SC'``, ``'ST'``).
    look_direction : str, optional
        Look direction (``'RIGHT'`` or ``'LEFT'``).
    polarization_mode : str, optional
        Polarization class (``'SINGLE'``, ``'DUAL'``, ``'TWIN'``,
        ``'QUAD'``).
    polarization_list : List[str], optional
        List of polarization channels (e.g., ``['HH']``,
        ``['HH', 'VV']``).
    orbit_direction : str, optional
        Orbit direction (``'ASCENDING'`` or ``'DESCENDING'``).
    absolute_orbit : int, optional
        Absolute orbit number.
    start_time_utc : str, optional
        Acquisition start time (ISO 8601 UTC).
    stop_time_utc : str, optional
        Acquisition stop time (ISO 8601 UTC).
    generation_time : str, optional
        Product generation time (ISO 8601 UTC).
    processor_version : str, optional
        SAR processor version string.
    """

    mission: Optional[str] = None
    satellite: Optional[str] = None
    product_type: Optional[str] = None
    imaging_mode: Optional[str] = None
    look_direction: Optional[str] = None
    polarization_mode: Optional[str] = None
    polarization_list: Optional[List[str]] = None
    orbit_direction: Optional[str] = None
    absolute_orbit: Optional[int] = None
    start_time_utc: Optional[str] = None
    stop_time_utc: Optional[str] = None
    generation_time: Optional[str] = None
    processor_version: Optional[str] = None


# ===================================================================
# Scene info (from productInfo/sceneInfo)
# ===================================================================

@dataclass
class TSXSceneInfo:
    """Scene geographic and geometric information.

    Parameters
    ----------
    center_lat : float, optional
        Scene center latitude in degrees.
    center_lon : float, optional
        Scene center longitude in degrees.
    scene_extent : List[LatLonHAE], optional
        Scene corner coordinates (typically 4 corners).
    incidence_angle_near : float, optional
        Near-range incidence angle in degrees.
    incidence_angle_far : float, optional
        Far-range incidence angle in degrees.
    incidence_angle_center : float, optional
        Center incidence angle in degrees.
    heading_angle : float, optional
        Platform heading angle in degrees from north.
    """

    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    scene_extent: Optional[List[LatLonHAE]] = None
    incidence_angle_near: Optional[float] = None
    incidence_angle_far: Optional[float] = None
    incidence_angle_center: Optional[float] = None
    heading_angle: Optional[float] = None


# ===================================================================
# Image info (from productInfo/imageDataInfo)
# ===================================================================

@dataclass
class TSXImageInfo:
    """Image dimensions and sampling parameters.

    Parameters
    ----------
    num_rows : int
        Number of rows (azimuth lines).
    num_cols : int
        Number of columns (range samples).
    row_spacing : float, optional
        Row (azimuth) pixel spacing in meters.
    col_spacing : float, optional
        Column (range) pixel spacing in meters.
    sample_type : str, optional
        Data sample type (``'COMPLEX'`` for SSC,
        ``'DETECTED'`` for MGD/GEC/EEC).
    data_format : str, optional
        Data storage format (``'COSAR'`` or ``'GEOTIFF'``).
    bits_per_sample : int, optional
        Bits per sample component (e.g., 16 for int16 I/Q).
    projection : str, optional
        Data projection (``'SLANTRANGE'``, ``'GROUNDRANGE'``,
        ``'MAP'``).
    """

    num_rows: int = 0
    num_cols: int = 0
    row_spacing: Optional[float] = None
    col_spacing: Optional[float] = None
    sample_type: Optional[str] = None
    data_format: Optional[str] = None
    bits_per_sample: Optional[int] = None
    projection: Optional[str] = None


# ===================================================================
# Radar parameters (from instrument/radarParameters)
# ===================================================================

@dataclass
class TSXRadarParams:
    """Radar instrument parameters.

    Parameters
    ----------
    center_frequency : float, optional
        Center frequency in Hz (~9.65 GHz for X-band).
    prf : float, optional
        Pulse Repetition Frequency in Hz.
    range_bandwidth : float, optional
        Range chirp bandwidth in Hz.
    chirp_duration : float, optional
        Chirp pulse duration in seconds.
    adc_sampling_rate : float, optional
        ADC sampling rate in Hz.
    """

    center_frequency: Optional[float] = None
    prf: Optional[float] = None
    range_bandwidth: Optional[float] = None
    chirp_duration: Optional[float] = None
    adc_sampling_rate: Optional[float] = None


# ===================================================================
# Orbit state vectors (from platform/orbit/stateVec)
# ===================================================================

@dataclass
class TSXOrbitStateVector:
    """Orbit state vector at a single time.

    Parameters
    ----------
    time_utc : str, optional
        UTC time (ISO 8601).
    position : XYZ, optional
        ECF position in meters.
    velocity : XYZ, optional
        ECF velocity in m/s.
    """

    time_utc: Optional[str] = None
    position: Optional[XYZ] = None
    velocity: Optional[XYZ] = None


# ===================================================================
# Geolocation grid (from GEOREF.xml)
# ===================================================================

@dataclass
class TSXGeoGridPoint:
    """Geolocation grid tie point from GEOREF.xml.

    The georeference annotation provides a grid of tie points mapping
    (line, pixel) to (latitude, longitude, height).  Bilinear
    interpolation over this grid provides pixel-to-geographic
    transforms.

    Parameters
    ----------
    line : float
        Azimuth line coordinate.
    pixel : float
        Range pixel coordinate.
    latitude : float
        WGS-84 latitude in degrees.
    longitude : float
        WGS-84 longitude in degrees.
    height : float
        Height above WGS-84 ellipsoid in meters.
    incidence_angle : float, optional
        Local incidence angle in degrees.
    """

    line: float = 0.0
    pixel: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0
    height: float = 0.0
    incidence_angle: Optional[float] = None


# ===================================================================
# Calibration (from CALDATA.xml)
# ===================================================================

@dataclass
class TSXCalibration:
    """Radiometric calibration data from CALDATA.xml.

    Apply the calibration constant as:
    ``sigma_nought = Ks * |DN|^2 / sin(incidence_angle)``

    Parameters
    ----------
    calibration_constant : float, optional
        Absolute radiometric calibration factor (Ks).
    noise_equivalent_beta_nought : float, optional
        NESZ in beta-nought convention.
    calibration_type : str, optional
        Calibration annotation type.
    """

    calibration_constant: Optional[float] = None
    noise_equivalent_beta_nought: Optional[float] = None
    calibration_type: Optional[str] = None


# ===================================================================
# Doppler info (from processing/doppler)
# ===================================================================

@dataclass
class TSXDopplerInfo:
    """Doppler centroid and rate information.

    Parameters
    ----------
    doppler_centroid_coefficients : np.ndarray, optional
        Polynomial coefficients for Doppler centroid vs. range time.
    reference_time : float, optional
        Reference time for the Doppler polynomial in seconds.
    """

    doppler_centroid_coefficients: Optional[np.ndarray] = None
    reference_time: Optional[float] = None


# ===================================================================
# Processing info (from processing/processingInfo)
# ===================================================================

@dataclass
class TSXProcessingInfo:
    """SAR processing parameters.

    Parameters
    ----------
    processor_version : str, optional
        SAR processor version string.
    processing_level : str, optional
        Processing level (e.g., ``'L1B'`` for SSC).
    range_looks : int, optional
        Number of range looks.
    azimuth_looks : int, optional
        Number of azimuth looks.
    """

    processor_version: Optional[str] = None
    processing_level: Optional[str] = None
    range_looks: Optional[int] = None
    azimuth_looks: Optional[int] = None


# ===================================================================
# Top-level metadata
# ===================================================================

@dataclass
class TerraSARMetadata(ImageMetadata):
    """Complete typed metadata for a TerraSAR-X / TanDEM-X product.

    Inherits universal fields (``format``, ``rows``, ``cols``, ``dtype``)
    from ``ImageMetadata`` and adds all TerraSAR-X annotation,
    geolocation, and calibration metadata as typed fields.

    Parameters
    ----------
    product_info : TSXProductInfo, optional
        Product-level metadata (mission, satellite, mode, times).
    scene_info : TSXSceneInfo, optional
        Scene geographic extent and incidence angles.
    image_info : TSXImageInfo, optional
        Image dimensions and sampling parameters.
    radar_params : TSXRadarParams, optional
        Radar instrument parameters.
    orbit_state_vectors : List[TSXOrbitStateVector], optional
        Orbit state vectors (position, velocity vs. time).
    geolocation_grid : List[TSXGeoGridPoint], optional
        Geolocation tie-point grid from GEOREF.xml.
    calibration : TSXCalibration, optional
        Radiometric calibration constants from CALDATA.xml.
    doppler_info : TSXDopplerInfo, optional
        Doppler centroid polynomial.
    processing_info : TSXProcessingInfo, optional
        SAR processor parameters.

    Examples
    --------
    >>> from grdl.IO.sar import TerraSARReader
    >>> with TerraSARReader('TSX1_SAR__SSC.../') as reader:
    ...     meta = reader.metadata
    ...     print(meta.product_info.satellite)   # 'TSX-1'
    ...     print(meta.radar_params.center_frequency)  # 9.65e9
    """

    product_info: Optional[TSXProductInfo] = None
    scene_info: Optional[TSXSceneInfo] = None
    image_info: Optional[TSXImageInfo] = None
    radar_params: Optional[TSXRadarParams] = None
    orbit_state_vectors: Optional[List[TSXOrbitStateVector]] = None
    geolocation_grid: Optional[List[TSXGeoGridPoint]] = None
    calibration: Optional[TSXCalibration] = None
    doppler_info: Optional[TSXDopplerInfo] = None
    processing_info: Optional[TSXProcessingInfo] = None
