# -*- coding: utf-8 -*-
"""
Sentinel-1 SLC Metadata - Typed metadata for Sentinel-1 IW SLC products.

Nested dataclasses representing the full annotation, calibration, and
noise metadata contained in a Sentinel-1 IW SLC SAFE product.  Each
swath+polarization combination within a SAFE archive maps to one
``Sentinel1SLCMetadata`` instance.

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
2026-02-16

Modified
--------
2026-02-16
"""

# Standard library
from dataclasses import dataclass, field
from typing import List, Optional

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.models.base import ImageMetadata
from grdl.IO.models.common import XYZ


# ===================================================================
# Product-level info (from adsHeader / manifest)
# ===================================================================

@dataclass
class S1SLCProductInfo:
    """Sentinel-1 SLC product-level metadata.

    Parameters
    ----------
    mission : str, optional
        Mission identifier (``'S1A'``, ``'S1B'``, ``'S1C'``).
    mode : str, optional
        Acquisition mode (``'IW'``, ``'EW'``, ``'SM'``).
    product_type : str, optional
        Product type (``'SLC'``).
    transmit_receive_polarization : str, optional
        Product polarization class (``'DV'``, ``'DH'``, ``'SV'``, ``'SH'``).
    start_time : str, optional
        Acquisition start time (ISO 8601 UTC).
    stop_time : str, optional
        Acquisition stop time (ISO 8601 UTC).
    absolute_orbit : int, optional
        Absolute orbit number.
    relative_orbit : int, optional
        Relative orbit number within the cycle.
    orbit_pass : str, optional
        Orbit direction (``'ASCENDING'`` or ``'DESCENDING'``).
    processing_facility : str, optional
        Name of the processing facility.
    processing_time : str, optional
        Processing date/time (ISO 8601 UTC).
    ipf_version : str, optional
        Instrument Processing Facility software version.
    """

    mission: Optional[str] = None
    mode: Optional[str] = None
    product_type: Optional[str] = None
    transmit_receive_polarization: Optional[str] = None
    start_time: Optional[str] = None
    stop_time: Optional[str] = None
    absolute_orbit: Optional[int] = None
    relative_orbit: Optional[int] = None
    orbit_pass: Optional[str] = None
    processing_facility: Optional[str] = None
    processing_time: Optional[str] = None
    ipf_version: Optional[str] = None


# ===================================================================
# Swath-level info (from imageAnnotation/imageInformation)
# ===================================================================

@dataclass
class S1SLCSwathInfo:
    """Per-swath / per-polarization image parameters.

    Parameters
    ----------
    swath : str, optional
        Swath identifier (``'IW1'``, ``'IW2'``, ``'IW3'``).
    polarization : str, optional
        Polarization channel (``'VV'``, ``'VH'``, ``'HH'``, ``'HV'``).
    lines : int
        Total lines in the measurement TIFF.
    samples : int
        Samples per line.
    range_pixel_spacing : float, optional
        Slant-range pixel spacing in meters.
    azimuth_pixel_spacing : float, optional
        Azimuth pixel spacing in meters.
    azimuth_time_interval : float, optional
        Line-to-line time interval in seconds (1 / PRF).
    slant_range_time : float, optional
        Two-way slant-range time to first pixel in seconds.
    incidence_angle_mid : float, optional
        Mid-swath incidence angle in degrees.
    azimuth_steering_rate : float, optional
        TOPS azimuth beam steering rate in degrees/s.
    range_sampling_rate : float, optional
        Range ADC sampling rate in Hz.
    radar_frequency : float, optional
        Carrier frequency in Hz (~5.405 GHz for C-band).
    azimuth_frequency : float, optional
        Pulse repetition frequency (PRF) in Hz.
    """

    swath: Optional[str] = None
    polarization: Optional[str] = None
    lines: int = 0
    samples: int = 0
    range_pixel_spacing: Optional[float] = None
    azimuth_pixel_spacing: Optional[float] = None
    azimuth_time_interval: Optional[float] = None
    slant_range_time: Optional[float] = None
    incidence_angle_mid: Optional[float] = None
    azimuth_steering_rate: Optional[float] = None
    range_sampling_rate: Optional[float] = None
    radar_frequency: Optional[float] = None
    azimuth_frequency: Optional[float] = None


# ===================================================================
# Burst descriptor (from swathTiming/burstList)
# ===================================================================

@dataclass
class S1SLCBurst:
    """Metadata for a single TOPS burst.

    Parameters
    ----------
    index : int
        Zero-based burst index.
    azimuth_time : str, optional
        Burst start azimuth time (ISO 8601 UTC).
    azimuth_anx_time : float, optional
        Time since ascending node crossing in seconds.
    sensing_time : str, optional
        Sensing time of the burst.
    byte_offset : int, optional
        Byte offset of the burst within the measurement TIFF.
    first_valid_sample : np.ndarray, optional
        Per-line first valid sample index. Shape ``(lines_per_burst,)``.
        Value of ``-1`` means the entire line is invalid.
    last_valid_sample : np.ndarray, optional
        Per-line last valid sample index. Shape ``(lines_per_burst,)``.
    first_line : int
        First line of this burst in the full swath image.
    last_line : int
        Last line (exclusive) of this burst in the full swath image.
    lines_per_burst : int
        Number of lines in this burst.
    samples_per_burst : int
        Number of samples per line in this burst.
    """

    index: int = 0
    azimuth_time: Optional[str] = None
    azimuth_anx_time: Optional[float] = None
    sensing_time: Optional[str] = None
    byte_offset: Optional[int] = None
    first_valid_sample: Optional[np.ndarray] = None
    last_valid_sample: Optional[np.ndarray] = None
    first_line: int = 0
    last_line: int = 0
    lines_per_burst: int = 0
    samples_per_burst: int = 0


# ===================================================================
# Orbit state vectors (from generalAnnotation/orbitList)
# ===================================================================

@dataclass
class S1SLCOrbitStateVector:
    """Orbit state vector at a single time.

    Parameters
    ----------
    time : str, optional
        UTC time (ISO 8601).
    position : XYZ, optional
        ECF position in meters.
    velocity : XYZ, optional
        ECF velocity in m/s.
    """

    time: Optional[str] = None
    position: Optional[XYZ] = None
    velocity: Optional[XYZ] = None


# ===================================================================
# Geolocation grid (from geolocationGrid)
# ===================================================================

@dataclass
class S1SLCGeoGridPoint:
    """Single point from the annotation geolocation grid.

    The annotation XML provides a regular grid of tie points mapping
    (line, pixel) to (latitude, longitude, height).  Bilinear
    interpolation over this grid provides pixel-to-geographic transforms.

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
    elevation_angle : float, optional
        Elevation angle in degrees.
    """

    line: float = 0.0
    pixel: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0
    height: float = 0.0
    incidence_angle: Optional[float] = None
    elevation_angle: Optional[float] = None


# ===================================================================
# Doppler centroid (from dopplerCentroid/dcEstimateList)
# ===================================================================

@dataclass
class S1SLCDopplerCentroid:
    """Doppler centroid frequency estimate.

    Parameters
    ----------
    azimuth_time : str, optional
        Reference azimuth time (ISO 8601 UTC).
    t0 : float, optional
        Two-way slant-range time origin for the polynomial in seconds.
    coefficients : np.ndarray, optional
        Polynomial coefficients ``[c0, c1, c2, ...]`` such that
        ``f_dc(t) = c0 + c1*(t - t0) + c2*(t - t0)^2 + ...``
        where ``t`` is two-way slant-range time in seconds.
    """

    azimuth_time: Optional[str] = None
    t0: Optional[float] = None
    coefficients: Optional[np.ndarray] = None


# ===================================================================
# Doppler FM rate (from generalAnnotation/azimuthFmRateList)
# ===================================================================

@dataclass
class S1SLCDopplerFmRate:
    """Azimuth FM rate polynomial.

    Parameters
    ----------
    azimuth_time : str, optional
        Reference azimuth time (ISO 8601 UTC).
    t0 : float, optional
        Two-way slant-range time origin for the polynomial in seconds.
    coefficients : np.ndarray, optional
        Polynomial coefficients for the azimuth FM rate (Hz/s).
    """

    azimuth_time: Optional[str] = None
    t0: Optional[float] = None
    coefficients: Optional[np.ndarray] = None


# ===================================================================
# Calibration vectors (from calibration XML)
# ===================================================================

@dataclass
class S1SLCCalibrationVector:
    """Radiometric calibration look-up table at a single azimuth time.

    Each vector provides calibration values at a set of range pixel
    positions.  Intermediate pixels are obtained by linear interpolation.
    Apply as: ``calibrated = abs(DN)^2 / LUT^2`` for power, or
    ``calibrated = DN / LUT`` for amplitude.

    Parameters
    ----------
    azimuth_time : str, optional
        Reference azimuth time (ISO 8601 UTC).
    line : int
        Image line corresponding to this vector.
    pixel : np.ndarray, optional
        Range pixel indices for the LUT samples.
    sigma_nought : np.ndarray, optional
        Sigma-nought calibration LUT values.
    beta_nought : np.ndarray, optional
        Beta-nought calibration LUT values.
    gamma : np.ndarray, optional
        Gamma calibration LUT values.
    dn : np.ndarray, optional
        Digital number calibration LUT values.
    """

    azimuth_time: Optional[str] = None
    line: int = 0
    pixel: Optional[np.ndarray] = None
    sigma_nought: Optional[np.ndarray] = None
    beta_nought: Optional[np.ndarray] = None
    gamma: Optional[np.ndarray] = None
    dn: Optional[np.ndarray] = None


# ===================================================================
# Noise vectors (from noise XML)
# ===================================================================

@dataclass
class S1SLCNoiseRangeVector:
    """Thermal noise estimate in the range direction.

    Parameters
    ----------
    azimuth_time : str, optional
        Reference azimuth time (ISO 8601 UTC).
    line : int
        Image line corresponding to this vector.
    pixel : np.ndarray, optional
        Range pixel indices.
    noise_range_lut : np.ndarray, optional
        Noise power LUT values (linear scale).
    """

    azimuth_time: Optional[str] = None
    line: int = 0
    pixel: Optional[np.ndarray] = None
    noise_range_lut: Optional[np.ndarray] = None


@dataclass
class S1SLCNoiseAzimuthVector:
    """Thermal noise estimate in the azimuth direction.

    Parameters
    ----------
    swath : str, optional
        Swath identifier.
    first_azimuth_line : int
        First azimuth line of the vector.
    last_azimuth_line : int
        Last azimuth line of the vector.
    first_range_sample : int
        First range sample of the vector.
    last_range_sample : int
        Last range sample of the vector.
    line : np.ndarray, optional
        Azimuth line indices.
    noise_azimuth_lut : np.ndarray, optional
        Noise power LUT values (linear scale).
    """

    swath: Optional[str] = None
    first_azimuth_line: int = 0
    last_azimuth_line: int = 0
    first_range_sample: int = 0
    last_range_sample: int = 0
    line: Optional[np.ndarray] = None
    noise_azimuth_lut: Optional[np.ndarray] = None


# ===================================================================
# Top-level metadata
# ===================================================================

@dataclass
class Sentinel1SLCMetadata(ImageMetadata):
    """Complete typed metadata for a Sentinel-1 IW SLC swath+polarization.

    Inherits universal fields (``format``, ``rows``, ``cols``, ``dtype``)
    from ``ImageMetadata`` and adds all Sentinel-1 SLC annotation,
    calibration, and noise metadata as typed fields.

    Parameters
    ----------
    product_info : S1SLCProductInfo, optional
        Product-level metadata (mission, orbit, times).
    swath_info : S1SLCSwathInfo, optional
        Swath and image geometry parameters.
    bursts : List[S1SLCBurst], optional
        Per-burst timing and valid-sample descriptors.
    orbit_state_vectors : List[S1SLCOrbitStateVector], optional
        Orbit state vectors (position, velocity vs time).
    geolocation_grid : List[S1SLCGeoGridPoint], optional
        Geolocation tie-point grid for pixel-to-latlon transforms.
    doppler_centroids : List[S1SLCDopplerCentroid], optional
        Doppler centroid frequency estimates.
    doppler_fm_rates : List[S1SLCDopplerFmRate], optional
        Azimuth FM rate polynomials.
    calibration_vectors : List[S1SLCCalibrationVector], optional
        Radiometric calibration LUTs.
    noise_range_vectors : List[S1SLCNoiseRangeVector], optional
        Range thermal noise vectors.
    noise_azimuth_vectors : List[S1SLCNoiseAzimuthVector], optional
        Azimuth thermal noise vectors.
    num_bursts : int
        Number of bursts in this swath.
    lines_per_burst : int
        Lines per burst (constant within a swath).
    samples_per_burst : int
        Samples per burst (constant within a swath).

    Examples
    --------
    >>> from grdl.IO.sar import Sentinel1SLCReader
    >>> with Sentinel1SLCReader('product.SAFE', swath='IW1') as reader:
    ...     meta = reader.metadata
    ...     print(meta.product_info.mission)  # 'S1A'
    ...     print(meta.num_bursts)            # 9
    ...     print(meta.swath_info.range_pixel_spacing)  # 2.33
    """

    product_info: Optional[S1SLCProductInfo] = None
    swath_info: Optional[S1SLCSwathInfo] = None
    bursts: Optional[List[S1SLCBurst]] = None
    orbit_state_vectors: Optional[List[S1SLCOrbitStateVector]] = None
    geolocation_grid: Optional[List[S1SLCGeoGridPoint]] = None
    doppler_centroids: Optional[List[S1SLCDopplerCentroid]] = None
    doppler_fm_rates: Optional[List[S1SLCDopplerFmRate]] = None
    calibration_vectors: Optional[List[S1SLCCalibrationVector]] = None
    noise_range_vectors: Optional[List[S1SLCNoiseRangeVector]] = None
    noise_azimuth_vectors: Optional[List[S1SLCNoiseAzimuthVector]] = None
    num_bursts: int = 0
    lines_per_burst: int = 0
    samples_per_burst: int = 0
