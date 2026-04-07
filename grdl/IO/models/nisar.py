# -*- coding: utf-8 -*-
"""
NISAR Metadata - Typed metadata for NASA NISAR L-band/S-band SAR data.

Nested dataclass hierarchy for NISAR RSLC (Range Doppler SLC) and
GSLC (Geocoded SLC) products.  Fields are extracted from the NISAR
HDF5 product structure under ``science/{LSAR|SSAR}/{RSLC|GSLC}/``.

Dependencies
------------
h5py

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
2026-02-25

Modified
--------
2026-02-25
"""

# Standard library
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.models.base import ImageMetadata


@dataclass
class NISARIdentification:
    """NISAR product identification metadata.

    Extracted from ``science/{band}/identification/`` group.

    Parameters
    ----------
    mission_id : str, optional
        Mission identifier (``'NISAR'`` or ``'ALOS'`` for simulated data).
    product_type : str, optional
        Product type (``'RSLC'``, ``'GSLC'``).
    radar_band : str, optional
        Radar frequency band character (``'L'`` or ``'S'``).
    look_direction : str, optional
        Antenna look direction (``'right'`` or ``'left'``).
    orbit_pass_direction : str, optional
        Orbit direction (``'ascending'`` or ``'descending'``).
    absolute_orbit_number : int, optional
        Absolute orbit number.
    track_number : int, optional
        Relative track number.
    frame_number : int, optional
        Frame number within the track.
    is_geocoded : bool, optional
        Whether the product is geocoded.
    processing_type : str, optional
        Processing type string.
    zero_doppler_start_time : str, optional
        Zero-Doppler start time (ISO 8601).
    zero_doppler_end_time : str, optional
        Zero-Doppler end time (ISO 8601).
    bounding_polygon : str, optional
        WKT bounding polygon string.
    granule_id : str, optional
        Unique granule identifier string.
    instrument_name : str, optional
        Instrument name (e.g., ``'PALSAR'``).
    processing_date_time : str, optional
        Processing date and time (ISO 8601).
    product_version : str, optional
        Product version string.
    """

    mission_id: Optional[str] = None
    product_type: Optional[str] = None
    radar_band: Optional[str] = None
    look_direction: Optional[str] = None
    orbit_pass_direction: Optional[str] = None
    absolute_orbit_number: Optional[int] = None
    track_number: Optional[int] = None
    frame_number: Optional[int] = None
    is_geocoded: Optional[bool] = None
    processing_type: Optional[str] = None
    zero_doppler_start_time: Optional[str] = None
    zero_doppler_end_time: Optional[str] = None
    bounding_polygon: Optional[str] = None
    granule_id: Optional[str] = None
    instrument_name: Optional[str] = None
    processing_date_time: Optional[str] = None
    product_version: Optional[str] = None


@dataclass
class NISAROrbit:
    """NISAR orbit state vectors.

    Extracted from ``science/{band}/{product}/metadata/orbit/`` group.

    Parameters
    ----------
    time : np.ndarray, optional
        Time vector, shape ``(N,)``, seconds since ``reference_epoch``.
    position : np.ndarray, optional
        ECF position vectors, shape ``(N, 3)``, meters.
    velocity : np.ndarray, optional
        ECF velocity vectors, shape ``(N, 3)``, m/s.
    reference_epoch : str, optional
        Time reference string (e.g., ``'seconds since 2008-10-12 00:00:00'``).
    orbit_type : str, optional
        Orbit determination type (e.g., ``'DOE'``).
    interp_method : str, optional
        Interpolation method (e.g., ``'Hermite'``).
    """

    time: Optional[np.ndarray] = None
    position: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    reference_epoch: Optional[str] = None
    orbit_type: Optional[str] = None
    interp_method: Optional[str] = None


@dataclass
class NISARAttitude:
    """NISAR spacecraft attitude data.

    Extracted from ``science/{band}/{product}/metadata/attitude/`` group.

    Parameters
    ----------
    time : np.ndarray, optional
        Time vector, shape ``(M,)``, seconds since ``reference_epoch``.
    quaternions : np.ndarray, optional
        Attitude quaternions, shape ``(M, 4)``.
    euler_angles : np.ndarray, optional
        Euler angles, shape ``(M, 3)``, degrees.
    reference_epoch : str, optional
        Time reference string.
    attitude_type : str, optional
        Attitude determination type.
    """

    time: Optional[np.ndarray] = None
    quaternions: Optional[np.ndarray] = None
    euler_angles: Optional[np.ndarray] = None
    reference_epoch: Optional[str] = None
    attitude_type: Optional[str] = None


@dataclass
class NISARSwathParameters:
    """NISAR RSLC swath/frequency parameters.

    Extracted from ``science/{band}/RSLC/swaths/frequency{X}/`` group.

    Parameters
    ----------
    acquired_center_frequency : float, optional
        Acquired center frequency, Hz.
    acquired_range_bandwidth : float, optional
        Acquired range bandwidth, Hz.
    processed_center_frequency : float, optional
        Processed center frequency, Hz.
    processed_range_bandwidth : float, optional
        Processed range bandwidth, Hz.
    processed_azimuth_bandwidth : float, optional
        Processed azimuth bandwidth, Hz.
    slant_range_spacing : float, optional
        Slant range sample spacing, meters.
    nominal_acquisition_prf : float, optional
        Nominal pulse repetition frequency, Hz.
    scene_center_along_track_spacing : float, optional
        Along-track spacing at scene center, meters.
    scene_center_ground_range_spacing : float, optional
        Ground-range spacing at scene center, meters.
    polarizations : List[str], optional
        Available polarizations (e.g., ``['HH', 'HV']``).
    number_of_sub_swaths : int, optional
        Number of sub-swaths.
    zero_doppler_time_spacing : float, optional
        Azimuth time spacing, seconds.
    slant_range : np.ndarray, optional
        Slant range vector, shape ``(cols,)``, meters.
    zero_doppler_time : np.ndarray, optional
        Zero-Doppler time vector, shape ``(rows,)``, seconds since epoch.
    zero_doppler_time_reference_epoch : str, optional
        Time reference for zero_doppler_time.
    doppler_centroid : np.ndarray, optional
        Doppler centroid 2D grid, shape ``(n_dc_az, n_dc_rng)``, Hz.
        Extracted from
        ``metadata/processingInformation/parameters/frequency{X}/dopplerCentroid``.
    doppler_centroid_slant_range : np.ndarray, optional
        Slant-range axis of the Doppler centroid grid, shape ``(n_dc_rng,)``, m.
    doppler_centroid_azimuth_time : np.ndarray, optional
        Azimuth time axis of the Doppler centroid grid,
        shape ``(n_dc_az,)``, seconds.
    """

    acquired_center_frequency: Optional[float] = None
    acquired_range_bandwidth: Optional[float] = None
    processed_center_frequency: Optional[float] = None
    processed_range_bandwidth: Optional[float] = None
    processed_azimuth_bandwidth: Optional[float] = None
    slant_range_spacing: Optional[float] = None
    nominal_acquisition_prf: Optional[float] = None
    scene_center_along_track_spacing: Optional[float] = None
    scene_center_ground_range_spacing: Optional[float] = None
    polarizations: Optional[List[str]] = None
    number_of_sub_swaths: Optional[int] = None
    zero_doppler_time_spacing: Optional[float] = None
    slant_range: Optional[np.ndarray] = None
    zero_doppler_time: Optional[np.ndarray] = None
    zero_doppler_time_reference_epoch: Optional[str] = None
    range_chirp_weighting: Optional[np.ndarray] = None
    azimuth_chirp_weighting: Optional[np.ndarray] = None
    doppler_centroid: Optional[np.ndarray] = None
    doppler_centroid_slant_range: Optional[np.ndarray] = None
    doppler_centroid_azimuth_time: Optional[np.ndarray] = None


@dataclass
class NISARGridParameters:
    """NISAR GSLC geocoded grid parameters.

    Extracted from ``science/{band}/GSLC/grids/frequency{X}/`` group.

    Parameters
    ----------
    x_coordinates : np.ndarray, optional
        Easting coordinate vector, shape ``(cols,)``.
    y_coordinates : np.ndarray, optional
        Northing coordinate vector, shape ``(rows,)``.
    x_coordinate_spacing : float, optional
        X pixel spacing, meters.
    y_coordinate_spacing : float, optional
        Y pixel spacing, meters.
    epsg : int, optional
        EPSG code of the map projection.
    center_frequency : float, optional
        Center frequency, Hz.
    range_bandwidth : float, optional
        Range bandwidth, Hz.
    azimuth_bandwidth : float, optional
        Azimuth bandwidth, Hz.
    slant_range_spacing : float, optional
        Original slant range spacing, meters.
    polarizations : List[str], optional
        Available polarizations.
    """

    x_coordinates: Optional[np.ndarray] = None
    y_coordinates: Optional[np.ndarray] = None
    x_coordinate_spacing: Optional[float] = None
    y_coordinate_spacing: Optional[float] = None
    epsg: Optional[int] = None
    center_frequency: Optional[float] = None
    range_bandwidth: Optional[float] = None
    azimuth_bandwidth: Optional[float] = None
    slant_range_spacing: Optional[float] = None
    polarizations: Optional[List[str]] = None


@dataclass
class NISARGeolocationGrid:
    """NISAR RSLC geolocation grid mapping pixel coordinates to geography.

    Extracted from ``science/{band}/RSLC/metadata/geolocationGrid/`` group.
    Arrays are 3-D with shape ``(heights, azimuth, range)`` where the
    height axis samples different terrain elevations.

    Parameters
    ----------
    coordinate_x : np.ndarray, optional
        Longitude (or easting) grid.
    coordinate_y : np.ndarray, optional
        Latitude (or northing) grid.
    epsg : int, optional
        EPSG code (4326 for geographic lon/lat).
    slant_range : np.ndarray, optional
        Slant range sample vector, shape ``(range,)``.
    zero_doppler_time : np.ndarray, optional
        Zero-Doppler time sample vector, shape ``(azimuth,)``.
    height_above_ellipsoid : np.ndarray, optional
        Height samples, shape ``(heights,)``, meters.
    incidence_angle : np.ndarray, optional
        Incidence angle grid, degrees.
    elevation_angle : np.ndarray, optional
        Elevation angle grid, degrees.
    """

    coordinate_x: Optional[np.ndarray] = None
    coordinate_y: Optional[np.ndarray] = None
    epsg: Optional[int] = None
    slant_range: Optional[np.ndarray] = None
    zero_doppler_time: Optional[np.ndarray] = None
    height_above_ellipsoid: Optional[np.ndarray] = None
    incidence_angle: Optional[np.ndarray] = None
    elevation_angle: Optional[np.ndarray] = None


@dataclass
class NISARCalibration:
    """NISAR radiometric calibration information.

    Extracted from the ``calibrationInformation/geometry/`` sub-group.

    Parameters
    ----------
    sigma0 : np.ndarray, optional
        Sigma-nought calibration grid.
    beta0 : np.ndarray, optional
        Beta-nought calibration grid.
    gamma0 : np.ndarray, optional
        Gamma-nought calibration grid.
    """

    sigma0: Optional[np.ndarray] = None
    beta0: Optional[np.ndarray] = None
    gamma0: Optional[np.ndarray] = None


@dataclass
class NISARProcessingInfo:
    """NISAR processing information.

    Extracted from ``metadata/processingInformation/`` sub-group.

    Parameters
    ----------
    software_version : str, optional
        Processing software version string.
    algorithms : Dict[str, Any], optional
        Algorithm parameters as key-value pairs.
    """

    software_version: Optional[str] = None
    algorithms: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class NISARMetadata(ImageMetadata):
    """Complete typed metadata for NASA NISAR SAR products.

    Supports both RSLC (range-Doppler SLC) and GSLC (geocoded SLC)
    product types for L-band (LSAR) and S-band (SSAR) instruments.
    Inherits from ``ImageMetadata`` for universal fields and dict-like
    access.

    Parameters
    ----------
    product_type : str, optional
        Product type (``'RSLC'`` or ``'GSLC'``).
    radar_band : str, optional
        Radar band group name (``'LSAR'`` or ``'SSAR'``).
    frequency : str, optional
        Active frequency sub-band (``'A'`` or ``'B'``).
    polarization : str, optional
        Active polarization channel (``'HH'``, ``'HV'``, ``'VH'``, ``'VV'``).
    available_frequencies : List[str], optional
        All frequencies present in the file.
    available_polarizations : List[str], optional
        All polarizations present for the active frequency.
    identification : NISARIdentification, optional
        Product identification metadata.
    orbit : NISAROrbit, optional
        Orbit state vectors.
    attitude : NISARAttitude, optional
        Spacecraft attitude data.
    swath_parameters : NISARSwathParameters, optional
        Swath parameters (RSLC only).
    grid_parameters : NISARGridParameters, optional
        Grid parameters (GSLC only).
    geolocation_grid : NISARGeolocationGrid, optional
        Geolocation grid (RSLC only).
    calibration : NISARCalibration, optional
        Radiometric calibration data.
    processing_info : NISARProcessingInfo, optional
        Processing information.

    Examples
    --------
    >>> meta = reader.metadata  # NISARMetadata
    >>> meta.product_type
    'RSLC'
    >>> meta.radar_band
    'LSAR'
    >>> meta.identification.look_direction
    'right'
    >>> meta.orbit.position.shape
    (8, 3)
    """

    product_type: Optional[str] = None
    radar_band: Optional[str] = None
    frequency: Optional[str] = None
    polarization: Optional[str] = None
    available_frequencies: Optional[List[str]] = None
    available_polarizations: Optional[List[str]] = None
    identification: Optional[NISARIdentification] = None
    orbit: Optional[NISAROrbit] = None
    attitude: Optional[NISARAttitude] = None
    swath_parameters: Optional[NISARSwathParameters] = None
    grid_parameters: Optional[NISARGridParameters] = None
    geolocation_grid: Optional[NISARGeolocationGrid] = None
    calibration: Optional[NISARCalibration] = None
    processing_info: Optional[NISARProcessingInfo] = None
