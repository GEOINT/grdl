# -*- coding: utf-8 -*-
"""
SIDD Metadata - Complete typed metadata for SIDD imagery.

Nested dataclasses mirroring all 13 sections of the NGA SIDD v2 standard
as implemented by sarpy/sarkit. Every field from the SIDD XML schema
is represented as a typed attribute.

ErrorStatistics, Radiometric, and MatchInfo sections reuse the
corresponding SICD types since they share the same schema.

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
from typing import Any, Dict, List, Optional

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.models.base import ImageMetadata
from grdl.IO.models.common import (
    XYZ,
    LatLon,
    LatLonHAE,
    RowCol,
    Poly1D,
    Poly2D,
    XYZPoly,
)
from grdl.IO.models.sicd import (
    SICDErrorStatistics,
    SICDRadiometric,
    SICDMatchInfo,
)


# ===================================================================
# Section 1: ProductCreation
# ===================================================================

@dataclass
class SIDDProcessorInformation:
    """Processor information for product creation.

    Parameters
    ----------
    application : str, optional
        Application name.
    processing_date_time : str, optional
        Processing date/time (ISO 8601).
    site : str, optional
        Processing site.
    profile : str, optional
        Processing profile.
    """

    application: Optional[str] = None
    processing_date_time: Optional[str] = None
    site: Optional[str] = None
    profile: Optional[str] = None


@dataclass
class SIDDClassification:
    """Product security classification.

    Parameters
    ----------
    classification : str, optional
        Classification level (``'U'``, ``'C'``, ``'R'``, ``'S'``,
        ``'TS'``).
    owner_producer : str, optional
        Owner/producer country code.
    des_version : int, optional
        DES version number.
    create_date : str, optional
        Classification creation date.
    sar_identifier : str, optional
        SAR identifier.
    sci_controls : str, optional
        SCI controls.
    dissemination_controls : str, optional
        Dissemination controls.
    releasable_to : str, optional
        Releasable-to marking.
    classified_by : str, optional
        Person/office who classified.
    derived_from : str, optional
        Derivation source.
    declass_date : str, optional
        Declassification date.
    declass_event : str, optional
        Declassification event.
    """

    classification: Optional[str] = None
    owner_producer: Optional[str] = None
    des_version: Optional[int] = None
    create_date: Optional[str] = None
    sar_identifier: Optional[str] = None
    sci_controls: Optional[str] = None
    dissemination_controls: Optional[str] = None
    releasable_to: Optional[str] = None
    classified_by: Optional[str] = None
    derived_from: Optional[str] = None
    declass_date: Optional[str] = None
    declass_event: Optional[str] = None


@dataclass
class SIDDProductCreation:
    """Product creation information.

    Parameters
    ----------
    processor_information : SIDDProcessorInformation, optional
        Processor information.
    classification : SIDDClassification, optional
        Security classification.
    product_name : str, optional
        Product name.
    product_class : str, optional
        Product class.
    product_type : str, optional
        Product type.
    """

    processor_information: Optional[SIDDProcessorInformation] = None
    classification: Optional[SIDDClassification] = None
    product_name: Optional[str] = None
    product_class: Optional[str] = None
    product_type: Optional[str] = None


# ===================================================================
# Section 2: Display
# ===================================================================

@dataclass
class SIDDDRAParameters:
    """Dynamic Range Adjustment parameters.

    Parameters
    ----------
    pmin : float, optional
        Minimum percentile (0-1).
    pmax : float, optional
        Maximum percentile (0-1).
    emin_modifier : float, optional
        Minimum enhancement modifier (0-1).
    emax_modifier : float, optional
        Maximum enhancement modifier (0-1).
    """

    pmin: Optional[float] = None
    pmax: Optional[float] = None
    emin_modifier: Optional[float] = None
    emax_modifier: Optional[float] = None


@dataclass
class SIDDDRAOverrides:
    """Dynamic Range Adjustment override values.

    Parameters
    ----------
    subtractor : float, optional
        Subtractor value (0-2047).
    multiplier : float, optional
        Multiplier value (0-2047).
    """

    subtractor: Optional[float] = None
    multiplier: Optional[float] = None


@dataclass
class SIDDDynamicRangeAdjustment:
    """Dynamic range adjustment parameters.

    Parameters
    ----------
    algorithm_type : str, optional
        DRA algorithm: ``'AUTO'``, ``'MANUAL'``, ``'NONE'``.
    band_stats_source : int, optional
        Band statistics source index.
    dra_parameters : SIDDDRAParameters, optional
        DRA parameters.
    dra_overrides : SIDDDRAOverrides, optional
        DRA overrides.
    """

    algorithm_type: Optional[str] = None
    band_stats_source: Optional[int] = None
    dra_parameters: Optional[SIDDDRAParameters] = None
    dra_overrides: Optional[SIDDDRAOverrides] = None


@dataclass
class SIDDDisplay:
    """Product display parameters.

    Parameters
    ----------
    pixel_type : str, optional
        Pixel type: ``'MONO8I'``, ``'MONO8LU'``, ``'MONO16I'``,
        ``'RGB24I'``, ``'RGB8LU'``.
    num_bands : int, optional
        Number of display bands.
    default_band_display : int, optional
        Default band for display.
    dynamic_range_adjustment : SIDDDynamicRangeAdjustment, optional
        Dynamic range adjustment for the first band.
    """

    pixel_type: Optional[str] = None
    num_bands: Optional[int] = None
    default_band_display: Optional[int] = None
    dynamic_range_adjustment: Optional[SIDDDynamicRangeAdjustment] = None


# ===================================================================
# Section 3: GeoData
# ===================================================================

@dataclass
class SIDDGeoData:
    """Geographic data.

    Parameters
    ----------
    earth_model : str
        Earth model identifier (default ``'WGS_84'``).
    image_corners : List[LatLon], optional
        Four image corner coordinates.
    valid_data : List[LatLon], optional
        Valid data polygon vertices.
    """

    earth_model: str = 'WGS_84'
    image_corners: Optional[List[LatLon]] = None
    valid_data: Optional[List[LatLon]] = None


# ===================================================================
# Section 4: Measurement
# ===================================================================

@dataclass
class SIDDReferencePoint:
    """Measurement reference point.

    Parameters
    ----------
    ecef : XYZ, optional
        ECF coordinates (meters).
    point : RowCol, optional
        Image pixel coordinates.
    name : str, optional
        Reference point name.
    """

    ecef: Optional[XYZ] = None
    point: Optional[RowCol] = None
    name: Optional[str] = None


@dataclass
class SIDDProductPlane:
    """Product measurement plane.

    Parameters
    ----------
    row_unit_vector : XYZ, optional
        Row unit vector in ECF.
    col_unit_vector : XYZ, optional
        Column unit vector in ECF.
    """

    row_unit_vector: Optional[XYZ] = None
    col_unit_vector: Optional[XYZ] = None


@dataclass
class SIDDPlaneProjection:
    """Plane projection parameters.

    Parameters
    ----------
    reference_point : SIDDReferencePoint, optional
        Reference point.
    sample_spacing : RowCol, optional
        Row/column sample spacing.
    time_coa_poly : Poly2D, optional
        Center of aperture time polynomial.
    product_plane : SIDDProductPlane, optional
        Product measurement plane.
    """

    reference_point: Optional[SIDDReferencePoint] = None
    sample_spacing: Optional[RowCol] = None
    time_coa_poly: Optional[Poly2D] = None
    product_plane: Optional[SIDDProductPlane] = None


@dataclass
class SIDDMeasurement:
    """Measurement parameters.

    Parameters
    ----------
    projection_type : str, optional
        Projection type: ``'PlaneProjection'``,
        ``'GeographicProjection'``, ``'CylindricalProjection'``,
        ``'PolynomialProjection'``.
    plane_projection : SIDDPlaneProjection, optional
        Plane projection parameters.
    pixel_footprint : RowCol, optional
        Pixel footprint (rows, cols).
    arp_flag : str, optional
        ARP type: ``'REALTIME'``, ``'PREDICTED'``,
        ``'POST PROCESSED'``.
    arp_poly : XYZPoly, optional
        ARP position polynomial.
    """

    projection_type: Optional[str] = None
    plane_projection: Optional[SIDDPlaneProjection] = None
    pixel_footprint: Optional[RowCol] = None
    arp_flag: Optional[str] = None
    arp_poly: Optional[XYZPoly] = None


# ===================================================================
# Section 5: ExploitationFeatures
# ===================================================================

@dataclass
class SIDDRadarMode:
    """Radar collection mode.

    Parameters
    ----------
    mode_type : str, optional
        Mode type: ``'SPOTLIGHT'``, ``'STRIPMAP'``, ``'SCANSAR'``.
    mode_id : str, optional
        Mode identifier string.
    """

    mode_type: Optional[str] = None
    mode_id: Optional[str] = None


@dataclass
class SIDDTxRcvPolarization:
    """Transmit/receive polarization pair.

    Parameters
    ----------
    tx_polarization : str, optional
        Transmit polarization (``'V'``, ``'H'``, ``'RHC'``, ``'LHC'``).
    rcv_polarization : str, optional
        Receive polarization.
    """

    tx_polarization: Optional[str] = None
    rcv_polarization: Optional[str] = None


@dataclass
class SIDDCollectionGeometry:
    """Collection geometry parameters.

    Parameters
    ----------
    azimuth : float, optional
        Azimuth angle (degrees, 0-360).
    slope : float, optional
        Slope angle (degrees, 0-90).
    squint : float, optional
        Squint angle (degrees).
    graze : float, optional
        Grazing angle (degrees, 0-90).
    tilt : float, optional
        Tilt angle (degrees).
    doppler_cone_angle : float, optional
        Doppler cone angle (degrees, 0-180).
    """

    azimuth: Optional[float] = None
    slope: Optional[float] = None
    squint: Optional[float] = None
    graze: Optional[float] = None
    tilt: Optional[float] = None
    doppler_cone_angle: Optional[float] = None


@dataclass
class SIDDAngleMagnitude:
    """Angle and magnitude pair.

    Parameters
    ----------
    angle : float, optional
        Angle (degrees).
    magnitude : float, optional
        Magnitude value.
    """

    angle: Optional[float] = None
    magnitude: Optional[float] = None


@dataclass
class SIDDCollectionPhenomenology:
    """Collection phenomenology parameters.

    Parameters
    ----------
    shadow : SIDDAngleMagnitude, optional
        Shadow angle and magnitude.
    layover : SIDDAngleMagnitude, optional
        Layover angle and magnitude.
    multi_path : float, optional
        Multi-path ground bounce.
    ground_track : float, optional
        Ground track angle (degrees).
    """

    shadow: Optional[SIDDAngleMagnitude] = None
    layover: Optional[SIDDAngleMagnitude] = None
    multi_path: Optional[float] = None
    ground_track: Optional[float] = None


@dataclass
class SIDDCollectionInfo:
    """Exploitation features collection information.

    Parameters
    ----------
    sensor_name : str, optional
        Sensor/collector name.
    radar_mode : SIDDRadarMode, optional
        Radar collection mode.
    collection_date_time : str, optional
        Collection date/time (ISO 8601).
    collection_duration : float, optional
        Collection duration (seconds).
    resolution_range : float, optional
        Range resolution (meters).
    resolution_azimuth : float, optional
        Azimuth resolution (meters).
    polarizations : List[SIDDTxRcvPolarization], optional
        Polarization list.
    geometry : SIDDCollectionGeometry, optional
        Collection geometry.
    phenomenology : SIDDCollectionPhenomenology, optional
        Collection phenomenology.
    identifier : str, optional
        Collection identifier.
    """

    sensor_name: Optional[str] = None
    radar_mode: Optional[SIDDRadarMode] = None
    collection_date_time: Optional[str] = None
    collection_duration: Optional[float] = None
    resolution_range: Optional[float] = None
    resolution_azimuth: Optional[float] = None
    polarizations: Optional[List[SIDDTxRcvPolarization]] = None
    geometry: Optional[SIDDCollectionGeometry] = None
    phenomenology: Optional[SIDDCollectionPhenomenology] = None
    identifier: Optional[str] = None


@dataclass
class SIDDProductResolution:
    """Product resolution.

    Parameters
    ----------
    row : float, optional
        Row resolution (meters).
    col : float, optional
        Column resolution (meters).
    """

    row: Optional[float] = None
    col: Optional[float] = None


@dataclass
class SIDDExploitationFeaturesProduct:
    """Exploitation features product entry.

    Parameters
    ----------
    resolution : SIDDProductResolution, optional
        Product resolution.
    ellipticity : float, optional
        Ellipticity of the resolution cell.
    north : float, optional
        North direction angle (degrees).
    """

    resolution: Optional[SIDDProductResolution] = None
    ellipticity: Optional[float] = None
    north: Optional[float] = None


@dataclass
class SIDDExploitationFeatures:
    """Exploitation features.

    Parameters
    ----------
    collections : List[SIDDCollectionInfo], optional
        Source collection information.
    products : List[SIDDExploitationFeaturesProduct], optional
        Product exploitation features.
    """

    collections: Optional[List[SIDDCollectionInfo]] = None
    products: Optional[List[SIDDExploitationFeaturesProduct]] = None


# ===================================================================
# Section 6: DownstreamReprocessing
# ===================================================================

@dataclass
class SIDDGeometricChip:
    """Geometric chip parameters.

    Parameters
    ----------
    chip_size : RowCol, optional
        Chip dimensions (rows, cols).
    original_upper_left : RowCol, optional
        Original upper-left corner.
    original_upper_right : RowCol, optional
        Original upper-right corner.
    original_lower_left : RowCol, optional
        Original lower-left corner.
    original_lower_right : RowCol, optional
        Original lower-right corner.
    """

    chip_size: Optional[RowCol] = None
    original_upper_left: Optional[RowCol] = None
    original_upper_right: Optional[RowCol] = None
    original_lower_left: Optional[RowCol] = None
    original_lower_right: Optional[RowCol] = None


@dataclass
class SIDDProcessingEvent:
    """Downstream processing event.

    Parameters
    ----------
    application_name : str, optional
        Application that performed processing.
    applied_date_time : str, optional
        Date/time processing was applied (ISO 8601).
    interpolation_method : str, optional
        Interpolation method used.
    descriptors : Dict[str, str], optional
        Processing descriptors.
    """

    application_name: Optional[str] = None
    applied_date_time: Optional[str] = None
    interpolation_method: Optional[str] = None
    descriptors: Optional[Dict[str, str]] = None


@dataclass
class SIDDDownstreamReprocessing:
    """Downstream reprocessing information.

    Parameters
    ----------
    geometric_chip : SIDDGeometricChip, optional
        Geometric chip parameters.
    processing_events : List[SIDDProcessingEvent], optional
        Processing events.
    """

    geometric_chip: Optional[SIDDGeometricChip] = None
    processing_events: Optional[List[SIDDProcessingEvent]] = None


# ===================================================================
# Section 7: Compression
# ===================================================================

@dataclass
class SIDDJPEG2000Subtype:
    """JPEG 2000 compression subtype.

    Parameters
    ----------
    num_wavelet_levels : int, optional
        Number of wavelet decomposition levels.
    num_bands : int, optional
        Number of image bands.
    """

    num_wavelet_levels: Optional[int] = None
    num_bands: Optional[int] = None


@dataclass
class SIDDCompression:
    """Compression information.

    Parameters
    ----------
    original : SIDDJpeg2000Subtype, optional
        Original JPEG 2000 parameters.
    parsed : SIDDJpeg2000Subtype, optional
        Parsed JPEG 2000 parameters.
    """

    original: Optional[SIDDJPEG2000Subtype] = None
    parsed: Optional[SIDDJPEG2000Subtype] = None


# ===================================================================
# Section 8: DigitalElevationData
# ===================================================================

@dataclass
class SIDDGeographicCoordinates:
    """Geographic coordinates for DEM.

    Parameters
    ----------
    longitude_density : float, optional
        Longitude density.
    latitude_density : float, optional
        Latitude density.
    reference_origin : LatLon, optional
        Reference origin point.
    """

    longitude_density: Optional[float] = None
    latitude_density: Optional[float] = None
    reference_origin: Optional[LatLon] = None


@dataclass
class SIDDGeopositioning:
    """DEM geopositioning parameters.

    Parameters
    ----------
    coordinate_system_type : str, optional
        ``'GCS'`` or ``'UTM'``.
    geodetic_datum : str, optional
        Geodetic datum name.
    reference_ellipsoid : str, optional
        Reference ellipsoid name.
    vertical_datum : str, optional
        Vertical datum name.
    sounding_datum : str, optional
        Sounding datum name.
    false_origin : int, optional
        False origin value.
    utm_grid_zone_number : int, optional
        UTM grid zone number.
    """

    coordinate_system_type: Optional[str] = None
    geodetic_datum: Optional[str] = None
    reference_ellipsoid: Optional[str] = None
    vertical_datum: Optional[str] = None
    sounding_datum: Optional[str] = None
    false_origin: Optional[int] = None
    utm_grid_zone_number: Optional[int] = None


@dataclass
class SIDDPositionalAccuracy:
    """Positional accuracy for DEM.

    Parameters
    ----------
    num_regions : int, optional
        Number of accuracy regions.
    absolute_horizontal : List[float], optional
        Absolute horizontal accuracy per region.
    absolute_vertical : List[float], optional
        Absolute vertical accuracy per region.
    point_to_point_horizontal : List[float], optional
        Point-to-point horizontal accuracy per region.
    point_to_point_vertical : List[float], optional
        Point-to-point vertical accuracy per region.
    """

    num_regions: Optional[int] = None
    absolute_horizontal: Optional[List[float]] = None
    absolute_vertical: Optional[List[float]] = None
    point_to_point_horizontal: Optional[List[float]] = None
    point_to_point_vertical: Optional[List[float]] = None


@dataclass
class SIDDDigitalElevationData:
    """Digital elevation data parameters.

    Parameters
    ----------
    geographic_coordinates : SIDDGeographicCoordinates, optional
        Geographic coordinates.
    geopositioning : SIDDGeopositioning, optional
        Geopositioning parameters.
    positional_accuracy : SIDDPositionalAccuracy, optional
        Positional accuracy.
    null_value : int, optional
        Null/no-data value for DEM.
    """

    geographic_coordinates: Optional[SIDDGeographicCoordinates] = None
    geopositioning: Optional[SIDDGeopositioning] = None
    positional_accuracy: Optional[SIDDPositionalAccuracy] = None
    null_value: Optional[int] = None


# ===================================================================
# Section 9: ProductProcessing
# ===================================================================

@dataclass
class SIDDProcessingModule:
    """Product processing module.

    Parameters
    ----------
    module_name : str, optional
        Processing module name.
    name : str, optional
        Display name.
    parameters : Dict[str, str], optional
        Module parameters.
    """

    module_name: Optional[str] = None
    name: Optional[str] = None
    parameters: Optional[Dict[str, str]] = None


@dataclass
class SIDDProductProcessing:
    """Product processing information.

    Parameters
    ----------
    processing_modules : List[SIDDProcessingModule], optional
        Processing modules applied.
    """

    processing_modules: Optional[List[SIDDProcessingModule]] = None


# ===================================================================
# Section 10: Annotations
# ===================================================================

@dataclass
class SIDDAnnotation:
    """Product annotation entry.

    Parameters
    ----------
    identifier : str, optional
        Annotation identifier.
    spatial_reference_system : str, optional
        Spatial reference system.
    objects : Dict[str, Any], optional
        Annotation objects.
    """

    identifier: Optional[str] = None
    spatial_reference_system: Optional[str] = None
    objects: Optional[Dict[str, Any]] = None


@dataclass
class SIDDAnnotations:
    """Product annotations.

    Parameters
    ----------
    annotations : List[SIDDAnnotation], optional
        Annotation entries.
    """

    annotations: Optional[List[SIDDAnnotation]] = None


# ===================================================================
# Top-level: SIDDMetadata
# ===================================================================

@dataclass
class SIDDMetadata(ImageMetadata):
    """Complete typed metadata for SIDD imagery.

    Contains all 13 sections of the SIDD v2 standard as nested
    dataclasses, plus backend and multi-image tracking fields.
    Inherits from ``ImageMetadata`` for universal fields and
    dict-like access.

    ErrorStatistics, Radiometric, and MatchInfo reuse the
    corresponding SICD types since they share the same NGA schema.

    Parameters
    ----------
    backend : str, optional
        Active reader backend (``'sarkit'``).
    num_images : int, optional
        Number of product images in the file.
    image_index : int, optional
        Active product image index.
    product_creation : SIDDProductCreation, optional
    display : SIDDDisplay, optional
    geo_data : SIDDGeoData, optional
    measurement : SIDDMeasurement, optional
    exploitation_features : SIDDExploitationFeatures, optional
    downstream_reprocessing : SIDDDownstreamReprocessing, optional
    error_statistics : SICDErrorStatistics, optional
        Reuses SICD error statistics type.
    radiometric : SICDRadiometric, optional
        Reuses SICD radiometric type.
    match_info : SICDMatchInfo, optional
        Reuses SICD match info type.
    compression : SIDDCompression, optional
    digital_elevation_data : SIDDDigitalElevationData, optional
    product_processing : SIDDProductProcessing, optional
    annotations : SIDDAnnotations, optional

    Examples
    --------
    >>> meta = reader.metadata  # SIDDMetadata
    >>> meta.display.pixel_type
    'MONO8I'
    >>> meta.exploitation_features.collections[0].sensor_name
    'SENSOR_X'
    >>> meta.measurement.plane_projection.sample_spacing.row
    0.5
    """

    backend: Optional[str] = None
    num_images: Optional[int] = None
    image_index: Optional[int] = None
    product_creation: Optional[SIDDProductCreation] = None
    display: Optional[SIDDDisplay] = None
    geo_data: Optional[SIDDGeoData] = None
    measurement: Optional[SIDDMeasurement] = None
    exploitation_features: Optional[SIDDExploitationFeatures] = None
    downstream_reprocessing: Optional[SIDDDownstreamReprocessing] = None
    error_statistics: Optional[SICDErrorStatistics] = None
    radiometric: Optional[SICDRadiometric] = None
    match_info: Optional[SICDMatchInfo] = None
    compression: Optional[SIDDCompression] = None
    digital_elevation_data: Optional[SIDDDigitalElevationData] = None
    product_processing: Optional[SIDDProductProcessing] = None
    annotations: Optional[SIDDAnnotations] = None
