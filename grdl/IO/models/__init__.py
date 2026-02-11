# -*- coding: utf-8 -*-
"""
IO Models - Typed metadata containers for imagery readers.

Re-exports all metadata classes from submodules for convenient access:

    from grdl.IO.models import ImageMetadata, SICDMetadata, SIDDMetadata

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

# Base
from grdl.IO.models.base import ImageMetadata

# Common primitives
from grdl.IO.models.common import (
    XYZ,
    LatLon,
    LatLonHAE,
    RowCol,
    Poly1D,
    Poly2D,
    XYZPoly,
)

# SICD
from grdl.IO.models.sicd import (
    SICDMetadata,
    SICDRadarMode,
    SICDCollectionInfo,
    SICDImageCreation,
    SICDFullImage,
    SICDImageData,
    SICDSCP,
    SICDGeoData,
    SICDWgtType,
    SICDDirParam,
    SICDGrid,
    SICDIPPSet,
    SICDTimeline,
    SICDPosition,
    SICDTxFrequency,
    SICDWaveformParams,
    SICDRcvChannel,
    SICDAreaCorner,
    SICDArea,
    SICDRadarCollection,
    SICDRcvChanProc,
    SICDTxFrequencyProc,
    SICDProcessingStep,
    SICDImageFormation,
    SICDSCPCOA,
    SICDNoiseLevel,
    SICDRadiometric,
    SICDEBType,
    SICDGainPhasePoly,
    SICDAntParam,
    SICDAntenna,
    SICDCompositeSCPError,
    SICDCorrCoefs,
    SICDErrorDecorrFunc,
    SICDPosVelErr,
    SICDRadarSensorError,
    SICDErrorStatistics,
    SICDMatchCollection,
    SICDMatchType,
    SICDMatchInfo,
    SICDRgAzComp,
    SICDSTDeskew,
    SICDPFA,
    SICDRMRef,
    SICDINCA,
    SICDRMA,
)

# SIDD
from grdl.IO.models.sidd import (
    SIDDMetadata,
    SIDDProcessorInformation,
    SIDDClassification,
    SIDDProductCreation,
    SIDDDRAParameters,
    SIDDDRAOverrides,
    SIDDDynamicRangeAdjustment,
    SIDDDisplay,
    SIDDGeoData,
    SIDDReferencePoint,
    SIDDProductPlane,
    SIDDPlaneProjection,
    SIDDMeasurement,
    SIDDRadarMode,
    SIDDTxRcvPolarization,
    SIDDCollectionGeometry,
    SIDDAngleMagnitude,
    SIDDCollectionPhenomenology,
    SIDDCollectionInfo,
    SIDDProductResolution,
    SIDDExploitationFeaturesProduct,
    SIDDExploitationFeatures,
    SIDDGeometricChip,
    SIDDProcessingEvent,
    SIDDDownstreamReprocessing,
    SIDDJPEG2000Subtype,
    SIDDCompression,
    SIDDGeographicCoordinates,
    SIDDGeopositioning,
    SIDDPositionalAccuracy,
    SIDDDigitalElevationData,
    SIDDProcessingModule,
    SIDDProductProcessing,
    SIDDAnnotation,
    SIDDAnnotations,
)

# BIOMASS
from grdl.IO.models.biomass import BIOMASSMetadata

# VIIRS
from grdl.IO.models.viirs import VIIRSMetadata

# ASTER
from grdl.IO.models.aster import ASTERMetadata

__all__ = [
    # Base
    'ImageMetadata',
    # Common
    'XYZ',
    'LatLon',
    'LatLonHAE',
    'RowCol',
    'Poly1D',
    'Poly2D',
    'XYZPoly',
    # SICD
    'SICDMetadata',
    'SICDRadarMode',
    'SICDCollectionInfo',
    'SICDImageCreation',
    'SICDFullImage',
    'SICDImageData',
    'SICDSCP',
    'SICDGeoData',
    'SICDWgtType',
    'SICDDirParam',
    'SICDGrid',
    'SICDIPPSet',
    'SICDTimeline',
    'SICDPosition',
    'SICDTxFrequency',
    'SICDWaveformParams',
    'SICDRcvChannel',
    'SICDAreaCorner',
    'SICDArea',
    'SICDRadarCollection',
    'SICDRcvChanProc',
    'SICDTxFrequencyProc',
    'SICDProcessingStep',
    'SICDImageFormation',
    'SICDSCPCOA',
    'SICDNoiseLevel',
    'SICDRadiometric',
    'SICDEBType',
    'SICDGainPhasePoly',
    'SICDAntParam',
    'SICDAntenna',
    'SICDCompositeSCPError',
    'SICDCorrCoefs',
    'SICDErrorDecorrFunc',
    'SICDPosVelErr',
    'SICDRadarSensorError',
    'SICDErrorStatistics',
    'SICDMatchCollection',
    'SICDMatchType',
    'SICDMatchInfo',
    'SICDRgAzComp',
    'SICDSTDeskew',
    'SICDPFA',
    'SICDRMRef',
    'SICDINCA',
    'SICDRMA',
    # SIDD
    'SIDDMetadata',
    'SIDDProcessorInformation',
    'SIDDClassification',
    'SIDDProductCreation',
    'SIDDDRAParameters',
    'SIDDDRAOverrides',
    'SIDDDynamicRangeAdjustment',
    'SIDDDisplay',
    'SIDDGeoData',
    'SIDDReferencePoint',
    'SIDDProductPlane',
    'SIDDPlaneProjection',
    'SIDDMeasurement',
    'SIDDRadarMode',
    'SIDDTxRcvPolarization',
    'SIDDCollectionGeometry',
    'SIDDAngleMagnitude',
    'SIDDCollectionPhenomenology',
    'SIDDCollectionInfo',
    'SIDDProductResolution',
    'SIDDExploitationFeaturesProduct',
    'SIDDExploitationFeatures',
    'SIDDGeometricChip',
    'SIDDProcessingEvent',
    'SIDDDownstreamReprocessing',
    'SIDDJPEG2000Subtype',
    'SIDDCompression',
    'SIDDGeographicCoordinates',
    'SIDDGeopositioning',
    'SIDDPositionalAccuracy',
    'SIDDDigitalElevationData',
    'SIDDProcessingModule',
    'SIDDProductProcessing',
    'SIDDAnnotation',
    'SIDDAnnotations',
    # BIOMASS
    'BIOMASSMetadata',
    # VIIRS
    'VIIRSMetadata',
    # ASTER
    'ASTERMetadata',
]
