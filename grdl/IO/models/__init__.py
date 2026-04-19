# -*- coding: utf-8 -*-
"""
IO Models - Typed metadata containers for imagery readers.

Re-exports all metadata classes from submodules for convenient access:

    from grdl.IO.models import ImageMetadata, SICDMetadata, SIDDMetadata

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
2026-02-10

Modified
--------
2026-03-29
"""

# Base
from grdl.IO.models.base import ImageMetadata, ChannelMetadata

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

# Sentinel-2
from grdl.IO.models.sentinel2 import (
    Sentinel2Metadata,
    S2ProductInfo,
    S2QualityIndicators,
    S2TileGeocoding,
    S2AngleGrid,
    S2MeanAngles,
    S2RadiometricInfo,
    S2SpectralBand,
    S2Footprint,
)

# Sentinel-1 SLC
from grdl.IO.models.sentinel1_slc import (
    Sentinel1SLCMetadata,
    S1SLCProductInfo,
    S1SLCSwathInfo,
    S1SLCBurst,
    S1SLCOrbitStateVector,
    S1SLCGeoGridPoint,
    S1SLCDopplerCentroid,
    S1SLCDopplerFmRate,
    S1SLCCalibrationVector,
    S1SLCNoiseRangeVector,
    S1SLCNoiseAzimuthVector,
)

# Sentinel-1 L0
from grdl.IO.models.sentinel1_l0 import (
    Sentinel1L0Metadata,
    Sentinel1Mission,
    Sentinel1Mode,
    Sentinel1Polarization,
    S1L0SwathID,
    S1L0OrbitStateVector,
    S1L0AttitudeRecord,
    S1L0RadarParameters,
    S1L0DownlinkInfo,
    S1L0BurstRecord,
    S1L0SwathParameters,
    S1L0InstrumentTiming,
    S1L0GeolocationGrid,
)

# TerraSAR-X / TanDEM-X
from grdl.IO.models.terrasar import (
    TerraSARMetadata,
    TSXProductInfo,
    TSXSceneInfo,
    TSXImageInfo,
    TSXRadarParams,
    TSXOrbitStateVector,
    TSXGeoGridPoint,
    TSXCalibration,
    TSXDopplerInfo,
    TSXProcessingInfo,
)

# CPHD
from grdl.IO.models.cphd import (
    CPHDMetadata,
    CPHDChannel,
    CPHDPVP,
    CPHDGlobal,
    CPHDCollectionInfo,
    CPHDTxWaveform,
    CPHDRcvParameters,
    CPHDAntennaPattern,
    CPHDSceneCoordinates,
    CPHDReferenceGeometry,
    CPHDDwellPolynomial,
    create_subaperture_metadata,
)

# CRSD
from grdl.IO.models.crsd import (
    CRSDMetadata,
    CRSDChannelParameters,
    CRSDDwellPolynomialSet,
    CRSDReferenceGeometry,
)

# NISAR
from grdl.IO.models.nisar import (
    NISARMetadata,
    NISARIdentification,
    NISAROrbit,
    NISARAttitude,
    NISARSwathParameters,
    NISARGridParameters,
    NISARGeolocationGrid,
    NISARCalibration,
    NISARDopplerCentroid,
    NISARProcessingInfo,
)

# EO NITF
from grdl.IO.models.eo_nitf import (
    EONITFMetadata,
    RPCCoefficients,
    RSMIdentification,
    RSMCoefficients,
    RSMSegmentGrid,
    CSEPHAMetadata,
    RSMGGAMetadata,
    RSMGGAGridPlane,
)

__all__ = [
    # Base
    'ImageMetadata',
    'ChannelMetadata',
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
    # Sentinel-2
    'Sentinel2Metadata',
    'S2ProductInfo',
    'S2QualityIndicators',
    'S2TileGeocoding',
    'S2AngleGrid',
    'S2MeanAngles',
    'S2RadiometricInfo',
    'S2SpectralBand',
    'S2Footprint',
    # Sentinel-1 SLC
    'Sentinel1SLCMetadata',
    'S1SLCProductInfo',
    'S1SLCSwathInfo',
    'S1SLCBurst',
    'S1SLCOrbitStateVector',
    'S1SLCGeoGridPoint',
    'S1SLCDopplerCentroid',
    'S1SLCDopplerFmRate',
    'S1SLCCalibrationVector',
    'S1SLCNoiseRangeVector',
    'S1SLCNoiseAzimuthVector',
    # Sentinel-1 L0
    'Sentinel1L0Metadata',
    'Sentinel1Mission',
    'Sentinel1Mode',
    'Sentinel1Polarization',
    'S1L0SwathID',
    'S1L0OrbitStateVector',
    'S1L0AttitudeRecord',
    'S1L0RadarParameters',
    'S1L0DownlinkInfo',
    'S1L0BurstRecord',
    'S1L0SwathParameters',
    'S1L0InstrumentTiming',
    'S1L0GeolocationGrid',
    # TerraSAR-X / TanDEM-X
    'TerraSARMetadata',
    'TSXProductInfo',
    'TSXSceneInfo',
    'TSXImageInfo',
    'TSXRadarParams',
    'TSXOrbitStateVector',
    'TSXGeoGridPoint',
    'TSXCalibration',
    'TSXDopplerInfo',
    'TSXProcessingInfo',
    # CPHD
    'CPHDMetadata',
    'CRSDMetadata',
    'CRSDChannelParameters',
    'CRSDDwellPolynomialSet',
    'CRSDReferenceGeometry',
    'CPHDChannel',
    'CPHDPVP',
    'CPHDGlobal',
    'CPHDCollectionInfo',
    'CPHDTxWaveform',
    'CPHDRcvParameters',
    'CPHDAntennaPattern',
    'CPHDSceneCoordinates',
    'CPHDReferenceGeometry',
    'CPHDDwellPolynomial',
    'create_subaperture_metadata',
    # NISAR
    'NISARMetadata',
    'NISARIdentification',
    'NISAROrbit',
    'NISARAttitude',
    'NISARSwathParameters',
    'NISARGridParameters',
    'NISARGeolocationGrid',
    'NISARCalibration',
    'NISARDopplerCentroid',
    'NISARProcessingInfo',
    # EO NITF
    'EONITFMetadata',
    'RPCCoefficients',
    'RSMIdentification',
    'RSMCoefficients',
    'RSMSegmentGrid',
    'CSEPHAMetadata',
    'RSMGGAMetadata',
    'RSMGGAGridPlane',
]
