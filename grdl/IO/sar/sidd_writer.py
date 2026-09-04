# -*- coding: utf-8 -*-
"""
SIDD Writer - Write derived SAR products in SIDD NITF format.

Converts GRDL's typed ``SIDDMetadata`` to sarpy's ``SIDDType`` v3 and
writes the product image data via sarpy's NITF writer backend.  All
twelve SIDD 3.0 sections are converted -- the five the schema requires
(ProductCreation, Display, GeoData, Measurement, ExploitationFeatures)
plus DownstreamReprocessing, ErrorStatistics, Radiometric, MatchInfo,
DigitalElevationData and ProductProcessing.  Metadata produced by
:func:`~grdl.IO.sar.sidd_builder.build_sidd_metadata` or read back from
an existing SIDD therefore round-trips without loss.  Compression is
not converted: these products are written uncompressed, so declaring a
J2K block would misdescribe the file.

Supported pixel types are ``MONO8I`` (uint8), ``MONO16I`` (uint16) and
``RGB24I`` (three-band uint8).  Multi-band arrays are accepted with the
band axis either leading (``(3, rows, cols)``, the orthorectifier's
layout) or trailing (``(rows, cols, 3)``); the axis is identified by
matching against the product's declared pixel footprint.

Dependencies
------------
sarpy

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
2026-03-07

Modified
--------
2026-09-03  Complete the GRDL -> sarpy metadata conversion (GeoData,
            Measurement, ExploitationFeatures, full Display), add band
            and dtype handling, and validate before constructing the
            sarpy writer so a rejected structure cannot leak.
"""

from __future__ import annotations

# Standard library
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.exceptions import DependencyError, ValidationError
from grdl.IO.base import ImageWriter
from grdl.IO.models import SIDDMetadata
from grdl.IO.models.common import LatLon, Poly2D, RowCol, XYZ, XYZPoly

try:
    from sarpy.io.product.sidd import (
        SIDDWriter as _SarpySIDDWriter,
        validate_sidd_for_writing as _validate_sidd,
    )
    from sarpy.io.product.sidd3_elements.SIDD import SIDDType
    from sarpy.io.product.sidd3_elements.ProductCreation import (
        ProcessorInformationType,
        ProductClassificationType,
        ProductCreationType,
    )
    from sarpy.io.product.sidd3_elements.Display import (
        DynamicRangeAdjustmentType,
        GeometricTransformType,
        InteractiveProcessingType,
        NonInteractiveProcessingType,
        OrientationType,
        ProductDisplayType,
        ProductGenerationOptionsType,
        RRDSType,
        ScalingType,
        SharpnessEnhancementType,
    )
    from sarpy.io.product.sidd3_elements.GeoData import GeoDataType
    from sarpy.io.product.sidd3_elements.Measurement import (
        GeographicProjectionType,
        MeasurementType,
        PlaneProjectionType,
        PolynomialProjectionType,
        ProductPlaneType,
    )
    from sarpy.io.product.sidd3_elements.ExploitationFeatures import (
        CollectionType,
        ExploitationFeaturesCollectionGeometryType,
        ExploitationFeaturesCollectionInformationType,
        ExploitationFeaturesCollectionPhenomenologyType,
        ExploitationFeaturesProductType,
        ExploitationFeaturesType,
        InputROIType,
        ProcTxRcvPolarizationType,
        TxRcvPolarizationType,
    )
    from sarpy.io.product.sidd3_elements.blocks import (
        AngleZeroToExclusive360MagnitudeType,
        ErrorStatisticsType,
        FilterBankType,
        FilterType,
        MatchInfoType,
        NewLookupTableType,
        Poly2DType,
        PredefinedFilterType,
        PredefinedLookupType,
        RadarModeType,
        RadiometricType,
        RangeAzimuthType,
        ReferencePointType,
        RowColDoubleType,
        XYZPolyType,
        XYZType,
    )
    from sarpy.io.product.sidd3_elements.DigitalElevationData import (
        DigitalElevationDataType,
        GeographicCoordinatesType,
        GeopositioningType,
        PositionalAccuracyType,
    )
    from sarpy.io.product.sidd3_elements.DownstreamReprocessing import (
        DownstreamReprocessingType,
        GeometricChipType,
        ProcessingEventType,
    )
    from sarpy.io.product.sidd3_elements.ProductProcessing import (
        ProcessingModuleType,
        ProductProcessingType,
    )
    from sarpy.io.complex.sicd_elements.ErrorStatistics import (
        CompositeSCPErrorType,
        CorrCoefsType,
        ErrorComponentsType,
        ErrorDecorrFuncType,
        IonoErrorType,
        PosVelErrType,
        RadarSensorErrorType,
        TropoErrorType,
        UnmodeledDecorrType,
        UnmodeledType,
    )
    from sarpy.io.complex.sicd_elements.MatchInfo import (
        MatchCollectionType,
        MatchType,
    )
    from sarpy.io.complex.sicd_elements.Radiometric import NoiseLevelType_
    from sarpy.io.xml.base import create_text_node as _create_text_node
    _HAS_SARPY_SIDD = True
except ImportError:
    _HAS_SARPY_SIDD = False

logger = logging.getLogger(__name__)

# Bands and numpy dtype required by each SIDD pixel type.
_PIXEL_LAYOUT = {
    'MONO8I': (1, np.uint8),
    'MONO16I': (1, np.uint16),
    'RGB24I': (3, np.uint8),
}


# ===================================================================
# Small conversion helpers: GRDL common types -> sarpy blocks
# ===================================================================

def _xyz(value: Optional[XYZ]) -> Optional['XYZType']:
    """Convert a GRDL ``XYZ`` to a sarpy ``XYZType``."""
    if value is None:
        return None
    return XYZType(X=value.x, Y=value.y, Z=value.z)


def _row_col(value: Optional[RowCol]) -> Optional['RowColDoubleType']:
    """Convert a GRDL ``RowCol`` to a sarpy ``RowColDoubleType``."""
    if value is None:
        return None
    return RowColDoubleType(Row=value.row, Col=value.col)


def _poly2d(value: Optional[Poly2D]) -> Optional['Poly2DType']:
    """Convert a GRDL ``Poly2D`` to a sarpy ``Poly2DType``."""
    if value is None or value.coefs is None:
        return None
    return Poly2DType(Coefs=np.asarray(value.coefs, dtype=np.float64))


def _xyz_poly(value: Optional[XYZPoly]) -> Optional['XYZPolyType']:
    """Convert a GRDL ``XYZPoly`` to a sarpy ``XYZPolyType``."""
    if value is None:
        return None
    if value.x is None or value.y is None or value.z is None:
        return None
    return XYZPolyType(
        X=np.asarray(value.x.coefs, dtype=np.float64),
        Y=np.asarray(value.y.coefs, dtype=np.float64),
        Z=np.asarray(value.z.coefs, dtype=np.float64),
    )


def _latlon_array(
    corners: Optional[List[LatLon]],
) -> Optional[np.ndarray]:
    """Stack a list of ``LatLon`` into an ``(N, 2)`` array."""
    if not corners:
        return None
    return np.array(
        [[c.lat, c.lon] for c in corners], dtype=np.float64,
    )


def _reference_point(value: Any) -> Optional['ReferencePointType']:
    """Convert a GRDL ``SIDDReferencePoint`` to sarpy's type."""
    if value is None or value.ecef is None or value.point is None:
        return None
    return ReferencePointType(
        ECEF=value.ecef.to_array(),
        Point=(value.point.row, value.point.col),
        name=value.name,
    )


# ===================================================================
# Section converters
# ===================================================================

def _convert_product_creation(pc: Any) -> Optional['ProductCreationType']:
    """Build sarpy ``ProductCreationType`` from the GRDL model.

    Parameters
    ----------
    pc : SIDDProductCreation or None
        GRDL product creation block.

    Returns
    -------
    ProductCreationType or None
        None when the input is None.
    """
    if pc is None:
        return None

    proc_info = None
    pi = pc.processor_information
    if pi is not None:
        proc_info = ProcessorInformationType(
            Application=pi.application,
            ProcessingDateTime=pi.processing_date_time,
            Site=pi.site,
            Profile=pi.profile,
        )

    cls = pc.classification
    classification = ProductClassificationType(
        DESVersion=(cls.des_version if cls and cls.des_version else 13),
        createDate=(cls.create_date if cls else None),
        classification=(cls.classification if cls else 'U') or 'U',
        ownerProducer=(cls.owner_producer if cls else 'USA') or 'USA',
        compliesWith='USGov',
        ISMCATCESVersion='201903',
        SCIcontrols=(cls.sci_controls if cls else None),
        SARIdentifier=(cls.sar_identifier if cls else None),
        disseminationControls=(
            cls.dissemination_controls if cls else None
        ),
        releasableTo=(cls.releasable_to if cls else None),
        classifiedBy=(cls.classified_by if cls else None),
        derivedFrom=(cls.derived_from if cls else None),
        declassDate=(cls.declass_date if cls else None),
        declassEvent=(cls.declass_event if cls else None),
    )

    return ProductCreationType(
        ProcessorInformation=proc_info,
        Classification=classification,
        ProductName=pc.product_name,
        ProductClass=pc.product_class,
        ProductType=pc.product_type,
    )


def _default_non_interactive(band: int) -> 'NonInteractiveProcessingType':
    """Default non-interactive processing block for one display band."""
    return NonInteractiveProcessingType(
        ProductGenerationOptions=ProductGenerationOptionsType(
            DataRemapping=NewLookupTableType(
                LUTName='DENSITY',
                Predefined=PredefinedLookupType(DatabaseName='DENSITY'),
            ),
        ),
        RRDS=RRDSType(DownsamplingMethod='DECIMATE'),
        band=band,
    )


def _default_interactive(
    band: int,
    dra: Any = None,
) -> 'InteractiveProcessingType':
    """Default interactive processing block for one display band.

    Parameters
    ----------
    band : int
        1-based band index.
    dra : SIDDDynamicRangeAdjustment or None
        Dynamic range adjustment carried by the GRDL model, if any.
    """
    bilinear = FilterBankType(
        Predefined=PredefinedFilterType(DatabaseName='BILINEAR'),
    )
    algorithm = 'NONE'
    band_stats_source = band
    if dra is not None:
        algorithm = dra.algorithm_type or 'NONE'
        if dra.band_stats_source is not None:
            band_stats_source = dra.band_stats_source

    return InteractiveProcessingType(
        GeometricTransform=GeometricTransformType(
            Scaling=ScalingType(
                AntiAlias=FilterType(
                    FilterName='AntiAlias',
                    FilterBank=bilinear,
                    Operation='CONVOLUTION',
                ),
                Interpolation=FilterType(
                    FilterName='Interpolation',
                    FilterBank=bilinear,
                    Operation='CORRELATION',
                ),
            ),
            Orientation=OrientationType(ShadowDirection='ARBITRARY'),
        ),
        SharpnessEnhancement=SharpnessEnhancementType(
            ModularTransferFunctionEnhancement=FilterType(
                FilterName='ModularTransferFunctionEnhancement',
                FilterBank=bilinear,
                Operation='CONVOLUTION',
            ),
        ),
        DynamicRangeAdjustment=DynamicRangeAdjustmentType(
            AlgorithmType=algorithm,
            BandStatsSource=band_stats_source,
        ),
        band=band,
    )


def _convert_display(display: Any) -> 'ProductDisplayType':
    """Build sarpy ``ProductDisplayType`` from the GRDL model.

    The SIDD 3.0 schema requires a NonInteractiveProcessing and an
    InteractiveProcessing block per display band.  The GRDL model does
    not carry those, so standard density-remap / bilinear blocks are
    generated for each band.

    Parameters
    ----------
    display : SIDDDisplay or None
        GRDL display block.

    Returns
    -------
    ProductDisplayType
        Populated display block.

    Raises
    ------
    ValidationError
        If the pixel type is unsupported.
    """
    pixel_type = (display.pixel_type if display else None) or 'MONO8I'
    if pixel_type not in _PIXEL_LAYOUT:
        raise ValidationError(
            f"Unsupported SIDD pixel type '{pixel_type}'. "
            f"Must be one of {sorted(_PIXEL_LAYOUT)}"
        )
    bands = _PIXEL_LAYOUT[pixel_type][0]

    dra = display.dynamic_range_adjustment if display else None
    return ProductDisplayType(
        PixelType=pixel_type,
        NumBands=bands,
        DefaultBandDisplay=(
            display.default_band_display if display else None
        ),
        NonInteractiveProcessing=[
            _default_non_interactive(i + 1) for i in range(bands)
        ],
        InteractiveProcessing=[
            _default_interactive(i + 1, dra) for i in range(bands)
        ],
    )


def _convert_geo_data(geo_data: Any) -> Optional['GeoDataType']:
    """Build sarpy ``GeoDataType`` from the GRDL model.

    Parameters
    ----------
    geo_data : SIDDGeoData or None
        GRDL geographic block.

    Returns
    -------
    GeoDataType or None
        None when no image corners are available -- sarpy needs them to
        write the NITF IGEOLO fields.
    """
    if geo_data is None:
        return None
    corners = _latlon_array(geo_data.image_corners)
    if corners is None:
        return None
    valid = _latlon_array(geo_data.valid_data)
    return GeoDataType(
        EarthModel=geo_data.earth_model or 'WGS_84',
        ImageCorners=corners,
        ValidData=(valid if valid is not None else corners),
    )


def _convert_measurement(meas: Any) -> Optional['MeasurementType']:
    """Build sarpy ``MeasurementType`` from the GRDL model.

    Converts whichever projection the GRDL model carries -- plane,
    geographic or polynomial -- along with the pixel footprint, ARP
    polynomial and valid data polygon.

    Parameters
    ----------
    meas : SIDDMeasurement or None
        GRDL measurement block.

    Returns
    -------
    MeasurementType or None
        None when the input is None.
    """
    if meas is None:
        return None

    plane = None
    pp = meas.plane_projection
    if pp is not None:
        product_plane = None
        if pp.product_plane is not None:
            product_plane = ProductPlaneType(
                RowUnitVector=(
                    pp.product_plane.row_unit_vector.to_array()
                    if pp.product_plane.row_unit_vector else None
                ),
                ColUnitVector=(
                    pp.product_plane.col_unit_vector.to_array()
                    if pp.product_plane.col_unit_vector else None
                ),
            )
        plane = PlaneProjectionType(
            ReferencePoint=_reference_point(pp.reference_point),
            SampleSpacing=_row_col(pp.sample_spacing),
            TimeCOAPoly=_poly2d(pp.time_coa_poly),
            ProductPlane=product_plane,
        )

    geographic = None
    gp = meas.geographic_projection
    if gp is not None:
        geographic = GeographicProjectionType(
            ReferencePoint=_reference_point(gp.reference_point),
            SampleSpacing=_row_col(gp.sample_spacing),
            TimeCOAPoly=_poly2d(gp.time_coa_poly),
        )

    polynomial = None
    yp = meas.polynomial_projection
    if yp is not None:
        polynomial = PolynomialProjectionType(
            ReferencePoint=_reference_point(yp.reference_point),
            RowColToLat=_poly2d(yp.row_col_to_lat),
            RowColToLon=_poly2d(yp.row_col_to_lon),
            RowColToAlt=_poly2d(yp.row_col_to_alt),
            LatLonToRow=_poly2d(yp.lat_lon_to_row),
            LatLonToCol=_poly2d(yp.lat_lon_to_col),
        )

    footprint = None
    if meas.pixel_footprint is not None:
        footprint = (
            int(meas.pixel_footprint.row), int(meas.pixel_footprint.col),
        )

    valid_data = None
    if meas.valid_data:
        valid_data = tuple(
            (int(round(v.row)), int(round(v.col))) for v in meas.valid_data
        )

    return MeasurementType(
        PlaneProjection=plane,
        GeographicProjection=geographic,
        PolynomialProjection=polynomial,
        PixelFootprint=footprint,
        ARPFlag=meas.arp_flag,
        ARPPoly=_xyz_poly(meas.arp_poly),
        ValidData=valid_data,
    )


def _convert_exploitation(ef: Any) -> Optional['ExploitationFeaturesType']:
    """Build sarpy ``ExploitationFeaturesType`` from the GRDL model.

    Parameters
    ----------
    ef : SIDDExploitationFeatures or None
        GRDL exploitation features block.

    Returns
    -------
    ExploitationFeaturesType or None
        None when the input carries no collections.
    """
    if ef is None or not ef.collections:
        return None

    collections = []
    for idx, coll in enumerate(ef.collections):
        radar_mode = None
        if coll.radar_mode is not None:
            radar_mode = RadarModeType(
                ModeType=coll.radar_mode.mode_type,
                ModeID=coll.radar_mode.mode_id,
            )

        resolution = None
        if (coll.resolution_range is not None
                and coll.resolution_azimuth is not None):
            resolution = RangeAzimuthType(
                Range=coll.resolution_range,
                Azimuth=coll.resolution_azimuth,
            )

        polarizations = None
        if coll.polarizations:
            polarizations = [
                TxRcvPolarizationType(
                    TxPolarization=p.tx_polarization,
                    RcvPolarization=p.rcv_polarization,
                )
                for p in coll.polarizations
            ]

        geometry = None
        if coll.geometry is not None:
            g = coll.geometry
            geometry = ExploitationFeaturesCollectionGeometryType(
                Azimuth=g.azimuth,
                Slope=g.slope,
                Squint=g.squint,
                Graze=g.graze,
                Tilt=g.tilt,
                DopplerConeAngle=g.doppler_cone_angle,
            )

        phenomenology = None
        if coll.phenomenology is not None:
            ph = coll.phenomenology
            phenomenology = ExploitationFeaturesCollectionPhenomenologyType(
                Shadow=_angle_magnitude(ph.shadow),
                Layover=_angle_magnitude(ph.layover),
                MultiPath=ph.multi_path,
                GroundTrack=ph.ground_track,
            )

        collections.append(CollectionType(
            identifier=coll.identifier or f'collection_{idx}',
            Information=ExploitationFeaturesCollectionInformationType(
                SensorName=coll.sensor_name,
                RadarMode=radar_mode,
                CollectionDateTime=coll.collection_date_time,
                CollectionDuration=coll.collection_duration,
                Resolution=resolution,
                InputROI=_convert_input_roi(coll.input_roi),
                Polarizations=polarizations,
            ),
            Geometry=geometry,
            Phenomenology=phenomenology,
        ))

    products = []
    for prod in (ef.products or []):
        resolution = None
        if prod.resolution is not None:
            resolution = RowColDoubleType(
                Row=prod.resolution.row, Col=prod.resolution.col,
            )
        proc_pols = None
        if prod.polarizations:
            proc_pols = [
                ProcTxRcvPolarizationType(
                    TxPolarizationProc=p.tx_polarization,
                    RcvPolarizationProc=p.rcv_polarization,
                )
                for p in prod.polarizations
            ]
        products.append(ExploitationFeaturesProductType(
            Resolution=resolution,
            Ellipticity=prod.ellipticity,
            Polarizations=proc_pols,
            North=prod.north,
        ))

    return ExploitationFeaturesType(
        Collections=collections, Products=products,
    )


def _angle_magnitude(
    value: Any,
) -> Optional['AngleZeroToExclusive360MagnitudeType']:
    """Convert a GRDL ``SIDDAngleMagnitude`` to sarpy's type."""
    if value is None or value.angle is None or value.magnitude is None:
        return None
    return AngleZeroToExclusive360MagnitudeType(
        Angle=value.angle, Magnitude=value.magnitude,
    )


def _convert_input_roi(roi: Any) -> Optional['InputROIType']:
    """Convert a GRDL ``SIDDInputROI`` to sarpy's type.

    Parameters
    ----------
    roi : SIDDInputROI or None
        Source image region the product was built from.

    Returns
    -------
    InputROIType or None
        None when the size or upper-left corner is missing; both are
        required by the schema.
    """
    if roi is None or roi.size is None or roi.upper_left is None:
        return None
    return InputROIType(
        Size=(int(roi.size.row), int(roi.size.col)),
        UpperLeft=(int(roi.upper_left.row), int(roi.upper_left.col)),
    )


def _convert_radiometric(rad: Any) -> Optional['RadiometricType']:
    """Convert GRDL radiometric calibration to sarpy's SIDD type.

    The polynomials must already be expressed in product pixel
    coordinates; :func:`~grdl.IO.sar.sidd_builder.build_sidd_metadata`
    refits them from the source before they reach here.

    Parameters
    ----------
    rad : SICDRadiometric or None
        Radiometric block.

    Returns
    -------
    RadiometricType or None
        None when the input is None.
    """
    if rad is None:
        return None

    noise_level = None
    if rad.noise_level is not None:
        noise_poly = _poly2d(rad.noise_level.noise_poly)
        if noise_poly is not None:
            noise_level = NoiseLevelType_(
                NoiseLevelType=rad.noise_level.noise_level_type,
                NoisePoly=noise_poly,
            )

    return RadiometricType(
        NoiseLevel=noise_level,
        RCSSFPoly=_poly2d(rad.rcs_sf_poly),
        SigmaZeroSFPoly=_poly2d(rad.sigma_zero_sf_poly),
        BetaZeroSFPoly=_poly2d(rad.beta_zero_sf_poly),
        GammaZeroSFPoly=_poly2d(rad.gamma_zero_sf_poly),
    )


def _convert_decorr(func: Any) -> Optional['ErrorDecorrFuncType']:
    """Convert a GRDL error decorrelation function to sarpy's type."""
    if func is None:
        return None
    if func.corr_coef_zero is None or func.decorr_rate is None:
        return None
    return ErrorDecorrFuncType(
        CorrCoefZero=func.corr_coef_zero, DecorrRate=func.decorr_rate,
    )


def _convert_error_statistics(err: Any) -> Optional['ErrorStatisticsType']:
    """Convert GRDL error statistics to sarpy's SIDD type.

    Error statistics describe the collection geometry rather than the
    image grid, so they carry across from the source SICD unchanged.

    Parameters
    ----------
    err : SICDErrorStatistics or None
        Error statistics block.

    Returns
    -------
    ErrorStatisticsType or None
        None when the input is None or carries no populated section.
    """
    if err is None:
        return None

    composite = None
    cs = err.composite_scp
    if cs is not None and None not in (cs.rg, cs.az, cs.rg_az):
        composite = CompositeSCPErrorType(
            Rg=cs.rg, Az=cs.az, RgAz=cs.rg_az,
        )

    pos_vel = None
    pv = err.monostatic
    if pv is not None and None not in (
            pv.frame, pv.p1, pv.p2, pv.p3, pv.v1, pv.v2, pv.v3):
        corr = None
        cc = pv.corr_coefs
        if cc is not None:
            fields = {
                'P1P2': cc.p1p2, 'P1P3': cc.p1p3, 'P1V1': cc.p1v1,
                'P1V2': cc.p1v2, 'P1V3': cc.p1v3, 'P2P3': cc.p2p3,
                'P2V1': cc.p2v1, 'P2V2': cc.p2v2, 'P2V3': cc.p2v3,
                'P3V1': cc.p3v1, 'P3V2': cc.p3v2, 'P3V3': cc.p3v3,
                'V1V2': cc.v1v2, 'V1V3': cc.v1v3, 'V2V3': cc.v2v3,
            }
            if all(v is not None for v in fields.values()):
                corr = CorrCoefsType(**fields)
        pos_vel = PosVelErrType(
            Frame=pv.frame,
            P1=pv.p1, P2=pv.p2, P3=pv.p3,
            V1=pv.v1, V2=pv.v2, V3=pv.v3,
            CorrCoefs=corr,
            PositionDecorr=_convert_decorr(pv.position_decorr),
        )

    radar_sensor = None
    rs = err.radar_sensor
    if rs is not None and rs.range_bias is not None:
        radar_sensor = RadarSensorErrorType(
            RangeBias=rs.range_bias,
            ClockFreqSF=rs.clock_freq_sf,
            TransmitFreqSF=rs.transmit_freq_sf,
            RangeBiasDecorr=_convert_decorr(rs.range_bias_decorr),
        )

    tropo = None
    tr = err.tropo_error
    if tr is not None and (tr.tropo_range_vertical is not None
                           or tr.tropo_range_slant is not None):
        tropo = TropoErrorType(
            TropoRangeVertical=tr.tropo_range_vertical,
            TropoRangeSlant=tr.tropo_range_slant,
            TropoRangeDecorr=_convert_decorr(tr.tropo_range_decorr),
        )

    iono = None
    io = err.iono_error
    if io is not None and io.iono_rg_rg_rate_cc is not None:
        iono = IonoErrorType(
            IonoRangeVertical=io.iono_range_vertical,
            IonoRangeRateVertical=io.iono_range_rate_vertical,
            IonoRgRgRateCC=io.iono_rg_rg_rate_cc,
            IonoRangeVertDecorr=_convert_decorr(
                io.iono_range_vert_decorr,
            ),
        )

    unmodeled = None
    un = err.unmodeled
    if un is not None and None not in (un.xrow, un.ycol, un.xrow_ycol):
        decorr = None
        ud = un.unmodeled_decorr
        if ud is not None and ud.xrow is not None and ud.ycol is not None:
            decorr = UnmodeledDecorrType(
                Xrow=_convert_decorr(ud.xrow),
                Ycol=_convert_decorr(ud.ycol),
            )
        unmodeled = UnmodeledType(
            Xrow=un.xrow, Ycol=un.ycol, XrowYcol=un.xrow_ycol,
            UnmodeledDecorr=decorr,
        )

    # The schema requires both PosVelErr and RadarSensor inside
    # Components, so a source carrying only one of them contributes no
    # Components block -- and Tropo/Iono live inside it.
    components = None
    if pos_vel is not None and radar_sensor is not None:
        components = ErrorComponentsType(
            PosVelErr=pos_vel, RadarSensor=radar_sensor,
            TropoError=tropo, IonoError=iono,
        )
    elif tropo is not None or iono is not None:
        logger.warning(
            "Source carries tropospheric or ionospheric errors but not "
            "both PosVelErr and RadarSensor; the Components block is "
            "omitted, since the schema requires both."
        )

    if composite is None and components is None and unmodeled is None:
        return None
    return ErrorStatisticsType(
        CompositeSCP=composite,
        Components=components,
        Unmodeled=unmodeled,
        AdditionalParms=err.additional_parms or None,
    )


def _convert_match_info(match: Any) -> Optional['MatchInfoType']:
    """Convert GRDL match info to sarpy's SIDD type.

    Parameters
    ----------
    match : SICDMatchInfo or None
        Match info block.

    Returns
    -------
    MatchInfoType or None
        None when the input carries no match types.
    """
    if match is None or not match.match_types:
        return None

    types = []
    for mt in match.match_types:
        if mt.type_id is None:
            continue
        collections = [
            MatchCollectionType(
                CoreName=mc.core_name,
                MatchIndex=mc.match_index,
                Parameters=mc.parameters,
            )
            for mc in (mt.match_collections or [])
            if mc.core_name is not None
        ]
        types.append(MatchType(
            TypeID=mt.type_id,
            CurrentIndex=mt.current_index,
            MatchCollections=collections or None,
        ))

    if not types:
        return None
    return MatchInfoType(MatchTypes=types)


class _OrderedProcessingModule(ProcessingModuleType):
    """A ``ProcessingModuleType`` that serializes in schema order.

    sarpy's implementation appends ``ModuleName`` *after* the generic
    field serialization, so a module carrying parameters emits
    ``ModuleParameter`` elements before ``ModuleName``.  The SIDD
    schema declares ``ModuleName`` first, followed by a choice of
    ``ModuleParameter`` or nested ``ProcessingModule`` elements, so
    sarpy's ordering fails validation.  This subclass emits the
    children in the declared order.
    """

    def to_node(
        self,
        doc: Any,
        tag: str,
        ns_key: Optional[str] = None,
        parent: Any = None,
        check_validity: bool = False,
        strict: bool = False,
        exclude: tuple = (),
    ) -> Any:
        """Serialize with ``ModuleName`` ahead of the parameters.

        Parameters
        ----------
        doc, tag, ns_key, parent, check_validity, strict, exclude
            As for ``sarpy.io.xml.base.Serializable.to_node``.

        Returns
        -------
        ElementTree.Element
            The module node.
        """
        # Build an empty node: hold back every child we place by hand.
        node = super(ProcessingModuleType, self).to_node(
            doc, tag, ns_key=ns_key, parent=parent,
            check_validity=check_validity, strict=strict,
            exclude=exclude + (
                'ModuleName', 'name', 'ModuleParameters',
                'ProcessingModules',
            ),
        )

        if self.ModuleName is not None:
            mn_key = self._child_xml_ns_key.get('ModuleName', ns_key)
            mn_tag = (
                '{}:ModuleName'.format(mn_key)
                if mn_key is not None and mn_key != 'default'
                else 'ModuleName'
            )
            mn_node = _create_text_node(doc, mn_tag, self.ModuleName,
                                        parent=node)
            if self.name is not None:
                mn_node.attrib['name'] = self.name

        if self.ModuleParameters is not None:
            mp_key = self._child_xml_ns_key.get('ModuleParameters', ns_key)
            self.ModuleParameters.to_node(
                doc, ns_key=mp_key, parent=node,
                check_validity=check_validity, strict=strict,
            )

        pm_key = self._child_xml_ns_key.get('ProcessingModules', ns_key)
        for entry in self.ProcessingModules:
            entry.to_node(doc, tag, ns_key=pm_key, parent=node,
                          strict=strict)
        return node


def _convert_product_processing(
    proc: Any,
) -> Optional['ProductProcessingType']:
    """Convert GRDL product processing modules to sarpy's type.

    Parameters
    ----------
    proc : SIDDProductProcessing or None
        Processing block.

    Returns
    -------
    ProductProcessingType or None
        None when there are no modules to record.
    """
    if proc is None or not proc.processing_modules:
        return None
    modules = [
        _OrderedProcessingModule(
            ModuleName=m.module_name,
            name=m.name or 'GRDL',
            ModuleParameters=m.parameters or {},
        )
        for m in proc.processing_modules
        if m.module_name is not None
    ]
    if not modules:
        return None
    return ProductProcessingType(ProcessingModules=modules)


def _convert_digital_elevation_data(
    ded: Any,
) -> Optional['DigitalElevationDataType']:
    """Convert GRDL digital elevation data to sarpy's type.

    Parameters
    ----------
    ded : SIDDDigitalElevationData or None
        Terrain model description.

    Returns
    -------
    DigitalElevationDataType or None
        None when the geographic coordinates or geopositioning blocks
        are missing; both are required by the schema.
    """
    if ded is None:
        return None
    gc = ded.geographic_coordinates
    gp = ded.geopositioning
    if gc is None or gp is None or gc.reference_origin is None:
        return None
    if gc.latitude_density is None or gc.longitude_density is None:
        return None

    accuracy = ded.positional_accuracy
    num_regions = 1 if accuracy is None else (accuracy.num_regions or 1)

    return DigitalElevationDataType(
        GeographicCoordinates=GeographicCoordinatesType(
            LongitudeDensity=gc.longitude_density,
            LatitudeDensity=gc.latitude_density,
            ReferenceOrigin=(
                gc.reference_origin.lat, gc.reference_origin.lon,
            ),
        ),
        Geopositioning=GeopositioningType(
            CoordinateSystemType=gp.coordinate_system_type or 'GGS',
            GeodeticDatum=(
                gp.geodetic_datum or 'World Geodetic System 1984'
            ),
            ReferenceEllipsoid=(
                gp.reference_ellipsoid or 'World Geodetic System 1984'
            ),
            VerticalDatum=gp.vertical_datum or 'HAE',
            SoundingDatum=gp.sounding_datum or 'MSL',
            FalseOrigin=(
                gp.false_origin if gp.false_origin is not None else 0
            ),
            UTMGridZoneNumber=gp.utm_grid_zone_number,
        ),
        PositionalAccuracy=PositionalAccuracyType(
            NumRegions=num_regions,
        ),
    )


def _convert_downstream(
    down: Any,
) -> Optional['DownstreamReprocessingType']:
    """Convert GRDL downstream reprocessing to sarpy's type.

    Parameters
    ----------
    down : SIDDDownstreamReprocessing or None
        Chip and processing event history.

    Returns
    -------
    DownstreamReprocessingType or None
        None when neither a chip nor an event is present.
    """
    if down is None:
        return None

    chip = None
    gc = down.geometric_chip
    if gc is not None and None not in (
            gc.chip_size, gc.original_upper_left, gc.original_upper_right,
            gc.original_lower_left, gc.original_lower_right):
        chip = GeometricChipType(
            ChipSize=(int(gc.chip_size.row), int(gc.chip_size.col)),
            OriginalUpperLeftCoordinate=(
                gc.original_upper_left.row, gc.original_upper_left.col,
            ),
            OriginalUpperRightCoordinate=(
                gc.original_upper_right.row, gc.original_upper_right.col,
            ),
            OriginalLowerLeftCoordinate=(
                gc.original_lower_left.row, gc.original_lower_left.col,
            ),
            OriginalLowerRightCoordinate=(
                gc.original_lower_right.row, gc.original_lower_right.col,
            ),
        )

    events = [
        ProcessingEventType(
            ApplicationName=ev.application_name,
            AppliedDateTime=ev.applied_date_time,
            InterpolationMethod=ev.interpolation_method,
            Descriptors=ev.descriptors,
        )
        for ev in (down.processing_events or [])
        if ev.application_name is not None
        and ev.applied_date_time is not None
    ]

    if chip is None and not events:
        return None
    return DownstreamReprocessingType(
        GeometricChip=chip, ProcessingEvents=events or None,
    )


def _sidd_metadata_to_sarpy(meta: SIDDMetadata) -> 'SIDDType':
    """Convert GRDL ``SIDDMetadata`` to a sarpy ``SIDDType`` v3.

    Every section the SIDD 3.0 schema requires is converted:
    ProductCreation, Display, GeoData, Measurement and
    ExploitationFeatures.  Sections absent from the GRDL model are left
    as None; :meth:`SIDDWriter.write` validates the result before
    handing it to sarpy.

    Parameters
    ----------
    meta : SIDDMetadata
        GRDL typed SIDD metadata.

    Returns
    -------
    SIDDType
        Sarpy SIDD v3 metadata object.
    """
    return SIDDType(
        ProductCreation=_convert_product_creation(meta.product_creation),
        Display=_convert_display(meta.display),
        GeoData=_convert_geo_data(meta.geo_data),
        Measurement=_convert_measurement(meta.measurement),
        ExploitationFeatures=_convert_exploitation(
            meta.exploitation_features,
        ),
        DownstreamReprocessing=_convert_downstream(
            meta.downstream_reprocessing,
        ),
        ErrorStatistics=_convert_error_statistics(meta.error_statistics),
        Radiometric=_convert_radiometric(meta.radiometric),
        MatchInfo=_convert_match_info(meta.match_info),
        DigitalElevationData=_convert_digital_elevation_data(
            meta.digital_elevation_data,
        ),
        ProductProcessing=_convert_product_processing(
            meta.product_processing,
        ),
    )


# ===================================================================
# SIDDWriter
# ===================================================================

class SIDDWriter(ImageWriter):
    """Write derived SAR imagery in SIDD NITF format.

    Accepts GRDL's ``SIDDMetadata`` and converts it to sarpy's internal
    SIDDType v3 for NITF writing.  Use
    :func:`~grdl.IO.sar.sidd_builder.build_sidd_metadata` to construct
    that metadata from an orthorectification output grid, or
    :meth:`~grdl.image_processing.ortho.ortho_builder.OrthoResult.save_sidd`
    to go straight from an ortho result to a file.

    Parameters
    ----------
    filepath : str or Path
        Output path for the SIDD NITF file.
    metadata : SIDDMetadata, optional
        Typed SIDD metadata.  Required unless
        :meth:`set_sarpy_metadata` is called before writing.

    Raises
    ------
    DependencyError
        If sarpy is not installed.

    Examples
    --------
    >>> from grdl.IO.sar import SIDDWriter
    >>> from grdl.IO.sar.sidd_builder import build_sidd_metadata
    >>> meta = build_sidd_metadata(
    ...     grid, product.shape, pixel_type='MONO8I',
    ...     source_metadata=reader.metadata, geolocation=geo,
    ... )
    >>> SIDDWriter('product.nitf', metadata=meta).write(product)
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        metadata: Optional[SIDDMetadata] = None,
    ) -> None:
        if not _HAS_SARPY_SIDD:
            raise DependencyError(
                "sarpy is required for SIDDWriter. "
                "Install with: pip install sarpy"
            )
        super().__init__(filepath, metadata)

        if metadata is not None:
            self._sarpy_meta = _sidd_metadata_to_sarpy(metadata)
        else:
            self._sarpy_meta = SIDDType()

    def set_sarpy_metadata(self, sidd_type: 'SIDDType') -> None:
        """Override with a raw sarpy SIDDType for advanced use.

        Parameters
        ----------
        sidd_type : SIDDType
            Fully populated sarpy SIDD v3 metadata object.
        """
        self._sarpy_meta = sidd_type

    def _expected_layout(self) -> Sequence[int]:
        """Rows, columns, bands and dtype required by the metadata.

        Returns
        -------
        tuple
            ``(rows, cols, bands, dtype)``.

        Raises
        ------
        ValidationError
            If the metadata lacks a pixel footprint or pixel type.
        """
        meas = self._sarpy_meta.Measurement
        display = self._sarpy_meta.Display
        if meas is None or meas.PixelFootprint is None:
            raise ValidationError(
                "SIDD metadata has no Measurement.PixelFootprint; the "
                "product size is unknown. Build metadata with "
                "grdl.IO.sar.sidd_builder.build_sidd_metadata()."
            )
        if display is None or display.PixelType is None:
            raise ValidationError(
                "SIDD metadata has no Display.PixelType."
            )
        bands, dtype = _PIXEL_LAYOUT[display.PixelType]
        return (
            int(meas.PixelFootprint.Row),
            int(meas.PixelFootprint.Col),
            bands,
            dtype,
        )

    def _prepare_data(self, data: np.ndarray) -> np.ndarray:
        """Validate the product array and put bands on the last axis.

        Parameters
        ----------
        data : np.ndarray
            Product raster.  Single band ``(rows, cols)``, or multi-band
            with the band axis leading or trailing.

        Returns
        -------
        np.ndarray
            C-contiguous array shaped ``(rows, cols)`` for single-band
            or ``(rows, cols, bands)`` for multi-band, as sarpy's
            band-interleaved-by-pixel NITF writer expects.

        Raises
        ------
        ValidationError
            If the shape or dtype does not match the metadata.
        """
        arr = np.asarray(data)
        rows, cols, bands, dtype = self._expected_layout()

        if bands == 1:
            if arr.shape != (rows, cols):
                raise ValidationError(
                    f"Product shape {arr.shape} does not match the "
                    f"declared pixel footprint ({rows}, {cols})"
                )
        else:
            if arr.ndim != 3:
                raise ValidationError(
                    f"Pixel type declares {bands} bands, but the product "
                    f"is {arr.ndim}D with shape {arr.shape}"
                )
            if arr.shape == (bands, rows, cols):
                arr = np.moveaxis(arr, 0, -1)
            elif arr.shape != (rows, cols, bands):
                raise ValidationError(
                    f"Product shape {arr.shape} matches neither "
                    f"({bands}, {rows}, {cols}) nor "
                    f"({rows}, {cols}, {bands})"
                )

        if arr.dtype != dtype:
            raise ValidationError(
                f"Pixel type requires dtype {np.dtype(dtype).name}, got "
                f"{arr.dtype}. Apply a stretch from grdl.contrast (for "
                f"example PercentileStretch or MangisDensity) to produce "
                f"display-ready integer samples before writing."
            )

        return np.ascontiguousarray(arr)

    def write(
        self,
        data: np.ndarray,
        geolocation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write the product image to a SIDD NITF file.

        Parameters
        ----------
        data : np.ndarray
            Product raster.  ``(rows, cols)`` for ``MONO8I`` /
            ``MONO16I``, or ``(3, rows, cols)`` / ``(rows, cols, 3)``
            for ``RGB24I``.
        geolocation : dict, optional
            Ignored -- SIDD geolocation lives in the metadata.

        Raises
        ------
        ValidationError
            If the array does not match the metadata, or the metadata is
            not complete enough to write a SIDD.
        """
        arr = self._prepare_data(data)

        # Validate before constructing the sarpy writer.  A structure
        # rejected inside the constructor leaves a partially built
        # object whose __del__ raises, masking the real error.
        try:
            _validate_sidd(self._sarpy_meta)
        except ValueError as exc:
            raise ValidationError(
                f"SIDD metadata is not complete enough to write: {exc} "
                f"Build it with "
                f"grdl.IO.sar.sidd_builder.build_sidd_metadata()."
            ) from exc

        writer = _SarpySIDDWriter(
            str(self.filepath),
            sidd_meta=self._sarpy_meta,
            check_existence=False,
        )
        try:
            writer.write_chip(arr, start_indices=(0, 0), index=0)
        finally:
            writer.close()

        logger.info(
            "Wrote SIDD %s (%d x %d, %s)",
            self.filepath.name, arr.shape[0], arr.shape[1],
            self._sarpy_meta.Display.PixelType,
        )

    def write_chip(
        self,
        data: np.ndarray,
        row_start: int,
        col_start: int,
        geolocation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write a chip to an existing SIDD file.

        Not supported for SIDD format — SIDD files must be written
        as complete images via ``write()``.

        Raises
        ------
        NotImplementedError
            Always raised; SIDD does not support incremental chip writes
            through this interface.
        """
        raise NotImplementedError(
            "SIDD format does not support chip-level writes. "
            "Use write() with the full image."
        )
