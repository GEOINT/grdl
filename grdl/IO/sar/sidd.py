# -*- coding: utf-8 -*-
"""
SIDD Reader - Sensor Independent Derived Data format.

NGA standard for derived SAR products in NITF containers.  Uses sarkit
as the primary backend with sarpy as fallback.  Populates all SIDD
metadata sections as nested dataclasses via ``SIDDMetadata``.

Dependencies
------------
sarkit (primary) or sarpy (fallback)

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
2026-02-09

Modified
--------
2026-03-07
"""

# Standard library
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.base import ImageReader
from grdl.IO.models import (
    SIDDMetadata,
    SIDDProductCreation,
    SIDDProcessorInformation,
    SIDDClassification,
    SIDDDisplay,
    SIDDDynamicRangeAdjustment,
    SIDDDRAParameters,
    SIDDDRAOverrides,
    SIDDGeoData,
    SIDDMeasurement,
    SIDDPlaneProjection,
    SIDDReferencePoint,
    SIDDProductPlane,
    SIDDExploitationFeatures,
    SIDDCollectionInfo,
    SIDDRadarMode,
    SIDDTxRcvPolarization,
    SIDDCollectionGeometry,
    SIDDCollectionPhenomenology,
    SIDDAngleMagnitude,
    SIDDExploitationFeaturesProduct,
    SIDDProductResolution,
    SIDDDownstreamReprocessing,
    SIDDGeometricChip,
    SIDDProcessingEvent,
    SIDDCompression,
    SIDDJPEG2000Subtype,
    SIDDDigitalElevationData,
    SIDDGeographicCoordinates,
    SIDDGeopositioning,
    SIDDPositionalAccuracy,
    SIDDProductProcessing,
    SIDDProcessingModule,
    SIDDAnnotations,
    SIDDAnnotation,
)
from grdl.IO.models.common import (
    XYZ,
    LatLon,
    RowCol,
    Poly2D,
    XYZPoly,
    Poly1D,
)
from grdl.IO.models.sicd import (
    SICDErrorStatistics,
    SICDRadiometric,
    SICDMatchInfo,
    SICDCompositeSCPError,
    SICDPosVelErr,
    SICDCorrCoefs,
    SICDRadarSensorError,
    SICDNoiseLevel,
    SICDMatchType,
    SICDMatchCollection,
)
from grdl.IO.sar._backend import require_sar_backend

logger = logging.getLogger(__name__)


# Pixel-type → dtype mapping shared by both backends
_PIXEL_DTYPE_MAP = {
    'MONO8I': 'uint8',
    'MONO8LU': 'uint8',
    'MONO16I': 'uint16',
    'RGB24I': 'uint8',
    'RGB8LU': 'uint8',
}


# ===================================================================
# XML extraction helpers (sarkit backend)
# ===================================================================

def _xml_float(elem: Optional[ET.Element], path: str) -> Optional[float]:
    val = elem.findtext(path) if elem is not None else None
    return float(val) if val is not None else None


def _xml_int(elem: Optional[ET.Element], path: str) -> Optional[int]:
    val = elem.findtext(path) if elem is not None else None
    return int(val) if val is not None else None


def _xml_str(elem: Optional[ET.Element], path: str) -> Optional[str]:
    return elem.findtext(path) if elem is not None else None


def _xml_xyz(elem: Optional[ET.Element], path: str) -> Optional[XYZ]:
    if elem is None:
        return None
    sub = elem.find(path)
    if sub is None:
        return None
    x = _xml_float(sub, '{*}X')
    y = _xml_float(sub, '{*}Y')
    z = _xml_float(sub, '{*}Z')
    if x is None and y is None and z is None:
        return None
    return XYZ(x=x or 0.0, y=y or 0.0, z=z or 0.0)


def _xml_rowcol(elem: Optional[ET.Element], path: str) -> Optional[RowCol]:
    if elem is None:
        return None
    sub = elem.find(path)
    if sub is None:
        return None
    row = _xml_float(sub, '{*}Row')
    col = _xml_float(sub, '{*}Col')
    if row is None:
        return None
    return RowCol(row=row, col=col or 0.0)


def _xml_latlon(elem: Optional[ET.Element], path: str) -> Optional[LatLon]:
    if elem is None:
        return None
    sub = elem.find(path)
    if sub is None:
        return None
    lat = _xml_float(sub, '{*}Lat')
    lon = _xml_float(sub, '{*}Lon')
    if lat is None:
        return None
    return LatLon(lat=lat, lon=lon or 0.0)


def _xml_poly1d(elem: Optional[ET.Element]) -> Optional[Poly1D]:
    if elem is None:
        return None
    order = int(elem.get('order1', '0'))
    coefs = np.zeros(order + 1)
    for coef_el in elem.findall('{*}Coef'):
        exp = int(coef_el.get('exponent1', '0'))
        coefs[exp] = float(coef_el.text)
    return Poly1D(coefs=coefs)


def _xml_poly2d(elem: Optional[ET.Element]) -> Optional[Poly2D]:
    if elem is None:
        return None
    order1 = int(elem.get('order1', '0'))
    order2 = int(elem.get('order2', '0'))
    coefs = np.zeros((order1 + 1, order2 + 1))
    for coef_el in elem.findall('{*}Coef'):
        exp1 = int(coef_el.get('exponent1', '0'))
        exp2 = int(coef_el.get('exponent2', '0'))
        coefs[exp1, exp2] = float(coef_el.text)
    return Poly2D(coefs=coefs)


def _xml_poly2d_at(
    elem: Optional[ET.Element], path: str,
) -> Optional[Poly2D]:
    if elem is None:
        return None
    return _xml_poly2d(elem.find(path))


def _xml_xyzpoly(
    elem: Optional[ET.Element], path: str,
) -> Optional[XYZPoly]:
    if elem is None:
        return None
    sub = elem.find(path)
    if sub is None:
        return None
    x = _xml_poly1d(sub.find('{*}X'))
    y = _xml_poly1d(sub.find('{*}Y'))
    z = _xml_poly1d(sub.find('{*}Z'))
    return XYZPoly(x=x, y=y, z=z)


# ===================================================================
# sarpy attribute helper
# ===================================================================

def _safe_get(obj: Any, attr: str) -> Any:
    """Safely get an attribute from a sarpy object."""
    if obj is None:
        return None
    return getattr(obj, attr, None)


def _sarpy_poly2d(obj: Any) -> Optional[Poly2D]:
    """Convert sarpy Poly2DType to Poly2D."""
    if obj is None:
        return None
    coefs = getattr(obj, 'Coefs', None)
    if coefs is None:
        return None
    return Poly2D(coefs=np.array(coefs))


# ===================================================================
# Section extractors — sarkit (XML)
# ===================================================================

def _extract_product_creation(
    xml: ET.Element,
) -> Optional[SIDDProductCreation]:
    """Extract ProductCreation from SIDD XML."""
    pc = xml.find('{*}ProductCreation')
    if pc is None:
        return None

    proc_info = None
    pi = pc.find('{*}ProcessorInformation')
    if pi is not None:
        proc_info = SIDDProcessorInformation(
            application=_xml_str(pi, '{*}Application'),
            processing_date_time=_xml_str(pi, '{*}ProcessingDateTime'),
            site=_xml_str(pi, '{*}Site'),
            profile=_xml_str(pi, '{*}Profile'),
        )

    classification = None
    cls_elem = pc.find('{*}Classification')
    if cls_elem is not None:
        classification = SIDDClassification(
            classification=cls_elem.get('classification'),
            owner_producer=cls_elem.get('ownerProducer'),
            des_version=_xml_int(cls_elem, '{*}DESVersion'),
            create_date=cls_elem.get('createDate'),
            sar_identifier=cls_elem.get('SARIdentifier'),
            sci_controls=cls_elem.get('SCIcontrols'),
            dissemination_controls=cls_elem.get('disseminationControls'),
            releasable_to=cls_elem.get('releasableTo'),
            classified_by=cls_elem.get('classifiedBy'),
            derived_from=cls_elem.get('derivedFrom'),
            declass_date=cls_elem.get('declassDate'),
            declass_event=cls_elem.get('declassEvent'),
        )
        if classification.classification is None:
            classification.classification = _xml_str(
                cls_elem, '{*}SecurityClassification'
            )

    return SIDDProductCreation(
        processor_information=proc_info,
        classification=classification,
        product_name=_xml_str(pc, '{*}ProductName'),
        product_class=_xml_str(pc, '{*}ProductClass'),
        product_type=_xml_str(pc, '{*}ProductType'),
    )


def _extract_display(xml: ET.Element) -> Optional[SIDDDisplay]:
    """Extract Display from SIDD XML."""
    disp = xml.find('{*}Display')
    if disp is None:
        return None

    dra = None
    ip_list = disp.findall('{*}InteractiveProcessing')
    if ip_list:
        dra_elem = ip_list[0].find('{*}DynamicRangeAdjustment')
        if dra_elem is not None:
            dra_params = None
            dp = dra_elem.find('{*}DRAParameters')
            if dp is not None:
                dra_params = SIDDDRAParameters(
                    pmin=_xml_float(dp, '{*}Pmin'),
                    pmax=_xml_float(dp, '{*}Pmax'),
                    emin_modifier=_xml_float(dp, '{*}EminModifier'),
                    emax_modifier=_xml_float(dp, '{*}EmaxModifier'),
                )
            dra_overrides = None
            do = dra_elem.find('{*}DRAOverrides')
            if do is not None:
                dra_overrides = SIDDDRAOverrides(
                    subtractor=_xml_float(do, '{*}Subtractor'),
                    multiplier=_xml_float(do, '{*}Multiplier'),
                )
            dra = SIDDDynamicRangeAdjustment(
                algorithm_type=_xml_str(dra_elem, '{*}AlgorithmType'),
                band_stats_source=_xml_int(dra_elem, '{*}BandStatsSource'),
                dra_parameters=dra_params,
                dra_overrides=dra_overrides,
            )

    return SIDDDisplay(
        pixel_type=_xml_str(disp, '{*}PixelType'),
        num_bands=_xml_int(disp, '{*}NumBands'),
        default_band_display=_xml_int(disp, '{*}DefaultBandDisplay'),
        dynamic_range_adjustment=dra,
    )


def _extract_geo_data(xml: ET.Element) -> Optional[SIDDGeoData]:
    """Extract GeoData from SIDD XML."""
    geo = xml.find('{*}GeoData')
    if geo is None:
        return None

    corners = None
    ic = geo.find('{*}ImageCorners')
    if ic is not None:
        corners = []
        for label in ('ICP', 'FRFC', 'FRLC', 'LRLC', 'LRFC'):
            for c in ic.findall('{*}' + label):
                lat = _xml_float(c, '{*}Lat')
                lon = _xml_float(c, '{*}Lon')
                if lat is not None:
                    corners.append(LatLon(lat=lat, lon=lon or 0.0))

    return SIDDGeoData(
        earth_model=_xml_str(geo, '{*}EarthModel') or 'WGS_84',
        image_corners=corners if corners else None,
    )


def _extract_measurement(xml: ET.Element) -> Optional[SIDDMeasurement]:
    """Extract Measurement from SIDD XML."""
    meas = xml.find('{*}Measurement')
    if meas is None:
        return None

    proj_type = None
    plane_proj = None

    pp = meas.find('{*}PlaneProjection')
    if pp is not None:
        proj_type = 'PlaneProjection'

        ref_point = None
        rp = pp.find('{*}ReferencePoint')
        if rp is not None:
            ref_point = SIDDReferencePoint(
                ecef=_xml_xyz(rp, '{*}ECEF') or _xml_xyz(rp, '{*}ECF'),
                point=_xml_rowcol(rp, '{*}Point'),
                name=rp.get('name'),
            )

        product_plane = None
        pplane = pp.find('{*}ProductPlane')
        if pplane is not None:
            product_plane = SIDDProductPlane(
                row_unit_vector=_xml_xyz(pplane, '{*}RowUnitVector'),
                col_unit_vector=_xml_xyz(pplane, '{*}ColUnitVector'),
            )

        plane_proj = SIDDPlaneProjection(
            reference_point=ref_point,
            sample_spacing=_xml_rowcol(pp, '{*}SampleSpacing'),
            time_coa_poly=_xml_poly2d_at(pp, '{*}TimeCOAPoly'),
            product_plane=product_plane,
        )
    elif meas.find('{*}GeographicProjection') is not None:
        proj_type = 'GeographicProjection'
    elif meas.find('{*}CylindricalProjection') is not None:
        proj_type = 'CylindricalProjection'
    elif meas.find('{*}PolynomialProjection') is not None:
        proj_type = 'PolynomialProjection'

    return SIDDMeasurement(
        projection_type=proj_type,
        plane_projection=plane_proj,
        pixel_footprint=_xml_rowcol(meas, '{*}PixelFootprint'),
        arp_flag=_xml_str(meas, '{*}ARPFlag'),
        arp_poly=_xml_xyzpoly(meas, '{*}ARPPoly'),
    )


def _extract_exploitation_features(
    xml: ET.Element,
) -> Optional[SIDDExploitationFeatures]:
    """Extract ExploitationFeatures from SIDD XML."""
    ef = xml.find('{*}ExploitationFeatures')
    if ef is None:
        return None

    collections = None
    coll_elems = ef.findall('{*}Collection')
    if coll_elems:
        collections = []
        for coll in coll_elems:
            info = coll.find('{*}Information')

            radar_mode = None
            rm = info.find('{*}RadarMode') if info is not None else None
            if rm is not None:
                radar_mode = SIDDRadarMode(
                    mode_type=_xml_str(rm, '{*}ModeType'),
                    mode_id=_xml_str(rm, '{*}ModeID'),
                )

            polarizations = None
            pol_elems = (
                info.findall('{*}Polarizations/{*}Polarization')
                if info is not None else []
            )
            if pol_elems:
                polarizations = []
                for pol in pol_elems:
                    polarizations.append(SIDDTxRcvPolarization(
                        tx_polarization=_xml_str(pol, '{*}TxPolarization'),
                        rcv_polarization=_xml_str(
                            pol, '{*}RcvPolarization'
                        ),
                    ))

            geometry = None
            geom = coll.find('{*}Geometry')
            if geom is not None:
                geometry = SIDDCollectionGeometry(
                    azimuth=_xml_float(geom, '{*}Azimuth'),
                    slope=_xml_float(geom, '{*}Slope'),
                    squint=_xml_float(geom, '{*}Squint'),
                    graze=_xml_float(geom, '{*}Graze'),
                    tilt=_xml_float(geom, '{*}Tilt'),
                    doppler_cone_angle=_xml_float(
                        geom, '{*}DopplerConeAngle'
                    ),
                )

            phenomenology = None
            phenom = coll.find('{*}Phenomenology')
            if phenom is not None:
                shadow = None
                sh = phenom.find('{*}Shadow')
                if sh is not None:
                    shadow = SIDDAngleMagnitude(
                        angle=_xml_float(sh, '{*}Angle'),
                        magnitude=_xml_float(sh, '{*}Magnitude'),
                    )
                layover = None
                lo = phenom.find('{*}Layover')
                if lo is not None:
                    layover = SIDDAngleMagnitude(
                        angle=_xml_float(lo, '{*}Angle'),
                        magnitude=_xml_float(lo, '{*}Magnitude'),
                    )
                phenomenology = SIDDCollectionPhenomenology(
                    shadow=shadow,
                    layover=layover,
                    multi_path=_xml_float(phenom, '{*}MultiPath'),
                    ground_track=_xml_float(phenom, '{*}GroundTrack'),
                )

            res_range = (
                _xml_float(info, '{*}Resolution/{*}Range')
                if info is not None else None
            )
            res_az = (
                _xml_float(info, '{*}Resolution/{*}Azimuth')
                if info is not None else None
            )

            collections.append(SIDDCollectionInfo(
                sensor_name=(
                    _xml_str(info, '{*}SensorName')
                    if info is not None else None
                ),
                radar_mode=radar_mode,
                collection_date_time=(
                    _xml_str(info, '{*}CollectionDateTime')
                    if info is not None else None
                ),
                collection_duration=(
                    _xml_float(info, '{*}CollectionDuration')
                    if info is not None else None
                ),
                resolution_range=res_range,
                resolution_azimuth=res_az,
                polarizations=polarizations,
                geometry=geometry,
                phenomenology=phenomenology,
                identifier=coll.get('identifier'),
            ))

    products = None
    prod_elems = ef.findall('{*}Product')
    if prod_elems:
        products = []
        for prod in prod_elems:
            resolution = None
            res = prod.find('{*}Resolution')
            if res is not None:
                resolution = SIDDProductResolution(
                    row=_xml_float(res, '{*}Row'),
                    col=_xml_float(res, '{*}Col'),
                )
            products.append(SIDDExploitationFeaturesProduct(
                resolution=resolution,
                ellipticity=_xml_float(prod, '{*}Ellipticity'),
                north=_xml_float(prod, '{*}North'),
            ))

    return SIDDExploitationFeatures(
        collections=collections,
        products=products,
    )


def _extract_downstream_reprocessing(
    xml: ET.Element,
) -> Optional[SIDDDownstreamReprocessing]:
    """Extract DownstreamReprocessing from SIDD XML."""
    dr = xml.find('{*}DownstreamReprocessing')
    if dr is None:
        return None

    geo_chip = None
    gc = dr.find('{*}GeometricChip')
    if gc is not None:
        geo_chip = SIDDGeometricChip(
            chip_size=_xml_rowcol(gc, '{*}ChipSize'),
            original_upper_left=_xml_rowcol(
                gc, '{*}OriginalUpperLeftCoordinate'
            ),
            original_upper_right=_xml_rowcol(
                gc, '{*}OriginalUpperRightCoordinate'
            ),
            original_lower_left=_xml_rowcol(
                gc, '{*}OriginalLowerLeftCoordinate'
            ),
            original_lower_right=_xml_rowcol(
                gc, '{*}OriginalLowerRightCoordinate'
            ),
        )

    events = None
    pe_elems = dr.findall('{*}ProcessingEvents/{*}ProcessingEvent')
    if pe_elems:
        events = []
        for pe in pe_elems:
            descriptors = None
            d_elems = pe.findall('{*}Descriptor')
            if d_elems:
                descriptors = {
                    d.get('name', ''): (d.text or '') for d in d_elems
                }
            events.append(SIDDProcessingEvent(
                application_name=_xml_str(pe, '{*}ApplicationName'),
                applied_date_time=_xml_str(pe, '{*}AppliedDateTime'),
                interpolation_method=_xml_str(
                    pe, '{*}InterpolationMethod'
                ),
                descriptors=descriptors,
            ))

    return SIDDDownstreamReprocessing(
        geometric_chip=geo_chip,
        processing_events=events,
    )


def _extract_compression(
    xml: ET.Element,
) -> Optional[SIDDCompression]:
    """Extract Compression from SIDD XML."""
    comp = xml.find('{*}Compression')
    if comp is None:
        return None

    def _j2k_subtype(parent: ET.Element, tag: str) -> Optional[SIDDJPEG2000Subtype]:
        sub = parent.find('{*}' + tag)
        if sub is None:
            return None
        return SIDDJPEG2000Subtype(
            num_wavelet_levels=_xml_int(sub, '{*}NumWaveletLevels'),
            num_bands=_xml_int(sub, '{*}NumBands'),
        )

    j2k = comp.find('{*}J2K')
    if j2k is not None:
        return SIDDCompression(
            original=_j2k_subtype(j2k, 'Original'),
            parsed=_j2k_subtype(j2k, 'Parsed'),
        )

    return SIDDCompression()


def _extract_digital_elevation_data(
    xml: ET.Element,
) -> Optional[SIDDDigitalElevationData]:
    """Extract DigitalElevationData from SIDD XML."""
    ded = xml.find('{*}DigitalElevationData')
    if ded is None:
        return None

    geo_coords = None
    gc = ded.find('{*}GeographicCoordinates')
    if gc is not None:
        geo_coords = SIDDGeographicCoordinates(
            longitude_density=_xml_float(gc, '{*}LongitudeDensity'),
            latitude_density=_xml_float(gc, '{*}LatitudeDensity'),
            reference_origin=_xml_latlon(gc, '{*}ReferenceOrigin'),
        )

    geopositioning = None
    gp = ded.find('{*}Geopositioning')
    if gp is not None:
        geopositioning = SIDDGeopositioning(
            coordinate_system_type=_xml_str(gp, '{*}CoordinateSystemType'),
            geodetic_datum=_xml_str(gp, '{*}GeodeticDatum'),
            reference_ellipsoid=_xml_str(gp, '{*}ReferenceEllipsoid'),
            vertical_datum=_xml_str(gp, '{*}VerticalDatum'),
            sounding_datum=_xml_str(gp, '{*}SoundingDatum'),
            false_origin=_xml_int(gp, '{*}FalseOrigin'),
            utm_grid_zone_number=_xml_int(gp, '{*}UTMGridZoneNumber'),
        )

    positional_accuracy = None
    pa = ded.find('{*}PositionalAccuracy')
    if pa is not None:
        num_regions = _xml_int(pa, '{*}NumRegions')
        abs_h_list = []
        abs_v_list = []
        p2p_h_list = []
        p2p_v_list = []
        for region in pa.findall('{*}AbsoluteAccuracy/{*}Region'):
            h = _xml_float(region, '{*}Horizontal')
            v = _xml_float(region, '{*}Vertical')
            if h is not None:
                abs_h_list.append(h)
            if v is not None:
                abs_v_list.append(v)
        for region in pa.findall('{*}PointToPointAccuracy/{*}Region'):
            h = _xml_float(region, '{*}Horizontal')
            v = _xml_float(region, '{*}Vertical')
            if h is not None:
                p2p_h_list.append(h)
            if v is not None:
                p2p_v_list.append(v)
        positional_accuracy = SIDDPositionalAccuracy(
            num_regions=num_regions,
            absolute_horizontal=abs_h_list or None,
            absolute_vertical=abs_v_list or None,
            point_to_point_horizontal=p2p_h_list or None,
            point_to_point_vertical=p2p_v_list or None,
        )

    return SIDDDigitalElevationData(
        geographic_coordinates=geo_coords,
        geopositioning=geopositioning,
        positional_accuracy=positional_accuracy,
        null_value=_xml_int(ded, '{*}NullValue'),
    )


def _extract_product_processing(
    xml: ET.Element,
) -> Optional[SIDDProductProcessing]:
    """Extract ProductProcessing from SIDD XML."""
    pp = xml.find('{*}ProductProcessing')
    if pp is None:
        return None

    modules = None
    pm_elems = pp.findall('{*}ProcessingModule')
    if pm_elems:
        modules = []
        for pm in pm_elems:
            params = None
            p_elems = pm.findall('{*}ModuleParameters/{*}Parameter')
            if not p_elems:
                p_elems = pm.findall('{*}Parameter')
            if p_elems:
                params = {
                    p.get('name', ''): (p.text or '') for p in p_elems
                }
            modules.append(SIDDProcessingModule(
                module_name=_xml_str(pm, '{*}ModuleName'),
                name=pm.get('name'),
                parameters=params,
            ))

    return SIDDProductProcessing(processing_modules=modules)


def _extract_annotations(
    xml: ET.Element,
) -> Optional[SIDDAnnotations]:
    """Extract Annotations from SIDD XML."""
    ann = xml.find('{*}Annotations')
    if ann is None:
        return None

    annotations = None
    ann_elems = ann.findall('{*}Annotation')
    if ann_elems:
        annotations = []
        for a in ann_elems:
            annotations.append(SIDDAnnotation(
                identifier=a.get('identifier') or _xml_str(
                    a, '{*}Identifier'
                ),
                spatial_reference_system=_xml_str(
                    a, '{*}SpatialReferenceSystem'
                ),
            ))

    return SIDDAnnotations(annotations=annotations)


def _extract_error_statistics_xml(
    xml: ET.Element,
) -> Optional[SICDErrorStatistics]:
    """Extract ErrorStatistics from SIDD XML (shared with SICD)."""
    es = xml.find('{*}ErrorStatistics')
    if es is None:
        return None

    composite_scp = None
    cs = es.find('{*}CompositeSCP')
    if cs is not None:
        composite_scp = SICDCompositeSCPError(
            rg=_xml_float(cs, '{*}Rg'),
            az=_xml_float(cs, '{*}Az'),
            rg_az=_xml_float(cs, '{*}RgAz'),
        )

    return SICDErrorStatistics(
        composite_scp=composite_scp,
    )


def _extract_radiometric_xml(
    xml: ET.Element,
) -> Optional[SICDRadiometric]:
    """Extract Radiometric from SIDD XML (shared with SICD)."""
    rad = xml.find('{*}Radiometric')
    if rad is None:
        return None

    noise_level = None
    nl = rad.find('{*}NoiseLevel')
    if nl is not None:
        noise_level = SICDNoiseLevel(
            noise_level_type=_xml_str(nl, '{*}NoiseLevelType'),
            noise_poly=_xml_poly2d(nl.find('{*}NoisePoly')),
        )

    return SICDRadiometric(
        noise_level=noise_level,
        rcs_sf_poly=_xml_poly2d(rad.find('{*}RCSSFPoly')),
        sigma_zero_sf_poly=_xml_poly2d(rad.find('{*}SigmaZeroSFPoly')),
        beta_zero_sf_poly=_xml_poly2d(rad.find('{*}BetaZeroSFPoly')),
        gamma_zero_sf_poly=_xml_poly2d(rad.find('{*}GammaZeroSFPoly')),
    )


def _extract_match_info_xml(
    xml: ET.Element,
) -> Optional[SICDMatchInfo]:
    """Extract MatchInfo from SIDD XML (shared with SICD)."""
    mi = xml.find('{*}MatchInfo')
    if mi is None:
        return None

    match_types = None
    mt_elems = mi.findall('{*}MatchType')
    if mt_elems:
        match_types = []
        for mt in mt_elems:
            collections = None
            mc_elems = mt.findall('{*}MatchCollection')
            if mc_elems:
                collections = []
                for mc in mc_elems:
                    collections.append(SICDMatchCollection(
                        core_name=_xml_str(mc, '{*}CoreName'),
                        match_index=_xml_int(mc, '{*}MatchIndex'),
                    ))
            match_types.append(SICDMatchType(
                type_id=_xml_str(mt, '{*}TypeID'),
                current_index=_xml_int(mt, '{*}CurrentIndex'),
                num_match_collections=_xml_int(
                    mt, '{*}NumMatchCollections'
                ),
                match_collections=collections,
            ))

    return SICDMatchInfo(match_types=match_types)


# ===================================================================
# Section extractors — sarpy
# ===================================================================

def _extract_product_creation_sarpy(
    sm: Any,
) -> Optional[SIDDProductCreation]:
    """Extract ProductCreation from sarpy SIDDType."""
    pc = _safe_get(sm, 'ProductCreation')
    if pc is None:
        return None

    proc_info = None
    pi = _safe_get(pc, 'ProcessorInformation')
    if pi is not None:
        proc_info = SIDDProcessorInformation(
            application=_safe_get(pi, 'Application'),
            processing_date_time=str(_safe_get(pi, 'ProcessingDateTime'))
            if _safe_get(pi, 'ProcessingDateTime') is not None else None,
            site=_safe_get(pi, 'Site'),
            profile=_safe_get(pi, 'Profile'),
        )

    classification = None
    cls_obj = _safe_get(pc, 'Classification')
    if cls_obj is not None:
        classification = SIDDClassification(
            classification=_safe_get(cls_obj, 'classification'),
            owner_producer=_safe_get(cls_obj, 'ownerProducer'),
            des_version=_safe_get(cls_obj, 'DESVersion'),
            create_date=_safe_get(cls_obj, 'createDate'),
        )

    return SIDDProductCreation(
        processor_information=proc_info,
        classification=classification,
        product_name=_safe_get(pc, 'ProductName'),
        product_class=_safe_get(pc, 'ProductClass'),
        product_type=_safe_get(pc, 'ProductType'),
    )


def _extract_display_sarpy(sm: Any) -> Optional[SIDDDisplay]:
    """Extract Display from sarpy SIDDType."""
    disp = _safe_get(sm, 'Display')
    if disp is None:
        return None

    pixel_type = _safe_get(disp, 'PixelType')
    num_bands = _safe_get(disp, 'NumBands')

    dra = None
    ip_list = _safe_get(disp, 'InteractiveProcessing')
    if ip_list and len(ip_list) > 0:
        dra_obj = _safe_get(ip_list[0], 'DynamicRangeAdjustment')
        if dra_obj is not None:
            dra_params = None
            dp = _safe_get(dra_obj, 'DRAParameters')
            if dp is not None:
                dra_params = SIDDDRAParameters(
                    pmin=_safe_get(dp, 'Pmin'),
                    pmax=_safe_get(dp, 'Pmax'),
                    emin_modifier=_safe_get(dp, 'EminModifier'),
                    emax_modifier=_safe_get(dp, 'EmaxModifier'),
                )
            dra_overrides = None
            do = _safe_get(dra_obj, 'DRAOverrides')
            if do is not None:
                dra_overrides = SIDDDRAOverrides(
                    subtractor=_safe_get(do, 'Subtractor'),
                    multiplier=_safe_get(do, 'Multiplier'),
                )
            dra = SIDDDynamicRangeAdjustment(
                algorithm_type=_safe_get(dra_obj, 'AlgorithmType'),
                band_stats_source=_safe_get(dra_obj, 'BandStatsSource'),
                dra_parameters=dra_params,
                dra_overrides=dra_overrides,
            )

    return SIDDDisplay(
        pixel_type=pixel_type,
        num_bands=num_bands,
        default_band_display=_safe_get(disp, 'DefaultBandDisplay'),
        dynamic_range_adjustment=dra,
    )


def _extract_geo_data_sarpy(sm: Any) -> Optional[SIDDGeoData]:
    """Extract GeoData from sarpy SIDDType."""
    geo = _safe_get(sm, 'GeoData')
    if geo is None:
        return None

    corners = None
    ic = _safe_get(geo, 'ImageCorners')
    if ic is not None:
        try:
            corners = []
            for c in ic:
                lat = getattr(c, 'Lat', None)
                lon = getattr(c, 'Lon', None)
                if lat is not None:
                    corners.append(LatLon(lat=float(lat), lon=float(lon or 0.0)))
        except (TypeError, AttributeError):
            corners = None

    return SIDDGeoData(
        earth_model=_safe_get(geo, 'EarthModel') or 'WGS_84',
        image_corners=corners if corners else None,
    )


def _extract_measurement_sarpy(sm: Any) -> Optional[SIDDMeasurement]:
    """Extract Measurement from sarpy SIDDType."""
    meas = _safe_get(sm, 'Measurement')
    if meas is None:
        return None

    proj_type = None
    plane_proj = None

    pp = _safe_get(meas, 'PlaneProjection')
    if pp is not None:
        proj_type = 'PlaneProjection'

        ref_point = None
        rp = _safe_get(pp, 'ReferencePoint')
        if rp is not None:
            ecf = _safe_get(rp, 'ECF')
            ecef = None
            if ecf is not None:
                ecef = XYZ(
                    x=float(ecf[0]) if ecf is not None else 0.0,
                    y=float(ecf[1]) if ecf is not None else 0.0,
                    z=float(ecf[2]) if ecf is not None else 0.0,
                )
            pt = _safe_get(rp, 'Point')
            point = None
            if pt is not None:
                point = RowCol(
                    row=float(_safe_get(pt, 'Row') or 0),
                    col=float(_safe_get(pt, 'Col') or 0),
                )
            ref_point = SIDDReferencePoint(
                ecef=ecef,
                point=point,
                name=_safe_get(rp, 'name'),
            )

        sample_spacing = None
        ss = _safe_get(pp, 'SampleSpacing')
        if ss is not None:
            sample_spacing = RowCol(
                row=float(_safe_get(ss, 'Row') or 0),
                col=float(_safe_get(ss, 'Col') or 0),
            )

        product_plane = None
        pplane = _safe_get(pp, 'ProductPlane')
        if pplane is not None:
            ruv = _safe_get(pplane, 'RowUnitVector')
            cuv = _safe_get(pplane, 'ColUnitVector')
            row_uv = None
            col_uv = None
            if ruv is not None:
                row_uv = XYZ(
                    x=float(ruv[0]), y=float(ruv[1]), z=float(ruv[2]),
                )
            if cuv is not None:
                col_uv = XYZ(
                    x=float(cuv[0]), y=float(cuv[1]), z=float(cuv[2]),
                )
            product_plane = SIDDProductPlane(
                row_unit_vector=row_uv,
                col_unit_vector=col_uv,
            )

        plane_proj = SIDDPlaneProjection(
            reference_point=ref_point,
            sample_spacing=sample_spacing,
            time_coa_poly=_sarpy_poly2d(_safe_get(pp, 'TimeCOAPoly')),
            product_plane=product_plane,
        )
    elif _safe_get(meas, 'GeographicProjection') is not None:
        proj_type = 'GeographicProjection'
    elif _safe_get(meas, 'CylindricalProjection') is not None:
        proj_type = 'CylindricalProjection'
    elif _safe_get(meas, 'PolynomialProjection') is not None:
        proj_type = 'PolynomialProjection'

    pixel_footprint = None
    pf = _safe_get(meas, 'PixelFootprint')
    if pf is not None:
        pixel_footprint = RowCol(
            row=float(_safe_get(pf, 'Row') or 0),
            col=float(_safe_get(pf, 'Col') or 0),
        )

    return SIDDMeasurement(
        projection_type=proj_type,
        plane_projection=plane_proj,
        pixel_footprint=pixel_footprint,
        arp_flag=_safe_get(meas, 'ARPFlag'),
        arp_poly=_sarpy_xyzpoly(_safe_get(meas, 'ARPPoly')),
    )


def _extract_exploitation_features_sarpy(
    sm: Any,
) -> Optional[SIDDExploitationFeatures]:
    """Extract ExploitationFeatures from sarpy SIDDType."""
    ef = _safe_get(sm, 'ExploitationFeatures')
    if ef is None:
        return None

    collections = None
    coll_list = _safe_get(ef, 'Collections')
    if coll_list:
        collections = []
        for coll in coll_list:
            info = _safe_get(coll, 'Information')

            radar_mode = None
            rm = _safe_get(info, 'RadarMode') if info is not None else None
            if rm is not None:
                radar_mode = SIDDRadarMode(
                    mode_type=_safe_get(rm, 'ModeType'),
                    mode_id=_safe_get(rm, 'ModeID'),
                )

            polarizations = None
            pol_list = (
                _safe_get(info, 'Polarizations')
                if info is not None else None
            )
            if pol_list:
                polarizations = []
                for pol in pol_list:
                    polarizations.append(SIDDTxRcvPolarization(
                        tx_polarization=_safe_get(pol, 'TxPolarization'),
                        rcv_polarization=_safe_get(pol, 'RcvPolarization'),
                    ))

            geometry = None
            geom = _safe_get(coll, 'Geometry')
            if geom is not None:
                geometry = SIDDCollectionGeometry(
                    azimuth=_safe_get(geom, 'Azimuth'),
                    slope=_safe_get(geom, 'Slope'),
                    squint=_safe_get(geom, 'Squint'),
                    graze=_safe_get(geom, 'Graze'),
                    tilt=_safe_get(geom, 'Tilt'),
                    doppler_cone_angle=_safe_get(geom, 'DopplerConeAngle'),
                )

            phenomenology = None
            phenom = _safe_get(coll, 'Phenomenology')
            if phenom is not None:
                shadow = None
                sh = _safe_get(phenom, 'Shadow')
                if sh is not None:
                    shadow = SIDDAngleMagnitude(
                        angle=_safe_get(sh, 'Angle'),
                        magnitude=_safe_get(sh, 'Magnitude'),
                    )
                layover = None
                lo = _safe_get(phenom, 'Layover')
                if lo is not None:
                    layover = SIDDAngleMagnitude(
                        angle=_safe_get(lo, 'Angle'),
                        magnitude=_safe_get(lo, 'Magnitude'),
                    )
                phenomenology = SIDDCollectionPhenomenology(
                    shadow=shadow,
                    layover=layover,
                    multi_path=_safe_get(phenom, 'MultiPath'),
                    ground_track=_safe_get(phenom, 'GroundTrack'),
                )

            res = _safe_get(info, 'Resolution') if info is not None else None

            collections.append(SIDDCollectionInfo(
                sensor_name=(
                    _safe_get(info, 'SensorName')
                    if info is not None else None
                ),
                radar_mode=radar_mode,
                collection_date_time=(
                    str(_safe_get(info, 'CollectionDateTime'))
                    if info is not None and _safe_get(
                        info, 'CollectionDateTime'
                    ) is not None
                    else None
                ),
                collection_duration=(
                    _safe_get(info, 'CollectionDuration')
                    if info is not None else None
                ),
                resolution_range=(
                    _safe_get(res, 'Range') if res is not None else None
                ),
                resolution_azimuth=(
                    _safe_get(res, 'Azimuth') if res is not None else None
                ),
                polarizations=polarizations,
                geometry=geometry,
                phenomenology=phenomenology,
                identifier=_safe_get(coll, 'identifier'),
            ))

    products = None
    prod_list = _safe_get(ef, 'Products')
    if prod_list:
        products = []
        for prod in prod_list:
            resolution = None
            res = _safe_get(prod, 'Resolution')
            if res is not None:
                resolution = SIDDProductResolution(
                    row=_safe_get(res, 'Row'),
                    col=_safe_get(res, 'Col'),
                )
            products.append(SIDDExploitationFeaturesProduct(
                resolution=resolution,
                ellipticity=_safe_get(prod, 'Ellipticity'),
                north=_safe_get(prod, 'North'),
            ))

    return SIDDExploitationFeatures(
        collections=collections,
        products=products,
    )


def _extract_downstream_reprocessing_sarpy(
    sm: Any,
) -> Optional[SIDDDownstreamReprocessing]:
    """Extract DownstreamReprocessing from sarpy SIDDType."""
    dr = _safe_get(sm, 'DownstreamReprocessing')
    if dr is None:
        return None

    geo_chip = None
    gc = _safe_get(dr, 'GeometricChip')
    if gc is not None:
        def _rc(obj):
            if obj is None:
                return None
            return RowCol(
                row=float(_safe_get(obj, 'Row') or 0),
                col=float(_safe_get(obj, 'Col') or 0),
            )
        geo_chip = SIDDGeometricChip(
            chip_size=_rc(_safe_get(gc, 'ChipSize')),
            original_upper_left=_rc(
                _safe_get(gc, 'OriginalUpperLeftCoordinate')
            ),
            original_upper_right=_rc(
                _safe_get(gc, 'OriginalUpperRightCoordinate')
            ),
            original_lower_left=_rc(
                _safe_get(gc, 'OriginalLowerLeftCoordinate')
            ),
            original_lower_right=_rc(
                _safe_get(gc, 'OriginalLowerRightCoordinate')
            ),
        )

    events = None
    pe_obj = _safe_get(dr, 'ProcessingEvents')
    if pe_obj:
        events = []
        pe_list = pe_obj if isinstance(pe_obj, (list, tuple)) else [pe_obj]
        for pe in pe_list:
            events.append(SIDDProcessingEvent(
                application_name=_safe_get(pe, 'ApplicationName'),
                applied_date_time=str(_safe_get(pe, 'AppliedDateTime'))
                if _safe_get(pe, 'AppliedDateTime') is not None else None,
                interpolation_method=_safe_get(pe, 'InterpolationMethod'),
            ))

    return SIDDDownstreamReprocessing(
        geometric_chip=geo_chip,
        processing_events=events,
    )


def _extract_compression_sarpy(sm: Any) -> Optional[SIDDCompression]:
    """Extract Compression from sarpy SIDDType."""
    comp = _safe_get(sm, 'Compression')
    if comp is None:
        return None

    def _j2k(parent, attr):
        sub = _safe_get(parent, attr)
        if sub is None:
            return None
        return SIDDJPEG2000Subtype(
            num_wavelet_levels=_safe_get(sub, 'NumWaveletLevels'),
            num_bands=_safe_get(sub, 'NumBands'),
        )

    j2k = _safe_get(comp, 'J2K')
    if j2k is not None:
        return SIDDCompression(
            original=_j2k(j2k, 'Original'),
            parsed=_j2k(j2k, 'Parsed'),
        )

    return SIDDCompression()


def _extract_digital_elevation_data_sarpy(
    sm: Any,
) -> Optional[SIDDDigitalElevationData]:
    """Extract DigitalElevationData from sarpy SIDDType."""
    ded = _safe_get(sm, 'DigitalElevationData')
    if ded is None:
        return None

    geo_coords = None
    gc = _safe_get(ded, 'GeographicCoordinates')
    if gc is not None:
        ref_origin = _safe_get(gc, 'ReferenceOrigin')
        origin = None
        if ref_origin is not None:
            origin = LatLon(
                lat=float(_safe_get(ref_origin, 'Lat') or 0),
                lon=float(_safe_get(ref_origin, 'Lon') or 0),
            )
        geo_coords = SIDDGeographicCoordinates(
            longitude_density=_safe_get(gc, 'LongitudeDensity'),
            latitude_density=_safe_get(gc, 'LatitudeDensity'),
            reference_origin=origin,
        )

    geopositioning = None
    gp = _safe_get(ded, 'Geopositioning')
    if gp is not None:
        geopositioning = SIDDGeopositioning(
            coordinate_system_type=_safe_get(gp, 'CoordinateSystemType'),
            geodetic_datum=_safe_get(gp, 'GeodeticDatum'),
            reference_ellipsoid=_safe_get(gp, 'ReferenceEllipsoid'),
            vertical_datum=_safe_get(gp, 'VerticalDatum'),
            sounding_datum=_safe_get(gp, 'SoundingDatum'),
            false_origin=_safe_get(gp, 'FalseOrigin'),
            utm_grid_zone_number=_safe_get(gp, 'UTMGridZoneNumber'),
        )

    return SIDDDigitalElevationData(
        geographic_coordinates=geo_coords,
        geopositioning=geopositioning,
        null_value=_safe_get(ded, 'NullValue'),
    )


def _extract_product_processing_sarpy(
    sm: Any,
) -> Optional[SIDDProductProcessing]:
    """Extract ProductProcessing from sarpy SIDDType."""
    pp = _safe_get(sm, 'ProductProcessing')
    if pp is None:
        return None

    modules = None
    pm_list = _safe_get(pp, 'ProcessingModules')
    if pm_list:
        modules = []
        for pm in pm_list:
            modules.append(SIDDProcessingModule(
                module_name=_safe_get(pm, 'ModuleName'),
                name=_safe_get(pm, 'name'),
            ))

    return SIDDProductProcessing(processing_modules=modules)


def _extract_error_statistics_sarpy(
    sm: Any,
) -> Optional[SICDErrorStatistics]:
    """Extract ErrorStatistics from sarpy SIDDType (shared type)."""
    es = _safe_get(sm, 'ErrorStatistics')
    if es is None:
        return None

    composite_scp = None
    cs = _safe_get(es, 'CompositeSCP')
    if cs is not None:
        composite_scp = SICDCompositeSCPError(
            rg=_safe_get(cs, 'Rg'),
            az=_safe_get(cs, 'Az'),
            rg_az=_safe_get(cs, 'RgAz'),
        )

    return SICDErrorStatistics(
        composite_scp=composite_scp,
    )


def _extract_radiometric_sarpy(
    sm: Any,
) -> Optional[SICDRadiometric]:
    """Extract Radiometric from sarpy SIDDType (shared type)."""
    rad = _safe_get(sm, 'Radiometric')
    if rad is None:
        return None

    noise_level = None
    nl = _safe_get(rad, 'NoiseLevel')
    if nl is not None:
        noise_level = SICDNoiseLevel(
            noise_level_type=_safe_get(nl, 'NoiseLevelType'),
            noise_poly=_sarpy_poly2d(_safe_get(nl, 'NoisePoly')),
        )

    return SICDRadiometric(
        noise_level=noise_level,
        rcs_sf_poly=_sarpy_poly2d(_safe_get(rad, 'RCSSFPoly')),
        sigma_zero_sf_poly=_sarpy_poly2d(
            _safe_get(rad, 'SigmaZeroSFPoly')
        ),
        beta_zero_sf_poly=_sarpy_poly2d(
            _safe_get(rad, 'BetaZeroSFPoly')
        ),
        gamma_zero_sf_poly=_sarpy_poly2d(
            _safe_get(rad, 'GammaZeroSFPoly')
        ),
    )


def _extract_match_info_sarpy(sm: Any) -> Optional[SICDMatchInfo]:
    """Extract MatchInfo from sarpy SIDDType (shared type)."""
    mi = _safe_get(sm, 'MatchInfo')
    if mi is None:
        return None

    match_types = None
    mt_list = _safe_get(mi, 'MatchTypes') or _safe_get(mi, 'MatchType')
    if mt_list:
        if not isinstance(mt_list, (list, tuple)):
            mt_list = [mt_list]
        match_types = []
        for mt in mt_list:
            collections = None
            mc_list = _safe_get(mt, 'MatchCollections')
            if mc_list:
                collections = []
                for mc in mc_list:
                    collections.append(SICDMatchCollection(
                        core_name=_safe_get(mc, 'CoreName'),
                        match_index=_safe_get(mc, 'MatchIndex'),
                    ))
            match_types.append(SICDMatchType(
                type_id=_safe_get(mt, 'TypeID'),
                current_index=_safe_get(mt, 'CurrentIndex'),
                num_match_collections=_safe_get(mt, 'NumMatchCollections'),
                match_collections=collections,
            ))

    return SICDMatchInfo(match_types=match_types)


# ===================================================================
# SIDDReader
# ===================================================================

class SIDDReader(ImageReader):
    """Read SIDD (Sensor Independent Derived Data) format.

    SIDD is the NGA standard for derived SAR products (processed
    imagery) in NITF containers.  A single file may contain multiple
    product images.  Uses sarkit as the primary backend with sarpy as
    fallback.

    Parameters
    ----------
    filepath : str or Path
        Path to the SIDD file (NITF container).
    image_index : int, optional
        Index of the product image to read (default 0).

    Attributes
    ----------
    filepath : Path
        Path to the SIDD file.
    metadata : SIDDMetadata
        Complete typed metadata with all SIDD sections.
    image_index : int
        Active product image index.
    backend : str
        Active backend (``'sarkit'`` or ``'sarpy'``).

    Raises
    ------
    ImportError
        If neither sarkit nor sarpy is installed.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not valid SIDD.

    Examples
    --------
    >>> from grdl.IO.sar import SIDDReader
    >>> with SIDDReader('derived.nitf') as reader:
    ...     chip = reader.read_chip(0, 512, 0, 512)
    ...     print(reader.metadata.display.pixel_type)
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        image_index: int = 0,
    ) -> None:
        self.backend = require_sar_backend('SIDD')
        logger.info("SIDD backend selected: %s", self.backend)
        self.image_index = image_index
        self._cached_image: Optional[np.ndarray] = None
        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Load SIDD metadata using the active backend."""
        if self.backend == 'sarkit':
            self._load_metadata_sarkit()
        else:
            self._load_metadata_sarpy()

    def _load_metadata_sarkit(self) -> None:
        """Load metadata via sarkit — all sections."""
        import sarkit.sidd

        try:
            self._file_handle = open(str(self.filepath), 'rb')
            self._reader = sarkit.sidd.NitfReader(self._file_handle)

            num_images = len(self._reader.metadata.images)

            if self.image_index >= num_images:
                raise ValueError(
                    f"image_index {self.image_index} out of range, "
                    f"file contains {num_images} product image(s)"
                )

            img_meta = self._reader.metadata.images[self.image_index]
            xml = img_meta.xmltree

            num_rows = xml.findtext(
                '{*}Measurement/{*}PixelFootprint/{*}Row'
            )
            num_cols = xml.findtext(
                '{*}Measurement/{*}PixelFootprint/{*}Col'
            )
            pixel_type = xml.findtext('{*}Display/{*}PixelType')
            dtype_str = _PIXEL_DTYPE_MAP.get(pixel_type, 'uint8')

            self.metadata = SIDDMetadata(
                format='SIDD',
                rows=int(num_rows) if num_rows else 0,
                cols=int(num_cols) if num_cols else 0,
                dtype=dtype_str,
                backend='sarkit',
                num_images=num_images,
                image_index=self.image_index,
                product_creation=_extract_product_creation(xml),
                display=_extract_display(xml),
                geo_data=_extract_geo_data(xml),
                measurement=_extract_measurement(xml),
                exploitation_features=_extract_exploitation_features(xml),
                downstream_reprocessing=(
                    _extract_downstream_reprocessing(xml)
                ),
                error_statistics=_extract_error_statistics_xml(xml),
                radiometric=_extract_radiometric_xml(xml),
                match_info=_extract_match_info_xml(xml),
                compression=_extract_compression(xml),
                digital_elevation_data=(
                    _extract_digital_elevation_data(xml)
                ),
                product_processing=_extract_product_processing(xml),
                annotations=_extract_annotations(xml),
            )

            self._xmltree = xml

            logger.info(
                "Loaded SIDD %s (%d x %d) via sarkit, image %d of %d",
                self.filepath.name,
                int(num_rows) if num_rows else 0,
                int(num_cols) if num_cols else 0,
                self.image_index,
                num_images,
            )

        except Exception as e:
            raise ValueError(f"Failed to load SIDD metadata: {e}") from e

    def _load_metadata_sarpy(self) -> None:
        """Load metadata via sarpy (fallback) — all sections."""
        from sarpy.io.product.sidd import SIDDReader as _SarpySIDDReader

        try:
            self._reader = _SarpySIDDReader(str(self.filepath))

            sidd_metas = self._reader.get_sidds_as_tuple()
            num_images = len(sidd_metas)

            if self.image_index >= num_images:
                raise ValueError(
                    f"image_index {self.image_index} out of range, "
                    f"file contains {num_images} product image(s)"
                )

            sm = sidd_metas[self.image_index]
            self._sarpy_meta = sm

            # Get dimensions from data_size
            data_sizes = self._reader.get_data_size_as_tuple()
            shape = data_sizes[self.image_index]
            num_rows = int(shape[0])
            num_cols = int(shape[1])

            pixel_type = _safe_get(
                _safe_get(sm, 'Display'), 'PixelType'
            )
            dtype_str = _PIXEL_DTYPE_MAP.get(pixel_type, 'uint8')

            self.metadata = SIDDMetadata(
                format='SIDD',
                rows=num_rows,
                cols=num_cols,
                dtype=dtype_str,
                backend='sarpy',
                num_images=num_images,
                image_index=self.image_index,
                product_creation=_extract_product_creation_sarpy(sm),
                display=_extract_display_sarpy(sm),
                geo_data=_extract_geo_data_sarpy(sm),
                measurement=_extract_measurement_sarpy(sm),
                exploitation_features=(
                    _extract_exploitation_features_sarpy(sm)
                ),
                downstream_reprocessing=(
                    _extract_downstream_reprocessing_sarpy(sm)
                ),
                error_statistics=_extract_error_statistics_sarpy(sm),
                radiometric=_extract_radiometric_sarpy(sm),
                match_info=_extract_match_info_sarpy(sm),
                compression=_extract_compression_sarpy(sm),
                digital_elevation_data=(
                    _extract_digital_elevation_data_sarpy(sm)
                ),
                product_processing=_extract_product_processing_sarpy(sm),
            )

            logger.info(
                "Loaded SIDD %s (%d x %d) via sarpy, image %d of %d",
                self.filepath.name, num_rows, num_cols,
                self.image_index, num_images,
            )

        except Exception as e:
            raise ValueError(f"Failed to load SIDD metadata: {e}") from e

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial chip from the SIDD product image.

        Parameters
        ----------
        row_start : int
            Starting row index (inclusive).
        row_end : int
            Ending row index (exclusive).
        col_start : int
            Starting column index (inclusive).
        col_end : int
            Ending column index (exclusive).
        bands : Optional[List[int]]
            Ignored for SIDD.

        Returns
        -------
        np.ndarray
            Image chip data.

        Raises
        ------
        ValueError
            If indices are out of bounds.
        """
        if row_start < 0 or col_start < 0:
            raise ValueError("Start indices must be non-negative")
        if row_end > self.metadata['rows'] or col_end > self.metadata['cols']:
            raise ValueError("End indices exceed image dimensions")

        if self.backend == 'sarkit':
            if self._cached_image is None:
                self._cached_image = self._reader.read_image(
                    self.image_index
                )
            return self._cached_image[row_start:row_end, col_start:col_end]
        else:
            return self._reader.read_chip(
                np.s_[row_start:row_end, col_start:col_end],
                index=self.image_index,
            )

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """Read the full SIDD product image.

        Parameters
        ----------
        bands : Optional[List[int]]
            Ignored for SIDD.

        Returns
        -------
        np.ndarray
            Full product image data.
        """
        if self.backend == 'sarkit':
            if self._cached_image is None:
                self._cached_image = self._reader.read_image(
                    self.image_index
                )
            return self._cached_image
        else:
            return self._reader.read_chip(
                np.s_[:, :], index=self.image_index,
            )

    def get_shape(self) -> Tuple[int, int]:
        """Get image dimensions.

        Returns
        -------
        Tuple[int, int]
            ``(rows, cols)``.
        """
        return (self.metadata['rows'], self.metadata['cols'])

    def get_dtype(self) -> np.dtype:
        """Get data type.

        Returns
        -------
        np.dtype
            Data type of the product image.
        """
        return np.dtype(self.metadata['dtype'])

    def close(self) -> None:
        """Close the reader and release resources."""
        self._cached_image = None
        if self.backend == 'sarkit':
            if hasattr(self, '_file_handle') and self._file_handle is not None:
                try:
                    self._file_handle.close()
                except Exception:
                    pass
                self._file_handle = None
        else:
            if hasattr(self, '_reader') and self._reader is not None:
                self._reader.close()
