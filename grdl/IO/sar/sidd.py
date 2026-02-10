# -*- coding: utf-8 -*-
"""
SIDD Reader - Sensor Independent Derived Data format.

NGA standard for derived SAR products in NITF containers. Uses sarkit
as the backend (no sarpy fallback). Populates all SIDD metadata
sections as nested dataclasses via ``SIDDMetadata``.

Dependencies
------------
sarkit

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
2026-02-10
"""

# Standard library
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
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
)
from grdl.IO.models.common import (
    XYZ,
    LatLon,
    RowCol,
    Poly2D,
    XYZPoly,
    Poly1D,
)
from grdl.IO.sar._backend import require_sarkit


# ===================================================================
# XML extraction helpers (reuse pattern from SICD reader)
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
# Section extractors
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
        # Fallback: classification may be a child element
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

    # Extract first band's DRA from InteractiveProcessing
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

    # Determine projection type
    proj_type = None
    plane_proj = None

    pp = meas.find('{*}PlaneProjection')
    if pp is not None:
        proj_type = 'PlaneProjection'

        ref_point = None
        rp = pp.find('{*}ReferencePoint')
        if rp is not None:
            ref_point = SIDDReferencePoint(
                ecef=_xml_xyz(rp, '{*}ECF'),
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


# ===================================================================
# SIDDReader
# ===================================================================

class SIDDReader(ImageReader):
    """Read SIDD (Sensor Independent Derived Data) format.

    SIDD is the NGA standard for derived SAR products (processed
    imagery) in NITF containers. A single file may contain multiple
    product images. Requires sarkit (no sarpy fallback).

    Parameters
    ----------
    filepath : str or Path
        Path to the SIDD file (NITF container).
    image_index : int, optional
        Index of the product image to read (default 0). SIDD files
        can contain multiple product images.

    Attributes
    ----------
    filepath : Path
        Path to the SIDD file.
    metadata : SIDDMetadata
        Complete typed metadata with all SIDD sections.
    image_index : int
        Active product image index.

    Raises
    ------
    ImportError
        If sarkit is not installed.
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
        require_sarkit('SIDD')
        self.image_index = image_index
        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Load SIDD metadata using sarkit — all sections."""
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

            # Extract dimensions
            num_rows = xml.findtext(
                '{*}Measurement/{*}PixelFootprint/{*}Row'
            )
            num_cols = xml.findtext(
                '{*}Measurement/{*}PixelFootprint/{*}Col'
            )
            pixel_type = xml.findtext('{*}Display/{*}PixelType')

            # Map pixel type to numpy dtype
            dtype_map = {
                'MONO8I': 'uint8',
                'MONO8LU': 'uint8',
                'MONO16I': 'uint16',
                'RGB24I': 'uint8',
                'RGB8LU': 'uint8',
            }
            dtype_str = dtype_map.get(pixel_type, 'uint8')

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
            )

            self._xmltree = xml

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

        data, _ = self._reader.read_image_sub_image(
            self.image_index,
            start_row=row_start,
            start_col=col_start,
            stop_row=row_end,
            stop_col=col_end,
        )
        return data

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
        return self._reader.read_image(self.image_index)

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
        if hasattr(self, '_reader') and self._reader is not None:
            self._reader.done()
        if hasattr(self, '_file_handle') and self._file_handle is not None:
            self._file_handle.close()
