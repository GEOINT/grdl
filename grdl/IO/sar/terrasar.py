# -*- coding: utf-8 -*-
"""
TerraSAR-X / TanDEM-X Reader - DLR TSX/TDX SAR product reader.

Reads TerraSAR-X and TanDEM-X products from the standard DLR directory
structure.  Handles both SSC products (COSAR binary complex data) and
detected products (MGD, GEC, EEC via GeoTIFF).  A single reader
instance opens one polarization channel.

Annotation, georef, and calibration XMLs are parsed into typed
``TerraSARMetadata`` dataclasses.  Calibration is **not** applied
automatically -- the calibration constant is stored in metadata for
the user to apply as needed.

Dependencies
------------
rasterio (only for detected products -- MGD/GEC/EEC via GeoTIFF)

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
from pathlib import Path
from typing import List, Optional, Tuple, Union
import struct
import xml.etree.ElementTree as ET

# Third-party
import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

# GRDL internal
from grdl.IO.base import ImageReader
from grdl.IO.models.common import XYZ, LatLonHAE
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


# ===================================================================
# XML parsing helpers
# ===================================================================

def _xml_text(
    elem: Optional[ET.Element], path: str,
) -> Optional[str]:
    """Extract text from XML path, returning None if absent."""
    if elem is None:
        return None
    return elem.findtext(path)


def _xml_float(
    elem: Optional[ET.Element], path: str,
) -> Optional[float]:
    """Extract float from XML path, returning None if absent."""
    val = _xml_text(elem, path)
    return float(val) if val is not None else None


def _xml_int(
    elem: Optional[ET.Element], path: str,
) -> Optional[int]:
    """Extract int from XML path, returning None if absent."""
    val = _xml_text(elem, path)
    return int(val) if val is not None else None


def _strip_namespace(root: ET.Element) -> ET.Element:
    """Strip XML namespace URIs from all tag names in the tree.

    TerraSAR-X annotation XMLs may include a default namespace
    declaration.  Stripping it allows ``findtext()`` to work with
    plain tag names.

    Parameters
    ----------
    root : ET.Element
        Root element of the parsed XML tree.

    Returns
    -------
    ET.Element
        Same root element with namespace prefixes removed.
    """
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
    return root


# ===================================================================
# Annotation XML section extractors
# ===================================================================

def _extract_product_info(root: ET.Element) -> TSXProductInfo:
    """Extract product-level metadata from generalHeader + productInfo."""
    hdr = root.find('generalHeader')
    pi = root.find('productInfo')
    acq = pi.find('acquisitionInfo') if pi is not None else None
    mi = pi.find('missionInfo') if pi is not None else None
    si = pi.find('sceneInfo') if pi is not None else None
    proc = root.find('processing')

    # Polarization list
    pol_list: Optional[List[str]] = None
    if acq is not None:
        pol_elems = acq.findall('polarisationList/polLayer')
        if pol_elems:
            pol_list = [p.text for p in pol_elems if p.text is not None]

    # TSX XMLs use 'mission' in generalHeader for the satellite ID
    # (e.g. 'TSX-1').  Some products may also include 'satellite'.
    mission_name = _xml_text(hdr, 'mission') or _xml_text(mi, 'mission')
    satellite_name = _xml_text(hdr, 'satellite') or mission_name

    return TSXProductInfo(
        mission=mission_name,
        satellite=satellite_name,
        generation_time=_xml_text(hdr, 'generationTime'),
        product_type=(
            _xml_text(pi, 'productVariantInfo/productVariant')
            or _xml_text(pi, 'productVariantInfo/productType')
        ) if pi is not None else None,
        imaging_mode=_xml_text(acq, 'imagingMode'),
        look_direction=_xml_text(acq, 'lookDirection'),
        polarization_mode=_xml_text(acq, 'polarisationMode'),
        polarization_list=pol_list,
        orbit_direction=_xml_text(mi, 'orbitDirection'),
        absolute_orbit=_xml_int(mi, 'absOrbit'),
        start_time_utc=_xml_text(si, 'start/timeUTC'),
        stop_time_utc=_xml_text(si, 'stop/timeUTC'),
        processor_version=_xml_text(
            proc, 'processingInfo/processorVersion'
        ) if proc is not None else None,
    )


def _extract_scene_info(root: ET.Element) -> TSXSceneInfo:
    """Extract scene geometry from productInfo/sceneInfo."""
    si = root.find('productInfo/sceneInfo')
    if si is None:
        return TSXSceneInfo()

    # Scene corners
    corners: Optional[List[LatLonHAE]] = None
    corner_elems = si.findall('sceneCornerCoord')
    if corner_elems:
        corners = []
        for c in corner_elems:
            corners.append(LatLonHAE(
                lat=float(c.findtext('lat') or 0),
                lon=float(c.findtext('lon') or 0),
                hae=float(c.findtext('height') or 0),
            ))

    center = si.find('sceneCenterCoord')

    return TSXSceneInfo(
        center_lat=_xml_float(center, 'lat'),
        center_lon=_xml_float(center, 'lon'),
        scene_extent=corners,
        incidence_angle_near=_xml_float(si, 'rangeNearIncidenceAngle'),
        incidence_angle_far=_xml_float(si, 'rangeFarIncidenceAngle'),
        incidence_angle_center=_xml_float(center, 'incidenceAngle')
        if center is not None else None,
        heading_angle=_xml_float(si, 'headingAngle'),
    )


def _extract_image_info(root: ET.Element) -> TSXImageInfo:
    """Extract image dimensions and type from productInfo/imageDataInfo."""
    idi = root.find('productInfo/imageDataInfo')
    raster = idi.find('imageRaster') if idi is not None else None

    # Try productSpecific for projection info on SSC products
    ps = root.find('productSpecific/complexImageInfo')
    projection = _xml_text(ps, 'projection') if ps is not None else None
    if projection is None and raster is not None:
        projection = _xml_text(raster, 'projection')

    return TSXImageInfo(
        num_rows=_xml_int(raster, 'numberOfRows') or 0,
        num_cols=_xml_int(raster, 'numberOfColumns') or 0,
        row_spacing=_xml_float(raster, 'rowSpacing'),
        col_spacing=_xml_float(raster, 'columnSpacing'),
        sample_type=_xml_text(idi, 'imageDataType'),
        data_format=_xml_text(idi, 'imageDataFormat'),
        bits_per_sample=_xml_int(raster, 'bitsPerSample'),
        projection=projection,
    )


def _extract_radar_params(root: ET.Element) -> TSXRadarParams:
    """Extract radar parameters from instrument/radarParameters.

    PRF and bandwidth may be under ``productSpecific/complexImageInfo``
    (as commonPRF / commonRSF) or nested inside
    ``instrument/settings/settingRecord/PRF``.  We search all locations.
    """
    rp = root.find('instrument/radarParameters')
    ps = root.find('productSpecific/complexImageInfo')

    # PRF: try complexImageInfo first, then settingRecord, then rp
    prf = _xml_float(ps, 'commonPRF') if ps is not None else None
    if prf is None:
        prf = _xml_float(
            root, 'instrument/settings/settingRecord/PRF'
        )
    if prf is None:
        prf = _xml_float(rp, 'prf')

    # Bandwidth: try settings/rxBandwidth, then rp/totalBandwidth
    bw = _xml_float(root, 'instrument/settings/rxBandwidth')
    if bw is None:
        bw = _xml_float(rp, 'totalBandwidth')

    # ADC sampling rate: try complexImageInfo/commonRSF, then settings
    adc = _xml_float(ps, 'commonRSF') if ps is not None else None
    if adc is None:
        adc = _xml_float(root, 'instrument/settings/RSF')
    if adc is None:
        adc = _xml_float(rp, 'adcSamplingRate')

    return TSXRadarParams(
        center_frequency=_xml_float(rp, 'centerFrequency'),
        prf=prf,
        range_bandwidth=bw,
        chirp_duration=_xml_float(rp, 'chirpDuration'),
        adc_sampling_rate=adc,
    )


def _extract_orbit_state_vectors(
    root: ET.Element,
) -> List[TSXOrbitStateVector]:
    """Extract orbit state vectors from platform/orbit/stateVec."""
    orbit = root.find('platform/orbit')
    if orbit is None:
        return []

    vectors: List[TSXOrbitStateVector] = []
    for sv in orbit.findall('stateVec'):
        position = XYZ(
            x=float(sv.findtext('posX') or 0),
            y=float(sv.findtext('posY') or 0),
            z=float(sv.findtext('posZ') or 0),
        )
        velocity = XYZ(
            x=float(sv.findtext('velX') or 0),
            y=float(sv.findtext('velY') or 0),
            z=float(sv.findtext('velZ') or 0),
        )
        vectors.append(TSXOrbitStateVector(
            time_utc=sv.findtext('timeUTC'),
            position=position,
            velocity=velocity,
        ))

    return vectors


def _extract_doppler_info(root: ET.Element) -> TSXDopplerInfo:
    """Extract Doppler centroid from processing/doppler."""
    doppler = root.find('processing/doppler')
    if doppler is None:
        return TSXDopplerInfo()

    # Doppler centroid polynomial coefficients
    coef_elem = doppler.find('dopplerCentroid')
    coefs: Optional[np.ndarray] = None
    ref_time: Optional[float] = None

    if coef_elem is not None:
        ref_time = _xml_float(coef_elem, 'referenceTime')
        coef_values = coef_elem.findall('coefficient')
        if coef_values:
            coefs = np.array(
                [float(c.text) for c in coef_values if c.text is not None],
                dtype=np.float64,
            )
        else:
            # Some products store as space-separated text
            coef_text = coef_elem.findtext('dopplerCentroidPolynomial')
            if coef_text is not None and coef_text.strip():
                coefs = np.array(
                    [float(x) for x in coef_text.split()],
                    dtype=np.float64,
                )

    return TSXDopplerInfo(
        doppler_centroid_coefficients=coefs,
        reference_time=ref_time,
    )


def _extract_processing_info(root: ET.Element) -> TSXProcessingInfo:
    """Extract processing info from processing/processingInfo."""
    pi = root.find('processing/processingInfo')
    if pi is None:
        return TSXProcessingInfo()

    return TSXProcessingInfo(
        processor_version=_xml_text(pi, 'processorVersion'),
        processing_level=_xml_text(pi, 'processingLevel'),
        range_looks=_xml_int(pi, 'rangeLooks'),
        azimuth_looks=_xml_int(pi, 'azimuthLooks'),
    )


# ===================================================================
# GEOREF.xml extractor
# ===================================================================

def _extract_geolocation_grid(
    georef_path: Path,
) -> List[TSXGeoGridPoint]:
    """Parse geolocation grid tie points from GEOREF.xml.

    Parameters
    ----------
    georef_path : Path
        Path to the GEOREF.xml file.

    Returns
    -------
    List[TSXGeoGridPoint]
        Geolocation grid points.
    """
    if not georef_path.exists():
        return []

    tree = ET.parse(str(georef_path))
    root = _strip_namespace(tree.getroot())

    points: List[TSXGeoGridPoint] = []
    for gp in root.iter('gridPoint'):
        points.append(TSXGeoGridPoint(
            line=float(gp.findtext('lin') or gp.findtext('line') or 0),
            pixel=float(gp.findtext('pix') or gp.findtext('pixel') or 0),
            latitude=float(gp.findtext('lat') or 0),
            longitude=float(gp.findtext('lon') or 0),
            height=float(gp.findtext('height') or 0),
            incidence_angle=_xml_float(gp, 'incidenceAngle'),
        ))

    return points


# ===================================================================
# CALDATA.xml extractor
# ===================================================================

def _extract_calibration(caldata_path: Path) -> TSXCalibration:
    """Parse calibration data from CALDATA.xml.

    Parameters
    ----------
    caldata_path : Path
        Path to the CALDATA.xml file.

    Returns
    -------
    TSXCalibration
        Calibration metadata.
    """
    if not caldata_path.exists():
        return TSXCalibration()

    tree = ET.parse(str(caldata_path))
    root = _strip_namespace(tree.getroot())

    return TSXCalibration(
        calibration_constant=_xml_float(root, './/calFactor'),
        noise_equivalent_beta_nought=_xml_float(
            root, './/noiseEquivalentBetaNought'
        ),
        calibration_type=_xml_text(root, './/calibrationType'),
    )


# ===================================================================
# Directory resolution
# ===================================================================

def _resolve_tsx_product_dir(
    filepath: Path,
) -> Tuple[Path, Path]:
    """Resolve path to the TSX product directory and main annotation XML.

    Accepts either:
    - The product directory directly (contains the main XML).
    - The main annotation XML file itself.
    - A data file within the product tree (walks up to find it).

    Parameters
    ----------
    filepath : Path
        User-provided path.

    Returns
    -------
    product_dir : Path
        The product directory root.
    main_xml : Path
        Path to the main annotation XML file.

    Raises
    ------
    ValueError
        If no valid TerraSAR-X product structure is found.
    """
    filepath = filepath.resolve()

    if filepath.is_dir():
        xml_file = _find_main_xml(filepath)
        if xml_file is not None:
            return filepath, xml_file
        raise ValueError(
            f"No TerraSAR-X annotation XML found in: {filepath}"
        )

    # File path: could be the main XML or a file inside the product
    if filepath.suffix.lower() == '.xml':
        parent = filepath.parent
        # Check if this XML is the main annotation
        xml_file = _find_main_xml(parent)
        if xml_file is not None:
            return parent, xml_file

    # Walk up from a data file (e.g., IMAGEDATA/*.cos)
    for parent in filepath.parents:
        xml_file = _find_main_xml(parent)
        if xml_file is not None:
            return parent, xml_file
        if parent == parent.parent:
            break

    raise ValueError(
        f"Could not find a TerraSAR-X product directory for: {filepath}"
    )


def _find_main_xml(directory: Path) -> Optional[Path]:
    """Find the main annotation XML in a product directory.

    Searches for XML files whose name starts with ``TSX1_SAR`` or
    ``TDX1_SAR``, or matches the directory name.

    Parameters
    ----------
    directory : Path
        Directory to search.

    Returns
    -------
    Path or None
        Main annotation XML path, or None if not found.
    """
    dir_name = directory.name
    for f in sorted(directory.iterdir()):
        if not f.is_file() or f.suffix.lower() != '.xml':
            continue
        name_upper = f.stem.upper()
        if name_upper.startswith(('TSX1_SAR', 'TDX1_SAR')):
            return f
        if f.stem == dir_name:
            return f
    return None


# ===================================================================
# COSAR binary format
# ===================================================================

# COSAR burst header: 8 x uint32 (big-endian)
#   BIB  – Bytes in Burst (total burst size)
#   RSRI – Range Sample Real Index (usually 0)
#   RSNB – Range Samples per line (number of complex samples)
#   AS   – Azimuth Samples (number of data lines)
#   BI   – Burst Index (1-based)
#   RTNB – Range Total Number of Bytes per line (annotation + data)
#   TNL  – Total Number of Lines (header lines + data lines)
#   Magic – b'CSAR'
_COSAR_HEADER_FMT = '>7I4s'
_COSAR_HEADER_SIZE = struct.calcsize(_COSAR_HEADER_FMT)  # 32 bytes


@dataclass
class _COSARHeader:
    """Parsed COSAR burst header."""

    rsnb: int           # range samples per line
    azimuth_lines: int  # number of data lines (AS)
    rtnb: int           # bytes per range line (including annotation)
    tnl: int            # total lines (header + data)
    header_lines: int   # TNL - AS
    annotation_bytes: int  # RTNB - RSNB * 4  (per data line prefix)
    sample_size: int    # bytes per I/Q component (derived)


def _read_cosar_header(filepath: Path) -> _COSARHeader:
    """Read COSAR burst header to get image dimensions.

    Parameters
    ----------
    filepath : Path
        Path to ``.cos`` file.

    Returns
    -------
    _COSARHeader
        Parsed header with image dimensions and layout.

    Raises
    ------
    ValueError
        If the file is too small, magic is wrong, or values
        are invalid.
    """
    with open(str(filepath), 'rb') as fh:
        header_bytes = fh.read(_COSAR_HEADER_SIZE)

    if len(header_bytes) < _COSAR_HEADER_SIZE:
        raise ValueError(
            f"COSAR file too small for header: {filepath}"
        )

    bib, rsri, rsnb, az_samples, bi, rtnb, tnl, magic = (
        struct.unpack(_COSAR_HEADER_FMT, header_bytes)
    )

    if magic != b'CSAR':
        raise ValueError(
            f"Invalid COSAR magic: {magic!r} (expected b'CSAR')"
        )

    if az_samples == 0 or rsnb == 0:
        raise ValueError(
            f"Invalid COSAR header: azimuth_lines={az_samples}, "
            f"range_samples={rsnb}"
        )

    header_lines = tnl - az_samples
    if header_lines < 0:
        raise ValueError(
            f"Invalid COSAR header: TNL={tnl} < AS={az_samples}"
        )

    # Derive sample size from RTNB and RSNB.
    # Each complex sample = 2 components (I + Q), each of sample_size
    # bytes, plus a small per-line annotation prefix.
    # Typically annotation = 8 bytes, sample_size = 2 (int16 I/Q).
    # RTNB = annotation + RSNB * 2 * sample_size
    # Try int16 first (most common), then float32.
    annot_i2 = rtnb - rsnb * 4
    annot_f4 = rtnb - rsnb * 8

    if annot_i2 >= 0 and annot_i2 < rsnb:
        sample_size = 2
        annotation_bytes = annot_i2
    elif annot_f4 >= 0 and annot_f4 < rsnb:
        sample_size = 4
        annotation_bytes = annot_f4
    else:
        raise ValueError(
            f"Cannot derive COSAR sample size: RTNB={rtnb}, "
            f"RSNB={rsnb}"
        )

    return _COSARHeader(
        rsnb=rsnb,
        azimuth_lines=az_samples,
        rtnb=rtnb,
        tnl=tnl,
        header_lines=header_lines,
        annotation_bytes=annotation_bytes,
        sample_size=sample_size,
    )


def _read_cosar_chip(
    filepath: Path,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    header: _COSARHeader,
) -> np.ndarray:
    """Read a chip from a COSAR binary file.

    Reads interleaved I/Q sample pairs and returns a complex64 array.

    Parameters
    ----------
    filepath : Path
        Path to ``.cos`` file.
    row_start : int
        Starting row (inclusive).
    row_end : int
        Ending row (exclusive).
    col_start : int
        Starting column (inclusive).
    col_end : int
        Ending column (exclusive).
    header : _COSARHeader
        Parsed COSAR header.

    Returns
    -------
    np.ndarray
        Complex64 array of shape ``(row_end - row_start,
        col_end - col_start)``.
    """
    rtnb = header.rtnb
    annotation_bytes = header.annotation_bytes
    sample_size = header.sample_size
    data_start_line = header.header_lines

    # Sample dtype for I/Q components (COSAR is big-endian)
    if sample_size == 2:
        sample_dtype = np.dtype('>i2')
    else:
        sample_dtype = np.dtype('>f4')

    chip_rows = row_end - row_start
    chip_cols = col_end - col_start
    result = np.empty((chip_rows, chip_cols), dtype=np.complex64)

    with open(str(filepath), 'rb') as fh:
        for i, row in enumerate(range(row_start, row_end)):
            # Offset: (header_lines + row) * RTNB + annotation
            # + col_start * 2 * sample_size  (I/Q pair)
            line_offset = (
                (data_start_line + row) * rtnb
                + annotation_bytes
                + col_start * 2 * sample_size
            )
            fh.seek(line_offset)

            # Read I/Q pairs for the requested columns
            raw = fh.read(chip_cols * 2 * sample_size)
            iq = np.frombuffer(raw, dtype=sample_dtype)
            iq = iq.reshape(chip_cols, 2)

            result[i] = (
                iq[:, 0].astype(np.float32)
                + 1j * iq[:, 1].astype(np.float32)
            )

    return result


# ===================================================================
# TerraSARReader
# ===================================================================

class TerraSARReader(ImageReader):
    """Read TerraSAR-X / TanDEM-X SAR products.

    Handles SSC products (COSAR binary, complex data) and detected
    products (MGD, GEC, EEC via GeoTIFF).  Each instance opens one
    polarization channel from the product directory.

    Parameters
    ----------
    filepath : str or Path
        Path to the product directory, main annotation XML, or a
        data file within the product tree.
    polarization : str
        Polarization to open (``'HH'``, ``'VV'``, ``'HV'``, ``'VH'``).
        Default ``'HH'``.

    Attributes
    ----------
    metadata : TerraSARMetadata
        Complete typed metadata with all annotation, geolocation,
        and calibration sections.

    Raises
    ------
    ImportError
        If rasterio is not installed and the product is a detected
        (GeoTIFF) type.
    FileNotFoundError
        If the product directory does not exist.
    ValueError
        If the product structure is invalid or the requested
        polarization is not available.

    Examples
    --------
    >>> from grdl.IO.sar import TerraSARReader
    >>> with TerraSARReader('TSX1_SAR__SSC.../') as reader:
    ...     chip = reader.read_chip(0, 1000, 0, 1000)
    ...     print(reader.metadata.product_info.imaging_mode)
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        polarization: str = 'HH',
    ) -> None:
        self._requested_polarization = polarization.upper()

        # Resolve product directory and main XML
        self._product_dir, self._main_xml_path = (
            _resolve_tsx_product_dir(Path(filepath))
        )

        # Pre-parse main XML to determine product type
        try:
            tree = ET.parse(str(self._main_xml_path))
            self._xmltree = _strip_namespace(tree.getroot())
        except ET.ParseError as e:
            raise ValueError(
                f"Failed to parse annotation XML: {e}"
            ) from e

        data_format = _xml_text(
            self._xmltree, 'productInfo/imageDataInfo/imageDataFormat'
        )
        self._is_cosar = (
            data_format is not None
            and data_format.upper() == 'COSAR'
        )

        if not self._is_cosar and not _HAS_RASTERIO:
            raise ImportError(
                "Reading TerraSAR-X detected products (GeoTIFF) requires "
                "rasterio. Install with: "
                "conda install -c conda-forge rasterio"
            )

        # Locate image data file for requested polarization
        imagedata_dir = self._product_dir / 'IMAGEDATA'
        if not imagedata_dir.is_dir():
            raise ValueError(
                f"Missing IMAGEDATA directory: {imagedata_dir}"
            )

        self._image_file = self._find_image_file(
            imagedata_dir, self._requested_polarization,
        )
        if self._image_file is None:
            raise ValueError(
                f"No image data file found for polarization "
                f"'{self._requested_polarization}' in {imagedata_dir}"
            )

        # Locate optional annotation files
        ann_dir = self._product_dir / 'ANNOTATION'
        self._georef_path: Optional[Path] = None
        self._caldata_path: Optional[Path] = None

        if ann_dir.is_dir():
            for name in ('GEOREF.xml', 'georef.xml'):
                candidate = ann_dir / name
                if candidate.exists():
                    self._georef_path = candidate
                    break

            cal_dir = ann_dir / 'CALIBRATION'
            if not cal_dir.is_dir():
                cal_dir = ann_dir / 'calibration'
            if cal_dir.is_dir():
                for name in ('CALDATA.xml', 'caldata.xml'):
                    candidate = cal_dir / name
                    if candidate.exists():
                        self._caldata_path = candidate
                        break

        # Read COSAR header if applicable
        self._cosar_header: Optional[_COSARHeader] = None
        self._rasterio_dataset = None

        if self._is_cosar:
            self._cosar_header = _read_cosar_header(self._image_file)

        # Initialize base (validates path, calls _load_metadata)
        super().__init__(self._product_dir)

    @staticmethod
    def _find_image_file(
        imagedata_dir: Path,
        polarization: str,
    ) -> Optional[Path]:
        """Find image data file for a given polarization.

        Searches IMAGEDATA/ for ``.cos`` (COSAR) or ``.tif``
        (GeoTIFF) files whose names contain the polarization string.

        Parameters
        ----------
        imagedata_dir : Path
            IMAGEDATA directory.
        polarization : str
            Uppercase polarization (e.g., ``'HH'``).

        Returns
        -------
        Path or None
            Path to matching image file.
        """
        pol_upper = polarization.upper()
        for f in sorted(imagedata_dir.iterdir()):
            if not f.is_file():
                continue
            if f.suffix.lower() not in ('.cos', '.tif', '.tiff'):
                continue
            if pol_upper in f.name.upper():
                return f
        return None

    def _load_metadata(self) -> None:
        """Parse annotation, GEOREF, and calibration XMLs."""
        try:
            root = self._xmltree

            product_info = _extract_product_info(root)
            scene_info = _extract_scene_info(root)
            image_info = _extract_image_info(root)
            radar_params = _extract_radar_params(root)
            orbit_vectors = _extract_orbit_state_vectors(root)
            doppler_info = _extract_doppler_info(root)
            processing_info = _extract_processing_info(root)

            # Geolocation grid (from GEOREF.xml)
            geo_grid: List[TSXGeoGridPoint] = []
            if self._georef_path is not None:
                geo_grid = _extract_geolocation_grid(self._georef_path)

            # Calibration (from CALDATA.xml)
            calibration: Optional[TSXCalibration] = None
            if self._caldata_path is not None:
                calibration = _extract_calibration(self._caldata_path)

            # Determine rows/cols
            if self._is_cosar and self._cosar_header is not None:
                rows = self._cosar_header.azimuth_lines
                cols = self._cosar_header.rsnb
            else:
                rows = image_info.num_rows
                cols = image_info.num_cols

            # Determine dtype
            if self._is_cosar:
                dtype_str = 'complex64'
            else:
                dtype_str = 'float32'

            # Open rasterio dataset for detected products
            if not self._is_cosar:
                self._rasterio_dataset = rasterio.open(
                    str(self._image_file)
                )
                # Cross-check dimensions; prefer TIFF as authoritative
                tiff_rows = self._rasterio_dataset.height
                tiff_cols = self._rasterio_dataset.width
                if tiff_rows != rows or tiff_cols != cols:
                    rows = tiff_rows
                    cols = tiff_cols
                # Use actual dtype from TIFF
                dtype_str = str(self._rasterio_dataset.dtypes[0])

            # Format string
            product_type = product_info.product_type or 'Unknown'
            format_str = f'TerraSAR-X_{product_type}'

            self.metadata = TerraSARMetadata(
                format=format_str,
                rows=rows,
                cols=cols,
                dtype=dtype_str,
                bands=1,
                product_info=product_info,
                scene_info=scene_info,
                image_info=image_info,
                radar_params=radar_params,
                orbit_state_vectors=(
                    orbit_vectors if orbit_vectors else None
                ),
                geolocation_grid=geo_grid if geo_grid else None,
                calibration=calibration,
                doppler_info=doppler_info,
                processing_info=processing_info,
            )

        except ET.ParseError as e:
            raise ValueError(
                f"Failed to parse annotation XML: {e}"
            ) from e

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial chip from the image data.

        For SSC products, reads COSAR binary and returns complex64.
        For detected products, reads GeoTIFF via rasterio.

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
            Ignored for single-polarization products.

        Returns
        -------
        np.ndarray
            Image chip with shape ``(rows, cols)``.
            dtype is ``complex64`` for SSC, format-dependent for
            detected.

        Raises
        ------
        ValueError
            If indices are out of bounds.
        """
        if row_start < 0 or col_start < 0:
            raise ValueError("Start indices must be non-negative")
        if row_end > self.metadata.rows or col_end > self.metadata.cols:
            raise ValueError(
                f"End indices ({row_end}, {col_end}) exceed image "
                f"dimensions ({self.metadata.rows}, {self.metadata.cols})"
            )

        if self._is_cosar:
            return _read_cosar_chip(
                self._image_file,
                row_start, row_end, col_start, col_end,
                self._cosar_header,
            )
        else:
            window = Window(
                col_start, row_start,
                col_end - col_start, row_end - row_start,
            )
            data = self._rasterio_dataset.read(1, window=window)
            return data

    def get_available_polarizations(self) -> List[str]:
        """Return polarizations available in the product.

        Returns
        -------
        List[str]
            E.g., ``['HH']`` or ``['HH', 'VV']``.
        """
        if (self.metadata.product_info is not None
                and self.metadata.product_info.polarization_list
                is not None):
            return list(self.metadata.product_info.polarization_list)
        return [self._requested_polarization]

    def get_shape(self) -> Tuple[int, int]:
        """Get image dimensions.

        Returns
        -------
        Tuple[int, int]
            ``(rows, cols)`` — total lines and samples.
        """
        return (self.metadata.rows, self.metadata.cols)

    def get_dtype(self) -> np.dtype:
        """Get data type.

        Returns
        -------
        np.dtype
            ``complex64`` for SSC, format-dependent for detected.
        """
        return np.dtype(self.metadata.dtype)

    def close(self) -> None:
        """Close resources."""
        if self._rasterio_dataset is not None:
            self._rasterio_dataset.close()
            self._rasterio_dataset = None


# ===================================================================
# Convenience function
# ===================================================================

def open_terrasar(
    filepath: Union[str, Path],
    polarization: str = 'HH',
) -> TerraSARReader:
    """Open a TerraSAR-X / TanDEM-X product.

    Factory function that validates the product structure and
    returns a ``TerraSARReader``.

    Parameters
    ----------
    filepath : str or Path
        Path to the product directory or main annotation XML.
    polarization : str
        Polarization to open. Default ``'HH'``.

    Returns
    -------
    TerraSARReader
        Reader instance.

    Examples
    --------
    >>> from grdl.IO.sar import open_terrasar
    >>> reader = open_terrasar('TSX1_SAR__SSC.../', polarization='HH')
    >>> chip = reader.read_chip(0, 512, 0, 512)
    >>> reader.close()
    """
    return TerraSARReader(filepath, polarization=polarization)
