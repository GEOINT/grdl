# -*- coding: utf-8 -*-
"""
Sentinel-1 SLC Reader - ESA Sentinel-1 IW SLC SAFE products.

Reads Sentinel-1 IW SLC data from the standard ESA SAFE directory
structure.  Each reader instance opens one swath+polarization
combination (one measurement TIFF), exposing burst-level and
chip-level access to the complex SLC data.

Annotation, calibration, and noise XMLs are parsed into typed
``Sentinel1SLCMetadata`` dataclasses.  Calibration is **not**
applied automatically — LUT vectors are stored in metadata for
the user to apply as needed.

Dependencies
------------
rasterio

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
from pathlib import Path
from typing import List, Optional, Tuple, Union
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
from grdl.IO.models.common import XYZ
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


def _parse_space_separated_ints(text: Optional[str]) -> Optional[np.ndarray]:
    """Parse whitespace-separated integers into an ndarray."""
    if text is None or not text.strip():
        return None
    return np.array([int(x) for x in text.split()], dtype=np.int32)


def _parse_space_separated_floats(text: Optional[str]) -> Optional[np.ndarray]:
    """Parse whitespace-separated floats into an ndarray."""
    if text is None or not text.strip():
        return None
    return np.array([float(x) for x in text.split()], dtype=np.float64)


# ===================================================================
# Annotation XML section extractors
# ===================================================================

def _extract_product_info(root: ET.Element) -> S1SLCProductInfo:
    """Extract product-level metadata from adsHeader + generalAnnotation."""
    hdr = root.find('adsHeader')
    prod = root.find('generalAnnotation/productInformation')

    return S1SLCProductInfo(
        mission=_xml_text(hdr, 'missionId'),
        mode=_xml_text(hdr, 'mode'),
        product_type=_xml_text(hdr, 'productType'),
        transmit_receive_polarization=None,
        start_time=_xml_text(hdr, 'startTime'),
        stop_time=_xml_text(hdr, 'stopTime'),
        absolute_orbit=_xml_int(hdr, 'absoluteOrbitNumber'),
        relative_orbit=None,
        orbit_pass=_xml_text(prod, 'pass') if prod is not None else None,
        processing_facility=None,
        processing_time=None,
        ipf_version=None,
    )


def _extract_swath_info(root: ET.Element) -> S1SLCSwathInfo:
    """Extract swath/image geometry from annotation XML."""
    hdr = root.find('adsHeader')
    img = root.find('imageAnnotation/imageInformation')
    prod = root.find('generalAnnotation/productInformation')

    return S1SLCSwathInfo(
        swath=_xml_text(hdr, 'swath'),
        polarization=_xml_text(hdr, 'polarisation'),
        lines=_xml_int(img, 'numberOfLines') or 0,
        samples=_xml_int(img, 'numberOfSamples') or 0,
        range_pixel_spacing=_xml_float(img, 'rangePixelSpacing'),
        azimuth_pixel_spacing=_xml_float(img, 'azimuthPixelSpacing'),
        azimuth_time_interval=_xml_float(img, 'azimuthTimeInterval'),
        slant_range_time=_xml_float(img, 'slantRangeTime'),
        incidence_angle_mid=_xml_float(img, 'incidenceAngleMidSwath'),
        azimuth_steering_rate=_xml_float(prod, 'azimuthSteeringRate')
        if prod is not None else None,
        range_sampling_rate=_xml_float(prod, 'rangeSamplingRate')
        if prod is not None else None,
        radar_frequency=_xml_float(prod, 'radarFrequency')
        if prod is not None else None,
        azimuth_frequency=_xml_float(img, 'azimuthFrequency'),
    )


def _extract_bursts(root: ET.Element) -> Tuple[List[S1SLCBurst], int, int]:
    """Extract burst list from swathTiming.

    Returns
    -------
    bursts : List[S1SLCBurst]
    lines_per_burst : int
    samples_per_burst : int
    """
    st = root.find('swathTiming')
    lines_per_burst = _xml_int(st, 'linesPerBurst') or 0
    samples_per_burst = _xml_int(st, 'samplesPerBurst') or 0

    bursts: List[S1SLCBurst] = []
    burst_list = st.find('burstList') if st is not None else None
    if burst_list is None:
        return bursts, lines_per_burst, samples_per_burst

    for i, b in enumerate(burst_list.findall('burst')):
        fvs = _parse_space_separated_ints(b.findtext('firstValidSample'))
        lvs = _parse_space_separated_ints(b.findtext('lastValidSample'))

        bursts.append(S1SLCBurst(
            index=i,
            azimuth_time=b.findtext('azimuthTime'),
            azimuth_anx_time=_xml_float(b, 'azimuthAnxTime'),
            sensing_time=b.findtext('sensingTime'),
            byte_offset=_xml_int(b, 'byteOffset'),
            first_valid_sample=fvs,
            last_valid_sample=lvs,
            first_line=i * lines_per_burst,
            last_line=(i + 1) * lines_per_burst,
            lines_per_burst=lines_per_burst,
            samples_per_burst=samples_per_burst,
        ))

    return bursts, lines_per_burst, samples_per_burst


def _extract_orbit(root: ET.Element) -> List[S1SLCOrbitStateVector]:
    """Extract orbit state vectors from generalAnnotation/orbitList."""
    orbit_list = root.find('generalAnnotation/orbitList')
    if orbit_list is None:
        return []

    vectors: List[S1SLCOrbitStateVector] = []
    for orb in orbit_list.findall('orbit'):
        pos_elem = orb.find('position')
        vel_elem = orb.find('velocity')

        position = None
        if pos_elem is not None:
            position = XYZ(
                x=float(pos_elem.findtext('x') or 0),
                y=float(pos_elem.findtext('y') or 0),
                z=float(pos_elem.findtext('z') or 0),
            )

        velocity = None
        if vel_elem is not None:
            velocity = XYZ(
                x=float(vel_elem.findtext('x') or 0),
                y=float(vel_elem.findtext('y') or 0),
                z=float(vel_elem.findtext('z') or 0),
            )

        vectors.append(S1SLCOrbitStateVector(
            time=orb.findtext('time'),
            position=position,
            velocity=velocity,
        ))

    return vectors


def _extract_geolocation_grid(
    root: ET.Element,
) -> List[S1SLCGeoGridPoint]:
    """Extract geolocation grid from geolocationGrid."""
    grid_list = root.find(
        'geolocationGrid/geolocationGridPointList'
    )
    if grid_list is None:
        return []

    points: List[S1SLCGeoGridPoint] = []
    for gp in grid_list.findall('geolocationGridPoint'):
        points.append(S1SLCGeoGridPoint(
            line=float(gp.findtext('line') or 0),
            pixel=float(gp.findtext('pixel') or 0),
            latitude=float(gp.findtext('latitude') or 0),
            longitude=float(gp.findtext('longitude') or 0),
            height=float(gp.findtext('height') or 0),
            incidence_angle=_xml_float(gp, 'incidenceAngle'),
            elevation_angle=_xml_float(gp, 'elevationAngle'),
        ))

    return points


def _extract_doppler_centroids(
    root: ET.Element,
) -> List[S1SLCDopplerCentroid]:
    """Extract Doppler centroid estimates from dopplerCentroid."""
    dc_list = root.find('dopplerCentroid/dcEstimateList')
    if dc_list is None:
        return []

    estimates: List[S1SLCDopplerCentroid] = []
    for dc in dc_list.findall('dcEstimate'):
        # Use the data-derived polynomial (more accurate than geometry)
        coefs = _parse_space_separated_floats(
            dc.findtext('dataDcPolynomial')
        )

        estimates.append(S1SLCDopplerCentroid(
            azimuth_time=dc.findtext('azimuthTime'),
            t0=_xml_float(dc, 't0'),
            coefficients=coefs,
        ))

    return estimates


def _extract_doppler_fm_rates(
    root: ET.Element,
) -> List[S1SLCDopplerFmRate]:
    """Extract azimuth FM rate polynomials from generalAnnotation."""
    fm_list = root.find('generalAnnotation/azimuthFmRateList')
    if fm_list is None:
        return []

    rates: List[S1SLCDopplerFmRate] = []
    for fm in fm_list.findall('azimuthFmRate'):
        coefs = _parse_space_separated_floats(
            fm.findtext('azimuthFmRatePolynomial')
        )

        rates.append(S1SLCDopplerFmRate(
            azimuth_time=fm.findtext('azimuthTime'),
            t0=_xml_float(fm, 't0'),
            coefficients=coefs,
        ))

    return rates


# ===================================================================
# Calibration XML extractor
# ===================================================================

def _extract_calibration_vectors(
    cal_path: Path,
) -> List[S1SLCCalibrationVector]:
    """Parse calibration XML into calibration vectors."""
    if not cal_path.exists():
        return []

    tree = ET.parse(str(cal_path))
    root = tree.getroot()
    vec_list = root.find('calibrationVectorList')
    if vec_list is None:
        return []

    vectors: List[S1SLCCalibrationVector] = []
    for cv in vec_list.findall('calibrationVector'):
        vectors.append(S1SLCCalibrationVector(
            azimuth_time=cv.findtext('azimuthTime'),
            line=int(cv.findtext('line') or 0),
            pixel=_parse_space_separated_ints(cv.findtext('pixel')),
            sigma_nought=_parse_space_separated_floats(
                cv.findtext('sigmaNought')
            ),
            beta_nought=_parse_space_separated_floats(
                cv.findtext('betaNought')
            ),
            gamma=_parse_space_separated_floats(cv.findtext('gamma')),
            dn=_parse_space_separated_floats(cv.findtext('dn')),
        ))

    return vectors


# ===================================================================
# Noise XML extractor
# ===================================================================

def _extract_noise_vectors(
    noise_path: Path,
) -> Tuple[List[S1SLCNoiseRangeVector], List[S1SLCNoiseAzimuthVector]]:
    """Parse noise XML into range and azimuth noise vectors."""
    if not noise_path.exists():
        return [], []

    tree = ET.parse(str(noise_path))
    root = tree.getroot()

    # Range noise
    range_vectors: List[S1SLCNoiseRangeVector] = []
    rng_list = root.find('noiseRangeVectorList')
    if rng_list is not None:
        for rv in rng_list.findall('noiseRangeVector'):
            range_vectors.append(S1SLCNoiseRangeVector(
                azimuth_time=rv.findtext('azimuthTime'),
                line=int(rv.findtext('line') or 0),
                pixel=_parse_space_separated_ints(rv.findtext('pixel')),
                noise_range_lut=_parse_space_separated_floats(
                    rv.findtext('noiseRangeLut')
                ),
            ))

    # Azimuth noise
    azimuth_vectors: List[S1SLCNoiseAzimuthVector] = []
    az_list = root.find('noiseAzimuthVectorList')
    if az_list is not None:
        for av in az_list.findall('noiseAzimuthVector'):
            azimuth_vectors.append(S1SLCNoiseAzimuthVector(
                swath=av.findtext('swath'),
                first_azimuth_line=int(
                    av.findtext('firstAzimuthLine') or 0
                ),
                last_azimuth_line=int(
                    av.findtext('lastAzimuthLine') or 0
                ),
                first_range_sample=int(
                    av.findtext('firstRangeSample') or 0
                ),
                last_range_sample=int(
                    av.findtext('lastRangeSample') or 0
                ),
                line=_parse_space_separated_ints(av.findtext('line')),
                noise_azimuth_lut=_parse_space_separated_floats(
                    av.findtext('noiseAzimuthLut')
                ),
            ))

    return range_vectors, azimuth_vectors


# ===================================================================
# SAFE directory resolution
# ===================================================================

def _resolve_safe_dir(filepath: Path) -> Path:
    """Resolve path to the .SAFE directory.

    Accepts either:
    - A ``.SAFE`` directory directly.
    - A measurement TIFF within the SAFE tree (walks up to find it).

    Parameters
    ----------
    filepath : Path
        User-provided path.

    Returns
    -------
    Path
        The ``.SAFE`` directory.

    Raises
    ------
    ValueError
        If no valid SAFE directory can be found.
    """
    filepath = filepath.resolve()

    # Direct .SAFE directory
    if filepath.is_dir():
        if (filepath / 'manifest.safe').exists():
            return filepath
        raise ValueError(
            f"Directory does not contain manifest.safe: {filepath}"
        )

    # File inside a SAFE structure — walk up
    for parent in filepath.parents:
        if (parent / 'manifest.safe').exists():
            return parent
        # Stop at filesystem root
        if parent == parent.parent:
            break

    raise ValueError(
        f"Could not find a .SAFE directory for: {filepath}"
    )


def _find_matching_file(
    directory: Path,
    pattern_parts: List[str],
) -> Optional[Path]:
    """Find a file in directory whose name contains all pattern parts.

    Parameters
    ----------
    directory : Path
        Directory to search.
    pattern_parts : list of str
        Substrings that must all appear in the filename (case-insensitive).

    Returns
    -------
    Path or None
        First matching file, or None if not found.
    """
    for f in sorted(directory.iterdir()):
        if f.is_file():
            name_lower = f.name.lower()
            if all(p.lower() in name_lower for p in pattern_parts):
                return f
    return None


def _discover_available(
    annotation_dir: Path,
) -> Tuple[List[str], List[str]]:
    """Discover available swaths and polarizations from annotation dir.

    Returns
    -------
    swaths : List[str]
        Unique swath IDs found (e.g., ``['IW1', 'IW2', 'IW3']``).
    polarizations : List[str]
        Unique polarizations found (e.g., ``['VV', 'VH']``).
    """
    swaths: set = set()
    pols: set = set()

    for f in annotation_dir.iterdir():
        if not f.is_file() or not f.suffix.lower() == '.xml':
            continue
        name = f.name.lower()
        # Match pattern: s1{a,b,c}-iw{1,2,3}-slc-{vv,vh,...}-...
        for sw in ('iw1', 'iw2', 'iw3', 'ew1', 'ew2', 'ew3',
                    'ew4', 'ew5', 's1', 's2', 's3', 's4', 's5', 's6'):
            if f'-{sw}-' in name:
                swaths.add(sw.upper())
                break
        for pol in ('vv', 'vh', 'hh', 'hv'):
            if f'-{pol}-' in name:
                pols.add(pol.upper())
                break

    return sorted(swaths), sorted(pols)


# ===================================================================
# Sentinel1SLCReader
# ===================================================================

class Sentinel1SLCReader(ImageReader):
    """Read Sentinel-1 IW SLC data from ESA SAFE products.

    Each instance opens one swath+polarization combination from the
    SAFE archive. The measurement TIFF is accessed via rasterio; the
    annotation, calibration, and noise XMLs are parsed into typed
    ``Sentinel1SLCMetadata``.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.SAFE`` directory, or to a measurement TIFF
        within it (the reader walks up to find the SAFE root).
    swath : str
        Swath to open (``'IW1'``, ``'IW2'``, ``'IW3'``). Default
        ``'IW1'``.
    polarization : str
        Polarization to open (``'VV'``, ``'VH'``, ``'HH'``, ``'HV'``).
        Default ``'VV'``.

    Attributes
    ----------
    metadata : Sentinel1SLCMetadata
        Complete typed metadata with all annotation, calibration, and
        noise sections.

    Raises
    ------
    ImportError
        If rasterio is not installed.
    FileNotFoundError
        If the SAFE directory does not exist.
    ValueError
        If the requested swath/polarization is not found in the product,
        or the SAFE structure is invalid.

    Examples
    --------
    >>> from grdl.IO.sar import Sentinel1SLCReader
    >>> with Sentinel1SLCReader('product.SAFE', swath='IW1') as reader:
    ...     chip = reader.read_chip(0, 1000, 0, 1000)
    ...     burst = reader.read_burst(0)
    ...     print(reader.get_burst_count())
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        swath: str = 'IW1',
        polarization: str = 'VV',
    ) -> None:
        if not _HAS_RASTERIO:
            raise ImportError(
                "Reading Sentinel-1 SLC requires rasterio. "
                "Install with: conda install -c conda-forge rasterio"
            )

        self._swath = swath.upper()
        self._polarization = polarization.upper()

        # Resolve SAFE directory
        self._safe_dir = _resolve_safe_dir(Path(filepath))

        # Validate structure
        ann_dir = self._safe_dir / 'annotation'
        meas_dir = self._safe_dir / 'measurement'
        if not ann_dir.is_dir():
            raise ValueError(
                f"Missing annotation directory: {ann_dir}"
            )
        if not meas_dir.is_dir():
            raise ValueError(
                f"Missing measurement directory: {meas_dir}"
            )

        # Discover available swaths/polarizations
        self._available_swaths, self._available_polarizations = (
            _discover_available(ann_dir)
        )

        # Locate matching files
        swath_lower = self._swath.lower()
        pol_lower = self._polarization.lower()
        match_parts = [swath_lower, pol_lower, 'slc']

        self._annotation_path = _find_matching_file(ann_dir, match_parts)
        if self._annotation_path is None:
            raise ValueError(
                f"No annotation XML found for swath={self._swath}, "
                f"polarization={self._polarization} in {ann_dir}. "
                f"Available: swaths={self._available_swaths}, "
                f"polarizations={self._available_polarizations}"
            )

        self._measurement_path = _find_matching_file(
            meas_dir, [swath_lower, pol_lower]
        )
        if self._measurement_path is None:
            raise ValueError(
                f"No measurement TIFF found for swath={self._swath}, "
                f"polarization={self._polarization} in {meas_dir}"
            )

        # Calibration and noise paths (optional)
        cal_dir = ann_dir / 'calibration'
        self._calibration_path = _find_matching_file(
            cal_dir, ['calibration', swath_lower, pol_lower]
        ) if cal_dir.is_dir() else None

        self._noise_path = _find_matching_file(
            cal_dir, ['noise', swath_lower, pol_lower]
        ) if cal_dir.is_dir() else None

        # Initialize base (validates path existence, calls _load_metadata)
        super().__init__(self._safe_dir)

    @staticmethod
    def _to_complex(data: np.ndarray) -> np.ndarray:
        """Convert raw TIFF data to complex64.

        Handles multiple possible rasterio representations of Sentinel-1
        SLC complex integer data (CInt16):

        1. Already complex (GDAL CFloat32/CFloat64) → cast to complex64.
        2. Two-band int16 (I in band 0, Q in band 1) → combine.
        3. Structured dtype with 'real'/'imag' fields → combine.

        Parameters
        ----------
        data : np.ndarray
            Raw pixel data from rasterio.

        Returns
        -------
        np.ndarray
            Complex-valued array (``complex64``).
        """
        # Already complex (CInt16 read natively, CFloat32/64)
        if np.iscomplexobj(data):
            result = data.astype(np.complex64)
            # Squeeze leading band dimension: (1, rows, cols) → (rows, cols)
            if result.ndim == 3 and result.shape[0] == 1:
                result = result[0]
            return result
        # Structured dtype with real/imag fields (some GDAL versions)
        if data.dtype.names and 'real' in data.dtype.names:
            return (data['real'].astype(np.float32)
                    + 1j * data['imag'].astype(np.float32))
        # Two-band int16: band 0 = I, band 1 = Q
        if data.ndim == 3 and data.shape[0] == 2:
            return (data[0].astype(np.float32)
                    + 1j * data[1].astype(np.float32))
        return data.astype(np.complex64)

    def _load_metadata(self) -> None:
        """Parse annotation, calibration, noise XMLs and open TIFF."""
        try:
            # Parse annotation XML
            tree = ET.parse(str(self._annotation_path))
            root = tree.getroot()
            self._xmltree = root

            product_info = _extract_product_info(root)
            swath_info = _extract_swath_info(root)
            bursts, lines_per_burst, samples_per_burst = (
                _extract_bursts(root)
            )
            orbit_vectors = _extract_orbit(root)
            geo_grid = _extract_geolocation_grid(root)
            dc_estimates = _extract_doppler_centroids(root)
            fm_rates = _extract_doppler_fm_rates(root)

            # Parse calibration XML
            cal_vectors: List[S1SLCCalibrationVector] = []
            if self._calibration_path is not None:
                cal_vectors = _extract_calibration_vectors(
                    self._calibration_path
                )

            # Parse noise XML
            noise_range: List[S1SLCNoiseRangeVector] = []
            noise_azimuth: List[S1SLCNoiseAzimuthVector] = []
            if self._noise_path is not None:
                noise_range, noise_azimuth = _extract_noise_vectors(
                    self._noise_path
                )

            # Open measurement TIFF
            self._dataset = rasterio.open(str(self._measurement_path))

            # Cross-check dimensions
            tiff_rows = self._dataset.height
            tiff_cols = self._dataset.width
            xml_rows = swath_info.lines
            xml_cols = swath_info.samples

            if tiff_rows != xml_rows or tiff_cols != xml_cols:
                raise ValueError(
                    f"TIFF dimensions ({tiff_rows}x{tiff_cols}) do not "
                    f"match annotation XML ({xml_rows}x{xml_cols})"
                )

            self.metadata = Sentinel1SLCMetadata(
                format='Sentinel-1_IW_SLC',
                rows=xml_rows,
                cols=xml_cols,
                dtype='complex64',
                bands=1,
                product_info=product_info,
                swath_info=swath_info,
                bursts=bursts if bursts else None,
                orbit_state_vectors=orbit_vectors if orbit_vectors else None,
                geolocation_grid=geo_grid if geo_grid else None,
                doppler_centroids=dc_estimates if dc_estimates else None,
                doppler_fm_rates=fm_rates if fm_rates else None,
                calibration_vectors=cal_vectors if cal_vectors else None,
                noise_range_vectors=noise_range if noise_range else None,
                noise_azimuth_vectors=(
                    noise_azimuth if noise_azimuth else None
                ),
                num_bursts=len(bursts),
                lines_per_burst=lines_per_burst,
                samples_per_burst=samples_per_burst,
            )

        except ET.ParseError as e:
            raise ValueError(
                f"Failed to parse annotation XML: {e}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Failed to load Sentinel-1 SLC metadata: {e}"
            ) from e

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial chip from the SLC measurement TIFF.

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
            Ignored for SLC (single complex band).

        Returns
        -------
        np.ndarray
            Complex-valued image chip with shape ``(rows, cols)``
            and dtype ``complex64``.

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

        window = Window(
            col_start, row_start,
            col_end - col_start, row_end - row_start,
        )
        raw = self._dataset.read(window=window)
        return self._to_complex(raw)

    def read_burst(
        self,
        burst_index: int,
        apply_valid_mask: bool = True,
    ) -> np.ndarray:
        """Read a single burst with optional valid-sample masking.

        Parameters
        ----------
        burst_index : int
            Zero-based burst index.
        apply_valid_mask : bool
            If True (default), zero out invalid border samples using
            the ``firstValidSample`` / ``lastValidSample`` arrays from
            the annotation XML.

        Returns
        -------
        np.ndarray
            Complex-valued burst data with shape
            ``(lines_per_burst, samples_per_burst)``.

        Raises
        ------
        IndexError
            If burst_index is out of range.
        """
        if self.metadata.bursts is None or not self.metadata.bursts:
            raise ValueError("No burst information available")
        if burst_index < 0 or burst_index >= len(self.metadata.bursts):
            raise IndexError(
                f"Burst index {burst_index} out of range "
                f"[0, {len(self.metadata.bursts) - 1}]"
            )

        burst = self.metadata.bursts[burst_index]
        data = self.read_chip(
            burst.first_line, burst.last_line,
            0, self.metadata.cols,
        )

        if apply_valid_mask:
            fvs = burst.first_valid_sample
            lvs = burst.last_valid_sample
            if fvs is not None and lvs is not None:
                for i in range(data.shape[0]):
                    if fvs[i] < 0:
                        # Entire line invalid
                        data[i, :] = 0
                    else:
                        if fvs[i] > 0:
                            data[i, :fvs[i]] = 0
                        if lvs[i] < data.shape[1] - 1:
                            data[i, lvs[i] + 1:] = 0

        return data

    def get_burst_count(self) -> int:
        """Return the number of bursts in this swath.

        Returns
        -------
        int
            Number of bursts.
        """
        return self.metadata.num_bursts

    def get_available_swaths(self) -> List[str]:
        """Return swath IDs available in the SAFE product.

        Returns
        -------
        List[str]
            E.g., ``['IW1', 'IW2', 'IW3']``.
        """
        return list(self._available_swaths)

    def get_available_polarizations(self) -> List[str]:
        """Return polarizations available in the SAFE product.

        Returns
        -------
        List[str]
            E.g., ``['VH', 'VV']``.
        """
        return list(self._available_polarizations)

    def get_shape(self) -> Tuple[int, int]:
        """Get full swath image dimensions.

        Returns
        -------
        Tuple[int, int]
            ``(rows, cols)`` — total lines and samples for this swath.
        """
        return (self.metadata.rows, self.metadata.cols)

    def get_dtype(self) -> np.dtype:
        """Get data type.

        Returns
        -------
        np.dtype
            ``complex64`` for SLC data.
        """
        return np.dtype('complex64')

    def close(self) -> None:
        """Close the rasterio dataset and release resources."""
        if hasattr(self, '_dataset') and self._dataset is not None:
            self._dataset.close()
