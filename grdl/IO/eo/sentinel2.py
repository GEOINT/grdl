# -*- coding: utf-8 -*-
"""
Sentinel-2 Reader - Read Sentinel-2 MSI JPEG2000 products.

Sensor-specific reader for Sentinel-2A/2B/2C multispectral imagery.
Wraps JP2Reader for pixel access and extracts Sentinel-2 metadata from
filename patterns, JPEG2000 tags, and SAFE archive XML (``MTD_MSIL*.xml``,
``MTD_TL.xml``) into a typed ``Sentinel2Metadata`` dataclass.

Supports Level-1C (TOA Reflectance) and Level-2A (Surface Reflectance)
products in both standalone JP2 format and SAFE archive structure.

Dependencies
------------
rasterio or glymur (via JP2Reader)

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
2026-02-11

Modified
--------
2026-03-27  Full PSD-based XML parsing for SAFE archives.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union
from xml.etree import ElementTree as ET

import numpy as np

from grdl.IO.base import ImageReader
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
from grdl.IO.jpeg2000 import JP2Reader

logger = logging.getLogger(__name__)


# Band wavelength lookup table (center, min, max) in nm
BAND_WAVELENGTHS = {
    'B01': (443, 432, 453),   # Coastal aerosol
    'B02': (490, 458, 523),   # Blue
    'B03': (560, 543, 578),   # Green
    'B04': (665, 650, 680),   # Red
    'B05': (705, 698, 713),   # Red edge 1
    'B06': (740, 733, 748),   # Red edge 2
    'B07': (783, 773, 793),   # Red edge 3
    'B08': (842, 785, 900),   # NIR
    'B8A': (865, 855, 875),   # NIR narrow
    'B09': (945, 935, 955),   # Water vapor
    'B10': (1375, 1360, 1390), # Cirrus
    'B11': (1610, 1565, 1655), # SWIR 1
    'B12': (2190, 2100, 2280), # SWIR 2
}

BAND_PURPOSES = {
    'B01': 'Coastal aerosol',
    'B02': 'Blue',
    'B03': 'Green',
    'B04': 'Red',
    'B05': 'Red edge 1',
    'B06': 'Red edge 2',
    'B07': 'Red edge 3',
    'B08': 'NIR',
    'B8A': 'NIR narrow',
    'B09': 'Water vapor',
    'B10': 'Cirrus',
    'B11': 'SWIR 1',
    'B12': 'SWIR 2',
}

BAND_RESOLUTIONS = {
    'B01': 60, 'B02': 10, 'B03': 10, 'B04': 10,
    'B05': 20, 'B06': 20, 'B07': 20, 'B08': 10,
    'B8A': 20, 'B09': 60, 'B10': 60, 'B11': 20, 'B12': 20,
}


def _parse_sentinel2_filename(filepath: Path) -> dict:
    """Parse Sentinel-2 metadata from filename.

    Handles three naming conventions:

    1. Standalone band JP2:
       T10SEG_20240115T184719_B04_10m.jp2

    2. SAFE archive directory:
       S2A_MSIL2A_20240115T184719_N0510_R070_T10SEG_20240115T201234.SAFE

    3. Band file within SAFE structure:
       S2*.SAFE/GRANULE/.../IMG_DATA/R10m/T10SEG_20240115T184719_B04_10m.jp2

    Parameters
    ----------
    filepath : Path
        Path to Sentinel-2 file or directory.

    Returns
    -------
    dict
        Extracted metadata fields. Empty dict if parsing fails.
    """
    result = {}
    name = filepath.name
    parts = filepath.parts

    # Check parent directories for SAFE archive name
    for part in parts:
        if part.endswith('.SAFE'):
            match = re.match(
                r'(S2[ABC])_'
                r'(MSIL[12][CA])_'
                r'([0-9]{8}T[0-9]{6})_'
                r'(N[0-9]{4})_'
                r'R([0-9]{3})_'
                r'(T[0-9]{2}[A-Z]{3})_'
                r'([0-9]{8}T[0-9]{6})'
                r'(?:\.SAFE)?$',
                part
            )
            if match:
                result['satellite'] = match.group(1)
                result['product_type'] = match.group(2)
                result['sensing_datetime'] = match.group(3)
                result['baseline_processing'] = match.group(4)
                result['relative_orbit'] = int(match.group(5))
                result['mgrs_tile_id'] = match.group(6)
                result['product_discriminator'] = match.group(7)
                result['processing_level'] = (
                    'L1C' if 'L1C' in match.group(2) else 'L2A'
                )
                break

    # Standalone or in-SAFE band file
    band_match = re.match(
        r'(T[0-9]{2}[A-Z]{3})_'
        r'([0-9]{8}T[0-9]{6})_'
        r'(B[0-9]{1,2}A?)_'
        r'([0-9]{2})m',
        name
    )
    if band_match:
        if 'mgrs_tile_id' not in result:
            result['mgrs_tile_id'] = band_match.group(1)
        if 'sensing_datetime' not in result:
            result['sensing_datetime'] = band_match.group(2)
        result['band_id'] = band_match.group(3)
        result['resolution_tier'] = int(band_match.group(4))

    return result


def _parse_mgrs_tile(mgrs_id: str) -> Tuple[Optional[int], Optional[str]]:
    """Extract UTM zone and latitude band from MGRS tile ID."""
    if not mgrs_id or len(mgrs_id) < 4:
        return None, None
    try:
        return int(mgrs_id[1:3]), mgrs_id[3]
    except (ValueError, IndexError):
        return None, None


def _resolve_jp2_from_safe(safe_dir: Path) -> Optional[Path]:
    """Find a representative JP2 band file inside a SAFE directory."""
    for res_dir in ('R10m', 'R20m', 'R60m'):
        candidates = sorted(safe_dir.rglob(
            f'GRANULE/*/IMG_DATA/{res_dir}/T*_B*.jp2'
        ))
        if candidates:
            return candidates[0]

    candidates = sorted(safe_dir.rglob('GRANULE/*/IMG_DATA/T*_B*.jp2'))
    if candidates:
        return candidates[0]

    fallback = sorted(safe_dir.rglob('*.jp2'))
    return fallback[0] if fallback else None


def _find_safe_dir(filepath: Path) -> Optional[Path]:
    """Walk up from a JP2 file to find the enclosing .SAFE directory."""
    for parent in filepath.parents:
        if parent.suffix == '.SAFE':
            return parent
    return None


def _find_product_xml(safe_dir: Path) -> Optional[Path]:
    """Find MTD_MSIL{1C,2A}.xml in a SAFE directory."""
    for xml in safe_dir.glob('MTD_MSIL*.xml'):
        return xml
    return None


def _find_tile_xml(safe_dir: Path) -> Optional[Path]:
    """Find MTD_TL.xml in the GRANULE subdirectory."""
    candidates = list(safe_dir.rglob('GRANULE/*/MTD_TL.xml'))
    return candidates[0] if candidates else None


# ── XML parsing helpers ──────────────────────────────────────────────

def _text(el: Optional[ET.Element], tag: str) -> Optional[str]:
    """Get text content of a child element, or None."""
    if el is None:
        return None
    child = el.find(tag)
    return child.text.strip() if child is not None and child.text else None


def _float(el: Optional[ET.Element], tag: str) -> Optional[float]:
    """Get float value of a child element, or None."""
    t = _text(el, tag)
    return float(t) if t is not None else None


def _int(el: Optional[ET.Element], tag: str) -> Optional[int]:
    """Get int value of a child element, or None."""
    t = _text(el, tag)
    return int(t) if t is not None else None


def _parse_product_xml(xml_path: Path) -> Tuple[
    Optional[S2ProductInfo],
    Optional[S2QualityIndicators],
    Optional[S2RadiometricInfo],
    Optional[S2Footprint],
]:
    """Parse product-level MTD_MSIL{1C,2A}.xml."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        logger.warning("Failed to parse product XML: %s", xml_path)
        return None, None, None, None

    # Strip namespace for easier xpath
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'

    def _find(path: str) -> Optional[ET.Element]:
        """Find element with or without namespace."""
        el = root.find(path)
        if el is None and ns:
            el = root.find(path.replace('//', f'//{ns}').replace('/', f'/{ns}'))
        return el

    # ── Product Info ──────────────────────────────────────────────
    prod_info = S2ProductInfo()

    # Try multiple paths — PSD structure varies by baseline
    gi = root.find('.//General_Info') or root.find(f'.//{ns}General_Info')
    if gi is None:
        # Flat structure: elements directly under root
        gi = root

    pi = (gi.find('Product_Info') or gi.find(f'{ns}Product_Info')
          or gi.find('.//Product_Info') or gi)

    prod_info.product_type = _text(pi, 'PRODUCT_TYPE')
    prod_info.processing_level = _text(pi, 'PROCESSING_LEVEL')
    prod_info.generation_time = _text(pi, 'GENERATION_TIME')
    prod_info.processing_baseline = _text(pi, 'PROCESSING_BASELINE')

    dt = (pi.find('Datatake') or pi.find(f'{ns}Datatake')
          or pi.find('.//Datatake'))
    if dt is not None:
        prod_info.spacecraft_name = _text(dt, 'SPACECRAFT_NAME')
        prod_info.datatake_type = _text(dt, 'DATATAKE_TYPE')
        prod_info.datatake_sensing_start = _text(dt, 'DATATAKE_SENSING_START')
        orbit_num = _text(dt, 'SENSING_ORBIT_NUMBER')
        if orbit_num:
            try:
                prod_info.sensing_orbit_number = int(orbit_num)
            except ValueError:
                pass
        prod_info.sensing_orbit_direction = _text(dt, 'SENSING_ORBIT_DIRECTION')

    prod_info.product_start_time = _text(pi, 'PRODUCT_START_TIME')
    prod_info.product_stop_time = _text(pi, 'PRODUCT_STOP_TIME')
    prod_info.product_doi = _text(pi, 'PRODUCT_DOI')

    # ── Quality Indicators ────────────────────────────────────────
    quality = S2QualityIndicators()
    qi = root.find('.//Quality_Indicators_Info')
    if qi is None and ns:
        qi = root.find(f'.//{ns}Quality_Indicators_Info')

    if qi is not None:
        quality.cloud_coverage_assessment = _float(qi, 'Cloud_Coverage_Assessment')

        # Image content QI
        ic = qi.find('.//Image_Content_QI') or qi.find('Image_Content_QI')
        if ic is not None:
            quality.nodata_pixel_percentage = _float(ic, 'NODATA_PIXEL_PERCENTAGE')
            quality.saturated_defective_pixel_percentage = _float(
                ic, 'SATURATED_DEFECTIVE_PIXEL_PERCENTAGE')
            quality.dark_features_percentage = _float(ic, 'DARK_FEATURES_PERCENTAGE')
            quality.cloud_shadow_percentage = _float(ic, 'CLOUD_SHADOW_PERCENTAGE')
            quality.vegetation_percentage = _float(ic, 'VEGETATION_PERCENTAGE')
            quality.not_vegetated_percentage = _float(ic, 'NOT_VEGETATED_PERCENTAGE')
            quality.water_percentage = _float(ic, 'WATER_PERCENTAGE')
            quality.unclassified_percentage = _float(ic, 'UNCLASSIFIED_PERCENTAGE')
            quality.medium_proba_clouds_percentage = _float(
                ic, 'MEDIUM_PROBA_CLOUDS_PERCENTAGE')
            quality.high_proba_clouds_percentage = _float(
                ic, 'HIGH_PROBA_CLOUDS_PERCENTAGE')
            quality.thin_cirrus_percentage = _float(ic, 'THIN_CIRRUS_PERCENTAGE')
            quality.snow_ice_percentage = _float(ic, 'SNOW_ICE_PERCENTAGE')

    # ── Radiometric Info ──────────────────────────────────────────
    radiometric = None
    pic = (gi.find('.//Product_Image_Characteristics')
           or gi.find('Product_Image_Characteristics'))
    if pic is not None:
        radiometric = S2RadiometricInfo()
        qv = _float(pic, 'QUANTIFICATION_VALUE')
        if qv is None:
            # L2A path
            qv_el = pic.find('.//BOA_QUANTIFICATION_VALUE')
            if qv_el is not None and qv_el.text:
                qv = float(qv_el.text)
        radiometric.quantification_value = qv
        radiometric.u_sun = _float(pic, 'U')
        radiometric.reflectance_conversion_factor = _float(
            pic, 'Reflectance_Conversion/U')

        # Per-band radio add offset (L2A baseline >= 04.00)
        offsets = {}
        for rao in pic.findall('.//RADIO_ADD_OFFSET'):
            bid = rao.get('band_id')
            if bid is not None and rao.text:
                try:
                    band_name = f"B{int(bid):02d}" if bid != '8A' else 'B8A'
                    offsets[band_name] = int(rao.text)
                except ValueError:
                    pass
        if offsets:
            radiometric.radio_add_offset = offsets

        # Per-band solar irradiance
        irradiance = {}
        for si in pic.findall('.//SOLAR_IRRADIANCE'):
            bid = si.get('bandId')
            if bid is not None and si.text:
                try:
                    irradiance[f"B{int(bid):02d}" if bid != '8A' else 'B8A'] = float(si.text)
                except ValueError:
                    pass
        if irradiance:
            radiometric.irradiance_values = irradiance

    # ── Product Footprint ─────────────────────────────────────────
    footprint = None
    fp_el = root.find('.//Product_Footprint')
    if fp_el is None and ns:
        fp_el = root.find(f'.//{ns}Product_Footprint')
    if fp_el is not None:
        coords_el = fp_el.find('.//EXT_POS_LIST')
        if coords_el is not None and coords_el.text:
            values = coords_el.text.strip().split()
            coords = []
            for i in range(0, len(values) - 1, 2):
                try:
                    coords.append((float(values[i]), float(values[i + 1])))
                except ValueError:
                    pass
            if coords:
                footprint = S2Footprint(coordinates=coords, crs='EPSG:4326')

    return prod_info, quality, radiometric, footprint


def _parse_tile_xml(xml_path: Path) -> Tuple[
    Optional[S2TileGeocoding],
    Optional[S2AngleGrid],
    Optional[List[S2AngleGrid]],
    Optional[S2MeanAngles],
]:
    """Parse granule-level MTD_TL.xml."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        logger.warning("Failed to parse tile XML: %s", xml_path)
        return None, None, None, None

    # ── Tile Geocoding ────────────────────────────────────────────
    geocoding = None
    tg = root.find('.//Tile_Geocoding')
    if tg is not None:
        geocoding = S2TileGeocoding()
        geocoding.horizontal_cs_name = _text(tg, 'HORIZONTAL_CS_NAME')
        geocoding.horizontal_cs_code = _text(tg, 'HORIZONTAL_CS_CODE')

        for size_el in tg.findall('Size'):
            res = size_el.get('resolution')
            nrows = _int(size_el, 'NROWS')
            ncols = _int(size_el, 'NCOLS')
            if res == '10':
                geocoding.nrows_10m = nrows
                geocoding.ncols_10m = ncols
            elif res == '20':
                geocoding.nrows_20m = nrows
                geocoding.ncols_20m = ncols
            elif res == '60':
                geocoding.nrows_60m = nrows
                geocoding.ncols_60m = ncols

        for geo_el in tg.findall('Geoposition'):
            res = geo_el.get('resolution')
            ulx = _float(geo_el, 'ULX')
            uly = _float(geo_el, 'ULY')
            xdim = _float(geo_el, 'XDIM')
            ydim = _float(geo_el, 'YDIM')
            if res == '10':
                geocoding.ulx = ulx
                geocoding.uly = uly
                geocoding.x_dim_10m = xdim
                geocoding.y_dim_10m = ydim
            elif res == '20':
                geocoding.x_dim_20m = xdim
                geocoding.y_dim_20m = ydim
            elif res == '60':
                geocoding.x_dim_60m = xdim
                geocoding.y_dim_60m = ydim
            # Use 10m geoposition for ULX/ULY if not yet set
            if geocoding.ulx is None:
                geocoding.ulx = ulx
                geocoding.uly = uly

    # ── Sun Angles ────────────────────────────────────────────────
    sun_grid = None
    mean_angles = None

    ta = root.find('.//Tile_Angles') or root.find('.//Sun_Angles_Grid')
    if ta is None:
        # Some products put angles directly under Geometric_Info
        ta = root.find('.//Geometric_Info')

    sa = None
    if ta is not None:
        sa = ta.find('Sun_Angles_Grid')
        if sa is None:
            sa = ta.find('.//Sun_Angles_Grid')

    if sa is not None:
        sun_grid = _parse_angle_grid_element(sa)

    # ── Mean Sun Angle ────────────────────────────────────────────
    msa = None
    if ta is not None:
        msa = ta.find('Mean_Sun_Angle')
        if msa is None:
            msa = ta.find('.//Mean_Sun_Angle')
    if msa is not None:
        mean_angles = S2MeanAngles(
            sun_zenith=_float(msa, 'ZENITH_ANGLE'),
            sun_azimuth=_float(msa, 'AZIMUTH_ANGLE'),
        )

    # ── Mean Viewing Angles ───────────────────────────────────────
    if ta is not None:
        # Compute mean from per-band mean viewing angles
        zen_vals, azi_vals = [], []
        for mvi in (ta.findall('.//Mean_Viewing_Incidence_Angle')
                    or ta.findall('Mean_Viewing_Incidence_Angle')):
            z = _float(mvi, 'ZENITH_ANGLE')
            a = _float(mvi, 'AZIMUTH_ANGLE')
            if z is not None:
                zen_vals.append(z)
            if a is not None:
                azi_vals.append(a)
        if zen_vals:
            if mean_angles is None:
                mean_angles = S2MeanAngles()
            mean_angles.viewing_zenith = float(np.mean(zen_vals))
            mean_angles.viewing_azimuth = float(np.mean(azi_vals))

    # ── Per-Band Viewing Angle Grids ──────────────────────────────
    viewing_grids = []
    if ta is not None:
        for vag in (ta.findall('.//Viewing_Incidence_Angles_Grids')
                    or ta.findall('Viewing_Incidence_Angles_Grids')):
            grid = _parse_angle_grid_element(vag)
            if grid is not None:
                bid = vag.get('bandId')
                did = vag.get('detectorId')
                if bid is not None:
                    try:
                        grid.band_id = int(bid)
                    except ValueError:
                        pass
                if did is not None:
                    try:
                        grid.detector_id = int(did)
                    except ValueError:
                        pass
                viewing_grids.append(grid)

    return (
        geocoding,
        sun_grid,
        viewing_grids if viewing_grids else None,
        mean_angles,
    )


def _parse_angle_grid_element(el: ET.Element) -> Optional[S2AngleGrid]:
    """Parse a Sun_Angles_Grid or Viewing_Incidence_Angles_Grids element."""
    zenith_el = el.find('Zenith')
    azimuth_el = el.find('Azimuth')

    if zenith_el is None and azimuth_el is None:
        return None

    grid = S2AngleGrid()

    if zenith_el is not None:
        grid.col_step = _float(zenith_el, 'COL_STEP')
        grid.row_step = _float(zenith_el, 'ROW_STEP')
        grid.zenith = _parse_values_list(zenith_el)

    if azimuth_el is not None:
        grid.azimuth = _parse_values_list(azimuth_el)
        if grid.col_step is None:
            grid.col_step = _float(azimuth_el, 'COL_STEP')
            grid.row_step = _float(azimuth_el, 'ROW_STEP')

    return grid


def _parse_values_list(parent: ET.Element) -> Optional[np.ndarray]:
    """Parse a <Values_List> element into a 2D numpy array."""
    vl = parent.find('Values_List')
    if vl is None:
        return None

    rows = []
    for val_el in vl.findall('VALUES'):
        if val_el.text:
            row = []
            for v in val_el.text.strip().split():
                try:
                    row.append(float(v))
                except ValueError:
                    row.append(np.nan)
            rows.append(row)

    if not rows:
        return None

    return np.array(rows, dtype=np.float64)


def _build_spectral_bands() -> List[S2SpectralBand]:
    """Build the static spectral band table from PSD Table 4."""
    bands = []
    for bid, (center, wl_min, wl_max) in BAND_WAVELENGTHS.items():
        bands.append(S2SpectralBand(
            band_id=bid,
            resolution=BAND_RESOLUTIONS[bid],
            wavelength_center=float(center),
            wavelength_min=float(wl_min),
            wavelength_max=float(wl_max),
            bandwidth=float(wl_max - wl_min),
            purpose=BAND_PURPOSES.get(bid, ''),
        ))
    return bands


class Sentinel2Reader(ImageReader):
    """Read Sentinel-2 MSI JPEG2000 products.

    Wraps JP2Reader for pixel access and extracts Sentinel-2-specific
    metadata from filename patterns, JPEG2000 tags, and SAFE archive XML
    into a typed ``Sentinel2Metadata`` dataclass.

    Parameters
    ----------
    filepath : str or Path
        Path to a Sentinel-2 JP2 file, or a ``.SAFE`` directory.
    backend : str, optional
        JP2 backend (``'rasterio'``, ``'glymur'``, ``'auto'``).

    Attributes
    ----------
    filepath : Path
        Path to the JP2 file.
    metadata : Sentinel2Metadata
        Typed Sentinel-2 metadata.
    jp2_reader : JP2Reader
        Wrapped JP2Reader for pixel access.

    Examples
    --------
    >>> with Sentinel2Reader('product.SAFE') as reader:
    ...     print(reader.metadata.satellite)
    ...     print(reader.metadata.quality.cloud_coverage_assessment)
    ...     print(reader.metadata.mean_angles.sun_zenith)
    ...     chip = reader.read_chip(0, 1000, 0, 1000)
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        backend: str = 'auto',
    ) -> None:
        self._backend = backend
        self.jp2_reader: Optional[JP2Reader] = None

        resolved = Path(filepath)
        self._safe_dir: Optional[Path] = None

        if resolved.is_dir() and resolved.suffix == '.SAFE':
            self._safe_dir = resolved
            jp2 = _resolve_jp2_from_safe(resolved)
            if jp2 is None:
                raise ValueError(
                    f"No JP2 band files found in SAFE directory: {resolved}"
                )
            logger.info("Resolved SAFE directory to band file: %s", jp2.name)
            filepath = jp2

        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Load metadata from JP2, filename, and SAFE XML."""
        # Open JP2
        try:
            self.jp2_reader = JP2Reader(self.filepath, backend=self._backend)
        except Exception as e:
            raise ValueError(
                f"Failed to open Sentinel-2 JP2: {self.filepath}: {e}"
            ) from e

        jp2_meta = self.jp2_reader.metadata
        rows = jp2_meta['rows']
        cols = jp2_meta['cols']
        dtype = jp2_meta['dtype']
        bands = jp2_meta.get('bands', 1)

        # ── Filename parsing ──────────────────────────────────────
        parsed = _parse_sentinel2_filename(self.filepath)

        # CRS from rasterio
        crs_str = None
        if hasattr(self.jp2_reader, 'dataset') and self.jp2_reader.dataset:
            ds = self.jp2_reader.dataset
            if hasattr(ds, 'crs') and ds.crs:
                crs_str = str(ds.crs)

        # MGRS → UTM zone + latitude band
        utm_zone, latitude_band = None, None
        if 'mgrs_tile_id' in parsed:
            utm_zone, latitude_band = _parse_mgrs_tile(parsed['mgrs_tile_id'])

        # Wavelength lookup
        wavelength_center, wavelength_range = None, None
        if 'band_id' in parsed and parsed['band_id'] in BAND_WAVELENGTHS:
            center, wl_min, wl_max = BAND_WAVELENGTHS[parsed['band_id']]
            wavelength_center = float(center)
            wavelength_range = (float(wl_min), float(wl_max))

        # Format sensing datetime
        sensing_dt = parsed.get('sensing_datetime')
        if sensing_dt and 'T' in sensing_dt and len(sensing_dt) == 15:
            sensing_dt = (f"{sensing_dt[:4]}-{sensing_dt[4:6]}-"
                          f"{sensing_dt[6:11]}:{sensing_dt[11:13]}:"
                          f"{sensing_dt[13:]}")

        # ── Geolocation from rasterio (first-class fields) ────────
        transform = None
        bounds = None
        pixel_resolution = None
        if hasattr(self.jp2_reader, 'dataset') and self.jp2_reader.dataset:
            ds = self.jp2_reader.dataset
            if hasattr(ds, 'transform'):
                transform = ds.transform
            if hasattr(ds, 'bounds'):
                bounds = tuple(ds.bounds)
            if hasattr(ds, 'res'):
                pixel_resolution = ds.res

        extras = jp2_meta.get('extras', {}).copy()

        # ── SAFE XML parsing ──────────────────────────────────────
        product_info = None
        quality = None
        radiometric = None
        footprint = None
        tile_geocoding = None
        sun_angles = None
        viewing_angles = None
        mean_angles = None

        safe_dir = self._safe_dir or _find_safe_dir(self.filepath)

        if safe_dir is not None:
            # Product-level XML
            product_xml = _find_product_xml(safe_dir)
            if product_xml is not None:
                product_info, quality, radiometric, footprint = (
                    _parse_product_xml(product_xml)
                )
                logger.debug("Parsed product XML: %s", product_xml.name)

            # Tile-level XML
            tile_xml = _find_tile_xml(safe_dir)
            if tile_xml is not None:
                tile_geocoding, sun_angles, viewing_angles, mean_angles = (
                    _parse_tile_xml(tile_xml)
                )
                logger.debug("Parsed tile XML: %s", tile_xml.name)

                # Use tile geocoding CRS if rasterio didn't provide one
                if crs_str is None and tile_geocoding is not None:
                    crs_str = tile_geocoding.horizontal_cs_code

        # ── Build metadata ────────────────────────────────────────
        self.metadata = Sentinel2Metadata(
            format=f"Sentinel-2_{parsed.get('processing_level', 'L2A')}",
            rows=rows,
            cols=cols,
            dtype=dtype,
            bands=bands,
            crs=crs_str,
            nodata=0,
            extras=extras,
            # Filename-derived
            satellite=parsed.get('satellite'),
            processing_level=parsed.get('processing_level'),
            product_type=parsed.get('product_type'),
            sensing_datetime=sensing_dt,
            mgrs_tile_id=parsed.get('mgrs_tile_id'),
            band_id=parsed.get('band_id'),
            resolution_tier=parsed.get('resolution_tier'),
            baseline_processing=parsed.get('baseline_processing'),
            relative_orbit=parsed.get('relative_orbit'),
            product_discriminator=parsed.get('product_discriminator'),
            utm_zone=utm_zone,
            latitude_band=latitude_band,
            wavelength_center=wavelength_center,
            wavelength_range=wavelength_range,
            # Geolocation (first-class)
            transform=transform,
            bounds=bounds,
            pixel_resolution=pixel_resolution,
            # Product XML
            product_info=product_info,
            quality=quality,
            radiometric=radiometric,
            footprint=footprint,
            # Tile XML
            tile_geocoding=tile_geocoding,
            sun_angles=sun_angles,
            viewing_angles=viewing_angles,
            mean_angles=mean_angles,
            # Static
            spectral_bands=_build_spectral_bands(),
        )

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial chip from the Sentinel-2 JP2 file."""
        return self.jp2_reader.read_chip(
            row_start, row_end, col_start, col_end, bands=bands
        )

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """Read the entire Sentinel-2 image."""
        return self.jp2_reader.read_full(bands=bands)

    def get_shape(self) -> Tuple[int, ...]:
        """Get image dimensions."""
        return self.jp2_reader.get_shape()

    def get_dtype(self) -> np.dtype:
        """Get the data type."""
        return self.jp2_reader.get_dtype()

    def close(self) -> None:
        """Close the wrapped JP2Reader."""
        if self.jp2_reader is not None:
            self.jp2_reader.close()
