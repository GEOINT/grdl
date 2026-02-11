# -*- coding: utf-8 -*-
"""
ASTER Reader - Read ASTER L1T and GDEM GeoTIFF products.

Sensor-specific reader for ASTER (Advanced Spaceborne Thermal Emission
and Reflection Radiometer) products. Wraps ``GeoTIFFReader`` for pixel
access and extracts ASTER-specific metadata from GeoTIFF tags and
companion XML files into a typed ``ASTERMetadata`` dataclass.

Supports AST_L1T (Registered Radiance at Sensor) thermal infrared
bands and ASTGTM (Global Digital Elevation Model) products stored as
standard GeoTIFF files.

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
2026-02-10

Modified
--------
2026-02-10
"""

# Standard library
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
from grdl.IO.models import ASTERMetadata


def _parse_aster_xml(xml_path: Path) -> Dict[str, Any]:
    """Parse ASTER companion XML metadata.

    Extracts acquisition, orbital, solar geometry, and quality fields
    from the ASTER L1T or GDEM XML metadata file.

    Parameters
    ----------
    xml_path : Path
        Path to the companion XML file.

    Returns
    -------
    Dict[str, Any]
        Extracted metadata fields.
    """
    result: Dict[str, Any] = {}

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return result

    # Strip namespace prefixes for easier searching
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'

    def _find_text(tag: str) -> Optional[str]:
        """Search for a tag value anywhere in the tree."""
        # Try direct path
        elem = root.find(f".//{ns}{tag}")
        if elem is not None and elem.text:
            return elem.text.strip()
        # Try without namespace
        elem = root.find(f".//{tag}")
        if elem is not None and elem.text:
            return elem.text.strip()
        return None

    def _find_attr_value(name: str) -> Optional[str]:
        """Search for an attribute value in GranuleMetaDataFile format."""
        # ASTER XML uses <PSA><PSAName>key</PSAName><PSAValue>val</PSAValue>
        for psa in root.iter(f'{ns}PSA'):
            psa_name = psa.findtext(f'{ns}PSAName', '')
            if psa_name == name:
                return psa.findtext(f'{ns}PSAValue', '')
        for psa in root.iter('PSA'):
            psa_name = psa.findtext('PSAName', '')
            if psa_name == name:
                return psa.findtext('PSAValue', '')
        return None

    # Acquisition date/time
    cal_date = _find_text('CalendarDate') or _find_attr_value('ASTERProcessing.CalendarDate')
    if cal_date:
        result['acquisition_date'] = cal_date
    time_of_day = _find_text('TimeofDay') or _find_attr_value('ASTERProcessing.TimeOfDay')
    if time_of_day:
        result['acquisition_time'] = time_of_day

    # Entity and granule IDs
    entity_id = _find_text('LocalGranuleID') or _find_text('ShortName')
    if entity_id:
        result['entity_id'] = entity_id
    local_granule = _find_text('LocalGranuleID')
    if local_granule:
        result['local_granule_id'] = local_granule

    # Orbit
    orbit_dir = _find_attr_value('ASTERMapProjection.OrbitDirection')
    if orbit_dir:
        result['orbit_direction'] = orbit_dir
    wrs_path = _find_attr_value('WRS.Path') or _find_text('OrbitPath')
    if wrs_path:
        try:
            result['wrs_path'] = int(wrs_path)
        except ValueError:
            pass
    wrs_row = _find_attr_value('WRS.Row') or _find_text('OrbitRow')
    if wrs_row:
        try:
            result['wrs_row'] = int(wrs_row)
        except ValueError:
            pass

    # Scene center
    center_lat = _find_attr_value('ASTERMapProjection.SceneCenterLatitude')
    if center_lat:
        try:
            result['scene_center_lat'] = float(center_lat)
        except ValueError:
            pass
    center_lon = _find_attr_value('ASTERMapProjection.SceneCenterLongitude')
    if center_lon:
        try:
            result['scene_center_lon'] = float(center_lon)
        except ValueError:
            pass

    # Corner coordinates
    corners: Dict[str, Tuple[float, float]] = {}
    for corner_key, lat_key, lon_key in [
        ('UL', 'UpperLeftCornerLatitude', 'UpperLeftCornerLongitude'),
        ('UR', 'UpperRightCornerLatitude', 'UpperRightCornerLongitude'),
        ('LR', 'LowerRightCornerLatitude', 'LowerRightCornerLongitude'),
        ('LL', 'LowerLeftCornerLatitude', 'LowerLeftCornerLongitude'),
    ]:
        lat_val = _find_attr_value(lat_key) or _find_text(lat_key)
        lon_val = _find_attr_value(lon_key) or _find_text(lon_key)
        if lat_val and lon_val:
            try:
                corners[corner_key] = (float(lat_val), float(lon_val))
            except ValueError:
                pass
    if corners:
        result['corner_coords'] = corners

    # Solar geometry
    sun_az = _find_attr_value('Solar.Azimuth') or _find_text('SolarAzimuth')
    if sun_az:
        try:
            result['sun_azimuth'] = float(sun_az)
        except ValueError:
            pass
    sun_el = _find_attr_value('Solar.Elevation') or _find_text('SolarElevation')
    if sun_el:
        try:
            result['sun_elevation'] = float(sun_el)
        except ValueError:
            pass

    # Cloud cover
    cloud = _find_text('SceneCloudCoverage') or _find_attr_value('CloudCoverage')
    if cloud:
        try:
            result['cloud_cover'] = float(cloud)
        except ValueError:
            pass

    # Correction level
    corr = _find_attr_value('CorrectionLevelID')
    if corr:
        result['correction_level'] = corr

    return result


def _detect_aster_product_type(filepath: Path) -> str:
    """Detect ASTER product type from filename.

    Parameters
    ----------
    filepath : Path
        Path to the ASTER GeoTIFF file.

    Returns
    -------
    str
        Product type identifier (``'L1T'`` or ``'GDEM'``).
    """
    name = filepath.name.upper()
    if 'AST_L1T' in name or 'L1T' in name:
        return 'L1T'
    if 'ASTGTM' in name or 'GDEM' in name or '_DEM' in name:
        return 'GDEM'
    return 'ASTER'


def _detect_band_availability(filepath: Path) -> Dict[str, Optional[bool]]:
    """Detect which ASTER subsystems are present from filename.

    Parameters
    ----------
    filepath : Path
        Path to the ASTER GeoTIFF file.

    Returns
    -------
    Dict[str, Optional[bool]]
        Keys: vnir_available, swir_available, tir_available.
    """
    name = filepath.name.upper()
    result: Dict[str, Optional[bool]] = {
        'vnir_available': None,
        'swir_available': None,
        'tir_available': None,
    }

    # TIR bands: 10-14
    if any(f'B{b}' in name or f'BAND{b}' in name for b in range(10, 15)):
        result['tir_available'] = True
    # VNIR bands: 1-3N, 3B
    if any(f'B{b}' in name or f'BAND{b}' in name for b in range(1, 4)):
        result['vnir_available'] = True
    if 'B3N' in name or 'B3B' in name:
        result['vnir_available'] = True
    # SWIR bands: 4-9
    if any(f'B{b}' in name or f'BAND{b}' in name for b in range(4, 10)):
        result['swir_available'] = True

    # L1T products may contain all subsystems
    if 'L1T' in name and all(v is None for v in result.values()):
        result['tir_available'] = True

    return result


class ASTERReader(ImageReader):
    """Read ASTER L1T and GDEM GeoTIFF products.

    Wraps ``GeoTIFFReader`` for pixel access and extracts ASTER-specific
    metadata from GeoTIFF tags and companion XML files into a typed
    ``ASTERMetadata`` dataclass.

    Parameters
    ----------
    filepath : str or Path
        Path to the ASTER GeoTIFF file.

    Attributes
    ----------
    filepath : Path
        Path to the image file.
    metadata : ASTERMetadata
        Typed ASTER metadata with acquisition, orbital, and sensor fields.
    dataset : rasterio.DatasetReader
        Rasterio dataset for direct access.

    Raises
    ------
    ImportError
        If rasterio is not installed.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be opened as a GeoTIFF.

    Examples
    --------
    >>> from grdl.IO.ir import ASTERReader
    >>> with ASTERReader('AST_L1T_00305042006.tif') as reader:
    ...     print(reader.metadata.processing_level)
    ...     print(reader.metadata.acquisition_date)
    ...     chip = reader.read_chip(0, 512, 0, 512)
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        if not _HAS_RASTERIO:
            raise ImportError(
                "rasterio is required for ASTER reading. "
                "Install with: pip install rasterio"
            )
        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Load ASTER metadata from GeoTIFF tags and companion XML."""
        try:
            self.dataset = rasterio.open(str(self.filepath))
        except Exception as e:
            raise ValueError(
                f"Failed to open ASTER GeoTIFF: {self.filepath}: {e}"
            ) from e

        rows = self.dataset.height
        cols = self.dataset.width
        bands = self.dataset.count
        dtype = str(self.dataset.dtypes[0])
        crs_str = str(self.dataset.crs) if self.dataset.crs else None
        nodata = self.dataset.nodata

        extras: Dict[str, Any] = {
            'transform': self.dataset.transform,
            'bounds': self.dataset.bounds,
            'resolution': self.dataset.res,
        }

        # Detect product type from filename
        product_type = _detect_aster_product_type(self.filepath)

        # Detect band availability from filename
        band_info = _detect_band_availability(self.filepath)

        # Parse companion XML if present
        xml_fields: Dict[str, Any] = {}
        xml_candidates = list(self.filepath.parent.glob(
            self.filepath.stem + '*.xml'
        ))
        if not xml_candidates:
            xml_candidates = list(self.filepath.parent.glob('*.xml'))
        for xml_path in xml_candidates:
            parsed = _parse_aster_xml(xml_path)
            if parsed:
                xml_fields.update(parsed)
                break

        self.metadata = ASTERMetadata(
            format=f'ASTER_{product_type}',
            rows=rows,
            cols=cols,
            dtype=dtype,
            bands=bands,
            crs=crs_str,
            nodata=nodata,
            extras=extras,
            processing_level=product_type,
            acquisition_date=xml_fields.get('acquisition_date'),
            acquisition_time=xml_fields.get('acquisition_time'),
            entity_id=xml_fields.get('entity_id'),
            local_granule_id=xml_fields.get('local_granule_id'),
            orbit_direction=xml_fields.get('orbit_direction'),
            wrs_path=xml_fields.get('wrs_path'),
            wrs_row=xml_fields.get('wrs_row'),
            scene_center_lat=xml_fields.get('scene_center_lat'),
            scene_center_lon=xml_fields.get('scene_center_lon'),
            corner_coords=xml_fields.get('corner_coords'),
            sun_azimuth=xml_fields.get('sun_azimuth'),
            sun_elevation=xml_fields.get('sun_elevation'),
            cloud_cover=xml_fields.get('cloud_cover'),
            vnir_available=band_info.get('vnir_available'),
            swir_available=band_info.get('swir_available'),
            tir_available=band_info.get('tir_available'),
            correction_level=xml_fields.get('correction_level'),
        )

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial chip from the ASTER product.

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
            Band indices to read (0-based). If None, read all bands.

        Returns
        -------
        np.ndarray
            Image chip with shape ``(rows, cols)`` for single band or
            ``(bands, rows, cols)`` for multi-band.

        Raises
        ------
        ValueError
            If indices are out of bounds.
        """
        if row_start < 0 or col_start < 0:
            raise ValueError("Start indices must be non-negative")
        if row_end > self.metadata.rows or col_end > self.metadata.cols:
            raise ValueError("End indices exceed image dimensions")

        window = Window(
            col_start, row_start,
            col_end - col_start, row_end - row_start,
        )

        if bands is None:
            data = self.dataset.read(window=window)
        else:
            data = self.dataset.read([b + 1 for b in bands], window=window)

        if data.shape[0] == 1:
            return data[0]
        return data

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """Read the entire ASTER image.

        Parameters
        ----------
        bands : Optional[List[int]]
            Band indices to read (0-based). If None, read all bands.

        Returns
        -------
        np.ndarray
            Full image data.
        """
        if bands is None:
            data = self.dataset.read()
        else:
            data = self.dataset.read([b + 1 for b in bands])

        if data.shape[0] == 1:
            return data[0]
        return data

    def get_shape(self) -> Tuple[int, ...]:
        """Get image dimensions.

        Returns
        -------
        Tuple[int, ...]
            ``(rows, cols)`` for single band or
            ``(rows, cols, bands)`` for multi-band.
        """
        if self.metadata.bands == 1:
            return (self.metadata.rows, self.metadata.cols)
        return (self.metadata.rows, self.metadata.cols, self.metadata.bands)

    def get_dtype(self) -> np.dtype:
        """Get the data type of the image.

        Returns
        -------
        np.dtype
        """
        return np.dtype(self.metadata.dtype)

    def close(self) -> None:
        """Close the rasterio dataset."""
        if hasattr(self, 'dataset') and self.dataset is not None:
            self.dataset.close()
