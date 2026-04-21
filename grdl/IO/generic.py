# -*- coding: utf-8 -*-
"""
Generic Image Reader - GDAL fallback reader with universal open_any entry point.

Provides ``GDALFallbackReader``, a reader that uses GDAL (via rasterio)
to open any GDAL-supported raster format and extract all available
metadata.  Classifies the data modality (SAR, EO, IR, MSI, HSI) using
GDAL header clues such as NITF ICAT, data type, and band count.

Also provides ``open_any()``, a universal entry point that tries all
specialized GRDL readers first, falls back to GDAL, and finally
delegates to ``InvasiveProbeReader`` for truly unknown formats.

Dependencies
------------
rasterio (optional)

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
2026-04-01
"""

from __future__ import annotations

# Standard library
import importlib
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
import logging

from grdl.exceptions import DependencyError, UnsupportedFormatError
from grdl.IO.base import ImageReader
from grdl.IO.models import ImageMetadata

_log = logging.getLogger(__name__)
from grdl.IO.performance import (
    ReadConfig,
    _ensure_gdal_threads,
    _resolve_workers,
    chunked_parallel_read,
    parallel_band_read,
)
from grdl.IO.probe import InvasiveProbeReader


# ===================================================================
# Modality classification
# ===================================================================

_SAR_KEYWORDS = frozenset({
    'SAR', 'RADAR', 'SICD', 'SIDD', 'CPHD', 'CRSD',
})
_EO_KEYWORDS = frozenset({
    'VIS', 'VISUAL', 'PAN', 'PANCHROMATIC', 'EO', 'OPTICAL',
})
_IR_KEYWORDS = frozenset({
    'IR', 'THERMAL', 'INFRARED', 'TIR', 'SWIR', 'MWIR', 'LWIR',
})
_MSI_KEYWORDS = frozenset({
    'MS', 'MULTISPECTRAL', 'MULTI',
})
_HSI_KEYWORDS = frozenset({
    'HYPERSPECTRAL', 'HSI', 'IMAGING_SPECTROMETER',
})


def _classify_modality(
    driver: str,
    dtype: str,
    bands: int,
    tags: Dict[str, str],
) -> Tuple[Optional[str], str, List[str]]:
    """Classify data modality from GDAL metadata clues.

    Parameters
    ----------
    driver : str
        GDAL driver name (e.g., ``'GTiff'``, ``'NITF'``).
    dtype : str
        Primary data type string (e.g., ``'float32'``, ``'complex64'``).
    bands : int
        Number of image bands.
    tags : Dict[str, str]
        GDAL metadata tags (default domain).

    Returns
    -------
    modality : str or None
        Classified modality: ``'SAR'``, ``'EO'``, ``'IR'``, ``'MSI'``,
        ``'HSI'``, or ``None`` if unknown.
    confidence : str
        Classification confidence: ``'high'``, ``'medium'``, or ``'low'``.
    clues : List[str]
        Human-readable list of evidence supporting the classification.
    """
    clues: List[str] = []
    modality: Optional[str] = None
    confidence = 'low'

    # --- NITF-specific fields ---
    icat = tags.get('NITF_ICAT', '').upper().strip()
    irep = tags.get('NITF_IREP', '').upper().strip()
    ftitle = tags.get('NITF_FTITLE', '').upper()
    isorce = tags.get('NITF_ISORCE', '').upper()

    # NITF ICAT is the most reliable indicator
    if icat:
        for kw_set, mod_name in [
            (_SAR_KEYWORDS, 'SAR'),
            (_IR_KEYWORDS, 'IR'),
            (_MSI_KEYWORDS, 'MSI'),
            (_HSI_KEYWORDS, 'HSI'),
            (_EO_KEYWORDS, 'EO'),
        ]:
            if icat in kw_set or any(kw in icat for kw in kw_set):
                modality = mod_name
                confidence = 'high'
                clues.append(f"NITF ICAT='{icat}'")
                break

    # Search other NITF text fields for keywords
    if modality is None:
        text_fields = f"{irep} {ftitle} {isorce}"
        for kw_set, mod_name in [
            (_SAR_KEYWORDS, 'SAR'),
            (_IR_KEYWORDS, 'IR'),
            (_MSI_KEYWORDS, 'MSI'),
            (_HSI_KEYWORDS, 'HSI'),
            (_EO_KEYWORDS, 'EO'),
        ]:
            for kw in kw_set:
                if kw in text_fields:
                    modality = mod_name
                    confidence = 'medium'
                    clues.append(f"keyword '{kw}' in NITF metadata")
                    break
            if modality is not None:
                break

    # --- Data type ---
    if 'complex' in dtype:
        if modality is None:
            modality = 'SAR'
            confidence = 'high'
        clues.append(f"complex dtype '{dtype}'")

    # --- GDAL driver hints ---
    _sar_drivers = {'COSAR', 'TSX', 'RS2', 'RCM', 'SAFE', 'SAR_CEOS'}
    if driver in _sar_drivers:
        if modality is None:
            modality = 'SAR'
            confidence = 'high'
        clues.append(f"GDAL driver '{driver}'")

    if driver == 'SENTINEL2':
        if modality is None:
            modality = 'MSI'
            confidence = 'high'
        clues.append(f"GDAL driver '{driver}'")

    # --- Band count heuristics ---
    if modality is None:
        if bands > 100:
            modality = 'HSI'
            confidence = 'medium'
            clues.append(f"{bands} bands (hyperspectral range)")
        elif bands > 10:
            modality = 'MSI'
            confidence = 'medium'
            clues.append(f"{bands} bands (multispectral range)")
        elif bands in (3, 4):
            modality = 'EO'
            confidence = 'low'
            clues.append(f"{bands} bands (RGB/RGBA)")
        elif bands == 1:
            clues.append("single band (ambiguous)")

    if not clues:
        clues.append("no classification clues found")

    return modality, confidence, clues


# ===================================================================
# GDAL metadata extraction
# ===================================================================

def _extract_gdal_metadata(
    ds: 'rasterio.DatasetReader',
) -> Dict[str, Any]:
    """Extract all available metadata from a GDAL dataset.

    Collects geospatial info, band descriptions, NITF header fields,
    compression, overviews, and metadata from multiple GDAL domains.

    Parameters
    ----------
    ds : rasterio.DatasetReader
        Open rasterio dataset.

    Returns
    -------
    Dict[str, Any]
        Flat dictionary of all extracted metadata.
    """
    meta: Dict[str, Any] = {}

    # Driver / format
    meta['gdal_driver'] = ds.driver

    # Geospatial
    if ds.transform:
        meta['transform'] = ds.transform
    if ds.bounds:
        meta['bounds'] = ds.bounds
    if ds.res:
        meta['resolution'] = ds.res

    # Band descriptions
    if ds.descriptions and any(d is not None for d in ds.descriptions):
        meta['band_descriptions'] = list(ds.descriptions)

    # Color interpretation
    if ds.colorinterp:
        meta['color_interpretation'] = [
            ci.name for ci in ds.colorinterp
        ]

    # Compression
    if ds.compression:
        meta['compression'] = (
            ds.compression.name
            if hasattr(ds.compression, 'name')
            else str(ds.compression)
        )

    # Interleaving
    if ds.interleaving:
        meta['interleaving'] = (
            ds.interleaving.name
            if hasattr(ds.interleaving, 'name')
            else str(ds.interleaving)
        )

    # Per-band dtypes (useful if mixed)
    if len(set(ds.dtypes)) > 1:
        meta['band_dtypes'] = list(ds.dtypes)

    # Block / tile sizes
    if hasattr(ds, 'block_shapes'):
        meta['block_shapes'] = ds.block_shapes

    # Overviews
    for i in range(1, ds.count + 1):
        ovr = ds.overviews(i)
        if ovr:
            meta['overview_levels'] = ovr
            break

    # Default metadata domain tags
    default_tags = ds.tags()
    if default_tags:
        for k, v in default_tags.items():
            meta[f'tag_{k}'] = v

    # Additional GDAL metadata domains
    for domain in ('NITF_METADATA', 'xml:DES', 'RPC', 'IMAGERY',
                    'IMD', 'xml:XMP'):
        try:
            domain_tags = ds.tags(ns=domain)
            if domain_tags:
                for k, v in domain_tags.items():
                    meta[f'{domain}_{k}'] = v
        except Exception:
            pass

    return meta


# ===================================================================
# Driver-to-format mapping
# ===================================================================

_DRIVER_FORMAT_MAP: Dict[str, str] = {
    'GTiff': 'GeoTIFF',
    'NITF': 'NITF',
    'HDF5': 'HDF5',
    'HDF5Image': 'HDF5',
    'JP2OpenJPEG': 'JPEG2000',
    'JP2KAK': 'JPEG2000',
    'JP2ECW': 'JPEG2000',
    'JPEG2000': 'JPEG2000',
    'PNG': 'PNG',
    'JPEG': 'JPEG',
    'BMP': 'BMP',
    'MrSID': 'MrSID',
    'ECW': 'ECW',
    'ENVI': 'ENVI',
    'EHdr': 'ESRI_HDR',
    'HFA': 'ERDAS_IMG',
    'VRT': 'VRT',
    'PCIDSK': 'PCIDSK',
    'netCDF': 'NetCDF',
    'GRIB': 'GRIB',
    'ERS': 'ERS',
    'FITS': 'FITS',
    'CEOS': 'CEOS',
    'SAR_CEOS': 'SAR_CEOS',
    'DIMAP': 'DIMAP',
    'SENTINEL2': 'Sentinel-2',
    'COSAR': 'COSAR',
    'TSX': 'TerraSAR-X',
    'RS2': 'RADARSAT-2',
    'RCM': 'RCM',
    'SAFE': 'Sentinel-1',
}


def _driver_to_format(driver: str) -> str:
    """Map GDAL driver name to human-readable format string.

    Parameters
    ----------
    driver : str
        GDAL driver short name.

    Returns
    -------
    str
        Human-readable format name. Falls back to the raw driver
        name if no mapping exists.
    """
    return _DRIVER_FORMAT_MAP.get(driver, driver)


# ===================================================================
# GDALFallbackReader
# ===================================================================

class GDALFallbackReader(ImageReader):
    """Read any GDAL-supported raster with best-effort metadata.

    Fallback reader that uses GDAL (via rasterio) to open raster files
    that no specialized GRDL reader handles.  Extracts all available
    GDAL metadata and classifies the data modality (SAR, EO, IR, MSI,
    HSI) using header clues like NITF ICAT, data type, and band count.

    Parameters
    ----------
    filepath : str or Path
        Path to the raster file.

    Attributes
    ----------
    filepath : Path
        Path to the raster file.
    metadata : ImageMetadata
        Best-effort metadata with classification in extras.
    detected_modality : str or None
        Classified modality (``'SAR'``, ``'EO'``, ``'IR'``, ``'MSI'``,
        ``'HSI'``, or ``None``).
    classification_confidence : str
        Confidence level (``'high'``, ``'medium'``, ``'low'``).
    classification_clues : List[str]
        Reasons supporting the classification.

    Raises
    ------
    ImportError
        If rasterio is not installed.
    ValueError
        If GDAL cannot open the file.

    Examples
    --------
    >>> from grdl.IO.generic import GDALFallbackReader
    >>> with GDALFallbackReader('unknown_file.dat') as reader:
    ...     print(reader.detected_modality)
    ...     print(reader.classification_clues)
    ...     chip = reader.read_chip(0, 512, 0, 512)
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        read_config: Optional[ReadConfig] = None,
    ) -> None:
        if not _HAS_RASTERIO:
            raise DependencyError(
                "rasterio is required for GDAL fallback reading. "
                "Install with: pip install rasterio"
            )
        self.read_config = read_config or ReadConfig()
        self.detected_modality: Optional[str] = None
        self.classification_confidence: str = 'low'
        self.classification_clues: List[str] = []
        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Open file with GDAL and extract all available metadata."""
        try:
            self._dataset = rasterio.open(str(self.filepath))
        except Exception as e:
            raise ValueError(
                f"GDAL cannot open {self.filepath}: {e}"
            ) from e

        ds = self._dataset

        # Extract all GDAL metadata
        extras = _extract_gdal_metadata(ds)

        # Classify modality
        tags = ds.tags() or {}
        modality, confidence, clues = _classify_modality(
            driver=ds.driver,
            dtype=str(ds.dtypes[0]),
            bands=ds.count,
            tags=tags,
        )

        self.detected_modality = modality
        self.classification_confidence = confidence
        self.classification_clues = clues

        extras['detected_modality'] = modality
        extras['classification_confidence'] = confidence
        extras['classification_clues'] = clues

        # Map GDAL driver to format string
        format_name = _driver_to_format(ds.driver)

        self.metadata = ImageMetadata(
            format=format_name,
            rows=ds.height,
            cols=ds.width,
            dtype=str(ds.dtypes[0]),
            bands=ds.count,
            crs=str(ds.crs) if ds.crs else None,
            nodata=ds.nodata,
            extras=extras,
        )

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial chip from the raster.

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

        cfg = self.read_config
        if cfg.parallel:
            _ensure_gdal_threads(cfg)
            workers = _resolve_workers(cfg)
            n_pixels = (row_end - row_start) * (col_end - col_start)

            if bands is None:
                band_indices = list(range(1, self._dataset.count + 1))
            else:
                band_indices = [b + 1 for b in bands]

            if n_pixels >= cfg.chunk_threshold:
                data = chunked_parallel_read(
                    self._dataset, window, band_indices, workers)
            elif len(band_indices) > 1:
                data = parallel_band_read(
                    self._dataset, window, band_indices, workers)
            else:
                data = self._dataset.read(band_indices, window=window)
        else:
            if bands is None:
                data = self._dataset.read(window=window)
            else:
                data = self._dataset.read(
                    [b + 1 for b in bands], window=window,
                )

        if data.shape[0] == 1:
            return data[0]
        return data

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """Read the entire raster.

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
            data = self._dataset.read()
        else:
            data = self._dataset.read([b + 1 for b in bands])

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
        if hasattr(self, '_dataset') and self._dataset is not None:
            self._dataset.close()
            self._dataset = None


# ===================================================================
# Retry logic: GDAL-identified format → specific reader
# ===================================================================

def _retry_identified_reader(
    filepath: Path,
    fallback: GDALFallbackReader,
) -> Optional[ImageReader]:
    """Try a specialized reader based on GDAL classification.

    After the GDAL fallback reader identifies the format/modality, this
    function attempts to open the file with the correct specialized
    reader.  This catches cases where the generic opener cascade skipped
    the right reader (e.g., NITF file with ``.dat`` extension).

    Parameters
    ----------
    filepath : Path
        Path to the file.
    fallback : GDALFallbackReader
        Already-opened GDAL fallback reader with classification.

    Returns
    -------
    ImageReader or None
        Specialized reader if one succeeds, or None.
    """
    driver = fallback.metadata.get('gdal_driver')
    ext = filepath.suffix.lower()

    # NITF with non-standard extension — try NITFReader
    if driver == 'NITF' and ext not in ('.nitf', '.ntf', '.nsf'):
        try:
            mod = importlib.import_module('grdl.IO.nitf')
            return mod.NITFReader(filepath)
        except Exception:
            pass

    # HDF5 with non-standard extension — try HDF5Reader
    if driver in ('HDF5', 'HDF5Image') and ext not in (
        '.h5', '.he5', '.hdf5', '.hdf',
    ):
        try:
            mod = importlib.import_module('grdl.IO.hdf5')
            return mod.HDF5Reader(filepath)
        except Exception:
            pass

    # JPEG2000 with non-standard extension — try JP2Reader
    if driver in ('JP2OpenJPEG', 'JP2KAK', 'JP2ECW', 'JPEG2000') and (
        ext not in ('.jp2', '.j2k', '.j2c', '.jpx')
    ):
        try:
            mod = importlib.import_module('grdl.IO.jpeg2000')
            return mod.JP2Reader(filepath)
        except Exception:
            pass

    return None

# ===================================================================
# open_any — universal entry point
# ===================================================================

def open_any(filepath: Union[str, Path]) -> ImageReader:
    """Open any supported imagery file.

    Tries all specialized GRDL readers first (SAR, EO, IR,
    multispectral, and base format readers).  If none succeed, falls
    back to GDAL to read the file, classifies the data modality, and
    retries the correct specialized reader.  If GDAL also fails,
    performs invasive file probing as a last resort.

    Parameters
    ----------
    filepath : str or Path
        Path to the image file or directory.

    Returns
    -------
    ImageReader
        The most appropriate reader instance.  Returns
        ``GDALFallbackReader`` if no specialized reader can open the
        file but GDAL can, or ``InvasiveProbeReader`` if only
        non-cooperative probing succeeds.

    Raises
    ------
    ValueError
        If no reader (including invasive probing) can open the file.

    Examples
    --------
    >>> from grdl.IO.generic import open_any
    >>> with open_any('scene.ntf') as reader:
    ...     print(reader.metadata.format)
    ...     chip = reader.read_chip(0, 512, 0, 512)

    >>> # Unknown format — invasive probing
    >>> with open_any('mystery.dat') as reader:
    ...     print(reader.metadata.get('probe_clues'))
    ...     print(reader.metadata.get('probe_loading_strategy'))
    """
    filepath = Path(filepath)

    # --- Phase 1: Try modality-specific openers ---
    _openers = [
        ('grdl.IO.sar', 'open_sar'),
        ('grdl.IO.eo', 'open_eo'),
        ('grdl.IO.ir', 'open_ir'),
        ('grdl.IO.multispectral', 'open_multispectral'),
    ]

    # Keep the open_sar error: it carries actionable guidance (e.g. "use
    # Open Directory") and is chained into the final exception so grdk
    # can surface it in the error dialog.
    _sar_error: Optional[Exception] = None
    for mod_path, func_name in _openers:
        try:
            mod = importlib.import_module(mod_path)
            opener = getattr(mod, func_name)
            return opener(filepath)
        except UnsupportedFormatError:
            raise  # known wrong level — propagate immediately, skip all fallbacks
        except (ValueError, ImportError, Exception) as exc:
            if func_name == 'open_sar' and isinstance(exc, ValueError):
                _sar_error = exc
            # All other opener errors are silently skipped — they simply
            # indicate the format doesn't match, not a user-facing problem.

    # --- Phase 2: Try base format readers directly ---
    _readers = [
        ('grdl.IO.geotiff', 'GeoTIFFReader'),
        ('grdl.IO.nitf', 'NITFReader'),
        ('grdl.IO.hdf5', 'HDF5Reader'),
        ('grdl.IO.jpeg2000', 'JP2Reader'),
    ]

    for mod_path, cls_name in _readers:
        try:
            mod = importlib.import_module(mod_path)
            reader_cls = getattr(mod, cls_name)
            return reader_cls(filepath)
        except (ValueError, ImportError, Exception):
            pass

    # --- Phase 3: GDAL fallback with classification ---
    try:
        fallback = GDALFallbackReader(filepath)
    except (ImportError, ValueError):
        fallback = None

    if fallback is not None:
        # --- Phase 4: Retry identified reader ---
        reader = _retry_identified_reader(filepath, fallback)
        if reader is not None:
            fallback.close()
            return reader
        return fallback

    # --- Phase 5: Invasive probe (non-cooperative) ---
    try:
        return InvasiveProbeReader(filepath)
    except (ValueError, ImportError):
        pass

    _log.warning(
        "open_any: all readers failed for %s. "
        "Directory-structured formats (Sentinel-1 .SAFE, BIOMASS, TerraSAR-X) "
        "must be opened as a directory via 'Open'.",
        filepath,
    )
    # Build the error message.  If open_sar produced a specific rejection
    # message (e.g. "use Open to select the product directory"), surface it
    # directly so the GUI shows the actionable text rather than a generic
    # wrapper.  Fall back to a generic message when no specialised reader
    # even attempted the file.
    generic_msg = (
        f"No reader can open '{filepath.name}'. "
        "Tried all specialized readers, GDAL fallback, and invasive file probing. "
        "For directory-structured formats (Sentinel-1 .SAFE, BIOMASS, TerraSAR-X), "
        "use 'Open' and select the product directory."
    )
    if _sar_error:
        sar_msg = str(_sar_error)
        raise ValueError(f"{sar_msg}\n\n({generic_msg})") from _sar_error
    raise ValueError(generic_msg)
