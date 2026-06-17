# -*- coding: utf-8 -*-
"""
IO Module - Input/Output Operations for Geospatial Imagery.

Handles reading and writing various geospatial data formats. Base data
format readers (GeoTIFF, NITF, HDF5) live at this level.  Modality-specific
readers are organized into submodules (``sar/``, ``eo/``, ``ir/``,
``multispectral/``).

Dependencies
------------
rasterio
h5py
sarkit (primary) or sarpy (fallback)
requests

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
2026-01-30

Modified
--------
2026-06-17  Add warnings import, tighten open_reader fallback exceptions,
            and export register_reader/register_writer APIs.
2026-05-27
"""

# Standard library
import importlib
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Third-party
import numpy as np

# Base classes and models
from grdl.IO.base import ImageReader, ImageWriter, CatalogInterface
from grdl.IO.models import (
    ImageMetadata, SICDMetadata, SIDDMetadata, BIOMASSMetadata,
    VIIRSMetadata, ASTERMetadata, TerraSARMetadata, NISARMetadata,
)

# Base format readers (IO level)
from grdl.IO.geotiff import GeoTIFFReader
from grdl.IO.hdf5 import HDF5Reader
from grdl.IO.jpeg2000 import JP2Reader
from grdl.IO.nitf import NITFReader

# Base format writers
from grdl.IO.geotiff import GeoTIFFWriter
from grdl.IO.hdf5 import HDF5Writer
from grdl.IO.nitf import NITFWriter
from grdl.IO.numpy_io import NumpyWriter
from grdl.IO.png import PngWriter

# SAR submodule
from grdl.IO.sar import (
    SICDReader,
    CPHDReader,
    CRSDReader,
    SIDDReader,
    BIOMASSL1Reader,
    Sentinel1SLCReader,
    TerraSARReader,
    NISARReader,
    SICDCollectionReader,
    open_sar,
    open_biomass,
    open_terrasar,
    open_nisar,
    open_sicd_collection,
)

# SAR writers
from grdl.IO.sar.sicd_writer import SICDWriter
from grdl.IO.sar.sidd_writer import SIDDWriter

# Catalog submodule
from grdl.IO.catalog import (
    BIOMASSCatalog,
    Sentinel1SLCCatalog,
    TerraSARCatalog,
    NISARCatalog,
    Sentinel2Catalog,
    ASTERCatalog,
    VIIRSCatalog,
    load_credentials,
)

# Generic / GDAL fallback
from grdl.IO.generic import GDALFallbackReader, open_any

# Invasive probe reader
from grdl.IO.probe import InvasiveProbeReader

# EO submodule
from grdl.IO.eo import open_eo, EONITFReader

# IR submodule
from grdl.IO.ir import ASTERReader, open_ir

# Multispectral submodule
from grdl.IO.multispectral import VIIRSReader, open_multispectral

# GMTI submodule (STANAG 4607)
from grdl.IO.gmti import (
    STANAG4607Reader,
    STANAG4607Writer,
    open_gmti,
    dwell_footprint_polygon,
    ground_relative_velocity,
    filter_target_reports,
    summarize as gmti_summarize,
)
from grdl.IO.models.stanag4607 import STANAG4607Metadata


def _normalize_reader_key(format_key: str) -> str:
    """Normalize reader registry key.

    Reader keys are case-insensitive and canonicalized to hyphen form so
    callers may pass either ``sentinel1-slc`` or ``sentinel1_slc``.
    """
    return format_key.strip().lower().replace('_', '-')


# Reader registry: maps format keys → (module_path, class_name).
# All imports are lazy — optional dependencies are only loaded when
# the corresponding reader is actually requested.
_READER_REGISTRY: Dict[str, tuple] = {
    # Base raster formats
    'geotiff':        ('grdl.IO.geotiff',                'GeoTIFFReader'),
    'nitf':           ('grdl.IO.nitf',                   'NITFReader'),
    'hdf5':           ('grdl.IO.hdf5',                   'HDF5Reader'),
    'jpeg2000':       ('grdl.IO.jpeg2000',               'JP2Reader'),
    # NGA SAR formats (sarkit primary, sarpy fallback)
    'sicd':           ('grdl.IO.sar.sicd',               'SICDReader'),
    'cphd':           ('grdl.IO.sar.cphd',               'CPHDReader'),
    'crsd':           ('grdl.IO.sar.crsd',               'CRSDReader'),
    'sidd':           ('grdl.IO.sar.sidd',               'SIDDReader'),
    # Sensor-specific SAR readers
    'biomass':        ('grdl.IO.sar.biomass',            'BIOMASSL1Reader'),
    'sentinel1-slc':  ('grdl.IO.sar.sentinel1_slc',      'Sentinel1SLCReader'),
    'sentinel1-l0':   ('grdl.IO.sar.sentinel1_l0',       'Sentinel1L0Reader'),
    'terrasar':       ('grdl.IO.sar.terrasar',           'TerraSARReader'),
    'nisar':          ('grdl.IO.sar.nisar',              'NISARReader'),
    # EO readers
    'sentinel2':      ('grdl.IO.eo.sentinel2',           'Sentinel2Reader'),
    'eo-nitf':        ('grdl.IO.eo.nitf',                'EONITFReader'),
    # IR / thermal readers
    'aster':          ('grdl.IO.ir.aster',               'ASTERReader'),
    # Multispectral readers
    'viirs':          ('grdl.IO.multispectral.viirs',    'VIIRSReader'),
    # GMTI
    'stanag4607':     ('grdl.IO.gmti.stanag4607',        'STANAG4607Reader'),
    # Generic fallbacks (explicit use only)
    'gdal':           ('grdl.IO.generic',                'GDALFallbackReader'),
    'probe':          ('grdl.IO.probe',                  'InvasiveProbeReader'),
}


def register_reader(
    format: str,
    module_path: str,
    class_name: str,
    overwrite: bool = False,
) -> None:
    """Register a reader class in the reader factory registry.

    Parameters
    ----------
    format : str
        Reader format key. Case-insensitive; underscores normalize to hyphens.
    module_path : str
        Import path of the module containing the reader class.
    class_name : str
        Class name of the reader implementation.
    overwrite : bool, default=False
        If False, raising ValueError when the key already exists.

    Raises
    ------
    ValueError
        If *format* already exists and *overwrite* is False.
    """
    key = _normalize_reader_key(format)
    if not overwrite and key in _READER_REGISTRY:
        raise ValueError(
            f"Reader format {format!r} is already registered. "
            f"Use overwrite=True to replace it."
        )
    _READER_REGISTRY[key] = (module_path, class_name)


def get_reader(
    format: str,
    filepath: Union[str, Path],
) -> ImageReader:
    """Create an ImageReader for the given format.

    Registry-based factory that maps format strings to concrete reader
    classes via lazy imports.  Counterpart to :func:`get_writer`.

    Parameters
    ----------
    format : str
        Reader format key.  Case-insensitive.  Examples: ``'sicd'``,
        ``'geotiff'``, ``'sentinel1-slc'``, ``'nisar'``.
        Call :func:`list_reader_formats` for all supported keys.
    filepath : str or Path
        Path to the image file or directory.

    Returns
    -------
    ImageReader
        Concrete reader instance for the requested format.

    Raises
    ------
    ValueError
        If *format* is not a recognized registry key.
    ImportError
        If the optional dependency for that reader is not installed.
        The exception message names the missing package and refers to
        ``requirements-optional.txt``.

    Examples
    --------
    >>> from grdl.IO import get_reader
    >>> with get_reader('sicd', 'image.nitf') as reader:
    ...     chip = reader.read_chip(0, 512, 0, 512)

    >>> reader = get_reader('geotiff', 'scene.tif')
    >>> chip = reader.read_chip(0, 1000, 0, 1000)
    >>> reader.close()
    """
    key = _normalize_reader_key(format)
    if key not in _READER_REGISTRY:
        raise ValueError(
            f"Unknown reader format: {format!r}. "
            f"Supported formats: {sorted(_READER_REGISTRY.keys())}. "
            "Use open_reader(filepath) for automatic format detection."
        )
    module_path, class_name = _READER_REGISTRY[key]
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Reader '{format}' requires optional dependencies that are "
            f"not installed: {exc}. "
            "See requirements-optional.txt for install instructions."
        ) from exc
    reader_cls = getattr(module, class_name)
    return reader_cls(filepath)


def list_reader_formats() -> List[str]:
    """Return all registered reader format keys.

    Returns
    -------
    List[str]
        Sorted list of format keys accepted by :func:`get_reader`.
    """
    return sorted(_READER_REGISTRY.keys())


# Writer registry: maps format strings to (module_path, class_name)
_WRITER_REGISTRY: Dict[str, tuple] = {
    'geotiff': ('grdl.IO.geotiff', 'GeoTIFFWriter'),
    'numpy': ('grdl.IO.numpy_io', 'NumpyWriter'),
    'png': ('grdl.IO.png', 'PngWriter'),
    'hdf5': ('grdl.IO.hdf5', 'HDF5Writer'),
    'nitf': ('grdl.IO.nitf', 'NITFWriter'),
    # Note: STANAG4607Writer is not registered here — it operates on a
    # typed metadata container, not an ndarray, so it does not fit the
    # ``get_writer(path, metadata=ImageMetadata).write(data)`` contract.
    # Use ``grdl.IO.STANAG4607Writer`` directly instead.
}


def register_writer(
    format: str,
    module_path: str,
    class_name: str,
    overwrite: bool = False,
) -> None:
    """Register a writer class in the writer factory registry.

    Parameters
    ----------
    format : str
        Writer format key. Case-insensitive.
    module_path : str
        Import path of the module containing the writer class.
    class_name : str
        Class name of the writer implementation.
    overwrite : bool, default=False
        If False, raising ValueError when the key already exists.

    Raises
    ------
    ValueError
        If *format* already exists and *overwrite* is False.
    """
    key = format.strip().lower()
    if not overwrite and key in _WRITER_REGISTRY:
        raise ValueError(
            f"Writer format {format!r} is already registered. "
            f"Use overwrite=True to replace it."
        )
    _WRITER_REGISTRY[key] = (module_path, class_name)


def get_writer(
    format: str,
    filepath: Union[str, Path],
    metadata: Optional[ImageMetadata] = None,
) -> ImageWriter:
    """Create an ImageWriter for the given format.

    Factory function that maps format strings to concrete writer
    classes.  Uses lazy imports so optional dependencies are only
    required when the corresponding writer is requested.

    Parameters
    ----------
    format : str
        Output format.  One of ``'geotiff'``, ``'numpy'``, ``'png'``,
        ``'hdf5'``.
    filepath : str or Path
        Output file path.
    metadata : ImageMetadata, optional
        Typed metadata passed to the writer constructor.

    Returns
    -------
    ImageWriter
        Concrete writer instance for the requested format.

    Raises
    ------
    ValueError
        If *format* is not a recognized format string.
    """
    key = format.strip().lower()
    if key not in _WRITER_REGISTRY:
        raise ValueError(
            f"Unknown writer format: {format!r}. "
            f"Supported formats: {sorted(_WRITER_REGISTRY.keys())}"
        )
    module_path, class_name = _WRITER_REGISTRY[key]
    module = importlib.import_module(module_path)
    writer_cls = getattr(module, class_name)
    return writer_cls(filepath, metadata=metadata)


# Extension-to-format mapping for writer auto-detection
_EXTENSION_MAP: Dict[str, str] = {
    '.tif': 'geotiff',
    '.tiff': 'geotiff',
    '.geotiff': 'geotiff',
    '.npy': 'numpy',
    '.png': 'png',
    '.h5': 'hdf5',
    '.hdf5': 'hdf5',
    '.he5': 'hdf5',
    '.nitf': 'nitf',
    '.ntf': 'nitf',
}


# Reader extension-to-format mapping for open_reader() auto-detection.
# Maps lowercase file extension → _READER_REGISTRY key.
_READER_EXTENSION_MAP: Dict[str, str] = {
    '.tif': 'geotiff',
    '.tiff': 'geotiff',
    '.geotiff': 'geotiff',
    '.h5': 'hdf5',
    '.hdf5': 'hdf5',
    '.he5': 'hdf5',
    '.hdf': 'hdf5',
    '.nitf': 'nitf',
    '.ntf': 'nitf',
    '.nsf': 'nitf',
    '.jp2': 'jpeg2000',
    '.j2k': 'jpeg2000',
    '.j2c': 'jpeg2000',
    '.jpx': 'jpeg2000',
}


def write(
    data: np.ndarray,
    path: Union[str, Path],
    metadata: Optional[ImageMetadata] = None,
    format: Optional[str] = None,
    geolocation: Optional[Dict[str, Any]] = None,
) -> None:
    """Write array data to a file, auto-detecting format from extension."""
    path = Path(path)
    if format is None:
        ext = path.suffix.lower()
        if ext not in _EXTENSION_MAP:
            raise ValueError(
                f"Cannot determine writer format from extension '{ext}'. "
                f"Supported extensions: {sorted(_EXTENSION_MAP.keys())}. "
                "Provide an explicit format= argument."
            )
        format = _EXTENSION_MAP[ext]

    with get_writer(format, path, metadata=metadata) as writer:
        writer.write(data, geolocation=geolocation)


def open_reader(filepath: Union[str, Path]) -> ImageReader:
    """Open any supported image file, auto-detecting format."""
    filepath = Path(filepath)
    ext = filepath.suffix.lower()
    _missing_lib_msg: Optional[str] = None

    # 1. Extension-mapped fast path through the reader registry
    if ext in _READER_EXTENSION_MAP:
        fmt = _READER_EXTENSION_MAP[ext]
        try:
            return get_reader(fmt, filepath)
        except ImportError as exc:
            # Specialized library missing — record and fall through.
            _missing_lib_msg = str(exc)
        except ValueError:
            # Format mismatch from a specific reader (try generic cascade).
            pass

    # 2. Delegate to open_any for NITF sniffing, modality cascades,
    #    GDAL fallback, and invasive probing.
    try:
        reader = open_any(filepath)
    except ValueError:
        if _missing_lib_msg:
            raise ValueError(
                f"Could not open '{filepath}': {_missing_lib_msg}. "
                "Install the missing package and retry, or check "
                "requirements-optional.txt."
            )
        raise ValueError(
            f"Could not open '{filepath}'. No reader (specialized, GDAL, "
            "or invasive probing) could handle this file. Check that it is "
            "a supported format and all required libraries are installed."
        )

    # Surface missing-library message when fallback path succeeded.
    if _missing_lib_msg:
        warnings.warn(
            f"Opened '{filepath.name}' via fallback reader "
            f"({type(reader).__name__}) because a specialized reader "
            f"dependency is not installed: {_missing_lib_msg}. "
            "Install the missing package for richer metadata and stricter "
            "format validation.",
            UserWarning,
            stacklevel=2,
        )

    return reader


def open_image(filepath: Union[str, Path]) -> ImageReader:
    """Open any supported raster image file.

    .. deprecated::
        Use :func:`open_reader` instead. ``open_image`` is a compatibility
        alias and will be removed in a future release.
    """
    warnings.warn(
        "open_image() is deprecated and will be removed in a future release. "
        "Use open_reader() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return open_reader(filepath)


__all__ = [
    # Base classes and models
    'ImageReader',
    'ImageWriter',
    'CatalogInterface',
    'ImageMetadata',
    'SICDMetadata',
    'SIDDMetadata',
    'BIOMASSMetadata',
    'VIIRSMetadata',
    'ASTERMetadata',
    # Base format readers
    'GeoTIFFReader',
    'HDF5Reader',
    'JP2Reader',
    'NITFReader',
    # SAR readers
    'SICDReader',
    'CPHDReader',
    'CRSDReader',
    'SIDDReader',
    'Sentinel1SLCReader',
    'SICDCollectionReader',
    # BIOMASS
    'BIOMASSL1Reader',
    'BIOMASSCatalog',
    # Catalogs
    'Sentinel1SLCCatalog',
    'TerraSARCatalog',
    'NISARCatalog',
    'Sentinel2Catalog',
    'ASTERCatalog',
    'VIIRSCatalog',
    # TerraSAR-X / TanDEM-X
    'TerraSARMetadata',
    'TerraSARReader',
    'open_terrasar',
    # NISAR
    'NISARReader',
    'NISARMetadata',
    'open_nisar',
    # EO readers
    'EONITFReader',
    # IR readers
    'ASTERReader',
    # Multispectral readers
    'VIIRSReader',
    # GMTI (STANAG 4607)
    'STANAG4607Reader',
    'STANAG4607Writer',
    'STANAG4607Metadata',
    'open_gmti',
    'dwell_footprint_polygon',
    'ground_relative_velocity',
    'filter_target_reports',
    'gmti_summarize',
    # Generic / GDAL fallback / invasive probe
    'GDALFallbackReader',
    'InvasiveProbeReader',
    'open_any',
    # Writers
    'GeoTIFFWriter',
    'HDF5Writer',
    'NITFWriter',
    'NumpyWriter',
    'PngWriter',
    'SICDWriter',
    'SIDDWriter',
    # Reader factory
    'register_reader',
    'get_reader',
    'list_reader_formats',
    # Writer factory and convenience
    'register_writer',
    'get_writer',
    'write',
    # Convenience functions
    'open_reader',
    'open_image',  # deprecated alias for open_reader
    'open_sar',
    'open_sicd_collection',
    'open_biomass',
    'open_eo',
    'open_ir',
    'open_multispectral',
    'load_credentials',
]
