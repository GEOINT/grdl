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
2026-05-27
"""

# Standard library
import importlib
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
    'eo-nitf':        ('grdl.IO.eo.nitf',               'EONITFReader'),
    # IR / thermal readers
    'aster':          ('grdl.IO.ir.aster',              'ASTERReader'),
    # Multispectral readers
    'viirs':          ('grdl.IO.multispectral.viirs',    'VIIRSReader'),
    # GMTI
    'stanag4607':     ('grdl.IO.gmti.stanag4607',       'STANAG4607Reader'),
    # Generic fallbacks (explicit use only)
    'gdal':           ('grdl.IO.generic',               'GDALFallbackReader'),
    'probe':          ('grdl.IO.probe',                  'InvasiveProbeReader'),
}


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
    key = format.lower()
    if key not in _READER_REGISTRY:
        raise ValueError(
            f"Unknown reader format: {format!r}. "
            f"Supported formats: {sorted(_READER_REGISTRY.keys())}. "
            f"Use open_reader(filepath) for automatic format detection."
        )
    module_path, class_name = _READER_REGISTRY[key]
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Reader '{format}' requires optional dependencies that are "
            f"not installed: {exc}. "
            f"See requirements-optional.txt for install instructions."
        ) from exc
    reader_cls = getattr(module, class_name)
    return reader_cls(filepath)


def list_reader_formats() -> List[str]:
    """Return all registered reader format keys.

    Returns
    -------
    List[str]
        Sorted list of format keys accepted by :func:`get_reader`.

    Examples
    --------
    >>> from grdl.IO import list_reader_formats
    >>> list_reader_formats()
    ['aster', 'biomass', 'cphd', 'crsd', 'eo-nitf', ...]
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

    Examples
    --------
    >>> from grdl.IO import get_writer
    >>> writer = get_writer('geotiff', 'output.tif')
    """
    key = format.lower()
    if key not in _WRITER_REGISTRY:
        raise ValueError(
            f"Unknown writer format: {format!r}. "
            f"Supported formats: {sorted(_WRITER_REGISTRY.keys())}"
        )
    module_path, class_name = _WRITER_REGISTRY[key]
    module = importlib.import_module(module_path)
    writer_cls = getattr(module, class_name)
    return writer_cls(filepath, metadata=metadata)


# Extension-to-format mapping for auto-detection
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
    '.tif':    'geotiff',
    '.tiff':   'geotiff',
    '.geotiff':'geotiff',
    '.h5':     'hdf5',
    '.hdf5':   'hdf5',
    '.he5':    'hdf5',
    '.hdf':    'hdf5',
    '.nitf':   'nitf',
    '.ntf':    'nitf',
    '.nsf':    'nitf',
    '.jp2':    'jpeg2000',
    '.j2k':    'jpeg2000',
    '.j2c':    'jpeg2000',
    '.jpx':    'jpeg2000',
}


def write(
    data: np.ndarray,
    path: Union[str, Path],
    metadata: Optional[ImageMetadata] = None,
    format: Optional[str] = None,
    geolocation: Optional[Dict[str, Any]] = None,
) -> None:
    """Write array data to a file, auto-detecting format from extension.

    Convenience function that creates the appropriate writer, writes
    the data, and closes the writer.  For finer control (chip writes,
    multiple datasets), use :func:`get_writer` directly.

    Parameters
    ----------
    data : np.ndarray
        Image data to write.
    path : str or Path
        Output file path.
    metadata : ImageMetadata, optional
        Typed metadata passed to the writer.
    format : str, optional
        Output format override.  If ``None``, auto-detected from the
        file extension.
    geolocation : Dict[str, Any], optional
        Geolocation information (CRS, transform) for geospatial formats.

    Raises
    ------
    ValueError
        If *format* is ``None`` and the extension is not recognized.

    Examples
    --------
    >>> from grdl.IO import write
    >>> import numpy as np
    >>> data = np.random.rand(64, 64).astype(np.float32)
    >>> write(data, 'output.tif')
    >>> write(data, 'output.npy')
    >>> write(data, 'result.dat', format='numpy')
    """
    path = Path(path)
    if format is None:
        ext = path.suffix.lower()
        if ext not in _EXTENSION_MAP:
            raise ValueError(
                f"Cannot determine writer format from extension '{ext}'. "
                f"Supported extensions: {sorted(_EXTENSION_MAP.keys())}. "
                f"Provide an explicit format= argument."
            )
        format = _EXTENSION_MAP[ext]

    with get_writer(format, path, metadata=metadata) as writer:
        writer.write(data, geolocation=geolocation)


def open_reader(filepath: Union[str, Path]) -> ImageReader:
    """Open any supported image file, auto-detecting format.

    Extension-to-format dispatch via :data:`_READER_EXTENSION_MAP` and
    :func:`get_reader`.  Falls through to
    :func:`~grdl.IO.generic.open_any` (NITF sniffing, modality cascades,
    GDAL fallback, invasive probing) when the extension is unknown or when
    the primary reader raises :exc:`ImportError` due to a missing
    dependency.

    A :class:`UserWarning` is emitted — via :func:`~grdl.IO.generic.open_any`
    — when execution reaches the GDAL or invasive-probe fallback because a
    specialized library is not installed.  The warning message names the
    missing packages so the user can take corrective action.

    For SAR-specific auto-detection (SICD/CPHD/CRSD/SIDD/Sentinel-1 etc.),
    :func:`open_sar` is preferred.

    Parameters
    ----------
    filepath : str or Path
        Path to the image file or directory.

    Returns
    -------
    ImageReader
        Most appropriate reader for the file.

    Raises
    ------
    ValueError
        If no reader can open the file.

    Warns
    -----
    UserWarning
        When a specialized reader failed due to a missing dependency
        and GDAL or :class:`~grdl.IO.probe.InvasiveProbeReader` is used
        as fallback.

    Examples
    --------
    >>> from grdl.IO import open_reader
    >>> with open_reader('scene.tif') as reader:
    ...     chip = reader.read_chip(0, 512, 0, 512)

    >>> # Explicit format is faster and unambiguous:
    >>> reader = get_reader('sicd', 'image.nitf')
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()
    _missing_lib_msg: Optional[str] = None

    # 1. Extension-mapped fast path through the reader registry
    if ext in _READER_EXTENSION_MAP:
        fmt = _READER_EXTENSION_MAP[ext]
        try:
            return get_reader(fmt, filepath)
        except ImportError as exc:
            # Specialized library missing — record the message and fall
            # through to open_any() so the user still gets a result.
            _missing_lib_msg = str(exc)
        except (ValueError, Exception):
            pass

    # 2. Delegate to open_any for NITF sniffing, modality cascades,
    #    GDAL fallback, and invasive probing.  open_any() emits its own
    #    UserWarning when it reaches the GDAL / probe fallback tiers.
    from grdl.IO.generic import open_any
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

    # Surface the missing-library message as a warning even when
    # open_any() succeeded via a lower-tier fallback.
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
        Use :func:`open_reader` instead.  ``open_image`` is a thin
        compatibility alias and will be removed in a future release.

    Parameters
    ----------
    filepath : str or Path
        Path to raster image file.

    Returns
    -------
    ImageReader
        Appropriate reader instance.
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
    'get_reader',
    'list_reader_formats',
    # Writer factory and convenience
    'get_writer',
    'write',
    # Convenience functions
    'open_reader',
    'open_image',       # deprecated alias for open_reader
    'open_sar',
    'open_sicd_collection',
    'open_biomass',
    'open_eo',
    'open_ir',
    'open_multispectral',
    'load_credentials',
]
