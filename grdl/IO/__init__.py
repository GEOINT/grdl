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
duane.d.smalley@gmail.com

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
2026-02-19
"""

# Standard library
import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Third-party
import numpy as np

# Base classes and models
from grdl.IO.base import ImageReader, ImageWriter, CatalogInterface
from grdl.IO.models import (
    ImageMetadata, SICDMetadata, SIDDMetadata, BIOMASSMetadata,
    VIIRSMetadata, ASTERMetadata, TerraSARMetadata,
)

# Base format readers (IO level)
from grdl.IO.geotiff import GeoTIFFReader
from grdl.IO.hdf5 import HDF5Reader
from grdl.IO.jpeg2000 import JP2Reader
from grdl.IO.nitf import NITFReader

# SAR submodule
from grdl.IO.sar import (
    SICDReader,
    CPHDReader,
    CRSDReader,
    SIDDReader,
    BIOMASSL1Reader,
    BIOMASSCatalog,
    TerraSARReader,
    open_sar,
    open_biomass,
    open_terrasar,
    load_credentials,
)

# EO submodule
from grdl.IO.eo import open_eo

# IR submodule
from grdl.IO.ir import ASTERReader, open_ir

# Multispectral submodule
from grdl.IO.multispectral import VIIRSReader, open_multispectral


# Writer registry: maps format strings to (module_path, class_name)
_WRITER_REGISTRY: Dict[str, tuple] = {
    'geotiff': ('grdl.IO.geotiff', 'GeoTIFFWriter'),
    'numpy': ('grdl.IO.numpy_io', 'NumpyWriter'),
    'png': ('grdl.IO.png', 'PngWriter'),
    'hdf5': ('grdl.IO.hdf5', 'HDF5Writer'),
    'nitf': ('grdl.IO.nitf', 'NITFWriter'),
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


def open_image(filepath: Union[str, Path]) -> ImageReader:
    """Open any supported raster image file.

    Tries GeoTIFF first, then NITF.  For SAR-specific formats
    (SICD, CPHD, CRSD, SIDD), use ``open_sar()`` instead.

    Parameters
    ----------
    filepath : str or Path
        Path to raster image file.

    Returns
    -------
    ImageReader
        Appropriate reader instance.

    Raises
    ------
    ValueError
        If format cannot be determined or is unsupported.

    Examples
    --------
    >>> from grdl.IO import open_image
    >>> reader = open_image('scene.tif')
    >>> chip = reader.read_chip(0, 512, 0, 512)
    >>> reader.close()
    """
    filepath = Path(filepath)

    # Try GeoTIFF first (most common)
    if filepath.suffix.lower() in ('.tif', '.tiff', '.geotiff'):
        try:
            return GeoTIFFReader(filepath)
        except (ValueError, ImportError):
            pass

    # Try NITF
    if filepath.suffix.lower() in ('.nitf', '.ntf', '.nsf'):
        try:
            return NITFReader(filepath)
        except (ValueError, ImportError):
            pass

    # Try HDF5
    if filepath.suffix.lower() in ('.h5', '.he5', '.hdf5', '.hdf'):
        try:
            return HDF5Reader(filepath)
        except (ValueError, ImportError):
            pass

    # Try JPEG2000
    if filepath.suffix.lower() in ('.jp2', '.j2k', '.j2c', '.jpx'):
        try:
            return JP2Reader(filepath)
        except (ValueError, ImportError):
            pass

    # Try GeoTIFF as fallback for unknown extensions
    try:
        return GeoTIFFReader(filepath)
    except (ValueError, ImportError):
        pass

    raise ValueError(
        f"Could not open {filepath}. "
        "Ensure file is a valid GeoTIFF, NITF, HDF5, or JPEG2000 and the required "
        "library (rasterio, h5py, glymur) is installed. "
        "For SAR-specific formats (SICD, CPHD, CRSD), use open_sar()."
    )


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
    # BIOMASS
    'BIOMASSL1Reader',
    'BIOMASSCatalog',
    # TerraSAR-X / TanDEM-X
    'TerraSARMetadata',
    'TerraSARReader',
    'open_terrasar',
    # IR readers
    'ASTERReader',
    # Multispectral readers
    'VIIRSReader',
    # Writer factory and convenience
    'get_writer',
    'write',
    # Convenience functions
    'open_image',
    'open_sar',
    'open_biomass',
    'open_eo',
    'open_ir',
    'open_multispectral',
    'load_credentials',
]
