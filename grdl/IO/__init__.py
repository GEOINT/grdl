# -*- coding: utf-8 -*-
"""
IO Module - Input/Output Operations for Geospatial Imagery.

Handles reading and writing various geospatial data formats. Base data
format readers (GeoTIFF, NITF) live at this level.  Modality-specific
readers are organized into submodules (``sar/``).

Dependencies
------------
rasterio
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
2026-02-09
"""

# Standard library
from pathlib import Path
from typing import Union

# Base classes
from grdl.IO.base import ImageReader, ImageWriter, CatalogInterface

# Base format readers (IO level)
from grdl.IO.geotiff import GeoTIFFReader
from grdl.IO.nitf import NITFReader

# SAR submodule
from grdl.IO.sar import (
    SICDReader,
    CPHDReader,
    CRSDReader,
    SIDDReader,
    BIOMASSL1Reader,
    BIOMASSCatalog,
    open_sar,
    open_biomass,
    load_credentials,
)


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

    # Try GeoTIFF as fallback for unknown extensions
    try:
        return GeoTIFFReader(filepath)
    except (ValueError, ImportError):
        pass

    raise ValueError(
        f"Could not open {filepath}. "
        "Ensure file is a valid GeoTIFF or NITF and rasterio is installed. "
        "For SAR-specific formats (SICD, CPHD, CRSD), use open_sar()."
    )


__all__ = [
    # Base classes
    'ImageReader',
    'ImageWriter',
    'CatalogInterface',
    # Base format readers
    'GeoTIFFReader',
    'NITFReader',
    # SAR readers
    'SICDReader',
    'CPHDReader',
    'CRSDReader',
    'SIDDReader',
    # BIOMASS
    'BIOMASSL1Reader',
    'BIOMASSCatalog',
    # Convenience functions
    'open_image',
    'open_sar',
    'open_biomass',
    'load_credentials',
]
