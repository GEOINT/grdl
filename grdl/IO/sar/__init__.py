# -*- coding: utf-8 -*-
"""
SAR Readers - Synthetic Aperture Radar format readers.

Provides readers for NGA SAR standards (SICD, CPHD, CRSD, SIDD), ESA
BIOMASS products, and a convenience ``open_sar()`` auto-detection
function.  Uses sarkit as the primary backend for NGA formats with
sarpy as fallback for SICD and CPHD.

Dependencies
------------
sarkit (primary) or sarpy (fallback)
rasterio (for BIOMASS and GeoTIFF fallback)

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
2026-02-09
"""

# Standard library
from pathlib import Path
from typing import Union

# SAR readers and writers
from grdl.IO.sar.sicd import SICDReader
from grdl.IO.sar.sicd_writer import SICDWriter
from grdl.IO.sar.cphd import CPHDReader
from grdl.IO.sar.crsd import CRSDReader
from grdl.IO.sar.sidd import SIDDReader

# BIOMASS
from grdl.IO.sar.biomass import BIOMASSL1Reader, open_biomass
from grdl.IO.sar.biomass_catalog import BIOMASSCatalog, load_credentials

# Metadata models
from grdl.IO.models import SICDMetadata, SIDDMetadata, BIOMASSMetadata

# Base class (for return type)
from grdl.IO.base import ImageReader

# Backend detection flags
from grdl.IO.sar._backend import _HAS_SARKIT, _HAS_SARPY


def open_sar(filepath: Union[str, Path]) -> ImageReader:
    """Auto-detect SAR format and return appropriate reader.

    Tries NGA formats (SICD, CPHD, CRSD, SIDD) first, then falls
    back to GeoTIFFReader for SAR GRD products.

    Parameters
    ----------
    filepath : str or Path
        Path to SAR file.

    Returns
    -------
    ImageReader
        Appropriate SAR reader instance.

    Raises
    ------
    ValueError
        If format cannot be determined or is unsupported.

    Examples
    --------
    >>> reader = open_sar('image.nitf')
    >>> print(f"Format: {reader.metadata['format']}")
    >>> chip = reader.read_chip(0, 1000, 0, 1000)
    >>> reader.close()
    """
    filepath = Path(filepath)

    # Try SICD (NITF container with complex SAR)
    if _HAS_SARKIT or _HAS_SARPY:
        try:
            return SICDReader(filepath)
        except (ValueError, ImportError, Exception):
            pass

        # Try CPHD
        try:
            return CPHDReader(filepath)
        except (ValueError, ImportError, Exception):
            pass

    # Try CRSD (sarkit-only)
    if _HAS_SARKIT:
        try:
            return CRSDReader(filepath)
        except (ValueError, ImportError, Exception):
            pass

        # Try SIDD
        try:
            return SIDDReader(filepath)
        except (ValueError, ImportError, Exception):
            pass

    # Try GeoTIFF fallback (SAR GRD products)
    if filepath.suffix.lower() in ('.tif', '.tiff'):
        try:
            from grdl.IO.geotiff import GeoTIFFReader
            return GeoTIFFReader(filepath)
        except (ValueError, ImportError, Exception):
            pass

    raise ValueError(
        f"Could not determine SAR format for {filepath}. "
        "Ensure file is valid SICD, CPHD, CRSD, SIDD, or GeoTIFF "
        "format and required libraries (sarkit, sarpy, rasterio) "
        "are installed."
    )


__all__ = [
    # NGA SAR formats
    'SICDReader',
    'SICDWriter',
    'CPHDReader',
    'CRSDReader',
    'SIDDReader',
    # BIOMASS
    'BIOMASSL1Reader',
    'BIOMASSCatalog',
    # Metadata models
    'SICDMetadata',
    'SIDDMetadata',
    'BIOMASSMetadata',
    # Convenience functions
    'open_sar',
    'open_biomass',
    'load_credentials',
]
