# -*- coding: utf-8 -*-
"""
Multispectral Readers - Multispectral and hyperspectral imagery readers.

Provides readers for multispectral and hyperspectral satellite products.
Currently supports VIIRS products (VNP46A1 nighttime lights, VNP13A1
vegetation indices, surface reflectance) in HDF5 format.

Planned sensors: MODIS, ASTER full multispectral, EMIT, PRISMA, EnMAP,
DESIS, HISUI.

Dependencies
------------
h5py

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
from pathlib import Path
from typing import Union

# Multispectral readers
from grdl.IO.multispectral.viirs import VIIRSReader

# Metadata models
from grdl.IO.models import VIIRSMetadata

# Base class (for return type)
from grdl.IO.base import ImageReader

# Backend detection flags
from grdl.IO.multispectral._backend import _HAS_H5PY, _HAS_XARRAY, _HAS_SPECTRAL


def open_multispectral(filepath: Union[str, Path]) -> ImageReader:
    """Auto-detect multispectral format and return appropriate reader.

    Tries VIIRS detection first (HDF5), then falls back to generic
    HDF5Reader.

    Parameters
    ----------
    filepath : str or Path
        Path to multispectral image file.

    Returns
    -------
    ImageReader
        Appropriate multispectral reader instance.

    Raises
    ------
    ValueError
        If format cannot be determined or is unsupported.

    Examples
    --------
    >>> reader = open_multispectral('VNP46A1.h5')
    >>> print(reader.metadata.product_short_name)
    >>> reader.close()
    """
    filepath = Path(filepath)
    name_upper = filepath.name.upper()

    # Try VIIRS (HDF5)
    if filepath.suffix.lower() in ('.h5', '.he5', '.hdf5'):
        if any(tag in name_upper for tag in ('VNP', 'VJ1', 'VJ2', 'VIIRS')):
            try:
                return VIIRSReader(filepath)
            except (ValueError, ImportError):
                pass

    # HDF5 fallback — try VIIRSReader for any HDF5 file
    if filepath.suffix.lower() in ('.h5', '.he5', '.hdf5'):
        try:
            return VIIRSReader(filepath)
        except (ValueError, ImportError):
            pass

    raise ValueError(
        f"Could not determine multispectral format for {filepath}. "
        "Ensure file is a valid HDF5 file and h5py is installed."
    )


__all__ = [
    # Readers
    'VIIRSReader',
    # Metadata
    'VIIRSMetadata',
    # Convenience
    'open_multispectral',
]
