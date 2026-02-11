# -*- coding: utf-8 -*-
"""
IR Readers - Thermal and infrared imagery readers.

Provides readers for thermal/infrared satellite products. Currently
supports ASTER L1T thermal bands and ASTGTM elevation models.

Planned sensors: ECOSTRESS, Landsat TIRS, VIIRS thermal bands,
GOES thermal.

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
from pathlib import Path
from typing import Union

# IR readers
from grdl.IO.ir.aster import ASTERReader

# Metadata models
from grdl.IO.models import ASTERMetadata

# Base class (for return type)
from grdl.IO.base import ImageReader

# Backend detection flags
from grdl.IO.ir._backend import _HAS_RASTERIO, _HAS_H5PY


def open_ir(filepath: Union[str, Path]) -> ImageReader:
    """Auto-detect IR/thermal format and return appropriate reader.

    Tries ASTER detection first, then falls back to GeoTIFFReader.

    Parameters
    ----------
    filepath : str or Path
        Path to IR/thermal image file.

    Returns
    -------
    ImageReader
        Appropriate IR reader instance.

    Raises
    ------
    ValueError
        If format cannot be determined or is unsupported.

    Examples
    --------
    >>> reader = open_ir('AST_L1T_00305042006.tif')
    >>> print(reader.metadata.processing_level)
    >>> reader.close()
    """
    filepath = Path(filepath)
    name_upper = filepath.name.upper()

    # Try ASTER (L1T or GDEM)
    if any(tag in name_upper for tag in ('AST_L1T', 'ASTGTM', 'ASTER', 'AST_')):
        try:
            return ASTERReader(filepath)
        except (ValueError, ImportError):
            pass

    # GeoTIFF fallback for unknown IR products
    if filepath.suffix.lower() in ('.tif', '.tiff'):
        try:
            return ASTERReader(filepath)
        except (ValueError, ImportError):
            pass

    raise ValueError(
        f"Could not determine IR format for {filepath}. "
        "Ensure file is a valid ASTER GeoTIFF and rasterio is installed."
    )


__all__ = [
    # Readers
    'ASTERReader',
    # Metadata
    'ASTERMetadata',
    # Convenience
    'open_ir',
]
