# -*- coding: utf-8 -*-
"""
EO Readers - Electro-Optical imagery readers.

Provides readers for visible, panchromatic, and VIS/NIR satellite
imagery. Planned sensors include Landsat OLI, Sentinel-2, HLS,
WorldView, PlanetScope, Pleiades/SPOT, and NAIP.

Dependencies
------------
rasterio
glymur (for JPEG2000 formats)

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

# Base class (for return type)
from grdl.IO.base import ImageReader

# Backend detection flags
from grdl.IO.eo._backend import _HAS_RASTERIO, _HAS_GLYMUR


def open_eo(filepath: Union[str, Path]) -> ImageReader:
    """Auto-detect EO format and return appropriate reader.

    Currently falls back to GeoTIFFReader for all EO products.
    Sensor-specific readers will be added as they are implemented.

    Parameters
    ----------
    filepath : str or Path
        Path to EO image file.

    Returns
    -------
    ImageReader
        Appropriate EO reader instance.

    Raises
    ------
    ValueError
        If format cannot be determined or is unsupported.

    Examples
    --------
    >>> reader = open_eo('LC08_B4.TIF')
    >>> chip = reader.read_chip(0, 1000, 0, 1000)
    >>> reader.close()
    """
    filepath = Path(filepath)

    # GeoTIFF fallback (most EO products)
    if filepath.suffix.lower() in ('.tif', '.tiff', '.geotiff'):
        try:
            from grdl.IO.geotiff import GeoTIFFReader
            return GeoTIFFReader(filepath)
        except (ValueError, ImportError) as e:
            raise ValueError(
                f"Failed to open EO file {filepath}: {e}"
            ) from e

    # JPEG2000 fallback (Sentinel-2, Pleiades)
    if filepath.suffix.lower() in ('.jp2', '.j2k'):
        try:
            from grdl.IO.jpeg2000 import JP2Reader
            return JP2Reader(filepath)
        except (ValueError, ImportError) as e:
            raise ValueError(
                f"Failed to open EO file {filepath}: {e}"
            ) from e

    raise ValueError(
        f"Could not determine EO format for {filepath}. "
        "Ensure file is a valid GeoTIFF or JPEG2000 and the required "
        "library (rasterio, glymur) is installed."
    )


__all__ = [
    'open_eo',
]
