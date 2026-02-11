# -*- coding: utf-8 -*-
"""
EO Geolocation Backend Detection - Detect available EO geolocation libraries.

Probes for rasterio (specifically ``rasterio.transform.Affine``) and pyproj
at import time. Provides boolean flags and a helper function that EO
geolocation implementations use to verify required packages are installed
before constructing geolocation objects.

Dependencies
------------
rasterio
pyproj

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
2026-02-11

Modified
--------
2026-02-11
"""

# Standard library
from typing import List

_HAS_RASTERIO = False
_HAS_PYPROJ = False

try:
    from rasterio.transform import Affine  # noqa: F401
    _HAS_RASTERIO = True
except ImportError:
    pass

try:
    import pyproj  # noqa: F401
    _HAS_PYPROJ = True
except ImportError:
    pass


def require_affine_backend() -> None:
    """Verify that all packages required for affine geolocation are installed.

    Checks for both rasterio (provides ``Affine`` transform) and pyproj
    (provides CRS-aware coordinate transformations). Raises a single
    ``ImportError`` listing all missing packages so users can install
    everything in one step.

    Raises
    ------
    ImportError
        If rasterio or pyproj (or both) are not installed. The message
        lists all missing packages with installation instructions.
    """
    missing: List[str] = []
    if not _HAS_RASTERIO:
        missing.append('rasterio')
    if not _HAS_PYPROJ:
        missing.append('pyproj')

    if missing:
        packages = ' '.join(missing)
        raise ImportError(
            f"AffineGeolocation requires {', '.join(missing)}. "
            f"Install with: pip install {packages}"
        )
