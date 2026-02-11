# -*- coding: utf-8 -*-
"""
Elevation Backend Detection - Detect available DEM/elevation libraries.

Probes for rasterio at import time. Provides a boolean flag and a helper
function that elevation model implementations use to verify required
packages are installed before constructing elevation objects.

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
2026-02-11

Modified
--------
2026-02-11
"""

_HAS_RASTERIO = False

try:
    import rasterio  # noqa: F401
    _HAS_RASTERIO = True
except ImportError:
    pass


def require_elevation_backend() -> None:
    """Verify that rasterio is installed for DEM/DTED elevation lookup.

    Raises
    ------
    ImportError
        If rasterio is not installed. The message includes installation
        instructions.
    """
    if not _HAS_RASTERIO:
        raise ImportError(
            "DEM/DTED elevation lookup requires rasterio. "
            "Install with: pip install rasterio"
        )
