# -*- coding: utf-8 -*-
"""
Elevation Module - DEM/DTED terrain elevation lookup and geoid correction.

Provides abstract and concrete elevation models for querying terrain
heights from various DEM data sources. Supports DTED tiles, GeoTIFF
DEMs, and constant-height fallbacks. Optional geoid correction converts
MSL heights to HAE using EGM96 undulation grids.

Key Classes
-----------
- ElevationModel: Abstract base class for all elevation models
- ConstantElevation: Returns a fixed height (default fallback)
- DTEDElevation: Reads DTED Level 0/1/2 tiles via rasterio
- GeoTIFFDEM: Reads a single GeoTIFF DEM file via rasterio
- GeoidCorrection: EGM96 geoid undulation lookup

Usage
-----
    >>> from grdl.geolocation.elevation import ConstantElevation
    >>> elev = ConstantElevation(height=100.0)
    >>> elev.get_elevation(34.05, -118.25)
    100.0

    >>> from grdl.geolocation.elevation import DTEDElevation
    >>> elev = DTEDElevation('/data/dted')
    >>> elev.get_elevation(34.05, -118.25)
    432.0

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

from grdl.geolocation.elevation.base import ElevationModel
from grdl.geolocation.elevation.constant import ConstantElevation

# Optional: may fail if rasterio not available
try:
    from grdl.geolocation.elevation.dted import DTEDElevation
except ImportError:
    pass

try:
    from grdl.geolocation.elevation.geotiff_dem import GeoTIFFDEM
except ImportError:
    pass

try:
    from grdl.geolocation.elevation.geoid import GeoidCorrection
except ImportError:
    pass

__all__ = [
    'ElevationModel',
    'ConstantElevation',
    'DTEDElevation',
    'GeoTIFFDEM',
    'GeoidCorrection',
]
