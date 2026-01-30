# -*- coding: utf-8 -*-
"""
Geolocation Module - Coordinate transformations for geospatial imagery.

Provides interfaces and implementations for transforming between image pixel
coordinates and geographic coordinates (latitude/longitude).

Key Classes
-----------
- Geolocation: Abstract base class for coordinate transformations
- NoGeolocation: Fallback for imagery without geolocation

Usage
-----
Create a geolocation object from an imagery reader:

    >>> from grdl.IO import open_biomass
    >>> from grdl.geolocation import Geolocation
    >>>
    >>> with open_biomass('path/to/biomass_product') as reader:
    >>>     geo = Geolocation.from_reader(reader)
    >>>     lat, lon, height = geo.pixel_to_latlon(1000, 500)
    >>>     print(f"Pixel (1000, 500) -> ({lat:.6f}, {lon:.6f})")

Modules
-------
- base: Abstract base classes
- sar: SAR-specific geolocation (slant range, SICD, GRD)
- eo: EO-specific geolocation (geocoded rasters)
- utils: Utility functions

Dependencies
------------
scipy

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
2026-01-30
"""

from grdl.geolocation.base import Geolocation, NoGeolocation

__all__ = [
    'Geolocation',
    'NoGeolocation',
]

__version__ = '0.1.0'