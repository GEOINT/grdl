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
Create a geolocation object from an imagery reader. The ``pixel_to_latlon``
and ``latlon_to_pixel`` methods accept scalar, list, or ndarray inputs and
return matching types (scalar in → scalar out, array in → array out):

    >>> from grdl.IO import open_biomass
    >>> from grdl.geolocation.sar.gcp import GCPGeolocation
    >>> import numpy as np
    >>>
    >>> with open_biomass('path/to/biomass_product') as reader:
    >>>     geo = GCPGeolocation(
    ...         reader.metadata['gcps'],
    ...         (reader.metadata['rows'], reader.metadata['cols']),
    ...     )
    >>>
    >>>     # Single pixel (returns scalars)
    >>>     lat, lon, height = geo.pixel_to_latlon(1000, 500)
    >>>
    >>>     # Array of pixels (returns arrays, vectorized)
    >>>     lats, lons, heights = geo.pixel_to_latlon(
    ...         np.array([100, 200, 300]), np.array([400, 500, 600])
    ...     )

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