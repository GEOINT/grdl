# -*- coding: utf-8 -*-
"""
EO Geolocation Module - Coordinate transformations for electro-optical imagery.

Handles geolocation for optical and multispectral imagery, typically
using affine transforms for geocoded rasters.

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
2026-02-11
"""

from grdl.geolocation.eo.affine import AffineGeolocation
from grdl.geolocation.eo.rpc import RPCGeolocation
from grdl.geolocation.eo.rsm import RSMGeolocation

__all__ = [
    'AffineGeolocation',
    'RPCGeolocation',
    'RSMGeolocation',
]