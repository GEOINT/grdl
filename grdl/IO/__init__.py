# -*- coding: utf-8 -*-
"""
IO Module - Input/Output Operations for Geospatial Imagery

Handles reading and writing various geospatial data formats including SAR,
EO imagery, and geospatial vector data.

Dependencies:
    Core: numpy
    Optional: rasterio, gdal, shapely

Author: Duane Smalley
License: MIT
Created: 2026-01-30
Modified: 2026-01-30
"""

# Import SAR readers
from grdl.IO.sar import (
    SICDReader,
    CPHDReader,
    GRDReader,
    open_sar,
)

# Public API
__all__ = [
    # SAR readers
    'SICDReader',
    'CPHDReader',
    'GRDReader',
    'open_sar',
]
