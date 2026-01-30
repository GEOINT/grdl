# -*- coding: utf-8 -*-
"""
IO Module - Input/Output Operations for Geospatial Imagery.

Handles reading and writing various geospatial data formats including SAR,
EO imagery, and geospatial vector data. Includes catalog search and download
via the ESA MAAP STAC API.

Dependencies
------------
rasterio
sarpy
requests

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

# Import SAR readers
from grdl.IO.sar import (
    SICDReader,
    CPHDReader,
    GRDReader,
    open_sar,
)

# Import BIOMASS readers
from grdl.IO.biomass import (
    BIOMASSL1Reader,
    open_biomass,
)

# Import catalog tools
from grdl.IO.catalog import (
    BIOMASSCatalog,
    load_credentials,
)

# Public API
__all__ = [
    # SAR readers
    'SICDReader',
    'CPHDReader',
    'GRDReader',
    'open_sar',
    # BIOMASS readers
    'BIOMASSL1Reader',
    'open_biomass',
    # Catalog tools
    'BIOMASSCatalog',
    'load_credentials',
]
