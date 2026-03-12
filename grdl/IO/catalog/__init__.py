# -*- coding: utf-8 -*-
"""
Catalog Module - Local discovery, metadata indexing, and SQLite cataloging.

Provides CatalogInterface subclasses for all GRDL-supported sensors.
Each catalog discovers sensor-specific files on disk, extracts metadata
via the corresponding Reader, and maintains a local SQLite database for
offline filtering and querying.

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
2026-03-11

Modified
--------
2026-03-12
"""

from grdl.IO.catalog.remote_utils import (
    load_credentials,
    get_cdse_token,
    get_earthdata_token,
    download_file,
)
from grdl.IO.catalog.biomass_catalog import BIOMASSCatalog
from grdl.IO.catalog.sentinel1_catalog import Sentinel1SLCCatalog
from grdl.IO.catalog.terrasar_catalog import TerraSARCatalog
from grdl.IO.catalog.nisar_catalog import NISARCatalog
from grdl.IO.catalog.sentinel2_catalog import Sentinel2Catalog
from grdl.IO.catalog.aster_catalog import ASTERCatalog
from grdl.IO.catalog.viirs_catalog import VIIRSCatalog

__all__ = [
    'load_credentials',
    'get_cdse_token',
    'get_earthdata_token',
    'download_file',
    'BIOMASSCatalog',
    'Sentinel1SLCCatalog',
    'TerraSARCatalog',
    'NISARCatalog',
    'Sentinel2Catalog',
    'ASTERCatalog',
    'VIIRSCatalog',
]
