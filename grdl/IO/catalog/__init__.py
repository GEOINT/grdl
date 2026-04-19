# -*- coding: utf-8 -*-
"""
Catalog Module - Local discovery, remote query, download, and SQLite cataloging.

Provides CatalogInterface subclasses for all GRDL-supported sensors.
Each catalog discovers sensor-specific files on disk, queries remote
archives (ESA CDSE, NASA Earthdata, ESA MAAP), downloads products with
authenticated streaming, and maintains a local SQLite database for
offline filtering and querying.

All catalogs follow the same two-step workflow: **query** to search a
remote archive and index results into the local SQLite database, then
**download** a specific product by its ID.

All query parameters are optional. Calling with no arguments returns
up to 50 of the most recent products from the remote archive.

Dependencies
------------
requests (for remote queries and downloads)

Examples
--------
**Sentinel-1 SLC** (ESA Copernicus Data Space):

>>> from grdl.IO.catalog import Sentinel1SLCCatalog
>>> cat = Sentinel1SLCCatalog('/data/sentinel1')
>>> results = cat.query_cdse(
...     start_date='2025-12-01',
...     end_date='2025-12-07',
...     bbox=(-5.0, 40.0, 0.0, 42.0),
...     max_results=10,
... )
>>> path = cat.download_product(results[0]['Id'])

**Sentinel-2** (ESA Copernicus Data Space):

>>> from grdl.IO.catalog import Sentinel2Catalog
>>> cat = Sentinel2Catalog('/data/sentinel2')
>>> results = cat.query_cdse(
...     start_date='2025-12-01',
...     end_date='2025-12-07',
...     bbox=(-5.0, 40.0, 0.0, 42.0),
...     mgrs_tile_id='T30TYN',
...     processing_level='MSIL2A',
...     max_results=10,
... )
>>> path = cat.download_product(results[0]['Id'])

**NISAR** (NASA Earthdata / ASF DAAC):

>>> from grdl.IO.catalog import NISARCatalog
>>> cat = NISARCatalog('/data/nisar')
>>> results = cat.query_earthdata(
...     start_date='2026-01-20',
...     end_date='2026-01-21',
...     bbox=(-120.0, 33.0, -117.0, 35.0),
...     product_type='RSLC',
...     max_results=5,
... )
>>> path = cat.download_product(results[0]['id'])

**ASTER** (NASA Earthdata / LP DAAC):

>>> from grdl.IO.catalog import ASTERCatalog
>>> cat = ASTERCatalog('/data/aster')
>>> results = cat.query_earthdata(
...     start_date='2026-01-01',
...     end_date='2026-02-01',
...     cloud_cover_max=20.0,
...     max_results=10,
... )
>>> path = cat.download_product(results[0]['id'])

**VIIRS** (NASA Earthdata / LAADS DAAC):

>>> from grdl.IO.catalog import VIIRSCatalog
>>> cat = VIIRSCatalog('/data/viirs')
>>> results = cat.query_earthdata(
...     product_short_name='VNP02DNB',
...     max_results=5,
... )
>>> path = cat.download_product(results[0]['id'])

**BIOMASS** (ESA MAAP):

>>> from grdl.IO.catalog import BIOMASSCatalog
>>> cat = BIOMASSCatalog('/data/biomass')
>>> results = cat.query_esa(
...     bbox=(115.5, -31.5, 116.8, -30.5),
...     max_results=10,
... )
>>> path = cat.download_product(results[0]['id'])

**TerraSAR-X** (local discovery only -- no remote API):

>>> from grdl.IO.catalog import TerraSARCatalog
>>> cat = TerraSARCatalog('/data/terrasar')
>>> cat.discover_images()

**Credentials** are loaded from ``~/.config/geoint/credentials.json``:

.. code-block:: json

    {
        "esa_copernicus": {"username": "...", "password": "..."},
        "nasa_earthdata": {"username": "...", "password": "..."},
        "esa_maap": {"offline_token": "..."}
    }

Environment variable fallbacks: ``COPERNICUS_USERNAME``,
``COPERNICUS_PASSWORD``, ``EARTHDATA_USERNAME``, ``EARTHDATA_PASSWORD``,
``ESA_MAAP_OFFLINE_TOKEN``.

**SQLite catalog databases** are stored by default in
``~/.config/geoint/catalogs/``::

    ~/.config/geoint/
    ├── credentials.json
    └── catalogs/
        ├── sentinel1_slc.db
        ├── sentinel2.db
        ├── nisar.db
        ├── aster.db
        ├── viirs.db
        ├── biomass.db
        └── terrasar.db

The directory is created automatically on first use. Pass a custom
``db_path`` to the constructor to override the default location.

**bbox format**: ``(west, south, east, north)`` in decimal degrees
(WGS-84). West/East are longitude (-180 to 180), South/North are
latitude (-90 to 90).

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

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
2026-03-13
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
