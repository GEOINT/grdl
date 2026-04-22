# -*- coding: utf-8 -*-
"""
IO Import Tests - Verify public API surface after IO restructure.

Replaces 39 individual ``is not None`` / ``issubclass`` / ``callable``
tests with a single parametrized smoke test plus ``__all__`` verification.

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
2026-02-09

Modified
--------
2026-02-10
"""

import importlib

import pytest


# -- Parametrized import smoke test ----------------------------------------

@pytest.mark.parametrize("name", [
    'ImageReader', 'ImageWriter', 'CatalogInterface', 'ImageMetadata',
    'SICDMetadata', 'SIDDMetadata', 'BIOMASSMetadata',
    'VIIRSMetadata', 'ASTERMetadata',
    'GeoTIFFReader', 'HDF5Reader', 'JP2Reader', 'NITFReader',
    'SICDReader', 'CPHDReader', 'CRSDReader', 'SIDDReader',
    'BIOMASSL1Reader',
    'BIOMASSCatalog', 'Sentinel1SLCCatalog', 'TerraSARCatalog',
    'NISARCatalog', 'Sentinel2Catalog', 'ASTERCatalog', 'VIIRSCatalog',
    'ASTERReader', 'VIIRSReader',
    'open_image', 'open_sar', 'open_biomass', 'load_credentials',
    'open_eo', 'open_ir', 'open_multispectral',
])
def test_io_public_name_importable(name):
    """Every name in IO.__all__ is importable and is a class or callable."""
    io_mod = importlib.import_module('grdl.IO')
    obj = getattr(io_mod, name)
    assert callable(obj), f"grdl.IO.{name} is not callable"


# -- __all__ verification -------------------------------------------------

def test_io_all_exports():
    """IO __all__ contains expected symbols."""
    import grdl.IO as io_mod
    expected = [
        'ImageReader', 'ImageWriter', 'CatalogInterface', 'ImageMetadata',
        'SICDMetadata', 'SIDDMetadata', 'BIOMASSMetadata',
        'VIIRSMetadata', 'ASTERMetadata',
        'GeoTIFFReader', 'HDF5Reader', 'JP2Reader', 'NITFReader',
        'SICDReader', 'CPHDReader', 'CRSDReader', 'SIDDReader',
        'BIOMASSL1Reader',
        'BIOMASSCatalog', 'Sentinel1SLCCatalog', 'TerraSARCatalog',
        'NISARCatalog', 'Sentinel2Catalog', 'ASTERCatalog', 'VIIRSCatalog',
        'ASTERReader', 'VIIRSReader',
        'open_image', 'open_sar', 'open_biomass', 'load_credentials',
        'open_eo', 'open_ir', 'open_multispectral',
    ]
    for name in expected:
        assert name in io_mod.__all__, f"'{name}' missing from IO.__all__"


def test_sar_all_exports():
    """SAR __all__ contains expected symbols."""
    import grdl.IO.sar as sar_mod
    expected = [
        'SICDReader', 'CPHDReader', 'CRSDReader', 'SIDDReader',
        'BIOMASSL1Reader',
        'SICDMetadata', 'SIDDMetadata', 'BIOMASSMetadata',
        'open_sar', 'open_biomass',
    ]
    for name in expected:
        assert name in sar_mod.__all__, f"'{name}' missing from sar.__all__"
    # Catalog classes live in grdl.IO.catalog, not grdl.IO.sar
    assert 'BIOMASSCatalog' not in sar_mod.__all__
    assert 'load_credentials' not in sar_mod.__all__


def test_catalog_all_exports():
    """Catalog __all__ contains expected symbols."""
    import grdl.IO.catalog as cat_mod
    expected = [
        'BIOMASSCatalog', 'load_credentials',
        'Sentinel1SLCCatalog', 'TerraSARCatalog', 'NISARCatalog',
        'Sentinel2Catalog', 'ASTERCatalog', 'VIIRSCatalog',
    ]
    for name in expected:
        assert name in cat_mod.__all__, f"'{name}' missing from catalog.__all__"
