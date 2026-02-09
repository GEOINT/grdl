# -*- coding: utf-8 -*-
"""
IO Import Tests - Verify all IO module imports work correctly.

Tests that all public API classes and functions are importable from
their documented paths after the IO restructure.

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
2026-02-09

Modified
--------
2026-02-09
"""

import pytest


# -- Top-level IO imports --------------------------------------------------

def test_import_base_classes():
    """Base ABCs importable from grdl.IO."""
    from grdl.IO import ImageReader, ImageWriter, CatalogInterface
    assert ImageReader is not None
    assert ImageWriter is not None
    assert CatalogInterface is not None


def test_import_base_format_readers():
    """GeoTIFF and NITF readers importable from grdl.IO."""
    from grdl.IO import GeoTIFFReader, NITFReader
    assert GeoTIFFReader is not None
    assert NITFReader is not None


def test_import_sar_readers():
    """SAR readers importable from grdl.IO."""
    from grdl.IO import SICDReader, CPHDReader, CRSDReader, SIDDReader
    assert SICDReader is not None
    assert CPHDReader is not None
    assert CRSDReader is not None
    assert SIDDReader is not None


def test_import_biomass():
    """BIOMASS classes importable from grdl.IO."""
    from grdl.IO import BIOMASSL1Reader, BIOMASSCatalog
    assert BIOMASSL1Reader is not None
    assert BIOMASSCatalog is not None


def test_import_convenience_functions():
    """Convenience functions importable from grdl.IO."""
    from grdl.IO import open_image, open_sar, open_biomass, load_credentials
    assert callable(open_image)
    assert callable(open_sar)
    assert callable(open_biomass)
    assert callable(load_credentials)


# -- Direct submodule imports ----------------------------------------------

def test_import_geotiff_direct():
    """GeoTIFFReader importable from grdl.IO.geotiff."""
    from grdl.IO.geotiff import GeoTIFFReader
    assert GeoTIFFReader is not None


def test_import_nitf_direct():
    """NITFReader importable from grdl.IO.nitf."""
    from grdl.IO.nitf import NITFReader
    assert NITFReader is not None


def test_import_sar_submodule():
    """SAR readers importable from grdl.IO.sar."""
    from grdl.IO.sar import (
        SICDReader, CPHDReader, CRSDReader, SIDDReader,
        BIOMASSL1Reader, BIOMASSCatalog,
        open_sar, open_biomass, load_credentials,
    )
    assert SICDReader is not None
    assert CPHDReader is not None
    assert CRSDReader is not None
    assert SIDDReader is not None
    assert BIOMASSL1Reader is not None
    assert BIOMASSCatalog is not None


def test_import_sar_individual_modules():
    """Individual SAR module imports work."""
    from grdl.IO.sar.sicd import SICDReader
    from grdl.IO.sar.cphd import CPHDReader
    from grdl.IO.sar.crsd import CRSDReader
    from grdl.IO.sar.sidd import SIDDReader
    from grdl.IO.sar.biomass import BIOMASSL1Reader
    from grdl.IO.sar.biomass_catalog import BIOMASSCatalog, load_credentials

    assert SICDReader is not None
    assert CPHDReader is not None
    assert CRSDReader is not None
    assert SIDDReader is not None
    assert BIOMASSL1Reader is not None
    assert BIOMASSCatalog is not None
    assert callable(load_credentials)


def test_import_backend_flags():
    """Backend detection flags importable."""
    from grdl.IO.sar._backend import _HAS_SARKIT, _HAS_SARPY
    assert isinstance(_HAS_SARKIT, bool)
    assert isinstance(_HAS_SARPY, bool)


# -- ABC inheritance -------------------------------------------------------

def test_geotiff_is_image_reader():
    """GeoTIFFReader inherits from ImageReader."""
    from grdl.IO.base import ImageReader
    from grdl.IO.geotiff import GeoTIFFReader
    assert issubclass(GeoTIFFReader, ImageReader)


def test_nitf_is_image_reader():
    """NITFReader inherits from ImageReader."""
    from grdl.IO.base import ImageReader
    from grdl.IO.nitf import NITFReader
    assert issubclass(NITFReader, ImageReader)


def test_sicd_is_image_reader():
    """SICDReader inherits from ImageReader."""
    from grdl.IO.base import ImageReader
    from grdl.IO.sar.sicd import SICDReader
    assert issubclass(SICDReader, ImageReader)


def test_cphd_is_image_reader():
    """CPHDReader inherits from ImageReader."""
    from grdl.IO.base import ImageReader
    from grdl.IO.sar.cphd import CPHDReader
    assert issubclass(CPHDReader, ImageReader)


def test_crsd_is_image_reader():
    """CRSDReader inherits from ImageReader."""
    from grdl.IO.base import ImageReader
    from grdl.IO.sar.crsd import CRSDReader
    assert issubclass(CRSDReader, ImageReader)


def test_sidd_is_image_reader():
    """SIDDReader inherits from ImageReader."""
    from grdl.IO.base import ImageReader
    from grdl.IO.sar.sidd import SIDDReader
    assert issubclass(SIDDReader, ImageReader)


def test_biomass_is_image_reader():
    """BIOMASSL1Reader inherits from ImageReader."""
    from grdl.IO.base import ImageReader
    from grdl.IO.sar.biomass import BIOMASSL1Reader
    assert issubclass(BIOMASSL1Reader, ImageReader)


def test_catalog_is_catalog_interface():
    """BIOMASSCatalog inherits from CatalogInterface."""
    from grdl.IO.base import CatalogInterface
    from grdl.IO.sar.biomass_catalog import BIOMASSCatalog
    assert issubclass(BIOMASSCatalog, CatalogInterface)


# -- __all__ verification -------------------------------------------------

def test_io_all_exports():
    """IO __all__ contains expected symbols."""
    import grdl.IO as io_mod
    expected = [
        'ImageReader', 'ImageWriter', 'CatalogInterface',
        'GeoTIFFReader', 'NITFReader',
        'SICDReader', 'CPHDReader', 'CRSDReader', 'SIDDReader',
        'BIOMASSL1Reader', 'BIOMASSCatalog',
        'open_image', 'open_sar', 'open_biomass', 'load_credentials',
    ]
    for name in expected:
        assert name in io_mod.__all__, f"'{name}' missing from IO.__all__"


def test_sar_all_exports():
    """SAR __all__ contains expected symbols."""
    import grdl.IO.sar as sar_mod
    expected = [
        'SICDReader', 'CPHDReader', 'CRSDReader', 'SIDDReader',
        'BIOMASSL1Reader', 'BIOMASSCatalog',
        'open_sar', 'open_biomass', 'load_credentials',
    ]
    for name in expected:
        assert name in sar_mod.__all__, f"'{name}' missing from sar.__all__"
