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
2026-02-10
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
    """GeoTIFF, NITF, and HDF5 readers importable from grdl.IO."""
    from grdl.IO import GeoTIFFReader, NITFReader, HDF5Reader
    assert GeoTIFFReader is not None
    assert NITFReader is not None
    assert HDF5Reader is not None


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


def test_import_ir_readers():
    """IR readers importable from grdl.IO."""
    from grdl.IO import ASTERReader
    assert ASTERReader is not None


def test_import_multispectral_readers():
    """Multispectral readers importable from grdl.IO."""
    from grdl.IO import VIIRSReader
    assert VIIRSReader is not None


def test_import_new_metadata_models():
    """VIIRS and ASTER metadata importable from grdl.IO."""
    from grdl.IO import VIIRSMetadata, ASTERMetadata
    assert VIIRSMetadata is not None
    assert ASTERMetadata is not None


def test_import_convenience_functions():
    """Convenience functions importable from grdl.IO."""
    from grdl.IO import (
        open_image, open_sar, open_biomass, load_credentials,
        open_eo, open_ir, open_multispectral,
    )
    assert callable(open_image)
    assert callable(open_sar)
    assert callable(open_biomass)
    assert callable(load_credentials)
    assert callable(open_eo)
    assert callable(open_ir)
    assert callable(open_multispectral)


# -- Direct submodule imports ----------------------------------------------

def test_import_geotiff_direct():
    """GeoTIFFReader importable from grdl.IO.geotiff."""
    from grdl.IO.geotiff import GeoTIFFReader
    assert GeoTIFFReader is not None


def test_import_nitf_direct():
    """NITFReader importable from grdl.IO.nitf."""
    from grdl.IO.nitf import NITFReader
    assert NITFReader is not None


def test_import_hdf5_direct():
    """HDF5Reader importable from grdl.IO.hdf5."""
    from grdl.IO.hdf5 import HDF5Reader
    assert HDF5Reader is not None


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


def test_import_ir_submodule():
    """IR readers importable from grdl.IO.ir."""
    from grdl.IO.ir import ASTERReader, ASTERMetadata, open_ir
    assert ASTERReader is not None
    assert ASTERMetadata is not None
    assert callable(open_ir)


def test_import_ir_individual_module():
    """Individual IR module imports work."""
    from grdl.IO.ir.aster import ASTERReader
    assert ASTERReader is not None


def test_import_multispectral_submodule():
    """Multispectral readers importable from grdl.IO.multispectral."""
    from grdl.IO.multispectral import VIIRSReader, VIIRSMetadata, open_multispectral
    assert VIIRSReader is not None
    assert VIIRSMetadata is not None
    assert callable(open_multispectral)


def test_import_multispectral_individual_module():
    """Individual multispectral module imports work."""
    from grdl.IO.multispectral.viirs import VIIRSReader
    assert VIIRSReader is not None


def test_import_eo_submodule():
    """EO scaffold importable from grdl.IO.eo."""
    from grdl.IO.eo import open_eo
    assert callable(open_eo)


def test_import_backend_flags():
    """Backend detection flags importable."""
    from grdl.IO.sar._backend import _HAS_SARKIT, _HAS_SARPY
    assert isinstance(_HAS_SARKIT, bool)
    assert isinstance(_HAS_SARPY, bool)


def test_import_ir_backend_flags():
    """IR backend detection flags importable."""
    from grdl.IO.ir._backend import _HAS_RASTERIO, _HAS_H5PY
    assert isinstance(_HAS_RASTERIO, bool)
    assert isinstance(_HAS_H5PY, bool)


def test_import_eo_backend_flags():
    """EO backend detection flags importable."""
    from grdl.IO.eo._backend import _HAS_RASTERIO, _HAS_GLYMUR
    assert isinstance(_HAS_RASTERIO, bool)
    assert isinstance(_HAS_GLYMUR, bool)


def test_import_multispectral_backend_flags():
    """Multispectral backend detection flags importable."""
    from grdl.IO.multispectral._backend import _HAS_H5PY, _HAS_XARRAY, _HAS_SPECTRAL
    assert isinstance(_HAS_H5PY, bool)
    assert isinstance(_HAS_XARRAY, bool)
    assert isinstance(_HAS_SPECTRAL, bool)


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


def test_hdf5_is_image_reader():
    """HDF5Reader inherits from ImageReader."""
    from grdl.IO.base import ImageReader
    from grdl.IO.hdf5 import HDF5Reader
    assert issubclass(HDF5Reader, ImageReader)


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


def test_aster_is_image_reader():
    """ASTERReader inherits from ImageReader."""
    from grdl.IO.base import ImageReader
    from grdl.IO.ir.aster import ASTERReader
    assert issubclass(ASTERReader, ImageReader)


def test_viirs_is_image_reader():
    """VIIRSReader inherits from ImageReader."""
    from grdl.IO.base import ImageReader
    from grdl.IO.multispectral.viirs import VIIRSReader
    assert issubclass(VIIRSReader, ImageReader)


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
        'BIOMASSL1Reader', 'BIOMASSCatalog',
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
        'BIOMASSL1Reader', 'BIOMASSCatalog',
        'SICDMetadata', 'SIDDMetadata', 'BIOMASSMetadata',
        'open_sar', 'open_biomass', 'load_credentials',
    ]
    for name in expected:
        assert name in sar_mod.__all__, f"'{name}' missing from sar.__all__"


# -- Metadata model imports ------------------------------------------------

def test_import_metadata_models_from_io():
    """Metadata subclasses importable from grdl.IO."""
    from grdl.IO import SICDMetadata, SIDDMetadata, BIOMASSMetadata
    assert SICDMetadata is not None
    assert SIDDMetadata is not None
    assert BIOMASSMetadata is not None


def test_import_metadata_models_from_models():
    """Metadata subclasses importable from grdl.IO.models."""
    from grdl.IO.models import (
        ImageMetadata, SICDMetadata, SIDDMetadata, BIOMASSMetadata,
        VIIRSMetadata, ASTERMetadata,
        XYZ, LatLon, LatLonHAE, RowCol, Poly1D, Poly2D, XYZPoly,
    )
    assert ImageMetadata is not None
    assert SICDMetadata is not None
    assert SIDDMetadata is not None
    assert BIOMASSMetadata is not None
    assert VIIRSMetadata is not None
    assert ASTERMetadata is not None
    assert XYZ is not None
    assert LatLon is not None
    assert LatLonHAE is not None
    assert RowCol is not None
    assert Poly1D is not None
    assert Poly2D is not None
    assert XYZPoly is not None


def test_import_metadata_models_from_sar():
    """Metadata subclasses importable from grdl.IO.sar."""
    from grdl.IO.sar import SICDMetadata, SIDDMetadata, BIOMASSMetadata
    assert SICDMetadata is not None
    assert SIDDMetadata is not None
    assert BIOMASSMetadata is not None


def test_import_metadata_models_from_ir():
    """ASTERMetadata importable from grdl.IO.ir."""
    from grdl.IO.ir import ASTERMetadata
    assert ASTERMetadata is not None


def test_import_metadata_models_from_multispectral():
    """VIIRSMetadata importable from grdl.IO.multispectral."""
    from grdl.IO.multispectral import VIIRSMetadata
    assert VIIRSMetadata is not None


def test_import_metadata_models_direct():
    """Metadata models importable directly from their module files."""
    from grdl.IO.models.viirs import VIIRSMetadata
    from grdl.IO.models.aster import ASTERMetadata
    assert VIIRSMetadata is not None
    assert ASTERMetadata is not None
