# -*- coding: utf-8 -*-
"""
IR Reader Tests - Unit tests for ASTERReader.

Uses synthetic GeoTIFF files created with rasterio and companion XML
files for testing ASTER-specific metadata extraction.

Dependencies
------------
pytest
rasterio

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
2026-02-10

Modified
--------
2026-02-10
"""

import pytest
import numpy as np
from pathlib import Path

try:
    import rasterio
    from rasterio.transform import from_bounds
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

pytestmark = pytest.mark.skipif(
    not _HAS_RASTERIO, reason="rasterio not installed"
)


@pytest.fixture
def aster_l1t_tiff(tmp_path):
    """Create a synthetic ASTER L1T GeoTIFF for testing."""
    filepath = tmp_path / "AST_L1T_00305042006_B10.tif"
    data = np.random.rand(100, 200).astype(np.float32)
    transform = from_bounds(-119.0, 34.0, -118.0, 35.0, 200, 100)

    with rasterio.open(
        str(filepath), 'w', driver='GTiff',
        height=100, width=200, count=1,
        dtype='float32', crs='EPSG:4326',
        transform=transform,
    ) as ds:
        ds.write(data, 1)

    return filepath, data


@pytest.fixture
def aster_gdem_tiff(tmp_path):
    """Create a synthetic ASTER GDEM GeoTIFF for testing."""
    filepath = tmp_path / "ASTGTM_N34W119_dem.tif"
    data = np.random.randint(0, 3000, size=(100, 200)).astype(np.int16)
    transform = from_bounds(-119.0, 34.0, -118.0, 35.0, 200, 100)

    with rasterio.open(
        str(filepath), 'w', driver='GTiff',
        height=100, width=200, count=1,
        dtype='int16', crs='EPSG:4326',
        transform=transform, nodata=-9999,
    ) as ds:
        ds.write(data, 1)

    return filepath, data


@pytest.fixture
def aster_with_xml(tmp_path):
    """Create an ASTER L1T GeoTIFF with companion XML metadata."""
    filepath = tmp_path / "AST_L1T_00305042006_B10.tif"
    data = np.random.rand(100, 200).astype(np.float32)
    transform = from_bounds(-119.0, 34.0, -118.0, 35.0, 200, 100)

    with rasterio.open(
        str(filepath), 'w', driver='GTiff',
        height=100, width=200, count=1,
        dtype='float32', crs='EPSG:4326',
        transform=transform,
    ) as ds:
        ds.write(data, 1)

    # Write companion XML
    xml_path = tmp_path / "AST_L1T_00305042006_B10.xml"
    xml_path.write_text("""\
<?xml version="1.0" encoding="UTF-8"?>
<GranuleMetaDataFile>
  <CalendarDate>2006/05/04</CalendarDate>
  <TimeofDay>10:30:45</TimeofDay>
  <LocalGranuleID>AST_L1T_00305042006</LocalGranuleID>
  <SceneCloudCoverage>15.0</SceneCloudCoverage>
  <PSA><PSAName>Solar.Azimuth</PSAName><PSAValue>135.5</PSAValue></PSA>
  <PSA><PSAName>Solar.Elevation</PSAName><PSAValue>62.3</PSAValue></PSA>
  <PSA><PSAName>ASTERMapProjection.OrbitDirection</PSAName><PSAValue>DE</PSAValue></PSA>
  <PSA><PSAName>CorrectionLevelID</PSAName><PSAValue>L1T</PSAValue></PSA>
  <UpperLeftCornerLatitude>35.0</UpperLeftCornerLatitude>
  <UpperLeftCornerLongitude>-119.0</UpperLeftCornerLongitude>
  <LowerRightCornerLatitude>34.0</LowerRightCornerLatitude>
  <LowerRightCornerLongitude>-118.0</LowerRightCornerLongitude>
</GranuleMetaDataFile>
""")

    return filepath, data


@pytest.fixture
def multi_band_aster(tmp_path):
    """Create a multi-band ASTER GeoTIFF."""
    filepath = tmp_path / "AST_L1T_00305042006_TIR.tif"
    data = np.random.rand(5, 80, 120).astype(np.float32)
    transform = from_bounds(-119.0, 34.0, -118.0, 35.0, 120, 80)

    with rasterio.open(
        str(filepath), 'w', driver='GTiff',
        height=80, width=120, count=5,
        dtype='float32', crs='EPSG:4326',
        transform=transform,
    ) as ds:
        ds.write(data)

    return filepath, data


# -- Metadata tests ---------------------------------------------------------

def test_metadata_l1t(aster_l1t_tiff):
    """Metadata extracted correctly for ASTER L1T product."""
    from grdl.IO.ir.aster import ASTERReader

    filepath, _ = aster_l1t_tiff
    with ASTERReader(filepath) as reader:
        assert reader.metadata.format == 'ASTER_L1T'
        assert reader.metadata.rows == 100
        assert reader.metadata.cols == 200
        assert reader.metadata.bands == 1
        assert reader.metadata.processing_level == 'L1T'
        assert reader.metadata.tir_available is True


def test_metadata_gdem(aster_gdem_tiff):
    """Metadata extracted correctly for ASTER GDEM product."""
    from grdl.IO.ir.aster import ASTERReader

    filepath, _ = aster_gdem_tiff
    with ASTERReader(filepath) as reader:
        assert reader.metadata.format == 'ASTER_GDEM'
        assert reader.metadata.processing_level == 'GDEM'
        assert reader.metadata.dtype == 'int16'
        assert reader.metadata.nodata == -9999
        assert 'EPSG:4326' in reader.metadata.crs


def test_metadata_xml_extraction(aster_with_xml):
    """ASTER-specific metadata extracted from companion XML."""
    from grdl.IO.ir.aster import ASTERReader

    filepath, _ = aster_with_xml
    with ASTERReader(filepath) as reader:
        meta = reader.metadata
        assert meta.acquisition_date == '2006/05/04'
        assert meta.acquisition_time == '10:30:45'
        assert meta.local_granule_id == 'AST_L1T_00305042006'
        assert meta.cloud_cover == 15.0
        assert meta.sun_azimuth == 135.5
        assert meta.sun_elevation == 62.3
        assert meta.orbit_direction == 'DE'
        assert meta.correction_level == 'L1T'


def test_metadata_corner_coords(aster_with_xml):
    """Corner coordinates extracted from companion XML."""
    from grdl.IO.ir.aster import ASTERReader

    filepath, _ = aster_with_xml
    with ASTERReader(filepath) as reader:
        corners = reader.metadata.corner_coords
        assert corners is not None
        assert 'UL' in corners
        assert corners['UL'] == (35.0, -119.0)
        assert 'LR' in corners
        assert corners['LR'] == (34.0, -118.0)


def test_metadata_no_xml(aster_l1t_tiff):
    """Reader works without companion XML (fields are None)."""
    from grdl.IO.ir.aster import ASTERReader

    filepath, _ = aster_l1t_tiff
    with ASTERReader(filepath) as reader:
        assert reader.metadata.acquisition_date is None
        assert reader.metadata.sun_azimuth is None
        assert reader.metadata.corner_coords is None


def test_metadata_is_aster_type(aster_l1t_tiff):
    """Metadata is ASTERMetadata instance."""
    from grdl.IO.ir.aster import ASTERReader
    from grdl.IO.models import ASTERMetadata

    filepath, _ = aster_l1t_tiff
    with ASTERReader(filepath) as reader:
        assert isinstance(reader.metadata, ASTERMetadata)


def test_metadata_dict_compat(aster_l1t_tiff):
    """ASTERMetadata supports dict-like access."""
    from grdl.IO.ir.aster import ASTERReader

    filepath, _ = aster_l1t_tiff
    with ASTERReader(filepath) as reader:
        assert reader.metadata['format'] == 'ASTER_L1T'
        assert reader.metadata['rows'] == 100
        assert 'processing_level' in reader.metadata


# -- Shape and dtype tests --------------------------------------------------

def test_get_shape_single_band(aster_l1t_tiff):
    """get_shape returns (rows, cols) for single band."""
    from grdl.IO.ir.aster import ASTERReader

    filepath, _ = aster_l1t_tiff
    with ASTERReader(filepath) as reader:
        assert reader.get_shape() == (100, 200)


def test_get_shape_multi_band(multi_band_aster):
    """get_shape returns (rows, cols, bands) for multi-band."""
    from grdl.IO.ir.aster import ASTERReader

    filepath, _ = multi_band_aster
    with ASTERReader(filepath) as reader:
        assert reader.get_shape() == (80, 120, 5)


def test_get_dtype(aster_l1t_tiff):
    """get_dtype returns correct numpy dtype."""
    from grdl.IO.ir.aster import ASTERReader

    filepath, _ = aster_l1t_tiff
    with ASTERReader(filepath) as reader:
        assert reader.get_dtype() == np.dtype('float32')


# -- Read tests --------------------------------------------------------------

def test_read_chip_single_band(aster_l1t_tiff):
    """read_chip returns correct data for single band."""
    from grdl.IO.ir.aster import ASTERReader

    filepath, original = aster_l1t_tiff
    with ASTERReader(filepath) as reader:
        chip = reader.read_chip(10, 30, 20, 50)
        assert chip.shape == (20, 30)
        np.testing.assert_array_almost_equal(
            chip, original[10:30, 20:50]
        )


def test_read_chip_multi_band(multi_band_aster):
    """read_chip returns correct data for multi-band."""
    from grdl.IO.ir.aster import ASTERReader

    filepath, original = multi_band_aster
    with ASTERReader(filepath) as reader:
        chip = reader.read_chip(0, 40, 0, 60)
        assert chip.shape == (5, 40, 60)
        np.testing.assert_array_almost_equal(
            chip, original[:, :40, :60]
        )


def test_read_chip_band_selection(multi_band_aster):
    """read_chip with specific bands returns correct data."""
    from grdl.IO.ir.aster import ASTERReader

    filepath, original = multi_band_aster
    with ASTERReader(filepath) as reader:
        chip = reader.read_chip(0, 40, 0, 60, bands=[2])
        assert chip.shape == (40, 60)
        np.testing.assert_array_almost_equal(
            chip, original[2, :40, :60]
        )


def test_read_full_single_band(aster_l1t_tiff):
    """read_full returns entire image for single band."""
    from grdl.IO.ir.aster import ASTERReader

    filepath, original = aster_l1t_tiff
    with ASTERReader(filepath) as reader:
        full = reader.read_full()
        assert full.shape == (100, 200)
        np.testing.assert_array_almost_equal(full, original)


# -- Validation tests --------------------------------------------------------

def test_read_chip_negative_start_raises(aster_l1t_tiff):
    """Negative start indices raise ValueError."""
    from grdl.IO.ir.aster import ASTERReader

    filepath, _ = aster_l1t_tiff
    with ASTERReader(filepath) as reader:
        with pytest.raises(ValueError, match="non-negative"):
            reader.read_chip(-1, 10, 0, 10)


def test_read_chip_out_of_bounds_raises(aster_l1t_tiff):
    """Out-of-bounds end indices raise ValueError."""
    from grdl.IO.ir.aster import ASTERReader

    filepath, _ = aster_l1t_tiff
    with ASTERReader(filepath) as reader:
        with pytest.raises(ValueError, match="exceed"):
            reader.read_chip(0, 200, 0, 10)


def test_file_not_found():
    """FileNotFoundError for non-existent file."""
    from grdl.IO.ir.aster import ASTERReader

    with pytest.raises(FileNotFoundError):
        ASTERReader('/nonexistent/file.tif')


# -- Context manager and ABC tests ------------------------------------------

def test_context_manager(aster_l1t_tiff):
    """Context manager opens and closes cleanly."""
    from grdl.IO.ir.aster import ASTERReader

    filepath, _ = aster_l1t_tiff
    with ASTERReader(filepath) as reader:
        assert reader.metadata.rows == 100


def test_close_idempotent(aster_l1t_tiff):
    """Calling close() multiple times does not raise."""
    from grdl.IO.ir.aster import ASTERReader

    filepath, _ = aster_l1t_tiff
    reader = ASTERReader(filepath)
    reader.close()
    reader.close()


def test_is_image_reader_subclass():
    """ASTERReader is a subclass of ImageReader."""
    from grdl.IO.base import ImageReader
    from grdl.IO.ir.aster import ASTERReader
    assert issubclass(ASTERReader, ImageReader)


# -- open_ir auto-detection --------------------------------------------------

def test_open_ir_aster(aster_l1t_tiff):
    """open_ir auto-detects ASTER products."""
    from grdl.IO.ir import open_ir

    filepath, _ = aster_l1t_tiff
    with open_ir(filepath) as reader:
        assert reader.metadata.format == 'ASTER_L1T'
