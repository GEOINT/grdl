# -*- coding: utf-8 -*-
"""
Tests for grdl.IO.generic — GDALFallbackReader and open_any().

Tests classification logic, metadata extraction, GDAL fallback reader,
and open_any dispatch.  Uses mocks to avoid requiring real imagery files.

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
2026-03-07

Modified
--------
2026-03-07
"""

# Standard library
from pathlib import Path
from unittest import mock

# Third-party
import numpy as np
import pytest

# Module under test
from grdl.IO.generic import (
    _classify_modality,
    _driver_to_format,
    _extract_gdal_metadata,
    _retry_identified_reader,
    GDALFallbackReader,
    open_any,
)
from grdl.IO.models import ImageMetadata


# ===================================================================
# Test _classify_modality
# ===================================================================

class TestClassifyModality:
    """Tests for the modality classification function."""

    def test_nitf_icat_sar(self):
        modality, confidence, clues = _classify_modality(
            driver='NITF', dtype='complex64', bands=1,
            tags={'NITF_ICAT': 'SAR'},
        )
        assert modality == 'SAR'
        assert confidence == 'high'
        assert any('ICAT' in c for c in clues)

    def test_nitf_icat_ir(self):
        modality, confidence, clues = _classify_modality(
            driver='NITF', dtype='float32', bands=1,
            tags={'NITF_ICAT': 'IR'},
        )
        assert modality == 'IR'
        assert confidence == 'high'

    def test_nitf_icat_multispectral(self):
        modality, confidence, clues = _classify_modality(
            driver='NITF', dtype='uint16', bands=4,
            tags={'NITF_ICAT': 'MS'},
        )
        assert modality == 'MSI'
        assert confidence == 'high'

    def test_nitf_icat_eo(self):
        modality, confidence, clues = _classify_modality(
            driver='NITF', dtype='uint8', bands=3,
            tags={'NITF_ICAT': 'VIS'},
        )
        assert modality == 'EO'
        assert confidence == 'high'

    def test_complex_dtype_implies_sar(self):
        modality, confidence, clues = _classify_modality(
            driver='GTiff', dtype='complex64', bands=1,
            tags={},
        )
        assert modality == 'SAR'
        assert confidence == 'high'
        assert any('complex' in c for c in clues)

    def test_complex128_implies_sar(self):
        modality, confidence, clues = _classify_modality(
            driver='ENVI', dtype='complex128', bands=1,
            tags={},
        )
        assert modality == 'SAR'
        assert confidence == 'high'

    def test_sar_driver(self):
        modality, confidence, clues = _classify_modality(
            driver='COSAR', dtype='float32', bands=1,
            tags={},
        )
        assert modality == 'SAR'
        assert confidence == 'high'
        assert any('COSAR' in c for c in clues)

    def test_sentinel2_driver(self):
        modality, confidence, clues = _classify_modality(
            driver='SENTINEL2', dtype='uint16', bands=13,
            tags={},
        )
        assert modality == 'MSI'
        assert confidence == 'high'

    def test_many_bands_hyperspectral(self):
        modality, confidence, clues = _classify_modality(
            driver='ENVI', dtype='float32', bands=224,
            tags={},
        )
        assert modality == 'HSI'
        assert confidence == 'medium'
        assert any('224' in c for c in clues)

    def test_moderate_bands_multispectral(self):
        modality, confidence, clues = _classify_modality(
            driver='GTiff', dtype='uint16', bands=13,
            tags={},
        )
        assert modality == 'MSI'
        assert confidence == 'medium'

    def test_rgb_bands_eo(self):
        modality, confidence, clues = _classify_modality(
            driver='GTiff', dtype='uint8', bands=3,
            tags={},
        )
        assert modality == 'EO'
        assert confidence == 'low'
        assert any('RGB' in c for c in clues)

    def test_rgba_bands_eo(self):
        modality, confidence, clues = _classify_modality(
            driver='GTiff', dtype='uint8', bands=4,
            tags={},
        )
        assert modality == 'EO'
        assert confidence == 'low'

    def test_single_band_ambiguous(self):
        modality, confidence, clues = _classify_modality(
            driver='GTiff', dtype='float32', bands=1,
            tags={},
        )
        assert modality is None
        assert any('ambiguous' in c for c in clues)

    def test_no_clues(self):
        modality, confidence, clues = _classify_modality(
            driver='VRT', dtype='uint16', bands=2,
            tags={},
        )
        assert modality is None
        assert len(clues) > 0

    def test_nitf_ftitle_keyword(self):
        modality, confidence, clues = _classify_modality(
            driver='NITF', dtype='float32', bands=1,
            tags={'NITF_FTITLE': 'SAR imagery product'},
        )
        assert modality == 'SAR'
        assert confidence == 'medium'

    def test_nitf_isorce_keyword(self):
        modality, confidence, clues = _classify_modality(
            driver='NITF', dtype='uint8', bands=1,
            tags={'NITF_ISORCE': 'RADAR SENSOR X'},
        )
        assert modality == 'SAR'
        assert confidence == 'medium'


# ===================================================================
# Test _driver_to_format
# ===================================================================

class TestDriverToFormat:
    """Tests for GDAL driver-to-format mapping."""

    def test_known_drivers(self):
        assert _driver_to_format('GTiff') == 'GeoTIFF'
        assert _driver_to_format('NITF') == 'NITF'
        assert _driver_to_format('HDF5') == 'HDF5'
        assert _driver_to_format('JP2OpenJPEG') == 'JPEG2000'
        assert _driver_to_format('ENVI') == 'ENVI'
        assert _driver_to_format('HFA') == 'ERDAS_IMG'
        assert _driver_to_format('COSAR') == 'COSAR'
        assert _driver_to_format('TSX') == 'TerraSAR-X'

    def test_unknown_driver_passthrough(self):
        assert _driver_to_format('SomeObscureDriver') == 'SomeObscureDriver'


# ===================================================================
# Test _extract_gdal_metadata
# ===================================================================

class TestExtractGDALMetadata:
    """Tests for GDAL metadata extraction."""

    def test_extracts_driver(self):
        ds = mock.MagicMock()
        ds.driver = 'GTiff'
        ds.transform = None
        ds.bounds = None
        ds.res = None
        ds.descriptions = None
        ds.colorinterp = None
        ds.compression = None
        ds.interleaving = None
        ds.dtypes = ('float32',)
        ds.count = 1
        ds.block_shapes = None
        ds.overviews.return_value = []
        ds.tags.return_value = {}

        meta = _extract_gdal_metadata(ds)
        assert meta['gdal_driver'] == 'GTiff'

    def test_extracts_geospatial(self):
        ds = mock.MagicMock()
        ds.driver = 'GTiff'
        ds.transform = 'some_affine'
        ds.bounds = (0, 0, 1, 1)
        ds.res = (0.001, 0.001)
        ds.descriptions = None
        ds.colorinterp = None
        ds.compression = None
        ds.interleaving = None
        ds.dtypes = ('float32',)
        ds.count = 1
        ds.block_shapes = None
        ds.overviews.return_value = []
        ds.tags.return_value = {}

        meta = _extract_gdal_metadata(ds)
        assert meta['transform'] == 'some_affine'
        assert meta['bounds'] == (0, 0, 1, 1)
        assert meta['resolution'] == (0.001, 0.001)

    def test_extracts_tags(self):
        ds = mock.MagicMock()
        ds.driver = 'NITF'
        ds.transform = None
        ds.bounds = None
        ds.res = None
        ds.descriptions = None
        ds.colorinterp = None
        ds.compression = None
        ds.interleaving = None
        ds.dtypes = ('uint8',)
        ds.count = 1
        ds.block_shapes = None
        ds.overviews.return_value = []
        ds.tags.return_value = {
            'NITF_ICAT': 'SAR',
            'NITF_IREP': 'NODISPLY',
        }

        meta = _extract_gdal_metadata(ds)
        assert meta['tag_NITF_ICAT'] == 'SAR'
        assert meta['tag_NITF_IREP'] == 'NODISPLY'


# ===================================================================
# Test GDALFallbackReader
# ===================================================================

class TestGDALFallbackReader:
    """Tests for the GDALFallbackReader class."""

    def test_requires_rasterio(self):
        with mock.patch('grdl.IO.generic._HAS_RASTERIO', False):
            with pytest.raises(ImportError, match='rasterio'):
                GDALFallbackReader('/some/file.dat')

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            GDALFallbackReader('/nonexistent/path/file.tif')

    @mock.patch('grdl.IO.generic.rasterio')
    def test_reads_with_gdal(self, mock_rasterio, tmp_path):
        # Create a dummy file
        dummy = tmp_path / 'test.dat'
        dummy.write_bytes(b'\x00' * 100)

        # Mock rasterio dataset
        mock_ds = mock.MagicMock()
        mock_ds.driver = 'GTiff'
        mock_ds.height = 64
        mock_ds.width = 128
        mock_ds.count = 1
        mock_ds.dtypes = ('float32',)
        mock_ds.crs = None
        mock_ds.nodata = None
        mock_ds.transform = None
        mock_ds.bounds = None
        mock_ds.res = None
        mock_ds.descriptions = None
        mock_ds.colorinterp = None
        mock_ds.compression = None
        mock_ds.interleaving = None
        mock_ds.block_shapes = None
        mock_ds.overviews.return_value = []
        mock_ds.tags.return_value = {}

        mock_rasterio.open.return_value = mock_ds

        reader = GDALFallbackReader(str(dummy))
        assert reader.metadata.format == 'GeoTIFF'
        assert reader.metadata.rows == 64
        assert reader.metadata.cols == 128
        assert reader.metadata.dtype == 'float32'
        reader.close()

    @mock.patch('grdl.IO.generic.rasterio')
    def test_classifies_sar(self, mock_rasterio, tmp_path):
        dummy = tmp_path / 'test.nitf'
        dummy.write_bytes(b'\x00' * 100)

        mock_ds = mock.MagicMock()
        mock_ds.driver = 'NITF'
        mock_ds.height = 512
        mock_ds.width = 512
        mock_ds.count = 1
        mock_ds.dtypes = ('complex64',)
        mock_ds.crs = None
        mock_ds.nodata = None
        mock_ds.transform = None
        mock_ds.bounds = None
        mock_ds.res = None
        mock_ds.descriptions = None
        mock_ds.colorinterp = None
        mock_ds.compression = None
        mock_ds.interleaving = None
        mock_ds.block_shapes = None
        mock_ds.overviews.return_value = []
        mock_ds.tags.return_value = {'NITF_ICAT': 'SAR'}

        mock_rasterio.open.return_value = mock_ds

        reader = GDALFallbackReader(str(dummy))
        assert reader.detected_modality == 'SAR'
        assert reader.classification_confidence == 'high'
        assert reader.metadata.format == 'NITF'
        assert reader.metadata.get('detected_modality') == 'SAR'
        reader.close()

    @mock.patch('grdl.IO.generic.rasterio')
    def test_read_chip(self, mock_rasterio, tmp_path):
        dummy = tmp_path / 'test.tif'
        dummy.write_bytes(b'\x00' * 100)

        mock_ds = mock.MagicMock()
        mock_ds.driver = 'GTiff'
        mock_ds.height = 64
        mock_ds.width = 64
        mock_ds.count = 1
        mock_ds.dtypes = ('uint8',)
        mock_ds.crs = None
        mock_ds.nodata = None
        mock_ds.transform = None
        mock_ds.bounds = None
        mock_ds.res = None
        mock_ds.descriptions = None
        mock_ds.colorinterp = None
        mock_ds.compression = None
        mock_ds.interleaving = None
        mock_ds.block_shapes = None
        mock_ds.overviews.return_value = []
        mock_ds.tags.return_value = {}

        chip = np.zeros((1, 32, 32), dtype=np.uint8)
        mock_ds.read.return_value = chip
        mock_rasterio.open.return_value = mock_ds

        reader = GDALFallbackReader(str(dummy))
        result = reader.read_chip(0, 32, 0, 32)
        assert result.shape == (32, 32)
        reader.close()

    @mock.patch('grdl.IO.generic.rasterio')
    def test_read_chip_multiband(self, mock_rasterio, tmp_path):
        dummy = tmp_path / 'test.tif'
        dummy.write_bytes(b'\x00' * 100)

        mock_ds = mock.MagicMock()
        mock_ds.driver = 'GTiff'
        mock_ds.height = 64
        mock_ds.width = 64
        mock_ds.count = 3
        mock_ds.dtypes = ('uint8',)
        mock_ds.crs = None
        mock_ds.nodata = None
        mock_ds.transform = None
        mock_ds.bounds = None
        mock_ds.res = None
        mock_ds.descriptions = None
        mock_ds.colorinterp = None
        mock_ds.compression = None
        mock_ds.interleaving = None
        mock_ds.block_shapes = None
        mock_ds.overviews.return_value = []
        mock_ds.tags.return_value = {}

        chip = np.zeros((3, 32, 32), dtype=np.uint8)
        mock_ds.read.return_value = chip
        mock_rasterio.open.return_value = mock_ds

        reader = GDALFallbackReader(str(dummy))
        result = reader.read_chip(0, 32, 0, 32)
        assert result.shape == (3, 32, 32)
        assert reader.get_shape() == (64, 64, 3)
        reader.close()

    @mock.patch('grdl.IO.generic.rasterio')
    def test_read_chip_bounds_check(self, mock_rasterio, tmp_path):
        dummy = tmp_path / 'test.tif'
        dummy.write_bytes(b'\x00' * 100)

        mock_ds = mock.MagicMock()
        mock_ds.driver = 'GTiff'
        mock_ds.height = 64
        mock_ds.width = 64
        mock_ds.count = 1
        mock_ds.dtypes = ('float32',)
        mock_ds.crs = None
        mock_ds.nodata = None
        mock_ds.transform = None
        mock_ds.bounds = None
        mock_ds.res = None
        mock_ds.descriptions = None
        mock_ds.colorinterp = None
        mock_ds.compression = None
        mock_ds.interleaving = None
        mock_ds.block_shapes = None
        mock_ds.overviews.return_value = []
        mock_ds.tags.return_value = {}
        mock_rasterio.open.return_value = mock_ds

        reader = GDALFallbackReader(str(dummy))

        with pytest.raises(ValueError, match='non-negative'):
            reader.read_chip(-1, 10, 0, 10)
        with pytest.raises(ValueError, match='exceed'):
            reader.read_chip(0, 100, 0, 10)
        reader.close()

    @mock.patch('grdl.IO.generic.rasterio')
    def test_get_dtype(self, mock_rasterio, tmp_path):
        dummy = tmp_path / 'test.tif'
        dummy.write_bytes(b'\x00' * 100)

        mock_ds = mock.MagicMock()
        mock_ds.driver = 'GTiff'
        mock_ds.height = 10
        mock_ds.width = 10
        mock_ds.count = 1
        mock_ds.dtypes = ('complex64',)
        mock_ds.crs = None
        mock_ds.nodata = None
        mock_ds.transform = None
        mock_ds.bounds = None
        mock_ds.res = None
        mock_ds.descriptions = None
        mock_ds.colorinterp = None
        mock_ds.compression = None
        mock_ds.interleaving = None
        mock_ds.block_shapes = None
        mock_ds.overviews.return_value = []
        mock_ds.tags.return_value = {}
        mock_rasterio.open.return_value = mock_ds

        reader = GDALFallbackReader(str(dummy))
        assert reader.get_dtype() == np.dtype('complex64')
        reader.close()

    @mock.patch('grdl.IO.generic.rasterio')
    def test_context_manager(self, mock_rasterio, tmp_path):
        dummy = tmp_path / 'test.tif'
        dummy.write_bytes(b'\x00' * 100)

        mock_ds = mock.MagicMock()
        mock_ds.driver = 'GTiff'
        mock_ds.height = 10
        mock_ds.width = 10
        mock_ds.count = 1
        mock_ds.dtypes = ('uint8',)
        mock_ds.crs = None
        mock_ds.nodata = None
        mock_ds.transform = None
        mock_ds.bounds = None
        mock_ds.res = None
        mock_ds.descriptions = None
        mock_ds.colorinterp = None
        mock_ds.compression = None
        mock_ds.interleaving = None
        mock_ds.block_shapes = None
        mock_ds.overviews.return_value = []
        mock_ds.tags.return_value = {}
        mock_rasterio.open.return_value = mock_ds

        with GDALFallbackReader(str(dummy)) as reader:
            assert reader.metadata.rows == 10

        mock_ds.close.assert_called_once()


# ===================================================================
# Test open_any
# ===================================================================

class TestOpenAny:
    """Tests for the open_any universal opener."""

    def test_file_not_found(self):
        with pytest.raises((FileNotFoundError, ValueError)):
            open_any('/nonexistent/path/file.tif')

    @mock.patch('grdl.IO.generic.importlib')
    @mock.patch('grdl.IO.generic.GDALFallbackReader')
    def test_tries_specialized_first(self, mock_fallback, mock_importlib,
                                     tmp_path):
        dummy = tmp_path / 'test.nitf'
        dummy.write_bytes(b'\x00' * 100)

        # Make open_sar succeed
        mock_reader = mock.MagicMock()
        mock_mod = mock.MagicMock()
        mock_mod.open_sar.return_value = mock_reader
        mock_importlib.import_module.return_value = mock_mod

        result = open_any(str(dummy))
        assert result is mock_reader
        mock_fallback.assert_not_called()

    @mock.patch('grdl.IO.generic.importlib')
    @mock.patch('grdl.IO.generic.GDALFallbackReader')
    def test_falls_back_to_gdal(self, mock_fallback_cls, mock_importlib,
                                tmp_path):
        dummy = tmp_path / 'test.dat'
        dummy.write_bytes(b'\x00' * 100)

        # All openers and readers fail
        mock_mod = mock.MagicMock()
        mock_mod.open_sar.side_effect = ValueError("not SAR")
        mock_mod.open_eo.side_effect = ValueError("not EO")
        mock_mod.open_ir.side_effect = ValueError("not IR")
        mock_mod.open_multispectral.side_effect = ValueError("not MS")
        mock_mod.GeoTIFFReader.side_effect = ValueError("not GeoTIFF")
        mock_mod.NITFReader.side_effect = ValueError("not NITF")
        mock_mod.HDF5Reader.side_effect = ValueError("not HDF5")
        mock_mod.JP2Reader.side_effect = ValueError("not JP2")
        mock_importlib.import_module.return_value = mock_mod

        # GDAL fallback succeeds
        mock_fallback = mock.MagicMock()
        mock_fallback.metadata = ImageMetadata(
            format='ENVI', rows=100, cols=100, dtype='float32',
        )
        mock_fallback.detected_modality = None
        mock_fallback_cls.return_value = mock_fallback

        result = open_any(str(dummy))
        assert result is mock_fallback

    @mock.patch('grdl.IO.generic.importlib')
    @mock.patch('grdl.IO.generic.InvasiveProbeReader')
    @mock.patch('grdl.IO.generic.GDALFallbackReader')
    def test_raises_when_nothing_works(self, mock_fallback_cls,
                                       mock_probe_cls,
                                       mock_importlib, tmp_path):
        dummy = tmp_path / 'test.xyz'
        dummy.write_bytes(b'\x00' * 100)

        # All openers fail
        mock_mod = mock.MagicMock()
        for attr in ['open_sar', 'open_eo', 'open_ir', 'open_multispectral',
                      'GeoTIFFReader', 'NITFReader', 'HDF5Reader',
                      'JP2Reader']:
            setattr(mock_mod, attr, mock.MagicMock(
                side_effect=ValueError("nope")
            ))
        mock_importlib.import_module.return_value = mock_mod

        # GDAL fallback also fails
        mock_fallback_cls.side_effect = ValueError("GDAL can't open")

        # Invasive probe also fails
        mock_probe_cls.side_effect = ValueError("probe failed")

        with pytest.raises(ValueError, match='No reader can open'):
            open_any(str(dummy))


# ===================================================================
# Test _retry_identified_reader
# ===================================================================

class TestRetryIdentifiedReader:
    """Tests for the retry-after-classification logic."""

    def test_nitf_with_nonstandard_extension(self, tmp_path):
        dummy = tmp_path / 'test.dat'
        dummy.write_bytes(b'\x00' * 100)

        fallback = mock.MagicMock()
        fallback.metadata = ImageMetadata(
            format='NITF', rows=100, cols=100, dtype='uint8',
            extras={'gdal_driver': 'NITF'},
        )

        # Mock the NITFReader import
        with mock.patch('grdl.IO.generic.importlib') as mock_imp:
            mock_reader = mock.MagicMock()
            mock_mod = mock.MagicMock()
            mock_mod.NITFReader.return_value = mock_reader
            mock_imp.import_module.return_value = mock_mod

            result = _retry_identified_reader(dummy, fallback)
            assert result is mock_reader

    def test_standard_nitf_extension_no_retry(self, tmp_path):
        dummy = tmp_path / 'test.nitf'
        dummy.write_bytes(b'\x00' * 100)

        fallback = mock.MagicMock()
        fallback.metadata = ImageMetadata(
            format='NITF', rows=100, cols=100, dtype='uint8',
            extras={'gdal_driver': 'NITF'},
        )

        result = _retry_identified_reader(dummy, fallback)
        assert result is None

    def test_geotiff_no_retry(self, tmp_path):
        dummy = tmp_path / 'test.tif'
        dummy.write_bytes(b'\x00' * 100)

        fallback = mock.MagicMock()
        fallback.metadata = ImageMetadata(
            format='GeoTIFF', rows=100, cols=100, dtype='float32',
            extras={'gdal_driver': 'GTiff'},
        )

        result = _retry_identified_reader(dummy, fallback)
        assert result is None

    def test_hdf5_with_nonstandard_extension(self, tmp_path):
        dummy = tmp_path / 'product.nc4'
        dummy.write_bytes(b'\x00' * 100)

        fallback = mock.MagicMock()
        fallback.metadata = ImageMetadata(
            format='HDF5', rows=100, cols=100, dtype='float32',
            extras={'gdal_driver': 'HDF5'},
        )

        with mock.patch('grdl.IO.generic.importlib') as mock_imp:
            mock_reader = mock.MagicMock()
            mock_mod = mock.MagicMock()
            mock_mod.HDF5Reader.return_value = mock_reader
            mock_imp.import_module.return_value = mock_mod

            result = _retry_identified_reader(dummy, fallback)
            assert result is mock_reader

    def test_retry_failure_returns_none(self, tmp_path):
        dummy = tmp_path / 'test.dat'
        dummy.write_bytes(b'\x00' * 100)

        fallback = mock.MagicMock()
        fallback.metadata = ImageMetadata(
            format='NITF', rows=100, cols=100, dtype='uint8',
            extras={'gdal_driver': 'NITF'},
        )

        with mock.patch('grdl.IO.generic.importlib') as mock_imp:
            mock_mod = mock.MagicMock()
            mock_mod.NITFReader.side_effect = ValueError("can't open")
            mock_imp.import_module.return_value = mock_mod

            result = _retry_identified_reader(dummy, fallback)
            assert result is None
