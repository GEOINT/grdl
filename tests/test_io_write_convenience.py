# -*- coding: utf-8 -*-
"""
Write Convenience Function Tests - Unit tests for grdl.IO.write().

Tests auto-detection from file extension, explicit format override,
error handling, and round-trip verification for all supported formats.

Dependencies
------------
pytest

Author
------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-11

Modified
--------
2026-02-11
"""

import pytest
import numpy as np


class TestWriteAutoDetection:
    """Tests for format auto-detection from file extension."""

    def test_autodetect_tif(self, tmp_path):
        """'.tif' extension auto-detects to geotiff format."""
        pytest.importorskip('rasterio')
        from grdl.IO import write
        from grdl.IO.geotiff import GeoTIFFReader

        data = np.random.rand(16, 16).astype(np.float32)
        path = tmp_path / "test.tif"
        write(data, path)

        with GeoTIFFReader(path) as reader:
            result = reader.read_full()
            np.testing.assert_array_almost_equal(result, data)

    def test_autodetect_tiff(self, tmp_path):
        """'.tiff' extension auto-detects to geotiff format."""
        pytest.importorskip('rasterio')
        from grdl.IO import write
        from grdl.IO.geotiff import GeoTIFFReader

        data = np.random.rand(16, 16).astype(np.float32)
        path = tmp_path / "test.tiff"
        write(data, path)

        with GeoTIFFReader(path) as reader:
            result = reader.read_full()
            np.testing.assert_array_almost_equal(result, data)

    def test_autodetect_npy(self, tmp_path):
        """'.npy' extension auto-detects to numpy format."""
        from grdl.IO import write

        data = np.random.rand(16, 16).astype(np.float32)
        path = tmp_path / "test.npy"
        write(data, path)

        result = np.load(str(path))
        np.testing.assert_array_equal(result, data)

    def test_autodetect_png(self, tmp_path):
        """'.png' extension auto-detects to png format."""
        pytest.importorskip('PIL')
        from grdl.IO import write
        from PIL import Image

        data = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        path = tmp_path / "test.png"
        write(data, path)

        result = np.array(Image.open(str(path)))
        np.testing.assert_array_equal(result, data)

    def test_autodetect_nitf(self, tmp_path):
        """'.nitf' extension auto-detects to nitf format."""
        pytest.importorskip('rasterio')
        from grdl.IO import write
        from grdl.IO.nitf import NITFReader

        data = np.random.rand(16, 16).astype(np.float32)
        path = tmp_path / "test.nitf"
        write(data, path)

        with NITFReader(path) as reader:
            result = reader.read_full()
            np.testing.assert_array_almost_equal(result, data)

    def test_autodetect_ntf(self, tmp_path):
        """'.ntf' extension auto-detects to nitf format."""
        pytest.importorskip('rasterio')
        from grdl.IO import write
        from grdl.IO.nitf import NITFReader

        data = np.random.rand(16, 16).astype(np.float32)
        path = tmp_path / "test.ntf"
        write(data, path)

        with NITFReader(path) as reader:
            result = reader.read_full()
            np.testing.assert_array_almost_equal(result, data)

    def test_autodetect_h5(self, tmp_path):
        """'.h5' extension auto-detects to hdf5 format."""
        pytest.importorskip('h5py')
        from grdl.IO import write
        import h5py

        data = np.random.rand(16, 16).astype(np.float32)
        path = tmp_path / "test.h5"
        write(data, path)

        with h5py.File(str(path), 'r') as f:
            result = f['data'][:]
            np.testing.assert_array_almost_equal(result, data)


class TestWriteExplicitFormat:
    """Tests for explicit format override."""

    def test_explicit_format_overrides_extension(self, tmp_path):
        """Explicit format= overrides extension detection."""
        from grdl.IO import write

        data = np.random.rand(8, 8).astype(np.float32)
        # Use .npy extension so np.save doesn't append another .npy
        path = tmp_path / "test.npy"
        write(data, path, format='numpy')

        result = np.load(str(path))
        np.testing.assert_array_equal(result, data)

    def test_unknown_extension_with_format_succeeds(self, tmp_path):
        """Unknown extension works when format= is provided."""
        pytest.importorskip('rasterio')
        from grdl.IO import write
        from grdl.IO.geotiff import GeoTIFFReader

        data = np.random.rand(8, 8).astype(np.float32)
        path = tmp_path / "output.bin"
        write(data, path, format='geotiff')

        with GeoTIFFReader(path) as reader:
            result = reader.read_full()
            np.testing.assert_array_almost_equal(result, data)


class TestWriteErrorHandling:
    """Tests for error cases."""

    def test_unknown_extension_no_format_raises(self, tmp_path):
        """Unknown extension without format= raises ValueError."""
        from grdl.IO import write

        data = np.zeros((8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="Cannot determine writer format"):
            write(data, tmp_path / "output.xyz")

    def test_unknown_format_string_raises(self, tmp_path):
        """Unknown format string raises ValueError."""
        from grdl.IO import write

        data = np.zeros((8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown writer format"):
            write(data, tmp_path / "output.dat", format='bmp')


class TestWriteMetadataPassthrough:
    """Tests for metadata and geolocation passthrough."""

    def test_metadata_included_in_numpy_sidecar(self, tmp_path):
        """Metadata is written to the numpy sidecar JSON."""
        import json
        from grdl.IO import write

        data = np.zeros((8, 8), dtype=np.float32)
        path = tmp_path / "meta.npy"
        write(data, path, metadata={'source': 'pipeline_v2'})

        sidecar = json.loads((tmp_path / "meta.npy.json").read_text())
        assert sidecar['source'] == 'pipeline_v2'

    def test_geolocation_persists_in_geotiff(self, tmp_path):
        """CRS and transform persist through write/read."""
        pytest.importorskip('rasterio')
        from rasterio.transform import from_bounds
        from grdl.IO import write
        from grdl.IO.geotiff import GeoTIFFReader

        data = np.random.rand(32, 48).astype(np.float32)
        path = tmp_path / "geo.tif"
        transform = from_bounds(10.0, 20.0, 11.0, 21.0, 48, 32)

        write(data, path, geolocation={
            'crs': 'EPSG:4326',
            'transform': transform,
        })

        with GeoTIFFReader(path) as reader:
            assert 'EPSG:4326' in reader.metadata['crs']
            result = reader.read_full()
            np.testing.assert_array_almost_equal(result, data)
