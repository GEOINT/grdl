# -*- coding: utf-8 -*-
"""
HDF5 Writer Tests - Unit tests for HDF5Writer.

Tests round-trip writes, multiple datasets, compression, and
metadata attributes.

Dependencies
------------
pytest
h5py

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

import numpy as np
import pytest

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

pytestmark = pytest.mark.skipif(
    not _HAS_H5PY, reason="h5py not installed"
)


class TestHDF5WriterBasic:
    """Basic write and round-trip tests."""

    def test_write_float32_roundtrip(self, tmp_path):
        """Write float32 array, read back, verify equality."""
        from grdl.IO.hdf5 import HDF5Writer

        data = np.random.rand(32, 48).astype(np.float32)
        filepath = tmp_path / "basic.h5"

        with HDF5Writer(filepath) as writer:
            writer.write(data)

        with h5py.File(str(filepath), 'r') as f:
            result = f['data'][()]
            np.testing.assert_array_equal(result, data)

    def test_write_3d_roundtrip(self, tmp_path):
        """Write 3D multi-band array, read back, verify equality."""
        from grdl.IO.hdf5 import HDF5Writer

        data = np.random.rand(3, 16, 24).astype(np.float64)
        filepath = tmp_path / "multi.h5"

        with HDF5Writer(filepath) as writer:
            writer.write(data)

        with h5py.File(str(filepath), 'r') as f:
            result = f['data'][()]
            np.testing.assert_array_equal(result, data)

    def test_custom_dataset_name(self, tmp_path):
        """Custom dataset name from metadata is respected."""
        from grdl.IO.hdf5 import HDF5Writer
        from grdl.IO.models import ImageMetadata

        data = np.zeros((8, 8), dtype=np.float32)
        filepath = tmp_path / "named.h5"
        meta = ImageMetadata(
            format='HDF5', rows=8, cols=8, dtype='float32',
            extras={'dataset_name': 'imagery'},
        )

        with HDF5Writer(filepath, metadata=meta) as w:
            w.write(data)

        with h5py.File(str(filepath), 'r') as f:
            assert 'imagery' in f
            np.testing.assert_array_equal(f['imagery'][()], data)


class TestHDF5WriterCompression:
    """Compression tests."""

    def test_gzip_compression(self, tmp_path):
        """gzip compression is applied and data survives round-trip."""
        from grdl.IO.hdf5 import HDF5Writer
        from grdl.IO.models import ImageMetadata

        data = np.random.rand(32, 32).astype(np.float32)
        filepath = tmp_path / "gzip.h5"
        meta = ImageMetadata(
            format='HDF5', rows=32, cols=32, dtype='float32',
            extras={'compression': 'gzip'},
        )

        with HDF5Writer(filepath, metadata=meta) as w:
            w.write(data)

        with h5py.File(str(filepath), 'r') as f:
            assert f['data'].compression == 'gzip'
            np.testing.assert_array_equal(f['data'][()], data)

    def test_lzf_compression(self, tmp_path):
        """lzf compression is applied and data survives round-trip."""
        from grdl.IO.hdf5 import HDF5Writer
        from grdl.IO.models import ImageMetadata

        data = np.random.rand(32, 32).astype(np.float32)
        filepath = tmp_path / "lzf.h5"
        meta = ImageMetadata(
            format='HDF5', rows=32, cols=32, dtype='float32',
            extras={'compression': 'lzf'},
        )

        with HDF5Writer(filepath, metadata=meta) as w:
            w.write(data)

        with h5py.File(str(filepath), 'r') as f:
            assert f['data'].compression == 'lzf'
            np.testing.assert_array_equal(f['data'][()], data)


class TestHDF5WriterAttributes:
    """Metadata attribute tests."""

    def test_attributes_written(self, tmp_path):
        """Custom attributes attached to dataset."""
        from grdl.IO.hdf5 import HDF5Writer
        from grdl.IO.models import ImageMetadata

        data = np.zeros((8, 8), dtype=np.float32)
        filepath = tmp_path / "attrs.h5"

        attrs = {'description': 'test data', 'version': 1}
        meta = ImageMetadata(
            format='HDF5', rows=8, cols=8, dtype='float32',
            extras={'attributes': attrs},
        )
        with HDF5Writer(filepath, metadata=meta) as w:
            w.write(data)

        with h5py.File(str(filepath), 'r') as f:
            assert f['data'].attrs['description'] == 'test data'
            assert f['data'].attrs['version'] == 1

    def test_geolocation_as_attributes(self, tmp_path):
        """Geolocation info stored as prefixed dataset attributes."""
        from grdl.IO.hdf5 import HDF5Writer

        data = np.zeros((8, 8), dtype=np.float32)
        filepath = tmp_path / "geo.h5"

        with HDF5Writer(filepath) as w:
            w.write(data, geolocation={'crs': 'EPSG:4326'})

        with h5py.File(str(filepath), 'r') as f:
            assert f['data'].attrs['geo_crs'] == 'EPSG:4326'


class TestHDF5WriterMultiDataset:
    """Multiple dataset tests."""

    def test_write_dataset_additional(self, tmp_path):
        """Write primary + additional named datasets."""
        from grdl.IO.hdf5 import HDF5Writer

        primary = np.random.rand(16, 16).astype(np.float32)
        mask = np.ones((16, 16), dtype=np.uint8)
        filepath = tmp_path / "multi_ds.h5"

        with HDF5Writer(filepath) as w:
            w.write(primary)
            w.write_dataset('mask', mask, attributes={'type': 'binary'})

        with h5py.File(str(filepath), 'r') as f:
            np.testing.assert_array_equal(f['data'][()], primary)
            np.testing.assert_array_equal(f['mask'][()], mask)
            assert f['mask'].attrs['type'] == 'binary'


class TestHDF5WriterEdgeCases:
    """Edge cases and error handling."""

    def test_implements_image_writer(self, tmp_path):
        """HDF5Writer is an ImageWriter subclass."""
        from grdl.IO.hdf5 import HDF5Writer
        from grdl.IO.base import ImageWriter

        writer = HDF5Writer(tmp_path / "test.h5")
        assert isinstance(writer, ImageWriter)

    def test_write_chip_without_file_raises(self, tmp_path):
        """write_chip before write() raises IOError."""
        from grdl.IO.hdf5 import HDF5Writer

        filepath = tmp_path / "no_file.h5"
        with HDF5Writer(filepath) as w:
            with pytest.raises(IOError, match="not open"):
                w.write_chip(np.zeros((4, 4)), 0, 0)

    def test_context_manager_closes(self, tmp_path):
        """Context manager closes file handle on exit."""
        from grdl.IO.hdf5 import HDF5Writer

        filepath = tmp_path / "ctx.h5"
        writer = HDF5Writer(filepath)
        with writer:
            writer.write(np.zeros((8, 8), dtype=np.float32))
        assert writer._file is None
