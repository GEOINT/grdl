# -*- coding: utf-8 -*-
"""
Writer Registry Tests - Unit tests for get_writer() factory function.

Tests that get_writer() returns the correct writer class for each
supported format string and raises ValueError for unknown formats.

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

from grdl.IO.base import ImageWriter


class TestGetWriter:
    """Tests for the get_writer() factory function."""

    def test_get_writer_numpy(self, tmp_path):
        """get_writer('numpy') returns NumpyWriter instance."""
        from grdl.IO import get_writer
        from grdl.IO.numpy_io import NumpyWriter

        writer = get_writer('numpy', tmp_path / 'test.npy')
        assert isinstance(writer, NumpyWriter)
        assert isinstance(writer, ImageWriter)

    def test_get_writer_png(self, tmp_path):
        """get_writer('png') returns PngWriter instance."""
        pytest.importorskip('PIL')
        from grdl.IO import get_writer
        from grdl.IO.png import PngWriter

        writer = get_writer('png', tmp_path / 'test.png')
        assert isinstance(writer, PngWriter)
        assert isinstance(writer, ImageWriter)

    def test_get_writer_geotiff(self, tmp_path):
        """get_writer('geotiff') returns GeoTIFFWriter instance."""
        pytest.importorskip('rasterio')
        from grdl.IO import get_writer
        from grdl.IO.geotiff import GeoTIFFWriter

        writer = get_writer('geotiff', tmp_path / 'test.tif')
        assert isinstance(writer, GeoTIFFWriter)
        assert isinstance(writer, ImageWriter)

    def test_get_writer_hdf5(self, tmp_path):
        """get_writer('hdf5') returns HDF5Writer instance."""
        pytest.importorskip('h5py')
        from grdl.IO import get_writer
        from grdl.IO.hdf5 import HDF5Writer

        writer = get_writer('hdf5', tmp_path / 'test.h5')
        assert isinstance(writer, HDF5Writer)
        assert isinstance(writer, ImageWriter)

    def test_get_writer_case_insensitive(self, tmp_path):
        """Format string is case-insensitive."""
        from grdl.IO import get_writer

        writer = get_writer('NUMPY', tmp_path / 'test.npy')
        assert isinstance(writer, ImageWriter)

    def test_get_writer_unknown_format_raises(self, tmp_path):
        """Unknown format raises ValueError."""
        from grdl.IO import get_writer

        with pytest.raises(ValueError, match="Unknown writer format"):
            get_writer('bmp', tmp_path / 'test.bmp')

    def test_get_writer_nitf(self, tmp_path):
        """get_writer('nitf') returns NITFWriter instance."""
        pytest.importorskip('rasterio')
        from grdl.IO import get_writer
        from grdl.IO.nitf import NITFWriter

        writer = get_writer('nitf', tmp_path / 'test.nitf')
        assert isinstance(writer, NITFWriter)
        assert isinstance(writer, ImageWriter)

    def test_get_writer_with_metadata(self, tmp_path):
        """Metadata is passed through to writer."""
        from grdl.IO import get_writer
        from grdl.IO.models import ImageMetadata

        meta = ImageMetadata(
            format='numpy', rows=0, cols=0, dtype='float32',
            extras={'source': 'pipeline'},
        )
        writer = get_writer(
            'numpy', tmp_path / 'test.npy',
            metadata=meta
        )
        assert writer.metadata['source'] == 'pipeline'
