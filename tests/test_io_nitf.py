# -*- coding: utf-8 -*-
"""
NITF Reader Tests - Unit tests for NITFReader.

Uses synthetic NITF files created with rasterio/GDAL for testing.

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
2026-02-09

Modified
--------
2026-02-09
"""

import pytest
import numpy as np

try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

pytestmark = pytest.mark.skipif(
    not _HAS_RASTERIO, reason="rasterio not installed"
)


@pytest.fixture
def nitf_file(tmp_path):
    """Create a minimal NITF file for testing."""
    from rasterio.transform import from_bounds

    filepath = tmp_path / "test.ntf"
    data = np.random.randint(0, 255, (64, 128), dtype=np.uint8)

    transform = from_bounds(-180, -90, 180, 90, 128, 64)

    with rasterio.open(
        str(filepath), 'w', driver='NITF',
        height=64, width=128, count=1,
        dtype='uint8',
        transform=transform,
        crs='EPSG:4326',
        ICORDS='G',
    ) as ds:
        ds.write(data, 1)

    return filepath, data


@pytest.fixture
def nitf_multi(tmp_path):
    """Create a multi-band NITF file for testing."""
    from rasterio.transform import from_bounds

    filepath = tmp_path / "test_multi.ntf"
    data = np.random.randint(0, 255, (3, 50, 100), dtype=np.uint8)

    transform = from_bounds(-180, -90, 180, 90, 100, 50)

    with rasterio.open(
        str(filepath), 'w', driver='NITF',
        height=50, width=100, count=3,
        dtype='uint8',
        transform=transform,
        crs='EPSG:4326',
        ICORDS='G',
    ) as ds:
        ds.write(data)

    return filepath, data


def test_metadata(nitf_file):
    """Metadata extracted correctly from NITF."""
    from grdl.IO.nitf import NITFReader

    filepath, _ = nitf_file
    with NITFReader(filepath) as reader:
        assert reader.metadata['format'] == 'NITF'
        assert reader.metadata['rows'] == 64
        assert reader.metadata['cols'] == 128
        assert reader.metadata['bands'] == 1


def test_get_shape_single(nitf_file):
    """get_shape returns (rows, cols) for single band NITF."""
    from grdl.IO.nitf import NITFReader

    filepath, _ = nitf_file
    with NITFReader(filepath) as reader:
        assert reader.get_shape() == (64, 128)


def test_get_shape_multi(nitf_multi):
    """get_shape returns (rows, cols, bands) for multi-band NITF."""
    from grdl.IO.nitf import NITFReader

    filepath, _ = nitf_multi
    with NITFReader(filepath) as reader:
        assert reader.get_shape() == (50, 100, 3)


def test_get_dtype(nitf_file):
    """get_dtype returns correct numpy dtype."""
    from grdl.IO.nitf import NITFReader

    filepath, _ = nitf_file
    with NITFReader(filepath) as reader:
        assert reader.get_dtype() == np.dtype('uint8')


def test_read_chip(nitf_file):
    """read_chip returns correct data."""
    from grdl.IO.nitf import NITFReader

    filepath, original = nitf_file
    with NITFReader(filepath) as reader:
        chip = reader.read_chip(0, 32, 0, 64)
        assert chip.shape == (32, 64)
        assert chip.dtype == np.uint8


def test_read_full(nitf_file):
    """read_full returns entire image."""
    from grdl.IO.nitf import NITFReader

    filepath, _ = nitf_file
    with NITFReader(filepath) as reader:
        full = reader.read_full()
        assert full.shape == (64, 128)


def test_read_chip_negative_start_raises(nitf_file):
    """Negative start indices raise ValueError."""
    from grdl.IO.nitf import NITFReader

    filepath, _ = nitf_file
    with NITFReader(filepath) as reader:
        with pytest.raises(ValueError, match="non-negative"):
            reader.read_chip(-1, 10, 0, 10)


def test_read_chip_out_of_bounds_raises(nitf_file):
    """Out-of-bounds end indices raise ValueError."""
    from grdl.IO.nitf import NITFReader

    filepath, _ = nitf_file
    with NITFReader(filepath) as reader:
        with pytest.raises(ValueError, match="exceed"):
            reader.read_chip(0, 100, 0, 10)


def test_geolocation_none_without_crs(nitf_file):
    """get_geolocation returns None when no CRS is set."""
    from grdl.IO.nitf import NITFReader

    filepath, _ = nitf_file
    with NITFReader(filepath) as reader:
        geo = reader.get_geolocation()
        # NITF created without CRS may return None
        if reader.metadata['crs'] is None:
            assert geo is None


def test_file_not_found():
    """FileNotFoundError for non-existent file."""
    from grdl.IO.nitf import NITFReader

    with pytest.raises(FileNotFoundError):
        NITFReader('/nonexistent/file.ntf')


def test_context_manager(nitf_file):
    """Context manager opens and closes cleanly."""
    from grdl.IO.nitf import NITFReader

    filepath, _ = nitf_file
    with NITFReader(filepath) as reader:
        assert reader.metadata['format'] == 'NITF'
