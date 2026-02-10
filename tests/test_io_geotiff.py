# -*- coding: utf-8 -*-
"""
GeoTIFF Reader Tests - Unit tests for GeoTIFFReader.

Uses synthetic GeoTIFF files created with rasterio for testing.

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
def single_band_tiff(tmp_path):
    """Create a single-band GeoTIFF for testing."""
    filepath = tmp_path / "test_single.tif"
    data = np.random.rand(100, 200).astype(np.float32)
    transform = from_bounds(0, 0, 1, 1, 200, 100)

    with rasterio.open(
        str(filepath), 'w', driver='GTiff',
        height=100, width=200, count=1,
        dtype='float32', crs='EPSG:4326',
        transform=transform,
    ) as ds:
        ds.write(data, 1)

    return filepath, data


@pytest.fixture
def multi_band_tiff(tmp_path):
    """Create a 3-band GeoTIFF for testing."""
    filepath = tmp_path / "test_multi.tif"
    data = np.random.rand(3, 80, 120).astype(np.float32)
    transform = from_bounds(10, 20, 11, 21, 120, 80)

    with rasterio.open(
        str(filepath), 'w', driver='GTiff',
        height=80, width=120, count=3,
        dtype='float32', crs='EPSG:4326',
        transform=transform,
    ) as ds:
        ds.write(data)

    return filepath, data


def test_metadata_single_band(single_band_tiff):
    """Metadata extracted correctly from single-band GeoTIFF."""
    from grdl.IO.geotiff import GeoTIFFReader

    filepath, _ = single_band_tiff
    with GeoTIFFReader(filepath) as reader:
        assert reader.metadata['format'] == 'GeoTIFF'
        assert reader.metadata['rows'] == 100
        assert reader.metadata['cols'] == 200
        assert reader.metadata['bands'] == 1
        assert reader.metadata['dtype'] == 'float32'
        assert 'EPSG:4326' in reader.metadata['crs']


def test_metadata_multi_band(multi_band_tiff):
    """Metadata extracted correctly from multi-band GeoTIFF."""
    from grdl.IO.geotiff import GeoTIFFReader

    filepath, _ = multi_band_tiff
    with GeoTIFFReader(filepath) as reader:
        assert reader.metadata['bands'] == 3
        assert reader.metadata['rows'] == 80
        assert reader.metadata['cols'] == 120


def test_get_shape_single_band(single_band_tiff):
    """get_shape returns (rows, cols) for single band."""
    from grdl.IO.geotiff import GeoTIFFReader

    filepath, _ = single_band_tiff
    with GeoTIFFReader(filepath) as reader:
        assert reader.get_shape() == (100, 200)


def test_get_shape_multi_band(multi_band_tiff):
    """get_shape returns (rows, cols, bands) for multi-band."""
    from grdl.IO.geotiff import GeoTIFFReader

    filepath, _ = multi_band_tiff
    with GeoTIFFReader(filepath) as reader:
        assert reader.get_shape() == (80, 120, 3)


def test_get_dtype(single_band_tiff):
    """get_dtype returns correct numpy dtype."""
    from grdl.IO.geotiff import GeoTIFFReader

    filepath, _ = single_band_tiff
    with GeoTIFFReader(filepath) as reader:
        assert reader.get_dtype() == np.dtype('float32')


def test_read_chip_single_band(single_band_tiff):
    """read_chip returns correct data for single band."""
    from grdl.IO.geotiff import GeoTIFFReader

    filepath, original = single_band_tiff
    with GeoTIFFReader(filepath) as reader:
        chip = reader.read_chip(10, 30, 20, 50)
        assert chip.shape == (20, 30)
        np.testing.assert_array_almost_equal(
            chip, original[10:30, 20:50]
        )


def test_read_chip_multi_band(multi_band_tiff):
    """read_chip returns correct data for multi-band."""
    from grdl.IO.geotiff import GeoTIFFReader

    filepath, original = multi_band_tiff
    with GeoTIFFReader(filepath) as reader:
        chip = reader.read_chip(0, 40, 0, 60)
        assert chip.shape == (3, 40, 60)
        np.testing.assert_array_almost_equal(
            chip, original[:, :40, :60]
        )


def test_read_chip_band_selection(multi_band_tiff):
    """read_chip with specific bands returns correct data."""
    from grdl.IO.geotiff import GeoTIFFReader

    filepath, original = multi_band_tiff
    with GeoTIFFReader(filepath) as reader:
        chip = reader.read_chip(0, 40, 0, 60, bands=[1])
        assert chip.shape == (40, 60)
        np.testing.assert_array_almost_equal(
            chip, original[1, :40, :60]
        )


def test_read_full_single_band(single_band_tiff):
    """read_full returns entire image for single band."""
    from grdl.IO.geotiff import GeoTIFFReader

    filepath, original = single_band_tiff
    with GeoTIFFReader(filepath) as reader:
        full = reader.read_full()
        assert full.shape == (100, 200)
        np.testing.assert_array_almost_equal(full, original)


def test_read_chip_negative_start_raises(single_band_tiff):
    """Negative start indices raise ValueError."""
    from grdl.IO.geotiff import GeoTIFFReader

    filepath, _ = single_band_tiff
    with GeoTIFFReader(filepath) as reader:
        with pytest.raises(ValueError, match="non-negative"):
            reader.read_chip(-1, 10, 0, 10)


def test_read_chip_out_of_bounds_raises(single_band_tiff):
    """Out-of-bounds end indices raise ValueError."""
    from grdl.IO.geotiff import GeoTIFFReader

    filepath, _ = single_band_tiff
    with GeoTIFFReader(filepath) as reader:
        with pytest.raises(ValueError, match="exceed"):
            reader.read_chip(0, 200, 0, 10)


def test_geolocation(single_band_tiff):
    """get_geolocation returns CRS and transform info."""
    from grdl.IO.geotiff import GeoTIFFReader

    filepath, _ = single_band_tiff
    with GeoTIFFReader(filepath) as reader:
        geo = reader.get_geolocation()
        assert geo is not None
        assert 'crs' in geo
        assert 'transform' in geo
        assert 'bounds' in geo
        assert 'resolution' in geo


def test_file_not_found():
    """FileNotFoundError for non-existent file."""
    from grdl.IO.geotiff import GeoTIFFReader

    with pytest.raises(FileNotFoundError):
        GeoTIFFReader('/nonexistent/file.tif')


def test_context_manager(single_band_tiff):
    """Context manager opens and closes cleanly."""
    from grdl.IO.geotiff import GeoTIFFReader

    filepath, _ = single_band_tiff
    with GeoTIFFReader(filepath) as reader:
        assert reader.metadata['rows'] == 100
    # After context exit, dataset should be closed


def test_open_image_geotiff(single_band_tiff):
    """open_image auto-detects GeoTIFF files."""
    from grdl.IO import open_image

    filepath, _ = single_band_tiff
    with open_image(filepath) as reader:
        assert reader.metadata['format'] == 'GeoTIFF'
