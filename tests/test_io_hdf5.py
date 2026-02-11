# -*- coding: utf-8 -*-
"""
HDF5 Reader Tests - Unit tests for HDF5Reader.

Uses synthetic HDF5 files created with h5py for testing.

Dependencies
------------
pytest
h5py

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
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

pytestmark = pytest.mark.skipif(
    not _HAS_H5PY, reason="h5py not installed"
)


@pytest.fixture
def single_band_h5(tmp_path):
    """Create an HDF5 file with a single 2D dataset."""
    filepath = tmp_path / "test_single.h5"
    data = np.random.rand(100, 200).astype(np.float32)

    with h5py.File(str(filepath), "w") as f:
        f.attrs["producer"] = "test"
        ds = f.create_dataset("image", data=data)
        ds.attrs["units"] = "reflectance"

    return filepath, data


@pytest.fixture
def multi_band_h5(tmp_path):
    """Create an HDF5 file with a 3D dataset (bands, rows, cols)."""
    filepath = tmp_path / "test_multi.h5"
    data = np.random.rand(3, 80, 120).astype(np.float32)

    with h5py.File(str(filepath), "w") as f:
        f.create_dataset("bands", data=data)

    return filepath, data


@pytest.fixture
def nested_h5(tmp_path):
    """Create an HDF5 file with datasets in nested groups."""
    filepath = tmp_path / "test_nested.h5"
    data_a = np.random.rand(50, 60).astype(np.float32)
    data_b = np.random.rand(4, 50, 60).astype(np.float64)

    with h5py.File(str(filepath), "w") as f:
        grp = f.create_group("HDFEOS/GRIDS/VNP_Grid")
        grp.create_dataset("NDVI", data=data_a)
        grp.create_dataset("Bands", data=data_b)
        # Also add a 1D dataset that should be skipped by auto-detect
        f.create_dataset("timestamps", data=np.arange(100))

    return filepath, data_a, data_b


@pytest.fixture
def geo_h5(tmp_path):
    """Create an HDF5 file with geolocation attributes."""
    filepath = tmp_path / "test_geo.h5"
    data = np.random.rand(64, 64).astype(np.float32)

    with h5py.File(str(filepath), "w") as f:
        f.attrs["crs"] = "EPSG:4326"
        f.attrs["geospatial_lat_min"] = -31.5
        f.attrs["geospatial_lat_max"] = -30.5
        f.attrs["geospatial_lon_min"] = 115.5
        f.attrs["geospatial_lon_max"] = 116.8
        ds = f.create_dataset("data", data=data)
        ds.attrs["scale_factor"] = 0.0001

    return filepath, data


# -- Metadata tests ---------------------------------------------------------

def test_metadata_single_band(single_band_h5):
    """Metadata extracted correctly from single 2D dataset."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, _ = single_band_h5
    with HDF5Reader(filepath) as reader:
        assert reader.metadata['format'] == 'HDF5'
        assert reader.metadata['rows'] == 100
        assert reader.metadata['cols'] == 200
        assert reader.metadata['bands'] == 1
        assert reader.metadata['dtype'] == 'float32'
        assert reader.metadata['dataset_path'] == '/image'


def test_metadata_multi_band(multi_band_h5):
    """Metadata extracted correctly from 3D dataset."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, _ = multi_band_h5
    with HDF5Reader(filepath) as reader:
        assert reader.metadata['bands'] == 3
        assert reader.metadata['rows'] == 80
        assert reader.metadata['cols'] == 120


def test_metadata_includes_attributes(single_band_h5):
    """HDF5 attributes propagated to metadata extras."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, _ = single_band_h5
    with HDF5Reader(filepath) as reader:
        assert reader.metadata['units'] == 'reflectance'
        assert reader.metadata['file_producer'] == 'test'


# -- Auto-detection tests ---------------------------------------------------

def test_auto_detect_selects_first_2d(nested_h5):
    """Auto-detect picks a 2D+ dataset, skipping 1D."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, data_a, _ = nested_h5
    with HDF5Reader(filepath) as reader:
        # Should select one of the 2D+ datasets, not the 1D timestamps
        assert reader.dataset_path in (
            '/HDFEOS/GRIDS/VNP_Grid/NDVI',
            '/HDFEOS/GRIDS/VNP_Grid/Bands',
        )
        assert reader.metadata['rows'] == 50
        assert reader.metadata['cols'] == 60


def test_explicit_dataset_path(nested_h5):
    """Explicit dataset_path selects the requested dataset."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, _, data_b = nested_h5
    with HDF5Reader(filepath, dataset_path='/HDFEOS/GRIDS/VNP_Grid/Bands') as reader:
        assert reader.metadata['bands'] == 4
        assert reader.metadata['rows'] == 50
        assert reader.metadata['cols'] == 60


def test_invalid_dataset_path_raises(single_band_h5):
    """Invalid dataset_path raises ValueError with available paths."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, _ = single_band_h5
    with pytest.raises(ValueError, match="not found"):
        HDF5Reader(filepath, dataset_path='/nonexistent')


def test_no_suitable_dataset_raises(tmp_path):
    """File with only 1D data raises ValueError."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath = tmp_path / "only_1d.h5"
    with h5py.File(str(filepath), "w") as f:
        f.create_dataset("vector", data=np.arange(100))

    with pytest.raises(ValueError, match="No 2D"):
        HDF5Reader(filepath)


# -- Shape and dtype tests --------------------------------------------------

def test_get_shape_single_band(single_band_h5):
    """get_shape returns (rows, cols) for single band."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, _ = single_band_h5
    with HDF5Reader(filepath) as reader:
        assert reader.get_shape() == (100, 200)


def test_get_shape_multi_band(multi_band_h5):
    """get_shape returns (rows, cols, bands) for multi-band."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, _ = multi_band_h5
    with HDF5Reader(filepath) as reader:
        assert reader.get_shape() == (80, 120, 3)


def test_get_dtype(single_band_h5):
    """get_dtype returns correct numpy dtype."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, _ = single_band_h5
    with HDF5Reader(filepath) as reader:
        assert reader.get_dtype() == np.dtype('float32')


# -- Read tests --------------------------------------------------------------

def test_read_chip_single_band(single_band_h5):
    """read_chip returns correct data for single 2D dataset."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, original = single_band_h5
    with HDF5Reader(filepath) as reader:
        chip = reader.read_chip(10, 30, 20, 50)
        assert chip.shape == (20, 30)
        np.testing.assert_array_almost_equal(
            chip, original[10:30, 20:50]
        )


def test_read_chip_multi_band(multi_band_h5):
    """read_chip returns correct data for 3D dataset."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, original = multi_band_h5
    with HDF5Reader(filepath) as reader:
        chip = reader.read_chip(0, 40, 0, 60)
        assert chip.shape == (3, 40, 60)
        np.testing.assert_array_almost_equal(
            chip, original[:, :40, :60]
        )


def test_read_chip_band_selection(multi_band_h5):
    """read_chip with specific bands returns correct data."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, original = multi_band_h5
    with HDF5Reader(filepath) as reader:
        chip = reader.read_chip(0, 40, 0, 60, bands=[1])
        assert chip.shape == (40, 60)
        np.testing.assert_array_almost_equal(
            chip, original[1, :40, :60]
        )


def test_read_full_single_band(single_band_h5):
    """read_full returns entire image for single band."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, original = single_band_h5
    with HDF5Reader(filepath) as reader:
        full = reader.read_full()
        assert full.shape == (100, 200)
        np.testing.assert_array_almost_equal(full, original)


def test_read_full_multi_band(multi_band_h5):
    """read_full returns all bands."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, original = multi_band_h5
    with HDF5Reader(filepath) as reader:
        full = reader.read_full()
        assert full.shape == (3, 80, 120)
        np.testing.assert_array_almost_equal(full, original)


def test_read_full_band_selection(multi_band_h5):
    """read_full with band selection returns correct subset."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, original = multi_band_h5
    with HDF5Reader(filepath) as reader:
        data = reader.read_full(bands=[0, 2])
        assert data.shape == (2, 80, 120)
        np.testing.assert_array_almost_equal(data[0], original[0])
        np.testing.assert_array_almost_equal(data[1], original[2])


# -- Validation tests --------------------------------------------------------

def test_read_chip_negative_start_raises(single_band_h5):
    """Negative start indices raise ValueError."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, _ = single_band_h5
    with HDF5Reader(filepath) as reader:
        with pytest.raises(ValueError, match="non-negative"):
            reader.read_chip(-1, 10, 0, 10)


def test_read_chip_out_of_bounds_raises(single_band_h5):
    """Out-of-bounds end indices raise ValueError."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, _ = single_band_h5
    with HDF5Reader(filepath) as reader:
        with pytest.raises(ValueError, match="exceed"):
            reader.read_chip(0, 200, 0, 10)


def test_file_not_found():
    """FileNotFoundError for non-existent file."""
    from grdl.IO.hdf5 import HDF5Reader

    with pytest.raises(FileNotFoundError):
        HDF5Reader('/nonexistent/file.h5')


# -- Context manager and cleanup tests ---------------------------------------

def test_context_manager(single_band_h5):
    """Context manager opens and closes cleanly."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, _ = single_band_h5
    with HDF5Reader(filepath) as reader:
        assert reader.metadata['rows'] == 100


def test_close_idempotent(single_band_h5):
    """Calling close() multiple times does not raise."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, _ = single_band_h5
    reader = HDF5Reader(filepath)
    reader.close()
    reader.close()


# -- list_datasets utility tests ---------------------------------------------

def test_list_datasets(nested_h5):
    """list_datasets returns all 2D+ datasets with paths and shapes."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, _, _ = nested_h5
    datasets = HDF5Reader.list_datasets(filepath)
    paths = [p for p, _, _ in datasets]
    assert '/HDFEOS/GRIDS/VNP_Grid/NDVI' in paths
    assert '/HDFEOS/GRIDS/VNP_Grid/Bands' in paths
    # 1D timestamps should NOT appear
    assert '/timestamps' not in paths


def test_list_datasets_min_ndim(nested_h5):
    """list_datasets with min_ndim=1 includes 1D arrays."""
    from grdl.IO.hdf5 import HDF5Reader

    filepath, _, _ = nested_h5
    datasets = HDF5Reader.list_datasets(filepath, min_ndim=1)
    paths = [p for p, _, _ in datasets]
    assert '/timestamps' in paths


def test_list_datasets_file_not_found():
    """list_datasets raises FileNotFoundError for missing file."""
    from grdl.IO.hdf5 import HDF5Reader

    with pytest.raises(FileNotFoundError):
        HDF5Reader.list_datasets('/nonexistent/file.h5')


# -- ImageReader ABC contract ------------------------------------------------

def test_is_image_reader_subclass():
    """HDF5Reader is a subclass of ImageReader."""
    from grdl.IO.base import ImageReader
    from grdl.IO.hdf5 import HDF5Reader
    assert issubclass(HDF5Reader, ImageReader)
