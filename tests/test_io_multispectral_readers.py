# -*- coding: utf-8 -*-
"""
Multispectral Reader Tests - Unit tests for VIIRSReader.

Uses synthetic HDF5 files created with h5py for testing VIIRS-specific
metadata extraction.

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
def viirs_nightlights_h5(tmp_path):
    """Create a synthetic VIIRS VNP46A1 nighttime lights HDF5 file."""
    filepath = tmp_path / "VNP46A1.A2024001.h09v05.002.h5"
    data = np.random.rand(100, 200).astype(np.float32)

    with h5py.File(str(filepath), "w") as f:
        # File-level attributes
        f.attrs["SatelliteName"] = "Suomi NPP"
        f.attrs["InstrumentName"] = "VIIRS"
        f.attrs["ShortName"] = "VNP46A1"
        f.attrs["LongName"] = "VIIRS/NPP Daily Gridded DNB"
        f.attrs["ProcessingLevel"] = "L3"
        f.attrs["VersionId"] = "002"
        f.attrs["RangeBeginningDate"] = "2024-01-01"
        f.attrs["RangeBeginningTime"] = "00:00:00"
        f.attrs["RangeEndingDate"] = "2024-01-01"
        f.attrs["RangeEndingTime"] = "23:59:59"
        f.attrs["DayNightFlag"] = "Night"
        f.attrs["HorizontalTileNumber"] = 9
        f.attrs["VerticalTileNumber"] = 5
        f.attrs["NorthBoundingCoordinate"] = 40.0
        f.attrs["SouthBoundingCoordinate"] = 30.0
        f.attrs["EastBoundingCoordinate"] = -80.0
        f.attrs["WestBoundingCoordinate"] = -90.0

        # Dataset with calibration attributes
        grp = f.create_group("HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields")
        ds = grp.create_dataset("DNB_At_Sensor_Radiance", data=data)
        ds.attrs["long_name"] = "At Sensor Radiance for DNB"
        ds.attrs["units"] = "nW/(cm^2 sr)"
        ds.attrs["scale_factor"] = np.float32(0.1)
        ds.attrs["add_offset"] = np.float32(0.0)
        ds.attrs["_FillValue"] = np.float32(-999.0)
        ds.attrs["valid_range"] = np.array([0.0, 200.0], dtype=np.float32)

    return filepath, data


@pytest.fixture
def viirs_vegetation_h5(tmp_path):
    """Create a synthetic VIIRS VNP13A1 vegetation index HDF5 file."""
    filepath = tmp_path / "VNP13A1.A2024001.h09v05.002.h5"
    data = np.random.randint(-2000, 10000, size=(80, 120)).astype(np.int16)

    with h5py.File(str(filepath), "w") as f:
        f.attrs["SatelliteName"] = "Suomi NPP"
        f.attrs["ShortName"] = "VNP13A1"
        f.attrs["DayNightFlag"] = "Day"
        f.attrs["OrbitNumber"] = 12345

        grp = f.create_group("HDFEOS/GRIDS/VNP_Grid_16Day/Data Fields")
        ds = grp.create_dataset("NDVI", data=data)
        ds.attrs["long_name"] = "NDVI"
        ds.attrs["units"] = "NDVI"
        ds.attrs["scale_factor"] = np.float32(0.0001)
        ds.attrs["_FillValue"] = np.int16(-15000)
        ds.attrs["valid_range"] = np.array([-2000, 10000], dtype=np.int16)

    return filepath, data


@pytest.fixture
def viirs_multiband_h5(tmp_path):
    """Create a synthetic multi-band VIIRS HDF5 file."""
    filepath = tmp_path / "VNP09GA.h5"
    data = np.random.rand(3, 50, 60).astype(np.float32)

    with h5py.File(str(filepath), "w") as f:
        f.attrs["SatelliteName"] = "NOAA-20"
        f.attrs["ShortName"] = "VNP09GA"
        ds = f.create_dataset("surface_reflectance", data=data)
        ds.attrs["long_name"] = "Surface Reflectance"

    return filepath, data


# -- Metadata tests ---------------------------------------------------------

def test_metadata_nightlights(viirs_nightlights_h5):
    """Metadata extracted correctly for VNP46A1 product."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, _ = viirs_nightlights_h5
    with VIIRSReader(filepath) as reader:
        meta = reader.metadata
        assert meta.format == 'VIIRS'
        assert meta.rows == 100
        assert meta.cols == 200
        assert meta.bands == 1
        assert meta.satellite_name == 'Suomi NPP'
        assert meta.instrument_name == 'VIIRS'
        assert meta.product_short_name == 'VNP46A1'
        assert meta.processing_level == 'L3'
        assert meta.day_night_flag == 'Night'
        assert meta.collection_version == '002'


def test_metadata_temporal(viirs_nightlights_h5):
    """Temporal metadata extracted correctly."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, _ = viirs_nightlights_h5
    with VIIRSReader(filepath) as reader:
        meta = reader.metadata
        assert '2024-01-01' in meta.start_datetime
        assert '2024-01-01' in meta.end_datetime


def test_metadata_geospatial(viirs_nightlights_h5):
    """Geospatial bounds extracted correctly."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, _ = viirs_nightlights_h5
    with VIIRSReader(filepath) as reader:
        bounds = reader.metadata.geospatial_bounds
        assert bounds is not None
        assert bounds == (30.0, 40.0, -90.0, -80.0)


def test_metadata_tile_numbers(viirs_nightlights_h5):
    """Tile numbers extracted as strings."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, _ = viirs_nightlights_h5
    with VIIRSReader(filepath) as reader:
        assert reader.metadata.horizontal_tile_number == '9'
        assert reader.metadata.vertical_tile_number == '5'


def test_metadata_calibration(viirs_nightlights_h5):
    """Dataset-level calibration attributes extracted."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, _ = viirs_nightlights_h5
    with VIIRSReader(filepath) as reader:
        meta = reader.metadata
        assert meta.dataset_long_name == 'At Sensor Radiance for DNB'
        assert meta.dataset_units == 'nW/(cm^2 sr)'
        assert meta.scale_factor == pytest.approx(0.1)
        assert meta.add_offset == pytest.approx(0.0)
        assert meta.fill_value == pytest.approx(-999.0)
        assert meta.valid_range == (0.0, 200.0)


def test_metadata_orbit_number(viirs_vegetation_h5):
    """Orbit number extracted as int."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, _ = viirs_vegetation_h5
    with VIIRSReader(filepath) as reader:
        assert reader.metadata.orbit_number == 12345


def test_metadata_is_viirs_type(viirs_nightlights_h5):
    """Metadata is VIIRSMetadata instance."""
    from grdl.IO.multispectral.viirs import VIIRSReader
    from grdl.IO.models import VIIRSMetadata

    filepath, _ = viirs_nightlights_h5
    with VIIRSReader(filepath) as reader:
        assert isinstance(reader.metadata, VIIRSMetadata)


def test_metadata_dict_compat(viirs_nightlights_h5):
    """VIIRSMetadata supports dict-like access."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, _ = viirs_nightlights_h5
    with VIIRSReader(filepath) as reader:
        assert reader.metadata['format'] == 'VIIRS'
        assert reader.metadata['rows'] == 100
        assert 'satellite_name' in reader.metadata


def test_metadata_dataset_path(viirs_nightlights_h5):
    """Auto-detected dataset path stored in metadata."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, _ = viirs_nightlights_h5
    with VIIRSReader(filepath) as reader:
        assert reader.metadata.dataset_path is not None
        assert 'DNB_At_Sensor_Radiance' in reader.metadata.dataset_path


# -- Shape and dtype tests --------------------------------------------------

def test_get_shape_single_band(viirs_nightlights_h5):
    """get_shape returns (rows, cols) for single band."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, _ = viirs_nightlights_h5
    with VIIRSReader(filepath) as reader:
        assert reader.get_shape() == (100, 200)


def test_get_shape_multi_band(viirs_multiband_h5):
    """get_shape returns (rows, cols, bands) for multi-band."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, _ = viirs_multiband_h5
    with VIIRSReader(filepath) as reader:
        assert reader.get_shape() == (50, 60, 3)


def test_get_dtype(viirs_nightlights_h5):
    """get_dtype returns correct numpy dtype."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, _ = viirs_nightlights_h5
    with VIIRSReader(filepath) as reader:
        assert reader.get_dtype() == np.dtype('float32')


# -- Read tests --------------------------------------------------------------

def test_read_chip_single_band(viirs_nightlights_h5):
    """read_chip returns correct data for single 2D dataset."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, original = viirs_nightlights_h5
    with VIIRSReader(filepath) as reader:
        chip = reader.read_chip(10, 30, 20, 50)
        assert chip.shape == (20, 30)
        np.testing.assert_array_almost_equal(
            chip, original[10:30, 20:50]
        )


def test_read_chip_multi_band(viirs_multiband_h5):
    """read_chip returns correct data for 3D dataset."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, original = viirs_multiband_h5
    with VIIRSReader(filepath) as reader:
        chip = reader.read_chip(0, 25, 0, 30)
        assert chip.shape == (3, 25, 30)
        np.testing.assert_array_almost_equal(
            chip, original[:, :25, :30]
        )


def test_read_chip_band_selection(viirs_multiband_h5):
    """read_chip with specific bands returns correct data."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, original = viirs_multiband_h5
    with VIIRSReader(filepath) as reader:
        chip = reader.read_chip(0, 25, 0, 30, bands=[1])
        assert chip.shape == (25, 30)
        np.testing.assert_array_almost_equal(
            chip, original[1, :25, :30]
        )


def test_read_full_single_band(viirs_nightlights_h5):
    """read_full returns entire image for single band."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, original = viirs_nightlights_h5
    with VIIRSReader(filepath) as reader:
        full = reader.read_full()
        assert full.shape == (100, 200)
        np.testing.assert_array_almost_equal(full, original)


def test_read_full_multi_band(viirs_multiband_h5):
    """read_full returns all bands."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, original = viirs_multiband_h5
    with VIIRSReader(filepath) as reader:
        full = reader.read_full()
        assert full.shape == (3, 50, 60)
        np.testing.assert_array_almost_equal(full, original)


# -- Explicit dataset path tests --------------------------------------------

def test_explicit_dataset_path(viirs_nightlights_h5):
    """Explicit dataset_path selects the requested dataset."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, _ = viirs_nightlights_h5
    ds_path = '/HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/DNB_At_Sensor_Radiance'
    with VIIRSReader(filepath, dataset_path=ds_path) as reader:
        assert reader.dataset_path == ds_path
        assert reader.metadata.rows == 100


def test_invalid_dataset_path_raises(viirs_nightlights_h5):
    """Invalid dataset_path raises ValueError."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, _ = viirs_nightlights_h5
    with pytest.raises(ValueError, match="not found"):
        VIIRSReader(filepath, dataset_path='/nonexistent')


# -- Validation tests --------------------------------------------------------

def test_read_chip_negative_start_raises(viirs_nightlights_h5):
    """Negative start indices raise ValueError."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, _ = viirs_nightlights_h5
    with VIIRSReader(filepath) as reader:
        with pytest.raises(ValueError, match="non-negative"):
            reader.read_chip(-1, 10, 0, 10)


def test_read_chip_out_of_bounds_raises(viirs_nightlights_h5):
    """Out-of-bounds end indices raise ValueError."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, _ = viirs_nightlights_h5
    with VIIRSReader(filepath) as reader:
        with pytest.raises(ValueError, match="exceed"):
            reader.read_chip(0, 200, 0, 10)


def test_file_not_found():
    """FileNotFoundError for non-existent file."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    with pytest.raises(FileNotFoundError):
        VIIRSReader('/nonexistent/file.h5')


# -- Context manager and ABC tests ------------------------------------------

def test_context_manager(viirs_nightlights_h5):
    """Context manager opens and closes cleanly."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, _ = viirs_nightlights_h5
    with VIIRSReader(filepath) as reader:
        assert reader.metadata.rows == 100


def test_close_idempotent(viirs_nightlights_h5):
    """Calling close() multiple times does not raise."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, _ = viirs_nightlights_h5
    reader = VIIRSReader(filepath)
    reader.close()
    reader.close()


def test_is_image_reader_subclass():
    """VIIRSReader is a subclass of ImageReader."""
    from grdl.IO.base import ImageReader
    from grdl.IO.multispectral.viirs import VIIRSReader
    assert issubclass(VIIRSReader, ImageReader)


# -- list_datasets utility tests ---------------------------------------------

def test_list_datasets(viirs_nightlights_h5):
    """list_datasets returns 2D+ datasets."""
    from grdl.IO.multispectral.viirs import VIIRSReader

    filepath, _ = viirs_nightlights_h5
    datasets = VIIRSReader.list_datasets(filepath)
    paths = [p for p, _, _ in datasets]
    assert any('DNB_At_Sensor_Radiance' in p for p in paths)


# -- open_multispectral auto-detection ---------------------------------------

def test_open_multispectral_viirs(viirs_nightlights_h5):
    """open_multispectral auto-detects VIIRS products."""
    from grdl.IO.multispectral import open_multispectral

    filepath, _ = viirs_nightlights_h5
    with open_multispectral(filepath) as reader:
        assert reader.metadata.format == 'VIIRS'
