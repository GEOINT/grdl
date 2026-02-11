# -*- coding: utf-8 -*-
"""
Sentinel-2 Reader Tests - Unit tests for Sentinel2Reader.

Uses synthetic JPEG2000 files with Sentinel-2 naming conventions for testing.
Tests metadata extraction, filename parsing, and integration with open_eo().

Dependencies
------------
pytest
rasterio or glymur

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
2026-02-11

Modified
--------
2026-02-11
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

try:
    import glymur
    _HAS_GLYMUR = True
except ImportError:
    _HAS_GLYMUR = False

pytestmark = pytest.mark.skipif(
    not (_HAS_RASTERIO or _HAS_GLYMUR),
    reason="Neither rasterio nor glymur installed"
)


# =============================================================================
# Test Fixtures - Create Synthetic Sentinel-2 Files
# =============================================================================

@pytest.fixture
def sentinel2_standalone_band(tmp_path):
    """Create a synthetic Sentinel-2 standalone band file."""
    if not _HAS_RASTERIO:
        pytest.skip("rasterio not installed")

    # Use Sentinel-2 naming convention
    filepath = tmp_path / "T10SEG_20240115T184719_B04_10m.jp2"

    # Sentinel-2 characteristics: 10980x10980 for 10m bands, uint16, 15-bit values
    data = np.random.randint(0, 32767, size=(10980, 10980), dtype=np.uint16)

    # UTM Zone 10N for tile T10SEG
    transform = from_bounds(500000, 4000000, 609800, 4109800, 10980, 10980)

    try:
        with rasterio.open(
            str(filepath), 'w', driver='JP2OpenJPEG',
            height=10980, width=10980, count=1,
            dtype='uint16', crs='EPSG:32610',  # UTM Zone 10N
            transform=transform,
            **{'REVERSIBLE': 'YES', 'QUALITY': '100'}
        ) as ds:
            ds.write(data, 1)
    except Exception:
        pytest.skip("GDAL JP2 driver not available")

    return filepath, data


@pytest.fixture
def sentinel2_b8a_band(tmp_path):
    """Create a synthetic Sentinel-2 B8A (NIR narrow) band file."""
    if not _HAS_RASTERIO:
        pytest.skip("rasterio not installed")

    # B8A is a 20m resolution band
    filepath = tmp_path / "T32TQR_20230710T103619_B8A_20m.jp2"

    # 20m bands are 5490x5490
    data = np.random.randint(0, 32767, size=(5490, 5490), dtype=np.uint16)
    transform = from_bounds(300000, 5000000, 409800, 5109800, 5490, 5490)

    try:
        with rasterio.open(
            str(filepath), 'w', driver='JP2OpenJPEG',
            height=5490, width=5490, count=1,
            dtype='uint16', crs='EPSG:32632',  # UTM Zone 32N
            transform=transform,
            **{'REVERSIBLE': 'YES', 'QUALITY': '100'}
        ) as ds:
            ds.write(data, 1)
    except Exception:
        pytest.skip("GDAL JP2 driver not available")

    return filepath, data


@pytest.fixture
def sentinel2_safe_structure(tmp_path):
    """Create a synthetic Sentinel-2 file within SAFE directory structure."""
    if not _HAS_RASTERIO:
        pytest.skip("rasterio not installed")

    # Create SAFE directory structure
    safe_dir = tmp_path / 'S2A_MSIL2A_20240115T184719_N0510_R070_T10SEG_20240115T201234.SAFE'
    granule_dir = safe_dir / 'GRANULE' / 'L2A_T10SEG_A044567_20240115T184719' / 'IMG_DATA' / 'R10m'
    granule_dir.mkdir(parents=True)

    filepath = granule_dir / 'T10SEG_20240115T184719_B04_10m.jp2'

    # Small size for faster tests
    data = np.random.randint(0, 32767, size=(1000, 1000), dtype=np.uint16)
    transform = from_bounds(500000, 4000000, 510000, 4010000, 1000, 1000)

    try:
        with rasterio.open(
            str(filepath), 'w', driver='JP2OpenJPEG',
            height=1000, width=1000, count=1,
            dtype='uint16', crs='EPSG:32610',
            transform=transform,
            **{'REVERSIBLE': 'YES', 'QUALITY': '100'}
        ) as ds:
            ds.write(data, 1)
    except Exception:
        pytest.skip("GDAL JP2 driver not available")

    return filepath, data


@pytest.fixture
def non_sentinel2_jp2(tmp_path):
    """Create a non-Sentinel-2 JP2 file (e.g., Pleiades-style)."""
    if not _HAS_RASTERIO:
        pytest.skip("rasterio not installed")

    filepath = tmp_path / "pleiades_image_12345.jp2"
    data = np.random.randint(0, 4096, size=(500, 500), dtype=np.uint16)
    transform = from_bounds(0, 0, 1, 1, 500, 500)

    try:
        with rasterio.open(
            str(filepath), 'w', driver='JP2OpenJPEG',
            height=500, width=500, count=1,
            dtype='uint16', crs='EPSG:4326',
            transform=transform,
            **{'REVERSIBLE': 'YES', 'QUALITY': '100'}
        ) as ds:
            ds.write(data, 1)
    except Exception:
        pytest.skip("GDAL JP2 driver not available")

    return filepath, data


# =============================================================================
# Metadata Extraction Tests
# =============================================================================

def test_metadata_standalone_band(sentinel2_standalone_band):
    """Metadata extracted correctly from standalone Sentinel-2 band."""
    from grdl.IO.eo.sentinel2 import Sentinel2Reader

    filepath, _ = sentinel2_standalone_band
    with Sentinel2Reader(filepath) as reader:
        # Base metadata from JP2
        assert reader.metadata.format == 'Sentinel-2_L2A'
        assert reader.metadata.rows == 10980
        assert reader.metadata.cols == 10980
        assert reader.metadata.dtype == 'uint16'
        assert reader.metadata.bands == 1
        # CRS might be None if JP2 driver doesn't support it
        if reader.metadata.crs:
            assert 'EPSG:32610' in reader.metadata.crs or '32610' in reader.metadata.crs
        assert reader.metadata.nodata == 0

        # Sentinel-2 specific metadata from filename
        assert reader.metadata.mgrs_tile_id == 'T10SEG'
        assert reader.metadata.sensing_datetime == '2024-01-15T18:47:19'
        assert reader.metadata.band_id == 'B04'
        assert reader.metadata.resolution_tier == 10
        assert reader.metadata.utm_zone == 10
        assert reader.metadata.latitude_band == 'S'

        # Band wavelength lookup
        assert reader.metadata.wavelength_center == 665  # Red band
        assert reader.metadata.wavelength_range == (650, 680)


def test_metadata_b8a_band(sentinel2_b8a_band):
    """Metadata extracted correctly from B8A (NIR narrow) band."""
    from grdl.IO.eo.sentinel2 import Sentinel2Reader

    filepath, _ = sentinel2_b8a_band
    with Sentinel2Reader(filepath) as reader:
        assert reader.metadata.mgrs_tile_id == 'T32TQR'
        assert reader.metadata.band_id == 'B8A'
        assert reader.metadata.resolution_tier == 20
        assert reader.metadata.utm_zone == 32
        assert reader.metadata.latitude_band == 'T'

        # B8A wavelength
        assert reader.metadata.wavelength_center == 865
        assert reader.metadata.wavelength_range == (855, 875)


def test_metadata_safe_structure(sentinel2_safe_structure):
    """Metadata extracted from SAFE directory structure."""
    from grdl.IO.eo.sentinel2 import Sentinel2Reader

    filepath, _ = sentinel2_safe_structure
    with Sentinel2Reader(filepath) as reader:
        # SAFE-level metadata
        assert reader.metadata.satellite == 'S2A'
        assert reader.metadata.product_type == 'MSIL2A'
        assert reader.metadata.processing_level == 'L2A'
        assert reader.metadata.baseline_processing == 'N0510'
        assert reader.metadata.relative_orbit == 70
        assert reader.metadata.product_discriminator == '20240115T201234'

        # Band-level metadata
        assert reader.metadata.band_id == 'B04'
        assert reader.metadata.resolution_tier == 10
        assert reader.metadata.mgrs_tile_id == 'T10SEG'


def test_metadata_missing_for_non_sentinel2(non_sentinel2_jp2):
    """Non-Sentinel-2 files have None for Sentinel-2 specific fields."""
    from grdl.IO.eo.sentinel2 import Sentinel2Reader

    filepath, _ = non_sentinel2_jp2
    with Sentinel2Reader(filepath) as reader:
        # Base fields should still work (from JP2Reader)
        assert reader.metadata.rows == 500
        assert reader.metadata.cols == 500
        assert reader.metadata.dtype == 'uint16'

        # Sentinel-2 specific fields should be None
        assert reader.metadata.satellite is None
        assert reader.metadata.mgrs_tile_id is None
        assert reader.metadata.band_id is None
        assert reader.metadata.wavelength_center is None


# =============================================================================
# Data Reading Tests
# =============================================================================

def test_read_chip_standalone(sentinel2_standalone_band):
    """read_chip returns correct data from standalone band."""
    from grdl.IO.eo.sentinel2 import Sentinel2Reader

    filepath, original = sentinel2_standalone_band
    with Sentinel2Reader(filepath) as reader:
        chip = reader.read_chip(0, 1000, 0, 1000)
        assert chip.shape == (1000, 1000)
        assert chip.dtype == np.uint16

        # Verify values are in 15-bit range (Sentinel-2 characteristic)
        assert chip.max() <= 32767
        assert chip.min() >= 0


def test_read_full_small_file(sentinel2_safe_structure):
    """read_full returns entire image."""
    from grdl.IO.eo.sentinel2 import Sentinel2Reader

    filepath, original = sentinel2_safe_structure
    with Sentinel2Reader(filepath) as reader:
        full = reader.read_full()
        assert full.shape == (1000, 1000)
        assert full.dtype == np.uint16


def test_get_shape_standalone(sentinel2_standalone_band):
    """get_shape returns correct dimensions."""
    from grdl.IO.eo.sentinel2 import Sentinel2Reader

    filepath, _ = sentinel2_standalone_band
    with Sentinel2Reader(filepath) as reader:
        assert reader.get_shape() == (10980, 10980)


def test_get_dtype_standalone(sentinel2_standalone_band):
    """get_dtype returns uint16 for Sentinel-2."""
    from grdl.IO.eo.sentinel2 import Sentinel2Reader

    filepath, _ = sentinel2_standalone_band
    with Sentinel2Reader(filepath) as reader:
        assert reader.get_dtype() == np.dtype('uint16')


# =============================================================================
# Detection and Integration Tests
# =============================================================================

def test_detection_standalone_band():
    """Standalone Sentinel-2 band is detected correctly."""
    from grdl.IO.eo import _is_sentinel2

    assert _is_sentinel2(Path('T10SEG_20240115T184719_B04_10m.jp2'))
    assert _is_sentinel2(Path('T32TQR_20230710T103619_B8A_20m.jp2'))


def test_detection_safe_archive():
    """SAFE archive is detected correctly."""
    from grdl.IO.eo import _is_sentinel2

    assert _is_sentinel2(Path('S2A_MSIL2A_20240115T184719_N0510_R070_T10SEG_20240115T201234.SAFE'))
    assert _is_sentinel2(Path('S2B_MSIL1C_20230601T100029_N0509_R122_T33UUP_20230601T121000.SAFE'))


def test_detection_within_safe():
    """Files within SAFE structure are detected."""
    from grdl.IO.eo import _is_sentinel2

    path = Path('S2A_MSIL2A_20240115T184719_N0510_R070_T10SEG_20240115T201234.SAFE/'
                'GRANULE/L2A_T10SEG_A044567_20240115T184719/'
                'IMG_DATA/R10m/T10SEG_20240115T184719_B04_10m.jp2')
    assert _is_sentinel2(path)


def test_detection_rejects_non_sentinel2():
    """Non-Sentinel-2 JP2 files are not detected as Sentinel-2."""
    from grdl.IO.eo import _is_sentinel2

    assert not _is_sentinel2(Path('pleiades_image.jp2'))
    assert not _is_sentinel2(Path('spot_scene.jp2'))
    assert not _is_sentinel2(Path('enmap_L2A_data.jp2'))
    assert not _is_sentinel2(Path('random_file.jp2'))


def test_open_eo_detects_sentinel2(sentinel2_standalone_band):
    """open_eo() auto-detects Sentinel-2 files."""
    from grdl.IO.eo import open_eo
    from grdl.IO.eo.sentinel2 import Sentinel2Reader

    filepath, _ = sentinel2_standalone_band
    reader = open_eo(filepath)

    assert isinstance(reader, Sentinel2Reader)
    assert reader.metadata.band_id == 'B04'
    reader.close()


def test_open_eo_uses_jp2reader_for_non_sentinel2(non_sentinel2_jp2):
    """open_eo() uses generic JP2Reader for non-Sentinel-2 files."""
    from grdl.IO.eo import open_eo
    from grdl.IO.eo.sentinel2 import Sentinel2Reader

    filepath, _ = non_sentinel2_jp2
    reader = open_eo(filepath)

    # Should be JP2Reader, not Sentinel2Reader
    assert not isinstance(reader, Sentinel2Reader)
    assert reader.metadata.format == 'JPEG2000'
    reader.close()


# =============================================================================
# Filename Parsing Tests (Unit Tests)
# =============================================================================

def test_parse_standalone_filename():
    """Parse standalone Sentinel-2 filename."""
    from grdl.IO.eo.sentinel2 import _parse_sentinel2_filename

    result = _parse_sentinel2_filename(Path('T10SEG_20240115T184719_B04_10m.jp2'))

    assert result['mgrs_tile_id'] == 'T10SEG'
    assert result['sensing_datetime'] == '20240115T184719'
    assert result['band_id'] == 'B04'
    assert result['resolution_tier'] == 10


def test_parse_safe_archive_filename():
    """Parse SAFE archive directory name."""
    from grdl.IO.eo.sentinel2 import _parse_sentinel2_filename

    result = _parse_sentinel2_filename(
        Path('S2A_MSIL2A_20240115T184719_N0510_R070_T10SEG_20240115T201234.SAFE')
    )

    assert result['satellite'] == 'S2A'
    assert result['product_type'] == 'MSIL2A'
    assert result['processing_level'] == 'L2A'
    assert result['baseline_processing'] == 'N0510'
    assert result['relative_orbit'] == 70


def test_parse_invalid_filename_returns_empty():
    """Invalid filenames return empty dict."""
    from grdl.IO.eo.sentinel2 import _parse_sentinel2_filename

    result = _parse_sentinel2_filename(Path('invalid_filename.jp2'))
    assert result == {}


def test_parse_mgrs_tile():
    """Parse MGRS tile ID to extract UTM zone and latitude band."""
    from grdl.IO.eo.sentinel2 import _parse_mgrs_tile

    utm_zone, lat_band = _parse_mgrs_tile('T10SEG')
    assert utm_zone == 10
    assert lat_band == 'S'

    utm_zone, lat_band = _parse_mgrs_tile('T32TQR')
    assert utm_zone == 32
    assert lat_band == 'T'


def test_parse_invalid_mgrs_tile():
    """Invalid MGRS tile IDs return None."""
    from grdl.IO.eo.sentinel2 import _parse_mgrs_tile

    assert _parse_mgrs_tile(None) == (None, None)
    assert _parse_mgrs_tile('') == (None, None)
    assert _parse_mgrs_tile('ABC') == (None, None)


# =============================================================================
# Band Wavelength Tests
# =============================================================================

def test_band_wavelengths_complete():
    """All 13 Sentinel-2 bands have wavelength data."""
    from grdl.IO.eo.sentinel2 import BAND_WAVELENGTHS

    expected_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06',
                      'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']

    for band_id in expected_bands:
        assert band_id in BAND_WAVELENGTHS
        center, wl_min, wl_max = BAND_WAVELENGTHS[band_id]
        assert wl_min < center < wl_max


def test_band_wavelength_specific_values():
    """Spot check specific band wavelengths."""
    from grdl.IO.eo.sentinel2 import BAND_WAVELENGTHS

    # Red band
    assert BAND_WAVELENGTHS['B04'] == (665, 650, 680)
    # NIR
    assert BAND_WAVELENGTHS['B08'] == (842, 785, 900)
    # NIR narrow
    assert BAND_WAVELENGTHS['B8A'] == (865, 855, 875)
