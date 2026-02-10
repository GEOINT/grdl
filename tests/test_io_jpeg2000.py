# -*- coding: utf-8 -*-
"""
JPEG2000 Reader Tests - Unit tests for JP2Reader.

Uses synthetic JPEG2000 files created with rasterio or glymur for testing.

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

try:
    import glymur
    _HAS_GLYMUR = True
except ImportError:
    _HAS_GLYMUR = False

pytestmark = pytest.mark.skipif(
    not (_HAS_RASTERIO or _HAS_GLYMUR),
    reason="Neither rasterio nor glymur installed"
)


@pytest.fixture
def single_band_jp2_rasterio(tmp_path):
    """Create a single-band JP2 file using rasterio."""
    if not _HAS_RASTERIO:
        pytest.skip("rasterio not installed")

    filepath = tmp_path / "test_single.jp2"
    data = np.random.randint(0, 4096, size=(100, 200), dtype=np.uint16)
    transform = from_bounds(0, 0, 1, 1, 200, 100)

    try:
        # Create with lossless compression using proper rasterio syntax
        with rasterio.open(
            str(filepath), 'w', driver='JP2OpenJPEG',
            height=100, width=200, count=1,
            dtype='uint16', crs='EPSG:4326',
            transform=transform,
            **{'REVERSIBLE': 'YES', 'QUALITY': '100'}  # Truly lossless
        ) as ds:
            ds.write(data, 1)
    except Exception:
        # JP2 driver might not be available
        pytest.skip("GDAL JP2 driver not available")

    return filepath, data


@pytest.fixture
def multi_band_jp2_rasterio(tmp_path):
    """Create a 3-band JP2 file using rasterio."""
    if not _HAS_RASTERIO:
        pytest.skip("rasterio not installed")

    filepath = tmp_path / "test_multi.jp2"
    data = np.random.randint(0, 256, size=(3, 80, 120), dtype=np.uint8)
    transform = from_bounds(10, 20, 11, 21, 120, 80)

    try:
        # Create with lossless compression using proper rasterio syntax
        with rasterio.open(
            str(filepath), 'w', driver='JP2OpenJPEG',
            height=80, width=120, count=3,
            dtype='uint8', crs='EPSG:4326',
            transform=transform,
            **{'REVERSIBLE': 'YES', 'QUALITY': '100'}  # Truly lossless
        ) as ds:
            ds.write(data)
    except Exception:
        pytest.skip("GDAL JP2 driver not available")

    return filepath, data


@pytest.fixture
def single_band_jp2_glymur(tmp_path):
    """Create a single-band JP2 file using glymur."""
    if not _HAS_GLYMUR:
        pytest.skip("glymur not installed")

    filepath = tmp_path / "test_glymur.jp2"
    data = np.random.randint(0, 4096, size=(100, 200), dtype=np.uint16)

    try:
        glymur.Jp2k(str(filepath), data=data)
    except Exception as e:
        pytest.skip(f"glymur JP2 creation failed: {e}")

    return filepath, data


# -- Metadata tests ---------------------------------------------------------

@pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed")
def test_metadata_single_band_rasterio(single_band_jp2_rasterio):
    """Metadata extracted correctly from single-band JP2 with rasterio."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, _ = single_band_jp2_rasterio
    with JP2Reader(filepath) as reader:
        assert reader.metadata['format'] == 'JPEG2000'
        assert reader.metadata['rows'] == 100
        assert reader.metadata['cols'] == 200
        assert reader.metadata['bands'] == 1
        assert reader.metadata['dtype'] == 'uint16'
        assert reader.backend == 'rasterio'


@pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed")
def test_metadata_multi_band_rasterio(multi_band_jp2_rasterio):
    """Metadata extracted correctly from multi-band JP2 with rasterio."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, _ = multi_band_jp2_rasterio
    with JP2Reader(filepath) as reader:
        assert reader.metadata['bands'] == 3
        assert reader.metadata['rows'] == 80
        assert reader.metadata['cols'] == 120
        assert reader.backend == 'rasterio'


@pytest.mark.skipif(not _HAS_GLYMUR, reason="glymur not installed")
def test_metadata_single_band_glymur(single_band_jp2_glymur):
    """Metadata extracted correctly from single-band JP2 with glymur."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, _ = single_band_jp2_glymur
    with JP2Reader(filepath, backend='glymur') as reader:
        assert reader.metadata['format'] == 'JPEG2000'
        assert reader.metadata['rows'] == 100
        assert reader.metadata['cols'] == 200
        assert reader.metadata['bands'] == 1
        assert reader.backend == 'glymur'


# -- Backend selection tests ------------------------------------------------

@pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed")
def test_backend_explicit_rasterio(single_band_jp2_rasterio):
    """Explicit backend='rasterio' uses rasterio."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, _ = single_band_jp2_rasterio
    with JP2Reader(filepath, backend='rasterio') as reader:
        assert reader.backend == 'rasterio'


@pytest.mark.skipif(not _HAS_GLYMUR, reason="glymur not installed")
def test_backend_explicit_glymur(single_band_jp2_glymur):
    """Explicit backend='glymur' uses glymur."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, _ = single_band_jp2_glymur
    with JP2Reader(filepath, backend='glymur') as reader:
        assert reader.backend == 'glymur'


@pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed")
def test_backend_auto_prefers_rasterio(single_band_jp2_rasterio):
    """Auto backend prefers rasterio when available."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, _ = single_band_jp2_rasterio
    with JP2Reader(filepath, backend='auto') as reader:
        assert reader.backend == 'rasterio'


# -- Shape and dtype tests --------------------------------------------------

@pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed")
def test_get_shape_single_band(single_band_jp2_rasterio):
    """get_shape returns (rows, cols) for single band."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, _ = single_band_jp2_rasterio
    with JP2Reader(filepath) as reader:
        assert reader.get_shape() == (100, 200)


@pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed")
def test_get_shape_multi_band(multi_band_jp2_rasterio):
    """get_shape returns (rows, cols, bands) for multi-band."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, _ = multi_band_jp2_rasterio
    with JP2Reader(filepath) as reader:
        assert reader.get_shape() == (80, 120, 3)


@pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed")
def test_get_dtype(single_band_jp2_rasterio):
    """get_dtype returns correct numpy dtype."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, _ = single_band_jp2_rasterio
    with JP2Reader(filepath) as reader:
        assert reader.get_dtype() == np.dtype('uint16')


# -- Read tests (rasterio backend) -------------------------------------------

@pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed")
def test_read_chip_single_band_rasterio(single_band_jp2_rasterio):
    """read_chip returns correct data for single band with rasterio."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, original = single_band_jp2_rasterio
    with JP2Reader(filepath) as reader:
        chip = reader.read_chip(10, 30, 20, 50)
        assert chip.shape == (20, 30)
        # Compare against ground truth (JP2-encoded data read back)
        np.testing.assert_array_equal(chip, original[10:30, 20:50])


@pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed")
def test_read_chip_multi_band_rasterio(multi_band_jp2_rasterio):
    """read_chip returns correct data for multi-band with rasterio."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, original = multi_band_jp2_rasterio
    with JP2Reader(filepath) as reader:
        chip = reader.read_chip(0, 40, 0, 60)
        assert chip.shape == (3, 40, 60)
        # Compare against ground truth (JP2-encoded data read back)
        np.testing.assert_array_equal(chip, original[:, :40, :60])


@pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed")
def test_read_chip_band_selection_rasterio(multi_band_jp2_rasterio):
    """read_chip with specific bands returns correct data with rasterio."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, original = multi_band_jp2_rasterio
    with JP2Reader(filepath) as reader:
        chip = reader.read_chip(0, 40, 0, 60, bands=[1])
        assert chip.shape == (40, 60)
        # Compare against ground truth (JP2-encoded data read back)
        np.testing.assert_array_equal(chip, original[1, :40, :60])


@pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed")
def test_read_full_single_band_rasterio(single_band_jp2_rasterio):
    """read_full returns entire image for single band with rasterio."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, original = single_band_jp2_rasterio
    with JP2Reader(filepath) as reader:
        full = reader.read_full()
        assert full.shape == (100, 200)
        # Compare against ground truth (JP2-encoded data read back)
        np.testing.assert_array_equal(full, original)


# -- Read tests (glymur backend) ---------------------------------------------

@pytest.mark.skipif(not _HAS_GLYMUR, reason="glymur not installed")
def test_read_chip_single_band_glymur(single_band_jp2_glymur):
    """read_chip returns correct data for single band with glymur."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, original = single_band_jp2_glymur
    with JP2Reader(filepath, backend='glymur') as reader:
        chip = reader.read_chip(10, 30, 20, 50)
        assert chip.shape == (20, 30)
        # Compare against ground truth (JP2-encoded data read back)
        np.testing.assert_array_equal(chip, original[10:30, 20:50])


@pytest.mark.skipif(not _HAS_GLYMUR, reason="glymur not installed")
def test_read_full_single_band_glymur(single_band_jp2_glymur):
    """read_full returns entire image for single band with glymur."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, original = single_band_jp2_glymur
    with JP2Reader(filepath, backend='glymur') as reader:
        full = reader.read_full()
        assert full.shape == (100, 200)
        # Compare against ground truth (JP2-encoded data read back)
        np.testing.assert_array_equal(full, original)


# -- Validation tests --------------------------------------------------------

@pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed")
def test_read_chip_negative_start_raises(single_band_jp2_rasterio):
    """Negative start indices raise ValueError."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, _ = single_band_jp2_rasterio
    with JP2Reader(filepath) as reader:
        with pytest.raises(ValueError, match="non-negative"):
            reader.read_chip(-1, 10, 0, 10)


@pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed")
def test_read_chip_out_of_bounds_raises(single_band_jp2_rasterio):
    """Out-of-bounds end indices raise ValueError."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, _ = single_band_jp2_rasterio
    with JP2Reader(filepath) as reader:
        with pytest.raises(ValueError, match="exceed"):
            reader.read_chip(0, 200, 0, 10)


def test_file_not_found():
    """FileNotFoundError for non-existent file."""
    from grdl.IO.jpeg2000 import JP2Reader

    with pytest.raises(FileNotFoundError):
        JP2Reader('/nonexistent/file.jp2')


def test_no_backend_available(monkeypatch):
    """ImportError when no backend is available."""
    from grdl.IO import jpeg2000

    # Mock both backends as unavailable
    monkeypatch.setattr(jpeg2000, '_HAS_RASTERIO', False)
    monkeypatch.setattr(jpeg2000, '_HAS_GLYMUR', False)

    # Need to create a dummy file that exists
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jp2', delete=False) as f:
        filepath = f.name

    try:
        with pytest.raises(ImportError, match="Either rasterio or glymur"):
            jpeg2000.JP2Reader(filepath)
    finally:
        Path(filepath).unlink()


# -- Geolocation tests -------------------------------------------------------

@pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed")
def test_geolocation_rasterio(single_band_jp2_rasterio):
    """get_geolocation returns CRS and transform info with rasterio."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, _ = single_band_jp2_rasterio
    with JP2Reader(filepath) as reader:
        geo = reader.get_geolocation()
        assert geo is not None
        assert 'crs' in geo
        assert 'transform' in geo
        assert 'bounds' in geo
        assert 'EPSG:4326' in geo['crs']


@pytest.mark.skipif(not _HAS_GLYMUR, reason="glymur not installed")
def test_geolocation_none_with_glymur(single_band_jp2_glymur):
    """get_geolocation returns None with glymur backend."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, _ = single_band_jp2_glymur
    with JP2Reader(filepath, backend='glymur') as reader:
        geo = reader.get_geolocation()
        # Glymur backend doesn't extract geolocation
        assert geo is None


# -- Context manager and cleanup tests ---------------------------------------

@pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed")
def test_context_manager(single_band_jp2_rasterio):
    """Context manager opens and closes cleanly."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, _ = single_band_jp2_rasterio
    with JP2Reader(filepath) as reader:
        assert reader.metadata['rows'] == 100


@pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed")
def test_close_idempotent(single_band_jp2_rasterio):
    """Calling close() multiple times does not raise."""
    from grdl.IO.jpeg2000 import JP2Reader

    filepath, _ = single_band_jp2_rasterio
    reader = JP2Reader(filepath)
    reader.close()
    reader.close()


# -- ImageReader ABC contract ------------------------------------------------

def test_is_image_reader_subclass():
    """JP2Reader is a subclass of ImageReader."""
    from grdl.IO.base import ImageReader
    from grdl.IO.jpeg2000 import JP2Reader
    assert issubclass(JP2Reader, ImageReader)
