# -*- coding: utf-8 -*-
"""
GeoTIFF Writer Tests - Unit tests for GeoTIFFWriter.

Round-trip tests: write array, read back, assert equal. Covers
single-band, multi-band, with/without geolocation metadata, and
various dtypes.

Dependencies
------------
pytest
rasterio

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

try:
    import rasterio
    from rasterio.transform import from_bounds
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

pytestmark = pytest.mark.skipif(
    not _HAS_RASTERIO, reason="rasterio not installed"
)


class TestGeoTIFFWriterSingleBand:
    """Single-band write and round-trip tests."""

    def test_write_float32_roundtrip(self, tmp_path):
        """Write float32 single-band, read back, verify equality."""
        from grdl.IO.geotiff import GeoTIFFWriter, GeoTIFFReader

        data = np.random.rand(32, 48).astype(np.float32)
        filepath = tmp_path / "single_f32.tif"

        with GeoTIFFWriter(filepath) as writer:
            writer.write(data)

        with GeoTIFFReader(filepath) as reader:
            result = reader.read_full()
            assert result.shape == (32, 48)
            np.testing.assert_array_almost_equal(result, data)

    def test_write_float64_roundtrip(self, tmp_path):
        """Write float64 single-band, read back, verify equality."""
        from grdl.IO.geotiff import GeoTIFFWriter, GeoTIFFReader

        data = np.random.rand(16, 24).astype(np.float64)
        filepath = tmp_path / "single_f64.tif"

        with GeoTIFFWriter(filepath) as writer:
            writer.write(data)

        with GeoTIFFReader(filepath) as reader:
            result = reader.read_full()
            np.testing.assert_array_almost_equal(result, data)

    def test_write_uint8_roundtrip(self, tmp_path):
        """Write uint8 single-band, read back, verify equality."""
        from grdl.IO.geotiff import GeoTIFFWriter, GeoTIFFReader

        data = np.random.randint(0, 256, (20, 30), dtype=np.uint8)
        filepath = tmp_path / "single_u8.tif"

        with GeoTIFFWriter(filepath) as writer:
            writer.write(data)

        with GeoTIFFReader(filepath) as reader:
            result = reader.read_full()
            np.testing.assert_array_equal(result, data)

    def test_write_uint16_roundtrip(self, tmp_path):
        """Write uint16 single-band, read back, verify equality."""
        from grdl.IO.geotiff import GeoTIFFWriter, GeoTIFFReader

        data = np.random.randint(0, 65536, (24, 36), dtype=np.uint16)
        filepath = tmp_path / "single_u16.tif"

        with GeoTIFFWriter(filepath) as writer:
            writer.write(data)

        with GeoTIFFReader(filepath) as reader:
            result = reader.read_full()
            np.testing.assert_array_equal(result, data)


class TestGeoTIFFWriterMultiBand:
    """Multi-band write and round-trip tests."""

    def test_write_multiband_uint8_roundtrip(self, tmp_path):
        """Write 3-band uint8, read back, verify equality."""
        from grdl.IO.geotiff import GeoTIFFWriter, GeoTIFFReader

        data = np.random.randint(0, 256, (3, 32, 48), dtype=np.uint8)
        filepath = tmp_path / "multi_u8.tif"

        with GeoTIFFWriter(filepath) as writer:
            writer.write(data)

        with GeoTIFFReader(filepath) as reader:
            result = reader.read_full()
            assert result.shape == (3, 32, 48)
            np.testing.assert_array_equal(result, data)

    def test_write_multiband_float32_roundtrip(self, tmp_path):
        """Write 4-band float32, read back, verify equality."""
        from grdl.IO.geotiff import GeoTIFFWriter, GeoTIFFReader

        data = np.random.rand(4, 16, 24).astype(np.float32)
        filepath = tmp_path / "multi_f32.tif"

        with GeoTIFFWriter(filepath) as writer:
            writer.write(data)

        with GeoTIFFReader(filepath) as reader:
            result = reader.read_full()
            assert result.shape == (4, 16, 24)
            np.testing.assert_array_almost_equal(result, data)


class TestGeoTIFFWriterGeolocation:
    """Tests with geolocation metadata."""

    def test_write_with_crs_and_transform(self, tmp_path):
        """Write with CRS and transform, verify they persist."""
        from grdl.IO.geotiff import GeoTIFFWriter, GeoTIFFReader

        data = np.random.rand(32, 48).astype(np.float32)
        filepath = tmp_path / "geo.tif"
        transform = from_bounds(10.0, 20.0, 11.0, 21.0, 48, 32)

        with GeoTIFFWriter(filepath) as writer:
            writer.write(data, geolocation={
                'crs': 'EPSG:4326',
                'transform': transform,
            })

        with GeoTIFFReader(filepath) as reader:
            assert 'EPSG:4326' in reader.metadata['crs']
            result = reader.read_full()
            np.testing.assert_array_almost_equal(result, data)

    def test_write_without_geolocation(self, tmp_path):
        """Write without geolocation, CRS should be None."""
        from grdl.IO.geotiff import GeoTIFFWriter, GeoTIFFReader

        data = np.random.rand(16, 16).astype(np.float32)
        filepath = tmp_path / "nogeo.tif"

        with GeoTIFFWriter(filepath) as writer:
            writer.write(data)

        with GeoTIFFReader(filepath) as reader:
            assert reader.metadata['crs'] is None


class TestGeoTIFFWriterEdgeCases:
    """Edge cases and error handling."""

    def test_nan_values_preserved(self, tmp_path):
        """NaN values survive round-trip."""
        from grdl.IO.geotiff import GeoTIFFWriter, GeoTIFFReader

        data = np.array([[1.0, np.nan], [np.nan, 2.0]], dtype=np.float32)
        filepath = tmp_path / "nan.tif"

        with GeoTIFFWriter(filepath) as writer:
            writer.write(data)

        with GeoTIFFReader(filepath) as reader:
            result = reader.read_full()
            assert np.isnan(result[0, 1])
            assert np.isnan(result[1, 0])
            assert result[0, 0] == 1.0
            assert result[1, 1] == 2.0

    def test_invalid_dimensions_raises(self, tmp_path):
        """4D array raises ValueError."""
        from grdl.IO.geotiff import GeoTIFFWriter

        data = np.zeros((2, 3, 4, 5), dtype=np.float32)
        filepath = tmp_path / "bad.tif"

        with GeoTIFFWriter(filepath) as writer:
            with pytest.raises(ValueError, match="2D or 3D"):
                writer.write(data)

    def test_context_manager(self, tmp_path):
        """Context manager protocol works correctly."""
        from grdl.IO.geotiff import GeoTIFFWriter
        from grdl.IO.base import ImageWriter

        filepath = tmp_path / "ctx.tif"
        writer = GeoTIFFWriter(filepath)
        assert isinstance(writer, ImageWriter)
        with writer:
            writer.write(np.zeros((8, 8), dtype=np.float32))

    def test_write_chip(self, tmp_path):
        """write_chip writes a partial region to existing file."""
        from grdl.IO.geotiff import GeoTIFFWriter, GeoTIFFReader

        data = np.zeros((32, 32), dtype=np.float32)
        filepath = tmp_path / "chip.tif"

        with GeoTIFFWriter(filepath) as writer:
            writer.write(data)

        chip = np.ones((8, 8), dtype=np.float32) * 42.0
        with GeoTIFFWriter(filepath) as writer:
            writer.write_chip(chip, row_start=4, col_start=4)

        with GeoTIFFReader(filepath) as reader:
            result = reader.read_full()
            np.testing.assert_array_almost_equal(
                result[4:12, 4:12], chip
            )
            assert result[0, 0] == 0.0
