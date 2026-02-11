# -*- coding: utf-8 -*-
"""
PNG Writer Tests - Unit tests for PngWriter.

Tests uint8 grayscale and RGB writes, float auto-normalization, and
edge cases.

Dependencies
------------
pytest
Pillow

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

import warnings

import numpy as np
import pytest

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

pytestmark = pytest.mark.skipif(
    not _HAS_PIL, reason="Pillow not installed"
)


class TestPngWriterGrayscale:
    """Grayscale PNG write tests."""

    def test_write_uint8_grayscale_roundtrip(self, tmp_path):
        """Write uint8 grayscale, read back, verify equality."""
        from grdl.IO.png import PngWriter

        data = np.random.randint(0, 256, (32, 48), dtype=np.uint8)
        filepath = tmp_path / "gray.png"

        with PngWriter(filepath) as writer:
            writer.write(data)

        img = Image.open(str(filepath))
        result = np.array(img)
        np.testing.assert_array_equal(result, data)

    def test_write_float_grayscale_auto_normalizes(self, tmp_path):
        """Float grayscale auto-normalized to uint8 with warning."""
        from grdl.IO.png import PngWriter

        data = np.linspace(0.0, 1.0, 64 * 64).reshape(64, 64).astype(np.float32)
        filepath = tmp_path / "float_gray.png"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with PngWriter(filepath) as writer:
                writer.write(data)
            assert len(w) == 1
            assert "auto-normalized" in str(w[0].message)

        img = Image.open(str(filepath))
        result = np.array(img)
        assert result.dtype == np.uint8
        assert result.min() == 0
        assert result.max() == 255


class TestPngWriterRGB:
    """RGB PNG write tests."""

    def test_write_uint8_rgb_roundtrip(self, tmp_path):
        """Write uint8 RGB, read back, verify equality."""
        from grdl.IO.png import PngWriter

        data = np.random.randint(0, 256, (32, 48, 3), dtype=np.uint8)
        filepath = tmp_path / "rgb.png"

        with PngWriter(filepath) as writer:
            writer.write(data)

        img = Image.open(str(filepath))
        result = np.array(img)
        np.testing.assert_array_equal(result, data)

    def test_write_float_rgb_auto_normalizes(self, tmp_path):
        """Float RGB auto-normalized to uint8 with warning."""
        from grdl.IO.png import PngWriter

        data = np.random.rand(16, 16, 3).astype(np.float64)
        filepath = tmp_path / "float_rgb.png"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with PngWriter(filepath) as writer:
                writer.write(data)
            assert len(w) == 1
            assert "auto-normalized" in str(w[0].message)

        img = Image.open(str(filepath))
        result = np.array(img)
        assert result.dtype == np.uint8


class TestPngWriterEdgeCases:
    """Edge cases and error handling."""

    def test_invalid_shape_raises(self, tmp_path):
        """Non-grayscale/RGB shape raises ValueError."""
        from grdl.IO.png import PngWriter

        data = np.zeros((16, 16, 4), dtype=np.uint8)
        filepath = tmp_path / "bad.png"

        with PngWriter(filepath) as writer:
            with pytest.raises(ValueError, match="grayscale.*RGB"):
                writer.write(data)

    def test_write_chip_raises(self, tmp_path):
        """write_chip raises NotImplementedError."""
        from grdl.IO.png import PngWriter

        filepath = tmp_path / "chip.png"
        with PngWriter(filepath) as writer:
            with pytest.raises(NotImplementedError, match="partial"):
                writer.write_chip(np.zeros((4, 4), dtype=np.uint8), 0, 0)

    def test_implements_image_writer(self, tmp_path):
        """PngWriter is an ImageWriter subclass."""
        from grdl.IO.png import PngWriter
        from grdl.IO.base import ImageWriter

        writer = PngWriter(tmp_path / "test.png")
        assert isinstance(writer, ImageWriter)

    def test_constant_float_array(self, tmp_path):
        """Constant float array (dmax == dmin) produces all-zero output."""
        from grdl.IO.png import PngWriter

        data = np.full((16, 16), 5.0, dtype=np.float32)
        filepath = tmp_path / "const.png"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with PngWriter(filepath) as writer:
                writer.write(data)

        img = Image.open(str(filepath))
        result = np.array(img)
        assert result.max() == 0
