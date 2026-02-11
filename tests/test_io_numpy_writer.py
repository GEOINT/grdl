# -*- coding: utf-8 -*-
"""
NumPy Writer Tests - Unit tests for NumpyWriter.

Tests .npy and .npz round-trips and JSON sidecar metadata.

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

import json

import numpy as np
import pytest


class TestNumpyWriterNpy:
    """Tests for .npy single-array writes."""

    def test_write_npy_roundtrip(self, tmp_path):
        """Write .npy, read back, verify equality."""
        from grdl.IO.numpy_io import NumpyWriter

        data = np.random.rand(32, 48).astype(np.float32)
        filepath = tmp_path / "test.npy"

        with NumpyWriter(filepath) as writer:
            writer.write(data)

        result = np.load(str(filepath))
        np.testing.assert_array_equal(result, data)

    def test_write_npy_sidecar_created(self, tmp_path):
        """JSON sidecar file is created alongside .npy."""
        from grdl.IO.numpy_io import NumpyWriter
        from grdl.IO.models import ImageMetadata

        data = np.random.rand(16, 24).astype(np.float64)
        filepath = tmp_path / "meta.npy"
        meta = ImageMetadata(
            format='numpy', rows=16, cols=24, dtype='float64',
            extras={'source': 'test'},
        )

        with NumpyWriter(filepath, metadata=meta) as writer:
            writer.write(data)

        sidecar_path = tmp_path / "meta.npy.json"
        assert sidecar_path.exists()

        with open(sidecar_path) as f:
            meta = json.load(f)

        assert meta['shape'] == [16, 24]
        assert meta['dtype'] == 'float64'
        assert meta['source'] == 'test'

    def test_write_npy_sidecar_with_geolocation(self, tmp_path):
        """Geolocation is included in sidecar when provided."""
        from grdl.IO.numpy_io import NumpyWriter

        data = np.zeros((8, 8), dtype=np.float32)
        filepath = tmp_path / "geo.npy"

        with NumpyWriter(filepath) as writer:
            writer.write(data, geolocation={'crs': 'EPSG:4326'})

        sidecar_path = tmp_path / "geo.npy.json"
        with open(sidecar_path) as f:
            meta = json.load(f)
        assert meta['geolocation']['crs'] == 'EPSG:4326'

    def test_write_3d_array(self, tmp_path):
        """3D array round-trips correctly."""
        from grdl.IO.numpy_io import NumpyWriter

        data = np.random.rand(3, 16, 16).astype(np.float32)
        filepath = tmp_path / "bands.npy"

        with NumpyWriter(filepath) as writer:
            writer.write(data)

        result = np.load(str(filepath))
        np.testing.assert_array_equal(result, data)


class TestNumpyWriterNpz:
    """Tests for .npz multi-array writes."""

    def test_write_npz_roundtrip(self, tmp_path):
        """Write .npz with multiple arrays, read back, verify."""
        from grdl.IO.numpy_io import NumpyWriter

        arrays = {
            'red': np.random.rand(16, 16).astype(np.float32),
            'green': np.random.rand(16, 16).astype(np.float32),
            'blue': np.random.rand(16, 16).astype(np.float32),
        }
        filepath = tmp_path / "multi.npz"

        with NumpyWriter(filepath) as writer:
            writer.write_npz(arrays)

        # np.savez adds .npz extension if not present
        actual_path = filepath
        if not actual_path.exists():
            actual_path = filepath.with_suffix('.npz')

        loaded = np.load(str(actual_path))
        for name, expected in arrays.items():
            np.testing.assert_array_equal(loaded[name], expected)

    def test_write_npz_sidecar(self, tmp_path):
        """JSON sidecar includes array_names for .npz."""
        from grdl.IO.numpy_io import NumpyWriter

        arrays = {
            'data': np.zeros((8, 8), dtype=np.float32),
            'mask': np.ones((8, 8), dtype=np.uint8),
        }
        filepath = tmp_path / "sidecar.npz"

        with NumpyWriter(filepath) as writer:
            writer.write_npz(arrays)

        sidecar_path = filepath.parent / (filepath.name + '.json')
        with open(sidecar_path) as f:
            meta = json.load(f)

        assert 'array_names' in meta
        assert set(meta['array_names']) == {'data', 'mask'}


class TestNumpyWriterEdgeCases:
    """Edge cases and error handling."""

    def test_write_chip_raises(self, tmp_path):
        """write_chip raises NotImplementedError."""
        from grdl.IO.numpy_io import NumpyWriter

        filepath = tmp_path / "chip.npy"
        with NumpyWriter(filepath) as writer:
            with pytest.raises(NotImplementedError, match="partial"):
                writer.write_chip(np.zeros((4, 4)), 0, 0)

    def test_implements_image_writer(self, tmp_path):
        """NumpyWriter is an ImageWriter subclass."""
        from grdl.IO.numpy_io import NumpyWriter
        from grdl.IO.base import ImageWriter

        writer = NumpyWriter(tmp_path / "test.npy")
        assert isinstance(writer, ImageWriter)
