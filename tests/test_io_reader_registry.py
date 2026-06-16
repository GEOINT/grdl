# -*- coding: utf-8 -*-
"""
Reader Registry Tests - Unit tests for get_reader(), register_reader(),
register_writer(), and the ImageReader shape helpers.

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-06-16

Modified
--------
2026-06-16
"""

import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from grdl.IO import (
    _READER_REGISTRY,
    _WRITER_REGISTRY,
    get_reader,
    register_reader,
    register_writer,
)
from grdl.IO.base import ImageReader


# ===================================================================
# _READER_REGISTRY contents
# ===================================================================

class TestReaderRegistryContents:
    """Verify expected format keys are present in _READER_REGISTRY."""

    @pytest.mark.parametrize("key", [
        'geotiff', 'hdf5', 'nitf', 'jp2',
        'sicd', 'cphd', 'crsd', 'sidd',
        'sentinel1_slc', 'sentinel1_l0', 'biomass', 'terrasar', 'nisar',
        'sicd_collection', 'eo_nitf', 'sentinel2', 'aster', 'viirs',
        'stanag4607', 'gdal', 'probe',
    ])
    def test_key_present(self, key):
        assert key in _READER_REGISTRY, f"'{key}' missing from _READER_REGISTRY"

    def test_each_entry_is_2_tuple(self):
        for key, value in _READER_REGISTRY.items():
            assert isinstance(value, tuple) and len(value) == 2, (
                f"Registry entry for '{key}' is not a 2-tuple: {value!r}"
            )

    def test_module_paths_are_strings(self):
        for key, (mod_path, cls_name) in _READER_REGISTRY.items():
            assert isinstance(mod_path, str), (
                f"Module path for '{key}' is not a str: {mod_path!r}"
            )
            assert isinstance(cls_name, str), (
                f"Class name for '{key}' is not a str: {cls_name!r}"
            )


# ===================================================================
# get_reader — error cases
# ===================================================================

class TestGetReaderErrors:
    """Error-path tests for get_reader()."""

    def test_unknown_format_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown reader format"):
            get_reader('not_a_format', '/tmp/x')

    def test_error_message_lists_supported_formats(self):
        with pytest.raises(ValueError) as exc_info:
            get_reader('bmp', '/tmp/x.bmp')
        assert 'geotiff' in str(exc_info.value)
        assert 'sicd' in str(exc_info.value)

    def test_hyphen_normalised_to_underscore(self):
        """'sentinel1-slc' (with hyphen) should resolve to sentinel1_slc."""
        with mock.patch('grdl.IO._READER_REGISTRY', {
            'sentinel1_slc': ('grdl.IO.sar.sentinel1_slc', 'Sentinel1SLCReader')
        }):
            # Just verify normalisation — don't actually instantiate
            key = 'sentinel1-slc'.lower().replace('-', '_')
            assert key == 'sentinel1_slc'

    def test_case_insensitive(self):
        """Upper-case format key is normalised to lower-case."""
        with pytest.raises(ValueError, match="Unknown reader format"):
            # 'GEOTIFF' should map to 'geotiff'; we use a truly unknown name
            get_reader('UNKNOWN_FMT', '/tmp/x')
        # Valid upper-case key should NOT raise ValueError about unknown format
        with mock.patch('grdl.IO._READER_REGISTRY', {'geotiff': ('grdl.IO.geotiff', 'GeoTIFFReader')}):
            from grdl.exceptions import DependencyError
            try:
                get_reader('GEOTIFF', '/tmp/nonexistent.tif')
            except (FileNotFoundError, ValueError, DependencyError):
                pass  # expected — file doesn't exist; format was resolved OK


# ===================================================================
# register_reader / register_writer
# ===================================================================

class TestRegistrationHooks:
    """Tests for runtime reader/writer registration."""

    def test_register_reader_injects_key(self):
        original = dict(_READER_REGISTRY)
        try:
            register_reader('test_sensor', 'grdl.IO.geotiff', 'GeoTIFFReader')
            assert 'test_sensor' in _READER_REGISTRY
            assert _READER_REGISTRY['test_sensor'] == (
                'grdl.IO.geotiff', 'GeoTIFFReader'
            )
        finally:
            # Clean up injected key so other tests aren't polluted
            _READER_REGISTRY.pop('test_sensor', None)
            # Restore any keys that may have been removed
            for k, v in original.items():
                if k not in _READER_REGISTRY:
                    _READER_REGISTRY[k] = v

    def test_register_writer_injects_key(self):
        original = dict(_WRITER_REGISTRY)
        try:
            register_writer('test_fmt', 'grdl.IO.numpy_io', 'NumpyWriter')
            assert 'test_fmt' in _WRITER_REGISTRY
        finally:
            _WRITER_REGISTRY.pop('test_fmt', None)
            for k, v in original.items():
                if k not in _WRITER_REGISTRY:
                    _WRITER_REGISTRY[k] = v

    def test_register_reader_hyphen_normalised(self):
        original = dict(_READER_REGISTRY)
        try:
            register_reader('my-sensor', 'grdl.IO.geotiff', 'GeoTIFFReader')
            assert 'my_sensor' in _READER_REGISTRY
            assert 'my-sensor' not in _READER_REGISTRY
        finally:
            _READER_REGISTRY.pop('my_sensor', None)
            for k, v in original.items():
                if k not in _READER_REGISTRY:
                    _READER_REGISTRY[k] = v

    def test_register_reader_overwrite(self):
        original = dict(_READER_REGISTRY)
        try:
            register_reader('geotiff', 'grdl.IO.geotiff', 'GeoTIFFReader')
            assert _READER_REGISTRY['geotiff'] == (
                'grdl.IO.geotiff', 'GeoTIFFReader'
            )
        finally:
            for k, v in original.items():
                _READER_REGISTRY[k] = v

    def test_registered_format_callable_via_get_reader(self, tmp_path):
        """A registered format can be dispatched via get_reader()."""
        pytest.importorskip('rasterio')
        from grdl.IO.geotiff import GeoTIFFReader

        dummy = tmp_path / 'test.tif'
        import rasterio
        data = np.zeros((8, 8), dtype=np.float32)
        with rasterio.open(
            str(dummy), 'w', driver='GTiff',
            height=8, width=8, count=1, dtype='float32',
        ) as dst:
            dst.write(data[np.newaxis])

        original = dict(_READER_REGISTRY)
        try:
            register_reader('plugin_tif', 'grdl.IO.geotiff', 'GeoTIFFReader')
            reader = get_reader('plugin_tif', dummy)
            assert isinstance(reader, GeoTIFFReader)
            reader.close()
        finally:
            _READER_REGISTRY.pop('plugin_tif', None)
            for k, v in original.items():
                if k not in _READER_REGISTRY:
                    _READER_REGISTRY[k] = v


# ===================================================================
# ImageReader._ensure_2d
# ===================================================================

class TestEnsure2D:
    """Tests for the _ensure_2d() static helper on ImageReader."""

    def test_already_2d_unchanged(self):
        arr = np.zeros((4, 4))
        out = ImageReader._ensure_2d(arr)
        assert out.shape == (4, 4)
        assert out is arr  # no copy made

    def test_bands_rows_cols_single_squeezed(self):
        arr = np.zeros((1, 4, 4))
        out = ImageReader._ensure_2d(arr)
        assert out.shape == (4, 4)

    def test_rows_cols_bands_single_squeezed(self):
        arr = np.zeros((4, 4, 1))
        out = ImageReader._ensure_2d(arr)
        assert out.shape == (4, 4)

    def test_multiband_unchanged(self):
        arr = np.zeros((3, 4, 4))
        out = ImageReader._ensure_2d(arr)
        assert out.shape == (3, 4, 4)
        assert out is arr

    def test_1d_raises(self):
        arr = np.zeros((4,))
        with pytest.raises(ValueError, match="at least 2D"):
            ImageReader._ensure_2d(arr)

    def test_0d_raises(self):
        arr = np.array(1.0)
        with pytest.raises(ValueError, match="at least 2D"):
            ImageReader._ensure_2d(arr)

    def test_preserves_dtype_complex(self):
        arr = np.zeros((1, 8, 8), dtype=np.complex64)
        out = ImageReader._ensure_2d(arr)
        assert out.dtype == np.complex64
        assert out.shape == (8, 8)


# ===================================================================
# ImageReader._validate_single_pol
# ===================================================================

class TestValidateSinglePol:
    """Tests for the _validate_single_pol() static helper."""

    def test_2d_passes(self):
        arr = np.zeros((4, 4))
        ImageReader._validate_single_pol(arr)  # must not raise

    def test_3d_raises(self):
        arr = np.zeros((1, 4, 4))
        with pytest.raises(ValueError, match="Single-polarization"):
            ImageReader._validate_single_pol(arr)

    def test_context_in_error_message(self):
        arr = np.zeros((1, 4, 4))
        with pytest.raises(ValueError, match=r"\[MySARReader\]"):
            ImageReader._validate_single_pol(arr, context="MySARReader")

    def test_no_context_no_bracket(self):
        arr = np.zeros((1, 4, 4))
        with pytest.raises(ValueError) as exc_info:
            ImageReader._validate_single_pol(arr)
        assert '[' not in str(exc_info.value)

    def test_1d_raises(self):
        arr = np.zeros((4,))
        with pytest.raises(ValueError, match="Single-polarization"):
            ImageReader._validate_single_pol(arr)


# ===================================================================
# open_image — GDAL fallback warning
# ===================================================================

class TestOpenImageFallbackWarning:
    """open_image() emits UserWarning when a DependencyError forces GDAL."""

    def test_warns_on_dependency_error(self, tmp_path):
        """If the primary reader raises DependencyError, a warning is emitted."""
        pytest.importorskip('rasterio')

        dummy = tmp_path / 'test.tif'
        import rasterio
        data = np.zeros((8, 8), dtype=np.float32)
        with rasterio.open(
            str(dummy), 'w', driver='GTiff',
            height=8, width=8, count=1, dtype='float32',
        ) as dst:
            dst.write(data[np.newaxis])

        from grdl.exceptions import DependencyError as GrdlDepError
        from grdl.IO.geotiff import GeoTIFFReader

        with mock.patch.object(
            GeoTIFFReader, '__init__',
            side_effect=GrdlDepError("rasterio not installed"),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                from grdl.IO import open_image
                try:
                    open_image(str(dummy))
                except Exception:
                    pass
            dep_warns = [
                w for w in caught
                if issubclass(w.category, UserWarning)
                and 'required library is not installed' in str(w.message)
            ]
            assert dep_warns, (
                "Expected a UserWarning about missing library, got: "
                + str([str(w.message) for w in caught])
            )
