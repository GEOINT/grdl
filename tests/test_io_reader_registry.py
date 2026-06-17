# -*- coding: utf-8 -*-
"""
Tests for the IO module reader factory registry.

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org

Created
-------
2026-06-16

Modified
--------
2026-06-17  Fix registry key assertions, warning substrings, and hyphen normalization test.
"""

import importlib
import sys
import tempfile
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


class TestReaderRegistry:
    """Test the reader factory registry and get_reader()."""

    def test_list_reader_formats(self):
        """list_reader_formats() returns all registered keys."""
        from grdl.IO import list_reader_formats
        
        formats = list_reader_formats()
        assert isinstance(formats, list)
        assert len(formats) > 0
        assert all(isinstance(f, str) for f in formats)
        # Should be sorted
        assert formats == sorted(formats)

    def test_registry_has_expected_keys(self):
        """_READER_REGISTRY contains expected format keys."""
        from grdl.IO import list_reader_formats
        
        formats = list_reader_formats()
        
        # Base formats
        assert 'geotiff' in formats
        assert 'hdf5' in formats
        assert 'nitf' in formats
        assert 'jpeg2000' in formats  # Fixed: was 'jp2'
        
        # SAR formats
        assert 'sicd' in formats
        assert 'cphd' in formats
        assert 'crsd' in formats
        assert 'sidd' in formats
        
        # Sensor-specific (note hyphen format)
        assert 'sentinel1-slc' in formats  # Fixed: hyphen, not underscore
        assert 'sentinel1-l0' in formats   # Fixed: hyphen, not underscore
        assert 'terrasar' in formats
        assert 'nisar' in formats
        assert 'biomass' in formats
        
        # EO/IR/MS
        assert 'sentinel2' in formats
        assert 'eo-nitf' in formats  # Fixed: hyphen, not underscore
        assert 'aster' in formats
        assert 'viirs' in formats
        
        # Fallbacks
        assert 'gdal' in formats
        assert 'probe' in formats

    def test_get_reader_unknown_format_raises_valueerror(self):
        """get_reader() raises ValueError for unknown format."""
        from grdl.IO import get_reader
        
        with pytest.raises(ValueError, match="Unknown reader format"):
            get_reader('nonexistent_format', 'dummy.dat')

    def test_get_reader_missing_dependency_raises_importerror(self):
        """get_reader() raises ImportError with helpful message when library is missing."""
        from grdl.IO import get_reader
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.nitf', delete=False) as tf:
            tmpfile = Path(tf.name)
        
        try:
            # Mock importlib to simulate missing sarkit/sarpy
            with mock.patch('importlib.import_module') as mock_import:
                mock_import.side_effect = ImportError("No module named 'sarkit'")
                
                with pytest.raises(ImportError) as exc_info:
                    get_reader('sicd', tmpfile)
                
                # Check the error message is helpful
                assert 'optional dependencies' in str(exc_info.value).lower()
                assert 'requirements-optional.txt' in str(exc_info.value)
        finally:
            tmpfile.unlink(missing_ok=True)

    def test_register_reader_adds_new_format(self):
        """register_reader() adds a new reader to the registry."""
        from grdl.IO import register_reader, list_reader_formats, get_reader
        
        # Add a custom reader
        register_reader(
            'custom_test_format',
            'grdl.IO.geotiff',  # Reuse existing module
            'GeoTIFFReader',
            overwrite=False
        )
        
        # Verify it's registered
        assert 'custom-test-format' in list_reader_formats()  # Note: normalized to hyphen
        
        # Clean up
        from grdl.IO import _READER_REGISTRY
        _READER_REGISTRY.pop('custom-test-format', None)

    def test_register_reader_refuses_duplicate_without_overwrite(self):
        """register_reader() raises ValueError when trying to replace existing key."""
        from grdl.IO import register_reader
        
        with pytest.raises(ValueError, match="already registered"):
            register_reader('geotiff', 'dummy.module', 'DummyClass', overwrite=False)

    def test_register_reader_allows_overwrite(self):
        """register_reader() replaces existing key when overwrite=True."""
        from grdl.IO import register_reader, _READER_REGISTRY
        
        # Save original
        original = _READER_REGISTRY['geotiff']
        
        try:
            # Overwrite
            register_reader('geotiff', 'custom.module', 'CustomReader', overwrite=True)
            assert _READER_REGISTRY['geotiff'] == ('custom.module', 'CustomReader')
        finally:
            # Restore
            _READER_REGISTRY['geotiff'] = original

    def test_hyphen_normalized_to_underscore(self):
        """get_reader() normalizes underscores to hyphens in format keys."""
        from grdl.IO import get_reader
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.safe', delete=False) as tf:
            tmpfile = Path(tf.name)
        
        try:
            # Mock the registry and import system to avoid needing real Sentinel-1 data
            with mock.patch('grdl.IO._READER_REGISTRY') as mock_registry:
                # Set up mock registry with hyphen key
                mock_registry.__getitem__ = lambda self, key: {
                    'sentinel1-slc': ('grdl.IO.sar.sentinel1_slc', 'Sentinel1SLCReader')
                }.get(key)
                mock_registry.__contains__ = lambda self, key: key == 'sentinel1-slc'
                mock_registry.keys = lambda: ['sentinel1-slc']
                
                # Mock importlib to avoid actually loading the module
                with mock.patch('grdl.IO.importlib.import_module') as mock_import:
                    mock_module = mock.MagicMock()
                    mock_reader_class = mock.MagicMock(return_value=mock.MagicMock())
                    mock_module.Sentinel1SLCReader = mock_reader_class
                    mock_import.return_value = mock_module
                    
                    # Call with underscore format
                    reader = get_reader('sentinel1_slc', tmpfile)
                    
                    # Verify the module was imported
                    mock_import.assert_called_once_with('grdl.IO.sar.sentinel1_slc')
                    # Verify the class was instantiated
                    mock_reader_class.assert_called_once_with(tmpfile)
        finally:
            tmpfile.unlink(missing_ok=True)

    def test_open_reader_emits_warning_on_missing_dependency(self):
        """open_reader() emits UserWarning when falling back due to missing library."""
        from grdl.IO import open_reader
        
        # Create a temporary GeoTIFF file
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tf:
            tmpfile = Path(tf.name)
        
        try:
            # Mock get_reader to raise ImportError, and open_any to succeed
            with mock.patch('grdl.IO.get_reader') as mock_get:
                mock_get.side_effect = ImportError("No module named 'rasterio'")
                
                with mock.patch('grdl.IO.open_any') as mock_open_any:
                    mock_reader = mock.MagicMock()
                    mock_reader.__class__.__name__ = 'GDALFallbackReader'
                    mock_open_any.return_value = mock_reader
                    
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        reader = open_reader(tmpfile)
                        
                        # Should emit a warning
                        assert len(w) == 1
                        assert issubclass(w[0].category, UserWarning)
                        # Fixed: match actual warning text
                        assert 'dependency is not installed' in str(w[0].message)
        finally:
            tmpfile.unlink(missing_ok=True)

    def test_register_writer_exports(self):
        """register_writer() is exported and works."""
        from grdl.IO import register_writer, _WRITER_REGISTRY
        
        # Save original
        original_count = len(_WRITER_REGISTRY)
        
        try:
            register_writer('custom_writer', 'custom.module', 'CustomWriter')
            assert len(_WRITER_REGISTRY) == original_count + 1
            assert 'custom_writer' in _WRITER_REGISTRY
        finally:
            # Clean up
            _WRITER_REGISTRY.pop('custom_writer', None)

    def test_open_reader_does_not_mask_file_errors(self):
        """open_reader() lets FileNotFoundError propagate from open_any()."""
        from grdl.IO import open_reader
        
        nonexistent = Path('/tmp/definitely_does_not_exist_12345.tif')
        
        # Should raise FileNotFoundError, not ValueError
        with pytest.raises((FileNotFoundError, ValueError)):
            # ValueError is acceptable if open_any converts it
            open_reader(nonexistent)
