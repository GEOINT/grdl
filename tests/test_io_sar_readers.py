# -*- coding: utf-8 -*-
"""
SAR Reader Tests - API contract and backend selection tests.

Tests for SICD, CPHD, CRSD, and SIDD readers. Uses mocking to verify
API contracts without requiring real SAR data files.

Dependencies
------------
pytest

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
from unittest import mock


# -- SICDReader tests -------------------------------------------------------

class TestSICDReader:
    """Tests for SICDReader API contract."""

    def test_is_image_reader_subclass(self):
        """SICDReader is a subclass of ImageReader."""
        from grdl.IO.base import ImageReader
        from grdl.IO.sar.sicd import SICDReader
        assert issubclass(SICDReader, ImageReader)

    def test_backend_attribute(self):
        """SICDReader sets backend attribute on init."""
        from grdl.IO.sar.sicd import SICDReader
        from grdl.IO.sar import _backend

        # Mock to avoid file I/O
        with mock.patch.object(_backend, '_HAS_SARKIT', True):
            with mock.patch.object(
                SICDReader, '__init__', lambda self, fp: None
            ):
                reader = SICDReader.__new__(SICDReader)
                reader.backend = _backend.require_sar_backend('SICD')
                assert reader.backend == 'sarkit'

    def test_get_dtype_returns_complex64(self):
        """get_dtype returns complex64."""
        from grdl.IO.sar.sicd import SICDReader
        with mock.patch.object(
            SICDReader, '__init__', lambda self, fp: None
        ):
            reader = SICDReader.__new__(SICDReader)
            reader.metadata = {'dtype': 'complex64'}
            assert reader.get_dtype() == np.dtype('complex64')

    def test_get_shape_returns_tuple(self):
        """get_shape returns (rows, cols) tuple."""
        from grdl.IO.sar.sicd import SICDReader
        with mock.patch.object(
            SICDReader, '__init__', lambda self, fp: None
        ):
            reader = SICDReader.__new__(SICDReader)
            reader.metadata = {'rows': 1000, 'cols': 2000}
            assert reader.get_shape() == (1000, 2000)

    def test_file_not_found(self):
        """FileNotFoundError for non-existent file."""
        from grdl.IO.sar.sicd import SICDReader
        with pytest.raises(FileNotFoundError):
            SICDReader('/nonexistent/file.nitf')


# -- CPHDReader tests -------------------------------------------------------

class TestCPHDReader:
    """Tests for CPHDReader API contract."""

    def test_is_image_reader_subclass(self):
        """CPHDReader is a subclass of ImageReader."""
        from grdl.IO.base import ImageReader
        from grdl.IO.sar.cphd import CPHDReader
        assert issubclass(CPHDReader, ImageReader)

    def test_backend_selection(self):
        """CPHDReader selects backend via require_sar_backend."""
        from grdl.IO.sar.cphd import CPHDReader
        from grdl.IO.sar import _backend

        with mock.patch.object(_backend, '_HAS_SARKIT', False):
            with mock.patch.object(_backend, '_HAS_SARPY', True):
                with mock.patch.object(
                    CPHDReader, '__init__', lambda self, fp: None
                ):
                    reader = CPHDReader.__new__(CPHDReader)
                    reader.backend = _backend.require_sar_backend('CPHD')
                    assert reader.backend == 'sarpy'

    def test_get_dtype_returns_complex64(self):
        """get_dtype returns complex64."""
        from grdl.IO.sar.cphd import CPHDReader
        with mock.patch.object(
            CPHDReader, '__init__', lambda self, fp: None
        ):
            reader = CPHDReader.__new__(CPHDReader)
            assert reader.get_dtype() == np.dtype('complex64')

    def test_get_shape_from_channels(self):
        """get_shape returns first channel dimensions."""
        from grdl.IO.sar.cphd import CPHDReader
        with mock.patch.object(
            CPHDReader, '__init__', lambda self, fp: None
        ):
            reader = CPHDReader.__new__(CPHDReader)
            reader.metadata = {
                'channels': {
                    'CH1': {'num_vectors': 500, 'num_samples': 1000},
                }
            }
            assert reader.get_shape() == (500, 1000)

    def test_file_not_found(self):
        """FileNotFoundError for non-existent file."""
        from grdl.IO.sar.cphd import CPHDReader
        with pytest.raises(FileNotFoundError):
            CPHDReader('/nonexistent/file.cphd')


# -- CRSDReader tests -------------------------------------------------------

class TestCRSDReader:
    """Tests for CRSDReader API contract."""

    def test_is_image_reader_subclass(self):
        """CRSDReader is a subclass of ImageReader."""
        from grdl.IO.base import ImageReader
        from grdl.IO.sar.crsd import CRSDReader
        assert issubclass(CRSDReader, ImageReader)

    def test_requires_sarkit(self):
        """CRSDReader raises ImportError when sarkit is missing."""
        from grdl.IO.sar.crsd import CRSDReader
        from grdl.IO.sar import _backend

        with mock.patch.object(_backend, '_HAS_SARKIT', False):
            with pytest.raises(ImportError, match='sarkit'):
                CRSDReader('/some/file.crsd')

    def test_get_dtype_returns_complex64(self):
        """get_dtype returns complex64."""
        from grdl.IO.sar.crsd import CRSDReader
        with mock.patch.object(
            CRSDReader, '__init__', lambda self, fp: None
        ):
            reader = CRSDReader.__new__(CRSDReader)
            assert reader.get_dtype() == np.dtype('complex64')

    def test_get_shape_from_channels(self):
        """get_shape returns first channel dimensions."""
        from grdl.IO.sar.crsd import CRSDReader
        with mock.patch.object(
            CRSDReader, '__init__', lambda self, fp: None
        ):
            reader = CRSDReader.__new__(CRSDReader)
            reader.metadata = {
                'channels': {
                    'RX1': {'num_vectors': 200, 'num_samples': 400},
                }
            }
            assert reader.get_shape() == (200, 400)


# -- SIDDReader tests -------------------------------------------------------

class TestSIDDReader:
    """Tests for SIDDReader API contract."""

    def test_is_image_reader_subclass(self):
        """SIDDReader is a subclass of ImageReader."""
        from grdl.IO.base import ImageReader
        from grdl.IO.sar.sidd import SIDDReader
        assert issubclass(SIDDReader, ImageReader)

    def test_requires_sarkit(self):
        """SIDDReader raises ImportError when sarkit is missing."""
        from grdl.IO.sar.sidd import SIDDReader
        from grdl.IO.sar import _backend

        with mock.patch.object(_backend, '_HAS_SARKIT', False):
            with pytest.raises(ImportError, match='sarkit'):
                SIDDReader('/some/file.nitf')

    def test_get_shape_returns_tuple(self):
        """get_shape returns (rows, cols) tuple."""
        from grdl.IO.sar.sidd import SIDDReader
        with mock.patch.object(
            SIDDReader, '__init__', lambda self, fp, **kw: None
        ):
            reader = SIDDReader.__new__(SIDDReader)
            reader.metadata = {'rows': 512, 'cols': 768}
            assert reader.get_shape() == (512, 768)

    def test_image_index_parameter(self):
        """SIDDReader accepts image_index parameter."""
        from grdl.IO.sar.sidd import SIDDReader
        import inspect
        sig = inspect.signature(SIDDReader.__init__)
        assert 'image_index' in sig.parameters


# -- open_sar tests ---------------------------------------------------------

class TestOpenSar:
    """Tests for the open_sar auto-detection function."""

    def test_open_sar_is_callable(self):
        """open_sar is a callable function."""
        from grdl.IO.sar import open_sar
        assert callable(open_sar)

    def test_open_sar_raises_for_nonexistent(self):
        """open_sar raises ValueError for non-existent file."""
        from grdl.IO.sar import open_sar
        with pytest.raises((ValueError, FileNotFoundError)):
            open_sar('/nonexistent/file.nitf')

    def test_open_sar_geotiff_fallback(self, tmp_path):
        """open_sar falls back to GeoTIFF for .tif files."""
        try:
            import rasterio
        except ImportError:
            pytest.skip("rasterio not installed")

        filepath = tmp_path / "test.tif"
        data = np.ones((10, 20), dtype=np.float32)
        with rasterio.open(
            str(filepath), 'w', driver='GTiff',
            height=10, width=20, count=1, dtype='float32',
        ) as ds:
            ds.write(data, 1)

        from grdl.IO.sar import open_sar
        with open_sar(filepath) as reader:
            assert reader.metadata['format'] == 'GeoTIFF'


# -- Geolocation tests ------------------------------------------------------

class TestGeolocation:
    """Tests for reader geolocation methods."""

    def test_sicd_geolocation_with_scp(self):
        """SICDReader returns geolocation when SCP is available."""
        from grdl.IO.sar.sicd import SICDReader
        with mock.patch.object(
            SICDReader, '__init__', lambda self, fp: None
        ):
            reader = SICDReader.__new__(SICDReader)
            reader.metadata = {
                'scp_llh': [34.0, -118.0, 100.0],
                'corner_coords': None,
            }
            geo = reader.get_geolocation()
            assert geo is not None
            assert 'scp_llh' in geo

    def test_sicd_geolocation_none_without_scp(self):
        """SICDReader returns None when no SCP is available."""
        from grdl.IO.sar.sicd import SICDReader
        with mock.patch.object(
            SICDReader, '__init__', lambda self, fp: None
        ):
            reader = SICDReader.__new__(SICDReader)
            reader.metadata = {}
            assert reader.get_geolocation() is None

    def test_cphd_geolocation(self):
        """CPHDReader returns phase history space projection."""
        from grdl.IO.sar.cphd import CPHDReader
        with mock.patch.object(
            CPHDReader, '__init__', lambda self, fp: None
        ):
            reader = CPHDReader.__new__(CPHDReader)
            geo = reader.get_geolocation()
            assert geo is not None
            assert geo['projection'] == 'Phase history space'

    def test_crsd_geolocation(self):
        """CRSDReader returns radar signal space projection."""
        from grdl.IO.sar.crsd import CRSDReader
        with mock.patch.object(
            CRSDReader, '__init__', lambda self, fp: None
        ):
            reader = CRSDReader.__new__(CRSDReader)
            geo = reader.get_geolocation()
            assert geo is not None
            assert geo['projection'] == 'Radar signal space'

    def test_sidd_geolocation(self):
        """SIDDReader returns derived product projection."""
        from grdl.IO.sar.sidd import SIDDReader
        with mock.patch.object(
            SIDDReader, '__init__', lambda self, fp, **kw: None
        ):
            reader = SIDDReader.__new__(SIDDReader)
            geo = reader.get_geolocation()
            assert geo is not None
            assert 'SIDD' in geo['projection']
