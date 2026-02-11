# -*- coding: utf-8 -*-
"""
SAR Reader Tests - Error handling and integration tests.

Tests real error paths and integration behavior for SICD, CPHD, CRSD,
and SIDD readers.  Mock-heavy getter tests that bypass __init__ and
verify manually-set attributes have been removed.

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
2026-02-10
"""

import pytest
import numpy as np
from unittest import mock


# -- SICDReader tests -------------------------------------------------------

class TestSICDReader:
    """Tests for SICDReader error handling."""

    def test_file_not_found(self):
        """FileNotFoundError for non-existent file."""
        from grdl.IO.sar.sicd import SICDReader
        with pytest.raises(FileNotFoundError):
            SICDReader('/nonexistent/file.nitf')


# -- CPHDReader tests -------------------------------------------------------

class TestCPHDReader:
    """Tests for CPHDReader error handling."""

    def test_file_not_found(self):
        """FileNotFoundError for non-existent file."""
        from grdl.IO.sar.cphd import CPHDReader
        with pytest.raises(FileNotFoundError):
            CPHDReader('/nonexistent/file.cphd')


# -- CRSDReader tests -------------------------------------------------------

class TestCRSDReader:
    """Tests for CRSDReader error handling."""

    def test_requires_sarkit(self):
        """CRSDReader raises ImportError when sarkit is missing."""
        from grdl.IO.sar.crsd import CRSDReader
        from grdl.IO.sar import _backend

        with mock.patch.object(_backend, '_HAS_SARKIT', False):
            with pytest.raises(ImportError, match='sarkit'):
                CRSDReader('/some/file.crsd')


# -- SIDDReader tests -------------------------------------------------------

class TestSIDDReader:
    """Tests for SIDDReader error handling."""

    def test_requires_sarkit(self):
        """SIDDReader raises ImportError when sarkit is missing."""
        from grdl.IO.sar.sidd import SIDDReader
        from grdl.IO.sar import _backend

        with mock.patch.object(_backend, '_HAS_SARKIT', False):
            with pytest.raises(ImportError, match='sarkit'):
                SIDDReader('/some/file.nitf')


# -- open_sar tests ---------------------------------------------------------

class TestOpenSar:
    """Tests for the open_sar auto-detection function."""

    def test_open_sar_raises_for_nonexistent(self):
        """open_sar raises ValueError for non-existent file."""
        from grdl.IO.sar import open_sar
        with pytest.raises((ValueError, FileNotFoundError)):
            open_sar('/nonexistent/file.nitf')

    def test_open_sar_geotiff_fallback(self, tmp_path):
        """open_sar falls back to GeoTIFF for .tif files."""
        try:
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            pytest.skip("rasterio not installed")

        filepath = tmp_path / "test.tif"
        data = np.ones((10, 20), dtype=np.float32)

        transform = from_bounds(-180, -90, 180, 90, 20, 10)

        with rasterio.open(
            str(filepath), 'w', driver='GTiff',
            height=10, width=20, count=1, dtype='float32',
            transform=transform,
            crs='EPSG:4326',
        ) as ds:
            ds.write(data, 1)

        from grdl.IO.sar import open_sar
        with open_sar(filepath) as reader:
            assert reader.metadata['format'] == 'GeoTIFF'
