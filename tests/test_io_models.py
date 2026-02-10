# -*- coding: utf-8 -*-
"""
IO Models Tests - Unit tests for ImageMetadata dataclass.

Verifies typed attribute access, dict-like backward compatibility,
extras handling, serialization, and subclass field discovery.

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

# Standard library
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Third-party
import pytest

# GRDL internal
from grdl.IO.models import ImageMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_meta():
    """ImageMetadata with only required fields."""
    return ImageMetadata(
        format='GeoTIFF',
        rows=1024,
        cols=2048,
        dtype='float32',
    )


@pytest.fixture
def full_meta():
    """ImageMetadata with all fields populated."""
    return ImageMetadata(
        format='GeoTIFF',
        rows=1024,
        cols=2048,
        dtype='float32',
        bands=3,
        crs='EPSG:4326',
        nodata=-9999.0,
        extras={
            'transform': 'affine_obj',
            'resolution': (10.0, 10.0),
        },
    )


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------

def test_required_fields(basic_meta):
    """Required fields accessible as attributes."""
    assert basic_meta.format == 'GeoTIFF'
    assert basic_meta.rows == 1024
    assert basic_meta.cols == 2048
    assert basic_meta.dtype == 'float32'


# ---------------------------------------------------------------------------
# Optional fields
# ---------------------------------------------------------------------------

def test_optional_fields_default_none(basic_meta):
    """Optional fields default to None."""
    assert basic_meta.bands is None
    assert basic_meta.crs is None
    assert basic_meta.nodata is None


def test_optional_fields_set(full_meta):
    """Optional fields set correctly."""
    assert full_meta.bands == 3
    assert full_meta.crs == 'EPSG:4326'
    assert full_meta.nodata == -9999.0


def test_extras_default_empty():
    """Extras defaults to empty dict."""
    meta = ImageMetadata(format='X', rows=1, cols=1, dtype='uint8')
    assert meta.extras == {}


# ---------------------------------------------------------------------------
# Dict-like access: __getitem__
# ---------------------------------------------------------------------------

def test_dict_access_typed_field(full_meta):
    """Dict access resolves typed fields."""
    assert full_meta['rows'] == 1024
    assert full_meta['crs'] == 'EPSG:4326'


def test_dict_access_extras(full_meta):
    """Dict access resolves extras keys."""
    assert full_meta['transform'] == 'affine_obj'
    assert full_meta['resolution'] == (10.0, 10.0)


def test_dict_access_keyerror(basic_meta):
    """Dict access raises KeyError for missing keys."""
    with pytest.raises(KeyError):
        basic_meta['missing_key']


# ---------------------------------------------------------------------------
# Dict-like access: __setitem__
# ---------------------------------------------------------------------------

def test_setitem_typed_field(basic_meta):
    """Setting a typed field via dict syntax updates attribute."""
    basic_meta['bands'] = 4
    assert basic_meta.bands == 4
    assert basic_meta['bands'] == 4


def test_setitem_extras(basic_meta):
    """Setting unknown key goes into extras."""
    basic_meta['custom_key'] = 'custom_value'
    assert basic_meta.extras['custom_key'] == 'custom_value'
    assert basic_meta['custom_key'] == 'custom_value'


# ---------------------------------------------------------------------------
# Dict-like access: __contains__
# ---------------------------------------------------------------------------

def test_contains_typed_field(full_meta):
    """Contains returns True for non-None typed fields."""
    assert 'rows' in full_meta
    assert 'crs' in full_meta


def test_contains_optional_none(basic_meta):
    """Contains returns False for None-valued optional fields."""
    assert 'crs' not in basic_meta
    assert 'nodata' not in basic_meta


def test_contains_extras(full_meta):
    """Contains checks extras dict."""
    assert 'transform' in full_meta
    assert 'nonexistent' not in full_meta


# ---------------------------------------------------------------------------
# Dict-like access: get()
# ---------------------------------------------------------------------------

def test_get_typed_field(full_meta):
    """get() returns typed field values."""
    assert full_meta.get('rows') == 1024


def test_get_extras(full_meta):
    """get() returns extras values."""
    assert full_meta.get('transform') == 'affine_obj'


def test_get_with_default(basic_meta):
    """get() returns default for missing keys."""
    assert basic_meta.get('missing', 42) == 42


def test_get_none_field_returns_default(basic_meta):
    """get() returns default for None-valued typed fields."""
    assert basic_meta.get('crs', 'default_crs') == 'default_crs'


# ---------------------------------------------------------------------------
# keys()
# ---------------------------------------------------------------------------

def test_keys_basic(basic_meta):
    """keys() includes non-None typed fields only."""
    k = basic_meta.keys()
    assert 'format' in k
    assert 'rows' in k
    assert 'cols' in k
    assert 'dtype' in k
    assert 'crs' not in k
    assert 'nodata' not in k


def test_keys_with_extras(full_meta):
    """keys() includes both typed fields and extras."""
    k = full_meta.keys()
    assert 'format' in k
    assert 'bands' in k
    assert 'transform' in k
    assert 'resolution' in k


# ---------------------------------------------------------------------------
# to_dict() / from_dict()
# ---------------------------------------------------------------------------

def test_values(full_meta):
    """values() returns values matching keys() order."""
    k = full_meta.keys()
    v = full_meta.values()
    assert len(k) == len(v)
    assert v[0] == 'GeoTIFF'  # format
    assert (10.0, 10.0) in v  # resolution from extras


def test_items(full_meta):
    """items() returns (key, value) pairs."""
    items = full_meta.items()
    item_dict = dict(items)
    assert item_dict['format'] == 'GeoTIFF'
    assert item_dict['rows'] == 1024
    assert item_dict['transform'] == 'affine_obj'


def test_to_dict(full_meta):
    """to_dict() returns flat dictionary with extras merged."""
    d = full_meta.to_dict()
    assert d['format'] == 'GeoTIFF'
    assert d['rows'] == 1024
    assert d['bands'] == 3
    assert d['transform'] == 'affine_obj'
    assert 'extras' not in d


def test_to_dict_excludes_none(basic_meta):
    """to_dict() excludes None-valued optional fields."""
    d = basic_meta.to_dict()
    assert 'crs' not in d
    assert 'nodata' not in d


def test_from_dict():
    """from_dict() constructs ImageMetadata from plain dict."""
    d = {
        'format': 'HDF5',
        'rows': 512,
        'cols': 512,
        'dtype': 'int16',
        'bands': 1,
        'dataset_path': '/data/NDVI',
    }
    meta = ImageMetadata.from_dict(d)
    assert meta.format == 'HDF5'
    assert meta.rows == 512
    assert meta.bands == 1
    assert meta['dataset_path'] == '/data/NDVI'


def test_from_dict_roundtrip(full_meta):
    """from_dict(to_dict()) preserves data."""
    d = full_meta.to_dict()
    restored = ImageMetadata.from_dict(d)
    assert restored.format == full_meta.format
    assert restored.rows == full_meta.rows
    assert restored.bands == full_meta.bands
    assert restored['transform'] == full_meta['transform']


# ---------------------------------------------------------------------------
# __iter__ / __len__
# ---------------------------------------------------------------------------

def test_iter(full_meta):
    """Iterating yields keys, enabling dict(meta) compatibility."""
    keys = list(full_meta)
    assert 'format' in keys
    assert 'transform' in keys


def test_dict_conversion(full_meta):
    """dict(meta) works via __iter__ + __getitem__."""
    d = dict(full_meta)
    assert d['format'] == 'GeoTIFF'
    assert d['transform'] == 'affine_obj'


def test_len(basic_meta, full_meta):
    """len() returns number of available keys."""
    assert len(basic_meta) == 4  # format, rows, cols, dtype
    assert len(full_meta) == 9  # + bands, crs, nodata, transform, resolution


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

def test_repr(basic_meta):
    """repr includes class name and field values."""
    r = repr(basic_meta)
    assert 'ImageMetadata' in r
    assert 'GeoTIFF' in r
    assert '1024' in r


# ---------------------------------------------------------------------------
# Subclass field discovery
# ---------------------------------------------------------------------------

@dataclass
class MockSensorMetadata(ImageMetadata):
    """Test subclass with additional typed fields."""
    backend: Optional[str] = None
    pixel_type: Optional[str] = None


def test_subclass_fields_in_dict_access():
    """Subclass typed fields accessible via dict-like access."""
    meta = MockSensorMetadata(
        format='SICD',
        rows=100,
        cols=200,
        dtype='complex64',
        backend='sarkit',
        pixel_type='RE32F_IM32F',
    )
    assert meta['backend'] == 'sarkit'
    assert meta['pixel_type'] == 'RE32F_IM32F'
    assert 'backend' in meta
    assert meta.get('backend') == 'sarkit'


def test_subclass_none_fields_not_in_contains():
    """Subclass None-valued typed fields return False for 'in'."""
    meta = MockSensorMetadata(
        format='SICD',
        rows=100,
        cols=200,
        dtype='complex64',
    )
    assert 'backend' not in meta
    assert meta.get('backend', 'default') == 'default'


def test_subclass_keys_include_typed():
    """keys() discovers subclass typed fields."""
    meta = MockSensorMetadata(
        format='SICD',
        rows=100,
        cols=200,
        dtype='complex64',
        backend='sarkit',
    )
    k = meta.keys()
    assert 'backend' in k
    assert 'pixel_type' not in k  # None, so excluded


def test_subclass_from_dict():
    """from_dict() on subclass extracts subclass typed fields."""
    d = {
        'format': 'SICD',
        'rows': 100,
        'cols': 200,
        'dtype': 'complex64',
        'backend': 'sarkit',
        'collector_name': 'SENSOR_X',
    }
    meta = MockSensorMetadata.from_dict(d)
    assert meta.backend == 'sarkit'
    assert meta['collector_name'] == 'SENSOR_X'


# ---------------------------------------------------------------------------
# Extras dict independence
# ---------------------------------------------------------------------------

def test_extras_dict_is_independent():
    """Each instance gets its own extras dict."""
    meta1 = ImageMetadata(format='A', rows=1, cols=1, dtype='uint8')
    meta2 = ImageMetadata(format='B', rows=2, cols=2, dtype='uint8')
    meta1.extras['key1'] = 'val1'
    assert 'key1' not in meta2.extras


# ---------------------------------------------------------------------------
# Import from grdl.IO
# ---------------------------------------------------------------------------

def test_import_from_io_package():
    """ImageMetadata importable from grdl.IO."""
    from grdl.IO import ImageMetadata as IM
    assert IM is ImageMetadata
