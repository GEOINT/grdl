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
import numpy as np
import pytest

# GRDL internal
from grdl.IO.models import (
    ImageMetadata,
    SICDMetadata,
    SIDDMetadata,
    BIOMASSMetadata,
    XYZ,
    LatLon,
    LatLonHAE,
    RowCol,
    Poly1D,
    Poly2D,
    XYZPoly,
    SICDCollectionInfo,
    SICDRadarMode,
    SICDImageData,
    SICDFullImage,
    SICDSCP,
    SICDGeoData,
    SICDGrid,
    SICDDirParam,
    SICDSCPCOA,
    SIDDProductCreation,
    SIDDProcessorInformation,
    SIDDClassification,
    SIDDDisplay,
    SIDDMeasurement,
    SIDDExploitationFeatures,
)


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


# ---------------------------------------------------------------------------
# Common primitive types
# ---------------------------------------------------------------------------

class TestCommonTypes:
    """Tests for shared primitive dataclasses."""

    def test_xyz_defaults(self):
        """XYZ defaults to zeros."""
        p = XYZ()
        assert p.x == 0.0
        assert p.y == 0.0
        assert p.z == 0.0

    def test_xyz_with_values(self):
        """XYZ stores values."""
        p = XYZ(x=1.0, y=2.0, z=3.0)
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.z == 3.0

    def test_latlon(self):
        """LatLon stores lat/lon."""
        ll = LatLon(lat=34.0, lon=-118.0)
        assert ll.lat == 34.0
        assert ll.lon == -118.0

    def test_latlonhae(self):
        """LatLonHAE stores lat/lon/hae."""
        llh = LatLonHAE(lat=34.0, lon=-118.0, hae=100.0)
        assert llh.hae == 100.0

    def test_rowcol(self):
        """RowCol stores row/col."""
        rc = RowCol(row=10.5, col=20.5)
        assert rc.row == 10.5
        assert rc.col == 20.5

    def test_poly1d(self):
        """Poly1D stores coefficient array."""
        coefs = np.array([1.0, 2.0, 3.0])
        p = Poly1D(coefs=coefs)
        np.testing.assert_array_equal(p.coefs, coefs)

    def test_poly1d_none(self):
        """Poly1D defaults to None."""
        p = Poly1D()
        assert p.coefs is None

    def test_poly2d(self):
        """Poly2D stores 2D coefficient array."""
        coefs = np.array([[1.0, 2.0], [3.0, 4.0]])
        p = Poly2D(coefs=coefs)
        assert p.coefs.shape == (2, 2)

    def test_xyzpoly(self):
        """XYZPoly stores three Poly1D components."""
        xp = XYZPoly(
            x=Poly1D(coefs=np.array([1.0])),
            y=Poly1D(coefs=np.array([2.0])),
            z=Poly1D(coefs=np.array([3.0])),
        )
        assert xp.x.coefs[0] == 1.0
        assert xp.y.coefs[0] == 2.0
        assert xp.z.coefs[0] == 3.0


# ---------------------------------------------------------------------------
# SICDMetadata
# ---------------------------------------------------------------------------

class TestSICDMetadata:
    """Tests for SICDMetadata typed subclass."""

    @pytest.fixture
    def sicd_meta(self):
        """Minimal SICDMetadata for testing."""
        return SICDMetadata(
            format='SICD',
            rows=1000,
            cols=2000,
            dtype='complex64',
            backend='sarkit',
            collection_info=SICDCollectionInfo(
                collector_name='SENSOR_X',
                core_name='CORE_001',
                radar_mode=SICDRadarMode(
                    mode_type='SPOTLIGHT',
                    mode_id='SP1',
                ),
                classification='UNCLASSIFIED',
            ),
            image_data=SICDImageData(
                pixel_type='RE32F_IM32F',
                num_rows=1000,
                num_cols=2000,
                first_row=0,
                first_col=0,
                full_image=SICDFullImage(num_rows=1000, num_cols=2000),
                scp_pixel=RowCol(row=500.0, col=1000.0),
            ),
            geo_data=SICDGeoData(
                earth_model='WGS_84',
                scp=SICDSCP(
                    ecf=XYZ(x=-2414827.0, y=-4730767.0, z=3543746.0),
                    llh=LatLonHAE(lat=33.9, lon=-117.2, hae=350.0),
                ),
                image_corners=[
                    LatLon(lat=33.8, lon=-117.3),
                    LatLon(lat=33.8, lon=-117.1),
                    LatLon(lat=34.0, lon=-117.1),
                    LatLon(lat=34.0, lon=-117.3),
                ],
            ),
            scpcoa=SICDSCPCOA(
                scp_time=0.5,
                slant_range=600000.0,
                graze_ang=45.0,
                incidence_ang=45.0,
                side_of_track='L',
            ),
        )

    def test_inherits_image_metadata(self):
        """SICDMetadata is subclass of ImageMetadata."""
        assert issubclass(SICDMetadata, ImageMetadata)

    def test_required_fields(self, sicd_meta):
        """Base fields accessible."""
        assert sicd_meta.format == 'SICD'
        assert sicd_meta.rows == 1000
        assert sicd_meta.cols == 2000
        assert sicd_meta.dtype == 'complex64'

    def test_backend(self, sicd_meta):
        """Backend field accessible."""
        assert sicd_meta.backend == 'sarkit'

    def test_nested_collection_info(self, sicd_meta):
        """Nested collection_info accessible."""
        ci = sicd_meta.collection_info
        assert ci.collector_name == 'SENSOR_X'
        assert ci.core_name == 'CORE_001'
        assert ci.radar_mode.mode_type == 'SPOTLIGHT'
        assert ci.radar_mode.mode_id == 'SP1'
        assert ci.classification == 'UNCLASSIFIED'

    def test_nested_image_data(self, sicd_meta):
        """Nested image_data accessible."""
        assert sicd_meta.image_data.pixel_type == 'RE32F_IM32F'
        assert sicd_meta.image_data.full_image.num_rows == 1000
        assert sicd_meta.image_data.scp_pixel.row == 500.0

    def test_nested_geo_data(self, sicd_meta):
        """Nested geo_data with SCP accessible."""
        scp = sicd_meta.geo_data.scp
        assert scp.ecf.x == pytest.approx(-2414827.0)
        assert scp.llh.lat == pytest.approx(33.9)
        assert len(sicd_meta.geo_data.image_corners) == 4

    def test_nested_scpcoa(self, sicd_meta):
        """Nested SCPCOA accessible."""
        assert sicd_meta.scpcoa.graze_ang == pytest.approx(45.0)
        assert sicd_meta.scpcoa.side_of_track == 'L'

    def test_optional_sections_default_none(self, sicd_meta):
        """Unset optional sections default to None."""
        assert sicd_meta.grid is None
        assert sicd_meta.timeline is None
        assert sicd_meta.position is None
        assert sicd_meta.antenna is None
        assert sicd_meta.error_statistics is None
        assert sicd_meta.rma is None

    def test_dict_access_nested(self, sicd_meta):
        """Dict-like access resolves subclass fields."""
        assert sicd_meta['backend'] == 'sarkit'
        assert 'collection_info' in sicd_meta
        assert 'grid' not in sicd_meta  # None

    def test_keys_includes_sicd_fields(self, sicd_meta):
        """keys() includes populated SICD-specific fields."""
        k = sicd_meta.keys()
        assert 'backend' in k
        assert 'collection_info' in k
        assert 'image_data' in k
        assert 'geo_data' in k
        assert 'scpcoa' in k
        # None sections excluded
        assert 'grid' not in k
        assert 'antenna' not in k

    def test_grid_deep_nesting(self):
        """Grid with DirParam deeply nested."""
        meta = SICDMetadata(
            format='SICD',
            rows=100,
            cols=200,
            dtype='complex64',
            grid=SICDGrid(
                image_plane='SLANT',
                type='RGAZIM',
                row=SICDDirParam(ss=0.5, imp_resp_wid=0.6, sgn=-1),
                col=SICDDirParam(ss=0.5, imp_resp_wid=0.7, sgn=-1),
            ),
        )
        assert meta.grid.row.ss == pytest.approx(0.5)
        assert meta.grid.col.imp_resp_wid == pytest.approx(0.7)
        assert meta.grid.image_plane == 'SLANT'


# ---------------------------------------------------------------------------
# SIDDMetadata
# ---------------------------------------------------------------------------

class TestSIDDMetadata:
    """Tests for SIDDMetadata typed subclass."""

    @pytest.fixture
    def sidd_meta(self):
        """Minimal SIDDMetadata for testing."""
        return SIDDMetadata(
            format='SIDD',
            rows=512,
            cols=768,
            dtype='uint8',
            backend='sarkit',
            num_images=1,
            image_index=0,
            product_creation=SIDDProductCreation(
                processor_information=SIDDProcessorInformation(
                    application='TestProcessor',
                    processing_date_time='2026-01-15T12:00:00Z',
                ),
                classification=SIDDClassification(
                    classification='UNCLASSIFIED',
                ),
                product_name='Test Product',
                product_class='Detected Image',
            ),
            display=SIDDDisplay(
                pixel_type='MONO8I',
                num_bands=1,
            ),
        )

    def test_inherits_image_metadata(self):
        """SIDDMetadata is subclass of ImageMetadata."""
        assert issubclass(SIDDMetadata, ImageMetadata)

    def test_required_fields(self, sidd_meta):
        """Base fields accessible."""
        assert sidd_meta.format == 'SIDD'
        assert sidd_meta.rows == 512
        assert sidd_meta.cols == 768

    def test_sidd_specific_fields(self, sidd_meta):
        """SIDD-specific fields accessible."""
        assert sidd_meta.num_images == 1
        assert sidd_meta.image_index == 0

    def test_nested_product_creation(self, sidd_meta):
        """Nested product_creation accessible."""
        pc = sidd_meta.product_creation
        assert pc.product_name == 'Test Product'
        assert pc.processor_information.application == 'TestProcessor'
        assert pc.classification.classification == 'UNCLASSIFIED'

    def test_nested_display(self, sidd_meta):
        """Nested display accessible."""
        assert sidd_meta.display.pixel_type == 'MONO8I'
        assert sidd_meta.display.num_bands == 1

    def test_optional_sections_default_none(self, sidd_meta):
        """Unset optional sections default to None."""
        assert sidd_meta.geo_data is None
        assert sidd_meta.measurement is None
        assert sidd_meta.exploitation_features is None
        assert sidd_meta.compression is None

    def test_dict_access(self, sidd_meta):
        """Dict-like access resolves SIDD fields."""
        assert sidd_meta['backend'] == 'sarkit'
        assert sidd_meta['num_images'] == 1
        assert 'product_creation' in sidd_meta
        assert 'measurement' not in sidd_meta


# ---------------------------------------------------------------------------
# BIOMASSMetadata
# ---------------------------------------------------------------------------

class TestBIOMASSMetadata:
    """Tests for BIOMASSMetadata typed subclass."""

    @pytest.fixture
    def biomass_meta(self):
        """Minimal BIOMASSMetadata for testing."""
        return BIOMASSMetadata(
            format='BIOMASS_L1_SCS',
            rows=5000,
            cols=10000,
            dtype='complex64',
            bands=4,
            mission='BIOMASS',
            swath='S1',
            product_type='SCS',
            start_time='2026-01-15T10:30:00Z',
            stop_time='2026-01-15T10:31:00Z',
            orbit_number=1234,
            orbit_pass='ASCENDING',
            polarizations=['HH', 'HV', 'VH', 'VV'],
            num_polarizations=4,
            range_pixel_spacing=6.25,
            azimuth_pixel_spacing=6.25,
            pixel_type='32 bit Float',
            pixel_representation='Abs Phase',
            projection='Slant Range',
            nodata_value=-9999.0,
            prf=3000.0,
        )

    def test_inherits_image_metadata(self):
        """BIOMASSMetadata is subclass of ImageMetadata."""
        assert issubclass(BIOMASSMetadata, ImageMetadata)

    def test_required_fields(self, biomass_meta):
        """Base fields accessible."""
        assert biomass_meta.format == 'BIOMASS_L1_SCS'
        assert biomass_meta.rows == 5000
        assert biomass_meta.bands == 4

    def test_biomass_fields(self, biomass_meta):
        """BIOMASS-specific typed fields accessible."""
        assert biomass_meta.mission == 'BIOMASS'
        assert biomass_meta.swath == 'S1'
        assert biomass_meta.product_type == 'SCS'
        assert biomass_meta.orbit_number == 1234
        assert biomass_meta.orbit_pass == 'ASCENDING'
        assert biomass_meta.polarizations == ['HH', 'HV', 'VH', 'VV']
        assert biomass_meta.num_polarizations == 4
        assert biomass_meta.range_pixel_spacing == pytest.approx(6.25)
        assert biomass_meta.azimuth_pixel_spacing == pytest.approx(6.25)
        assert biomass_meta.nodata_value == pytest.approx(-9999.0)
        assert biomass_meta.prf == pytest.approx(3000.0)

    def test_optional_fields_default_none(self):
        """Unset BIOMASS fields default to None."""
        meta = BIOMASSMetadata(
            format='BIOMASS_L1_SCS',
            rows=100,
            cols=100,
            dtype='complex64',
        )
        assert meta.mission is None
        assert meta.polarizations is None
        assert meta.corner_coords is None
        assert meta.gcps is None

    def test_corner_coords(self):
        """Corner coordinates stored as dict."""
        meta = BIOMASSMetadata(
            format='BIOMASS_L1_SCS',
            rows=100,
            cols=100,
            dtype='complex64',
            corner_coords={
                'corner1': (10.0, 20.0),
                'corner2': (10.0, 21.0),
                'corner3': (11.0, 21.0),
                'corner4': (11.0, 20.0),
            },
        )
        assert meta.corner_coords['corner1'] == (10.0, 20.0)
        assert len(meta.corner_coords) == 4

    def test_dict_access(self, biomass_meta):
        """Dict-like access resolves BIOMASS fields."""
        assert biomass_meta['mission'] == 'BIOMASS'
        assert biomass_meta['prf'] == pytest.approx(3000.0)
        assert 'polarizations' in biomass_meta
        assert 'gcps' not in biomass_meta  # None

    def test_keys_includes_biomass_fields(self, biomass_meta):
        """keys() includes populated BIOMASS-specific fields."""
        k = biomass_meta.keys()
        assert 'mission' in k
        assert 'polarizations' in k
        assert 'range_pixel_spacing' in k
        # None fields excluded
        assert 'corner_coords' not in k
        assert 'gcps' not in k
