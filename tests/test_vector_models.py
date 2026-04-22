# -*- coding: utf-8 -*-
"""Tests for grdl.vector.models — Feature, FieldSchema, FeatureSet."""

import warnings

import pytest
from shapely.geometry import Point, Polygon, box, LineString

from grdl.vector.models import Feature, FieldSchema, FeatureSet


# =====================================================================
# Feature
# =====================================================================

class TestFeature:
    def test_create_with_defaults(self):
        f = Feature(geometry=Point(1, 2))
        assert f.geometry.geom_type == 'Point'
        assert f.properties == {}
        assert f.id is not None

    def test_create_with_properties(self):
        f = Feature(
            geometry=Point(1, 2),
            properties={'name': 'test', 'value': 42},
            feature_id='feat-1',
        )
        assert f.properties['name'] == 'test'
        assert f.id == 'feat-1'

    def test_to_geojson_feature(self):
        f = Feature(
            geometry=Point(1, 2),
            properties={'label': 'A'},
            feature_id='f1',
        )
        gj = f.to_geojson_feature()
        assert gj['type'] == 'Feature'
        assert gj['id'] == 'f1'
        assert gj['geometry']['type'] == 'Point'
        assert gj['properties']['label'] == 'A'

    def test_from_geojson_feature(self):
        gj = {
            'type': 'Feature',
            'id': 'x1',
            'geometry': {'type': 'Point', 'coordinates': [10, 20]},
            'properties': {'kind': 'city'},
        }
        f = Feature.from_geojson_feature(gj)
        assert f.id == 'x1'
        assert f.geometry.x == 10
        assert f.geometry.y == 20
        assert f.properties['kind'] == 'city'

    def test_repr(self):
        f = Feature(geometry=Point(0, 0), feature_id='r1')
        r = repr(f)
        assert 'Feature' in r
        assert 'Point' in r


# =====================================================================
# FieldSchema
# =====================================================================

class TestFieldSchema:
    def test_create(self):
        fs = FieldSchema(name='value', dtype='float', description='A value')
        assert fs.name == 'value'
        assert fs.dtype == 'float'
        assert fs.nullable is True

    def test_round_trip_dict(self):
        fs = FieldSchema(name='x', dtype='int', nullable=False)
        d = fs.to_dict()
        fs2 = FieldSchema.from_dict(d)
        assert fs2.name == 'x'
        assert fs2.dtype == 'int'
        assert fs2.nullable is False


# =====================================================================
# FeatureSet
# =====================================================================

def _make_feature_set():
    """Helper to create a small test FeatureSet."""
    features = [
        Feature(
            geometry=Point(0, 0),
            properties={'label': 'A', 'score': 0.9},
            feature_id='0',
        ),
        Feature(
            geometry=Point(5, 5),
            properties={'label': 'B', 'score': 0.5},
            feature_id='1',
        ),
        Feature(
            geometry=box(10, 10, 12, 12),
            properties={'label': 'C', 'score': 0.7},
            feature_id='2',
        ),
    ]
    return FeatureSet(
        features=features,
        crs='EPSG:4326',
        schema=[FieldSchema('label', 'str'), FieldSchema('score', 'float')],
        metadata={'source': 'test'},
    )


class TestFeatureSet:
    def test_len_iter_getitem(self):
        fs = _make_feature_set()
        assert len(fs) == 3
        assert fs.count == 3
        items = list(fs)
        assert len(items) == 3
        assert fs[0].id == '0'

    def test_bounds(self):
        fs = _make_feature_set()
        b = fs.bounds
        assert b[0] == 0.0  # minx
        assert b[1] == 0.0  # miny
        assert b[2] == 12.0  # maxx
        assert b[3] == 12.0  # maxy

    def test_bounds_empty(self):
        fs = FeatureSet(features=[])
        assert fs.bounds is None

    def test_geometry_types(self):
        fs = _make_feature_set()
        types = fs.geometry_types
        assert 'Point' in types
        assert 'Polygon' in types

    def test_get_geometries(self):
        fs = _make_feature_set()
        geoms = fs.get_geometries()
        assert len(geoms) == 3
        assert geoms[0].geom_type == 'Point'

    def test_get_property_array(self):
        fs = _make_feature_set()
        labels = fs.get_property_array('label')
        assert labels == ['A', 'B', 'C']

    def test_get_property_array_missing(self):
        fs = _make_feature_set()
        vals = fs.get_property_array('nonexistent')
        assert vals == [None, None, None]


class TestFeatureSetFiltering:
    def test_filter_by_bbox(self):
        fs = _make_feature_set()
        result = fs.filter_by_bbox(-1, -1, 1, 1)
        assert len(result) == 1
        assert result[0].id == '0'

    def test_filter_by_property(self):
        fs = _make_feature_set()
        result = fs.filter_by_property('label', 'B')
        assert len(result) == 1
        assert result[0].properties['label'] == 'B'

    def test_filter_by_geometry_intersects(self):
        fs = _make_feature_set()
        query_geom = box(-1, -1, 1, 1)
        result = fs.filter_by_geometry(query_geom, predicate='intersects')
        assert len(result) == 1
        assert result[0].id == '0'

    def test_filter_by_geometry_contains(self):
        fs = _make_feature_set()
        query_geom = box(-1, -1, 6, 6)
        result = fs.filter_by_geometry(query_geom, predicate='contains')
        assert len(result) == 2  # contains Point(0,0) and Point(5,5)


class TestFeatureSetGeoJSON:
    def test_round_trip(self):
        fs = _make_feature_set()
        gj = fs.to_geojson()

        assert gj['type'] == 'FeatureCollection'
        assert len(gj['features']) == 3

        fs2 = FeatureSet.from_geojson(gj)
        assert len(fs2) == 3
        assert fs2.crs == 'EPSG:4326'
        assert len(fs2.schema) == 2
        assert fs2[0].properties['label'] == 'A'
        assert fs2.metadata.get('source') == 'test'

    def test_round_trip_preserves_geometry(self):
        fs = _make_feature_set()
        gj = fs.to_geojson()
        fs2 = FeatureSet.from_geojson(gj)
        # Point geometry preserved
        assert fs2[0].geometry.x == 0.0
        assert fs2[0].geometry.y == 0.0
        # Polygon geometry preserved
        assert fs2[2].geometry.geom_type == 'Polygon'


class TestDetectionSetBridge:
    def test_from_detection_set(self):
        from grdl.image_processing.detection.models import (
            Detection, DetectionSet,
        )
        dets = [
            Detection(
                pixel_geometry=Point(10, 20),
                properties={'label': 'target'},
                confidence=0.85,
                geo_geometry=Point(-77.0, 38.0),
            ),
            Detection(
                pixel_geometry=Point(30, 40),
                properties={'label': 'clutter'},
                confidence=0.3,
            ),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ds = DetectionSet(
                detections=dets,
                detector_name='TestDetector',
                detector_version='1.0.0',
                output_fields=('label',),
            )

        fs = FeatureSet.from_detection_set(ds, use_geo_geometry=True)
        assert len(fs) == 2
        # First detection has geo_geometry -> should use it
        assert fs[0].geometry.x == -77.0
        # Second detection has no geo_geometry -> uses pixel
        assert fs[1].geometry.x == 30.0
        assert fs[0].properties['confidence'] == 0.85
        assert fs.metadata['source_detector'] == 'TestDetector'

    def test_to_detection_set(self):
        fs = _make_feature_set()
        # Add a confidence property for bridge
        fs.features[0].properties['confidence'] = 0.99

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ds = fs.to_detection_set(
                detector_name='VectorBridge',
                detector_version='1.0.0',
            )

        assert len(ds) == 3
        assert ds.detector_name == 'VectorBridge'
        # First detection had confidence extracted
        assert ds[0].confidence == 0.99
        # Second detection had no confidence
        assert ds[1].confidence is None

    def test_round_trip_detection_set(self):
        from grdl.image_processing.detection.models import (
            Detection, DetectionSet,
        )
        dets = [
            Detection(
                pixel_geometry=Point(1, 2),
                properties={'val': 10},
                confidence=0.5,
            ),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ds = DetectionSet(
                detections=dets,
                detector_name='D',
                detector_version='0.1.0',
                output_fields=('val',),
            )

        fs = FeatureSet.from_detection_set(ds, use_geo_geometry=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ds2 = fs.to_detection_set(
                detector_name='D',
                detector_version='0.1.0',
            )
        assert len(ds2) == 1
        assert ds2[0].properties['val'] == 10
        assert ds2[0].confidence == 0.5


class TestGeoDataFrameBridge:
    def test_to_geodataframe(self):
        pytest.importorskip('geopandas')
        fs = _make_feature_set()
        gdf = fs.to_geodataframe()
        assert len(gdf) == 3
        assert 'label' in gdf.columns

    def test_from_geodataframe(self):
        gpd = pytest.importorskip('geopandas')
        gdf = gpd.GeoDataFrame(
            {'name': ['X', 'Y']},
            geometry=[Point(0, 0), Point(1, 1)],
            crs='EPSG:4326',
        )
        fs = FeatureSet.from_geodataframe(gdf)
        assert len(fs) == 2
        assert fs[0].properties['name'] == 'X'

    def test_round_trip_geodataframe(self):
        pytest.importorskip('geopandas')
        fs = _make_feature_set()
        gdf = fs.to_geodataframe()
        fs2 = FeatureSet.from_geodataframe(gdf)
        assert len(fs2) == len(fs)
        labels = fs2.get_property_array('label')
        assert set(labels) == {'A', 'B', 'C'}
