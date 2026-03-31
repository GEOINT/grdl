# -*- coding: utf-8 -*-
"""Tests for grdl.vector.spatial — Spatial operators."""

import pytest
from shapely.geometry import Point, box, Polygon

from grdl.vector.models import Feature, FeatureSet
from grdl.vector.spatial import (
    BufferOperator,
    IntersectionOperator,
    UnionOperator,
    DissolveOperator,
    SpatialJoinOperator,
    ClipOperator,
    CentroidOperator,
    ConvexHullOperator,
)


def _make_points():
    """Three point features."""
    return FeatureSet(features=[
        Feature(geometry=Point(0, 0), properties={'group': 'A', 'val': 1}, feature_id='p0'),
        Feature(geometry=Point(10, 10), properties={'group': 'B', 'val': 2}, feature_id='p1'),
        Feature(geometry=Point(20, 20), properties={'group': 'A', 'val': 3}, feature_id='p2'),
    ])


def _make_polygons():
    """Two overlapping box features."""
    return FeatureSet(features=[
        Feature(geometry=box(0, 0, 10, 10), properties={'name': 'left'}, feature_id='b0'),
        Feature(geometry=box(5, 5, 15, 15), properties={'name': 'right'}, feature_id='b1'),
    ])


class TestBufferOperator:
    def test_buffer_points(self):
        fs = _make_points()
        op = BufferOperator(distance=1.0, resolution=4)
        result = op.process(fs)
        assert len(result) == 3
        for f in result:
            assert f.geometry.geom_type == 'Polygon'
            assert f.geometry.area > 0

    def test_buffer_preserves_properties(self):
        fs = _make_points()
        op = BufferOperator(distance=0.5)
        result = op.process(fs)
        assert result[0].properties['group'] == 'A'
        assert result[0].id == 'p0'

    def test_buffer_version_tag(self):
        assert BufferOperator.__processor_version__ == '1.0.0'
        assert BufferOperator.__processor_tags__['category'].value == 'analyze'


class TestIntersectionOperator:
    def test_intersection(self):
        fs = _make_polygons()
        left = FeatureSet(features=[fs[0]])
        right = FeatureSet(features=[fs[1]])
        op = IntersectionOperator()
        result = op.process(left, overlay=right)
        assert len(result) == 1
        # Intersection of box(0,0,10,10) and box(5,5,15,15) = box(5,5,10,10)
        assert result[0].geometry.area == pytest.approx(25.0)

    def test_intersection_no_overlay(self):
        op = IntersectionOperator()
        with pytest.raises(ValueError, match='overlay'):
            op.process(_make_points())


class TestUnionOperator:
    def test_union_polygons(self):
        fs = _make_polygons()
        op = UnionOperator()
        result = op.process(fs)
        assert len(result) == 1
        # Union area = 100 + 100 - 25 = 175
        assert result[0].geometry.area == pytest.approx(175.0)

    def test_union_empty(self):
        op = UnionOperator()
        result = op.process(FeatureSet(features=[]))
        assert len(result) == 0


class TestDissolveOperator:
    def test_dissolve_by_group(self):
        fs = _make_points()
        # Buffer first so we have polygons to dissolve
        buffered = BufferOperator(distance=1.0).process(fs)
        op = DissolveOperator(by='group')
        result = op.process(buffered)
        # Two groups: A and B
        assert len(result) == 2
        groups = {f.properties['group'] for f in result}
        assert groups == {'A', 'B'}

    def test_dissolve_no_field(self):
        op = DissolveOperator()
        with pytest.raises(ValueError, match='by'):
            op.process(_make_points())


class TestSpatialJoinOperator:
    def test_inner_join(self):
        left = FeatureSet(features=[
            Feature(geometry=Point(5, 5), properties={'left_val': 1}, feature_id='l0'),
            Feature(geometry=Point(50, 50), properties={'left_val': 2}, feature_id='l1'),
        ])
        right = FeatureSet(features=[
            Feature(geometry=box(0, 0, 10, 10), properties={'right_val': 'X'}, feature_id='r0'),
        ])
        op = SpatialJoinOperator(predicate='intersects', how='inner')
        result = op.process(left, overlay=right)
        # Only Point(5,5) intersects box(0,0,10,10)
        assert len(result) == 1
        assert result[0].properties['left_val'] == 1
        assert result[0].properties['right_val'] == 'X'

    def test_left_join(self):
        left = FeatureSet(features=[
            Feature(geometry=Point(5, 5), properties={'v': 1}, feature_id='l0'),
            Feature(geometry=Point(50, 50), properties={'v': 2}, feature_id='l1'),
        ])
        right = FeatureSet(features=[
            Feature(geometry=box(0, 0, 10, 10), properties={'r': 'X'}, feature_id='r0'),
        ])
        op = SpatialJoinOperator(predicate='intersects', how='left')
        result = op.process(left, overlay=right)
        assert len(result) == 2  # Both left features kept

    def test_no_overlay(self):
        op = SpatialJoinOperator()
        with pytest.raises(ValueError, match='overlay'):
            op.process(_make_points())


class TestClipOperator:
    def test_clip(self):
        fs = _make_polygons()
        clip_geom = box(0, 0, 8, 8)
        op = ClipOperator()
        result = op.process(fs, clip_geometry=clip_geom)
        assert len(result) == 2
        # First box fully inside clip -> area = 64 (8x8 portion of 0-10)
        # Actually: intersection of box(0,0,10,10) with box(0,0,8,8) = box(0,0,8,8) -> 64
        assert result[0].geometry.area == pytest.approx(64.0)
        # Second box: intersection of box(5,5,15,15) with box(0,0,8,8) = box(5,5,8,8) -> 9
        assert result[1].geometry.area == pytest.approx(9.0)

    def test_clip_no_geometry(self):
        op = ClipOperator()
        with pytest.raises(ValueError, match='clip_geometry'):
            op.process(_make_points())


class TestCentroidOperator:
    def test_centroid(self):
        fs = _make_polygons()
        op = CentroidOperator()
        result = op.process(fs)
        assert len(result) == 2
        for f in result:
            assert f.geometry.geom_type == 'Point'
        # Centroid of box(0,0,10,10) = (5, 5)
        assert result[0].geometry.x == pytest.approx(5.0)
        assert result[0].geometry.y == pytest.approx(5.0)

    def test_centroid_preserves_properties(self):
        fs = _make_polygons()
        op = CentroidOperator()
        result = op.process(fs)
        assert result[0].properties['name'] == 'left'
        assert result[0].id == 'b0'


class TestConvexHullOperator:
    def test_convex_hull(self):
        # Create an L-shaped polygon
        l_shape = Polygon([(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)])
        fs = FeatureSet(features=[
            Feature(geometry=l_shape, properties={'shape': 'L'}, feature_id='L1'),
        ])
        op = ConvexHullOperator()
        result = op.process(fs)
        assert len(result) == 1
        hull = result[0].geometry
        # Convex hull of L-shape should be larger than original
        assert hull.area >= l_shape.area
        assert hull.geom_type == 'Polygon'
        assert result[0].properties['shape'] == 'L'


class TestVectorProcessorExecute:
    def test_execute_with_feature_set(self):
        """Test execute() protocol dispatch."""
        import dataclasses

        @dataclasses.dataclass
        class FakeMetadata:
            rows: int = 0
            cols: int = 0

        fs = _make_points()
        op = CentroidOperator()
        result, meta = op.execute(FakeMetadata(), fs)
        assert isinstance(result, FeatureSet)

    def test_execute_rejects_ndarray(self):
        import numpy as np
        import dataclasses

        @dataclasses.dataclass
        class FakeMetadata:
            rows: int = 0
            cols: int = 0

        op = CentroidOperator()
        with pytest.raises(TypeError, match='numpy'):
            op.execute(FakeMetadata(), np.zeros((10, 10)))
