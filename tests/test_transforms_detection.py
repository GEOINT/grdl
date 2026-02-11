# -*- coding: utf-8 -*-
"""
Tests for the transforms.detection bridge module and RegistrationResult
vector-transform methods.

Tests transform_pixel_geometry, transform_detection, and
transform_detection_set for Point and Polygon (including box) geometries,
both forward and inverse directions, and bbox refit/polygon modes.

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
2026-02-11

Modified
--------
2026-02-11
"""

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from grdl.coregistration.base import RegistrationResult
from grdl.image_processing.detection.models import Detection, DetectionSet
from grdl.transforms.detection import (
    transform_pixel_geometry,
    transform_detection,
    transform_detection_set,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_affine_result():
    """RegistrationResult with identity affine (2x3)."""
    matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    return RegistrationResult(
        transform_matrix=matrix,
        residual_rms=0.0,
        num_matches=10,
        inlier_ratio=1.0,
    )


@pytest.fixture
def translation_affine_result():
    """RegistrationResult with translation of (+5, +3) in (row, col)."""
    # [row', col'] = [row + 5, col + 3]
    matrix = np.array([[1, 0, 5], [0, 1, 3]], dtype=np.float64)
    return RegistrationResult(
        transform_matrix=matrix,
        residual_rms=0.1,
        num_matches=20,
        inlier_ratio=0.95,
    )


@pytest.fixture
def rotation_90_result():
    """RegistrationResult with 90-degree CCW rotation.

    (row, col) -> (-col, row)
    """
    matrix = np.array([[0, -1, 0], [1, 0, 0]], dtype=np.float64)
    return RegistrationResult(
        transform_matrix=matrix,
        residual_rms=0.0,
        num_matches=10,
        inlier_ratio=1.0,
    )


@pytest.fixture
def identity_projective_result():
    """RegistrationResult with identity homography (3x3)."""
    matrix = np.eye(3, dtype=np.float64)
    return RegistrationResult(
        transform_matrix=matrix,
        residual_rms=0.0,
        num_matches=10,
        inlier_ratio=1.0,
    )


@pytest.fixture
def translation_projective_result():
    """RegistrationResult with translation homography (+5, +3) in (row, col)."""
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 2] = 5.0
    matrix[1, 2] = 3.0
    return RegistrationResult(
        transform_matrix=matrix,
        residual_rms=0.1,
        num_matches=15,
        inlier_ratio=0.9,
    )


@pytest.fixture
def sample_detection_set():
    """DetectionSet with one Point and one box detection."""
    # Point at pixel (row=10, col=20) -> shapely Point(x=20, y=10)
    d1 = Detection(
        pixel_geometry=Point(20.0, 10.0),
        properties={'score': 0.9},
        confidence=0.9,
    )
    # Box: col_min=200, row_min=100, col_max=230, row_max=120
    d2 = Detection(
        pixel_geometry=box(200.0, 100.0, 230.0, 120.0),
        properties={'score': 0.7},
        confidence=0.7,
    )
    return DetectionSet(
        detections=[d1, d2],
        detector_name='TestDetector',
        detector_version='0.1.0',
        output_fields=('score',),
        metadata={'source': 'test'},
    )


# ---------------------------------------------------------------------------
# RegistrationResult.transform_points tests
# ---------------------------------------------------------------------------

class TestRegistrationResultTransformPoints:

    def test_identity_affine(self, identity_affine_result):
        pts = np.array([[10, 20], [30, 40]], dtype=np.float64)
        result = identity_affine_result.transform_points(pts)
        np.testing.assert_array_almost_equal(result, pts)

    def test_translation_affine(self, translation_affine_result):
        pts = np.array([[10, 20], [30, 40]], dtype=np.float64)
        result = translation_affine_result.transform_points(pts)
        expected = np.array([[15, 23], [35, 43]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_identity_projective(self, identity_projective_result):
        pts = np.array([[10, 20], [30, 40]], dtype=np.float64)
        result = identity_projective_result.transform_points(pts)
        np.testing.assert_array_almost_equal(result, pts)

    def test_translation_projective(self, translation_projective_result):
        pts = np.array([[10, 20]], dtype=np.float64)
        result = translation_projective_result.transform_points(pts)
        expected = np.array([[15, 23]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_round_trip_affine(self, translation_affine_result):
        pts = np.array([[10, 20], [50, 60]], dtype=np.float64)
        forward = translation_affine_result.transform_points(pts)
        back = translation_affine_result.transform_points(forward, inverse=True)
        np.testing.assert_array_almost_equal(back, pts)

    def test_round_trip_projective(self, translation_projective_result):
        pts = np.array([[10, 20], [50, 60]], dtype=np.float64)
        forward = translation_projective_result.transform_points(pts)
        back = translation_projective_result.transform_points(
            forward, inverse=True,
        )
        np.testing.assert_array_almost_equal(back, pts)

    def test_inverse_transform_matrix_shape_affine(
        self, translation_affine_result,
    ):
        inv = translation_affine_result.inverse_transform_matrix
        assert inv.shape == (3, 3)

    def test_inverse_transform_matrix_shape_projective(
        self, translation_projective_result,
    ):
        inv = translation_projective_result.inverse_transform_matrix
        assert inv.shape == (3, 3)


# ---------------------------------------------------------------------------
# transform_pixel_geometry — Point
# ---------------------------------------------------------------------------

class TestTransformPixelGeometryPoint:

    def test_identity_preserves_point(self, identity_affine_result):
        # Point at row=10, col=20 -> shapely (x=20, y=10)
        geom = Point(20.0, 10.0)
        out = transform_pixel_geometry(geom, identity_affine_result)
        assert out.geom_type == 'Point'
        assert pytest.approx(out.x) == 20.0
        assert pytest.approx(out.y) == 10.0

    def test_translation_shifts_point(self, translation_affine_result):
        # Point at row=10, col=20
        # After +5 row, +3 col: row=15, col=23
        geom = Point(20.0, 10.0)
        out = transform_pixel_geometry(geom, translation_affine_result)
        assert out.geom_type == 'Point'
        assert pytest.approx(out.x) == 23.0  # col
        assert pytest.approx(out.y) == 15.0  # row

    def test_projective_point(self, translation_projective_result):
        geom = Point(20.0, 10.0)
        out = transform_pixel_geometry(geom, translation_projective_result)
        assert pytest.approx(out.x) == 23.0
        assert pytest.approx(out.y) == 15.0

    def test_returns_new_geometry(self, translation_affine_result):
        geom = Point(20.0, 10.0)
        out = transform_pixel_geometry(geom, translation_affine_result)
        assert out is not geom
        # Original unchanged
        assert geom.x == 20.0
        assert geom.y == 10.0


# ---------------------------------------------------------------------------
# transform_pixel_geometry — Polygon (box / axis-aligned rect)
# ---------------------------------------------------------------------------

class TestTransformPixelGeometryBox:

    def test_identity_preserves_box(self, identity_affine_result):
        # box(minx, miny, maxx, maxy) = box(col_min, row_min, col_max, row_max)
        geom = box(20.0, 10.0, 50.0, 30.0)
        out = transform_pixel_geometry(geom, identity_affine_result)
        assert out.geom_type == 'Polygon'
        assert pytest.approx(out.bounds) == (20.0, 10.0, 50.0, 30.0)

    def test_translation_shifts_box(self, translation_affine_result):
        # box at col=[20,50], row=[10,30]
        # +5 row, +3 col -> col=[23,53], row=[15,35]
        geom = box(20.0, 10.0, 50.0, 30.0)
        out = transform_pixel_geometry(geom, translation_affine_result)
        assert out.geom_type == 'Polygon'
        assert pytest.approx(out.bounds) == (23.0, 15.0, 53.0, 35.0)

    def test_refit_preserves_axis_aligned(self, translation_affine_result):
        geom = box(20.0, 10.0, 50.0, 30.0)
        out = transform_pixel_geometry(
            geom, translation_affine_result, bbox_mode='refit',
        )
        # Should still be a valid axis-aligned box after translation
        bounds = out.bounds
        coords = np.array(out.exterior.coords)
        xs = set(np.round(coords[:-1, 0], 10))
        ys = set(np.round(coords[:-1, 1], 10))
        assert len(xs) == 2  # only 2 unique x values
        assert len(ys) == 2  # only 2 unique y values

    def test_polygon_mode_preserves_exact_shape(
        self, translation_affine_result,
    ):
        geom = box(20.0, 10.0, 50.0, 30.0)
        out = transform_pixel_geometry(
            geom, translation_affine_result, bbox_mode='polygon',
        )
        assert out.geom_type == 'Polygon'

    def test_rotation_refit_contains_all_corners(self, rotation_90_result):
        """After rotation, re-fit box must contain all transformed corners."""
        geom = box(0.0, 0.0, 20.0, 10.0)
        refit_out = transform_pixel_geometry(
            geom, rotation_90_result, bbox_mode='refit',
        )
        poly_out = transform_pixel_geometry(
            geom, rotation_90_result, bbox_mode='polygon',
        )
        # All polygon vertices must be inside (or on) the re-fit box
        for coord in poly_out.exterior.coords[:-1]:
            p = Point(coord)
            assert refit_out.contains(p) or refit_out.touches(p), (
                f"Re-fit box does not contain transformed corner {coord}"
            )

    def test_invalid_bbox_mode(self, translation_affine_result):
        geom = box(20.0, 10.0, 50.0, 30.0)
        with pytest.raises(ValueError, match="bbox_mode"):
            transform_pixel_geometry(
                geom, translation_affine_result, bbox_mode='invalid',
            )


# ---------------------------------------------------------------------------
# transform_pixel_geometry — general Polygon
# ---------------------------------------------------------------------------

class TestTransformPixelGeometryPolygon:

    def test_identity_preserves_polygon(self, identity_affine_result):
        # Triangle in shapely (col, row) convention
        geom = Polygon([(0, 0), (10, 0), (10, 10), (0, 0)])
        out = transform_pixel_geometry(geom, identity_affine_result)
        assert out.geom_type == 'Polygon'
        original_coords = np.array(geom.exterior.coords)
        out_coords = np.array(out.exterior.coords)
        np.testing.assert_array_almost_equal(out_coords, original_coords)

    def test_translation_shifts_polygon(self, translation_affine_result):
        # Triangle: shapely coords (col, row)
        # Vertices: (col=0, row=0), (col=10, row=0), (col=10, row=10)
        geom = Polygon([(0, 0), (10, 0), (10, 10), (0, 0)])
        out = transform_pixel_geometry(
            geom, translation_affine_result, bbox_mode='polygon',
        )
        # +5 row, +3 col -> (3, 5), (13, 5), (13, 15)
        coords = np.array(out.exterior.coords)
        expected = np.array([
            [3.0, 5.0], [13.0, 5.0], [13.0, 15.0], [3.0, 5.0],
        ])
        np.testing.assert_array_almost_equal(coords, expected)

    def test_non_rect_polygon_always_returns_polygon(
        self, translation_affine_result,
    ):
        """Non-rectangular polygons return Polygon even with refit mode."""
        geom = Polygon([(0, 0), (10, 0), (5, 10), (0, 0)])
        out = transform_pixel_geometry(
            geom, translation_affine_result, bbox_mode='refit',
        )
        assert out.geom_type == 'Polygon'


# ---------------------------------------------------------------------------
# transform_pixel_geometry — inverse direction
# ---------------------------------------------------------------------------

class TestTransformPixelGeometryInverse:

    def test_round_trip_point(self, translation_affine_result):
        geom = Point(20.0, 10.0)
        forward = transform_pixel_geometry(geom, translation_affine_result)
        back = transform_pixel_geometry(
            forward, translation_affine_result, inverse=True,
        )
        assert pytest.approx(back.x) == geom.x
        assert pytest.approx(back.y) == geom.y

    def test_round_trip_box(self, translation_affine_result):
        geom = box(20.0, 10.0, 50.0, 30.0)
        forward = transform_pixel_geometry(geom, translation_affine_result)
        back = transform_pixel_geometry(
            forward, translation_affine_result, inverse=True,
        )
        assert pytest.approx(back.bounds, abs=1e-10) == geom.bounds

    def test_round_trip_polygon(self, translation_affine_result):
        geom = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        forward = transform_pixel_geometry(
            geom, translation_affine_result, bbox_mode='polygon',
        )
        back = transform_pixel_geometry(
            forward, translation_affine_result,
            inverse=True, bbox_mode='polygon',
        )
        original = np.array(geom.exterior.coords)
        result = np.array(back.exterior.coords)
        np.testing.assert_array_almost_equal(result, original)

    def test_round_trip_projective(self, translation_projective_result):
        geom = Point(50.0, 25.0)
        forward = transform_pixel_geometry(
            geom, translation_projective_result,
        )
        back = transform_pixel_geometry(
            forward, translation_projective_result, inverse=True,
        )
        assert pytest.approx(back.x) == geom.x
        assert pytest.approx(back.y) == geom.y


# ---------------------------------------------------------------------------
# transform_detection
# ---------------------------------------------------------------------------

class TestTransformDetection:

    def test_pixel_geometry_transformed(self, translation_affine_result):
        det = Detection(
            pixel_geometry=Point(20.0, 10.0),
            properties={'score': 0.8},
            confidence=0.8,
        )
        out = transform_detection(det, translation_affine_result)
        assert pytest.approx(out.pixel_geometry.x) == 23.0
        assert pytest.approx(out.pixel_geometry.y) == 15.0

    def test_properties_preserved(self, translation_affine_result):
        det = Detection(
            pixel_geometry=Point(20.0, 10.0),
            properties={'score': 0.8, 'label': 'vehicle'},
            confidence=0.8,
        )
        out = transform_detection(det, translation_affine_result)
        assert out.properties == {'score': 0.8, 'label': 'vehicle'}
        assert out.confidence == 0.8

    def test_geo_geometry_preserved(self, translation_affine_result):
        geo = Point(-75.0, 40.0)  # shapely: (lon, lat)
        det = Detection(
            pixel_geometry=Point(20.0, 10.0),
            properties={'score': 0.8},
            confidence=0.8,
            geo_geometry=geo,
        )
        out = transform_detection(det, translation_affine_result)
        assert out.geo_geometry is geo  # same object, not transformed

    def test_original_unchanged(self, translation_affine_result):
        det = Detection(
            pixel_geometry=Point(20.0, 10.0),
            properties={'score': 0.8},
            confidence=0.8,
        )
        transform_detection(det, translation_affine_result)
        assert det.pixel_geometry.x == 20.0
        assert det.pixel_geometry.y == 10.0

    def test_returns_new_detection(self, translation_affine_result):
        det = Detection(
            pixel_geometry=Point(20.0, 10.0),
            properties={'score': 0.8},
            confidence=0.8,
        )
        out = transform_detection(det, translation_affine_result)
        assert out is not det


# ---------------------------------------------------------------------------
# transform_detection_set
# ---------------------------------------------------------------------------

class TestTransformDetectionSet:

    def test_all_detections_transformed(
        self, sample_detection_set, translation_affine_result,
    ):
        out = transform_detection_set(
            sample_detection_set, translation_affine_result,
        )
        assert len(out) == len(sample_detection_set)
        # Point: row=10->15, col=20->23 -> shapely (x=23, y=15)
        assert pytest.approx(out[0].pixel_geometry.x) == 23.0
        assert pytest.approx(out[0].pixel_geometry.y) == 15.0
        # Box: bounds should shift by +3 col, +5 row
        original_bounds = sample_detection_set[1].pixel_geometry.bounds
        new_bounds = out[1].pixel_geometry.bounds
        assert pytest.approx(new_bounds[0]) == original_bounds[0] + 3.0  # minx
        assert pytest.approx(new_bounds[1]) == original_bounds[1] + 5.0  # miny
        assert pytest.approx(new_bounds[2]) == original_bounds[2] + 3.0  # maxx
        assert pytest.approx(new_bounds[3]) == original_bounds[3] + 5.0  # maxy

    def test_metadata_preserved_and_extended(
        self, sample_detection_set, translation_affine_result,
    ):
        out = transform_detection_set(
            sample_detection_set, translation_affine_result,
        )
        assert out.metadata['source'] == 'test'
        assert out.metadata['coordinate_transform_applied'] is True
        assert out.metadata['transform_direction'] == 'forward'

    def test_inverse_metadata(
        self, sample_detection_set, translation_affine_result,
    ):
        out = transform_detection_set(
            sample_detection_set, translation_affine_result, inverse=True,
        )
        assert out.metadata['transform_direction'] == 'inverse'

    def test_detector_info_preserved(
        self, sample_detection_set, translation_affine_result,
    ):
        out = transform_detection_set(
            sample_detection_set, translation_affine_result,
        )
        assert out.detector_name == 'TestDetector'
        assert out.detector_version == '0.1.0'
        assert out.output_fields == ('score',)

    def test_empty_set(self, translation_affine_result):
        empty = DetectionSet(
            detections=[],
            detector_name='Empty',
            detector_version='0.0.0',
            output_fields=(),
        )
        out = transform_detection_set(empty, translation_affine_result)
        assert len(out) == 0
        assert out.detector_name == 'Empty'

    def test_original_unchanged(
        self, sample_detection_set, translation_affine_result,
    ):
        original_x = sample_detection_set[0].pixel_geometry.x
        original_y = sample_detection_set[0].pixel_geometry.y
        transform_detection_set(
            sample_detection_set, translation_affine_result,
        )
        assert sample_detection_set[0].pixel_geometry.x == original_x
        assert sample_detection_set[0].pixel_geometry.y == original_y
