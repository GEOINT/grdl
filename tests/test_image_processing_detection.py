# -*- coding: utf-8 -*-
"""
Image Detection Tests.

Tests for the detection data models (Geometry, OutputField, OutputSchema,
Detection, DetectionSet), the ImageDetector ABC, and geo-registration.
All tests use synthetic data.

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
2026-02-06

Modified
--------
2026-02-06
"""

from typing import Any, Optional, Tuple

import numpy as np
import pytest

from grdl.image_processing.base import ImageProcessor
from grdl.image_processing.detection.base import ImageDetector
from grdl.image_processing.detection.models import (
    Detection,
    DetectionSet,
    Geometry,
    OutputField,
    OutputSchema,
)
from grdl.image_processing.versioning import (
    DetectionInputSpec,
    processor_version,
)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class TestGeometry:
    """Test Geometry creation and GeoJSON export."""

    def test_point_creation(self):
        g = Geometry.point(10.0, 20.0)
        assert g.geometry_type == 'Point'
        np.testing.assert_array_equal(g.pixel_coordinates, [10.0, 20.0])
        assert g.geographic_coordinates is None

    def test_point_with_geo(self):
        g = Geometry.point(10.0, 20.0, lat=45.0, lon=-73.0)
        np.testing.assert_array_equal(
            g.geographic_coordinates, [45.0, -73.0]
        )

    def test_bounding_box_creation(self):
        g = Geometry.bounding_box(5.0, 10.0, 50.0, 100.0)
        assert g.geometry_type == 'BoundingBox'
        np.testing.assert_array_equal(
            g.pixel_coordinates, [5.0, 10.0, 50.0, 100.0]
        )

    def test_polygon_creation(self):
        verts = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=np.float64)
        g = Geometry.polygon(verts)
        assert g.geometry_type == 'Polygon'
        assert g.pixel_coordinates.shape == (4, 2)

    def test_polygon_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            Geometry.polygon(np.array([1, 2, 3]))

    def test_invalid_geometry_type_raises(self):
        with pytest.raises(ValueError, match="geometry_type"):
            Geometry('Circle', np.array([0, 0]))

    def test_non_array_raises(self):
        with pytest.raises(TypeError, match="ndarray"):
            Geometry('Point', [10, 20])

    def test_to_geojson_point_pixel(self):
        g = Geometry.point(10.0, 20.0)
        gj = g.to_geojson()
        assert gj['type'] == 'Point'
        # GeoJSON: [col/lon, row/lat]
        assert gj['coordinates'] == [20.0, 10.0]

    def test_to_geojson_point_geo(self):
        g = Geometry.point(10.0, 20.0, lat=45.0, lon=-73.0)
        gj = g.to_geojson()
        assert gj['type'] == 'Point'
        # GeoJSON: [lon, lat]
        assert gj['coordinates'] == [-73.0, 45.0]

    def test_to_geojson_bbox(self):
        g = Geometry.bounding_box(0.0, 0.0, 10.0, 20.0)
        gj = g.to_geojson()
        assert gj['type'] == 'Polygon'
        assert len(gj['coordinates']) == 1  # one ring
        ring = gj['coordinates'][0]
        assert len(ring) == 5  # closed ring
        assert ring[0] == ring[-1]  # closure

    def test_to_geojson_polygon(self):
        verts = np.array([[0, 0], [0, 10], [10, 10]], dtype=np.float64)
        g = Geometry.polygon(verts)
        gj = g.to_geojson()
        assert gj['type'] == 'Polygon'
        ring = gj['coordinates'][0]
        # 3 vertices + closure = 4 points
        assert len(ring) == 4
        assert ring[0] == ring[-1]

    def test_repr(self):
        g = Geometry.point(1.0, 2.0)
        r = repr(g)
        assert 'Point' in r


# ---------------------------------------------------------------------------
# OutputField
# ---------------------------------------------------------------------------

class TestOutputField:
    """Test OutputField construction."""

    def test_construction(self):
        f = OutputField('magnitude', 'float', 'Signal magnitude', units='dB')
        assert f.name == 'magnitude'
        assert f.dtype == 'float'
        assert f.description == 'Signal magnitude'
        assert f.units == 'dB'

    def test_without_units(self):
        f = OutputField('label', 'str', 'Class label')
        assert f.units is None

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="dtype"):
            OutputField('x', 'complex', 'Bad type')

    def test_repr(self):
        f = OutputField('x', 'float', 'desc', units='m')
        r = repr(f)
        assert 'x' in r
        assert 'm' in r


# ---------------------------------------------------------------------------
# OutputSchema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Test OutputSchema construction and validation."""

    @pytest.fixture
    def phenomenological_schema(self):
        return OutputSchema(fields=(
            OutputField('change_magnitude', 'float', 'Change in dB', units='dB'),
            OutputField('coherence_loss', 'float', 'Coherence loss ratio'),
        ))

    @pytest.fixture
    def classification_schema(self):
        return OutputSchema(fields=(
            OutputField('label', 'str', 'Semantic class label'),
            OutputField('label_confidence', 'float', 'Classification confidence'),
        ))

    def test_field_names(self, phenomenological_schema):
        assert phenomenological_schema.field_names == (
            'change_magnitude', 'coherence_loss'
        )

    def test_validate_valid_properties(self, phenomenological_schema):
        props = {'change_magnitude': 3.5, 'coherence_loss': 0.2}
        # Should not raise
        phenomenological_schema.validate_properties(props)

    def test_validate_missing_field_raises(self, phenomenological_schema):
        with pytest.raises(ValueError, match="Missing"):
            phenomenological_schema.validate_properties(
                {'change_magnitude': 3.5}
            )

    def test_validate_wrong_type_raises(self, classification_schema):
        with pytest.raises(ValueError, match="type"):
            classification_schema.validate_properties(
                {'label': 123, 'label_confidence': 0.9}
            )

    def test_classification_schema_valid(self, classification_schema):
        props = {'label': 'building', 'label_confidence': 0.95}
        # Should not raise
        classification_schema.validate_properties(props)

    def test_int_accepted_as_float(self, phenomenological_schema):
        """int values are valid for float fields."""
        props = {'change_magnitude': 3, 'coherence_loss': 0}
        phenomenological_schema.validate_properties(props)

    def test_repr(self, phenomenological_schema):
        r = repr(phenomenological_schema)
        assert 'change_magnitude' in r


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestDetection:
    """Test Detection construction and GeoJSON export."""

    def test_construction(self):
        g = Geometry.point(10.0, 20.0)
        d = Detection(g, {'label': 'car'}, confidence=0.9)
        assert d.geometry is g
        assert d.properties == {'label': 'car'}
        assert d.confidence == 0.9

    def test_confidence_optional(self):
        g = Geometry.point(5.0, 5.0)
        d = Detection(g, {'value': 1.0})
        assert d.confidence is None

    def test_to_geojson_feature(self):
        g = Geometry.point(10.0, 20.0, lat=45.0, lon=-73.0)
        d = Detection(g, {'label': 'tree'}, confidence=0.8)
        feature = d.to_geojson_feature()
        assert feature['type'] == 'Feature'
        assert feature['geometry']['type'] == 'Point'
        assert feature['properties']['label'] == 'tree'
        assert feature['properties']['confidence'] == 0.8

    def test_to_geojson_feature_no_confidence(self):
        g = Geometry.point(10.0, 20.0)
        d = Detection(g, {'value': 3.0})
        feature = d.to_geojson_feature()
        assert 'confidence' not in feature['properties']
        assert feature['properties']['value'] == 3.0

    def test_repr(self):
        g = Geometry.point(1.0, 2.0)
        d = Detection(g, {}, confidence=0.5)
        r = repr(d)
        assert 'Point' in r
        assert '0.5' in r


# ---------------------------------------------------------------------------
# DetectionSet
# ---------------------------------------------------------------------------

class TestDetectionSet:
    """Test DetectionSet construction, protocol, and methods."""

    @pytest.fixture
    def sample_schema(self):
        return OutputSchema(fields=(
            OutputField('label', 'str', 'Class label'),
        ))

    @pytest.fixture
    def sample_set(self, sample_schema):
        detections = [
            Detection(Geometry.point(i, i), {'label': f'obj_{i}'},
                       confidence=i * 0.25)
            for i in range(4)
        ]
        return DetectionSet(
            detections=detections,
            detector_name='TestDetector',
            detector_version='1.0.0',
            output_schema=sample_schema,
        )

    def test_len(self, sample_set):
        assert len(sample_set) == 4

    def test_iter(self, sample_set):
        items = list(sample_set)
        assert len(items) == 4
        assert all(isinstance(d, Detection) for d in items)

    def test_getitem(self, sample_set):
        d = sample_set[2]
        assert d.properties['label'] == 'obj_2'

    def test_to_geojson(self, sample_set):
        gj = sample_set.to_geojson()
        assert gj['type'] == 'FeatureCollection'
        assert len(gj['features']) == 4
        assert gj['properties']['detector_name'] == 'TestDetector'
        assert gj['properties']['detector_version'] == '1.0.0'

    def test_filter_by_confidence(self, sample_set):
        filtered = sample_set.filter_by_confidence(0.5)
        assert len(filtered) == 2  # confidence 0.5 and 0.75
        assert filtered.detector_name == 'TestDetector'

    def test_filter_excludes_none_confidence(self, sample_schema):
        detections = [
            Detection(Geometry.point(0, 0), {'label': 'a'}),  # None confidence
            Detection(Geometry.point(1, 1), {'label': 'b'}, confidence=0.9),
        ]
        ds = DetectionSet(detections, 'Det', '1.0', sample_schema)
        filtered = ds.filter_by_confidence(0.0)
        assert len(filtered) == 1

    def test_empty_set(self, sample_schema):
        ds = DetectionSet([], 'Empty', '1.0', sample_schema)
        assert len(ds) == 0
        gj = ds.to_geojson()
        assert gj['features'] == []

    def test_metadata(self, sample_schema):
        ds = DetectionSet(
            [], 'Det', '1.0', sample_schema,
            metadata={'processing_time_ms': 42},
        )
        assert ds.metadata['processing_time_ms'] == 42
        gj = ds.to_geojson()
        assert gj['properties']['processing_time_ms'] == 42

    def test_repr(self, sample_set):
        r = repr(sample_set)
        assert 'TestDetector' in r
        assert '4' in r


# ---------------------------------------------------------------------------
# ImageDetector ABC contract
# ---------------------------------------------------------------------------

@processor_version('1.0.0')
class DummyDetector(ImageDetector):
    """Concrete detector for testing: marks pixels above threshold as points."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    @property
    def output_schema(self) -> OutputSchema:
        return OutputSchema(fields=(
            OutputField('intensity', 'float', 'Pixel intensity'),
        ))

    def detect(
        self,
        source: np.ndarray,
        geolocation: Optional[Any] = None,
        **kwargs: Any,
    ) -> DetectionSet:
        self._validate_detection_inputs(kwargs)
        rows, cols = np.where(source > self.threshold)
        detections = []
        for r, c in zip(rows, cols):
            geom = Geometry.point(float(r), float(c))
            props = {'intensity': float(source[r, c])}
            detections.append(Detection(geom, props, confidence=float(source[r, c])))

        ds = DetectionSet(
            detections=detections,
            detector_name=self.__class__.__name__,
            detector_version=getattr(self.__class__, '__processor_version__', 'unknown'),
            output_schema=self.output_schema,
        )

        if geolocation is not None:
            self._geo_register_detections(ds, geolocation)

        return ds


class TestImageDetectorABC:
    """Verify ImageDetector ABC contract."""

    def test_is_subclass_of_image_processor(self):
        assert issubclass(ImageDetector, ImageProcessor)

    def test_is_abstract(self):
        """Cannot instantiate ImageDetector directly."""
        with pytest.raises(TypeError):
            ImageDetector()

    def test_concrete_satisfies_interface(self):
        detector = DummyDetector(threshold=0.5)
        assert isinstance(detector, ImageDetector)
        assert isinstance(detector, ImageProcessor)

    def test_detect_returns_detection_set(self):
        detector = DummyDetector(threshold=0.5)
        image = np.array([[0.1, 0.9], [0.3, 0.7]], dtype=np.float64)
        result = detector.detect(image)
        assert isinstance(result, DetectionSet)

    def test_detect_finds_correct_pixels(self):
        detector = DummyDetector(threshold=0.5)
        image = np.array([[0.1, 0.9], [0.3, 0.7]], dtype=np.float64)
        result = detector.detect(image)
        assert len(result) == 2  # pixels at (0,1) and (1,1)

    def test_output_schema_property(self):
        detector = DummyDetector()
        schema = detector.output_schema
        assert isinstance(schema, OutputSchema)
        assert 'intensity' in schema.field_names

    def test_detect_without_geolocation(self):
        detector = DummyDetector(threshold=0.0)
        image = np.array([[1.0]], dtype=np.float64)
        result = detector.detect(image)
        assert len(result) == 1
        assert result[0].geometry.geographic_coordinates is None

    def test_detector_version(self):
        assert DummyDetector.__processor_version__ == '1.0.0'

    def test_detection_set_metadata(self):
        detector = DummyDetector(threshold=0.0)
        image = np.array([[1.0]], dtype=np.float64)
        result = detector.detect(image)
        assert result.detector_name == 'DummyDetector'
        assert result.detector_version == '1.0.0'


# ---------------------------------------------------------------------------
# Geo-registration
# ---------------------------------------------------------------------------

class _MockGeolocation:
    """Minimal mock for Geolocation that does a simple affine transform."""

    def image_to_latlon(self, row, col, height=0.0):
        """Simple transform: lat = row * 0.01, lon = col * 0.01."""
        if isinstance(row, np.ndarray):
            lats = row * 0.01
            lons = col * 0.01
            heights = np.full_like(row, height)
            return lats, lons, heights
        return float(row) * 0.01, float(col) * 0.01, height


class TestGeoRegistration:
    """Test _geo_register_detections."""

    def test_point_geo_registration(self):
        detector = DummyDetector(threshold=0.0)
        image = np.array([[1.0]], dtype=np.float64)
        geo = _MockGeolocation()
        result = detector.detect(image, geolocation=geo)

        assert len(result) == 1
        d = result[0]
        assert d.geometry.geographic_coordinates is not None
        lat, lon = d.geometry.geographic_coordinates
        assert lat == pytest.approx(0.0)  # row=0 * 0.01
        assert lon == pytest.approx(0.0)  # col=0 * 0.01

    def test_multi_point_geo_registration(self):
        detector = DummyDetector(threshold=0.0)
        image = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ], dtype=np.float64)
        geo = _MockGeolocation()
        result = detector.detect(image, geolocation=geo)

        assert len(result) == 4
        # Check pixel (1, 1) -> lat=0.01, lon=0.01
        d_11 = [d for d in result
                 if d.geometry.pixel_coordinates[0] == 1.0
                 and d.geometry.pixel_coordinates[1] == 1.0][0]
        lat, lon = d_11.geometry.geographic_coordinates
        assert lat == pytest.approx(0.01)
        assert lon == pytest.approx(0.01)

    def test_empty_detections_no_error(self):
        detector = DummyDetector(threshold=100.0)
        image = np.array([[0.5]], dtype=np.float64)
        geo = _MockGeolocation()
        result = detector.detect(image, geolocation=geo)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Detection input flow through detector
# ---------------------------------------------------------------------------

@processor_version('1.0.0')
class _BiasedDetector(ImageDetector):
    """Detector that accepts prior detections to bias threshold."""

    @property
    def output_schema(self) -> OutputSchema:
        return OutputSchema(fields=(
            OutputField('value', 'float', 'Detection value'),
        ))

    @property
    def detection_input_specs(self) -> Tuple[DetectionInputSpec, ...]:
        return (
            DetectionInputSpec(
                'prior_detections', required=False,
                description='Prior detections to bias threshold',
            ),
        )

    def detect(self, source, geolocation=None, **kwargs):
        self._validate_detection_inputs(kwargs)
        prior = self._get_detection_input('prior_detections', kwargs)

        threshold = 0.5
        if prior is not None and len(prior) > 0:
            threshold = 0.1  # lower threshold when priors exist

        rows, cols = np.where(source > threshold)
        detections = [
            Detection(
                Geometry.point(float(r), float(c)),
                {'value': float(source[r, c])},
            )
            for r, c in zip(rows, cols)
        ]
        return DetectionSet(
            detections=detections,
            detector_name='BiasedDetector',
            detector_version='1.0.0',
            output_schema=self.output_schema,
        )


class TestDetectionInputFlow:
    """Test detection-to-processor input flow."""

    def test_detect_without_prior(self):
        detector = _BiasedDetector()
        image = np.array([[0.3, 0.7]], dtype=np.float64)
        result = detector.detect(image)
        assert len(result) == 1  # only 0.7 > 0.5

    def test_detect_with_prior(self):
        detector = _BiasedDetector()
        image = np.array([[0.3, 0.7]], dtype=np.float64)

        # Create a dummy prior DetectionSet
        prior_schema = OutputSchema(fields=(
            OutputField('label', 'str', 'Label'),
        ))
        prior = DetectionSet(
            [Detection(Geometry.point(0, 0), {'label': 'x'})],
            'PriorDet', '1.0', prior_schema,
        )

        result = detector.detect(image, prior_detections=prior)
        assert len(result) == 2  # both 0.3 and 0.7 > 0.1 (biased threshold)
