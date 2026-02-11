# -*- coding: utf-8 -*-
"""
Image Detection Tests.

Tests for the detection data models (Detection, DetectionSet), the
ImageDetector ABC, geo-registration, the data dictionary (FieldDefinition,
DATA_DICTIONARY, Fields), and detection input flow.
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
2026-02-11
"""

import warnings
from typing import Any, Optional, Tuple

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from grdl.image_processing.base import ImageProcessor
from grdl.image_processing.detection.base import ImageDetector
from grdl.image_processing.detection.models import (
    Detection,
    DetectionSet,
)
from grdl.image_processing.detection.fields import (
    DATA_DICTIONARY,
    FieldDefinition,
    Fields,
    is_dictionary_field,
    list_fields,
    lookup_field,
)
from grdl.image_processing.versioning import (
    DetectionInputSpec,
    processor_version,
)


# ---------------------------------------------------------------------------
# FieldDefinition
# ---------------------------------------------------------------------------

class TestFieldDefinition:
    """Test FieldDefinition construction."""

    def test_construction(self):
        f = FieldDefinition('sar.sigma0', 'float', 'Sigma naught', 'dB')
        assert f.name == 'sar.sigma0'
        assert f.dtype == 'float'
        assert f.description == 'Sigma naught'
        assert f.units == 'dB'
        assert f.domain == 'sar'

    def test_without_units(self):
        f = FieldDefinition('identity.label', 'str', 'Class label')
        assert f.units is None
        assert f.domain == 'identity'

    def test_domain_extraction(self):
        f = FieldDefinition('physical.rcs_db', 'float', 'RCS in dB', 'dBsm')
        assert f.domain == 'physical'

    def test_repr(self):
        f = FieldDefinition('sar.sigma0', 'float', 'Sigma naught', 'dB')
        r = repr(f)
        assert 'sar.sigma0' in r
        assert 'dB' in r


# ---------------------------------------------------------------------------
# Data Dictionary
# ---------------------------------------------------------------------------

class TestDataDictionary:
    """Test DATA_DICTIONARY contents and helpers."""

    def test_dictionary_not_empty(self):
        assert len(DATA_DICTIONARY) > 0

    def test_all_entries_are_field_definitions(self):
        for name, field in DATA_DICTIONARY.items():
            assert isinstance(field, FieldDefinition)
            assert field.name == name

    def test_domains_present(self):
        domains = {f.domain for f in DATA_DICTIONARY.values()}
        expected = {
            'physical', 'sar', 'spectral', 'volume',
            'identity', 'trait', 'temporal', 'context',
        }
        assert expected == domains

    def test_lookup_field_existing(self):
        f = lookup_field('sar.coherence')
        assert f is not None
        assert f.name == 'sar.coherence'
        assert f.dtype == 'float'

    def test_lookup_field_missing(self):
        assert lookup_field('nonexistent.field') is None

    def test_is_dictionary_field(self):
        assert is_dictionary_field('identity.label') is True
        assert is_dictionary_field('custom_field') is False

    def test_list_fields_all(self):
        all_fields = list_fields()
        assert len(all_fields) == len(DATA_DICTIONARY)
        # Sorted by name
        names = [f.name for f in all_fields]
        assert names == sorted(names)

    def test_list_fields_by_domain(self):
        sar_fields = list_fields(domain='sar')
        assert len(sar_fields) > 0
        assert all(f.domain == 'sar' for f in sar_fields)

    def test_list_fields_empty_domain(self):
        result = list_fields(domain='nonexistent')
        assert result == []


# ---------------------------------------------------------------------------
# Fields Accessor
# ---------------------------------------------------------------------------

class TestFieldsAccessor:
    """Test Fields constant accessor class."""

    def test_sar_field(self):
        assert Fields.sar.CHANGE_MAGNITUDE == 'sar.change_magnitude'

    def test_identity_field(self):
        assert Fields.identity.LABEL == 'identity.label'

    def test_physical_field(self):
        assert Fields.physical.RCS_DB == 'physical.rcs_db'

    def test_spectral_field(self):
        assert Fields.spectral.NDVI == 'spectral.ndvi'

    def test_volume_field(self):
        assert Fields.volume.CANOPY_HEIGHT == 'volume.canopy_height'

    def test_trait_field(self):
        assert Fields.trait.COLOR == 'trait.color'

    def test_temporal_field(self):
        assert Fields.temporal.FIRST_SEEN == 'temporal.first_seen'

    def test_context_field(self):
        assert Fields.context.ELEVATION == 'context.elevation'

    def test_all_accessor_values_are_in_dictionary(self):
        """Every Fields constant should map to a DATA_DICTIONARY entry."""
        for domain_name in [
            'physical', 'sar', 'spectral', 'volume',
            'identity', 'trait', 'temporal', 'context',
        ]:
            domain_obj = getattr(Fields, domain_name)
            for attr in dir(domain_obj):
                if attr.startswith('_'):
                    continue
                value = getattr(domain_obj, attr)
                assert is_dictionary_field(value), (
                    f"Fields.{domain_name}.{attr} = {value!r} "
                    f"not in DATA_DICTIONARY"
                )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestDetection:
    """Test Detection construction and GeoJSON export."""

    def test_construction_point(self):
        geom = Point(20.0, 10.0)  # x=col=20, y=row=10
        d = Detection(geom, {'identity.label': 'car'}, confidence=0.9)
        assert d.pixel_geometry is geom
        assert d.properties == {'identity.label': 'car'}
        assert d.confidence == 0.9
        assert d.geo_geometry is None

    def test_construction_with_geo(self):
        pixel = Point(20.0, 10.0)
        geo = Point(-73.0, 45.0)  # x=lon, y=lat
        d = Detection(pixel, {}, geo_geometry=geo)
        assert d.geo_geometry is geo

    def test_confidence_optional(self):
        d = Detection(Point(5.0, 5.0), {'value': 1.0})
        assert d.confidence is None

    def test_to_geojson_feature_with_geo(self):
        pixel = Point(20.0, 10.0)
        geo = Point(-73.0, 45.0)
        d = Detection(
            pixel, {'identity.label': 'tree'}, confidence=0.8,
            geo_geometry=geo,
        )
        feature = d.to_geojson_feature()
        assert feature['type'] == 'Feature'
        assert feature['geometry']['type'] == 'Point'
        # GeoJSON coords from shapely mapping: [lon, lat] = [-73, 45]
        assert feature['geometry']['coordinates'] == (-73.0, 45.0)
        assert feature['properties']['identity.label'] == 'tree'
        assert feature['properties']['confidence'] == 0.8

    def test_to_geojson_feature_pixel_fallback(self):
        d = Detection(Point(20.0, 10.0), {'value': 3.0})
        feature = d.to_geojson_feature()
        assert feature['geometry']['type'] == 'Point'
        # Falls back to pixel geometry: (col=20, row=10)
        assert feature['geometry']['coordinates'] == (20.0, 10.0)
        assert 'confidence' not in feature['properties']

    def test_to_geojson_feature_no_confidence(self):
        d = Detection(Point(20.0, 10.0), {'value': 3.0})
        feature = d.to_geojson_feature()
        assert 'confidence' not in feature['properties']
        assert feature['properties']['value'] == 3.0

    def test_to_geojson_bbox(self):
        """Bounding box via shapely.geometry.box()."""
        geom = box(10.0, 5.0, 100.0, 50.0)  # minx, miny, maxx, maxy
        d = Detection(geom, {})
        feature = d.to_geojson_feature()
        assert feature['geometry']['type'] == 'Polygon'

    def test_to_geojson_polygon(self):
        verts = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        geom = Polygon(verts)
        d = Detection(geom, {})
        feature = d.to_geojson_feature()
        assert feature['geometry']['type'] == 'Polygon'
        coords = feature['geometry']['coordinates']
        assert len(coords) == 1  # one ring
        assert len(coords[0]) == 5  # closed ring

    def test_repr(self):
        d = Detection(Point(1.0, 2.0), {}, confidence=0.5)
        r = repr(d)
        assert 'Point' in r
        assert '0.5' in r


# ---------------------------------------------------------------------------
# DetectionSet
# ---------------------------------------------------------------------------

class TestDetectionSet:
    """Test DetectionSet construction, protocol, and methods."""

    @pytest.fixture
    def sample_set(self):
        detections = [
            Detection(
                Point(float(i), float(i)),
                {Fields.identity.LABEL: f'obj_{i}'},
                confidence=i * 0.25,
            )
            for i in range(4)
        ]
        return DetectionSet(
            detections=detections,
            detector_name='TestDetector',
            detector_version='1.0.0',
            output_fields=(Fields.identity.LABEL,),
        )

    def test_len(self, sample_set):
        assert len(sample_set) == 4

    def test_iter(self, sample_set):
        items = list(sample_set)
        assert len(items) == 4
        assert all(isinstance(d, Detection) for d in items)

    def test_getitem(self, sample_set):
        d = sample_set[2]
        assert d.properties[Fields.identity.LABEL] == 'obj_2'

    def test_to_geojson(self, sample_set):
        gj = sample_set.to_geojson()
        assert gj['type'] == 'FeatureCollection'
        assert len(gj['features']) == 4
        assert gj['properties']['detector_name'] == 'TestDetector'
        assert gj['properties']['detector_version'] == '1.0.0'
        assert gj['properties']['output_fields'] == (Fields.identity.LABEL,)

    def test_filter_by_confidence(self, sample_set):
        filtered = sample_set.filter_by_confidence(0.5)
        assert len(filtered) == 2  # confidence 0.5 and 0.75
        assert filtered.detector_name == 'TestDetector'

    def test_filter_excludes_none_confidence(self):
        detections = [
            Detection(Point(0, 0), {Fields.identity.LABEL: 'a'}),
            Detection(
                Point(1, 1), {Fields.identity.LABEL: 'b'}, confidence=0.9,
            ),
        ]
        ds = DetectionSet(
            detections, 'Det', '1.0',
            output_fields=(Fields.identity.LABEL,),
        )
        filtered = ds.filter_by_confidence(0.0)
        assert len(filtered) == 1

    def test_empty_set(self):
        ds = DetectionSet([], 'Empty', '1.0')
        assert len(ds) == 0
        gj = ds.to_geojson()
        assert gj['features'] == []

    def test_metadata(self):
        ds = DetectionSet(
            [], 'Det', '1.0',
            metadata={'processing_time_ms': 42},
        )
        assert ds.metadata['processing_time_ms'] == 42
        gj = ds.to_geojson()
        assert gj['properties']['processing_time_ms'] == 42

    def test_non_dictionary_field_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            DetectionSet(
                [], 'Det', '1.0',
                output_fields=('custom_nonstandard_field',),
            )
            assert len(w) == 1
            assert 'custom_nonstandard_field' in str(w[0].message)
            assert 'data dictionary' in str(w[0].message)

    def test_dictionary_field_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            DetectionSet(
                [], 'Det', '1.0',
                output_fields=(Fields.sar.CHANGE_MAGNITUDE,),
            )
            assert len(w) == 0

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
    def output_fields(self) -> Tuple[str, ...]:
        return ('intensity',)

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
            geom = Point(float(c), float(r))  # x=col, y=row
            props = {'intensity': float(source[r, c])}
            detections.append(
                Detection(geom, props, confidence=float(source[r, c]))
            )

        ds = DetectionSet(
            detections=detections,
            detector_name=self.__class__.__name__,
            detector_version=getattr(
                self.__class__, '__processor_version__', 'unknown'
            ),
            output_fields=self.output_fields,
        )

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

    def test_output_fields_property(self):
        detector = DummyDetector()
        fields = detector.output_fields
        assert isinstance(fields, tuple)
        assert 'intensity' in fields

    def test_detect_without_geolocation(self):
        detector = DummyDetector(threshold=0.0)
        image = np.array([[1.0]], dtype=np.float64)
        result = detector.detect(image)
        assert len(result) == 1
        assert result[0].geo_geometry is None

    def test_detector_version(self):
        assert DummyDetector.__processor_version__ == '1.0.0'

    def test_detection_set_metadata(self):
        detector = DummyDetector(threshold=0.0)
        image = np.array([[1.0]], dtype=np.float64)
        result = detector.detect(image)
        assert result.detector_name == 'DummyDetector'
        assert result.detector_version == '1.0.0'




# ---------------------------------------------------------------------------
# Detection input flow through detector
# ---------------------------------------------------------------------------

@processor_version('1.0.0')
class _BiasedDetector(ImageDetector):
    """Detector that accepts prior detections to bias threshold."""

    @property
    def output_fields(self) -> Tuple[str, ...]:
        return ('value',)

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
                Point(float(c), float(r)),
                {'value': float(source[r, c])},
            )
            for r, c in zip(rows, cols)
        ]
        return DetectionSet(
            detections=detections,
            detector_name='BiasedDetector',
            detector_version='1.0.0',
            output_fields=self.output_fields,
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
        prior = DetectionSet(
            [Detection(Point(0, 0), {Fields.identity.LABEL: 'x'})],
            'PriorDet', '1.0',
            output_fields=(Fields.identity.LABEL,),
        )

        result = detector.detect(image, prior_detections=prior)
        assert len(result) == 2  # both 0.3 and 0.7 > 0.1 (biased threshold)
