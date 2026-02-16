# -*- coding: utf-8 -*-
"""
Tests for CFAR detection module.

Covers validation, all four CFAR variants, cross-variant comparisons,
geolocation integration, runtime parameter overrides, and processor
metadata contracts.

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
2026-02-16

Modified
--------
2026-02-16
"""

import numpy as np
import pytest

from grdl.exceptions import ValidationError
from grdl.image_processing.detection import (
    CACFARDetector,
    CFARDetector,
    DetectionSet,
    Fields,
    GOCFARDetector,
    OSCFARDetector,
    SOCFARDetector,
    is_dictionary_field,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def uniform_bg_with_target():
    """200x200 image: background=10.0, 10x10 bright square at center=50.0."""
    image = np.full((200, 200), 10.0)
    image[95:105, 95:105] = 50.0
    return image


@pytest.fixture
def clutter_edge_image():
    """200x200: left half=10.0, right half=30.0, bright target in low region."""
    image = np.zeros((200, 200))
    image[:, :100] = 10.0
    image[:, 100:] = 30.0
    # Bright target well inside the low-clutter region
    image[100, 50] = 80.0
    return image


@pytest.fixture
def interfering_targets_image():
    """200x200 bg=10.0 with clustered interfering targets + isolated target."""
    image = np.full((200, 200), 10.0)
    # Dense cluster of bright targets
    for r, c in [(100, 100), (100, 105), (105, 100), (105, 105), (102, 102)]:
        image[r, c] = 60.0
    # Isolated target far from cluster
    image[50, 50] = 60.0
    return image


@pytest.fixture
def uniform_image():
    """200x200 uniform image at 10.0 (no targets)."""
    return np.full((200, 200), 10.0)


class _MockGeolocation:
    """Simple mock for pixel-to-geographic transforms."""

    def image_to_latlon(self, row, col, height=0.0):
        return (40.0 + float(row) * 0.001, -74.0 + float(col) * 0.001, 0.0)


# ===================================================================
# Validation tests
# ===================================================================

class TestCFARValidation:
    """Tests for CFAR parameter validation.

    Invalid parameters may be caught by the GRDL ``Annotated`` param
    system (raises ``ValueError``/``TypeError``) or by the custom
    ``__init__`` validation helpers (raises ``ValidationError``).
    """

    def test_guard_geq_training_raises(self):
        with pytest.raises((ValidationError, ValueError)):
            CACFARDetector(guard_cells=12, training_cells=12)

    def test_guard_greater_than_training_raises(self):
        with pytest.raises((ValidationError, ValueError)):
            CACFARDetector(guard_cells=15, training_cells=12)

    def test_negative_guard_raises(self):
        with pytest.raises((ValidationError, ValueError)):
            CACFARDetector(guard_cells=-1, training_cells=12)

    def test_zero_guard_raises(self):
        with pytest.raises((ValidationError, ValueError)):
            CACFARDetector(guard_cells=0, training_cells=12)

    def test_pfa_zero_raises(self):
        with pytest.raises((ValidationError, ValueError)):
            CACFARDetector(pfa=0.0)

    def test_pfa_one_raises(self):
        with pytest.raises((ValidationError, ValueError)):
            CACFARDetector(pfa=1.0)

    def test_pfa_negative_raises(self):
        with pytest.raises((ValidationError, ValueError)):
            CACFARDetector(pfa=-0.01)

    def test_invalid_assumption_raises(self):
        with pytest.raises((ValidationError, ValueError)):
            CACFARDetector(assumption='rayleigh')

    def test_valid_params_accepted(self):
        det = CACFARDetector(
            guard_cells=3, training_cells=12,
            pfa=1e-3, min_pixels=9, assumption='gaussian',
        )
        assert det.guard_cells == 3
        assert det.training_cells == 12

    def test_requires_2d_input(self, uniform_bg_with_target):
        det = CACFARDetector()
        with pytest.raises(ValidationError, match="2D"):
            det.detect(np.ones((3, 100, 100)))


# ===================================================================
# CA-CFAR tests
# ===================================================================

class TestCACFARDetector:
    """Tests for Cell-Averaging CFAR."""

    def test_detects_bright_target(self, uniform_bg_with_target):
        det = CACFARDetector(pfa=1e-3, min_pixels=5)
        result = det.detect(uniform_bg_with_target)
        assert isinstance(result, DetectionSet)
        assert len(result) >= 1

    def test_no_detections_on_uniform(self, uniform_image):
        det = CACFARDetector(pfa=1e-3, min_pixels=5)
        result = det.detect(uniform_image)
        assert len(result) == 0

    def test_output_is_detection_set(self, uniform_bg_with_target):
        det = CACFARDetector()
        result = det.detect(uniform_bg_with_target)
        assert isinstance(result, DetectionSet)
        assert result.detector_name == 'CACFARDetector'
        assert result.detector_version == '1.0.0'

    def test_detection_has_correct_fields(self, uniform_bg_with_target):
        det = CACFARDetector(pfa=1e-3, min_pixels=5)
        result = det.detect(uniform_bg_with_target)
        assert len(result) >= 1
        d = result[0]
        assert Fields.sar.SIGMA0 in d.properties
        assert Fields.identity.IS_TARGET in d.properties
        assert Fields.physical.AREA in d.properties
        assert Fields.physical.LENGTH in d.properties
        assert Fields.physical.WIDTH in d.properties
        assert d.properties[Fields.identity.IS_TARGET] is True

    def test_output_fields_in_data_dictionary(self):
        det = CACFARDetector()
        for field in det.output_fields:
            assert is_dictionary_field(field), f"{field} not in data dictionary"

    def test_processor_version_set(self):
        assert CACFARDetector.__processor_version__ == '1.0.0'

    def test_processor_tags_set(self):
        tags = CACFARDetector.__processor_tags__
        assert 'modalities' in tags
        assert 'category' in tags

    def test_min_pixels_filters_small(self, uniform_bg_with_target):
        det_loose = CACFARDetector(pfa=1e-3, min_pixels=1)
        det_strict = CACFARDetector(pfa=1e-3, min_pixels=200)
        result_loose = det_loose.detect(uniform_bg_with_target)
        result_strict = det_strict.detect(uniform_bg_with_target)
        assert len(result_loose) >= len(result_strict)

    def test_gaussian_assumption(self, uniform_bg_with_target):
        det = CACFARDetector(pfa=1e-3, assumption='gaussian')
        result = det.detect(uniform_bg_with_target)
        assert result.metadata['assumption'] == 'gaussian'

    def test_exponential_assumption(self):
        # Exponential model needs a much brighter target relative to
        # background (threshold = alpha * bg_mean, alpha ~7 for pfa=1e-3).
        image = np.full((200, 200), 10.0)
        image[95:105, 95:105] = 200.0  # 20x background
        det = CACFARDetector(pfa=1e-3, assumption='exponential', min_pixels=5)
        result = det.detect(image)
        assert result.metadata['assumption'] == 'exponential'
        assert len(result) >= 1

    def test_runtime_param_override(self, uniform_bg_with_target):
        det = CACFARDetector(pfa=1e-3, min_pixels=5)
        result_default = det.detect(uniform_bg_with_target)
        # Very strict min_pixels should filter more
        result_strict = det.detect(uniform_bg_with_target, min_pixels=500)
        assert len(result_default) >= len(result_strict)

    def test_geolocation_populates_geo(self, uniform_bg_with_target):
        det = CACFARDetector(pfa=1e-3, min_pixels=5)
        geo = _MockGeolocation()
        result = det.detect(uniform_bg_with_target, geolocation=geo)
        assert len(result) >= 1
        d = result[0]
        assert d.geo_geometry is not None

    def test_confidence_in_0_1(self, uniform_bg_with_target):
        det = CACFARDetector(pfa=1e-3, min_pixels=5)
        result = det.detect(uniform_bg_with_target)
        for d in result:
            assert 0.0 <= d.confidence <= 1.0

    def test_metadata_in_detection_set(self, uniform_bg_with_target):
        det = CACFARDetector(pfa=1e-3, min_pixels=5)
        result = det.detect(uniform_bg_with_target)
        assert 'guard_cells' in result.metadata
        assert 'training_cells' in result.metadata
        assert 'pfa' in result.metadata
        assert 'image_shape' in result.metadata

    def test_detection_pixel_geometry_is_box(self, uniform_bg_with_target):
        det = CACFARDetector(pfa=1e-3, min_pixels=5)
        result = det.detect(uniform_bg_with_target)
        assert len(result) >= 1
        geom = result[0].pixel_geometry
        assert geom.geom_type == 'Polygon'
        # Bounding box should have positive area
        assert geom.area > 0

    def test_geojson_export(self, uniform_bg_with_target):
        det = CACFARDetector(pfa=1e-3, min_pixels=5)
        result = det.detect(uniform_bg_with_target)
        geojson = result.to_geojson()
        assert geojson['type'] == 'FeatureCollection'
        assert len(geojson['features']) == len(result)


# ===================================================================
# GO-CFAR tests
# ===================================================================

class TestGOCFARDetector:
    """Tests for Greatest-Of CFAR."""

    def test_detects_bright_target(self, uniform_bg_with_target):
        det = GOCFARDetector(pfa=1e-3, min_pixels=5)
        result = det.detect(uniform_bg_with_target)
        assert len(result) >= 1

    def test_no_detections_on_uniform(self, uniform_image):
        det = GOCFARDetector(pfa=1e-3, min_pixels=5)
        result = det.detect(uniform_image)
        assert len(result) == 0

    def test_output_is_detection_set(self, uniform_bg_with_target):
        det = GOCFARDetector()
        result = det.detect(uniform_bg_with_target)
        assert isinstance(result, DetectionSet)
        assert result.detector_name == 'GOCFARDetector'

    def test_processor_version(self):
        assert GOCFARDetector.__processor_version__ == '1.0.0'


# ===================================================================
# SO-CFAR tests
# ===================================================================

class TestSOCFARDetector:
    """Tests for Smallest-Of CFAR."""

    def test_detects_bright_target(self, uniform_bg_with_target):
        det = SOCFARDetector(pfa=1e-3, min_pixels=5)
        result = det.detect(uniform_bg_with_target)
        assert len(result) >= 1

    def test_no_detections_on_uniform(self, uniform_image):
        det = SOCFARDetector(pfa=1e-3, min_pixels=5)
        result = det.detect(uniform_image)
        assert len(result) == 0

    def test_output_is_detection_set(self, uniform_bg_with_target):
        det = SOCFARDetector()
        result = det.detect(uniform_bg_with_target)
        assert isinstance(result, DetectionSet)
        assert result.detector_name == 'SOCFARDetector'

    def test_processor_version(self):
        assert SOCFARDetector.__processor_version__ == '1.0.0'


# ===================================================================
# OS-CFAR tests
# ===================================================================

class TestOSCFARDetector:
    """Tests for Ordered-Statistics CFAR."""

    def test_detects_bright_target(self, uniform_bg_with_target):
        det = OSCFARDetector(pfa=1e-3, min_pixels=5, percentile=75.0)
        result = det.detect(uniform_bg_with_target)
        assert len(result) >= 1

    def test_no_detections_on_uniform(self, uniform_image):
        det = OSCFARDetector(pfa=1e-3, min_pixels=5)
        result = det.detect(uniform_image)
        assert len(result) == 0

    def test_percentile_parameter(self):
        det = OSCFARDetector(percentile=50.0)
        assert det.percentile == 50.0

    def test_output_is_detection_set(self, uniform_bg_with_target):
        det = OSCFARDetector()
        result = det.detect(uniform_bg_with_target)
        assert isinstance(result, DetectionSet)
        assert result.detector_name == 'OSCFARDetector'

    def test_processor_version(self):
        assert OSCFARDetector.__processor_version__ == '1.0.0'


# ===================================================================
# Cross-variant comparisons
# ===================================================================

class TestCFARCrossVariant:
    """Compare behavior across CFAR variants."""

    def test_all_detect_obvious_target(self, uniform_bg_with_target):
        """All variants should detect a 40-dB-above-background target."""
        for cls in (CACFARDetector, GOCFARDetector, SOCFARDetector, OSCFARDetector):
            det = cls(pfa=1e-3, min_pixels=5)
            result = det.detect(uniform_bg_with_target)
            assert len(result) >= 1, f"{cls.__name__} missed obvious target"

    def test_all_empty_on_uniform(self, uniform_image):
        """No variant should detect anything in a uniform image."""
        for cls in (CACFARDetector, GOCFARDetector, SOCFARDetector, OSCFARDetector):
            det = cls(pfa=1e-3, min_pixels=5)
            result = det.detect(uniform_image)
            assert len(result) == 0, f"{cls.__name__} false alarm on uniform"

    def test_go_fewer_detections_than_so_at_edge(self, clutter_edge_image):
        """GO-CFAR (conservative) should produce <= detections than SO-CFAR at edges."""
        go = GOCFARDetector(pfa=1e-2, min_pixels=1)
        so = SOCFARDetector(pfa=1e-2, min_pixels=1)
        go_result = go.detect(clutter_edge_image)
        so_result = so.detect(clutter_edge_image)
        # SO is more aggressive, should have >= detections
        assert len(so_result) >= len(go_result)

    def test_cfar_detector_is_abstract(self):
        """CFARDetector cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CFARDetector()
