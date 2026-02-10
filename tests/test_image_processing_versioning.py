# -*- coding: utf-8 -*-
"""
Processor Versioning Tests.

Tests for the @processor_version decorator, __init_subclass__ version
warnings on ImageProcessor, and DetectionInputSpec validation.

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

import warnings

import numpy as np
import pytest

from grdl.image_processing.base import ImageProcessor, ImageTransform
from grdl.image_processing.versioning import (
    DetectionInputSpec,
    processor_tags,
    processor_version,
)
from grdl.vocabulary import (
    DetectionType,
    ImageModality,
    ProcessorCategory,
    SegmentationType,
)


# ---------------------------------------------------------------------------
# @processor_version decorator
# ---------------------------------------------------------------------------

class TestProcessorVersionDecorator:
    """Test that @processor_version stamps the version correctly."""

    def test_stamps_version_on_class(self):
        """Decorated class has __processor_version__ attribute."""
        @processor_version('2.1.0')
        class _Versioned(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        assert hasattr(_Versioned, '__processor_version__')
        assert _Versioned.__processor_version__ == '2.1.0'

    def test_version_string_preserved_exactly(self):
        """Version string is exactly as provided."""
        @processor_version('0.0.1-alpha')
        class _Alpha(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        assert _Alpha.__processor_version__ == '0.0.1-alpha'

    def test_class_still_instantiable(self):
        """Decorated class can be instantiated normally."""
        @processor_version('1.0.0')
        class _Inst(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        obj = _Inst()
        result = obj.apply(np.zeros((2, 2)))
        assert result.shape == (2, 2)

    def test_works_on_image_processor_subclass(self):
        """Decorator works on direct ImageProcessor subclasses."""
        @processor_version('1.0.0')
        class _Direct(ImageProcessor):
            pass

        assert _Direct.__processor_version__ == '1.0.0'

    def test_works_on_plain_class(self):
        """Decorator works on any class, not just ImageProcessor."""
        @processor_version('3.0.0')
        class _Plain:
            pass

        assert _Plain.__processor_version__ == '3.0.0'

    def test_decorated_class_is_same_class(self):
        """Decorator returns the same class object, not a wrapper."""
        class _Original(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        decorated = processor_version('1.0.0')(_Original)
        assert decorated is _Original


# ---------------------------------------------------------------------------
# Version warning at instantiation
# ---------------------------------------------------------------------------

class TestMissingVersionWarning:
    """Test that unversioned concrete subclasses trigger UserWarning at instantiation."""

    def test_warns_for_undecorated_concrete_class(self):
        """Concrete subclass without @processor_version triggers warning on first instantiation."""
        class _Unversioned(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        # Clear the warned set so our class gets checked
        ImageProcessor._version_warned_classes.discard(_Unversioned)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _Unversioned()

            version_warnings = [
                x for x in w
                if issubclass(x.category, UserWarning)
                and 'processor version' in str(x.message).lower()
            ]
            assert len(version_warnings) == 1
            assert '_Unversioned' in str(version_warnings[0].message)

    def test_warns_only_once(self):
        """Warning fires only on first instantiation, not subsequent ones."""
        class _OnceOnly(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        ImageProcessor._version_warned_classes.discard(_OnceOnly)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _OnceOnly()
            _OnceOnly()
            _OnceOnly()

            version_warnings = [
                x for x in w
                if issubclass(x.category, UserWarning)
                and 'processor version' in str(x.message).lower()
            ]
            assert len(version_warnings) == 1

    def test_no_warning_for_decorated_class(self):
        """Decorated class does not trigger version warning."""
        @processor_version('1.0.0')
        class _Versioned(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        ImageProcessor._version_warned_classes.discard(_Versioned)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _Versioned()

            version_warnings = [
                x for x in w
                if issubclass(x.category, UserWarning)
                and 'processor version' in str(x.message).lower()
            ]
            assert len(version_warnings) == 0

    def test_no_warning_for_abstract_subclass(self):
        """Abstract class cannot be instantiated, so no warning fires."""
        from abc import abstractmethod

        class _AbstractMiddle(ImageProcessor):
            @abstractmethod
            def process(self):
                ...

        with pytest.raises(TypeError):
            _AbstractMiddle()

    def test_existing_classes_have_versions(self):
        """Orthorectifier and PauliDecomposition are decorated."""
        from grdl.image_processing.decomposition.pauli import PauliDecomposition
        from grdl.image_processing.ortho.ortho import Orthorectifier

        assert hasattr(PauliDecomposition, '__processor_version__')
        assert hasattr(Orthorectifier, '__processor_version__')


# ---------------------------------------------------------------------------
# ImageProcessor hierarchy
# ---------------------------------------------------------------------------

class TestImageProcessorHierarchy:
    """Verify the ImageProcessor class hierarchy."""

    def test_image_transform_is_image_processor(self):
        assert issubclass(ImageTransform, ImageProcessor)

    def test_polarimetric_decomposition_is_image_processor(self):
        from grdl.image_processing.decomposition.base import PolarimetricDecomposition
        assert issubclass(PolarimetricDecomposition, ImageProcessor)

    def test_image_detector_is_image_processor(self):
        from grdl.image_processing.detection.base import ImageDetector
        assert issubclass(ImageDetector, ImageProcessor)

    def test_pauli_isinstance_image_processor(self):
        from grdl.image_processing.decomposition.pauli import PauliDecomposition
        pauli = PauliDecomposition()
        assert isinstance(pauli, ImageProcessor)


# ---------------------------------------------------------------------------
# DetectionInputSpec
# ---------------------------------------------------------------------------

class TestDetectionInputSpec:
    """Test DetectionInputSpec construction and repr."""

    def test_construction(self):
        spec = DetectionInputSpec(
            name='prior_detections',
            required=True,
            description='Detections from a prior pass',
        )
        assert spec.name == 'prior_detections'
        assert spec.required is True
        assert spec.description == 'Detections from a prior pass'

    def test_repr(self):
        spec = DetectionInputSpec('mask', False, 'Region mask')
        r = repr(spec)
        assert 'mask' in r
        assert 'False' in r

    def test_optional_spec(self):
        spec = DetectionInputSpec('hints', False, 'Optional hints')
        assert spec.required is False


# ---------------------------------------------------------------------------
# Detection input validation on ImageProcessor
# ---------------------------------------------------------------------------

class TestDetectionInputValidation:
    """Test detection input declaration and validation."""

    def test_default_specs_empty(self):
        """ImageProcessor subclass has no detection inputs by default."""
        @processor_version('1.0.0')
        class _NoInputs(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        obj = _NoInputs()
        assert obj.detection_input_specs == ()

    def test_custom_specs(self):
        """Subclass can declare detection input specs."""
        @processor_version('1.0.0')
        class _WithInputs(ImageTransform):
            @property
            def detection_input_specs(self):
                return (
                    DetectionInputSpec('regions', True, 'Target regions'),
                )

            def apply(self, source, **kwargs):
                return source

        obj = _WithInputs()
        assert len(obj.detection_input_specs) == 1
        assert obj.detection_input_specs[0].name == 'regions'

    def test_validate_passes_when_required_present(self):
        """Validation passes when required inputs are in kwargs."""
        @processor_version('1.0.0')
        class _RequiredInput(ImageTransform):
            @property
            def detection_input_specs(self):
                return (
                    DetectionInputSpec('targets', True, 'Required targets'),
                )

            def apply(self, source, **kwargs):
                self._validate_detection_inputs(kwargs)
                return source

        obj = _RequiredInput()
        # Should not raise
        obj._validate_detection_inputs({'targets': 'dummy_detection_set'})

    def test_validate_raises_when_required_missing(self):
        """Validation raises ValueError when required input is missing."""
        @processor_version('1.0.0')
        class _RequiredInput(ImageTransform):
            @property
            def detection_input_specs(self):
                return (
                    DetectionInputSpec('targets', True, 'Required targets'),
                )

            def apply(self, source, **kwargs):
                self._validate_detection_inputs(kwargs)
                return source

        obj = _RequiredInput()
        with pytest.raises(ValueError, match="targets"):
            obj._validate_detection_inputs({})

    def test_validate_passes_when_optional_missing(self):
        """Validation passes when optional input is missing."""
        @processor_version('1.0.0')
        class _OptionalInput(ImageTransform):
            @property
            def detection_input_specs(self):
                return (
                    DetectionInputSpec('hints', False, 'Optional hints'),
                )

            def apply(self, source, **kwargs):
                self._validate_detection_inputs(kwargs)
                return source

        obj = _OptionalInput()
        # Should not raise
        obj._validate_detection_inputs({})

    def test_get_detection_input_present(self):
        """_get_detection_input returns the value when present."""
        @processor_version('1.0.0')
        class _Getter(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        obj = _Getter()
        result = obj._get_detection_input('targets', {'targets': 'value'})
        assert result == 'value'

    def test_get_detection_input_missing(self):
        """_get_detection_input returns None when missing."""
        @processor_version('1.0.0')
        class _Getter(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        obj = _Getter()
        result = obj._get_detection_input('targets', {})
        assert result is None


# ---------------------------------------------------------------------------
# Import isolation
# ---------------------------------------------------------------------------

class TestImports:
    """Verify public API imports work."""

    def test_import_processor_version(self):
        from grdl.image_processing import processor_version as pv
        assert pv is processor_version

    def test_import_detection_input_spec(self):
        from grdl.image_processing import DetectionInputSpec as DIS
        assert DIS is DetectionInputSpec

    def test_import_image_processor(self):
        from grdl.image_processing import ImageProcessor as IP
        assert IP is ImageProcessor

    def test_import_vocabulary_enums(self):
        from grdl.image_processing import (
            ImageModality as IM,
            ProcessorCategory as PC,
            DetectionType as DT,
            SegmentationType as ST,
        )
        assert IM is ImageModality
        assert PC is ProcessorCategory
        assert DT is DetectionType
        assert ST is SegmentationType


# ---------------------------------------------------------------------------
# @processor_tags decorator
# ---------------------------------------------------------------------------

class TestProcessorTagsDecorator:
    """Test that @processor_tags stamps enum-based tags correctly."""

    def test_stamps_tags_on_class(self):
        @processor_tags(
            modalities=[ImageModality.SAR, ImageModality.PAN],
            category=ProcessorCategory.FILTERS,
        )
        @processor_version('1.0.0')
        class _Tagged(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        assert hasattr(_Tagged, '__processor_tags__')
        tags = _Tagged.__processor_tags__
        assert ImageModality.SAR in tags['modalities']
        assert ImageModality.PAN in tags['modalities']
        assert tags['category'] is ProcessorCategory.FILTERS

    def test_modalities_stored_as_tuple(self):
        @processor_tags(modalities=[ImageModality.EO])
        @processor_version('1.0.0')
        class _T(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        assert isinstance(_T.__processor_tags__['modalities'], tuple)

    def test_empty_tags(self):
        @processor_tags()
        @processor_version('1.0.0')
        class _Empty(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        tags = _Empty.__processor_tags__
        assert tags['modalities'] == ()
        assert tags['category'] is None
        assert tags['description'] is None
        assert tags['detection_types'] == ()
        assert tags['segmentation_types'] == ()

    def test_detection_and_segmentation_types(self):
        @processor_tags(
            detection_types=[DetectionType.CLASSIFICATION],
            segmentation_types=[SegmentationType.SEMANTIC],
        )
        @processor_version('1.0.0')
        class _Detector(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        tags = _Detector.__processor_tags__
        assert DetectionType.CLASSIFICATION in tags['detection_types']
        assert SegmentationType.SEMANTIC in tags['segmentation_types']

    def test_rejects_string_modality(self):
        with pytest.raises(TypeError, match="ImageModality"):
            @processor_tags(modalities=['SAR'])
            class _Bad(ImageTransform):
                def apply(self, source, **kwargs):
                    return source

    def test_rejects_string_category(self):
        with pytest.raises(TypeError, match="ProcessorCategory"):
            @processor_tags(category='filters')
            class _Bad(ImageTransform):
                def apply(self, source, **kwargs):
                    return source

    def test_rejects_string_detection_type(self):
        with pytest.raises(TypeError, match="DetectionType"):
            @processor_tags(detection_types=['classification'])
            class _Bad(ImageTransform):
                def apply(self, source, **kwargs):
                    return source

    def test_description_still_accepts_string(self):
        @processor_tags(description="A useful filter")
        @processor_version('1.0.0')
        class _Desc(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        assert _Desc.__processor_tags__['description'] == "A useful filter"

    def test_decorated_class_is_same_class(self):
        class _Original(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        decorated = processor_tags(
            modalities=[ImageModality.SAR],
        )(_Original)
        assert decorated is _Original
