# -*- coding: utf-8 -*-
"""
Tunable Parameter Tests.

Tests for TunableParameterSpec declaration, validation, type checking,
default value resolution, range enforcement, and choices constraints
on ImageProcessor subclasses.

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

import numpy as np
import pytest

from grdl.image_processing.base import ImageProcessor, ImageTransform
from grdl.image_processing.versioning import (
    TunableParameterSpec,
    _NO_DEFAULT,
    _NoDefault,
    processor_version,
)


# ---------------------------------------------------------------------------
# TunableParameterSpec construction
# ---------------------------------------------------------------------------

class TestTunableParameterSpecConstruction:
    """Test TunableParameterSpec construction and attributes."""

    def test_basic_construction(self):
        """Spec stores name, param_type, default, and description."""
        spec = TunableParameterSpec(
            name='threshold',
            param_type=float,
            default=0.5,
            description='Detection threshold',
        )
        assert spec.name == 'threshold'
        assert spec.param_type is float
        assert spec.default == 0.5
        assert spec.description == 'Detection threshold'

    def test_required_when_no_default(self):
        """Spec is required when no default is provided."""
        spec = TunableParameterSpec(
            name='mode',
            param_type=str,
            description='Processing mode',
        )
        assert spec.required is True

    def test_not_required_when_default_provided(self):
        """Spec is optional when a default is provided."""
        spec = TunableParameterSpec(
            name='alpha',
            param_type=float,
            default=0.5,
        )
        assert spec.required is False

    def test_none_is_valid_default(self):
        """None as default means optional, not required."""
        spec = TunableParameterSpec(
            name='mask',
            param_type=object,
            default=None,
        )
        assert spec.required is False
        assert spec.default is None

    def test_min_max_stored(self):
        """Range constraints are stored on the spec."""
        spec = TunableParameterSpec(
            name='gain',
            param_type=float,
            default=1.0,
            min_value=0.0,
            max_value=10.0,
        )
        assert spec.min_value == 0.0
        assert spec.max_value == 10.0

    def test_choices_stored(self):
        """Choices constraint is stored on the spec."""
        spec = TunableParameterSpec(
            name='method',
            param_type=str,
            default='hard',
            choices=('hard', 'soft'),
        )
        assert spec.choices == ('hard', 'soft')

    def test_range_and_choices_mutually_exclusive(self):
        """Cannot specify both range and choices."""
        with pytest.raises(ValueError, match="cannot specify both"):
            TunableParameterSpec(
                name='x',
                param_type=float,
                default=1.0,
                min_value=0.0,
                choices=(1.0, 2.0),
            )

    def test_repr_required(self):
        """Repr for required spec omits default."""
        spec = TunableParameterSpec(
            name='mode',
            param_type=str,
        )
        r = repr(spec)
        assert 'mode' in r
        assert 'required=True' in r
        assert 'default=' not in r

    def test_repr_optional_with_constraints(self):
        """Repr for optional spec shows default and constraints."""
        spec = TunableParameterSpec(
            name='gain',
            param_type=float,
            default=1.0,
            min_value=0.0,
            max_value=10.0,
        )
        r = repr(spec)
        assert 'required=False' in r
        assert 'default=1.0' in r
        assert 'min_value=0.0' in r
        assert 'max_value=10.0' in r


# ---------------------------------------------------------------------------
# Default specs on ImageProcessor
# ---------------------------------------------------------------------------

class TestTunableParameterDefaultSpecs:
    """Test default tunable_parameter_specs behavior."""

    def test_default_specs_empty(self):
        """ImageTransform subclass returns empty tuple by default."""
        @processor_version('1.0.0')
        class _NoTunables(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        obj = _NoTunables()
        assert obj.tunable_parameter_specs == ()

    def test_custom_specs(self):
        """Subclass can override tunable_parameter_specs."""
        @processor_version('1.0.0')
        class _WithTunables(ImageTransform):
            @property
            def tunable_parameter_specs(self):
                return (
                    TunableParameterSpec('threshold', float, 0.5, 'A threshold'),
                )

            def apply(self, source, **kwargs):
                return source

        obj = _WithTunables()
        assert len(obj.tunable_parameter_specs) == 1
        assert obj.tunable_parameter_specs[0].name == 'threshold'


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestTunableParameterValidation:
    """Test _validate_tunable_parameters on ImageProcessor."""

    def _make_processor(self, specs):
        """Create a versioned processor with given tunable specs."""
        @processor_version('1.0.0')
        class _Proc(ImageTransform):
            @property
            def tunable_parameter_specs(self_inner):
                return specs

            def apply(self_inner, source, **kwargs):
                return source

        return _Proc()

    def test_validate_passes_all_present(self):
        """No error when all required params are provided."""
        proc = self._make_processor((
            TunableParameterSpec('x', float),
        ))
        proc._validate_tunable_parameters({'x': 1.0})

    def test_validate_raises_required_missing(self):
        """ValueError when required param is missing."""
        proc = self._make_processor((
            TunableParameterSpec('x', float, description='Required x'),
        ))
        with pytest.raises(ValueError, match="x"):
            proc._validate_tunable_parameters({})

    def test_validate_passes_optional_missing(self):
        """No error when optional param is missing."""
        proc = self._make_processor((
            TunableParameterSpec('x', float, default=0.5),
        ))
        proc._validate_tunable_parameters({})

    def test_validate_type_correct(self):
        """No error for correct type."""
        proc = self._make_processor((
            TunableParameterSpec('x', float, default=0.5),
        ))
        proc._validate_tunable_parameters({'x': 0.7})

    def test_validate_int_accepted_as_float(self):
        """int is accepted when param_type is float."""
        proc = self._make_processor((
            TunableParameterSpec('x', float, default=0.5),
        ))
        proc._validate_tunable_parameters({'x': 1})

    def test_validate_type_wrong_raises(self):
        """TypeError when value has wrong type."""
        proc = self._make_processor((
            TunableParameterSpec('x', float, default=0.5),
        ))
        with pytest.raises(TypeError, match="x"):
            proc._validate_tunable_parameters({'x': 'not_a_float'})

    def test_validate_min_value_passes(self):
        """No error when value equals min_value."""
        proc = self._make_processor((
            TunableParameterSpec('x', float, default=0.5, min_value=0.0),
        ))
        proc._validate_tunable_parameters({'x': 0.0})

    def test_validate_min_value_below_raises(self):
        """ValueError when value is below min_value."""
        proc = self._make_processor((
            TunableParameterSpec('x', float, default=0.5, min_value=0.0),
        ))
        with pytest.raises(ValueError, match="below minimum"):
            proc._validate_tunable_parameters({'x': -0.1})

    def test_validate_max_value_passes(self):
        """No error when value equals max_value."""
        proc = self._make_processor((
            TunableParameterSpec('x', float, default=0.5, max_value=1.0),
        ))
        proc._validate_tunable_parameters({'x': 1.0})

    def test_validate_max_value_above_raises(self):
        """ValueError when value is above max_value."""
        proc = self._make_processor((
            TunableParameterSpec('x', float, default=0.5, max_value=1.0),
        ))
        with pytest.raises(ValueError, match="above maximum"):
            proc._validate_tunable_parameters({'x': 1.1})

    def test_validate_choices_valid(self):
        """No error when value is in choices."""
        proc = self._make_processor((
            TunableParameterSpec('m', str, default='a', choices=('a', 'b')),
        ))
        proc._validate_tunable_parameters({'m': 'b'})

    def test_validate_choices_invalid_raises(self):
        """ValueError when value is not in choices."""
        proc = self._make_processor((
            TunableParameterSpec('m', str, default='a', choices=('a', 'b')),
        ))
        with pytest.raises(ValueError, match="not in allowed choices"):
            proc._validate_tunable_parameters({'m': 'c'})

    def test_validate_multiple_params(self):
        """Validates all params, not just the first."""
        proc = self._make_processor((
            TunableParameterSpec('x', float, default=0.5, min_value=0.0),
            TunableParameterSpec('m', str, default='a', choices=('a', 'b')),
        ))
        # Both valid
        proc._validate_tunable_parameters({'x': 0.5, 'm': 'a'})
        # Second invalid
        with pytest.raises(ValueError, match="not in allowed choices"):
            proc._validate_tunable_parameters({'x': 0.5, 'm': 'c'})


# ---------------------------------------------------------------------------
# _get_tunable_parameter
# ---------------------------------------------------------------------------

class TestGetTunableParameter:
    """Test _get_tunable_parameter default resolution."""

    def _make_processor(self, specs):
        @processor_version('1.0.0')
        class _Proc(ImageTransform):
            @property
            def tunable_parameter_specs(self_inner):
                return specs

            def apply(self_inner, source, **kwargs):
                return source

        return _Proc()

    def test_returns_value_when_present(self):
        """Returns the kwarg value when provided."""
        proc = self._make_processor((
            TunableParameterSpec('x', float, default=0.5),
        ))
        assert proc._get_tunable_parameter('x', {'x': 0.9}) == 0.9

    def test_returns_default_when_absent(self):
        """Returns spec default when kwarg is not provided."""
        proc = self._make_processor((
            TunableParameterSpec('x', float, default=0.5),
        ))
        assert proc._get_tunable_parameter('x', {}) == 0.5

    def test_returns_none_when_no_spec(self):
        """Returns None when no matching spec exists."""
        proc = self._make_processor(())
        assert proc._get_tunable_parameter('unknown', {}) is None

    def test_returns_none_default_correctly(self):
        """When default is None, returns None (not confused with no-spec)."""
        proc = self._make_processor((
            TunableParameterSpec('x', object, default=None),
        ))
        assert proc._get_tunable_parameter('x', {}) is None


# ---------------------------------------------------------------------------
# Integration: concrete processor using tunable parameters
# ---------------------------------------------------------------------------

@processor_version('1.0.0')
class _ThresholdTransform(ImageTransform):
    """Test transform that uses tunable threshold and method params."""

    @property
    def tunable_parameter_specs(self):
        return (
            TunableParameterSpec(
                name='threshold',
                param_type=float,
                default=0.5,
                description='Pixel intensity threshold',
                min_value=0.0,
                max_value=1.0,
            ),
            TunableParameterSpec(
                name='method',
                param_type=str,
                default='hard',
                description='Thresholding method',
                choices=('hard', 'soft'),
            ),
        )

    def apply(self, source, **kwargs):
        self._validate_tunable_parameters(kwargs)
        threshold = self._get_tunable_parameter('threshold', kwargs)
        method = self._get_tunable_parameter('method', kwargs)
        if method == 'hard':
            return (source > threshold).astype(source.dtype)
        else:
            return np.clip(source - threshold, 0, None)


class TestTunableParameterIntegration:
    """End-to-end tests with a concrete tunable processor."""

    def test_apply_with_defaults(self):
        """apply() uses default threshold=0.5, method='hard'."""
        proc = _ThresholdTransform()
        img = np.array([[0.3, 0.7], [0.5, 0.9]])
        result = proc.apply(img)
        expected = (img > 0.5).astype(img.dtype)
        np.testing.assert_array_equal(result, expected)

    def test_apply_with_custom_values(self):
        """apply() uses provided threshold and method."""
        proc = _ThresholdTransform()
        img = np.array([[0.3, 0.7], [0.5, 0.9]])
        result = proc.apply(img, threshold=0.3, method='soft')
        expected = np.clip(img - 0.3, 0, None)
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_with_invalid_type_raises(self):
        """TypeError when threshold is a string."""
        proc = _ThresholdTransform()
        img = np.zeros((2, 2))
        with pytest.raises(TypeError, match="threshold"):
            proc.apply(img, threshold='high')

    def test_apply_with_out_of_range_raises(self):
        """ValueError when threshold exceeds max."""
        proc = _ThresholdTransform()
        img = np.zeros((2, 2))
        with pytest.raises(ValueError, match="above maximum"):
            proc.apply(img, threshold=2.0)

    def test_apply_with_invalid_choice_raises(self):
        """ValueError when method is not in choices."""
        proc = _ThresholdTransform()
        img = np.zeros((2, 2))
        with pytest.raises(ValueError, match="not in allowed choices"):
            proc.apply(img, method='fuzzy')


# ---------------------------------------------------------------------------
# Import checks
# ---------------------------------------------------------------------------

class TestTunableParameterImports:
    """Verify public API imports work."""

    def test_import_from_versioning(self):
        from grdl.image_processing.versioning import TunableParameterSpec as TPS
        assert TPS is TunableParameterSpec

    def test_import_from_image_processing(self):
        from grdl.image_processing import TunableParameterSpec as TPS
        assert TPS is TunableParameterSpec
