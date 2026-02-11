# -*- coding: utf-8 -*-
"""
Annotated Tunable Parameter Tests.

Tests for the typing.Annotated-based tunable parameter system: constraint
markers (Range, Options, Desc), ParamSpec introspection, __init_subclass__
annotation collection, auto-generated __init__, __post_init__ hook,
_resolve_params runtime resolution, validation, and inheritance.

Author
------
Steven Siebert

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

from typing import Annotated

import numpy as np
import pytest

from grdl.image_processing.base import ImageProcessor, ImageTransform
from grdl.image_processing.params import (
    Desc,
    Options,
    ParamMeta,
    ParamSpec,
    Range,
    collect_param_specs,
    _make_init,
)
from grdl.image_processing.versioning import processor_version


# ---------------------------------------------------------------------------
# Constraint marker construction
# ---------------------------------------------------------------------------

class TestRange:
    """Test Range constraint marker."""

    def test_basic(self):
        r = Range(min=0.0, max=1.0)
        assert r.min == 0.0
        assert r.max == 1.0

    def test_defaults_none(self):
        r = Range()
        assert r.min is None
        assert r.max is None

    def test_min_only(self):
        r = Range(min=0)
        assert r.min == 0
        assert r.max is None

    def test_max_only(self):
        r = Range(max=100)
        assert r.min is None
        assert r.max == 100

    def test_is_param_meta(self):
        assert isinstance(Range(), ParamMeta)

    def test_repr(self):
        assert 'min=0' in repr(Range(min=0, max=1))


class TestOptions:
    """Test Options constraint marker."""

    def test_basic(self):
        o = Options('a', 'b', 'c')
        assert o.choices == ('a', 'b', 'c')

    def test_single_choice(self):
        o = Options('only')
        assert o.choices == ('only',)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            Options()

    def test_is_param_meta(self):
        assert isinstance(Options('x'), ParamMeta)


class TestDesc:
    """Test Desc constraint marker."""

    def test_basic(self):
        d = Desc('A description')
        assert d.text == 'A description'

    def test_is_param_meta(self):
        assert isinstance(Desc('x'), ParamMeta)


# ---------------------------------------------------------------------------
# ParamSpec construction and validation
# ---------------------------------------------------------------------------

class TestParamSpec:
    """Test ParamSpec introspection data class."""

    def test_basic_construction(self):
        spec = ParamSpec(
            name='threshold', param_type=float, default=0.5,
            has_default=True, description='A threshold',
            min_value=0.0, max_value=1.0, choices=None,
        )
        assert spec.name == 'threshold'
        assert spec.param_type is float
        assert spec.default == 0.5
        assert spec.required is False
        assert spec.min_value == 0.0
        assert spec.max_value == 1.0

    def test_required_when_no_default(self):
        spec = ParamSpec(
            name='mode', param_type=str, default=None,
            has_default=False, description='', min_value=None,
            max_value=None, choices=None,
        )
        assert spec.required is True

    def test_validate_type_correct(self):
        spec = ParamSpec('x', float, 0.5, True, '', None, None, None)
        spec.validate(0.7)  # no error

    def test_validate_int_accepted_as_float(self):
        spec = ParamSpec('x', float, 0.5, True, '', None, None, None)
        spec.validate(1)  # no error

    def test_validate_type_wrong_raises(self):
        spec = ParamSpec('x', float, 0.5, True, '', None, None, None)
        with pytest.raises(TypeError, match="x"):
            spec.validate('bad')

    def test_validate_min_boundary(self):
        spec = ParamSpec('x', float, 0.5, True, '', 0.0, None, None)
        spec.validate(0.0)  # exact boundary OK
        with pytest.raises(ValueError, match="below minimum"):
            spec.validate(-0.1)

    def test_validate_max_boundary(self):
        spec = ParamSpec('x', float, 0.5, True, '', None, 1.0, None)
        spec.validate(1.0)  # exact boundary OK
        with pytest.raises(ValueError, match="above maximum"):
            spec.validate(1.1)

    def test_validate_choices_valid(self):
        spec = ParamSpec('m', str, 'a', True, '', None, None, ('a', 'b'))
        spec.validate('b')  # no error

    def test_validate_choices_invalid(self):
        spec = ParamSpec('m', str, 'a', True, '', None, None, ('a', 'b'))
        with pytest.raises(ValueError, match="not in allowed choices"):
            spec.validate('c')

    def test_repr_required(self):
        spec = ParamSpec('x', float, None, False, '', None, None, None)
        r = repr(spec)
        assert 'required=True' in r
        assert 'default=' not in r

    def test_repr_optional(self):
        spec = ParamSpec('x', float, 0.5, True, '', 0.0, 1.0, None)
        r = repr(spec)
        assert 'required=False' in r
        assert 'default=0.5' in r
        assert 'min_value=0.0' in r
        assert 'max_value=1.0' in r


# ---------------------------------------------------------------------------
# collect_param_specs
# ---------------------------------------------------------------------------

class TestCollectParamSpecs:
    """Test annotation collection from Annotated class-body fields."""

    def test_empty_class(self):
        class C:
            pass
        assert collect_param_specs(C) == ()

    def test_plain_annotations_ignored(self):
        class C:
            x: float = 1.0
        assert collect_param_specs(C) == ()

    def test_annotated_without_param_meta_ignored(self):
        class C:
            x: Annotated[float, "just a string"] = 1.0
        assert collect_param_specs(C) == ()

    def test_single_annotated_field(self):
        class C:
            x: Annotated[float, Range(min=0), Desc('test')] = 1.0

        specs = collect_param_specs(C)
        assert len(specs) == 1
        assert specs[0].name == 'x'
        assert specs[0].param_type is float
        assert specs[0].default == 1.0
        assert specs[0].required is False
        assert specs[0].min_value == 0
        assert specs[0].description == 'test'

    def test_multiple_fields(self):
        class C:
            a: Annotated[float, Range(min=0)] = 1.0
            b: Annotated[str, Options('x', 'y')] = 'x'

        specs = collect_param_specs(C)
        assert len(specs) == 2
        assert specs[0].name == 'a'
        assert specs[1].name == 'b'
        assert specs[1].choices == ('x', 'y')

    def test_required_field_no_default(self):
        class C:
            x: Annotated[float, Desc('required')]

        specs = collect_param_specs(C)
        assert len(specs) == 1
        assert specs[0].required is True

    def test_range_and_options_mutually_exclusive(self):
        with pytest.raises(TypeError, match="mutually exclusive"):
            class C:
                x: Annotated[float, Range(min=0), Options(1, 2)] = 1.0
            collect_param_specs(C)

    def test_desc_only(self):
        """Desc alone is enough to mark as tunable."""
        class C:
            x: Annotated[float, Desc('hello')] = 5.0

        specs = collect_param_specs(C)
        assert len(specs) == 1
        assert specs[0].description == 'hello'
        assert specs[0].min_value is None
        assert specs[0].choices is None

    def test_inheritance_merges_specs(self):
        class Parent:
            a: Annotated[float, Range(min=0)] = 1.0

        class Child(Parent):
            b: Annotated[int, Range(min=1)] = 5

        specs = collect_param_specs(Child)
        assert len(specs) == 2
        names = [s.name for s in specs]
        assert 'a' in names
        assert 'b' in names

    def test_child_overrides_parent_param(self):
        class Parent:
            x: Annotated[float, Range(max=10)] = 1.0

        class Child(Parent):
            x: Annotated[float, Range(max=100)] = 5.0

        specs = collect_param_specs(Child)
        assert len(specs) == 1
        assert specs[0].name == 'x'
        assert specs[0].max_value == 100
        assert specs[0].default == 5.0


# ---------------------------------------------------------------------------
# Auto-generated __init__
# ---------------------------------------------------------------------------

class TestMakeInit:
    """Test _make_init auto-generated constructor."""

    def test_sets_attributes(self):
        @processor_version('1.0.0')
        class P(ImageTransform):
            threshold: Annotated[float, Range(min=0, max=1)] = 0.5

            def apply(self, source, **kwargs):
                return source

        p = P()
        assert p.threshold == 0.5

    def test_custom_value(self):
        @processor_version('1.0.0')
        class P(ImageTransform):
            threshold: Annotated[float, Range(min=0, max=1)] = 0.5

            def apply(self, source, **kwargs):
                return source

        p = P(threshold=0.8)
        assert p.threshold == 0.8

    def test_required_missing_raises(self):
        @processor_version('1.0.0')
        class P(ImageTransform):
            threshold: Annotated[float, Desc('required')]

            def apply(self, source, **kwargs):
                return source

        with pytest.raises(TypeError, match="missing required"):
            P()

    def test_validation_in_init(self):
        @processor_version('1.0.0')
        class P(ImageTransform):
            threshold: Annotated[float, Range(min=0, max=1)] = 0.5

            def apply(self, source, **kwargs):
                return source

        with pytest.raises(ValueError, match="above maximum"):
            P(threshold=2.0)

    def test_unexpected_kwargs_raises(self):
        @processor_version('1.0.0')
        class P(ImageTransform):
            threshold: Annotated[float, Range(min=0, max=1)] = 0.5

            def apply(self, source, **kwargs):
                return source

        with pytest.raises(TypeError, match="unexpected"):
            P(nonexistent=42)

    def test_choices_validation_in_init(self):
        @processor_version('1.0.0')
        class P(ImageTransform):
            method: Annotated[str, Options('a', 'b')] = 'a'

            def apply(self, source, **kwargs):
                return source

        P(method='b')  # OK
        with pytest.raises(ValueError, match="not in allowed choices"):
            P(method='c')

    def test_signature_introspectable(self):
        """Generated __init__ has a proper inspect.Signature."""
        import inspect

        @processor_version('1.0.0')
        class P(ImageTransform):
            x: Annotated[float, Range(min=0)] = 1.0
            y: Annotated[int, Desc('count')] = 5

            def apply(self, source, **kwargs):
                return source

        sig = inspect.signature(P)
        params = list(sig.parameters)
        assert 'x' in params
        assert 'y' in params


# ---------------------------------------------------------------------------
# __post_init__ hook
# ---------------------------------------------------------------------------

class TestPostInit:
    """Test __post_init__ is called after attribute assignment."""

    def test_post_init_called(self):
        @processor_version('1.0.0')
        class P(ImageTransform):
            method: Annotated[str, Options('a', 'b', 'A', 'B')] = 'A'

            def __post_init__(self):
                self.method = self.method.lower()

            def apply(self, source, **kwargs):
                return source

        p = P(method='A')
        assert p.method == 'a'

    def test_post_init_can_derive_state(self):
        @processor_version('1.0.0')
        class P(ImageTransform):
            radius: Annotated[float, Range(min=0), Desc('radius')] = 5.5

            def __post_init__(self):
                self._int_radius = max(1, int(round(self.radius)))

            def apply(self, source, **kwargs):
                return source

        p = P(radius=3.2)
        assert p._int_radius == 3


# ---------------------------------------------------------------------------
# Custom __init__ preserved
# ---------------------------------------------------------------------------

class TestCustomInit:
    """Class with own __init__ keeps it; __param_specs__ still built."""

    def test_custom_init_not_overwritten(self):
        @processor_version('1.0.0')
        class P(ImageTransform):
            sigma: Annotated[float, Range(min=0.1), Desc('sigma')] = 2.0

            def __init__(self, sigma=2.0, extra='custom'):
                self.sigma = sigma
                self.extra = extra

            def apply(self, source, **kwargs):
                return source

        p = P(sigma=3.0, extra='hello')
        assert p.sigma == 3.0
        assert p.extra == 'hello'

    def test_param_specs_still_built(self):
        @processor_version('1.0.0')
        class P(ImageTransform):
            sigma: Annotated[float, Range(min=0.1), Desc('sigma')] = 2.0

            def __init__(self, sigma=2.0):
                self.sigma = sigma

            def apply(self, source, **kwargs):
                return source

        assert len(P.__param_specs__) == 1
        assert P.__param_specs__[0].name == 'sigma'


# ---------------------------------------------------------------------------
# __param_specs__ class attribute
# ---------------------------------------------------------------------------

class TestParamSpecsAttribute:
    """Test __param_specs__ is set correctly on classes."""

    def test_no_annotated_fields(self):
        @processor_version('1.0.0')
        class P(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        assert P.__param_specs__ == ()

    def test_base_class_empty(self):
        assert ImageProcessor.__param_specs__ == ()

    def test_specs_on_class_not_instance(self):
        """__param_specs__ is readable from the class directly."""
        @processor_version('1.0.0')
        class P(ImageTransform):
            x: Annotated[float, Desc('test')] = 1.0

            def apply(self, source, **kwargs):
                return source

        assert len(P.__param_specs__) == 1
        assert P.__param_specs__[0].name == 'x'


# ---------------------------------------------------------------------------
# _resolve_params
# ---------------------------------------------------------------------------

class TestResolveParams:
    """Test _resolve_params runtime resolution."""

    def _make_processor(self):
        @processor_version('1.0.0')
        class P(ImageTransform):
            threshold: Annotated[float, Range(min=0, max=1), Desc('t')] = 0.5
            method: Annotated[str, Options('hard', 'soft'), Desc('m')] = 'hard'

            def apply(self, source, **kwargs):
                return source

        return P

    def test_defaults_resolved(self):
        P = self._make_processor()
        p = P()
        params = p._resolve_params({})
        assert params == {'threshold': 0.5, 'method': 'hard'}

    def test_kwargs_override(self):
        P = self._make_processor()
        p = P()
        params = p._resolve_params({'threshold': 0.8})
        assert params['threshold'] == 0.8
        assert params['method'] == 'hard'

    def test_construction_override_persists(self):
        P = self._make_processor()
        p = P(threshold=0.3)
        params = p._resolve_params({})
        assert params['threshold'] == 0.3

    def test_kwargs_override_construction(self):
        """Runtime kwargs override construction-time values."""
        P = self._make_processor()
        p = P(threshold=0.3)
        params = p._resolve_params({'threshold': 0.9})
        assert params['threshold'] == 0.9

    def test_validation_on_resolve(self):
        P = self._make_processor()
        p = P()
        with pytest.raises(ValueError, match="above maximum"):
            p._resolve_params({'threshold': 2.0})

    def test_type_error_on_resolve(self):
        P = self._make_processor()
        p = P()
        with pytest.raises(TypeError, match="threshold"):
            p._resolve_params({'threshold': 'bad'})

    def test_non_param_kwargs_ignored(self):
        """Non-param kwargs (e.g. progress_callback) pass through."""
        P = self._make_processor()
        p = P()
        params = p._resolve_params({
            'threshold': 0.7,
            'progress_callback': lambda f: None,
        })
        assert params == {'threshold': 0.7, 'method': 'hard'}


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------

class TestInheritance:
    """Test parameter inheritance across processor hierarchy."""

    def test_child_inherits_parent_params(self):
        @processor_version('1.0.0')
        class Parent(ImageTransform):
            sigma: Annotated[float, Range(min=0), Desc('sigma')] = 2.0

            def apply(self, source, **kwargs):
                return source

        @processor_version('1.0.0')
        class Child(Parent):
            extra: Annotated[int, Range(min=1), Desc('extra')] = 5

        assert len(Child.__param_specs__) == 2
        names = [s.name for s in Child.__param_specs__]
        assert 'sigma' in names
        assert 'extra' in names

        c = Child(sigma=3.0, extra=10)
        assert c.sigma == 3.0
        assert c.extra == 10

    def test_child_override_narrows_constraint(self):
        @processor_version('1.0.0')
        class Parent(ImageTransform):
            x: Annotated[float, Range(max=100)] = 1.0

            def apply(self, source, **kwargs):
                return source

        @processor_version('1.0.0')
        class Child(Parent):
            x: Annotated[float, Range(max=50)] = 1.0

        assert len(Child.__param_specs__) == 1
        assert Child.__param_specs__[0].max_value == 50


# ---------------------------------------------------------------------------
# Integration: concrete processor end-to-end
# ---------------------------------------------------------------------------

@processor_version('1.0.0')
class _ThresholdTransform(ImageTransform):
    """Test transform with Annotated tunable parameters."""

    threshold: Annotated[float, Range(min=0.0, max=1.0),
                          Desc('Pixel intensity threshold')] = 0.5
    method: Annotated[str, Options('hard', 'soft'),
                       Desc('Thresholding method')] = 'hard'

    def apply(self, source, **kwargs):
        params = self._resolve_params(kwargs)
        threshold = params['threshold']
        method = params['method']
        if method == 'hard':
            return (source > threshold).astype(source.dtype)
        else:
            return np.clip(source - threshold, 0, None)


class TestAnnotatedIntegration:
    """End-to-end tests with a concrete Annotated processor."""

    def test_apply_with_defaults(self):
        proc = _ThresholdTransform()
        img = np.array([[0.3, 0.7], [0.5, 0.9]])
        result = proc.apply(img)
        expected = (img > 0.5).astype(img.dtype)
        np.testing.assert_array_equal(result, expected)

    def test_apply_with_runtime_override(self):
        proc = _ThresholdTransform()
        img = np.array([[0.3, 0.7], [0.5, 0.9]])
        result = proc.apply(img, threshold=0.3, method='soft')
        expected = np.clip(img - 0.3, 0, None)
        np.testing.assert_array_almost_equal(result, expected)

    def test_construction_override(self):
        proc = _ThresholdTransform(threshold=0.1)
        assert proc.threshold == 0.1
        img = np.array([[0.05, 0.2], [0.1, 0.3]])
        result = proc.apply(img)
        expected = (img > 0.1).astype(img.dtype)
        np.testing.assert_array_equal(result, expected)

    def test_runtime_overrides_construction(self):
        proc = _ThresholdTransform(threshold=0.1)
        img = np.array([[0.3, 0.7], [0.5, 0.9]])
        result = proc.apply(img, threshold=0.6)
        expected = (img > 0.6).astype(img.dtype)
        np.testing.assert_array_equal(result, expected)

    def test_invalid_type_raises(self):
        proc = _ThresholdTransform()
        img = np.zeros((2, 2))
        with pytest.raises(TypeError, match="threshold"):
            proc.apply(img, threshold='high')

    def test_out_of_range_raises(self):
        proc = _ThresholdTransform()
        img = np.zeros((2, 2))
        with pytest.raises(ValueError, match="above maximum"):
            proc.apply(img, threshold=2.0)

    def test_invalid_choice_raises(self):
        proc = _ThresholdTransform()
        img = np.zeros((2, 2))
        with pytest.raises(ValueError, match="not in allowed choices"):
            proc.apply(img, method='fuzzy')

    def test_param_specs_introspectable(self):
        specs = _ThresholdTransform.__param_specs__
        assert len(specs) == 2
        assert specs[0].name == 'threshold'
        assert specs[0].param_type is float
        assert specs[0].min_value == 0.0
        assert specs[0].max_value == 1.0
        assert specs[1].name == 'method'
        assert specs[1].choices == ('hard', 'soft')


# ---------------------------------------------------------------------------
# Import checks
# ---------------------------------------------------------------------------

class TestAnnotatedImports:
    """Verify public API imports work."""

    def test_import_constraint_types(self):
        from grdl.image_processing.params import Range, Options, Desc, ParamSpec
        assert Range is not None
        assert Options is not None
        assert Desc is not None
        assert ParamSpec is not None

    def test_param_specs_on_base(self):
        """ImageProcessor has __param_specs__ as empty tuple."""
        assert hasattr(ImageProcessor, '__param_specs__')
        assert ImageProcessor.__param_specs__ == ()
