# -*- coding: utf-8 -*-
"""
Global Processor Decorator Tests - Unit tests for @globalprocessor.

Tests method decoration, __is_global_callback__ flag, collection into
__global_callbacks__ tuple, __has_global_pass__ flag, inheritance,
and non-decorated method exclusion.

Dependencies
------------
pytest

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
2026-02-11

Modified
--------
2026-02-11
"""

import warnings

import numpy as np
import pytest

from grdl.image_processing.versioning import globalprocessor, processor_version


class TestGlobalProcessorDecoratorFlag:
    """Tests that @globalprocessor sets the method flag."""

    def test_decorated_method_has_flag(self):
        """Decorated method has __is_global_callback__ = True."""
        @globalprocessor
        def my_method(self, source):
            pass

        assert my_method.__is_global_callback__ is True

    def test_undecorated_method_lacks_flag(self):
        """Undecorated method does not have __is_global_callback__."""
        def my_method(self, source):
            pass

        assert not hasattr(my_method, '__is_global_callback__')

    def test_decorator_returns_same_function(self):
        """Decorator returns the original function object."""
        def my_method(self, source):
            pass

        result = globalprocessor(my_method)
        assert result is my_method


class TestGlobalCallbacksCollection:
    """Tests for __global_callbacks__ and __has_global_pass__."""

    def test_class_with_no_global_callbacks(self):
        """Class without @globalprocessor has empty callbacks."""
        from grdl.image_processing.base import ImageTransform

        @processor_version('1.0.0')
        class Plain(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        assert Plain.__global_callbacks__ == ()
        assert Plain.__has_global_pass__ is False

    def test_class_with_one_global_callback(self):
        """Class with one @globalprocessor method collected correctly."""
        from grdl.image_processing.base import ImageTransform

        @processor_version('1.0.0')
        class OneCallback(ImageTransform):
            @globalprocessor
            def compute_stats(self, source):
                return source.mean()

            def apply(self, source, **kwargs):
                return source

        assert 'compute_stats' in OneCallback.__global_callbacks__
        assert len(OneCallback.__global_callbacks__) == 1
        assert OneCallback.__has_global_pass__ is True

    def test_class_with_multiple_global_callbacks(self):
        """Class with multiple @globalprocessor methods collected."""
        from grdl.image_processing.base import ImageTransform

        @processor_version('1.0.0')
        class MultiCallback(ImageTransform):
            @globalprocessor
            def compute_mean(self, source):
                return source.mean()

            @globalprocessor
            def compute_std(self, source):
                return source.std()

            def apply(self, source, **kwargs):
                return source

        assert len(MultiCallback.__global_callbacks__) == 2
        assert 'compute_mean' in MultiCallback.__global_callbacks__
        assert 'compute_std' in MultiCallback.__global_callbacks__
        assert MultiCallback.__has_global_pass__ is True

    def test_non_decorated_methods_excluded(self):
        """Non-@globalprocessor methods are NOT in __global_callbacks__."""
        from grdl.image_processing.base import ImageTransform

        @processor_version('1.0.0')
        class Mixed(ImageTransform):
            @globalprocessor
            def compute_stats(self, source):
                return source.mean()

            def helper(self):
                pass

            def apply(self, source, **kwargs):
                return source

        assert 'compute_stats' in Mixed.__global_callbacks__
        assert 'helper' not in Mixed.__global_callbacks__
        assert 'apply' not in Mixed.__global_callbacks__


class TestGlobalCallbacksInheritance:
    """Tests that subclasses inherit parent's global callbacks."""

    def test_subclass_inherits_parent_callbacks(self):
        """Subclass gets parent's global callbacks."""
        from grdl.image_processing.base import ImageTransform

        @processor_version('1.0.0')
        class Parent(ImageTransform):
            @globalprocessor
            def parent_callback(self, source):
                return source.mean()

            def apply(self, source, **kwargs):
                return source

        @processor_version('1.0.0')
        class Child(Parent):
            pass

        assert 'parent_callback' in Child.__global_callbacks__
        assert Child.__has_global_pass__ is True

    def test_subclass_adds_own_callbacks(self):
        """Subclass adds its own callbacks to parent's."""
        from grdl.image_processing.base import ImageTransform

        @processor_version('1.0.0')
        class Parent(ImageTransform):
            @globalprocessor
            def parent_callback(self, source):
                return source.mean()

            def apply(self, source, **kwargs):
                return source

        @processor_version('1.0.0')
        class Child(Parent):
            @globalprocessor
            def child_callback(self, source):
                return source.std()

        assert 'parent_callback' in Child.__global_callbacks__
        assert 'child_callback' in Child.__global_callbacks__
        assert len(Child.__global_callbacks__) == 2

    def test_parent_not_affected_by_child(self):
        """Parent's callbacks are unchanged after child definition."""
        from grdl.image_processing.base import ImageTransform

        @processor_version('1.0.0')
        class Parent(ImageTransform):
            @globalprocessor
            def parent_callback(self, source):
                return source.mean()

            def apply(self, source, **kwargs):
                return source

        @processor_version('1.0.0')
        class Child(Parent):
            @globalprocessor
            def child_callback(self, source):
                return source.std()

        assert Parent.__global_callbacks__ == ('parent_callback',)
        assert 'child_callback' not in Parent.__global_callbacks__
