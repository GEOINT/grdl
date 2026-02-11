# -*- coding: utf-8 -*-
"""
Processor Tags Phase Tests - Unit tests for @processor_tags phases parameter.

Tests that the phases parameter stores ExecutionPhase tuples correctly,
defaults to empty tuple, and rejects invalid types at decorator time.

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

import pytest

from grdl.vocabulary import ExecutionPhase, ProcessorCategory
from grdl.image_processing.versioning import processor_tags, processor_version


class TestProcessorTagsPhases:
    """Tests for the phases parameter in @processor_tags."""

    def test_phases_stored_correctly(self):
        """phases parameter stored as tuple in __processor_tags__."""
        from grdl.image_processing.base import ImageTransform

        @processor_version('1.0.0')
        @processor_tags(phases=[
            ExecutionPhase.TILE_PROCESSING,
            ExecutionPhase.GLOBAL_PROCESSING,
        ])
        class MyProcessor(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        tags = MyProcessor.__processor_tags__
        assert 'phases' in tags
        assert tags['phases'] == (
            ExecutionPhase.TILE_PROCESSING,
            ExecutionPhase.GLOBAL_PROCESSING,
        )

    def test_single_phase(self):
        """Single phase stored as 1-element tuple."""
        from grdl.image_processing.base import ImageTransform

        @processor_version('1.0.0')
        @processor_tags(phases=[ExecutionPhase.IO])
        class IOProcessor(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        assert IOProcessor.__processor_tags__['phases'] == (
            ExecutionPhase.IO,
        )

    def test_phases_default_empty_tuple(self):
        """Missing phases defaults to empty tuple."""
        from grdl.image_processing.base import ImageTransform

        @processor_version('1.0.0')
        @processor_tags(category=ProcessorCategory.ENHANCE)
        class NoPhase(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        assert NoPhase.__processor_tags__['phases'] == ()

    def test_phases_none_defaults_empty_tuple(self):
        """Explicit phases=None defaults to empty tuple."""
        from grdl.image_processing.base import ImageTransform

        @processor_version('1.0.0')
        @processor_tags(phases=None)
        class NonePhase(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        assert NonePhase.__processor_tags__['phases'] == ()

    def test_invalid_phase_raises_type_error(self):
        """Non-ExecutionPhase value raises TypeError at decorator time."""
        with pytest.raises(TypeError, match="ExecutionPhase"):
            @processor_tags(phases=['not_a_phase'])
            class Bad:
                pass

    def test_invalid_phase_mixed_raises(self):
        """Mix of valid and invalid phases raises TypeError."""
        with pytest.raises(TypeError, match="ExecutionPhase"):
            @processor_tags(phases=[
                ExecutionPhase.IO,
                'invalid',
            ])
            class Bad:
                pass

    def test_phases_with_other_tags(self):
        """phases works alongside modalities and category."""
        from grdl.image_processing.base import ImageTransform
        from grdl.vocabulary import ImageModality

        @processor_version('1.0.0')
        @processor_tags(
            modalities=[ImageModality.SAR],
            category=ProcessorCategory.ENHANCE,
            phases=[ExecutionPhase.TILE_PROCESSING],
        )
        class FullTags(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        tags = FullTags.__processor_tags__
        assert tags['modalities'] == (ImageModality.SAR,)
        assert tags['category'] == ProcessorCategory.ENHANCE
        assert tags['phases'] == (ExecutionPhase.TILE_PROCESSING,)

    def test_all_phases_accepted(self):
        """All 8 ExecutionPhase values are accepted."""
        from grdl.image_processing.base import ImageTransform

        all_phases = list(ExecutionPhase)

        @processor_version('1.0.0')
        @processor_tags(phases=all_phases)
        class AllPhases(ImageTransform):
            def apply(self, source, **kwargs):
                return source

        assert len(AllPhases.__processor_tags__['phases']) == 8
        assert set(AllPhases.__processor_tags__['phases']) == set(
            ExecutionPhase
        )
