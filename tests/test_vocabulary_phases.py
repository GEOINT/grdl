# -*- coding: utf-8 -*-
"""
Vocabulary Phase and Format Enum Tests - Unit tests for ExecutionPhase
and OutputFormat enums.

Tests that all expected enum values are present and importable from
both grdl.vocabulary and grdl.

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


class TestExecutionPhaseEnum:
    """Tests for ExecutionPhase enum."""

    def test_all_eight_values_present(self):
        """ExecutionPhase has exactly 8 members."""
        from grdl.vocabulary import ExecutionPhase

        expected = {
            'IO', 'GLOBAL_PROCESSING', 'DATA_PREP', 'TILING',
            'TILE_PROCESSING', 'EXTRACTION', 'VECTOR_PROCESSING',
            'FINALIZATION',
        }
        actual = {m.name for m in ExecutionPhase}
        assert actual == expected

    def test_values_are_lowercase_strings(self):
        """Each ExecutionPhase value is a lowercase string."""
        from grdl.vocabulary import ExecutionPhase

        for member in ExecutionPhase:
            assert isinstance(member.value, str)
            assert member.value == member.value.lower()

    def test_importable_from_grdl(self):
        """ExecutionPhase is importable from grdl top-level."""
        from grdl import ExecutionPhase

        assert ExecutionPhase.IO.value == 'io'

    def test_importable_from_vocabulary(self):
        """ExecutionPhase is importable from grdl.vocabulary."""
        from grdl.vocabulary import ExecutionPhase

        assert ExecutionPhase.TILE_PROCESSING.value == 'tile_processing'

    def test_individual_members(self):
        """Spot-check individual member values."""
        from grdl.vocabulary import ExecutionPhase

        assert ExecutionPhase.IO.value == 'io'
        assert ExecutionPhase.GLOBAL_PROCESSING.value == 'global_processing'
        assert ExecutionPhase.DATA_PREP.value == 'data_prep'
        assert ExecutionPhase.TILING.value == 'tiling'
        assert ExecutionPhase.TILE_PROCESSING.value == 'tile_processing'
        assert ExecutionPhase.EXTRACTION.value == 'extraction'
        assert ExecutionPhase.VECTOR_PROCESSING.value == 'vector_processing'
        assert ExecutionPhase.FINALIZATION.value == 'finalization'


class TestOutputFormatEnum:
    """Tests for OutputFormat enum."""

    def test_all_five_values_present(self):
        """OutputFormat has exactly 5 members."""
        from grdl.vocabulary import OutputFormat

        expected = {'GEOTIFF', 'NUMPY', 'PNG', 'HDF5', 'NITF'}
        actual = {m.name for m in OutputFormat}
        assert actual == expected

    def test_values_are_lowercase_strings(self):
        """Each OutputFormat value is a lowercase string."""
        from grdl.vocabulary import OutputFormat

        for member in OutputFormat:
            assert isinstance(member.value, str)
            assert member.value == member.value.lower()

    def test_importable_from_grdl(self):
        """OutputFormat is importable from grdl top-level."""
        from grdl import OutputFormat

        assert OutputFormat.GEOTIFF.value == 'geotiff'

    def test_importable_from_vocabulary(self):
        """OutputFormat is importable from grdl.vocabulary."""
        from grdl.vocabulary import OutputFormat

        assert OutputFormat.PNG.value == 'png'

    def test_individual_members(self):
        """Spot-check individual member values."""
        from grdl.vocabulary import OutputFormat

        assert OutputFormat.GEOTIFF.value == 'geotiff'
        assert OutputFormat.NUMPY.value == 'numpy'
        assert OutputFormat.PNG.value == 'png'
        assert OutputFormat.HDF5.value == 'hdf5'
        assert OutputFormat.NITF.value == 'nitf'
