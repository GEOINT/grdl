# -*- coding: utf-8 -*-
"""
Tests for the ChipExtractor class.

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
2026-02-09

Modified
--------
2026-02-09
"""

import numpy as np
import pytest

from grdl.data_prep import ChipExtractor, ChipRegion


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestChipExtractorInit:
    """Test ChipExtractor constructor validation."""

    def test_valid_construction(self):
        ext = ChipExtractor(nrows=100, ncols=200)
        assert ext.nrows == 100
        assert ext.ncols == 200
        assert ext.shape == (100, 200)

    def test_non_int_raises_type_error(self):
        with pytest.raises(TypeError, match="must be int"):
            ChipExtractor(nrows=100.0, ncols=200)

    def test_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="must be positive"):
            ChipExtractor(nrows=0, ncols=100)

    def test_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="must be positive"):
            ChipExtractor(nrows=100, ncols=-5)

    def test_repr(self):
        ext = ChipExtractor(nrows=50, ncols=75)
        assert repr(ext) == "ChipExtractor(nrows=50, ncols=75)"


# ---------------------------------------------------------------------------
# chip_at_point -- scalar
# ---------------------------------------------------------------------------

class TestChipAtPointScalar:
    """Test chip_at_point with scalar inputs."""

    def test_center_of_image(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        region = ext.chip_at_point(50, 50, row_width=20, col_width=20)
        assert isinstance(region, ChipRegion)
        assert region == ChipRegion(40, 40, 60, 60)

    def test_asymmetric_chip(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        region = ext.chip_at_point(50, 50, row_width=10, col_width=30)
        assert region == ChipRegion(45, 35, 55, 65)

    def test_top_left_corner_snapped(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        region = ext.chip_at_point(5, 5, row_width=20, col_width=20)
        # Snaps to start at 0, maintains full 20-pixel width
        assert region == ChipRegion(0, 0, 20, 20)

    def test_bottom_right_corner_snapped(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        region = ext.chip_at_point(95, 95, row_width=20, col_width=20)
        # Snaps to end at 100, maintains full 20-pixel width
        assert region == ChipRegion(80, 80, 100, 100)

    def test_users_example_top_edge(self):
        """Row 25 in a 500-row image with chip size 100 -> 0-100."""
        ext = ChipExtractor(nrows=500, ncols=500)
        region = ext.chip_at_point(25, 250, row_width=100, col_width=100)
        assert region.row_start == 0
        assert region.row_end == 100

    def test_users_example_bottom_edge(self):
        """Row 475 in a 500-row image with chip size 100 -> 400-500."""
        ext = ChipExtractor(nrows=500, ncols=500)
        region = ext.chip_at_point(475, 250, row_width=100, col_width=100)
        assert region.row_start == 400
        assert region.row_end == 500

    def test_exact_corner_zero(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        region = ext.chip_at_point(0, 0, row_width=10, col_width=10)
        assert region.row_start == 0
        assert region.col_start == 0

    def test_float_input_rounded(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        region = ext.chip_at_point(50.7, 49.3, row_width=10, col_width=10)
        # 50.7 rounds to 51, 49.3 rounds to 49
        assert region == ChipRegion(46, 44, 56, 54)

    def test_usable_for_slicing(self):
        """ChipRegion can directly slice a numpy array."""
        image = np.arange(100 * 100).reshape(100, 100)
        ext = ChipExtractor(nrows=100, ncols=100)
        r = ext.chip_at_point(50, 50, row_width=10, col_width=10)
        chip = image[r.row_start:r.row_end, r.col_start:r.col_end]
        assert chip.shape == (10, 10)

    def test_tuple_unpacking(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        rs, cs, re, ce = ext.chip_at_point(50, 50, row_width=10, col_width=10)
        assert rs == 45
        assert cs == 45
        assert re == 55
        assert ce == 55


# ---------------------------------------------------------------------------
# chip_at_point -- array
# ---------------------------------------------------------------------------

class TestChipAtPointArray:
    """Test chip_at_point with array/list inputs."""

    def test_list_input(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        regions = ext.chip_at_point([50, 10], [50, 10],
                                    row_width=20, col_width=20)
        assert isinstance(regions, list)
        assert len(regions) == 2
        assert regions[0] == ChipRegion(40, 40, 60, 60)
        # Edge point snaps: maintains full 20x20 size
        assert regions[1] == ChipRegion(0, 0, 20, 20)

    def test_ndarray_input(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        rows = np.array([50, 10, 90])
        cols = np.array([50, 10, 90])
        regions = ext.chip_at_point(rows, cols, row_width=10, col_width=10)
        assert len(regions) == 3

    def test_mixed_interior_and_edge(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        regions = ext.chip_at_point([50, 2], [50, 2],
                                    row_width=20, col_width=20)
        # Interior point: full chip
        assert regions[0].row_end - regions[0].row_start == 20
        # Edge point: snapped, still full chip
        assert regions[1].row_end - regions[1].row_start == 20
        assert regions[1].row_start == 0


# ---------------------------------------------------------------------------
# chip_at_point -- validation
# ---------------------------------------------------------------------------

class TestChipAtPointValidation:
    """Test chip_at_point input validation."""

    def test_point_outside_image_raises(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        with pytest.raises(ValueError, match="row values must be in"):
            ext.chip_at_point(100, 50, row_width=10, col_width=10)

    def test_negative_point_raises(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        with pytest.raises(ValueError, match="row values must be in"):
            ext.chip_at_point(-1, 50, row_width=10, col_width=10)

    def test_col_outside_raises(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        with pytest.raises(ValueError, match="col values must be in"):
            ext.chip_at_point(50, 100, row_width=10, col_width=10)

    def test_non_int_width_raises(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        with pytest.raises(TypeError, match="row_width must be int"):
            ext.chip_at_point(50, 50, row_width=10.0, col_width=10)

    def test_zero_width_raises(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        with pytest.raises(ValueError, match="col_width must be positive"):
            ext.chip_at_point(50, 50, row_width=10, col_width=0)


# ---------------------------------------------------------------------------
# chip_positions
# ---------------------------------------------------------------------------

class TestChipPositions:
    """Test chip_positions for whole-image chunking."""

    def test_even_division(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        regions = ext.chip_positions(row_width=25, col_width=25)
        assert len(regions) == 16
        for r in regions:
            assert r.row_end - r.row_start == 25
            assert r.col_end - r.col_start == 25

    def test_uneven_division_snaps(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        regions = ext.chip_positions(row_width=30, col_width=30)
        # 4 row positions: 0, 30, 60, 90
        # 4 col positions: 0, 30, 60, 90
        assert len(regions) == 16
        # Last chip snaps inward to maintain full 30x30 size
        last = regions[-1]
        assert last.row_end - last.row_start == 30
        assert last.col_end - last.col_start == 30
        # Snapped: starts at 70 instead of 90
        assert last == ChipRegion(70, 70, 100, 100)

    def test_chip_larger_than_image(self):
        ext = ChipExtractor(nrows=50, ncols=50)
        regions = ext.chip_positions(row_width=100, col_width=100)
        assert len(regions) == 1
        assert regions[0] == ChipRegion(0, 0, 50, 50)

    def test_row_major_order(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        regions = ext.chip_positions(row_width=50, col_width=50)
        assert len(regions) == 4
        # Row 0: (0,0), (0,50)
        assert regions[0] == ChipRegion(0, 0, 50, 50)
        assert regions[1] == ChipRegion(0, 50, 50, 100)
        # Row 1: (50,0), (50,50)
        assert regions[2] == ChipRegion(50, 0, 100, 50)
        assert regions[3] == ChipRegion(50, 50, 100, 100)

    def test_single_pixel_chips(self):
        ext = ChipExtractor(nrows=3, ncols=3)
        regions = ext.chip_positions(row_width=1, col_width=1)
        assert len(regions) == 9

    def test_full_coverage(self):
        """Every pixel is covered. Edge chips may overlap due to snapping."""
        ext = ChipExtractor(nrows=100, ncols=100)
        regions = ext.chip_positions(row_width=30, col_width=30)
        covered = np.zeros((100, 100), dtype=int)
        for r in regions:
            covered[r.row_start:r.row_end, r.col_start:r.col_end] += 1
        assert np.all(covered >= 1)

    def test_full_coverage_even_no_overlap(self):
        """Even division produces no overlap."""
        ext = ChipExtractor(nrows=100, ncols=100)
        regions = ext.chip_positions(row_width=25, col_width=25)
        covered = np.zeros((100, 100), dtype=int)
        for r in regions:
            covered[r.row_start:r.row_end, r.col_start:r.col_end] += 1
        assert np.all(covered == 1)

    def test_all_chips_uniform_size(self):
        """All chips maintain full requested size regardless of position."""
        ext = ChipExtractor(nrows=100, ncols=100)
        regions = ext.chip_positions(row_width=30, col_width=30)
        for r in regions:
            assert r.row_end - r.row_start == 30
            assert r.col_end - r.col_start == 30

    def test_validation_zero_width(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        with pytest.raises(ValueError, match="must be positive"):
            ext.chip_positions(row_width=0, col_width=10)

    def test_validation_non_int(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        with pytest.raises(TypeError, match="must be int"):
            ext.chip_positions(row_width=10.5, col_width=10)


# ---------------------------------------------------------------------------
# ChipRegion
# ---------------------------------------------------------------------------

class TestChipRegion:
    """Test ChipRegion named tuple."""

    def test_fields(self):
        r = ChipRegion(10, 20, 30, 40)
        assert r.row_start == 10
        assert r.col_start == 20
        assert r.row_end == 30
        assert r.col_end == 40

    def test_equality(self):
        r1 = ChipRegion(10, 20, 30, 40)
        r2 = ChipRegion(10, 20, 30, 40)
        assert r1 == r2

    def test_immutable(self):
        r = ChipRegion(10, 20, 30, 40)
        with pytest.raises(AttributeError):
            r.row_start = 5
