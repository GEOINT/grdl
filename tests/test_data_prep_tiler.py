# -*- coding: utf-8 -*-
"""
Tests for the Tiler class.

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

from grdl.data_prep import Tiler, ChipRegion


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestTilerInit:
    """Test Tiler constructor validation."""

    def test_valid_construction(self):
        tiler = Tiler(nrows=100, ncols=200, tile_size=32)
        assert tiler.nrows == 100
        assert tiler.ncols == 200
        assert tiler.tile_size == (32, 32)
        assert tiler.stride == (32, 32)

    def test_tuple_tile_size(self):
        tiler = Tiler(nrows=100, ncols=100, tile_size=(32, 64))
        assert tiler.tile_size == (32, 64)

    def test_explicit_stride(self):
        tiler = Tiler(nrows=100, ncols=100, tile_size=32, stride=16)
        assert tiler.stride == (16, 16)

    def test_stride_defaults_to_tile_size(self):
        tiler = Tiler(nrows=100, ncols=100, tile_size=32)
        assert tiler.stride == tiler.tile_size

    def test_stride_exceeds_tile_size_raises(self):
        with pytest.raises(ValueError, match="must not exceed"):
            Tiler(nrows=100, ncols=100, tile_size=32, stride=64)

    def test_non_int_nrows_raises(self):
        with pytest.raises(TypeError, match="must be int"):
            Tiler(nrows=100.0, ncols=100, tile_size=32)

    def test_zero_tile_size_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            Tiler(nrows=100, ncols=100, tile_size=0)

    def test_negative_ncols_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            Tiler(nrows=100, ncols=-10, tile_size=32)

    def test_repr(self):
        tiler = Tiler(nrows=100, ncols=200, tile_size=32, stride=16)
        expected = "Tiler(nrows=100, ncols=200, tile_size=(32, 32), stride=(16, 16))"
        assert repr(tiler) == expected

    def test_shape_property(self):
        tiler = Tiler(nrows=100, ncols=200, tile_size=32)
        assert tiler.shape == (100, 200)


# ---------------------------------------------------------------------------
# tile_positions -- no overlap
# ---------------------------------------------------------------------------

class TestTilePositionsNoOverlap:
    """Test tile_positions with stride equal to tile_size."""

    def test_even_division(self):
        tiler = Tiler(nrows=100, ncols=100, tile_size=50)
        regions = tiler.tile_positions()
        assert len(regions) == 4
        assert regions[0] == ChipRegion(0, 0, 50, 50)
        assert regions[1] == ChipRegion(0, 50, 50, 100)
        assert regions[2] == ChipRegion(50, 0, 100, 50)
        assert regions[3] == ChipRegion(50, 50, 100, 100)

    def test_uneven_division_snaps_edge(self):
        tiler = Tiler(nrows=100, ncols=100, tile_size=64)
        regions = tiler.tile_positions()
        assert len(regions) == 4
        # Last tile snaps inward to maintain full 64x64 size
        last = regions[-1]
        assert last.row_end == 100
        assert last.col_end == 100
        assert last.row_end - last.row_start == 64
        assert last == ChipRegion(36, 36, 100, 100)

    def test_tile_larger_than_image(self):
        tiler = Tiler(nrows=50, ncols=50, tile_size=100)
        regions = tiler.tile_positions()
        assert len(regions) == 1
        assert regions[0] == ChipRegion(0, 0, 50, 50)

    def test_tile_equals_image(self):
        tiler = Tiler(nrows=100, ncols=100, tile_size=100)
        regions = tiler.tile_positions()
        assert len(regions) == 1
        assert regions[0] == ChipRegion(0, 0, 100, 100)

    def test_row_major_order(self):
        tiler = Tiler(nrows=90, ncols=90, tile_size=30)
        regions = tiler.tile_positions()
        assert len(regions) == 9
        # Verify row-major: row changes slower than col
        for i in range(len(regions) - 1):
            curr_row = regions[i].row_start
            next_row = regions[i + 1].row_start
            curr_col = regions[i].col_start
            next_col = regions[i + 1].col_start
            if curr_row == next_row:
                assert next_col > curr_col

    def test_returns_chip_region_type(self):
        tiler = Tiler(nrows=100, ncols=100, tile_size=50)
        regions = tiler.tile_positions()
        for r in regions:
            assert isinstance(r, ChipRegion)

    def test_all_tiles_uniform_size(self):
        """All tiles maintain full tile_size regardless of position."""
        tiler = Tiler(nrows=100, ncols=100, tile_size=64)
        regions = tiler.tile_positions()
        for r in regions:
            assert r.row_end - r.row_start == 64
            assert r.col_end - r.col_start == 64


# ---------------------------------------------------------------------------
# tile_positions -- with overlap
# ---------------------------------------------------------------------------

class TestTilePositionsOverlap:
    """Test tile_positions with stride less than tile_size."""

    def test_half_overlap(self):
        tiler = Tiler(nrows=100, ncols=100, tile_size=32, stride=16)
        regions = tiler.tile_positions()
        # First two tiles in the same row should overlap by 16 pixels
        assert regions[0].col_end - regions[1].col_start == 16

    def test_overlap_full_coverage(self):
        """Every pixel should be covered by at least one tile."""
        tiler = Tiler(nrows=100, ncols=100, tile_size=32, stride=16)
        regions = tiler.tile_positions()
        covered = np.zeros((100, 100), dtype=int)
        for r in regions:
            covered[r.row_start:r.row_end, r.col_start:r.col_end] += 1
        assert np.all(covered >= 1)

    def test_interior_pixels_covered_multiple_times(self):
        """Interior pixels should be covered more than once with overlap."""
        tiler = Tiler(nrows=100, ncols=100, tile_size=32, stride=16)
        regions = tiler.tile_positions()
        covered = np.zeros((100, 100), dtype=int)
        for r in regions:
            covered[r.row_start:r.row_end, r.col_start:r.col_end] += 1
        # Interior pixel (50, 50) should be covered by multiple tiles
        assert covered[50, 50] > 1

    def test_asymmetric_tile_and_stride(self):
        tiler = Tiler(nrows=100, ncols=100,
                      tile_size=(32, 64), stride=(16, 32))
        regions = tiler.tile_positions()
        # Verify tile dimensions respect tile_size
        first = regions[0]
        assert first.row_end - first.row_start == 32
        assert first.col_end - first.col_start == 64

    def test_stride_equals_one(self):
        """Stride of 1 produces many heavily overlapping tiles."""
        tiler = Tiler(nrows=10, ncols=10, tile_size=5, stride=1)
        regions = tiler.tile_positions()
        # 6 row positions * 6 col positions = 36 tiles
        assert len(regions) == 36


# ---------------------------------------------------------------------------
# Inherits ChipBase
# ---------------------------------------------------------------------------

class TestTilerInheritance:
    """Verify Tiler inherits from ChipBase properly."""

    def test_is_chip_base(self):
        from grdl.data_prep import ChipBase
        tiler = Tiler(nrows=100, ncols=100, tile_size=32)
        assert isinstance(tiler, ChipBase)

    def test_snap_region_available(self):
        tiler = Tiler(nrows=100, ncols=100, tile_size=32)
        region = tiler._snap_region(-5, -5, 50, 50)
        assert region == ChipRegion(0, 0, 50, 50)
