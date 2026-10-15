# -*- coding: utf-8 -*-
"""
Tests for halo tiling and the Stitcher.

Author
------
Jason Fritz
43161141+stryder-vtx@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-10-07

Modified
--------
2026-10-07
"""

import numpy as np
import pytest
from scipy.ndimage import uniform_filter

from grdl.data_prep import (
    ChipRegion, Stitcher, Tiler, memmap_sink, total_halo,
)


class TestHaloTiles:

    def test_cores_partition_image(self):
        tiles = Tiler(50, 70, tile_size=16).halo_tiles(3)
        cover = np.zeros((50, 70), dtype=int)
        for t in tiles:
            c = t.core
            cover[c.row_start:c.row_end, c.col_start:c.col_end] += 1
        assert (cover == 1).all()

    def test_read_region_clipped_at_edges(self):
        tiles = Tiler(32, 32, tile_size=16).halo_tiles(4)
        assert tiles[0].read == ChipRegion(0, 0, 20, 20)
        assert tiles[-1].read == ChipRegion(12, 12, 32, 32)

    def test_crop_selects_core(self):
        img = np.arange(40 * 40).reshape(40, 40)
        for t in Tiler(40, 40, tile_size=16).halo_tiles((2, 5)):
            read = img[t.read.row_start:t.read.row_end,
                       t.read.col_start:t.read.col_end]
            core = img[t.core.row_start:t.core.row_end,
                       t.core.col_start:t.core.col_end]
            np.testing.assert_array_equal(read[t.crop], core)

    def test_zero_halo_equals_partition(self):
        tiler = Tiler(30, 30, tile_size=8)
        assert [t.core for t in tiler.halo_tiles(0)] == \
            tiler.partition_positions()

    def test_negative_halo_rejected(self):
        with pytest.raises(ValueError):
            Tiler(10, 10, tile_size=5).halo_tiles(-1)

    def test_overlapping_cores_follow_stride(self):
        tiler = Tiler(40, 40, tile_size=20, stride=10)
        tiles = tiler.halo_tiles(2, overlapping=True)
        assert [t.core for t in tiles] == tiler.tile_positions()
        assert tiles[0].read == ChipRegion(0, 0, 22, 22)

    def test_total_halo_sums(self):
        assert total_halo(3, (1, 2), 0) == (4, 5)


class TestStitcherMethods:

    @pytest.mark.parametrize(
        'method', ['weighted', 'mean', 'first', 'last', 'max', 'min'])
    def test_constant_overlap_recovered(self, method):
        full = np.full((40, 50), 7.0, dtype=np.float32)
        st = Stitcher(40, 50, method=method)
        for r in Tiler(40, 50, tile_size=20, stride=10).tile_positions():
            st.add(r, full[r.row_start:r.row_end, r.col_start:r.col_end])
        np.testing.assert_allclose(st.finalize(), full)

    def test_weighted_averages_overlap(self):
        st = Stitcher(1, 10, method='mean')
        st.add(ChipRegion(0, 0, 1, 10), np.zeros((1, 10)))
        st.add(ChipRegion(0, 5, 1, 10), np.full((1, 5), 4.0))
        out = st.finalize()
        np.testing.assert_allclose(out[0, :5], 0.0)
        np.testing.assert_allclose(out[0, 5:], 2.0)

    def test_weighted_favors_tile_interior(self):
        st = Stitcher(1, 10, method='weighted')
        st.add(ChipRegion(0, 0, 1, 6), np.zeros((1, 6)))
        st.add(ChipRegion(0, 4, 1, 10), np.ones((1, 6)))
        out = st.finalize()[0]
        assert out[4] < 0.5 < out[5]

    def test_first_and_last(self):
        a, b = np.zeros((1, 6)), np.ones((1, 6))
        first = Stitcher(1, 10, method='first')
        last = Stitcher(1, 10, method='last')
        for st in (first, last):
            st.add(ChipRegion(0, 0, 1, 6), a)
            st.add(ChipRegion(0, 4, 1, 10), b)
        assert first.finalize()[0, 4] == 0.0
        assert last.finalize()[0, 4] == 1.0

    def test_max_min(self):
        for method, expect in (('max', 3.0), ('min', 1.0)):
            st = Stitcher(1, 4, method=method)
            st.add(ChipRegion(0, 0, 1, 4), np.full((1, 4), 1.0))
            st.add(ChipRegion(0, 0, 1, 4), np.full((1, 4), 3.0))
            assert (st.finalize() == expect).all()

    def test_max_rejects_complex(self):
        st = Stitcher(1, 2, method='max')
        with pytest.raises(ValueError):
            st.add(ChipRegion(0, 0, 1, 2), np.ones((1, 2), np.complex64))

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            Stitcher(4, 4, method='nope')


class TestStitcherShapes:

    def test_complex_matrix_tiles(self):
        rng = np.random.default_rng(0)
        full = (rng.normal(size=(3, 3, 30, 30))
                + 1j * rng.normal(size=(3, 3, 30, 30))).astype(np.complex64)
        st = Stitcher(30, 30)
        for t in Tiler(30, 30, tile_size=12).halo_tiles(0):
            c = t.core
            st.add(c, full[..., c.row_start:c.row_end, c.col_start:c.col_end])
        out = st.finalize()
        assert out.dtype == np.complex64
        np.testing.assert_allclose(out, full, rtol=1e-5)

    def test_dict_tiles(self):
        st = Stitcher(8, 8)
        for t in Tiler(8, 8, tile_size=4).halo_tiles(0):
            c = t.core
            shape = (c.row_end - c.row_start, c.col_end - c.col_start)
            st.add(c, {'a': np.ones(shape), 'b': np.full((2,) + shape, 2.0)})
        out = st.finalize()
        assert out['a'].shape == (8, 8) and out['b'].shape == (2, 8, 8)

    def test_integer_input_promoted(self):
        st = Stitcher(2, 2)
        st.add(ChipRegion(0, 0, 2, 2), np.ones((2, 2), dtype=np.uint8))
        assert st.finalize().dtype == np.float32

    def test_gap_detected(self):
        st = Stitcher(4, 4)
        st.add(ChipRegion(0, 0, 2, 4), np.ones((2, 4)))
        with pytest.raises(ValueError, match='not covered'):
            st.finalize()

    def test_gap_allowed(self):
        st = Stitcher(4, 4)
        st.add(ChipRegion(0, 0, 2, 4), np.ones((2, 4)))
        assert st.finalize(allow_gaps=True).shape == (4, 4)

    def test_shape_mismatch(self):
        st = Stitcher(4, 4)
        with pytest.raises(ValueError):
            st.add(ChipRegion(0, 0, 2, 2), np.ones((3, 3)))

    def test_out_of_bounds(self):
        st = Stitcher(4, 4)
        with pytest.raises(ValueError):
            st.add(ChipRegion(0, 0, 5, 4), np.ones((5, 4)))

    def test_mixed_array_and_dict(self):
        st = Stitcher(4, 4)
        st.add(ChipRegion(0, 0, 2, 4), np.ones((2, 4)))
        with pytest.raises(ValueError):
            st.add(ChipRegion(2, 0, 4, 4), {'a': np.ones((2, 4))})

    def test_memmap_sink(self, tmp_path):
        st = Stitcher(6, 6, sink=memmap_sink(tmp_path))
        st.add(ChipRegion(0, 0, 6, 6), np.ones((2, 6, 6), np.float32))
        out = st.finalize()
        assert isinstance(out, np.memmap)
        assert (tmp_path / 'out.dat').exists()


class TestHaloEquivalence:
    """Tiled processing with a halo reproduces the full-image result."""

    @pytest.mark.parametrize('tile', [16, 23, 64])
    def test_boxcar_matches_full(self, tile):
        rng = np.random.default_rng(1)
        img = rng.normal(size=(2, 70, 90)).astype(np.float32)
        win = 7

        def proc(a):
            return uniform_filter(a, size=(1, win, win), mode='reflect')

        expected = proc(img)
        st = Stitcher(70, 90)
        for t in Tiler(70, 90, tile_size=tile).halo_tiles(win // 2):
            r = t.read
            block = img[:, r.row_start:r.row_end, r.col_start:r.col_end]
            st.add_halo_tile(t, proc(block))
        np.testing.assert_allclose(st.finalize(), expected, atol=1e-5)

    def test_insufficient_halo_differs(self):
        rng = np.random.default_rng(2)
        img = rng.normal(size=(40, 40)).astype(np.float32)
        expected = uniform_filter(img, size=7, mode='reflect')
        st = Stitcher(40, 40)
        for t in Tiler(40, 40, tile_size=20).halo_tiles(0):
            r = t.read
            block = img[r.row_start:r.row_end, r.col_start:r.col_end]
            st.add_halo_tile(t, uniform_filter(block, size=7, mode='reflect'))
        assert not np.allclose(st.finalize(), expected, atol=1e-5)
