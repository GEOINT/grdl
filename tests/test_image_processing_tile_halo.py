# -*- coding: utf-8 -*-
"""
Tests that declared tile halos reproduce full-image results.

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

from grdl.data_prep import Stitcher, Tiler
from grdl.image_processing.decomposition import (
    CoherencyMatrix, CovarianceMatrix, FreemanDurden3C, PauliDecomposition,
    Yamaguchi4C,
)
from grdl.image_processing.filters import (
    ComplexLeeFilter, IDANFilter, LeeFilter, MedianFilter, RefinedLeeFilter,
)
from grdl.image_processing.tiling import get_tile_halo

ROWS, COLS = 83, 101


@pytest.fixture(scope='module')
def quad():
    rng = np.random.default_rng(7)
    shape = (4, ROWS, COLS)
    cube = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    cube[1] = cube[2] = 0.15 * cube[1] + 0.05j * cube[0]
    return cube.astype(np.complex64)


def _tiled(proc, data, fn, tile=32):
    """Run fn(block) per halo tile over the last two axes and stitch."""
    halo = get_tile_halo(proc)
    assert halo is not None
    st = Stitcher(ROWS, COLS)
    for t in Tiler(ROWS, COLS, tile_size=tile).halo_tiles(halo):
        r = t.read
        block = data[..., r.row_start:r.row_end, r.col_start:r.col_end]
        st.add_halo_tile(t, fn(block))
    return st.finalize()


class TestDeclaredHalos:

    @pytest.mark.parametrize('proc,expected', [
        (CovarianceMatrix(window_size=7), (3, 3)),
        (CoherencyMatrix(window_size=1), (0, 0)),
        (PauliDecomposition(window_size=5), (2, 2)),
        (FreemanDurden3C(window_size=9), (4, 4)),
        (RefinedLeeFilter(kernel_size=7), (3, 3)),
        (LeeFilter(kernel_size=5, enl=4.0), (2, 2)),
    ])
    def test_halo_values(self, proc, expected):
        assert get_tile_halo(proc) == expected

    def test_auto_enl_is_global(self):
        assert get_tile_halo(LeeFilter(kernel_size=7)) is None
        assert get_tile_halo(ComplexLeeFilter(kernel_size=7)) is None

    def test_idan_is_global(self):
        assert get_tile_halo(IDANFilter(kernel_size=7, enl=2.0)) is None

    def test_undeclared_is_global(self):
        assert get_tile_halo(MedianFilter(kernel_size=3)) is None
        assert get_tile_halo(object()) is None

    def test_callable_halo(self):
        class P:
            def tile_halo(self):
                return (1, 2)
        assert get_tile_halo(P()) == (1, 2)


class TestHaloEquivalence:

    @pytest.mark.parametrize('cls', [CovarianceMatrix, CoherencyMatrix])
    @pytest.mark.parametrize('ws', [1, 7])
    def test_matrix_builders(self, quad, cls, ws):
        proc = cls(window_size=ws)
        full = proc.compute(quad)
        tiled = _tiled(proc, quad, proc.compute, tile=29)
        np.testing.assert_allclose(tiled, full, rtol=1e-4, atol=1e-4)

    def test_refined_lee_on_c3(self, quad):
        c3 = CovarianceMatrix(window_size=3).compute(quad)
        proc = RefinedLeeFilter(kernel_size=7)
        full = proc.filter_matrix(c3)
        tiled = _tiled(proc, c3, proc.filter_matrix, tile=32)
        np.testing.assert_allclose(tiled, full, rtol=1e-4, atol=1e-4)

    def test_lee_with_known_enl(self, quad):
        intensity = np.abs(quad[0]) ** 2
        proc = LeeFilter(kernel_size=7, enl=4.0)
        full = proc.apply(intensity)
        tiled = _tiled(proc, intensity, proc.apply, tile=32)
        np.testing.assert_allclose(tiled, full, rtol=1e-4, atol=1e-4)

    def test_freeman_durden_from_channels(self, quad):
        self._check_decompose(FreemanDurden3C(window_size=5), quad, 32)

    def test_yamaguchi_from_channels(self, quad):
        self._check_decompose(Yamaguchi4C(window_size=5), quad, 32)

    @staticmethod
    def _check_decompose(proc, quad, tile, rtol=1e-3):
        full = proc.decompose(*quad)

        halo = get_tile_halo(proc)
        st = Stitcher(ROWS, COLS)
        for t in Tiler(ROWS, COLS, tile_size=tile).halo_tiles(halo):
            r = t.read
            sl = (slice(None), slice(r.row_start, r.row_end),
                  slice(r.col_start, r.col_end))
            st.add_halo_tile(t, proc.decompose(*quad[sl]))
        tiled = st.finalize()
        for key, value in full.items():
            np.testing.assert_allclose(tiled[key], value, rtol=rtol,
                                       atol=1e-3, err_msg=key)

    def test_pauli_with_window(self, quad):
        proc = PauliDecomposition(window_size=5)
        full = proc.decompose(*quad)
        halo = get_tile_halo(proc)
        st = Stitcher(ROWS, COLS)
        for t in Tiler(ROWS, COLS, tile_size=40).halo_tiles(halo):
            r = t.read
            sl = (slice(None), slice(r.row_start, r.row_end),
                  slice(r.col_start, r.col_end))
            st.add_halo_tile(t, proc.decompose(*quad[sl]))
        tiled = st.finalize()
        for key, value in full.items():
            np.testing.assert_allclose(tiled[key], value, rtol=1e-4,
                                       atol=1e-4, err_msg=key)
