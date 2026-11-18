"""
Tests that declared tile halos reproduce full-image results for core GRDL filters.

Author
------
Jason Fritz
43161141+stryder-vtx@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.
"""

import numpy as np
import pytest

from grdl.data_prep import Stitcher, Tiler
from grdl.image_processing.filters import (
    MaxFilter,
    MeanFilter,
    MedianFilter,
    MinFilter,
    StdDevFilter,
)
from grdl.image_processing.filters.linear import boxcar_mean
from grdl.image_processing.tiling import get_tile_halo

ROWS, COLS = 83, 101


@pytest.fixture(scope="module")
def sample_cube():
    rng = np.random.default_rng(7)
    shape = (4, ROWS, COLS)
    cube = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return cube.astype(np.complex64)


def _tiled(proc, data, fn, tile=32):
    """Run fn(block) per halo tile over the last two axes and stitch."""
    halo = get_tile_halo(proc)
    assert halo is not None
    st = Stitcher(ROWS, COLS)
    for t in Tiler(ROWS, COLS, tile_size=tile).halo_tiles(halo):
        r = t.read
        block = data[..., r.row_start : r.row_end, r.col_start : r.col_end]
        st.add_halo_tile(t, fn(block))
    return st.finalize()


class TestDeclaredHalos:
    @pytest.mark.parametrize(
        "proc,expected",
        [
            (MeanFilter(kernel_size=7), (3, 3)),
            (MeanFilter(kernel_size=5), (2, 2)),
            (MeanFilter(kernel_size=3), (1, 1)),
        ],
    )
    def test_halo_values(self, proc, expected):
        assert get_tile_halo(proc) == expected

    def test_mean_wrap_is_global(self):
        assert get_tile_halo(MeanFilter(kernel_size=7, mode="wrap")) is None

    def test_undeclared_is_global(self):
        assert get_tile_halo(MedianFilter(kernel_size=3)) is None
        assert get_tile_halo(StdDevFilter(kernel_size=5)) is None
        assert get_tile_halo(MinFilter(kernel_size=5)) is None
        assert get_tile_halo(MaxFilter(kernel_size=5)) is None
        assert get_tile_halo(object()) is None

    def test_callable_halo(self):
        class P:
            def tile_halo(self):
                return (1, 2)

        assert get_tile_halo(P()) == (1, 2)


class TestHaloEquivalence:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("window", [3, 7])
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest"])
    def test_mean_filter(self, sample_cube, dtype, window, mode):
        source = np.abs(sample_cube[0]).astype(dtype)
        proc = MeanFilter(kernel_size=window, mode=mode)
        full = proc.apply(source)
        tiled = _tiled(proc, source, proc.apply, tile=29)
        tolerance = 1e-6 if dtype == np.float32 else 1e-12
        np.testing.assert_allclose(tiled, full, rtol=tolerance, atol=tolerance)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    @pytest.mark.parametrize("window", [1, 7])
    def test_complex_boxcar(self, sample_cube, dtype, window):
        source = sample_cube[0].astype(dtype)

        class _BoxcarProc:
            def tile_halo(self):
                pad = window // 2
                return (pad, pad)

        full = boxcar_mean(source, window)
        tiled = _tiled(_BoxcarProc(), source, lambda block: boxcar_mean(block, window), tile=29)
        tolerance = 1e-6 if dtype == np.complex64 else 1e-12
        np.testing.assert_allclose(tiled, full, rtol=tolerance, atol=tolerance)
