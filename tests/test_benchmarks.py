"""
Performance benchmarks for GRDL image processing components.

Uses pytest-benchmark to track execution time for key operations.
Run with: ``pytest tests/test_benchmarks.py --benchmark-only``

Skip benchmarks during normal test runs with:
``pytest tests/ --benchmark-disable``

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
2026-02-06

Modified
--------
2026-02-10
"""

import tracemalloc

import numpy as np
import pytest
from scipy.ndimage import uniform_filter

from grdl.image_processing.decomposition.pol_matrix import (
    build_target_vector_c,
    build_target_vector_t,
    compute_dual_elements,
    compute_matrix,
)

# Mark all tests in this module as benchmarks so they can be skipped
# during normal test runs: pytest tests/ -m "not benchmark"
pytestmark = pytest.mark.benchmark


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_image():
    """64x64 random image for fast benchmarks."""
    rng = np.random.RandomState(42)
    return rng.rand(64, 64) * 255.0


@pytest.fixture
def medium_image():
    """256x256 random image for realistic benchmarks."""
    rng = np.random.RandomState(42)
    return rng.rand(256, 256) * 255.0


@pytest.fixture
def stack_3d():
    """5-band 64x64 image stack."""
    rng = np.random.RandomState(42)
    return rng.rand(5, 64, 64) * 255.0


# ---------------------------------------------------------------------------
# Data Prep Benchmarks
# ---------------------------------------------------------------------------


class TestDataPrepBenchmarks:
    """Benchmark data prep utilities."""

    def test_tiler_tile(self, benchmark, medium_image):
        from grdl.data_prep import Tiler

        tiler = Tiler(nrows=256, ncols=256, tile_size=64, stride=32)
        benchmark(tiler.tile_positions)

    def test_normalizer_minmax(self, benchmark, medium_image):
        from grdl.data_prep import Normalizer

        norm = Normalizer(method="minmax")
        benchmark(norm.normalize, medium_image)

    def test_normalizer_zscore(self, benchmark, medium_image):
        from grdl.data_prep import Normalizer

        norm = Normalizer(method="zscore")
        benchmark(norm.normalize, medium_image)


def _previous_dual_elements(components, window):
    """Historical element kernel, with precision-matched target components."""
    first, second = components
    diagonal_first = uniform_filter(np.abs(first) ** 2, size=window)
    diagonal_second = uniform_filter(np.abs(second) ** 2, size=window)
    real = uniform_filter((first * second.conj()).real, size=window)
    imag = uniform_filter((first * second.conj()).imag, size=window)
    return diagonal_first, real + 1j * imag, diagonal_second


def _previous_full_matrix(components, window):
    """Historical allocation/filter strategy, without its complex64 downcast."""
    result = np.zeros(
        (len(components), len(components), *components[0].shape), dtype=components[0].dtype
    )
    for row, first in enumerate(components):
        for col in range(row, len(components)):
            product = first * components[col].conj()
            element = uniform_filter(product.real, size=window).astype(
                product.real.dtype
            ) + 1j * uniform_filter(product.imag, size=window).astype(product.imag.dtype)
            result[row, col] = element
            if row != col:
                result[col, row] = element.conj()
    return result


def _shared_full_matrix(components, window):
    return compute_matrix(components, window, False)


@pytest.mark.parametrize(
    "basis,dimension,elements_only",
    [
        ("C", 2, True),
        ("T", 2, True),
        ("C", 2, False),
        ("T", 2, False),
        ("C", 3, False),
        ("T", 3, False),
    ],
)
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("window", [1, 7])
@pytest.mark.parametrize("implementation", ["previous", "shared"])
def test_polarimetric_formation(
    benchmark, basis, dimension, elements_only, dtype, window, implementation
):
    """Compare formation kernels; input/target allocation is outside measurement."""
    rng = np.random.default_rng(107)
    channels = (
        rng.normal(size=(dimension, 256, 256)) + 1j * rng.normal(size=(dimension, 256, 256))
    ).astype(dtype)
    builder = build_target_vector_c if basis == "C" else build_target_vector_t
    components = builder(channels)
    previous = _previous_dual_elements if elements_only else _previous_full_matrix
    shared = compute_dual_elements if elements_only else _shared_full_matrix
    operation = previous if implementation == "previous" else shared
    result = benchmark.pedantic(
        operation, args=(components, window), rounds=9, iterations=1, warmup_rounds=2
    )
    tolerance = 2e-6 if dtype == np.complex64 else 5e-14
    np.testing.assert_allclose(result, previous(components, window), rtol=tolerance, atol=tolerance)
    tracemalloc.start()
    try:
        measured = operation(components, window)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    del measured
    benchmark.extra_info["peak_traced_bytes"] = peak
    benchmark.extra_info["input_shape"] = list(channels.shape)
    benchmark.extra_info["baseline"] = "historical kernel; precision matched; targets precomputed"
