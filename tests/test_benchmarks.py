# -*- coding: utf-8 -*-
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

import numpy as np
import pytest

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
        norm = Normalizer(method='minmax')
        benchmark(norm.normalize, medium_image)

    def test_normalizer_zscore(self, benchmark, medium_image):
        from grdl.data_prep import Normalizer
        norm = Normalizer(method='zscore')
        benchmark(norm.normalize, medium_image)
