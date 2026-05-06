# -*- coding: utf-8 -*-
"""
Tests for grdl.shapes.backend dispatch hierarchy.

The harness is NumPy-only on most CI machines. Tests verify:

- Default backend resolves successfully and reports the expected
  string identifiers.
- ``set_backend`` overrides are honoured.
- Missing-backend requests emit a UserWarning and fall back cleanly.
- batch_map returns results in input order across parallel modes.

Dependencies
------------
pytest

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-04-18

Modified
--------
2026-04-18
"""

import warnings

import numpy as np
import pytest

from grdl.shapes.backend import (
    batch_map,
    cupy_available,
    detect_backend,
    get_array_module,
    numba_available,
    set_backend,
)


@pytest.fixture(autouse=True)
def _restore_backend():
    """Reset backend after each test so tests stay independent."""
    original = detect_backend()
    yield
    set_backend(
        array=original.array,
        jit=original.jit,
        parallel=original.parallel,
        gpu_threshold=original.gpu_threshold,
        max_workers=original.max_workers,
    )


class TestDetectAndSet:
    def test_default_resolves(self):
        backend = detect_backend()
        assert backend.array in ('cupy', 'numpy')
        assert backend.jit in ('numba', 'none')
        assert backend.parallel in ('process', 'thread', 'none')

    def test_force_numpy(self):
        backend = set_backend(array='numpy')
        assert backend.array == 'numpy'
        assert get_array_module(1_000_000) is np

    def test_force_missing_cupy_falls_back(self):
        if cupy_available():
            pytest.skip("cupy is installed; cannot test fallback")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            backend = set_backend(array='cupy')
            assert backend.array == 'numpy'
            assert any('cupy' in str(item.message) for item in w)

    def test_force_missing_numba_falls_back(self):
        if numba_available():
            pytest.skip("numba is installed; cannot test fallback")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            backend = set_backend(jit='numba')
            assert backend.jit == 'none'
            assert any('numba' in str(item.message) for item in w)


class TestBatchMap:
    def test_sequential(self):
        set_backend(parallel='none')
        out = batch_map(lambda x: x * 2, [1, 2, 3, 4])
        assert out == [2, 4, 6, 8]

    def test_thread_preserves_order(self):
        set_backend(parallel='thread')
        out = batch_map(lambda x: x ** 2, [1, 2, 3, 4])
        assert out == [1, 4, 9, 16]

    def test_empty_input(self):
        assert batch_map(lambda x: x, []) == []

    def test_single_input(self):
        assert batch_map(lambda x: x + 1, [5]) == [6]
