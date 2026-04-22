# -*- coding: utf-8 -*-
"""
Tests for grdl.IO.performance — ReadConfig and parallel helpers.

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org

Created
-------
2026-04-01

Modified
--------
2026-04-01
"""

import os

import pytest

from grdl.IO.performance import (
    ReadConfig,
    configure_gdal_threads,
    _resolve_workers,
)


class TestReadConfig:
    """Test ReadConfig defaults and behavior."""

    def test_defaults(self):
        cfg = ReadConfig()
        assert cfg.parallel is False
        assert cfg.max_workers is None
        assert cfg.gdal_num_threads is None
        assert cfg.chunk_threshold == 4_000_000

    def test_custom_values(self):
        cfg = ReadConfig(parallel=True, max_workers=4,
                         gdal_num_threads=8, chunk_threshold=1_000_000)
        assert cfg.parallel is True
        assert cfg.max_workers == 4
        assert cfg.gdal_num_threads == 8
        assert cfg.chunk_threshold == 1_000_000


class TestResolveWorkers:
    """Test _resolve_workers."""

    def test_explicit_workers(self):
        cfg = ReadConfig(max_workers=6)
        assert _resolve_workers(cfg) == 6

    def test_default_workers(self):
        cfg = ReadConfig()
        workers = _resolve_workers(cfg)
        cpu = os.cpu_count() or 4
        assert workers == max(1, cpu - 1)


class TestConfigureGDALThreads:
    """Test configure_gdal_threads sets env var."""

    def test_sets_env_var(self):
        configure_gdal_threads(4)
        assert os.environ.get('GDAL_NUM_THREADS') == '4'

    def test_zero_sets_all_cpus(self):
        configure_gdal_threads(0)
        assert os.environ.get('GDAL_NUM_THREADS') == 'ALL_CPUS'
