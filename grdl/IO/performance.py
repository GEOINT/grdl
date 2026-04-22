# -*- coding: utf-8 -*-
"""
IO Performance - Opt-in parallel reading utilities for imagery readers.

Provides a ``ReadConfig`` dataclass for configuring parallel I/O and
helper functions for multi-threaded band reads, chunked window reads,
and GDAL thread configuration.

All parallel features are **opt-in** — the default ``ReadConfig()``
preserves single-threaded behavior for full backward compatibility.

Dependencies
------------
rasterio (for parallel band and chunked reads)

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-04-01

Modified
--------
2026-04-01
"""

# Standard library
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# Third-party
import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False


@dataclass
class ReadConfig:
    """Configuration for parallel IO operations.

    All fields default to non-parallel behavior.  Pass an instance
    with ``parallel=True`` to any reader's constructor to enable
    multi-threaded reads.

    Parameters
    ----------
    parallel : bool
        Enable parallel reading.  Default ``False``.
    max_workers : int or None
        Thread pool size.  ``None`` uses ``os.cpu_count() - 1``.
    gdal_num_threads : int or None
        Set ``GDAL_NUM_THREADS`` for internal multi-threaded
        decompression (JPEG2000, DEFLATE, LZW).  ``None`` leaves
        the environment variable unchanged.
    chunk_threshold : int
        Minimum total pixels in a ``read_chip`` window before
        chunked parallel read is attempted.  Default 4,000,000
        (roughly 2000 × 2000).
    """

    parallel: bool = False
    max_workers: Optional[int] = None
    gdal_num_threads: Optional[int] = None
    chunk_threshold: int = 4_000_000


_gdal_threads_configured = False


def configure_gdal_threads(n: int) -> None:
    """Set GDAL_NUM_THREADS for multi-threaded decompression.

    This is a **process-wide** setting.  Call once at application
    startup, not per-reader.  rasterio (GDAL) already respects this
    environment variable for formats that support parallel
    decompression (JPEG2000, DEFLATE, LZW, ZSTD).

    Parameters
    ----------
    n : int
        Number of GDAL decompression threads.  Use ``0`` or ``ALL_CPUS``
        for automatic selection.
    """
    global _gdal_threads_configured
    if n <= 0:
        os.environ['GDAL_NUM_THREADS'] = 'ALL_CPUS'
    else:
        os.environ['GDAL_NUM_THREADS'] = str(n)
    _gdal_threads_configured = True


def _ensure_gdal_threads(config: ReadConfig) -> None:
    """Apply GDAL thread config once if specified."""
    global _gdal_threads_configured
    if (config.gdal_num_threads is not None
            and not _gdal_threads_configured):
        configure_gdal_threads(config.gdal_num_threads)


def _resolve_workers(config: ReadConfig) -> int:
    """Resolve the number of worker threads."""
    if config.max_workers is not None:
        return config.max_workers
    cpu = os.cpu_count() or 4
    return max(1, cpu - 1)


def parallel_band_read(
    dataset: 'rasterio.DatasetReader',
    window: 'Window',
    band_indices: Sequence[int],
    max_workers: int,
) -> np.ndarray:
    """Read multiple bands in parallel threads.

    rasterio releases the GIL during I/O, so threads scale well
    for multi-band reads from a single dataset.

    Parameters
    ----------
    dataset : rasterio.DatasetReader
        Open rasterio dataset.
    window : rasterio.windows.Window
        Spatial window to read.
    band_indices : sequence of int
        1-based band indices to read.
    max_workers : int
        Number of threads.

    Returns
    -------
    np.ndarray
        Shape ``(len(band_indices), window_height, window_width)``.
    """
    n_bands = len(band_indices)
    height = int(window.height)
    width = int(window.width)
    dtype = dataset.dtypes[0]
    out = np.empty((n_bands, height, width), dtype=dtype)
    filepath = dataset.name  # each thread opens its own handle

    def _read_one(args: Tuple[int, int]) -> None:
        idx, band = args
        with rasterio.open(filepath) as ds:
            out[idx] = ds.read(band, window=window)

    with ThreadPoolExecutor(max_workers=min(max_workers, n_bands)) as pool:
        list(pool.map(_read_one, enumerate(band_indices)))

    return out


def chunked_parallel_read(
    dataset: 'rasterio.DatasetReader',
    window: 'Window',
    bands: Optional[List[int]],
    max_workers: int,
) -> np.ndarray:
    """Read a large window by splitting into tile-aligned sub-windows.

    Splits the requested window into sub-windows aligned to the
    dataset's internal block structure, reads them concurrently,
    and assembles the result.

    Parameters
    ----------
    dataset : rasterio.DatasetReader
        Open rasterio dataset.
    window : rasterio.windows.Window
        Full spatial window to read.
    bands : list of int or None
        1-based band indices (None = all bands).
    max_workers : int
        Number of threads.

    Returns
    -------
    np.ndarray
        Assembled image data.
    """
    # Determine block size from dataset
    block_shapes = dataset.block_shapes
    block_h, block_w = block_shapes[0] if block_shapes else (512, 512)
    # Ensure reasonable minimum
    block_h = max(block_h, 256)
    block_w = max(block_w, 256)

    col_off = int(window.col_off)
    row_off = int(window.row_off)
    width = int(window.width)
    height = int(window.height)

    if bands is None:
        bands_list = list(range(1, dataset.count + 1))
    else:
        bands_list = bands

    n_bands = len(bands_list)
    dtype = dataset.dtypes[0]
    out = np.empty((n_bands, height, width), dtype=dtype)

    # Build sub-window list
    sub_windows = []
    for r_start in range(0, height, block_h):
        r_end = min(r_start + block_h, height)
        for c_start in range(0, width, block_w):
            c_end = min(c_start + block_w, width)
            sub_win = Window(
                col_off + c_start, row_off + r_start,
                c_end - c_start, r_end - r_start,
            )
            sub_windows.append((r_start, r_end, c_start, c_end, sub_win))

    filepath = dataset.name  # each thread opens its own handle

    def _read_sub(args: Tuple[int, int, int, int, 'Window']) -> None:
        r_s, r_e, c_s, c_e, sw = args
        with rasterio.open(filepath) as ds:
            data = ds.read(bands_list, window=sw)
        out[:, r_s:r_e, c_s:c_e] = data

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(_read_sub, sub_windows))

    return out
