# -*- coding: utf-8 -*-
"""
Data Preparation Module - Chip and tile index computation, and normalization.

Provides utilities for planning chip and tile layouts within bounded images,
and for normalizing pixel intensities. Chip and tile classes return index
bounds (``ChipRegion`` named tuples), not pixel data.

Key Classes
-----------
- ChipBase: ABC for image dimension management and coordinate clipping
- ChipExtractor: Point-centered and whole-image chip region computation
- Tiler: Stride-based overlapping tiles, plus non-overlapping partition
- ChipRegion: Named tuple for clipped image region bounds
- Normalizer: Per-chip or per-image intensity normalization (incl. streaming)
- StreamingStats: Online accumulator for full-image statistics
- StatsResult: Finalized streaming statistics container
- compute_image_statistics: Tiled (optionally parallel) full-image stats

Usage
-----
Compute chip regions centered at points:

    >>> from grdl.data_prep import ChipExtractor
    >>> ext = ChipExtractor(nrows=1000, ncols=2000)
    >>> region = ext.chip_at_point(500, 1000, row_width=64, col_width=64)
    >>> chip = image[region.row_start:region.row_end,
    ...              region.col_start:region.col_end]

Partition an image into non-overlapping chips:

    >>> regions = ext.chip_positions(row_width=256, col_width=256)

Compute overlapping tile positions (or a non-overlapping partition):

    >>> from grdl.data_prep import Tiler
    >>> tiler = Tiler(nrows=1000, ncols=2000, tile_size=256, stride=128)
    >>> tile_regions = tiler.tile_positions()        # overlapping
    >>> cover = Tiler(1000, 2000, tile_size=256).partition_positions()  # exact

Normalize an image to [0, 1] range:

    >>> from grdl.data_prep import Normalizer
    >>> norm = Normalizer(method='minmax')
    >>> normalized = norm.normalize(image)

Build a normalization baseline from a full image without loading it:

    >>> norm = Normalizer(method='zscore')
    >>> norm.fit_streaming('scene.nitf', mask='nonzero_finite')
    >>> chip_norm = norm.transform(chip)

Compute full-image statistics directly (tiled, optionally parallel):

    >>> from grdl.data_prep import compute_image_statistics
    >>> stats = compute_image_statistics('scene.nitf', mask='metadata',
    ...                                  percentiles=[1, 99])

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
2026-02-06

Modified
--------
2026-02-09
"""

from grdl.data_prep.base import ChipBase, ChipRegion
from grdl.data_prep.chip_extractor import ChipExtractor
from grdl.data_prep.tiler import Tiler
from grdl.data_prep.normalizer import Normalizer
from grdl.data_prep.streaming_stats import (
    StatsResult,
    StreamingStats,
    compute_image_statistics,
    build_valid_mask,
)

__all__ = [
    'ChipBase',
    'ChipRegion',
    'ChipExtractor',
    'Tiler',
    'Normalizer',
    'StatsResult',
    'StreamingStats',
    'compute_image_statistics',
    'build_valid_mask',
]
