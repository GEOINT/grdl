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
- Tiler: Stride-based overlapping tile region computation
- ChipRegion: Named tuple for clipped image region bounds
- Normalizer: Per-chip or per-image intensity normalization

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

Compute overlapping tile positions:

    >>> from grdl.data_prep import Tiler
    >>> tiler = Tiler(nrows=1000, ncols=2000, tile_size=256, stride=128)
    >>> tile_regions = tiler.tile_positions()

Normalize an image to [0, 1] range:

    >>> from grdl.data_prep import Normalizer
    >>> norm = Normalizer(method='minmax')
    >>> normalized = norm.normalize(image)

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
2026-02-06

Modified
--------
2026-02-09
"""

from grdl.data_prep.base import ChipBase, ChipRegion
from grdl.data_prep.chip_extractor import ChipExtractor
from grdl.data_prep.tiler import Tiler
from grdl.data_prep.normalizer import Normalizer

__all__ = [
    'ChipBase',
    'ChipRegion',
    'ChipExtractor',
    'Tiler',
    'Normalizer',
]
