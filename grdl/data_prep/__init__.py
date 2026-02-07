# -*- coding: utf-8 -*-
"""
Data Preparation Module - Tiling, chip extraction, and normalization utilities.

Provides concrete utilities for formatting imagery into ML/AI pipeline-ready
formats. Includes configurable image tiling with overlap support, chip
extraction at point and polygon locations, and per-chip or per-image intensity
normalization with fit/transform semantics.

Key Classes
-----------
- Tiler: Split images into overlapping tiles with configurable stride
- ChipExtractor: Extract image chips at specified locations
- Normalizer: Per-chip or per-image intensity normalization

Usage
-----
Tile a large image into overlapping 256x256 chips:

    >>> from grdl.data_prep import Tiler
    >>> tiler = Tiler(tile_size=256, stride=128)
    >>> tiles = tiler.tile(image)
    >>> reconstructed = tiler.untile(tiles, image.shape)

Extract chips centered at point locations:

    >>> from grdl.data_prep import ChipExtractor
    >>> extractor = ChipExtractor(chip_size=64)
    >>> chips = extractor.extract_at_points(image, points)

Normalize an image to [0, 1] range:

    >>> from grdl.data_prep import Normalizer
    >>> norm = Normalizer(method='minmax')
    >>> normalized = norm.normalize(image)

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
2026-02-06
"""

from grdl.data_prep.tiler import Tiler
from grdl.data_prep.chip_extractor import ChipExtractor
from grdl.data_prep.normalizer import Normalizer

__all__ = [
    'Tiler',
    'ChipExtractor',
    'Normalizer',
]
