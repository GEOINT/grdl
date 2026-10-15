# -*- coding: utf-8 -*-
"""
Tiling declarations for image processors.

A processor advertises whether it can run on image tiles by exposing a
``tile_halo`` attribute (or zero-argument callable):

- an ``int`` or ``(row, col)`` pair: tile-safe when each tile is read with
  that many extra pixels on every side (``0`` for pixelwise operations);
- ``None``: needs the whole image (global statistics, adaptive neighborhoods).

A processor that declares nothing is treated as global.

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

# Standard library
from typing import Any, Optional, Tuple

# GRDL internal
from grdl.data_prep.tiler import _normalize_halo


def get_tile_halo(processor: Any) -> Optional[Tuple[int, int]]:
    """Return a processor's ``(row, col)`` tile halo, or None if global.

    Parameters
    ----------
    processor : object
        Processor instance, or a class that defines ``tile_halo`` as a
        plain value.

    Returns
    -------
    Tuple[int, int] or None
        None means the processor cannot run tile by tile.
    """
    value = getattr(processor, 'tile_halo', None)
    if callable(value):
        value = value()
    if value is None:
        return None
    return _normalize_halo(value)


class WindowedTileHalo:
    """Mixin for boxcar-windowed processors: halo is ``window_size // 2``.

    Valid when the window average uses reflect padding at the array edge,
    so a tile read with this halo matches a full-image run.
    """

    @property
    def tile_halo(self) -> Tuple[int, int]:
        half = int(getattr(self, 'window_size', 1)) // 2
        return (half, half)
