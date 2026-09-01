# -*- coding: utf-8 -*-
"""
Peak-memory prediction for an orthorectification run.

Orthorectification is one of the few operations in the library whose
memory cost is not obvious from its inputs: the output grid can be far
larger than the source, and each tile pulls a source window whose size
depends on the geometry rather than on the tile.  A grid that looks
modest can need tens of gigabytes, and on a platform without memory
overcommit that takes the machine down rather than raising.

``estimate_ortho_memory`` answers "what will this cost?" before the
allocation happens, so a caller can pick a tile size, lower the worker
count, or coarsen the grid instead of finding out the hard way.

The per-tile figures are **calibrated from measurement, not derived**.
They are dominated by transients whose lifetime is set by the allocator
and by how many threads are in flight, not by any single array the code
controls.  Treat the result as an order-of-magnitude guard, not a
guarantee -- it exists to catch a run that obviously will not fit.

Dependencies
------------
numpy

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
2026-08-31

Modified
--------
2026-08-31
"""

# Standard library
import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.data_prep import Tiler
from grdl.data_prep.base import ChipRegion

logger = logging.getLogger(__name__)

# The tiled and untiled paths need different coefficients, because
# their transients scale differently.  Fitted against peak RSS on real
# collects (8866x8867 and 13736x41610).
#
# Tiled: the working set per in-flight tile is dominated by the
# projection batch buffers, which are allocated per worker *thread*
# inside the mapping stage, so the cost is close to fixed per tile
# rather than proportional to it.  Expressed per pixel of a 1024 tile,
# which is the auto-tiling default.  Predicts 5.87 GiB against 5.97
# measured on a 572 Mpx complex grid.
TILE_BYTES_PER_PX = 600

# Untiled: the mapping stage chunks into row-strips sized by worker
# count, so the big transients are strip-bounded, not grid-bounded.
# What does scale with the grid is the cached mapping (17 B/px), the
# chip-relative coordinate copies (16 B/px) and the backend's kernel
# output (8 B/px).  Rounded up to stay conservative: predicts ~35 GiB
# against 25 measured, which is the right direction for a guard.
UNTILED_BYTES_PER_PX = 50

# Floor independent of tile size: the resampling backend's runtime plus
# one in-flight projection batch per worker thread.
WORKING_FLOOR_BYTES = 1 << 30

# Bytes held per *source* pixel while a tile is read.  A complex reader
# whose chips are converted to magnitude holds both forms at once, and
# the resampling backends promote the chip to float64.
SOURCE_BYTES_PER_PX = 12


@dataclass(frozen=True)
class MemoryEstimate:
    """Predicted peak memory for an orthorectification run.

    Attributes
    ----------
    output_bytes : int
        The full output array, allocated once.  This is the floor: it
        cannot be reduced except by coarsening the grid or restricting
        the region.
    tile_bytes : int
        Per-tile working set, paid once per worker.  For an untiled
        plan this is the whole-grid working set.
    source_chip_bytes : int
        Largest per-tile source window over the plan, paid once per
        worker.
    floor_bytes : int
        Cost independent of tile size.
    peak_bytes : int
        Predicted peak.
    n_tiles : int
        Tiles in the plan.
    tile_size : Tuple[int, int]
        Tile dimensions, or the full grid when untiled.
    workers : int
        Concurrent tiles assumed.
    tiled : bool
        Whether the plan is tiled at all.
    """

    output_bytes: int
    tile_bytes: int
    source_chip_bytes: int
    floor_bytes: int
    peak_bytes: int
    n_tiles: int
    tile_size: Tuple[int, int]
    workers: int
    tiled: bool

    @property
    def peak_gb(self) -> float:
        """Predicted peak in gibibytes."""
        return self.peak_bytes / float(1 << 30)

    def report(self) -> str:
        """One-line human-readable summary.

        Returns
        -------
        str
        """
        gb = 1.0 / (1 << 30)
        mb = 1.0 / (1 << 20)
        how = (f'{self.n_tiles} tiles of '
               f'{self.tile_size[0]}x{self.tile_size[1]}'
               if self.tiled else 'untiled')
        workers = f' x{self.workers} workers' if self.workers > 1 else ''
        return (
            f'{how}{workers}  output {self.output_bytes * gb:.2f} GiB  '
            f'per-tile {self.tile_bytes * mb:.0f} MiB + chip '
            f'{self.source_chip_bytes * mb:.0f} MiB  '
            f'=> peak ~{self.peak_gb:.2f} GiB'
        )


def predict_source_window(
    output_grid: Any,
    geolocation: Any,
    region: ChipRegion,
    *,
    probe: int = 9,
    pad: int = 3,
) -> Optional[Tuple[int, int, int, int]]:
    """Predict the source window one output tile will read.

    The grid-to-source map is smooth, so the extrema over a rectangular
    tile lie on its boundary; sampling the perimeter is enough and costs
    a few dozen projections.  This matters because the window is set by
    geometry, not by tile size: a coarse output grid over a fine source
    can pull hundreds of millions of source pixels for one tile.

    Parameters
    ----------
    output_grid : OutputGridProtocol
        Grid being produced.
    geolocation : Geolocation
        Source image geolocation.
    region : ChipRegion
        The output tile.
    probe : int, default=9
        Samples per tile edge.
    pad : int, default=3
        Interpolation halo in source pixels.

    Returns
    -------
    Optional[Tuple[int, int, int, int]]
        ``(row_start, row_end, col_start, col_end)`` clipped to the
        source, or ``None`` when the tile maps entirely outside it.
    """
    r0, c0 = region.row_start, region.col_start
    r1, c1 = region.row_end, region.col_end

    edge = np.linspace(0.0, 1.0, probe)
    rows = np.concatenate([
        np.full(probe, r0 + 0.5), np.full(probe, r1 - 0.5),
        r0 + 0.5 + edge * (r1 - r0 - 1.0),
        r0 + 0.5 + edge * (r1 - r0 - 1.0),
    ])
    cols = np.concatenate([
        c0 + 0.5 + edge * (c1 - c0 - 1.0),
        c0 + 0.5 + edge * (c1 - c0 - 1.0),
        np.full(probe, c0 + 0.5), np.full(probe, c1 - 0.5),
    ])

    lats, lons = output_grid.image_to_latlon(rows, cols)
    source = geolocation.latlon_to_image(
        np.column_stack([np.asarray(lats), np.asarray(lons)]),
    )
    good = np.isfinite(source[:, 0]) & np.isfinite(source[:, 1])
    if not good.any():
        return None

    n_rows, n_cols = geolocation.shape
    sr0 = max(0, int(np.floor(source[good, 0].min())) - pad)
    sr1 = min(n_rows, int(np.ceil(source[good, 0].max())) + pad + 1)
    sc0 = max(0, int(np.floor(source[good, 1].min())) - pad)
    sc1 = min(n_cols, int(np.ceil(source[good, 1].max())) + pad + 1)
    if sr1 <= sr0 or sc1 <= sc0:
        return None
    return sr0, sr1, sc0, sc1


def estimate_ortho_memory(
    output_grid: Any,
    geolocation: Optional[Any] = None,
    *,
    tile_size: Optional[Union[int, Tuple[int, int]]] = None,
    dtype: Any = np.float32,
    bands: int = 1,
    workers: int = 1,
    source_bytes_per_px: int = SOURCE_BYTES_PER_PX,
    sample_tiles: int = 24,
) -> MemoryEstimate:
    """Predict peak memory for orthorectifying onto a grid.

    Parameters
    ----------
    output_grid : OutputGridProtocol
        Grid to be produced.
    geolocation : Geolocation, optional
        Source geolocation.  When given, per-tile source windows are
        measured by projection rather than guessed, which is the term
        that varies most between collects.
    tile_size : int or (int, int), optional
        Tile dimensions.  ``None`` estimates the untiled path, where
        the whole grid is one "tile" -- which is exactly why that path
        is expensive.
    dtype : np.dtype, default=float32
        Output dtype.
    bands : int, default=1
        Output band count.
    workers : int, default=1
        Concurrent tiles; per-tile terms are paid once per worker.
    source_bytes_per_px : int
        Bytes held per source pixel while a tile is in flight.
    sample_tiles : int, default=24
        Tiles sampled when measuring source windows.  The largest
        window found is used.

    Returns
    -------
    MemoryEstimate

    Examples
    --------
    >>> est = estimate_ortho_memory(grid, geo, tile_size=1024, workers=4)
    >>> print(est.report())
    >>> if est.peak_gb > 8.0:
    ...     est = estimate_ortho_memory(grid, geo, tile_size=512)
    """
    rows, cols = int(output_grid.rows), int(output_grid.cols)
    itemsize = np.dtype(dtype).itemsize
    output_bytes = rows * cols * itemsize * max(1, bands)

    tiled = tile_size is not None
    if tiled:
        tiler = Tiler(rows, cols, tile_size=tile_size)
        regions: List[ChipRegion] = tiler.partition_positions()
    else:
        regions = [ChipRegion(0, 0, rows, cols)]

    tile_px = max(
        (r.row_end - r.row_start) * (r.col_end - r.col_start)
        for r in regions
    )
    tile_dims = (
        regions[0].row_end - regions[0].row_start,
        regions[0].col_end - regions[0].col_start,
    )

    chip_px = 0
    if geolocation is not None:
        step = max(1, len(regions) // max(1, sample_tiles))
        for region in regions[::step]:
            bbox = predict_source_window(output_grid, geolocation, region)
            if bbox is not None:
                r_a, r_b, c_a, c_b = bbox
                chip_px = max(chip_px, (r_b - r_a) * (c_b - c_a))
    else:
        chip_px = tile_px

    n_workers = max(1, int(workers))
    per_px = TILE_BYTES_PER_PX if tiled else UNTILED_BYTES_PER_PX
    per_worker = (tile_px * per_px
                  + chip_px * source_bytes_per_px
                  + tile_px * itemsize * max(1, bands))

    estimate = MemoryEstimate(
        output_bytes=output_bytes,
        tile_bytes=tile_px * TILE_BYTES_PER_PX,
        source_chip_bytes=chip_px * source_bytes_per_px,
        floor_bytes=WORKING_FLOOR_BYTES,
        peak_bytes=(output_bytes + WORKING_FLOOR_BYTES
                    + n_workers * per_worker),
        n_tiles=len(regions),
        tile_size=tile_dims,
        workers=n_workers,
        tiled=tiled,
    )
    logger.debug("Ortho memory estimate: %s", estimate.report())
    return estimate


def default_workers() -> int:
    """A conservative default for concurrent tiles.

    The stages inside a tile are already threaded but only reach two to
    four cores, so a few concurrent tiles is what actually saturates a
    machine.  Returns are flat beyond about four while memory keeps
    scaling, so this caps there rather than using the core count.

    Returns
    -------
    int
    """
    return max(1, min(4, (os.cpu_count() or 2) - 1))
