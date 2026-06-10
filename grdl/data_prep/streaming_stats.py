# -*- coding: utf-8 -*-
"""
Streaming full-image statistics for normalization baselines.

Computes mean / variance / std / min / max exactly via Chan's batch-parallel
merge (each tile reduced with vectorized numpy, then merged) and estimates
percentiles from a fixed-bin histogram. The default ``'float32'`` histogram
bins values by the top bits of their IEEE-754 total-order key -- a fixed,
data-independent geometry with uniform relative resolution (~0.55%% per bin at
65536 bins) over the entire float32 range -- so percentiles come out of the
same single pass as the exact moments. Explicit ``'log'``/``'linear'``
histograms over a data-driven range remain available (two passes). Designed to
consume any GRDL ``ImageReader`` tile-by-tile so a full-image normalization
baseline (z-score / percentile clip) can be computed without holding the whole
image in memory, and to scale across CPU cores for large imagery.

The reductions are pure numpy element-wise/reduction ops and never call BLAS,
so a threaded BLAS backend cannot parallelize them. :func:`compute_image_statistics`
instead fans tiles out across processes (each worker owns its own reader and a
local :class:`StreamingStats`, merged in the parent) when the job is large
enough to amortize process-pool startup, and runs serially otherwise.

Valid-pixel selection is generic across readers:
  * ``'none'``           -- every finite pixel
  * ``'nonzero_finite'`` -- drop zeros and non-finite (per-tile, data-driven;
    the zero test runs on the raw pixels, before any value transform)
  * ``'metadata'``       -- rasterize the reader's valid-data polygon
  * ``'both'``           -- metadata polygon AND nonzero_finite

Dependencies
------------
grdl.IO (readers), and -- only for ``'metadata'`` masking --
grdl.geolocation and grdl.shapes (imported lazily).

Author
------
Duane Smalley
duane.d.smalley@gmail.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-06-07

Modified
--------
2026-06-09
"""

# Standard library
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.data_prep.base import ChipRegion
from grdl.data_prep.tiler import Tiler
from grdl.exceptions import ValidationError

__all__ = [
    'StatsResult',
    'StreamingStats',
    'compute_image_statistics',
    'build_valid_mask',
    'VALUE_TRANSFORMS',
    'MASK_STRATEGIES',
    'PARALLEL_MIN_PIXELS',
]


# ---------------------------------------------------------------------------
# Value transforms (complex -> real scalar for statistics)
# ---------------------------------------------------------------------------

def _magnitude(chip: np.ndarray) -> np.ndarray:
    """Magnitude ``|z|`` for complex input; passthrough for real input."""
    if np.iscomplexobj(chip):
        return np.abs(chip)
    return chip


def _power(chip: np.ndarray) -> np.ndarray:
    """Squared magnitude (power) for complex input; square for real input."""
    if np.iscomplexobj(chip):
        return (chip.real * chip.real) + (chip.imag * chip.imag)
    return chip.astype(np.float64) ** 2


def _decibel(chip: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """Decibels ``20*log10(|z|)`` with a small floor to avoid ``log(0)``.

    Computed in the magnitude's native float dtype (float32 stays float32 --
    log10 in float64 is ~3x slower and doubles memory bandwidth).
    """
    mag = _magnitude(chip)
    if not np.issubdtype(mag.dtype, np.floating):
        mag = mag.astype(np.float64)
    return 20.0 * np.log10(np.maximum(mag, mag.dtype.type(floor)))


def _identity(chip: np.ndarray) -> np.ndarray:
    """Passthrough. Requires real input."""
    return chip


VALUE_TRANSFORMS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    'auto': _magnitude,       # magnitude for complex, passthrough for real
    'magnitude': _magnitude,
    'power': _power,
    'decibel': _decibel,
    'identity': _identity,
}

MASK_STRATEGIES = ('none', 'nonzero_finite', 'metadata', 'both')

# Below this many tile pixels, in-process serial accumulation beats a process
# pool: the per-job numpy work is smaller than the spawn + import + per-worker
# reader-open overhead. Tunable per machine (core count / disk speed).
PARALLEL_MIN_PIXELS = 500_000_000


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class StatsResult:
    """Finalized statistics over the accumulated valid pixels.

    Attributes
    ----------
    count : int
        Number of valid samples accumulated.
    mean, var, std : float
        Sample mean, population variance, and standard deviation.
    minimum, maximum : float
        Extremes over the valid samples.
    percentiles : dict of float -> float
        Requested quantiles (key in [0, 100]) to estimated value. Empty when
        no histogram was configured.
    """

    count: int
    mean: float
    var: float
    std: float
    minimum: float
    maximum: float
    percentiles: Dict[float, float] = field(default_factory=dict)

    @property
    def l2_norm(self) -> float:
        """Exact L2 norm ``sqrt(sum(x**2))`` derived from count/mean/var."""
        if self.count <= 0:
            return float('nan')
        return float(np.sqrt(self.count * (self.var + self.mean * self.mean)))

    def summary(self) -> str:
        """One-line human-readable summary."""
        pct = '  '.join(
            f'p{q:g}={v:.6g}' for q, v in sorted(self.percentiles.items())
        )
        base = (
            f'n={self.count:,}  mean={self.mean:.6g}  std={self.std:.6g}  '
            f'min={self.minimum:.6g}  max={self.maximum:.6g}'
        )
        return base + (f'\n        {pct}' if pct else '')


# ---------------------------------------------------------------------------
# Streaming accumulator
# ---------------------------------------------------------------------------

class StreamingStats:
    """Online accumulator for full-image statistics.

    Exact for count/mean/var/std/min/max via Chan's parallel (batch) merge.
    Percentiles are estimated from a fixed-bin histogram. The default
    ``'float32'`` spacing bins values by the top bits of their IEEE-754
    total-order key: a data-independent geometry with ~0.55%% relative bin
    width (at 65536 bins) over the full float32 range, requiring no a-priori
    value range -- percentiles are available from a single pass. ``'log'`` and
    ``'linear'`` spacings bin over an explicit ``hist_range`` instead
    (typically from a cheap first pass over min/max).

    The accumulator is associative: independent instances over disjoint pixel
    subsets can be combined with :meth:`merge` to give the same result as a
    single pass. This is what allows process-parallel accumulation.

    Parameters
    ----------
    percentiles : sequence of float, optional
        Quantiles in [0, 100] to estimate. Requires a configured histogram.
    hist_range : (float, float), optional
        ``(low, high)`` value range for ``'log'``/``'linear'`` spacing.
        Ignored by ``'float32'`` spacing (its geometry is fixed). If omitted
        and spacing is not ``'float32'``, no histogram is built and
        percentiles are unavailable.
    n_bins : int
        Number of histogram bins. Default 65536. Must be a power of two
        between 256 and 2**32 for ``'float32'`` spacing.
    hist_spacing : {'float32', 'log', 'linear'}
        Bin geometry. ``'float32'`` (default) needs no range and is exact in
        ordering (values quantize to the float32 grid; magnitudes beyond
        float32 range saturate). ``'log'`` suits magnitude/power imagery when
        an explicit range is preferred.

    Raises
    ------
    ValidationError
        If ``hist_spacing`` is not recognized, or ``n_bins`` is invalid for
        ``'float32'`` spacing.
    """

    def __init__(
        self,
        percentiles: Optional[Sequence[float]] = None,
        hist_range: Optional[Tuple[float, float]] = None,
        n_bins: int = 65536,
        hist_spacing: str = 'float32',
    ) -> None:
        self._n: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0
        self._min: float = np.inf
        self._max: float = -np.inf

        self.percentiles = list(percentiles) if percentiles is not None else []
        self._n_bins = int(n_bins)
        self._spacing = hist_spacing
        self._counts: Optional[np.ndarray] = None
        self._lo: float = 0.0
        self._span: float = 0.0
        self._shift: int = 0
        if hist_spacing == 'float32':
            n = self._n_bins
            if n & (n - 1) or not (256 <= n <= 2 ** 32):
                raise ValidationError(
                    "'float32' spacing requires n_bins to be a power of two "
                    f"in [256, 2**32]; got {n}"
                )
            self._shift = 32 - (n.bit_length() - 1)
            # Histogram only materialized when percentiles were requested or
            # an (ignored) range was supplied -- mirrors log/linear gating.
            if self.percentiles or hist_range is not None:
                self._counts = np.zeros(n, dtype=np.int64)
        elif hist_range is not None:
            low, high = float(hist_range[0]), float(hist_range[1])
            if hist_spacing == 'log':
                low = max(low, np.finfo(np.float64).tiny)
                self._lo = float(np.log10(low))
                self._span = float(np.log10(high)) - self._lo
            elif hist_spacing == 'linear':
                self._lo = low
                self._span = high - low
            else:
                raise ValidationError(f"unknown hist_spacing: {hist_spacing!r}")
            if self._span <= 0.0:
                self._span = 1.0
            self._counts = np.zeros(self._n_bins, dtype=np.int64)
        elif hist_spacing not in ('log', 'linear'):
            raise ValidationError(f"unknown hist_spacing: {hist_spacing!r}")

    def _bin_edge(self, i: int) -> float:
        """Value at the left edge of bin ``i`` in the original domain."""
        if self._spacing == 'float32':
            # Invert the total-order key: k >= 2**31 is the non-negative
            # side (bits = k - 2**31), below it the negative side (bits = ~k).
            k = min(int(i) << self._shift, 0xFFFFFFFF)
            bits = np.uint32(k - 0x80000000 if k >= 0x80000000 else ~np.uint32(k))
            val = float(bits.view(np.float32))
            if not np.isfinite(val):
                fmax = float(np.finfo(np.float32).max)
                val = fmax if k >= 0x80000000 else -fmax
            return val
        pos = self._lo + (i / self._n_bins) * self._span
        return float(10.0 ** pos) if self._spacing == 'log' else float(pos)

    # -- accumulation -------------------------------------------------------

    def update(self, values: np.ndarray) -> None:
        """Fold a batch of real-valued samples into the running statistics.

        Parameters
        ----------
        values : np.ndarray
            Samples of any shape (flattened internally). Non-finite entries
            are dropped defensively, so NaN/inf never corrupt the result.
        """
        x = np.asarray(values).ravel()
        if x.size == 0:
            return
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(np.float64)
        finite = np.isfinite(x)
        if not finite.all():
            x = x[finite]
            if x.size == 0:
                return

        n_b = x.size
        mean_b = float(x.mean())
        m2_b = float(((x - mean_b) ** 2).sum())

        if self._n == 0:
            self._n, self._mean, self._m2 = n_b, mean_b, m2_b
        else:
            n_a, mean_a, m2_a = self._n, self._mean, self._m2
            n = n_a + n_b
            delta = mean_b - mean_a
            self._mean = mean_a + delta * (n_b / n)
            self._m2 = m2_a + m2_b + delta * delta * (n_a * n_b / n)
            self._n = n

        bmin = float(x.min())
        bmax = float(x.max())
        if bmin < self._min:
            self._min = bmin
        if bmax > self._max:
            self._max = bmax

        if self._counts is not None:
            if self._spacing == 'float32':
                b = np.ascontiguousarray(
                    x.astype(np.float32, copy=False)
                ).view(np.uint32)
                if bmin > 0.0:
                    # Strictly positive (typical for SAR magnitude/power):
                    # the key is just bits | sign-offset -- one shift + add.
                    idx = (b >> np.uint32(self._shift)).astype(np.intp)
                    idx += self._n_bins >> 1
                else:
                    # General path: IEEE-754 total-order key is monotonic in
                    # value across negatives, +/-0, and positives.
                    neg = b >= np.uint32(0x80000000)
                    key = np.where(neg, ~b, b | np.uint32(0x80000000))
                    idx = (key >> np.uint32(self._shift)).astype(np.intp)
            else:
                if self._spacing == 'log':
                    # Floor in the array's own dtype -- a float64 floor would
                    # silently promote a float32 tile (3x slower log10).
                    pos = np.log10(np.maximum(x, np.finfo(x.dtype).tiny))
                else:
                    pos = x
                idx = ((pos - self._lo) / self._span * self._n_bins
                       ).astype(np.intp)
                np.clip(idx, 0, self._n_bins - 1, out=idx)
            self._counts += np.bincount(idx, minlength=self._n_bins)

    def merge(self, other: 'StreamingStats') -> None:
        """Merge another accumulator (e.g. from a worker) in place.

        Parameters
        ----------
        other : StreamingStats
            Accumulator over a disjoint pixel subset. Must share histogram
            geometry for percentiles to combine meaningfully.
        """
        if other._n == 0:
            return
        if self._n == 0:
            self._n, self._mean, self._m2 = other._n, other._mean, other._m2
        else:
            n_a, mean_a, m2_a = self._n, self._mean, self._m2
            n_b, mean_b, m2_b = other._n, other._mean, other._m2
            n = n_a + n_b
            delta = mean_b - mean_a
            self._mean = mean_a + delta * (n_b / n)
            self._m2 = m2_a + m2_b + delta * delta * (n_a * n_b / n)
            self._n = n
        self._min = min(self._min, other._min)
        self._max = max(self._max, other._max)
        if self._counts is not None and other._counts is not None:
            self._counts += other._counts

    # -- finalize -----------------------------------------------------------

    def _percentile_from_hist(self, q: float) -> float:
        """Linear-interpolated percentile from the cumulative histogram."""
        assert self._counts is not None
        total = self._counts.sum()
        if total == 0:
            return float('nan')
        target = (q / 100.0) * total
        cum = np.cumsum(self._counts)
        idx = int(np.searchsorted(cum, target, side='left'))
        idx = min(idx, self._n_bins - 1)
        below = cum[idx - 1] if idx > 0 else 0
        in_bin = self._counts[idx]
        lo, hi = self._bin_edge(idx), self._bin_edge(idx + 1)
        if in_bin == 0:
            return float(lo)
        frac = (target - below) / in_bin
        return float(lo + frac * (hi - lo))

    def result(self) -> StatsResult:
        """Return the finalized statistics.

        Returns
        -------
        StatsResult
            Mean/var/std/min/max are exact; percentiles (if a histogram was
            configured) are histogram-interpolated.
        """
        var = (self._m2 / self._n) if self._n > 0 else float('nan')
        pcts: Dict[float, float] = {}
        if self._counts is not None:
            for q in self.percentiles:
                pcts[q] = self._percentile_from_hist(q)
        return StatsResult(
            count=self._n,
            mean=self._mean if self._n else float('nan'),
            var=var,
            std=float(np.sqrt(var)) if self._n else float('nan'),
            minimum=self._min if self._n else float('nan'),
            maximum=self._max if self._n else float('nan'),
            percentiles=pcts,
        )


# ---------------------------------------------------------------------------
# Valid-pixel selection (generic across all readers)
# ---------------------------------------------------------------------------

def _valid_polygon(reader: object) -> Optional[np.ndarray]:
    """Return a reader's valid-data polygon as ``(N, 2)`` pixel vertices.

    Prefers the authoritative pixel-space valid-data polygon
    (``metadata.image_data.valid_data`` -- row/col vertices, no geolocation
    required); falls back to the geographic polygon
    (``metadata.geo_data.valid_data`` -- lat/lon vertices projected via
    :func:`grdl.geolocation.create_geolocation`). Returns ``None`` if the
    reader exposes no valid-data polygon.
    """
    meta = getattr(reader, 'metadata', None)

    # 1. Pixel-space valid-data polygon (preferred -- exact, no geolocation).
    image_data = getattr(meta, 'image_data', None) if meta is not None else None
    pix = getattr(image_data, 'valid_data', None) if image_data is not None else None
    if pix:
        first_row = float(getattr(image_data, 'first_row', 0) or 0)
        first_col = float(getattr(image_data, 'first_col', 0) or 0)
        return np.array(
            [[v.row - first_row, v.col - first_col] for v in pix],
            dtype=np.float64,
        )

    # 2. Geographic valid-data polygon -> pixel via geolocation.
    geo_data = getattr(meta, 'geo_data', None) if meta is not None else None
    valid = getattr(geo_data, 'valid_data', None) if geo_data is not None else None
    if valid:
        from grdl.geolocation import create_geolocation

        latlon = np.array([[v.lat, v.lon] for v in valid], dtype=np.float64)
        geo = create_geolocation(reader)
        pixels = geo.latlon_to_image(latlon)  # (N, 2) -> [row, col]
        return np.asarray(pixels[:, :2], dtype=np.float64)

    return None


def build_valid_mask(reader: object) -> Optional[np.ndarray]:
    """Rasterize a reader's valid-data polygon to a full-image boolean mask.

    Generic across GRDL readers; see :func:`_valid_polygon` for which
    polygon is used. Returns ``None`` if the reader exposes no valid-data
    polygon.

    Note: accumulation rasterizes the polygon tile-by-tile rather than
    materializing this full-image mask (which costs ``rows * cols`` bytes
    per worker on large imagery).

    Parameters
    ----------
    reader : ImageReader
        Any GRDL reader with populated ``.metadata`` and ``.get_shape()``.

    Returns
    -------
    np.ndarray of bool, shape (rows, cols), or None
    """
    from grdl.shapes import rasterize_polygon

    poly = _valid_polygon(reader)
    if poly is None:
        return None
    shape = reader.get_shape()
    return rasterize_polygon(poly, (int(shape[0]), int(shape[1])))


def _tile_mask(poly: np.ndarray, region: ChipRegion) -> np.ndarray:
    """Rasterize the valid-data polygon over a single tile.

    Integer translation of the vertices commutes with the scanline fill, so
    the tile interior matches the corresponding slice of the full-image mask
    -- without ever materializing ``rows * cols`` bytes per worker. Pixels
    exactly on the polygon outline may differ from :func:`build_valid_mask`
    by sub-pixel rounding where edges cross the tile window (the perimeter
    pass clips the polygon to the window before rounding); the disagreement
    is O(polygon perimeter) pixels, on the boundary where validity is
    inherently ambiguous. Tiles wholly outside the polygon's bounding box
    return all-False without rasterizing (the caller skips reading those
    tiles entirely).
    """
    from grdl.shapes import rasterize_polygon

    r0, c0, r1, c1 = (region.row_start, region.col_start,
                      region.row_end, region.col_end)
    rmin, cmin = poly.min(axis=0)
    rmax, cmax = poly.max(axis=0)
    if r1 <= rmin or r0 > rmax or c1 <= cmin or c0 > cmax:
        return np.zeros((r1 - r0, c1 - c0), dtype=bool)
    return rasterize_polygon(poly - [r0, c0], (r1 - r0, c1 - c0))


def _select(
    values: np.ndarray,
    strategy: str,
    meta_mask_tile: Optional[np.ndarray],
    raw: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the 1-D vector of valid samples for a value tile.

    ``raw`` is the untransformed chip; when given, the nonzero test runs
    against it so zero-fill pixels are excluded even when the transform maps
    them to a nonzero value (e.g. decibel's ``log(0)`` floor).
    """
    keep = np.isfinite(values)
    nonzero_src = values if raw is None else raw
    if strategy == 'none':
        pass
    elif strategy == 'nonzero_finite':
        keep &= nonzero_src != 0
    elif strategy in ('metadata', 'both'):
        if meta_mask_tile is None:
            raise ValidationError(f"strategy {strategy!r} requires a valid-data mask")
        keep &= meta_mask_tile
        if strategy == 'both':
            keep &= nonzero_src != 0
    else:
        raise ValidationError(f"unknown masking strategy: {strategy!r}")
    return values[keep]


# ---------------------------------------------------------------------------
# Accumulation over a reader (serial core, parallel worker, dispatcher)
# ---------------------------------------------------------------------------

def _accumulate_reader(
    reader: object,
    regions: List[ChipRegion],
    transform: Callable[[np.ndarray], np.ndarray],
    strategy: str,
    band: int,
    poly: Optional[np.ndarray],
    acc: StreamingStats,
) -> None:
    """Fold an open reader's tile block into ``acc`` (in place)."""
    for reg in regions:
        mtile = None
        if poly is not None:
            mtile = _tile_mask(poly, reg)
            if not mtile.any():
                continue  # no valid pixels in this tile -- skip the read
        r0, c0, r1, c1 = reg.row_start, reg.col_start, reg.row_end, reg.col_end
        chip = reader.read_chip(r0, r1, c0, c1)
        if chip.ndim == 3:
            chip = chip[..., band]
        vals = transform(chip)
        acc.update(_select(vals, strategy, mtile, raw=chip))


def _new_acc(
    percentiles: Optional[Sequence[float]],
    hist_spec: Optional[Tuple[Optional[float], Optional[float], int, str]],
) -> StreamingStats:
    if hist_spec is not None:
        low, high, n_bins, spacing = hist_spec
        return StreamingStats(
            percentiles=percentiles,
            hist_range=(low, high) if low is not None else None,
            n_bins=n_bins, hist_spacing=spacing,
        )
    return StreamingStats()


def _worker(task: tuple) -> StreamingStats:
    """Process-pool worker: open reader, accumulate an assigned tile block."""
    (path, regions, transform_name, strategy, band,
     percentiles, hist_spec) = task

    from grdl.IO.generic import open_any

    reader = open_any(path)
    transform = VALUE_TRANSFORMS[transform_name]
    poly = None
    if strategy in ('metadata', 'both'):
        poly = _valid_polygon(reader)
    acc = _new_acc(percentiles, hist_spec)
    _accumulate_reader(reader, regions, transform, strategy, band, poly, acc)
    if hasattr(reader, 'close'):
        reader.close()
    return acc


def _chunk(seq: list, n: int) -> List[list]:
    """Split ``seq`` into ``n`` near-equal contiguous chunks."""
    n = max(1, min(n, len(seq)))
    size = len(seq) / n
    return [seq[round(i * size):round((i + 1) * size)] for i in range(n)]


def _accumulate(
    path: str,
    regions: List[ChipRegion],
    transform_name: str,
    strategy: str,
    band: int,
    percentiles: Optional[Sequence[float]],
    hist_spec: Optional[Tuple[Optional[float], Optional[float], int, str]],
    use_parallel: bool,
    n_workers: int,
) -> StreamingStats:
    """One accumulation pass, serial or process-parallel, over ``path``."""
    if not use_parallel:
        from grdl.IO.generic import open_any

        reader = open_any(path)
        transform = VALUE_TRANSFORMS[transform_name]
        poly = None
        if strategy in ('metadata', 'both'):
            poly = _valid_polygon(reader)
        acc = _new_acc(percentiles, hist_spec)
        _accumulate_reader(reader, regions, transform, strategy, band,
                           poly, acc)
        if hasattr(reader, 'close'):
            reader.close()
        return acc

    # Pin BLAS/OpenMP to one thread per worker: the reductions don't use BLAS,
    # but this prevents N_workers x N_blas_threads oversubscription if any op
    # ever does. Spawned children inherit the env at import time.
    for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(var, '1')

    tasks = [
        (path, chunk, transform_name, strategy, band,
         list(percentiles or []), hist_spec)
        for chunk in _chunk(regions, n_workers)
    ]
    merged = _new_acc(percentiles, hist_spec)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for partial in pool.map(_worker, tasks):
            merged.merge(partial)
    return merged


def _resolve_path(source: Union[str, object]) -> Optional[str]:
    """Return a filesystem path for ``source`` (str or reader) or None."""
    if isinstance(source, str):
        return source
    fp = getattr(source, 'filepath', None)
    return str(fp) if fp is not None else None


def compute_image_statistics(
    source: Union[str, object],
    *,
    tile: Union[int, Tuple[int, int]] = 2048,
    transform: str = 'auto',
    mask: str = 'none',
    percentiles: Optional[Sequence[float]] = None,
    n_bins: int = 65536,
    hist_spacing: str = 'auto',
    parallel: Union[str, bool] = 'auto',
    n_workers: Optional[int] = None,
    band: int = 0,
    min_pixels: int = PARALLEL_MIN_PIXELS,
) -> StatsResult:
    """Compute exact full-image statistics over valid pixels, tile by tile.

    Mean/std/min/max are exact (Chan merge); percentiles are histogram-based.
    With the default ``'auto'`` (``'float32'``) histogram, percentiles come
    out of the same single pass as the moments -- the image is read exactly
    once. ``'log'``/``'linear'`` spacing instead sizes the histogram from the
    pass-1 data range and re-reads the image for a second pass. Accumulation
    runs serially or across processes depending on ``parallel``.

    Parameters
    ----------
    source : str or ImageReader
        Image path, or an open GRDL reader. A path is required for parallel
        execution (readers are not picklable); if a reader without a
        ``filepath`` is given, execution falls back to serial.
    tile : int or (int, int)
        Tile size for the non-overlapping partition.
    transform : str
        Value transform key from :data:`VALUE_TRANSFORMS` (``'auto'`` uses
        magnitude for complex data, passthrough for real).
    mask : {'none', 'nonzero_finite', 'metadata', 'both'}
        Valid-pixel selection. ``'metadata'``/``'both'`` rasterize the
        reader's valid-data polygon tile-by-tile; tiles wholly outside the
        polygon are skipped without reading.
    percentiles : sequence of float, optional
        Quantiles in [0, 100] to estimate.
    n_bins : int
        Histogram bins for percentile estimation. Must be a power of two for
        ``'auto'``/``'float32'`` spacing.
    hist_spacing : {'auto', 'float32', 'log', 'linear'}
        Histogram bin geometry. ``'auto'`` resolves to ``'float32'``
        (single-pass, ~0.55%% relative bin width at 65536 bins, no data-range
        pass). ``'log'``/``'linear'`` keep the two-pass data-range histogram.
    parallel : {'auto', True, False}
        ``'auto'`` parallelizes only when the tile area is at least
        ``min_pixels`` (process-pool startup is otherwise a net loss).
    n_workers : int, optional
        Process count for parallel execution. Defaults to ``os.cpu_count()``.
    band : int
        Band index for multi-band imagery.
    min_pixels : int
        Pixel-count threshold for the ``'auto'`` parallel decision.

    Returns
    -------
    StatsResult
        Finalized statistics over the selected pixels.

    Raises
    ------
    ValidationError
        If ``transform`` or ``mask`` is unknown.
    """
    if transform not in VALUE_TRANSFORMS:
        raise ValidationError(
            f"transform must be one of {tuple(VALUE_TRANSFORMS)}; got {transform!r}"
        )
    if mask not in MASK_STRATEGIES:
        raise ValidationError(
            f"mask must be one of {MASK_STRATEGIES}; got {mask!r}"
        )
    spacing = 'float32' if hist_spacing == 'auto' else hist_spacing
    if spacing not in ('float32', 'log', 'linear'):
        raise ValidationError(
            "hist_spacing must be one of ('auto', 'float32', 'log', "
            f"'linear'); got {hist_spacing!r}"
        )

    # Acquire shape (and keep a reader only for the serial / shape path).
    own_reader = None
    if isinstance(source, str):
        from grdl.IO.generic import open_any

        own_reader = open_any(source)
        reader = own_reader
    else:
        reader = source
    shape = reader.get_shape()
    rows, cols = int(shape[0]), int(shape[1])
    regions = Tiler(rows, cols, tile_size=tile).partition_positions()

    path = _resolve_path(source)
    workers = n_workers or os.cpu_count() or 1
    total = rows * cols
    if parallel == 'auto':
        use_parallel = (path is not None and workers > 1
                        and total >= min_pixels)
    else:
        use_parallel = bool(parallel) and path is not None and workers > 1

    # Parallel needs a path; release any reader we opened so workers own theirs.
    if use_parallel and own_reader is not None and hasattr(own_reader, 'close'):
        own_reader.close()
        own_reader = None

    # Serial reuses the open reader (caller's or the one we opened); parallel
    # workers extract the polygon from their own readers.
    poly = None
    if not use_parallel and mask in ('metadata', 'both'):
        poly = _valid_polygon(reader)

    def _run(hist_spec, pcts):
        if use_parallel:
            return _accumulate(path, regions, transform, mask, band,
                               pcts, hist_spec, True, workers)
        acc = _new_acc(pcts, hist_spec)
        _accumulate_reader(reader, regions, VALUE_TRANSFORMS[transform],
                           mask, band, poly, acc)
        return acc

    try:
        if percentiles and spacing == 'float32':
            # Single pass: the float32 histogram needs no data range, so the
            # exact moments and the percentile histogram share one read.
            acc = _run((None, None, n_bins, 'float32'), list(percentiles))
            return acc.result()

        # Pass 1: exact mean/std/min/max (no histogram).
        acc1 = _run(None, None)
        res = acc1.result()

        # Pass 2: percentiles, with a histogram sized from pass-1 range.
        if percentiles:
            lo, hi = res.minimum, res.maximum
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                acc2 = _run((lo, hi, n_bins, spacing), list(percentiles))
                res.percentiles = acc2.result().percentiles
        return res
    finally:
        if own_reader is not None and hasattr(own_reader, 'close'):
            own_reader.close()
