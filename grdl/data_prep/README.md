# Data Preparation Module

Index-only chip/tile planning, intensity normalization, and full-image statistics for ML/AI pipelines. `ChipExtractor` and `Tiler` compute *where* chips and tiles fall within an image, returning `ChipRegion` named tuples with clipped pixel bounds -- they never touch pixel data. `Normalizer` handles per-chip or per-image intensity scaling with fit/transform semantics, including a memory-bounded streaming fit over full images. `compute_image_statistics` and `StreamingStats` compute exact, valid-masked, optionally parallel full-image statistics tile by tile.

---

## Quick Start

```python
import numpy as np
from grdl.data_prep import ChipExtractor, Tiler, Normalizer

image = np.random.rand(1000, 2000)

# Extract a 64x64 chip centered at (500, 1000)
ext = ChipExtractor(nrows=1000, ncols=2000)
region = ext.chip_at_point(500, 1000, row_width=64, col_width=64)
chip = image[region.row_start:region.row_end, region.col_start:region.col_end]

# Normalize the chip to [0, 1]
norm = Normalizer(method='minmax')
chip_normalized = norm.normalize(chip)
```

---

## ChipRegion

All chip/tile methods return `ChipRegion` named tuples. Use them directly for numpy slicing:

```python
from grdl.data_prep import ChipRegion

region = ChipRegion(row_start=100, col_start=200, row_end=164, col_end=264)
chip = image[region.row_start:region.row_end, region.col_start:region.col_end]
```

Fields, in tuple order: `row_start`, `col_start` (inclusive), `row_end`, `col_end` (exclusive). All values are guaranteed within image bounds (`row_start >= 0`, `row_end <= nrows`, etc.), so the slice never raises or wraps. Because it is a `NamedTuple`, it also unpacks positionally:

```python
r0, c0, r1, c1 = region
```

---

## ChipBase (ABC)

`ChipExtractor` and `Tiler` both inherit `ChipBase`, which manages image dimensions and provides inward snapping. You rarely instantiate it directly, but its contract defines the shared behavior.

```python
base.nrows     # number of image rows
base.ncols     # number of image columns
base.shape     # (nrows, ncols)
```

Construction validates dimensions: `nrows`/`ncols` must be positive `int` (else `TypeError`/`ValueError`).

The protected `_snap_region(row_start, col_start, row_width, col_width)` is the core primitive. It slides a requested window **inward** so it fits entirely within `[0, nrows] x [0, ncols]` while preserving the requested size. If the requested width exceeds the image dimension, it clamps to the full extent in that dimension:

```text
500-row image, 100-tall chip near the top edge:
  _snap_region(-25, 0, 100, 100) -> ChipRegion(0,   0, 100, 100)
500-row image, 100-tall chip near the bottom edge:
  _snap_region(425, 0, 100, 100) -> ChipRegion(400, 0, 500, 100)
```

This inward-snap (rather than clipping to a smaller chip) is why edge chips keep their full requested size, at the cost of overlapping their neighbors at the boundary.

---

## ChipExtractor

Point-centered chip extraction and whole-image partitioning.

```python
ext = ChipExtractor(nrows, ncols)
```

### chip_at_point -- extract around a location

```python
region = ext.chip_at_point(row, col, row_width, col_width)
```

The chip is centered on `(row, col)` (rounded to nearest int). Near-edge points snap inward to keep the full requested size. Accepts scalar **or** array-like centers; returns a single `ChipRegion` for scalar input and a `list[ChipRegion]` for array/list input.

```python
ext = ChipExtractor(nrows=1000, ncols=2000)

# Single point -> one ChipRegion
region = ext.chip_at_point(500, 1000, row_width=128, col_width=128)

# Multiple points -> list of ChipRegion
regions = ext.chip_at_point([100, 500, 900], [200, 1000, 1800],
                            row_width=64, col_width=64)

# Near-edge point snaps inward to maintain full chip size
edge_region = ext.chip_at_point(10, 10, row_width=64, col_width=64)
# -> ChipRegion(row_start=0, col_start=0, row_end=64, col_end=64)
```

Raises `TypeError` if `row_width`/`col_width` is not `int`, and `ValueError` if they are not positive or if any center point lies outside `[0, nrows)` x `[0, ncols)`.

### chip_positions -- partition the full image

```python
regions = ext.chip_positions(row_width=256, col_width=256)
```

Lays out chips row-major from the top-left. Edge chips snap inward to maintain full size (so they may overlap their neighbors at the boundary). When the image is smaller than the requested chip, a single full-image chip is returned.

---

## Tiler

Stride-based tile-grid computation. Construction:

```python
from grdl.data_prep import Tiler

Tiler(nrows, ncols, tile_size, stride=None)
```

- `tile_size`: `int` (square) or `(tile_rows, tile_cols)`.
- `stride`: `int` or `(stride_rows, stride_cols)`. Defaults to `tile_size` (no overlap). **Must not exceed `tile_size`** in either dimension (else `ValueError`).

Properties: `tiler.tile_size`, `tiler.stride` (both `(rows, cols)` tuples), plus the inherited `nrows`/`ncols`/`shape`.

### `tile_positions()` vs `partition_positions()` -- read this

`Tiler` exposes two layout methods with **fundamentally different guarantees**. Choosing the wrong one silently corrupts per-pixel results.

| Method | Tiles overlap? | Edge handling | Covers each pixel | Use for |
|--------|----------------|---------------|-------------------|---------|
| `tile_positions()` | Yes, when `stride < tile_size`; edge tiles overlap regardless | Snap **inward** to keep constant `tile_size` | **One or more** times | Sliding-window inference, feature extraction, anything where a constant tile shape matters |
| `partition_positions()` | **Never** | Clip edge tiles **smaller** than `tile_size` | **Exactly once** | Per-pixel aggregation: statistics, histograms, masks, reductions -- anywhere double-counting edge pixels is wrong |

**`tile_positions()`** -- overlapping coverage, constant tile size:

```python
tiler = Tiler(nrows=100, ncols=100, tile_size=64)   # stride defaults to 64
regions = tiler.tile_positions()
# len == 4; the last tile is snapped inward to stay 64x64:
#   ChipRegion(row_start=36, col_start=36, row_end=100, col_end=100)
# -> the 36..64 band is covered twice. Every tile is exactly 64x64.

# 50% overlap (stride = tile_size / 2)
overlap = Tiler(1000, 2000, tile_size=256, stride=128).tile_positions()

# Rectangular tiles and strides
rect = Tiler(1000, 2000, tile_size=(256, 512), stride=(128, 256)).tile_positions()
```

**`partition_positions()`** -- strict non-overlapping partition (`stride` is ignored):

```python
tiler = Tiler(nrows=100, ncols=100, tile_size=64)
regions = tiler.partition_positions()
# len == 4; the last tile is CLIPPED (smaller), not snapped:
#   ChipRegion(row_start=64, col_start=64, row_end=100, col_end=100)  # 36x36
# -> the union is exactly the full image, every pixel counted once.
```

Because `partition_positions()` covers every pixel exactly once, it is the correct tiler for `compute_image_statistics` and any reduction. `tile_positions()` would double-count the snapped-inward edge band.

### When to use Tiler vs ChipExtractor

| Task | Use |
|------|-----|
| Extract chips at specific detection / point locations | `ChipExtractor.chip_at_point` |
| Constant-size chip grid (edge chips may overlap) | `ChipExtractor.chip_positions` or `Tiler.tile_positions` |
| Sliding window with explicit overlap for inference | `Tiler(..., stride=<small>).tile_positions()` |
| Non-overlapping exact partition for per-pixel stats | `Tiler.partition_positions()` |

---

## Normalizer

Four normalization methods with stateless and fit/transform modes.

```python
Normalizer(method='minmax', percentile_low=2.0, percentile_high=98.0, epsilon=1e-10)
```

| Method | Output range | Description |
|--------|-------------|-------------|
| `'minmax'` | [0, 1] | Scale by `(x - min) / (max - min)` |
| `'zscore'` | unbounded | `(x - mean) / std` |
| `'percentile'` | [0, 1] | Clip to `[percentile_low, percentile_high]`, then scale to [0, 1] |
| `'unit_norm'` | unbounded | Divide by the array's L2 norm |
| `'mad'` | unbounded | Robust z-score: `(x - median) / (1.4826 * MAD)`, where `MAD = median(|x - median(x)|)` |

The `'mad'` method is the outlier-resistant counterpart of `'zscore'`: the median and the Median Absolute Deviation each have a 50% breakdown point, so a handful of extreme pixels (speckle spikes, hot pixels, saturated returns) barely move the center or scale. The `1.4826` constant (`1 / Phi^-1(0.75)`, exported as `grdl.data_prep.MAD_TO_STD`) makes `1.4826 * MAD` converge to the standard deviation for normally distributed data, so a MAD z-score is directly comparable to an ordinary z-score.

`epsilon` guards against divide-by-zero: when the relevant spread (range / std / percentile range / L2 norm / scaled MAD) falls below `epsilon`, the method returns an all-zeros array. All methods return `float64` regardless of input dtype. Invalid `method` or percentile bounds raise `ValueError`. Inputs must be `np.ndarray` (else `TypeError`).

Read-only properties: `norm.method` and `norm.is_fitted`.

### Stateless (per-chip)

`normalize()` computes statistics from the input and applies them immediately -- nothing is stored. Works on any-shape arrays.

```python
norm = Normalizer(method='minmax')
normalized = norm.normalize(chip)
```

### Fit / transform (reusable statistics)

`fit()` computes and stores `min`, `max`, `mean`, `std`, the percentile values, and the L2 norm from a reference array; `transform()` then applies those stored parameters to new data (e.g., normalize a test set with training statistics). `transform()` before `fit()` raises `ProcessorError`.

```python
norm = Normalizer(method='zscore')
norm.fit(training_data)                       # store mean/std from the reference
test_normalized = norm.transform(test_data)   # apply the same statistics

normalized = norm.fit_transform(data)         # fit then transform in one call
```

### Percentile clipping

```python
norm = Normalizer(method='percentile', percentile_low=2.0, percentile_high=98.0)
normalized = norm.normalize(image)            # clip outliers, then scale to [0, 1]
```

### Streaming fit over a full image

`fit_streaming()` builds the same fitted parameters as `fit()` but streams the image tile by tile through `compute_image_statistics` -- never holding the whole image in memory, and optionally fanning tiles across CPU cores. Use it to derive a normalization baseline from imagery too large (or too slow) to load whole.

```python
norm = Normalizer(method='zscore')
norm.fit_streaming('scene.nitf', mask='nonzero_finite')   # exact mean/std, masked
chip_norm = norm.transform(chip)
```

Full signature:

```python
norm.fit_streaming(
    source,                 # image path (str) or an open GRDL reader
    *,
    tile=2048,              # tile size for the non-overlapping partition
    transform='auto',       # 'auto'|'magnitude'|'power'|'decibel'|'identity'
    mask='none',            # 'none'|'nonzero_finite'|'metadata'|'both'
    n_bins=65536,           # histogram bins (only for method='percentile')
    hist_spacing='auto',    # 'auto'|'float32'|'log'|'linear'
    parallel='auto',        # 'auto'|True|False
    n_workers=None,         # process count for parallel runs
    band=0,                 # band index for multi-band imagery
)
```

- `transform='auto'` takes the magnitude of complex imagery and passes real imagery through.
- For SAR imagery with zero fill, prefer `mask='nonzero_finite'` or `mask='metadata'` (the sensor valid-data polygon) so the baseline isn't dominated by fill pixels.
- The histogram (`n_bins`, `hist_spacing`) is only consulted for `method='percentile'`; min/max/mean/std are always exact.
- A path enables parallel execution (open readers are not picklable).
- Raises `ProcessorError` if no valid pixels remain after masking.

---

## Full-Image Statistics Baseline

`compute_image_statistics` is the purpose-built way to get a mean/std/percentile baseline over a whole image without loading it into memory. It reads the image as a **non-overlapping partition** (`Tiler.partition_positions`), so every valid pixel is counted exactly once, accumulates exact moments via Chan's parallel merge, estimates percentiles from a fixed-bin histogram, and -- for large imagery -- fans tiles out across processes.

```python
from grdl.data_prep import compute_image_statistics

stats = compute_image_statistics('scene.nitf', mask='metadata', percentiles=[1, 99])
print(stats.summary())
print(stats.mean, stats.std, stats.minimum, stats.maximum)
print(stats.percentiles[1], stats.percentiles[99])
print(stats.l2_norm)
```

Full signature:

```python
compute_image_statistics(
    source,                 # image path (str) or an open GRDL reader
    *,
    tile=2048,              # int or (rows, cols) partition tile size
    transform='auto',       # value transform: 'auto'|'magnitude'|'power'|'decibel'|'identity'
    mask='none',            # valid-pixel selection (see below)
    percentiles=None,       # sequence of quantiles in [0, 100], or None
    n_bins=65536,           # histogram bins (power of two for 'float32' spacing)
    hist_spacing='auto',    # 'auto'|'float32'|'log'|'linear'
    parallel='auto',        # 'auto'|True|False
    n_workers=None,         # defaults to os.cpu_count()
    band=0,                 # band index for multi-band imagery
    min_pixels=PARALLEL_MIN_PIXELS,   # threshold for the 'auto' parallel decision
    mad=False,              # also compute the median and Median Absolute Deviation
) -> StatsResult
```

Set `mad=True` to also populate `median` and `mad` (the robust spread). The MAD is taken about the median, so it needs the median first: this costs **one extra read** of the image (a deviation pass over `|x - median|`, binned in a non-negative single-pass float32 histogram). The median is computed even when no value percentiles are requested, and is *not* added to `percentiles` unless the caller explicitly asks for `50.0`.

### StatsResult

```python
@dataclass
class StatsResult:
    count: int                       # number of valid samples accumulated
    mean: float
    var: float                       # population variance
    std: float
    minimum: float
    maximum: float
    percentiles: dict[float, float]  # {quantile -> value}, empty if none requested
    median: float                    # 50th percentile; nan unless mad=True
    mad: float                       # median(|x - median|); nan unless mad=True

    @property
    def l2_norm(self) -> float       # exact sqrt(sum(x**2)) derived from count/mean/var

    @property
    def mad_std(self) -> float       # 1.4826 * mad (std-consistent robust spread)

    def summary(self) -> str         # one-line human-readable summary
```

`mean`/`var`/`std`/`minimum`/`maximum` are **exact**; percentiles, `median`, and `mad` are histogram-interpolated. When no valid pixels are found, `count == 0` and the float fields are `nan`. `median`/`mad` stay `nan` unless `mad=True` was passed.

### Value transforms

`transform` maps each (possibly complex) tile to the real scalar that statistics run on:

| Value | Effect |
|-------|--------|
| `'auto'` | Magnitude `\|z\|` for complex input; passthrough for real (the usual choice) |
| `'magnitude'` | `\|z\|` |
| `'power'` | `\|z\|^2` |
| `'decibel'` | `20*log10(\|z\|)` with a small floor |
| `'identity'` | Passthrough (requires real input) |

### Valid-pixel selection (`mask`)

NaN/inf are always excluded. The strategy adds further selection:

| Mode | Keeps |
|------|-------|
| `'none'` | Every finite pixel |
| `'nonzero_finite'` | Drops zeros and non-finite (the zero test runs on the **raw** pixels, before the value transform -- so decibel's `log(0)` floor doesn't sneak fill pixels back in) |
| `'metadata'` | Pixels inside the reader's valid-data polygon (rasterized tile-by-tile; tiles wholly outside it are skipped without reading) |
| `'both'` | `'metadata'` AND `'nonzero_finite'` |

`'metadata'`/`'both'` need a reader exposing a valid-data polygon; `build_valid_mask(reader)` returns the full-image boolean mask (or `None`) if you want it directly.

### Histogram spacing and read passes

| `hist_spacing` | Passes over the image | Notes |
|----------------|-----------------------|-------|
| `'auto'` / `'float32'` (default) | **1** | Bins by the top bits of the IEEE-754 total-order key -- fixed, data-independent geometry (~0.55% relative bin width at 65536 bins). Percentiles come out of the same single read as the moments. `n_bins` must be a power of two in `[256, 2**32]`. |
| `'log'` / `'linear'` | 2 | Sizes the histogram from a first-pass data range, then re-reads for the percentile pass. |

If no `percentiles` are requested, only the (single) moments pass runs.

### Parallelism

With `parallel='auto'`, the job runs across processes only when the tile area is at least `min_pixels` (`PARALLEL_MIN_PIXELS = 500_000_000`) -- below that, process-pool startup is a net loss and it runs serially. Parallel execution requires a filesystem path (readers are not picklable); passing an open reader without a `filepath` falls back to serial. `parallel=True`/`False` forces the decision.

### Process-parallel building block: StreamingStats

`StreamingStats` is the online accumulator underneath everything above. It is **associative** -- independent instances over disjoint pixel subsets merge into the same result as a single pass -- which is what makes process-parallel accumulation exact.

```python
from grdl.data_prep import StreamingStats

acc = StreamingStats(percentiles=[1, 99], n_bins=65536, hist_spacing='float32')
acc.update(tile_a.ravel())       # fold a batch of real samples (non-finite dropped)
acc.update(tile_b.ravel())

other = StreamingStats(percentiles=[1, 99])
other.update(tile_c.ravel())
acc.merge(other)                 # combine a worker's partial result

result = acc.result()            # -> StatsResult
```

For `'log'`/`'linear'` spacing, pass `hist_range=(low, high)`; `'float32'` spacing needs no range. `update()` accepts any-shape input (flattened internally) and defensively drops NaN/inf, so they never corrupt the running mean/var/min/max.
