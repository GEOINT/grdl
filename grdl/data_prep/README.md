# Data Preparation Module

Index-only chip/tile planning and intensity normalization for ML/AI pipelines. `ChipExtractor` and `Tiler` compute where chips and tiles fall within an image, returning `ChipRegion` named tuples with clipped pixel bounds -- they never touch pixel data. `Normalizer` handles per-chip or per-image intensity scaling with fit/transform semantics.

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

Fields: `row_start`, `col_start` (inclusive), `row_end`, `col_end` (exclusive). All values are guaranteed within image bounds.

---

## ChipExtractor

Point-centered chip extraction and whole-image partitioning.

### chip_at_point -- extract around a location

```python
ext = ChipExtractor(nrows=1000, ncols=2000)

# Single point (returns one ChipRegion)
region = ext.chip_at_point(500, 1000, row_width=128, col_width=128)

# Multiple points (returns list of ChipRegion)
regions = ext.chip_at_point([100, 500, 900], [200, 1000, 1800],
                            row_width=64, col_width=64)

# Near-edge points snap inward to maintain full chip size
edge_region = ext.chip_at_point(10, 10, row_width=64, col_width=64)
# row_start=0, col_start=0 (snapped, still 64x64)
```

### chip_positions -- partition the full image

```python
regions = ext.chip_positions(row_width=256, col_width=256)
# Returns non-overlapping chips covering the image.
# Edge chips snap inward to maintain full size (may overlap neighbors).
```

---

## Tiler

Stride-based overlapping tile grid computation.

```python
from grdl.data_prep import Tiler

# Non-overlapping tiles (stride = tile_size)
tiler = Tiler(nrows=1000, ncols=2000, tile_size=256)
regions = tiler.tile_positions()

# 50% overlapping tiles
tiler = Tiler(nrows=1000, ncols=2000, tile_size=256, stride=128)
regions = tiler.tile_positions()

# Rectangular tiles
tiler = Tiler(nrows=1000, ncols=2000, tile_size=(256, 512), stride=(128, 256))
regions = tiler.tile_positions()
```

Tiles are laid out row-major from top-left. Edge tiles snap inward to maintain full tile size.

### When to use Tiler vs ChipExtractor

| Task | Use |
|------|-----|
| Extract chips at specific detection locations | `ChipExtractor.chip_at_point` |
| Partition image into non-overlapping blocks | `ChipExtractor.chip_positions` |
| Sliding-window with overlap for inference | `Tiler` |

---

## Normalizer

Four normalization methods with stateless and fit/transform modes.

| Method | Output range | Description |
|--------|-------------|-------------|
| `'minmax'` | [0, 1] | Scale by min/max |
| `'zscore'` | unbounded | Subtract mean, divide by std |
| `'percentile'` | [0, 1] | Clip to percentile range, then scale |
| `'unit_norm'` | unbounded | Divide by L2 norm |

### Stateless (per-chip)

```python
norm = Normalizer(method='minmax')
normalized = norm.normalize(chip)
```

### Fit/Transform (reusable statistics)

```python
norm = Normalizer(method='zscore')
norm.fit(training_data)                   # compute mean/std from training set
test_normalized = norm.transform(test_data)  # apply same statistics

# Or in one step
normalized = norm.fit_transform(data)
```

### Percentile clipping

```python
norm = Normalizer(method='percentile', percentile_low=2.0, percentile_high=98.0)
normalized = norm.normalize(image)  # clips outliers, then scales to [0, 1]
```
