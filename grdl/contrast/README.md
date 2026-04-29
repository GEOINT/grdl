# `grdl.contrast` — Display Contrast Operators

Multi-modal view-time contrast adjustment for SAR, EO, MSI, PAN, and HSI
imagery. Wraps the full `sarpy.visualization.remap` SAR catalog plus
generic stretches (linear, percentile, gamma, sigmoid, histogram, CLAHE)
behind the standard GRDL `ImageTransform` API.

All operators:

- Inherit `ImageTransform` and use `@processor_version` /
  `@processor_tags` metadata for runtime catalog discovery.
- Accept complex or real input — complex inputs have their magnitude
  taken automatically.
- Return `np.ndarray` of dtype `float32` in `[0, 1]` — directly
  consumable by `matplotlib.pyplot.imshow(cmap=...)`.
- Support cross-tile consistency via per-call kwargs (`data_mean`,
  `min_value`/`max_value`, `stats=(min, max, changeover)`).

---

## Quick Start

```python
import numpy as np
from grdl.contrast import (
    auto_select,
    MangisDensity, NRLStretch, Brighter,
    LinearStretch, PercentileStretch,
    GammaCorrection, SigmoidStretch,
    HistogramEqualization, CLAHE,
)

# 1) Auto-select an operator from reader metadata
operator_name = auto_select(reader.metadata)
# SAR → 'brighter', EO NITF → 'gamma', MSI/unknown → 'percentile'

# 2) Apply directly — no Pipeline required
sar_disp = NRLStretch().apply(sar_amplitude)          # complex or real
eo_disp  = PercentileStretch(plow=2, phigh=98).apply(eo_image)
mid_disp = GammaCorrection(gamma=1.4).apply(eo_disp)  # chain manually

# 3) Cross-tile consistency — pre-compute stats once, reuse across chips
mean = float(np.mean(np.abs(full_image)))
remap = MangisDensity()
chip_a_disp = remap.apply(chip_a, data_mean=mean)
chip_b_disp = remap.apply(chip_b, data_mean=mean)     # same brightness
```

---

## Operator Catalog

### SAR — sarpy.visualization.remap ports

| Class | Origin | Notes |
|-------|--------|-------|
| `MangisDensity` | `Density` | Kevin Mangis 1994 log-density. `dmin` (range floor), `mmult` (contrast). Pre-compute `data_mean` for cross-tile consistency. |
| `Brighter` | `Brighter` | Preset: `dmin=60`, `mmult=40`. The default for SAR via `auto_select`. |
| `Darker` | `Darker` | Preset: `dmin=0`, `mmult=40`. |
| `HighContrast` | `High_Contrast` | Preset: `dmin=30`, `mmult=4` (lower mmult = harder contrast). |
| `GDM` | `GDM` | Generalized Density Mapping with graze/slope-aware cutoffs. Requires `graze_deg`, `slope_deg`. |
| `PEDF` | `PEDF` | Density with the top half compressed by 0.5× — preserves bright detail. |
| `NRLStretch` | `NRL` | Linear up to `changeover` (typically the 99th percentile), then `log2` to max. Knee defaults to 80% of output range. |
| `LogStretch` | `Logarithmic` | Bounded `log2((clip(x, min, max) − min) / (max − min) + 1)`. |
| `ToDecibels` (re-export) | — | `20·log10(\|x\| + ε)` clamped to `floor_db`. |

### Generic — works on any modality

| Class | Notes |
|-------|-------|
| `LinearStretch` | `(x − min) / (max − min)`, clipped to `[0, 1]`. Non-finite samples saturate to `1.0`. |
| `PercentileStretch` (re-export) | 2/99 (configurable) percentile clip. |
| `GammaCorrection` | `x^(1/gamma)` — expects input already in `[0, 1]`. `gamma > 1` brightens, `< 1` darkens. |
| `SigmoidStretch` | Logistic S-curve, rescaled so `x=0 → y=0` and `x=1 → y=1`. |
| `HistogramEqualization` | Global CDF remap. NaN-aware. |
| `CLAHE` | Contrast-Limited Adaptive Histogram Equalization (scikit-image). 2D-only. |

### Skipped from sarpy

`LUT8bit` (matplotlib `cmap=` covers it) and the `_register_defaults` /
named-registry pattern (YAGNI for v1).

---

## Auto-Selection by Modality

```python
from grdl.contrast import auto_select

# Maps metadata class name → recommended operator name.
auto_select(sicd_meta)     # → 'brighter' (SAR default)
auto_select(eo_nitf_meta)  # → 'gamma'
auto_select(s2_meta)       # → 'percentile'
auto_select(None)          # → 'percentile' (universal safe default)
```

| Metadata class | Returns | Why |
|---|---|---|
| `SICDMetadata`, `SIDDMetadata`, `CPHDMetadata`, `CRSDMetadata`, `BIOMASSMetadata`, `Sentinel1SLCMetadata`, `TerraSARMetadata`, `NISARMetadata` | `'brighter'` | Mangis preset (`dmin=60`, `mmult=40`) — bright SAR display without per-image tuning |
| `EONITFMetadata` | `'gamma'` | Pooled 2/99 percentile + gamma=1.4 — typical 8-bit optical |
| `Sentinel2Metadata`, `VIIRSMetadata`, `ASTERMetadata` | `'percentile'` | Per-band 2/99 clip — predictable across bands |
| Unknown / `None` | `'percentile'` | Universal safe default |

`auto_select` is dispatched at module level with no extras-gated imports —
it runs even when SAR or EO extras aren't installed.

---

## Cross-Tile Consistency

Sarpy supports pre-computing stats on a representative sample and
reusing them across chips so brightness stays uniform. GRDL captures
this as **kwargs at `apply()` time**, not as fit/transform state:

| Operator | Stat kwargs |
|----------|-------------|
| `MangisDensity`, `Brighter`, `Darker`, `HighContrast`, `PEDF` | `data_mean=` |
| `GDM` | `data_mean=`, `data_median=` |
| `LinearStretch`, `LogStretch` | `min_value=`, `max_value=` |
| `NRLStretch` | `stats=(min, max, changeover)` |

When omitted, each operator computes the stat on the input array (good
for one-shots; less consistent across tiles).

---

## Output Convention

`apply()` returns `float32` in `[0, 1]` — composable, matplotlib-ready,
and lossless for the common `imshow(vmin=0, vmax=1)` path. For uint8 /
uint16 quantization at save time, use `clip_cast()` from
`grdl.contrast.base`:

```python
from grdl.contrast import clip_cast

uint8_out = clip_cast(stretched, dtype='uint8', max_value=255)
```

---

## Composition (no Pipeline required)

Operators are independent. Chain them by direct calls:

```python
norm    = LinearStretch(min_value=0, max_value=1).apply(stretched)
bright  = GammaCorrection(gamma=1.4).apply(norm)
detail  = CLAHE(kernel_size=64, clip_limit=0.02).apply(bright)
```

Or use the existing `grdl.image_processing.pipeline.Pipeline` if you
prefer a declarative chain. `grdl.contrast` does not depend on it.

---

## Worked Example — `point_roi_ortho.py`

`grdl/example/ortho/point_roi_ortho.py` ships a YAML-configured demo
that pools shared stats across the source chip and the orthorectified
result, then applies an auto-selected operator so both display panels
show identical dynamic range:

```yaml
# point_roi_ortho.yaml
display_contrast: auto       # auto | percentile | linear | log
                             # | mangis | brighter | darker | high_contrast
                             # | nrl | gamma | histogram | clahe
```

The script:

1. Calls `auto_select(reader.metadata)` to resolve the operator.
2. Pools the finite samples from both panels.
3. Threads pooled stats through the operator's correct kwargs.
4. Renders both panels at `vmin=0, vmax=1` for visual consistency.

---

## Installation

CLAHE requires scikit-image; everything else is numpy-only:

```bash
pip install grdl[contrast]   # numpy + scipy + scikit-image
```

---

## Architecture & Design

For class hierarchy, paradigm details, sarpy-port equivalence, and the
per-call stats threading model, see
[ARCHITECTURE.md](ARCHITECTURE.md).
