# `grdl.contrast` — Architecture

*Modified: 2026-04-26*

## Overview

`grdl.contrast` is a peer top-level module providing display-time
dynamic range adjustment for any GRDL modality. It ports the full
`sarpy.visualization.remap` catalog as concrete `ImageTransform`
subclasses and adds multi-modal stretches (gamma, sigmoid, histogram,
CLAHE) using the same paradigm.

**Design principles:**

- **No new abstractions.** Every operator is an `ImageTransform`
  subclass — same paradigm as `image_processing/intensity.py`,
  `image_processing/decomposition/`, `image_processing/detection/`.
- **No Pipeline requirement.** Operators stand alone. Users chain by
  direct call or use the existing
  `grdl.image_processing.pipeline.Pipeline` if they want.
- **No fit/transform lifecycle.** Cross-tile consistency comes from
  pre-computed stats passed as `apply()` kwargs (mirrors sarpy's
  `raw_call(data, data_mean=...)` shape).
- **Float `[0, 1]` output convention.** Composable; matplotlib reads it
  via `imshow(vmin=0, vmax=1)`. `clip_cast()` provides uint8/uint16
  quantization when needed at save time.

---

## File Structure

```
grdl/contrast/
├── __init__.py        Re-exports + module docstring
├── auto.py            auto_select(metadata) → operator name
├── base.py            clip_cast(), linear_map(), nan_safe_stats()
│
├── linear.py          LinearStretch          (sarpy Linear port)
├── logarithmic.py     LogStretch             (sarpy Logarithmic port)
├── density.py         MangisDensity, Brighter, Darker, HighContrast,
│                      GDM, PEDF              (sarpy Density family + GDM + PEDF)
├── nrl.py             NRLStretch             (sarpy NRL port)
│
├── gamma.py           GammaCorrection
├── sigmoid.py         SigmoidStretch
├── histogram.py       HistogramEqualization, CLAHE (skimage)
│
├── percentile.py      Re-export of PercentileStretch (intensity.py)
└── decibel.py         Re-export of ToDecibels        (intensity.py)
```

---

## Class Hierarchy

```
ImageTransform (ABC)                        image_processing/base.py
│
├── LinearStretch                           contrast/linear.py
├── LogStretch                              contrast/logarithmic.py
│
├── MangisDensity                           contrast/density.py
│   ├── Brighter        (preset dmin=60, mmult=40)
│   ├── Darker          (preset dmin=0,  mmult=40)
│   └── HighContrast    (preset dmin=30, mmult=4)
│
├── GDM (Generalized Density Mapping)       contrast/density.py
├── PEDF (Piecewise Extended Density)       contrast/density.py
├── NRLStretch                              contrast/nrl.py
│
├── GammaCorrection                         contrast/gamma.py
├── SigmoidStretch                          contrast/sigmoid.py
├── HistogramEqualization                   contrast/histogram.py
├── CLAHE                                   contrast/histogram.py
│
├── PercentileStretch (re-export)           image_processing/intensity.py
└── ToDecibels        (re-export)           image_processing/intensity.py
```

Every concrete class:

- `@processor_version('1.0.0')` — version stamp
- `@processor_tags(modalities=[...], category=PC.ENHANCE, description=...)`
- `Annotated` class-body fields for tunable parameters with `Range` /
  `Options` / `Desc` markers
- `apply(source, **kwargs) -> ndarray`

---

## sarpy.visualization.remap Equivalence

Tests under `tests/test_contrast_logarithmic.py`,
`tests/test_contrast_density.py`, and `tests/test_contrast_nrl.py`
include numerical regression tests that compare GRDL output against
sarpy on identical inputs (sarpy ships in the `[sar]` extra and in the
default test environment).

Tolerance: ≤ 1e-3 to 1e-5 absolute, depending on the operator.
Differences arise from:

1. **Float vs uint8 final cast.** Sarpy returns uint8 (`max_output_value
   = 255`); GRDL returns float divided by 255. The float path preserves
   precision before display.
2. **No `clip_cast` step inside `raw_call`.** Sarpy's `call()` method
   wraps `raw_call` with `clip_cast` to uint8; GRDL's `apply()`
   substitutes a final `np.clip(..., 0.0, 1.0)` to keep the result in
   the canonical `[0, 1]` range.

---

## Stats Threading (Cross-Tile Consistency)

Sarpy lets you pre-compute global stats on a representative sample and
reuse them across chips. GRDL captures this **without new state** —
stats are kwargs at `apply()` time.

| Operator | Kwarg | Sarpy equivalent |
|----------|-------|-----------------|
| `MangisDensity`, `Brighter`, `Darker`, `HighContrast`, `PEDF` | `data_mean=` | `Density.raw_call(data, data_mean=...)` |
| `GDM` | `data_mean=`, `data_median=` | `GDM.raw_call(data, data_mean=, data_median=)` |
| `LinearStretch`, `LogStretch` | `min_value=`, `max_value=` | `Linear.raw_call(data, min_value=, max_value=)` |
| `NRLStretch` | `stats=(min, max, changeover)` | `NRL.raw_call(data, stats=...)` |

When omitted, each operator computes the stat on its own input. The
class-body `Annotated` defaults (`dmin`, `mmult`, `gamma`, `knee`,
`percentile`, ...) are merged via `_resolve_params(kwargs)` from
`ImageProcessor`.

---

## Auto-Selection Dispatch

`auto.py` maps metadata class names to recommended operator names. Class
name detection (rather than `isinstance` checks) avoids importing the
concrete metadata classes here — no circular dependency, no
extras-gating.

```python
_SAR_METADATA = frozenset({
    'SICDMetadata', 'SIDDMetadata', 'CPHDMetadata', 'CRSDMetadata',
    'BIOMASSMetadata', 'Sentinel1SLCMetadata', 'TerraSARMetadata',
    'NISARMetadata',
})
_MSI_METADATA = frozenset({'Sentinel2Metadata', 'VIIRSMetadata', 'ASTERMetadata'})
_EO_METADATA  = frozenset({'EONITFMetadata'})

def auto_select(metadata) -> str:
    if metadata is None:                     return 'percentile'
    name = type(metadata).__name__
    if name in _SAR_METADATA:                return 'brighter'
    if name in _EO_METADATA:                 return 'gamma'
    if name in _MSI_METADATA:                return 'percentile'
    return 'percentile'
```

The returned string is intentionally a **slug** (not an instance) so
callers can route through their own dispatcher without coupling to a
specific operator class. The example
`grdl/example/ortho/point_roi_ortho.py` demonstrates the dispatch
pattern.

---

## NaN / Zero / Non-Finite Handling

Ported verbatim from sarpy where applicable:

```python
finite_mask = np.isfinite(amplitude)
zero_mask   = (amplitude == 0)
use_mask    = finite_mask & ~zero_mask
out[~finite_mask] = max_output_value      # saturated white
out[zero_mask]    = 0
out[use_mask]     = ...                    # operator-specific
```

`HistogramEqualization` preserves NaN in the output. `CLAHE`
normalizes finite samples to `[0, 1]` before delegating to
`skimage.exposure.equalize_adapthist`.

---

## Output Convention

`apply()` always returns `float32` in `[0, 1]`. Two reasons:

1. **Composability.** Operators chain without intermediate casts; the
   output of one is a valid input to the next.
2. **matplotlib-friendliness.** `imshow(arr, vmin=0, vmax=1)` is the
   simplest path to a correct display.

For uint8/uint16 saving, callers use:

```python
from grdl.contrast import clip_cast
out_uint8 = clip_cast(stretched, dtype='uint8', max_value=255)
```

---

## Discovery & Catalog Integration

Each operator carries `__processor_tags__` so downstream tooling
(`grdl-runtime`, `grdk`) can filter/search:

```python
from grdl.contrast import MangisDensity
MangisDensity.__processor_tags__
# {'modalities': (<ImageModality.SAR>,),
#  'category': <ProcessorCategory.ENHANCE>,
#  'description': 'Mangis 1994 logarithmic density remap (sarpy Density).',
#  ...}
```

`Brighter`, `Darker`, `HighContrast` each carry their own tags
(separate decorators on the subclass) so they appear independently in
discovery — they aren't hidden behind `MangisDensity` despite being
subclasses.

---

## Dependencies

| Operator | Required | Optional |
|----------|----------|----------|
| All except `CLAHE` | `numpy`, `scipy` (transitive via `image_processing.base`) | — |
| `CLAHE` | + `scikit-image>=0.19.0` | install with `pip install grdl[contrast]` |

`auto_select`, `clip_cast`, `linear_map`, `nan_safe_stats` are pure
numpy.

---

## Testing

Per-operator tests live in `tests/test_contrast_*.py`. Coverage:

- Identity / known-input output on a small synthetic array
- NaN / zero / non-finite handling
- Per-call kwargs vs instance default precedence
- For SAR ports: numerical regression against `sarpy.visualization.remap`
  on identical inputs

```bash
pytest tests/test_contrast_*.py -v          # all 65 tests
pytest tests/test_contrast_density.py -v    # MangisDensity + presets + GDM + PEDF
pytest tests/test_contrast_nrl.py -v        # NRLStretch + sarpy regression
```

---

## Adding a New Operator

1. Create `grdl/contrast/<operator>.py` with the standard file header
   (encoding, docstring with title, dependencies, author, license,
   created/modified dates).
2. Inherit `ImageTransform` from `grdl.image_processing.base`.
3. Add `@processor_version('1.0.0')` and `@processor_tags(...)`.
4. Declare tunable parameters as `Annotated` class-body fields with
   `Range`, `Options`, `Desc` markers from
   `grdl.image_processing.params`.
5. Implement `apply(source, **kwargs)` returning `float32` in `[0, 1]`.
   Handle complex input by taking magnitude. Use `_resolve_params(kwargs)`
   to merge instance defaults with runtime overrides.
6. Re-export from `grdl/contrast/__init__.py`.
7. Add tests in `tests/test_contrast_<operator>.py`.
8. Reference the new operator in [README.md](README.md) and update
   `auto.py` if it should be the new default for any modality.
