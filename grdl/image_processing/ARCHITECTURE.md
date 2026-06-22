# Image Processing Module — Architecture

*Modified: 2026-03-10*

## Overview

The `image_processing` module provides all raster-to-raster transforms,
raster-to-vector detections, and spectral decompositions in GRDL. Every
processor inherits from `ImageProcessor`, which provides version
tracking, tunable parameter validation, and a uniform `execute()`
protocol for grdl-runtime integration.

---

## Directory Layout

```
image_processing/
├── __init__.py              # Top-level re-exports
├── base.py                  # ImageProcessor, ImageTransform, BandwiseTransformMixin
├── params.py                # Range, Options, Desc, ParamSpec
├── versioning.py            # @processor_version, @processor_tags, @globalprocessor
├── intensity.py             # ToDecibels, PercentileStretch
├── pipeline.py              # Pipeline (sequential transform composition)
│
├── filters/                 # Spatial image filters
│   ├── linear.py            #   MeanFilter, GaussianFilter
│   ├── rank.py              #   MedianFilter, MinFilter, MaxFilter
│   ├── statistical.py       #   StdDevFilter
│   ├── speckle.py           #   LeeFilter, ComplexLeeFilter
│   └── phase.py             #   PhaseGradientFilter
│
├── ortho/                   # Geometric correction / orthorectification
│   ├── ortho.py             #   OutputGridProtocol, validate_sub_grid_indices,
│   │                        #   GeographicGrid, Orthorectifier
│   ├── enu_grid.py          #   ENUGrid (satisfies OutputGridProtocol)
│   ├── utm_grid.py          #   UTMGrid (satisfies OutputGridProtocol)
│   ├── web_mercator_grid.py #   WebMercatorGrid (satisfies OutputGridProtocol)
│   ├── ortho_builder.py     #   orthorectify(), OrthoBuilder, OrthoResult (builder + ROI + tiling)
│   ├── roi.py               #   orthorectify_point_roi(), PointRoiResult (point-centered ortho)
│   ├── accelerated.py       #   resample(), detect_backend() (multi-backend dispatch)
│   └── resolution.py        #   compute_output_resolution (SICD, BIOMASS dispatch)
│
├── decomposition/           # Polarimetric decompositions
│   ├── base.py              #   PolarimetricDecomposition (ABC)
│   ├── pauli.py             #   PauliDecomposition (quad-pol)
│   ├── pol_matrix.py        #   Scattering/coherency matrix helpers
│   └── dual_pol_halpha.py   #   DualPolHAlpha (dual-pol)
│
├── detection/               # Sparse vector detectors
│   ├── base.py              #   ImageDetector (ABC)
│   ├── models.py            #   Detection, DetectionSet
│   ├── fields.py            #   FieldDefinition, Fields, DATA_DICTIONARY
│   └── cfar/                #   CFAR detector family
│       ├── _base.py         #     CFARDetector (template method)
│       ├── ca_cfar.py       #     CACFARDetector
│       ├── go_cfar.py       #     GOCFARDetector
│       ├── so_cfar.py       #     SOCFARDetector
│       └── os_cfar.py       #     OSCFARDetector
│
└── sar/                     # SAR-specific (requires SICDMetadata)
    ├── sublook.py           #   SublookDecomposition (1D)
    ├── multilook.py         #   MultilookDecomposition (2D, uses Tiler)
    └── image_formation/     #   Phase-history to image algorithms
        ├── base.py          #     ImageFormationAlgorithm (ABC)
        ├── geometry.py      #     CollectionGeometry
        ├── polar_grid.py    #     PolarGrid
        ├── pfa.py           #     PolarFormatAlgorithm
        ├── stripmap_pfa.py  #     StripmapPFA
        ├── rda.py           #     RangeDopplerAlgorithm
        ├── ffbp.py          #     FastBackProjection
        └── subaperture.py   #     SubaperturePartitioner
```

---

## Class Hierarchy

```
ImageProcessor (ABC)
│
├── ImageTransform (ABC) ── dense raster → raster
│   ├── BandwiseTransformMixin ── auto-handles 3D band stacks
│   │   ├── MeanFilter          [FILTERS]
│   │   ├── GaussianFilter      [FILTERS]
│   │   ├── MedianFilter        [FILTERS]
│   │   ├── MinFilter           [FILTERS]
│   │   ├── MaxFilter           [FILTERS]
│   │   ├── StdDevFilter        [FILTERS]
│   │   ├── LeeFilter           [FILTERS, SAR]
│   │   ├── ComplexLeeFilter    [FILTERS, SAR]
│   │   └── PhaseGradientFilter [FILTERS, SAR]
│   ├── ToDecibels              [ENHANCE]
│   ├── PercentileStretch       [ENHANCE]
│   ├── Orthorectifier          [GEOM_CORRECT]
│   └── Pipeline                (sequential composition)
│
├── OrthoBuilder ── builder orchestrator (ROI, tiling, auto-resolution)
│   └── OrthoResult ── output container (data, grid, geo metadata)
│
├── ImageDetector (ABC) ── raster → DetectionSet
│   └── CFARDetector (ABC, template method)
│       ├── CACFARDetector      [FIND_MAXIMA, SAR]
│       ├── GOCFARDetector      [FIND_MAXIMA, SAR]
│       ├── SOCFARDetector      [FIND_MAXIMA, SAR]
│       └── OSCFARDetector      [FIND_MAXIMA, SAR]
│
├── PolarimetricDecomposition (ABC) ── scattering matrix → components
│   ├── PauliDecomposition      (quad-pol)
│   └── DualPolHAlpha           (dual-pol H/Alpha eigendecomposition)
│
├── SublookDecomposition        [SAR] ── 1D sub-aperture (no category tag)
├── MultilookDecomposition      [STACKS, SAR] ── 2D sub-aperture
├── DominanceFeatures           [ANALYZE, SAR] ── aperture dominance/entropy
│
└── CSIProcessor                [ENHANCE, SAR] ── coherent shape index

ImageFormationAlgorithm (ABC) ── phase history → complex image
├── PolarFormatAlgorithm        (spotlight PFA)
├── StripmapPFA                 (stripmap PFA)
├── RangeDopplerAlgorithm       (stripmap RDA)
└── FastBackProjection          (FFT-accelerated BP)
```

Tags in `[brackets]` are `ProcessorCategory` / `ImageModality` values
set by `@processor_tags`.

**Image-formation algorithms are not yet registered in the processor
metadata catalog.** `ImageFormationAlgorithm` is a standalone ABC (not an
`ImageProcessor` subclass), and `PolarFormatAlgorithm`,
`RangeDopplerAlgorithm`, `FastBackProjection`, and `StripmapPFA` do
**not** currently carry `@processor_version` / `@processor_tags`. Their
contract is `form_image(signal, geometry)` and `get_output_grid()`, not
`apply()` / `__param_specs__`. They are therefore invisible to
grdl-runtime catalog discovery and grdk UI generation until tagged.

---

## When to Use What

### Filtering

| Need | Use | Why |
|------|-----|-----|
| General smoothing | `MeanFilter` | Fast separable O(N), uniform box |
| Edge-preserving smoothing | `GaussianFilter` | Weighted by distance, less ringing |
| Remove impulse noise | `MedianFilter` | Rank filter, eliminates outliers |
| SAR speckle (intensity) | `LeeFilter` | Adaptive Ci-weighting, ENL-aware |
| SAR speckle (complex SLC) | `ComplexLeeFilter` | Preserves interferometric phase |
| Phase fringe rate | `PhaseGradientFilter` | Windowed conjugate-multiply gradient |
| Local texture measure | `StdDevFilter` | O(N) variance decomposition trick |
| Morphological erosion | `MinFilter` | Local minimum |
| Morphological dilation | `MaxFilter` | Local maximum |

### Intensity / Enhancement

| Need | Use |
|------|-----|
| Log-scale display of SAR | `ToDecibels` |
| Contrast stretch for display | `PercentileStretch` |

### Decomposition

| Data | Use | Output |
|------|-----|--------|
| Quad-pol (HH, HV, VH, VV) | `PauliDecomposition` | surface, double-bounce, volume |
| Dual-pol (co + cross) | `DualPolHAlpha` | entropy, alpha, anisotropy, span |

### Sub-aperture Analysis

| Need | Use | Output shape |
|------|-----|-------------|
| 1D spectral splitting (azimuth or range) | `SublookDecomposition` | `(N, rows, cols)` |
| 2D spectral splitting (M x N grid) | `MultilookDecomposition` | `(M, N, rows, cols)` |

`MultilookDecomposition` uses `Tiler` from `grdl.data_prep` to partition
the 2D frequency support into rectangular bins. This is the Tiler's
first cross-module consumer, demonstrating the GRDL pattern of composing
generic utilities at the application level.

### Detection

| Clutter environment | Use |
|---------------------|-----|
| Homogeneous background | `CACFARDetector` |
| Clutter edges (suppress false alarms) | `GOCFARDetector` |
| Weak targets near edges | `SOCFARDetector` |
| Interfering strong targets | `OSCFARDetector` |

### Image Formation

| Collection mode | Use |
|-----------------|-----|
| Spotlight SAR (CPHD) | `PolarFormatAlgorithm` |
| Stripmap SAR (CPHD) | `StripmapPFA` or `RangeDopplerAlgorithm` |
| General backprojection | `FastBackProjection` |

### Geometric Correction / Orthorectification

| Need | Use |
|------|-----|
| Full ortho (recommended) | `orthorectify()` — keyword-argument function, auto-resolution, DEM |
| Ortho a geographic sub-region | `orthorectify(roi=(min_lat, max_lat, min_lon, max_lon))` |
| Memory-efficient large output | `orthorectify(tile_size=2048)` |
| ROI + tiling (composable) | `orthorectify(roi=(...), tile_size=2048)` |
| ENU output in meters | `orthorectify(enu_grid=dict(pixel_size_m=1.0))` |
| Low-level mapping + resample | `Orthorectifier` + `GeographicGrid` (compute_mapping / apply) |
| Auto-compute output resolution | `compute_output_resolution(metadata)` |

**`orthorectify()`** is the recommended entry point. It handles resolution
computation, output grid construction, DEM integration, ROI restriction,
and tiled processing via keyword arguments:

```python
geo.elevation = dem                          # DEM terrain correction (lives on geolocation)

result = orthorectify(
    geolocation=geo,
    reader=reader,
    metadata=reader.metadata,        # auto-resolution from SICD/BIOMASS
    roi=(36.0, 36.1, -75.8, -75.7), # geographic sub-region
    tile_size=2048,                  # memory-efficient tiling
    interpolation='nearest',
)
# result.data, result.output_grid, result.geolocation_metadata
```

**Tiling** partitions the output grid using `grdl.data_prep.Tiler`,
processes each tile independently (bounded mapping memory), and
assembles into the full output array. Each tile's `GeographicGrid` is
extracted via `GeographicGrid.sub_grid()`.

`OrthoBuilder` (fluent builder) is still available for advanced cases
requiring partial configuration or builder reuse.

### Composition

| Need | Use |
|------|-----|
| Chain multiple transforms | `Pipeline([step1, step2, ...])` |

---

## Key Design Patterns

### 1. Processor Metadata (`@processor_version` + `@processor_tags`)

Every concrete processor declares its version and capabilities:

```python
@processor_version('1.0.0')
@processor_tags(
    modalities=[ImageModality.SAR],
    category=ProcessorCategory.FILTERS,
    description='Adaptive SAR speckle filter',
)
class LeeFilter(BandwiseTransformMixin, ImageTransform):
    ...
```

- `__processor_version__` — semantic version, bumped on algorithm change.
- `__processor_tags__` — dict of modalities, category, description.
  Used by grdl-runtime for catalog filtering and grdk for UI discovery.

### 2. Tunable Parameters via `Annotated`

```python
class LeeFilter(BandwiseTransformMixin, ImageTransform):
    kernel_size: Annotated[int, Range(min=3, max=101),
                           Desc('Kernel size')] = 7
    enl: Annotated[float, Range(min=0.5, max=100.0),
                   Desc('Equivalent number of looks')] = 1.0
```

- `__init_subclass__` calls `collect_param_specs(cls)` at class
  definition → builds `__param_specs__` tuple of `ParamSpec`.
- If no custom `__init__`, one is auto-generated with keyword-only
  args, defaults, and constraint validation.
- `_resolve_params(kwargs)` merges instance defaults with runtime
  overrides.
- grdk reads `__param_specs__` to generate UI controls.

### 3. BandwiseTransformMixin

All filters subclass `(BandwiseTransformMixin, ImageTransform)`:

```
apply(source)  →  if 3D: for each band → _apply_2d(band)
                  if 2D: _apply_2d(source)
```

Subclasses implement only `_apply_2d()`. The mixin handles band
iteration, dtype preservation, and shape validation.

### 4. Template Method (CFAR)

`CFARDetector` implements the full detection pipeline. Subclasses
override only `_estimate_background()`:

```
detect(image)
  → _resolve_params()
  → _estimate_background(image, guard, training)   ← subclass hook
  → adaptive threshold from (bg_mean, bg_std, pfa)
  → binary mask → connected components → DetectionSet
```

### 5. Cross-Module Composition (Tiler → MultilookDecomposition)

`MultilookDecomposition` uses `grdl.data_prep.Tiler` to partition the
2D frequency support into a rectangular grid of bins:

```
MultilookDecomposition.__init__(metadata, looks_rg=3, looks_az=3)
  → _compute_support_geometry()   per dimension
  → _build_tiler(n_support_rg, n_support_az)
      → Tiler(nrows=..., ncols=..., tile_size=..., stride=...)

decompose(image)
  → fft2 + fftshift
  → deweight (separable 2D inverse-weight)
  → tiler.tile_positions()  →  M*N ChipRegion instances
  → for each region: extract sub-band → ifft2
  → result shape: (looks_rg, looks_az, rows, cols)
```

Dependency direction: `image_processing.sar` → `data_prep.Tiler`
(specific depends on generic).

### 6. Universal `execute()` Protocol

Every `ImageProcessor` exposes `execute(metadata, source, **kwargs) ->
(result, updated_metadata)` for grdl-runtime. The base implementation
sets `self._metadata` (so domain methods can read `self.metadata`) then
probes for `apply` / `detect` / `decompose`. Each ABC overrides it:

- `ImageTransform.execute` delegates to `apply()` and rebuilds output
  metadata (rows/cols/bands/dtype/axis_order, channel descriptors) via
  `_update_metadata_for_result` + `_infer_3d_axis_order`.
- `ImageDetector.execute` delegates to `detect()` and returns metadata
  unchanged (output is a `DetectionSet`).
- `PolarimetricDecomposition.execute` extracts quad-pol channels from a
  4-band source (or `shh/shv/svh/svv` kwargs) and returns a component
  dict plus per-component channel metadata.
- `Pipeline.execute` threads updated metadata forward step to step.

Concrete classes implement only the domain method, never `execute()`.

### 7. Version-Check on First Instantiation

`ImageProcessor.__new__` warns once per class (tracked in
`_version_warned_classes`) if a concrete class lacks
`__processor_version__`. The check runs in `__new__` (not
`__init_subclass__`) so decorators have already been applied. Abstract
classes are skipped.

### 8. Detection-Input Flow

Any processor can declare upstream `DetectionSet` inputs by overriding
`detection_input_specs` (tuple of `DetectionInputSpec`). Inputs arrive as
`**kwargs`; `_validate_detection_inputs(kwargs)` enforces required ones
and `_get_detection_input(name, kwargs)` retrieves them. This lets a
detector bias a downstream transform without signature changes.

### 9. Detection Data Models

`detection/models.py` defines `Detection` (shapely `pixel_geometry` +
optional `geo_geometry`, `properties` dict, optional `confidence`;
`to_geojson_feature()`) and `DetectionSet` (`__len__` / `__iter__` /
`__getitem__`, `to_geojson()`, `filter_by_confidence()`,
`filter_by_region(shape, geolocation, mode=...)`). Pixel coords use
shapely `(x=col, y=row)`; geographic coords use `(x=lon, y=lat)`. Field
names in `properties` should come from the `Fields` data dictionary
(`detection/fields.py`, `DATA_DICTIONARY`); non-dictionary names emit a
`UserWarning`. `CFARDetector._build_detections` emits boxes carrying
`sar.sigma0`, `identity.is_target`, and `physical.area/length/width`.

---

## Processor Catalog

| Class | Version | Category | Modalities | Parameters |
|-------|---------|----------|-----------|------------|
| `MeanFilter` | 1.0.0 | FILTERS | all | kernel_size, mode |
| `GaussianFilter` | 1.0.0 | FILTERS | all | sigma, mode |
| `MedianFilter` | 1.0.0 | FILTERS | all | kernel_size, mode |
| `MinFilter` | 1.0.0 | FILTERS | all | kernel_size, mode |
| `MaxFilter` | 1.0.0 | FILTERS | all | kernel_size, mode |
| `StdDevFilter` | 1.0.0 | FILTERS | all | kernel_size, mode |
| `LeeFilter` | 2.0.0 | FILTERS | SAR | kernel_size, enl, mode |
| `ComplexLeeFilter` | 2.0.0 | FILTERS | SAR | kernel_size, enl, mode |
| `PhaseGradientFilter` | 1.0.0 | FILTERS | SAR | kernel_size, direction |
| `ToDecibels` | 1.0.0 | ENHANCE | all | floor_db |
| `PercentileStretch` | 1.0.0 | ENHANCE | all | plow, phigh |
| `Orthorectifier` | 0.1.0 | GEOM_CORRECT | all | interpolation |
| `OrthoBuilder` | — | — | all | source, geolocation, resolution, roi, tile_size, interpolation, nodata |
| `GeographicGrid` | — | — | — | min/max lat/lon, pixel sizes; `sub_grid()`, `from_geolocation()` |
| `PauliDecomposition` | 0.1.0 | — | SAR | — |
| `DualPolHAlpha` | 1.0.0 | — | SAR | window_size |
| `CACFARDetector` | 1.0.0 | FIND_MAXIMA | SAR | guard_cells, training_cells, pfa, min_pixels, assumption |
| `GOCFARDetector` | 1.0.0 | FIND_MAXIMA | SAR | guard_cells, training_cells, pfa, min_pixels, assumption |
| `SOCFARDetector` | 1.0.0 | FIND_MAXIMA | SAR | guard_cells, training_cells, pfa, min_pixels, assumption |
| `OSCFARDetector` | 1.0.0 | FIND_MAXIMA | SAR | guard_cells, training_cells, pfa, min_pixels, assumption, percentile |
| `SublookDecomposition` | 0.1.0 | — | SAR | num_looks, dimension, overlap, deweight, deskew |
| `MultilookDecomposition` | 0.1.0 | STACKS | SAR | looks_rg, looks_az, overlap, deweight |
| `CSIProcessor` | 0.2.0 | ENHANCE | SAR | dimension, overlap, deweight, normalization, plow, phigh, floor_db |
| `DominanceFeatures` | 1.0.0 | ANALYZE | SAR | num_looks, dimension, window_size, dom_window |

All five CFAR parameters are declared as `Annotated` fields on the
`CFARDetector` base; `assumption` selects `'gaussian'` (dB input,
threshold = mean + alpha*std) vs `'exponential'` (linear input,
threshold = alpha*mean). The `SublookDecomposition` `deskew` flag centers
the phase history via `DeltaKCOAPoly` before sub-band cutting (required
for stripmap / ScanSAR-TOPS, near no-op for spotlight).

> **Not in this catalog:** the image-formation algorithms
> (`PolarFormatAlgorithm`, `RangeDopplerAlgorithm`, `FastBackProjection`,
> `StripmapPFA`) are not `ImageProcessor` subclasses and carry no
> `@processor_version` / `@processor_tags` — they are absent from
> runtime/grdk discovery.

---

## Axis Conventions

SAR processors use a consistent axis mapping between image domain,
frequency domain, and SICD metadata:

| Domain | Rows (axis 0) | Cols (axis 1) |
|--------|---------------|---------------|
| Image pixels | range | azimuth |
| Spectrum (after `fft2` + `fftshift`) | range freq | azimuth freq |
| SICD metadata | `grid.row` = range | `grid.col` = azimuth |
| Tiler (multilook) | `nrows` = range bins | `ncols` = azimuth bins |
| MultilookDecomposition output | `looks_rg` (dim 0) | `looks_az` (dim 1) |

---

## Adding a New Processor

1. Choose the base class:
   - Dense raster transform → `ImageTransform` (or `BandwiseTransformMixin`)
   - Sparse vector detector → `ImageDetector`
   - Polarimetric decomposition → `PolarimetricDecomposition`
   - SAR spectral analysis → `ImageProcessor` directly
   - Image formation → `ImageFormationAlgorithm`

2. Add `@processor_version('x.y.z')` and `@processor_tags(...)`.

3. Declare tunable parameters as `Annotated` class-body fields.

4. Implement the abstract method (`apply`, `detect`, `decompose`,
   or `form_image`).

5. Export from the sub-module and top-level `__init__.py`.

6. Write tests in `tests/test_image_processing_<submodule>.py`.

See `CLAUDE.md` § "Adding a New Module" and § "Processor Metadata"
for full details.
