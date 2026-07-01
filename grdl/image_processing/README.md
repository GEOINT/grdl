# `grdl.image_processing` — Image Transforms, Detection, Decomposition, Formation

Top-level module for everything that turns one image into another or
into a structured output. Concrete sub-domains:

- **Filters** — spatial and statistical filters (Lee, Gaussian, median,
  phase gradient, ...)
- **Decomposition** — polarimetric (Pauli quad-pol, DualPolHAlpha)
- **Detection** — `Detection` / `DetectionSet` data models + CFAR
  variants (CA, GO, SO, OS)
- **SAR** — sublook decomposition, CSI RGB composite, dominance
  features, image formation (PFA, RDA, FFBP, StripmapPFA)
- **Ortho** — orthorectification, output grids, accelerated resampling,
  point-ROI helper. See [ortho/README.md](ortho/README.md).
- **Intensity** — `ToDecibels`, `PercentileStretch` (also re-exported
  from [`grdl.contrast`](../contrast/README.md))
- **Versioning & metadata** — `@processor_version`, `@processor_tags`,
  `Annotated` tunable parameters
- **Pipeline** — sequential composition (optional; everything also
  works without it)

For class hierarchies, ABC contracts, GPU/numba/torch backend
dispatch, and the processor metadata system, see
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## Quick Tour

### Polarimetric decomposition

```python
from grdl.image_processing import PauliDecomposition

pauli = PauliDecomposition()
components = pauli.decompose(shh, shv, svh, svv)
rgb = pauli.to_rgb(components)        # (rows, cols, 3) float32 [0, 1]
```

### Sub-aperture decomposition + CSI

```python
from grdl.image_processing.sar import (
    SublookDecomposition, CSIProcessor, compute_dominance,
)

# num_looks, dimension ('azimuth'|'range'), overlap, deweight, deskew.
# deskew centers the phase history via DeltaKCOAPoly before cutting
# sub-bands -- required for stripmap / ScanSAR-TOPS, a near no-op for
# spotlight (default True).
sublook = SublookDecomposition(reader.metadata, num_looks=7, deskew=True)
looks   = sublook.decompose(image)    # (7, rows, cols) complex
mag     = sublook.to_magnitude(looks)

# Color sub-aperture (CSI) RGB. normalization: 'none'|'log'|'percentile'.
csi     = CSIProcessor(reader.metadata, dimension='azimuth',
                       overlap=0.5, normalization='log')
csi_rgb = csi.apply(image)            # (3, rows, cols)

# Aperture-power dominance feature
dom = compute_dominance(looks, window_size=7, dom_window=3)  # (rows, cols)
```

### CFAR detection

```python
from grdl.image_processing.detection.cfar import CACFARDetector

# Constructor params: guard_cells, training_cells, pfa, min_pixels, assumption
detector = CACFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
detections = detector.detect(image)    # DetectionSet

# Geo-registered output, optional ROI gating, GeoJSON export
detections = detector.detect(image, geolocation=geo, valid_mask=roi_mask)
strong = detections.filter_by_confidence(0.5)
feature_collection = detections.to_geojson()
```

`detect()` accepts `assumption='gaussian'` for dB-scale input (threshold =
mean + alpha * std) or `assumption='exponential'` for linear-scale input
(threshold = alpha * mean). `OSCFARDetector` adds a `percentile` parameter
(default 75.0) for ordered-statistics background estimation.

### Orthorectification

```python
from grdl.image_processing.ortho import orthorectify, orthorectify_point_roi

# Full-image
geo.elevation = open_elevation('/data/dted/')
result = orthorectify(geolocation=geo, reader=reader, interpolation='lanczos')

# Point-ROI single-call helper (any modality)
result = orthorectify_point_roi(
    reader=reader, lat=34.05, lon=-118.25,
    width_m=500, height_m=500, pixel_size_m=0.25,
)
```

### SAR image formation

```python
from grdl.image_processing.sar.image_formation import (
    PolarFormatAlgorithm, RangeDopplerAlgorithm,
    FastBackProjection, StripmapPFA, PolarGrid, CollectionGeometry,
)

# Spotlight PFA. PolarGrid now defaults scene_sizing='full' (data-
# unambiguous full-scene extent); 'toa' sizes to the receive window.
geometry = CollectionGeometry(...)         # per-pulse state from CPHD
grid = PolarGrid(geometry, scene_sizing='full')
pfa = PolarFormatAlgorithm(grid=grid, weighting='taylor')
image = pfa.form_image(signal, geometry)   # complex SAR image
```

> **Note:** the image-formation algorithms (`PolarFormatAlgorithm`,
> `RangeDopplerAlgorithm`, `FastBackProjection`, `StripmapPFA`) do **not**
> yet carry `@processor_version` / `@processor_tags`, so they are not
> registered in the processor-metadata catalog used by grdl-runtime and
> grdk. They share the `ImageFormationAlgorithm` ABC
> (`form_image()`, `get_output_grid()`) rather than `ImageProcessor`.

### Pipeline (optional)

```python
from grdl.image_processing import Pipeline
from grdl.image_processing.intensity import ToDecibels, PercentileStretch

display = Pipeline([ToDecibels(), PercentileStretch()]).apply(sar_image)
```

---

## Detection Data Models

Detectors return a `DetectionSet` of `Detection` objects (see
[`detection/models.py`](detection/models.py)). Geometry is carried as
shapely objects in two coordinate spaces:

- **pixel space**: shapely `(x, y)` = `(col, row)`, origin top-left.
- **geographic space** (`geo_geometry`): shapely `(x, y)` =
  `(longitude, latitude)` in WGS84 (GeoJSON order). Populated when a
  `geolocation` is passed to `detect()`.

```python
det = DetectionSet([...], detector_name='CACFAR', detector_version='1.0.0',
                   output_fields=(Fields.sar.SIGMA0,))
len(det); det[0]; [d for d in det]          # len / index / iter
det.filter_by_confidence(0.5)               # → new DetectionSet
det.filter_by_region(shape, geo, mode='centroid')  # 'intersects' | 'contained'
det.to_geojson()                            # FeatureCollection dict
det[0].to_geojson_feature()                 # single Feature dict
```

### Fields data dictionary

Detection `properties` keys should come from the GRDL data dictionary
([`detection/fields.py`](detection/fields.py)). Non-dictionary names emit
a `UserWarning` at `DetectionSet` construction. Access standardized names
via dot-notation domains for IDE autocomplete:

```python
from grdl.image_processing.detection.fields import Fields

Fields.sar.SIGMA0            # 'sar.sigma0'
Fields.sar.CHANGE_MAGNITUDE  # 'sar.change_magnitude'
Fields.sar.COHERENCE         # 'sar.coherence'
Fields.sar.POL_ENTROPY       # 'sar.pol_entropy'
Fields.identity.IS_TARGET    # 'identity.is_target'
Fields.physical.AREA         # 'physical.area'
```

Domains: `physical`, `sar`, `gmti`, `spectral`, `volume`, `identity`,
`trait`, `temporal`, `context`. Use `lookup_field(name)`,
`is_dictionary_field(name)`, and `list_fields(domain=...)` to introspect.

---

## Sub-Module READMEs

| Sub-module | README |
|-----------|--------|
| `ortho/` | [ortho/README.md](ortho/README.md) |

ABCs, processor-metadata machinery, and backend dispatch are documented
in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Adding a Processor

Every concrete `ImageProcessor` subclass should declare:

1. `@processor_version('x.y.z')` — algorithm version stamp
2. `@processor_tags(modalities=[...], category=..., description='...')`
   — capability metadata
3. `Annotated` class-body fields with `Range` / `Options` / `Desc` for
   tunable parameters

```python
from typing import Annotated, Any
import numpy as np

from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Range, Options, Desc
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_version('1.0.0')
@processor_tags(
    modalities=[IM.SAR, IM.PAN],
    category=PC.FILTERS,
    description='Adaptive edge-preserving smoothing',
)
class MyFilter(ImageTransform):
    sigma: Annotated[float, Range(min=0.1, max=100.0),
                     Desc('Gaussian sigma')] = 2.0

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        params = self._resolve_params(kwargs)
        # ...
```

At class-definition time, `ImageProcessor.__init_subclass__` calls
`collect_param_specs(cls)` to gather every `Annotated` field carrying a
`ParamMeta` marker (`Range`, `Options`, `Desc`) into the
`__param_specs__` tuple, and auto-generates a keyword-only `__init__`
(unless the subclass defines its own — common when non-tunable
constructor args like metadata are needed). At runtime,
`self._resolve_params(kwargs)` merges instance defaults with per-call
overrides and validates each value via `ParamSpec.validate` (type, range,
choices). `Range` and `Options` are mutually exclusive on one field.

`@processor_tags` accepts the full capability set, all enum values
validated eagerly at decorator time (typos fail at import):

| Argument | Type | Purpose |
|----------|------|---------|
| `modalities` | `Sequence[ImageModality]` | Supported imagery types |
| `category` | `ProcessorCategory` | Functional grouping |
| `description` | `str` | Human-readable purpose |
| `detection_types` | `Sequence[DetectionType]` | For `ImageDetector` outputs |
| `segmentation_types` | `Sequence[SegmentationType]` | For segmentation outputs |
| `phases` | `Sequence[ExecutionPhase]` | Compatible pipeline phases |
| `gpu_capability` | `GpuCapability` | `REQUIRED` / `PREFERRED` / `CPU_ONLY` |
| `required_bands` | `int` | Band count the processor needs |
| `pol_mode` | `PolarimetricMode` | Required polarimetric mode |

`@processor_version()` with no argument infers the version from the
installed `grdl` package metadata. Methods decorated with
`@globalprocessor` are collected into `__global_callbacks__`
(`__has_global_pass__` becomes `True`), signaling the executor to stream
the full image through those callbacks before running `apply()`.

`grdl-runtime` reads `__processor_tags__` for catalog discovery; `grdk`
reads `__param_specs__` to build dynamic UI controls. See [CLAUDE.md](../../CLAUDE.md)
for the full processor metadata convention.
