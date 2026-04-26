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
from grdl.image_processing.sar import SublookDecomposition, CSIProcessor

sublook = SublookDecomposition(reader.metadata, num_looks=7)
looks   = sublook.decompose(image)    # (7, rows, cols) complex

csi     = CSIProcessor(reader.metadata, dimension='azimuth')
csi_rgb = csi.apply(image)            # (rows, cols, 3)
```

### CFAR detection

```python
from grdl.image_processing.detection.cfar import CACFARDetector

detector = CACFARDetector(guard_cells=2, background_cells=4, pfa=1e-6)
detections = detector.detect(image)    # DetectionSet
```

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
    FastBackProjection, StripmapPFA,
)
```

### Pipeline (optional)

```python
from grdl.image_processing import Pipeline
from grdl.image_processing.intensity import ToDecibels, PercentileStretch

display = Pipeline([ToDecibels(), PercentileStretch()]).apply(sar_image)
```

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

`grdl-runtime` reads `__processor_tags__` for catalog discovery; `grdk`
reads `__param_specs__` to build dynamic UI controls. See [CLAUDE.md](../../CLAUDE.md)
for the full processor metadata convention.
