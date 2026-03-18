# GRDL Development Guide

## Project Overview

GRDL (GEOINT Rapid Development Library) is a modular Python library for geospatial intelligence workflows. It operates on any 2D correlated imagery -- SAR, EO, MSI, hyperspectral, space-based, or terrestrial.

This is a **library, not a framework**. Every module must be independently usable.

## Development Environment

**Python Environment:** Always use the `grdl` conda environment for all Python operations (testing, development, package installation).

```bash
conda activate grdl
python tests/test_io_biomass.py --quick
```

## Architecture Rules

### Module Design

- One concept per module. If a module does two things, split it.
- Modularity is enforced through **abstract base classes**. Each domain defines ABCs that establish the contract. Concrete implementations inherit from them.
- Sensor-agnostic by default. Sensor-specific logic (SAR, EO, MSI) goes in dedicated submodules under its sensor directory, implementing the domain's ABC.
- No global state. No singletons. No module-level side effects on import.
- All public functions and classes operate on numpy arrays or standard Python types as their primary data interface.

### Use GRDL Modules for Their Purpose

Every GRDL module owns a specific responsibility. **Always use the purpose-built module instead of writing ad-hoc code.** If a GRDL module exists for the task, use it.

| Task | Use this | Not this |
|------|----------|----------|
| Load any imagery format | `grdl.IO` readers (`SICDReader`, `NISARReader`, `Sentinel2Reader`, `EONITFReader`, ...) | Raw `rasterio.open()` / `h5py.File()` |
| Write imagery to disk | `grdl.IO` writers (`GeoTIFFWriter`, `SICDWriter`, `NumpyWriter`, ...) | Raw write calls |
| Open any supported format | `grdl.IO.generic.open_any()` | Manual format detection |
| Plan chip/tile regions | `grdl.data_prep.ChipExtractor` or `Tiler` | Hand-rolled `for r in range(0, rows, sz):` loops |
| Normalize for ML | `grdl.data_prep.Normalizer` | Inline min-max arithmetic |
| Image to lat/lon | `grdl.geolocation` (`AffineGeolocation`, `SICDGeolocation`, `RPCGeolocation`, `RSMGeolocation`, ...) | Manual affine math or GCP interpolation |
| EO NITF geolocation (RPC/RSM) | `grdl.geolocation.eo.rpc` / `grdl.geolocation.eo.rsm` | Manual RPC polynomial evaluation |
| Coordinate conversion | `grdl.geolocation.coordinates` (geodetic/ECEF/ENU) | Manual WGS84 math |
| Terrain elevation lookup | `grdl.geolocation.elevation` (`DTEDElevation`, `GeoTIFFDEM`) | Raw `rasterio.open()` on DEM tiles |
| SAR decomposition | `grdl.image_processing` (`PauliDecomposition`, `DualPolHAlpha`) | Manual complex arithmetic |
| Sub-aperture dominance | `grdl.image_processing.sar.dominance` (`compute_dominance`) | Manual sublook power ratios |
| CSI RGB composite | `grdl.image_processing.sar.CSIProcessor` | Ad-hoc HSV mapping |
| CFAR target detection | `grdl.image_processing.detection.cfar` (CA, GO, SO, OS) | Manual threshold loops |
| SAR image formation | `grdl.image_processing.sar.image_formation` (PFA, RDA, FFBP) | Custom FFT pipelines |
| Interpolation / resampling | `grdl.interpolation` (`PolyphaseInterpolator`, `LanczosInterpolator`, ...) | Manual sinc convolution |
| Image alignment | `grdl.coregistration` | Custom OpenCV wrappers |
| Transform detection geometries | `grdl.transforms` | Manual coordinate mapping |

Modules handle edge cases (boundary snapping, band indexing, lazy loading, resource cleanup) that ad-hoc code misses. **Compose them at the application level** — each module does its job, the application wires them together. See `grdl/example/image_processing/sar/sublook_compare.py` and `grdl/example/image_processing/sar/csi_detection_overlay.py` for full integration examples.

### Fail Fast

- All imports at the top of the file. If a dependency is missing, the module fails on import -- not buried in a runtime call.
- Validate inputs early. Raise clear exceptions before doing work with bad data.
- Never silently coerce types or swallow errors.

### Abstract Base Classes

Each domain directory defines its contracts via ABCs in a `base.py` file:

```python
from abc import ABC, abstractmethod
import numpy as np

class ImageFilter(ABC):
    """Base class for all spatial image filters."""

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the filter to a 2D image array.

        Parameters
        ----------
        image : np.ndarray
            2D array of pixel values. Shape (rows, cols).

        Returns
        -------
        np.ndarray
            Filtered image.
        """
        ...
```

Concrete implementations inherit and implement:

```python
from .base import ImageFilter

class MedianFilter(ImageFilter):
    def __init__(self, kernel_size: int = 3) -> None:
        ...

    def apply(self, image: np.ndarray) -> np.ndarray:
        ...
```

### Processor Metadata (Highly Recommended)

Every concrete `ImageProcessor` subclass (transforms, detectors, decompositions) should declare metadata via decorators and `Annotated` type hints. This metadata is what makes processors discoverable in grdl-runtime's catalog and configurable in grdk's UI. **Always add this metadata when creating new processors.**

#### 1. `@processor_version` — Algorithm version stamp

```python
from grdl.image_processing.versioning import processor_version

@processor_version('1.0.0')
class MyFilter(ImageTransform):
    ...
```

- Sets `__processor_version__` on the class.
- A runtime warning fires at first instantiation if this decorator is missing.
- Use semantic versioning. Bump when the algorithm or output format changes.

#### 2. `@processor_tags` — Capability discovery metadata

```python
from grdl.image_processing.versioning import processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC

@processor_tags(
    modalities=[IM.SAR, IM.PAN],
    category=PC.FILTERS,
    description='Adaptive edge-preserving smoothing',
)
class MyFilter(ImageTransform):
    ...
```

- Sets `__processor_tags__` dict on the class.
- `modalities`: which imagery types the processor supports (enum from `grdl.vocabulary.ImageModality`).
- `category`: functional grouping (enum from `grdl.vocabulary.ProcessorCategory`).
- `description`: short human-readable purpose string.
- Also accepts `detection_types` and `segmentation_types` for detector/segmentation processors.
- All enum values are validated eagerly at decorator time (fail-fast on typos).
- grdl-runtime uses these tags for catalog filtering; grdk uses them for widget discovery.

#### 3. `Annotated` Tunable Parameters — Declarative constraints

Declare tunable parameters as class-body annotations with constraint markers from `grdl.image_processing.params`:

```python
from typing import Annotated
from grdl.image_processing.params import Range, Options, Desc

@processor_version('1.0.0')
@processor_tags(modalities=[IM.PAN], category=PC.FILTERS)
class MyFilter(ImageTransform):
    sigma: Annotated[float, Range(min=0.1, max=100.0),
                     Desc('Gaussian sigma')] = 2.0
    method: Annotated[str, Options('bilateral', 'guided'),
                      Desc('Filter algorithm')] = 'bilateral'

    def apply(self, source, **kwargs):
        params = self._resolve_params(kwargs)
        # Use params['sigma'], params['method']
        ...
```

**Constraint markers:**

| Marker | Purpose | Example |
|--------|---------|---------|
| `Range(min=, max=)` | Inclusive numeric bounds | `Range(min=0.0, max=1.0)` |
| `Options(...)` | Discrete allowed values | `Options('nearest', 'bilinear')` |
| `Desc('...')` | Human-readable label for GUIs | `Desc('Kernel radius in pixels')` |

**How the machinery works:**

- `ImageProcessor.__init_subclass__` calls `collect_param_specs(cls)` → populates `cls.__param_specs__`.
- If no custom `__init__` is defined, one is auto-generated with keyword-only args, defaults, and validation.
- `_resolve_params(kwargs)` merges instance defaults with runtime overrides and validates all values.
- `Range` and `Options` are mutually exclusive on the same parameter.
- grdk reads `__param_specs__` to build dynamic slider/dropdown/checkbox controls.

**When to use a custom `__init__`:** If the processor needs non-tunable constructor arguments (e.g., metadata objects, geolocation references), define a custom `__init__`. The `__param_specs__` tuple is still built for introspection, but the auto-generated init is skipped.

#### Complete Example

```python
from typing import Annotated, Any
import numpy as np
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.image_processing.params import Range, Options, Desc
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC

@processor_version('1.0.0')
@processor_tags(
    modalities=[IM.PAN, IM.SAR, IM.MSI],
    category=PC.FILTERS,
    description='Configurable spatial median filter',
)
class MedianFilter(ImageTransform):
    radius: Annotated[int, Range(min=1, max=50),
                      Desc('Filter kernel radius')] = 2
    boundary: Annotated[str, Options('reflect', 'constant', 'wrap'),
                        Desc('Boundary handling mode')] = 'reflect'

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        params = self._resolve_params(kwargs)
        # params['radius'] and params['boundary'] are validated
        ...
```

### Directory Structure

```
GRDL/
  grdl/
    exceptions.py            # Custom exception hierarchy (GrdlError → ValidationError, ProcessorError, etc.)
    py.typed                 # PEP 561 type stub marker
    IO/                      # Input/Output — format readers and writers
      base.py                # ImageReader / ImageWriter / CatalogInterface ABCs
      geotiff.py             # GeoTIFFReader, GeoTIFFWriter (rasterio)
      hdf5.py                # HDF5Reader, HDF5Writer (h5py)
      jpeg2000.py            # JP2Reader (glymur)
      nitf.py                # NITFReader, NITFWriter (rasterio/GDAL)
      numpy_io.py            # NumpyWriter (.npy / .npz)
      png.py                 # PngWriter
      generic.py             # GDALFallbackReader, open_any()
      probe.py               # InvasiveProbeReader (format sniffing)
      models/                # Typed metadata dataclasses
        base.py              # ImageMetadata base class
        common.py            # Shared primitives (XYZ w/ vector math, callable Poly1D/Poly2D/XYZPoly, ...)
        sicd.py              # SICDMetadata (~35 nested dataclasses)
        sidd.py              # SIDDMetadata (~25 nested dataclasses)
        cphd.py              # CPHDMetadata
        biomass.py           # BIOMASSMetadata
        viirs.py             # VIIRSMetadata
        aster.py             # ASTERMetadata
        sentinel1_slc.py     # Sentinel1SLCMetadata
        sentinel2.py         # Sentinel2Metadata
        terrasar.py          # TerraSARMetadata
        nisar.py             # NISARMetadata
        eo_nitf.py           # EONITFMetadata, RPCCoefficients, RSMCoefficients
      sar/                   # SAR modality submodule
        _backend.py          # sarkit/sarpy availability
        sicd.py              # SICDReader (sarkit/sarpy)
        sicd_writer.py       # SICDWriter
        cphd.py              # CPHDReader (sarkit/sarpy)
        crsd.py              # CRSDReader (sarkit)
        sidd.py              # SIDDReader (sarkit)
        sidd_writer.py       # SIDDWriter
        biomass.py           # BIOMASSL1Reader
        biomass_catalog.py   # BIOMASSCatalog
        sentinel1_slc.py     # Sentinel1SLCReader
        terrasar.py          # TerraSARReader, open_terrasar()
        nisar.py             # NISARReader, open_nisar()
      ir/                    # IR/thermal modality submodule
        _backend.py          # rasterio/h5py availability
        aster.py             # ASTERReader (L1T, GDEM)
      multispectral/         # Multispectral modality submodule
        _backend.py          # h5py/xarray/spectral availability
        viirs.py             # VIIRSReader
      eo/                    # EO modality submodule
        _backend.py          # rasterio/glymur availability
        sentinel2.py         # Sentinel2Reader
        nitf.py              # EONITFReader (RPC/RSM extraction)
    geolocation/             # Image-to-geographic coordinate transforms with DEM integration
      base.py                # Geolocation ABC, NoGeolocation, iterative DEM refinement
      utils.py               # Footprint, bounds, distance helpers
      coordinates.py         # geodetic_to_ecef, ecef_to_geodetic, geodetic_to_enu, enu_to_geodetic
      projection.py          # COAProjection, image_to_ground_hae/dem, ground_to_image, wgs84_norm
      __init__.py            # Re-exports all public classes
      sar/                   # SAR geolocation submodule
        _backend.py          # sarpy/sarkit availability probing
        gcp.py               # GCPGeolocation (BIOMASS Delaunay interpolation)
        sicd.py              # SICDGeolocation (native R/Rdot, sarpy, or sarkit)
        sidd.py              # SIDDGeolocation (SIDD imagery)
        nisar.py             # NISARGeolocation
        sentinel1_slc.py     # Sentinel1SLCGeolocation
        __init__.py
      eo/                    # EO geolocation submodule
        _backend.py          # rasterio/pyproj availability probing
        affine.py            # AffineGeolocation (geocoded rasters, affine + pyproj)
        rpc.py               # RPCGeolocation (RPC00B rational polynomials)
        rsm.py               # RSMGeolocation (RSMPCA replacement sensor model)
        __init__.py
      elevation/             # Terrain elevation models
        _backend.py          # rasterio availability probing
        base.py              # ElevationModel ABC
        constant.py          # ConstantElevation (fixed-height fallback)
        dted.py              # DTEDElevation (DTED tiles via rasterio)
        geotiff_dem.py       # GeoTIFFDEM (GeoTIFF DEM via rasterio)
        geoid.py             # GeoidCorrection (EGM96 geoid undulation lookup)
        __init__.py
    image_processing/        # Image transforms, detection, formation
      base.py                # ImageProcessor, ImageTransform, BandwiseTransformMixin ABCs
      params.py              # Range, Options, Desc, ParamSpec
      versioning.py          # @processor_version, @processor_tags
      pipeline.py            # Pipeline (sequential transform composition)
      ortho/                 # Orthorectification
        ortho.py             # Orthorectifier, OutputGrid, OrthoPipeline, OrthoResult
        enu_grid.py          # ENUGrid (local East-North-Up grid)
        accelerated.py       # resample(), detect_backend()
      decomposition/         # Polarimetric decomposition
        base.py              # PolarimetricDecomposition ABC
        pauli.py             # PauliDecomposition (quad-pol)
        dual_pol.py          # DualPolHAlpha (dual-pol H/Alpha)
      detection/             # Target detection
        base.py              # ImageDetector ABC
        models.py            # Detection, DetectionSet
        fields.py            # Data dictionary (Fields.sar, Fields.physical, ...)
        cfar/                # CFAR detector variants
          _base.py           # CFARDetector ABC
          ca.py              # CACFARDetector (Cell-Averaging)
          go.py              # GOCFARDetector (Greatest-Of)
          so.py              # SOCFARDetector (Smallest-Of)
          os.py              # OSCFARDetector (Ordered-Statistics)
      sar/                   # SAR-specific transforms
        sublook.py           # SublookDecomposition (sub-aperture splitting)
        csi.py               # CSIProcessor (Coherent Shape Index RGB)
        dominance.py         # DominanceFeatures, compute_dominance, compute_sublook_entropy
        image_formation/     # SAR image formation algorithms
          pfa.py             # PolarFormatAlgorithm
          rda.py             # RangeDopplerAlgorithm
          ffbp.py            # FastBackProjection
          stripmap_pfa.py    # StripmapPFA
    interpolation/           # 1D bandwidth-preserving interpolation kernels
      base.py                # Interpolator, KernelInterpolator ABCs
      lanczos.py             # LanczosInterpolator
      windowed_sinc.py       # KaiserSincInterpolator
      lagrange.py            # LagrangeInterpolator
      farrow.py              # FarrowInterpolator
      polyphase.py           # PolyphaseInterpolator
      thiran.py              # ThiranDelayFilter (IIR allpass)
    transforms/              # Detection geometry transforms
      detection.py           # transform_pixel_geometry, transform_detection, transform_detection_set
    data_prep/               # ML/AI data preparation — index-only chip/tile planning
      base.py                # ChipBase ABC, ChipRegion NamedTuple, shared helpers
      tiler.py               # Tiler (stride-based tile region computation)
      chip_extractor.py      # ChipExtractor (point-centered and whole-image chip regions)
      normalizer.py          # Normalizer (minmax, zscore, percentile, unit_norm)
    coregistration/          # Image alignment and registration
      affine.py              # Affine transform alignment
      projective.py          # Projective transform alignment
      feature_match.py       # Feature-based matching (OpenCV)
  tests/
    conftest.py              # Shared pytest fixtures (synthetic images)
    test_<domain>_<module>.py
    test_benchmarks.py       # Performance benchmarks (pytest-benchmark)
  pyproject.toml             # Package config, dependencies, tool settings
  example_images/            # Small sample data for tests and demos
```

Domain directories map to the module areas defined in the README:

| Directory | Domain |
|-----------|--------|
| `IO/` | Format readers and writers (base formats + `sar/`, `ir/`, `multispectral/`, `eo/` modality submodules + writers) |
| `IO/models/` | Typed metadata dataclasses (`SICDMetadata`, `SIDDMetadata`, `CPHDMetadata`, `BIOMASSMetadata`, `VIIRSMetadata`, `ASTERMetadata`, `Sentinel1SLCMetadata`, `Sentinel2Metadata`, `TerraSARMetadata`, `NISARMetadata`, `EONITFMetadata` + `RPCCoefficients`/`RSMCoefficients`) |
| `geolocation/` | Image-to-geographic coordinate transforms with DEM integration (`sar/`, `eo/`, `elevation/` submodules + coordinate utilities + native R/Rdot projection engine + RPC/RSM geolocation) |
| `image_processing/` | Orthorectification (+ ENUGrid, accelerated resampling), polarimetric decomposition (Pauli, DualPolHAlpha), SAR sublook/CSI/dominance, image formation (PFA, RDA, FFBP), CFAR detection, versioning, tunable parameters, pipeline |
| `interpolation/` | 1D bandwidth-preserving interpolation kernels (Lanczos, Kaiser sinc, Lagrange, Farrow, Polyphase, Thiran) |
| `transforms/` | Detection geometry transforms (apply coregistration to vector detections) |
| `data_prep/` | Index-only chip/tile planning (`ChipExtractor`, `Tiler`) and normalization (`Normalizer`) for ML/AI pipelines |
| `coregistration/` | Affine, projective, and feature-matching image alignment |
| `exceptions.py` | Custom exception hierarchy (GrdlError, ValidationError, ProcessorError, etc.) |
| `sensors/` | Sensor-specific operations (subdirs: `sar/`, `eo/`, `msi/`) -- planned |
| `ml/` | Feature extraction, annotation, dataset builders -- planned |

## File Header Standard

Every `.py` file in the repository must begin with a standardized header block. This header appears before all imports and code.

### Header Format

```python
# -*- coding: utf-8 -*-
"""
<Module title -- short description of what this file contains.>

<Extended description if needed. Explain purpose, scope, and any important
context about the module's role in the library.>

Dependencies
------------
<List any third-party dependencies beyond numpy, one per line.>

Author
------
<Author name and email. Retrieve from the local OS user profile or
prompt the user at the start of the session. Do not assume a default.>

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
<YYYY-MM-DD>

Modified
--------
<YYYY-MM-DD>
"""
```

### Header Rules

- **Encoding declaration**: Always include `# -*- coding: utf-8 -*-` as the first line.
- **Module title**: First line of the docstring is a concise summary (one line, under 79 characters).
- **Dependencies**: List only non-numpy third-party packages. Omit this section if only numpy is required.
- **Author**: Retrieve the author name and email from the local OS user profile, IDE settings, or git config. If unavailable, prompt the user at the start of the session. Do not hardcode a default author. Additional contributors append their name and email on new lines.
- **License**: Always `MIT License` with the copyright line. Do not duplicate the full license text.
- **Created**: The date the file was first written, in `YYYY-MM-DD` format.
- **Modified**: The date of the most recent substantive change, in `YYYY-MM-DD` format. Update this on every edit.

### Complete Example

```python
# -*- coding: utf-8 -*-
"""
Median spatial filter for 2D imagery.

Provides a configurable median filter operating on single-band raster
arrays. Implements the ImageFilter ABC from the image_processing domain.

Dependencies
------------
scipy

Author
------
<Your Name>
<your.email@example.com>

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2025-01-30

Modified
--------
2025-01-30
"""

# Standard library
from typing import Optional

# Third-party
import numpy as np
from scipy.ndimage import median_filter

# GRDL internal
from .base import ImageFilter


class MedianFilter(ImageFilter):
    ...
```

## Python Standards

This project follows PEP 8, PEP 257, and PEP 484 strictly.

### Naming Conventions (PEP 8)

- Directories and files: `snake_case`
- Classes: `PascalCase`
- Functions, methods, and variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private/internal names: prefix with `_`
- Protected members intended for subclass use: prefix with `_`

### Type Hints (PEP 484)

- All public function signatures must have full type annotations.
- Use `typing` module types where appropriate (`Optional`, `Union`, `Sequence`, etc.).
- Use `np.ndarray` for array parameters. Add shape and dtype expectations in the docstring.

### Docstrings (PEP 257 / NumPy Style)

All public modules, classes, and functions require docstrings. Use NumPy-style formatting:

```python
def apply_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply a spatial median filter to a 2D image array.

    Parameters
    ----------
    image : np.ndarray
        2D array of pixel values. Shape (rows, cols).
    kernel_size : int
        Side length of the square filter kernel in pixels. Must be odd.
        Default is 3.

    Returns
    -------
    np.ndarray
        Filtered image with same shape and dtype as input.

    Raises
    ------
    ValueError
        If kernel_size is even or image is not 2D.
    """
```

### Imports (PEP 8)

Imports are organized in three groups, separated by blank lines:

```python
# Standard library
import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple

# Third-party
import numpy as np
from scipy.ndimage import median_filter

# GRDL internal
from grdl.image_processing.base import ImageFilter
```

### What Not to Do

- Do not add docstrings or type hints to code you did not write or modify.
- Do not refactor adjacent code while fixing a bug. Stay scoped.
- Do not add logging, telemetry, or print statements unless the module's purpose requires it.
- Do not create utility grab-bag modules. If a helper doesn't belong to a domain, the domain is missing.

## Performance Optimization

Performance is critical for processing large geospatial imagery. Follow these guidelines:

### Vectorized Operations

- **Always prefer vectorized numpy operations over Python loops.** Vectorized code is 10-100x faster for array operations.
- Use numpy broadcasting and universal functions (ufuncs) wherever possible.
- Process data in batches rather than element-by-element.

**Bad (loops):**
```python
def transform_pixels(geo, rows, cols):
    lats = []
    lons = []
    for i in range(len(rows)):
        lat, lon, _ = geo.image_to_latlon(rows[i], cols[i])
        lats.append(lat)
        lons.append(lon)
    return np.array(lats), np.array(lons)
```

**Good (pass arrays directly -- the same method handles both):**
```python
def transform_pixels(geo, rows: np.ndarray, cols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lats, lons, _ = geo.image_to_latlon(rows, cols)
    return lats, lons
```

### Unified Scalar/Array Methods

- Public methods should accept scalar, list, or ndarray inputs and return matching types.
- Implement a single abstract method operating on arrays. The base class handles scalar/array dispatch.
- Do not provide separate `*_batch()` methods. One method handles both cases.

**Example:**
```python
class Geolocation(ABC):
    @abstractmethod
    def _image_to_latlon_array(
        self, rows: np.ndarray, cols: np.ndarray, height: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Subclasses implement vectorized array transform."""
        pass

    def image_to_latlon(
        self,
        row: Union[float, list, np.ndarray],
        col: Union[float, list, np.ndarray],
        height: float = 0.0
    ) -> Union[Tuple[float, float, float],
               Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Public method: scalar in → scalar out, array in → array out."""
        scalar = _is_scalar(row) and _is_scalar(col)
        rows_arr = _to_array(row)
        cols_arr = _to_array(col)
        lats, lons, heights = self._image_to_latlon_array(rows_arr, cols_arr, height)
        if scalar:
            return (float(lats[0]), float(lons[0]), float(heights[0]))
        return lats, lons, heights
```

### NumPy Best Practices

- Use `np.column_stack()`, `np.vstack()`, `np.hstack()` for array construction instead of loops
- Use array slicing and boolean indexing for filtering instead of loops
- Leverage `np.where()`, `np.select()` for conditional operations
- Use `np.apply_along_axis()` only as a last resort (still slower than pure vectorization)
- Pre-allocate arrays when size is known: `result = np.empty((n, m))` instead of appending

### Memory Efficiency

- Avoid unnecessary array copies. Use views where possible.
- Process data in chunks for very large arrays that don't fit in memory.
- Be explicit about dtypes to avoid unnecessary conversions: `dtype=np.float32` vs `dtype=np.float64`

### Performance Testing

- For performance-critical code, verify speedup with `%%timeit` in notebooks or `timeit` module
- Use `pytest-benchmark` for repeatable benchmarks: `pytest tests/test_benchmarks.py --benchmark-only`
- Skip benchmarks during normal test runs: `pytest tests/ --benchmark-disable`
- Target at least 10x speedup for vectorized versions over naive loops
- Document performance characteristics in docstrings when relevant (e.g., "< 1ms per transform")

## Dependency Management

### Source of Truth: `pyproject.toml`

**`pyproject.toml` is the single source of truth** for all dependencies. All package metadata, dependencies, and optional extras are defined here. This file drives PyPI publication and is read by build tools.

### Keeping Files in Sync

Three files must be kept synchronized:

| File | Purpose | How to Update |
|------|---------|---------------|
| `pyproject.toml` | **Source of truth** — package metadata, all dependencies, extras | Edit directly; this is the authoritative definition |
| `requirements.txt` (if it exists) | Development convenience — pinned versions for reproducible environments | `pip freeze > requirements.txt` after updating dependencies in `pyproject.toml` and installing |
| `.github/workflows/publish.yml` | PyPI publication — **DO NOT EDIT this file manually** (it extracts version from `pyproject.toml` automatically) | No action needed; the workflow reads `version` from `pyproject.toml` |

**Dependency workflow:**
1. Update dependencies in `pyproject.toml` (add new packages, change versions, create/rename extras)
2. Install dependencies: `pip install -e ".[all,dev]"` (or appropriate extras for your work)
3. If `requirements.txt` exists in this project, regenerate it: `pip freeze > requirements.txt`
4. Commit both files

### Publishing to PyPI

The publish workflow (`.github/workflows/publish.yml`) triggers on **GitHub Release creation** (`release: [published]`), **not** on tag push alone. Pushing a tag without creating a release will not publish.

**Release steps:**
1. Bump `version` in `pyproject.toml` (semantic versioning: `major.minor.patch`)
2. Commit and push:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to X.Y.Z"
   git push origin main
   ```
3. Create a git tag matching the version:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
4. Create a GitHub Release from the tag — this triggers the publish workflow:
   ```bash
   gh release create vX.Y.Z --title "vX.Y.Z" --notes "Release notes here"
   ```
5. Verify the workflow succeeded:
   ```bash
   gh run list --limit 1
   ```

The workflow builds wheels via `python -m build` and publishes to PyPI with OIDC trusted publishing (no API keys). Artifacts are available at [pypi.org/p/grdl](https://pypi.org/p/grdl).

### Dependency Rules

**Core** (always required):
- `numpy>=1.20.0` — common array type across all modules.
- `scipy>=1.7.0` — interpolation, ndimage filters, scientific computing.

**Optional** (install per module group):

| Extra | Packages | Modules |
|-------|----------|---------|
| `sar` | `sarpy`, `sarkit` | `grdl.IO.sar` (SICD, CPHD, CRSD, SIDD) |
| `eo` | `rasterio`, `glymur` | `grdl.IO.geotiff`, `grdl.IO.jpeg2000`, `grdl.IO.eo` |
| `hdf5` | `h5py` | `grdl.IO.hdf5` |
| `multispectral` | `h5py`, `xarray`, `spectral` | `grdl.IO.multispectral` |
| `ir` | `rasterio`, `h5py` | `grdl.IO.ir` |
| `biomass` | `rasterio`, `requests` | `grdl.IO.sar.biomass`, `grdl.IO.sar.biomass_catalog` |
| `geolocation` | `pyproj` | `grdl.geolocation.eo`, `grdl.geolocation.elevation` |
| `coregistration` | `opencv-python-headless` | `grdl.coregistration.feature_match` |
| `examples` | `matplotlib` | `grdl.example.*` |
| `all` | everything above | full installation |

**Dev** (testing & code quality): `pytest`, `pytest-cov`, `pytest-benchmark`, `ruff`, `black`, `mypy`.

**Environment setup** (conda-forge preferred):
```bash
conda env create -f environment.yml   # first time
conda activate grdl
pip install -e .
```

**Import rules:**
- All imports go at the top of the file. If a dependency is not installed, the module must fail immediately on import with a clear `ImportError`.
- **Optional dependencies** (e.g., `sarpy`, `rasterio`, `h5py`) may use a `try`/`except` guard at the module level to allow partial installation. In this case, the module must raise a clear `ImportError` with installation instructions at construction time (i.e., in `__init__`), not silently degrade. This pattern allows `import grdl` to succeed even when only a subset of optional dependencies are installed.
- Each module that requires dependencies beyond numpy must document them in the file header's Dependencies section.
- Never add dependencies for convenience. Prefer numpy over pulling in pandas for a single operation.

## Testing

- Tests live in `tests/` and follow the naming pattern `test_<domain>_<module>.py`.
- Use `pytest`. No `unittest.TestCase` subclasses.
- Test with small synthetic data, not real imagery files. If a test needs sample data, use `example_images/` or generate arrays inline.
- Each public function gets at least one test for the expected path and one for an error condition.
- ABC contracts should have tests that verify concrete subclasses satisfy the interface.

### Running Tests

```bash
pytest tests/ -v                               # Full suite
pytest tests/ -v --benchmark-disable           # Skip benchmarks
pytest tests/test_benchmarks.py --benchmark-only  # Benchmarks only
```

## Git Practices

- Commit messages: imperative mood, one line, under 72 characters. Body if needed.
- One logical change per commit. Do not mix unrelated changes.
- Branch naming: `<type>/<short-description>` (e.g. `feature/lee-filter`, `fix/coordinate-overflow`)

## Adding a New Module -- Checklist

1. Identify which domain directory it belongs to.
2. If the domain directory is new, create `base.py` with the ABC(s) defining the contract.
3. Create the concrete module file with the standard file header (encoding, docstring with title, dependencies, author, license, created/modified dates).
4. Place all imports at the top of the file, below the header. Fail fast if dependencies are missing.
5. Inherit from the domain's ABC. Implement all abstract methods.
6. **For ImageProcessor subclasses** (transforms, detectors, decompositions):
   - Add `@processor_version('x.y.z')` decorator.
   - Add `@processor_tags(modalities=[...], category=..., description='...')` decorator.
   - Declare tunable parameters as `Annotated` class-body fields with `Range`, `Options`, and `Desc` markers.
   - Use `self._resolve_params(kwargs)` in `apply()`/`detect()` to merge and validate parameters.
   - See the "Processor Metadata" section above for full details and examples.
7. Write full docstrings (NumPy style) and type hints on all public members.
8. Add exports to `__init__.py`.
9. Write tests in `tests/`.
10. Verify the module works in isolation: `from grdl.<domain>.<module> import <Thing>`.
