# GRDL Development Guide

## Project Overview

GRDL (GEOINT Rapid Development Library) is a modular Python library for geospatial intelligence workflows. It operates on any 2D correlated imagery -- SAR, EO, MSI, hyperspectral, space-based, or terrestrial.

This is a **library, not a framework**. Every module must be independently usable.

## Development Environment

**Python Environment:** Always use the `starlight` conda environment for all Python operations (testing, development, package installation).

```bash
conda activate starlight
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
| Load any imagery format | `grdl.IO` readers | Raw `rasterio.open()` / `h5py.File()` |
| Plan chip/tile regions | `grdl.data_prep.ChipExtractor` or `Tiler` | Hand-rolled `for r in range(0, rows, sz):` loops |
| Normalize for ML | `grdl.data_prep.Normalizer` | Inline min-max arithmetic |
| Pixel to lat/lon | `grdl.geolocation` | Manual GCP interpolation |
| Spatial filters, contrast | `grdl.imagej` | Re-implementing algorithms |
| SAR decomposition | `grdl.image_processing` | Manual complex arithmetic |
| Image alignment | `grdl.coregistration` | Custom OpenCV wrappers |

Modules handle edge cases (boundary snapping, band indexing, lazy loading, resource cleanup) that ad-hoc code misses. **Compose them at the application level** — each module does its job, the application wires them together. See `grdl/example/image_processing/sar/sublook_compare.py` for a full integration example.

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

### Directory Structure

```
GRDL/
  grdl/
    exceptions.py            # Custom exception hierarchy (GrdlError → ValidationError, ProcessorError, etc.)
    py.typed                 # PEP 561 type stub marker
    IO/                      # Input/Output — format readers and writers
      base.py                # ImageReader / ImageWriter / CatalogInterface ABCs
      geotiff.py             # GeoTIFFReader (rasterio), open_image()
      hdf5.py                # HDF5Reader (h5py)
      jpeg2000.py            # JP2Reader (glymur)
      nitf.py                # NITFReader (rasterio/GDAL)
      models/                # Typed metadata dataclasses
        base.py              # ImageMetadata base class
        common.py            # Shared primitives (XYZ, LatLonHAE, Poly2D, ...)
        sicd.py              # SICDMetadata (~35 nested dataclasses)
        sidd.py              # SIDDMetadata (~25 nested dataclasses)
        biomass.py           # BIOMASSMetadata
        viirs.py             # VIIRSMetadata
        aster.py             # ASTERMetadata
      sar/                   # SAR modality submodule
        _backend.py          # sarkit/sarpy availability
        sicd.py, cphd.py     # SICDReader, CPHDReader (sarkit/sarpy)
        crsd.py, sidd.py     # CRSDReader, SIDDReader (sarkit)
        biomass.py           # BIOMASSL1Reader
        biomass_catalog.py   # BIOMASSCatalog
      ir/                    # IR/thermal modality submodule
        _backend.py          # rasterio/h5py availability
        aster.py             # ASTERReader (L1T, GDEM)
      multispectral/         # Multispectral modality submodule
        _backend.py          # h5py/xarray/spectral availability
        viirs.py             # VIIRSReader
      eo/                    # EO modality submodule (scaffold)
        _backend.py          # rasterio/glymur availability
    <domain>/                # Other top-level module areas
      base.py                # ABCs defining the domain's contracts
      <submodule>.py         # Concrete implementations
      __init__.py            # Expose public API
      <subdomain>/           # Nested subdomains (e.g. detection/, ortho/, decomposition/)
    imagej/                  # ImageJ/Fiji ports (organized by ImageJ menu category)
      __init__.py            # Barrel re-exports all 12 components (backward compat)
      _taxonomy.py           # Category constants shared with GRDK
      filters/               # Process > Filters (RankFilters, UnsharpMask)
      background/            # Process > Subtract Background (RollingBallBackground)
      binary/                # Process > Binary (MorphologicalFilter)
      enhance/               # Process > Enhance Contrast (CLAHE, GammaCorrection)
      edges/                 # Process > Find Edges (EdgeDetector)
      fft/                   # Process > FFT (FFTBandpassFilter)
      find_maxima/           # Process > Find Maxima (FindMaxima)
      threshold/             # Image > Adjust > Threshold (AutoLocalThreshold)
      segmentation/          # Plugins > Segmentation (StatisticalRegionMerging)
      stacks/                # Image > Stacks (ZProjection)
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
| `IO/` | Format readers and writers (base formats + `sar/`, `ir/`, `multispectral/`, `eo/` modality submodules) |
| `IO/models/` | Typed metadata dataclasses (`SICDMetadata`, `SIDDMetadata`, `BIOMASSMetadata`, `VIIRSMetadata`, `ASTERMetadata`) |
| `geolocation/` | Pixel-to-geographic coordinate transforms |
| `image_processing/` | Orthorectification, polarimetric decomposition, SAR sublook, detection, versioning, pipeline, transforms |
| `imagej/` | ImageJ/Fiji algorithm ports -- 12 classic algorithms in 10 subdirectories matching ImageJ menu hierarchy |
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
        lat, lon, _ = geo.pixel_to_latlon(rows[i], cols[i])
        lats.append(lat)
        lons.append(lon)
    return np.array(lats), np.array(lons)
```

**Good (pass arrays directly -- the same method handles both):**
```python
def transform_pixels(geo, rows: np.ndarray, cols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lats, lons, _ = geo.pixel_to_latlon(rows, cols)
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
    def _pixel_to_latlon_array(
        self, rows: np.ndarray, cols: np.ndarray, height: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Subclasses implement vectorized array transform."""
        pass

    def pixel_to_latlon(
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
        lats, lons, heights = self._pixel_to_latlon_array(rows_arr, cols_arr, height)
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

## Dependencies

- `numpy` is always available. It is the common data type across all modules.
- All imports go at the top of the file. If a dependency is not installed, the module must fail immediately on import with a clear `ImportError`.
- **Optional dependencies** (e.g., `sarpy`, `rasterio`, `scipy`) may use a `try`/`except` guard at the module level to allow partial installation. In this case, the module must raise a clear `ImportError` with installation instructions at construction time (i.e., in `__init__`), not silently degrade. This pattern allows `import grdl` to succeed even when only a subset of optional dependencies are installed.
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
6. Write full docstrings (NumPy style) and type hints on all public members.
7. Add exports to `__init__.py`.
8. Write tests in `tests/`.
9. Verify the module works in isolation: `from grdl.<domain>.<module> import <Thing>`.
