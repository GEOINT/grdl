# GRDL Development Guide

## Project Overview

GRDL (GEOINT Rapid Development Library) is a modular Python library for geospatial intelligence workflows. It operates on any 2D correlated imagery -- SAR, EO, MSI, hyperspectral, space-based, or terrestrial.

This is a **library, not a framework**. Every module must be independently usable.

## Architecture Rules

### Module Design

- One concept per module. If a module does two things, split it.
- Modularity is enforced through **abstract base classes**. Each domain defines ABCs that establish the contract. Concrete implementations inherit from them.
- Sensor-agnostic by default. Sensor-specific logic (SAR, EO, MSI) goes in dedicated submodules under its sensor directory, implementing the domain's ABC.
- No global state. No singletons. No module-level side effects on import.
- All public functions and classes operate on numpy arrays or standard Python types as their primary data interface.

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
  <domain>/                  # Top-level module area (e.g. image_processing/)
    base.py                  # ABCs defining the domain's contracts
    <submodule>.py           # Concrete implementations
    __init__.py              # Expose public API
  tests/
    test_<domain>_<module>.py
  example_images/            # Small sample data for tests and demos
```

Domain directories map to the module areas defined in the README:

| Directory | Domain |
|-----------|--------|
| `image_processing/` | Filtering, enhancement, normalization, transforms |
| `geospatial/` | Coordinate conversions, projections, pixel-to-geo mapping |
| `data_prep/` | Chunking, tiling, resampling, ML pipeline formatting |
| `sensors/` | Sensor-specific operations (subdirs: `sar/`, `eo/`, `msi/`) |
| `ml/` | Feature extraction, annotation, dataset builders |
| `io/` | Format readers and writers |

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
Duane Smalley, PhD
duane.d.smalley@gmail.com

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
- **Author**: `Duane Smalley, PhD` and `duane.d.smalley@gmail.com` for all original work. Additional contributors append their name and email on new lines.
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
Duane Smalley, PhD
duane.d.smalley@gmail.com

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

## Dependencies

- `numpy` is always available. It is the common data type across all modules.
- All imports go at the top of the file. If a dependency is not installed, the module must fail immediately on import with a clear `ImportError`.
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
pytest tests/ -v
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
