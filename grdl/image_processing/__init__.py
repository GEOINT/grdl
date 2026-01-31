# -*- coding: utf-8 -*-
"""
Image Processing Module - Geometric and radiometric transforms for imagery.

Provides interfaces and implementations for image transforms including
orthorectification (geometric reprojection to ground-referenced grids),
filtering, enhancement, and normalization.

Key Classes
-----------
- ImageTransform: Abstract base class for all image transforms
- Orthorectifier: Orthorectify imagery from native geometry to geographic grid
- OutputGrid: Specification for an orthorectified output grid
- PolarimetricDecomposition: ABC for polarimetric decomposition methods
- PauliDecomposition: Quad-pol Pauli basis decomposition

Usage
-----
Orthorectify a BIOMASS SAR image to a regular geographic grid:

    >>> from grdl.IO import BIOMASSL1Reader
    >>> from grdl.geolocation import Geolocation
    >>> from grdl.image_processing import Orthorectifier, OutputGrid
    >>>
    >>> with BIOMASSL1Reader('path/to/product') as reader:
    >>>     geo = Geolocation.from_reader(reader)
    >>>     grid = OutputGrid.from_geolocation(geo, pixel_size_lat=0.001,
    ...                                        pixel_size_lon=0.001)
    >>>     ortho = Orthorectifier(geo, grid)
    >>>     result = ortho.apply_from_reader(reader, bands=[0])

Pauli decomposition of quad-pol SAR data:

    >>> from grdl.image_processing import PauliDecomposition
    >>> pauli = PauliDecomposition()
    >>> components = pauli.decompose(shh, shv, svh, svv)
    >>> rgb = pauli.to_rgb(components)

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
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-01-30

Modified
--------
2026-01-30
"""

from grdl.image_processing.base import ImageTransform
from grdl.image_processing.ortho import Orthorectifier, OutputGrid
from grdl.image_processing.decomposition import (
    PolarimetricDecomposition,
    PauliDecomposition,
)

__all__ = [
    'ImageTransform',
    'Orthorectifier',
    'OutputGrid',
    'PolarimetricDecomposition',
    'PauliDecomposition',
]
