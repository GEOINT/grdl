# -*- coding: utf-8 -*-
"""
Image Processing Module - Geometric and radiometric transforms for imagery.

Provides interfaces and implementations for image processors including
dense raster transforms (orthorectification, filtering, enhancement),
polarimetric decompositions, and sparse vector detectors. All processor
types inherit from ``ImageProcessor`` which provides version checking
and detection input flow.

Key Classes
-----------
- ImageProcessor: Common base class for all processor types
- ImageTransform: ABC for dense raster transforms (ndarray -> ndarray)
- ImageDetector: ABC for sparse vector detectors (ndarray -> DetectionSet)
- Orthorectifier: Orthorectify imagery from native geometry to geographic grid
- OutputGrid: Specification for an orthorectified output grid
- PolarimetricDecomposition: ABC for polarimetric decomposition methods
- PauliDecomposition: Quad-pol Pauli basis decomposition
- Detection, DetectionSet, Geometry: Detection output data models
- OutputField, OutputSchema: Self-declared output format declarations
- processor_version: Version decorator for all processor types
- DetectionInputSpec: Declaration of detection inputs a processor accepts
- TunableParameterSpec: Declaration of tunable parameters a processor accepts

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
2026-02-06
"""

from grdl.image_processing.base import ImageProcessor, ImageTransform, BandwiseTransformMixin
from grdl.image_processing.ortho import Orthorectifier, OutputGrid
from grdl.image_processing.decomposition import (
    PolarimetricDecomposition,
    PauliDecomposition,
)
from grdl.image_processing.detection import (
    ImageDetector,
    Detection,
    DetectionSet,
    Geometry,
    OutputField,
    OutputSchema,
)
from grdl.image_processing.pipeline import Pipeline
from grdl.image_processing.versioning import (
    processor_version,
    processor_tags,
    DetectionInputSpec,
    TunableParameterSpec,
)
from grdl.vocabulary import (
    ImageModality,
    ProcessorCategory,
    DetectionType,
    SegmentationType,
)

__all__ = [
    'ImageProcessor',
    'ImageTransform',
    'BandwiseTransformMixin',
    'Orthorectifier',
    'OutputGrid',
    'PolarimetricDecomposition',
    'PauliDecomposition',
    'ImageDetector',
    'Detection',
    'DetectionSet',
    'Geometry',
    'OutputField',
    'OutputSchema',
    'Pipeline',
    'processor_version',
    'processor_tags',
    'DetectionInputSpec',
    'TunableParameterSpec',
    'ImageModality',
    'ProcessorCategory',
    'DetectionType',
    'SegmentationType',
]
