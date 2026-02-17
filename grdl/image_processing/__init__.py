# -*- coding: utf-8 -*-
"""
Image Processing Module - Geometric, radiometric, and SAR-specific transforms.

Provides interfaces and implementations for image processors including
dense raster transforms (orthorectification, filtering, enhancement),
polarimetric decompositions, SAR sub-aperture decomposition, and sparse
vector detectors. All processor types inherit from ``ImageProcessor``
which provides version checking, detection input flow, and tunable
parameter validation.

Sub-modules
-----------
ortho/
    Orthorectification from native acquisition geometry to geographic grids.
decomposition/
    Polarimetric decomposition of quad-pol SAR scattering matrices.
detection/
    Sparse geo-registered vector detections (points, bounding boxes).
sar/
    SAR-specific transforms requiring sensor metadata (sublook, etc.).

Key Classes
-----------
- ImageProcessor: Common base class for all processor types
- ImageTransform: ABC for dense raster transforms (ndarray -> ndarray)
- BandwiseTransformMixin: Auto-apply 2D transforms across 3D band stacks
- ImageDetector: ABC for sparse vector detectors (ndarray -> DetectionSet)
- Orthorectifier: Orthorectify imagery from native geometry to geographic grid
- OutputGrid: Specification for an orthorectified output grid
- PolarimetricDecomposition: ABC for polarimetric decomposition methods
- PauliDecomposition: Quad-pol Pauli basis decomposition
- SublookDecomposition: Sub-aperture spectral splitting of complex SAR imagery
- Detection, DetectionSet: Detection output data models
- FieldDefinition, Fields, DATA_DICTIONARY: Standardized data dictionary
- Pipeline: Sequential composition of ImageTransform steps
- processor_version: Version decorator for all processor types
- processor_tags: Capability metadata decorator (modalities, category)
- DetectionInputSpec: Declaration of detection inputs a processor accepts
- Range, Options, Desc: Annotated constraint markers for tunable parameters
- ParamSpec: Introspection data class for tunable parameters

Usage
-----
Orthorectify a BIOMASS SAR image to a regular geographic grid:

    >>> from grdl.IO import BIOMASSL1Reader
    >>> from grdl.geolocation.sar.gcp import GCPGeolocation
    >>> from grdl.image_processing import Orthorectifier, OutputGrid
    >>>
    >>> with BIOMASSL1Reader('path/to/product') as reader:
    ...     geo = GCPGeolocation(
    ...         reader.metadata['gcps'],
    ...         (reader.metadata['rows'], reader.metadata['cols']),
    ...     )
    ...     grid = OutputGrid.from_geolocation(geo, pixel_size_lat=0.001,
    ...                                        pixel_size_lon=0.001)
    ...     ortho = Orthorectifier(geo, grid)
    ...     result = ortho.apply_from_reader(reader, bands=[0])

Pauli decomposition of quad-pol SAR data:

    >>> from grdl.image_processing import PauliDecomposition
    >>> pauli = PauliDecomposition()
    >>> components = pauli.decompose(shh, shv, svh, svv)
    >>> rgb = pauli.to_rgb(components)

Sub-aperture (sublook) decomposition of complex SAR imagery:

    >>> from grdl.IO.sar import SICDReader
    >>> from grdl.image_processing import SublookDecomposition
    >>>
    >>> with SICDReader('image.nitf') as reader:
    ...     image = reader.read_full()
    ...     sublook = SublookDecomposition(reader.metadata, num_looks=3,
    ...                                    overlap=0.1)
    ...     looks = sublook.decompose(image)   # (3, rows, cols) complex
    ...     db = sublook.to_db(looks)          # (3, rows, cols) float

Dependencies
------------
scipy
torch (optional, for GPU-accelerated SAR transforms)

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
2026-02-10
"""

from grdl.image_processing.base import ImageProcessor, ImageTransform, BandwiseTransformMixin
from grdl.image_processing.ortho import Orthorectifier, OutputGrid
from grdl.image_processing.decomposition import (
    PolarimetricDecomposition,
    PauliDecomposition,
    DualPolHAlpha,
)
from grdl.image_processing.detection import (
    ImageDetector,
    Detection,
    DetectionSet,
    FieldDefinition,
    Fields,
    DATA_DICTIONARY,
)
from grdl.image_processing.sar import SublookDecomposition, MultilookDecomposition
from grdl.image_processing.intensity import ToDecibels, PercentileStretch
from grdl.image_processing.filters import (
    MeanFilter,
    GaussianFilter,
    MedianFilter,
    MinFilter,
    MaxFilter,
    StdDevFilter,
    LeeFilter,
    ComplexLeeFilter,
    PhaseGradientFilter,
)
from grdl.image_processing.pipeline import Pipeline
from grdl.image_processing.versioning import (
    processor_version,
    processor_tags,
    globalprocessor,
    DetectionInputSpec,
)
from grdl.image_processing.params import (
    Range,
    Options,
    Desc,
    ParamSpec,
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
    'DualPolHAlpha',
    'ImageDetector',
    'Detection',
    'DetectionSet',
    'FieldDefinition',
    'Fields',
    'DATA_DICTIONARY',
    'SublookDecomposition',
    'MultilookDecomposition',
    'ToDecibels',
    'PercentileStretch',
    'MeanFilter',
    'GaussianFilter',
    'MedianFilter',
    'MinFilter',
    'MaxFilter',
    'StdDevFilter',
    'LeeFilter',
    'ComplexLeeFilter',
    'PhaseGradientFilter',
    'Pipeline',
    'processor_version',
    'processor_tags',
    'globalprocessor',
    'DetectionInputSpec',
    'Range',
    'Options',
    'Desc',
    'ParamSpec',
    'ImageModality',
    'ProcessorCategory',
    'DetectionType',
    'SegmentationType',
]
