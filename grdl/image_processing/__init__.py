# -*- coding: utf-8 -*-
"""
Image Processing Module - Transforms, detectors, and decompositions.

Provides interfaces and implementations for image processors spanning
dense raster transforms (orthorectification, filtering, enhancement),
polarimetric decompositions, SAR sub-aperture decomposition, CFAR
target detection, and composable pipelines. All processor types inherit
from ``ImageProcessor`` which provides version checking, tunable
parameter validation, and metadata management.

Sub-modules
-----------
filters/
    Spatial image filters -- linear (mean, Gaussian), rank (median, min,
    max), statistical (std-dev), adaptive SAR speckle (Lee, complex Lee),
    and phase gradient. All auto-handle 3D band stacks via
    ``BandwiseTransformMixin``.
intensity.py
    Radiometric transforms -- ``ToDecibels``, ``PercentileStretch``.
ortho/
    Orthorectification from native acquisition geometry to geographic grids.
    ``OrthoPipeline`` builder with ROI, tiled processing, auto-resolution,
    and DEM terrain correction. ``compute_output_resolution`` dispatches
    on metadata type (SICD, BIOMASS).
decomposition/
    Polarimetric decompositions -- quad-pol Pauli, dual-pol H/Alpha.
detection/
    Sparse geo-registered vector detections and CFAR detector family
    (CA, GO, SO, OS).
sar/
    SAR-specific transforms requiring ``SICDMetadata`` -- 1D sublook,
    2D multilook, and image formation algorithms (PFA, RDA, FFBP).
pipeline.py
    Sequential composition of ``ImageTransform`` steps.
versioning.py
    ``@processor_version`` and ``@processor_tags`` decorators.
params.py
    ``Range``, ``Options``, ``Desc`` constraint markers for tunable
    parameters via ``Annotated`` type hints.

Key Classes
-----------
Base infrastructure:
    ``ImageProcessor``, ``ImageTransform``, ``BandwiseTransformMixin``,
    ``Pipeline``, ``processor_version``, ``processor_tags``,
    ``Range``, ``Options``, ``Desc``, ``ParamSpec``

Filters:
    ``MeanFilter``, ``GaussianFilter``, ``MedianFilter``, ``MinFilter``,
    ``MaxFilter``, ``StdDevFilter``, ``LeeFilter``, ``ComplexLeeFilter``,
    ``PhaseGradientFilter``

Intensity:
    ``ToDecibels``, ``PercentileStretch``

Ortho:
    ``OrthoPipeline`` (recommended), ``OrthoResult``, ``Orthorectifier``,
    ``OutputGrid``, ``compute_output_resolution``

Decomposition:
    ``PolarimetricDecomposition`` (ABC), ``PauliDecomposition``,
    ``DualPolHAlpha``

Detection:
    ``ImageDetector`` (ABC), ``Detection``, ``DetectionSet``,
    ``CFARDetector`` (ABC), ``CACFARDetector``, ``GOCFARDetector``,
    ``SOCFARDetector``, ``OSCFARDetector``

SAR:
    ``SublookDecomposition``, ``MultilookDecomposition``,
    ``ImageFormationAlgorithm`` (ABC), ``PolarFormatAlgorithm``,
    ``StripmapPFA``, ``RangeDopplerAlgorithm``, ``FastBackProjection``

Usage
-----
Orthorectify a SICD image via OrthoPipeline (recommended):

    >>> from grdl.IO.sar import SICDReader
    >>> from grdl.geolocation.sar.sicd import SICDGeolocation
    >>> from grdl.image_processing import OrthoPipeline
    >>>
    >>> with SICDReader('image.nitf') as reader:
    ...     geo = SICDGeolocation.from_reader(reader)
    ...     result = (
    ...         OrthoPipeline()
    ...         .with_reader(reader)
    ...         .with_geolocation(geo)
    ...         .with_metadata(reader.metadata)       # auto-resolution
    ...         .with_interpolation('nearest')
    ...         .run()
    ...     )
    ...     # result.data, result.output_grid

Ortho with ROI (geographic sub-region) and tiling (memory-efficient):

    >>> result = (
    ...     OrthoPipeline()
    ...     .with_source_array(mag)
    ...     .with_geolocation(geo)
    ...     .with_resolution(0.001, 0.001)
    ...     .with_roi(36.0, 36.1, -75.8, -75.7)    # geographic subset
    ...     .with_tile_size(2048)                    # bounded mapping memory
    ...     .with_elevation(dem)                     # DEM terrain correction
    ...     .run()
    ... )

Pauli decomposition of quad-pol SAR data:

    >>> from grdl.image_processing import PauliDecomposition
    >>> pauli = PauliDecomposition()
    >>> components = pauli.decompose(shh, shv, svh, svv)
    >>> rgb = pauli.to_rgb(components)

Sub-aperture decomposition of complex SAR imagery:

    >>> from grdl.IO.sar import SICDReader
    >>> from grdl.image_processing import SublookDecomposition
    >>>
    >>> with SICDReader('image.nitf') as reader:
    ...     image = reader.read_full()
    ...     sublook = SublookDecomposition(reader.metadata, num_looks=3,
    ...                                    overlap=0.1)
    ...     looks = sublook.decompose(image)   # (3, rows, cols) complex

2D multi-look decomposition into M x N sub-aperture grid:

    >>> from grdl.image_processing import MultilookDecomposition
    >>>
    >>> with SICDReader('image.nitf') as reader:
    ...     image = reader.read_full()
    ...     ml = MultilookDecomposition(reader.metadata,
    ...                                looks_rg=3, looks_az=3)
    ...     grid = ml.decompose(image)     # (3, 3, rows, cols) complex
    ...     flat = ml.to_flat_stack(grid)  # (9, rows, cols) complex

CFAR target detection:

    >>> from grdl.image_processing.detection import CACFARDetector
    >>>
    >>> detector = CACFARDetector(guard_cells=4, training_cells=16, pfa=1e-6)
    >>> detections = detector.detect(intensity_image, geolocation=geo)

See ``architecture.md`` in this directory for the full class hierarchy
and design rationale.

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
2026-02-18
"""

from grdl.image_processing.base import ImageProcessor, ImageTransform, BandwiseTransformMixin
from grdl.image_processing.ortho import (
    Orthorectifier, OutputGrid, OrthoPipeline, OrthoResult,
    compute_output_resolution,
)
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
from grdl.image_processing.sar import SublookDecomposition, MultilookDecomposition, CSIProcessor
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
    'OrthoPipeline',
    'OrthoResult',
    'compute_output_resolution',
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
    'CSIProcessor',
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
