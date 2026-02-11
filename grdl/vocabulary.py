# -*- coding: utf-8 -*-
"""
Vocabulary - Canonical enums for the GRDL framework.

Defines the single source of truth for controlled vocabularies used across
grdl, grdl-runtime, and grdk: image modalities, processor categories,
detection types, and segmentation types.  All three packages import from
this module so that tag values are guaranteed consistent and typo-free.

Author
------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-10

Modified
--------
2026-02-10
"""

from enum import Enum


class ImageModality(Enum):
    """Supported image modalities for processor and workflow tagging.

    Processors and workflows are tagged with one or more modalities
    indicating what types of imagery they are designed for.
    """

    PAN = "PAN"
    SAR = "SAR"
    MSI = "MSI"
    HSI = "HSI"
    SWIR = "SWIR"
    MWIR = "MWIR"
    LWIR = "LWIR"
    EO = "EO"
    LIDAR = "LIDAR"
    FMV = "FMV"


class ProcessorCategory(Enum):
    """Processing categories for processor tagging.

    Each value corresponds to a functional grouping of image processing
    operations.
    """

    FILTERS = "filters"
    BACKGROUND = "background"
    BINARY = "binary"
    ENHANCE = "enhance"
    EDGES = "edges"
    FFT = "fft"
    FIND_MAXIMA = "find_maxima"
    THRESHOLD = "threshold"
    SEGMENTATION = "segmentation"
    STACKS = "stacks"
    MATH = "math"
    ANALYZE = "analyze"
    NOISE = "noise"


class DetectionType(Enum):
    """Type/fidelity of detection a detector processor performs.

    Used to describe the output characteristics of ``ImageDetector``
    subclasses at the component level.
    """

    PHENOMENON_SIGNATURE = "phenomenon_signature"
    CHARACTERIZATION = "characterization"
    CLASSIFICATION = "classification"


class SegmentationType(Enum):
    """Type of segmentation a segmentor processor produces.

    Used to describe the output characteristics of segmentation
    processors at the component level.
    """

    INSTANCE = "instance"
    SEMANTIC = "semantic"
    PANOPTIC = "panoptic"


class ExecutionPhase(Enum):
    """Pipeline execution phases for processor and workflow tagging.

    Each value corresponds to a distinct stage in the grdl-runtime
    execution pipeline.  Processors tagged with one or more phases
    are restricted to those stages.
    """

    IO = "io"
    GLOBAL_PROCESSING = "global_processing"
    DATA_PREP = "data_prep"
    TILING = "tiling"
    TILE_PROCESSING = "tile_processing"
    EXTRACTION = "extraction"
    VECTOR_PROCESSING = "vector_processing"
    FINALIZATION = "finalization"


class OutputFormat(Enum):
    """Supported output file formats for write configuration.

    Used by tap-out nodes and write steps to select the appropriate
    ``ImageWriter`` implementation.
    """

    GEOTIFF = "geotiff"
    NUMPY = "numpy"
    PNG = "png"
    HDF5 = "hdf5"
    NITF = "nitf"
