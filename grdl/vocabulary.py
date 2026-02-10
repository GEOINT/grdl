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
    operations (and, for ImageJ ports, to the subdirectory under
    ``grdl/imagej/``).
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
