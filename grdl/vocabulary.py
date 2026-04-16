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
    DECOMPOSITION = "decomposition"
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


class GpuCapability(Enum):
    """GPU capability declaration for processor tagging.

    Declares a processor's relationship with GPU hardware.  Set via
    ``@processor_tags(gpu_capability=GpuCapability.PREFERRED)`` and read
    by the execution resolver to plan hardware-aware execution paths.

    Intended to replace the legacy ``__gpu_compatible__`` class attribute.
    """

    REQUIRED = "required"
    """Processor cannot run without a GPU (e.g., a PyTorch model)."""

    PREFERRED = "preferred"
    """Processor benefits from GPU but has a CPU fallback path."""

    CPU_ONLY = "cpu_only"
    """Processor uses CPU-only libraries (e.g., scipy) and cannot use GPU."""


class BandExpansion(Enum):
    """Strategy for expanding fewer input bands to match a processor's requirement.

    Used by grdl-runtime's band adaptation system when the incoming image
    has fewer bands than a processor declares via
    ``@processor_tags(required_bands=N)``.
    """

    REPEAT = "repeat"
    """Tile available bands cyclically (e.g., 1-band → [B, B, B])."""

    ZERO_PAD = "zero_pad"
    """Fill missing bands with zeros."""


class BandReduction(Enum):
    """Strategy for reducing excess input bands to match a processor's requirement.

    Used by grdl-runtime's band adaptation system when the incoming image
    has more bands than a processor declares via
    ``@processor_tags(required_bands=N)``.
    """

    FIRST_N = "first_n"
    """Take the first N bands."""

    MEAN = "mean"
    """Average across all bands."""

    MEDIAN = "median"
    """Median across all bands."""

    MAX = "max"
    """Maximum across all bands."""

    MIN = "min"
    """Minimum across all bands."""

    LUMINANCE = "luminance"
    """Weighted luminance: 0.2126*R + 0.7152*G + 0.0722*B (3→1 only)."""


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


class PolarimetricMode(Enum):
    """Polarimetric collection mode for SAR processor and workflow tagging.

    Declares the polarimetric diversity required by a processor or workflow.
    Used by grdk widgets to gate actions on quad-pol completeness and by
    grdl-runtime to validate workflow–dataset compatibility.
    """

    SINGLE_POL = "single_pol"
    """Single polarization (e.g., HH-only, VV-only)."""

    DUAL_POL = "dual_pol"
    """Dual polarization (co-pol + cross-pol, e.g., HH+HV or VV+VH)."""

    COMPACT_POL = "compact_pol"
    """Compact/hybrid polarimetry (e.g., Pi/4, CTLR modes)."""

    QUAD_POL = "quad_pol"
    """Full quad-polarization: HH, HV, VH, VV simultaneously."""

    @classmethod
    def from_reader(cls, reader) -> 'PolarimetricMode | None':
        """Classify the polarimetric mode of *reader* from its metadata.

        Inspects ``reader.metadata.channel_metadata`` for multi-band
        readers (e.g. NISAR opened with ``polarizations='all'``, BIOMASS)
        and ``reader.metadata.polarization`` for single-band readers.
        No array data is loaded.

        Parameters
        ----------
        reader : ImageReader
            Any grdl reader instance.

        Returns
        -------
        PolarimetricMode or None
            - ``QUAD_POL`` — all four HH/HV/VH/VV channels present.
            - ``DUAL_POL`` — exactly two channels from a recognised pair.
            - ``SINGLE_POL`` — one polarization channel detected.
            - ``None`` — no polarization metadata found.
        """
        _QUAD = frozenset({'HH', 'HV', 'VH', 'VV'})
        _DUAL_PAIRS = (
            frozenset({'HH', 'HV'}),
            frozenset({'VV', 'VH'}),
            frozenset({'HH', 'VV'}),
            frozenset({'HV', 'VH'}),
        )
        meta = getattr(reader, 'metadata', None)
        channel_metadata = getattr(meta, 'channel_metadata', None)
        if channel_metadata:
            pols = frozenset(
                ch.polarization.strip().upper()
                for ch in channel_metadata
                if isinstance(getattr(ch, 'polarization', None), str)
                and ch.polarization.strip()
            )
            if _QUAD.issubset(pols):
                return cls.QUAD_POL
            if len(pols) == 2 and pols in _DUAL_PAIRS:
                return cls.DUAL_POL
            if len(pols) == 1:
                return cls.SINGLE_POL
            return None

        pol = getattr(meta, 'polarization', None)
        if isinstance(pol, str) and pol.strip():
            return cls.SINGLE_POL
        return None


class DataType(str, Enum):
    """Logical data type flowing through a processing pipeline.

    Used to tag edges and ports in grdl-runtime workflow graphs so
    the executor knows which processor family to dispatch to.
    """

    RASTER = "raster"
    FEATURE_SET = "feature_set"
    DETECTION_SET = "detection_set"
    METADATA = "metadata"
