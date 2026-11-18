# -*- coding: utf-8 -*-
"""
Image Processing Module - Core Framework & Modality-Agnostic Processors.

Provides base processor abstractions (``ImageProcessor``, ``ImageTransform``,
``ImageDetector``, ``PolarimetricDecomposition``, ``ImageFormationAlgorithm``),
parameter & versioning decorators, generic spatial filters, intensity
transforms, orthorectification, and sequential ``Pipeline`` composition.

Modality-specific processor catalogs live in ``grdl-sar`` and ``grdl-eo``
and are lazily re-exported here for backwards compatibility.

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.
"""

from importlib import import_module
import warnings

from grdl.image_processing._compat import install_shim_finder

install_shim_finder()

from grdl.image_processing.base import (
    BandwiseTransformMixin,
    ImageProcessor,
    ImageTransform,
)
from grdl.image_processing.decomposition.base import PolarimetricDecomposition
from grdl.image_processing.detection import (
    DATA_DICTIONARY,
    Detection,
    DetectionSet,
    FieldDefinition,
    Fields,
    ImageDetector,
)
from grdl.image_processing.filters import (
    GaussianFilter,
    MaxFilter,
    MeanFilter,
    MedianFilter,
    MinFilter,
    StdDevFilter,
)
from grdl.image_processing.intensity import PercentileStretch, ToDecibels
from grdl.image_processing.ortho import (
    GeographicGrid,
    OrthoBuilder,
    Orthorectifier,
    OrthoResult,
    OutputGrid,
    OutputGridProtocol,
    compute_output_resolution,
)
from grdl.image_processing.params import Desc, Options, ParamSpec, Range
from grdl.image_processing.pipeline import Pipeline
from grdl.image_processing.sar.image_formation.base import ImageFormationAlgorithm
from grdl.image_processing.versioning import (
    DetectionInputSpec,
    globalprocessor,
    processor_tags,
    processor_version,
)
from grdl.vocabulary import (
    DetectionType,
    ImageModality,
    ProcessorCategory,
    SegmentationType,
)

_SAR_EXPORTS = {
    "PauliDecomposition",
    "DualPolHAlpha",
    "CompactPolDecompositionBase",
    "CompactPolDegreeOfPolarization",
    "CompactPolModelFree3C",
    "CompactPolSOmega",
    "CompactPolMChi",
    "CompactPolMDelta",
    "SublookDecomposition",
    "MultilookDecomposition",
    "CSIProcessor",
    "LeeFilter",
    "ComplexLeeFilter",
    "PhaseGradientFilter",
    "FrostFilter",
    "KuanFilter",
    "BoxcarFilter",
    "CACFARDetector",
    "GOCFARDetector",
    "SOCFARDetector",
    "OSCFARDetector",
    "PolarFormatAlgorithm",
    "StripmapPFA",
    "RangeDopplerAlgorithm",
    "FastBackProjection",
}

_EO_EXPORTS = {
    "SobelFilter",
    "LaplacianFilter",
    "UnsharpMaskFilter",
    "HighPassFilter",
    "BroveyPanSharpenTransform",
    "SimpleMeanFusionTransform",
    "brovey_pan_sharpen",
    "simple_mean_fusion",
    "SpectralIndexTransform",
    "compute_spectral_index",
}


def __getattr__(name: str):
    if name in _SAR_EXPORTS:
        warnings.warn(
            f"grdl.image_processing.{name} is deprecated and has moved to grdl-sar; "
            f"import from grdl_sar instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            return getattr(import_module("grdl_sar"), name)
        except ImportError as exc:
            raise ImportError(
                f"{name} has moved to grdl-sar. Install with: pip install grdl-sar"
            ) from exc
    if name in _EO_EXPORTS:
        warnings.warn(
            f"grdl.image_processing.{name} is deprecated and has moved to grdl-eo; "
            f"import from grdl_eo instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            return getattr(import_module("grdl_eo"), name)
        except ImportError as exc:
            raise ImportError(
                f"{name} has moved to grdl-eo. Install with: pip install grdl-eo"
            ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ImageProcessor",
    "ImageTransform",
    "BandwiseTransformMixin",
    "Orthorectifier",
    "GeographicGrid",
    "OutputGrid",
    "OutputGridProtocol",
    "OrthoBuilder",
    "OrthoResult",
    "compute_output_resolution",
    "PolarimetricDecomposition",
    "ImageDetector",
    "Detection",
    "DetectionSet",
    "FieldDefinition",
    "Fields",
    "DATA_DICTIONARY",
    "ImageFormationAlgorithm",
    "ToDecibels",
    "PercentileStretch",
    "MeanFilter",
    "GaussianFilter",
    "MedianFilter",
    "MinFilter",
    "MaxFilter",
    "StdDevFilter",
    "Pipeline",
    "processor_version",
    "processor_tags",
    "globalprocessor",
    "DetectionInputSpec",
    "Range",
    "Options",
    "Desc",
    "ParamSpec",
    "ImageModality",
    "ProcessorCategory",
    "DetectionType",
    "SegmentationType",
]
