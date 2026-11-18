"""
Spatial Filters - Core linear, rank, and statistical filters, plus
compatibility re-exports from ``grdl-sar`` and ``grdl-eo``.
"""

from importlib import import_module
from typing import TYPE_CHECKING
import warnings

from grdl.image_processing.filters.linear import GaussianFilter, MeanFilter
from grdl.image_processing.filters.rank import MaxFilter, MedianFilter, MinFilter
from grdl.image_processing.filters.statistical import StdDevFilter

if TYPE_CHECKING:
    from grdl_eo.filters.edge import (
        HighPassFilter,
        LaplacianFilter,
        SobelFilter,
        UnsharpMaskFilter,
    )
    from grdl_sar.filters.enhanced_lee import EnhancedLeeFilter
    from grdl_sar.filters.idan import IDANFilter
    from grdl_sar.filters.lee_sigma import LeeSigmaFilter
    from grdl_sar.filters.phase import PhaseGradientFilter
    from grdl_sar.filters.refined_lee import RefinedLeeFilter
    from grdl_sar.filters.sar_base import SARFilter
    from grdl_sar.filters.sar_speckle import BoxcarFilter, FrostFilter, KuanFilter
    from grdl_sar.filters.standard_lee import ComplexLeeFilter, LeeFilter
    from grdl_sar.filters.wishart import WishartFilter

_EO_FILTERS = {
    "SobelFilter",
    "LaplacianFilter",
    "UnsharpMaskFilter",
    "HighPassFilter",
}
_SAR_FILTERS = {
    "SARFilter",
    "LeeFilter",
    "ComplexLeeFilter",
    "EnhancedLeeFilter",
    "LeeSigmaFilter",
    "RefinedLeeFilter",
    "IDANFilter",
    "PhaseGradientFilter",
    "WishartFilter",
    "FrostFilter",
    "KuanFilter",
    "BoxcarFilter",
}


def __getattr__(name: str):
    if name in _EO_FILTERS:
        warnings.warn(
            f"grdl.image_processing.filters.{name} is deprecated and has moved to grdl-eo; "
            "import from grdl_eo.filters instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            return getattr(import_module("grdl_eo.filters"), name)
        except ImportError as exc:
            raise ImportError(
                f"{name} has moved to grdl-eo. Install with: pip install grdl-eo"
            ) from exc
    if name in _SAR_FILTERS:
        warnings.warn(
            f"grdl.image_processing.filters.{name} is deprecated and has moved to grdl-sar; "
            "import from grdl_sar.filters instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            return getattr(import_module("grdl_sar.filters"), name)
        except ImportError as exc:
            raise ImportError(
                f"{name} has moved to grdl-sar. Install with: pip install grdl-sar"
            ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MeanFilter",
    "GaussianFilter",
    "MedianFilter",
    "MinFilter",
    "MaxFilter",
    "StdDevFilter",
    "SARFilter",
    "LeeFilter",
    "ComplexLeeFilter",
    "EnhancedLeeFilter",
    "LeeSigmaFilter",
    "RefinedLeeFilter",
    "IDANFilter",
    "PhaseGradientFilter",
    "WishartFilter",
    "SobelFilter",
    "LaplacianFilter",
    "UnsharpMaskFilter",
    "HighPassFilter",
    "FrostFilter",
    "KuanFilter",
    "BoxcarFilter",
]
