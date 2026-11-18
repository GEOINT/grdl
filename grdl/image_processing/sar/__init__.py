# -*- coding: utf-8 -*-
"""
SAR Sub-module Compatibility Re-exports.

Base ABC ``ImageFormationAlgorithm`` remains in ``grdl``. Concrete SAR
processors live in ``grdl-sar`` (``grdl_sar.subaperture``,
``grdl_sar.image_formation``).
"""

from importlib import import_module
import warnings

from grdl.image_processing.sar.image_formation.base import ImageFormationAlgorithm


def __getattr__(name: str):
    for mod_name in ("grdl_sar.subaperture", "grdl_sar.image_formation"):
        try:
            mod = import_module(mod_name)
            if hasattr(mod, name):
                warnings.warn(
                    f"grdl.image_processing.sar.{name} is deprecated and has moved to {mod_name}; "
                    f"import from {mod_name} instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return getattr(mod, name)
        except ImportError:
            continue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ImageFormationAlgorithm",
]
