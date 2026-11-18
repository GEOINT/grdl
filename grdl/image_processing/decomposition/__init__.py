# -*- coding: utf-8 -*-
"""
Polarimetric Decomposition Base & Compatibility Re-exports.

``PolarimetricDecomposition`` lives in ``grdl`` as the core framework ABC.
Concrete decompositions live in ``grdl-sar`` (``grdl_sar.decomposition``).
"""

from importlib import import_module
import warnings

from grdl.image_processing.decomposition.base import PolarimetricDecomposition


def __getattr__(name: str):
    warnings.warn(
        f"grdl.image_processing.decomposition.{name} is deprecated and has moved to grdl-sar; "
        "import from grdl_sar.decomposition instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        mod = import_module("grdl_sar.decomposition")
        if hasattr(mod, name):
            return getattr(mod, name)
    except ImportError as exc:
        raise ImportError(
            f"{name} has moved to grdl-sar. Install with: pip install grdl-sar"
        ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PolarimetricDecomposition",
]
