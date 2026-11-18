# -*- coding: utf-8 -*-
"""
Detection Sub-module - Core detection interfaces and data models.

Provides ``ImageDetector`` ABC, ``Detection``, ``DetectionSet``, and
``DATA_DICTIONARY`` / ``Fields``.  SAR CFAR detectors live in ``grdl-sar``
(``grdl_sar.detection``).
"""

from importlib import import_module
import warnings

from grdl.image_processing.detection.base import ImageDetector
from grdl.image_processing.detection.fields import (
    DATA_DICTIONARY,
    FieldDefinition,
    Fields,
    is_dictionary_field,
    list_fields,
    lookup_field,
)
from grdl.image_processing.detection.models import Detection, DetectionSet

_SAR_CFAR = {
    "CFARDetector",
    "CACFARDetector",
    "GOCFARDetector",
    "SOCFARDetector",
    "OSCFARDetector",
}


def __getattr__(name: str):
    if name in _SAR_CFAR:
        warnings.warn(
            f"grdl.image_processing.detection.{name} is deprecated and has moved to grdl-sar; "
            "import from grdl_sar.detection instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            return getattr(import_module("grdl_sar.detection"), name)
        except ImportError as exc:
            raise ImportError(
                f"{name} has moved to grdl-sar. Install with: pip install grdl-sar"
            ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ImageDetector",
    "Detection",
    "DetectionSet",
    "FieldDefinition",
    "Fields",
    "DATA_DICTIONARY",
    "lookup_field",
    "is_dictionary_field",
    "list_fields",
    "CFARDetector",
    "CACFARDetector",
    "GOCFARDetector",
    "SOCFARDetector",
    "OSCFARDetector",
]
