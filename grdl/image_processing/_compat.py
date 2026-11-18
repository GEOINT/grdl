# -*- coding: utf-8 -*-
"""
Compatibility Redirector for Migrated Modality Processors.

Intercepts legacy imports under ``grdl.image_processing.*`` for processors
that moved to ``grdl-sar`` and ``grdl-eo``, emits a ``DeprecationWarning``,
and aliases the target module into ``sys.modules`` so both public and private
attributes resolve identically.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import warnings
from typing import Sequence

_EXACT_REDIRECTS: dict[str, tuple[str, str]] = {
    # EO / MSI / HSI
    "grdl.image_processing.fusion": ("grdl_eo.fusion", "grdl-eo"),
    "grdl.image_processing.fusion.pansharpen": ("grdl_eo.fusion.pansharpen", "grdl-eo"),
    "grdl.image_processing.spectral": ("grdl_eo.spectral", "grdl-eo"),
    "grdl.image_processing.spectral.indices": ("grdl_eo.spectral.indices", "grdl-eo"),
    "grdl.image_processing.filters.edge": ("grdl_eo.filters.edge", "grdl-eo"),
    # SAR filters
    "grdl.image_processing.filters.base": ("grdl_sar.filters.base", "grdl-sar"),
    "grdl.image_processing.filters.sar_base": ("grdl_sar.filters.sar_base", "grdl-sar"),
    "grdl.image_processing.filters.standard_lee": ("grdl_sar.filters.standard_lee", "grdl-sar"),
    "grdl.image_processing.filters.enhanced_lee": ("grdl_sar.filters.enhanced_lee", "grdl-sar"),
    "grdl.image_processing.filters.lee_sigma": ("grdl_sar.filters.lee_sigma", "grdl-sar"),
    "grdl.image_processing.filters.refined_lee": ("grdl_sar.filters.refined_lee", "grdl-sar"),
    "grdl.image_processing.filters.idan": ("grdl_sar.filters.idan", "grdl-sar"),
    "grdl.image_processing.filters.wishart": ("grdl_sar.filters.wishart", "grdl-sar"),
    "grdl.image_processing.filters.sar_speckle": ("grdl_sar.filters.sar_speckle", "grdl-sar"),
    "grdl.image_processing.filters.phase": ("grdl_sar.filters.phase", "grdl-sar"),
    # SAR subaperture
    "grdl.image_processing.sar.csi": ("grdl_sar.subaperture.csi", "grdl-sar"),
    "grdl.image_processing.sar.dominance": ("grdl_sar.subaperture.dominance", "grdl-sar"),
    "grdl.image_processing.sar.multilook": ("grdl_sar.subaperture.multilook", "grdl-sar"),
    "grdl.image_processing.sar.sublook": ("grdl_sar.subaperture.sublook", "grdl-sar"),
    # SAR detection CFAR
    "grdl.image_processing.detection.cfar": ("grdl_sar.detection.cfar", "grdl-sar"),
}

_PREFIX_REDIRECTS: tuple[tuple[str, str, str, frozenset[str]], ...] = (
    (
        "grdl.image_processing.decomposition.",
        "grdl_sar.decomposition.",
        "grdl-sar",
        frozenset({"grdl.image_processing.decomposition.base"}),
    ),
    (
        "grdl.image_processing.detection.cfar.",
        "grdl_sar.detection.cfar.",
        "grdl-sar",
        frozenset(),
    ),
    (
        "grdl.image_processing.sar.image_formation.",
        "grdl_sar.image_formation.",
        "grdl-sar",
        frozenset({"grdl.image_processing.sar.image_formation.base"}),
    ),
)


def _resolve_target(fullname: str) -> tuple[str, str] | None:
    if fullname in _EXACT_REDIRECTS:
        return _EXACT_REDIRECTS[fullname]
    for old_prefix, new_prefix, pkg_hint, exclusions in _PREFIX_REDIRECTS:
        if fullname.startswith(old_prefix) and fullname not in exclusions:
            suffix = fullname[len(old_prefix) :]
            return f"{new_prefix}{suffix}", pkg_hint
    return None


class _MovedModuleLoader(importlib.abc.Loader):
    """Loader that forwards a moved module and emits ``DeprecationWarning``."""

    def __init__(self, target_name: str, pkg_hint: str) -> None:
        self.target_name = target_name
        self.pkg_hint = pkg_hint

    def create_module(self, spec: importlib.machinery.ModuleSpec):
        return None  # Use default module creation so target.__name__ is never mutated

    def exec_module(self, module) -> None:
        warnings.warn(
            f"{module.__name__} is deprecated and has moved to {self.target_name} "
            f"({self.pkg_hint}); import from {self.target_name} instead.",
            DeprecationWarning,
            stacklevel=4,
        )
        try:
            target = importlib.import_module(self.target_name)
        except ImportError as exc:
            raise ImportError(
                f"{module.__name__} has moved to {self.pkg_hint}. "
                f"Install with: pip install {self.pkg_hint}"
            ) from exc
        for key, value in target.__dict__.items():
            if key in {"__name__", "__spec__", "__loader__", "__package__", "__path__", "__file__"}:
                continue
            module.__dict__[key] = value
        module.__dict__["__getattr__"] = lambda name, _t=target: getattr(_t, name)


class _ModalityCompatFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder redirecting legacy SAR/EO imports."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target=None,
    ) -> importlib.machinery.ModuleSpec | None:
        resolved = _resolve_target(fullname)
        if resolved is None:
            return None
        target_name, pkg_hint = resolved
        is_pkg = target_name.count(".") <= 1 or fullname.endswith(".cfar")
        return importlib.machinery.ModuleSpec(
            name=fullname,
            loader=_MovedModuleLoader(target_name, pkg_hint),
            is_package=is_pkg,
        )


_FINDER = _ModalityCompatFinder()


def install_shim_finder() -> None:
    """Register the compatibility meta-path finder once."""
    if _FINDER not in sys.meta_path:
        sys.meta_path.append(_FINDER)
