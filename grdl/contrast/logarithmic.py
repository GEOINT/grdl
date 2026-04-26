# -*- coding: utf-8 -*-
"""
Bounded logarithmic contrast stretch.

Direct port of ``sarpy.visualization.remap.Logarithmic``.  Maps amplitude
to ``[0, 1]`` via ``log2((clip(x, min, max) - min) / (max - min) + 1)`` —
log scaling with explicit lower and upper thresholds.

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-04-26

Modified
--------
2026-04-26
"""

# Standard library
from typing import Annotated, Any, Optional

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_version('1.0.0')
@processor_tags(
    modalities=[IM.SAR, IM.EO, IM.MSI],
    category=PC.ENHANCE,
    description='Bounded log2 stretch (sarpy Logarithmic).',
)
class LogStretch(ImageTransform):
    """Bounded log2 contrast stretch.

    Computes ``log2((clip(x, min, max) - min) / (max - min) + 1)`` over
    finite, non-zero samples — output naturally falls in ``[0, 1]``.
    Zero samples map to ``0``; non-finite samples map to ``1`` (saturated).

    Parameters
    ----------
    min_value : float, optional
        Lower clip threshold.  ``None`` → derive from input.
    max_value : float, optional
        Upper clip threshold.  ``None`` → derive from input.
    """

    min_value: Annotated[
        float, Desc('Lower clip threshold; None → input min'),
    ] = None
    max_value: Annotated[
        float, Desc('Upper clip threshold; None → input max'),
    ] = None

    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> None:
        self.min_value = min_value
        self.max_value = max_value

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        amplitude = np.abs(source) if np.iscomplexobj(source) else source
        amplitude = np.asarray(amplitude)

        out = np.empty(amplitude.shape, dtype=np.float64)
        finite_mask = np.isfinite(amplitude)
        zero_mask = (amplitude == 0)
        use_mask = finite_mask & (~zero_mask)

        out[~finite_mask] = 1.0
        out[zero_mask] = 0.0

        if np.any(use_mask):
            data = amplitude[use_mask]
            lo, hi = self._resolve_extrema(data, kwargs)

            if lo == hi:
                out[use_mask] = 0.0
            else:
                clipped = (np.clip(data, lo, hi) - lo) / (hi - lo) + 1.0
                out[use_mask] = np.log2(clipped)

        return out.astype(np.float32)

    def _resolve_extrema(
        self,
        finite_data: np.ndarray,
        kwargs: dict,
    ) -> tuple:
        lo = kwargs.get('min_value', self.min_value)
        hi = kwargs.get('max_value', self.max_value)
        if lo is None:
            lo = float(np.min(finite_data))
        if hi is None:
            hi = float(np.max(finite_data))
        if lo > hi:
            lo, hi = hi, lo
        return float(lo), float(hi)
