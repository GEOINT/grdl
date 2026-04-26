# -*- coding: utf-8 -*-
"""
Linear contrast stretch.

Maps amplitude to ``[0, 1]`` linearly between explicit ``min_value`` and
``max_value`` thresholds (or the input's observed extrema when omitted).
Direct port of ``sarpy.visualization.remap.Linear``, with the same
finite/non-finite handling: non-finite samples receive the maximum
output value (saturated white) so they remain visible against a clipped
background.

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
from grdl.contrast.base import linear_map
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_version('1.0.0')
@processor_tags(
    modalities=[IM.SAR, IM.EO, IM.MSI, IM.PAN, IM.HSI],
    category=PC.ENHANCE,
    description='Linear contrast stretch with optional explicit min/max.',
)
class LinearStretch(ImageTransform):
    """Linear stretch to ``[0, 1]``.

    Computes ``(|x| - min) / (max - min)``, clipped to ``[0, 1]``.  When
    ``min_value`` / ``max_value`` are ``None``, uses the input's finite
    min/max.  Non-finite samples are mapped to ``1.0`` (saturated).

    Parameters
    ----------
    min_value : float, optional
        Lower threshold.  ``None`` → derive from input extrema.
    max_value : float, optional
        Upper threshold.  ``None`` → derive from input extrema.

    Examples
    --------
    >>> stretch = LinearStretch()
    >>> out = stretch.apply(amplitude_image)
    >>>
    >>> # Pre-computed bounds for tile-consistent display
    >>> stretch = LinearStretch(min_value=2.0, max_value=512.0)
    >>> tile = stretch.apply(chip)
    """

    min_value: Annotated[
        float, Desc('Lower threshold; None → input min'),
    ] = None
    max_value: Annotated[
        float, Desc('Upper threshold; None → input max'),
    ] = None

    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> None:
        self.min_value = min_value
        self.max_value = max_value

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply linear stretch.

        Parameters
        ----------
        source : np.ndarray
            Real or complex array of any shape.  Complex input has its
            magnitude taken first.
        **kwargs
            ``min_value``, ``max_value`` — per-call overrides of instance
            thresholds.

        Returns
        -------
        np.ndarray
            ``float32`` array in ``[0, 1]``, same shape as input.
        """
        amplitude = np.abs(source) if np.iscomplexobj(source) else source
        amplitude = np.asarray(amplitude)

        finite_mask = np.isfinite(amplitude)
        out = np.empty(amplitude.shape, dtype=np.float64)
        out[~finite_mask] = 1.0   # saturated white

        if np.any(finite_mask):
            finite_data = amplitude[finite_mask]
            lo, hi = self._resolve_extrema(finite_data, kwargs)

            if lo == hi:
                out[finite_mask] = 0.0
            else:
                out[finite_mask] = linear_map(finite_data, lo, hi)

        return out.astype(np.float32)

    def _resolve_extrema(
        self,
        finite_data: np.ndarray,
        kwargs: dict,
    ) -> tuple:
        """Order: kwargs > instance > computed-from-input."""
        lo = kwargs.get('min_value', self.min_value)
        hi = kwargs.get('max_value', self.max_value)
        if lo is None:
            lo = float(np.min(finite_data))
        if hi is None:
            hi = float(np.max(finite_data))
        if lo > hi:
            lo, hi = hi, lo
        return float(lo), float(hi)
