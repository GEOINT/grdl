# -*- coding: utf-8 -*-
"""
Gamma correction.

Applies a power-law transform ``y = x ** (1 / gamma)``.  Expects input
already normalized to ``[0, 1]`` (e.g. from ``LinearStretch`` or
``PercentileStretch``).  Use ``gamma > 1`` to brighten midtones, ``< 1``
to darken.

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
from typing import Annotated, Any

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_version('1.0.0')
@processor_tags(
    modalities=[IM.SAR, IM.EO, IM.MSI, IM.PAN, IM.HSI],
    category=PC.ENHANCE,
    description='Gamma (power-law) correction on [0, 1] input.',
)
class GammaCorrection(ImageTransform):
    """Gamma (power-law) correction.

    Computes ``y = x ** (1 / gamma)`` clipped to ``[0, 1]``.  Input is
    expected to already be normalized to ``[0, 1]``; values outside that
    range are clipped before the power transform.

    Parameters
    ----------
    gamma : float, default=1.0
        Gamma exponent.  ``> 1`` brightens, ``< 1`` darkens, ``1`` is
        identity.

    Examples
    --------
    >>> stretched = LinearStretch().apply(image)
    >>> bright    = GammaCorrection(gamma=1.5).apply(stretched)
    """

    gamma: Annotated[
        float, Range(min=0.01, max=100.0),
        Desc('Gamma exponent (>1 brightens, <1 darkens)'),
    ] = 1.0

    def __init__(self, gamma: float = 1.0) -> None:
        if gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.gamma = float(gamma)

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        params = self._resolve_params(kwargs)
        x = np.abs(source) if np.iscomplexobj(source) else np.asarray(source)
        clipped = np.clip(x, 0.0, 1.0)
        out = np.power(clipped, 1.0 / params['gamma'])
        return np.clip(out, 0.0, 1.0).astype(np.float32)
