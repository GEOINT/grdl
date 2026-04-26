# -*- coding: utf-8 -*-
"""
Sigmoid (S-curve) contrast stretch.

Applies a logistic curve ``y = 1 / (1 + exp(-slope * (x - center)))``
followed by linear rescale so ``y`` lands exactly in ``[0, 1]``.
Smoother than hard percentile clipping; useful for compressing extremes
without introducing flat saturation.

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
    description='Sigmoid (logistic S-curve) stretch on [0, 1] input.',
)
class SigmoidStretch(ImageTransform):
    """Logistic S-curve stretch on ``[0, 1]`` input.

    Computes ``raw = 1 / (1 + exp(-slope * (x - center)))`` then rescales
    so that ``x = 0`` → ``y = 0`` and ``x = 1`` → ``y = 1``.

    Parameters
    ----------
    center : float, default=0.5
        Midpoint of the S-curve in input units.
    slope : float, default=10.0
        Steepness.  Higher → sharper transition.

    Examples
    --------
    >>> norm = PercentileStretch().apply(image)
    >>> out  = SigmoidStretch(center=0.5, slope=12.0).apply(norm)
    """

    center: Annotated[
        float, Range(min=0.0, max=1.0),
        Desc('Midpoint of the S-curve in [0, 1]'),
    ] = 0.5
    slope: Annotated[
        float, Range(min=0.1, max=100.0),
        Desc('Curve steepness'),
    ] = 10.0

    def __init__(self, center: float = 0.5, slope: float = 10.0) -> None:
        if not (0.0 <= center <= 1.0):
            raise ValueError(f"center must be in [0, 1], got {center}")
        if slope <= 0.0:
            raise ValueError(f"slope must be positive, got {slope}")
        self.center = float(center)
        self.slope = float(slope)

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        params = self._resolve_params(kwargs)
        center = params['center']
        slope = params['slope']

        x = np.abs(source) if np.iscomplexobj(source) else np.asarray(source)
        x = np.clip(x, 0.0, 1.0)
        sig = 1.0 / (1.0 + np.exp(-slope * (x - center)))

        # Rescale endpoints so the curve hits 0 at x=0 and 1 at x=1.
        y0 = 1.0 / (1.0 + np.exp(slope * center))
        y1 = 1.0 / (1.0 + np.exp(-slope * (1.0 - center)))
        if y1 - y0 < np.finfo(np.float64).eps:
            return np.zeros(x.shape, dtype=np.float32)
        out = (sig - y0) / (y1 - y0)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
