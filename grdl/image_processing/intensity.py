# -*- coding: utf-8 -*-
"""
Intensity Transforms - Magnitude, dB, and contrast stretch operations.

Provides reusable ``ImageTransform`` components for converting complex or
real-valued imagery to display-ready representations:

- ``ToDecibels``: magnitude in dB — ``20 * log10(|x| + floor)``
- ``PercentileStretch``: percentile-based contrast normalization to [0, 1]

Both work on arrays of any shape (2D single image, 3D stacks, etc.).

Author
------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-11
"""

# Standard library
from typing import Annotated, Any

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ProcessorCategory


@processor_version('1.0.0')
@processor_tags(category=ProcessorCategory.ENHANCE)
class ToDecibels(ImageTransform):
    """Convert complex or real-valued imagery to magnitude in decibels.

    Computes ``20 * log10(|source| + eps)`` where *eps* prevents log of
    zero.  Works on complex-valued arrays (extracts magnitude first) and
    real-valued arrays alike.  Accepts any shape.

    Parameters
    ----------
    floor_db : float
        Minimum dB value.  Pixels below this are clamped.
        Default is ``-60.0``.

    Examples
    --------
    >>> import numpy as np
    >>> from grdl.image_processing.intensity import ToDecibels
    >>>
    >>> to_db = ToDecibels(floor_db=-50.0)
    >>> db = to_db.apply(complex_image)
    """

    __gpu_compatible__ = True

    floor_db: Annotated[float, Range(max=0.0), Desc('Minimum dB floor')] = -60.0

    def __init__(self, floor_db: float = -60.0) -> None:
        self.floor_db = floor_db

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply dB conversion.

        Parameters
        ----------
        source : np.ndarray
            Complex or real-valued array of any shape.

        Returns
        -------
        np.ndarray
            Real-valued dB magnitude array, same shape, dtype float64.
        """
        params = self._resolve_params(kwargs)
        floor = params['floor_db']

        mag = np.abs(source)
        db = 20.0 * np.log10(mag + np.finfo(np.float64).tiny)
        np.maximum(db, floor, out=db)
        return db


@processor_version('1.0.0')
@processor_tags(category=ProcessorCategory.ENHANCE)
class PercentileStretch(ImageTransform):
    """Percentile-based contrast stretch to [0, 1].

    Clips values to the [*plow*, *phigh*] percentile range computed from
    the input, then linearly rescales to [0, 1].  Works on any shape.

    Parameters
    ----------
    plow : float
        Lower percentile (0–100).  Default ``2.0``.
    phigh : float
        Upper percentile (0–100).  Default ``98.0``.

    Examples
    --------
    >>> import numpy as np
    >>> from grdl.image_processing.intensity import PercentileStretch
    >>>
    >>> stretch = PercentileStretch(plow=1.0, phigh=99.0)
    >>> vis = stretch.apply(db_image)
    """

    __gpu_compatible__ = True

    plow: Annotated[float, Range(min=0.0, max=100.0), Desc('Lower percentile')] = 2.0
    phigh: Annotated[float, Range(min=0.0, max=100.0), Desc('Upper percentile')] = 98.0

    def __init__(self, plow: float = 2.0, phigh: float = 98.0) -> None:
        if plow >= phigh:
            raise ValueError(
                f"plow ({plow}) must be less than phigh ({phigh})"
            )
        self.plow = plow
        self.phigh = phigh

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply percentile stretch.

        Parameters
        ----------
        source : np.ndarray
            Real-valued array of any shape.

        Returns
        -------
        np.ndarray
            Array normalized to [0, 1], dtype float32.
        """
        params = self._resolve_params(kwargs)
        plow = params['plow']
        phigh = params['phigh']

        vmin = np.percentile(source, plow)
        vmax = np.percentile(source, phigh)
        if vmax - vmin < np.finfo(np.float32).eps:
            return np.zeros_like(source, dtype=np.float32)
        out = (source - vmin) / (vmax - vmin)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
