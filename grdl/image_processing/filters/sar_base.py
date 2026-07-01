# -*- coding: utf-8 -*-
"""
SAR Filter Base - Abstract base class for SAR-specific spatial filters.

Provides shared parameters (``kernel_size``, ``enl``) and the ``_estimate_enl``
helper used by all SAR scalar filters.  Matrix-aware (polarimetric) filters
inherit from this class and implement ``apply()`` directly; scalar (per-band)
filters combine this base with ``BandwiseTransformMixin`` and implement
``_apply_2d()``.

Author
------
Jason Fritz, PhD
43161141+stryder-vtx@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-06-24

Modified
--------
2026-06-24
"""

# Standard library
import logging
from typing import Annotated

# Third-party
import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False
    cp = None

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.filters._validation import validate_kernel_size
from grdl.image_processing.params import Desc, Range

logger = logging.getLogger(__name__)


class SARFilter(ImageTransform):
    """Abstract base class for SAR-specific spatial filters.

    Declares the common ``kernel_size`` and ``enl`` parameters shared by all
    SAR scalar and polarimetric filters, and provides the ``_estimate_enl``
    static helper.

    Scalar (per-band) subclasses add ``BandwiseTransformMixin`` to their
    inheritance and implement ``_apply_2d()``.  Matrix-aware (covariance /
    coherency) subclasses inherit this class directly and implement
    ``apply()`` themselves, since they must process all channels jointly.

    Parameters
    ----------
    kernel_size : int
        Square kernel side length in pixels.  Must be odd and >= 3.
        Default is 7.
    enl : float
        Equivalent Number of Looks.  Controls the noise threshold.
        If 0.0, automatically estimated from the image by treating
        the bottom 10 % of Ci² values as homogeneous regions.
        Default is 0.0 (auto-estimate).
    """

    kernel_size: Annotated[int, Range(min=3, max=31),
                           Desc('Square kernel side length (odd)')] = 7
    enl: Annotated[float, Range(min=0.0),
                   Desc('Equivalent Number of Looks (0 = auto)')] = 0.0

    def __init__(
        self,
        kernel_size: int = 7,
        enl: float = 0.0,
    ) -> None:
        validate_kernel_size(kernel_size)
        self.kernel_size = kernel_size
        self.enl = enl

    @staticmethod
    def _estimate_enl(
        ci2: np.ndarray,
        homogeneous_fraction: float = 0.1,
    ) -> float:
        """Estimate Equivalent Number of Looks from coefficient of variation.

        Sorts Ci² values and treats the lowest fraction as homogeneous
        regions, then estimates ENL = 1 / mean(Ci²_homogeneous).

        Parameters
        ----------
        ci2 : np.ndarray
            Coefficient of variation squared, Ci² = var(I) / E[I]².
            Accepts cupy arrays; computation is performed on the same device.
        homogeneous_fraction : float
            Fraction of lowest-Ci² pixels assumed homogeneous.
            Default is 0.1 (10 %).

        Returns
        -------
        float
            Estimated ENL (>= 1.0 for valid SAR data).
        """
        _is_gpu = _HAS_CUPY and isinstance(ci2, cp.ndarray)
        xp = cp if _is_gpu else np
        sorted_vals = xp.sort(ci2.ravel())
        n_homog = max(1, int(len(sorted_vals) * homogeneous_fraction))
        mean_ci2 = float(xp.mean(sorted_vals[:n_homog]))
        if mean_ci2 > 0:
            return 1.0 / mean_ci2
        return 1.0
