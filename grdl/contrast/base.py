# -*- coding: utf-8 -*-
"""
Contrast Module Helpers - clip-and-cast, linear remap, NaN-safe stats.

Small numerical utilities shared by every operator in :mod:`grdl.contrast`.
Ported from ``sarpy.visualization.remap`` (``clip_cast``, ``_linear_map``,
``_nrl_stats``) so SAR remaps stay byte-compatible with sarpy on identical
inputs.

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
from typing import Optional, Tuple, Union

# Third-party
import numpy as np


def clip_cast(
    array: np.ndarray,
    dtype: Union[str, np.dtype, np.number] = 'uint8',
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
) -> np.ndarray:
    """Clip values into the integer dtype's valid range, then cast.

    Avoids the silent wraparound of ``arr.astype('uint8')`` when input
    exceeds the dtype range.  When ``min_value``/``max_value`` are given,
    they are intersected with the dtype's intrinsic limits.

    Parameters
    ----------
    array : np.ndarray
        Real-valued source array.
    dtype : str or np.dtype, default='uint8'
        Target integer dtype.
    min_value : int or float, optional
        Additional lower clip bound (intersected with dtype min).
    max_value : int or float, optional
        Additional upper clip bound (intersected with dtype max).

    Returns
    -------
    np.ndarray
        Clipped and cast array, same shape as input.
    """
    np_type = np.dtype(dtype)
    type_min = np.iinfo(np_type).min
    type_max = np.iinfo(np_type).max
    lo = type_min if min_value is None else max(min_value, type_min)
    hi = type_max if max_value is None else min(max_value, type_max)
    return np.clip(array, lo, hi).astype(np_type)


def linear_map(
    data: np.ndarray,
    min_value: float,
    max_value: float,
) -> np.ndarray:
    """Map *data* linearly to ``[0, 1]`` with clipping.

    Computes ``clip((data - min_value) / (max_value - min_value), 0, 1)``.
    Caller must ensure ``max_value > min_value``.
    """
    return np.clip(
        (data - min_value) / float(max_value - min_value),
        0.0, 1.0,
    )


def nan_safe_stats(
    amplitude: np.ndarray,
    percentile: Union[int, float] = 99.0,
) -> Tuple[float, float, float]:
    """Return ``(min, max, percentile)`` over finite values only.

    Parameters
    ----------
    amplitude : np.ndarray
        Real-valued array (may contain NaN/inf).
    percentile : float, default=99.0
        Percentile to compute alongside min/max.

    Returns
    -------
    (min, max, p) : tuple of float
        All zero when no finite values are present.
    """
    finite_mask = np.isfinite(amplitude)
    if not np.any(finite_mask):
        return 0.0, 0.0, 0.0
    finite = amplitude[finite_mask]
    return (
        float(np.min(finite)),
        float(np.max(finite)),
        float(np.percentile(finite, percentile)),
    )
