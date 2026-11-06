"""Utility helpers for compact-pol decomposition implementations."""

from typing import cast

import numpy as np


def compact_handedness_s3(s3: np.ndarray, chi: float) -> np.ndarray:
    """Adapt canonical S3 to compact algorithms' transmit-relative convention.

    Negative ellipticity reverses S3. Nonnegative ellipticity, including
    zero, retains S3 so existing component meanings and angles are preserved.
    """
    return s3 if chi >= 0 else -s3


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Safely divide arrays with tiny-value protection on denominator."""
    eps = np.finfo(np.float64).tiny
    return cast(np.ndarray, numerator / np.where(np.abs(denominator) > eps, denominator, eps))
