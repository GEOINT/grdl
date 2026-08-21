# -*- coding: utf-8 -*-
"""Utility helpers for compact-pol decomposition implementations."""

import numpy as np


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Safely divide arrays with tiny-value protection on denominator."""
    eps = np.finfo(np.float64).tiny
    return numerator / np.where(np.abs(denominator) > eps, denominator, eps)
