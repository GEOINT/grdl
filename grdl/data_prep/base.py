# -*- coding: utf-8 -*-
"""
Data Preparation Base - Shared utilities for data preparation modules.

This module contains shared helper functions used across the data_prep
domain. Unlike other GRDL domains, data_prep does not define ABCs because
its classes (Tiler, ChipExtractor, Normalizer) are concrete utility classes
rather than extensible hierarchies.

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
2026-02-06

Modified
--------
2026-02-06
"""

# Standard library
from typing import Tuple, Union

# Third-party
import numpy as np


def _normalize_pair(
    value: Union[int, Tuple[int, int]], name: str
) -> Tuple[int, int]:
    """Convert an int or (int, int) tuple to a validated (int, int) pair.

    Parameters
    ----------
    value : int or Tuple[int, int]
        Scalar or pair value. If scalar, both elements are set equal.
    name : str
        Parameter name for error messages.

    Returns
    -------
    Tuple[int, int]
        Validated (rows, cols) pair.

    Raises
    ------
    TypeError
        If value is not int or tuple of two ints.
    ValueError
        If any element is not positive.
    """
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return (value, value)
    if isinstance(value, tuple) and len(value) == 2:
        r, c = value
        if not isinstance(r, int) or not isinstance(c, int):
            raise TypeError(f"{name} tuple elements must be int")
        if r <= 0 or c <= 0:
            raise ValueError(
                f"{name} elements must be positive, got ({r}, {c})"
            )
        return (r, c)
    raise TypeError(
        f"{name} must be int or Tuple[int, int], got {type(value).__name__}"
    )


def _validate_image(image: np.ndarray) -> None:
    """Validate that an array is a 2D or 3D image.

    Parameters
    ----------
    image : np.ndarray
        Array to validate.

    Raises
    ------
    TypeError
        If input is not a numpy ndarray.
    ValueError
        If input is not 2D or 3D.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"image must be np.ndarray, got {type(image).__name__}"
        )
    if image.ndim not in (2, 3):
        raise ValueError(
            f"image must be 2D (rows, cols) or 3D (bands, rows, cols), "
            f"got {image.ndim}D"
        )
