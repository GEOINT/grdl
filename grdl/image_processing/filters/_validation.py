# -*- coding: utf-8 -*-
"""
Filter Validation Helpers - Shared kernel size and boundary mode validation.

Provides reusable validation functions for spatial image filters. All filter
classes in this subpackage call these helpers to enforce consistent constraints
on kernel size (odd, >= 3) and scipy.ndimage boundary modes.

Author
------
Duane Smalley, PhD
duane.d.smalley@gmail.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-11

Modified
--------
2026-02-11
"""

# GRDL internal
from grdl.exceptions import ValidationError


BOUNDARY_MODES = ('reflect', 'constant', 'nearest', 'wrap')


def validate_kernel_size(kernel_size: int, name: str = 'kernel_size') -> None:
    """Validate that kernel size is an odd integer >= 3.

    Parameters
    ----------
    kernel_size : int
        The kernel size to validate.
    name : str
        Parameter name for error messages. Default ``'kernel_size'``.

    Raises
    ------
    ValidationError
        If ``kernel_size`` is not an integer, is less than 3, or is even.
    """
    if not isinstance(kernel_size, int):
        raise ValidationError(
            f"{name} must be an integer, got {type(kernel_size).__name__}"
        )
    if kernel_size < 3:
        raise ValidationError(
            f"{name} must be >= 3, got {kernel_size}"
        )
    if kernel_size % 2 == 0:
        raise ValidationError(
            f"{name} must be odd, got {kernel_size}"
        )


def validate_mode(mode: str) -> None:
    """Validate that boundary mode is supported by scipy.ndimage.

    Parameters
    ----------
    mode : str
        Boundary handling mode to validate.

    Raises
    ------
    ValidationError
        If ``mode`` is not one of the supported scipy.ndimage modes.
    """
    if mode not in BOUNDARY_MODES:
        raise ValidationError(
            f"mode must be one of {BOUNDARY_MODES}, got {mode!r}"
        )
