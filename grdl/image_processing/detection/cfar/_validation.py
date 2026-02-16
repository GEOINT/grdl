# -*- coding: utf-8 -*-
"""
CFAR Validation Helpers - Shared parameter validation for CFAR detectors.

Provides validation functions for CFAR detector parameters: window
geometry, probability of false alarm, and statistical model assumption.
All functions raise ``ValidationError`` on invalid input.

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
2026-02-16

Modified
--------
2026-02-16
"""

from grdl.exceptions import ValidationError


ASSUMPTIONS = ('gaussian', 'exponential')
"""Allowed statistical model assumptions for threshold computation."""


def validate_cfar_window(guard_cells: int, training_cells: int) -> None:
    """Validate CFAR guard and training window geometry.

    Parameters
    ----------
    guard_cells : int
        Guard band half-width in pixels. Must be positive.
    training_cells : int
        Training annulus half-width in pixels. Must exceed guard_cells.

    Raises
    ------
    ValidationError
        If guard_cells >= training_cells, or either is not a positive
        integer.
    """
    if not isinstance(guard_cells, int) or guard_cells < 1:
        raise ValidationError(
            f"guard_cells must be a positive integer, got {guard_cells!r}"
        )
    if not isinstance(training_cells, int) or training_cells < 1:
        raise ValidationError(
            f"training_cells must be a positive integer, got {training_cells!r}"
        )
    if guard_cells >= training_cells:
        raise ValidationError(
            f"guard_cells ({guard_cells}) must be less than "
            f"training_cells ({training_cells})"
        )


def validate_pfa(pfa: float) -> None:
    """Validate probability of false alarm.

    Parameters
    ----------
    pfa : float
        Probability of false alarm. Must satisfy 0 < pfa < 1.

    Raises
    ------
    ValidationError
        If pfa is out of range.
    """
    if not isinstance(pfa, (int, float)) or pfa <= 0.0 or pfa >= 1.0:
        raise ValidationError(
            f"pfa must satisfy 0 < pfa < 1, got {pfa!r}"
        )


def validate_assumption(assumption: str) -> None:
    """Validate statistical model assumption.

    Parameters
    ----------
    assumption : str
        One of ``'gaussian'`` or ``'exponential'``.

    Raises
    ------
    ValidationError
        If assumption is not a recognized value.
    """
    if assumption not in ASSUMPTIONS:
        raise ValidationError(
            f"assumption must be one of {ASSUMPTIONS}, got {assumption!r}"
        )
