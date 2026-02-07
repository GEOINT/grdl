# -*- coding: utf-8 -*-
"""
GRDL Exception Hierarchy - Domain-specific exceptions for GRDL operations.

Provides a small exception hierarchy that lets downstream consumers (e.g.,
GRDK's WorkflowExecutor) catch GRDL-specific errors distinctly from
Python built-in exceptions. All GRDL exceptions subclass both ``GrdlError``
and the appropriate built-in exception for backward compatibility.

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


class GrdlError(Exception):
    """Base exception for all GRDL errors."""


class ValidationError(GrdlError, ValueError):
    """Invalid input data, parameters, or configuration.

    Raised for shape mismatches, out-of-range parameters, invalid
    method names, and other input validation failures.
    """


class ProcessorError(GrdlError, RuntimeError):
    """Algorithm or processing failure during apply()/detect().

    Raised when a processor encounters a non-recoverable error
    during execution (not an input validation issue).
    """


class DependencyError(GrdlError, ImportError):
    """Missing optional dependency required for a specific module.

    Raised when a module requires an optional package (sarpy,
    rasterio, opencv) that is not installed.
    """


class GeolocationError(GrdlError, RuntimeError):
    """Coordinate transformation or geolocation failure.

    Raised for out-of-bounds transforms, missing GCPs, or
    unsupported geometry types.
    """
