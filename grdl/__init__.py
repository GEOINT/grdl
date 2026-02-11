# -*- coding: utf-8 -*-
"""
GRDL - GEOINT Rapid Development Library.

A modular Python library of fundamental building blocks for deriving geospatial
intelligence from any 2D correlated imagery -- SAR, EO, MSI, hyperspectral,
space-based, or terrestrial.

Dependencies
------------
numpy
scipy

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
2026-01-30

Modified
--------
2026-01-30
"""

__version__ = "0.1.0"
__author__ = "Duane Smalley"

from grdl.exceptions import (
    GrdlError,
    ValidationError,
    ProcessorError,
    DependencyError,
    GeolocationError,
)
from grdl.vocabulary import (
    ImageModality,
    ProcessorCategory,
    DetectionType,
    SegmentationType,
    ExecutionPhase,
    OutputFormat,
)

# Re-export coregistration classes for convenience
try:
    from grdl.coregistration import (
        CoRegistration,
        AffineCoRegistration,
        ProjectiveCoRegistration,
        RegistrationResult,
    )
    _COREG_EXPORTS = [
        'CoRegistration',
        'AffineCoRegistration',
        'ProjectiveCoRegistration',
        'RegistrationResult',
    ]
except ImportError:
    _COREG_EXPORTS = []

try:
    from grdl.coregistration import FeatureMatchCoRegistration
    _COREG_EXPORTS.append('FeatureMatchCoRegistration')
except ImportError:
    pass

__all__ = [
    'GrdlError',
    'ValidationError',
    'ProcessorError',
    'DependencyError',
    'GeolocationError',
    'ImageModality',
    'ProcessorCategory',
    'DetectionType',
    'SegmentationType',
    'ExecutionPhase',
    'OutputFormat',
    *_COREG_EXPORTS,
]
