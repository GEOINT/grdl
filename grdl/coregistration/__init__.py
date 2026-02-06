# -*- coding: utf-8 -*-
"""
Co-Registration Module - Image-to-image alignment for multi-temporal stacks.

Provides interfaces and implementations for registering a moving image to a
fixed (reference) image. Co-registration is the prerequisite step for building
multi-temporal image stacks in which all images share a common pixel space.

Key Classes
-----------
- CoRegistration: Abstract base class for co-registration algorithms
- RegistrationResult: Result container with transform matrix and quality metrics
- AffineCoRegistration: Affine transform estimation from control points
- ProjectiveCoRegistration: Homography estimation from control points
- FeatureMatchCoRegistration: Automated feature-based registration (OpenCV)

Usage
-----
Co-register a moving image to a fixed reference using feature matching:

    >>> from grdl.coregistration import FeatureMatchCoRegistration
    >>> coreg = FeatureMatchCoRegistration(method='orb', max_features=1000)
    >>> result = coreg.estimate(fixed_image, moving_image)
    >>> aligned = coreg.apply(moving_image, result)

Dependencies
------------
scipy
opencv-python-headless (optional, for FeatureMatchCoRegistration)

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
2026-02-06

Modified
--------
2026-02-06
"""

from grdl.coregistration.base import CoRegistration, RegistrationResult
from grdl.coregistration.affine import AffineCoRegistration
from grdl.coregistration.projective import ProjectiveCoRegistration

__all__ = [
    'CoRegistration',
    'RegistrationResult',
    'AffineCoRegistration',
    'ProjectiveCoRegistration',
]

# FeatureMatchCoRegistration requires opencv-python-headless.
# Import conditionally to avoid hard dependency.
try:
    from grdl.coregistration.feature_match import FeatureMatchCoRegistration
    __all__.append('FeatureMatchCoRegistration')
except ImportError:
    pass

__version__ = '0.1.0'
