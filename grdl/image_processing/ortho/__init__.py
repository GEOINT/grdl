# -*- coding: utf-8 -*-
"""
Orthorectification Sub-module - Geometric reprojection to geographic grids.

Provides orthorectification for imagery in native acquisition geometry
(SAR slant range, oblique EO, etc.) to regular ground-projected grids.

Key Classes
-----------
- Orthorectifier: Orthorectify imagery using a Geolocation object
- OutputGrid: Specification for the output geographic grid

Dependencies
------------
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

from grdl.image_processing.ortho.ortho import Orthorectifier, OutputGrid

__all__ = [
    'Orthorectifier',
    'OutputGrid',
]
