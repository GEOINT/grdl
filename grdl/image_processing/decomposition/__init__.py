# -*- coding: utf-8 -*-
"""
Polarimetric Decomposition Sub-module - Scattering matrix decompositions.

Provides polarimetric decomposition methods for quad-pol SAR imagery.
Decompositions operate on the complex-valued scattering matrix [S] and
return named complex components preserving phase and interference.

Key Classes
-----------
- PolarimetricDecomposition: ABC for all polarimetric decompositions
- PauliDecomposition: Quad-pol Pauli basis decomposition

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

from grdl.image_processing.decomposition.base import PolarimetricDecomposition
from grdl.image_processing.decomposition.pauli import PauliDecomposition
from grdl.image_processing.decomposition.dual_pol_halpha import DualPolHAlpha

__all__ = [
    'PolarimetricDecomposition',
    'PauliDecomposition',
    'DualPolHAlpha',
]
