# -*- coding: utf-8 -*-
"""
Polarimetric Decomposition Sub-module - Scattering matrix decompositions.

Provides polarimetric decomposition methods for SAR imagery. Quad-pol
decompositions operate on the full scattering matrix [S]; dual-pol
decompositions operate on a 2-channel coherency matrix.

Key Classes
-----------
- PolarimetricDecomposition: ABC for all polarimetric decompositions
- PauliDecomposition: Quad-pol Pauli basis (surface / double-bounce / volume)
- DualPolHAlpha: Dual-pol entropy, alpha angle, anisotropy, and span

When to Use What
----------------
- **Quad-pol data (HH, HV, VH, VV):** ``PauliDecomposition`` decomposes
  the scattering matrix into three orthogonal Pauli basis components.
  RGB composite: R=double-bounce, G=volume, B=surface.

- **Dual-pol data (co-pol + cross-pol):** ``DualPolHAlpha`` computes
  the 2x2 coherency matrix eigendecomposition yielding entropy (H),
  mean scattering angle (alpha), anisotropy (A), and total span.
  Entropy near 0 means single dominant mechanism; near 1 means random.

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
2026-02-17
"""

from grdl.image_processing.decomposition.base import PolarimetricDecomposition
from grdl.image_processing.decomposition.pauli import PauliDecomposition
from grdl.image_processing.decomposition.dual_pol_halpha import DualPolHAlpha

__all__ = [
    'PolarimetricDecomposition',
    'PauliDecomposition',
    'DualPolHAlpha',
]
