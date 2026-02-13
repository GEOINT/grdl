# -*- coding: utf-8 -*-
"""
Interpolation - Bandwidth-preserving 1D interpolation functions.

Provides kernel-based interpolators with a uniform callable signature
``(x_old, y_old, x_new) -> y_new``. All interpolators handle real and
complex data, support mildly non-uniform input grids, and fill
out-of-bounds points with zero.

Available interpolators:

- ``LanczosInterpolator`` / ``lanczos_interpolator`` — Lanczos-windowed
  sinc with configurable number of lobes.
- ``KaiserSincInterpolator`` / ``windowed_sinc_interpolator`` —
  Kaiser-windowed sinc with adjustable sidelobe control.
- ``LagrangeInterpolator`` / ``lagrange_interpolator`` — Maximally flat
  FIR fractional delay filter (Laakso et al. Eq. 42).
- ``FarrowInterpolator`` / ``farrow_interpolator`` — Farrow structure
  polynomial coefficient approximation (Laakso et al. Eqs. 59-63).
- ``PolyphaseInterpolator`` / ``polyphase_interpolator`` —
  Pre-computed filter bank (Kaiser-windowed sinc phases) with table
  lookup + linear phase interpolation.  Fastest FIR method for PFA.
- ``ThiranDelayFilter`` / ``thiran_delay`` — IIR allpass fractional
  delay filter with maximally flat group delay (Thiran, 1971;
  Laakso et al. Eq. 86). Operates on uniformly-sampled signals.

Base classes:

- ``Interpolator`` — ABC for all interpolators.
- ``KernelInterpolator`` — Template for kernel-based methods (handles
  neighbor gathering, normalization, OOB fill).

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
2026-02-12

Modified
--------
2026-02-12
"""

from grdl.interpolation.base import Interpolator, KernelInterpolator
from grdl.interpolation.lanczos import LanczosInterpolator, lanczos_interpolator
from grdl.interpolation.windowed_sinc import (
    KaiserSincInterpolator,
    windowed_sinc_interpolator,
)
from grdl.interpolation.lagrange import LagrangeInterpolator, lagrange_interpolator
from grdl.interpolation.farrow import FarrowInterpolator, farrow_interpolator
from grdl.interpolation.polyphase import (
    PolyphaseInterpolator,
    polyphase_interpolator,
)
from grdl.interpolation.thiran import ThiranDelayFilter, thiran_delay

__all__ = [
    'Interpolator',
    'KernelInterpolator',
    'LanczosInterpolator',
    'lanczos_interpolator',
    'KaiserSincInterpolator',
    'windowed_sinc_interpolator',
    'LagrangeInterpolator',
    'lagrange_interpolator',
    'FarrowInterpolator',
    'farrow_interpolator',
    'PolyphaseInterpolator',
    'polyphase_interpolator',
    'ThiranDelayFilter',
    'thiran_delay',
]
