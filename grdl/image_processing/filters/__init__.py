# -*- coding: utf-8 -*-
"""
Spatial Filters - Statistical, rank, and adaptive spatial image filters.

Provides general-purpose spatial image filters operating on single-band or
multi-band raster arrays. All filters inherit from ``BandwiseTransformMixin``
and ``ImageTransform``, automatically handling 3D ``(bands, rows, cols)``
band stacks by applying the 2D filter to each band independently.

Linear Filters
    ``MeanFilter`` — box/uniform averaging (separable, O(N) per pixel)
    ``GaussianFilter`` — Gaussian smoothing (separable, O(N) per pixel)

Rank Filters
    ``MedianFilter`` — rank-based median (edge-preserving denoising)
    ``MinFilter`` — local minimum (morphological erosion)
    ``MaxFilter`` — local maximum (morphological dilation)

Statistical Filters
    ``StdDevFilter`` — local standard deviation via variance decomposition

Adaptive Filters
    ``LeeFilter`` — SAR speckle filter for real-valued intensity
    ``ComplexLeeFilter`` — SAR speckle filter for complex SLC data

Phase Filters
    ``PhaseGradientFilter`` — windowed phase gradient for complex SAR data

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
2026-02-11

Modified
--------
2026-02-11
"""

from grdl.image_processing.filters.linear import GaussianFilter, MeanFilter
from grdl.image_processing.filters.rank import MaxFilter, MedianFilter, MinFilter
from grdl.image_processing.filters.statistical import StdDevFilter
from grdl.image_processing.filters.speckle import ComplexLeeFilter, LeeFilter
from grdl.image_processing.filters.phase import PhaseGradientFilter

__all__ = [
    'MeanFilter',
    'GaussianFilter',
    'MedianFilter',
    'MinFilter',
    'MaxFilter',
    'StdDevFilter',
    'LeeFilter',
    'ComplexLeeFilter',
    'PhaseGradientFilter',
]
