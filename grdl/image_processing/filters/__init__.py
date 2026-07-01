# -*- coding: utf-8 -*-
"""
Spatial Filters - Statistical, rank, and adaptive spatial image filters.

Provides general-purpose and SAR-specific spatial image filters.

General Filters (scalar, band-independent)
    Linear
        ``MeanFilter`` — box/uniform averaging (separable, O(N) per pixel)
        ``GaussianFilter`` — Gaussian smoothing (separable, O(N) per pixel)
    Rank
        ``MedianFilter`` — rank-based median (edge-preserving denoising)
        ``MinFilter`` — local minimum (morphological erosion)
        ``MaxFilter`` — local maximum (morphological dilation)
    Statistical
        ``StdDevFilter`` — local standard deviation via variance decomposition

SAR Scalar Filters (per-band; inherit ``BandwiseTransformMixin + SARFilter``)
    ``LeeFilter`` — Lee adaptive speckle filter for real-valued intensity
    ``ComplexLeeFilter`` — Lee adaptive speckle filter for complex SLC data
    ``LeeSigmaFilter`` — Lee Sigma filter with probability-based neighbour selection

Phase Filters
    ``PhaseGradientFilter`` — windowed phase gradient for complex SAR data

SAR Polarimetric Filters (matrix-aware; inherit ``SARFilter`` directly)
    (future: further polarimetric filters)

Dependencies
------------
scipy

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

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
from grdl.image_processing.filters.sar_base import SARFilter
from grdl.image_processing.filters.standard_lee import ComplexLeeFilter, LeeFilter
from grdl.image_processing.filters.enhanced_lee import EnhancedLeeFilter
from grdl.image_processing.filters.lee_sigma import LeeSigmaFilter
from grdl.image_processing.filters.refined_lee import RefinedLeeFilter
from grdl.image_processing.filters.phase import PhaseGradientFilter

__all__ = [
    'MeanFilter',
    'GaussianFilter',
    'MedianFilter',
    'MinFilter',
    'MaxFilter',
    'StdDevFilter',
    'SARFilter',
    'LeeFilter',
    'ComplexLeeFilter',
    'EnhancedLeeFilter',
    'LeeSigmaFilter',
    'RefinedLeeFilter',
    'PhaseGradientFilter',
]
