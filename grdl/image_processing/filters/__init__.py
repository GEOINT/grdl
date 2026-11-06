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
    Edge & Sharpening
        ``SobelFilter`` — 3x3 Sobel gradient operator
        ``LaplacianFilter`` — discrete Laplacian second-derivative edge detector
        ``UnsharpMaskFilter`` — unsharp masking spatial sharpening
        ``HighPassFilter`` — fine spatial detail filter

SAR Scalar Filters (per-band; inherit ``BandwiseTransformMixin + SARFilter``)
    ``LeeFilter`` — Lee adaptive speckle filter for real-valued intensity
    ``ComplexLeeFilter`` — Lee adaptive speckle filter for complex SLC data
    ``EnhancedLeeFilter`` — amplitude-domain smoothing with target protection
    ``LeeSigmaFilter`` — Lee Sigma filter with probability-based neighbour selection
    ``FrostFilter`` — exponentially damped adaptive speckle filter
    ``KuanFilter`` — MMSE multiplicative speckle filter
    ``BoxcarFilter`` — uniform moving average spatial filter

Phase Filters
    ``PhaseGradientFilter`` — windowed phase gradient for complex SAR data

SAR Polarimetric Filters (joint matrix processing via ``PolarimetricFilter``)
    ``RefinedLeeFilter`` — directional MMSE filtering using polarimetric span
    ``IDANFilter`` — adaptive neighborhoods; also supports independent image bands
    ``WishartFilter`` — Complex Wishart filter for polarimetric covariance matrices

All filters expose apply(). Independent bands use YX/CYX arrays; joint
polarimetric inputs use CCYX matrices or explicitly described CYX SLC cubes.
See README.md in this directory for input kinds, examples, and migration notes.

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

from importlib import import_module
from typing import TYPE_CHECKING

from grdl.image_processing.filters.enhanced_lee import EnhancedLeeFilter
from grdl.image_processing.filters.lee_sigma import LeeSigmaFilter
from grdl.image_processing.filters.linear import GaussianFilter, MeanFilter
from grdl.image_processing.filters.phase import PhaseGradientFilter
from grdl.image_processing.filters.rank import MaxFilter, MedianFilter, MinFilter
from grdl.image_processing.filters.sar_base import SARFilter
from grdl.image_processing.filters.standard_lee import ComplexLeeFilter, LeeFilter
from grdl.image_processing.filters.statistical import StdDevFilter
from grdl.image_processing.filters.edge import (
    SobelFilter,
    LaplacianFilter,
    UnsharpMaskFilter,
    HighPassFilter,
)
from grdl.image_processing.filters.sar_speckle import (
    FrostFilter,
    KuanFilter,
    BoxcarFilter,
)

if TYPE_CHECKING:
    from grdl.image_processing.filters.idan import IDANFilter
    from grdl.image_processing.filters.refined_lee import RefinedLeeFilter
    from grdl.image_processing.filters.wishart import WishartFilter


def __getattr__(name: str):
    """Load joint filters on demand so linear kernels can serve pol_matrix.

    Joint filters depend on matrix preparation; importing them eagerly would
    cycle when matrix construction imports the shared linear boxcar kernel.
    """
    modules = {
        "RefinedLeeFilter": "refined_lee",
        "IDANFilter": "idan",
        "WishartFilter": "wishart",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{modules[name]}"), name)


__all__ = [
    "MeanFilter",
    "GaussianFilter",
    "MedianFilter",
    "MinFilter",
    "MaxFilter",
    "StdDevFilter",
    "SARFilter",
    "LeeFilter",
    "ComplexLeeFilter",
    "EnhancedLeeFilter",
    "LeeSigmaFilter",
    "RefinedLeeFilter",
    "IDANFilter",
    "PhaseGradientFilter",
    "WishartFilter",
    "SobelFilter",
    "LaplacianFilter",
    "UnsharpMaskFilter",
    "HighPassFilter",
    "FrostFilter",
    "KuanFilter",
    "BoxcarFilter",
]
