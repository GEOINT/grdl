# -*- coding: utf-8 -*-
"""
ImageJ/Fiji Ports - Classic image processing algorithms ported from ImageJ/Fiji.

Pure-NumPy reimplementations of widely used ImageJ and Fiji image processing
algorithms, selected for relevance to remotely sensed imagery (PAN, MSI, HSI,
SAR, thermal). Each class mirrors the original ImageJ/Fiji algorithm as closely
as possible, preserving default parameter values and algorithmic behavior.

All ported components inherit from ``ImageTransform`` and carry attribution
to the original ImageJ/Fiji authors. Version strings mirror the original
source version from which the port was derived.

Components
----------
Spatial Filters:
- RollingBallBackground: Background subtraction via Sternberg's rolling ball
- UnsharpMask: Gaussian-based sharpening
- RankFilters: Median, Min, Max, Mean, Variance, Despeckle
- MorphologicalFilter: Erode, Dilate, Open, Close, TopHat, BlackHat, Gradient

Contrast & Enhancement:
- CLAHE: Contrast Limited Adaptive Histogram Equalization
- GammaCorrection: Power-law intensity transform

Thresholding & Segmentation:
- AutoLocalThreshold: Local thresholding (Bernsen, Niblack, Sauvola, etc.)
- StatisticalRegionMerging: SRM region-based segmentation

Edge & Feature Detection:
- EdgeDetector: Sobel, Prewitt, Roberts, LoG, Scharr
- FindMaxima: Prominence-based peak/target detection

Frequency Domain:
- FFTBandpassFilter: Frequency-domain bandpass and stripe suppression

Stack Operations:
- ZProjection: Stack projection (max, mean, median, min, sum, std)

Attribution
-----------
ImageJ is developed by Wayne Rasband at the U.S. National Institutes of Health.
ImageJ 1.x source code is in the public domain.

Fiji plugins (CLAHE, Auto Local Threshold, Statistical Region Merging) are
distributed under GPL-2. This module provides independent reimplementations
in NumPy, not derivative works of the GPL source, but follows the same
published algorithms and cites the original authors.

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

from grdl.imagej.rolling_ball import RollingBallBackground
from grdl.imagej.clahe import CLAHE
from grdl.imagej.auto_local_threshold import AutoLocalThreshold
from grdl.imagej.unsharp_mask import UnsharpMask
from grdl.imagej.fft_bandpass import FFTBandpassFilter
from grdl.imagej.z_projection import ZProjection
from grdl.imagej.rank_filters import RankFilters
from grdl.imagej.morphology import MorphologicalFilter
from grdl.imagej.edge_detection import EdgeDetector
from grdl.imagej.gamma import GammaCorrection
from grdl.imagej.find_maxima import FindMaxima
from grdl.imagej.statistical_region_merging import StatisticalRegionMerging

__all__ = [
    'RollingBallBackground',
    'CLAHE',
    'AutoLocalThreshold',
    'UnsharpMask',
    'FFTBandpassFilter',
    'ZProjection',
    'RankFilters',
    'MorphologicalFilter',
    'EdgeDetector',
    'GammaCorrection',
    'FindMaxima',
    'StatisticalRegionMerging',
]
