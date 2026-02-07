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

Components (organized by ImageJ menu category)
------------------------------------------------
Process > Filters (filters/):
- RankFilters: Median, Min, Max, Mean, Variance, Despeckle
- UnsharpMask: Gaussian-based sharpening

Process > Subtract Background (background/):
- RollingBallBackground: Background subtraction via Sternberg's rolling ball

Process > Binary (binary/):
- MorphologicalFilter: Erode, Dilate, Open, Close, TopHat, BlackHat, Gradient

Process > Enhance Contrast (enhance/):
- CLAHE: Contrast Limited Adaptive Histogram Equalization
- GammaCorrection: Power-law intensity transform

Process > Find Edges (edges/):
- EdgeDetector: Sobel, Prewitt, Roberts, LoG, Scharr

Process > FFT (fft/):
- FFTBandpassFilter: Frequency-domain bandpass and stripe suppression

Process > Find Maxima (find_maxima/):
- FindMaxima: Prominence-based peak/target detection

Image > Adjust > Threshold (threshold/):
- AutoLocalThreshold: Local thresholding (Bernsen, Niblack, Sauvola, etc.)

Plugins > Segmentation (segmentation/):
- StatisticalRegionMerging: SRM region-based segmentation

Image > Stacks (stacks/):
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
Steven Siebert

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

# Process > Filters
from grdl.imagej.filters import RankFilters, UnsharpMask

# Process > Subtract Background
from grdl.imagej.background import RollingBallBackground

# Process > Binary
from grdl.imagej.binary import MorphologicalFilter

# Process > Enhance Contrast
from grdl.imagej.enhance import CLAHE, GammaCorrection

# Process > Find Edges
from grdl.imagej.edges import EdgeDetector

# Process > FFT
from grdl.imagej.fft import FFTBandpassFilter

# Process > Find Maxima
from grdl.imagej.find_maxima import FindMaxima

# Image > Adjust > Threshold
from grdl.imagej.threshold import AutoLocalThreshold

# Plugins > Segmentation
from grdl.imagej.segmentation import StatisticalRegionMerging

# Image > Stacks
from grdl.imagej.stacks import ZProjection

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
