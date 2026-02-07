# -*- coding: utf-8 -*-
"""
Rank Filters - Port of ImageJ's RankFilters plugin.

Implements spatial rank-order filters: Median, Min (Erosion), Max
(Dilation), Mean, Variance, and Despeckle (3x3 Median). Each filter
operates on a local circular neighborhood and selects a statistic
from the ranked pixel values within that window.

Particularly useful for:
- SAR speckle reduction (Median, Despeckle)
- Morphological erosion/dilation of continuous-valued imagery (Min/Max)
- Texture analysis via local variance (Variance)
- Noise estimation for adaptive filtering (Variance)
- Pre-processing MSI/HSI bands before classification (Median)
- Hot/dead pixel removal in thermal and PAN sensors (Median)

Attribution
-----------
ImageJ implementation: Michael Schmid (Vienna University of Technology),
based on earlier code by Wayne Rasband. Source:
``ij/plugin/filter/RankFilters.java`` in ImageJ 1.54j.
ImageJ 1.x source is in the public domain.

Dependencies
------------
scipy

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

# Standard library
from typing import Any

# Third-party
import numpy as np
from scipy.ndimage import (
    median_filter,
    minimum_filter,
    maximum_filter,
    uniform_filter,
)

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version, processor_tags


RANK_METHODS = ('median', 'min', 'max', 'mean', 'variance', 'despeckle')


@processor_tags(modalities=['SAR', 'PAN', 'EO', 'MSI', 'HSI', 'thermal'], category='filters')
@processor_version('1.54j')
class RankFilters(ImageTransform):
    """Rank-order spatial filters, ported from ImageJ 1.54j.

    Applies a spatial rank filter with a square neighborhood of side
    length ``2 * radius + 1``.

    Parameters
    ----------
    method : str
        Filter type. One of:

        - ``'median'``: Median value in the window. Excellent for
          impulse noise and moderate SAR speckle.
        - ``'min'``: Minimum value (grayscale erosion). Shrinks bright
          regions.
        - ``'max'``: Maximum value (grayscale dilation). Expands bright
          regions.
        - ``'mean'``: Arithmetic mean in the window. Equivalent to a
          box filter / uniform filter.
        - ``'variance'``: Local variance in the window. High values
          indicate texture or edges.
        - ``'despeckle'``: 3x3 median filter (radius forced to 1).
          ImageJ's Process > Noise > Despeckle.

    radius : int
        Half-size of the filter window. Window is
        ``(2*radius+1) x (2*radius+1)`` pixels. Default 2 (5x5).
        Ignored for ``'despeckle'`` which always uses radius=1.

    Notes
    -----
    Port of ``ij/plugin/filter/RankFilters.java`` from ImageJ 1.54j
    (public domain). Original implementation by Michael Schmid.

    ImageJ uses a circular (disk) kernel; this port uses a square
    kernel via scipy.ndimage for performance. The difference is minor
    for small radii and negligible for most remote sensing applications.

    Examples
    --------
    SAR speckle reduction with 5x5 median:

    >>> from grdl.imagej import RankFilters
    >>> mf = RankFilters(method='median', radius=2)
    >>> denoised = mf.apply(sar_amplitude)

    Texture map via local variance:

    >>> vf = RankFilters(method='variance', radius=3)
    >>> texture = vf.apply(pan_image)
    """

    __imagej_source__ = 'ij/plugin/filter/RankFilters.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = False

    def __init__(
        self,
        method: str = 'median',
        radius: int = 2,
    ) -> None:
        method_lower = method.lower()
        if method_lower not in RANK_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Must be one of {RANK_METHODS}"
            )
        if radius < 1 and method_lower != 'despeckle':
            raise ValueError(f"radius must be >= 1, got {radius}")

        self.method = method_lower
        self.radius = 1 if method_lower == 'despeckle' else radius

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply rank filter to a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Filtered image, dtype float64, same shape as input.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(
                f"Expected 2D image, got shape {source.shape}"
            )

        image = source.astype(np.float64)
        size = 2 * self.radius + 1

        if self.method in ('median', 'despeckle'):
            return median_filter(image, size=size, mode='nearest')

        elif self.method == 'min':
            return minimum_filter(image, size=size, mode='nearest')

        elif self.method == 'max':
            return maximum_filter(image, size=size, mode='nearest')

        elif self.method == 'mean':
            return uniform_filter(image, size=size, mode='nearest')

        elif self.method == 'variance':
            mean = uniform_filter(image, size=size, mode='nearest')
            mean_sq = uniform_filter(image * image, size=size, mode='nearest')
            return np.maximum(mean_sq - mean * mean, 0.0)

        raise ValueError(f"Unknown method: {self.method}")
