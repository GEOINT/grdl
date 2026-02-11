# -*- coding: utf-8 -*-
"""
Linear Spatial Filters - Mean (box) and Gaussian smoothing filters.

Provides separable linear convolution filters backed by scipy's C-optimized
implementations. Both filters run in O(N) per pixel regardless of kernel
size due to separable decomposition into two 1D passes.

- ``MeanFilter``: Uniform averaging via ``scipy.ndimage.uniform_filter``
- ``GaussianFilter``: Gaussian smoothing via ``scipy.ndimage.gaussian_filter``

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

# Standard library
from typing import Annotated, Any

# Third-party
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

# GRDL internal
from grdl.image_processing.base import BandwiseTransformMixin, ImageTransform
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.image_processing.filters._validation import (
    validate_kernel_size,
    validate_mode,
)
from grdl.vocabulary import ProcessorCategory


@processor_version('1.0.0')
@processor_tags(category=ProcessorCategory.FILTERS)
class MeanFilter(BandwiseTransformMixin, ImageTransform):
    """Spatial mean (box) filter using separable uniform convolution.

    Replaces each pixel with the arithmetic mean of its local neighborhood.
    Backed by ``scipy.ndimage.uniform_filter`` which decomposes into two
    1D passes, making it O(N) per pixel regardless of kernel size.

    Parameters
    ----------
    kernel_size : int
        Square kernel side length in pixels. Must be odd and >= 3.
        Default is 3.
    mode : str
        Boundary handling mode. One of ``'reflect'``, ``'constant'``,
        ``'nearest'``, ``'wrap'``. Default is ``'reflect'``.

    Examples
    --------
    >>> from grdl.image_processing.filters import MeanFilter
    >>> f = MeanFilter(kernel_size=5)
    >>> smoothed = f.apply(image)
    """

    __gpu_compatible__ = False

    kernel_size: Annotated[int, Range(min=3, max=101),
                           Desc('Square kernel side length (odd)')] = 3
    mode: Annotated[str, Options('reflect', 'constant', 'nearest', 'wrap'),
                    Desc('Boundary handling mode')] = 'reflect'

    def __init__(self, kernel_size: int = 3, mode: str = 'reflect') -> None:
        validate_kernel_size(kernel_size)
        validate_mode(mode)
        self.kernel_size = kernel_size
        self.mode = mode

    def _apply_2d(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply mean filter to a single 2D band.

        Parameters
        ----------
        source : np.ndarray
            2D image array, shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Smoothed image, same shape, float32 or float64.
        """
        params = self._resolve_params(kwargs)
        ks = params['kernel_size']
        mode = params['mode']
        validate_kernel_size(ks)
        validate_mode(mode)

        if source.dtype == np.float64:
            working = source.astype(np.float64)
        else:
            working = source.astype(np.float32)
        return uniform_filter(working, size=ks, mode=mode)


@processor_version('1.0.0')
@processor_tags(category=ProcessorCategory.FILTERS)
class GaussianFilter(BandwiseTransformMixin, ImageTransform):
    """Gaussian smoothing filter using separable convolution.

    Applies a Gaussian blur with the specified standard deviation. Backed
    by ``scipy.ndimage.gaussian_filter`` which decomposes into two 1D
    passes, making it O(N) per pixel regardless of sigma.

    Parameters
    ----------
    sigma : float
        Gaussian standard deviation in pixels. Default is 1.0.
    truncate : float
        Truncate the filter at this many standard deviations. Controls
        the effective kernel extent. Default is 4.0.
    mode : str
        Boundary handling mode. One of ``'reflect'``, ``'constant'``,
        ``'nearest'``, ``'wrap'``. Default is ``'reflect'``.

    Examples
    --------
    >>> from grdl.image_processing.filters import GaussianFilter
    >>> f = GaussianFilter(sigma=2.0)
    >>> smoothed = f.apply(image)
    """

    __gpu_compatible__ = False

    sigma: Annotated[float, Range(min=0.1, max=100.0),
                     Desc('Gaussian standard deviation')] = 1.0
    truncate: Annotated[float, Range(min=1.0, max=10.0),
                        Desc('Truncate filter at this many sigmas')] = 4.0
    mode: Annotated[str, Options('reflect', 'constant', 'nearest', 'wrap'),
                    Desc('Boundary handling mode')] = 'reflect'

    def __init__(
        self,
        sigma: float = 1.0,
        truncate: float = 4.0,
        mode: str = 'reflect',
    ) -> None:
        validate_mode(mode)
        self.sigma = sigma
        self.truncate = truncate
        self.mode = mode

    def _apply_2d(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Gaussian filter to a single 2D band.

        Parameters
        ----------
        source : np.ndarray
            2D image array, shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Smoothed image, same shape, float32 or float64.
        """
        params = self._resolve_params(kwargs)
        mode = params['mode']
        validate_mode(mode)

        if source.dtype == np.float64:
            working = source.astype(np.float64)
        else:
            working = source.astype(np.float32)
        return gaussian_filter(
            working,
            sigma=params['sigma'],
            truncate=params['truncate'],
            mode=mode,
        )
