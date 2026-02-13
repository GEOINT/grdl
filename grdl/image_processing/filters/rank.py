# -*- coding: utf-8 -*-
"""
Rank Filters - Median, minimum, and maximum spatial filters.

Provides non-linear rank-based filters backed by scipy's C-optimized
implementations. Non-separable (O(k^2) per pixel) but highly optimized
in C for common kernel sizes.

- ``MedianFilter``: Rank-based median for noise removal
- ``MinFilter``: Local minimum (morphological erosion)
- ``MaxFilter``: Local maximum (morphological dilation)

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
from scipy.ndimage import maximum_filter, median_filter, minimum_filter

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
class MedianFilter(BandwiseTransformMixin, ImageTransform):
    """Spatial median filter for noise removal.

    Replaces each pixel with the median of its local neighborhood.
    Excellent for removing salt-and-pepper noise while preserving edges.
    Backed by ``scipy.ndimage.median_filter`` (C-optimized).

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
    >>> from grdl.image_processing.filters import MedianFilter
    >>> f = MedianFilter(kernel_size=5)
    >>> denoised = f.apply(noisy_image)
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
        """Apply median filter to a single 2D band.

        Parameters
        ----------
        source : np.ndarray
            2D image array, shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Filtered image, same shape and dtype as input.
        """
        params = self._resolve_params(kwargs)
        ks = params['kernel_size']
        mode = params['mode']
        validate_kernel_size(ks)
        validate_mode(mode)
        return median_filter(source, size=ks, mode=mode)


@processor_version('1.0.0')
@processor_tags(category=ProcessorCategory.FILTERS)
class MinFilter(BandwiseTransformMixin, ImageTransform):
    """Local minimum filter (morphological erosion).

    Replaces each pixel with the minimum value in its local neighborhood.
    Useful for dark feature detection, removing bright noise, and as a
    morphological erosion operator. Backed by
    ``scipy.ndimage.minimum_filter`` (C-optimized).

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
    >>> from grdl.image_processing.filters import MinFilter
    >>> f = MinFilter(kernel_size=3)
    >>> eroded = f.apply(image)
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
        """Apply minimum filter to a single 2D band.

        Parameters
        ----------
        source : np.ndarray
            2D image array, shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Filtered image, same shape and dtype as input.
        """
        params = self._resolve_params(kwargs)
        ks = params['kernel_size']
        mode = params['mode']
        validate_kernel_size(ks)
        validate_mode(mode)
        return minimum_filter(source, size=ks, mode=mode)


@processor_version('1.0.0')
@processor_tags(category=ProcessorCategory.FILTERS)
class MaxFilter(BandwiseTransformMixin, ImageTransform):
    """Local maximum filter (morphological dilation).

    Replaces each pixel with the maximum value in its local neighborhood.
    Useful for bright feature detection, removing dark noise, and as a
    morphological dilation operator. Backed by
    ``scipy.ndimage.maximum_filter`` (C-optimized).

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
    >>> from grdl.image_processing.filters import MaxFilter
    >>> f = MaxFilter(kernel_size=3)
    >>> dilated = f.apply(image)
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
        """Apply maximum filter to a single 2D band.

        Parameters
        ----------
        source : np.ndarray
            2D image array, shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Filtered image, same shape and dtype as input.
        """
        params = self._resolve_params(kwargs)
        ks = params['kernel_size']
        mode = params['mode']
        validate_kernel_size(ks)
        validate_mode(mode)
        return maximum_filter(source, size=ks, mode=mode)
