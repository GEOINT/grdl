# -*- coding: utf-8 -*-
"""
SAR Adaptive Speckle Filters - Frost, Kuan, and Boxcar filters.

Implements standard SAR spatial speckle suppression algorithms:
- ``FrostFilter``: Exponentially damped adaptive speckle filter
- ``KuanFilter``: Minimum mean square error (MMSE) multiplicative speckle filter
- ``BoxcarFilter``: Moving average spatial smoothing filter

Author
------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.
"""

from typing import Annotated, Any
import numpy as np
from scipy import ndimage

from grdl.image_processing.base import BandwiseTransformMixin, ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.vocabulary import ImageModality, ProcessorCategory


@processor_version("1.0.0")
@processor_tags(category=ProcessorCategory.FILTERS, modalities=[ImageModality.SAR])
class FrostFilter(BandwiseTransformMixin, ImageTransform):
    """Frost exponentially damped adaptive speckle filter for SAR imagery."""

    window_size: Annotated[int, Range(min=3, max=31), Desc("Square filter window size (odd integer)")] = 5
    damping_factor: Annotated[float, Range(min=0.1, max=10.0), Desc("Exponential damping factor")] = 2.0

    def __init__(self, window_size: int = 5, damping_factor: float = 2.0) -> None:
        self.window_size = int(window_size)
        self.damping_factor = float(damping_factor)

    def _apply_2d(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        img = source.astype(np.float64)
        local_mean = ndimage.uniform_filter(img, size=self.window_size)
        local_sq_mean = ndimage.uniform_filter(img ** 2, size=self.window_size)
        local_var = np.maximum(0.0, local_sq_mean - local_mean ** 2)

        ci = np.sqrt(local_var) / np.maximum(1e-7, local_mean)
        weight = np.exp(-self.damping_factor * ci)
        filtered = local_mean * weight + img * (1.0 - weight)
        return filtered.astype(source.dtype)


@processor_version("1.0.0")
@processor_tags(category=ProcessorCategory.FILTERS, modalities=[ImageModality.SAR])
class KuanFilter(BandwiseTransformMixin, ImageTransform):
    """Kuan adaptive speckle filter (exact MMSE formulation)."""

    window_size: Annotated[int, Range(min=3, max=31), Desc("Square filter window size (odd integer)")] = 5
    num_looks: Annotated[int, Range(min=1, max=64), Desc("Equivalent number of looks (ENL)")] = 1

    def __init__(self, window_size: int = 5, num_looks: int = 1) -> None:
        self.window_size = int(window_size)
        self.num_looks = int(num_looks)

    def _apply_2d(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        img = source.astype(np.float64)
        local_mean = ndimage.uniform_filter(img, size=self.window_size)
        local_sq_mean = ndimage.uniform_filter(img ** 2, size=self.window_size)
        local_var = np.maximum(0.0, local_sq_mean - local_mean ** 2)

        cu = 1.0 / np.sqrt(float(max(1, self.num_looks)))
        ci = np.sqrt(local_var) / np.maximum(1e-7, local_mean)
        weight = np.maximum(0.0, 1.0 - (cu ** 2) / np.maximum(1e-7, ci ** 2)) / (1.0 + cu ** 2)
        weight = np.clip(weight, 0.0, 1.0)

        filtered = local_mean + weight * (img - local_mean)
        return filtered.astype(source.dtype)


@processor_version("1.0.0")
@processor_tags(category=ProcessorCategory.FILTERS, modalities=[ImageModality.SAR, ImageModality.PAN, ImageModality.MSI])
class BoxcarFilter(BandwiseTransformMixin, ImageTransform):
    """Boxcar (uniform average) spatial filter."""

    window_size: Annotated[int, Range(min=1, max=31), Desc("Square filter window size")] = 3

    def __init__(self, window_size: int = 3) -> None:
        self.window_size = int(window_size)

    def _apply_2d(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        return ndimage.uniform_filter(source, size=self.window_size).astype(source.dtype)
