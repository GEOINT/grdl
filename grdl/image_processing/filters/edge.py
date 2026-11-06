# -*- coding: utf-8 -*-
"""
Edge and Spatial Enhancement Filters - Sobel, Laplacian, Unsharp Mask, and High Pass.

Provides edge detection and spatial sharpening transforms for optical, SAR, and
multispectral imagery. Backed by scipy's C-optimized ndimage filters.

- ``SobelFilter``: First-derivative gradient magnitude / directional operator
- ``LaplacianFilter``: Second-derivative discrete Laplacian edge detector
- ``UnsharpMaskFilter``: Contrast enhancement via blurred subtraction
- ``HighPassFilter``: Fine spatial detail extraction

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
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.vocabulary import ImageModality, ProcessorCategory


@processor_version("1.0.0")
@processor_tags(
    category=ProcessorCategory.EDGES,
    modalities=[ImageModality.PAN, ImageModality.EO, ImageModality.MSI, ImageModality.SAR],
)
class SobelFilter(BandwiseTransformMixin, ImageTransform):
    """Sobel edge enhancement and gradient filter.

    Computes 3x3 Sobel gradient components or the composite gradient magnitude:
    sqrt(Gx^2 + Gy^2).
    """

    output_type: Annotated[
        str, Options("magnitude", "gradient_x", "gradient_y"), Desc("Gradient component")
    ] = "magnitude"
    mode: Annotated[
        str, Options("reflect", "constant", "nearest", "mirror", "wrap"), Desc("Boundary mode")
    ] = "reflect"

    def __init__(self, output_type: str = "magnitude", mode: str = "reflect") -> None:
        self.output_type = output_type
        self.mode = mode

    def _apply_2d(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        gx = ndimage.sobel(source, axis=1, mode=self.mode)
        gy = ndimage.sobel(source, axis=0, mode=self.mode)
        if self.output_type == "gradient_x":
            return gx.astype(np.float32)
        elif self.output_type == "gradient_y":
            return gy.astype(np.float32)
        return np.hypot(gx, gy).astype(np.float32)


@processor_version("1.0.0")
@processor_tags(
    category=ProcessorCategory.EDGES,
    modalities=[ImageModality.PAN, ImageModality.EO, ImageModality.MSI, ImageModality.SAR],
)
class LaplacianFilter(BandwiseTransformMixin, ImageTransform):
    """Discrete Laplacian second-derivative edge detector."""

    mode: Annotated[
        str, Options("reflect", "constant", "nearest", "mirror", "wrap"), Desc("Boundary mode")
    ] = "reflect"

    def __init__(self, mode: str = "reflect") -> None:
        self.mode = mode

    def _apply_2d(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.abs(ndimage.laplace(source, mode=self.mode)).astype(np.float32)


@processor_version("1.0.0")
@processor_tags(
    category=ProcessorCategory.ENHANCE,
    modalities=[ImageModality.PAN, ImageModality.EO, ImageModality.MSI],
)
class UnsharpMaskFilter(BandwiseTransformMixin, ImageTransform):
    """Unsharp masking spatial sharpening filter: source + amount * (source - blurred)."""

    radius: Annotated[float, Range(min=0.1, max=50.0), Desc("Gaussian blur sigma")] = 1.0
    amount: Annotated[float, Range(min=0.1, max=10.0), Desc("Sharpening strength")] = 1.5
    mode: Annotated[
        str, Options("reflect", "constant", "nearest", "mirror", "wrap"), Desc("Boundary mode")
    ] = "reflect"

    def __init__(self, radius: float = 1.0, amount: float = 1.5, mode: str = "reflect") -> None:
        self.radius = radius
        self.amount = amount
        self.mode = mode

    def _apply_2d(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        blurred = ndimage.gaussian_filter(source, sigma=self.radius, mode=self.mode)
        sharpened = source + self.amount * (source - blurred)
        return sharpened.astype(source.dtype)


@processor_version("1.0.0")
@processor_tags(
    category=ProcessorCategory.FILTERS,
    modalities=[ImageModality.PAN, ImageModality.EO, ImageModality.MSI],
)
class HighPassFilter(BandwiseTransformMixin, ImageTransform):
    """High-pass filter retaining fine spatial details: source - gaussian_blur(source)."""

    cutoff: Annotated[float, Range(min=0.1, max=50.0), Desc("Gaussian cutoff sigma")] = 2.0
    mode: Annotated[
        str, Options("reflect", "constant", "nearest", "mirror", "wrap"), Desc("Boundary mode")
    ] = "reflect"

    def __init__(self, cutoff: float = 2.0, mode: str = "reflect") -> None:
        self.cutoff = cutoff
        self.mode = mode

    def _apply_2d(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        blurred = ndimage.gaussian_filter(source, sigma=self.cutoff, mode=self.mode)
        return (source - blurred).astype(np.float32)
