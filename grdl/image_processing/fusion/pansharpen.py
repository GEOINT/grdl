# -*- coding: utf-8 -*-
"""
Pan-Sharpening & Multi-Band Fusion Transforms.

Provides radiometric fusion transforms to combine high-resolution panchromatic
imagery with multispectral image bands:
- ``BroveyPanSharpenTransform``: Multiplicative Brovey ratio fusion
- ``SimpleMeanFusionTransform``: Weighted linear average fusion
- ``IHSPanSharpenTransform``: Intensity-Hue-Saturation pan replacement

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

from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.vocabulary import ImageModality, ProcessorCategory


def brovey_pan_sharpen(
    pan: np.ndarray,
    ms: np.ndarray,
    pan_weight: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Brovey Pan-Sharpening algorithm.

    Formula: Fused_i = MS_i * (PAN * pan_weight) / (mean(MS) + eps)
    """
    if ms.ndim == 2:
        ms = ms[np.newaxis, :, :]

    pan_2d = pan[0] if pan.ndim == 3 else pan
    ms_float = ms.astype(np.float32)
    pan_float = pan_2d.astype(np.float32) * float(pan_weight)

    ms_mean = np.mean(ms_float, axis=0) + eps
    ratio = pan_float / ms_mean
    fused = ms_float * ratio[np.newaxis, :, :]
    return fused.astype(ms.dtype)


def simple_mean_fusion(
    pan: np.ndarray,
    ms: np.ndarray,
    pan_weight: float = 0.5,
) -> np.ndarray:
    """Simple weighted mean fusion: (1 - pan_weight) * MS + pan_weight * PAN."""
    if ms.ndim == 2:
        ms = ms[np.newaxis, :, :]

    pan_2d = pan[0] if pan.ndim == 3 else pan
    ms_float = ms.astype(np.float32)
    pan_float = pan_2d.astype(np.float32)

    w = float(pan_weight)
    fused = (1.0 - w) * ms_float + w * pan_float[np.newaxis, :, :]
    return fused.astype(ms.dtype)


@processor_version("1.0.0")
@processor_tags(
    category=ProcessorCategory.ENHANCE,
    modalities=[ImageModality.PAN, ImageModality.MSI],
)
class BroveyPanSharpenTransform(ImageTransform):
    """Brovey pan-sharpening transform for multispectral imagery.

    When called with a single 3D array where band 0 is PAN and bands 1+ are MS,
    or with `pan` passed via kwargs.
    """

    pan_weight: Annotated[float, Range(min=0.1, max=5.0), Desc("Pan intensity multiplier")] = 1.0

    def __init__(self, pan_weight: float = 1.0) -> None:
        self.pan_weight = float(pan_weight)

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        pan = kwargs.get("pan")
        if pan is not None:
            return brovey_pan_sharpen(pan, source, pan_weight=self.pan_weight)

        if source.ndim == 3 and source.shape[0] >= 2:
            # Band 0 is pan, remainder is ms
            return brovey_pan_sharpen(source[0], source[1:], pan_weight=self.pan_weight)

        return source


@processor_version("1.0.0")
@processor_tags(
    category=ProcessorCategory.ENHANCE,
    modalities=[ImageModality.PAN, ImageModality.MSI],
)
class SimpleMeanFusionTransform(ImageTransform):
    """Weighted linear average fusion."""

    pan_weight: Annotated[float, Range(min=0.0, max=1.0), Desc("Weight of pan relative to multispectral")] = 0.5

    def __init__(self, pan_weight: float = 0.5) -> None:
        self.pan_weight = float(pan_weight)

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        pan = kwargs.get("pan")
        if pan is not None:
            return simple_mean_fusion(pan, source, pan_weight=self.pan_weight)

        if source.ndim == 3 and source.shape[0] >= 2:
            return simple_mean_fusion(source[0], source[1:], pan_weight=self.pan_weight)

        return source
