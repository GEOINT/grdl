# -*- coding: utf-8 -*-
"""
Spectral Indices & Band Math Transforms.

Provides standard multispectral and hyperspectral vegetation, water, and soil indices:
- NDVI: Normalized Difference Vegetation Index
- NDWI: Normalized Difference Water Index
- NDBI: Normalized Difference Built-up Index
- SAVI: Soil-Adjusted Vegetation Index
- RATIO: Band A / Band B Simple Ratio

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
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.vocabulary import ImageModality, ProcessorCategory


def compute_spectral_index(
    data: np.ndarray,
    index_type: str = "NDVI",
    nir_idx: int = 3,
    red_idx: int = 2,
    green_idx: int = 1,
    swir_idx: int = 5,
    savi_l: float = 0.5,
) -> np.ndarray:
    """Compute standard spectral index from a 3D (C, H, W) array."""
    if data.ndim != 3:
        raise ValueError("Data must be 3D (C, H, W) for spectral index calculation")

    num_bands = data.shape[0]

    def _get_band(idx: int) -> np.ndarray:
        return data[min(max(0, idx), num_bands - 1)].astype(np.float32)

    index_type = index_type.upper()
    eps = 1e-7

    if index_type == "NDVI":
        nir = _get_band(nir_idx)
        red = _get_band(red_idx)
        denom = nir + red
        denom[denom == 0] = eps
        return (nir - red) / denom

    elif index_type == "NDWI":
        green = _get_band(green_idx)
        nir = _get_band(nir_idx)
        denom = green + nir
        denom[denom == 0] = eps
        return (green - nir) / denom

    elif index_type == "NDBI":
        swir = _get_band(swir_idx)
        nir = _get_band(nir_idx)
        denom = swir + nir
        denom[denom == 0] = eps
        return (swir - nir) / denom

    elif index_type == "SAVI":
        nir = _get_band(nir_idx)
        red = _get_band(red_idx)
        denom = nir + red + float(savi_l)
        denom[denom == 0] = eps
        return ((nir - red) / denom) * (1.0 + float(savi_l))

    elif index_type == "RATIO":
        band_a = _get_band(nir_idx)
        band_b = _get_band(red_idx)
        denom = band_b.copy()
        denom[denom == 0] = eps
        return band_a / denom

    else:
        raise ValueError(f"Unsupported index type: {index_type}")


@processor_version("1.0.0")
@processor_tags(
    category=ProcessorCategory.MATH,
    modalities=[ImageModality.MSI, ImageModality.HSI],
)
class SpectralIndexTransform(ImageTransform):
    """Spectral Index Transform (NDVI, NDWI, NDBI, SAVI, RATIO)."""

    index_type: Annotated[
        str, Options("NDVI", "NDWI", "NDBI", "SAVI", "RATIO"), Desc("Spectral index type")
    ] = "NDVI"
    nir_idx: Annotated[int, Range(min=0, max=1000), Desc("NIR / Band 1 zero-based index")] = 3
    red_idx: Annotated[int, Range(min=0, max=1000), Desc("Red / Band 2 zero-based index")] = 2
    green_idx: Annotated[int, Range(min=0, max=1000), Desc("Green zero-based index")] = 1
    swir_idx: Annotated[int, Range(min=0, max=1000), Desc("SWIR zero-based index")] = 5
    savi_l: Annotated[float, Range(min=0.0, max=1.0), Desc("Soil adjustment factor for SAVI")] = 0.5

    def __init__(
        self,
        index_type: str = "NDVI",
        nir_idx: int = 3,
        red_idx: int = 2,
        green_idx: int = 1,
        swir_idx: int = 5,
        savi_l: float = 0.5,
    ) -> None:
        self.index_type = index_type
        self.nir_idx = int(nir_idx)
        self.red_idx = int(red_idx)
        self.green_idx = int(green_idx)
        self.swir_idx = int(swir_idx)
        self.savi_l = float(savi_l)

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        if source.ndim == 2:
            return source
        return compute_spectral_index(
            source,
            index_type=self.index_type,
            nir_idx=self.nir_idx,
            red_idx=self.red_idx,
            green_idx=self.green_idx,
            swir_idx=self.swir_idx,
            savi_l=self.savi_l,
        )
