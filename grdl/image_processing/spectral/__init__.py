# -*- coding: utf-8 -*-
"""
Spectral Image Processing Module for Multispectral and Hyperspectral Imagery.
"""

from grdl.image_processing.spectral.indices import (
    SpectralIndexTransform,
    compute_spectral_index,
)

__all__ = [
    "SpectralIndexTransform",
    "compute_spectral_index",
]
