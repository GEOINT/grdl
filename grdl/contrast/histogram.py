# -*- coding: utf-8 -*-
"""
Histogram-based contrast adjustments.

- :class:`HistogramEqualization` — global CDF remap (numpy only).
- :class:`CLAHE` — Contrast-Limited Adaptive Histogram Equalization
  (delegates to :func:`skimage.exposure.equalize_adapthist`).

Both return ``float32`` in ``[0, 1]``.

Dependencies
------------
scikit-image (CLAHE only)

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-04-26

Modified
--------
2026-04-26
"""

# Standard library
from typing import Annotated, Any

# Third-party
import numpy as np

try:
    from skimage.exposure import equalize_adapthist as _skimage_clahe
    _HAS_SKIMAGE = True
except ImportError:
    _skimage_clahe = None
    _HAS_SKIMAGE = False

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


# =====================================================================
# Histogram equalization (global)
# =====================================================================


@processor_version('1.0.0')
@processor_tags(
    modalities=[IM.SAR, IM.EO, IM.MSI, IM.PAN, IM.HSI],
    category=PC.ENHANCE,
    description='Global histogram equalization (CDF remap).',
)
class HistogramEqualization(ImageTransform):
    """Global histogram equalization.

    Computes the CDF over finite samples and remaps the input through it.
    Output is ``float32`` in ``[0, 1]``.  Non-finite input samples are
    preserved as ``NaN`` in the output.

    Parameters
    ----------
    n_bins : int, default=256
        Number of histogram bins for the CDF.
    """

    n_bins: Annotated[
        int, Range(min=8, max=65536),
        Desc('Histogram bin count'),
    ] = 256

    def __init__(self, n_bins: int = 256) -> None:
        if n_bins < 8:
            raise ValueError(f"n_bins must be >= 8, got {n_bins}")
        self.n_bins = int(n_bins)

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        params = self._resolve_params(kwargs)
        n_bins = params['n_bins']

        amplitude = np.abs(source) if np.iscomplexobj(source) else source
        amplitude = np.asarray(amplitude, dtype=np.float64)
        finite_mask = np.isfinite(amplitude)
        finite = amplitude[finite_mask]
        if finite.size == 0:
            return np.full(amplitude.shape, np.nan, dtype=np.float32)

        hist, edges = np.histogram(finite, bins=n_bins)
        cdf = np.cumsum(hist).astype(np.float64)
        if cdf[-1] == 0:
            return np.zeros(amplitude.shape, dtype=np.float32)
        cdf = cdf / cdf[-1]

        # Use bin centers as interpolation x-grid.
        centers = 0.5 * (edges[:-1] + edges[1:])
        out = np.full(amplitude.shape, np.nan, dtype=np.float64)
        out[finite_mask] = np.interp(amplitude[finite_mask], centers, cdf)
        return out.astype(np.float32)


# =====================================================================
# CLAHE — Contrast-Limited Adaptive Histogram Equalization
# =====================================================================


@processor_version('1.0.0')
@processor_tags(
    modalities=[IM.SAR, IM.EO, IM.MSI, IM.PAN, IM.HSI],
    category=PC.ENHANCE,
    description='Contrast-Limited Adaptive Histogram Equalization (skimage).',
)
class CLAHE(ImageTransform):
    """Contrast-Limited Adaptive Histogram Equalization.

    Wraps :func:`skimage.exposure.equalize_adapthist`.  Operates only on
    2D (grayscale) input.  Input is normalized to ``[0, 1]`` internally
    using its finite min/max before CLAHE is applied.

    Parameters
    ----------
    kernel_size : int, default=64
        Tile side length in pixels.  Smaller → more local adaptation.
    clip_limit : float, default=0.01
        Clipping limit (normalized contrast cap, in ``[0, 1]``).

    Raises
    ------
    ImportError
        On construction if scikit-image is not installed.
    """

    kernel_size: Annotated[
        int, Range(min=4, max=4096),
        Desc('CLAHE tile side length in pixels'),
    ] = 64
    clip_limit: Annotated[
        float, Range(min=0.0, max=1.0),
        Desc('Normalized contrast clipping limit'),
    ] = 0.01

    def __init__(
        self, kernel_size: int = 64, clip_limit: float = 0.01,
    ) -> None:
        if not _HAS_SKIMAGE:
            raise ImportError(
                "CLAHE requires scikit-image. Install with: "
                "pip install scikit-image>=0.20"
            )
        if kernel_size < 4:
            raise ValueError(
                f"kernel_size must be >= 4, got {kernel_size}"
            )
        if not (0.0 <= clip_limit <= 1.0):
            raise ValueError(
                f"clip_limit must be in [0, 1], got {clip_limit}"
            )
        self.kernel_size = int(kernel_size)
        self.clip_limit = float(clip_limit)

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        params = self._resolve_params(kwargs)

        amplitude = np.abs(source) if np.iscomplexobj(source) else source
        amplitude = np.asarray(amplitude, dtype=np.float64)
        if amplitude.ndim != 2:
            raise ValueError(
                f"CLAHE requires a 2D array, got shape {amplitude.shape}"
            )

        finite_mask = np.isfinite(amplitude)
        if not np.any(finite_mask):
            return np.zeros(amplitude.shape, dtype=np.float32)

        finite = amplitude[finite_mask]
        lo = float(np.min(finite))
        hi = float(np.max(finite))

        norm = np.zeros(amplitude.shape, dtype=np.float64)
        if hi - lo > np.finfo(np.float64).eps:
            norm[finite_mask] = (finite - lo) / (hi - lo)

        out = _skimage_clahe(
            norm,
            kernel_size=params['kernel_size'],
            clip_limit=params['clip_limit'],
        )
        return out.astype(np.float32)
