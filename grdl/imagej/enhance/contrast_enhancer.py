# -*- coding: utf-8 -*-
"""
Contrast Enhancer - Port of ImageJ's Process > Enhance Contrast.

Implements linear histogram stretching with configurable saturation
percentage. This is ImageJ's primary tool for adjusting brightness and
contrast, distinct from CLAHE (which is adaptive/local). The enhancer
stretches the histogram so that a specified percentage of pixels are
saturated at both ends, maximizing the use of the dynamic range.

Particularly useful for:
- Stretching SAR amplitude images for visual interpretation
- Display normalization of PAN/EO imagery with narrow dynamic range
- Preparing thermal imagery for visual analysis
- Normalizing individual MSI/HSI bands before false-color display
- Standardizing intensity ranges across multi-date image series
- Quick contrast enhancement for manual interpretation workflows

Attribution
-----------
ImageJ implementation: Wayne Rasband (NIH).
Source: ``ij/plugin/ContrastEnhancer.java`` in ImageJ 1.54j.
ImageJ 1.x source is in the public domain.

Author
------
Jason Fritz

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-09

Modified
--------
2026-02-09
"""

# Standard library
from typing import Annotated, Any

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.ENHANCE)
@processor_version('1.54j')
class ContrastEnhancer(ImageTransform):
    """Linear histogram stretching, ported from ImageJ 1.54j.

    Stretches the image histogram so that a specified fraction of
    pixels is saturated (clipped) at the low and high ends.

    Parameters
    ----------
    saturated : float
        Percentage of pixels to saturate (total, split equally between
        low and high ends). ImageJ default is 0.35%. For example,
        ``saturated=0.35`` clips the darkest 0.175% and brightest
        0.175% of pixels. Must be in [0, 100). Default is 0.35.
    equalize : bool
        If True, apply histogram equalization instead of linear
        stretching. Default is False (linear stretch).
    normalize : bool
        If True, output is normalized to [0, 1]. If False (default),
        output retains the input's value range.

    Attributes
    ----------
    min_val_ : float
        The low clip value used (set after ``apply()``).
    max_val_ : float
        The high clip value used (set after ``apply()``).

    Notes
    -----
    Port of ``ij/plugin/ContrastEnhancer.java`` from ImageJ 1.54j
    (public domain). Original by Wayne Rasband.

    The ``equalize=True`` mode performs global histogram equalization,
    not adaptive. For adaptive (local) enhancement, use ``CLAHE``
    instead.

    Examples
    --------
    Default contrast stretching:

    >>> from grdl.imagej import ContrastEnhancer
    >>> ce = ContrastEnhancer(saturated=0.35)
    >>> enhanced = ce.apply(sar_amplitude)

    More aggressive stretching:

    >>> ce = ContrastEnhancer(saturated=2.0)
    >>> enhanced = ce.apply(thermal_image)

    Normalize to [0, 1]:

    >>> ce = ContrastEnhancer(saturated=0.5, normalize=True)
    >>> normalized = ce.apply(pan_image)
    """

    __imagej_source__ = 'ij/plugin/ContrastEnhancer.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = True

    saturated: Annotated[float, Range(min=0.0, max=99.99),
                          Desc('Saturation percentage')] = 0.35
    equalize: Annotated[bool, Desc('Histogram equalization instead of stretch')] = False
    normalize: Annotated[bool, Desc('Normalize output to [0, 1]')] = False

    def __post_init__(self):
        self.min_val_: float = 0.0
        self.max_val_: float = 0.0

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply contrast enhancement to a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Enhanced image, dtype float64, same shape as input.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(
                f"Expected 2D image, got shape {source.shape}"
            )

        p = self._resolve_params(kwargs)

        image = source.astype(np.float64)

        if p['equalize']:
            return self._equalize(image, p)

        return self._linear_stretch(image, p)

    def _linear_stretch(self, image: np.ndarray, p: dict) -> np.ndarray:
        """Apply linear histogram stretching with saturation."""
        vmin = image.min()
        vmax = image.max()

        if vmax - vmin < 1e-15:
            self.min_val_ = vmin
            self.max_val_ = vmax
            return image.copy()

        # Compute percentile clip values
        half_sat = p['saturated'] / 2.0
        low_clip = np.percentile(image, half_sat)
        high_clip = np.percentile(image, 100.0 - half_sat)

        self.min_val_ = low_clip
        self.max_val_ = high_clip

        if high_clip - low_clip < 1e-15:
            return image.copy()

        # Clip and stretch
        result = np.clip(image, low_clip, high_clip)

        if p['normalize']:
            result = (result - low_clip) / (high_clip - low_clip)
        else:
            result = (result - low_clip) / (high_clip - low_clip) * \
                     (vmax - vmin) + vmin

        return result

    def _equalize(self, image: np.ndarray, p: dict) -> np.ndarray:
        """Apply global histogram equalization."""
        vmin = image.min()
        vmax = image.max()

        if vmax - vmin < 1e-15:
            self.min_val_ = vmin
            self.max_val_ = vmax
            return image.copy()

        n_bins = 256
        hist, bin_edges = np.histogram(image, bins=n_bins,
                                       range=(vmin, vmax))
        cdf = np.cumsum(hist).astype(np.float64)
        cdf_min = cdf[cdf > 0].min()
        total = cdf[-1]

        if total - cdf_min == 0:
            return image.copy()

        # Normalize CDF to [0, 1]
        cdf_norm = (cdf - cdf_min) / (total - cdf_min)

        # Map pixel values through CDF
        bin_idx = np.clip(
            ((image - vmin) / (vmax - vmin) * (n_bins - 1)).astype(int),
            0, n_bins - 1
        )
        result = cdf_norm[bin_idx]

        self.min_val_ = vmin
        self.max_val_ = vmax

        if not p['normalize']:
            result = result * (vmax - vmin) + vmin

        return result
