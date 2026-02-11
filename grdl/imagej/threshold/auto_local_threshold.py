# -*- coding: utf-8 -*-
"""
Auto Local Threshold - Port of Fiji's Auto_Local_Threshold plugin.

Implements eight local thresholding methods that compute a threshold
for each pixel based on statistics of its local neighborhood. These
methods adapt to spatially varying illumination and contrast, making
them essential for segmenting remotely sensed imagery with non-uniform
backgrounds.

Particularly useful for:
- Segmenting targets in SAR imagery with speckle and varying backscatter
- Extracting features from PAN imagery with shadow/illumination gradients
- Cloud/land/water masking in MSI data
- Detecting thermal anomalies against variable background temperatures
- Edge and feature extraction in HSI band ratios

Methods
-------
- Bernsen: Local contrast range decision
- Mean: Local mean offset
- Median: Local median offset
- MidGrey: Local midrange offset
- Niblack: Mean + k * standard deviation
- Sauvola: Mean * (1 + k * (stddev/r - 1))
- Phansalkar: Sauvola modification for low-contrast imagery
- Contrast: Contrast-based binary decision

Attribution
-----------
Fiji implementation: Gabriel Landini (University of Birmingham, UK).
Source: ``Auto_Local_Threshold.java`` (Fiji, GPL-2).

This is an independent NumPy reimplementation following the published
thresholding algorithms. References for each method are provided in
the class docstring.

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
from typing import Annotated, Any, Optional

# Third-party
import numpy as np
from scipy.ndimage import uniform_filter, median_filter

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


def _local_stats(image: np.ndarray, radius: int):
    """Compute local mean and standard deviation using uniform filter.

    Parameters
    ----------
    image : np.ndarray
        2D float64 image.
    radius : int
        Window half-size. Window is ``(2*radius+1) x (2*radius+1)``.

    Returns
    -------
    mean : np.ndarray
        Local mean.
    std : np.ndarray
        Local standard deviation.
    """
    size = 2 * radius + 1
    mean = uniform_filter(image, size=size, mode='nearest')
    mean_sq = uniform_filter(image * image, size=size, mode='nearest')
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(variance)
    return mean, std


def _local_min_max(image: np.ndarray, radius: int):
    """Compute local min and max using a rolling window.

    Uses the morphological approach: min via erosion conceptually.
    For efficiency, uses a padded sliding-window approach.

    Parameters
    ----------
    image : np.ndarray
        2D float64 image.
    radius : int
        Window half-size.

    Returns
    -------
    local_min : np.ndarray
        Local minimum values.
    local_max : np.ndarray
        Local maximum values.
    """
    from scipy.ndimage import minimum_filter, maximum_filter
    size = 2 * radius + 1
    local_min = minimum_filter(image, size=size, mode='nearest')
    local_max = maximum_filter(image, size=size, mode='nearest')
    return local_min, local_max


METHODS = (
    'bernsen', 'mean', 'median', 'midgrey',
    'niblack', 'sauvola', 'phansalkar', 'contrast',
)


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR], category=PC.THRESHOLD)
@processor_version('1.10.1')
class AutoLocalThreshold(ImageTransform):
    """Auto Local Threshold, ported from Fiji's Auto_Local_Threshold v1.10.1.

    Computes a per-pixel threshold based on local neighborhood statistics.
    Pixels above the local threshold are set to 1.0 (foreground), below
    to 0.0 (background).

    Parameters
    ----------
    method : str
        Thresholding method. One of:

        - ``'bernsen'``: ``T = (max + min) / 2`` if local contrast
          ``(max - min) >= contrast_threshold``, else 128 (background).
          Reference: J. Bernsen, "Dynamic thresholding of grey-level
          images", Proc. ICPR 1986.

        - ``'mean'``: ``T = mean - c``.

        - ``'median'``: ``T = median - c``.

        - ``'midgrey'``: ``T = (max + min) / 2 - c``.

        - ``'niblack'``: ``T = mean + k * stddev``.
          Reference: W. Niblack, "An Introduction to Digital Image
          Processing", Prentice Hall, 1986.

        - ``'sauvola'``: ``T = mean * (1 + k * (stddev / r - 1))``.
          Reference: J. Sauvola and M. Pietikainen, "Adaptive document
          image binarization", Pattern Recognition 33(2), 2000.

        - ``'phansalkar'``: ``T = mean * (1 + p * exp(-q * mean) +
          k * (stddev / r - 1))``. Reference: N. Phansalkar et al.,
          "Adaptive local thresholding for detection of nuclei in
          diversity stained cytology images", ICCSP 2011.

        - ``'contrast'``: If ``(max - min) >= contrast_threshold``:
          ``T = (max + min) / 2``, else pixel is background.

    radius : int
        Half-size of the local window. Window is ``(2*radius+1)^2``.
        Fiji default is 15.
    k : float
        Parameter for Niblack, Sauvola, Phansalkar. Controls sensitivity
        to local standard deviation. Niblack default: -0.2, Sauvola
        default: 0.5, Phansalkar default: 0.25.
    r : float
        Dynamic range of standard deviation, used by Sauvola and
        Phansalkar. Default 128 (for 8-bit imagery; scale proportionally
        for other ranges).
    c : float
        Offset constant for Mean, Median, MidGrey methods. Default 0.
    contrast_threshold : float
        Minimum local contrast for Bernsen and Contrast methods.
        Default 15 (8-bit scale).
    p : float
        Phansalkar ``p`` parameter. Default 2.0.
    q : float
        Phansalkar ``q`` parameter. Default 10.0.

    Notes
    -----
    Independent reimplementation of the algorithms as described in the
    referenced papers and in Fiji's ``Auto_Local_Threshold.java`` v1.10.1
    by Gabriel Landini (GPL-2).

    Input images are processed as float64 internally. For best results
    with methods that use absolute thresholds (Bernsen, Contrast), scale
    the image to [0, 255] range or adjust ``contrast_threshold`` and
    ``r`` accordingly.

    Examples
    --------
    >>> from grdl.imagej import AutoLocalThreshold
    >>> alt = AutoLocalThreshold(method='sauvola', radius=15, k=0.5)
    >>> binary = alt.apply(pan_image)
    """

    __imagej_source__ = 'fiji/Auto_Local_Threshold.java'
    __imagej_version__ = '1.10.1'
    __gpu_compatible__ = False

    method: Annotated[str, Options(*METHODS), Desc('Thresholding method')] = 'sauvola'
    radius: Annotated[int, Range(min=1), Desc('Local window half-size')] = 15
    k: Annotated[float, Desc('Sensitivity to local std deviation')] = 0.5
    r: Annotated[float, Desc('Dynamic range of standard deviation')] = 128.0
    c: Annotated[float, Desc('Offset constant for Mean/Median/MidGrey')] = 0.0
    contrast_threshold: Annotated[float, Desc('Minimum local contrast for Bernsen/Contrast')] = 15.0
    p: Annotated[float, Desc('Phansalkar p parameter')] = 2.0
    q: Annotated[float, Desc('Phansalkar q parameter')] = 10.0

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply local thresholding to a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Binary image, dtype float64: 1.0 for foreground (pixel
            above local threshold), 0.0 for background.

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

        if p['method'] == 'bernsen':
            return self._bernsen(image)
        elif p['method'] == 'mean':
            return self._mean(image)
        elif p['method'] == 'median':
            return self._median(image)
        elif p['method'] == 'midgrey':
            return self._midgrey(image)
        elif p['method'] == 'niblack':
            return self._niblack(image)
        elif p['method'] == 'sauvola':
            return self._sauvola(image)
        elif p['method'] == 'phansalkar':
            return self._phansalkar(image)
        elif p['method'] == 'contrast':
            return self._contrast(image)
        else:
            raise ValueError(f"Unknown method: {p['method']}")

    def _bernsen(self, image: np.ndarray) -> np.ndarray:
        local_min, local_max = _local_min_max(image, self.radius)
        contrast = local_max - local_min
        midrange = (local_max + local_min) / 2.0

        result = np.where(
            contrast >= self.contrast_threshold,
            np.where(image > midrange, 1.0, 0.0),
            0.0,
        )
        return result

    def _mean(self, image: np.ndarray) -> np.ndarray:
        mean, _ = _local_stats(image, self.radius)
        threshold = mean - self.c
        return np.where(image > threshold, 1.0, 0.0)

    def _median(self, image: np.ndarray) -> np.ndarray:
        size = 2 * self.radius + 1
        med = median_filter(image, size=size, mode='nearest')
        threshold = med - self.c
        return np.where(image > threshold, 1.0, 0.0)

    def _midgrey(self, image: np.ndarray) -> np.ndarray:
        local_min, local_max = _local_min_max(image, self.radius)
        threshold = (local_max + local_min) / 2.0 - self.c
        return np.where(image > threshold, 1.0, 0.0)

    def _niblack(self, image: np.ndarray) -> np.ndarray:
        mean, std = _local_stats(image, self.radius)
        threshold = mean + self.k * std
        return np.where(image > threshold, 1.0, 0.0)

    def _sauvola(self, image: np.ndarray) -> np.ndarray:
        mean, std = _local_stats(image, self.radius)
        threshold = mean * (1.0 + self.k * (std / self.r - 1.0))
        return np.where(image > threshold, 1.0, 0.0)

    def _phansalkar(self, image: np.ndarray) -> np.ndarray:
        mean, std = _local_stats(image, self.radius)
        threshold = mean * (
            1.0 + self.p * np.exp(-self.q * mean)
            + self.k * (std / self.r - 1.0)
        )
        return np.where(image > threshold, 1.0, 0.0)

    def _contrast(self, image: np.ndarray) -> np.ndarray:
        local_min, local_max = _local_min_max(image, self.radius)
        contrast = local_max - local_min
        midrange = (local_max + local_min) / 2.0

        result = np.where(
            contrast >= self.contrast_threshold,
            np.where(image > midrange, 1.0, 0.0),
            0.0,
        )
        return result
