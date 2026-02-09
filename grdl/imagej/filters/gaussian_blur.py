# -*- coding: utf-8 -*-
"""
Gaussian Blur - Port of ImageJ's Process > Filters > Gaussian Blur.

Applies an isotropic or anisotropic Gaussian smoothing kernel to a 2D
image. This is the single most commonly used preprocessing filter in
ImageJ, employed for noise suppression, scale-space construction, and
anti-aliasing prior to downsampling.

Particularly useful for:
- SAR speckle pre-smoothing before edge detection or thresholding
- Noise reduction in PAN/EO imagery before feature extraction
- Scale-space pyramid construction for multi-scale analysis
- Anti-aliasing before subsampling MSI/HSI bands
- Thermal noise suppression while preserving large-scale gradients
- Pre-filtering before gradient computation (Canny-style workflows)

Attribution
-----------
ImageJ implementation: Michael Schmid (Vienna University of Technology).
Source: ``ij/plugin/filter/GaussianBlur.java`` in ImageJ 1.54j.
ImageJ 1.x source is in the public domain.

The ImageJ implementation uses separable 1D convolutions with a
kernel truncated at ``3 * sigma`` (accuracy parameter controls this).
This port delegates to ``scipy.ndimage.gaussian_filter`` which uses
the same separable approach with configurable truncation.

Dependencies
------------
scipy

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
from typing import Any, Optional, Tuple, Union

# Third-party
import numpy as np
from scipy.ndimage import gaussian_filter

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version, processor_tags


@processor_tags(modalities=['SAR', 'PAN', 'EO', 'MSI', 'HSI', 'thermal'],
                category='filters')
@processor_version('1.54j')
class GaussianBlur(ImageTransform):
    """Gaussian smoothing filter, ported from ImageJ 1.54j.

    Convolves the image with a 2D Gaussian kernel. The kernel is
    separable, so the 2D convolution is performed as two sequential
    1D convolutions (row-wise then column-wise) for efficiency.

    Supports complex-valued input (SLC/SICD SAR data). When the input
    is complex, the real and imaginary components are filtered
    independently, preserving phase information.

    Parameters
    ----------
    sigma : float or tuple of float
        Standard deviation of the Gaussian kernel in pixels.
        A single float applies the same sigma in both dimensions
        (isotropic). A 2-tuple ``(sigma_y, sigma_x)`` applies
        different sigmas along rows and columns (anisotropic).
        ImageJ's "Radius" field corresponds to ``sigma``.
        Must be > 0. Default is 2.0.
    accuracy : float
        Controls kernel truncation radius as a multiple of sigma.
        The kernel extends to ``accuracy * sigma`` pixels from center.
        ImageJ default is 0.01, which corresponds to approximately
        ``ceil(2.5 / accuracy)`` ≈ 3.0 sigma. This port uses the
        ``truncate`` parameter of ``scipy.ndimage.gaussian_filter``.
        Default is 4.0 (truncate at 4 sigma, slightly more conservative
        than ImageJ's default of ~3 sigma).

    Notes
    -----
    Port of ``ij/plugin/filter/GaussianBlur.java`` from ImageJ 1.54j
    (public domain). Original implementation by Michael Schmid.

    The ImageJ version uses ``float[]`` intermediate buffers and a
    custom downscale→blur→upscale strategy for very large sigma
    values. This port relies on scipy's optimized Gaussian filter
    which handles large sigma directly.

    For sigma = 0, the image is returned unchanged (identity).

    For complex-valued input (e.g. SAR SLC or NGA SICD), scipy's
    ``gaussian_filter`` filters real and imaginary parts independently.
    This is equivalent to convolving the complex signal with a real-
    valued Gaussian kernel, which is the standard approach for
    coherent speckle reduction in the complex domain.

    Examples
    --------
    Standard isotropic blur:

    >>> from grdl.imagej import GaussianBlur
    >>> blur = GaussianBlur(sigma=2.0)
    >>> smoothed = blur.apply(noisy_pan)

    Anisotropic blur (more smoothing in azimuth for SAR):

    >>> blur = GaussianBlur(sigma=(4.0, 1.5))
    >>> smoothed = blur.apply(sar_amplitude)

    Smooth complex SLC data (preserves phase):

    >>> blur = GaussianBlur(sigma=1.5)
    >>> smoothed_slc = blur.apply(sicd_complex)
    """

    __imagej_source__ = 'ij/plugin/filter/GaussianBlur.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = True

    def __init__(
        self,
        sigma: Union[float, Tuple[float, float]] = 2.0,
        accuracy: float = 4.0,
    ) -> None:
        if isinstance(sigma, (list, tuple)):
            if len(sigma) != 2:
                raise ValueError(
                    f"sigma tuple must have 2 elements, got {len(sigma)}"
                )
            if any(s < 0 for s in sigma):
                raise ValueError(
                    f"sigma values must be >= 0, got {sigma}"
                )
            self.sigma = tuple(float(s) for s in sigma)
        else:
            if sigma < 0:
                raise ValueError(f"sigma must be >= 0, got {sigma}")
            self.sigma = float(sigma)

        if accuracy <= 0:
            raise ValueError(f"accuracy must be > 0, got {accuracy}")
        self.accuracy = accuracy

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Gaussian blur to a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``. Supports both
            real-valued and complex-valued (SLC/SICD) input.

        Returns
        -------
        np.ndarray
            Smoothed image, same shape as input. dtype is complex128
            for complex input, float64 otherwise.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(
                f"Expected 2D image, got shape {source.shape}"
            )

        if np.iscomplexobj(source):
            image = source.astype(np.complex128)
        else:
            image = source.astype(np.float64)

        # Identity for zero sigma
        if isinstance(self.sigma, tuple):
            if all(s == 0 for s in self.sigma):
                return image.copy()
        elif self.sigma == 0:
            return image.copy()

        return gaussian_filter(
            image,
            sigma=self.sigma,
            truncate=self.accuracy,
            mode='nearest',
        )
