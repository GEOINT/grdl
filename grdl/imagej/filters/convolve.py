# -*- coding: utf-8 -*-
"""
Convolve - Port of ImageJ's Process > Filters > Convolve.

Applies a user-defined 2D convolution kernel to an image. This is the
general-purpose spatial convolution in ImageJ, allowing arbitrary
filter design through custom kernel matrices.

Particularly useful for:
- Custom directional filters for SAR imagery (e.g. azimuth streaks)
- Application of published convolution kernels to PAN/EO data
- Implementing non-standard edge detectors for specific targets
- Matched filtering with known point-spread functions
- Custom texture kernels for MSI/HSI feature extraction
- Deconvolution preprocessing with known PSF approximations

Attribution
-----------
ImageJ implementation: Wayne Rasband (NIH).
Source: ``ij/plugin/filter/Convolver.java`` in ImageJ 1.54j.
ImageJ 1.x source is in the public domain.

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
from typing import Any

# Third-party
import numpy as np
from scipy.ndimage import convolve

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.FILTERS)
@processor_version('1.54j')
class Convolver(ImageTransform):
    """General-purpose 2D convolution, ported from ImageJ 1.54j.

    Convolves a 2D image with an arbitrary kernel matrix. The kernel
    is optionally normalized so that its elements sum to 1 (preserving
    image brightness), matching ImageJ's default "Normalize Kernel"
    behavior.

    Supports complex-valued input (SLC/SICD SAR data). The real-valued
    kernel is convolved with the complex image, filtering real and
    imaginary components independently.

    Parameters
    ----------
    kernel : np.ndarray
        2D convolution kernel. Must have odd dimensions. Common
        examples: 3x3 sharpening, 5x5 Gaussian, directional filters.
    normalize : bool
        If True (default), the kernel is normalized so its elements
        sum to 1.0 before convolution. Set to False for derivative
        kernels (whose elements sum to 0) or other kernels where
        normalization is inappropriate. Matches ImageJ's
        "Normalize Kernel" checkbox.

    Notes
    -----
    Port of ``ij/plugin/filter/Convolver.java`` from ImageJ 1.54j
    (public domain). Original implementation by Wayne Rasband.

    ImageJ uses zero-padded boundaries by default. This port uses
    ``mode='nearest'`` (replicate border pixels), which generally
    produces fewer artifacts in remote sensing imagery.

    For very large kernels, consider using ``FFTBandpassFilter``
    instead, which operates in the frequency domain.

    For complex-valued input (e.g. SAR SLC or NGA SICD), the
    convolution is applied to the complex signal directly. With a
    real-valued kernel this is equivalent to independently filtering
    the I and Q channels, preserving interferometric phase.

    Examples
    --------
    Apply a 3x3 Laplacian for edge enhancement:

    >>> from grdl.imagej import Convolver
    >>> laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    >>> conv = Convolver(kernel=laplacian, normalize=False)
    >>> edges = conv.apply(pan_image)

    Apply a custom directional filter:

    >>> kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    >>> conv = Convolver(kernel=kernel, normalize=False)
    >>> horizontal_edges = conv.apply(sar_image)

    Smooth complex SLC data with a weighted kernel:

    >>> kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    >>> conv = Convolver(kernel=kernel, normalize=True)
    >>> smoothed_slc = conv.apply(sicd_complex)
    """

    __imagej_source__ = 'ij/plugin/filter/Convolver.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = True

    def __init__(
        self,
        kernel: np.ndarray,
        normalize: bool = True,
    ) -> None:
        kernel = np.asarray(kernel, dtype=np.float64)
        if kernel.ndim != 2:
            raise ValueError(
                f"kernel must be 2D, got {kernel.ndim}D"
            )
        if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            raise ValueError(
                f"kernel dimensions must be odd, got {kernel.shape}"
            )
        if kernel.size == 0:
            raise ValueError("kernel must not be empty")

        if normalize:
            ksum = kernel.sum()
            if abs(ksum) > 1e-15:
                kernel = kernel / ksum

        self.kernel = kernel

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply 2D convolution to an image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``. Supports both
            real-valued and complex-valued (SLC/SICD) input.

        Returns
        -------
        np.ndarray
            Convolved image, same shape as input. dtype is complex128
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
        return convolve(image, self.kernel, mode='nearest')
