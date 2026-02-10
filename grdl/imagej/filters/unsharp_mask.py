# -*- coding: utf-8 -*-
"""
Unsharp Mask - Port of ImageJ's UnsharpMask plugin.

Enhances image sharpness by subtracting a Gaussian-blurred version of
the image from the original, then adding a weighted portion back. This
amplifies high-frequency detail (edges, textures) while preserving the
overall intensity profile.

Particularly useful for:
- Sharpening PAN imagery before pan-sharpening fusion
- Enhancing edge detail in EO satellite imagery
- Improving texture visibility in SAR amplitude images
- Sharpening thermal imagery for feature extraction
- Pre-processing MSI bands before visual interpretation

Attribution
-----------
ImageJ implementation: Based on the standard unsharp mask algorithm.
Source: ``ij/plugin/filter/UnsharpMask.java`` in ImageJ 1.54j.
ImageJ 1.x source is in the public domain.

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
from typing import Any

# Third-party
import numpy as np
from scipy.ndimage import gaussian_filter

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version, processor_tags, TunableParameterSpec
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.SWIR, IM.MWIR, IM.LWIR], category=PC.FILTERS)
@processor_version('1.54j')
class UnsharpMask(ImageTransform):
    """Unsharp mask sharpening filter, ported from ImageJ 1.54j.

    Computes: ``output = image + weight * (image - gaussian_blur(image, sigma))``

    Equivalently: ``output = (1 + weight) * image - weight * blurred``

    Parameters
    ----------
    sigma : float
        Gaussian blur sigma (standard deviation) in pixels. Controls the
        scale of detail that is sharpened. Larger sigma sharpens coarser
        features. ImageJ default is 1.0.
    weight : float
        Sharpening strength (mask weight). Higher values produce stronger
        sharpening. ImageJ default is 0.6. A value of 0.0 returns the
        original image unchanged.

    Notes
    -----
    Port of ``ij/plugin/filter/UnsharpMask.java`` from ImageJ 1.54j
    (public domain). The algorithm is the standard photographic unsharp
    mask technique applied digitally.

    The Gaussian blur uses scipy.ndimage.gaussian_filter with 'nearest'
    boundary mode, matching ImageJ's boundary handling.

    Examples
    --------
    >>> from grdl.imagej import UnsharpMask
    >>> usm = UnsharpMask(sigma=2.0, weight=0.6)
    >>> sharpened = usm.apply(pan_image)
    """

    __imagej_source__ = 'ij/plugin/filter/UnsharpMask.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = False

    def __init__(
        self,
        sigma: float = 1.0,
        weight: float = 0.6,
    ) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if weight < 0:
            raise ValueError(f"weight must be >= 0, got {weight}")

        self.sigma = sigma
        self.weight = weight

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply unsharp mask sharpening.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Sharpened image, dtype float64, same shape as input.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(
                f"Expected 2D image, got shape {source.shape}"
            )

        image = source.astype(np.float64)
        blurred = gaussian_filter(image, sigma=self.sigma, mode='nearest')

        # output = image + weight * (image - blurred)
        sharpened = image + self.weight * (image - blurred)

        return sharpened
