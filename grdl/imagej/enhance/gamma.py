# -*- coding: utf-8 -*-
"""
Gamma Correction - Port of ImageJ's Process > Math > Gamma function.

Applies a power-law (gamma) transform to pixel intensities:
``output = input^gamma`` (after normalization to [0, 1]).

Gamma < 1.0 brightens dark regions (compresses highlights, expands
shadows). Gamma > 1.0 darkens bright regions (expands highlights,
compresses shadows). Gamma = 1.0 is identity.

Particularly useful for:
- Adjusting dynamic range of SAR amplitude imagery for visualization
- Compensating non-linear sensor response in PAN/EO cameras
- Enhancing dark regions in nighttime PAN imagery (gamma < 1)
- Improving contrast in thermal imagery with narrow dynamic range
- Display normalization for HSI false-color composites
- Radiometric correction of multi-look SAR products

Attribution
-----------
ImageJ implementation: Wayne Rasband (NIH).
Source: ``ij/process/FloatProcessor.java`` (``gamma()`` method)
in ImageJ 1.54j. ImageJ 1.x source is in the public domain.

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
from typing import Annotated, Any

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_version('1.54j')
@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.ENHANCE)
class GammaCorrection(ImageTransform):
    """Power-law gamma correction, ported from ImageJ 1.54j.

    Normalizes pixel values to [0, 1], applies the gamma power law,
    then scales back to the original range.

    ``output = ((input - min) / (max - min))^gamma * (max - min) + min``

    Parameters
    ----------
    gamma : float
        Gamma exponent. Must be > 0.

        - ``gamma < 1.0``: Brightens (expand shadows, compress highlights).
        - ``gamma = 1.0``: Identity (no change).
        - ``gamma > 1.0``: Darkens (compress shadows, expand highlights).

        ImageJ default is 0.5 (brighten). Common values: 0.4-0.6 for
        SAR amplitude display, 1.5-2.5 for suppressing bright features.

    Notes
    -----
    Port of the gamma function from ``ij/process/FloatProcessor.java``
    in ImageJ 1.54j (public domain). Original by Wayne Rasband.

    For constant-valued images (max == min), returns the input unchanged.

    Examples
    --------
    Brighten SAR amplitude for display:

    >>> from grdl.imagej import GammaCorrection
    >>> gc = GammaCorrection(gamma=0.4)
    >>> display_image = gc.apply(sar_db)

    Darken to suppress bright targets:

    >>> gc = GammaCorrection(gamma=2.0)
    >>> suppressed = gc.apply(pan_image)
    """

    __imagej_source__ = 'ij/process/FloatProcessor.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = True

    gamma: Annotated[float, Range(min=0.001), Desc('Gamma exponent')] = 0.5

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply gamma correction to a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Gamma-corrected image, dtype float64, same value range
            as input.

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
        vmin = image.min()
        vmax = image.max()

        if vmax - vmin < 1e-15:
            return image.copy()

        # Normalize to [0, 1], apply gamma, scale back
        normalized = (image - vmin) / (vmax - vmin)
        corrected = np.power(normalized, p['gamma'])
        return corrected * (vmax - vmin) + vmin
