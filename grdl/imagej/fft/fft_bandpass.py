# -*- coding: utf-8 -*-
"""
FFT Bandpass Filter - Port of ImageJ's FFT Filter plugin.

Applies frequency-domain bandpass filtering to suppress structures
outside a specified spatial scale range. The filter removes both
large-scale illumination gradients (low frequencies) and small-scale
noise (high frequencies), leaving features of interest in the
intermediate spatial frequency band.

Optionally suppresses horizontal or vertical stripe artifacts via
directional filtering in the frequency domain.

Particularly useful for:
- Removing scan-line artifacts (stripes) in satellite push-broom sensors
- Suppressing speckle noise in SAR imagery (high-frequency cutoff)
- Removing illumination gradients in PAN/EO imagery (low-frequency cutoff)
- Isolating features at specific spatial scales in MSI/HSI data
- De-striping thermal imagery from whisk-broom scanners

Attribution
-----------
ImageJ implementation: Joachim Walter (University of Heidelberg).
Source: ``ij/plugin/filter/FFTFilter.java`` in ImageJ 1.54j.
ImageJ 1.x source is in the public domain.

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
from typing import Any, Optional

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version, processor_tags, TunableParameterSpec


def _make_bandpass_mask(
    shape: tuple,
    filter_large: float,
    filter_small: float,
) -> np.ndarray:
    """Build a 2D bandpass mask in frequency domain.

    Creates a mask that passes spatial frequencies between
    ``1/filter_large`` and ``1/filter_small``. Uses smooth Gaussian
    roll-off (not hard cutoffs) matching ImageJ's approach.

    Parameters
    ----------
    shape : tuple
        (rows, cols) of the FFT array.
    filter_large : float
        Cutoff size for large structures (pixels). Structures larger
        than this are suppressed (high-pass component).
    filter_small : float
        Cutoff size for small structures (pixels). Structures smaller
        than this are suppressed (low-pass component).

    Returns
    -------
    np.ndarray
        2D float64 mask, shape ``shape``, values in [0, 1].
    """
    rows, cols = shape
    cy, cx = rows // 2, cols // 2

    # Frequency coordinates (centered)
    y = np.arange(rows, dtype=np.float64) - cy
    x = np.arange(cols, dtype=np.float64) - cx
    Y, X = np.meshgrid(y, x, indexing='ij')
    dist = np.sqrt(Y * Y + X * X)

    # Avoid division by zero at DC
    dist_safe = np.where(dist == 0, 1e-10, dist)

    mask = np.ones(shape, dtype=np.float64)

    # High-pass: suppress frequencies below 1/filter_large
    if filter_large > 0:
        cutoff_large = rows / filter_large
        # Gaussian roll-off
        mask *= 1.0 - np.exp(-(dist_safe * dist_safe) / (2.0 * cutoff_large * cutoff_large))

    # Low-pass: suppress frequencies above 1/filter_small
    if filter_small > 0:
        cutoff_small = rows / filter_small
        mask *= np.exp(-(dist_safe * dist_safe) / (2.0 * cutoff_small * cutoff_small))

    return mask


def _make_stripe_mask(
    shape: tuple,
    direction: str,
    tolerance: float,
) -> np.ndarray:
    """Build a directional stripe suppression mask.

    Suppresses frequency components along a specific direction to
    remove stripe artifacts (e.g., satellite scan lines).

    Parameters
    ----------
    shape : tuple
        (rows, cols) of the FFT array.
    direction : str
        ``'horizontal'`` suppresses horizontal stripes (vertical
        frequency axis), ``'vertical'`` suppresses vertical stripes
        (horizontal frequency axis).
    tolerance : float
        Angular tolerance in degrees. Wider tolerance suppresses
        more off-axis components. Default 5.0.

    Returns
    -------
    np.ndarray
        2D float64 mask, values in [0, 1].
    """
    rows, cols = shape
    cy, cx = rows // 2, cols // 2

    y = np.arange(rows, dtype=np.float64) - cy
    x = np.arange(cols, dtype=np.float64) - cx
    Y, X = np.meshgrid(y, x, indexing='ij')

    # Angle from center
    angle = np.degrees(np.arctan2(Y, X))

    tol = tolerance

    if direction == 'horizontal':
        # Horizontal stripes → suppress along vertical axis (90°/270°)
        mask = np.where(
            (np.abs(np.abs(angle) - 90) < tol) & (np.abs(Y) > 1),
            0.0, 1.0
        )
    elif direction == 'vertical':
        # Vertical stripes → suppress along horizontal axis (0°/180°)
        vert_mask = (
            ((np.abs(angle) < tol) | (np.abs(angle) > 180 - tol))
            & (np.abs(X) > 1)
        )
        mask = np.where(vert_mask, 0.0, 1.0)
    else:
        mask = np.ones(shape, dtype=np.float64)

    return mask


@processor_tags(modalities=['SAR', 'PAN', 'EO', 'MSI', 'thermal'], category='fft')
@processor_version('1.54j')
class FFTBandpassFilter(ImageTransform):
    """FFT bandpass filter, ported from ImageJ 1.54j.

    Applies a frequency-domain bandpass filter that preserves structures
    between ``filter_small`` and ``filter_large`` pixels in size.
    Optionally removes directional stripe artifacts.

    Parameters
    ----------
    filter_large : float
        Size of largest structures to keep (pixels). Structures larger
        than this are filtered out (removes background gradients).
        ImageJ default is 40. Set to 0 to disable high-pass component.
    filter_small : float
        Size of smallest structures to keep (pixels). Structures smaller
        than this are filtered out (removes noise). ImageJ default is 3.
        Set to 0 to disable low-pass component.
    suppress_stripes : str or None
        Direction of stripes to suppress: ``'horizontal'``,
        ``'vertical'``, or ``None`` (no suppression). Default None.
    stripe_tolerance : float
        Angular tolerance for stripe suppression in degrees. ImageJ
        default is 5.0.
    autoscale : bool
        If True, normalize the result to the same mean and standard
        deviation as the input. Matches ImageJ's "Autoscale after
        filtering" option. Default True.

    Notes
    -----
    Port of ``ij/plugin/filter/FFTFilter.java`` from ImageJ 1.54j
    (public domain). Original implementation by Joachim Walter.

    The input image is zero-padded to the next power of two for
    efficient FFT computation, then cropped back to original size.

    Examples
    --------
    Remove background gradients and noise from SAR amplitude:

    >>> from grdl.imagej import FFTBandpassFilter
    >>> bp = FFTBandpassFilter(filter_large=40, filter_small=3)
    >>> filtered = bp.apply(sar_image)

    Remove horizontal scan-line artifacts:

    >>> bp = FFTBandpassFilter(filter_large=0, filter_small=0,
    ...                        suppress_stripes='horizontal')
    >>> destriped = bp.apply(thermal_image)
    """

    __imagej_source__ = 'ij/plugin/filter/FFTFilter.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = True

    def __init__(
        self,
        filter_large: float = 40.0,
        filter_small: float = 3.0,
        suppress_stripes: Optional[str] = None,
        stripe_tolerance: float = 5.0,
        autoscale: bool = True,
    ) -> None:
        if filter_large < 0:
            raise ValueError(f"filter_large must be >= 0, got {filter_large}")
        if filter_small < 0:
            raise ValueError(f"filter_small must be >= 0, got {filter_small}")
        if suppress_stripes is not None:
            if suppress_stripes not in ('horizontal', 'vertical'):
                raise ValueError(
                    f"suppress_stripes must be 'horizontal', 'vertical', "
                    f"or None, got {suppress_stripes!r}"
                )

        self.filter_large = filter_large
        self.filter_small = filter_small
        self.suppress_stripes = suppress_stripes
        self.stripe_tolerance = stripe_tolerance
        self.autoscale = autoscale

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply FFT bandpass filter.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Filtered image, dtype float64, same shape as input.

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
        orig_mean = image.mean()
        orig_std = image.std()
        orig_shape = image.shape

        # Pad to next power of 2 for efficient FFT
        rows, cols = image.shape
        pad_rows = int(2 ** np.ceil(np.log2(rows)))
        pad_cols = int(2 ** np.ceil(np.log2(cols)))

        padded = np.zeros((pad_rows, pad_cols), dtype=np.float64)
        padded[:rows, :cols] = image

        # Mirror-pad edges to reduce ringing
        if rows < pad_rows:
            padded[rows:, :cols] = image[rows - 1::-1, :][:pad_rows - rows, :]
        if cols < pad_cols:
            padded[:rows, cols:] = image[:, cols - 1::-1][:, :pad_cols - cols]

        # Forward FFT
        fft_data = np.fft.fftshift(np.fft.fft2(padded))

        # Build combined mask
        mask = np.ones((pad_rows, pad_cols), dtype=np.float64)

        if self.filter_large > 0 or self.filter_small > 0:
            bp_mask = _make_bandpass_mask(
                (pad_rows, pad_cols), self.filter_large, self.filter_small
            )
            mask *= bp_mask

        if self.suppress_stripes is not None:
            stripe_mask = _make_stripe_mask(
                (pad_rows, pad_cols),
                self.suppress_stripes,
                self.stripe_tolerance,
            )
            mask *= stripe_mask

        # Apply mask
        fft_data *= mask

        # Inverse FFT
        result_padded = np.real(np.fft.ifft2(np.fft.ifftshift(fft_data)))

        # Crop to original size
        result = result_padded[:rows, :cols]

        # Autoscale to match input statistics
        if self.autoscale:
            result_std = result.std()
            result_mean = result.mean()
            if result_std > 1e-15:
                result = (result - result_mean) / result_std * orig_std + orig_mean

        return result
