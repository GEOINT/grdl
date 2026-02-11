# -*- coding: utf-8 -*-
"""
Phase Gradient Filter - Windowed phase gradient estimation for complex imagery.

Computes the local phase gradient of complex-valued data (e.g., SAR SLC)
using conjugate multiplication with shifted neighbors followed by windowed
averaging. This approach avoids explicit phase unwrapping: by averaging
complex conjugate products over a local window *before* extracting the
angle, phase wrapping artifacts are suppressed.

Algorithm
---------
For complex signal ``z``:

1. Row gradient products: ``conj(z[r, c]) * z[r+1, c]``
2. Col gradient products: ``conj(z[r, c]) * z[r, c+1]``
3. Smooth each product field over a ``(kernel_size x kernel_size)`` window
4. Extract phase: ``angle(smoothed_product)`` → radians/pixel
5. Return row gradient, col gradient, or gradient magnitude

Dependencies
------------
scipy

Author
------
Duane Smalley, PhD
duane.d.smalley@gmail.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-11

Modified
--------
2026-02-11
"""

# Standard library
from typing import Annotated, Any

# Third-party
import numpy as np
from scipy.ndimage import uniform_filter

# GRDL internal
from grdl.exceptions import ValidationError
from grdl.image_processing.base import BandwiseTransformMixin, ImageTransform
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.image_processing.filters._validation import (
    validate_kernel_size,
    validate_mode,
)
from grdl.vocabulary import ImageModality, ProcessorCategory


@processor_version('1.0.0')
@processor_tags(category=ProcessorCategory.FILTERS,
                modalities=[ImageModality.SAR])
class PhaseGradientFilter(BandwiseTransformMixin, ImageTransform):
    """Windowed phase gradient filter for complex-valued imagery.

    Estimates the local spatial phase gradient of complex data using
    conjugate multiplication with shifted neighbors, followed by
    windowed averaging to suppress noise and phase wrapping artifacts.

    The conjugate product approach computes the phase difference between
    adjacent pixels without explicit unwrapping::

        Δφ_row = angle( mean_window( conj(z[r,c]) * z[r+1,c] ) )
        Δφ_col = angle( mean_window( conj(z[r,c]) * z[r,c+1] ) )

    Averaging the complex products over a window before extracting
    the angle provides robust gradient estimates even in noisy SAR data.

    Parameters
    ----------
    kernel_size : int
        Square smoothing window side length in pixels. Must be odd
        and >= 3. Larger windows produce smoother gradient estimates.
        Default is 3.
    direction : str
        Which gradient component to return:

        - ``'magnitude'``: gradient magnitude ``sqrt(Δφ_row² + Δφ_col²)``
        - ``'row'``: phase gradient in the row (azimuth) direction
        - ``'col'``: phase gradient in the column (range) direction

        Default is ``'magnitude'``.
    mode : str
        Boundary handling mode for the smoothing window. One of
        ``'reflect'``, ``'constant'``, ``'nearest'``, ``'wrap'``.
        Default is ``'reflect'``.

    Raises
    ------
    ValidationError
        If input is not complex-valued.

    Examples
    --------
    >>> from grdl.image_processing.filters import PhaseGradientFilter
    >>> pgf = PhaseGradientFilter(kernel_size=5, direction='magnitude')
    >>> grad_mag = pgf.apply(complex_sar_image)

    Row and column gradients separately:

    >>> row_grad = PhaseGradientFilter(direction='row').apply(slc)
    >>> col_grad = PhaseGradientFilter(direction='col').apply(slc)
    """

    __gpu_compatible__ = False

    kernel_size: Annotated[int, Range(min=3, max=101),
                           Desc('Smoothing window side length (odd)')] = 3
    direction: Annotated[str, Options('magnitude', 'row', 'col'),
                         Desc('Gradient component to return')] = 'magnitude'
    mode: Annotated[str, Options('reflect', 'constant', 'nearest', 'wrap'),
                    Desc('Boundary handling mode')] = 'reflect'

    def __init__(
        self,
        kernel_size: int = 3,
        direction: str = 'magnitude',
        mode: str = 'reflect',
    ) -> None:
        validate_kernel_size(kernel_size)
        validate_mode(mode)
        if direction not in ('magnitude', 'row', 'col'):
            raise ValidationError(
                f"direction must be 'magnitude', 'row', or 'col', "
                f"got {direction!r}"
            )
        self.kernel_size = kernel_size
        self.direction = direction
        self.mode = mode

    def _apply_2d(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute windowed phase gradient of a single 2D complex band.

        Parameters
        ----------
        source : np.ndarray
            2D complex-valued array, shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Phase gradient in radians/pixel, same shape, float64.
            Values in ``[-π, π]`` for row/col direction, or
            ``[0, π√2]`` for magnitude.

        Raises
        ------
        ValidationError
            If ``source`` is not complex-valued.
        """
        if not np.iscomplexobj(source):
            raise ValidationError(
                "PhaseGradientFilter requires complex-valued input. "
                f"Got dtype {source.dtype}"
            )

        params = self._resolve_params(kwargs)
        ks = params['kernel_size']
        direction = params['direction']
        mode = params['mode']
        validate_kernel_size(ks)
        validate_mode(mode)

        z = source.astype(np.complex128)

        # Conjugate products with shifted neighbors (forward phase difference)
        # Row gradient: conj(z[r,c]) * z[r+1,c] → φ[r+1] - φ[r]
        conj_prod_row = np.conj(z[:-1, :]) * z[1:, :]
        # Pad last row to preserve shape
        conj_prod_row = np.vstack([
            conj_prod_row,
            conj_prod_row[-1:, :],
        ])

        # Col gradient: conj(z[r,c]) * z[r,c+1] → φ[c+1] - φ[c]
        conj_prod_col = np.conj(z[:, :-1]) * z[:, 1:]
        # Pad last column to preserve shape
        conj_prod_col = np.hstack([
            conj_prod_col,
            conj_prod_col[:, -1:],
        ])

        # Smooth complex products over window (real and imag separately)
        if direction in ('magnitude', 'row'):
            smooth_row_re = uniform_filter(conj_prod_row.real, size=ks, mode=mode)
            smooth_row_im = uniform_filter(conj_prod_row.imag, size=ks, mode=mode)
            grad_row = np.arctan2(smooth_row_im, smooth_row_re)

        if direction in ('magnitude', 'col'):
            smooth_col_re = uniform_filter(conj_prod_col.real, size=ks, mode=mode)
            smooth_col_im = uniform_filter(conj_prod_col.imag, size=ks, mode=mode)
            grad_col = np.arctan2(smooth_col_im, smooth_col_re)

        if direction == 'row':
            return grad_row
        elif direction == 'col':
            return grad_col
        else:
            return np.sqrt(grad_row * grad_row + grad_col * grad_col)
