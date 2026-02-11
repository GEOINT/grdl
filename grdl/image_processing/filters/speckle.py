# -*- coding: utf-8 -*-
"""
Speckle Filters - Lee adaptive filters for SAR imagery.

Provides real-valued and complex-valued Lee speckle filters for SAR imagery.

``LeeFilter`` operates on real-valued intensity/amplitude data, weighting
between the local mean and the observed value based on variance ratio.

``ComplexLeeFilter`` operates on complex-valued SLC data, applying the same
adaptive weighting in the complex domain to suppress speckle while
preserving interferometric phase.

Uses ``scipy.ndimage.uniform_filter`` for efficient local statistics.

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
2026-02-12
"""

# Standard library
from typing import Annotated, Any

# Third-party
import numpy as np
from scipy.ndimage import uniform_filter

# GRDL internal
from grdl.image_processing.base import BandwiseTransformMixin, ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.exceptions import ValidationError
from grdl.image_processing.filters._validation import validate_kernel_size
from grdl.vocabulary import ImageModality, ProcessorCategory


@processor_version('1.0.0')
@processor_tags(category=ProcessorCategory.FILTERS,
                modalities=[ImageModality.SAR])
class LeeFilter(BandwiseTransformMixin, ImageTransform):
    """Lee adaptive speckle filter for SAR imagery.

    Preserves edges by weighting between the local mean and the observed
    pixel value based on the ratio of local variance to expected noise
    variance::

        output = mean + weight * (pixel - mean)
        weight = max(0, 1 - noise_var / local_var)

    When local variance is high (edge/target), ``weight -> 1`` (pass
    through original). When local variance is low (homogeneous region),
    ``weight -> 0`` (smooth to local mean).

    Parameters
    ----------
    kernel_size : int
        Square kernel side length in pixels. Must be odd and >= 3.
        Default is 7.
    noise_variance : float
        Expected noise variance. For single-look SAR intensity data,
        use 1.0 (multiplicative speckle model). If 0.0, the noise
        variance is estimated from the image as the mean of local
        variances. Default is 0.0 (auto-estimate).

    Examples
    --------
    >>> from grdl.image_processing.filters import LeeFilter
    >>> lee = LeeFilter(kernel_size=7)
    >>> despeckled = lee.apply(sar_intensity)

    With known noise variance:

    >>> lee = LeeFilter(kernel_size=7, noise_variance=1.0)
    >>> despeckled = lee.apply(sar_intensity)
    """

    __gpu_compatible__ = True

    kernel_size: Annotated[int, Range(min=3, max=31),
                           Desc('Square kernel side length (odd)')] = 7
    noise_variance: Annotated[float, Range(min=0.0),
                              Desc('Expected noise variance (0 = auto)')] = 0.0

    def __init__(
        self,
        kernel_size: int = 7,
        noise_variance: float = 0.0,
    ) -> None:
        validate_kernel_size(kernel_size)
        self.kernel_size = kernel_size
        self.noise_variance = noise_variance

    def _apply_2d(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Lee speckle filter to a single 2D band.

        Parameters
        ----------
        source : np.ndarray
            2D image array, shape ``(rows, cols)``. Typically SAR
            intensity or amplitude data.

        Returns
        -------
        np.ndarray
            Despeckled image, same shape, float64.
        """
        params = self._resolve_params(kwargs)
        ks = params['kernel_size']
        noise_var = params['noise_variance']
        validate_kernel_size(ks)

        # Float64 for numerical stability of variance computation
        x = source.astype(np.float64)
        local_mean = uniform_filter(x, size=ks, mode='reflect')
        local_mean_sq = uniform_filter(x * x, size=ks, mode='reflect')
        local_var = local_mean_sq - local_mean * local_mean
        np.maximum(local_var, 0.0, out=local_var)

        if noise_var <= 0.0:
            # Auto-estimate: mean of local variances
            noise_var = float(np.mean(local_var))

        # Adaptive weight: high local_var -> pass through; low -> smooth
        # Use safe division to avoid RuntimeWarning on zero-variance pixels
        safe_var = np.where(local_var > 0, local_var, 1.0)
        weight = np.where(
            local_var > 0,
            np.maximum(0.0, 1.0 - noise_var / safe_var),
            0.0,
        )
        return local_mean + weight * (x - local_mean)


@processor_version('1.0.0')
@processor_tags(category=ProcessorCategory.FILTERS,
                modalities=[ImageModality.SAR])
class ComplexLeeFilter(BandwiseTransformMixin, ImageTransform):
    """Lee adaptive speckle filter for complex-valued SAR SLC data.

    Applies the Lee adaptive weighting in the complex domain, suppressing
    speckle noise while preserving interferometric phase. The adaptive
    weight is derived from the intensity variance (same as ``LeeFilter``),
    but applied to the complex signal::

        z_out = E[z] + weight * (z - E[z])
        weight = max(0, 1 - noise_var / var(|z|²))

    where ``E[z]`` is the local complex mean (computed by smoothing real
    and imaginary parts independently).

    When local intensity variance is high (edge/target), ``weight -> 1``
    (pass through original complex value). When variance is low
    (homogeneous clutter), ``weight -> 0`` (smooth to local complex mean).

    Parameters
    ----------
    kernel_size : int
        Square kernel side length in pixels. Must be odd and >= 3.
        Default is 7.
    noise_variance : float
        Expected noise variance of the intensity ``|z|²``. If 0.0,
        the noise variance is estimated from the image as the mean of
        local intensity variances. Default is 0.0 (auto-estimate).

    Raises
    ------
    ValidationError
        If input is not complex-valued.

    Examples
    --------
    >>> from grdl.image_processing.filters import ComplexLeeFilter
    >>> clf = ComplexLeeFilter(kernel_size=7)
    >>> despeckled_slc = clf.apply(complex_sar_image)

    Phase is preserved — useful before interferometric operations:

    >>> import numpy as np
    >>> np.angle(despeckled_slc)  # phase field still meaningful
    """

    __gpu_compatible__ = False

    kernel_size: Annotated[int, Range(min=3, max=31),
                           Desc('Square kernel side length (odd)')] = 7
    noise_variance: Annotated[float, Range(min=0.0),
                              Desc('Expected intensity noise variance '
                                   '(0 = auto)')] = 0.0

    def __init__(
        self,
        kernel_size: int = 7,
        noise_variance: float = 0.0,
    ) -> None:
        validate_kernel_size(kernel_size)
        self.kernel_size = kernel_size
        self.noise_variance = noise_variance

    def _apply_2d(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply complex Lee speckle filter to a single 2D band.

        Parameters
        ----------
        source : np.ndarray
            2D complex-valued array, shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Despeckled complex image, same shape, complex128.

        Raises
        ------
        ValidationError
            If ``source`` is not complex-valued.
        """
        if not np.iscomplexobj(source):
            raise ValidationError(
                "ComplexLeeFilter requires complex-valued input. "
                f"Got dtype {source.dtype}"
            )

        params = self._resolve_params(kwargs)
        ks = params['kernel_size']
        noise_var = params['noise_variance']
        validate_kernel_size(ks)

        z = source.astype(np.complex128)

        # Local complex mean (smooth real and imag independently)
        local_mean_re = uniform_filter(z.real, size=ks, mode='reflect')
        local_mean_im = uniform_filter(z.imag, size=ks, mode='reflect')
        local_mean_z = local_mean_re + 1j * local_mean_im

        # Intensity-based adaptive weight
        intensity = z.real * z.real + z.imag * z.imag  # |z|²
        local_mean_I = uniform_filter(intensity, size=ks, mode='reflect')
        local_mean_I2 = uniform_filter(
            intensity * intensity, size=ks, mode='reflect',
        )
        local_var_I = local_mean_I2 - local_mean_I * local_mean_I
        np.maximum(local_var_I, 0.0, out=local_var_I)

        if noise_var <= 0.0:
            noise_var = float(np.mean(local_var_I))

        # Adaptive weight from intensity variance
        safe_var = np.where(local_var_I > 0, local_var_I, 1.0)
        weight = np.where(
            local_var_I > 0,
            np.maximum(0.0, 1.0 - noise_var / safe_var),
            0.0,
        )

        return local_mean_z + weight * (z - local_mean_z)
