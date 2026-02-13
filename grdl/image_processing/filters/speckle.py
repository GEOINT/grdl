# -*- coding: utf-8 -*-
"""
Speckle Filters - Adaptive Lee filters for SAR imagery.

Provides real-valued and complex-valued Lee speckle filters for SAR imagery
using coefficient of variation (Ci²) weighting with automatic ENL estimation
from homogeneous regions.

``LeeFilter`` operates on real-valued intensity/amplitude data. The adaptive
weight is derived from the local coefficient of variation relative to the
estimated noise level (1/ENL).

``ComplexLeeFilter`` operates on complex-valued SLC data, applying the same
Ci²-based weighting in the complex domain to suppress speckle while
preserving interferometric phase.

Algorithm
---------
Both filters compute local statistics via ``uniform_filter`` and derive::

    Ci² = var(I) / E[I]²        (local coefficient of variation)
    Cn² = 1 / ENL               (noise coefficient of variation)
    W   = clamp(1 - Cn²/Ci², 0, 1)   (adaptive weight)
    out = E[x] + W * (x - E[x])

ENL is either user-supplied or automatically estimated from the bottom 10%
of Ci² values (assumed homogeneous regions).

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
from grdl.image_processing.base import BandwiseTransformMixin, ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.exceptions import ValidationError
from grdl.image_processing.filters._validation import validate_kernel_size
from grdl.vocabulary import ImageModality, ProcessorCategory


# ===================================================================
# ENL estimation helper
# ===================================================================

def _estimate_enl(ci2: np.ndarray, homogeneous_fraction: float = 0.1) -> float:
    """Estimate Equivalent Number of Looks from coefficient of variation.

    Sorts Ci² values and treats the lowest fraction as homogeneous
    regions, then estimates ENL = 1 / mean(Ci²_homogeneous).

    Parameters
    ----------
    ci2 : np.ndarray
        Coefficient of variation squared, Ci² = var(I) / E[I]².
    homogeneous_fraction : float
        Fraction of lowest-Ci² pixels assumed homogeneous.
        Default is 0.1 (10%).

    Returns
    -------
    float
        Estimated ENL (>= 1.0 for valid SAR data).
    """
    sorted_vals = np.sort(ci2.ravel())
    n_homog = max(1, int(len(sorted_vals) * homogeneous_fraction))
    mean_ci2 = float(np.mean(sorted_vals[:n_homog]))
    if mean_ci2 > 0:
        return 1.0 / mean_ci2
    return 1.0


@processor_version('2.0.0')
@processor_tags(category=ProcessorCategory.FILTERS,
                modalities=[ImageModality.SAR])
class LeeFilter(BandwiseTransformMixin, ImageTransform):
    """Lee adaptive speckle filter for SAR imagery.

    Uses coefficient of variation (Ci²) weighting with automatic ENL
    estimation from homogeneous regions. Preserves edges by weighting
    between the local mean and the observed pixel value::

        Ci² = var(I) / E[I]²          (local coefficient of variation)
        Cn² = 1 / ENL                 (noise coefficient of variation)
        W   = clamp(1 - Cn²/Ci², 0, 1)
        out = E[x] + W * (x - E[x])

    When local Ci² is high (edge/target), ``W -> 1`` (pass through).
    When Ci² is low (homogeneous), ``W -> 0`` (smooth to local mean).

    Parameters
    ----------
    kernel_size : int
        Square kernel side length in pixels. Must be odd and >= 3.
        Default is 7.
    enl : float
        Equivalent Number of Looks. Controls noise threshold.
        If 0.0, automatically estimated from the image by treating
        the bottom 10% of Ci² values as homogeneous regions.
        Default is 0.0 (auto-estimate).

    Examples
    --------
    >>> from grdl.image_processing.filters import LeeFilter
    >>> lee = LeeFilter(kernel_size=7)
    >>> despeckled = lee.apply(sar_intensity)

    With known ENL (e.g., 4-look data):

    >>> lee = LeeFilter(kernel_size=7, enl=4.0)
    >>> despeckled = lee.apply(sar_intensity)
    """

    __gpu_compatible__ = True

    kernel_size: Annotated[int, Range(min=3, max=31),
                           Desc('Square kernel side length (odd)')] = 7
    enl: Annotated[float, Range(min=0.0),
                   Desc('Equivalent Number of Looks (0 = auto)')] = 0.0

    def __init__(
        self,
        kernel_size: int = 7,
        enl: float = 0.0,
    ) -> None:
        validate_kernel_size(kernel_size)
        self.kernel_size = kernel_size
        self.enl = enl

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
        enl = params['enl']
        validate_kernel_size(ks)

        eps = np.finfo(np.float64).tiny
        x = source.astype(np.float64)

        # Local statistics via variance decomposition: var = E[X²] - E[X]²
        local_mean = uniform_filter(x, size=ks, mode='reflect')
        local_mean_sq = uniform_filter(x * x, size=ks, mode='reflect')
        local_var = local_mean_sq - local_mean * local_mean
        np.maximum(local_var, 0.0, out=local_var)

        # Coefficient of variation squared: Ci² = var / mean²
        ci2 = local_var / (local_mean * local_mean + eps)

        # ENL estimation or use provided value
        if enl <= 0.0:
            estimated_enl = _estimate_enl(ci2)
        else:
            estimated_enl = enl

        # Noise coefficient of variation: Cn² = 1/ENL
        cn2 = 1.0 / estimated_enl

        # Adaptive weight: W = clamp(1 - Cn²/Ci², 0, 1)
        weight = np.clip(1.0 - cn2 / (ci2 + eps), 0.0, 1.0)

        return local_mean + weight * (x - local_mean)


@processor_version('2.0.0')
@processor_tags(category=ProcessorCategory.FILTERS,
                modalities=[ImageModality.SAR])
class ComplexLeeFilter(BandwiseTransformMixin, ImageTransform):
    """Lee adaptive speckle filter for complex-valued SAR SLC data.

    Uses coefficient of variation (Ci²) weighting with automatic ENL
    estimation from homogeneous regions. Applies the adaptive weight
    in the complex domain to suppress speckle while preserving
    interferometric phase::

        Ci² = var(|z|²) / E[|z|²]²   (intensity coeff. of variation)
        Cn² = 1 / ENL                 (noise coeff. of variation)
        W   = clamp(1 - Cn²/Ci², 0, 1)
        z_out = E[z] + W * (z - E[z])

    where ``E[z]`` is the local complex mean (real and imaginary
    parts smoothed independently).

    When local Ci² is high (edge/target), ``W -> 1`` (pass through
    original complex value). When Ci² is low (homogeneous clutter),
    ``W -> 0`` (smooth to local complex mean).

    Parameters
    ----------
    kernel_size : int
        Square kernel side length in pixels. Must be odd and >= 3.
        Default is 7.
    enl : float
        Equivalent Number of Looks. Controls noise threshold.
        If 0.0, automatically estimated from the image by treating
        the bottom 10% of Ci² values as homogeneous regions.
        Default is 0.0 (auto-estimate).

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

    With known ENL (e.g., 4-look data):

    >>> clf = ComplexLeeFilter(kernel_size=7, enl=4.0)
    >>> despeckled_slc = clf.apply(complex_sar_image)
    """

    __gpu_compatible__ = False

    kernel_size: Annotated[int, Range(min=3, max=31),
                           Desc('Square kernel side length (odd)')] = 7
    enl: Annotated[float, Range(min=0.0),
                   Desc('Equivalent Number of Looks (0 = auto)')] = 0.0

    def __init__(
        self,
        kernel_size: int = 7,
        enl: float = 0.0,
    ) -> None:
        validate_kernel_size(kernel_size)
        self.kernel_size = kernel_size
        self.enl = enl

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
        enl = params['enl']
        validate_kernel_size(ks)

        eps = np.finfo(np.float64).tiny
        z = source.astype(np.complex128)

        # Local complex mean (smooth real and imag independently)
        local_mean_re = uniform_filter(z.real, size=ks, mode='reflect')
        local_mean_im = uniform_filter(z.imag, size=ks, mode='reflect')
        local_mean_z = local_mean_re + 1j * local_mean_im

        # Intensity statistics: I = |z|²
        intensity = z.real * z.real + z.imag * z.imag
        local_mean_I = uniform_filter(intensity, size=ks, mode='reflect')
        local_mean_I2 = uniform_filter(
            intensity * intensity, size=ks, mode='reflect',
        )
        local_var_I = local_mean_I2 - local_mean_I * local_mean_I
        np.maximum(local_var_I, 0.0, out=local_var_I)

        # Coefficient of variation squared: Ci² = var(I) / E[I]²
        ci2 = local_var_I / (local_mean_I * local_mean_I + eps)

        # ENL estimation or use provided value
        if enl <= 0.0:
            estimated_enl = _estimate_enl(ci2)
        else:
            estimated_enl = enl

        # Noise coefficient of variation: Cn² = 1/ENL
        cn2 = 1.0 / estimated_enl

        # Adaptive weight: W = clamp(1 - Cn²/Ci², 0, 1)
        weight = np.clip(1.0 - cn2 / (ci2 + eps), 0.0, 1.0)

        return local_mean_z + weight * (z - local_mean_z)
