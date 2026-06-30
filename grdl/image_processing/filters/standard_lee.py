# -*- coding: utf-8 -*-
"""
Standard Lee Filters - Adaptive Lee speckle filters for SAR imagery.

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
Jason Fritz, PhD
43161141+stryder-vtx@users.noreply.github.com

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
2026-06-24
"""

# Standard library
import logging
from typing import Any

# Third-party
import numpy as np
from scipy.ndimage import uniform_filter as _scipy_uniform_filter

try:
    import cupy as cp
    import cupyx.scipy.ndimage as _cupyx_ndimage
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False
    cp = None
    _cupyx_ndimage = None

# GRDL internal
from grdl.image_processing.base import BandwiseTransformMixin
from grdl.image_processing.filters._validation import validate_kernel_size
from grdl.image_processing.filters.sar_base import SARFilter
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.exceptions import ValidationError
from grdl.vocabulary import ImageModality, ProcessorCategory

logger = logging.getLogger(__name__)


@processor_version('2.0.0')
@processor_tags(category=ProcessorCategory.FILTERS,
                modalities=[ImageModality.SAR])
class LeeFilter(BandwiseTransformMixin, SARFilter):
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

    def __init__(
        self,
        kernel_size: int = 7,
        enl: float = 0.0,
    ) -> None:
        super().__init__(kernel_size=kernel_size, enl=enl)

    def _apply_2d(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Lee speckle filter to a single 2D band.

        Parameters
        ----------
        source : np.ndarray
            2D image array, shape ``(rows, cols)``. Typically SAR
            intensity or amplitude data. Accepts cupy arrays.

        Returns
        -------
        np.ndarray
            Despeckled image, same shape, float64.
        """
        params = self._resolve_params(kwargs)
        ks = params['kernel_size']
        enl = params['enl']
        validate_kernel_size(ks)

        if np.iscomplexobj(source):
            raise ValidationError(
                "LeeFilter requires real-valued amplitude or intensity input. "
                "For complex SLC data, use ComplexLeeFilter."
            )

        _is_gpu = _HAS_CUPY and isinstance(source, cp.ndarray)
        xp = cp if _is_gpu else np
        ndi_uniform = _cupyx_ndimage.uniform_filter if _is_gpu else _scipy_uniform_filter

        eps = xp.finfo(xp.float64).tiny
        x = source.astype(xp.float64)

        # Local statistics via variance decomposition: var = E[X²] - E[X]²
        local_mean = ndi_uniform(x, size=ks, mode='reflect')
        local_mean_sq = ndi_uniform(x * x, size=ks, mode='reflect')
        local_var = local_mean_sq - local_mean * local_mean
        xp.maximum(local_var, 0.0, out=local_var)

        # Coefficient of variation squared: Ci² = var / mean²
        ci2 = local_var / (local_mean * local_mean + eps)

        # ENL estimation or use provided value
        if enl <= 0.0:
            estimated_enl = self._estimate_enl(ci2)
            logger.debug("LeeFilter: auto-estimated ENL=%.2f", estimated_enl)
        else:
            estimated_enl = enl
            logger.debug("LeeFilter: user-provided ENL=%.2f", estimated_enl)

        # Noise coefficient of variation: Cn² = 1/ENL
        cn2 = 1.0 / estimated_enl

        # Adaptive weight: W = clamp(1 - Cn²/Ci², 0, 1)
        weight = xp.clip(1.0 - cn2 / (ci2 + eps), 0.0, 1.0)

        return local_mean + weight * (x - local_mean)


@processor_version('2.0.0')
@processor_tags(category=ProcessorCategory.FILTERS,
                modalities=[ImageModality.SAR])
class ComplexLeeFilter(BandwiseTransformMixin, SARFilter):
    """Phase-preserving Lee adaptive speckle filter for complex SAR SLC data.

    Applies the Lee MMSE filter to the **amplitude** ``a = |z|`` and then
    reconstructs a complex output by pairing the filtered amplitude with the
    original pixel phase::\n
        a       = |z|                            (instantaneous amplitude)
        Ci²     = var_local(a) / E_local[a]²    (amplitude coeff. of variation)
        Cn²     = 1 / ENL                       (noise coeff. of variation)
        W       = clamp(1 - Cn²/Ci², 0, 1)
        a_out   = E_local[a] + W * (a - E_local[a])
        z_out   = a_out * exp(j·angle(z))

    Working in the amplitude domain (like ``LeeFilter``) produces smoother
    MMSE weights than the intensity domain, reducing blocky edge-boundary
    artifacts.  The original interferometric phase ``angle(z)`` is preserved
    exactly.

    The output amplitude is numerically identical to ``LeeFilter`` applied to
    ``np.abs(slc)``; only the complex phase is added back.  This is the
    appropriate choice for single-pol SLC pre-processing (InSAR, PolSAR).

    When ``Ci² >> Cn²`` (bright target / edge), ``W → 1`` and the output
    amplitude tracks the observed value.  When ``Ci² ≈ Cn²`` (homogeneous
    speckle), ``W → 0`` and the output amplitude approaches the local mean \u2014
    reducing speckle correctly.

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

    __gpu_compatible__ = True

    def __init__(
        self,
        kernel_size: int = 7,
        enl: float = 0.0,
    ) -> None:
        super().__init__(kernel_size=kernel_size, enl=enl)

    def _apply_2d(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply complex Lee speckle filter to a single 2D band.

        Parameters
        ----------
        source : np.ndarray
            2D complex-valued array, shape ``(rows, cols)``.
            Accepts cupy arrays.

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

        _is_gpu = _HAS_CUPY and isinstance(source, cp.ndarray)
        xp = cp if _is_gpu else np
        ndi_uniform = _cupyx_ndimage.uniform_filter if _is_gpu else _scipy_uniform_filter

        eps = xp.finfo(xp.float64).tiny
        z = source.astype(xp.complex128)

        # Amplitude and intensity
        intensity = z.real * z.real + z.imag * z.imag
        amp = xp.sqrt(intensity)

        # Local statistics in amplitude domain (smoother Ci², fewer
        # edge-boundary artifacts than intensity domain)
        local_mean_amp = ndi_uniform(amp, size=ks, mode='reflect')
        local_mean_I   = ndi_uniform(intensity, size=ks, mode='reflect')
        # var(a) = E[a²] - E[a]² = E[I] - E[a]²
        local_var_amp = local_mean_I - local_mean_amp * local_mean_amp
        xp.maximum(local_var_amp, 0.0, out=local_var_amp)

        # Coefficient of variation squared: Ci² = var(a) / E[a]²
        ci2 = local_var_amp / (local_mean_amp * local_mean_amp + eps)

        # ENL estimation or use provided value
        if enl <= 0.0:
            estimated_enl = self._estimate_enl(ci2)
        else:
            estimated_enl = enl

        logger.debug(
            "ComplexLeeFilter: input shape %s, ENL=%.2f",
            source.shape, estimated_enl,
        )

        # Noise coefficient of variation: Cn² = 1/ENL
        cn2 = 1.0 / estimated_enl

        # Adaptive MMSE weight: W = clamp(1 - Cn²/Ci², 0, 1)
        weight = xp.clip(1.0 - cn2 / (ci2 + eps), 0.0, 1.0)

        # Amplitude MMSE (identical to LeeFilter applied to |z|)
        amp_filtered = local_mean_amp + weight * (amp - local_mean_amp)
        xp.maximum(amp_filtered, 0.0, out=amp_filtered)

        # Reconstruct: filtered amplitude × original unit phasor
        unit_phasor = z / (amp + eps)
        return amp_filtered * unit_phasor
