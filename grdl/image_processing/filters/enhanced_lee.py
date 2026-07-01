# -*- coding: utf-8 -*-
"""
Enhanced Lee Filter - Adaptive SAR speckle filter with hard edge/target thresholds.

Implements the Enhanced Lee filter (Lopes et al., 1990) which classifies each
pixel as homogeneous, heterogeneous, or a point target and applies a
structurally-appropriate response in each case:

    cu    = 1 / sqrt(ENL)          — expected CV in pure homogeneous speckle
    cmax  = sqrt(1 + 2 / ENL)      — threshold for isolated bright targets
    ci    = std_local(a) / mean_local(a)   (local amplitude CV)

    if ci <= cu:    out = mean_local               (smooth — homogeneous)
    if ci >= cmax:  out = pixel                    (preserve — point target)
    else:           W = exp(-damp * (ci - cu) / (cmax - ci))
                    out = mean_local * W + pixel * (1 - W)  (mixed)

Key advantages over the standard Lee MMSE filter:
  - Hard point-target protection via ``cmax`` threshold.
  - Hard homogeneous-region smoothing via ``cu`` threshold.
  - Exponential transition weight gives smoother spatial appearance.
  - ``damp`` parameter controls transition steepness.

For **complex SLC** input the filter is applied to the **amplitude** ``|z|``
and the original phase is preserved (same strategy as ``ComplexLeeFilter``
and ``LeeSigmaFilter``).

References
----------
Lopes, A., Touzi, R., and Nezry, E. (1990). Adaptive speckle filters and
    scene heterogeneity. IEEE Transactions on Geoscience and Remote Sensing,
    28(6), 992–1000.

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
2026-06-30

Modified
--------
2026-06-30
"""

# Standard library
import logging
import math
from typing import Annotated, Any

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
from grdl.image_processing.params import Desc, Range
from grdl.vocabulary import ImageModality, ProcessorCategory

logger = logging.getLogger(__name__)


@processor_version('1.0.0')
@processor_tags(category=ProcessorCategory.FILTERS,
                modalities=[ImageModality.SAR])
class EnhancedLeeFilter(BandwiseTransformMixin, SARFilter):
    """Enhanced Lee adaptive speckle filter (Lopes et al., 1990).

    Classifies each pixel into one of three regimes based on the local
    amplitude coefficient of variation ``ci = std_local(a) / mean_local(a)``::

        cu   = 1 / sqrt(ENL)        — CV threshold: homogeneous speckle
        cmax = sqrt(1 + 2 / ENL)    — CV threshold: isolated point target

        ci <= cu    →  out = mean_local                  (fully smooth)
        ci >= cmax  →  out = pixel                       (fully preserve)
        cu < ci < cmax:
            W   = exp(-damp * (ci - cu) / (cmax - ci))
            out = mean_local * W + pixel * (1 - W)

    Advantages over ``LeeFilter`` (MMSE):

    * **Point-target protection** — ``cmax`` prevents any smoothing of
      bright isolated pixels (ships, corner reflectors, buildings).
    * **Sharper homogeneous smoothing** — ``cu`` hard threshold guarantees
      the full local mean in pure speckle.
    * **Smoother transitions** — exponential weight vs. linear MMSE weight.

    For **complex SLC** input the filter is applied to amplitude ``|z|``
    and the original interferometric phase is preserved exactly.

    Parameters
    ----------
    kernel_size : int
        Square kernel side length in pixels.  Must be odd and >= 3.
        Default is 7.
    enl : float
        Equivalent Number of Looks.  If 0.0, automatically estimated
        from the image.  Default is 0.0 (auto-estimate).
    damp : float
        Exponential damping factor >= 0.  Controls the steepness of the
        mixed-region weight.  Higher values push weight towards 0 (more
        smoothing of mixed regions); 0 gives a uniform weight of 1/e.
        Default is 1.0.

    Examples
    --------
    >>> from grdl.image_processing.filters import EnhancedLeeFilter
    >>> elf = EnhancedLeeFilter(kernel_size=7, damp=1.0)
    >>> filtered = elf.apply(slc)          # complex SLC — phase preserved
    >>> filtered = elf.apply(np.abs(slc))  # amplitude

    References
    ----------
    Lopes, A., Touzi, R., and Nezry, E. (1990). Adaptive speckle filters and
        scene heterogeneity. IEEE Transactions on Geoscience and Remote Sensing,
        28(6), 992–1000.
    """

    __gpu_compatible__ = True

    kernel_size: Annotated[int, Range(min=3, max=31),
                           Desc('Square kernel side length (odd)')] = 7
    enl: Annotated[float, Range(min=0.0),
                   Desc('Equivalent Number of Looks (0 = auto)')] = 0.0
    damp: Annotated[float, Range(min=0.0),
                    Desc('Exponential damping factor for the mixed-region weight')] = 1.0

    def __init__(
        self,
        kernel_size: int = 7,
        enl: float = 0.0,
        damp: float = 1.0,
    ) -> None:
        super().__init__(kernel_size=kernel_size, enl=enl)
        if damp < 0.0:
            raise ValidationError(f"damp must be >= 0, got {damp}")
        self.damp = damp

    def _apply_2d(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Enhanced Lee filter to a single 2D band.

        Parameters
        ----------
        source : np.ndarray
            2D array, shape ``(rows, cols)``.  Real (amplitude) or complex
            (SLC).  Accepts cupy arrays when GPU is available.

        Returns
        -------
        np.ndarray
            Despeckled image, same shape.  dtype float64 for real input,
            complex128 for complex input.
        """
        params = self._resolve_params(kwargs)
        ks   = params['kernel_size']
        enl  = params['enl']
        damp = params['damp']
        validate_kernel_size(ks)

        _is_gpu = _HAS_CUPY and isinstance(source, cp.ndarray)
        xp = cp if _is_gpu else np
        _ndi = _cupyx_ndimage.uniform_filter if _is_gpu else _scipy_uniform_filter

        eps = xp.finfo(xp.float64).tiny
        is_complex = np.iscomplexobj(source)

        if is_complex:
            z   = source.astype(xp.complex128)
            amp = xp.sqrt(z.real * z.real + z.imag * z.imag)
        else:
            z   = None
            amp = xp.abs(source).astype(xp.float64)

        # Local amplitude statistics
        local_mean  = _ndi(amp,       size=ks, mode='reflect')
        local_mean2 = _ndi(amp * amp, size=ks, mode='reflect')
        local_var   = xp.maximum(local_mean2 - local_mean * local_mean, 0.0)

        # Ci² = var / mean² ; ci = sqrt(Ci²)
        ci2 = local_var / (local_mean * local_mean + eps)
        ci  = xp.sqrt(ci2)

        # ENL estimation from amplitude Ci²
        if enl <= 0.0:
            estimated_enl = self._estimate_enl(ci2)
            logger.debug("EnhancedLeeFilter: auto-estimated ENL=%.2f", estimated_enl)
        else:
            estimated_enl = float(enl)
            logger.debug("EnhancedLeeFilter: user-provided ENL=%.2f", estimated_enl)

        cu   = 1.0 / math.sqrt(estimated_enl)       # homogeneous threshold
        cmax = math.sqrt(1.0 + 2.0 / estimated_enl) # point-target threshold

        logger.debug(
            "EnhancedLeeFilter: ENL=%.2f, cu=%.4f, cmax=%.4f",
            estimated_enl, cu, cmax,
        )

        # Mixed-region weight W: 1 = smooth (use mean), 0 = preserve (use pixel)
        #   W = exp(-damp * (ci - cu) / (cmax - ci))
        gap     = xp.maximum(cmax - ci, eps)                   # avoid div/0 as ci→cmax
        exp_arg = -float(damp) * xp.maximum(ci - cu, 0.0) / gap
        W_mixed = xp.exp(xp.clip(exp_arg, -500.0, 0.0))

        # Regime classification:
        #   ci <= cu   → W = 1 (full smoothing)
        #   ci >= cmax → W = 0 (point target, no smoothing)
        #   otherwise  → W = W_mixed  (exponential blend)
        in_mixed = (ci > cu) & (ci < cmax)
        W = xp.where(ci <= cu, 1.0, xp.where(in_mixed, W_mixed, 0.0))

        # out = mean * W + pixel * (1 - W)  =  pixel + W * (mean - pixel)
        amp_filtered = amp + W * (local_mean - amp)

        if is_complex:
            unit_phasor = z / (amp + eps)
            return amp_filtered * unit_phasor
        else:
            return amp_filtered
