# -*- coding: utf-8 -*-
"""
Lee Sigma Filter - Sigma-interval adaptive speckle filter for SAR imagery.

The Lee Sigma filter (Lee 1983) selects only those neighboring pixels whose
intensity falls within a probability-based confidence interval around the
center pixel intensity, avoiding contamination from edge-adjacent pixels
when estimating local statistics.  The MMSE Lee weighting is then applied
using only the selected pixels.

Selection interval ``[c_lo * I, c_hi * I]``
where ``c_lo`` and ``c_hi`` are the ``(1-sigma)/2`` and ``(1+sigma)/2``
quantiles of the ``Gamma(ENL, 1/ENL)`` single-look speckle distribution.
For example, ``sigma=0.9`` selects pixels within the 5th–95th percentile
range of the expected speckle distribution for the observed intensity.

For complex SLC input the selection criterion is applied to intensity
``|z|²`` while the adaptive weight is applied in the complex domain,
preserving interferometric phase.

References
----------
Lee, J.-S. (1983). Digital image smoothing and the sigma filter.
    Computer Vision, Graphics, and Image Processing, 24(2), 255–269.

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
2026-06-30

Modified
--------
2026-06-30
"""

# Standard library
import logging
from typing import Annotated, Any

# Third-party
import numpy as np
from scipy.ndimage import uniform_filter as _scipy_uniform_filter
from scipy.stats import gamma as _gamma_dist

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
class LeeSigmaFilter(BandwiseTransformMixin, SARFilter):
    """Lee Sigma adaptive speckle filter for SAR imagery.

    Reduces speckle contamination from edge-adjacent pixels by selecting
    only those neighbors whose intensity falls within a probability-based
    confidence interval around the center pixel::

        c_lo, c_hi = Gamma(ENL, 1/ENL).ppf([(1-sigma)/2, (1+sigma)/2])
        valid neighbors: I_nb in [c_lo * I_center, c_hi * I_center]

    Local statistics (mean and variance) are estimated from valid
    neighbors only.  An MMSE Lee weight is then derived::

        Ci²_s = var_s / mean_s²       (sigma-selected coefficient of variation)
        Cn²   = 1 / ENL               (noise coefficient of variation)
        W     = clamp(1 - Cn²/Ci²_s, 0, 1)
        out   = mean_s + W * (I - mean_s)

    When fewer than ``min_valid`` neighbors pass the sigma test, the
    full-kernel mean is used as the fallback reference.

    For **complex SLC** input the selection criterion uses intensity
    ``|z|²`` while the adaptive weight is applied in the complex domain,
    preserving interferometric phase (identical to ``ComplexLeeFilter``
    but with sigma-based neighbour selection).

    Parameters
    ----------
    kernel_size : int
        Square kernel side length in pixels. Must be odd and >= 3.
        Default is 7.
    enl : float
        Equivalent Number of Looks. If 0.0, automatically estimated
        from the image. Default is 0.0 (auto-estimate).
    sigma : float
        Probability coverage of the selection interval in (0, 1).
        ``sigma=0.9`` selects pixels within the 5th–95th percentile
        range of the Gamma(ENL) speckle distribution.
        Default is 0.9.
    min_valid : int
        Minimum number of neighbors that must pass the sigma test for
        the sigma statistics to be used.  If fewer pass, the full-
        kernel mean is substituted.  Default is 1.

    Examples
    --------
    Filter a real-valued intensity image:

    >>> from grdl.image_processing.filters import LeeSigmaFilter
    >>> lsf = LeeSigmaFilter(kernel_size=7, sigma=0.9)
    >>> despeckled = lsf.apply(sar_intensity)

    Filter a complex SLC (phase-preserving):

    >>> lsf = LeeSigmaFilter(kernel_size=7, enl=1.0, sigma=0.9)
    >>> despeckled_slc = lsf.apply(slc)

    References
    ----------
    Lee, J.-S. (1983). Digital image smoothing and the sigma filter.
    Computer Vision, Graphics, and Image Processing, 24(2), 255–269.
    """

    __gpu_compatible__ = False

    kernel_size: Annotated[int, Range(min=3, max=31),
                           Desc('Square kernel side length (odd)')] = 7
    enl: Annotated[float, Range(min=0.0),
                   Desc('Equivalent Number of Looks (0 = auto)')] = 0.0
    sigma: Annotated[float, Range(min=0.01, max=0.99),
                     Desc('Probability coverage of the sigma selection interval')] = 0.9
    min_valid: Annotated[int, Range(min=1, max=9801),
                         Desc('Minimum valid neighbours for sigma statistics')] = 1

    def __init__(
        self,
        kernel_size: int = 7,
        enl: float = 0.0,
        sigma: float = 0.9,
        min_valid: int = 1,
    ) -> None:
        super().__init__(kernel_size=kernel_size, enl=enl)
        if not (0.0 < sigma < 1.0):
            raise ValidationError(
                f"sigma must be in (0, 1), got {sigma}"
            )
        if not isinstance(min_valid, int) or min_valid < 1:
            raise ValidationError(
                f"min_valid must be an integer >= 1, got {min_valid}"
            )
        self.sigma = sigma
        self.min_valid = min_valid

    def _apply_2d(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Lee Sigma filter to a single 2D band.

        For both **real** and **complex** input, all statistics and MMSE
        are computed in the **amplitude** domain ``a = |z|`` (or ``|source|``
        for real input).  The sigma selection bounds are applied to amplitude
        values.  For complex input the filtered amplitude is paired with the
        original pixel phase, preserving interferometric phase exactly.  The
        output amplitude is numerically identical to applying the filter to
        ``np.abs(slc)``.

        Parameters
        ----------
        source : np.ndarray
            2D array, shape ``(rows, cols)``.  Real (amplitude) or complex
            (SLC).

        Returns
        -------
        np.ndarray
            Despeckled image, same shape.  dtype float64 for real input,
            complex128 for complex input.
        """
        params = self._resolve_params(kwargs)
        ks = params['kernel_size']
        enl = params['enl']
        sigma = params['sigma']
        min_valid = params['min_valid']
        validate_kernel_size(ks)

        eps = np.finfo(np.float64).tiny
        is_complex = np.iscomplexobj(source)

        if is_complex:
            z = source.astype(np.complex128)
            amp = np.sqrt(z.real * z.real + z.imag * z.imag)  # |z|
            I_select = amp   # use amplitude for sigma bounds (same as real branch)
        else:
            z = None
            amp = np.abs(source).astype(np.float64)
            I_select = amp   # real branch: amplitude throughout

        rows, cols = amp.shape

        # -- Full-kernel amplitude statistics for ENL estimation --
        local_mean_amp  = _scipy_uniform_filter(amp, size=ks, mode='reflect')
        local_mean_amp2 = _scipy_uniform_filter(amp * amp, size=ks, mode='reflect')
        local_var_amp   = np.maximum(
            local_mean_amp2 - local_mean_amp * local_mean_amp, 0.0
        )
        ci2_full = local_var_amp / (local_mean_amp * local_mean_amp + eps)

        # -- ENL estimation or user-provided --
        if enl <= 0.0:
            enl_est = self._estimate_enl(ci2_full)
            logger.debug("LeeSigmaFilter: auto-estimated ENL=%.2f", enl_est)
        else:
            enl_est = float(enl)
            logger.debug("LeeSigmaFilter: user-provided ENL=%.2f", enl_est)

        # -- Sigma selection bounds from Gamma(ENL, 1/ENL) distribution --
        # These bounds are applied to I_select (intensity for complex, amplitude for real).
        alpha = (1.0 - sigma) / 2.0
        c_lo = float(_gamma_dist.ppf(alpha, a=enl_est, scale=1.0 / enl_est))
        c_hi = float(_gamma_dist.ppf(1.0 - alpha, a=enl_est, scale=1.0 / enl_est))
        logger.debug(
            "LeeSigmaFilter: sigma=%.2f, ENL=%.2f → c_lo=%.4f, c_hi=%.4f",
            sigma, enl_est, c_lo, c_hi,
        )

        lo = I_select * c_lo
        hi = I_select * c_hi

        # -- Accumulate sigma-selected statistics in amplitude domain --
        pad = ks // 2
        I_pad   = np.pad(I_select, pad, mode='reflect')  # for selection test
        amp_pad = np.pad(amp, pad, mode='reflect')        # amplitude to accumulate

        sum_amp  = np.zeros((rows, cols), dtype=np.float64)  # sum a
        sum_amp2 = np.zeros((rows, cols), dtype=np.float64)  # sum a² (for var)
        count    = np.zeros((rows, cols), dtype=np.float64)

        for dy in range(ks):
            for dx in range(ks):
                nb_I   = I_pad  [dy:dy + rows, dx:dx + cols]
                nb_amp = amp_pad[dy:dy + rows, dx:dx + cols]
                valid  = ((nb_I >= lo) & (nb_I <= hi)).astype(np.float64)
                sum_amp  += nb_amp * valid
                sum_amp2 += nb_amp * nb_amp * valid
                count    += valid

        # -- Sigma-selected amplitude statistics --
        safe_count      = np.maximum(count, 1.0)
        mean_amp_sigma  = sum_amp / safe_count
        var_amp_sigma   = np.maximum(
            sum_amp2 / safe_count - mean_amp_sigma * mean_amp_sigma, 0.0
        )

        # -- MMSE Lee weight from sigma-selected amplitude statistics --
        ci2_sigma = var_amp_sigma / (mean_amp_sigma * mean_amp_sigma + eps)
        cn2    = 1.0 / enl_est
        weight = np.clip(1.0 - cn2 / (ci2_sigma + eps), 0.0, 1.0)

        # Fall back to full-kernel amplitude mean where count < min_valid
        use_sigma   = count >= float(min_valid)
        mean_amp_ref = np.where(use_sigma, mean_amp_sigma, local_mean_amp)

        # -- Amplitude MMSE --
        amp_filtered = mean_amp_ref + weight * (amp - mean_amp_ref)
        np.maximum(amp_filtered, 0.0, out=amp_filtered)

        if is_complex:
            # Reconstruct: filtered amplitude × original unit phasor
            unit_phasor = z / (amp + eps)
            return amp_filtered * unit_phasor
        else:
            return amp_filtered
