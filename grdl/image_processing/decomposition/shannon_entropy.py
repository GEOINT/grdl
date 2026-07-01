# -*- coding: utf-8 -*-
"""
Shannon Entropy Decomposition (Full-Pol) — intensity and polarimetric entropy.

Decomposes the information content of the 3x3 coherency matrix [T3] into
two orthogonal contributions:

    H_I  = 3 * log(e * π * I / 3)        (intensity Shannon entropy)
    H_P  = log(1 - m)                     (polarimetric Shannon entropy)
    H    = H_I + H_P  (via nansum)

where I = trace([T3]) is the total scattered power and

    m = 1 - 27 * det([T3]) / trace([T3])^3

is the intermediate polarimetric measure (note: *without* sqrt, unlike the
Barakat DoP in ``DegreeOfPolarization``).

H_P captures the polarimetric disorder (0 for fully random, negative for
concentrated scattering); H_I reflects the absolute intensity.

References
----------
Morio, J., Gini, F., Réfrégier, P., and Goudail, F. (2009).
    "A Shannon entropy interpretation of the Shannon capacity,"
    IEEE Signal Processing Letters, 16(3), pp. 193–196.
Pottier, E. and Lee, J.S. (2009).  "Unsupervised classification scheme
    of POLSAR images based on the complex Wishart distribution and
    H/A/Alpha polarimetric decomposition theorem."

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
from typing import Annotated, Dict, Tuple, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.decomposition.base import PolarimetricDecomposition
from grdl.image_processing.decomposition.pol_matrix import CoherencyMatrix
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.image_processing.params import Range, Desc
from grdl.vocabulary import ImageModality

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata

logger = logging.getLogger(__name__)


@processor_version('1.0.0')
@processor_tags(modalities=[ImageModality.SAR])
class ShannonEntropy(PolarimetricDecomposition):
    """Shannon entropy decomposition of the full-pol coherency matrix [T3].

    Separates the total information content of [T3] into an intensity
    component H_I (dependent only on total power) and a polarimetric
    component H_P (dependent only on the relative distribution of power
    across scattering mechanisms):

        H_I = 3 * log(e * π * I / 3)
        H_P = log(1 - m)
        H   = H_I + H_P

    where ``I = trace([T3])`` and
    ``m = 1 - 27 * det([T3]) / trace([T3])^3``.

    Degenerate pixels (fully polarised or zero-power) yield NaN for
    H_P and/or H_I; ``H_total`` is formed with ``nansum`` so that a
    valid contribution from the other term is preserved.

    Parameters
    ----------
    window_size : int
        Side length of the boxcar averaging window for [T3].
        Must be odd and >= 3.  Default 7.

    Examples
    --------
    >>> from grdl.image_processing.decomposition import ShannonEntropy
    >>> she = ShannonEntropy(window_size=7)
    >>> comp = she.decompose(shh, shv, svh, svv)
    >>> print(comp.keys())  # H_total, H_intensity, H_polarimetric

    References
    ----------
    Morio, J., Gini, F., Réfrégier, P., and Goudail, F. (2009).
        "A Shannon entropy interpretation of the Shannon capacity,"
        IEEE Signal Processing Letters, 16(3), pp. 193–196.
    """

    __gpu_compatible__ = False

    window_size: Annotated[int, Range(min=3, max=31),
                           Desc('Boxcar averaging window size')] = 7

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def component_names(self) -> Tuple[str, ...]:
        return ('H_total', 'H_intensity', 'H_polarimetric')

    def decompose(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: np.ndarray,
        svv: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Decompose quad-pol data into Shannon entropy components.

        Parameters
        ----------
        shh, shv, svh, svv : np.ndarray
            Complex scattering matrix channels, each shape ``(rows, cols)``.

        Returns
        -------
        Dict[str, np.ndarray]
            Keys: ``'H_total'``, ``'H_intensity'``, ``'H_polarimetric'``.
        """
        self._validate_scattering_matrix(shh, shv, svh, svv)
        channels = np.stack([shh, shv, svh, svv], axis=0)
        t3 = CoherencyMatrix(window_size=self.window_size).compute(channels)
        return self.decompose_from_t3(t3)

    def decompose_from_t3(self, t3: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute Shannon entropy from a pre-computed coherency matrix [T3].

        Parameters
        ----------
        t3 : np.ndarray
            Shape ``(3, 3, rows, cols)``, complex.

        Returns
        -------
        Dict[str, np.ndarray]
            Keys: ``'H_total'``, ``'H_intensity'``, ``'H_polarimetric'``.
            H_P and H_I may contain NaN for degenerate pixels.
        """
        if t3.ndim != 4 or t3.shape[:2] != (3, 3):
            raise ValueError(
                f"Expected t3 shape (3, 3, rows, cols), got {t3.shape}"
            )
        rows, cols = t3.shape[2], t3.shape[3]

        t3_flat = t3.transpose(2, 3, 0, 1).reshape(-1, 3, 3)  # (N, 3, 3)

        det_v = np.linalg.det(t3_flat).real              # (N,)
        trace_v = (
            t3_flat[:, 0, 0].real
            + t3_flat[:, 1, 1].real
            + t3_flat[:, 2, 2].real
        )                                                  # (N,)

        eps = 1e-8
        I = np.maximum(trace_v, eps)

        # Intermediate polarimetric measure (not Barakat sqrt)
        m = 1.0 - 27.0 * det_v / (I ** 3 + eps)
        one_minus_m = 1.0 - m  # = 27*det / I^3

        # Polarimetric entropy: log(1 - m) = log(27*det/I^3)
        with np.errstate(divide='ignore', invalid='ignore'):
            HSP = np.where(
                one_minus_m > eps,
                np.log(np.abs(one_minus_m)),
                np.nan,
            )
        HSP[~np.isfinite(HSP)] = np.nan

        # Intensity entropy: 3 * log(e * π * I / 3)
        with np.errstate(divide='ignore', invalid='ignore'):
            HSI = 3.0 * np.log(np.e * np.pi * I / 3.0)
        HSI[~np.isfinite(HSI)] = np.nan

        # Total: nansum so a NaN in one component doesn't cancel a valid other
        HS = np.nansum(np.stack([HSP, HSI], axis=0), axis=0)

        logger.debug("ShannonEntropy: mean H_total = %.3f", float(np.nanmean(HS)))
        return {
            'H_total':       HS.reshape(rows, cols),
            'H_intensity':   HSI.reshape(rows, cols),
            'H_polarimetric': HSP.reshape(rows, cols),
        }

    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        """Create an RGB composite from Shannon entropy components.

        - **Red**: H_total (percentile stretched)
        - **Green**: H_intensity (percentile stretched)
        - **Blue**: H_polarimetric (percentile stretched)
        """
        from grdl.IO.models.base import ImageMetadata, ChannelMetadata

        def _stretch(arr):
            finite = arr[np.isfinite(arr)]
            lo = np.percentile(finite, percentile_low)
            hi = np.percentile(finite, percentile_high)
            return np.clip((arr - lo) / max(hi - lo, 1e-8), 0.0, 1.0).astype(
                np.float32
            )

        r = _stretch(components['H_total'])
        g = _stretch(components['H_intensity'])
        b = _stretch(components['H_polarimetric'])
        rgb = np.stack([r, g, b], axis=0)

        meta = ImageMetadata(
            format='ShannonEntropyRGB',
            rows=rgb.shape[1],
            cols=rgb.shape[2],
            bands=3,
            dtype='float32',
            axis_order='CYX',
            channel_metadata=[
                ChannelMetadata(index=0, name='H_total', role='rgb_red'),
                ChannelMetadata(index=1, name='H_intensity', role='rgb_green'),
                ChannelMetadata(
                    index=2, name='H_polarimetric', role='rgb_blue'
                ),
            ],
        )
        return rgb, meta
