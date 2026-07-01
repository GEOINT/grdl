# -*- coding: utf-8 -*-
"""
Neumann Polarimetric Decomposition (Full-Pol) — scattering diversity parameters.

Extracts four polarimetric descriptors from the full-pol coherency matrix [T3]
using the coherent-matrix de-orientation approach of Neumann et al. (2010):

    psi        — polarization orientation angle (terrain slope / azimuth tilt),
                  degrees, range ±45°
    delta_mod  — degree of polarization (modulus), ≥ 0
    delta_pha  — phase difference between the dominant and secondary
                  scattering mechanisms, degrees
    tau        — scattering diversity / depolarization index, ∈ [0, 1]

The algorithm rotates [T3] about the antenna boresight to minimise T23,
then extracts the descriptors from the de-oriented matrix.

References
----------
Neumann, M., Ferro-Famil, L., and Reigber, A. (2010). "Estimation of
    forest structure, ground, and canopy layer characteristics from
    multibaseline polarimetric interferometric SAR data," IEEE Trans.
    Geoscience and Remote Sensing, 48(3), pp. 1086–1104.

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
class NeumannDecomposition(PolarimetricDecomposition):
    """Neumann polarimetric decomposition (scattering diversity parameters).

    Extracts four descriptors from [T3] via coherent de-orientation:

    - **psi**: polarization orientation angle [deg], range ±45°.
      Measures azimuth tilt / terrain slope.

    - **delta_mod**: modulus of the coherence between dominant and
      secondary scattering mechanisms (degree of polarization), ≥ 0.

    - **delta_pha**: phase difference between mechanisms [deg].

    - **tau**: scattering diversity / depolarization index, ∈ [0, 1].
      tau = 0: single dominant mechanism;
      tau = 1: equal contribution from all mechanisms.

    Parameters
    ----------
    window_size : int
        Side length of the boxcar averaging window for [T3].
        Must be odd and >= 3.  Default 7.

    Examples
    --------
    >>> from grdl.image_processing.decomposition import NeumannDecomposition
    >>> neu = NeumannDecomposition(window_size=7)
    >>> comp = neu.decompose(shh, shv, svh, svv)
    >>> print(comp.keys())  # psi, delta_mod, delta_pha, tau

    From a pre-computed coherency matrix:

    >>> t3 = CoherencyMatrix(window_size=7).compute(channels)
    >>> comp = neu.decompose_from_t3(t3)

    References
    ----------
    Neumann, M., Ferro-Famil, L., and Reigber, A. (2010). "Estimation of
        forest structure, ground, and canopy layer characteristics from
        multibaseline polarimetric interferometric SAR data," IEEE Trans.
        Geoscience and Remote Sensing, 48(3), pp. 1086–1104.
    """

    __gpu_compatible__ = False

    window_size: Annotated[int, Range(min=3, max=31),
                           Desc('Boxcar averaging window size')] = 7

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def component_names(self) -> Tuple[str, ...]:
        return ('psi', 'delta_mod', 'delta_pha', 'tau')

    def decompose(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: np.ndarray,
        svv: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Decompose quad-pol data into Neumann parameters.

        Parameters
        ----------
        shh, shv, svh, svv : np.ndarray
            Complex scattering matrix channels, shape ``(rows, cols)``.

        Returns
        -------
        Dict[str, np.ndarray]
            Keys: ``'psi'``, ``'delta_mod'``, ``'delta_pha'``, ``'tau'``.
        """
        self._validate_scattering_matrix(shh, shv, svh, svv)
        channels = np.stack([shh, shv, svh, svv], axis=0)
        t3 = CoherencyMatrix(window_size=self.window_size).compute(channels)
        return self.decompose_from_t3(t3)

    def decompose_from_t3(self, t3: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute Neumann parameters from a pre-computed [T3].

        Parameters
        ----------
        t3 : np.ndarray
            Shape ``(3, 3, rows, cols)``, complex.

        Returns
        -------
        Dict[str, np.ndarray]
            Keys: ``'psi'``, ``'delta_mod'``, ``'delta_pha'``, ``'tau'``.
        """
        if t3.ndim != 4 or t3.shape[:2] != (3, 3):
            raise ValueError(
                f"Expected t3 shape (3, 3, rows, cols), got {t3.shape}"
            )

        eps = float(np.finfo(np.float64).eps)

        # Extract T3 elements
        T11 = np.real(t3[0, 0])          # (rows, cols)
        T12 = t3[0, 1]                   # complex
        T13 = t3[0, 2]                   # complex
        T22 = np.real(t3[1, 1])
        T23 = t3[1, 2]                   # complex
        T33 = np.real(t3[2, 2])

        # -- Polarization orientation angle --
        # Phi = 0.25 * (pi + arctan2(-2*Re(T23), Re(T33) - Re(T22)))
        T23_re = np.real(T23)
        T23_im = np.imag(T23)
        Phi = 0.25 * (np.pi + np.arctan2(-2.0 * T23_re, T33 - T22))

        # Constrain to ±45°
        Phi = np.where(Phi > np.pi / 4.0, Phi - np.pi / 2.0, Phi)

        psi = np.degrees(Phi)            # convert to degrees

        # -- De-orientation by 2*Phi --
        cos2 = np.cos(2.0 * Phi)
        sin2 = np.sin(2.0 * Phi)
        sin4 = np.sin(4.0 * Phi)

        T12_re = np.real(T12)
        T12_im = np.imag(T12)
        T13_re = np.real(T13)
        T13_im = np.imag(T13)

        # Rotated T12 (T01 in 0-indexed notation)
        T12re0 = T12_re * cos2 - T13_re * sin2
        T12im0 = T12_im * cos2 - T13_im * sin2

        # Rotated diagonal elements
        T220 = T22 * cos2 ** 2 - T23_re * sin4 + T33 * sin2 ** 2
        T330 = T22 * sin2 ** 2 - T23_re * sin4 + T33 * cos2 ** 2

        # -- Neumann descriptors --
        delta_mod = np.sqrt(
            np.maximum((T220 + T330) / (T11 + eps), 0.0)
        )
        delta_pha = np.degrees(np.arctan2(T12im0, T12re0 + eps))

        T12_abs = np.sqrt(T12re0 ** 2 + T12im0 ** 2)
        tau = 1.0 - (T12_abs / (T11 + eps)) / (delta_mod + eps)
        tau = np.clip(tau, 0.0, 1.0)

        logger.debug(
            "NeumannDecomposition: mean psi=%.1f°, mean delta_mod=%.3f",
            float(np.nanmean(psi)), float(np.nanmean(delta_mod)),
        )
        return {
            'psi':       psi,
            'delta_mod': delta_mod,
            'delta_pha': delta_pha,
            'tau':       tau,
        }

    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        """Create an RGB composite from Neumann parameters.

        - **Red**: psi (orientation angle, ±45° → [0, 1])
        - **Green**: delta_mod (degree of polarization, percentile stretched)
        - **Blue**: tau (scattering diversity [0, 1])
        """
        from grdl.IO.models.base import ImageMetadata, ChannelMetadata

        def _stretch(arr, lo=None, hi=None):
            if lo is None:
                lo = np.nanpercentile(arr, percentile_low)
            if hi is None:
                hi = np.nanpercentile(arr, percentile_high)
            return np.clip(
                (arr - lo) / max(hi - lo, 1e-8), 0.0, 1.0
            ).astype(np.float32)

        r = _stretch(components['psi'], lo=-45.0, hi=45.0)
        g = _stretch(components['delta_mod'])
        b = np.clip(components['tau'], 0.0, 1.0).astype(np.float32)
        rgb = np.stack([r, g, b], axis=0)

        meta = ImageMetadata(
            format='NeumannRGB',
            rows=rgb.shape[1],
            cols=rgb.shape[2],
            bands=3,
            dtype='float32',
            axis_order='CYX',
            channel_metadata=[
                ChannelMetadata(index=0, name='psi', role='rgb_red'),
                ChannelMetadata(index=1, name='delta_mod', role='rgb_green'),
                ChannelMetadata(index=2, name='tau', role='rgb_blue'),
            ],
        )
        return rgb, meta
