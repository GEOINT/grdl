# -*- coding: utf-8 -*-
"""
Touzi Coherent Target Scattering Vector Model (CTSVM) Decomposition.

Performs the Touzi decomposition (Touzi, 2007) of the 3x3 coherency matrix
[T3] into per-eigenvector Coherent Target Scattering Vector Model (CTSVM)
parameters plus probability-weighted means.

For each of the three eigenvectors v_k of [T3], four CTSVM parameters are
extracted:

    alpha_k  — scattering type angle [0, 90]° (surface → dihedral)
    phi_k    — phase of target scattering mechanism [−90, 90]°
    tau_k    — helicity (target symmetry) [−45, 45]°
    psi_k    — target orientation angle [−45, 45]°

Probability-weighted mean values (alpha_mean, phi_mean, tau_mean, psi_mean)
provide single-pixel summary scalars analogous to the Cloude-Pottier alpha.

References
----------
Touzi, R. (2007). "Target scattering decomposition in terms of roll-invariant
    target parameters," IEEE Transactions on Geoscience and Remote Sensing,
    45(1), pp. 73–84.

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
class TouziDecomposition(PolarimetricDecomposition):
    """Touzi Coherent Target Scattering Vector Model (CTSVM) decomposition.

    Decomposes quad-pol SAR data via eigenanalysis of [T3] into three sets
    of CTSVM parameters — one per eigenvector — plus probability-weighted
    mean values:

    Per-eigenvector (k = 1, 2, 3):

    - **alpha_k**: scattering type angle [0, 90]°.
      0 = surface, 45 = dipole/volume, 90 = dihedral.

    - **phi_k**: target scattering mechanism phase [−90, 90]°.

    - **tau_k**: helicity / target asymmetry [−45, 45]°.
      0 = symmetric, ±45 = maximum helicity.

    - **psi_k**: target orientation angle [−45, 45]°.

    Weighted-mean scalars:
    ``alpha_mean``, ``phi_mean``, ``tau_mean``, ``psi_mean``.

    Parameters
    ----------
    window_size : int
        Side length of the boxcar averaging window for [T3].
        Must be odd and >= 3.  Default 7.

    Examples
    --------
    >>> from grdl.image_processing.decomposition import TouziDecomposition
    >>> touzi = TouziDecomposition(window_size=7)
    >>> comp = touzi.decompose(shh, shv, svh, svv)
    >>> print(comp['alpha_mean'].shape)

    References
    ----------
    Touzi, R. (2007). "Target scattering decomposition in terms of roll-
        invariant target parameters," IEEE Transactions on Geoscience and
        Remote Sensing, 45(1), pp. 73–84.
    """

    __gpu_compatible__ = False

    window_size: Annotated[int, Range(min=1, max=31),
                           Desc('Boxcar averaging window size')] = 7

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def component_names(self) -> Tuple[str, ...]:
        return (
            'alpha1', 'alpha2', 'alpha3',
            'phi1',   'phi2',   'phi3',
            'tau1',   'tau2',   'tau3',
            'psi1',   'psi2',   'psi3',
            'alpha_mean', 'phi_mean', 'tau_mean', 'psi_mean',
        )

    def decompose(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: np.ndarray,
        svv: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Decompose quad-pol data into Touzi CTSVM parameters.

        Parameters
        ----------
        shh, shv, svh, svv : np.ndarray
            Complex scattering matrix channels, shape ``(rows, cols)``.

        Returns
        -------
        Dict[str, np.ndarray]
            16 real float64 arrays (all in degrees); see ``component_names``.
        """
        self._validate_scattering_matrix(shh, shv, svh, svv)
        self._validate_internal_matrix_window_size('decompose_from_t3')
        channels = np.stack([shh, shv, svh, svv], axis=0)
        t3 = CoherencyMatrix(window_size=self.window_size).compute(channels)
        return self.decompose_from_t3(t3)

    def decompose_from_t3(self, t3: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute Touzi parameters from a pre-computed [T3].

        Parameters
        ----------
        t3 : np.ndarray
            Shape ``(3, 3, rows, cols)``, complex.

        Returns
        -------
        Dict[str, np.ndarray]
            16 real float64 arrays (all in degrees).
        """
        if t3.ndim != 4 or t3.shape[:2] != (3, 3):
            raise ValueError(
                f"Expected t3 shape (3, 3, rows, cols), got {t3.shape}"
            )
        rows, cols = t3.shape[2], t3.shape[3]
        N = rows * cols
        eps = 1e-8
        pi = np.pi

        # -- Reshape to (N, 3, 3) for batch eigendecomposition --
        M_flat = t3.transpose(2, 3, 0, 1).reshape(N, 3, 3)

        # -- Eigendecomposition (use eig, not eigh: eigenvectors need
        #    complex phases for the TSVM parameter extraction) --
        lambda_vals, V = np.linalg.eig(M_flat)  # (N,3), (N,3,3)

        # Sort by descending real eigenvalue
        idx = np.argsort(lambda_vals.real, axis=1)[:, ::-1]
        lambda_sorted = np.take_along_axis(lambda_vals.real, idx, axis=1)
        # Rearrange eigenvectors to match sorted eigenvalue order
        # V[:, :, k] is k-th eigenvector; need to permute last axis
        V_sorted = V[np.arange(N)[:, None, None],
                     np.arange(3)[None, :, None],
                     idx[:, None, :]]   # (N, 3, 3)

        # Clip negative eigenvalues (numerical noise)
        np.clip(lambda_sorted, 0.0, None, out=lambda_sorted)

        # -- Remove global phase of each eigenvector (phase of V[0,:]) --
        phase = np.arctan2(
            V_sorted[:, 0, :].imag,
            eps + V_sorted[:, 0, :].real,
        )  # (N, 3)
        V_sorted = V_sorted * np.exp(-1j * phase[:, np.newaxis, :])

        # -- Compute psi: orientation angle from V[1] and V[2] --
        psi = 0.5 * np.arctan2(
            V_sorted[:, 2, :].real,
            eps + V_sorted[:, 1, :].real,
        )  # (N, 3)

        # -- Rotate V[1,2] components by 2*psi --
        cos2psi = np.cos(2.0 * psi)   # (N, 3)
        sin2psi = np.sin(2.0 * psi)
        V1r = V_sorted[:, 1, :].real
        V1i = V_sorted[:, 1, :].imag
        V2r = V_sorted[:, 2, :].real
        V2i = V_sorted[:, 2, :].imag

        V_sorted[:, 1, :] = (
            V1r * cos2psi + V2r * sin2psi
            + 1j * (V1i * cos2psi + V2i * sin2psi)
        )
        V_sorted[:, 2, :] = (
            -V1r * sin2psi + V2r * cos2psi
            + 1j * (-V1i * sin2psi + V2i * cos2psi)
        )

        # -- tau: helicity from V[2].imag and V[0].real --
        tau = 0.5 * np.arctan2(
            -V_sorted[:, 2, :].imag,
            eps + V_sorted[:, 0, :].real,
        )  # (N, 3)

        # -- phi: phase of V[1] after psi-rotation --
        phi = np.arctan2(
            V_sorted[:, 1, :].imag,
            eps + V_sorted[:, 1, :].real,
        )  # (N, 3)

        # -- Rotate V[0,2] by 2*tau to extract alpha --
        cos2tau = np.cos(2.0 * tau)
        sin2tau = np.sin(2.0 * tau)
        V0r = V_sorted[:, 0, :].real
        V0i = V_sorted[:, 0, :].imag
        V2r = V_sorted[:, 2, :].real
        V2i = V_sorted[:, 2, :].imag

        V_sorted[:, 0, :] = (
            V0r * cos2tau - V2i * sin2tau
            + 1j * (V0i * cos2tau + V2r * sin2tau)
        )
        V_sorted[:, 2, :] = (
            -V0i * sin2tau + V2r * cos2tau
            + 1j * (V0r * sin2tau + V2i * cos2tau)
        )

        # -- Alpha: scattering type from V[0].real after tau-rotation --
        alpha = np.arccos(
            np.clip(V_sorted[:, 0, :].real, -1.0, 1.0)
        )  # (N, 3)

        # -- Sign correction: flip tau and phi when |psi| > pi/4 --
        flip = (psi < -pi / 4.0) | (psi > pi / 4.0)
        tau = np.where(flip, -tau, tau)
        phi = np.where(flip, -phi, phi)

        # -- Probability weights --
        total = lambda_sorted.sum(axis=1, keepdims=True)
        safe_total = np.where(total > 0.0, total, 1.0)
        p = np.clip(lambda_sorted / safe_total, 0.0, 1.0)  # (N, 3)

        # -- Probability-weighted means --
        alpha_mean = (alpha * p).sum(axis=1)
        phi_mean   = (phi   * p).sum(axis=1)
        tau_mean   = (tau   * p).sum(axis=1)
        psi_mean   = (psi   * p).sum(axis=1)

        # -- Convert to degrees and reshape --
        to_deg = 180.0 / pi
        shape = (rows, cols)

        a  = (alpha * to_deg).T.reshape((3,) + shape)  # (3, rows, cols)
        ph = (phi   * to_deg).T.reshape((3,) + shape)
        ta = (tau   * to_deg).T.reshape((3,) + shape)
        ps = (psi   * to_deg).T.reshape((3,) + shape)

        logger.debug(
            "TouziDecomposition: mean alpha_mean=%.1f°, mean tau_mean=%.1f°",
            float(np.nanmean(alpha_mean * to_deg)),
            float(np.nanmean(tau_mean * to_deg)),
        )
        return {
            'alpha1': a[0],  'alpha2': a[1],  'alpha3': a[2],
            'phi1':   ph[0], 'phi2':   ph[1], 'phi3':   ph[2],
            'tau1':   ta[0], 'tau2':   ta[1], 'tau3':   ta[2],
            'psi1':   ps[0], 'psi2':   ps[1], 'psi3':   ps[2],
            'alpha_mean': (alpha_mean * to_deg).reshape(shape),
            'phi_mean':   (phi_mean   * to_deg).reshape(shape),
            'tau_mean':   (tau_mean   * to_deg).reshape(shape),
            'psi_mean':   (psi_mean   * to_deg).reshape(shape),
        }

    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        """Create an RGB composite from Touzi mean parameters.

        - **Red**: alpha_mean [0, 90°]
        - **Green**: |tau_mean| [0, 45°]
        - **Blue**: psi_mean [−45, 45°]
        """
        from grdl.IO.models.base import ImageMetadata, ChannelMetadata

        def _stretch(arr, lo, hi):
            return np.clip(
                (arr - lo) / max(hi - lo, 1e-8), 0.0, 1.0
            ).astype(np.float32)

        r = _stretch(components['alpha_mean'], 0.0, 90.0)
        g = _stretch(np.abs(components['tau_mean']), 0.0, 45.0)
        b = _stretch(components['psi_mean'], -45.0, 45.0)
        rgb = np.stack([r, g, b], axis=0)

        meta = ImageMetadata(
            format='TouziRGB',
            rows=rgb.shape[1],
            cols=rgb.shape[2],
            bands=3,
            dtype='float32',
            axis_order='CYX',
            channel_metadata=[
                ChannelMetadata(index=0, name='alpha_mean', role='rgb_red'),
                ChannelMetadata(index=1, name='tau_mean',   role='rgb_green'),
                ChannelMetadata(index=2, name='psi_mean',   role='rgb_blue'),
            ],
        )
        return rgb, meta
