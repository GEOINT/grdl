# -*- coding: utf-8 -*-
"""
Praks Polarimetric Parameters (Full-Pol) — Frobenius-norm based descriptors.

Extracts seven polarimetric descriptors from the normalised 3x3 covariance
matrix [C3]/span (Praks et al., 2009):

    frobenius_norm          — ||M||_F^2  where M = C3/trace(C3)
    scattering_predominance — sqrt(||M||_F^2)
    scattering_diversity    — 1.5 * (1 - ||M||_F^2)
    degree_purity           — 2 * sqrt(max(||M||_F^2 - 0.25, 0))
    depolarization_index    — 1 - degree_purity / sqrt(3)
    alpha                   — arccos(clip(Re(M[0,0]), -1, 1)) [deg]
    entropy                 — 2.52 + 0.78*log|det(M + 0.16*I)| / log(3)

These parameters are alternatives to the Cloude-Pottier H/A/Alpha
that avoid eigendecomposition and are more robust under low-ENL conditions.

References
----------
Praks, J., Koeniguer, E.C. and Hallikainen, M.T. (2009). "Alternatives to
    target entropy and alpha angle in SAR polarimetry," IEEE Transactions on
    Geoscience and Remote Sensing, 47(7), pp. 2262–2274.

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
from grdl.image_processing.decomposition.pol_matrix import CovarianceMatrix
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.image_processing.params import Range, Desc
from grdl.vocabulary import ImageModality

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata

logger = logging.getLogger(__name__)


@processor_version('1.0.0')
@processor_tags(modalities=[ImageModality.SAR])
class PraksParameters(PolarimetricDecomposition):
    """Praks polarimetric parameters — Frobenius-norm alternatives to H/A/Alpha.

    Derives seven descriptors from the span-normalised covariance matrix
    ``M = [C3] / trace([C3])``:

    - **frobenius_norm**: ``||M||_F^2 = sum(|M_ij|^2)`` ∈ [1/3, 1].
      1/3 for fully random, 1 for a single-scatterer.

    - **scattering_predominance**: ``sqrt(||M||_F^2)`` — single dominant
      scattering mechanism strength.

    - **scattering_diversity**: ``1.5 * (1 - ||M||_F^2)`` — multi-mechanism
      complexity.

    - **degree_purity**: ``2 * sqrt(max(||M||_F^2 - 0.25, 0))`` ∈ [0, 1].

    - **depolarization_index**: ``1 - degree_purity / sqrt(3)`` ∈ [0, 1].

    - **alpha**: ``arccos(clip(Re(M[0,0]), -1, 1))`` [deg] ∈ [0, 90°].

    - **entropy**: determinant-based entropy
      ``2.52 + 0.78 * log|det(M + 0.16 I)| / log(3)``.

    Parameters
    ----------
    window_size : int
        Side length of the boxcar averaging window for [C3].
        Must be odd and >= 3.  Default 7.

    Examples
    --------
    >>> from grdl.image_processing.decomposition import PraksParameters
    >>> praks = PraksParameters(window_size=7)
    >>> comp = praks.decompose(shh, shv, svh, svv)
    >>> print(comp.keys())

    From a pre-computed covariance matrix:

    >>> from grdl.image_processing.decomposition import CovarianceMatrix
    >>> c3 = CovarianceMatrix(window_size=7).compute(channels)
    >>> comp = praks.decompose_from_c3(c3)

    References
    ----------
    Praks, J., Koeniguer, E.C. and Hallikainen, M.T. (2009). "Alternatives to
        target entropy and alpha angle in SAR polarimetry," IEEE Transactions on
        Geoscience and Remote Sensing, 47(7), pp. 2262–2274.
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
            'frobenius_norm',
            'scattering_predominance',
            'scattering_diversity',
            'degree_purity',
            'depolarization_index',
            'alpha',
            'entropy',
        )

    def decompose(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: np.ndarray,
        svv: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Decompose quad-pol data into Praks polarimetric parameters.

        Parameters
        ----------
        shh, shv, svh, svv : np.ndarray
            Complex scattering matrix channels, shape ``(rows, cols)``.

        Returns
        -------
        Dict[str, np.ndarray]
            Seven real float64 arrays; see class docstring for keys.
        """
        self._validate_scattering_matrix(shh, shv, svh, svv)
        self._validate_internal_matrix_window_size('decompose_from_c3')
        channels = np.stack([shh, shv, svh, svv], axis=0)
        c3 = CovarianceMatrix(window_size=self.window_size).compute(channels)
        return self.decompose_from_c3(c3)

    def decompose_from_c3(self, c3: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute Praks parameters from a pre-computed covariance matrix [C3].

        Parameters
        ----------
        c3 : np.ndarray
            Shape ``(3, 3, rows, cols)``, complex.

        Returns
        -------
        Dict[str, np.ndarray]
            Seven real float64 arrays.
        """
        if c3.ndim != 4 or c3.shape[:2] != (3, 3):
            raise ValueError(
                f"Expected c3 shape (3, 3, rows, cols), got {c3.shape}"
            )
        rows, cols = c3.shape[2], c3.shape[3]

        eps = 1e-12

        # Span (total power): trace of C3, real
        span = (
            np.real(c3[0, 0]) + np.real(c3[1, 1]) + np.real(c3[2, 2])
        )
        safe_span = np.maximum(span, eps)[np.newaxis, np.newaxis, :, :]

        # Span-normalised matrix M = C3 / span  (3, 3, rows, cols)
        M = c3 / safe_span

        # Frobenius norm of M: sum(|M_ij|^2) over i, j
        frobenius_norm = np.sum(np.abs(M) ** 2, axis=(0, 1))  # (rows, cols)

        # Scattering predominance and diversity
        scattering_predominance = np.sqrt(frobenius_norm)
        scattering_diversity = 1.5 * (1.0 - frobenius_norm)

        # Degree of purity and depolarization index
        safe_term = np.maximum(frobenius_norm - 0.25, 0.0)
        degree_purity = 2.0 * np.sqrt(safe_term)
        depolarization_index = 1.0 - degree_purity / np.sqrt(3.0)

        # Alpha angle (degrees): arccos(Re(M[0,0]))
        alpha = np.degrees(
            np.arccos(np.clip(np.real(M[0, 0]), -1.0, 1.0))
        )

        # Entropy: det-based — add 0.16 to each diagonal
        M_adj = M.copy()
        for i in range(3):
            M_adj[i, i] += 0.16  # broadcasts over (rows, cols)
        M_adj_yx = M_adj.transpose(2, 3, 0, 1)  # (rows, cols, 3, 3)
        det = np.linalg.det(M_adj_yx)            # (rows, cols), complex
        entropy = (
            2.52
            + 0.78 * np.log(np.abs(det) + eps) / np.log(3.0)
        )

        logger.debug(
            "PraksParameters: mean frobenius_norm=%.3f, mean alpha=%.1f°",
            float(np.nanmean(frobenius_norm)), float(np.nanmean(alpha)),
        )
        return {
            'frobenius_norm':          frobenius_norm,
            'scattering_predominance': scattering_predominance,
            'scattering_diversity':    scattering_diversity,
            'degree_purity':           degree_purity,
            'depolarization_index':    depolarization_index,
            'alpha':                   alpha,
            'entropy':                 np.real(entropy),
        }

    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        """Create an RGB composite from Praks parameters.

        - **Red**: alpha (angle, 0–90° normalised to [0, 1])
        - **Green**: scattering_diversity (percentile stretched)
        - **Blue**: depolarization_index (percentile stretched)
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

        r = _stretch(components['alpha'], lo=0.0, hi=90.0)
        g = _stretch(components['scattering_diversity'])
        b = _stretch(components['depolarization_index'])
        rgb = np.stack([r, g, b], axis=0)

        meta = ImageMetadata(
            format='PraksRGB',
            rows=rgb.shape[1],
            cols=rgb.shape[2],
            bands=3,
            dtype='float32',
            axis_order='CYX',
            channel_metadata=[
                ChannelMetadata(index=0, name='alpha', role='rgb_red'),
                ChannelMetadata(
                    index=1, name='scattering_diversity', role='rgb_green'
                ),
                ChannelMetadata(
                    index=2, name='depolarization_index', role='rgb_blue'
                ),
            ],
        )
        return rgb, meta
