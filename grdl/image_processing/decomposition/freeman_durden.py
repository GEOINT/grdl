# -*- coding: utf-8 -*-
"""
Freeman-Durden 3-Component Decomposition - Model-based power decomposition
of the 3x3 covariance matrix [C3] into surface, double-bounce, and volume
scattering components.

Given quad-pol (HH, HV, VH, VV) complex SLC channels, builds the spatially
averaged 3x3 covariance matrix [C3] and decomposes the total scattered
power into three physical scattering mechanisms:

- **Surface (Ps)**: Single-bounce Bragg surface scattering (odd-bounce).
- **Double-bounce (Pd)**: Dihedral/corner reflector scattering (even-bounce).
- **Volume (Pv)**: Randomly-oriented dipole cloud (vegetation/canopy).

The volume component is estimated first from the cross-pol power, then
subtracted from [C3]. The residual is partitioned into surface and
double-bounce based on the sign of Re(C13).

References
----------
Freeman, A. and Durden, S.L. (1998), "A three-component scattering model
for polarimetric SAR data," IEEE Trans. Geoscience and Remote Sensing,
36(3), pp.963-973.

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
2026-06-25

Modified
--------
2026-06-25
"""

# Standard library
import logging
from typing import Annotated, Dict, Optional, Tuple, TYPE_CHECKING

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
class FreemanDurden3C(PolarimetricDecomposition):
    """Freeman-Durden 3-component model-based power decomposition.

    Decomposes quad-pol SAR data into three physical scattering power
    components via the covariance matrix [C3]:

    - **surface** (Ps): Single-bounce Bragg surface scattering.
      Dominates for smooth surfaces, bare soil, calm water.

    - **double_bounce** (Pd): Dihedral corner reflector scattering.
      Dominates for buildings, tree trunks over ground, urban areas.

    - **volume** (Pv): Randomly-oriented thin dipole cloud.
      Dominates for vegetation canopy, rough surfaces.

    The algorithm estimates Pv first from cross-pol power
    (``Pv = 8/3 * C22`` after the volume model subtraction), then
    solves for Ps and Pd from the remaining co-pol covariance
    elements. The sign of ``Re(C13)`` determines which mechanism
    dominates the residual.

    Parameters
    ----------
    window_size : int
        Side length of the square boxcar averaging window used to
        estimate the covariance matrix [C3].  Must be odd and >= 3.
        Default 7.  Ignored when using ``decompose_from_c3()``.

    Examples
    --------
    From scattering matrix channels:

    >>> from grdl.image_processing.decomposition import FreemanDurden3C
    >>> fd = FreemanDurden3C(window_size=9)
    >>> components = fd.decompose(shh, shv, svh, svv)
    >>> print(components['surface'].shape)

    From a pre-computed (possibly filtered) covariance matrix:

    >>> from grdl.image_processing.decomposition import CovarianceMatrix
    >>> c3 = CovarianceMatrix(window_size=1).compute(channels)
    >>> # ... apply speckle filter to c3 ...
    >>> components = fd.decompose_from_c3(c3)
    """

    window_size: Annotated[int, Range(min=3, max=31),
                           Desc('Boxcar averaging window size')] = 7

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def component_names(self) -> Tuple[str, ...]:
        """Names of decomposition output components."""
        return ('surface', 'double_bounce', 'volume', 'span')

    def decompose(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: np.ndarray,
        svv: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Decompose quad-pol data into Freeman-Durden power components.

        Builds [C3] internally using boxcar averaging with
        ``window_size``, then performs the 3-component decomposition.

        Parameters
        ----------
        shh, shv, svh, svv : np.ndarray
            Complex scattering matrix channels, each shape ``(rows, cols)``.

        Returns
        -------
        Dict[str, np.ndarray]
            Keys: ``'surface'``, ``'double_bounce'``, ``'volume'``,
            ``'span'``.  All real-valued float64 arrays.
        """
        self._validate_scattering_matrix(shh, shv, svh, svv)

        # Build [C3] via CovarianceMatrix
        channels = np.stack([shh, shv, svh, svv], axis=0)
        cov = CovarianceMatrix(window_size=self.window_size)
        c3_ccyx = cov.compute(channels)  # (3, 3, rows, cols)

        return self.decompose_from_c3(c3_ccyx)

    def decompose_from_c3(
        self,
        c3: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Decompose a pre-computed covariance matrix [C3].

        Use this when you want to apply a different speckle filter
        (e.g. Refined Lee, NL-SAR) to [C3] before decomposition.

        Parameters
        ----------
        c3 : np.ndarray
            Covariance matrix, shape ``(3, 3, rows, cols)``, complex.
            Must be Hermitian (``c3[i,j] == conj(c3[j,i])``).

        Returns
        -------
        Dict[str, np.ndarray]
            Keys: ``'surface'``, ``'double_bounce'``, ``'volume'``,
            ``'span'``.
        """
        if c3.ndim != 4 or c3.shape[0] != 3 or c3.shape[1] != 3:
            raise ValueError(
                f"Expected c3 shape (3, 3, rows, cols), got {c3.shape}"
            )

        rows, cols = c3.shape[2], c3.shape[3]

        return self._freeman_durden_3c(c3, rows, cols)

    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        """Create an RGB composite from Freeman-Durden components.

        Standard convention:

        - **Red**: Double-bounce (Pd)
        - **Green**: Volume (Pv)
        - **Blue**: Surface (Ps)

        Parameters
        ----------
        components : Dict[str, np.ndarray]
            Output of ``decompose()`` or ``decompose_from_c3()``.
        representation : str
            ``'db'`` (default), ``'power'``, or ``'magnitude'``.
        percentile_low : float
            Lower percentile for stretch. Default 2.0.
        percentile_high : float
            Upper percentile for stretch. Default 98.0.

        Returns
        -------
        tuple[np.ndarray, ImageMetadata]
            ``(rgb, metadata)`` — rgb shape ``(3, rows, cols)``, float32.
        """
        from grdl.IO.models.base import ImageMetadata, ChannelMetadata

        required = {'surface', 'double_bounce', 'volume'}
        missing = required - set(components.keys())
        if missing:
            raise ValueError(f"Missing component keys: {missing}")

        # R=Pd, G=Pv, B=Ps
        pd = components['double_bounce']
        pv = components['volume']
        ps = components['surface']

        if representation == 'db':
            with np.errstate(divide='ignore', invalid='ignore'):
                pd = 10.0 * np.log10(np.maximum(pd, 1e-10))
                pv = 10.0 * np.log10(np.maximum(pv, 1e-10))
                ps = 10.0 * np.log10(np.maximum(ps, 1e-10))

        def _stretch(arr):
            lo = np.nanpercentile(arr, percentile_low)
            hi = np.nanpercentile(arr, percentile_high)
            return np.clip((arr - lo) / max(hi - lo, 1e-8), 0.0, 1.0).astype(np.float32)

        r = _stretch(pd)
        g = _stretch(pv)
        b = _stretch(ps)

        rgb = np.stack([r, g, b], axis=0)

        meta = ImageMetadata(
            format='RGB',
            rows=rgb.shape[1],
            cols=rgb.shape[2],
            bands=3,
            dtype='float32',
            axis_order='CYX',
            channel_metadata=[
                ChannelMetadata(index=0, name='double_bounce',
                                role='rgb_red'),
                ChannelMetadata(index=1, name='volume',
                                role='rgb_green'),
                ChannelMetadata(index=2, name='surface',
                                role='rgb_blue'),
            ],
        )
        return rgb, meta

    # ------------------------------------------------------------------
    # Internal algorithm
    # ------------------------------------------------------------------

    @staticmethod
    def _freeman_durden_3c(
        c3: np.ndarray,
        rows: int,
        cols: int,
    ) -> Dict[str, np.ndarray]:
        """Core Freeman-Durden 3-component algorithm.

        Operates on the covariance matrix [C3] in CCYX format.

        The [C3] covariance matrix elements are:
            C11 = <|S_HH|²>
            C13 = <S_HH · S_VV*>
            C22 = <|S_HV|²>  (= <|S_VH|²> under reciprocity)
            C33 = <|S_VV|²>

        Algorithm (Freeman & Durden 1998):
        1. Volume: Fv = (3/2) · C22
           Subtract volume contribution from covariance elements.
        2. If residual C11 or C33 ≤ 0, volume dominates — reassign
           all power to volume.
        3. Data conditioning: ensure |C13|² ≤ C11·C33.
        4. Odd-bounce (Re(C13) ≥ 0): solve for Fd first, then Fs.
           Even-bounce (Re(C13) < 0): solve for Fs first, then Fd.
        5. Compute scattering powers:
           Ps = Fs · (1 + |β|²)
           Pd = Fd · (1 + |α|²)
           Pv = (8/3) · Fv

        Parameters
        ----------
        c3 : np.ndarray
            Shape ``(3, 3, rows, cols)``, complex.

        Returns
        -------
        Dict[str, np.ndarray]
        """
        eps = 1e-10

        # Extract covariance matrix elements
        C11 = np.real(c3[0, 0]).copy()
        C13_re = np.real(c3[0, 2]).copy()
        C13_im = np.imag(c3[0, 2]).copy()
        C22 = np.real(c3[1, 1]).copy()
        C33 = np.real(c3[2, 2]).copy()

        span = C11 + C22 + C33
        span_max = np.nanmax(span)

        # -- Step 1: Volume scattering estimate --
        Fv = 1.5 * C22  # (3/2) * C22

        # Subtract volume contribution from co-pol elements
        C11_r = C11 - Fv
        C33_r = C33 - Fv
        C13_re_r = C13_re - Fv / 3.0

        # -- Step 2: Handle volume-dominated pixels --
        vol_dominated = (C11_r <= eps) | (C33_r <= eps)

        # For volume-dominated pixels, redistribute all power to volume
        Fv_adjusted = Fv.copy()
        Fv_adjusted[vol_dominated] = (
            3.0 * (C11[vol_dominated] + C22[vol_dominated]
                   + C33[vol_dominated] + 2.0 * Fv[vol_dominated])
            / 8.0
        )

        # -- Step 3: Data conditioning --
        # Ensure |C13|² ≤ C11·C33 (physical realizability)
        rtemp = C13_re_r**2 + C13_im**2
        product = C11_r * C33_r
        non_realizable = rtemp > product
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.where(
                non_realizable & (rtemp > eps),
                np.sqrt(np.maximum(product, 0.0) / rtemp),
                1.0,
            )
        C13_re_r *= scale
        C13_im *= scale

        # -- Step 4: Surface / double-bounce decomposition --
        Fd = np.zeros((rows, cols), dtype=np.float64)
        Fs = np.zeros((rows, cols), dtype=np.float64)
        alpha_coeff = np.zeros((rows, cols), dtype=np.float64)
        beta_coeff = np.zeros((rows, cols), dtype=np.float64)

        # Odd-bounce dominant: Re(C13) ≥ 0
        odd = (C13_re_r >= 0.0) & ~vol_dominated
        if np.any(odd):
            c11o = C11_r[odd]
            c33o = C33_r[odd]
            c13ro = C13_re_r[odd]
            c13io = C13_im[odd]

            denom = c11o + c33o + 2.0 * c13ro
            Fd[odd] = (c11o * c33o - c13ro**2 - c13io**2) / np.maximum(denom, eps)
            Fs[odd] = c33o - Fd[odd]
            alpha_coeff[odd] = -1.0
            fs_safe = np.maximum(Fs[odd], eps)
            beta_coeff[odd] = np.sqrt(
                (Fd[odd] + c13ro)**2 + c13io**2
            ) / fs_safe

        # Even-bounce dominant: Re(C13) < 0
        even = (C13_re_r < 0.0) & ~vol_dominated
        if np.any(even):
            c11e = C11_r[even]
            c33e = C33_r[even]
            c13re = C13_re_r[even]
            c13ie = C13_im[even]

            denom = c11e + c33e - 2.0 * c13re
            Fs[even] = (c11e * c33e - c13re**2 - c13ie**2) / np.maximum(denom, eps)
            Fd[even] = c33e - Fs[even]
            beta_coeff[even] = 1.0
            fd_safe = np.maximum(Fd[even], eps)
            alpha_coeff[even] = np.sqrt(
                (Fs[even] - c13re)**2 + c13ie**2
            ) / fd_safe

        # -- Step 5: Compute scattering powers --
        Ps = Fs * (1.0 + beta_coeff**2)
        Pd = Fd * (1.0 + alpha_coeff**2)
        Pv = (8.0 / 3.0) * Fv_adjusted

        # Clip to valid range
        np.clip(Ps, 0.0, span_max, out=Ps)
        np.clip(Pd, 0.0, span_max, out=Pd)
        np.clip(Pv, 0.0, span_max, out=Pv)

        # Set all-zero pixels to NaN (no valid scattering)
        zero_mask = (Ps == 0.0) & (Pd == 0.0) & (Pv == 0.0)
        Ps[zero_mask] = np.nan
        Pd[zero_mask] = np.nan
        Pv[zero_mask] = np.nan

        return {
            'surface': Ps,
            'double_bounce': Pd,
            'volume': Pv,
            'span': span,
        }
