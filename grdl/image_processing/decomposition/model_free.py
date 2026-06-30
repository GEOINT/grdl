# -*- coding: utf-8 -*-
"""
Model-Free Scattering Power Decomposition - MF3CF and MF4CF.

Implements the Dey et al. model-free 3- and 4-component scattering
power decompositions operating on the 3x3 coherency matrix [T3].

Unlike model-based methods (Freeman-Durden), these decompositions use
the scattering type parameter θ_FP (and helicity τ_FP for 4-component)
derived directly from T3 elements without assuming specific scattering
models.

References
----------
Dey, S., Bhattacharya, A., Ratha, D., Mandal, D., and Frery, A.C.
(2020), "Target characterization and scattering power decomposition
for full and compact polarimetric SAR data," IEEE Geoscience and
Remote Sensing Letters, 18(6), pp.1048-1052.

Dey, S., Bhattacharya, A., Ratha, D., Mandal, D., McNairn, H.,
Lopez-Sanchez, J.M., and Rao, Y.S. (2021), "Model-free four component
scattering power decomposition for polarimetric SAR data," IEEE Journal
of Selected Topics in Applied Earth Observations and Remote Sensing, 14,
pp.3887-3898.

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
2026-06-29

Modified
--------
2026-06-29
"""

# Standard library
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


# ======================================================================
# MF3CF — Model-Free 3-Component Full-Pol Decomposition
# ======================================================================

@processor_version('1.0.0')
@processor_tags(modalities=[ImageModality.SAR])
class ModelFree3C(PolarimetricDecomposition):
    """Model-free 3-component scattering power decomposition (MF3CF).

    Decomposes quad-pol SAR data into three scattering power components
    using the coherency matrix [T3] without assuming specific scattering
    models. Uses the scattering type parameter θ_FP to partition total
    power into surface, double-bounce, and volume components.

    - **surface** (Ps): Odd-bounce / surface scattering.
    - **double_bounce** (Pd): Even-bounce / dihedral scattering.
    - **volume** (Pv): Diffuse / volume scattering.

    The scattering type parameter θ_FP ∈ [-45°, +45°]:
    - θ_FP ≈ -45°: dominant odd-bounce (surface)
    - θ_FP ≈ 0°: dominant volume (diffuse)
    - θ_FP ≈ +45°: dominant even-bounce (dihedral)

    Parameters
    ----------
    window_size : int
        Side length of the square boxcar averaging window used to
        estimate the coherency matrix [T3].  Must be odd and >= 3.
        Default 7.  Ignored when using ``decompose_from_t3()``.

    Examples
    --------
    From scattering matrix channels:

    >>> from grdl.image_processing.decomposition import ModelFree3C
    >>> mf3 = ModelFree3C(window_size=7)
    >>> components = mf3.decompose(shh, shv, svh, svv)
    >>> print(components['surface'].shape)

    From a pre-computed (possibly filtered) coherency matrix:

    >>> from grdl.image_processing.decomposition import CoherencyMatrix
    >>> t3 = CoherencyMatrix(window_size=1).compute(channels)
    >>> # ... apply speckle filter to t3 ...
    >>> components = mf3.decompose_from_t3(t3)
    """

    window_size: Annotated[int, Range(min=3, max=31),
                           Desc('Boxcar averaging window size')] = 7

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def component_names(self) -> Tuple[str, ...]:
        """Names of decomposition output components."""
        return ('surface', 'double_bounce', 'volume', 'span', 'theta_fp')

    def decompose(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: np.ndarray,
        svv: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Decompose quad-pol data into MF3CF power components.

        Builds [T3] internally using boxcar averaging with
        ``window_size``, then performs the 3-component decomposition.

        Parameters
        ----------
        shh, shv, svh, svv : np.ndarray
            Complex scattering matrix channels, each shape ``(rows, cols)``.

        Returns
        -------
        Dict[str, np.ndarray]
            Keys: ``'surface'``, ``'double_bounce'``, ``'volume'``,
            ``'span'``, ``'theta_fp'``.
            All real-valued float64 arrays.
        """
        self._validate_scattering_matrix(shh, shv, svh, svv)

        # Build [T3] via CoherencyMatrix
        channels = np.stack([shh, shv, svh, svv], axis=0)
        coh = CoherencyMatrix(window_size=self.window_size)
        t3 = coh.compute(channels)  # (3, 3, rows, cols)

        return self.decompose_from_t3(t3)

    def decompose_from_t3(
        self,
        t3: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Decompose a pre-computed coherency matrix [T3].

        Use this when you want to apply a different speckle filter
        (e.g. Refined Lee, NL-SAR) to [T3] before decomposition.

        Parameters
        ----------
        t3 : np.ndarray
            Coherency matrix, shape ``(3, 3, rows, cols)``, complex.
            Must be Hermitian (``t3[i,j] == conj(t3[j,i])``).

        Returns
        -------
        Dict[str, np.ndarray]
            Keys: ``'surface'``, ``'double_bounce'``, ``'volume'``,
            ``'span'``, ``'theta_fp'``.
        """
        if t3.ndim != 4 or t3.shape[0] != 3 or t3.shape[1] != 3:
            raise ValueError(
                f"Expected t3 shape (3, 3, rows, cols), got {t3.shape}"
            )

        return self._mf3cf(t3)

    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        """Create an RGB composite from MF3CF components.

        Standard convention (same as Freeman-Durden):

        - **Red**: Double-bounce (Pd)
        - **Green**: Volume (Pv)
        - **Blue**: Surface (Ps)

        Parameters
        ----------
        components : Dict[str, np.ndarray]
            Output of ``decompose()`` or ``decompose_from_t3()``.
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

        pd = components['double_bounce'].copy()
        pv = components['volume'].copy()
        ps = components['surface'].copy()

        if representation == 'db':
            with np.errstate(divide='ignore', invalid='ignore'):
                pd = 10.0 * np.log10(np.maximum(pd, 1e-10))
                pv = 10.0 * np.log10(np.maximum(pv, 1e-10))
                ps = 10.0 * np.log10(np.maximum(ps, 1e-10))

        def _stretch(arr):
            lo = np.nanpercentile(arr, percentile_low)
            hi = np.nanpercentile(arr, percentile_high)
            return np.clip((arr - lo) / max(hi - lo, 1e-8), 0.0, 1.0).astype(np.float32)

        rgb = np.stack([_stretch(pd), _stretch(pv), _stretch(ps)], axis=0)

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

    def __repr__(self) -> str:
        return f"ModelFree3C(window_size={self.window_size})"

    # ------------------------------------------------------------------
    # Internal algorithm
    # ------------------------------------------------------------------

    @staticmethod
    def _mf3cf(t3: np.ndarray) -> Dict[str, np.ndarray]:
        """Core MF3CF algorithm (Dey et al. 2020).

        The scattering type parameter:
            θ_FP = 2 · arctan((T22 + T33 - T11) / (T11 - T22 + T33))

        Power partition using θ_FP:
            Span = T11 + T22 + T33
            Pv = Span · (1 - |3·cos(θ_FP) - 1| / 2)  — diffuse
            Remainder = Span - Pv
            If θ_FP > 0: Pd = Remainder · sin²(θ_FP), Ps = Remainder · cos²(θ_FP)
            If θ_FP < 0: Ps = Remainder · sin²(θ_FP), Pd = Remainder · cos²(θ_FP)
            If θ_FP = 0: Ps = Pd = Remainder / 2

        Refined power partition (from polsartools/Dey et al.):
            m = DoP (degree of polarization from T3)
            Pv = Span · (1 - m)  — depolarized component
            Ps = m · Span · cos²(θ_FP) · (1 + tan(θ_FP)) / 2  — odd-bounce
            Pd = m · Span · sin²(θ_FP) · (1 + cot(θ_FP)) / 2  — even-bounce

        Actually implementing the polsartools formulation which uses:
            θ_FP ∈ [-π/4, π/4]
            Ps = (m · Span · (1 + sin2θ)) / 2  when θ_FP < 0
            Pd = (m · Span · (1 - sin2θ)) / 2  when θ_FP > 0
            Pv = Span · (1 - m)
        """
        # Extract T3 diagonal (real) and off-diagonal elements
        T11 = np.real(t3[0, 0])
        T22 = np.real(t3[1, 1])
        T33 = np.real(t3[2, 2])
        T12 = t3[0, 1]
        T13 = t3[0, 2]
        T23 = t3[1, 2]

        span = T11 + T22 + T33

        # Scattering type parameter θ_FP
        # θ_FP = 2 · arctan((T22 + T33 - T11) / (T11 - T22 + T33))
        # Range: [-π/4, π/4] after proper bounding
        numer = T22 + T33 - T11
        denom = T11 - T22 + T33
        with np.errstate(divide='ignore', invalid='ignore'):
            theta_fp = np.arctan2(numer, denom)
        # The full arctan2 gives [-π, π]; the physical range is [-π/4, π/4]
        # but we keep the raw computation and clip to valid range
        np.clip(theta_fp, -np.pi / 4.0, np.pi / 4.0, out=theta_fp)

        # Degree of polarization (DoP) from T3
        # DoP = sqrt(1 - 27·det(T3) / trace(T3)³)
        # Compute via Barakat formula: m² = 1 - 27·|T|/Tr³
        # where |T| is the determinant
        # Using the 3x3 determinant formula
        det_T = (
            T11 * (T22 * T33 - np.abs(T23)**2)
            - T12 * (np.conj(T12) * T33 - T23 * np.conj(T13))
            + T13 * (np.conj(T12) * np.conj(T23) - T22 * np.conj(T13))
        )
        det_T = np.real(det_T)

        trace_cubed = span**3
        with np.errstate(divide='ignore', invalid='ignore'):
            m_sq = 1.0 - 27.0 * det_T / np.where(
                trace_cubed > 0, trace_cubed, 1.0
            )
        # Clamp to [0, 1] for numerical safety
        np.clip(m_sq, 0.0, 1.0, out=m_sq)
        m = np.sqrt(m_sq)

        # Power components (polsartools formulation)
        # sin(2θ) for the partition
        sin2theta = np.sin(2.0 * theta_fp)

        # Volume (depolarized)
        Pv = span * (1.0 - m)

        # Polarized component
        m_span = m * span

        # Surface: dominates when θ_FP < 0
        # Double-bounce: dominates when θ_FP > 0
        # Using symmetric formulation:
        #   Pd = m·Span·(1 - sin2θ)/2   [for θ>0, sin2θ>0, so Pd < m·Span/2]
        #   Ps = m·Span·(1 + sin2θ)/2   [for θ<0, sin2θ<0, so Ps > m·Span/2]
        # Wait — need to match polsartools exactly.
        # From polsartools mf3cf.py:
        #   val = (m * Span * (1 + sin_delta)) / 2
        #   where sin_delta = sin(2*theta)
        #   Pd = val when theta > 0 → Pd = m*S*(1+sin2θ)/2
        #   Ps = m*S - Pd when theta > 0
        #   Ps = val when theta < 0 → Ps = m*S*(1+sin2θ)/2
        #   Pd = m*S - Ps when theta < 0
        # But sin2θ < 0 when θ < 0, so (1+sin2θ)/2 < 0.5

        # Actually from the paper (Dey 2020, eq 13-15):
        #   When θ_FP > 0 (even-bounce dominant):
        #     Pd = m·Span·sin²(θ_FP)·(1 + 1/tan(θ_FP)) / 2
        #     Actually simplified: Pd = m·Span·(1+sin(2θ))/2
        #     Ps = m·Span - Pd
        #   When θ_FP < 0 (odd-bounce dominant):
        #     Ps = m·Span·(1-sin(2θ))/2  [sin(2θ)<0, so 1-sin2θ > 1]
        #     Wait that gives Ps > m·Span which is wrong.
        #
        # Let me use the polsartools implementation directly:
        # From polsartools fp/mf3cf.py process_chunk_mf3cf:
        #   sin_delta = np.sin(2 * theta_FP)
        #   val = (m_FP * Span * (1 + sin_delta)) / 2
        #   Pd = np.where(theta_FP > 0, val, m_FP * Span - val)
        #   Ps = np.where(theta_FP > 0, m_FP * Span - val, val)
        #   Pv = Span * (1 - m_FP)

        val = m_span * (1.0 + sin2theta) / 2.0

        Pd = np.where(theta_fp > 0, val, m_span - val)
        Ps = np.where(theta_fp > 0, m_span - val, val)

        # Clip to non-negative
        np.maximum(Ps, 0.0, out=Ps)
        np.maximum(Pd, 0.0, out=Pd)
        np.maximum(Pv, 0.0, out=Pv)

        # Set zero-span pixels to NaN
        zero_mask = span <= 0.0
        Ps[zero_mask] = np.nan
        Pd[zero_mask] = np.nan
        Pv[zero_mask] = np.nan

        return {
            'surface': Ps,
            'double_bounce': Pd,
            'volume': Pv,
            'span': span,
            'theta_fp': np.degrees(theta_fp),
        }


# ======================================================================
# MF4CF — Model-Free 4-Component Full-Pol Decomposition
# ======================================================================

@processor_version('1.0.0')
@processor_tags(modalities=[ImageModality.SAR])
class ModelFree4C(PolarimetricDecomposition):
    """Model-free 4-component scattering power decomposition (MF4CF).

    Extends MF3CF with a helix scattering component derived from the
    helicity angle τ_FP. Decomposes quad-pol SAR data into four
    scattering power components using the coherency matrix [T3].

    - **surface** (Ps): Odd-bounce / surface scattering.
    - **double_bounce** (Pd): Even-bounce / dihedral scattering.
    - **volume** (Pv): Diffuse / volume scattering.
    - **helix** (Pc): Asymmetric / helix scattering.

    The helicity parameter τ_FP ∈ [-45°, +45°]:
    - |τ_FP| ≈ 0°: negligible helix contribution
    - |τ_FP| ≈ 45°: dominant helix scattering (asymmetric targets)

    Parameters
    ----------
    window_size : int
        Side length of the square boxcar averaging window used to
        estimate the coherency matrix [T3].  Must be odd and >= 3.
        Default 7.  Ignored when using ``decompose_from_t3()``.

    Examples
    --------
    From scattering matrix channels:

    >>> from grdl.image_processing.decomposition import ModelFree4C
    >>> mf4 = ModelFree4C(window_size=7)
    >>> components = mf4.decompose(shh, shv, svh, svv)
    >>> print(components['helix'].shape)

    From a pre-computed (possibly filtered) coherency matrix:

    >>> from grdl.image_processing.decomposition import CoherencyMatrix
    >>> t3 = CoherencyMatrix(window_size=1).compute(channels)
    >>> # ... apply speckle filter to t3 ...
    >>> components = mf4.decompose_from_t3(t3)
    """

    window_size: Annotated[int, Range(min=3, max=31),
                           Desc('Boxcar averaging window size')] = 7

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def component_names(self) -> Tuple[str, ...]:
        """Names of decomposition output components."""
        return ('surface', 'double_bounce', 'volume', 'helix',
                'span', 'theta_fp', 'tau_fp')

    def decompose(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: np.ndarray,
        svv: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Decompose quad-pol data into MF4CF power components.

        Builds [T3] internally using boxcar averaging with
        ``window_size``, then performs the 4-component decomposition.

        Parameters
        ----------
        shh, shv, svh, svv : np.ndarray
            Complex scattering matrix channels, each shape ``(rows, cols)``.

        Returns
        -------
        Dict[str, np.ndarray]
            Keys: ``'surface'``, ``'double_bounce'``, ``'volume'``,
            ``'helix'``, ``'span'``, ``'theta_fp'``, ``'tau_fp'``.
            All real-valued float64 arrays.
        """
        self._validate_scattering_matrix(shh, shv, svh, svv)

        # Build [T3] via CoherencyMatrix
        channels = np.stack([shh, shv, svh, svv], axis=0)
        coh = CoherencyMatrix(window_size=self.window_size)
        t3 = coh.compute(channels)  # (3, 3, rows, cols)

        return self.decompose_from_t3(t3)

    def decompose_from_t3(
        self,
        t3: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Decompose a pre-computed coherency matrix [T3].

        Use this when you want to apply a different speckle filter
        (e.g. Refined Lee, NL-SAR) to [T3] before decomposition.

        Parameters
        ----------
        t3 : np.ndarray
            Coherency matrix, shape ``(3, 3, rows, cols)``, complex.
            Must be Hermitian (``t3[i,j] == conj(t3[j,i])``).

        Returns
        -------
        Dict[str, np.ndarray]
            Keys: ``'surface'``, ``'double_bounce'``, ``'volume'``,
            ``'helix'``, ``'span'``, ``'theta_fp'``, ``'tau_fp'``.
        """
        if t3.ndim != 4 or t3.shape[0] != 3 or t3.shape[1] != 3:
            raise ValueError(
                f"Expected t3 shape (3, 3, rows, cols), got {t3.shape}"
            )

        return self._mf4cf(t3)

    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        """Create an RGB composite from MF4CF components.

        Standard convention (same as Freeman-Durden):

        - **Red**: Double-bounce (Pd)
        - **Green**: Volume (Pv)
        - **Blue**: Surface (Ps)

        The helix component is not shown in the RGB but is available
        in the components dict.

        Parameters
        ----------
        components : Dict[str, np.ndarray]
            Output of ``decompose()`` or ``decompose_from_t3()``.
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

        pd = components['double_bounce'].copy()
        pv = components['volume'].copy()
        ps = components['surface'].copy()

        if representation == 'db':
            with np.errstate(divide='ignore', invalid='ignore'):
                pd = 10.0 * np.log10(np.maximum(pd, 1e-10))
                pv = 10.0 * np.log10(np.maximum(pv, 1e-10))
                ps = 10.0 * np.log10(np.maximum(ps, 1e-10))

        def _stretch(arr):
            lo = np.nanpercentile(arr, percentile_low)
            hi = np.nanpercentile(arr, percentile_high)
            return np.clip((arr - lo) / max(hi - lo, 1e-8), 0.0, 1.0).astype(np.float32)

        rgb = np.stack([_stretch(pd), _stretch(pv), _stretch(ps)], axis=0)

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

    def __repr__(self) -> str:
        return f"ModelFree4C(window_size={self.window_size})"

    # ------------------------------------------------------------------
    # Internal algorithm
    # ------------------------------------------------------------------

    @staticmethod
    def _mf4cf(t3: np.ndarray) -> Dict[str, np.ndarray]:
        """Core MF4CF algorithm (Dey et al. 2021).

        Extends MF3CF with helicity parameter τ_FP:
            τ_FP = arctan(2·Im(T23) / (T22 - T33))

        Power partition:
            Pc = m · Span · sin²(2τ_FP) / 2  — helix
            Remainder partitioned as in MF3CF using θ_FP.

        Parameters
        ----------
        t3 : np.ndarray
            Shape ``(3, 3, rows, cols)``, complex.

        Returns
        -------
        Dict[str, np.ndarray]
        """
        # Extract T3 elements
        T11 = np.real(t3[0, 0])
        T22 = np.real(t3[1, 1])
        T33 = np.real(t3[2, 2])
        T12 = t3[0, 1]
        T13 = t3[0, 2]
        T23 = t3[1, 2]

        span = T11 + T22 + T33

        # --- Scattering type parameter θ_FP ---
        numer_theta = T22 + T33 - T11
        denom_theta = T11 - T22 + T33
        with np.errstate(divide='ignore', invalid='ignore'):
            theta_fp = np.arctan2(numer_theta, denom_theta)
        np.clip(theta_fp, -np.pi / 4.0, np.pi / 4.0, out=theta_fp)

        # --- Helicity parameter τ_FP ---
        # τ_FP = arctan(2·Im(T23) / (T22 - T33))
        numer_tau = 2.0 * np.imag(T23)
        denom_tau = T22 - T33
        with np.errstate(divide='ignore', invalid='ignore'):
            tau_fp = np.arctan2(numer_tau, denom_tau) / 2.0
        # τ_FP range: [-π/4, π/4]
        np.clip(tau_fp, -np.pi / 4.0, np.pi / 4.0, out=tau_fp)

        # --- Degree of polarization (DoP) ---
        det_T = (
            T11 * (T22 * T33 - np.abs(T23)**2)
            - T12 * (np.conj(T12) * T33 - T23 * np.conj(T13))
            + T13 * (np.conj(T12) * np.conj(T23) - T22 * np.conj(T13))
        )
        det_T = np.real(det_T)

        trace_cubed = span**3
        with np.errstate(divide='ignore', invalid='ignore'):
            m_sq = 1.0 - 27.0 * det_T / np.where(
                trace_cubed > 0, trace_cubed, 1.0
            )
        np.clip(m_sq, 0.0, 1.0, out=m_sq)
        m = np.sqrt(m_sq)

        # --- Power components ---
        m_span = m * span
        sin2theta = np.sin(2.0 * theta_fp)

        # Helix power: Pc = m·Span·sin²(2τ) / 2
        # From polsartools mf4cf.py:
        #   Pc = (m_FP * Span * (1 - sin_delta_c)) / 2
        #   where sin_delta_c = cos(2*tau_FP)
        # So: Pc = m·Span·(1 - cos(2τ))/2 = m·Span·sin²(τ)
        # Actually: 1 - cos(2τ) = 2sin²(τ), so Pc = m·Span·sin²(τ)
        # Wait, let me re-derive from polsartools:
        #   sin_delta_c = np.cos(2 * tau_FP)
        #   val_c = (m_FP * Span * (1 - sin_delta_c)) / 2
        #   Pc = val_c
        cos2tau = np.cos(2.0 * tau_fp)
        Pc = m_span * (1.0 - cos2tau) / 2.0

        # Volume (depolarized)
        Pv = span * (1.0 - m)

        # Remaining polarized power after helix subtraction
        # From polsartools: the 3-component partition uses (m_span - Pc)
        # val = (m_FP * Span - Pc) * (1 + sin_delta) / 2
        m_span_residual = m_span - Pc
        val = m_span_residual * (1.0 + sin2theta) / 2.0

        Pd = np.where(theta_fp > 0, val, m_span_residual - val)
        Ps = np.where(theta_fp > 0, m_span_residual - val, val)

        # Clip to non-negative
        np.maximum(Ps, 0.0, out=Ps)
        np.maximum(Pd, 0.0, out=Pd)
        np.maximum(Pv, 0.0, out=Pv)
        np.maximum(Pc, 0.0, out=Pc)

        # Set zero-span pixels to NaN
        zero_mask = span <= 0.0
        Ps[zero_mask] = np.nan
        Pd[zero_mask] = np.nan
        Pv[zero_mask] = np.nan
        Pc[zero_mask] = np.nan

        return {
            'surface': Ps,
            'double_bounce': Pd,
            'volume': Pv,
            'helix': Pc,
            'span': span,
            'theta_fp': np.degrees(theta_fp),
            'tau_fp': np.degrees(tau_fp),
        }
