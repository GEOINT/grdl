# -*- coding: utf-8 -*-
"""
Yamaguchi 4-Component Scattering Power Decomposition (Full-Pol).

Implements the Yamaguchi et al. (2005, 2011) four-component decomposition
of the 3x3 coherency matrix [T3], separating total backscattered power into:

    surface        (Ps) — odd-bounce Bragg surface scattering
    double_bounce  (Pd) — even-bounce dihedral/corner reflector
    volume         (Pv) — randomly-oriented dipole volume
    helix          (Pc) — helix/circularly-polarised component

Three model variants are supported via the ``model`` parameter:

    'y4o' — original Yamaguchi 4-component (Y4O)
    'y4r' — rotation-corrected Yamaguchi (Y4R): applies a unitary
             rotation to minimise T23 before decomposing
    'y4s' — extended volume scattering model (Y4S): Y4R with an
             additional test to select the volume model type

The per-pixel decision logic is JIT-compiled with numba when available
(``pip install grdl[polsar]``), otherwise falls back to a pure Python loop
with a runtime warning.

References
----------
Yamaguchi, Y., Moriyama, T., Ishido, M., and Yamada, H. (2005). "Four-
    component scattering model for polarimetric SAR image decomposition,"
    IEEE Transactions on Geoscience and Remote Sensing, 43(8), pp. 1699–1706.
Yamaguchi, Y., Sato, A., Boerner, W.M., Sato, R., and Yamada, H. (2011).
    "Four-component scattering power decomposition with rotation of coherency
    matrix," IEEE Transactions on Geoscience and Remote Sensing, 49(6),
    pp. 2251–2258.

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
import warnings
from typing import Annotated, Dict, Tuple, TYPE_CHECKING

# Third-party
import numpy as np

# Optional: numba for JIT-compiled pixel loop
try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

# GRDL internal
from grdl.image_processing.decomposition.base import PolarimetricDecomposition
from grdl.image_processing.decomposition.pol_matrix import (
    CoherencyMatrix,
    CovarianceMatrix,
)
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.image_processing.params import Options, Range, Desc
from grdl.vocabulary import ImageModality

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata

logger = logging.getLogger(__name__)

# Model integer codes used inside the JIT kernel
_MODEL_Y4O = 0
_MODEL_Y4R = 1
_MODEL_Y4S = 2

# ---------------------------------------------------------------------------
# Numba JIT (or plain Python fallback) pixel-level kernel
# ---------------------------------------------------------------------------

def _yam4c_pixel_impl(
    t11r: float,
    t12r: float, t12i: float,
    t13r: float, t13i: float,
    t22r: float,
    t23r: float, t23i: float,
    t33r: float,
    model_int: int,
    span_min: float,
    span_max: float,
) -> tuple:
    """Yamaguchi 4-component decomposition for a single pixel.

    Accepts real-valued components of the T3 matrix elements so the
    function signature is compatible with numba.njit.

    Parameters
    ----------
    t11r, t22r, t33r : float
        Real diagonal T3 elements.
    t12r, t12i, t13r, t13i, t23r, t23i : float
        Real and imaginary parts of the off-diagonal T3 elements.
    model_int : int
        0 = Y4O, 1 = Y4R, 2 = Y4S.
    span_min, span_max : float
        Span limits for clipping output powers.

    Returns
    -------
    (odd, dbl, vol, hlx) : tuple of float
        Surface, double-bounce, volume, and helix power for this pixel.
    """
    eps = 1e-6

    # --- Optional unitary rotation (Y4R / Y4S) ---
    if model_int >= _MODEL_Y4R:
        denom = t22r - t33r
        if abs(denom) < eps:
            teta = 0.0
        else:
            teta = 0.5 * np.arctan(2.0 * t23r / denom)
        ct = np.cos(teta)
        st = np.sin(teta)
        c2 = ct * ct
        s2 = st * st
        cs = ct * st

        # Rotated T12, T13, T23 (real and imaginary), T22, T33
        t12r_new = t12r * ct + t13r * st
        t12i_new = t12i * ct + t13i * st
        t13r_new = -t12r * st + t13r * ct
        t13i_new = -t12i * st + t13i * ct
        t22_new  = t22r * c2 + 2.0 * t23r * cs + t33r * s2
        t23r_new = (
            -t22r * cs + t23r * (c2 - s2) + t33r * cs
        )
        t23i_new = t23i  # imaginary part of T23 is unchanged
        t33_new  = t22r * s2 + t33r * c2 - 2.0 * t23r * cs

        t12r = t12r_new; t12i = t12i_new
        t13r = t13r_new; t13i = t13i_new
        t22r = t22_new
        t23r = t23r_new; t23i = t23i_new
        t33r = t33_new

    # Helix component
    Pc = 2.0 * abs(t23i)

    # Total power
    TP = t11r + t22r + t33r

    # Determine volume scattering model (Y4O/Y4R: HV_type = 1)
    # Y4S: test whether to use HV_type 2 (double-bounce dominated volume)
    HV_type = 1
    if model_int == _MODEL_Y4S:
        C1 = t11r - t22r + (7.0 / 8.0) * t33r + (Pc / 16.0)
        if C1 <= 0.0:
            HV_type = 2

    # Volume power estimate
    if HV_type == 1:
        ratio_num = t11r + t22r - 2.0 * t12r
        ratio_den = t11r + t22r + 2.0 * t12r
        if ratio_den < eps:
            ratio = 0.0
        else:
            r = ratio_num / ratio_den
            ratio = 10.0 * np.log10(max(r, eps))

        two_t33_pc = 2.0 * t33r - Pc
        if -2.0 < ratio <= 2.0:
            Pv = 2.0 * two_t33_pc
        else:
            Pv = (15.0 / 8.0) * two_t33_pc
    else:  # HV_type == 2 (Y4S only)
        Pv = (15.0 / 16.0) * (2.0 * t33r - Pc)

    # --- 3-component Freeman fallback when Pv < 0 ---
    if Pv < 0.0:
        # Re-derive in the lexicographic covariance (Freeman & Durden basis)
        HHHH   = (t11r + 2.0 * t12r + t22r) / 2.0
        HHVVre = (t11r - t22r) / 2.0
        HHVVim = -t12i
        HVHV   = t33r / 2.0
        VVVV   = (t11r - 2.0 * t12r + t22r) / 2.0

        ratio2_num = VVVV
        ratio2_den = HHHH
        if ratio2_den < eps:
            ratio2 = 0.0
        else:
            ratio2 = 10.0 * np.log10(max(VVVV / HHHH, eps))

        if ratio2 <= -2.0:
            FV = 15.0 * HVHV / 4.0
            HHHH   -= 8.0 * FV / 15.0
            VVVV   -= 3.0 * FV / 15.0
            HHVVre -= 2.0 * FV / 15.0
        elif ratio2 > 2.0:
            FV = 15.0 * HVHV / 4.0
            HHHH   -= 3.0 * FV / 15.0
            VVVV   -= 8.0 * FV / 15.0
            HHVVre -= 2.0 * FV / 15.0
        else:
            FV = 8.0 * HVHV / 2.0
            HHHH   -= 3.0 * FV / 8.0
            VVVV   -= 3.0 * FV / 8.0
            HHVVre -= 1.0 * FV / 8.0

        if HHHH <= eps or VVVV <= eps:
            # Volume dominates everything
            if -2.0 < ratio2 <= 2.0:
                FV_full = (
                    (HHHH + 3.0 * FV / 8.0)
                    + HVHV
                    + (VVVV + 3.0 * FV / 8.0)
                )
            elif ratio2 <= -2.0:
                FV_full = (
                    (HHHH + 8.0 * FV / 15.0)
                    + HVHV
                    + (VVVV + 3.0 * FV / 15.0)
                )
            else:
                FV_full = (
                    (HHHH + 3.0 * FV / 15.0)
                    + HVHV
                    + (VVVV + 8.0 * FV / 15.0)
                )
            odd = max(min(FV_full, span_max), span_min)
            return odd, span_min, span_min, span_min

        # Data conditioning: |C13|^2 <= C11*C33
        rtemp = HHVVre * HHVVre + HHVVim * HHVVim
        if rtemp > HHHH * VVVV:
            scale = np.sqrt(HHHH * VVVV / max(rtemp, eps))
            HHVVre *= scale
            HHVVim *= scale

        FD = 0.0; FS = 0.0
        BETre = 0.0; BETim = 0.0
        ALPre = 0.0; ALPim = 0.0

        if HHVVre >= 0.0:  # odd-bounce dominant
            ALPre = -1.0
            FD = (
                HHHH * VVVV - HHVVre * HHVVre - HHVVim * HHVVim
            ) / (HHHH + VVVV + 2.0 * HHVVre)
            FS = VVVV - FD
            if abs(FS) > eps:
                BETre = (FD + HHVVre) / FS
                BETim = HHVVim / FS
        else:               # even-bounce dominant
            BETre = 1.0
            FS = (
                HHHH * VVVV - HHVVre * HHVVre - HHVVim * HHVVim
            ) / (HHHH + VVVV - 2.0 * HHVVre)
            FD = VVVV - FS
            if abs(FD) > eps:
                ALPre = (HHVVre - FS) / FD
                ALPim = HHVVim / FD

        Ps = FS * (1.0 + BETre * BETre + BETim * BETim)
        Pd = FD * (1.0 + ALPre * ALPre + ALPim * ALPim)

        odd = max(min(max(Ps, 0.0), span_max), span_min)
        dbl = max(min(max(Pd, 0.0), span_max), span_min)
        vol = max(min(max(FV, 0.0), span_max), span_min)
        return odd, dbl, vol, span_min  # no helix

    # --- 4-component Yamaguchi ---
    # Residual C parameters (T3 → effective C terms)
    Cre = t12r + t13r
    Cim = t12i + t13i

    if HV_type == 1:
        S = t11r - Pv / 2.0
        D = TP - Pv - Pc - S
        if -2.0 < ratio <= 2.0:
            pass  # Cre already computed
        elif ratio <= -2.0:
            Cre -= Pv / 6.0
        else:
            Cre += Pv / 6.0

        if (Pv + Pc) > TP:
            Ps = 0.0
            Pd = 0.0
            Pv = TP - Pc
        else:
            CO = 2.0 * t11r + Pc - TP
            if CO > 0.0:
                denom_ps = S if abs(S) > eps else eps
                Ps = S + (Cre * Cre + Cim * Cim) / denom_ps
                Pd = D - (Cre * Cre + Cim * Cim) / denom_ps
            else:
                denom_pd = D if abs(D) > eps else eps
                Pd = D + (Cre * Cre + Cim * Cim) / denom_pd
                Ps = S - (Cre * Cre + Cim * Cim) / denom_pd

            if Ps < 0.0:
                if Pd < 0.0:
                    Ps = 0.0; Pd = 0.0
                    Pv = TP - Pc
                else:
                    Ps = 0.0
                    Pd = TP - Pv - Pc
            elif Pd < 0.0:
                Pd = 0.0
                Ps = TP - Pv - Pc

    else:  # HV_type == 2  (Y4S only)
        S = t11r
        D = TP - Pv - Pc - S
        denom_pd = D if abs(D) > eps else eps
        Pd = D + (Cre * Cre + Cim * Cim) / denom_pd
        Ps = S - (Cre * Cre + Cim * Cim) / denom_pd
        if Ps < 0.0:
            if Pd < 0.0:
                Ps = 0.0; Pd = 0.0
                Pv = TP - Pc
            else:
                Ps = 0.0
                Pd = TP - Pv - Pc
        elif Pd < 0.0:
            Pd = 0.0
            Ps = TP - Pv - Pc

    # Non-negativity and span clipping
    Ps = max(Ps, 0.0); Pd = max(Pd, 0.0)
    Pv = max(Pv, 0.0); Pc_out = max(Pc, 0.0)

    odd = max(min(Ps, span_max), span_min)
    dbl = max(min(Pd, span_max), span_min)
    vol = max(min(Pv, span_max), span_min)
    hlx = max(min(Pc_out, span_max), span_min)
    return odd, dbl, vol, hlx


# Build the JIT-compiled (or fallback) batch function at import time.
# The batch function fills pre-allocated output arrays using prange
# (numba) or range (fallback).

if _HAS_NUMBA:
    _yam4c_pixel_nb = numba.njit(_yam4c_pixel_impl)

    @numba.njit(parallel=True)
    def _yam4c_batch(
        t11r, t12r, t12i, t13r, t13i, t22r, t23r, t23i, t33r,
        model_int, span_min, span_max,
        odd_out, dbl_out, vol_out, hlx_out,
    ):
        N = t11r.shape[0]
        for i in numba.prange(N):
            o, d, v, h = _yam4c_pixel_nb(
                t11r[i], t12r[i], t12i[i],
                t13r[i], t13i[i], t22r[i],
                t23r[i], t23i[i], t33r[i],
                model_int, span_min, span_max,
            )
            odd_out[i] = o
            dbl_out[i] = d
            vol_out[i] = v
            hlx_out[i] = h

else:
    def _yam4c_batch(
        t11r, t12r, t12i, t13r, t13i, t22r, t23r, t23i, t33r,
        model_int, span_min, span_max,
        odd_out, dbl_out, vol_out, hlx_out,
    ):
        N = t11r.shape[0]
        for i in range(N):
            o, d, v, h = _yam4c_pixel_impl(
                t11r[i], t12r[i], t12i[i],
                t13r[i], t13i[i], t22r[i],
                t23r[i], t23i[i], t33r[i],
                model_int, span_min, span_max,
            )
            odd_out[i] = o
            dbl_out[i] = d
            vol_out[i] = v
            hlx_out[i] = h


# ---------------------------------------------------------------------------
# T3 → C3 conversion helper (D = 1/√2 * [[1,0,1],[1,0,-1],[0,√2,0]])
# ---------------------------------------------------------------------------

def _t3_to_c3(t3: np.ndarray) -> np.ndarray:
    """Convert T3 (3,3,rows,cols) coherency matrix to C3 covariance matrix."""
    D = (1.0 / np.sqrt(2.0)) * np.array(
        [[1, 0, 1], [1, 0, -1], [0, np.sqrt(2), 0]], dtype=np.complex128
    )
    # T3 in (rows,cols,3,3) for broadcasting
    t3_yx = t3.transpose(2, 3, 0, 1)           # (rows, cols, 3, 3)
    c3_yx = D.T @ t3_yx @ D                     # (rows, cols, 3, 3)
    return c3_yx.transpose(2, 3, 0, 1)          # (3, 3, rows, cols)


# ---------------------------------------------------------------------------
# Decomposition class
# ---------------------------------------------------------------------------

@processor_version('1.0.0')
@processor_tags(modalities=[ImageModality.SAR])
class Yamaguchi4C(PolarimetricDecomposition):
    """Yamaguchi four-component scattering power decomposition.

    Separates total backscattered power into four physical components:

    - **surface** (Ps): single-bounce Bragg surface scattering.
    - **double_bounce** (Pd): dihedral corner-reflector scattering.
    - **volume** (Pv): randomly-oriented dipole cloud (vegetation).
    - **helix** (Pc): circularly-polarised helix component.

    Three model variants are available via ``model``:

    - ``'y4o'``: Original Yamaguchi (2005).  No rotation.
    - ``'y4r'``: Rotation-corrected Yamaguchi (2011).  A unitary rotation
      minimising T23 is applied before decomposition, improving performance
      over oriented targets.
    - ``'y4s'``: Extended volume model (Y4S): applies rotation and selects
      the volume model type (surface or double-bounce dominated) based on
      an additional T3 element test.

    Parameters
    ----------
    window_size : int
        Boxcar averaging window side length (must be odd, >= 3).  Default 7.
    model : str
        One of ``'y4o'``, ``'y4r'``, ``'y4s'``.  Default ``'y4o'``.

    Notes
    -----
    The per-pixel decision logic is JIT-compiled with numba when
    ``grdl[polsar]`` is installed; otherwise a pure Python loop is used
    (expect ~10–50× slower on large arrays).

    Examples
    --------
    >>> from grdl.image_processing.decomposition import Yamaguchi4C
    >>> yam = Yamaguchi4C(window_size=7, model='y4r')
    >>> comp = yam.decompose(shh, shv, svh, svv)
    >>> print(comp['surface'].shape)

    From a pre-computed T3 matrix:

    >>> t3 = CoherencyMatrix(window_size=7).compute(channels)
    >>> comp = yam.decompose_from_t3(t3)

    References
    ----------
    Yamaguchi, Y., Moriyama, T., Ishido, M., and Yamada, H. (2005). "Four-
        component scattering model for polarimetric SAR image decomposition,"
        IEEE Transactions on Geoscience and Remote Sensing, 43(8), pp. 1699–1706.
    Yamaguchi, Y., Sato, A., Boerner, W.M., Sato, R., and Yamada, H. (2011).
        "Four-component scattering power decomposition with rotation of coherency
        matrix," IEEE Transactions on Geoscience and Remote Sensing, 49(6),
        pp. 2251–2258.
    """

    __gpu_compatible__ = False

    window_size: Annotated[int, Range(min=1, max=31),
                           Desc('Boxcar averaging window size')] = 7
    model: Annotated[str, Options('y4o', 'y4r', 'y4s'),
                     Desc("Decomposition variant: 'y4o' (original), "
                          "'y4r' (rotation-corrected), 'y4s' (extended volume)"
                          )] = 'y4o'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def component_names(self) -> Tuple[str, ...]:
        return ('surface', 'double_bounce', 'volume', 'helix', 'span')

    def decompose(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: np.ndarray,
        svv: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Decompose quad-pol data into Yamaguchi 4-component powers.

        Parameters
        ----------
        shh, shv, svh, svv : np.ndarray
            Complex scattering matrix channels, shape ``(rows, cols)``.

        Returns
        -------
        Dict[str, np.ndarray]
            Keys: ``'surface'``, ``'double_bounce'``, ``'volume'``,
            ``'helix'``, ``'span'``.  All real float64.
        """
        self._validate_scattering_matrix(shh, shv, svh, svv)
        self._validate_internal_matrix_window_size('decompose_from_t3')
        channels = np.stack([shh, shv, svh, svv], axis=0)
        t3 = CoherencyMatrix(window_size=self.window_size).compute(channels)
        return self.decompose_from_t3(t3)

    def decompose_from_t3(self, t3: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute Yamaguchi powers from a pre-computed [T3].

        Parameters
        ----------
        t3 : np.ndarray
            Shape ``(3, 3, rows, cols)``, complex.

        Returns
        -------
        Dict[str, np.ndarray]
            Keys: ``'surface'``, ``'double_bounce'``, ``'volume'``,
            ``'helix'``, ``'span'``.
        """
        if t3.ndim != 4 or t3.shape[:2] != (3, 3):
            raise ValueError(
                f"Expected t3 shape (3, 3, rows, cols), got {t3.shape}"
            )
        rows, cols = t3.shape[2], t3.shape[3]

        # Span from T3 diagonal
        span_2d = (
            np.real(t3[0, 0]) + np.real(t3[1, 1]) + np.real(t3[2, 2])
        )
        span_max = float(np.nanmax(span_2d))
        span_min = 0.0  # component powers are non-negative by construction

        if not _HAS_NUMBA:
            warnings.warn(
                "numba is not installed; Yamaguchi4C will use a slow Python "
                "pixel loop.  Install grdl[polsar] for JIT-accelerated "
                "performance.",
                stacklevel=3,
            )

        # Map model name to integer code
        model_map = {'y4o': _MODEL_Y4O, 'y4r': _MODEL_Y4R, 'y4s': _MODEL_Y4S}
        model_int = model_map.get(self.model, _MODEL_Y4O)

        # Flatten T3 to (N,) per-element arrays (real/imag split for numba)
        t3_yx = t3.transpose(2, 3, 0, 1).reshape(-1, 3, 3)  # (N, 3, 3)
        N = t3_yx.shape[0]

        t11r = np.ascontiguousarray(t3_yx[:, 0, 0].real)
        t12r = np.ascontiguousarray(t3_yx[:, 0, 1].real)
        t12i = np.ascontiguousarray(t3_yx[:, 0, 1].imag)
        t13r = np.ascontiguousarray(t3_yx[:, 0, 2].real)
        t13i = np.ascontiguousarray(t3_yx[:, 0, 2].imag)
        t22r = np.ascontiguousarray(t3_yx[:, 1, 1].real)
        t23r = np.ascontiguousarray(t3_yx[:, 1, 2].real)
        t23i = np.ascontiguousarray(t3_yx[:, 1, 2].imag)
        t33r = np.ascontiguousarray(t3_yx[:, 2, 2].real)

        odd_out = np.empty(N, dtype=np.float64)
        dbl_out = np.empty(N, dtype=np.float64)
        vol_out = np.empty(N, dtype=np.float64)
        hlx_out = np.empty(N, dtype=np.float64)

        _yam4c_batch(
            t11r, t12r, t12i, t13r, t13i, t22r, t23r, t23i, t33r,
            model_int, span_min, span_max,
            odd_out, dbl_out, vol_out, hlx_out,
        )

        logger.debug(
            "Yamaguchi4C (%s): mean Ps=%.2f, Pd=%.2f, Pv=%.2f, Pc=%.2f",
            self.model,
            float(np.nanmean(odd_out)), float(np.nanmean(dbl_out)),
            float(np.nanmean(vol_out)), float(np.nanmean(hlx_out)),
        )
        return {
            'surface':      odd_out.reshape(rows, cols),
            'double_bounce': dbl_out.reshape(rows, cols),
            'volume':        vol_out.reshape(rows, cols),
            'helix':         hlx_out.reshape(rows, cols),
            'span':          span_2d,
        }

    def decompose_from_c3(self, c3: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute Yamaguchi powers from a pre-computed covariance matrix [C3].

        Converts [C3] → [T3] internally then calls ``decompose_from_t3``.

        Parameters
        ----------
        c3 : np.ndarray
            Shape ``(3, 3, rows, cols)``, complex.

        Returns
        -------
        Dict[str, np.ndarray]
        """
        if c3.ndim != 4 or c3.shape[:2] != (3, 3):
            raise ValueError(
                f"Expected c3 shape (3, 3, rows, cols), got {c3.shape}"
            )
        D = (1.0 / np.sqrt(2.0)) * np.array(
            [[1, 0, 1], [1, 0, -1], [0, np.sqrt(2), 0]], dtype=np.complex128
        )
        c3_yx = c3.transpose(2, 3, 0, 1)       # (rows, cols, 3, 3)
        t3_yx = D @ c3_yx @ D.T                 # (rows, cols, 3, 3)
        t3 = t3_yx.transpose(2, 3, 0, 1)        # (3, 3, rows, cols)
        return self.decompose_from_t3(t3)

    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        """Create an RGB composite from Yamaguchi powers.

        Standard convention (same as Freeman-Durden):

        - **Red**: double_bounce (Pd)
        - **Green**: volume (Pv)
        - **Blue**: surface (Ps)
        """
        from grdl.IO.models.base import ImageMetadata, ChannelMetadata

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
            return np.clip(
                (arr - lo) / max(hi - lo, 1e-8), 0.0, 1.0
            ).astype(np.float32)

        rgb = np.stack([_stretch(pd), _stretch(pv), _stretch(ps)], axis=0)

        meta = ImageMetadata(
            format='Yamaguchi4CRGB',
            rows=rgb.shape[1],
            cols=rgb.shape[2],
            bands=3,
            dtype='float32',
            axis_order='CYX',
            channel_metadata=[
                ChannelMetadata(index=0, name='double_bounce', role='rgb_red'),
                ChannelMetadata(index=1, name='volume',        role='rgb_green'),
                ChannelMetadata(index=2, name='surface',       role='rgb_blue'),
            ],
        )
        return rgb, meta
