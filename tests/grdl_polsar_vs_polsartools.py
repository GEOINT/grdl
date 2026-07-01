#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRDL vs polsartools Cross-Validation Script.

Compares GRDL polarimetric decomposition outputs against polsartools
reference implementations (inlined to avoid GDAL dependency). Verifies
numerical equivalence for:
  1. H/A/Alpha (FullPolHAalpha vs polsartools process_chunk_halphafp)
  2. Freeman-Durden 3C (FreemanDurden3C vs polsartools process_chunk_free3c)
  3. Model-Free 3C/4C (ModelFree3C/4C vs polsartools mf3cf/mf4cf)
  4. Degree of Polarization (DegreeOfPolarization vs polsartools dop_fp)
  5. Shannon Entropy (ShannonEntropy vs polsartools shannon_h_fp)
  6. Neumann Decomposition (NeumannDecomposition vs polsartools neumann_parm)
  7. Praks Parameters (PraksParameters vs polsartools praks_parm_fp)
  8. Touzi Decomposition (TouziDecomposition vs polsartools touzi_decomposition)
  9. Yamaguchi 4C (Yamaguchi4C vs polsartools yam4c_fp_py; patch only — slow)

Usage:
    conda run -n grdx python tests/grdl_polsar_vs_polsartools.py
    conda run -n grdx python tests/grdl_polsar_vs_polsartools.py --chip-size 500

Author
------
Jason Fritz, PhD
43161141+stryder-vtx@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# ======================================================================
# Configuration
# ======================================================================

DEFAULT_NISAR_FILE = Path(
    '/data/sar/slc/nisar/l1_rslc/'
    '20260105T055924_20260105T055931/'
    'NISAR_L1_PR_RSLC_009_116_D_054_2005_QPDH_A_20260105T055924_20260105T055931'
    '_X05010_N_P_J_001.h5'
)


# ======================================================================
# Utility
# ======================================================================

def mse_compare(grdl_arr, pst_arr, name):
    """Compute MSE between GRDL and polsartools, ignoring NaN/Inf."""
    mask = np.isfinite(grdl_arr) & np.isfinite(pst_arr)
    n = mask.sum()
    if n == 0:
        print(f'  {name:14s}: NO VALID PIXELS')
        return float('nan')
    diff = grdl_arr[mask] - pst_arr[mask]
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    max_err = float(np.max(np.abs(diff)))
    print(f'  {name:14s}: MSE={mse:.2e}, MAE={mae:.2e}, '
          f'MaxErr={max_err:.2e} ({n} valid pixels)')
    return mse


# ======================================================================
# polsartools Reference Implementations (inlined)
# ======================================================================

def polsartools_halpha_fp(t3_ccyx):
    """Reproduce polsartools process_chunk_halphafp on (3,3,rows,cols) T3.

    Source: polsartools/polsar/fp/h_a_alpha_fp.py
    Key difference: uses np.linalg.eig (general) vs GRDL's eigh (Hermitian).
    """
    rows, cols = t3_ccyx.shape[2], t3_ccyx.shape[3]

    T = np.transpose(t3_ccyx, (2, 3, 0, 1))
    data = T.reshape(-1, 3, 3)
    data = np.nan_to_num(data, nan=0.0, posinf=0, neginf=0)

    evals_, evecs_ = np.linalg.eig(data)

    sorted_indices = np.argsort(evals_, axis=-1)[:, ::-1]
    evals = np.take_along_axis(evals_, sorted_indices, axis=-1)
    evecs = np.array([
        evecs_[i, :, sorted_indices[i]] for i in range(evecs_.shape[0])
    ])

    esum = evals[:, 0] + evals[:, 1] + evals[:, 2]
    eval_norm1 = np.real(evals[:, 0] / esum)
    eval_norm2 = np.real(evals[:, 1] / esum)
    eval_norm3 = np.real(evals[:, 2] / esum)
    eval_norm1[eval_norm1 < 0] = 0
    eval_norm2[eval_norm2 < 0] = 0
    eval_norm3[eval_norm3 < 0] = 0

    alpha1 = np.arccos(np.abs(evecs[:, 0, 0])) * 180 / np.pi
    alpha2 = np.arccos(np.abs(evecs[:, 0, 1])) * 180 / np.pi
    alpha3 = np.arccos(np.abs(evecs[:, 0, 2])) * 180 / np.pi

    alpha = np.real(
        eval_norm1 * alpha1 + eval_norm2 * alpha2 + eval_norm3 * alpha3
    )

    H = -(eval_norm1 * np.log10(eval_norm1) / np.log10(3)
          + eval_norm2 * np.log10(eval_norm2) / np.log10(3)
          + eval_norm3 * np.log10(eval_norm3) / np.log10(3))
    H = np.real(H)

    Anisotropy = np.real(
        (eval_norm2 - eval_norm3) / (eval_norm2 + eval_norm3)
    )

    return (H.reshape(rows, cols).astype(np.float32),
            alpha.reshape(rows, cols).astype(np.float32),
            Anisotropy.reshape(rows, cols).astype(np.float32))


def polsartools_freeman_3c(c3_ccyx):
    """Reproduce polsartools process_chunk_free3c on (3,3,rows,cols) C3.

    Source: polsartools/polsar/fp/freeman_3c.py
    """
    rows, cols = c3_ccyx.shape[2], c3_ccyx.shape[3]
    eps = 1e-10

    C11 = np.real(c3_ccyx[0, 0]).copy()
    C22 = np.real(c3_ccyx[1, 1]).copy()
    C33 = np.real(c3_ccyx[2, 2]).copy()
    C13_re = np.real(c3_ccyx[0, 2]).copy()
    C13_im = np.imag(c3_ccyx[0, 2]).copy()

    Span = C11 + C22 + C33
    SpanMax = np.nanmax(Span)

    C11_ = C11.copy()
    C33_ = C33.copy()
    C13_re_ = C13_re.copy()
    FV = 3.0 * C22 / 2.0

    C11_ -= FV
    C33_ -= FV
    C13_re_ -= FV / 3.0

    FD = np.zeros_like(C11_)
    FS = np.zeros_like(C11_)
    ALP = np.zeros_like(C11_)
    BET = np.zeros_like(C11_)

    mask_eps = (C11_ <= eps) | (C33_ <= eps)
    FV_eps = 3.0 * (
        C11[mask_eps] + C22[mask_eps] + C33[mask_eps] + 2 * FV[mask_eps]
    ) / 8.0
    FV[mask_eps] = FV_eps

    rtemp = C13_re_**2 + C13_im**2
    mask_non_realizable = rtemp > (C11_ * C33_)
    scale = np.sqrt((C11_ * C33_) / rtemp)
    scale[~mask_non_realizable] = 1.0
    C13_re_ *= scale
    C13_im *= scale

    mask_odd = (C13_re_ >= 0.0) & ~mask_eps
    mask_even = (C13_re_ < 0.0) & ~mask_eps

    FD[mask_odd] = (
        (C11_[mask_odd] * C33_[mask_odd]
         - C13_re_[mask_odd]**2 - C13_im[mask_odd]**2)
        / (C11_[mask_odd] + C33_[mask_odd] + 2 * C13_re_[mask_odd])
    )
    FS[mask_odd] = C33_[mask_odd] - FD[mask_odd]
    ALP[mask_odd] = -1.0
    BET[mask_odd] = (
        np.sqrt((FD[mask_odd] + C13_re_[mask_odd])**2
                + C13_im[mask_odd]**2)
        / FS[mask_odd]
    )

    FS[mask_even] = (
        (C11_[mask_even] * C33_[mask_even]
         - C13_re_[mask_even]**2 - C13_im[mask_even]**2)
        / (C11_[mask_even] + C33_[mask_even] - 2 * C13_re_[mask_even])
    )
    FD[mask_even] = C33_[mask_even] - FS[mask_even]
    BET[mask_even] = 1.0
    FD_safe = np.where(FD[mask_even] <= eps, eps, FD[mask_even])
    ALP[mask_even] = (
        np.sqrt((FS[mask_even] - C13_re_[mask_even])**2
                + C13_im[mask_even]**2)
        / FD_safe
    )

    odd = np.clip(FS * (1 + BET**2), 0, SpanMax).astype(np.float32)
    dbl = np.clip(FD * (1 + ALP**2), 0, SpanMax).astype(np.float32)
    vol = np.clip(8.0 * FV / 3.0, 0, SpanMax).astype(np.float32)

    zero_mask = (odd == 0) & (dbl == 0) & (vol == 0)
    odd[zero_mask] = np.nan
    dbl[zero_mask] = np.nan
    vol[zero_mask] = np.nan
    return odd, dbl, vol


def polsartools_mf3cf(t3_ccyx):
    """Reproduce polsartools process_chunk_mf3cf on (3,3,rows,cols) T3.

    Source: polsartools/polsar/fp/mf3cf.py
    """
    T11 = np.real(t3_ccyx[0, 0])
    T22 = np.real(t3_ccyx[1, 1])
    T33 = np.real(t3_ccyx[2, 2])
    T12 = t3_ccyx[0, 1]
    T13 = t3_ccyx[0, 2]
    T23 = t3_ccyx[1, 2]

    span = T11 + T22 + T33

    # theta_FP
    numer = T22 + T33 - T11
    denom = T11 - T22 + T33
    with np.errstate(divide='ignore', invalid='ignore'):
        theta_fp = np.arctan2(numer, denom)
    np.clip(theta_fp, -np.pi / 4.0, np.pi / 4.0, out=theta_fp)

    # Degree of polarization
    det_T = np.real(
        T11 * (T22 * T33 - np.abs(T23)**2)
        - T12 * (np.conj(T12) * T33 - T23 * np.conj(T13))
        + T13 * (np.conj(T12) * np.conj(T23) - T22 * np.conj(T13))
    )
    trace_cubed = span**3
    with np.errstate(divide='ignore', invalid='ignore'):
        m_sq = 1.0 - 27.0 * det_T / np.where(
            trace_cubed > 0, trace_cubed, 1.0
        )
    np.clip(m_sq, 0.0, 1.0, out=m_sq)
    m = np.sqrt(m_sq)

    # Power partition
    sin2theta = np.sin(2.0 * theta_fp)
    Pv = span * (1.0 - m)
    m_span = m * span
    val = m_span * (1.0 + sin2theta) / 2.0

    Pd = np.where(theta_fp > 0, val, m_span - val)
    Ps = np.where(theta_fp > 0, m_span - val, val)

    np.maximum(Ps, 0.0, out=Ps)
    np.maximum(Pd, 0.0, out=Pd)
    np.maximum(Pv, 0.0, out=Pv)

    return Ps, Pd, Pv, np.degrees(theta_fp)


def polsartools_mf4cf(t3_ccyx):
    """Reproduce polsartools process_chunk_mf4cf on (3,3,rows,cols) T3.

    Source: polsartools/polsar/fp/mf4cf.py
    """
    T11 = np.real(t3_ccyx[0, 0])
    T22 = np.real(t3_ccyx[1, 1])
    T33 = np.real(t3_ccyx[2, 2])
    T12 = t3_ccyx[0, 1]
    T13 = t3_ccyx[0, 2]
    T23 = t3_ccyx[1, 2]

    span = T11 + T22 + T33

    # theta_FP
    numer_theta = T22 + T33 - T11
    denom_theta = T11 - T22 + T33
    with np.errstate(divide='ignore', invalid='ignore'):
        theta_fp = np.arctan2(numer_theta, denom_theta)
    np.clip(theta_fp, -np.pi / 4.0, np.pi / 4.0, out=theta_fp)

    # tau_FP
    numer_tau = 2.0 * np.imag(T23)
    denom_tau = T22 - T33
    with np.errstate(divide='ignore', invalid='ignore'):
        tau_fp = np.arctan2(numer_tau, denom_tau) / 2.0
    np.clip(tau_fp, -np.pi / 4.0, np.pi / 4.0, out=tau_fp)

    # Degree of polarization
    det_T = np.real(
        T11 * (T22 * T33 - np.abs(T23)**2)
        - T12 * (np.conj(T12) * T33 - T23 * np.conj(T13))
        + T13 * (np.conj(T12) * np.conj(T23) - T22 * np.conj(T13))
    )
    trace_cubed = span**3
    with np.errstate(divide='ignore', invalid='ignore'):
        m_sq = 1.0 - 27.0 * det_T / np.where(
            trace_cubed > 0, trace_cubed, 1.0
        )
    np.clip(m_sq, 0.0, 1.0, out=m_sq)
    m = np.sqrt(m_sq)

    # Power partition
    m_span = m * span
    sin2theta = np.sin(2.0 * theta_fp)
    cos2tau = np.cos(2.0 * tau_fp)

    Pc = m_span * (1.0 - cos2tau) / 2.0
    Pv = span * (1.0 - m)

    m_span_residual = m_span - Pc
    val = m_span_residual * (1.0 + sin2theta) / 2.0

    Pd = np.where(theta_fp > 0, val, m_span_residual - val)
    Ps = np.where(theta_fp > 0, m_span_residual - val, val)

    np.maximum(Ps, 0.0, out=Ps)
    np.maximum(Pd, 0.0, out=Pd)
    np.maximum(Pv, 0.0, out=Pv)
    np.maximum(Pc, 0.0, out=Pc)

    return Ps, Pd, Pv, Pc, np.degrees(theta_fp), np.degrees(tau_fp)


# ======================================================================
# polsartools Refined Lee Reference Implementation (inlined)
# Source: polsartools/polsartools/preprocess/rflee_filter.py
# ======================================================================

def _pst_make_mask(window_size):
    """Build 8 directional half-plane masks (polsartools make_Mask)."""
    Mask = np.zeros((8, window_size, window_size), dtype=np.float32)

    Nmax = 0
    for k in range(window_size):
        for l in range((window_size - 1) // 2, window_size):
            Mask[Nmax, k, l] = 1.0

    Nmax = 4
    for k in range(window_size):
        for l in range(0, 1 + (window_size - 1) // 2):
            Mask[Nmax, k, l] = 1.0

    Nmax = 1
    for k in range(window_size):
        for l in range(k, window_size):
            Mask[Nmax, k, l] = 1.0

    Nmax = 5
    for k in range(window_size):
        for l in range(0, k + 1):
            Mask[Nmax, k, l] = 1.0

    Nmax = 2
    for k in range(0, 1 + (window_size - 1) // 2):
        for l in range(window_size):
            Mask[Nmax, k, l] = 1.0

    Nmax = 6
    for k in range((window_size - 1) // 2, window_size):
        for l in range(window_size):
            Mask[Nmax, k, l] = 1.0

    Nmax = 3
    for k in range(window_size):
        for l in range(0, window_size - k):
            Mask[Nmax, k, l] = 1.0

    Nmax = 7
    for k in range(window_size):
        for l in range(window_size - 1 - k, window_size):
            Mask[Nmax, k, l] = 1.0

    return Mask


def _pst_make_coeff(sigma2, Deplct, Nwindow_size, window_sizeM1S2,
                    Sub_Nlig, Sub_Ncol, span, Mask):
    """Per-pixel MMSE coefficient (polsartools make_Coeff)."""
    coeff = np.zeros((Sub_Nlig, Sub_Ncol), dtype=np.complex64)
    Nmax_arr = np.zeros((Sub_Nlig, Sub_Ncol), dtype=np.int32)

    for lig in range(Sub_Nlig):
        for col in range(Sub_Ncol):
            # 3x3 sub-window gradient
            subwin = np.zeros((3, 3), dtype=np.complex64)
            for k in range(3):
                for l in range(3):
                    s = 0.0
                    for kk in range(Nwindow_size):
                        for ll in range(Nwindow_size):
                            s += span[k * Deplct + kk + lig,
                                      l * Deplct + ll + col]
                    subwin[k, l] = s / (Nwindow_size * Nwindow_size)

            Dist = np.zeros(4, dtype=np.complex64)
            Dist[0] = (-subwin[0,0] + subwin[0,2] - subwin[1,0]
                       + subwin[1,2] - subwin[2,0] + subwin[2,2])
            Dist[1] = (subwin[0,1] + subwin[0,2] - subwin[1,0]
                       + subwin[1,2] - subwin[2,0] - subwin[2,1])
            Dist[2] = (subwin[0,0] + subwin[0,1] + subwin[0,2]
                       - subwin[2,0] - subwin[2,1] - subwin[2,2])
            Dist[3] = (subwin[0,0] + subwin[0,1] + subwin[1,0]
                       - subwin[1,2] - subwin[2,1] - subwin[2,2])

            MaxDist = -np.inf
            Nmax_val = 0
            for k in range(4):
                if MaxDist < abs(Dist[k]):
                    MaxDist = abs(Dist[k])
                    Nmax_val = k
            if Dist[Nmax_val] > 0.0:
                Nmax_val += 4
            Nmax_arr[lig, col] = Nmax_val

            m_span = 0.0
            m_span2 = 0.0
            Npoints = 0.0
            for k in range(-window_sizeM1S2, window_sizeM1S2 + 1):
                for l in range(-window_sizeM1S2, window_sizeM1S2 + 1):
                    if Mask[Nmax_val, window_sizeM1S2 + k, window_sizeM1S2 + l] == 1.0:
                        s = span[window_sizeM1S2 + k + lig,
                                 window_sizeM1S2 + l + col]
                        m_span += s
                        m_span2 += s * s
                        Npoints += 1.0
            m_span /= Npoints
            m_span2 /= Npoints

            v_span = m_span2 - m_span * m_span
            cv_span = np.sqrt(abs(v_span)) / (1e-8 + m_span)
            c = (cv_span * cv_span - sigma2) / (cv_span * cv_span * (1 + sigma2) + 1e-8)
            if c < 0.0:
                c = 0.0
            coeff[lig, col] = c

    return coeff, Nmax_arr


def polsartools_refined_lee_t3(t3_ccyx, window_size=7):
    """Refined Lee filter on (3,3,rows,cols) T3.

    Exact loop-for-loop transcription of polsartools
    process_chunk_refined_lee (rflee_filter.py).
    """
    if window_size % 2 == 0:
        window_size += 1

    # Convert T3 to polsartools chunk format and pad internally
    pad_tl = window_size // 2
    pad_br = window_size // 2 + 1

    def _pad(arr):
        return np.pad(arr, ((pad_tl, pad_br), (pad_tl, pad_br)),
                      mode='constant', constant_values=0)

    t11 = _pad(np.real(t3_ccyx[0, 0]).astype(np.float32))
    t12 = _pad(t3_ccyx[0, 1].astype(np.complex64))
    t13 = _pad(t3_ccyx[0, 2].astype(np.complex64))
    t22 = _pad(np.real(t3_ccyx[1, 1]).astype(np.float32))
    t23 = _pad(t3_ccyx[1, 2].astype(np.complex64))
    t33 = _pad(np.real(t3_ccyx[2, 2]).astype(np.float32))

    # Build M_in[9, Nlig_padded, Ncol_padded] — row-major T3 layout
    t21 = np.conj(t12)
    t31 = np.conj(t13)
    t32 = np.conj(t23)
    M_in = np.stack([t11, t12, t13, t21, t22, t23, t31, t32, t33], axis=0)

    NpolarOut, Nlig_padded, Ncol_padded = M_in.shape
    Nlig = Nlig_padded - window_size
    Ncol = Ncol_padded - window_size
    window_sizeM1S2 = (window_size - 1) // 2

    sigma2 = 1.0  # Nlook = 1
    window_params = {
        3: (1, 1), 5: (3, 1), 7: (3, 2), 9: (5, 2), 11: (5, 3),
        13: (5, 4), 15: (7, 4), 17: (7, 5), 19: (7, 6), 21: (9, 6),
        23: (9, 7), 25: (9, 8), 27: (11, 8), 29: (11, 9), 31: (11, 10),
    }
    Nwindow_size, Deplct = window_params[window_size]

    Mask = _pst_make_mask(window_size)
    span = np.real(M_in[0] + M_in[4] + M_in[8]).astype(np.float32)
    coeff, Nmax = _pst_make_coeff(
        sigma2, Deplct, Nwindow_size, window_sizeM1S2, Nlig, Ncol, span, Mask
    )

    M_out = np.zeros((NpolarOut, Nlig, Ncol), dtype=np.complex64)
    for lig in range(Nlig):
        for col in range(Ncol):
            for Np in range(NpolarOut):
                mean = 0.0 + 0.0j
                Npoints = 0.0
                for k in range(-window_sizeM1S2, window_sizeM1S2 + 1):
                    for l in range(-window_sizeM1S2, window_sizeM1S2 + 1):
                        if Mask[Nmax[lig, col],
                                window_sizeM1S2 + k,
                                window_sizeM1S2 + l] == 1.0:
                            mean += M_in[Np,
                                         window_sizeM1S2 + lig + k,
                                         window_sizeM1S2 + col + l]
                            Npoints += 1.0
                mean /= Npoints
                center = M_in[Np, window_sizeM1S2 + lig, window_sizeM1S2 + col]
                M_out[Np, lig, col] = mean + coeff[lig, col] * (center - mean)

    # Pack M_out back to (3,3,rows,cols) T3
    out = np.zeros((3, 3, Nlig, Ncol), dtype=np.complex64)
    out[0, 0] = np.real(M_out[0])
    out[0, 1] = M_out[1];  out[1, 0] = np.conj(M_out[1])
    out[0, 2] = M_out[2];  out[2, 0] = np.conj(M_out[2])
    out[1, 1] = np.real(M_out[4])
    out[1, 2] = M_out[5];  out[2, 1] = np.conj(M_out[5])
    out[2, 2] = np.real(M_out[8])
    return out


# ======================================================================
# polsartools T3 ↔ C3 conversion helpers (from convert_matrices.py)
# D = (1/√2) * [[1,0,1],[1,0,-1],[0,√2,0]]
# C3 = D.T @ T3 @ D  (T3_C3_mat)
# T3 = D  @ C3 @ D.T (C3_T3_mat)
# ======================================================================

_D = (1.0 / np.sqrt(2.0)) * np.array(
    [[1, 0, 1], [1, 0, -1], [0, np.sqrt(2), 0]], dtype=np.complex128
)


def _pst_t3_to_c3(t3_ccyx):
    """Convert (3,3,rows,cols) T3 → C3 using polsartools convention."""
    t3_yx = t3_ccyx.transpose(2, 3, 0, 1)   # (rows,cols,3,3)
    c3_yx = (_D.T) @ t3_yx @ _D             # (rows,cols,3,3)
    return c3_yx.transpose(2, 3, 0, 1)      # (3,3,rows,cols)


def _pst_c3_to_t3(c3_ccyx):
    """Convert (3,3,rows,cols) C3 → T3 using polsartools convention."""
    c3_yx = c3_ccyx.transpose(2, 3, 0, 1)   # (rows,cols,3,3)
    t3_yx = _D @ c3_yx @ (_D.T)             # (rows,cols,3,3)
    return t3_yx.transpose(2, 3, 0, 1)      # (3,3,rows,cols)


# ======================================================================
# polsartools New Decomposition Reference Implementations (inlined)
# ======================================================================

def polsartools_dop_fp(t3_ccyx):
    """Barakat Degree of Polarization.

    Source: polsartools/polsar/fp/dop_fp.py  process_chunk_dopfp (T3 path).

    Note: polsartools does NOT clip m to [0,1] — negative values under the
    sqrt produce NaN.  GRDL clips the radicand to [0,1] first.
    """
    rows, cols = t3_ccyx.shape[2], t3_ccyx.shape[3]
    reshaped = t3_ccyx.reshape(3, 3, -1).transpose(2, 0, 1)
    det_T3 = np.linalg.det(reshaped).reshape(rows, cols)
    trace_T3 = (np.real(t3_ccyx[0, 0])
                + np.real(t3_ccyx[1, 1])
                + np.real(t3_ccyx[2, 2]))
    m1 = np.real(np.sqrt(1.0 - 27.0 * (det_T3 / (trace_T3 ** 3))))
    return m1.astype(np.float32)


def polsartools_shannon_h_fp(t3_ccyx):
    """Shannon entropy components.

    Source: polsartools/polsar/fp/shannon_h_fp.py  proc_shannon_h_fp (T3 path).

    Returns: (H_total, H_intensity, H_polarimetric) — all float32.
    """
    rows, cols = t3_ccyx.shape[2], t3_ccyx.shape[3]
    data = t3_ccyx.transpose(2, 3, 0, 1).reshape(-1, 3, 3)
    data = np.nan_to_num(data, nan=0.0, posinf=0, neginf=0)

    evals_, _ = np.linalg.eig(data)
    sorted_idx = np.argsort(evals_, axis=-1)[:, ::-1]
    evals = np.take_along_axis(evals_, sorted_idx, axis=-1)

    eps = 1e-8
    D = evals[:, 0] * evals[:, 1] * evals[:, 2]
    I = evals[:, 0] + evals[:, 1] + evals[:, 2]

    DoP = np.ones(rows * cols) - 27.0 * D / (I * I * I + eps)
    condition = (1.0 - DoP) < eps
    HSP = np.where(condition, 0.0,
                   np.log(np.abs(np.ones(rows * cols) - DoP)))
    HSP[np.isinf(HSP)] = np.nan
    HSP[HSP == 0] = np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        HSI = 3.0 * np.log(np.e * np.pi * I / 3.0)
        HSI[np.isinf(HSI)] = np.nan
        HSI[HSI == 0] = np.nan

    HS = np.nansum(np.dstack((HSP, HSI)), 2)

    return (np.real(HS).reshape(rows, cols).astype(np.float32),
            np.real(HSI).reshape(rows, cols).astype(np.float32),
            np.real(HSP).reshape(rows, cols).astype(np.float32))


def polsartools_neumann_fp(t3_ccyx):
    """Neumann de-orientation parameters.

    Source: polsartools/polsar/fp/neumann_parm.py  process_chunk_neufp (T3 path).

    Returns: (psi_deg, delta_mod, delta_pha_deg, tau) — all float32.
    """
    rows, cols = t3_ccyx.shape[2], t3_ccyx.shape[3]

    T11 = np.real(t3_ccyx[0, 0])
    T12re = np.real(t3_ccyx[0, 1])
    T12im = np.imag(t3_ccyx[0, 1])
    T13re = np.real(t3_ccyx[0, 2])
    T13im = np.imag(t3_ccyx[0, 2])
    T22 = np.real(t3_ccyx[1, 1])
    T23re = np.real(t3_ccyx[1, 2])
    T33 = np.real(t3_ccyx[2, 2])

    Phi = 0.25 * (np.pi + np.arctan2(-2.0 * T23re, T33 - T22))
    Phi[Phi <= np.pi / 4.0] = Phi[Phi <= np.pi / 4.0]
    Phi[Phi > np.pi / 4.0] -= np.pi / 2.0

    cos2Phi = np.cos(2.0 * Phi)
    sin2Phi = np.sin(2.0 * Phi)
    sin4Phi = np.sin(4.0 * Phi)

    T12re_rot = T12re * cos2Phi - T13re * sin2Phi
    T12im_rot = T12im * cos2Phi - T13im * sin2Phi
    T22_rot = T22 * cos2Phi ** 2 - T23re * sin4Phi + T33 * sin2Phi ** 2
    T33_rot = T22 * sin2Phi ** 2 - T23re * sin4Phi + T33 * cos2Phi ** 2

    eps = np.finfo(float).eps
    delta_mod = np.sqrt((T22_rot + T33_rot) / (T11 + eps))
    delta_pha = np.arctan2(T12im_rot, T12re_rot) * 180.0 / np.pi
    tau = 1.0 - (
        (np.sqrt(T12re_rot ** 2 + T12im_rot ** 2) / T11)
        / (delta_mod + eps)
    )

    psi_deg = Phi * 180.0 / np.pi
    return (psi_deg.astype(np.float32), delta_mod.astype(np.float32),
            delta_pha.astype(np.float32), tau.astype(np.float32))


def polsartools_praks_fp(t3_ccyx):
    """Praks scattering parameters.

    Source: polsartools/polsar/fp/praks_parm_fp.py  process_chunk_praks (T3 path,
    then internal T3→C3 conversion).

    Returns: (frobenius_norm, scatt_pred, scatt_div, deg_pur, dep_ind,
              alpha_deg, entropy) — all float32.
    """
    # polsartools converts T3 → C3 first
    c3 = _pst_t3_to_c3(t3_ccyx)

    span = (np.real(c3[0, 0]) + np.real(c3[1, 1]) + np.real(c3[2, 2]))
    span_safe = np.real(span) + 1e-12

    M_norm = c3 / span_safe[np.newaxis, np.newaxis, :, :]
    FrobNorm = np.sum(np.abs(M_norm) ** 2, axis=(0, 1))
    ScattPred = np.sqrt(FrobNorm)
    ScattDiv = 1.5 * (1.0 - FrobNorm)
    safe_term = np.maximum(FrobNorm - 0.25, 0.0)
    DegPur = 2.0 * np.sqrt(safe_term)
    DepInd = 1.0 - 2.0 * np.sqrt(safe_term) / np.sqrt(3.0)
    Alpha = np.arccos(np.clip(np.real(M_norm[0, 0]), -1.0, 1.0)) * 180.0 / np.pi

    M_adj = M_norm.copy()
    for i in range(3):
        M_adj[i, i] += 0.16
    M_adj_yx = M_adj.transpose(2, 3, 0, 1)  # (rows,cols,3,3)
    det = np.linalg.det(M_adj_yx)
    Entropy = (2.52 + 0.78
               * np.log(np.abs(det) + 1e-12) / np.log(3.0))

    return (FrobNorm.astype(np.float32), ScattPred.astype(np.float32),
            ScattDiv.astype(np.float32), DegPur.astype(np.float32),
            DepInd.astype(np.float32), Alpha.astype(np.float32),
            np.real(Entropy).astype(np.float32))


def polsartools_touzi_fp(t3_ccyx):
    """Touzi TSVM decomposition parameters.

    Source: polsartools/polsar/fp/touzi_decomposition.py
    process_chunk_touzi (T3 path).

    Returns: (alpha1, alpha2, alpha3, phi1, phi2, phi3, tau1, tau2, tau3,
              psi1, psi2, psi3, alpha_mean, phi_mean, tau_mean, psi_mean)
             All float32, in degrees.
    """
    rows, cols = t3_ccyx.shape[2], t3_ccyx.shape[3]
    N = rows * cols
    eps = 1e-8

    M_flat = t3_ccyx.transpose(2, 3, 0, 1).reshape(N, 3, 3)
    M_flat = np.nan_to_num(M_flat, nan=0.0, posinf=0, neginf=0)

    lambda_vals, V = np.linalg.eig(M_flat)
    idx = np.argsort(lambda_vals.real, axis=1)[:, ::-1]
    lambda_sorted = np.take_along_axis(lambda_vals.real, idx, axis=1)
    V_sorted = np.array([V[i][:, idx[i]] for i in range(N)])

    lambda_sorted = np.clip(lambda_sorted, 0, None)

    phase = np.arctan2(V_sorted[:, 0, :].imag, eps + V_sorted[:, 0, :].real)
    V_sorted *= np.exp(-1j * phase[:, np.newaxis, :])

    psi = 0.5 * np.arctan2(V_sorted[:, 2, :].real,
                            eps + V_sorted[:, 1, :].real)

    cos2psi = np.cos(2.0 * psi)
    sin2psi = np.sin(2.0 * psi)
    V1r = V_sorted[:, 1, :].real;  V1i = V_sorted[:, 1, :].imag
    V2r = V_sorted[:, 2, :].real;  V2i = V_sorted[:, 2, :].imag
    V_sorted[:, 1, :] = (V1r * cos2psi + V2r * sin2psi
                         + 1j * (V1i * cos2psi + V2i * sin2psi))
    V_sorted[:, 2, :] = (-V1r * sin2psi + V2r * cos2psi
                         + 1j * (-V1i * sin2psi + V2i * cos2psi))

    tau = 0.5 * np.arctan2(-V_sorted[:, 2, :].imag,
                            eps + V_sorted[:, 0, :].real)
    phi = np.arctan2(V_sorted[:, 1, :].imag, eps + V_sorted[:, 1, :].real)

    cos2tau = np.cos(2.0 * tau);  sin2tau = np.sin(2.0 * tau)
    V0r = V_sorted[:, 0, :].real;  V0i = V_sorted[:, 0, :].imag
    V2r = V_sorted[:, 2, :].real;  V2i = V_sorted[:, 2, :].imag
    V_sorted[:, 0, :] = (V0r * cos2tau - V2i * sin2tau
                         + 1j * (V0i * cos2tau + V2r * sin2tau))
    V_sorted[:, 2, :] = (-V0i * sin2tau + V2r * cos2tau
                         + 1j * (V0r * sin2tau + V2i * cos2tau))

    alpha = np.arccos(np.clip(V_sorted[:, 0, :].real, -1.0, 1.0))

    flip = (psi < -np.pi / 4.0) | (psi > np.pi / 4.0)
    tau[flip] *= -1.0
    phi[flip] *= -1.0

    total_lambda = eps + np.sum(lambda_sorted, axis=1, keepdims=True)
    p = np.clip(lambda_sorted / total_lambda, 0.0, 1.0)

    alpha_mean = np.sum(alpha * p, axis=1)
    phi_mean   = np.sum(phi   * p, axis=1)
    tau_mean   = np.sum(tau   * p, axis=1)
    psi_mean   = np.sum(psi   * p, axis=1)

    to_deg = 180.0 / np.pi
    shape = (rows, cols)
    alphas = (alpha.T.reshape((3,) + shape) * to_deg).astype(np.float32)
    phis   = (phi.T.reshape((3,) + shape)   * to_deg).astype(np.float32)
    taus   = (tau.T.reshape((3,) + shape)   * to_deg).astype(np.float32)
    psis   = (psi.T.reshape((3,) + shape)   * to_deg).astype(np.float32)

    return (alphas[0], alphas[1], alphas[2],
            phis[0],   phis[1],   phis[2],
            taus[0],   taus[1],   taus[2],
            psis[0],   psis[1],   psis[2],
            (alpha_mean.reshape(shape) * to_deg).astype(np.float32),
            (phi_mean.reshape(shape)   * to_deg).astype(np.float32),
            (tau_mean.reshape(shape)   * to_deg).astype(np.float32),
            (psi_mean.reshape(shape)   * to_deg).astype(np.float32))


def _pst_unitary_rotation(T3, teta):
    """Unitary rotation applied to flat 9-element T3 array (polsartools)."""
    T12_re, T12_im = T3[1].real, T3[1].imag
    T13_re, T13_im = T3[2].real, T3[2].imag
    T22 = T3[4]; T23_re, T23_im = T3[5].real, T3[5].imag; T33 = T3[8]
    ct = np.cos(teta);  st = np.sin(teta)
    T3[1] = (T12_re * ct + T13_re * st) + 1j * (T12_im * ct + T13_im * st)
    T3[2] = (-T12_re * st + T13_re * ct) + 1j * (-T12_im * st + T13_im * ct)
    T3[4] = T22 * ct**2 + 2.0 * T23_re * ct * st + T33 * st**2
    T3[5] = ((-T22 * ct * st + T23_re * (ct**2 - st**2) + T33 * ct * st)
             + 1j * (T23_im * ct**2 + T23_im * st**2))
    T3[8] = T22 * st**2 + T33 * ct**2 - 2.0 * T23_re * ct * st
    return T3


def polsartools_yam4c_py(t3_ccyx, model=''):
    """Yamaguchi 4-component decomposition — pure Python pixel loop.

    Source: polsartools/polsar/fp/yam4c_fp_py.py  process_chunk_yam4cfp.

    model : '' or 'y4co' = Y4O (original)
            'y4cr'        = Y4R (rotation-corrected)
            'y4cs'        = Y4S (extended volume)

    Returns: (odd, dbl, vol, hlx) — all float32.

    WARNING: This is the pure Python pixel loop.  It is **very slow** on
    large arrays — call on a small patch only (e.g. 50×50).
    """
    rows, cols = t3_ccyx.shape[2], t3_ccyx.shape[3]
    span = (np.real(t3_ccyx[0, 0]) + np.real(t3_ccyx[1, 1])
            + np.real(t3_ccyx[2, 2]))
    SpanMax = float(np.nanmax(span))
    SpanMin = float(np.nanmax([np.nanmin(span), 1e-6]))
    eps = 1e-6

    # Flatten to (rows*cols, 9) for pixel-loop access — same as polsartools
    T_flat = t3_ccyx.transpose(2, 3, 0, 1).reshape(rows * cols, 9)

    M_odd = np.zeros(rows * cols)
    M_dbl = np.zeros(rows * cols)
    M_vol = np.zeros(rows * cols)
    M_hlx = np.zeros(rows * cols)

    for k in range(rows * cols):
        T3 = T_flat[k].copy()

        if model in ('y4cr', 'y4cs'):
            denom = T3[4].real - T3[8].real
            teta = (0.5 * np.arctan(2.0 * T3[5].real / denom)
                    if abs(denom) > eps else 0.0)
            T3 = _pst_unitary_rotation(T3, teta)

        if np.allclose(T3, 0, rtol=1e-8, atol=1e-10):
            M_odd[k] = np.nan;  M_dbl[k] = np.nan
            M_vol[k] = np.nan;  M_hlx[k] = np.nan
            continue

        # Extract diagonal elements as float — Hermitian T3 diagonals are real;
        # stripping the fp imaginary residuals prevents ComplexWarning throughout.
        t11 = float(T3[0].real)
        t22 = float(T3[4].real)
        t33 = float(T3[8].real)

        Pc = 2.0 * abs(T3[5].imag)
        HV_type = 1

        if model == 'y4cs':
            C1 = t11 - t22 + (7.0 / 8.0) * t33 + (Pc / 16.0)
            HV_type = 1 if C1 > 0.0 else 2

        if HV_type == 1:
            ratio = 10.0 * np.log10(
                (t11 + t22 - 2.0 * T3[1].real)
                / max(t11 + t22 + 2.0 * T3[1].real, eps)
            )
            two_t33_pc = 2.0 * t33 - Pc
            if -2.0 < ratio <= 2.0:
                Pv = 2.0 * two_t33_pc
            else:
                Pv = (15.0 / 8.0) * two_t33_pc
        else:
            Pv = (15.0 / 16.0) * (2.0 * t33 - Pc)
            ratio = 0.0

        TP = t11 + t22 + t33

        if Pv < 0.0:
            # Freeman-Yamaguchi 3C fallback
            BETre = BETim = ALPre = ALPim = FS = FD = FV = 0.0
            HHHH   = (t11 + 2.0 * T3[1].real + t22) / 2.0
            HHVVre = (t11 - t22) / 2.0
            HHVVim = -float(T3[1].imag)
            HVHV   = t33 / 2.0
            VVVV   = (t11 - 2.0 * T3[1].real + t22) / 2.0
            ratio  = 10.0 * np.log10(max(VVVV / max(HHHH, eps), eps))

            if ratio <= -2.0:
                FV = 15.0 * HVHV / 4.0
                HHHH -= 8.0 * FV / 15.0; VVVV -= 3.0 * FV / 15.0
                HHVVre -= 2.0 * FV / 15.0
            elif ratio > 2.0:
                FV = 15.0 * HVHV / 4.0
                HHHH -= 3.0 * FV / 15.0; VVVV -= 8.0 * FV / 15.0
                HHVVre -= 2.0 * FV / 15.0
            else:
                FV = 8.0 * HVHV / 2.0
                HHHH -= 3.0 * FV / 8.0; VVVV -= 3.0 * FV / 8.0
                HHVVre -= 1.0 * FV / 8.0

            if HHHH <= eps or VVVV <= eps:
                if -2.0 < ratio <= 2.0:
                    FV = ((HHHH + 3.0 * FV / 8.0) + HVHV
                          + (VVVV + 3.0 * FV / 8.0))
                elif ratio <= -2.0:
                    FV = ((HHHH + 8.0 * FV / 15.0) + HVHV
                          + (VVVV + 3.0 * FV / 15.0))
                else:
                    FV = ((HHHH + 3.0 * FV / 15.0) + HVHV
                          + (VVVV + 8.0 * FV / 15.0))
            else:
                rtemp = HHVVre ** 2 + HHVVim ** 2
                if rtemp > HHHH * VVVV:
                    sc = np.sqrt(HHHH * VVVV / max(rtemp, eps))
                    HHVVre *= sc;  HHVVim *= sc
                if HHVVre >= 0.0:
                    ALPre = -1.0; ALPim = 0.0
                    FD = ((HHHH * VVVV - HHVVre**2 - HHVVim**2)
                          / (HHHH + VVVV + 2.0 * HHVVre))
                    FS = VVVV - FD
                    BETre = (FD + HHVVre) / max(FS, eps)
                    BETim = HHVVim / max(FS, eps)
                else:
                    BETre = 1.0; BETim = 0.0
                    FS = ((HHHH * VVVV - HHVVre**2 - HHVVim**2)
                          / (HHHH + VVVV - 2.0 * HHVVre))
                    FD = VVVV - FS
                    ALPre = (HHVVre - FS) / max(FD, eps)
                    ALPim = HHVVim / max(FD, eps)

            M_odd[k] = np.clip(FS * (1.0 + BETre**2 + BETim**2), SpanMin, SpanMax)
            M_dbl[k] = np.clip(FD * (1.0 + ALPre**2 + ALPim**2), SpanMin, SpanMax)
            M_vol[k] = np.clip(FV, SpanMin, SpanMax)
            M_hlx[k] = 0.0
        else:
            if HV_type == 1:
                S = t11 - Pv / 2.0
                D = TP - Pv - Pc - S
                Cre = float(T3[1].real) + float(T3[2].real)
                Cim = float(T3[1].imag) + float(T3[2].imag)
                if ratio <= -2.0:
                    Cre -= Pv / 6.0
                elif ratio > 2.0:
                    Cre += Pv / 6.0
                if (Pv + Pc) > TP:
                    Ps = 0.0;  Pd = 0.0;  Pv = TP - Pc
                else:
                    CO = 2.0 * t11 + Pc - TP
                    if CO > 0.0:
                        Ps = S + (Cre**2 + Cim**2) / max(S, eps)
                        Pd = D - (Cre**2 + Cim**2) / max(S, eps)
                    else:
                        Pd = D + (Cre**2 + Cim**2) / max(D, eps)
                        Ps = S - (Cre**2 + Cim**2) / max(D, eps)
                if Ps < 0.0:
                    if Pd < 0.0:
                        Ps = 0.0;  Pd = 0.0;  Pv = TP - Pc
                    else:
                        Ps = 0.0;  Pd = TP - Pv - Pc
                elif Pd < 0.0:
                    Pd = 0.0;  Ps = TP - Pv - Pc
            else:  # HV_type == 2
                S = t11
                D = TP - Pv - Pc - S
                Cre = float(T3[1].real) + float(T3[2].real)
                Cim = float(T3[1].imag) + float(T3[2].imag)
                Pd = D + (Cre**2 + Cim**2) / max(D, eps)
                Ps = S - (Cre**2 + Cim**2) / max(D, eps)
                if Ps < 0.0:
                    if Pd < 0.0:
                        Ps = 0.0;  Pd = 0.0;  Pv = TP - Pc
                    else:
                        Ps = 0.0;  Pd = TP - Pv - Pc
                elif Pd < 0.0:
                    Pd = 0.0;  Ps = TP - Pv - Pc

            Ps = max(min(max(Ps, 0.0), SpanMax), SpanMin)
            Pd = max(min(max(Pd, 0.0), SpanMax), SpanMin)
            Pv = max(min(max(Pv, 0.0), SpanMax), SpanMin)
            Pc = max(min(max(Pc, 0.0), SpanMax), SpanMin)
            M_odd[k] = Ps;  M_dbl[k] = Pd
            M_vol[k] = Pv;  M_hlx[k] = Pc

    return (M_odd.reshape(rows, cols).astype(np.float32),
            M_dbl.reshape(rows, cols).astype(np.float32),
            M_vol.reshape(rows, cols).astype(np.float32),
            M_hlx.reshape(rows, cols).astype(np.float32))


# ======================================================================
# Data Loading
# ======================================================================

def load_nisar_chip(nisar_file, chip_size=500, frequency='A'):
    """Load a quad-pol chip from a NISAR RSLC file."""
    from grdl.IO.sar.nisar import NISARReader

    reader = NISARReader(nisar_file, frequency=frequency, polarizations='all')
    meta = reader.metadata
    print(f'Image: {meta.rows} x {meta.cols}, bands={meta.bands}')

    cr, cc = meta.rows // 2, meta.cols // 2
    half = chip_size // 2
    r0 = max(0, cr - half)
    c0 = max(0, cc - half)
    r1 = min(meta.rows, r0 + chip_size)
    c1 = min(meta.cols, c0 + chip_size)
    cube = reader.read_chip(r0, r1, c0, c1)
    reader.close()
    print(f'Chip: [{r0}:{r1}, {c0}:{c1}], shape={cube.shape}')

    pol_names = [cm.polarization for cm in meta.channel_metadata]
    pol_index = {p: i for i, p in enumerate(pol_names)}

    shh = cube[pol_index['HH']]
    shv = cube[pol_index['HV']]
    svh = cube[pol_index.get('VH', pol_index['HV'])]
    svv = cube[pol_index['VV']]

    return shh, shv, svh, svv


# ======================================================================
# Validation Functions
# ======================================================================

def validate_halpha(shh, shv, svh, svv, window_size=3):
    """Cross-validate FullPolHAalpha against polsartools."""
    from grdl.image_processing.decomposition import (
        FullPolHAalpha, CoherencyMatrix,
    )

    print('\n' + '=' * 60)
    print('H/A/Alpha — GRDL vs polsartools')
    print('=' * 60)

    # GRDL decomposition
    decomp = FullPolHAalpha(window_size=window_size)
    components = decomp.decompose(shh, shv, svh, svv)

    # Build T3 for polsartools
    channels = np.stack([shh, shv, svh, svv], axis=0)
    t3 = CoherencyMatrix(window_size=window_size).compute(channels)

    # polsartools
    H_pst, alpha_pst, aniso_pst = polsartools_halpha_fp(t3)

    print(f'\nGRDL Entropy:    [{components["entropy"].min():.4f}, '
          f'{components["entropy"].max():.4f}]')
    print(f'polsartools H:   [{np.nanmin(H_pst):.4f}, '
          f'{np.nanmax(H_pst):.4f}]')

    print('\nMSE comparison:')
    mse_H = mse_compare(components['entropy'], H_pst, 'Entropy')
    mse_alpha = mse_compare(components['alpha'], alpha_pst, 'Alpha (°)')
    mse_aniso = mse_compare(components['anisotropy'], aniso_pst, 'Anisotropy')

    return mse_H, mse_alpha, mse_aniso


def validate_freeman_durden(shh, shv, svh, svv, window_size=3):
    """Cross-validate FreemanDurden3C against polsartools."""
    from grdl.image_processing.decomposition import (
        FreemanDurden3C, CovarianceMatrix,
    )

    print('\n' + '=' * 60)
    print('Freeman-Durden 3C — GRDL vs polsartools')
    print('=' * 60)

    # GRDL decomposition
    fd = FreemanDurden3C(window_size=window_size)
    fd_components = fd.decompose(shh, shv, svh, svv)

    # Build C3 for polsartools
    channels = np.stack([shh, shv, svh, svv], axis=0)
    c3 = CovarianceMatrix(window_size=window_size).compute(channels)

    # polsartools
    ps_pst, pd_pst, pv_pst = polsartools_freeman_3c(c3)

    print(f'\nGRDL Ps: [{np.nanmin(fd_components["surface"]):.4f}, '
          f'{np.nanmax(fd_components["surface"]):.4f}]')
    print(f'pst  Ps: [{np.nanmin(ps_pst):.4f}, {np.nanmax(ps_pst):.4f}]')

    print('\nMSE comparison:')
    mse_ps = mse_compare(fd_components['surface'], ps_pst, 'Surface (Ps)')
    mse_pd = mse_compare(fd_components['double_bounce'], pd_pst, 'Dbl-bounce')
    mse_pv = mse_compare(fd_components['volume'], pv_pst, 'Volume (Pv)')

    return mse_ps, mse_pd, mse_pv


def validate_model_free(shh, shv, svh, svv, window_size=3):
    """Cross-validate ModelFree3C/4C against polsartools."""
    from grdl.image_processing.decomposition import (
        ModelFree3C, ModelFree4C, CoherencyMatrix,
    )

    print('\n' + '=' * 60)
    print('Model-Free 3C/4C — GRDL vs polsartools')
    print('=' * 60)

    # Build T3
    channels = np.stack([shh, shv, svh, svv], axis=0)
    t3 = CoherencyMatrix(window_size=window_size).compute(channels)

    # --- MF3CF ---
    mf3 = ModelFree3C(window_size=window_size)
    mf3_components = mf3.decompose_from_t3(t3)

    ps_pst, pd_pst, pv_pst, theta_pst = polsartools_mf3cf(t3)

    print('\n--- MF3CF ---')
    print('\nMSE comparison:')
    mse_ps = mse_compare(mf3_components['surface'], ps_pst, 'Surface (Ps)')
    mse_pd = mse_compare(mf3_components['double_bounce'], pd_pst, 'Dbl-bounce')
    mse_pv = mse_compare(mf3_components['volume'], pv_pst, 'Volume (Pv)')
    mse_theta = mse_compare(mf3_components['theta_fp'], theta_pst, 'θ_FP (°)')

    # --- MF4CF ---
    mf4 = ModelFree4C(window_size=window_size)
    mf4_components = mf4.decompose_from_t3(t3)

    ps4, pd4, pv4, pc4, theta4, tau4 = polsartools_mf4cf(t3)

    print('\n--- MF4CF ---')
    print('\nMSE comparison:')
    mse_compare(mf4_components['surface'], ps4, 'Surface (Ps)')
    mse_compare(mf4_components['double_bounce'], pd4, 'Dbl-bounce')
    mse_compare(mf4_components['volume'], pv4, 'Volume (Pv)')
    mse_compare(mf4_components['helix'], pc4, 'Helix (Pc)')
    mse_compare(mf4_components['theta_fp'], theta4, 'θ_FP (°)')
    mse_compare(mf4_components['tau_fp'], tau4, 'τ_FP (°)')

    return mse_ps, mse_pd, mse_pv


def validate_dop(shh, shv, svh, svv, window_size=3):
    """Cross-validate DegreeOfPolarization against polsartools.

    Expected result: very small MSE.  The only difference is that polsartools
    does NOT clip the radicand to [0,1] before sqrt, so negative values under
    the sqrt produce NaN instead of 0.  Both implementations agree on all
    finite pixels.
    """
    from grdl.image_processing.decomposition import (
        DegreeOfPolarization, CoherencyMatrix,
    )

    print('\n' + '=' * 60)
    print('Degree of Polarization — GRDL vs polsartools')
    print('=' * 60)

    channels = np.stack([shh, shv, svh, svv], axis=0)
    t3 = CoherencyMatrix(window_size=window_size).compute(channels)

    comp = DegreeOfPolarization(window_size=window_size).decompose(
        shh, shv, svh, svv
    )
    dop_pst = polsartools_dop_fp(t3)

    print(f'\nGRDL DoP:        [{np.nanmin(comp["dop"]):.4f}, '
          f'{np.nanmax(comp["dop"]):.4f}]')
    print(f'polsartools DoP: [{np.nanmin(dop_pst):.4f}, '
          f'{np.nanmax(dop_pst):.4f}]')
    print('\nMSE comparison:')
    return mse_compare(comp['dop'], dop_pst, 'DoP')


def validate_shannon_entropy(shh, shv, svh, svv, window_size=3):
    """Cross-validate ShannonEntropy against polsartools."""
    from grdl.image_processing.decomposition import (
        ShannonEntropy, CoherencyMatrix,
    )

    print('\n' + '=' * 60)
    print('Shannon Entropy — GRDL vs polsartools')
    print('=' * 60)

    channels = np.stack([shh, shv, svh, svv], axis=0)
    t3 = CoherencyMatrix(window_size=window_size).compute(channels)

    comp = ShannonEntropy(window_size=window_size).decompose(
        shh, shv, svh, svv
    )
    hs_pst, hsi_pst, hsp_pst = polsartools_shannon_h_fp(t3)

    print(f'\nGRDL H_total: [{np.nanmin(comp["H_total"]):.4f}, '
          f'{np.nanmax(comp["H_total"]):.4f}]')
    print(f'pst  H_total: [{np.nanmin(hs_pst):.4f}, {np.nanmax(hs_pst):.4f}]')
    print('\nMSE comparison:')
    mse_hs  = mse_compare(comp['H_total'],        hs_pst,  'H_total')
    mse_hsi = mse_compare(comp['H_intensity'],    hsi_pst, 'H_intensity')
    mse_hsp = mse_compare(comp['H_polarimetric'], hsp_pst, 'H_polarimetric')
    return mse_hs, mse_hsi, mse_hsp


def validate_neumann(shh, shv, svh, svv, window_size=3):
    """Cross-validate NeumannDecomposition against polsartools."""
    from grdl.image_processing.decomposition import (
        NeumannDecomposition, CoherencyMatrix,
    )

    print('\n' + '=' * 60)
    print('Neumann Decomposition — GRDL vs polsartools')
    print('=' * 60)

    channels = np.stack([shh, shv, svh, svv], axis=0)
    t3 = CoherencyMatrix(window_size=window_size).compute(channels)

    comp = NeumannDecomposition(window_size=window_size).decompose(
        shh, shv, svh, svv
    )
    psi_pst, dmod_pst, dpha_pst, tau_pst = polsartools_neumann_fp(t3)

    print(f'\nGRDL psi (°): [{np.nanmin(comp["psi"]):.3f}, '
          f'{np.nanmax(comp["psi"]):.3f}]')
    print(f'pst  psi (°): [{np.nanmin(psi_pst):.3f}, {np.nanmax(psi_pst):.3f}]')
    print('\nMSE comparison:')
    mse_psi  = mse_compare(comp['psi'],       psi_pst,  'psi (°)')
    mse_dmod = mse_compare(comp['delta_mod'], dmod_pst, 'delta_mod')
    mse_dpha = mse_compare(comp['delta_pha'], dpha_pst, 'delta_pha (°)')
    mse_tau  = mse_compare(comp['tau'],       tau_pst,  'tau')
    return mse_psi, mse_dmod, mse_dpha, mse_tau


def validate_praks(shh, shv, svh, svv, window_size=3):
    """Cross-validate PraksParameters against polsartools.

    Note: polsartools computes |det| as sqrt(re²+im²) then applies log,
    whereas GRDL uses log(|det|).  These are equivalent.  The entropy
    formula is otherwise identical.
    """
    from grdl.image_processing.decomposition import (
        PraksParameters, CoherencyMatrix,
    )

    print('\n' + '=' * 60)
    print('Praks Parameters — GRDL vs polsartools')
    print('=' * 60)

    channels = np.stack([shh, shv, svh, svv], axis=0)
    t3 = CoherencyMatrix(window_size=window_size).compute(channels)

    comp = PraksParameters(window_size=window_size).decompose(
        shh, shv, svh, svv
    )
    (fn_pst, sp_pst, sd_pst,
     dp_pst, di_pst, al_pst, en_pst) = polsartools_praks_fp(t3)

    print(f'\nGRDL FrobNorm: [{np.nanmin(comp["frobenius_norm"]):.4f}, '
          f'{np.nanmax(comp["frobenius_norm"]):.4f}]')
    print(f'pst  FrobNorm: [{np.nanmin(fn_pst):.4f}, {np.nanmax(fn_pst):.4f}]')
    print('\nMSE comparison:')
    mse_compare(comp['frobenius_norm'],          fn_pst, 'FrobNorm')
    mse_compare(comp['scattering_predominance'], sp_pst, 'ScattPred')
    mse_compare(comp['scattering_diversity'],    sd_pst, 'ScattDiv')
    mse_compare(comp['degree_purity'],           dp_pst, 'DegPurity')
    mse_compare(comp['depolarization_index'],    di_pst, 'DepIndex')
    mse_compare(comp['alpha'],                   al_pst, 'Alpha (°)')
    mse_fn = mse_compare(comp['entropy'],        en_pst, 'Entropy')
    return mse_fn


def validate_touzi(shh, shv, svh, svv, window_size=3):
    """Cross-validate TouziDecomposition against polsartools."""
    from grdl.image_processing.decomposition import (
        TouziDecomposition, CoherencyMatrix,
    )

    print('\n' + '=' * 60)
    print('Touzi TSVM Decomposition — GRDL vs polsartools')
    print('=' * 60)

    channels = np.stack([shh, shv, svh, svv], axis=0)
    t3 = CoherencyMatrix(window_size=window_size).compute(channels)

    comp = TouziDecomposition(window_size=window_size).decompose(
        shh, shv, svh, svv
    )
    (a1, a2, a3, ph1, ph2, ph3, ta1, ta2, ta3,
     ps1, ps2, ps3, am, phm, tm, psm) = polsartools_touzi_fp(t3)

    print(f'\nGRDL alpha_mean: [{np.nanmin(comp["alpha_mean"]):.3f}, '
          f'{np.nanmax(comp["alpha_mean"]):.3f}]')
    print(f'pst  alpha_mean: [{np.nanmin(am):.3f}, {np.nanmax(am):.3f}]')
    print('\nMSE comparison:')
    mse_compare(comp['alpha1'],     a1,  'alpha1 (°)')
    mse_compare(comp['alpha2'],     a2,  'alpha2 (°)')
    mse_compare(comp['alpha3'],     a3,  'alpha3 (°)')
    mse_compare(comp['phi1'],       ph1, 'phi1 (°)')
    mse_compare(comp['tau1'],       ta1, 'tau1 (°)')
    mse_compare(comp['psi1'],       ps1, 'psi1 (°)')
    mse_am = mse_compare(comp['alpha_mean'], am,  'alpha_mean (°)')
    mse_compare(comp['phi_mean'],   phm, 'phi_mean (°)')
    mse_compare(comp['tau_mean'],   tm,  'tau_mean (°)')
    mse_compare(comp['psi_mean'],   psm, 'psi_mean (°)')
    return mse_am


def validate_yamaguchi4c(shh, shv, svh, svv, window_size=3):
    """Cross-validate Yamaguchi4C against polsartools pure Python.

    The pure Python pixel loop in polsartools is very slow, so comparison
    is done on a central 50×50 sub-patch only.

    Expected differences (intentional GRDL design decisions):
    - GRDL uses 0.0 as output floor; polsartools clips to SpanMin > 0.
    - GRDL uses arctan2 with zero-denom guard for rotation angle;
      polsartools uses plain arctan (may divide by zero for T22 == T33).
    All three model variants (y4o, y4r, y4s) are compared.
    """
    import warnings
    from grdl.image_processing.decomposition import (
        Yamaguchi4C, CoherencyMatrix,
    )

    print('\n' + '=' * 60)
    print('Yamaguchi 4C — GRDL vs polsartools (patch comparison)')
    print('=' * 60)

    PATCH = 50
    r_rows, r_cols = shh.shape
    rp = min(r_rows, PATCH);  cp = min(r_cols, PATCH)
    r0 = (r_rows - rp) // 2;  c0 = (r_cols - cp) // 2

    shh_p = shh[r0:r0 + rp, c0:c0 + cp]
    shv_p = shv[r0:r0 + rp, c0:c0 + cp]
    svh_p = svh[r0:r0 + rp, c0:c0 + cp]
    svv_p = svv[r0:r0 + rp, c0:c0 + cp]

    channels_p = np.stack([shh_p, shv_p, svh_p, svv_p], axis=0)
    t3_p = CoherencyMatrix(window_size=window_size).compute(channels_p)

    print(f'\nUsing central {rp}×{cp} patch (polsartools loop is slow).')

    model_map = [('y4o', ''), ('y4r', 'y4cr'), ('y4s', 'y4cs')]
    for grdl_model, pst_model in model_map:
        print(f'\n--- {grdl_model.upper()} vs polsartools '
              f'{"y4co" if not pst_model else pst_model} ---')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            comp = Yamaguchi4C(window_size=window_size,
                               model=grdl_model).decompose_from_t3(t3_p)

        print('Running polsartools pure Python...')
        odd_pst, dbl_pst, vol_pst, hlx_pst = polsartools_yam4c_py(
            t3_p, model=pst_model
        )

        print('\nMSE comparison:')
        mse_compare(comp['surface'],       odd_pst, 'surface (Ps)')
        mse_compare(comp['double_bounce'], dbl_pst, 'double-bounce')
        mse_compare(comp['volume'],        vol_pst, 'volume (Pv)')
        mse_compare(comp['helix'],         hlx_pst, 'helix (Pc)')


def validate_refined_lee(shh, shv, svh, svv, window_size=7):
    """Cross-validate RefinedLeeFilter against polsartools.

    Computes a single-look T3 (outer product, no boxcar) and applies both
    implementations.  The polsartools reference uses pure Python loops and
    is slow; comparison is done on a central 50×50 sub-patch.

    Expected result: non-zero but small MSE (~1e-4 for T11).  The
    discrepancy is *not* a bug — it arises because polsartools accumulates
    span in float32 while GRDL uses float64.  For pixels where two
    directional gradient magnitudes are nearly equal, the float32/float64
    rounding difference flips which of the 8 half-plane masks is chosen.
    That discrete flip changes the mean entirely for that pixel, producing
    a large local error.  Pixels with a clear dominant gradient direction
    agree to floating-point precision.
    """
    from grdl.image_processing.filters import RefinedLeeFilter

    print('\n' + '=' * 60)
    print(f'Refined Lee Filter (window={window_size}) — GRDL vs polsartools')
    print('=' * 60)

    # Build single-look T3 from Pauli basis (no boxcar)
    k0 = (shh - svv) / np.sqrt(2)
    k1 = (shv + svh) / np.sqrt(2)
    k2 = (shh + svv) / np.sqrt(2)
    rows, cols = shh.shape
    t3 = np.zeros((3, 3, rows, cols), dtype=np.complex64)
    for i, ki in enumerate([k0, k1, k2]):
        for j, kj in enumerate([k0, k1, k2]):
            t3[i, j] = ki * np.conj(kj)

    span = np.real(t3[0, 0] + t3[1, 1] + t3[2, 2])
    print(f'\nInput T3 shape: {t3.shape}')
    print(f'Span range: [{span.min():.4f}, {span.max():.4f}]')
    print('NOTE: non-zero MSE is expected — see function docstring.')

    # GRDL filter (run on the full chip)
    rlf = RefinedLeeFilter(kernel_size=window_size)
    t3_grdl = rlf.filter_matrix(t3)

    # Extract central patch for polsartools reference (Python loops are slow)
    PATCH = 50
    if rows > PATCH or cols > PATCH:
        rp = min(rows, PATCH)
        cp = min(cols, PATCH)
        r0 = (rows - rp) // 2
        c0 = (cols - cp) // 2
        t3_patch = t3[:, :, r0:r0 + rp, c0:c0 + cp]
        t3_grdl_patch = t3_grdl[:, :, r0:r0 + rp, c0:c0 + cp]
        print(f'\nUsing central {rp}×{cp} patch for polsartools reference '
              f'(loop-based implementation).')
    else:
        t3_patch = t3
        t3_grdl_patch = t3_grdl

    print('Running polsartools reference...')
    t3_pst = polsartools_refined_lee_t3(t3_patch, window_size=window_size)

    # Compare unique T3 elements (upper triangle + diagonal)
    print('\nMSE comparison (T3 elements):')
    elements = [
        ('T11', 0, 0), ('T12', 0, 1), ('T13', 0, 2),
        ('T22', 1, 1), ('T23', 1, 2), ('T33', 2, 2),
    ]
    for name, i, j in elements:
        mse_compare(np.real(t3_grdl_patch[i, j]),
                    np.real(t3_pst[i, j]), f'{name}.re')
        if i != j:
            mse_compare(np.imag(t3_grdl_patch[i, j]),
                        np.imag(t3_pst[i, j]), f'{name}.im')


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GRDL vs polsartools cross-validation'
    )
    parser.add_argument(
        '--nisar-file', type=Path, default=DEFAULT_NISAR_FILE,
        help='Path to NISAR RSLC H5 file'
    )
    parser.add_argument(
        '--chip-size', type=int, default=500,
        help='Square chip side length (pixels)'
    )
    parser.add_argument(
        '--window-size', type=int, default=3,
        help='T3/C3 boxcar averaging window size'
    )
    parser.add_argument(
        '--filter-window-size', type=int, default=7,
        help='Refined Lee filter kernel size (odd, 3-31, default 7)'
    )
    args = parser.parse_args()

    if not args.nisar_file.exists():
        print(f'ERROR: NISAR file not found: {args.nisar_file}')
        print('Falling back to synthetic data (50x50)...')
        rng = np.random.default_rng(42)
        N = 50
        shh = (rng.standard_normal((N, N))
               + 1j * rng.standard_normal((N, N))).astype(np.complex64)
        shv = (0.3 * (rng.standard_normal((N, N))
               + 1j * rng.standard_normal((N, N)))).astype(np.complex64)
        svh = shv.copy()
        svv = (rng.standard_normal((N, N))
               + 1j * rng.standard_normal((N, N))).astype(np.complex64)
    else:
        shh, shv, svh, svv = load_nisar_chip(
            args.nisar_file, chip_size=args.chip_size
        )

    print(f'\nData shape: {shh.shape}')
    print(f'Window size: {args.window_size}')

    # Run all cross-validations
    validate_halpha(shh, shv, svh, svv, window_size=args.window_size)
    validate_freeman_durden(shh, shv, svh, svv, window_size=args.window_size)
    validate_model_free(shh, shv, svh, svv, window_size=args.window_size)
    validate_dop(shh, shv, svh, svv, window_size=args.window_size)
    validate_shannon_entropy(shh, shv, svh, svv, window_size=args.window_size)
    validate_neumann(shh, shv, svh, svv, window_size=args.window_size)
    validate_praks(shh, shv, svh, svv, window_size=args.window_size)
    validate_touzi(shh, shv, svh, svv, window_size=args.window_size)
    validate_yamaguchi4c(shh, shv, svh, svv, window_size=args.window_size)
    validate_refined_lee(shh, shv, svh, svv, window_size=args.filter_window_size)

    print('\n' + '=' * 60)
    print('Cross-validation complete.')
    print('=' * 60)


if __name__ == '__main__':
    main()
