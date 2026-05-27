# -*- coding: utf-8 -*-
"""
Generic CRSD PVP/PPP array builders and CI2 quantization.

Constructs structured numpy arrays for Per-Vector Parameters (PVP)
and Per-Pulse Parameters (PPP) from channel metadata, timing,
platform state vectors, and antenna geometry. This module is sensor-
agnostic and can be reused by any CRSD producer.

The dtypes are inferred from the CRSD XML tree via ``sarkit.crsd``
for binary compatibility with CRSD file output.

Dependencies
------------
sarkit

Author
------
Jason Fritz
43161141+stryder-vtx@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-05-27

Modified
--------
2026-05-27
"""

# Standard library
from typing import Protocol, Tuple, Union

# Third-party
import numpy as np
from lxml import etree

# GRDL internal
from grdl.IO.sar._backend import require_sarkit

require_sarkit('CRSD')
import sarkit.crsd


class CRSDChannelLike(Protocol):
    """Minimal channel metadata required for PVP/PPP construction."""

    num_vectors: int
    f0_ref: float
    bw_inst: float
    fx_freq0: float
    fx_bw: float
    tx_pulse_duration: float
    fx_rate: float


def _to_vector(
    value: Union[int, float, np.ndarray],
    n: int,
    dtype: np.dtype,
) -> np.ndarray:
    """Broadcast scalar or validate vector length to shape ``(n,)``."""
    if np.isscalar(value):
        return np.full(n, value, dtype=dtype)
    arr = np.asarray(value, dtype=dtype)
    if arr.shape != (n,):
        raise ValueError(
            f"Expected shape ({n},), got {arr.shape}",
        )
    return arr


def get_pvp_dtype(xmltree: etree._ElementTree) -> np.dtype:
    """Extract PVP structured dtype from CRSD XML tree."""
    return sarkit.crsd.get_pvp_dtype(xmltree)


def get_ppp_dtype(xmltree: etree._ElementTree) -> np.dtype:
    """Extract PPP structured dtype from CRSD XML tree."""
    return sarkit.crsd.get_ppp_dtype(xmltree)


def build_pvp_array(
    channel: CRSDChannelLike,
    pvp_dtype: np.dtype,
    rcv_times: np.ndarray,
    rcv_positions: np.ndarray,
    rcv_velocities: np.ndarray,
    rcv_acx: np.ndarray,
    rcv_acy: np.ndarray,
    amp_sf: np.ndarray,
    ref_time_offset: float = 0.0,
    tx_pulse_index: Union[int, np.ndarray, None] = None,
    signal_valid: Union[int, np.ndarray] = 1,
    dfic0: Union[float, np.ndarray] = 0.0,
    fic_rate: Union[float, np.ndarray] = 0.0,
    dgrgc: Union[float, np.ndarray] = 0.0,
    rcv_eb: Union[np.ndarray, None] = None,
) -> np.ndarray:
    """Build PVP structured array for a single channel.

    Parameters
    ----------
    channel : CRSDChannelLike
        Channel metadata.
    pvp_dtype : np.dtype
        Structured dtype from the CRSD XML.
    rcv_times : np.ndarray
        Receive start times relative to CollectionRefTime (s).
    rcv_positions : np.ndarray
        Receiver ECEF positions, shape ``(N, 3)``.
    rcv_velocities : np.ndarray
        Receiver ECEF velocities, shape ``(N, 3)``.
    rcv_acx : np.ndarray
        Antenna coordinate X vectors, shape ``(N, 3)``.
    rcv_acy : np.ndarray
        Antenna coordinate Y vectors, shape ``(N, 3)``.
    amp_sf : np.ndarray
        Amplitude scale factors, shape ``(N,)``.
    ref_time_offset : float, optional
        Constant offset added to ``rcv_times``.
    tx_pulse_index : int or np.ndarray, optional
        Tx pulse index field. Defaults to sequential ``0..N-1``.
    signal_valid : int or np.ndarray, optional
        SIGNAL field values. Defaults to ``1`` for all vectors.
    dfic0 : float or np.ndarray, optional
        DFIC0 field values.
    fic_rate : float or np.ndarray, optional
        FICRate field values.
    dgrgc : float or np.ndarray, optional
        DGRGC field values.
    rcv_eb : np.ndarray, optional
        Receive electrical boresight angles, shape ``(N, 2)``.
        Defaults to zeros.
    """
    n = channel.num_vectors
    pvp = np.zeros(n, dtype=pvp_dtype)

    if ref_time_offset != 0.0:
        rcv_times = np.asarray(rcv_times, dtype=np.float64) + ref_time_offset

    rcv_int = np.floor(rcv_times).astype(np.int64)
    rcv_frac = rcv_times - rcv_int
    pvp["RcvStart"]["Int"] = rcv_int
    pvp["RcvStart"]["Frac"] = rcv_frac

    pvp["RcvPos"][:, 0] = rcv_positions[:, 0]
    pvp["RcvPos"][:, 1] = rcv_positions[:, 1]
    pvp["RcvPos"][:, 2] = rcv_positions[:, 2]

    pvp["RcvVel"][:, 0] = rcv_velocities[:, 0]
    pvp["RcvVel"][:, 1] = rcv_velocities[:, 1]
    pvp["RcvVel"][:, 2] = rcv_velocities[:, 2]

    pvp["FRCV1"] = channel.f0_ref - channel.bw_inst / 2.0
    pvp["FRCV2"] = channel.f0_ref + channel.bw_inst / 2.0

    pvp["RefPhi0"]["Int"] = 0
    pvp["RefPhi0"]["Frac"] = 0.0

    pvp["RefFreq"] = channel.f0_ref

    pvp["DFIC0"] = _to_vector(dfic0, n, np.float64)
    pvp["FICRate"] = _to_vector(fic_rate, n, np.float64)
    pvp["DGRGC"] = _to_vector(dgrgc, n, np.float64)

    pvp["RcvACX"][:, 0] = rcv_acx[:, 0]
    pvp["RcvACX"][:, 1] = rcv_acx[:, 1]
    pvp["RcvACX"][:, 2] = rcv_acx[:, 2]
    pvp["RcvACY"][:, 0] = rcv_acy[:, 0]
    pvp["RcvACY"][:, 1] = rcv_acy[:, 1]
    pvp["RcvACY"][:, 2] = rcv_acy[:, 2]

    if rcv_eb is None:
        pvp["RcvEB"][:, 0] = 0.0
        pvp["RcvEB"][:, 1] = 0.0
    else:
        if rcv_eb.shape != (n, 2):
            raise ValueError(
                f"Expected rcv_eb shape ({n}, 2), got {rcv_eb.shape}",
            )
        pvp["RcvEB"][:, 0] = rcv_eb[:, 0]
        pvp["RcvEB"][:, 1] = rcv_eb[:, 1]

    pvp["SIGNAL"] = _to_vector(signal_valid, n, np.int64)
    pvp["AmpSF"] = amp_sf

    if tx_pulse_index is None:
        pvp["TxPulseIndex"] = np.arange(n)
    else:
        pvp["TxPulseIndex"] = _to_vector(
            tx_pulse_index, n, np.int64,
        )

    return pvp


def build_ppp_array(
    channel: CRSDChannelLike,
    ppp_dtype: np.dtype,
    tx_times: np.ndarray,
    tx_positions: np.ndarray,
    tx_velocities: np.ndarray,
    tx_acx: np.ndarray,
    tx_acy: np.ndarray,
    tx_rad_int: np.ndarray,
    fx_response_index: Union[int, np.ndarray] = 0,
    tx_eb: Union[np.ndarray, None] = None,
) -> np.ndarray:
    """Build PPP structured array for a single Tx sequence/channel."""
    n = channel.num_vectors
    ppp = np.zeros(n, dtype=ppp_dtype)

    tx_int = np.floor(tx_times).astype(np.int64)
    tx_frac = tx_times - tx_int
    ppp["TxTime"]["Int"] = tx_int
    ppp["TxTime"]["Frac"] = tx_frac

    ppp["TxPos"][:, 0] = tx_positions[:, 0]
    ppp["TxPos"][:, 1] = tx_positions[:, 1]
    ppp["TxPos"][:, 2] = tx_positions[:, 2]

    ppp["TxVel"][:, 0] = tx_velocities[:, 0]
    ppp["TxVel"][:, 1] = tx_velocities[:, 1]
    ppp["TxVel"][:, 2] = tx_velocities[:, 2]

    ppp["FX1"] = channel.fx_freq0
    ppp["FX2"] = channel.fx_freq0 + channel.fx_bw
    ppp["TXmt"] = channel.tx_pulse_duration

    ppp["PhiX0"]["Int"] = 0
    ppp["PhiX0"]["Frac"] = 0.0

    ppp["FxFreq0"] = channel.fx_freq0
    ppp["FxRate"] = channel.fx_rate
    ppp["TxRadInt"] = tx_rad_int

    ppp["TxACX"][:, 0] = tx_acx[:, 0]
    ppp["TxACX"][:, 1] = tx_acx[:, 1]
    ppp["TxACX"][:, 2] = tx_acx[:, 2]
    ppp["TxACY"][:, 0] = tx_acy[:, 0]
    ppp["TxACY"][:, 1] = tx_acy[:, 1]
    ppp["TxACY"][:, 2] = tx_acy[:, 2]

    if tx_eb is None:
        ppp["TxEB"][:, 0] = 0.0
        ppp["TxEB"][:, 1] = 0.0
    else:
        if tx_eb.shape != (n, 2):
            raise ValueError(
                f"Expected tx_eb shape ({n}, 2), got {tx_eb.shape}",
            )
        ppp["TxEB"][:, 0] = tx_eb[:, 0]
        ppp["TxEB"][:, 1] = tx_eb[:, 1]

    ppp["FxResponseIndex"] = _to_vector(
        fx_response_index, n, np.int64,
    )

    return ppp


def quantize_to_ci2(
    signal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize complex64 signal data to CI2 (int8 I/Q pairs)."""
    num_vectors = signal.shape[0]
    ci2_dtype = np.dtype([("real", "i1"), ("imag", "i1")])
    ci2 = np.empty(signal.shape, dtype=ci2_dtype)
    amp_sf = np.empty(num_vectors, dtype=np.float64)

    real = signal.real
    imag = signal.imag

    for i in range(num_vectors):
        row_real = real[i]
        row_imag = imag[i]
        max_abs = max(
            np.max(np.abs(row_real)),
            np.max(np.abs(row_imag)),
        )
        if max_abs == 0.0:
            amp_sf[i] = 1.0
            ci2[i]["real"] = 0
            ci2[i]["imag"] = 0
        else:
            scale = max_abs / 127.0
            amp_sf[i] = scale
            ci2[i]["real"] = np.clip(
                np.round(row_real / scale), -128, 127,
            ).astype(np.int8)
            ci2[i]["imag"] = np.clip(
                np.round(row_imag / scale), -128, 127,
            ).astype(np.int8)

    return ci2, amp_sf
