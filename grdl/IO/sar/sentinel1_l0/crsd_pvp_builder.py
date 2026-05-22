# -*- coding: utf-8 -*-
"""
CRSD PVP/PPP Array Builder for Sentinel-1 Level 0 Data.

Constructs structured numpy arrays for Per-Vector Parameters (PVP)
and Per-Pulse Parameters (PPP) from Sentinel-1 L0 burst metadata,
orbit interpolation, and antenna geometry.

The dtypes are inferred from the CRSD XML tree via ``sarkit.crsd``,
ensuring exact binary compatibility with the CRSD file format.

Dependencies
------------
sarkit

Author
------
James Fritz
jpfritz@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-05-22

Modified
--------
2026-05-22
"""

# Standard library
import logging
from typing import Tuple

# Third-party
import numpy as np
from lxml import etree

try:
    import sarkit.crsd
except ImportError as exc:
    raise ImportError(
        "sarkit is required for CRSD writing. "
        "Install with: pip install sarkit"
    ) from exc

# GRDL internal
from grdl.IO.sar.sentinel1_l0.crsd_metadata_builder import (
    BurstChannelInfo,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Dtype helpers
# =====================================================================


def get_pvp_dtype(xmltree: etree._ElementTree) -> np.dtype:
    """Extract PVP structured dtype from CRSD XML tree.

    Parameters
    ----------
    xmltree : lxml.etree.ElementTree
        CRSD metadata XML tree.

    Returns
    -------
    np.dtype
        Structured dtype for PVP records.
    """
    return sarkit.crsd.get_pvp_dtype(xmltree)


def get_ppp_dtype(xmltree: etree._ElementTree) -> np.dtype:
    """Extract PPP structured dtype from CRSD XML tree.

    Parameters
    ----------
    xmltree : lxml.etree.ElementTree
        CRSD metadata XML tree.

    Returns
    -------
    np.dtype
        Structured dtype for PPP records.
    """
    return sarkit.crsd.get_ppp_dtype(xmltree)


# =====================================================================
# PVP Builder
# =====================================================================


def build_pvp_array(
    channel: BurstChannelInfo,
    pvp_dtype: np.dtype,
    rcv_times: np.ndarray,
    rcv_positions: np.ndarray,
    rcv_velocities: np.ndarray,
    rcv_acx: np.ndarray,
    rcv_acy: np.ndarray,
    amp_sf: np.ndarray,
    ref_time_offset: float = 0.0,
) -> np.ndarray:
    """Build PVP structured array for a single channel/burst.

    Parameters
    ----------
    channel : BurstChannelInfo
        Channel metadata (frequencies, bandwidth, etc.).
    pvp_dtype : np.dtype
        Structured dtype from the CRSD XML.
    rcv_times : np.ndarray
        Receive start times relative to CollectionRefTime (s).
        Shape ``(N,)``.
    rcv_positions : np.ndarray
        Receiver ECEF positions at receive time.
        Shape ``(N, 3)``.
    rcv_velocities : np.ndarray
        Receiver ECEF velocities at receive time.
        Shape ``(N, 3)``.
    rcv_acx : np.ndarray
        Antenna coordinate frame X unit vectors.
        Shape ``(N, 3)``.
    rcv_acy : np.ndarray
        Antenna coordinate frame Y unit vectors.
        Shape ``(N, 3)``.
    amp_sf : np.ndarray
        Amplitude scale factors (FDBAQ reconstruction) per vector.
        Shape ``(N,)``.
    ref_time_offset : float
        Offset to add to times (for burst-relative timing).

    Returns
    -------
    np.ndarray
        Structured array of shape ``(N,)`` with ``pvp_dtype``.
    """
    n = channel.num_vectors
    pvp = np.zeros(n, dtype=pvp_dtype)

    # Receive start time (int + frac seconds)
    rcv_int = np.floor(rcv_times).astype(np.int64)
    rcv_frac = rcv_times - rcv_int
    pvp["RcvStart"]["Int"] = rcv_int
    pvp["RcvStart"]["Frac"] = rcv_frac

    # Receiver position (sub-array shape (3,))
    pvp["RcvPos"][:, 0] = rcv_positions[:, 0]
    pvp["RcvPos"][:, 1] = rcv_positions[:, 1]
    pvp["RcvPos"][:, 2] = rcv_positions[:, 2]

    # Receiver velocity
    pvp["RcvVel"][:, 0] = rcv_velocities[:, 0]
    pvp["RcvVel"][:, 1] = rcv_velocities[:, 1]
    pvp["RcvVel"][:, 2] = rcv_velocities[:, 2]

    # Receive frequency extents
    frcv1 = channel.f0_ref - channel.bw_inst / 2.0
    frcv2 = channel.f0_ref + channel.bw_inst / 2.0
    pvp["FRCV1"] = frcv1
    pvp["FRCV2"] = frcv2

    # Reference phase — zero (no compensation)
    pvp["RefPhi0"]["Int"] = 0
    pvp["RefPhi0"]["Frac"] = 0.0

    # Reference frequency
    pvp["RefFreq"] = channel.f0_ref

    # Compensation fields — all zero
    pvp["DFIC0"] = 0.0
    pvp["FICRate"] = 0.0
    pvp["DGRGC"] = 0.0

    # Antenna coordinate frame vectors
    pvp["RcvACX"][:, 0] = rcv_acx[:, 0]
    pvp["RcvACX"][:, 1] = rcv_acx[:, 1]
    pvp["RcvACX"][:, 2] = rcv_acx[:, 2]
    pvp["RcvACY"][:, 0] = rcv_acy[:, 0]
    pvp["RcvACY"][:, 1] = rcv_acy[:, 1]
    pvp["RcvACY"][:, 2] = rcv_acy[:, 2]

    # Electrical boresight — zero (no steering correction)
    pvp["RcvEB"][:, 0] = 0.0
    pvp["RcvEB"][:, 1] = 0.0

    # Signal validity — 1 = valid
    pvp["SIGNAL"] = 1

    # Amplitude scale factor
    pvp["AmpSF"] = amp_sf

    # Tx pulse index — sequential within burst
    pvp["TxPulseIndex"] = np.arange(n)

    return pvp


# =====================================================================
# PPP Builder
# =====================================================================


def build_ppp_array(
    channel: BurstChannelInfo,
    ppp_dtype: np.dtype,
    tx_times: np.ndarray,
    tx_positions: np.ndarray,
    tx_velocities: np.ndarray,
    tx_acx: np.ndarray,
    tx_acy: np.ndarray,
    tx_rad_int: np.ndarray,
) -> np.ndarray:
    """Build PPP structured array for a single TxSequence/burst.

    Parameters
    ----------
    channel : BurstChannelInfo
        Channel metadata (waveform parameters).
    ppp_dtype : np.dtype
        Structured dtype from the CRSD XML.
    tx_times : np.ndarray
        Transmit times relative to CollectionRefTime (s).
        Shape ``(N,)``.
    tx_positions : np.ndarray
        Transmitter ECEF positions at transmit time.
        Shape ``(N, 3)``.
    tx_velocities : np.ndarray
        Transmitter ECEF velocities at transmit time.
        Shape ``(N, 3)``.
    tx_acx : np.ndarray
        Antenna coordinate frame X unit vectors.
        Shape ``(N, 3)``.
    tx_acy : np.ndarray
        Antenna coordinate frame Y unit vectors.
        Shape ``(N, 3)``.
    tx_rad_int : np.ndarray
        Transmit radiated intensity per pulse.
        Shape ``(N,)``.

    Returns
    -------
    np.ndarray
        Structured array of shape ``(N,)`` with ``ppp_dtype``.
    """
    n = channel.num_vectors
    ppp = np.zeros(n, dtype=ppp_dtype)

    # Transmit time (int + frac)
    tx_int = np.floor(tx_times).astype(np.int64)
    tx_frac = tx_times - tx_int
    ppp["TxTime"]["Int"] = tx_int
    ppp["TxTime"]["Frac"] = tx_frac

    # Transmit position (sub-array shape (3,))
    ppp["TxPos"][:, 0] = tx_positions[:, 0]
    ppp["TxPos"][:, 1] = tx_positions[:, 1]
    ppp["TxPos"][:, 2] = tx_positions[:, 2]

    # Transmit velocity
    ppp["TxVel"][:, 0] = tx_velocities[:, 0]
    ppp["TxVel"][:, 1] = tx_velocities[:, 1]
    ppp["TxVel"][:, 2] = tx_velocities[:, 2]

    # Frequency extents — chirp start and end frequencies
    ppp["FX1"] = channel.fx_freq0
    ppp["FX2"] = channel.fx_freq0 + channel.fx_bw

    # Pulse duration
    ppp["TXmt"] = channel.tx_pulse_duration

    # Phase — zero (no compensation)
    ppp["PhiX0"]["Int"] = 0
    ppp["PhiX0"]["Frac"] = 0.0

    # Chirp parameters
    ppp["FxFreq0"] = channel.fx_freq0
    ppp["FxRate"] = channel.fx_rate

    # Radiated intensity
    ppp["TxRadInt"] = tx_rad_int

    # Antenna coordinate frame
    ppp["TxACX"][:, 0] = tx_acx[:, 0]
    ppp["TxACX"][:, 1] = tx_acx[:, 1]
    ppp["TxACX"][:, 2] = tx_acx[:, 2]
    ppp["TxACY"][:, 0] = tx_acy[:, 0]
    ppp["TxACY"][:, 1] = tx_acy[:, 1]
    ppp["TxACY"][:, 2] = tx_acy[:, 2]

    # Electrical boresight
    ppp["TxEB"][:, 0] = 0.0
    ppp["TxEB"][:, 1] = 0.0

    # Frequency response index — all 0 (single flat response)
    ppp["FxResponseIndex"] = 0

    return ppp


# =====================================================================
# Signal quantization
# =====================================================================


def quantize_to_ci2(
    signal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize complex64 signal data to CI2 (int8 I/Q pairs).

    Parameters
    ----------
    signal : np.ndarray
        Complex64 decoded I/Q data. Shape ``(num_vectors, num_samples)``.

    Returns
    -------
    ci2 : np.ndarray
        CI2 signal array with dtype
        ``[('real', 'i1'), ('imag', 'i1')]``.
        Shape ``(num_vectors, num_samples)``.
    amp_sf : np.ndarray
        Per-vector amplitude scale factor. Shape ``(num_vectors,)``.
        ``true_amplitude = ci2_value * amp_sf``.
    """
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
