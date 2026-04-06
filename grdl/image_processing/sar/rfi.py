# -*- coding: utf-8 -*-
"""
RFI Detection - Spectral methods for radio-frequency interference detection.

Provides four detection functions operating on Compensated Phase History
Data (CPHD) signal arrays.  Each function accepts a 2-D complex
``(n_az, n_rng)`` array from ``CPHDReader.read_full()`` and returns a
typed dataclass result.

Signal domain: all four functions expect the input to be in the
range-**frequency** × slow-time domain, matching the CPHD FX standard
(``DomainType=FX``).  The GRDL ``CPHDWriter`` and
``experiments/nisar/main.py`` both produce this format: step 3b' (Az-IFFT)
restores the azimuth axis to slow-time and step 3e (Rng-IFFT) is
omitted, so the range axis remains in the frequency domain.  Callers
should pass only valid (non-zero) pulse rows; leading zero-padded rows
from the NISAR RSLC-to-CPHD conversion degrade the noise-floor estimate.

Methods
-------
detect_mean_psd
    Average power spectral density test.  Flags range-frequency bins
    whose mean power exceeds the spectral noise floor by a configurable
    dB threshold.

detect_per_pulse
    Per-pulse spectral outlier detection.  Flags bins that exceed a
    running median envelope in each individual pulse, then aggregates
    flag counts and fractions across pulses.

detect_cross_pol
    Cross-polarisation power ratio test.  Flags bins where the co-pol /
    cross-pol power ratio deviates from the scene median by a
    configurable dB threshold.  Accepts a dict mapping polarisation
    name to signal array.

detect_eigenvalue
    Eigenvalue decomposition of per-block covariance matrices.  Flags
    range-frequency bins associated with a low-rank RFI subspace by
    thresholding the dominant-to-median eigenvalue ratio.

Dependencies
------------
scipy

Author
------
Ava Courtney
courtney-ava@zai.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-04-06

Modified
--------
2026-04-06
"""

# Standard library
import logging
from dataclasses import dataclass, field
from typing import Dict, List

# Third-party
import numpy as np
from scipy.ndimage import median_filter as _scipy_median_filter
from scipy.linalg import eigh as _scipy_eigh

# GRDL internal
from grdl.exceptions import ValidationError

logger = logging.getLogger(__name__)

# Small constant added before log10 to avoid log(0).
_EPS: float = 1e-30

# ===================================================================
# Result dataclasses
# ===================================================================

@dataclass
class MeanPSDResult:
    """Result of the mean power spectral density test.

    Parameters
    ----------
    flagged_bins : np.ndarray
        Indices of flagged range-frequency bins, shape ``(n_flagged,)``.
    bin_powers_db : np.ndarray
        Mean power in dB at each flagged bin, shape ``(n_flagged,)``.
    excess_db : np.ndarray
        Power excess above noise floor in dB at each flagged bin,
        shape ``(n_flagged,)``.
    mean_psd_db : np.ndarray
        Mean power spectrum in dB across all range-frequency bins,
        shape ``(n_rng,)``.
    noise_floor_db : float
        Estimated noise floor in dB (median of ``mean_psd_db``).
    """

    flagged_bins: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.intp))
    bin_powers_db: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    excess_db: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    mean_psd_db: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    noise_floor_db: float = 0.0


@dataclass
class PerPulseResult:
    """Result of the per-pulse spectral outlier detection.

    Parameters
    ----------
    flag_counts : np.ndarray
        Number of pulses in which each range-frequency bin was flagged,
        shape ``(n_rng,)``.
    flag_fraction : np.ndarray
        Fraction of pulses in which each bin was flagged, shape
        ``(n_rng,)``.  Values in ``[0, 1]``.
    mask : np.ndarray
        2-D boolean flag mask, shape ``(n_rng, n_az)``.  ``True``
        where a bin-pulse combination exceeded the spectral envelope.
    """

    flag_counts: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.intp))
    flag_fraction: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))


@dataclass
class CrossPolResult:
    """Result of the cross-polarisation power ratio test.

    All dicts are keyed by ratio name, e.g. ``'HH/HV'``.

    Parameters
    ----------
    flagged_bins : Dict[str, np.ndarray]
        Flagged range-frequency bin indices per ratio, each shape
        ``(n_flagged_ratio,)``.
    ratio_anomaly_db : Dict[str, np.ndarray]
        Absolute ratio anomaly in dB at each flagged bin, per ratio.
        Shape matches the corresponding ``flagged_bins`` entry.
    mean_ratio_db : Dict[str, np.ndarray]
        Full mean co/cross-pol power ratio profile in dB, per ratio,
        shape ``(n_rng,)``.
    """

    flagged_bins: Dict[str, np.ndarray] = field(default_factory=dict)
    ratio_anomaly_db: Dict[str, np.ndarray] = field(default_factory=dict)
    mean_ratio_db: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class EigenvalueResult:
    """Result of the eigenvalue decomposition test.

    Parameters
    ----------
    eigenvalue_profiles : np.ndarray
        Per-block eigenvalue vectors in ascending order, shape
        ``(n_blocks, block_size)``.
    flagged_bins : np.ndarray
        Union of flagged range-frequency bin indices across all blocks,
        shape ``(n_flagged,)``.
    rfi_rank : np.ndarray
        Estimated RFI subspace rank per block (number of eigenvalues
        exceeding the dominant-ratio threshold), shape ``(n_blocks,)``.
    dominant_ratio : np.ndarray
        Ratio of the largest eigenvalue to the median eigenvalue per
        block, shape ``(n_blocks,)``.
    """

    eigenvalue_profiles: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float64)
    )
    flagged_bins: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.intp))
    rfi_rank: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.intp))
    dominant_ratio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))


# ===================================================================
# Input validation helpers
# ===================================================================

def _validate_signal(signal: np.ndarray, name: str = 'signal') -> None:
    """Raise ValidationError if *signal* is not a 2-D complex array."""
    if signal.ndim != 2:
        raise ValidationError(
            f"{name} must be 2-D (n_az, n_rng), got {signal.ndim}-D "
            f"with shape {signal.shape}"
        )
    if not np.iscomplexobj(signal):
        raise ValidationError(
            f"{name} must be complex-valued, got dtype={signal.dtype}"
        )


def _validate_filter_size(filter_size: int) -> None:
    """Raise ValidationError if *filter_size* is not a valid odd integer ≥ 3."""
    if filter_size < 3:
        raise ValidationError(
            f"filter_size must be ≥ 3, got {filter_size}"
        )
    if filter_size % 2 == 0:
        raise ValidationError(
            f"filter_size must be odd, got {filter_size}"
        )


def _validate_block_size(block_size: int) -> None:
    """Raise ValidationError if *block_size* is less than 2."""
    if block_size < 2:
        raise ValidationError(
            f"block_size must be ≥ 2, got {block_size}"
        )


def _validate_signals_dict(signals: Dict[str, np.ndarray]) -> None:
    """Raise ValidationError if *signals* has fewer than 2 entries."""
    if len(signals) < 2:
        raise ValidationError(
            f"detect_cross_pol requires at least 2 polarisations, "
            f"got {len(signals)}: {list(signals.keys())}"
        )
    for pol, sig in signals.items():
        _validate_signal(sig, name=f"signals['{pol}']")


# ===================================================================
# Detector functions
# ===================================================================

def detect_mean_psd(
    signal: np.ndarray,
    *,
    threshold_db: float = 10.0,
) -> MeanPSDResult:
    """Mean power spectral density RFI test.

    Averages power across all slow-time pulses per range-frequency bin.
    Estimates the spectral noise floor as the median of the mean PSD in
    dB.  Flags any bin whose mean power exceeds the noise floor by more
    than *threshold_db*.

    Parameters
    ----------
    signal : np.ndarray
        Complex FX phase history data, shape ``(n_az, n_rng)``.
        Axis 0 is slow-time; axis 1 is range-frequency (CPHD FX domain).
        Pass only valid (non-zero) pulse rows for accurate results;
        leading zero-padded rows from NISAR RSLC conversion inflate the
        noise floor estimate.
    threshold_db : float
        Detection threshold in dB above the noise floor.  Default 10.0.

    Returns
    -------
    MeanPSDResult
        ``flagged_bins`` contains the indices of all bins exceeding the
        threshold.  ``mean_psd_db`` and ``noise_floor_db`` are always
        populated.

    Raises
    ------
    ValidationError
        If *signal* is not 2-D or not complex.
    """
    _validate_signal(signal)
    logger.debug(
        "detect_mean_psd: shape=%s threshold_db=%.1f", signal.shape, threshold_db
    )

    rng_freq = signal.astype(np.complex64)
    mean_pwr = np.mean(np.abs(rng_freq) ** 2, axis=0)  # (n_rng,)
    mean_psd_db = 10.0 * np.log10(mean_pwr + _EPS)
    noise_floor_db = float(np.median(mean_psd_db))
    excess_db = mean_psd_db - noise_floor_db

    flag_mask = excess_db > threshold_db
    flagged_bins = np.where(flag_mask)[0].astype(np.intp)

    logger.debug(
        "detect_mean_psd: noise_floor=%.1f dB, n_flagged=%d",
        noise_floor_db,
        len(flagged_bins),
    )

    return MeanPSDResult(
        flagged_bins=flagged_bins,
        bin_powers_db=mean_psd_db[flagged_bins],
        excess_db=excess_db[flagged_bins],
        mean_psd_db=mean_psd_db,
        noise_floor_db=noise_floor_db,
    )


def detect_per_pulse(
    signal: np.ndarray,
    *,
    threshold_db: float = 6.0,
    filter_size: int = 51,
) -> PerPulseResult:
    """Per-pulse spectral outlier RFI detection.

    For each slow-time pulse, fits a smooth spectral envelope using a
    running median filter along the range-frequency axis.  Flags bins
    where the per-pulse power exceeds the envelope by more than
    *threshold_db*.  Results are aggregated to a per-bin flag count
    and fraction across all pulses, and a 2-D mask.

    Parameters
    ----------
    signal : np.ndarray
        Complex FX phase history data, shape ``(n_az, n_rng)``.
        Axis 0 is slow-time; axis 1 is range-frequency (CPHD FX domain).
        Pass only valid (non-zero) pulse rows for accurate results.
    threshold_db : float
        Detection threshold in dB above the local spectral envelope.
        Default 6.0.
    filter_size : int
        Half-width of the running median filter along the range-
        frequency axis.  Must be an odd integer ≥ 3.  Larger values
        produce a smoother envelope, reducing sensitivity to spectrally
        broad RFI.  Default 51.

    Returns
    -------
    PerPulseResult
        ``mask`` has shape ``(n_rng, n_az)``.  ``flag_counts`` and
        ``flag_fraction`` have shape ``(n_rng,)``.

    Raises
    ------
    ValidationError
        If *signal* is not 2-D or not complex, or if *filter_size* is
        even or less than 3.
    """
    _validate_signal(signal)
    _validate_filter_size(filter_size)
    logger.debug(
        "detect_per_pulse: shape=%s threshold_db=%.1f filter_size=%d",
        signal.shape, threshold_db, filter_size,
    )

    n_az, n_rng = signal.shape
    rng_freq = signal.astype(np.complex64)                        # (n_az, n_rng)
    power_db = 10.0 * np.log10(np.abs(rng_freq) ** 2 + _EPS)    # (n_az, n_rng)

    # Vectorised: median filter applied to all pulses at once along range axis.
    envelope = _scipy_median_filter(power_db, size=(1, filter_size), mode='reflect')

    flag_mask_az_rng = (power_db - envelope) > threshold_db      # (n_az, n_rng)
    mask = flag_mask_az_rng.T                                     # (n_rng, n_az)

    flag_counts = mask.sum(axis=1).astype(np.intp)
    flag_fraction = flag_counts / max(n_az, 1)

    logger.debug(
        "detect_per_pulse: n_bins_flagged=%d (≥1 pulse)",
        int((flag_counts > 0).sum()),
    )

    return PerPulseResult(
        flag_counts=flag_counts,
        flag_fraction=flag_fraction,
        mask=mask,
    )


def detect_cross_pol(
    signals: Dict[str, np.ndarray],
    *,
    threshold_db: float = 5.0,
) -> CrossPolResult:
    """Cross-polarisation power ratio RFI test.

    For each available co/cross-pol pair, computes the mean PSD for
    each polarisation channel and flags range-frequency bins where the
    power ratio deviates from the scene median ratio by more than
    *threshold_db*.

    Ratio pairs evaluated (when both polarisations are present):
    ``'HH/HV'``, ``'VV/VH'``, ``'HH/VV'``.  Missing pairs are
    silently omitted from the result.

    Parameters
    ----------
    signals : Dict[str, np.ndarray]
        Mapping from polarisation name (``'HH'``, ``'HV'``, ``'VH'``,
        ``'VV'``) to complex FX phase history array, each shape
        ``(n_az, n_rng)`` (slow-time × range-frequency).  Must contain
        at least 2 entries.  Pass only valid (non-zero) pulse rows for
        accurate results.
    threshold_db : float
        Detection threshold in dB deviation from the scene median
        ratio.  Default 5.0.

    Returns
    -------
    CrossPolResult
        Populated for all ratio pairs where both polarisations are
        present.  ``mean_ratio_db`` is always included for every
        available ratio.

    Raises
    ------
    ValidationError
        If *signals* contains fewer than 2 entries, or if any signal
        array is not 2-D complex.
    """
    _validate_signals_dict(signals)
    logger.debug(
        "detect_cross_pol: pols=%s threshold_db=%.1f",
        list(signals.keys()),
        threshold_db,
    )

    # Compute mean PSD in dB for every provided polarisation.
    mean_psd_db: Dict[str, np.ndarray] = {}
    for pol, sig in signals.items():
        rng_freq = sig.astype(np.complex64)
        mean_pwr = np.mean(np.abs(rng_freq) ** 2, axis=0)
        mean_psd_db[pol] = 10.0 * np.log10(mean_pwr + _EPS)

    _RATIO_PAIRS: List[tuple] = [('HH', 'HV'), ('VV', 'VH'), ('HH', 'VV')]

    flagged_bins: Dict[str, np.ndarray] = {}
    ratio_anomaly_db: Dict[str, np.ndarray] = {}
    mean_ratio_db: Dict[str, np.ndarray] = {}

    for co, cross in _RATIO_PAIRS:
        if co not in mean_psd_db or cross not in mean_psd_db:
            continue
        ratio_key = f'{co}/{cross}'
        ratio_db = mean_psd_db[co] - mean_psd_db[cross]
        scene_median = float(np.median(ratio_db))
        anomaly = np.abs(ratio_db - scene_median)
        flag_mask = anomaly > threshold_db
        idx = np.where(flag_mask)[0].astype(np.intp)

        mean_ratio_db[ratio_key] = ratio_db
        flagged_bins[ratio_key] = idx
        ratio_anomaly_db[ratio_key] = anomaly[idx]

        logger.debug(
            "detect_cross_pol: %s median=%.1f dB, n_flagged=%d",
            ratio_key,
            scene_median,
            len(idx),
        )

    return CrossPolResult(
        flagged_bins=flagged_bins,
        ratio_anomaly_db=ratio_anomaly_db,
        mean_ratio_db=mean_ratio_db,
    )


def detect_eigenvalue(
    signal: np.ndarray,
    *,
    block_size: int = 32,
    threshold: float = 10.0,
) -> EigenvalueResult:
    """Eigenvalue decomposition RFI detection.

    Partitions the slow-time axis into non-overlapping blocks of
    *block_size* pulses.  For each block, computes the Hermitian sample
    covariance matrix ``C`` of the pulse vectors over the
    range-frequency dimension::

        C[b] = X[b] @ X[b].conj().T / n_rng

    where ``X[b]`` has shape ``(block_size, n_rng)`` and contains
    *block_size* consecutive slow-time pulses.  The dominant eigenvalue
    of ``C[b]`` relative to the median eigenvalue indicates the presence
    of a low-rank (narrowband) RFI subspace.

    Range-frequency bins associated with the RFI subspace are identified
    by projecting the spectra onto the dominant eigenvectors and flagging
    bins whose projection power exceeds the median across all bins.

    Parameters
    ----------
    signal : np.ndarray
        Complex FX phase history data, shape ``(n_az, n_rng)``.
        Axis 0 is slow-time; axis 1 is range-frequency (CPHD FX domain).
        Pass only valid (non-zero) pulse rows for accurate results.
        If ``block_size ≥ n_az``, the entire signal is treated as a
        single block.
    block_size : int
        Number of slow-time pulses per covariance block.  Must be ≥ 2.
        Default 32.
    threshold : float
        Dominant-to-median eigenvalue ratio above which a block is
        classified as containing RFI.  Default 10.0.

    Returns
    -------
    EigenvalueResult
        ``eigenvalue_profiles`` has shape ``(n_blocks, block_size)``.
        ``flagged_bins`` is a sorted union of flagged indices across
        all blocks.  ``rfi_rank`` and ``dominant_ratio`` have shape
        ``(n_blocks,)``.

    Raises
    ------
    ValidationError
        If *signal* is not 2-D or not complex, or if *block_size* < 2.
    """
    _validate_signal(signal)
    _validate_block_size(block_size)
    logger.debug(
        "detect_eigenvalue: shape=%s block_size=%d threshold=%.1f",
        signal.shape,
        block_size,
        threshold,
    )

    n_az, n_rng = signal.shape
    # Clamp block_size to n_az so a single block covers the whole array
    # when block_size >= n_az.
    effective_block = min(block_size, n_az)
    n_blocks = max(n_az // effective_block, 1)

    rng_freq = signal.astype(np.complex64)                       # (n_az, n_rng) already FX

    eigenvalue_profiles = np.zeros((n_blocks, effective_block), dtype=np.float64)
    dominant_ratio = np.zeros(n_blocks, dtype=np.float64)
    rfi_rank = np.zeros(n_blocks, dtype=np.intp)
    all_flagged: List[np.ndarray] = []

    for b in range(n_blocks):
        row_start = b * effective_block
        row_end = row_start + effective_block
        X = rng_freq[row_start:row_end, :]                   # (block_size, n_rng)

        # Hermitian covariance of Doppler bins over range-frequency content.
        C = (X @ X.conj().T).real / n_rng                   # (block_size, block_size)

        eigvals, eigvecs = _scipy_eigh(C)                    # ascending order

        med_eig = float(np.median(eigvals))
        if med_eig <= 0.0:
            dom_ratio = 1.0
        else:
            dom_ratio = float(eigvals[-1]) / med_eig

        eigenvalue_profiles[b] = eigvals
        dominant_ratio[b] = dom_ratio

        rank = int(np.sum(eigvals / max(med_eig, _EPS) > threshold))
        rfi_rank[b] = rank

        if rank > 0:
            # Project range-freq spectra columns onto top-rank eigenvectors.
            top_vecs = eigvecs[:, -rank:]                    # (block_size, rank)
            proj = np.abs(top_vecs.conj().T @ X) ** 2       # (rank, n_rng)
            proj_power = proj.sum(axis=0)                    # (n_rng,)
            proj_median = float(np.median(proj_power))
            if proj_median > 0.0:
                idx = np.where(proj_power / proj_median > threshold)[0].astype(np.intp)
                all_flagged.append(idx)

        logger.debug(
            "detect_eigenvalue: block %d/%d dom_ratio=%.1f rank=%d",
            b + 1,
            n_blocks,
            dom_ratio,
            rank,
        )

    if all_flagged:
        flagged_bins = np.unique(np.concatenate(all_flagged)).astype(np.intp)
    else:
        flagged_bins = np.array([], dtype=np.intp)

    return EigenvalueResult(
        eigenvalue_profiles=eigenvalue_profiles,
        flagged_bins=flagged_bins,
        rfi_rank=rfi_rank,
        dominant_ratio=dominant_ratio,
    )
