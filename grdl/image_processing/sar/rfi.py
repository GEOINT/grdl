# -*- coding: utf-8 -*-
"""
RFI Detection and Mitigation - Spectral methods for radio-frequency
interference detection and removal.

Provides four detection functions and two mitigation functions operating
on Compensated Phase History Data (CPHD) signal arrays.  Each detection
function accepts a 2-D complex ``(n_az, n_rng)`` array from
``CPHDReader.read_full()`` and returns a typed dataclass result.

Signal domain: all detection functions expect the input to be in the
range-**frequency** × slow-time domain, matching the CPHD FX standard
(``DomainType=FX``).  The GRDL ``CPHDWriter`` and
``experiments/nisar/main.py`` both produce this format: step 3b' (Az-IFFT)
restores the azimuth axis to slow-time and step 3e (Rng-IFFT) is
omitted, so the range axis remains in the frequency domain.  Callers
should pass only valid (non-zero) pulse rows; leading zero-padded rows
from the NISAR RSLC-to-CPHD conversion degrade the noise-floor estimate.
See ``CPHDPVP.trim_to_valid`` for a helper that strips such rows.

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

mitigate_notch
    Zero-fill flagged range-frequency bins (spectral notch filter).
    Fast and simple; preferred when flagged bins are narrow and isolated.

mitigate_interpolate
    Replace flagged bins by linear interpolation from their nearest
    unflagged neighbours, independently per slow-time pulse.  Avoids
    the spectral holes introduced by ``mitigate_notch``.

mitigate_vsff
    Variable Space Frequency Filter.  Two-phase algorithm that localises
    narrowband RFI in the spectral and azimuth dimensions, then applies
    a 2-D Band-Stop Filter whose notch depth per pulse is proportional
    to the estimated SAR antenna gain toward the RFI source.  Handles
    both FX-domain and TOA-domain CPHD inputs automatically.

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
2026-04-09
"""

# Standard library
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Third-party
import numpy as np
from scipy.ndimage import median_filter as _scipy_median_filter
from scipy.linalg import eigh as _scipy_eigh

# GRDL internal
from grdl.exceptions import ValidationError
from grdl.IO.models.cphd import CPHDMetadata

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


@dataclass
class VSFFResult:
    """Result of the Variable Space Frequency Filter RFI mitigation.

    Parameters
    ----------
    mitigated : np.ndarray
        RFI-mitigated signal, same shape and domain as the input to
        ``mitigate_vsff``, shape ``(n_az, n_rng)``.
    rfi_center_bin : int
        Estimated range-frequency bin index of the RFI centre frequency,
        identified from the mean PSD excess above the noise floor in
        Phase 1.
    rfi_bw_bins : int
        Estimated RFI 3 dB half-power bandwidth in range-frequency bins,
        derived from the 3 dB limits of the mean PSD peak in Phase 1.
    rfi_azimuth_idx : int
        Estimated slow-time (azimuth) index of the RFI source, found as
        the peak of the BPF-isolated power profile in Phase 1.
    attenuation_coeffs : np.ndarray
        ABSF(η) attenuation weighting array, shape ``(n_az,)``.  Values
        in ``[Amax, 1]``; 1 at pulses with maximum antenna gain (deepest
        notch), decreasing to ``Amax`` at minimum-gain pulses.  The
        applied BSF coefficient in the stop-band is ``1 - ABSF(η)``.
    mean_psd_db : np.ndarray
        Mean power spectral density profile in dB over slow-time, used
        for Phase 1 spectral parameter estimation, shape ``(n_rng,)``.
    gain_profile : np.ndarray
        Estimated antenna gain proxy G(η): integrated BPF-band power per
        slow-time pulse, shape ``(n_az,)``.  Used as the gain model for
        the variable-attenuation BSF in Phase 2.
    az_lo : int
        Lower slow-time (azimuth) index of the estimated 3 dB antenna
        beamwidth window.  The BSF is applied only to pulses in the
        closed interval ``[az_lo, az_hi]``.
    az_hi : int
        Upper slow-time (azimuth) index of the estimated 3 dB antenna
        beamwidth window.  Pulses outside ``[az_lo, az_hi]`` are passed
        through the filter unchanged.
    """

    mitigated: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.complex64)
    )
    rfi_center_bin: int = 0
    rfi_bw_bins: int = 0
    rfi_azimuth_idx: int = 0
    attenuation_coeffs: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    mean_psd_db: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    gain_profile: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    az_lo: int = 0
    az_hi: int = 0


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


def _check_zero_pulse_fraction(
    signal: np.ndarray,
    name: str = 'signal',
) -> None:
    """Warn if more than 80 % of slow-time pulses are all-zero.

    Zero-padded pulse rows distort noise-floor and covariance estimates.
    Callers should strip leading/trailing zero rows before calling any
    detector.  See ``CPHDPVP.trim_to_valid`` or
    ``CPHDPVP.first_valid_pulse`` for helpers that do this.
    """
    valid_count = int(np.any(signal != 0, axis=1).sum())
    zero_frac = 1.0 - valid_count / max(signal.shape[0], 1)
    if zero_frac > 0.80:
        logger.warning(
            "%s: %.0f%% of pulses are all-zero — pass only valid "
            "(non-zero) pulse rows for accurate noise-floor estimates "
            "(see CPHDPVP.trim_to_valid)",
            name,
            zero_frac * 100,
        )


def _check_domain_type(metadata: Optional[CPHDMetadata]) -> None:
    """Warn if *metadata* indicates a non-FX (time-domain) signal.

    Reads ``metadata.global_params.domain_type`` from the
    ``CPHDMetadata`` object returned by ``CPHDReader``.  The RFI
    detectors assume the range axis is in the frequency domain
    (``DomainType=FX``); TOA-domain data must be Fourier-transformed
    along the range axis before detection.
    """
    if metadata is None:
        return
    gp = metadata.global_params
    if gp is None:
        return
    domain = gp.domain_type
    if domain is not None and domain.upper() != 'FX':
        logger.warning(
            "CPHD DomainType is '%s'; all detectors require FX-domain "
            "(range-frequency) input.  Transform the range axis to the "
            "frequency domain before calling this function.",
            domain,
        )


# ===================================================================
# Detector functions
# ===================================================================

def detect_mean_psd(
    signal: np.ndarray,
    *,
    threshold_db: float = 10.0,
    metadata: Optional[CPHDMetadata] = None,
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
    metadata : CPHDMetadata, optional
        CPHD file metadata.  When provided,
        ``metadata.global_params.domain_type`` is checked; a warning is
        emitted if the domain is not ``'FX'``.

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
    _check_domain_type(metadata)
    _check_zero_pulse_fraction(signal)
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
    metadata: Optional[CPHDMetadata] = None,
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
    metadata : CPHDMetadata, optional
        CPHD file metadata.  When provided,
        ``metadata.global_params.domain_type`` is checked; a warning is
        emitted if the domain is not ``'FX'``.

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
    _check_domain_type(metadata)
    _check_zero_pulse_fraction(signal)
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
    threshold_db: float = 10.0,
    metadata: Optional[CPHDMetadata] = None,
) -> CrossPolResult:
    """Cross-polarisation power ratio RFI test.

    For each available co/cross-pol pair, computes the mean PSD for
    each polarisation channel and flags range-frequency bins where the
    power ratio deviates from the scene median ratio by more than
    *threshold_db*.

    The default threshold is 10 dB, which is appropriate for real
    heterogeneous SAR scenes where natural scene scattering mechanisms
    (vegetation, urban, mixed terrain) produce HH/HV spectral imbalance
    of several dB across the band.  Lower values (e.g. 5 dB) increase
    sensitivity to subtle RFI but also increase the false-alarm rate on
    complex scenes; raise the threshold when the baseline pol ratio
    exhibits high spectral variability.

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
    metadata : CPHDMetadata, optional
        CPHD file metadata.  When provided,
        ``metadata.global_params.domain_type`` is checked; a warning is
        emitted if the domain is not ``'FX'``.

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
    _check_domain_type(metadata)
    logger.debug(
        "detect_cross_pol: pols=%s threshold_db=%.1f",
        list(signals.keys()),
        threshold_db,
    )

    # Compute mean PSD in dB for every provided polarisation.
    mean_psd_db: Dict[str, np.ndarray] = {}
    for pol, sig in signals.items():
        _check_zero_pulse_fraction(sig, name=f"signals['{pol}']")
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

    if not mean_ratio_db:
        logger.warning(
            "detect_cross_pol: no ratio pairs evaluated — expected keys "
            "'HH', 'HV', 'VV', or 'VH'; got %s.  Result will be empty.",
            list(signals.keys()),
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
    metadata: Optional[CPHDMetadata] = None,
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
    metadata : CPHDMetadata, optional
        CPHD file metadata.  When provided,
        ``metadata.global_params.domain_type`` is checked; a warning is
        emitted if the domain is not ``'FX'``.

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
    _check_domain_type(metadata)
    _check_zero_pulse_fraction(signal)
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
    tail_pulses = n_az - n_blocks * effective_block
    if tail_pulses > 0:
        logger.debug(
            "detect_eigenvalue: %d tail pulse(s) beyond the last full "
            "block will not be processed",
            tail_pulses,
        )

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
        C = (X @ X.conj().T) / n_rng                        # (block_size, block_size)

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


# ===================================================================
# Mitigation functions
# ===================================================================

def mitigate_notch(
    signal: np.ndarray,
    flagged_bins: np.ndarray,
) -> np.ndarray:
    """Zero-fill flagged range-frequency bins (spectral notch filter).

    Creates a copy of *signal* and sets all flagged range-frequency
    columns to zero across every slow-time pulse.  This is the simplest
    and fastest RFI mitigation method; it is preferred when the flagged
    bins are narrow and well-isolated.

    For wider or contiguous flagged regions consider
    ``mitigate_interpolate``, which fills the gap by interpolating from
    neighbouring bins instead of zeroing.

    Parameters
    ----------
    signal : np.ndarray
        Complex FX phase history data, shape ``(n_az, n_rng)``.
    flagged_bins : np.ndarray
        Indices of range-frequency bins to zero out, shape
        ``(n_flagged,)``.  Bins outside ``[0, n_rng)`` are silently
        ignored.

    Returns
    -------
    np.ndarray
        Mitigated copy of *signal*; shape and dtype are preserved.

    Raises
    ------
    ValidationError
        If *signal* is not 2-D or not complex.
    """
    _validate_signal(signal)
    mitigated = signal.copy()
    if len(flagged_bins) > 0:
        valid_idx = flagged_bins[
            (flagged_bins >= 0) & (flagged_bins < signal.shape[1])
        ]
        mitigated[:, valid_idx] = 0.0
    return mitigated


def mitigate_interpolate(
    signal: np.ndarray,
    flagged_bins: np.ndarray,
) -> np.ndarray:
    """Replace flagged range-frequency bins by linear interpolation.

    For every slow-time pulse, fills each flagged range-frequency bin
    by linearly interpolating between the values of its nearest
    unflagged neighbours on the left and right.  Both the real and
    imaginary parts are interpolated independently, preserving complex
    structure.  The operation is vectorised over all pulses.

    Compared to ``mitigate_notch``, this avoids spectral holes and
    preserves the local power level at the cost of introducing
    correlated samples in heavily-flagged contiguous regions.

    Parameters
    ----------
    signal : np.ndarray
        Complex FX phase history data, shape ``(n_az, n_rng)``.
    flagged_bins : np.ndarray
        Indices of range-frequency bins to replace, shape
        ``(n_flagged,)``.  Bins outside ``[0, n_rng)`` are silently
        clipped.

    Returns
    -------
    np.ndarray
        Mitigated copy of *signal*; shape and dtype are preserved.

    Raises
    ------
    ValidationError
        If *signal* is not 2-D or not complex.
    """
    _validate_signal(signal)
    if len(flagged_bins) == 0:
        return signal.copy()

    n_az, n_rng = signal.shape

    # Clip and deduplicate bin indices.
    flagged_bins = np.unique(
        np.clip(flagged_bins, 0, n_rng - 1).astype(np.intp)
    )
    good_bins = np.setdiff1d(np.arange(n_rng, dtype=np.intp), flagged_bins)

    if len(good_bins) == 0:
        logger.warning(
            "mitigate_interpolate: all %d bins are flagged; "
            "returning zero array.",
            n_rng,
        )
        return np.zeros_like(signal)

    # For each flagged bin, find the nearest good bin to its left and
    # right using a binary search on the sorted good_bins array.
    insert_pos = np.searchsorted(good_bins, flagged_bins)
    left_idx = np.clip(insert_pos - 1, 0, len(good_bins) - 1)
    right_idx = np.clip(insert_pos,     0, len(good_bins) - 1)

    l_bins = good_bins[left_idx]   # (n_flagged,) left-neighbour bin indices
    r_bins = good_bins[right_idx]  # (n_flagged,) right-neighbour bin indices

    # Linear interpolation weights; treat same-neighbour (edge) case as
    # constant extrapolation.
    span = (r_bins - l_bins).astype(np.float32)
    same = span == 0
    span[same] = 1.0  # avoid divide-by-zero
    w_right = np.clip(
        (flagged_bins - l_bins).astype(np.float32) / span, 0.0, 1.0
    )  # (n_flagged,)
    w_left = 1.0 - w_right

    # Vectorised over all pulses: (n_az, n_flagged).
    mitigated = signal.copy()
    mitigated[:, flagged_bins] = (
        w_left[np.newaxis, :] * signal[:, l_bins]
        + w_right[np.newaxis, :] * signal[:, r_bins]
    )
    return mitigated


def mitigate_vsff(
    signal: np.ndarray,
    *,
    amax_db: float = 0.0,
    metadata: Optional[CPHDMetadata] = None,
) -> VSFFResult:
    """Variable Space Frequency Filter (VSFF) for narrowband RFI mitigation.

    Implements a two-phase algorithm that first localises the narrowband
    RFI in both the spectral (range-frequency) and spatial (slow-time /
    azimuth) dimensions, then suppresses it with a 2-D tuneable Band-Stop
    Filter (BSF) whose notch depth at each azimuth pulse is proportional
    to the estimated SAR antenna gain toward the RFI source.

    The algorithm operates on Level-0 / Compensated Phase History Data
    arranged as a 2-D ``(n_az, n_rng)`` complex matrix.

    **Domain handling** — both CPHD FX-domain (range-frequency) and
    TOA-domain (range-time) inputs are supported.  When *metadata* is
    supplied and ``metadata.global_params.domain_type == 'TOA'``, the
    range axis is FFT-transformed to the frequency domain before
    processing and IFFT-transformed back to TOA before the result is
    returned.  All intermediate computation is performed in the
    range-frequency domain.

    Phase 1 — Power-Based RFI Localisation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. Compute the mean power spectral density (PSD) across all
       slow-time pulses.
    2. Identify the RFI centre frequency bin and 3 dB half-power
       bandwidth from the peak of the PSD excess above the noise floor.
    3. Construct a rectangular band-pass filter (BPF) centred on the
       estimated frequency and apply it to isolate the RFI component.
    4. Compute per-pulse integrated power within the BPF band.  The
       resulting 1-D profile serves as a proxy for the SAR antenna gain
       as a function of slow-time.  The pulse with peak power is taken
       as the estimated azimuth location of the RFI source.

    Phase 2 — 2-D Spatial-Frequency Filtration
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. Derive the normalised antenna gain profile G_norm(η) from the
       BPF-isolated per-pulse power.
    2. Compute per-pulse ABSF attenuation coefficients following the
       formula from the VSFF specification::

           ABSF(η) = (1 − Amax) · G_norm(η) + Amax

       where ``G_norm`` is normalised to ``[0, 1]`` (0 at minimum gain,
       1 at maximum gain) and ``Amax`` is the linear-scale floor factor
       derived from *amax_db*.  ABSF ranges from ``Amax`` (minimum-gain
       pulses) to ``1`` (maximum-gain pulse), yielding a BSF pass
       coefficient of ``1 − ABSF`` in the stop-band.
    3. Estimate the 3 dB azimuth beamwidth from the gain profile and
       restrict active filtering to pulses within that window.
    4. Construct and apply the 2-D BSF::

           H(f, η) = 1 − ABSF(η) · W(f)

       where ``W(f) = 1`` inside the frequency stop-band (width ≈ RFI
       bandwidth, centred on the RFI frequency) and ``0`` outside.
       The stop-band is applied only to pulses within the azimuth
       beamwidth; pulses outside are passed unmodified.

    Parameters
    ----------
    signal : np.ndarray
        Complex phase history data, shape ``(n_az, n_rng)``.
        Axis 0 is slow-time (azimuth); axis 1 is the range axis (either
        range-frequency for FX domain or range-time for TOA domain).
        Pass only valid (non-zero) pulse rows for accurate noise-floor
        estimates; see ``CPHDPVP.trim_to_valid``.
    amax_db : float
        Stop-band attenuation floor in dB applied to pulses at the
        minimum antenna gain position.  Must be ≥ 0.  Default ``0.0``
        (transparent at minimum-gain pulses — the "0 dB minimum
        attenuation baseline" prescribed by the VSFF specification).
        Pulses at maximum antenna gain always receive complete
        suppression (∞ dB) regardless of this parameter.
    metadata : CPHDMetadata, optional
        CPHD file metadata.  When provided,
        ``metadata.global_params.domain_type`` is inspected to detect
        TOA-domain input and perform the necessary range-axis FFT/IFFT.
        If *metadata* is ``None`` the signal is assumed to be in the
        FX (range-frequency) domain.

    Returns
    -------
    VSFFResult
        ``mitigated`` preserves the input shape, dtype, and signal
        domain.  Diagnostic fields include the estimated RFI centre
        bin, bandwidth, azimuth index, ABSF coefficients, mean PSD, and
        gain profile.

    Raises
    ------
    ValidationError
        If *signal* is not 2-D or not complex, or if *amax_db* < 0.
    """
    _validate_signal(signal)
    if amax_db < 0.0:
        raise ValidationError(
            f"amax_db must be ≥ 0, got {amax_db}"
        )
    _check_zero_pulse_fraction(signal)

    # --- Domain detection and conversion ---
    domain_type: Optional[str] = None
    if metadata is not None and metadata.global_params is not None:
        dt = metadata.global_params.domain_type
        if dt is not None:
            domain_type = dt.upper()

    is_toa = domain_type == 'TOA'
    if is_toa:
        sig_fx = np.fft.fft(signal, axis=1).astype(np.complex64)
        logger.debug(
            "mitigate_vsff: TOA domain detected; FFT applied to range axis"
        )
    else:
        sig_fx = signal.astype(np.complex64)

    n_az, n_rng = sig_fx.shape

    # Linear-scale Amax parameter for the ABSF formula.
    # amax_linear = 0.0 → 0 dB floor at Gmin (spec default);
    # larger values raise the stop-band floor at minimum-gain pulses.
    amax_linear = float(1.0 - 10.0 ** (-amax_db / 20.0))

    logger.debug(
        "mitigate_vsff: shape=%s amax_db=%.1f amax_linear=%.4f domain=%s",
        signal.shape,
        amax_db,
        amax_linear,
        domain_type or 'FX (assumed)',
    )

    # ============================================================
    # Phase 1: Power-Based RFI Localisation
    # ============================================================

    # Step 1.1 — Mean PSD profile across slow-time
    mean_pwr = np.mean(np.abs(sig_fx) ** 2, axis=0)          # (n_rng,)
    mean_psd_db_arr = 10.0 * np.log10(mean_pwr + _EPS)       # (n_rng,)

    # Step 1.2 — RFI spectral parameters from PSD excess above noise floor
    noise_floor_db = float(np.median(mean_psd_db_arr))
    excess_db_arr = mean_psd_db_arr - noise_floor_db          # (n_rng,)

    peak_bin = int(np.argmax(excess_db_arr))
    peak_excess_db = float(excess_db_arr[peak_bin])

    logger.debug(
        "mitigate_vsff: PSD peak at bin %d, excess=%.1f dB",
        peak_bin,
        peak_excess_db,
    )

    # Guard: if no spectral peak rises above the noise floor, return a
    # clean copy and populate diagnostic fields with neutral values.
    if peak_excess_db <= 0.0:
        logger.warning(
            "mitigate_vsff: no spectral peak exceeds the noise floor; "
            "returning input copy without modification."
        )
        return VSFFResult(
            mitigated=signal.copy(),
            rfi_center_bin=peak_bin,
            rfi_bw_bins=0,
            rfi_azimuth_idx=0,
            attenuation_coeffs=np.zeros(n_az, dtype=np.float64),
            mean_psd_db=mean_psd_db_arr.astype(np.float64),
            gain_profile=np.zeros(n_az, dtype=np.float64),
            az_lo=0,
            az_hi=n_az - 1,
        )

    # 3 dB half-power threshold around the PSD peak
    half_power_db = peak_excess_db - 3.0

    # Scan leftward from peak to find lower 3 dB edge
    lo_bin = peak_bin
    while lo_bin > 0 and excess_db_arr[lo_bin - 1] >= half_power_db:
        lo_bin -= 1

    # Scan rightward from peak to find upper 3 dB edge
    hi_bin = peak_bin
    while hi_bin < n_rng - 1 and excess_db_arr[hi_bin + 1] >= half_power_db:
        hi_bin += 1

    bw_bins = max(hi_bin - lo_bin + 1, 1)
    center_bin = (lo_bin + hi_bin) // 2
    half_bw = max(bw_bins // 2, 1)

    logger.debug(
        "mitigate_vsff: center_bin=%d bw_bins=%d [lo=%d hi=%d]",
        center_bin,
        bw_bins,
        lo_bin,
        hi_bin,
    )

    # Step 1.3 — Isolation BPF centred on estimated RFI frequency
    bpf_lo = max(0, center_bin - half_bw)
    bpf_hi = min(n_rng, center_bin + half_bw + 1)
    bpf_w = np.zeros(n_rng, dtype=np.float32)
    bpf_w[bpf_lo:bpf_hi] = 1.0

    # Multiply by BPF window in frequency domain (= convolution in time)
    si_tilde = sig_fx * bpf_w[np.newaxis, :]                  # (n_az, n_rng)

    # Step 1.4 — Azimuth peak localisation via per-pulse BPF band power
    gain_profile = np.sum(
        np.abs(si_tilde) ** 2, axis=1
    ).astype(np.float64)                                       # (n_az,)
    eta_rfi = int(np.argmax(gain_profile))

    logger.debug(
        "mitigate_vsff: RFI azimuth peak at pulse index %d", eta_rfi
    )

    # ============================================================
    # Phase 2: 2-D Spatial-Frequency Filtration
    # ============================================================

    # Step 2.1 — Antenna gain profile parameters
    g_max = float(np.max(gain_profile))
    g_min = float(np.min(gain_profile))
    g_range = g_max - g_min

    # Step 2.3 — Normalised gain and ABSF coefficients
    # G_norm ∈ [0, 1]: 0 at Gmin, 1 at Gmax
    if g_range < _EPS:
        # Flat gain profile: treat all pulses as uniformly at Gmax
        g_norm = np.ones(n_az, dtype=np.float64)
    else:
        g_norm = (gain_profile - g_min) / g_range

    # ABSF(η) = (1 - Amax)·G_norm(η) + Amax, ∈ [Amax, 1]
    # BSF pass coefficient in stop-band = 1 - ABSF(η)
    absf = (1.0 - amax_linear) * g_norm + amax_linear         # (n_az,)

    # Step 2.4 — Azimuth beamwidth from 3 dB points of gain profile
    g_peak = float(gain_profile[eta_rfi])
    half_g = g_peak / 2.0                      # 3 dB threshold (linear power)

    az_lo = eta_rfi
    while az_lo > 0 and gain_profile[az_lo - 1] >= half_g:
        az_lo -= 1

    az_hi = eta_rfi
    while az_hi < n_az - 1 and gain_profile[az_hi + 1] >= half_g:
        az_hi += 1

    logger.debug(
        "mitigate_vsff: azimuth beamwidth %d pulses [az_lo=%d az_hi=%d]",
        az_hi - az_lo + 1,
        az_lo,
        az_hi,
    )

    # Build 2-D BSF:  H(f, η) = 1 − ABSF(η) · W(f)
    # W(f) = 1 inside frequency stop-band, 0 outside.
    # Only pulses within the estimated azimuth beamwidth are filtered;
    # pulses outside the beamwidth are passed unmodified (H = 1).
    H_2d = np.ones((n_az, n_rng), dtype=np.float32)
    bw_slice = slice(az_lo, az_hi + 1)
    H_2d[bw_slice, bpf_lo:bpf_hi] = (
        (1.0 - absf[az_lo:az_hi + 1]).astype(np.float32)[:, np.newaxis]
    )

    # Apply 2-D filter
    mitigated_fx = sig_fx * H_2d                               # (n_az, n_rng)

    # --- Domain restoration ---
    orig_dtype = signal.dtype
    if is_toa:
        mitigated = np.fft.ifft(mitigated_fx, axis=1).astype(orig_dtype)
    else:
        mitigated = mitigated_fx.astype(orig_dtype)

    return VSFFResult(
        mitigated=mitigated,
        rfi_center_bin=center_bin,
        rfi_bw_bins=bw_bins,
        rfi_azimuth_idx=eta_rfi,
        attenuation_coeffs=absf,
        mean_psd_db=mean_psd_db_arr.astype(np.float64),
        gain_profile=gain_profile,
        az_lo=az_lo,
        az_hi=az_hi,
    )
