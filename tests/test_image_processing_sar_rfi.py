# -*- coding: utf-8 -*-
"""
RFI Detector Tests - Tests for spectral RFI detection functions.

Covers all four RFI detection methods: mean PSD test, per-pulse
spectral outlier detection, cross-polarisation power ratio test, and
eigenvalue decomposition test.  All tests use synthetic data with
controlled RFI injected so results are verifiable.

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

import numpy as np
import pytest

from grdl.exceptions import ValidationError
from grdl.image_processing.sar.rfi import (
    CrossPolResult,
    EigenvalueResult,
    MeanPSDResult,
    PerPulseResult,
    detect_cross_pol,
    detect_eigenvalue,
    detect_mean_psd,
    detect_per_pulse,
)


# ===================================================================
# Shared synthetic data helpers
# ===================================================================

_N_AZ: int = 128
_N_RNG: int = 256
_RFI_BIN: int = 64      # canonical RFI injection bin in range-freq space


def _make_noise(seed: int = 42) -> np.ndarray:
    """Complex Gaussian noise in range-frequency domain, shape (_N_AZ, _N_RNG)."""
    rng = np.random.RandomState(seed)
    return (
        rng.randn(_N_AZ, _N_RNG) + 1j * rng.randn(_N_AZ, _N_RNG)
    ).astype(np.complex64) / np.sqrt(2)


def _make_rfi_signal(amplitude: float = 100.0, seed: int = 42) -> np.ndarray:
    """Noise + coherent narrowband RFI in range-frequency domain (CPHD FX).

    The RFI is a narrowband tone added directly at range-frequency bin
    *_RFI_BIN* in every pulse.  No FFT is needed in the detectors because
    the input is already in the frequency domain.
    """
    noise = _make_noise(seed)
    rfi = noise.copy()
    rfi[:, _RFI_BIN] += amplitude  # add spike at freq bin across all pulses
    return rfi


# ===================================================================
# TestMeanPSD
# ===================================================================

class TestMeanPSD:
    """Tests for detect_mean_psd."""

    def test_flags_known_rfi_bin(self):
        signal = _make_rfi_signal()
        result = detect_mean_psd(signal, threshold_db=10.0)
        assert isinstance(result, MeanPSDResult)
        assert _RFI_BIN in result.flagged_bins

    def test_low_far_on_clean_noise(self):
        signal = _make_noise()
        result = detect_mean_psd(signal, threshold_db=10.0)
        assert len(result.flagged_bins) == 0

    def test_all_zero_input_returns_empty(self):
        signal = np.zeros((_N_AZ, _N_RNG), dtype=np.complex64)
        result = detect_mean_psd(signal)
        assert len(result.flagged_bins) == 0

    def test_mean_psd_db_shape(self):
        signal = _make_noise()
        result = detect_mean_psd(signal)
        assert result.mean_psd_db.shape == (_N_RNG,)

    def test_noise_floor_db_is_scalar(self):
        signal = _make_noise()
        result = detect_mean_psd(signal)
        assert isinstance(result.noise_floor_db, float)

    def test_flagged_field_shapes_consistent(self):
        signal = _make_rfi_signal()
        result = detect_mean_psd(signal)
        n = len(result.flagged_bins)
        assert result.bin_powers_db.shape == (n,)
        assert result.excess_db.shape == (n,)

    def test_excess_db_positive(self):
        signal = _make_rfi_signal()
        result = detect_mean_psd(signal, threshold_db=10.0)
        assert np.all(result.excess_db > 10.0)

    def test_raises_on_1d_input(self):
        with pytest.raises(ValidationError):
            detect_mean_psd(np.ones(_N_RNG, dtype=np.complex64))

    def test_raises_on_real_input(self):
        with pytest.raises(ValidationError):
            detect_mean_psd(np.ones((_N_AZ, _N_RNG), dtype=np.float32))


# ===================================================================
# TestPerPulse
# ===================================================================

class TestPerPulse:
    """Tests for detect_per_pulse."""

    def test_flags_known_rfi_bin(self):
        signal = _make_rfi_signal()
        result = detect_per_pulse(signal, threshold_db=6.0)
        assert isinstance(result, PerPulseResult)
        assert result.flag_counts[_RFI_BIN] > 0

    def test_low_far_on_clean_noise(self):
        signal = _make_noise()
        # Use a more demanding threshold (10 dB) where pure white Gaussian
        # noise should produce near-zero false alarms.  At 6 dB the per-pulse
        # median-filter test is statistically expected to flag ~13% of bins
        # (the dB-space std of complex Gaussian noise is ~5.6 dB), so 6 dB
        # is the sensitivity tuning knob for real SAR data, not a FAR target.
        result = detect_per_pulse(signal, threshold_db=10.0)
        assert result.flag_fraction.max() < 0.05

    def test_mask_shape(self):
        signal = _make_noise()
        result = detect_per_pulse(signal)
        assert result.mask.shape == (_N_RNG, _N_AZ)

    def test_flag_counts_shape(self):
        signal = _make_noise()
        result = detect_per_pulse(signal)
        assert result.flag_counts.shape == (_N_RNG,)
        assert result.flag_fraction.shape == (_N_RNG,)

    def test_flag_fraction_in_range(self):
        signal = _make_rfi_signal()
        result = detect_per_pulse(signal)
        assert np.all(result.flag_fraction >= 0.0)
        assert np.all(result.flag_fraction <= 1.0)

    def test_mask_consistent_with_counts(self):
        signal = _make_rfi_signal()
        result = detect_per_pulse(signal)
        assert np.array_equal(result.mask.sum(axis=1), result.flag_counts)

    def test_raises_on_even_filter_size(self):
        signal = _make_noise()
        with pytest.raises(ValidationError):
            detect_per_pulse(signal, filter_size=50)

    def test_raises_on_filter_size_too_small(self):
        signal = _make_noise()
        with pytest.raises(ValidationError):
            detect_per_pulse(signal, filter_size=2)

    def test_raises_on_non_complex_input(self):
        with pytest.raises(ValidationError):
            detect_per_pulse(np.ones((_N_AZ, _N_RNG), dtype=np.float64))

    def test_raises_on_1d_input(self):
        with pytest.raises(ValidationError):
            detect_per_pulse(np.ones(_N_RNG, dtype=np.complex64))


# ===================================================================
# TestCrossPol
# ===================================================================

class TestCrossPol:
    """Tests for detect_cross_pol."""

    def _rfi_hh_only(self) -> dict:
        """HH with RFI at _RFI_BIN, HV with pure noise at same shape."""
        return {
            'HH': _make_rfi_signal(amplitude=100.0, seed=42),
            'HV': _make_noise(seed=7),
        }

    def test_flags_anomalous_ratio_bin(self):
        signals = self._rfi_hh_only()
        result = detect_cross_pol(signals, threshold_db=5.0)
        assert isinstance(result, CrossPolResult)
        assert 'HH/HV' in result.flagged_bins
        assert _RFI_BIN in result.flagged_bins['HH/HV']

    def test_low_far_on_clean_noise(self):
        signals = {'HH': _make_noise(seed=1), 'HV': _make_noise(seed=2)}
        result = detect_cross_pol(signals, threshold_db=5.0)
        assert 'HH/HV' in result.flagged_bins
        # Expect few or no flags on matching noise distributions.
        assert len(result.flagged_bins['HH/HV']) == 0

    def test_mean_ratio_db_shape(self):
        signals = self._rfi_hh_only()
        result = detect_cross_pol(signals)
        assert result.mean_ratio_db['HH/HV'].shape == (_N_RNG,)

    def test_flagged_and_anomaly_shapes_consistent(self):
        signals = self._rfi_hh_only()
        result = detect_cross_pol(signals)
        for key in result.flagged_bins:
            n = len(result.flagged_bins[key])
            assert result.ratio_anomaly_db[key].shape == (n,)

    def test_ratio_anomaly_above_threshold(self):
        signals = self._rfi_hh_only()
        result = detect_cross_pol(signals, threshold_db=5.0)
        for key, anom in result.ratio_anomaly_db.items():
            assert np.all(anom > 5.0)

    def test_missing_pair_silently_omitted(self):
        # VH absent → VV/VH and HH/VV omitted from result.
        signals = {'HH': _make_noise(seed=1), 'HV': _make_noise(seed=2)}
        result = detect_cross_pol(signals)
        assert 'VV/VH' not in result.flagged_bins
        assert 'HH/VV' not in result.flagged_bins

    def test_all_four_pols_produces_all_ratios(self):
        signals = {
            'HH': _make_noise(seed=1),
            'HV': _make_noise(seed=2),
            'VH': _make_noise(seed=3),
            'VV': _make_noise(seed=4),
        }
        result = detect_cross_pol(signals)
        for key in ('HH/HV', 'VV/VH', 'HH/VV'):
            assert key in result.mean_ratio_db

    def test_raises_on_single_pol(self):
        with pytest.raises(ValidationError):
            detect_cross_pol({'HH': _make_noise()})

    def test_raises_on_empty_dict(self):
        with pytest.raises(ValidationError):
            detect_cross_pol({})

    def test_raises_on_non_complex_signal(self):
        with pytest.raises(ValidationError):
            detect_cross_pol({
                'HH': np.ones((_N_AZ, _N_RNG), dtype=np.float32),
                'HV': _make_noise(),
            })


# ===================================================================
# TestEigenvalue
# ===================================================================

class TestEigenvalue:
    """Tests for detect_eigenvalue."""

    def test_rfi_yields_high_dominant_ratio(self):
        signal = _make_rfi_signal()
        result = detect_eigenvalue(signal, block_size=32, threshold=10.0)
        assert isinstance(result, EigenvalueResult)
        # At least some blocks should detect the coherent RFI tone.
        assert np.any(result.dominant_ratio > 10.0)

    def test_rfi_flags_bin(self):
        signal = _make_rfi_signal()
        result = detect_eigenvalue(signal, block_size=32, threshold=10.0)
        assert len(result.flagged_bins) > 0

    def test_clean_noise_has_low_rank(self):
        signal = _make_noise()
        result = detect_eigenvalue(signal, block_size=32, threshold=10.0)
        assert np.all(result.rfi_rank == 0)

    def test_clean_noise_few_flagged_bins(self):
        signal = _make_noise()
        result = detect_eigenvalue(signal, block_size=32, threshold=10.0)
        assert len(result.flagged_bins) == 0

    def test_eigenvalue_profiles_shape(self):
        signal = _make_noise()
        result = detect_eigenvalue(signal, block_size=32)
        n_blocks = _N_AZ // 32
        assert result.eigenvalue_profiles.shape == (n_blocks, 32)

    def test_dominant_ratio_shape(self):
        signal = _make_noise()
        result = detect_eigenvalue(signal, block_size=32)
        n_blocks = _N_AZ // 32
        assert result.dominant_ratio.shape == (n_blocks,)
        assert result.rfi_rank.shape == (n_blocks,)

    def test_block_size_larger_than_n_az_single_block(self):
        signal = _make_rfi_signal()
        # block_size > n_az → treated as single block.
        result = detect_eigenvalue(signal, block_size=_N_AZ + 100)
        assert result.eigenvalue_profiles.shape[0] == 1
        assert result.dominant_ratio.shape == (1,)

    def test_flagged_bins_sorted_unique(self):
        signal = _make_rfi_signal()
        result = detect_eigenvalue(signal, block_size=32, threshold=10.0)
        if len(result.flagged_bins) > 1:
            assert np.all(np.diff(result.flagged_bins) > 0)

    def test_raises_on_block_size_1(self):
        with pytest.raises(ValidationError):
            detect_eigenvalue(_make_noise(), block_size=1)

    def test_raises_on_non_complex_input(self):
        with pytest.raises(ValidationError):
            detect_eigenvalue(np.ones((_N_AZ, _N_RNG), dtype=np.float64))

    def test_raises_on_1d_input(self):
        with pytest.raises(ValidationError):
            detect_eigenvalue(np.ones(_N_RNG, dtype=np.complex64))
