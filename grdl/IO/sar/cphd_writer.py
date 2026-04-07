# -*- coding: utf-8 -*-
"""
CPHD Writer - Convert NISAR RSLC phase history to NGA CPHD 1.1.0 binary.

Inverts a focused NISAR RSLC SLC image back to approximate compensated
phase history data (pseudo-CPHD) by reversing the separable range /
azimuth matched-filter compression steps, populating a fully-typed
``CPHDMetadata`` structure from the NISAR product metadata, and writing
a standards-compliant CPHD 1.1.0 binary file via sarkit.

The decompression steps mirror the experiment in
``experiments/nisar/main.py``:

    3a  Azimuth FFT            (np.fft.fft, axis=0)
    3b  Azimuth decompression  (k_a orbit-derived; invert azimuth MF)
    3c  Inverse azimuth FFT    (np.fft.ifft, axis=0) -> slow-time
    3d  Range FFT              (np.fft.fft, axis=1)
    3e  Range decompression    (Kr estimated; invert range chirp MF)
    3f  Transpose              -> (n_range_freq, n_slow_time)

The output is in the range-frequency × slow-time domain (CPHD FX
standard, ``DomainType=FX``).

The ``write()`` method saves a binary CPHD 1.1.0 file (sarkit backend)
containing the signal array in (NumVectors, NumSamples) layout and all
mandatory PVP fields.

Known approximations
--------------------
*  NISAR L1 does not store transmit pulse duration ``Tp``.  Chirp rate
   is estimated as ``Kr = processedRangeBandwidth / Tp`` with the
   nominal L-band ``Tp = 35 us``.  Override via
   ``process_params['pulse_duration_s']``.
*  NISAR uses TDBP image formation.  The separable frequency-domain
   inversion is an approximation valid near boresight.
*  ``CPHDPVP.srp_pos`` is approximated from the ARP at mid-aperture
   when ``process_params['srp_ecf']`` is not supplied.
*  ``aFDOP``, ``aFRR1``, ``aFRR2``, ``TDTropoSRP`` default to 0.0.
*  ``TOA1`` / ``TOA2`` are estimated as 2*(R-R_srp)/c relative to the
   SRP range.
*  ``SC0`` = ``FX1`` per pulse; ``SCSS`` = bw / n_rng.

Dependencies
------------
sarkit (CPHD 1.1.0 writer backend)
lxml (XML construction)
scipy (CubicHermiteSpline orbit interpolation)

Author
------
GRDL Contributors

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-04-01
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from grdl.IO.base import ImageWriter
from grdl.IO.models.nisar import NISARMetadata
from grdl.IO.models.cphd import (
    CPHDChannel,
    CPHDCollectionInfo,
    CPHDGlobal,
    CPHDMetadata,
    CPHDPVP,
    CPHDRcvParameters,
    CPHDTxWaveform,
)
from grdl.exceptions import DependencyError, ProcessorError, ValidationError

logger = logging.getLogger(__name__)

# Speed of light (m/s) — matches experiments/nisar/main.py constant.
_C: float = 299_792_458.0

# NISAR nominal transmit pulse durations (not stored in L1 products).
# L-band (LSAR): 35 µs typical.  S-band (SSAR): 20 µs typical.
_NISAR_L_NOMINAL_PULSE_DURATION_S: float = 35e-6
_NISAR_S_NOMINAL_PULSE_DURATION_S: float = 20e-6

# Number of samples for the default uniform weighting function.
_WEIGHTING_SAMPLES: int = 256

# CPHD 1.1.0 XML namespace.
_CPHD_NS: str = 'http://api.nsgreg.nga.mil/schema/cphd/1.1.0'

# Mandatory PVP field layout for the CPHD 1.1.0 file written by this class.
# Tuple: (field_name, word_offset, word_size, binary_format_string)
# Each word is 8 bytes.  Total = 28 words * 8 = 224 bytes (NumBytesPVP).
_PVP_FIELDS: List[Tuple[str, int, int, str]] = [
    ('TxTime',     0,  1, 'F8'),
    ('TxPos',      1,  3, 'X=F8;Y=F8;Z=F8;'),
    ('TxVel',      4,  3, 'X=F8;Y=F8;Z=F8;'),
    ('RcvTime',    7,  1, 'F8'),
    ('RcvPos',     8,  3, 'X=F8;Y=F8;Z=F8;'),
    ('RcvVel',    11,  3, 'X=F8;Y=F8;Z=F8;'),
    ('SRPPos',    14,  3, 'X=F8;Y=F8;Z=F8;'),
    ('aFDOP',     17,  1, 'F8'),
    ('aFRR1',     18,  1, 'F8'),
    ('aFRR2',     19,  1, 'F8'),
    ('FX1',       20,  1, 'F8'),
    ('FX2',       21,  1, 'F8'),
    ('TOA1',      22,  1, 'F8'),
    ('TOA2',      23,  1, 'F8'),
    ('TDTropoSRP',24,  1, 'F8'),
    ('SC0',       25,  1, 'F8'),
    ('SCSS',      26,  1, 'F8'),
    ('SIGNAL',    27,  1, 'I8'),
]
_NUM_BYTES_PVP: int = 28 * 8  # 224


def _require_sarkit() -> Any:
    """Import sarkit.cphd or raise DependencyError."""
    try:
        import sarkit.cphd as sc
        return sc
    except ImportError as exc:
        raise DependencyError(
            "Writing CPHD files requires sarkit. "
            "Install with: pip install sarkit"
        ) from exc


def _require_lxml() -> Any:
    """Import lxml.etree or raise DependencyError."""
    try:
        import lxml.etree as et
        return et
    except ImportError as exc:
        raise DependencyError(
            "Writing CPHD files requires lxml. "
            "Install with: pip install lxml"
        ) from exc


class CPHDWriter(ImageWriter):
    """Convert NISAR RSLC imagery to NGA CPHD 1.1.0 phase history data.

    Inherits from ``ImageWriter`` and implements the standard ``write``
    and ``write_chip`` interface while providing the domain-specific
    ``nisar_to_cphd`` conversion method.

    Parameters
    ----------
    filepath : str or Path
        Default output path used by ``write()`` when no ``filename``
        override is given.  Should end in ``.cphd``.
    metadata : CPHDMetadata, optional
        Pre-built CPHD metadata.  Normally populated automatically by
        a preceding call to ``nisar_to_cphd()``.

    Examples
    --------
    >>> from grdl.IO.sar.nisar import NISARReader
    >>> from grdl.IO.sar.cphd_writer import CPHDWriter
    >>> with NISARReader('product.h5', frequency='A', polarization='HH') as rdr:
    ...     slc = rdr.read_full()
    ...     nisar_meta = rdr.metadata
    >>> writer = CPHDWriter('output_HH.cphd')
    >>> phase_history, cphd_meta = writer.nisar_to_cphd(slc, nisar_meta)
    >>> writer.write(phase_history)
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        metadata: Optional[CPHDMetadata] = None,
    ) -> None:
        super().__init__(filepath, metadata)

    # ------------------------------------------------------------------
    # Private signal-processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _interp_orbit(
        orb_t: np.ndarray,
        orb_pos: np.ndarray,
        orb_vel: np.ndarray,
        query_t: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Hermite-spline interpolate orbit state to arbitrary times.

        Uses ``scipy.interpolate.CubicHermiteSpline`` (same backend as
        ``experiments/nisar/main.py``) to interpolate ECF position and
        velocity independently per axis.

        Parameters
        ----------
        orb_t : np.ndarray
            Orbit epoch times, shape ``(K,)``, seconds since reference
            epoch — must match the frame of ``query_t``.
        orb_pos : np.ndarray
            ECF position at each epoch, shape ``(K, 3)``, metres.
        orb_vel : np.ndarray
            ECF velocity at each epoch, shape ``(K, 3)``, m/s.
        query_t : np.ndarray
            Times to evaluate, shape ``(N,)``.  Must lie within
            ``[orb_t[0], orb_t[-1]]``.

        Returns
        -------
        pos : np.ndarray
            Interpolated ECF positions, shape ``(N, 3)``, metres.
        vel : np.ndarray
            Interpolated ECF velocities, shape ``(N, 3)``, m/s.

        Raises
        ------
        ValidationError
            If ``query_t`` lies outside the coverage of ``orb_t``.
        """
        if float(query_t[0]) < float(orb_t[0]) or float(query_t[-1]) > float(orb_t[-1]):
            # Allow a 1 ms tolerance for floating-point epoch alignment.
            _eps = 1e-3
            if (float(query_t[0]) < float(orb_t[0]) - _eps
                    or float(query_t[-1]) > float(orb_t[-1]) + _eps):
                raise ValidationError(
                    f"query_t [{query_t[0]:.3f}, {query_t[-1]:.3f}] falls outside "
                    f"orbit coverage [{orb_t[0]:.3f}, {orb_t[-1]:.3f}]. "
                    "Extend the orbit state vector span or reduce the SLC window."
                )
        n = len(query_t)
        pos = np.zeros((n, 3), dtype=np.float64)
        vel = np.zeros((n, 3), dtype=np.float64)
        for ax in range(3):
            spl = CubicHermiteSpline(orb_t, orb_pos[:, ax], orb_vel[:, ax])
            pos[:, ax] = spl(query_t)
            vel[:, ax] = spl.derivative()(query_t)
        return pos, vel

    @staticmethod
    def _ka_from_orbit(
        orb_vel: np.ndarray,
        slant_range: np.ndarray,
        wavelength: float,
    ) -> np.ndarray:
        """Compute orbit-derived azimuth chirp rate k_a per range bin.

        Uses the flat-Earth approximation::

            k_a(R) = -2 * v_eff^2 / (lambda * R)

        Parameters
        ----------
        orb_vel : np.ndarray
            ECF velocity per pulse, shape ``(N_az, 3)``, m/s.
        slant_range : np.ndarray
            Slant range vector, shape ``(N_rng,)``, metres.
        wavelength : float
            Carrier wavelength, metres.

        Returns
        -------
        ka : np.ndarray
            Azimuth chirp rate per range bin, shape ``(N_rng,)``, Hz/s.
            Values are negative (LFM convention).
        """
        v_eff = float(np.linalg.norm(orb_vel, axis=1).mean())
        return -2.0 * v_eff**2 / (wavelength * slant_range)

    @staticmethod
    def _build_range_inv_filter(
        n_rng: int,
        fs: float,
        bw: float,
        kr: float,
        wgt: np.ndarray,
    ) -> np.ndarray:
        """Build the inverse range matched filter in the frequency domain.

        Constructs ``H(f_r) = exp(-j*pi*f_r^2/Kr) / W(f_r)`` within the
        processed bandwidth and zero outside.

        Parameters
        ----------
        n_rng : int
            Number of range samples (FFT size).
        fs : float
            Range sampling rate, Hz.
        bw : float
            Processed range bandwidth, Hz.
        kr : float
            Range chirp rate, Hz/s.
        wgt : np.ndarray
            1-D weighting function (any length; resampled to ``n_rng``).

        Returns
        -------
        H : np.ndarray
            Complex inverse filter, shape ``(n_rng,)``, dtype complex64.
        """
        fr = np.fft.fftfreq(n_rng, d=1.0 / fs)
        w = np.interp(
            np.linspace(0.0, 1.0, n_rng),
            np.linspace(0.0, 1.0, len(wgt)),
            wgt.astype(np.float64),
        )
        in_band = np.abs(fr) <= bw / 2.0
        w_safe = np.where(w > 1e-6, w, 1e-6)
        # Inverse range filter: conjugate of the forward MF exp(+j*pi*fr^2/Kr).
        # Undoes range chirp compression performed during NISAR L1 processing.
        H = np.where(in_band, np.exp(-1j * np.pi * fr**2 / kr) / w_safe, 0j)
        return H.astype(np.complex64)

    @staticmethod
    def _build_az_inv_filter(
        n_az: int,
        prf: float,
        ka: np.ndarray,
        fd0: Union[float, np.ndarray],
        az_bw: float,
        wgt: np.ndarray,
    ) -> np.ndarray:
        """Build the range-variant inverse azimuth matched filter (2-D).

        Constructs the complex conjugate of the forward azimuth matched filter.
        The forward MF is ``exp(+j*pi*(f_a-fd0)^2/k_a)``; since k_a < 0 this
        equals ``exp(-j*pi*(f_a-fd0)^2/|k_a|)``.  The inverse (decompressor)
        is therefore::

            H_inv(R, f_a) = exp(-j*pi*(f_a-fd0(R))^2 / k_a(R)) / W(f_a)
                          = exp(+j*pi*(f_a-fd0(R))^2 / |k_a(R)|) / W(f_a)

        applied for ``|f_a - fd0(R)| <= az_bw/2``, zero elsewhere.

        Parameters
        ----------
        n_az : int
            Number of azimuth samples (FFT size).
        prf : float
            Output pulse repetition frequency, Hz.
        ka : np.ndarray
            Azimuth chirp rate per range bin, shape ``(N_rng,)``, Hz/s.
        fd0 : float or np.ndarray
            Doppler centroid frequency, Hz.  Either a scalar (uniform
            across range) or a 1-D array of shape ``(N_rng,)`` for a
            range-variant centroid derived from the HDF5 dopplerCentroid
            grid.
        az_bw : float
            Processed azimuth bandwidth, Hz.
        wgt : np.ndarray
            1-D azimuth weighting (any length; resampled to ``n_az``).

        Returns
        -------
        H_az : np.ndarray
            Complex inverse filter, shape ``(N_rng, n_az)``, dtype complex64.
        """
        fa = np.fft.fftfreq(n_az, d=1.0 / prf)
        w_az = np.interp(
            np.linspace(0.0, 1.0, n_az),
            np.linspace(0.0, 1.0, len(wgt)),
            wgt.astype(np.float64),
        )
        w_safe = np.where(w_az > 1e-6, w_az, 1e-6)

        fd0_arr = np.asarray(fd0)
        if fd0_arr.ndim == 0:
            # Scalar centroid: df is (n_az,); broadcast against ka for ph2d.
            df = fa - float(fd0_arr)                        # (n_az,)
            in_band = np.abs(df) <= az_bw / 2.0            # (n_az,)
            # Inverse azimuth filter: conjugate of forward MF exp(+j*pi*df^2/ka).
            # With ka < 0 (orbit-derived), -pi*df^2/ka > 0 for df != 0,
            # so exp(+j * (-pi*df^2/ka)) applies positive quadratic phase —
            # the correct decompression (undo) operator.
            ph2d = -np.pi * df[np.newaxis, :]**2 / ka[:, np.newaxis]  # (n_rng, n_az)
            H_az = np.where(
                in_band[np.newaxis, :],
                np.exp(1j * ph2d) / np.where(in_band[np.newaxis, :], w_safe[np.newaxis, :], 1.0),
                0j,
            )
        else:
            # Range-variant centroid: fd0_arr is (N_rng,); df is (N_rng, n_az).
            df = fa[np.newaxis, :] - fd0_arr[:, np.newaxis]      # (n_rng, n_az)
            in_band = np.abs(df) <= az_bw / 2.0                  # (n_rng, n_az)
            # Same sign convention as scalar branch: -pi*df^2/ka gives
            # positive phase since ka < 0.
            ph2d = -np.pi * df**2 / ka[:, np.newaxis]            # (n_rng, n_az)
            H_az = np.where(
                in_band,
                np.exp(1j * ph2d) / np.where(in_band, w_safe[np.newaxis, :], 1.0),
                0j,
            )
        return H_az.astype(np.complex64)

    @staticmethod
    def _slc_to_phase_history(
        slc_az_rng: np.ndarray,
        H_az: np.ndarray,
        H_rng: np.ndarray,
        prf: float,
    ) -> np.ndarray:
        """Decompress a focused SLC to pseudo phase history via 6 steps.

        Implements the separable inversion of the SAR image-formation
        operator::

            Step 3a  Az-FFT              fft(slc, axis=0)
            Step 3b  Az-decompression    *= H_az.T
            Step 3c  Az-IFFT             ifft(s, axis=0)  -> slow-time
            Step 3d  Rng-FFT             fft(s, axis=1)
            Step 3e  Rng-decompression   *= H_rng
            Step 3f  Transpose           -> (N_rng, N_az)

        The output is in the range-**frequency** × slow-**time** domain,
        matching the CPHD FX standard (``DomainType=FX``).

        Parameters
        ----------
        slc_az_rng : np.ndarray
            Focused SLC, shape ``(N_az, N_rng)``.  Cast to complex64.
        H_az : np.ndarray
            Inverse azimuth filter, shape ``(N_rng, N_az)``, complex64.
        H_rng : np.ndarray
            Inverse range filter, shape ``(N_rng,)``, complex64.

        Returns
        -------
        phase_history : np.ndarray
            Pseudo phase history, shape ``(N_rng, N_az)``, complex64.
            Axis 0 is range-frequency; axis 1 is slow-time.

        Notes
        -----
        TDBP-to-frequency-domain inversion is accurate near boresight
        but degrades at large squint angles.
        """
        s = slc_az_rng.astype(np.complex64)
        n_az, n_rng = s.shape

        # --- DIAGNOSTIC (remove after validation) ---
        S_dop_check = np.fft.fft(s, axis=0)
        dop_power = np.mean(np.abs(S_dop_check)**2, axis=1)  # power vs az-freq
        peak_bin = int(np.argmax(dop_power))
        logger.debug(
            "Doppler power peak at bin %d of %d (expected near fd0 bin)",
            peak_bin, s.shape[0]
        )
        # In _slc_to_phase_history, replace the diagnostic block with:
        fa = np.fft.fftfreq(n_az, d=1.0 / prf)

        # First-moment centroid estimator — robust on focused SLC
        # Use middle 10% of range bins to avoid edge effects
        rng_start = n_rng * 45 // 100
        rng_end   = n_rng * 55 // 100
        power_mid = np.mean(np.abs(S_dop_check[:, rng_start:rng_end])**2, axis=1)
        total_pow = power_mid.sum()
        if total_pow > 0:
            measured_fd_centroid = float(np.sum(fa * power_mid) / total_pow)
        else:
            measured_fd_centroid = 0.0

        logger.debug(
            "Doppler centroid (first moment, mid-swath): %.2f Hz  "
            "[peak-bin method gave %.2f Hz — unreliable on focused SLC]",
            measured_fd_centroid,
            (fa[int(np.argmax(power_mid))])
        )

        # Check specific range bins (using the 2D FFT result)
        fa = np.fft.fftfreq(n_az, d=1.0 / prf)
        for rng_bin in [n_rng//4, n_rng//2, 3*n_rng//4]:
            col_power = np.abs(S_dop_check[:, rng_bin])**2
            # Smooth before peak-finding to suppress noise spikes
            col_smooth = np.convolve(col_power, np.ones(64)/64, mode='same')
            bin_idx = int(np.argmax(col_smooth))
            measured_fd = float(fa[bin_idx])
            logger.debug(
                "Rng bin %d: measured fd0=%.1f Hz (smoothed peak)",
                rng_bin, measured_fd
            )

        s = np.fft.fft(s, axis=0)           # 3a  (N_az, N_rng)
        s *= H_az.T                          # 3b  H_az:(N_rng,N_az) -> .T:(N_az,N_rng)

        # Energy check — normalise by N_az to compare same-domain means.
        # np.fft.fft is unnormalised: mean(|FFT(x)|^2) = N_az * mean(|x|^2).
        # Expected ratio after correct inverse filter: az_bw / prf (fraction of
        # azimuth bandwidth selected, typically 0.3–0.7). Values near 0 indicate
        # the fd0 is so misaligned that the in-band gate is empty.
        energy_after_az_norm = float(np.mean(np.abs(s)**2)) / n_az
        energy_before_az = float(np.mean(np.abs(slc_az_rng)**2))
        retention = energy_after_az_norm / (energy_before_az + 1e-30)
        logger.debug(
            "Az filter energy retention (normalised): %.3f "
            "(expect ≈ az_bw/prf, ~0.3–0.7; near 0 → fd0 misaligned)",
            retention
        )

        s = np.fft.ifft(s, axis=0)          # 3c  -> slow-time
        s = np.fft.fft(s, axis=1)           # 3d  -> range-frequency
        s *= H_rng[np.newaxis, :]            # 3e
        return s.T.copy()                    # 3f -> (N_rng, N_az)

    # ------------------------------------------------------------------
    # CPHD metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pvp(
        az_times: np.ndarray,
        pulse_pos: np.ndarray,
        pulse_vel: np.ndarray,
        fc: float,
        bw: float,
        srp_ecf: Optional[np.ndarray],
        approx_notes: List[str],
        invalid_mask: Optional[np.ndarray] = None,
    ) -> CPHDPVP:
        """Populate per-vector parameters from orbit and swath data.

        For a monostatic system (NISAR) transmitter and receiver
        positions/velocities are identical. The two-way travel time to
        the scene centre offsets ``rcv_time`` from ``tx_time``.

        Parameters
        ----------
        az_times : np.ndarray
            Zero-Doppler slow-time per pulse, shape ``(N_az,)``, seconds.
        pulse_pos : np.ndarray
            Interpolated ECF ARP position, shape ``(N_az, 3)``.
        pulse_vel : np.ndarray
            Interpolated ECF ARP velocity, shape ``(N_az, 3)``.
        fc : float
            Processed centre frequency, Hz.
        bw : float
            Processed range bandwidth, Hz.
        srp_ecf : np.ndarray or None
            Scene Reference Point ECF, shape ``(3,)``.  If ``None``
            the ARP at mid-aperture is used as a placeholder.
        approx_notes : List[str]
            Mutable list; approximation notes appended in place.

        Returns
        -------
        CPHDPVP
        """
        n_az = len(az_times)
        if srp_ecf is not None:
            srp_pos = np.tile(srp_ecf.astype(np.float64), (n_az, 1))
        else:
            srp_pos = np.tile(pulse_pos[n_az // 2], (n_az, 1))
            approx_notes.append(
                "pvp.srp_pos: ARP at mid-aperture used as placeholder. "
                "Supply process_params['srp_ecf'] (ECF metres) for a "
                "scene-derived value."
            )

        r_srp = np.linalg.norm(pulse_pos - srp_pos, axis=1)
        rcv_time = az_times + 2.0 * r_srp / _C

        # SIGNAL: 1 = valid pulse, 0 = missing/burst-gap pulse.
        signal = np.ones(n_az, dtype=np.float32)
        if invalid_mask is not None:
            signal[invalid_mask] = 0.0

        return CPHDPVP(
            tx_time=az_times.copy(),
            tx_pos=pulse_pos.copy(),
            tx_vel=pulse_vel.copy(),
            rcv_time=rcv_time,
            rcv_pos=pulse_pos.copy(),
            rcv_vel=pulse_vel.copy(),
            srp_pos=srp_pos,
            fx1=np.full(n_az, fc - bw / 2.0),
            fx2=np.full(n_az, fc + bw / 2.0),
            signal=signal,
        )

    @staticmethod
    def _build_cphd_metadata(
        nisar_meta: NISARMetadata,
        az_times: np.ndarray,
        pulse_pos: np.ndarray,
        pulse_vel: np.ndarray,
        n_rng: int,
        prf: float,
        kr: float,
        pulse_duration_s: float,
        fs: float,
        srp_ecf: Optional[np.ndarray],
        approx_notes: List[str],
        invalid_mask: Optional[np.ndarray] = None,
    ) -> CPHDMetadata:
        """Map NISARMetadata fields to a fully-populated CPHDMetadata.

        Parameters
        ----------
        nisar_meta : NISARMetadata
            Source NISAR metadata with ``swath_parameters`` and ``orbit``.
        az_times : np.ndarray
            Zero-Doppler times for the processed pulse block, seconds.
        pulse_pos : np.ndarray
            Per-pulse ECF ARP positions, shape ``(N_az, 3)``.
        pulse_vel : np.ndarray
            Per-pulse ECF ARP velocities, shape ``(N_az, 3)``.
        n_rng : int
            Number of range samples.
        prf : float
            Output pulse repetition frequency, Hz.
        kr : float
            Range chirp rate, Hz/s (signed; negative for up-chirp).
        pulse_duration_s : float
            Transmit pulse duration used to estimate ``kr``, seconds.
        fs : float
            Range sampling rate, Hz.
        srp_ecf : np.ndarray or None
            Scene Reference Point ECF.
        approx_notes : List[str]
            Mutable list of approximation notes.
        invalid_mask : np.ndarray or None
            Boolean array of shape ``(N_az,)`` marking all-zero (missing)
            azimuth lines.  Passed to ``_build_pvp`` to set ``SIGNAL=0``
            for those pulses.  ``None`` when no invalid lines are detected.

        Returns
        -------
        CPHDMetadata
        """
        sp = nisar_meta.swath_parameters
        orb = nisar_meta.orbit
        ident = nisar_meta.identification

        fc = sp.processed_center_frequency
        bw = sp.processed_range_bandwidth
        n_az = len(az_times)

        pvp = CPHDWriter._build_pvp(
            az_times=az_times,
            pulse_pos=pulse_pos,
            pulse_vel=pulse_vel,
            fc=fc,
            bw=bw,
            srp_ecf=srp_ecf,
            approx_notes=approx_notes,
            invalid_mask=invalid_mask,
        )

        channel = CPHDChannel(
            identifier=(
                f"{nisar_meta.radar_band}_freq{nisar_meta.frequency}"
                f"_{nisar_meta.polarization}"
            ),
            num_vectors=n_az,
            num_samples=n_rng,
        )

        return CPHDMetadata(
            format='NISAR_PSEUDO_CPHD',
            rows=n_az,
            cols=n_rng,
            dtype='complex64',
            channels=[channel],
            num_channels=1,
            pvp=pvp,
            global_params=CPHDGlobal(
                domain_type='FX',
                phase_sgn=-1,
                fx_band_min=fc - bw / 2.0,
                fx_band_max=fc + bw / 2.0,
            ),
            collection_info=CPHDCollectionInfo(
                collector_name='NISAR',
                core_name=ident.granule_id if ident is not None else None,
                classification='UNCLASSIFIED',
                collect_type='MONOSTATIC',
                radar_mode='STRIPMAP',
            ),
            tx_waveform=CPHDTxWaveform(
                lfm_rate=abs(kr),
                pulse_length=pulse_duration_s,
                identifier='NISAR_LFM',
            ),
            rcv_parameters=CPHDRcvParameters(
                sample_rate=fs,
                window_length=n_rng / fs,
                identifier='NISAR_RCV',
            ),
            extras={
                'approximations': approx_notes,
                'frequency': nisar_meta.frequency,
                'polarization': nisar_meta.polarization,
                'radar_band': nisar_meta.radar_band,
                'orbit_type': orb.orbit_type if orb is not None else None,
                'orbit_interp_method': orb.interp_method if orb is not None else None,
                'nisar_fc_hz': fc,
                'nisar_bw_hz': bw,
                'nisar_az_bw_hz': sp.processed_azimuth_bandwidth,
                'nisar_prf_hz': prf,
                'nisar_fs_hz': fs,
                'nisar_kr_hz_s': kr,
                'kr_note': (
                    f"Kr = processedRangeBandwidth / Tp "
                    f"(Tp={pulse_duration_s * 1e6:.0f} us — not stored in NISAR L1)"
                ),
                'tdbp_note': (
                    "TDBP image formation: separable freq-domain inversion "
                    "is approximate (valid near boresight)."
                ),
                'slant_range_min_m': (
                    float(sp.slant_range[0]) if sp.slant_range is not None else None
                ),
                'slant_range_max_m': (
                    float(sp.slant_range[-1]) if sp.slant_range is not None else None
                ),
                'az_time_start_s': float(az_times[0]),
                'az_time_end_s': float(az_times[-1]),
                # ISO 8601 UTC strings for CPHD XML CollectionStart
                'collection_start_utc': (
                    ident.zero_doppler_start_time
                    if ident is not None and ident.zero_doppler_start_time
                    else None
                ),
                'az_time_start_utc': (
                    ident.zero_doppler_start_time
                    if ident is not None and ident.zero_doppler_start_time
                    else None
                ),
                'n_az_processed': n_az,
                # MI-1: document the N_az amplitude scale introduced by the
                # non-normalised forward azimuth FFT (step 3a).  Downstream
                # IFP must divide by this factor to recover NGA-spec radiometry.
                'az_fft_scale_factor': float(n_az),
                'az_fft_scale_note': (
                    f"Phase history amplitudes are scaled by N_az={n_az} relative "
                    "to an ortho-normalised azimuth transform. "
                    "Downstream IFP must divide by N_az to recover NGA-spec radiometry."
                ),
            },
        )

    @staticmethod
    def validate_metadata(cphd_meta: CPHDMetadata) -> List[str]:
        """Check CPHDMetadata for None values in required CPHD fields.

        Parameters
        ----------
        cphd_meta : CPHDMetadata
            Metadata instance to inspect.

        Returns
        -------
        issues : List[str]
            Descriptions of each missing required field.  Empty list
            means the metadata is complete.
        """
        issues: List[str] = []

        if not cphd_meta.channels:
            issues.append("channels: empty — no channel descriptor")
        else:
            ch = cphd_meta.channels[0]
            if not ch.identifier:
                issues.append("channels[0].identifier: empty string")
            if ch.num_vectors == 0:
                issues.append("channels[0].num_vectors: 0")
            if ch.num_samples == 0:
                issues.append("channels[0].num_samples: 0")

        gp = cphd_meta.global_params
        if gp is None:
            issues.append("global_params: None")
        else:
            for attr in ('domain_type', 'fx_band_min', 'fx_band_max'):
                if getattr(gp, attr) is None:
                    issues.append(f"global_params.{attr}: None")

        ci = cphd_meta.collection_info
        if ci is None:
            issues.append("collection_info: None")
        else:
            if ci.collector_name is None:
                issues.append("collection_info.collector_name: None")
            if ci.collect_type is None:
                issues.append("collection_info.collect_type: None")

        tw = cphd_meta.tx_waveform
        if tw is None:
            issues.append("tx_waveform: None")
        else:
            if tw.lfm_rate is None:
                issues.append("tx_waveform.lfm_rate: None — Kr not estimated")
            if tw.pulse_length is None:
                issues.append("tx_waveform.pulse_length: None — Tp not set")

        rp = cphd_meta.rcv_parameters
        if rp is None:
            issues.append("rcv_parameters: None")
        else:
            if rp.sample_rate is None:
                issues.append("rcv_parameters.sample_rate: None — fs not set")
            if rp.window_length is None:
                issues.append("rcv_parameters.window_length: None")

        pvp = cphd_meta.pvp
        if pvp is None:
            issues.append("pvp: None — no per-vector parameters")
        else:
            for fname in ('tx_time', 'tx_pos', 'tx_vel',
                          'rcv_time', 'rcv_pos', 'rcv_vel',
                          'srp_pos', 'fx1', 'fx2'):
                if getattr(pvp, fname) is None:
                    issues.append(f"pvp.{fname}: None")

        return issues

    # ------------------------------------------------------------------
    # CPHD binary writer helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cphd_xml(
        cphd_meta: CPHDMetadata,
        channel_id: str,
        n_az: int,
        n_rng: int,
        az_times: np.ndarray,
    ) -> Any:
        """Build a minimal CPHD 1.1.0 lxml ElementTree from CPHDMetadata.

        Produces an XML tree accepted by ``sarkit.cphd.Writer``.  The tree
        includes CollectionID, Global, Data, PVP, and Channel sections.

        Parameters
        ----------
        cphd_meta : CPHDMetadata
            Populated CPHD metadata (must have global_params, collection_info).
        channel_id : str
            Channel identifier string (matches ``channels[0].identifier``).
        n_az : int
            Number of pulses (slow-time samples = NumVectors).
        n_rng : int
            Number of range samples (NumSamples).
        az_times : np.ndarray
            Per-pulse zero-Doppler times, shape ``(N_az,)``, seconds.

        Returns
        -------
        lxml.etree.ElementTree
        """
        et = _require_lxml()
        ns = _CPHD_NS
        N = lambda tag: f'{{{ns}}}{tag}'  # noqa: E731

        root = et.Element(N('CPHD'))

        # ---- CollectionID -------------------------------------------
        ci = cphd_meta.collection_info
        cid = et.SubElement(root, N('CollectionID'))
        et.SubElement(cid, N('CollectorName')).text = (
            ci.collector_name if ci and ci.collector_name else 'NISAR'
        )
        et.SubElement(cid, N('CoreName')).text = (
            ci.core_name if ci and ci.core_name else channel_id
        )
        et.SubElement(cid, N('CollectType')).text = (
            ci.collect_type if ci and ci.collect_type else 'MONOSTATIC'
        )
        rm = et.SubElement(cid, N('RadarMode'))
        et.SubElement(rm, N('ModeType')).text = (
            ci.radar_mode if ci and ci.radar_mode else 'STRIPMAP'
        )
        et.SubElement(cid, N('Classification')).text = (
            ci.classification if ci and ci.classification else 'UNCLASSIFIED'
        )
        et.SubElement(cid, N('ReleaseInfo')).text = ''

        # ---- Global -------------------------------------------------
        gp = cphd_meta.global_params
        fc = gp.center_frequency if gp else 0.0
        bw = gp.bandwidth if gp else 0.0
        glb = et.SubElement(root, N('Global'))
        et.SubElement(glb, N('DomainType')).text = (
            gp.domain_type if gp and gp.domain_type else 'FX'
        )
        et.SubElement(glb, N('SGN')).text = str(gp.phase_sgn if gp else -1)
        tl = et.SubElement(glb, N('Timeline'))
        t_start = float(az_times[0])
        t_end = float(az_times[-1])
        # Prefer the ISO 8601 collection-start stored from NISARIdentification;
        # fall back to TxTime1 only as a last resort (not spec-compliant but
        # avoids a hard failure when identification metadata is absent).
        collection_start_iso = cphd_meta.extras.get('collection_start_utc')
        if not isinstance(collection_start_iso, str) or not collection_start_iso:
            collection_start_iso = cphd_meta.extras.get('az_time_start_utc', '')
        if not collection_start_iso:
            collection_start_iso = 'UNKNOWN'  # surfaced clearly rather than silently wrong
        et.SubElement(tl, N('CollectionStart')).text = collection_start_iso
        et.SubElement(tl, N('TxTime1')).text = repr(t_start)
        et.SubElement(tl, N('TxTime2')).text = repr(t_end)
        fb = et.SubElement(glb, N('FxBand'))
        et.SubElement(fb, N('FxMin')).text = repr(
            gp.fx_band_min if gp and gp.fx_band_min is not None else fc - bw / 2.0
        )
        et.SubElement(fb, N('FxMax')).text = repr(
            gp.fx_band_max if gp and gp.fx_band_max is not None else fc + bw / 2.0
        )
        # TOASwath: 2*(R_far - R_near)/c relative to SRP range
        r_min = cphd_meta.extras.get('slant_range_min_m', 0.0) or 0.0
        r_max = cphd_meta.extras.get('slant_range_max_m', 1.0) or 1.0
        r_mid = 0.5 * (r_min + r_max)
        toa_half = (r_max - r_min) / _C
        toa_sw = et.SubElement(glb, N('TOASwath'))
        et.SubElement(toa_sw, N('TOAMin')).text = repr(-toa_half)
        et.SubElement(toa_sw, N('TOAMax')).text = repr(toa_half)

        # ---- Data ---------------------------------------------------
        data_el = et.SubElement(root, N('Data'))
        et.SubElement(data_el, N('SignalArrayFormat')).text = 'CF8'
        et.SubElement(data_el, N('NumBytesPVP')).text = str(_NUM_BYTES_PVP)
        et.SubElement(data_el, N('NumCPHDChannels')).text = '1'
        ch_data = et.SubElement(data_el, N('Channel'))
        et.SubElement(ch_data, N('Identifier')).text = channel_id
        et.SubElement(ch_data, N('NumVectors')).text = str(n_az)
        et.SubElement(ch_data, N('NumSamples')).text = str(n_rng)
        et.SubElement(ch_data, N('SignalArrayByteOffset')).text = '0'
        et.SubElement(ch_data, N('PVPArrayByteOffset')).text = '0'

        # ---- Channel (top-level, §7.5 of CPHD 1.1.0) ---------------
        # Distinct from Data/Channel (which carries byte offsets).
        ch_el = et.SubElement(root, N('Channel'))
        et.SubElement(ch_el, N('RefChId')).text = channel_id
        ch_params = et.SubElement(ch_el, N('Parameters'))
        et.SubElement(ch_params, N('Identifier')).text = channel_id
        # RefVectorIndex: 0-based index of the reference PVP vector.
        et.SubElement(ch_params, N('RefVectorIndex')).text = str(n_az // 2)
        # For pseudo-CPHD the FX band and TOA window are fixed per pulse.
        et.SubElement(ch_params, N('FXFixed')).text = 'true'
        et.SubElement(ch_params, N('TOAFixed')).text = 'true'
        et.SubElement(ch_params, N('SRPFixed')).text = 'false'

        # ---- PVP ----------------------------------------------------
        pvp_el = et.SubElement(root, N('PVP'))
        for fname, offset, size, fmt in _PVP_FIELDS:
            f_el = et.SubElement(pvp_el, N(fname))
            et.SubElement(f_el, N('Offset')).text = str(offset)
            et.SubElement(f_el, N('Size')).text = str(size)
            et.SubElement(f_el, N('Format')).text = fmt

        return et.ElementTree(root)

    @staticmethod
    def _build_pvp_array(
        pvp: CPHDPVP,
        pvp_dtype: Any,
        n_az: int,
        n_rng: int,
        bw: float,
        slant_range_min: float,
        slant_range_max: float,
    ) -> np.ndarray:
        """Build a structured PVP array for sarkit ``write_pvp()``.

        Fields present in ``pvp`` are copied directly.  Missing mandatory
        CPHD fields are derived or zeroed with notes in the module
        docstring.

        Parameters
        ----------
        pvp : CPHDPVP
            Per-vector parameters from ``nisar_to_cphd``.
        pvp_dtype : numpy.dtype
            Structured dtype returned by ``sarkit.cphd.get_pvp_dtype()``.
        n_az : int
            Number of pulses.
        n_rng : int
            Number of range samples.
        bw : float
            Processed range bandwidth, Hz.  Used to compute SCSS.
        slant_range_min : float
            Near-range slant range, metres.
        slant_range_max : float
            Far-range slant range, metres.

        Returns
        -------
        pvp_arr : np.ndarray
            Structured array, shape ``(n_az,)``, dtype ``pvp_dtype``.
        """
        pvp_arr = np.zeros(n_az, dtype=pvp_dtype)

        pvp_arr['TxTime']  = pvp.tx_time.astype(np.float64)
        pvp_arr['TxPos']   = pvp.tx_pos.astype(np.float64)
        pvp_arr['TxVel']   = pvp.tx_vel.astype(np.float64)
        pvp_arr['RcvTime'] = pvp.rcv_time.astype(np.float64)
        pvp_arr['RcvPos']  = pvp.rcv_pos.astype(np.float64)
        pvp_arr['RcvVel']  = pvp.rcv_vel.astype(np.float64)
        pvp_arr['SRPPos']  = pvp.srp_pos.astype(np.float64)
        pvp_arr['FX1']     = pvp.fx1.astype(np.float64)
        pvp_arr['FX2']     = pvp.fx2.astype(np.float64)

        # SIGNAL validity indicator (1 = valid).
        if pvp.signal is not None:
            pvp_arr['SIGNAL'] = pvp.signal.astype(np.int64)
        else:
            pvp_arr['SIGNAL'] = 1

        # SC0: start frequency of the receive window = FX1 per pulse.
        pvp_arr['SC0'] = pvp_arr['FX1']

        # SCSS: frequency step between adjacent samples.
        pvp_arr['SCSS'] = bw / n_rng if n_rng > 0 else 0.0

        # TOA1/TOA2: per-pulse two-way delay relative to the instantaneous
        # ARP-to-SRP slant range.  Using per-pulse r_srp(t) rather than a
        # fixed midpoint compensates for the platform moving across the
        # aperture (relevant for long CPI or squinted modes).
        r_srp_per_pulse = np.linalg.norm(
            pvp.tx_pos.astype(np.float64) - pvp.srp_pos.astype(np.float64), axis=1
        )  # (n_az,)
        pvp_arr['TOA1'] = 2.0 * (slant_range_min - r_srp_per_pulse) / _C
        pvp_arr['TOA2'] = 2.0 * (slant_range_max - r_srp_per_pulse) / _C

        # aFDOP, aFRR1, aFRR2, TDTropoSRP: defaulted to 0.0.
        # These are approximations; see module-level docstring.

        return pvp_arr

    # ------------------------------------------------------------------
    # Core conversion
    # ------------------------------------------------------------------

    def nisar_to_cphd(
            self,
            slc: np.ndarray,
            nisar_meta: NISARMetadata,
            process_params: Optional[Dict[str, Any]] = None,
        ) -> Tuple[np.ndarray, CPHDMetadata]:
            """Convert a NISAR RSLC SLC chip to pseudo-CPHD phase history.

            NOTE: If you pass a chunked SLC (subset of azimuth lines), all per-azimuth metadata arrays
            (e.g., zero_doppler_time) in nisar_meta must be sliced to match the chunk. Otherwise, validation
            or interpolation errors will occur.

            Parameters
            ----------
            slc : np.ndarray
                Complex SLC data, shape ``(N_az, N_rng)``.  Cast internally
                to complex64 before processing.
            nisar_meta : NISARMetadata
                Metadata from ``NISARReader.metadata``.  Must be an RSLC
                product (``swath_parameters`` and ``orbit`` populated).
            process_params : dict, optional
                Supplementary parameters not stored in the NISAR L1 product.

                ``chunk_az_start`` : int, default 0
                    The starting azimuth index of this chunk relative to the full SLC.
                    Used to correctly index the Doppler Centroid grid.

                ``pulse_duration_s`` : float, default 35e-6
                    Transmit pulse duration (seconds).  Used to estimate
                    ``Kr = processedRangeBandwidth / Tp``.
                    **Not stored in NISAR L1; default is nominal L-band.**

                ``doppler_centroid_hz`` : float, default 0.0
                    Mid-aperture Doppler centroid (Hz) for the azimuth
                    decompression filter.

                ``range_weighting`` : np.ndarray
                    1-D range chirp weighting function.  Read from
                    ``processingInformation/parameters/rangeChirpWeighting``.

                ``az_weighting`` : np.ndarray
                    1-D azimuth chirp weighting function.  Read from
                    ``processingInformation/parameters/azimuthChirpWeighting``.

                ``srp_ecf`` : np.ndarray, shape (3,)
                    Scene Reference Point ECF metres.

                ``prf_override`` : float
                    Override the effective PRF (Hz).

                ``doppler_centroid_grid`` : np.ndarray, shape (N_rng,)
                    Range-variant Doppler centroid, Hz.  If provided, a
                    separate centroid is applied per range bin, correcting
                    for scene Doppler variation across the swath (squint,
                    crab-angle).  Interpolate the 2D HDF5
                    ``processingInformation/parameters/frequencyX/dopplerCentroid``
                    grid onto the slant-range vector to produce this array.
                    Takes priority over ``doppler_centroid_hz`` if both are set.

                ``chirp_sign`` : int, default +1
                    Transmit chirp direction: ``+1`` for up-chirp (NISAR,
                    ascending LFM rate) or ``-1`` for down-chirp.  The
                    inverse range filter ``exp(-j*pi*f^2/Kr)`` handles
                    both signs correctly when ``Kr`` is signed.

                ``az_time_offset_s`` : float, default 0.0
                    **Advanced.** Offset (in seconds) to add to the azimuth time vector (``az_times``) before orbit interpolation. Use this to correct for mismatched reference epochs between the orbit and zero-Doppler time vectors. If the orbit and azimuth time epochs differ, set this to the required offset (e.g., ``az_time_offset_s = orbit_epoch - az_epoch``).

            Returns
            -------
            phase_history : np.ndarray
                Pseudo phase history, shape ``(N_rng, N_az)``, complex64.
                Axis 0 is range/frequency; axis 1 is slow-time/pulse index.
            cphd_meta : CPHDMetadata
                Fully populated CPHD metadata.  Approximation notes are in
                ``cphd_meta.extras['approximations']``.  Call
                ``CPHDWriter.validate_metadata(cphd_meta)`` to verify
                completeness before writing.

            Raises
            ------
            ValidationError
                If required metadata sections are absent, if the SLC shape
                is inconsistent with the swath parameters, or if
                ``srp_ecf`` has the wrong shape.
            ProcessorError
                If the decompression arithmetic raises an unexpected
                exception.
            """
            # ---- validation -----------------------------------------------
            if nisar_meta.swath_parameters is None:
                raise ValidationError(
                    "NISARMetadata.swath_parameters is None. "
                    "CPHDWriter requires an RSLC product opened via NISARReader."
                )
            if nisar_meta.orbit is None:
                raise ValidationError(
                    "NISARMetadata.orbit is None. "
                    "Orbit state vectors are required for phase-history inversion."
                )
            if slc.ndim != 2:
                raise ValidationError(
                    f"slc must be 2-D (N_az, N_rng), got shape {slc.shape}."
                )

            sp = nisar_meta.swath_parameters
            orb = nisar_meta.orbit
            n_az, n_rng = slc.shape

            if sp.slant_range is not None and n_rng != sp.slant_range.shape[0]:
                raise ValidationError(
                    f"slc has {n_rng} range samples but swath_parameters "
                    f"slant_range has {sp.slant_range.shape[0]} elements."
                )

            # ---- process_params defaults ----------------------------------
            pp = process_params or {}
            chunk_az_start = int(pp.get('chunk_az_start', 0))
            
            # Select band-appropriate default pulse duration if caller doesn't override.
            _default_tp = (
                _NISAR_S_NOMINAL_PULSE_DURATION_S
                if (nisar_meta.radar_band or '').upper() == 'S'
                else _NISAR_L_NOMINAL_PULSE_DURATION_S
            )
            pulse_duration_s = float(pp.get('pulse_duration_s', _default_tp))
            
            # chirp_sign: +1 = up-chirp (NISAR default), -1 = down-chirp.
            chirp_sign = int(pp.get('chirp_sign', +1))
            if chirp_sign not in (+1, -1):
                raise ValidationError(
                    "process_params['chirp_sign'] must be +1 (up-chirp) or -1 (down-chirp), "
                    f"got {chirp_sign}."
                )
                
            fd0 = float(pp.get('doppler_centroid_hz', 0.0))
            
            # doppler_centroid_grid: optional (N_rng,) array for range-variant centroid.
            fd0_grid_raw = pp.get('doppler_centroid_grid', None)
            fd0_grid: Optional[np.ndarray] = None
            
            if fd0_grid_raw is not None:
                # Explicit override supplied — validate and use directly.
                fd0_grid = np.asarray(fd0_grid_raw, dtype=np.float64)
                if fd0_grid.ndim != 1 or len(fd0_grid) != n_rng:
                    raise ValidationError(
                        f"process_params['doppler_centroid_grid'] must be a 1-D array "
                        f"of length {n_rng} (N_rng), got shape {fd0_grid.shape}."
                    )
            elif (
                sp.doppler_centroid is not None
                and sp.doppler_centroid_slant_range is not None
                and sp.slant_range is not None
            ):
                # Derive from NISARSwathParameters using chunk-aware azimuth indexing
                dc2d = sp.doppler_centroid
                dc_sr = sp.doppler_centroid_slant_range
                
                chunk_az_mid = chunk_az_start + n_az // 2
                
                if sp.doppler_centroid_azimuth_time is not None:
                    # Interpolate using the DC grid's own azimuth-time axis.
                    # chunk_az_mid is a *line index* in the full SLC — convert
                    # it to a time value via zero_doppler_time if available,
                    # otherwise fall through to the fraction-based branch.
                    if sp.zero_doppler_time is not None and len(sp.zero_doppler_time) > 0:
                        az_mid_idx = min(chunk_az_mid, len(sp.zero_doppler_time) - 1)
                        t_mid = float(sp.zero_doppler_time[az_mid_idx])
                    else:
                        # No time vector available to anchor the line index;
                        # raise so the caller knows they must provide one.
                        raise ValidationError(
                            "doppler_centroid_azimuth_time is present but "
                            "zero_doppler_time is absent: cannot map chunk line "
                            "index to a DC grid row without an azimuth time vector."
                        )
                    dc_row_frac = np.interp(
                        t_mid,
                        sp.doppler_centroid_azimuth_time.astype(np.float64),
                        np.arange(dc2d.shape[0], dtype=np.float64),
                    )
                    dc_row = int(np.clip(int(round(float(dc_row_frac))), 0, dc2d.shape[0] - 1))
                elif sp.zero_doppler_time is not None and len(sp.zero_doppler_time) > 0:
                    # Map chunk mid-line to a time, then scale that time into the DC grid
                    # using the swath's start/end times as anchors.
                    az_mid_idx = int(np.clip(chunk_az_mid, 0, len(sp.zero_doppler_time) - 1))
                    t_mid = float(sp.zero_doppler_time[az_mid_idx])
                    t_start = float(sp.zero_doppler_time[0])
                    t_end   = float(sp.zero_doppler_time[-1])
                    frac = (t_mid - t_start) / max(t_end - t_start, 1e-9)
                    # DC grid spans a wider time range than the swath — we don't know
                    # DC grid t_start/t_end without doppler_centroid_azimuth_time, so
                    # this is still approximate, but far better than line-index scaling.
                    dc_row = int(round(frac * (dc2d.shape[0] - 1)))
                    dc_row = int(np.clip(dc_row, 0, dc2d.shape[0] - 1))
                    logger.warning(
                        "doppler_centroid_azimuth_time unavailable — DC row estimated from "
                        "swath time fraction (row %d). Populate sp.doppler_centroid_azimuth_time "
                        "from HDF5 path .../processingInformation/parameters/frequencyA/zeroDopplerTime "
                        "for accurate row selection.",
                        dc_row
                    )
                else:
                    # Last resort: use middle of DC grid
                    dc_row = dc2d.shape[0] // 2
                    
                fd0_grid = np.interp(
                    sp.slant_range.astype(np.float64),
                    dc_sr.astype(np.float64),
                    dc2d[dc_row, :].astype(np.float64),
                )
                # dc_row logged after prf is resolved (folding requires prf).

            srp_ecf_raw = pp.get('srp_ecf', None)
            srp_ecf: Optional[np.ndarray] = None
            if srp_ecf_raw is not None:
                srp_ecf = np.asarray(srp_ecf_raw, dtype=np.float64)
                if srp_ecf.shape != (3,):
                    raise ValidationError(
                        f"process_params['srp_ecf'] must have shape (3,), "
                        f"got {srp_ecf.shape}."
                    )

            uniform = np.ones(_WEIGHTING_SAMPLES, dtype=np.float32)
            rng_wgt = np.asarray(pp.get('range_weighting', uniform), dtype=np.float64)
            az_wgt = np.asarray(pp.get('az_weighting', uniform), dtype=np.float64)

            # ---- derived scalars ----------------------------------------
            for _attr, _label in (
                ('processed_center_frequency', 'swath_parameters.processed_center_frequency'),
                ('processed_range_bandwidth',  'swath_parameters.processed_range_bandwidth'),
                ('slant_range_spacing',        'swath_parameters.slant_range_spacing'),
            ):
                if getattr(sp, _attr) is None:
                    raise ValidationError(
                        f"NISARMetadata.swath_parameters.{_attr} is None. "
                        "This field is required for phase-history inversion."
                    )

            fc = sp.processed_center_frequency
            bw = sp.processed_range_bandwidth
            az_bw = sp.processed_azimuth_bandwidth
            lam = _C / fc
            fs = _C / (2.0 * sp.slant_range_spacing)
            kr = chirp_sign * bw / pulse_duration_s  # signed: +ve = up-chirp (NISAR)

            prf_override = pp.get('prf_override', None)
            if prf_override is not None:
                prf = float(prf_override)
            elif sp.zero_doppler_time_spacing:
                prf = 1.0 / sp.zero_doppler_time_spacing
            else:
                prf = float(sp.nominal_acquisition_prf)

            if fd0_grid is not None and pp.get('doppler_centroid_grid') is None:
                # NISAR dopplerCentroid is already in (-PRF/2, +PRF/2].
                # Just verify it's in range — if values are outside, that indicates
                # a metadata convention difference that needs explicit handling.
                fd0_min, fd0_max = float(fd0_grid.min()), float(fd0_grid.max())
                if fd0_max > prf / 2.0 or fd0_min < -prf / 2.0:
                    # Values outside [-PRF/2, PRF/2) — fold once
                    fd0_grid = ((fd0_grid + prf / 2.0) % prf) - prf / 2.0
                    logger.debug(
                        "doppler_centroid_grid: values outside [-PRF/2, PRF/2), folded to "
                        "%.1f–%.1f Hz", float(fd0_grid.min()), float(fd0_grid.max())
                    )
                else:
                    logger.debug(
                        "doppler_centroid_grid: already in [-PRF/2, PRF/2): %.1f–%.1f Hz",
                        fd0_min, fd0_max
                    )
            # ---- az-time vector -----------------------------------------
            if sp.zero_doppler_time is not None:
                if len(sp.zero_doppler_time) != n_az:
                    logger.warning(
                        "zero_doppler_time has %d elements but slc has %d azimuth lines. "
                        "Truncating/padding to match slc. Ensure NISARMetadata is "
                        "pre-sliced to the correct chunk window for mid-scene subsets.",
                        len(sp.zero_doppler_time), n_az,
                    )
                az_times = sp.zero_doppler_time[:n_az].astype(np.float64)
            else:
                az_times = np.arange(n_az, dtype=np.float64) / prf

            # ---- orbit/az_times epoch consistency check -------------------
            orb_epoch = getattr(orb, 'reference_epoch', None)
            az_epoch = getattr(sp, 'zero_doppler_time_reference_epoch', None)
            if orb_epoch is not None and az_epoch is not None and orb_epoch != az_epoch:
                raise ValidationError(
                    f"Orbit time frame epoch ({orb_epoch}) does not match zero-Doppler time epoch ({az_epoch}). "
                    "You must convert az_times to the orbit reference frame or supply a time offset via process_params['az_time_offset_s']."
                )
            
            az_time_offset = float(pp.get('az_time_offset_s', 0.0))
            if az_time_offset != 0.0:
                az_times = az_times + az_time_offset

            # ---- missing pulse / burst gap detection ---------------------
            invalid_mask = ~np.any(slc != 0, axis=1)  # True where line is all zeros
            n_invalid = int(np.sum(invalid_mask))
            if n_invalid > 0:
                logger.warning(
                    "Detected %d all-zero azimuth lines (missing pulses / burst gaps). "
                    "These will be marked SIGNAL=0 in the PVP array.",
                    n_invalid,
                )

            # ---- approximation notes ------------------------------------
            az_fd0: Union[float, np.ndarray] = fd0_grid if fd0_grid is not None else fd0
            approx_notes: List[str] = [
                f"Kr = processedRangeBandwidth / Tp "
                f"(Tp = {pulse_duration_s * 1e6:.0f} us — not stored in NISAR L1).",
                "TDBP image formation: separable freq-domain inversion is "
                "approximate (valid near boresight).",
            ]
            if fd0_grid is not None:
                approx_notes.append(
                    f"doppler_centroid_grid: range-variant centroid applied "
                    f"({float(fd0_grid.min()):.1f} – {float(fd0_grid.max()):.1f} Hz)"
                    + (
                        " [auto-derived from NISARSwathParameters.doppler_centroid]."
                        if pp.get('doppler_centroid_grid') is None
                        else " [caller-supplied]."
                    )
                )
            elif fd0 == 0.0:
                approx_notes.append(
                    "doppler_centroid_hz defaulted to 0.0 Hz. "
                    "Pass process_params['doppler_centroid_grid'] (shape N_rng) or "
                    "'doppler_centroid_hz' for improved accuracy."
                )
            if chirp_sign == -1:
                approx_notes.append("chirp_sign=-1: down-chirp inversion applied.")
            if np.all(rng_wgt == 1.0):
                approx_notes.append(
                    "range_weighting defaulted to uniform (no tapering correction)."
                )
            if np.all(az_wgt == 1.0):
                approx_notes.append("az_weighting defaulted to uniform.")

            logger.info(
                "nisar_to_cphd: freq=%s pol=%s n_az=%d n_rng=%d "
                "fc=%.3f GHz bw=%.1f MHz fs=%.2f MHz prf=%.2f Hz kr=%.3e Hz/s",
                nisar_meta.frequency, nisar_meta.polarization,
                n_az, n_rng, fc / 1e9, bw / 1e6, fs / 1e6, prf, kr,
            )

            # ---- orbit interpolation ------------------------------------
            try:
                p_pos, p_vel = self._interp_orbit(
                    orb.time, orb.position, orb.velocity, az_times
                )
            except (ValidationError, Exception) as exc:
                raise ProcessorError(f"Orbit interpolation failed: {exc}") from exc

            # ---- inverse filters ----------------------------------------
            ka = self._ka_from_orbit(p_vel, sp.slant_range, lam)
            H_rng = self._build_range_inv_filter(n_rng, fs, bw, kr, rng_wgt)
            H_az = self._build_az_inv_filter(n_az, prf, ka, az_fd0, az_bw, az_wgt)

            # ---- decompression ------------------------------------------
            try:
                phase_history = self._slc_to_phase_history(slc, H_az, H_rng, prf)
            except Exception as exc:
                raise ProcessorError(
                    f"SLC decompression failed at step 3a-3f: {exc}"
                ) from exc

            # ---- CPHD metadata ------------------------------------------
            cphd_meta = self._build_cphd_metadata(
                nisar_meta=nisar_meta,
                az_times=az_times,
                pulse_pos=p_pos,
                pulse_vel=p_vel,
                n_rng=n_rng,
                prf=prf,
                kr=kr,
                pulse_duration_s=pulse_duration_s,
                fs=fs,
                srp_ecf=srp_ecf,
                approx_notes=approx_notes,
                invalid_mask=invalid_mask if n_invalid > 0 else None,
            )

            issues = self.validate_metadata(cphd_meta)
            if issues:
                logger.warning(
                    "CPHDMetadata completeness issues (%d):\n  %s",
                    len(issues), "\n  ".join(issues),
                )
            else:
                logger.debug("CPHDMetadata completeness check passed.")

            self.metadata = cphd_meta
            logger.info(
                "nisar_to_cphd complete: phase_history shape=%s dtype=%s",
                phase_history.shape, phase_history.dtype,
            )
            return phase_history, cphd_meta

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    def write(
        self,
        data: np.ndarray,
        geolocation: Optional[Dict[str, Any]] = None,
        *,
        cphd_meta: Optional[CPHDMetadata] = None,
        filename: Optional[Union[str, Path]] = None,
    ) -> None:
        """Write phase history and metadata to a CPHD 1.1.0 binary file.

        Satisfies ``ImageWriter.write()``.  Uses sarkit as the CPHD
        backend.  The ``data`` array (N_rng, N_az) is transposed to
        CPHD signal layout (NumVectors=N_az, NumSamples=N_rng) before
        writing.

        Parameters
        ----------
        data : np.ndarray
            Phase history, shape ``(N_rng, N_az)``, complex64.
            Produced by ``nisar_to_cphd()``.
        geolocation : dict, optional
            Ignored.  Present for ``ImageWriter`` ABC compatibility.
        cphd_meta : CPHDMetadata, optional
            CPHD metadata to embed.  Defaults to ``self.metadata``.
        filename : str or Path, optional
            Output path override.  Defaults to ``self.filepath``.

        Raises
        ------
        DependencyError
            If sarkit or lxml is not installed.
        ValidationError
            If no CPHDMetadata is available or required fields are missing.
        ProcessorError
            If the sarkit write operation fails.
        """
        sc = _require_sarkit()
        _require_lxml()

        meta = cphd_meta if cphd_meta is not None else self.metadata
        if meta is None:
            raise ValidationError(
                "No CPHDMetadata available. Either pass cphd_meta= or call "
                "nisar_to_cphd() first to populate self.metadata."
            )

        issues = self.validate_metadata(meta)
        if issues:
            raise ValidationError(
                f"CPHDMetadata has {len(issues)} missing required field(s):\n"
                + "\n".join(f"  {i}" for i in issues)
            )

        out_path = Path(filename) if filename is not None else self.filepath
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # CPHD signal layout: (NumVectors=N_az, NumSamples=N_rng)
        # phase_history from nisar_to_cphd is (N_rng, N_az), so transpose.
        if data.ndim != 2:
            raise ValidationError(
                f"data must be 2-D (N_rng, N_az), got shape {data.shape}."
            )
        signal_array = data.T.astype(np.complex64, copy=False)   # (N_az, N_rng)
        # Ensure C-contiguity for sarkit compatibility
        signal_array = np.ascontiguousarray(signal_array)
        n_az, n_rng = signal_array.shape

        channel_id = meta.channels[0].identifier if meta.channels else 'ch0'
        sp = meta.swath_parameters if hasattr(meta, 'swath_parameters') else None
        bw = (meta.global_params.bandwidth if meta.global_params else
              meta.extras.get('nisar_bw_hz', 1.0))
        r_min = meta.extras.get('slant_range_min_m', 0.0) or 0.0
        r_max = meta.extras.get('slant_range_max_m', 1.0) or 1.0

        pvp = meta.pvp
        if pvp is None:
            raise ValidationError("CPHDMetadata.pvp is None — cannot build PVP array.")

        # Recover az_times from pvp.tx_time for XML TxTime1/TxTime2
        az_times = pvp.tx_time if pvp.tx_time is not None else np.zeros(n_az)

        # Build lxml XML tree
        xmltree = self._build_cphd_xml(meta, channel_id, n_az, n_rng, az_times)

        # Build structured PVP array matching the XML-declared dtype
        pvp_dtype = sc.get_pvp_dtype(xmltree)
        pvp_array = self._build_pvp_array(
            pvp=pvp,
            pvp_dtype=pvp_dtype,
            n_az=n_az,
            n_rng=n_rng,
            bw=float(bw),
            slant_range_min=float(r_min),
            slant_range_max=float(r_max),
        )

        sarkit_meta = sc.Metadata(xmltree=xmltree)

        try:
            with open(str(out_path), 'wb') as fh:
                writer = sc.Writer(fh, sarkit_meta)
                writer.write_pvp(channel_id, pvp_array)
                writer.write_signal(channel_id, signal_array)
                writer.done()
        except Exception as exc:
            raise ProcessorError(
                f"sarkit CPHD write failed for {out_path}: {exc}"
            ) from exc

        size_mb = out_path.stat().st_size / 1e6
        logger.info("Saved %s (%.1f MB)", out_path.name, size_mb)

    def write_chip(
        self,
        data: np.ndarray,
        row_start: int,
        col_start: int,
        geolocation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Not supported — CPHD must be written as a complete array.

        Raises
        ------
        ValidationError
            Always.
        """
        raise ValidationError(
            "CPHDWriter.write_chip() is not supported. "
            "Phase history must be written as a complete array via write()."
        )
