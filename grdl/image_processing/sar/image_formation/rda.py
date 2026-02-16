# -*- coding: utf-8 -*-
"""
Range-Doppler Algorithm - Stripmap SAR image formation from CPHD.

Implements the Range-Doppler Algorithm (RDA) for forming complex SAR
images from CPHD (Compensated Phase History Data) in FX domain.
The single-reference pipeline processes the aperture coherently:

1. **Rephase** — compensate per-pulse SRP drift to a common reference.
2. **Range compress** — IFFT along fast-time to range-time domain.
3. **Deramp** — remove range-dependent residual quadratic phase so
   all targets have linear-only slow-time phase.
4. **Azimuth FFT** — transform to range-Doppler domain; since all
   targets are linear after deramp, the FFT focuses them directly.
5. **RCMC** — range cell migration correction via ``range_axis*(1/D-1)``.
6. **Azimuth windowing** — optional sidelobe control.

Output is in range-Doppler domain (RGZERO grid).  No matched filter
or IFFT is needed — the deramp + FFT approach avoids the bandwidth
limitation of the common-quadratic + matched-filter pipeline, whose
chirp bandwidth ``|Ka|*T`` exceeds the PRF for long strips.

Each stage is exposed as a public method for intermediate inspection,
following the same tapout pattern as ``PolarFormatAlgorithm``.

.. note:: Azimuth Compression

   CPHD rephasing removes the quadratic phase at R0, leaving only
   a residual at other ranges.

   **Single-reference mode** (``block_size=None``): a deramp step
   removes the range-dependent residual quadratic ``(1/R0 - 1/R)``
   *before* the azimuth FFT.  All targets then have linear-only
   slow-time phase, so the FFT focuses them directly — no matched
   filter or IFFT.  Output is in the range-Doppler (RGZERO) domain.
   This avoids the bandwidth limitation of the common-quadratic
   approach, whose chirp BW ``|Ka|*T`` exceeds the PRF for long
   strips.

   **Subaperture mode** (``block_size`` parameter) processes
   overlapping azimuth blocks, each locally rephased, with per-block
   common quadratic + matched filtering after RCMC, then
   Hann-weighted mosaicking.  Auto block sizing limits the block
   length so the chirp bandwidth ``|Ka|*N*PRI`` stays within the
   PRF (with 10 % guard band).

Dependencies
------------
scipy

Author
------
Duane Smalley, PhD
duane.d.smalley@gmail.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-12

Modified
--------
2026-02-15
"""


# Standard library
from typing import Any, Callable, Dict, Optional, Union

# Third-party
import numpy as np
from numpy.linalg import norm
from scipy.interpolate import interp1d
from scipy.signal.windows import taylor as _taylor_window

# GRDL internal
from grdl.image_processing.sar.image_formation.base import (
    ImageFormationAlgorithm,
)
from grdl.IO.models.cphd import CPHDMetadata, CPHDPVP


# Speed of light (m/s)
_C = 299792458.0

_WINDOW_FUNCTIONS = {
    'uniform': None,
    'taylor': lambda n: _taylor_window(n, nbar=4, sll=35, norm=True),
    'hamming': np.hamming,
    'hanning': np.hanning,
}


def _scipy_interp1d(
    x_old: np.ndarray,
    y_old: np.ndarray,
    x_new: np.ndarray,
) -> np.ndarray:
    """Default interpolation using scipy interp1d (linear).

    Parameters
    ----------
    x_old : np.ndarray
        Original sample coordinates.
    y_old : np.ndarray
        Original sample values (complex).
    x_new : np.ndarray
        Target sample coordinates.

    Returns
    -------
    np.ndarray
        Interpolated values at ``x_new``.
    """
    func = interp1d(
        x_old, y_old,
        kind='linear',
        bounds_error=False,
        fill_value=0.0,
    )
    return func(x_new)


class RangeDopplerAlgorithm(ImageFormationAlgorithm):
    """Range-Doppler Algorithm for stripmap SAR image formation.

    **Single-reference mode** (default): five-stage pipeline:

    0. Rephase to common reference SRP
    1. Range compression (IFFT along fast-time)
    1.5. Azimuth deramp (range-dependent residual quadratic removal)
    2. Azimuth FFT (slow-time to Doppler — focuses all targets)
    3. Differential RCMC (range cell migration correction)
    4. Azimuth windowing

    **Subaperture mode** (``block_size`` parameter): overlapping
    azimuth blocks, each locally rephased, baseband-demodulated,
    windowed, and FFT'd, then mosaicked with Hann blending.
    Produces well-focused imagery across the full strip.

    Each single-reference stage is a public method for inspection.

    Parameters
    ----------
    metadata : CPHDMetadata
        CPHD metadata with populated PVP arrays.
    interpolator : callable, optional
        Interpolation function ``(x_old, y_old, x_new) -> y_new``
        for RCMC. Defaults to linear ``scipy.interpolate.interp1d``.
    range_weighting : str or callable, optional
        Window applied before range IFFT. Built-in options:
        ``'uniform'``, ``'taylor'``, ``'hamming'``, ``'hanning'``,
        or a callable ``f(n) -> ndarray``.
    azimuth_weighting : str or callable, optional
        Window applied during azimuth compression. Same options.
    block_size : int or str, optional
        Number of pulses per subaperture block. When set to an int,
        ``form_image`` uses overlapping-block processing with local
        rephasing and Hann-weighted mosaicking. When ``'auto'``,
        the block size is computed from a phase error budget of
        pi/4 radians (depth-of-focus limited). When None (default),
        the full aperture is processed as a single reference point.
    overlap : float
        Overlap fraction between adjacent subaperture blocks,
        in [0.0, 1.0). Default 0.5 (50% overlap).
    apply_amp_sf : bool
        If True (default), apply the per-pulse AmpSF normalization
        from CPHD metadata before processing.
    trim_invalid : bool
        If True (default), discard pulses with invalid signal
        indicator (SIGNAL == 0) or anomalous AmpSF.
    antenna_compensation : bool
        If True (default False), apply antenna pattern compensation
        per subaperture block using the antenna gain polynomial from
        CPHD metadata. Requires ``antenna_pattern`` in metadata.
    verbose : bool
        Print per-stage diagnostics. Default True.

    Examples
    --------
    >>> from grdl.IO.sar import CPHDReader
    >>> from grdl.image_processing.sar.image_formation import (
    ...     RangeDopplerAlgorithm,
    ... )
    >>> with CPHDReader('stripmap.cphd') as reader:
    ...     meta = reader.metadata
    ...     signal = reader.read_full()
    >>> rda = RangeDopplerAlgorithm(
    ...     meta, range_weighting='taylor', block_size='auto',
    ... )
    >>> image = rda.form_image(signal, geometry=None)
    """

    def __init__(
        self,
        metadata: CPHDMetadata,
        interpolator: Optional[Callable] = None,
        range_weighting: Union[str, Callable, None] = None,
        azimuth_weighting: Union[str, Callable, None] = None,
        block_size: Union[int, str, None] = None,
        overlap: float = 0.5,
        apply_amp_sf: bool = True,
        trim_invalid: bool = True,
        antenna_compensation: bool = False,
        verbose: bool = True,
    ) -> None:
        if metadata.pvp is None:
            raise ValueError(
                "CPHDMetadata must have populated PVP arrays"
            )
        self._metadata = metadata
        self._interp = interpolator or _scipy_interp1d
        self._range_weight_func = self._resolve_weighting(range_weighting)
        self._az_weight_func = self._resolve_weighting(azimuth_weighting)
        self._overlap = overlap
        self._apply_amp_sf = apply_amp_sf
        self._trim_invalid = trim_invalid
        self._antenna_compensation = antenna_compensation
        self._verbose = verbose

        self._validate_metadata()
        self._extract_params()

        # Resolve block_size after _extract_params (needs wavelength, etc.)
        if block_size == 'auto':
            self._block_size = self._auto_block_size()
        else:
            self._block_size = block_size

        # Recompute azimuth resolution for actual processing block size
        if self._block_size is not None:
            n_eff = min(self._block_size, self._npulses)
            self._azimuth_resolution = (
                0.886 * self._wavelength * self._r0_center
                / (2.0 * self._v_eff * self._pri * n_eff)
            )

    # ------------------------------------------------------------------
    # Weighting resolution (same pattern as PFA)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_weighting(
        weighting: Union[str, Callable, None],
    ) -> Optional[Callable]:
        """Resolve weighting parameter to a window function.

        Parameters
        ----------
        weighting : str, callable, or None
            Window specification.

        Returns
        -------
        callable or None
            Window function ``f(n) -> ndarray``, or None for uniform.
        """
        if weighting is None or weighting == 'uniform':
            return None
        if callable(weighting):
            return weighting
        if isinstance(weighting, str):
            key = weighting.lower()
            if key not in _WINDOW_FUNCTIONS:
                raise ValueError(
                    f"Unknown weighting '{weighting}'. "
                    f"Options: {list(_WINDOW_FUNCTIONS.keys())}"
                )
            return _WINDOW_FUNCTIONS[key]
        raise TypeError(
            f"weighting must be str, callable, or None, "
            f"got {type(weighting)}"
        )

    # ------------------------------------------------------------------
    # Metadata validation
    # ------------------------------------------------------------------

    def _validate_metadata(self) -> None:
        """Validate CPHD metadata for RDA processing.

        Checks collect type, domain type, and bandwidth consistency
        with transmit waveform parameters.
        """
        ci = self._metadata.collection_info
        gp = self._metadata.global_params
        tx = self._metadata.tx_waveform
        pvp = self._metadata.pvp

        # Validate collect type is monostatic
        if ci is not None and ci.collect_type is not None:
            if ci.collect_type != 'MONOSTATIC':
                raise ValueError(
                    f"RDA requires MONOSTATIC collection, "
                    f"got '{ci.collect_type}'. Bistatic data "
                    f"needs a specialized processor."
                )

        # Validate FX domain
        if gp is not None and gp.domain_type is not None:
            if gp.domain_type != 'FX':
                raise ValueError(
                    f"RDA requires FX domain CPHD data, "
                    f"got '{gp.domain_type}'. Convert TOA data "
                    f"to FX via DFT before processing."
                )

        # Warn about radar mode
        if self._verbose and ci is not None:
            mode = ci.radar_mode
            if mode is not None and 'STRIP' not in mode.upper():
                print(
                    f"  [WARN] RDA is designed for STRIPMAP mode, "
                    f"got '{mode}'. Results may be suboptimal."
                )

        # Validate bandwidth consistency with waveform
        if tx is not None and tx.lfm_rate is not None:
            if tx.pulse_length is not None:
                chirp_bw = abs(tx.lfm_rate * tx.pulse_length)
                pvp_bw = float(abs(pvp.fx2[0] - pvp.fx1[0]))
                if pvp_bw > 0 and abs(chirp_bw - pvp_bw) / pvp_bw > 0.1:
                    if self._verbose:
                        print(
                            f"  [WARN] Bandwidth mismatch: "
                            f"chirp={chirp_bw / 1e6:.1f} MHz, "
                            f"PVP={pvp_bw / 1e6:.1f} MHz "
                            f"(>10% difference)"
                        )

    # ------------------------------------------------------------------
    # Physics-driven subaperture auto-sizing
    # ------------------------------------------------------------------

    def _auto_block_size(self) -> int:
        """Compute subaperture block size from physical constraints.

        Three constraints limit the block size, and the tightest
        one wins:

        1. **Range drift limit** — the slant range to the SRP
           varies along the strip.  The block must be short enough
           that the range variation stays within a practical
           tolerance (``_DRIFT_FACTOR`` range resolution cells):

               N_drift = factor * delta_r / |dR/dpulse|

           This is typically the tightest constraint for strip map
           CPHD and produces blocks of a few thousand pulses.

        2. **Chirp bandwidth limit** — the common-quadratic +
           matched-filter pipeline requires the azimuth chirp
           bandwidth ``|Ka| * N * PRI`` to stay well within the
           PRF:

               N_bw = 0.9 * PRF / (|Ka| * PRI)

        3. **Quartic phase budget** — the parabolic range
           approximation introduces a quartic residual.  For a
           pi/4 budget:

               N_quartic = (8 * lambda * R0^3 / (v^4 * PRI^4))^(1/4)

        Returns
        -------
        int
            Physics-driven block size in pulses (minimum 64).
        """
        _DRIFT_FACTOR = 10  # practical tolerance in range cells

        # 1. Range drift limit from SRP slant-range variation
        dr_per_pulse = float(
            np.mean(np.abs(np.diff(self._r_srp)))
        )
        if dr_per_pulse > 0:
            n_drift = int(
                _DRIFT_FACTOR * self._range_resolution
                / dr_per_pulse
            )
        else:
            n_drift = self._npulses

        # 2. Chirp BW limit: |Ka| * N * PRI < 0.9 * PRF
        ka_abs = abs(self._ka_ref)
        if ka_abs > 0:
            n_bw = int(0.9 * self._prf / (ka_abs * self._pri))
        else:
            n_bw = self._npulses

        # 3. Quartic residual after full dechirp
        numerator = (
            8.0 * self._wavelength * self._r0_center ** 3
        )
        denominator = self._v_eff ** 4 * self._pri ** 4
        n_quartic = int(np.power(numerator / denominator, 0.25))

        n_max = min(n_drift, n_bw, n_quartic)

        # Clamp to reasonable range
        n_max = max(64, min(n_max, self._npulses))

        if self._verbose:
            t_max = n_max * self._pri
            bw_hz = ka_abs * n_max * self._pri
            print(
                f"  Auto block size: {n_max} pulses "
                f"(T_max={t_max:.4f}s)"
            )
            print(
                f"    Drift limit: {n_drift} "
                f"(dR/pulse={dr_per_pulse:.4f} m, "
                f"tol={_DRIFT_FACTOR} cells)"
            )
            print(
                f"    BW limit: {n_bw}, "
                f"Quartic limit: {n_quartic} "
                f"(chirp BW={bw_hz:.0f} Hz, "
                f"PRF={self._prf:.0f} Hz)"
            )

        return n_max

    # ------------------------------------------------------------------
    # Parameter extraction
    # ------------------------------------------------------------------

    def _extract_params(self) -> None:
        """Extract physical parameters from CPHD metadata PVP arrays.

        Populates instance attributes used by all processing stages.
        """
        pvp = self._metadata.pvp
        gp = self._metadata.global_params

        # Phase sign convention from CPHD Global
        self._phase_sgn = gp.phase_sgn if gp is not None else -1

        # Time array
        self._mid_times = 0.5 * (pvp.tx_time + pvp.rcv_time)
        self._npulses = len(self._mid_times)
        self._nsamples = self._metadata.cols

        # Center frequency and wavelength
        if gp is not None and gp.center_frequency is not None:
            self._fc = gp.center_frequency
        else:
            self._fc = 0.5 * float(pvp.fx1[0] + pvp.fx2[0])
        self._wavelength = _C / self._fc
        self._bandwidth = float(abs(pvp.fx2[0] - pvp.fx1[0]))

        # PRF
        dt = np.diff(self._mid_times)
        self._pri = float(np.mean(dt))
        self._prf = 1.0 / self._pri

        # Platform state (monostatic)
        self._arp = 0.5 * (pvp.tx_pos + pvp.rcv_pos)
        self._varp = 0.5 * (pvp.tx_vel + pvp.rcv_vel)
        self._speed = float(np.mean(norm(self._varp, axis=1)))

        # Slant range per pulse
        self._r_srp = norm(self._arp - pvp.srp_pos, axis=1)
        self._r0_center = float(np.mean(self._r_srp))

        # Reference SRP (center of aperture)
        center_idx = self._npulses // 2
        self._ref_srp = pvp.srp_pos[center_idx].copy()

        # Doppler centroid
        if pvp.a_fdop is not None:
            self._f_dc = float(np.mean(pvp.a_fdop))
        else:
            u_range = pvp.srp_pos - self._arp
            u_range = u_range / norm(u_range, axis=1)[:, np.newaxis]
            rdot = np.sum(u_range * self._varp, axis=1)
            self._f_dc = float(-2.0 * np.mean(rdot) / self._wavelength)

        # Effective velocity (from Doppler rate if available)
        # a_frr1 is dimensionless — for CPHD data that has been fully
        # motion-compensated, these residuals are near zero and cannot
        # be used to derive v_eff.  Only trust the derivation when the
        # result is physically reasonable (> 100 m/s, within 50% of
        # platform speed).
        self._v_eff = self._speed
        if pvp.a_frr1 is not None:
            ka_pvp = float(np.mean(pvp.a_frr1))
            if abs(ka_pvp) > 1e-6:
                v_from_ka = float(np.sqrt(
                    abs(ka_pvp) * self._wavelength
                    * self._r0_center / 2.0
                ))
                # Sanity check: within 50% of platform speed
                if (v_from_ka > 100.0
                        and abs(v_from_ka - self._speed) / self._speed
                        < 0.5):
                    self._v_eff = v_from_ka
                elif self._verbose:
                    print(
                        f"  [INFO] a_frr1-derived v_eff "
                        f"({v_from_ka:.1f} m/s) rejected, "
                        f"using platform speed "
                        f"({self._speed:.1f} m/s)"
                    )

        # Range sample spacing after IFFT
        self._delta_r = _C / (2.0 * self._bandwidth)

        # Azimuth FM rate at reference range
        self._ka_ref = (
            -2.0 * self._v_eff ** 2
            / (self._wavelength * self._r0_center)
        )

        # Doppler frequency axis
        self._f_eta = np.fft.fftshift(
            np.fft.fftfreq(self._npulses, d=self._pri)
        )

        # Resolution estimates
        self._range_resolution = 0.886 * _C / (2.0 * self._bandwidth)
        self._azimuth_resolution = 0.886 * self._v_eff / abs(
            self._ka_ref * self._pri * self._npulses
        )

    # ------------------------------------------------------------------
    # Stage 0: Rephase to common reference
    # ------------------------------------------------------------------

    def rephase_to_reference(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """Rephase FX-domain signal from per-pulse SRP to common SRP.

        In CPHD, each pulse is compensated to its own SRP. For
        coherent azimuth processing, all pulses must be referenced
        to a single scene center point.

        The CPHD signal model phase is ``SGN * fx * ΔTOA`` (in cycles).
        The rephase factor to move from ``R_old`` to ``R_new`` is:

            ``exp(+j * SGN * 4π * fx * (R_new - R_old) / c)``

        Parameters
        ----------
        signal : np.ndarray
            Phase history data, shape ``(npulses, nsamples)``.

        Returns
        -------
        np.ndarray
            Rephased signal, same shape and dtype.
        """
        pvp = self._metadata.pvp
        npulses, nsamples = signal.shape

        # Range from ARP to original per-pulse SRP
        r_old = norm(self._arp - pvp.srp_pos, axis=1)

        # Range from ARP to fixed reference SRP
        r_new = norm(
            self._arp - self._ref_srp[np.newaxis, :], axis=1,
        )

        delta_r = r_new - r_old  # (npulses,)

        # Build frequency grid vectorized
        sc0 = pvp.sc0[:, np.newaxis]
        scss = pvp.scss[:, np.newaxis]
        sample_idx = np.arange(nsamples)[np.newaxis, :]
        freq = sample_idx * scss + sc0

        phase = np.exp(
            1j * self._phase_sgn * 4*np.pi * freq
            * delta_r[:, np.newaxis] / _C
        )

        return signal * phase

    # ------------------------------------------------------------------
    # Stage 1: Range compression
    # ------------------------------------------------------------------

    def range_compress(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """Range compression via IFFT along fast-time axis.

        The CPHD FX-domain data is already range-dechirped. Range
        compression transforms from frequency to range-time domain.
        Optional range weighting controls sidelobes.

        Parameters
        ----------
        signal : np.ndarray
            Rephased FX-domain signal, shape ``(npulses, nsamples)``.

        Returns
        -------
        np.ndarray
            Range-compressed signal, shape ``(npulses, nsamples)``.
        """
        data = signal.copy()

        if self._range_weight_func is not None:
            w_rg = self._range_weight_func(
                data.shape[1]
            ).astype(data.real.dtype)
            data *= w_rg[np.newaxis, :]

        rc = np.fft.fftshift(
            np.fft.ifft(
                np.fft.ifftshift(data, axes=1), axis=1,
            ),
            axes=1,
        )

        return rc

    # ------------------------------------------------------------------
    # Stage 2: Azimuth FFT
    # ------------------------------------------------------------------

    def azimuth_fft(
        self,
        range_compressed: np.ndarray,
    ) -> np.ndarray:
        """Transform from slow-time to Doppler frequency domain.

        FFT along the pulse (azimuth) axis to enter the
        range-Doppler domain where RCMC can be applied.

        Parameters
        ----------
        range_compressed : np.ndarray
            Range-compressed signal, shape ``(npulses, nsamples)``.

        Returns
        -------
        np.ndarray
            Range-Doppler domain signal, shape ``(npulses, nsamples)``.
        """
        return np.fft.fftshift(
            np.fft.fft(
                np.fft.ifftshift(range_compressed, axes=0),
                axis=0,
            ),
            axes=0,
        )

    # ------------------------------------------------------------------
    # Stage 3: RCMC
    # ------------------------------------------------------------------

    def rcmc(
        self,
        range_doppler: np.ndarray,
    ) -> np.ndarray:
        """Range Cell Migration Correction via interpolation.

        After rephasing to a common SRP, the bulk range migration
        at R0 is already corrected.  The residual migration is
        range-dependent:

            delta_R(f_eta, r) = range_axis(r) * (1/D - 1)
            D = sqrt(1 - (lambda * f_eta / (2 * v_eff))^2)

        where ``range_axis(r)`` is the range offset from R0 in
        meters.  This differential correction is all that remains
        after the rephase step.

        Parameters
        ----------
        range_doppler : np.ndarray
            Range-Doppler domain signal, shape ``(npulses, nsamples)``.

        Returns
        -------
        np.ndarray
            RCMC-corrected signal, same shape.
        """
        npulses, nsamples = range_doppler.shape
        result = np.zeros_like(range_doppler)

        # Range axis in meters (differential from SRP center)
        range_axis = (
            np.arange(nsamples) - nsamples / 2.0
        ) * self._delta_r

        for i in range(npulses):
            f_eta_i = self._f_eta[i]
            arg = (
                self._wavelength * f_eta_i / (2.0 * self._v_eff)
            ) ** 2

            if arg >= 1.0:
                continue

            D = np.sqrt(1.0 - arg)

            # Differential RCMC: range_axis * (1/D - 1)
            # Rephasing already corrects bulk migration at R0.
            delta_r_shift = range_axis * (1.0 / D - 1.0)

            x_old = range_axis
            y_old = range_doppler[i, :]
            x_new = range_axis + delta_r_shift

            result[i, :] = self._interp(x_old, y_old, x_new)

        return result

    # ------------------------------------------------------------------
    # Stage 4: Azimuth compression
    # ------------------------------------------------------------------

    def azimuth_compress(
        self,
        rcmc_data: np.ndarray,
    ) -> np.ndarray:
        """Azimuth windowing in range-Doppler domain.

        For CPHD data the azimuth FFT (stage 2) **is** the azimuth
        compression.  After rephasing to a common SRP, each target's
        slow-time phase is a linear Doppler tone:

            phi(t) = 4*pi*v*y / (lambda*R0) * t + const

        The FFT resolves each tone to its Doppler bin, producing a
        focused image in the range-Doppler domain.  No matched filter
        or additional IFFT is needed — an IFFT here would undo the
        focusing performed by the azimuth FFT.

        This method applies optional azimuth windowing only.

        Parameters
        ----------
        rcmc_data : np.ndarray
            RCMC-corrected range-Doppler domain signal,
            shape ``(npulses, nsamples)``.

        Returns
        -------
        np.ndarray
            Complex SAR image in range-Doppler domain,
            shape ``(npulses, nsamples)``.
        """
        if self._az_weight_func is None:
            return rcmc_data

        npulses = rcmc_data.shape[0]
        data = rcmc_data.copy()
        w_az = self._az_weight_func(npulses).astype(
            data.real.dtype
        )
        data *= w_az[:, np.newaxis]
        return data

    # ------------------------------------------------------------------
    # Stage 4b: Azimuth matched filter + IFFT
    # ------------------------------------------------------------------

    def _azimuth_matched_filter(
        self,
        rd_data: np.ndarray,
        f_eta: np.ndarray,
    ) -> np.ndarray:
        """Apply range-dependent azimuth matched filter in range-Doppler.

        After RCMC, each range bin contains a chirped Doppler spectrum.
        The matched filter compresses it:

            H(f_eta, R) = exp(+j * pi * f_eta^2 / K_a(R))
            K_a(R) = SGN * 2 * v^2 / (lambda * R)

        Parameters
        ----------
        rd_data : np.ndarray
            RCMC-corrected range-Doppler data, shape ``(npulses, nsamples)``.
        f_eta : np.ndarray
            Doppler frequency axis in Hz, shape ``(npulses,)``.

        Returns
        -------
        np.ndarray
            Matched-filtered data, same shape.
        """
        npulses, nsamples = rd_data.shape
        range_axis = (
            np.arange(nsamples) - nsamples / 2.0
        ) * self._delta_r

        # Physical range per bin: R = R0 + SGN * range_axis
        # (range_axis = SGN*(R-R0), so R = R0 + SGN*range_axis)
        r_abs = self._r0_center + self._phase_sgn * range_axis
        r_abs = np.maximum(r_abs, self._r0_center * 0.5)

        # K_a(R) = SGN * 2*v^2 / (lambda*R)  [Hz/s]
        ka_r = (
            self._phase_sgn * 2.0 * self._v_eff ** 2
            / (self._wavelength * r_abs)
        )

        # Matched filter: H = exp(+j * pi * f_eta^2 / K_a)
        # Shape: (npulses, nsamples) via broadcasting
        f_eta_sq = f_eta[:, np.newaxis] ** 2  # (npulses, 1)
        inv_ka = 1.0 / ka_r[np.newaxis, :]   # (1, nsamples)

        result = rd_data.copy()
        _CHUNK = 256
        for cs in range(0, nsamples, _CHUNK):
            ce = min(cs + _CHUNK, nsamples)
            phase = np.pi * f_eta_sq * inv_ka[:, cs:ce]
            result[:, cs:ce] *= np.exp(1j * phase)

        if self._verbose:
            max_phase = float(
                np.pi * np.max(f_eta ** 2)
                * np.max(np.abs(1.0 / ka_r))
            )
            print(f"  Matched filter max phase: {max_phase:.1f} rad "
                  f"({max_phase / np.pi:.1f}\u03c0)")

        return result

    def azimuth_ifft(
        self,
        rd_data: np.ndarray,
    ) -> np.ndarray:
        """Inverse FFT along azimuth to produce focused image.

        Parameters
        ----------
        rd_data : np.ndarray
            Range-Doppler domain data after matched filtering,
            shape ``(npulses, nsamples)``.

        Returns
        -------
        np.ndarray
            Focused complex SAR image, shape ``(npulses, nsamples)``.
        """
        return np.fft.fftshift(
            np.fft.ifft(
                np.fft.ifftshift(rd_data, axes=0), axis=0,
            ),
            axes=0,
        )

    # ------------------------------------------------------------------
    # Azimuth deramp
    # ------------------------------------------------------------------

    def _apply_dechirp(
        self,
        rc: np.ndarray,
    ) -> np.ndarray:
        """Remove range-dependent quadratic azimuth phase.

        After rephasing to a common SRP, the slow-time quadratic
        phase at range ``R = R0 + SGN * range_axis`` is:

            phi(n, R) = SGN * 2*pi*v^2*PRI^2/lambda
                        * (1/R0 - 1/R) * n^2

        This is zero at R0 (the rephase perfectly compensates the
        geometric quadratic there) and grows with range offset.

        The deramp multiplies by ``exp(1j * deramp_per_bin * n^2)``
        where:

            deramp_per_bin = -SGN * 2*pi*v^2*PRI^2/lambda
                             * (1/R0 - 1/R)

        which simplifies to:

            deramp_per_bin = -(2*pi*v^2*PRI^2 / (lambda*R0))
                             * range_axis / R

        After this correction, **all** targets have linear-only
        slow-time phase regardless of range.

        Parameters
        ----------
        rc : np.ndarray
            Range-compressed signal, shape ``(npulses, nsamples)``.

        Returns
        -------
        np.ndarray
            Deramped signal, same shape.
        """
        npulses, nsamples = rc.shape
        range_axis = (
            np.arange(nsamples) - nsamples / 2.0
        ) * self._delta_r

        # Physical range: R = R0 + SGN * range_axis
        r_actual = self._r0_center + self._phase_sgn * range_axis
        r_actual = np.maximum(r_actual, self._r0_center * 0.5)

        # Deramp: negate the signal's residual quadratic.
        # Signal has SGN * 2πv²PRI²/λ * (1/R0 - 1/R) * n².
        # Deramp applies -(that) = -deramp_base * range_axis / R.
        deramp_base = (
            2.0 * np.pi * self._v_eff ** 2 * self._pri ** 2
            / (self._wavelength * self._r0_center)
        )
        deramp_per_bin = -deramp_base * range_axis / r_actual

        n_offsets = np.arange(npulses) - npulses / 2.0
        n_sq = n_offsets ** 2

        result = rc.copy()
        _CHUNK = 512
        for ns in range(0, npulses, _CHUNK):
            ne = min(ns + _CHUNK, npulses)
            phase = (
                n_sq[ns:ne, np.newaxis]
                * deramp_per_bin[np.newaxis, :]
            )
            result[ns:ne, :] *= np.exp(1j * phase)

        if self._verbose:
            max_phase = float(
                n_sq.max() * np.max(np.abs(deramp_per_bin))
            )
            print(f"  Deramp max phase: {max_phase:.2f} rad "
                  f"({max_phase / np.pi:.1f}\u03c0)")

        return result

    # ------------------------------------------------------------------
    # Doppler centroid estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_doppler_centroid(
        range_compressed: np.ndarray,
        pri: float,
    ) -> float:
        """Estimate Doppler centroid frequency via autocorrelation.

        Uses lag-1 autocorrelation of the range-compressed signal
        to estimate the mean Doppler frequency.

        Parameters
        ----------
        range_compressed : np.ndarray
            Range-compressed signal, shape ``(npulses, nsamples)``.
        pri : float
            Pulse repetition interval in seconds.

        Returns
        -------
        float
            Estimated Doppler centroid frequency in Hz.
        """
        r1 = np.sum(
            range_compressed[1:, :]
            * np.conj(range_compressed[:-1, :])
        )
        return float(np.angle(r1) / (2.0 * np.pi * pri))

    # ------------------------------------------------------------------
    # Signal preprocessing
    # ------------------------------------------------------------------

    def _preprocess_signal(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """Apply AmpSF normalization and trim invalid pulses.

        Called at the start of ``form_image`` when ``apply_amp_sf``
        or ``trim_invalid`` is enabled.

        Parameters
        ----------
        signal : np.ndarray
            Raw phase history signal, shape ``(npulses, nsamples)``.

        Returns
        -------
        np.ndarray
            Preprocessed signal (possibly fewer rows if trimmed).
        """
        pvp = self._metadata.pvp
        npulses = signal.shape[0]

        # Identify valid pulses
        valid_mask = np.ones(npulses, dtype=bool)

        if self._trim_invalid:
            # Signal indicator
            if pvp.signal is not None:
                valid_mask &= pvp.signal > 0

            # AmpSF anomaly detection (>3 sigma from median)
            if pvp.amp_sf is not None:
                median_sf = float(np.median(pvp.amp_sf))
                if median_sf > 0:
                    deviation = np.abs(
                        pvp.amp_sf - median_sf
                    ) / median_sf
                    valid_mask &= deviation < 0.5

        n_invalid = int(np.sum(~valid_mask))
        if n_invalid > 0 and self._verbose:
            print(
                f"  Trimming {n_invalid} invalid pulses "
                f"({npulses} -> {npulses - n_invalid})"
            )

        # Apply AmpSF
        if self._apply_amp_sf and pvp.amp_sf is not None:
            amp_sf = pvp.amp_sf
            if self._trim_invalid:
                amp_sf = amp_sf[valid_mask]
            signal_out = signal[valid_mask] if n_invalid > 0 else signal.copy()
            signal_out = signal_out * amp_sf[:, np.newaxis]
        elif n_invalid > 0:
            signal_out = signal[valid_mask]
        else:
            signal_out = signal

        # Update internal per-pulse arrays to match trimmed signal
        if n_invalid > 0:
            self._valid_mask = valid_mask
            self._npulses_orig = npulses

            # Trim internal per-pulse arrays
            self._arp = self._arp[valid_mask]
            self._varp = self._varp[valid_mask]
            self._mid_times = self._mid_times[valid_mask]
            self._r_srp = self._r_srp[valid_mask]
            self._npulses = int(np.sum(valid_mask))

            # Recompute Doppler frequency axis for new pulse count
            self._f_eta = np.fft.fftshift(
                np.fft.fftfreq(self._npulses, d=self._pri)
            )

            # Trim the metadata PVP so downstream stages
            # (rephase, subaperture) see consistent arrays
            pvp_old = self._metadata.pvp
            fields_1d = [
                'tx_time', 'rcv_time', 'fx1', 'fx2', 'sc0',
                'scss', 'signal', 'a_fdop', 'a_frr1', 'a_frr2',
                'amp_sf', 'toa1', 'toa2',
            ]
            fields_2d = [
                'tx_pos', 'tx_vel', 'rcv_pos', 'rcv_vel',
                'srp_pos',
            ]
            pvp_kwargs = {}
            for name in fields_1d + fields_2d:
                arr = getattr(pvp_old, name)
                pvp_kwargs[name] = (
                    arr[valid_mask] if arr is not None else None
                )
            trimmed_pvp = CPHDPVP(**pvp_kwargs)

            self._metadata = CPHDMetadata(
                format=self._metadata.format,
                rows=self._npulses,
                cols=self._metadata.cols,
                dtype=self._metadata.dtype,
                channels=self._metadata.channels,
                pvp=trimmed_pvp,
                global_params=self._metadata.global_params,
                collection_info=self._metadata.collection_info,
                tx_waveform=self._metadata.tx_waveform,
                rcv_parameters=self._metadata.rcv_parameters,
                antenna_pattern=self._metadata.antenna_pattern,
                scene_coordinates=self._metadata.scene_coordinates,
                reference_geometry=self._metadata.reference_geometry,
                dwell=self._metadata.dwell,
                num_channels=self._metadata.num_channels,
                extras=self._metadata.extras,
            )
        else:
            self._valid_mask = None

        return signal_out

    # ------------------------------------------------------------------
    # Antenna pattern compensation
    # ------------------------------------------------------------------

    def _compute_antenna_gain(
        self,
        pulse_indices: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Compute per-pulse antenna gain from CPHD antenna pattern.

        Uses the antenna gain polynomial and ACF orientation to
        compute the directional cosine of each pulse's SRP relative
        to the antenna boresight, then evaluates the gain polynomial.

        Parameters
        ----------
        pulse_indices : np.ndarray
            Indices of pulses to compute gain for.

        Returns
        -------
        np.ndarray or None
            Per-pulse gain in linear scale, or None if antenna
            data is unavailable.
        """
        ant = self._metadata.antenna_pattern
        if ant is None or ant.gain_poly is None:
            return None
        if ant.acf_x_poly is None or ant.acf_y_poly is None:
            return None

        pvp = self._metadata.pvp
        mid_times = self._mid_times[pulse_indices]
        arp = self._arp[pulse_indices]

        # Reference time for ACF polynomials
        ref_time = float(np.mean(self._mid_times))

        # Evaluate ACF orientation at each pulse time
        dt = mid_times - ref_time
        acf_x = ant.acf_x_poly
        acf_y = ant.acf_y_poly
        n_order = acf_x.shape[0]

        # Polynomial evaluation: sum(coef[k] * dt^k)
        x_axis = np.zeros((len(dt), 3))
        y_axis = np.zeros((len(dt), 3))
        for k in range(n_order):
            x_axis += acf_x[k][np.newaxis, :] * (dt ** k)[:, np.newaxis]
            if k < acf_y.shape[0]:
                y_axis += (
                    acf_y[k][np.newaxis, :] * (dt ** k)[:, np.newaxis]
                )

        # Normalize ACF axes
        x_norm = norm(x_axis, axis=1)[:, np.newaxis]
        y_norm = norm(y_axis, axis=1)[:, np.newaxis]
        x_norm[x_norm == 0] = 1.0
        y_norm[y_norm == 0] = 1.0
        x_axis /= x_norm
        y_axis /= y_norm

        # Vector from ARP to SRP (line of sight)
        srp = pvp.srp_pos[pulse_indices]
        los = srp - arp
        los_norm = norm(los, axis=1)[:, np.newaxis]
        los_norm[los_norm == 0] = 1.0
        los /= los_norm

        # Directional cosines in ACF
        dcx = np.sum(los * x_axis, axis=1)
        dcy = np.sum(los * y_axis, axis=1)

        # Evaluate 2D gain polynomial (dB)
        gain_poly = ant.gain_poly
        gain_db = np.zeros(len(dt))
        for i in range(gain_poly.shape[0]):
            for j in range(gain_poly.shape[1]):
                if gain_poly[i, j] != 0:
                    gain_db += (
                        gain_poly[i, j] * dcx**i * dcy**j
                    )

        # Add boresight gain
        if ant.gain_zero is not None:
            gain_db += ant.gain_zero

        # Convert to linear and normalize to unit mean
        gain_linear = 10.0 ** (gain_db / 20.0)
        mean_gain = float(np.mean(gain_linear))
        if mean_gain > 0:
            gain_linear /= mean_gain

        return gain_linear

    # ------------------------------------------------------------------
    # Subaperture pipeline
    # ------------------------------------------------------------------

    def _form_image_subaperture(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """Form image using overlapping subaperture blocks.

        Each block is rephased to its local center SRP, range
        compressed, windowed, and azimuth-FFT'd.  The common
        quadratic restores Ka(R0) so the matched filter can
        compress the full chirp.  Block size is limited so the
        chirp BW stays within the PRF.

        Blocks are mosaicked via **hard crop and stitch**: only
        the centre ``stride`` rows of each focused block are
        kept (where the synthetic aperture is complete), then
        concatenated sequentially.  This avoids destructive
        interference from phase-incoherent overlap blending,
        since each block is rephased to a different SRP.

        Parameters
        ----------
        signal : np.ndarray
            CPHD FX-domain phase history,
            shape ``(npulses, nsamples)``.

        Returns
        -------
        np.ndarray
            Complex SAR image, shape ``(n_az, nsamples)`` where
            ``n_az = n_blocks * stride``.
        """
        pvp = self._metadata.pvp
        npulses, nsamples = signal.shape
        block_size = min(self._block_size, npulses)
        stride = max(1, int(block_size * (1.0 - self._overlap)))

        # Compute block start indices
        starts = list(range(0, npulses - block_size + 1, stride))
        if not starts or starts[-1] + block_size < npulses:
            starts.append(max(0, npulses - block_size))
        starts = sorted(set(starts))

        # Crop indices: keep the center `stride` rows from each
        # focused block.  Edge pulses have incomplete synthetic
        # apertures; the center is best focused.  Each output
        # pixel comes from exactly one block, avoiding
        # phase-incoherent blending between blocks rephased to
        # different SRPs.
        crop_start = (block_size - stride) // 2
        crop_end = crop_start + stride
        n_out_rows = len(starts) * stride

        if self._verbose:
            print(f"\nSubaperture RDA: {len(starts)} blocks of "
                  f"{block_size} pulses, stride {stride}, "
                  f"overlap {self._overlap:.0%}")
            print(f"  Crop: rows [{crop_start}:{crop_end}] "
                  f"per block ({stride} rows)")
            print(f"  Output grid: {n_out_rows} x {nsamples}")

        # Precompute RCMC shift ratios for block-sized Doppler axis
        f_eta_block = np.fft.fftshift(
            np.fft.fftfreq(block_size, d=self._pri)
        )
        range_axis = (
            np.arange(nsamples) - nsamples / 2.0
        ) * self._delta_r

        arg = (
            self._wavelength * f_eta_block / (2.0 * self._v_eff)
        ) ** 2
        valid_doppler = arg < 1.0
        D_inv_minus_1 = np.zeros(block_size)
        D_inv_minus_1[valid_doppler] = (
            1.0 / np.sqrt(1.0 - arg[valid_doppler]) - 1.0
        )

        if self._verbose:
            max_shift_m = float(
                np.max(np.abs(range_axis)) * np.max(D_inv_minus_1)
            )
            max_shift_bins = max_shift_m / self._delta_r
            print(f"  RCMC max shift: {max_shift_m:.2f} m "
                  f"({max_shift_bins:.1f} bins)")

        # Collect cropped block centres for concatenation
        blocks = []

        for b_idx, start in enumerate(starts):
            end = start + block_size
            block_sig = signal[start:end, :]

            # Antenna pattern compensation (before rephasing)
            if self._antenna_compensation:
                pulse_idx = np.arange(start, end)
                gain = self._compute_antenna_gain(pulse_idx)
                if gain is not None:
                    # Inverse gain: divide by antenna pattern
                    inv_gain = np.where(
                        gain > 0.01, 1.0 / gain, 0.0,
                    ).astype(np.float32)
                    block_sig = block_sig * inv_gain[:, np.newaxis]

            # Local center SRP from block's center pulse
            center_pulse = start + block_size // 2
            local_srp = pvp.srp_pos[center_pulse].copy()

            # Local reference range (ARP to local SRP)
            r0_local = float(norm(
                self._arp[center_pulse] - local_srp
            ))

            # Rephase to local SRP (vectorized)
            arp_block = self._arp[start:end]
            r_old = norm(
                arp_block - pvp.srp_pos[start:end], axis=1,
            )
            r_new = norm(
                arp_block - local_srp[np.newaxis, :], axis=1,
            )
            delta_r = r_new - r_old

            sc0 = pvp.sc0[start:end, np.newaxis]
            scss = pvp.scss[start:end, np.newaxis]
            sample_idx = np.arange(nsamples)[np.newaxis, :]
            freq = sample_idx * scss + sc0

            rephase = np.exp(
                1j * self._phase_sgn * 4.0 * np.pi * freq
                * delta_r[:, np.newaxis] / _C
            )
            rephased = block_sig * rephase

            # Range compress
            rc = rephased.copy()
            if self._range_weight_func is not None:
                w_rg = self._range_weight_func(nsamples).astype(
                    rc.real.dtype,
                )
                rc *= w_rg[np.newaxis, :]
            rc = np.fft.fftshift(
                np.fft.ifft(
                    np.fft.ifftshift(rc, axes=1), axis=1,
                ),
                axes=1,
            )

            # Common quadratic at local R0 (range-independent).
            # Rephasing removed the quadratic at this block's SRP;
            # restore it so the matched filter uses the full K_a.
            common_base_local = (
                self._phase_sgn * 2.0 * np.pi
                * self._v_eff ** 2 * self._pri ** 2
                / (self._wavelength * r0_local)
            )
            n_blk_off = (
                np.arange(block_size) - block_size / 2.0
            )
            rc *= np.exp(
                1j * common_base_local
                * n_blk_off[:, np.newaxis] ** 2
            )

            # Per-block Doppler centroid demodulation
            if pvp.a_fdop is not None:
                f_dc_block = float(
                    np.mean(pvp.a_fdop[start:end])
                )
            else:
                f_dc_block = self.estimate_doppler_centroid(
                    rc, self._pri,
                )
            if abs(f_dc_block) > 1.0:
                t_block = (
                    np.arange(block_size) * self._pri
                )[:, np.newaxis]
                demod = np.exp(
                    -1j * 2.0 * np.pi * f_dc_block * t_block
                ).astype(np.complex64)
                rc *= demod

            # Azimuth FFT → range-Doppler domain
            rd_block = np.fft.fftshift(
                np.fft.fft(
                    np.fft.ifftshift(rc, axes=0), axis=0,
                ),
                axes=0,
            )

            # Differential RCMC: range_axis * (1/D - 1)
            # Rephasing already corrects bulk migration at the SRP;
            # only the range-dependent residual remains.
            rcmc_block = np.zeros_like(rd_block)
            for i in range(block_size):
                if not valid_doppler[i]:
                    continue
                shift = range_axis * D_inv_minus_1[i]
                x_new = range_axis + shift
                rcmc_block[i, :] = self._interp(
                    range_axis, rd_block[i, :], x_new,
                )

            # Azimuth matched filter (range-dependent)
            mf_block = self._azimuth_matched_filter(
                rcmc_block, f_eta_block,
            )

            # Azimuth windowing (in Doppler domain)
            if self._az_weight_func is not None:
                w_az = self._az_weight_func(block_size).astype(
                    mf_block.real.dtype,
                )
                mf_block *= w_az[:, np.newaxis]

            # Azimuth IFFT → focused image block (full Doppler BW)
            image_block = self.azimuth_ifft(mf_block)

            if self._verbose:
                mag = np.abs(image_block)
                pm = (
                    mag.max() / mag.mean()
                    if mag.mean() > 0 else 0
                )
                print(
                    f"  Block {b_idx}: pulses [{start}:{end}], "
                    f"fdc={f_dc_block:.1f}Hz, peak/mean={pm:.1f}"
                )

            # Crop the well-focused centre of the block
            blocks.append(
                image_block[crop_start:crop_end, :].copy()
            )

        return np.concatenate(blocks, axis=0)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def form_image(
        self,
        signal: np.ndarray,
        geometry: Any,
    ) -> np.ndarray:
        """Run the full Range-Doppler Algorithm pipeline.

        Parameters
        ----------
        signal : np.ndarray
            CPHD FX-domain phase history,
            shape ``(npulses, nsamples)``.
        geometry : Any
            Ignored (geometry is computed internally from metadata).

        Returns
        -------
        np.ndarray
            Complex SAR image.  In single-reference mode the shape
            is ``(npulses, nsamples)``.  In subaperture mode the
            azimuth dimension depends on the strip length and
            Doppler bin spacing.
        """
        # Ensure complex type (sarkit may return structured int16)
        if signal.dtype.names:
            signal = (
                signal['real'].astype(np.float32)
                + 1j * signal['imag'].astype(np.float32)
            )

        if self._verbose:
            print(f"RDA: {self._npulses} pulses x "
                  f"{self._nsamples} samples")
            print(f"  Phase SGN: {self._phase_sgn:+d}")
            print(f"  Wavelength: {self._wavelength * 100:.2f} cm")
            print(f"  Bandwidth: {self._bandwidth / 1e6:.1f} MHz")
            print(f"  PRF: {self._prf:.1f} Hz")
            print(f"  Platform speed: {self._speed:.1f} m/s")
            print(f"  Effective velocity: {self._v_eff:.1f} m/s")
            print(f"  Reference range: "
                  f"{self._r0_center / 1e3:.1f} km")
            print(f"  Doppler centroid: {self._f_dc:.1f} Hz")
            print(f"  Az FM rate (ref): {self._ka_ref:.1f} Hz/s")
            print(f"  Range resolution: "
                  f"{self._range_resolution:.3f} m")
            print(f"  Azimuth resolution: "
                  f"{self._azimuth_resolution:.3f} m")
            if self._block_size is not None:
                print(f"  Mode: subaperture ({self._block_size} "
                      f"pulses, {self._overlap:.0%} overlap)")
            else:
                print("  Mode: single-reference (full aperture)")

            # Report metadata utilization
            pvp = self._metadata.pvp
            if pvp.amp_sf is not None:
                print(f"  AmpSF: available "
                      f"(apply={self._apply_amp_sf})")
            if self._metadata.antenna_pattern is not None:
                print(f"  Antenna pattern: available "
                      f"(compensate={self._antenna_compensation})")
            rg = self._metadata.reference_geometry
            if rg is not None and rg.graze_angle_deg is not None:
                print(f"  Graze angle: "
                      f"{rg.graze_angle_deg:.2f} deg")
            if rg is not None and rg.side_of_track is not None:
                print(f"  Side of track: {rg.side_of_track}")

        # Preprocess: apply AmpSF, trim invalid pulses
        signal = self._preprocess_signal(signal)

        # Dispatch to subaperture or single-reference pipeline
        if self._block_size is not None:
            image = self._form_image_subaperture(signal)
        else:
            if self._verbose:
                print("\nStage 0: Rephasing to common SRP...")
            rephased = self.rephase_to_reference(signal)

            if self._verbose:
                print("Stage 1: Range compression (IFFT)...")
            rc = self.range_compress(rephased)

            # Stage 1.5: Deramp — remove range-dependent residual
            # quadratic.  Rephasing removed the quadratic at R0;
            # targets at R ≠ R0 have a residual (1/R0 - 1/R) that
            # the deramp corrects.  After deramp, ALL targets have
            # linear-only slow-time phase and the azimuth FFT
            # focuses them directly — no matched filter or IFFT.
            if self._verbose:
                print("Stage 1.5: Azimuth deramp...")
            rc = self._apply_dechirp(rc)

            if self._verbose:
                print("Stage 2: Azimuth FFT...")
            rd = self.azimuth_fft(rc)

            if self._verbose:
                print("Stage 3: Range Cell Migration Correction...")
            rcmc_out = self.rcmc(rd)

            if self._verbose:
                print("Stage 4: Azimuth windowing...")
            image = self.azimuth_compress(rcmc_out)

        if self._verbose:
            print(f"\nImage formed: {image.shape}, "
                  f"dtype: {image.dtype}")

        return image

    # ------------------------------------------------------------------
    # Output grid
    # ------------------------------------------------------------------

    def get_output_grid(self) -> Dict[str, Any]:
        """Return output grid parameters for SICD metadata.

        The RDA output grid is RGZERO type (range-zero-Doppler).

        Returns
        -------
        Dict[str, Any]
            Grid parameters matching SICD Grid section fields.
        """
        grid = {
            'image_plane': 'SLANT',
            'type': 'RGZERO',
            'rg_ss': self._delta_r,
            'rg_imp_resp_bw': 2.0 * self._bandwidth / _C,
            'rg_imp_resp_wid': self._range_resolution,
            'rg_kctr': 2.0 * self._fc / _C,
            'az_ss': self._speed / self._prf,
            'az_imp_resp_wid': self._azimuth_resolution,
            'az_kctr': self._f_dc / self._speed,
            'range_resolution': self._range_resolution,
            'azimuth_resolution': self._azimuth_resolution,
            'rec_n_pulses': self._npulses,
            'rec_n_samples': self._nsamples,
            'image_form_algo': 'RDA',
            'center_frequency': self._fc,
            'wavelength': self._wavelength,
            'bandwidth': self._bandwidth,
            'prf': self._prf,
            'v_eff': self._v_eff,
            'r0_center': self._r0_center,
            'f_dc': self._f_dc,
            'ka_ref': self._ka_ref,
        }

        # Reference geometry
        rg = self._metadata.reference_geometry
        if rg is not None:
            if rg.graze_angle_deg is not None:
                grid['graze_angle_deg'] = rg.graze_angle_deg
            if rg.azimuth_angle_deg is not None:
                grid['azimuth_angle_deg'] = rg.azimuth_angle_deg
            if rg.side_of_track is not None:
                grid['side_of_track'] = rg.side_of_track
            if rg.twist_angle_deg is not None:
                grid['twist_angle_deg'] = rg.twist_angle_deg
            if rg.slope_angle_deg is not None:
                grid['slope_angle_deg'] = rg.slope_angle_deg

        # Scene coordinates
        sc = self._metadata.scene_coordinates
        if sc is not None:
            if sc.iarp_ecf is not None:
                grid['iarp_ecf'] = sc.iarp_ecf
            if sc.iarp_llh is not None:
                grid['iarp_llh'] = sc.iarp_llh
            if sc.corner_points is not None:
                grid['corner_points'] = sc.corner_points

        # Collection info
        ci = self._metadata.collection_info
        if ci is not None:
            if ci.collector_name is not None:
                grid['collector_name'] = ci.collector_name
            if ci.radar_mode is not None:
                grid['radar_mode'] = ci.radar_mode

        return grid

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def wavelength(self) -> float:
        """Wavelength in meters."""
        return self._wavelength

    @property
    def bandwidth(self) -> float:
        """Signal bandwidth in Hz."""
        return self._bandwidth

    @property
    def prf(self) -> float:
        """Pulse Repetition Frequency in Hz."""
        return self._prf

    @property
    def reference_range(self) -> float:
        """Reference slant range in meters."""
        return self._r0_center

    @property
    def doppler_centroid(self) -> float:
        """Doppler centroid frequency in Hz."""
        return self._f_dc

    @property
    def effective_velocity(self) -> float:
        """Effective platform velocity in m/s."""
        return self._v_eff

    @property
    def azimuth_fm_rate(self) -> float:
        """Azimuth FM rate at reference range in Hz/s."""
        return self._ka_ref

    @property
    def block_size(self) -> Optional[int]:
        """Subaperture block size in pulses, or None."""
        return self._block_size

    @property
    def overlap(self) -> float:
        """Subaperture overlap fraction."""
        return self._overlap
