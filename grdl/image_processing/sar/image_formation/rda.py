# -*- coding: utf-8 -*-
"""
Range-Doppler Algorithm - Standard stripmap SAR image formation.

Implements the Range-Doppler Algorithm (RDA) for forming complex SAR
images from CPHD (Compensated Phase History Data) in FX domain.
The five-stage pipeline processes the full aperture coherently:

1. **Rephase** — compensate per-pulse SRP drift to a common reference.
2. **Range compress** — IFFT along fast-time to range-time domain.
3. **Azimuth FFT** — transform to range-Doppler domain.
4. **RCMC** — residual range cell migration correction (differential
   range only; SRP migration already removed by CPHD compensation).
5. **Azimuth windowing** — optional sidelobe control (no IFFT; the
   azimuth FFT already focuses CPHD data to the image domain).

Each stage is exposed as a public method for intermediate inspection,
following the same tapout pattern as ``PolarFormatAlgorithm``.

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
2026-02-13

Modified
--------
2026-02-14
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
from grdl.IO.models.cphd import CPHDMetadata


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

    Five-stage pipeline operating on CPHD FX-domain data:

    0. Rephase to common reference SRP
    1. Range compression (IFFT along fast-time)
    2. Azimuth FFT (slow-time to Doppler domain)
    3. Residual RCMC (differential range migration correction)
    4. Azimuth windowing (no IFFT — azimuth FFT is the compression)

    Each stage is a public method for intermediate inspection.

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
    >>> rda = RangeDopplerAlgorithm(meta, range_weighting='taylor')
    >>> image = rda.form_image(signal, geometry=None)
    """

    def __init__(
        self,
        metadata: CPHDMetadata,
        interpolator: Optional[Callable] = None,
        range_weighting: Union[str, Callable, None] = None,
        azimuth_weighting: Union[str, Callable, None] = None,
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
        self._verbose = verbose

        self._extract_params()

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
        if pvp.a_frr1 is not None:
            ka_pvp = float(np.mean(pvp.a_frr1)) * self._wavelength / 2.0
            self._v_eff = float(np.sqrt(
                abs(ka_pvp) * self._wavelength * self._r0_center / 2.0
            ))
            if self._v_eff < 100.0:
                self._v_eff = self._speed
        else:
            self._v_eff = self._speed

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

            ``exp(-j * SGN * 4π * fx * (R_new - R_old) / c)``

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
            -1j * self._phase_sgn * 4.0 * np.pi * freq
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

        CPHD compensation removes the SRP's range migration entirely.
        The residual migration for other targets depends only on their
        **differential** range from the SRP (not the absolute range):

            delta_R(f_eta) = delta_r_bin * (1/D - 1)
            D = sqrt(1 - (lambda * f_eta / (2 * v_eff))^2)

        where ``delta_r_bin`` is the range offset from the SRP.
        This is typically much smaller than the uncompensated
        migration ``R0 * (1/D - 1)`` by a factor of ``delta_r / R0``.

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

            # Differential RCMC: migration proportional to range
            # offset from SRP, not absolute range
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
            Complex SAR image, shape ``(npulses, nsamples)``.
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

        if self._verbose:
            print("\nStage 0: Rephasing to common SRP...")
        rephased = self.rephase_to_reference(signal)

        if self._verbose:
            print("Stage 1: Range compression (IFFT)...")
        rc = self.range_compress(rephased)

        if self._verbose:
            print("Stage 2: Azimuth FFT...")
        rd = self.azimuth_fft(rc)

        if self._verbose:
            print("Stage 3: Range Cell Migration Correction...")
        rcmc_out = self.rcmc(rd)

        if self._verbose:
            print("Stage 4: Azimuth compression...")
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
        return {
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
