# -*- coding: utf-8 -*-
"""
Fast Factorized Back-Projection - Strip map SAR image formation.

Implements the Fast Factorized Back-Projection (FFBP) algorithm for
forming complex SAR images from CPHD (Compensated Phase History Data)
in FX domain.  FFBP is an exact time-domain algorithm with no
stationary-phase or RCMC approximations.

The pipeline:

1. **Range compress** — IFFT along fast-time to range-time domain.
2. **Leaf formation** — Direct back-projection of small subapertures
   (``leaf_size`` pulses each) onto local polar grids.
3. **Binary-tree merge** — Recursively merge adjacent subaperture
   images via 1-D angular resampling per range bin.
4. **Polar-to-rectangular** — Convert the root polar image to a
   slant-plane rectangular grid (range × cross-range).

Each stage is exposed as a public method for intermediate inspection,
following the same tapout pattern as ``RangeDopplerAlgorithm``.

Dependencies
------------
scipy
numba (optional, for parallel acceleration)

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
2026-02-16

Modified
--------
2026-02-16
"""

# Standard library
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

# Third-party
import numpy as np
from numpy.linalg import norm
from scipy.interpolate import interp1d
from scipy.signal.windows import taylor as _taylor_window

# Optional numba acceleration
try:
    import numba as nb
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

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
    """Default interpolation using scipy interp1d (linear)."""
    func = interp1d(
        x_old, y_old,
        kind='linear',
        bounds_error=False,
        fill_value=0.0,
        copy=False,
    )
    return func(x_new)


# ==================================================================
# Numba-accelerated kernels
# ==================================================================

if _HAS_NUMBA:

    @nb.njit(parallel=True, cache=True)
    def _nb_bp_leaf(
        rc, arp, r_srp, srp_center, u_range, u_xr,
        range_axis, angle_axis, r0, delta_r, fc, phase_sgn, c_light,
    ):
        """Back-project a leaf subaperture onto a polar grid.

        Parameters
        ----------
        rc : complex, shape (n_pulses, nsamples)
        arp : float64, shape (n_pulses, 3)
        r_srp : float64, shape (n_pulses,)
        srp_center : float64, shape (3,)
        u_range, u_xr : float64, shape (3,)
        range_axis : float64, shape (nsamples,)
        angle_axis : float64, shape (n_angular,)
        r0, delta_r, fc, c_light : float64 scalars
        phase_sgn : int64

        Returns
        -------
        complex128, shape (nsamples, n_angular)
        """
        nsamples = rc.shape[1]
        n_ang = angle_axis.shape[0]
        n_pulses = rc.shape[0]
        four_pi = 4.0 * np.pi

        image = np.zeros((nsamples, n_ang), dtype=np.complex128)

        for i in nb.prange(nsamples):
            r_off = range_axis[i]
            for j in range(n_ang):
                theta = angle_axis[j]
                # Pixel ECF position (inline)
                px0 = (srp_center[0]
                       + r_off * u_range[0]
                       + theta * r0 * u_xr[0])
                px1 = (srp_center[1]
                       + r_off * u_range[1]
                       + theta * r0 * u_xr[1])
                px2 = (srp_center[2]
                       + r_off * u_range[2]
                       + theta * r0 * u_xr[2])

                acc_re = 0.0
                acc_im = 0.0

                for n in range(n_pulses):
                    dx = px0 - arp[n, 0]
                    dy = px1 - arp[n, 1]
                    dz = px2 - arp[n, 2]
                    r_n = np.sqrt(dx * dx + dy * dy + dz * dz)
                    dr = r_n - r_srp[n]

                    # Range bin index
                    r_bin = dr / delta_r + nsamples * 0.5
                    idx = int(np.floor(r_bin))
                    if idx < 0 or idx >= nsamples - 1:
                        continue
                    frac = r_bin - idx

                    # Linear interpolation of complex range profile
                    rc_re = (rc[n, idx].real * (1.0 - frac)
                             + rc[n, idx + 1].real * frac)
                    rc_im = (rc[n, idx].imag * (1.0 - frac)
                             + rc[n, idx + 1].imag * frac)

                    # Phase correction
                    phi = (-phase_sgn * four_pi
                           * fc * dr / c_light)
                    cos_p = np.cos(phi)
                    sin_p = np.sin(phi)

                    acc_re += rc_re * cos_p - rc_im * sin_p
                    acc_im += rc_re * sin_p + rc_im * cos_p

                image[i, j] = acc_re + 1j * acc_im

        return image

    @nb.njit(parallel=True, cache=True)
    def _nb_merge_kernel(
        left_img, left_ang, right_img, right_ang,
        parent_ang, range_axis, r0, d_left, d_right,
    ):
        """Merge two subaperture nodes via angular interpolation.

        Parameters
        ----------
        left_img, right_img : complex, shape (n_rg, n_ang_child)
        left_ang, right_ang : float64, shape (n_ang_child,)
        parent_ang : float64, shape (n_ang_parent,)
        range_axis : float64, shape (n_rg,)
        r0 : float64
        d_left, d_right : float64

        Returns
        -------
        complex, shape (n_rg, n_ang_parent)
        """
        n_rg = left_img.shape[0]
        n_parent = parent_ang.shape[0]
        n_left = left_ang.shape[0]
        n_right = right_ang.shape[0]

        left_min = left_ang[0]
        left_step = ((left_ang[n_left - 1] - left_min)
                     / (n_left - 1)) if n_left > 1 else 1.0
        right_min = right_ang[0]
        right_step = ((right_ang[n_right - 1] - right_min)
                      / (n_right - 1)) if n_right > 1 else 1.0

        parent_img = np.zeros(
            (n_rg, n_parent), dtype=left_img.dtype,
        )

        for i in nb.prange(n_rg):
            r_val = r0 + range_axis[i]
            if r_val <= 0.0:
                continue
            dt_l = d_left / r_val
            dt_r = d_right / r_val

            for j in range(n_parent):
                theta_p = parent_ang[j]

                # Left child interpolation
                theta_l = theta_p - dt_l
                idx_f = (theta_l - left_min) / left_step
                idx = int(np.floor(idx_f))
                val_l = left_img.dtype.type(0)
                if 0 <= idx < n_left - 1:
                    frac = idx_f - idx
                    val_l = (left_img[i, idx] * (1.0 - frac)
                             + left_img[i, idx + 1] * frac)

                # Right child interpolation
                theta_r = theta_p - dt_r
                idx_f = (theta_r - right_min) / right_step
                idx = int(np.floor(idx_f))
                val_r = right_img.dtype.type(0)
                if 0 <= idx < n_right - 1:
                    frac = idx_f - idx
                    val_r = (right_img[i, idx] * (1.0 - frac)
                             + right_img[i, idx + 1] * frac)

                parent_img[i, j] = val_l + val_r

        return parent_img

    @nb.njit(parallel=True, cache=True)
    def _nb_polar_to_rect(
        polar_img, angle_axis, range_axis, xr_axis, r0,
    ):
        """Convert polar-grid image to rectangular slant-plane grid.

        Parameters
        ----------
        polar_img : complex, shape (n_rg, n_ang)
        angle_axis : float64, shape (n_ang,)
        range_axis : float64, shape (n_rg,)
        xr_axis : float64, shape (n_xr,)
        r0 : float64

        Returns
        -------
        complex, shape (n_xr, n_rg)
        """
        n_rg = range_axis.shape[0]
        n_xr = xr_axis.shape[0]
        n_ang = angle_axis.shape[0]

        ang_min = angle_axis[0]
        ang_step = ((angle_axis[n_ang - 1] - ang_min)
                    / (n_ang - 1)) if n_ang > 1 else 1.0

        image = np.zeros((n_xr, n_rg), dtype=polar_img.dtype)

        for i in nb.prange(n_rg):
            r_val = r0 + range_axis[i]
            if r_val <= 0.0:
                continue
            for j in range(n_xr):
                theta = xr_axis[j] / r_val
                idx_f = (theta - ang_min) / ang_step
                idx = int(np.floor(idx_f))
                if 0 <= idx < n_ang - 1:
                    frac = idx_f - idx
                    image[j, i] = (
                        polar_img[i, idx] * (1.0 - frac)
                        + polar_img[i, idx + 1] * frac
                    )

        return image


@dataclass
class _SubapertureNode:
    """Internal representation of an FFBP tree node.

    Stores a subaperture image on a polar grid (range × angle)
    together with the coordinate axes and aperture metadata.
    """

    image: np.ndarray
    """Complex image, shape ``(N_range, N_angular)``."""

    range_axis: np.ndarray
    """Range offsets in metres from scene centre, shape ``(N_range,)``."""

    angle_axis: np.ndarray
    """Angular samples in radians, shape ``(N_angular,)``."""

    center_ecf: np.ndarray
    """ECF position of the subaperture centre, shape ``(3,)``."""

    center_time: float
    """Slow-time of the centre pulse (seconds)."""

    pulse_start: int
    """First pulse index (inclusive)."""

    pulse_end: int
    """Last pulse index (exclusive)."""


class FastBackProjection(ImageFormationAlgorithm):
    """Fast Factorized Back-Projection for strip map SAR.

    Hierarchical back-projection that factorises the O(N_rg N_az P)
    direct back-projection into O(N_rg N_ang P) via binary-tree
    subaperture merging with 1-D angular resampling, where
    ``N_ang`` (the ``n_angular`` parameter) is typically much smaller
    than the full cross-range output dimension.

    Parameters
    ----------
    metadata : CPHDMetadata
        CPHD metadata with populated PVP arrays.
    interpolator : callable, optional
        Interpolation function ``(x_old, y_old, x_new) -> y_new``
        for range-profile resampling.  Defaults to linear interp1d.
    range_weighting : str or callable, optional
        Window applied before range IFFT.  Built-in options:
        ``'uniform'``, ``'taylor'``, ``'hamming'``, ``'hanning'``,
        or a callable ``f(n) -> ndarray``.
    leaf_size : int
        Number of pulses per leaf subaperture.  Smaller values give
        higher accuracy but slower processing.  Default 8.
    n_angular : int
        Number of angular samples per tree node.  Controls the
        cross-range sampling density during the merge stages.
        Default 128.
    cross_range_spacing : float, optional
        Output cross-range pixel spacing in metres.  If *None*
        (default), derived from Nyquist sampling of the azimuth
        resolution.
    apply_amp_sf : bool
        Apply per-pulse AmpSF normalisation.  Default True.
    trim_invalid : bool
        Discard invalid pulses.  Default True.
    verbose : bool
        Print per-stage diagnostics.  Default True.
    """

    def __init__(
        self,
        metadata: CPHDMetadata,
        interpolator: Optional[Callable] = None,
        range_weighting: Union[str, Callable, None] = None,
        leaf_size: int = 8,
        n_angular: int = 128,
        cross_range_spacing: Optional[float] = None,
        apply_amp_sf: bool = True,
        trim_invalid: bool = True,
        use_numba: bool = True,
        verbose: bool = True,
    ) -> None:
        if metadata.pvp is None:
            raise ValueError(
                "CPHDMetadata must have populated PVP arrays"
            )
        self._metadata = metadata
        self._interp = interpolator or _scipy_interp1d
        self._range_weight_func = self._resolve_window(range_weighting)
        self._leaf_size = max(2, leaf_size)
        self._n_angular = max(4, n_angular)
        self._xr_spacing = cross_range_spacing
        self._apply_amp_sf = apply_amp_sf
        self._trim_invalid = trim_invalid
        self._use_numba = use_numba and _HAS_NUMBA
        self._verbose = verbose

        self._extract_params()

    # ------------------------------------------------------------------
    # Window resolution (shared with RDA)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_window(
        weighting: Union[str, Callable, None],
    ) -> Optional[Callable]:
        """Resolve weighting parameter to a window function."""
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
    # Parameter extraction (mirrors RDA._extract_params)
    # ------------------------------------------------------------------

    def _extract_params(self) -> None:
        """Extract physical parameters from CPHD metadata."""
        pvp = self._metadata.pvp
        gp = self._metadata.global_params

        # Phase sign convention
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

        # Per-pulse SRP positions and slant range
        self._srp_pos = pvp.srp_pos.copy()
        self._r_srp = norm(self._arp - self._srp_pos, axis=1)
        self._r0_center = float(np.mean(self._r_srp))

        # Reference SRP (center of aperture — used for slant plane)
        center_idx = self._npulses // 2
        self._ref_srp = self._srp_pos[center_idx].copy()

        # Range sample spacing after IFFT
        self._delta_r = _C / (2.0 * self._bandwidth)

        # Resolution estimates
        self._range_resolution = 0.886 * _C / (2.0 * self._bandwidth)
        total_time = float(
            self._mid_times[-1] - self._mid_times[0]
        )
        if total_time > 0:
            self._azimuth_resolution = (
                0.886 * self._wavelength * self._r0_center
                / (2.0 * self._speed * total_time)
            )
        else:
            self._azimuth_resolution = self._speed * self._pri

        # Output cross-range spacing (Nyquist for the resolution)
        if self._xr_spacing is None:
            self._xr_spacing = self._azimuth_resolution / 2.0

        # Slant-plane unit vectors at mid-aperture
        self._build_slant_plane()

    def _build_slant_plane(self) -> None:
        """Compute range and cross-range unit vectors."""
        center_idx = self._npulses // 2
        arp_c = self._arp[center_idx]
        srp_c = self._ref_srp

        # Range unit vector: SRP → ARP direction (away from scene)
        u_rg = arp_c - srp_c
        r_mag = float(norm(u_rg))
        if r_mag > 0:
            u_rg = u_rg / r_mag
        else:
            u_rg = np.array([0.0, 0.0, 1.0])

        # Velocity at mid-aperture
        v_c = self._varp[center_idx]

        # Cross-range: component of velocity perpendicular to range
        u_xr = v_c - np.dot(v_c, u_rg) * u_rg
        xr_mag = float(norm(u_xr))
        if xr_mag > 0:
            u_xr = u_xr / xr_mag
        else:
            u_xr = np.array([1.0, 0.0, 0.0])

        self._u_range = u_rg
        self._u_cross_range = u_xr

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_signal(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """Apply AmpSF normalisation and trim invalid pulses."""
        pvp = self._metadata.pvp
        npulses = signal.shape[0]

        valid_mask = np.ones(npulses, dtype=bool)

        if self._trim_invalid:
            if pvp.signal is not None:
                valid_mask &= pvp.signal > 0
            if pvp.amp_sf is not None:
                median_sf = float(np.median(pvp.amp_sf))
                if median_sf > 0:
                    deviation = (
                        np.abs(pvp.amp_sf - median_sf) / median_sf
                    )
                    valid_mask &= deviation < 0.5

        n_invalid = int(np.sum(~valid_mask))
        if n_invalid > 0 and self._verbose:
            print(
                f"  Trimming {n_invalid} invalid pulses "
                f"({npulses} -> {npulses - n_invalid})"
            )

        if self._apply_amp_sf and pvp.amp_sf is not None:
            amp_sf = pvp.amp_sf
            if self._trim_invalid:
                amp_sf = amp_sf[valid_mask]
            signal_out = (
                signal[valid_mask] if n_invalid > 0
                else signal.copy()
            )
            signal_out = signal_out * amp_sf[:, np.newaxis]
        elif n_invalid > 0:
            signal_out = signal[valid_mask]
        else:
            signal_out = signal.copy()

        # Update internal arrays for trimmed pulse set
        if n_invalid > 0:
            self._arp = self._arp[valid_mask]
            self._varp = self._varp[valid_mask]
            self._r_srp = self._r_srp[valid_mask]
            self._mid_times = self._mid_times[valid_mask]
            self._npulses = int(np.sum(valid_mask))

        return signal_out

    # ------------------------------------------------------------------
    # Stage 1: Range compression
    # ------------------------------------------------------------------

    def range_compress(self, signal: np.ndarray) -> np.ndarray:
        """Range compression via IFFT along fast-time axis.

        Parameters
        ----------
        signal : np.ndarray
            FX-domain phase history, shape ``(npulses, nsamples)``.

        Returns
        -------
        np.ndarray
            Range-compressed data, shape ``(npulses, nsamples)``.
        """
        data = signal.copy()
        if self._range_weight_func is not None:
            w = self._range_weight_func(data.shape[1]).astype(
                data.real.dtype,
            )
            data *= w[np.newaxis, :]
        return np.fft.fftshift(
            np.fft.ifft(
                np.fft.ifftshift(data, axes=1), axis=1,
            ),
            axes=1,
        )

    # ------------------------------------------------------------------
    # Stage 2: Leaf formation (direct back-projection)
    # ------------------------------------------------------------------

    def _back_project_leaf(
        self,
        rc: np.ndarray,
        start: int,
        end: int,
    ) -> _SubapertureNode:
        """Back-project a small subaperture onto a local polar grid.

        Parameters
        ----------
        rc : np.ndarray
            Range-compressed data, shape ``(npulses, nsamples)``.
        start, end : int
            Pulse index range ``[start, end)`` for this leaf.

        Returns
        -------
        _SubapertureNode
            Leaf image on a polar grid.
        """
        pvp = self._metadata.pvp
        nsamples = rc.shape[1]
        n_leaf = end - start
        n_ang = self._n_angular

        center_pulse = (start + end) // 2
        center_ecf = self._arp[center_pulse].copy()
        center_time = float(self._mid_times[center_pulse])
        srp_center = self._srp_pos[center_pulse].copy()

        # Range axis: same for all nodes (metres from scene centre)
        range_axis = (
            np.arange(nsamples) - nsamples / 2.0
        ) * self._delta_r

        # Angular extent of this leaf
        leaf_time = n_leaf * self._pri
        half_extent = (
            self._speed * leaf_time / (2.0 * self._r0_center)
        )
        # Oversample by 2× for interpolation quality in merges
        angle_axis = np.linspace(
            -half_extent * 2.0, half_extent * 2.0, n_ang,
        )

        # -- Numba fast path --
        if self._use_numba:
            image = _nb_bp_leaf(
                np.ascontiguousarray(rc[start:end]),
                np.ascontiguousarray(self._arp[start:end],
                                     dtype=np.float64),
                np.ascontiguousarray(self._r_srp[start:end],
                                     dtype=np.float64),
                np.ascontiguousarray(srp_center, dtype=np.float64),
                np.ascontiguousarray(self._u_range,
                                     dtype=np.float64),
                np.ascontiguousarray(self._u_cross_range,
                                     dtype=np.float64),
                np.ascontiguousarray(range_axis, dtype=np.float64),
                np.ascontiguousarray(angle_axis, dtype=np.float64),
                float(self._r0_center), float(self._delta_r),
                float(self._fc), np.int64(self._phase_sgn),
                _C,
            ).astype(np.complex64)
            return _SubapertureNode(
                image=image,
                range_axis=range_axis,
                angle_axis=angle_axis,
                center_ecf=center_ecf,
                center_time=center_time,
                pulse_start=start,
                pulse_end=end,
            )

        # -- Numpy fallback --
        # Output image
        image = np.zeros((nsamples, n_ang), dtype=np.complex64)

        # Pixel positions: P = SRP + r * u_range + θ * R0 * u_xr
        # We work relative to scene centre (ref_srp)
        # shape: (nsamples, n_ang, 3)
        r_grid = range_axis[:, np.newaxis]         # (N_rg, 1)
        theta_grid = angle_axis[np.newaxis, :]     # (1, N_ang)

        # 3D positions of all grid pixels
        pixel_ecf = (
            srp_center[np.newaxis, np.newaxis, :]
            + r_grid[:, :, np.newaxis] * self._u_range[np.newaxis, np.newaxis, :]
            + (theta_grid[:, :, np.newaxis] * self._r0_center)
            * self._u_cross_range[np.newaxis, np.newaxis, :]
        )  # (N_rg, N_ang, 3)

        # Back-project each pulse
        for n in range(start, end):
            arp_n = self._arp[n]           # (3,)
            r_srp_n = self._r_srp[n]       # scalar

            # Range from ARP to each pixel
            diff = pixel_ecf - arp_n[np.newaxis, np.newaxis, :]
            r_n = np.sqrt(np.sum(diff ** 2, axis=2))  # (N_rg, N_ang)

            # Differential range (from pulse's SRP)
            delta_r_n = r_n - r_srp_n  # (N_rg, N_ang)

            # Range bin index (float)
            r_bin = delta_r_n / self._delta_r + nsamples / 2.0

            # Interpolate range-compressed data for this pulse
            rc_pulse = rc[n, :]  # (nsamples,)
            x_old = np.arange(nsamples, dtype=np.float64)

            # Flatten for interpolation, then reshape
            r_bin_flat = r_bin.ravel()
            interp_vals = self._interp(
                x_old, rc_pulse, r_bin_flat,
            ).reshape(r_bin.shape)

            # Phase correction: remove carrier phase
            phase = np.exp(
                -1j * self._phase_sgn * 4.0 * np.pi
                * self._fc * delta_r_n / _C
            )

            image += interp_vals * phase

        return _SubapertureNode(
            image=image,
            range_axis=range_axis,
            angle_axis=angle_axis,
            center_ecf=center_ecf,
            center_time=center_time,
            pulse_start=start,
            pulse_end=end,
        )

    def form_leaves(
        self,
        rc: np.ndarray,
    ) -> List[_SubapertureNode]:
        """Form all leaf subapertures via direct back-projection.

        Parameters
        ----------
        rc : np.ndarray
            Range-compressed data, shape ``(npulses, nsamples)``.

        Returns
        -------
        list of _SubapertureNode
            One node per leaf subaperture.
        """
        leaves = []
        n = self._npulses
        leaf = self._leaf_size
        starts = list(range(0, n, leaf))

        for i, s in enumerate(starts):
            e = min(s + leaf, n)
            if e - s < 2:
                continue
            node = self._back_project_leaf(rc, s, e)
            leaves.append(node)

            if self._verbose and (
                i % max(1, len(starts) // 10) == 0
                or i == len(starts) - 1
            ):
                print(
                    f"  Leaf {i + 1}/{len(starts)}: "
                    f"pulses [{s}:{e}]"
                )

        return leaves

    # ------------------------------------------------------------------
    # Stage 3: Binary-tree merge
    # ------------------------------------------------------------------

    def _merge_pair(
        self,
        left: _SubapertureNode,
        right: _SubapertureNode,
    ) -> _SubapertureNode:
        """Merge two adjacent subaperture nodes.

        Resamples both children's angular grids into a parent grid
        that spans their combined angular extent, using 1-D
        interpolation per range bin.

        Parameters
        ----------
        left, right : _SubapertureNode
            Adjacent children to merge.

        Returns
        -------
        _SubapertureNode
            Parent node with combined image.
        """
        n_ang = self._n_angular

        # Parent centre: midpoint of children
        parent_center_ecf = 0.5 * (
            left.center_ecf + right.center_ecf
        )
        parent_center_time = 0.5 * (
            left.center_time + right.center_time
        )

        # Along-track offset of each child from parent centre
        d_left = float(np.dot(
            left.center_ecf - parent_center_ecf,
            self._u_cross_range,
        ))
        d_right = float(np.dot(
            right.center_ecf - parent_center_ecf,
            self._u_cross_range,
        ))

        # Parent angular extent: union of children shifted to parent
        # frame.  At reference range R0, child angle θ_c maps to
        # parent angle θ_p = θ_c + d / R0.
        dt_left = d_left / self._r0_center
        dt_right = d_right / self._r0_center
        angle_min = min(
            left.angle_axis[0] + dt_left,
            right.angle_axis[0] + dt_right,
        )
        angle_max = max(
            left.angle_axis[-1] + dt_left,
            right.angle_axis[-1] + dt_right,
        )
        parent_angle_axis = np.linspace(
            angle_min, angle_max, n_ang,
        )

        # Range axis (shared)
        range_axis = left.range_axis
        n_rg = len(range_axis)

        # -- Numba fast path --
        if self._use_numba:
            parent_image = _nb_merge_kernel(
                np.ascontiguousarray(left.image),
                np.ascontiguousarray(left.angle_axis,
                                     dtype=np.float64),
                np.ascontiguousarray(right.image),
                np.ascontiguousarray(right.angle_axis,
                                     dtype=np.float64),
                np.ascontiguousarray(parent_angle_axis,
                                     dtype=np.float64),
                np.ascontiguousarray(range_axis,
                                     dtype=np.float64),
                float(self._r0_center),
                d_left, d_right,
            )
            return _SubapertureNode(
                image=parent_image,
                range_axis=range_axis,
                angle_axis=parent_angle_axis,
                center_ecf=parent_center_ecf,
                center_time=parent_center_time,
                pulse_start=left.pulse_start,
                pulse_end=right.pulse_end,
            )

        # -- Numpy fallback --
        parent_image = np.zeros(
            (n_rg, n_ang), dtype=np.complex64,
        )

        # For each range bin, 1-D angular interpolation
        for i in range(n_rg):
            r_val = self._r0_center + range_axis[i]
            if r_val <= 0:
                continue

            # Angular offset is range-dependent
            delta_theta_left = d_left / r_val
            delta_theta_right = d_right / r_val

            # Map parent angles to child local angles
            theta_in_left = parent_angle_axis - delta_theta_left
            theta_in_right = parent_angle_axis - delta_theta_right

            # Interpolate left child
            val_left = self._interp(
                left.angle_axis,
                left.image[i, :],
                theta_in_left,
            )

            # Interpolate right child
            val_right = self._interp(
                right.angle_axis,
                right.image[i, :],
                theta_in_right,
            )

            parent_image[i, :] = val_left + val_right

        return _SubapertureNode(
            image=parent_image,
            range_axis=range_axis,
            angle_axis=parent_angle_axis,
            center_ecf=parent_center_ecf,
            center_time=parent_center_time,
            pulse_start=left.pulse_start,
            pulse_end=right.pulse_end,
        )

    def merge_tree(
        self,
        leaves: List[_SubapertureNode],
    ) -> _SubapertureNode:
        """Merge leaf subapertures in a binary tree.

        Parameters
        ----------
        leaves : list of _SubapertureNode
            Leaf nodes from ``form_leaves``.

        Returns
        -------
        _SubapertureNode
            Root node with fully focused image.
        """
        nodes = list(leaves)
        level = 0

        while len(nodes) > 1:
            level += 1
            if self._verbose:
                print(
                    f"  Merge level {level}: "
                    f"{len(nodes)} nodes → "
                    f"{(len(nodes) + 1) // 2}"
                )

            merged = []
            i = 0
            while i < len(nodes):
                if i + 1 < len(nodes):
                    merged.append(
                        self._merge_pair(nodes[i], nodes[i + 1])
                    )
                    i += 2
                else:
                    # Odd node: carry forward
                    merged.append(nodes[i])
                    i += 1
            nodes = merged

        return nodes[0]

    # ------------------------------------------------------------------
    # Stage 4: Polar → rectangular conversion
    # ------------------------------------------------------------------

    def polar_to_rect(
        self,
        node: _SubapertureNode,
    ) -> np.ndarray:
        """Convert polar-grid image to rectangular slant-plane grid.

        Parameters
        ----------
        node : _SubapertureNode
            Root node from ``merge_tree``.

        Returns
        -------
        np.ndarray
            Complex SAR image, shape ``(N_xr, N_rg)``.
        """
        range_axis = node.range_axis
        angle_axis = node.angle_axis
        n_rg = len(range_axis)

        # Cross-range extent from angular extent at R0
        xr_min = angle_axis[0] * self._r0_center
        xr_max = angle_axis[-1] * self._r0_center
        xr_extent = xr_max - xr_min

        n_xr = max(1, int(xr_extent / self._xr_spacing))
        xr_axis = np.linspace(xr_min, xr_max, n_xr)

        if self._verbose:
            ang_extent_deg = np.degrees(
                angle_axis[-1] - angle_axis[0]
            )
            print(f"  Root angular extent: "
                  f"{ang_extent_deg:.4f} deg "
                  f"({angle_axis[-1] - angle_axis[0]:.6f} rad)")
            print(f"  Cross-range extent: "
                  f"{xr_extent:.1f} m")
            print(f"  Output: {n_xr} x {n_rg} "
                  f"(cross-range x range)")

        # -- Numba fast path --
        if self._use_numba:
            return _nb_polar_to_rect(
                np.ascontiguousarray(node.image),
                np.ascontiguousarray(angle_axis,
                                     dtype=np.float64),
                np.ascontiguousarray(range_axis,
                                     dtype=np.float64),
                np.ascontiguousarray(xr_axis,
                                     dtype=np.float64),
                float(self._r0_center),
            )

        # -- Numpy fallback --
        image = np.zeros((n_xr, n_rg), dtype=np.complex64)

        # For each range bin, interpolate from angle to cross-range
        for i in range(n_rg):
            r_val = self._r0_center + range_axis[i]
            if r_val <= 0:
                continue
            # Cross-range → angle mapping
            theta_target = xr_axis / r_val
            image[:, i] = self._interp(
                angle_axis, node.image[i, :], theta_target,
            )

        return image

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def form_image(
        self,
        signal: np.ndarray,
        geometry: Any = None,
    ) -> np.ndarray:
        """Run the full FFBP pipeline.

        Parameters
        ----------
        signal : np.ndarray
            CPHD FX-domain phase history,
            shape ``(npulses, nsamples)``.
        geometry : Any
            Ignored (geometry computed from metadata).

        Returns
        -------
        np.ndarray
            Complex SAR image, shape ``(N_xr, N_rg)``.
        """
        # Ensure complex type
        if signal.dtype.names:
            signal = (
                signal['real'].astype(np.float32)
                + 1j * signal['imag'].astype(np.float32)
            )

        if self._verbose:
            print(f"FFBP: {self._npulses} pulses x "
                  f"{self._nsamples} samples")
            print(f"  Phase SGN: {self._phase_sgn:+d}")
            print(f"  Wavelength: {self._wavelength * 100:.2f} cm")
            print(f"  Bandwidth: {self._bandwidth / 1e6:.1f} MHz")
            print(f"  PRF: {self._prf:.1f} Hz")
            print(f"  Platform speed: {self._speed:.1f} m/s")
            print(f"  Reference range: "
                  f"{self._r0_center / 1e3:.1f} km")
            print(f"  Range resolution: "
                  f"{self._range_resolution:.3f} m")
            print(f"  Azimuth resolution: "
                  f"{self._azimuth_resolution:.3f} m")
            print(f"  Leaf size: {self._leaf_size} pulses")
            print(f"  Angular samples: {self._n_angular}")
            print(f"  Cross-range spacing: "
                  f"{self._xr_spacing:.3f} m")
            print(f"  Numba acceleration: "
                  f"{'enabled' if self._use_numba else 'disabled'}")
            n_leaves = (self._npulses + self._leaf_size - 1) // self._leaf_size
            n_levels = int(np.ceil(np.log2(max(2, n_leaves))))
            print(f"  Tree: {n_leaves} leaves, "
                  f"{n_levels} merge levels")

        # Preprocess
        signal = self._preprocess_signal(signal)

        # Stage 1: Range compress
        if self._verbose:
            print("\nStage 1: Range compression...")
        rc = self.range_compress(signal)

        # Stage 2: Leaf formation
        if self._verbose:
            print("\nStage 2: Leaf back-projection...")
        leaves = self.form_leaves(rc)

        # Stage 3: Binary-tree merge
        if self._verbose:
            print("\nStage 3: Binary-tree merge...")
        root = self.merge_tree(leaves)

        # Stage 4: Polar → rectangular
        if self._verbose:
            print("\nStage 4: Polar to rectangular...")
        image = self.polar_to_rect(root)

        if self._verbose:
            print(f"\nImage formed: {image.shape}, "
                  f"dtype: {image.dtype}")

        return image

    # ------------------------------------------------------------------
    # Output grid metadata
    # ------------------------------------------------------------------

    def get_output_grid(self) -> Dict[str, Any]:
        """Return output grid parameters.

        Returns
        -------
        Dict[str, Any]
            Grid metadata including resolution, spacing, and
            reference geometry.
        """
        grid: Dict[str, Any] = {
            'grid_type': 'XRGYCR',
            'range_resolution': self._range_resolution,
            'azimuth_resolution': self._azimuth_resolution,
            'range_sample_spacing': self._delta_r,
            'cross_range_sample_spacing': self._xr_spacing,
            'algorithm': 'FFBP',
        }

        rg = self._metadata.reference_geometry
        if rg is not None:
            if rg.graze_angle_deg is not None:
                grid['graze_angle_deg'] = rg.graze_angle_deg
            if rg.side_of_track is not None:
                grid['side_of_track'] = rg.side_of_track

        sc = self._metadata.scene_coordinates
        if sc is not None and sc.iarp_llh is not None:
            grid['iarp_llh'] = sc.iarp_llh

        return grid
