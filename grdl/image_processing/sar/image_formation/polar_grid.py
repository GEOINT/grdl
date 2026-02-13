# -*- coding: utf-8 -*-
"""
Polar Grid - K-space annulus bounds and resampled grid dimensions.

Computes the spatial frequency bounds (ku, kv) of the CPHD polar
annulus, determines the inscribed or circumscribed rectangular grid,
and derives resolution and sampling parameters needed by the Polar
Format Algorithm and SICD Grid metadata.

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
2026-02-12
"""

# Standard library
from typing import Dict, Optional, Tuple

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.sar.image_formation.geometry import (
    CollectionGeometry,
)


class PolarGrid:
    """K-space polar grid for the Polar Format Algorithm.

    Computes the annular bounds in spatial frequency, the inscribed
    or circumscribed rectangular grid, and resolution/sampling
    metrics.

    Parameters
    ----------
    geometry : CollectionGeometry
        Computed collection geometry.
    grid_mode : str
        ``'inscribed'`` (default) or ``'circumscribed'``.
    range_oversample : float
        Range k-space oversampling factor.  1.0 = Nyquist rate,
        2.0 = twice Nyquist (twice as many range samples).
    azimuth_oversample : float
        Azimuth k-space oversampling factor.  1.0 = Nyquist rate,
        2.0 = twice Nyquist (twice as many azimuth samples).

    Attributes
    ----------
    kv_bounds : np.ndarray
        Range spatial frequency bounds ``[kv_min, kv_max]``.
    ku_bounds : np.ndarray
        Azimuth spatial frequency bounds ``[ku_min, ku_max]``.
    rec_n_samples : int
        Number of range samples in the rectangular grid.
    rec_n_pulses : int
        Number of azimuth samples in the rectangular grid.
    range_resolution : float
        Range resolution (meters).
    azimuth_resolution : float
        Azimuth resolution (meters).
    rg_ss, az_ss : float
        Range and azimuth sample spacing (1/m).
    rg_imp_resp_bw, az_imp_resp_bw : float
        Range and azimuth impulse response bandwidth (1/m).
    rg_kctr, az_kctr : float
        Range and azimuth k-space center (1/m).

    Examples
    --------
    >>> grid = PolarGrid(geometry, grid_mode='inscribed')
    >>> print(f"Range resolution: {grid.range_resolution:.2f} m")
    >>> print(f"Grid: {grid.rec_n_pulses} x {grid.rec_n_samples}")
    """

    def __init__(
        self,
        geometry: CollectionGeometry,
        grid_mode: str = 'inscribed',
        range_oversample: float = 1.0,
        azimuth_oversample: float = 1.0,
    ) -> None:
        self.geometry = geometry
        self.grid_mode = grid_mode.upper()
        self.range_oversample = range_oversample
        self.azimuth_oversample = azimuth_oversample

        # Spatial frequency conversion factor: k = 2f/c
        self.sf_conv = 2.0 / geometry.c

        self._compute_grid_limits()
        self._compute_grid_parameters()

    # ------------------------------------------------------------------
    # Grid limits (k-space bounds)
    # ------------------------------------------------------------------

    def _compute_grid_limits(self) -> None:
        """Compute inscribed or circumscribed k-space bounds.

        The annulus has four edges: inner arc (fx1, all pulses), outer arc
        (fx2, all pulses), first pulse radial (pulse 0, all freqs), last
        pulse radial (pulse N-1, all freqs).

        For the inscribed rectangle, the ku bounds must be valid at every
        kv in the rectangle — the tightest constraint is at ``kv_bounds[0]``
        (smallest kv) where the annulus ku extent is narrowest.

        All pulses are considered for ku extrema, not just first/last,
        to handle ground-plane collections where ``k_sf`` varies per pulse.
        """
        geo = self.geometry
        scale = geo.k_sf

        # kv bounds (range direction): fx1 and fx2 projected per pulse
        sf_v_fx1 = self.sf_conv * geo.fx1 * np.cos(geo.phi) * scale
        sf_v_fx2 = self.sf_conv * geo.fx2 * np.cos(geo.phi) * scale

        # ku at inner and outer frequency edges for ALL pulses
        # ku = sf_conv * freq * sin(phi) * k_sf
        sf_u_fx1 = self.sf_conv * geo.fx1 * np.sin(geo.phi) * scale
        sf_u_fx2 = self.sf_conv * geo.fx2 * np.sin(geo.phi) * scale

        if self.grid_mode == 'INSCRIBED':
            kv_min_idx = np.argmax(np.abs(sf_v_fx1))
            kv_max_idx = np.argmin(np.abs(sf_v_fx2))
            self.kv_bounds = np.array([
                sf_v_fx1[kv_min_idx], sf_v_fx2[kv_max_idx],
            ])

            # Inscribed ku bounds: the annulus ku extent is narrowest
            # at kv_bounds[0] (smallest kv). Use tan(phi) mapping:
            # ku = kv * tan(phi), so at kv_min the ku range is tightest.
            # Evaluate ku for all pulses at the inscribed kv_min.
            kv_min = self.kv_bounds[0]
            ku_at_kv_min = kv_min * np.tan(geo.phi)
            self.ku_bounds = np.array([
                np.min(ku_at_kv_min),
                np.max(ku_at_kv_min),
            ])

        elif self.grid_mode == 'CIRCUMSCRIBED':
            kv_min_idx = np.argmin(np.abs(sf_v_fx1))
            kv_max_idx = np.argmax(np.abs(sf_v_fx2))
            self.kv_bounds = np.array([
                sf_v_fx1[kv_min_idx], sf_v_fx2[kv_max_idx],
            ])

            # Circumscribed: widest ku extent across all pulses and freqs
            self.ku_bounds = np.array([
                min(np.min(sf_u_fx1), np.min(sf_u_fx2)),
                max(np.max(sf_u_fx1), np.max(sf_u_fx2)),
            ])
        else:
            raise ValueError(
                f"grid_mode must be 'inscribed' or 'circumscribed', "
                f"got '{self.grid_mode}'"
            )

    # ------------------------------------------------------------------
    # Grid parameters (resolution, sampling, dimensions)
    # ------------------------------------------------------------------

    def _compute_grid_parameters(self) -> None:
        """Compute resolution, sampling, and resampled grid dimensions.

        Resolution is determined entirely by the k-space bandwidth
        (kv/ku bounds).  The oversampling factors control only the
        number of samples in the rectangular grid — they change the
        sampling density without affecting bandwidth or resolution.
        """
        geo = self.geometry
        c = geo.c

        # Processed bandwidth
        self.proc_bw = float(abs(self.kv_bounds[1] - self.kv_bounds[0])
                             / self.sf_conv)

        # Range resolution (from k-space bandwidth, not sampling)
        self.range_resolution = c * 0.886 / (2 * self.proc_bw)

        # Azimuth resolution: 0.886 / ku_bandwidth (ku in cycles/m)
        center_freq = float(np.mean(self.kv_bounds) / self.sf_conv)
        self.center_wl = c / center_freq
        ku_span = float(abs(self.ku_bounds[1] - self.ku_bounds[0]))
        self.azimuth_resolution = 0.886 / ku_span

        # Range sampling: Nyquist from receive window length, then
        # scale by oversampling factor
        rcv_params = self._get_rcv_window()
        if rcv_params is not None:
            nyquist_freq_sampling = 2.0 / rcv_params / 0.886
        else:
            nyquist_freq_sampling = float(geo.fxss[0])

        self.rec_n_samples = max(
            1,
            int(self.proc_bw / nyquist_freq_sampling
                * self.range_oversample),
        )

        # Azimuth sampling: minimum ku step between adjacent pulses,
        # then scale by oversampling factor
        az_sf_sampling = float(np.min(np.abs(np.diff(
            self.kv_bounds[0] * np.tan(geo.phi),
        )))) * 0.889

        self.rec_n_pulses = max(
            1,
            int(ku_span / az_sf_sampling * self.azimuth_oversample),
        )

        # SICD Grid parameters
        self.rg_imp_resp_bw = float(self.kv_bounds[1] - self.kv_bounds[0])
        self.rg_imp_resp_wid = 0.886 / self.rg_imp_resp_bw
        self.rg_ss = 1.0 / self.rg_imp_resp_bw
        self.rg_kctr = float(np.mean(self.kv_bounds))
        self.rg_delta_k1 = float(self.kv_bounds[0])
        self.rg_delta_k2 = float(self.kv_bounds[1])

        self.az_imp_resp_bw = ku_span
        self.az_imp_resp_wid = 0.886 / self.az_imp_resp_bw
        self.az_ss = 1.0 / self.az_imp_resp_bw
        self.az_kctr = float(np.mean(self.ku_bounds))

    def _get_rcv_window(self) -> Optional[float]:
        """Get receive window length from metadata."""
        rcv = self._get_metadata_attr('rcv_parameters')
        if rcv is not None:
            return getattr(rcv, 'window_length', None)
        return None

    def _get_metadata_attr(self, name: str):
        """Safely get an attribute from the metadata."""
        return getattr(self.geometry._metadata, name, None)

    # ------------------------------------------------------------------
    # Polar grid sample accessors (used by PFA)
    # ------------------------------------------------------------------

    def get_kv_for_pulse(self, pulse: int) -> np.ndarray:
        """Return kv (range spatial freq) values for a given pulse.

        Parameters
        ----------
        pulse : int
            Pulse index.

        Returns
        -------
        np.ndarray
            kv values, shape ``(nsamples,)``.
        """
        geo = self.geometry
        scaling = geo.k_sf[pulse]
        freq = (
            np.arange(geo.nsamples) * geo.fxss[pulse] + geo.fx0[pulse]
        )
        return self.sf_conv * freq * np.cos(geo.phi[pulse]) * scaling

    def get_ku_for_sample(self, sample: int) -> np.ndarray:
        """Return ku (azimuth spatial freq) values for a given sample.

        Parameters
        ----------
        sample : int
            Sample index.

        Returns
        -------
        np.ndarray
            ku values, shape ``(npulses,)``.
        """
        geo = self.geometry
        scaling = geo.k_sf
        if sample == -1:
            sample = geo.nsamples - 1
        freq = sample * geo.fxss + geo.fx0
        return self.sf_conv * freq * np.sin(geo.phi) * scaling

    def get_output_grid(self) -> Dict[str, float]:
        """Return output grid parameters for SICD metadata.

        Returns
        -------
        Dict[str, float]
            Grid parameters matching SICD Grid section fields.
        """
        return {
            'image_plane': self.geometry.image_plane,
            'type': 'RGAZIM',
            'rg_ss': self.rg_ss,
            'rg_imp_resp_bw': self.rg_imp_resp_bw,
            'rg_imp_resp_wid': self.rg_imp_resp_wid,
            'rg_kctr': self.rg_kctr,
            'rg_delta_k1': self.rg_delta_k1,
            'rg_delta_k2': self.rg_delta_k2,
            'az_ss': self.az_ss,
            'az_imp_resp_bw': self.az_imp_resp_bw,
            'az_imp_resp_wid': self.az_imp_resp_wid,
            'az_kctr': self.az_kctr,
            'ku_min': float(self.ku_bounds[0]),
            'ku_max': float(self.ku_bounds[1]),
            'range_resolution': self.range_resolution,
            'azimuth_resolution': self.azimuth_resolution,
            'rec_n_pulses': self.rec_n_pulses,
            'rec_n_samples': self.rec_n_samples,
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(self, ax=None) -> None:
        """Plot the polar grid annulus and rectangular bounds.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        """
        try:
            import matplotlib
            matplotlib.use("QtAgg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting."
            )

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        geo = self.geometry

        # Plot annulus boundaries: first and last pulse arcs
        for pulse_idx in [0, -1]:
            p = pulse_idx if pulse_idx >= 0 else geo.npulses - 1
            scaling = geo.k_sf[p]
            freq = np.arange(geo.nsamples) * geo.fxss[p] + geo.fx0[p]
            sf = self.sf_conv * freq
            ku = sf * np.sin(geo.phi[p]) * scaling
            kv = sf * np.cos(geo.phi[p]) * scaling
            ax.scatter(ku, kv, s=1, alpha=0.5)

        # Plot first and last sample arcs
        for sample_idx in [0, -1]:
            s = sample_idx if sample_idx >= 0 else geo.nsamples - 1
            freq = s * geo.fxss + geo.fx0
            sf = self.sf_conv * freq
            ku = sf * np.sin(geo.phi) * geo.k_sf
            kv = sf * np.cos(geo.phi) * geo.k_sf
            ax.scatter(ku, kv, s=1, alpha=0.5)

        # Rectangular bounds
        ku0, ku1 = self.ku_bounds
        kv0, kv1 = self.kv_bounds
        rect_ku = [ku0, ku1, ku1, ku0, ku0]
        rect_kv = [kv0, kv0, kv1, kv1, kv0]
        ax.plot(rect_ku, rect_kv, 'r-', linewidth=2,
                label=f'{self.grid_mode} grid')

        ax.set_xlabel('Cross Range Spatial Frequency (1/m)')
        ax.set_ylabel('Range Spatial Frequency (1/m)')
        ax.set_title('Polar Grid')
        ax.legend()
