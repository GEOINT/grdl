# -*- coding: utf-8 -*-
"""
LFM Polyphase Interpolation — Resample an LFM chirp at fractional rates.

Generates a complex linear frequency modulated (LFM) chirp at a given
sampling rate, then resamples at two fractional rates (0.85x and 1.25x)
using both ``PolyphaseInterpolator`` and ``KaiserSincInterpolator`` for
comparison.  Plots time-domain and frequency-domain results overlaid.

Dependencies
------------
matplotlib

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

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
import sys
from pathlib import Path

# Third-party
import numpy as np

# GRDL
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from grdl.interpolation import KaiserSincInterpolator, PolyphaseInterpolator

# Original smalleyd branch implementation for comparison
from polyphaseinterpolation import Polyphase_Interpolation


# ── LFM chirp generator ─────────────────────────────────────────────


def generate_lfm(
    fs: float,
    duration: float,
    f_start: float,
    f_stop: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a complex LFM (linear frequency modulated) chirp.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    duration : float
        Pulse duration in seconds.
    f_start : float
        Start frequency in Hz.
    f_stop : float
        Stop frequency in Hz.

    Returns
    -------
    t : np.ndarray
        Time vector, shape ``(N,)``.
    signal : np.ndarray
        Complex LFM chirp, shape ``(N,)``.
    """
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    chirp_rate = (f_stop - f_start) / duration
    phase = 2.0 * np.pi * (f_start * t + 0.5 * chirp_rate * t ** 2)
    return t, np.exp(1j * phase)


# ── Plotting ─────────────────────────────────────────────────────────


def plot_lfm_interpolation(
    t_orig: np.ndarray,
    sig_orig: np.ndarray,
    t_down_poly: np.ndarray,
    sig_down_poly: np.ndarray,
    t_up_poly: np.ndarray,
    sig_up_poly: np.ndarray,
    t_down_kaiser: np.ndarray,
    sig_down_kaiser: np.ndarray,
    t_up_kaiser: np.ndarray,
    sig_up_kaiser: np.ndarray,
    t_down_orig: np.ndarray,
    sig_down_orig: np.ndarray,
    t_up_orig: np.ndarray,
    sig_up_orig: np.ndarray,
    fs: float,
) -> None:
    """Plot original, polyphase, Kaiser-sinc, and smalleyd interpolated signals.

    Parameters
    ----------
    t_orig : np.ndarray
        Original time vector.
    sig_orig : np.ndarray
        Original complex LFM signal.
    t_down_poly, sig_down_poly : np.ndarray
        GRDL Polyphase 0.85x time / signal.
    t_up_poly, sig_up_poly : np.ndarray
        GRDL Polyphase 1.25x time / signal.
    t_down_kaiser, sig_down_kaiser : np.ndarray
        GRDL Kaiser-sinc 0.85x time / signal.
    t_up_kaiser, sig_up_kaiser : np.ndarray
        GRDL Kaiser-sinc 1.25x time / signal.
    t_down_orig, sig_down_orig : np.ndarray
        Smalleyd Polyphase 0.85x time / signal.
    t_up_orig, sig_up_orig : np.ndarray
        Smalleyd Polyphase 1.25x time / signal.
    fs : float
        Original sampling rate in Hz.
    """
    import matplotlib
    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt  # noqa: E402

    t_us = 1e6  # seconds → microseconds

    fs_down = 0.85 * fs
    fs_up = 1.25 * fs

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    # ── Time domain — 0.85x ──────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(t_orig * t_us, sig_orig.real, linewidth=0.8,
            color="steelblue", label=f"Original ({fs / 1e6:.0f} MHz)")
    ax.plot(t_down_poly * t_us, sig_down_poly.real, linewidth=0.8,
            linestyle="--", color="firebrick", label="GRDL Polyphase")
    ax.plot(t_down_kaiser * t_us, sig_down_kaiser.real, linewidth=0.8,
            linestyle=":", color="darkorange", label="GRDL Kaiser-sinc")
    ax.plot(t_down_orig * t_us, sig_down_orig.real, linewidth=0.8,
            linestyle="-.", color="mediumpurple", label="Smalleyd Polyphase")
    ax.set_ylabel("Real Part")
    ax.set_title(f"Time Domain — 0.85x  (fs = {fs_down / 1e6:.1f} MHz)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Time domain — 1.25x ──────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(t_orig * t_us, sig_orig.real, linewidth=0.8,
            color="steelblue", label=f"Original ({fs / 1e6:.0f} MHz)")
    ax.plot(t_up_poly * t_us, sig_up_poly.real, linewidth=0.8,
            linestyle="--", color="seagreen", label="GRDL Polyphase")
    ax.plot(t_up_kaiser * t_us, sig_up_kaiser.real, linewidth=0.8,
            linestyle=":", color="darkorange", label="GRDL Kaiser-sinc")
    ax.plot(t_up_orig * t_us, sig_up_orig.real, linewidth=0.8,
            linestyle="-.", color="mediumpurple", label="Smalleyd Polyphase")
    ax.set_title(f"Time Domain — 1.25x  (fs = {fs_up / 1e6:.1f} MHz)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Spectrum helper ──────────────────────────────────────────────
    def _spectrum_db(sig: np.ndarray, sample_rate: float) -> tuple:
        n = len(sig)
        freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate))
        mag = np.abs(np.fft.fftshift(np.fft.fft(sig)))
        mag_db = 20.0 * np.log10(mag / mag.max() + 1e-12)
        return freqs, mag_db

    # ── Spectrum — 0.85x ─────────────────────────────────────────────
    ax = axes[1, 0]
    f_orig, db_orig = _spectrum_db(sig_orig, fs)
    f_dp, db_dp = _spectrum_db(sig_down_poly, fs_down)
    f_dk, db_dk = _spectrum_db(sig_down_kaiser, fs_down)
    f_do, db_do = _spectrum_db(sig_down_orig, fs_down)
    ax.plot(f_orig / 1e6, db_orig, linewidth=0.8,
            color="steelblue", label="Original")
    ax.plot(f_dp / 1e6, db_dp, linewidth=0.8,
            linestyle="--", color="firebrick", label="GRDL Polyphase")
    ax.plot(f_dk / 1e6, db_dk, linewidth=0.8,
            linestyle=":", color="darkorange", label="GRDL Kaiser-sinc")
    ax.plot(f_do / 1e6, db_do, linewidth=0.8,
            linestyle="-.", color="mediumpurple", label="Smalleyd Polyphase")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Frequency Spectrum — 0.85x", fontsize=11)
    ax.set_ylim(-80, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Spectrum — 1.25x ─────────────────────────────────────────────
    ax = axes[1, 1]
    f_up, db_up = _spectrum_db(sig_up_poly, fs_up)
    f_uk, db_uk = _spectrum_db(sig_up_kaiser, fs_up)
    f_uo, db_uo = _spectrum_db(sig_up_orig, fs_up)
    ax.plot(f_orig / 1e6, db_orig, linewidth=0.8,
            color="steelblue", label="Original")
    ax.plot(f_up / 1e6, db_up, linewidth=0.8,
            linestyle="--", color="seagreen", label="GRDL Polyphase")
    ax.plot(f_uk / 1e6, db_uk, linewidth=0.8,
            linestyle=":", color="darkorange", label="GRDL Kaiser-sinc")
    ax.plot(f_uo / 1e6, db_uo, linewidth=0.8,
            linestyle="-.", color="mediumpurple", label="Smalleyd Polyphase")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_title("Frequency Spectrum — 1.25x", fontsize=11)
    ax.set_ylim(-80, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "LFM Chirp — GRDL Polyphase vs Kaiser-Sinc vs Smalleyd Polyphase",
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    """Generate LFM chirp, resample at 0.85x and 1.25x, and plot."""
    # Signal parameters
    fs = 100e6          # 100 MHz sampling rate
    duration = 10e-6    # 10 µs pulse
    f_start = -20e6     # -20 MHz start
    f_stop = 20e6       #  20 MHz stop (40 MHz bandwidth)

    print(f"Generating LFM chirp: fs={fs / 1e6:.0f} MHz, "
          f"BW={abs(f_stop - f_start) / 1e6:.0f} MHz, "
          f"duration={duration * 1e6:.0f} µs")

    t_orig, sig_orig = generate_lfm(fs, duration, f_start, f_stop)
    print(f"  Original: {len(t_orig)} samples")

    # Sample indices as coordinates (uniform integer grid)
    x_orig = np.arange(len(t_orig), dtype=np.float64)

    # GRDL interpolators — matched kernel length and beta for fair comparison
    poly = PolyphaseInterpolator(kernel_length=8, num_phases=256, beta=6.0)
    kaiser = KaiserSincInterpolator(kernel_length=8, beta=6.0)

    # Smalleyd branch polyphase (Remez prototype, numba-accelerated)
    orig_poly = Polyphase_Interpolation()
    orig_poly.build_filter_function()

    # ── Resample at 0.85x ────────────────────────────────────────────
    rate_down = 0.85
    n_down = int(len(t_orig) * rate_down)
    x_down = np.linspace(0, len(t_orig) - 1, n_down)
    t_down = x_down / fs

    sig_down_poly = poly(x_orig, sig_orig, x_down)
    sig_down_kaiser = kaiser(x_orig, sig_orig, x_down)
    sig_down_orig = orig_poly.poly_interp(
        x_orig, sig_orig.astype(np.complex64),
        orig_poly.the_filter, x_down,
    )
    print(f"  0.85x rate: {n_down} samples (fs = {fs * rate_down / 1e6:.1f} MHz)")

    # ── Resample at 1.25x ────────────────────────────────────────────
    rate_up = 1.25
    n_up = int(len(t_orig) * rate_up)
    x_up = np.linspace(0, len(t_orig) - 1, n_up)
    t_up = x_up / fs

    sig_up_poly = poly(x_orig, sig_orig, x_up)
    sig_up_kaiser = kaiser(x_orig, sig_orig, x_up)
    sig_up_orig = orig_poly.poly_interp(
        x_orig, sig_orig.astype(np.complex64),
        orig_poly.the_filter, x_up,
    )
    print(f"  1.25x rate: {n_up} samples (fs = {fs * rate_up / 1e6:.1f} MHz)")

    # ── Plot ─────────────────────────────────────────────────────────
    plot_lfm_interpolation(
        t_orig, sig_orig,
        t_down, sig_down_poly, t_up, sig_up_poly,
        t_down, sig_down_kaiser, t_up, sig_up_kaiser,
        t_down, sig_down_orig, t_up, sig_up_orig,
        fs,
    )


if __name__ == "__main__":
    main()
