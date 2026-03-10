# -*- coding: utf-8 -*-
"""
Polyphase interpolation example — resample a chirp signal.

Demonstrates using ``grdl.interpolation.PolyphaseInterpolator`` to
resample a complex linear frequency modulated (LFM) chirp from one
sample rate to another.  Compares Kaiser and Remez prototype designs.

Dependencies
------------
matplotlib

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
2024-08-02

Modified
--------
2026-03-10
"""

# Third-party
import numpy as np
import matplotlib.pyplot as plt

# GRDL internal
from grdl.interpolation import PolyphaseInterpolator


def main() -> None:
    """Resample a chirp signal using polyphase interpolation."""
    # Generate a complex LFM chirp
    chirp_rate = 250.0  # Hz/s
    fs_in = 5e5  # Input sample rate (Hz)
    n_in = int(fs_in)
    x_in = np.arange(n_in) / fs_in
    signal_in = np.exp(1j * 2 * np.pi * chirp_rate * x_in ** 2).astype(
        np.complex64
    )

    # Define output sample grid (different rate)
    fs_out = 2.123e5
    n_out = int(fs_out)
    x_out = np.arange(n_out) / fs_out

    # --- Kaiser prototype (default) ---
    interp_kaiser = PolyphaseInterpolator(
        kernel_length=16, num_phases=2048, prototype='kaiser',
    )
    y_kaiser = interp_kaiser(x_in, signal_in, x_out)

    # --- Remez prototype (better stopband rejection) ---
    interp_remez = PolyphaseInterpolator(
        kernel_length=16, num_phases=2048, prototype='remez',
    )
    y_remez = interp_remez(x_in, signal_in, x_out)

    # --- Plot time-domain comparison ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].set_title('Time Domain')
    axes[0].plot(x_in[:500], np.real(signal_in[:500]), label='Input', alpha=0.7)
    axes[0].plot(x_out[:200], np.real(y_kaiser[:200]), '--', label='Kaiser')
    axes[0].plot(x_out[:200], np.real(y_remez[:200]), ':', label='Remez')
    axes[0].legend()
    axes[0].set_xlabel('Time (s)')

    # --- Plot spectral comparison ---
    in_freq = (np.arange(n_in) / n_in - 0.5) * fs_in
    in_spec = np.fft.fftshift(np.fft.fft(signal_in))

    out_freq_k = (np.arange(len(y_kaiser)) / len(y_kaiser) - 0.5) * fs_out
    out_spec_k = np.fft.fftshift(np.fft.fft(y_kaiser))

    out_freq_r = (np.arange(len(y_remez)) / len(y_remez) - 0.5) * fs_out
    out_spec_r = np.fft.fftshift(np.fft.fft(y_remez))

    axes[1].set_title('Spectrum')
    axes[1].plot(in_freq, np.abs(in_spec), label='Input', alpha=0.5)
    axes[1].plot(out_freq_k, np.abs(out_spec_k), '--', label='Kaiser')
    axes[1].plot(out_freq_r, np.abs(out_spec_r), ':', label='Remez')
    axes[1].legend()
    axes[1].set_xlabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
