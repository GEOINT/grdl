# -*- coding: utf-8 -*-
"""
CPHD Metadata Dump - Extract and save all metadata from a CPHD file.

Reads a CPHD file via ``CPHDReader`` and writes a comprehensive text
dump of all metadata fields, PVP summary statistics, and sample PVP
array values to a text file for inspection.

Dependencies
------------
sarkit or sarpy

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
2026-02-13
"""

# Standard library
import argparse
import sys
from pathlib import Path
from typing import Optional

# Third-party
import numpy as np
from numpy.linalg import norm

# GRDL
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from grdl.IO.sar import CPHDReader


def dump_metadata(filepath: Path, output: Optional[Path] = None) -> str:
    """Read CPHD metadata and produce a formatted text dump.

    Parameters
    ----------
    filepath : Path
        Path to the CPHD file.
    output : Path, optional
        Path to write the dump. If None, writes to
        ``<cphd_basename>_metadata.txt`` alongside the CPHD file.

    Returns
    -------
    str
        The formatted metadata text.
    """
    if output is None:
        output = filepath.with_suffix('.metadata.txt')

    with CPHDReader(filepath) as reader:
        meta = reader.metadata
        pvp = meta.pvp
        lines = []

        def _line(text: str = '') -> None:
            lines.append(text)

        def _header(title: str) -> None:
            _line()
            _line('=' * 70)
            _line(f'  {title}')
            _line('=' * 70)

        def _field(label: str, value, width: int = 30) -> None:
            _line(f'  {label:<{width}} : {value}')

        # ── Title ──
        _line('CPHD METADATA DUMP')
        _line(f'File: {filepath}')
        _line(f'Backend: {reader.backend}')

        # ── Collection Info ──
        _header('COLLECTION INFO')
        ci = meta.collection_info
        if ci is not None:
            _field('Collector Name', ci.collector_name)
            _field('Core Name', ci.core_name)
            _field('Classification', ci.classification)
            _field('Collect Type', ci.collect_type)
            _field('Radar Mode', ci.radar_mode)
            _field('Radar Mode ID', ci.radar_mode_id)
        else:
            _line('  (no collection info)')

        # ── Global Params ──
        _header('GLOBAL PARAMETERS')
        gp = meta.global_params
        if gp is not None:
            _field('Domain Type', gp.domain_type)
            _field('Phase SGN', f'{gp.phase_sgn:+d}')
            _field('FX Band Min (Hz)', gp.fx_band_min)
            _field('FX Band Max (Hz)', gp.fx_band_max)
            _field('Bandwidth (Hz)', gp.bandwidth)
            _field('Center Frequency (Hz)', gp.center_frequency)
            if gp.center_frequency:
                _field('Center Frequency (GHz)',
                       f'{gp.center_frequency / 1e9:.6f}')
                _field('Wavelength (m)',
                       f'{299792458.0 / gp.center_frequency:.6f}')
            _field('TOA Swath Min (s)', gp.toa_swath_min)
            _field('TOA Swath Max (s)', gp.toa_swath_max)
        else:
            _line('  (no global params)')

        # ── Channel Info ──
        _header('CHANNELS')
        _field('Num Channels', meta.num_channels)
        for i, ch in enumerate(meta.channels):
            _line(f'  --- Channel {i} ---')
            _field('  Identifier', ch.identifier)
            _field('  Num Vectors (pulses)', ch.num_vectors)
            _field('  Num Samples', ch.num_samples)
            _field('  Signal Byte Offset', ch.signal_array_byte_offset)

        # ── Tx Waveform ──
        _header('TRANSMIT WAVEFORM')
        tx = meta.tx_waveform
        if tx is not None:
            _field('Identifier', tx.identifier)
            _field('LFM Rate (Hz/s)', tx.lfm_rate)
            _field('Pulse Length (s)', tx.pulse_length)
            if tx.lfm_rate and tx.pulse_length:
                _field('Chirp Bandwidth (Hz)',
                       f'{tx.lfm_rate * tx.pulse_length:.2f}')
                _field('Chirp Bandwidth (MHz)',
                       f'{tx.lfm_rate * tx.pulse_length / 1e6:.4f}')
        else:
            _line('  (no tx waveform)')

        # ── Rcv Parameters ──
        _header('RECEIVE PARAMETERS')
        rcv = meta.rcv_parameters
        if rcv is not None:
            _field('Identifier', rcv.identifier)
            _field('Window Length (s)', rcv.window_length)
            _field('Sample Rate (Hz)', rcv.sample_rate)
            if rcv.sample_rate:
                _field('Sample Rate (MHz)',
                       f'{rcv.sample_rate / 1e6:.4f}')
        else:
            _line('  (no rcv parameters)')

        # ── PVP Summary ──
        _header('PER-VECTOR PARAMETERS (PVP) SUMMARY')
        if pvp is not None:
            _field('Num Vectors', pvp.num_vectors)
            _field('Midpoint Time (s)', f'{pvp.midpoint_time:.9f}')
            _field('First Valid Pulse', pvp.first_valid_pulse)

            # Time
            if pvp.tx_time is not None:
                _line()
                _line('  --- Timing ---')
                _field('TX Time [min]', f'{pvp.tx_time.min():.9f}')
                _field('TX Time [max]', f'{pvp.tx_time.max():.9f}')
                _field('TX Time Span (s)',
                       f'{pvp.tx_time.max() - pvp.tx_time.min():.9f}')
            if pvp.rcv_time is not None:
                _field('RCV Time [min]', f'{pvp.rcv_time.min():.9f}')
                _field('RCV Time [max]', f'{pvp.rcv_time.max():.9f}')
                _field('RCV Time Span (s)',
                       f'{pvp.rcv_time.max() - pvp.rcv_time.min():.9f}')
            if pvp.tx_time is not None and pvp.rcv_time is not None:
                mid_times = 0.5 * (pvp.tx_time + pvp.rcv_time)
                dt = np.diff(mid_times)
                _field('Avg Pulse Interval (s)', f'{np.mean(dt):.9f}')
                _field('Avg PRF (Hz)', f'{1.0 / np.mean(dt):.2f}')

            # SRP Position
            if pvp.srp_pos is not None:
                _line()
                _line('  --- SRP Position (ECF, meters) ---')
                _field('SRP[0]',
                       f'({pvp.srp_pos[0, 0]:.3f}, '
                       f'{pvp.srp_pos[0, 1]:.3f}, '
                       f'{pvp.srp_pos[0, 2]:.3f})')
                _field('SRP[-1]',
                       f'({pvp.srp_pos[-1, 0]:.3f}, '
                       f'{pvp.srp_pos[-1, 1]:.3f}, '
                       f'{pvp.srp_pos[-1, 2]:.3f})')
                drift = norm(pvp.srp_pos[-1] - pvp.srp_pos[0])
                _field('SRP Total Drift (m)', f'{drift:.3f}')
                # Per-pulse SRP drift
                per_pulse_drift = norm(
                    np.diff(pvp.srp_pos, axis=0), axis=1,
                )
                _field('SRP Drift/Pulse [min] (m)',
                       f'{per_pulse_drift.min():.6f}')
                _field('SRP Drift/Pulse [max] (m)',
                       f'{per_pulse_drift.max():.6f}')
                _field('SRP Drift/Pulse [mean] (m)',
                       f'{per_pulse_drift.mean():.6f}')
                # Determine stripmap vs spotlight
                if drift > 1.0:
                    _field('MODE DETECTION',
                           'STRIPMAP (SRP drifts significantly)')
                else:
                    _field('MODE DETECTION',
                           'SPOTLIGHT (SRP approximately fixed)')

            # ARP (Platform) Position
            if pvp.tx_pos is not None:
                _line()
                _line('  --- ARP Position (ECF, meters) ---')
                arp = pvp.tx_pos  # monostatic: tx_pos == arp
                _field('ARP[0]',
                       f'({arp[0, 0]:.3f}, '
                       f'{arp[0, 1]:.3f}, '
                       f'{arp[0, 2]:.3f})')
                _field('ARP[-1]',
                       f'({arp[-1, 0]:.3f}, '
                       f'{arp[-1, 1]:.3f}, '
                       f'{arp[-1, 2]:.3f})')
                arp_mag = norm(arp, axis=1)
                # Approximate altitude (distance from earth center - ~6371km)
                _field('ARP |R| [min] (m)',
                       f'{arp_mag.min():.3f}')
                _field('ARP |R| [max] (m)',
                       f'{arp_mag.max():.3f}')
                track_length = norm(arp[-1] - arp[0])
                _field('Platform Track Length (m)',
                       f'{track_length:.3f}')

            # Velocity
            if pvp.tx_vel is not None:
                _line()
                _line('  --- Platform Velocity (ECF, m/s) ---')
                vel = pvp.tx_vel
                vel_mag = norm(vel, axis=1)
                _field('Speed [min] (m/s)',
                       f'{vel_mag.min():.3f}')
                _field('Speed [max] (m/s)',
                       f'{vel_mag.max():.3f}')
                _field('Speed [mean] (m/s)',
                       f'{vel_mag.mean():.3f}')

            # Frequency
            _line()
            _line('  --- Frequency Parameters ---')
            if pvp.fx1 is not None:
                _field('FX1 [min] (Hz)', f'{pvp.fx1.min():.2f}')
                _field('FX1 [max] (Hz)', f'{pvp.fx1.max():.2f}')
            if pvp.fx2 is not None:
                _field('FX2 [min] (Hz)', f'{pvp.fx2.min():.2f}')
                _field('FX2 [max] (Hz)', f'{pvp.fx2.max():.2f}')
            if pvp.fx1 is not None and pvp.fx2 is not None:
                bw = pvp.fx2 - pvp.fx1
                _field('Per-Pulse BW [min] (Hz)',
                       f'{bw.min():.2f}')
                _field('Per-Pulse BW [max] (Hz)',
                       f'{bw.max():.2f}')
                _field('Per-Pulse BW [mean] (MHz)',
                       f'{bw.mean() / 1e6:.4f}')
            if pvp.sc0 is not None:
                _field('SC0 [min] (Hz)', f'{pvp.sc0.min():.2f}')
                _field('SC0 [max] (Hz)', f'{pvp.sc0.max():.2f}')
            if pvp.scss is not None:
                _field('SCSS [min] (Hz)', f'{pvp.scss.min():.6f}')
                _field('SCSS [max] (Hz)', f'{pvp.scss.max():.6f}')

            # Signal validity
            if pvp.signal is not None:
                _line()
                _line('  --- Signal Validity ---')
                n_valid = int(np.sum(pvp.signal > 0))
                _field('Valid Pulses',
                       f'{n_valid} / {pvp.num_vectors}')
                _field('Signal [min]', f'{pvp.signal.min()}')
                _field('Signal [max]', f'{pvp.signal.max()}')

            # Doppler
            if pvp.a_fdop is not None:
                _line()
                _line('  --- Doppler ---')
                _field('aFDOP [min]', f'{pvp.a_fdop.min():.6f}')
                _field('aFDOP [max]', f'{pvp.a_fdop.max():.6f}')
            if pvp.a_frr1 is not None:
                _field('aFRR1 [min]', f'{pvp.a_frr1.min():.6f}')
                _field('aFRR1 [max]', f'{pvp.a_frr1.max():.6f}')
            if pvp.a_frr2 is not None:
                _field('aFRR2 [min]', f'{pvp.a_frr2.min():.6f}')
                _field('aFRR2 [max]', f'{pvp.a_frr2.max():.6f}')

            # ── Raw PVP Samples ──
            _header('PVP ARRAY SAMPLES (first 5, last 5)')
            n = pvp.num_vectors
            indices = list(range(min(5, n))) + list(range(max(0, n - 5), n))
            indices = sorted(set(indices))

            pvp_fields = [
                ('tx_time', '1D'),
                ('rcv_time', '1D'),
                ('fx1', '1D'),
                ('fx2', '1D'),
                ('sc0', '1D'),
                ('scss', '1D'),
                ('signal', '1D'),
                ('a_fdop', '1D'),
                ('a_frr1', '1D'),
                ('a_frr2', '1D'),
                ('tx_pos', '2D'),
                ('tx_vel', '2D'),
                ('rcv_pos', '2D'),
                ('rcv_vel', '2D'),
                ('srp_pos', '2D'),
            ]

            for field_name, dim in pvp_fields:
                arr = getattr(pvp, field_name, None)
                if arr is None:
                    continue
                _line()
                _line(f'  --- {field_name} (shape {arr.shape}, '
                      f'dtype {arr.dtype}) ---')
                for idx in indices:
                    if dim == '1D':
                        _line(f'    [{idx:6d}] {arr[idx]}')
                    else:
                        _line(f'    [{idx:6d}] {arr[idx]}')

        else:
            _line('  (no PVP data)')

        # ── Slant Range ──
        if pvp is not None and pvp.srp_pos is not None and pvp.tx_pos is not None:
            _header('DERIVED GEOMETRY')
            r_srp_arp = norm(pvp.tx_pos - pvp.srp_pos, axis=1)
            _field('Slant Range [min] (m)', f'{r_srp_arp.min():.3f}')
            _field('Slant Range [max] (m)', f'{r_srp_arp.max():.3f}')
            _field('Slant Range [mean] (m)', f'{r_srp_arp.mean():.3f}')
            _field('Slant Range [mean] (km)',
                   f'{r_srp_arp.mean() / 1000:.3f}')

    text = '\n'.join(lines)
    output.write_text(text, encoding='utf-8')
    print(f'Metadata dumped to: {output}')
    return text


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Dump CPHD metadata to a text file.',
    )
    parser.add_argument(
        'filepath',
        type=Path,
        help='Path to the CPHD file.',
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output text file path. Default: <input>.metadata.txt',
    )
    args = parser.parse_args()
    dump_metadata(args.filepath, args.output)


if __name__ == '__main__':
    main()
