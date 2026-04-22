# -*- coding: utf-8 -*-
"""
NISAR Pauli Decomposition — Colorized Pauli RGB from NISAR RSLC quad-pol data.

Reads a NISAR L1 RSLC HDF5 file (frequency A, quad-pol HH/HV/VH/VV),
runs the Pauli decomposition, and saves a gamma-corrected RGB image
alongside an optional matplotlib figure.

Pauli basis mapping:
  R = |S_HH - S_VV| / sqrt(2)    double-bounce
  G = |2 * S_HV| / sqrt(2)       volume / random
  B = |S_HH + S_VV| / sqrt(2)    surface (odd-bounce)

Usage
-----
  python nisar_pauli_decomposition.py <nisar_h5_file>
  python nisar_pauli_decomposition.py <nisar_h5_file> --chip-size 2048
  python nisar_pauli_decomposition.py <nisar_h5_file> --gamma 0.5 --out-dir ./output
  python nisar_pauli_decomposition.py --help

Dependencies
------------
h5py, matplotlib (optional, for display)

Author
------
Viplob Banerjee / Duane Smalley
geoint.org

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-04-15
"""

# Standard library
import argparse
import sys
from pathlib import Path

# Third-party
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# GRDL
from grdl.IO.sar.nisar import NISARReader
from grdl.image_processing.decomposition.pauli import PauliDecomposition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_pauli_rgb(rgb_float: np.ndarray) -> np.ndarray:
    """Convert (3, rows, cols) float32 [0,1] to (rows, cols, 3) uint8."""
    return (np.clip(rgb_float.transpose(1, 2, 0), 0.0, 1.0) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Pauli decomposition RGB from NISAR RSLC quad-pol data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'filepath',
        help='Path to a NISAR L1 RSLC HDF5 file.',
    )
    parser.add_argument(
        '--frequency', default='A',
        help='Frequency sub-band to read (A or B).',
    )
    parser.add_argument(
        '--chip-size', type=int, default=None,
        help='Read only a centre chip of this many rows×cols. '
             'Reads the full scene if omitted.',
    )
    parser.add_argument(
        '--out-dir', type=Path, default=Path('.'),
        help='Directory to write output images.',
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Skip generating the matplotlib figure.',
    )

    args = parser.parse_args(argv)
    filepath = Path(args.filepath)
    if not filepath.exists():
        parser.error(f"File not found: {filepath}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Open reader in multi-pol CYX mode
    # ------------------------------------------------------------------
    print(f"Opening {filepath.name}  frequency={args.frequency} ...")
    reader = NISARReader(
        filepath,
        frequency=args.frequency,
        polarizations='all',          # builds CYX cube for all available pols
    )

    pols = [cm.polarization for cm in reader.metadata.channel_metadata]
    print(f"  Available polarizations : {pols}")
    print(f"  Image size              : {reader.metadata.rows} × {reader.metadata.cols} px")
    print(f"  axis_order              : {reader.metadata.axis_order}")

    # Require at least HH, HV, VV for Pauli
    required = {'HH', 'HV', 'VV'}
    if not required.issubset(set(pols)):
        sys.exit(
            f"Pauli decomposition requires HH, HV, VV.  "
            f"Only found: {pols}"
        )

    # ------------------------------------------------------------------
    # 2. Read data (chip or full scene)
    # ------------------------------------------------------------------
    rows, cols = reader.metadata.rows, reader.metadata.cols

    if args.chip_size is not None:
        cs = args.chip_size
        r0 = max(0, rows // 2 - cs // 2)
        c0 = max(0, cols // 2 - cs // 2)
        r1 = min(rows, r0 + cs)
        c1 = min(cols, c0 + cs)
        print(f"Reading chip [{r0}:{r1}, {c0}:{c1}] ...")
        cube = reader.read_chip(r0, r1, c0, c1)   # (C, chip_rows, chip_cols)
    else:
        print("Reading full scene (this may take a moment for large files) ...")
        cube = reader.read_full()                  # (C, rows, cols)

    metadata = reader.metadata
    reader.close()

    print(f"  Cube shape: {cube.shape}  dtype: {cube.dtype}")

    # ------------------------------------------------------------------
    # 3. Pauli decomposition
    # ------------------------------------------------------------------
    print("Running Pauli decomposition ...")
    pauli = PauliDecomposition()

    # Extract individual polarization channels from the CYX cube
    pol_names = [cm.polarization.upper() for cm in metadata.channel_metadata]
    def _get(pol):
        idx = pol_names.index(pol)
        return cube[idx]

    shh = _get('HH')
    shv = _get('HV')
    svh = _get('VH') if 'VH' in pol_names else shv   # reciprocity fallback
    svv = _get('VV')

    components = pauli.decompose(shh, shv, svh, svv)
    # components: dict with keys 'surface', 'double_bounce', 'volume'
    print(f"  Component shapes: { {k: v.shape for k, v in components.items()} }")

    # Build [0,1] float32 RGB using Pauli's built-in stretch
    # Channel mapping: R=double-bounce, G=volume, B=surface
    rgb_float = pauli.to_rgb(components, representation='db',
                             percentile_low=2.0, percentile_high=98.0)
    # rgb_float: (3, rows, cols) float32 in [0, 1]
    print(f"  RGB float shape : {rgb_float.shape}")

    # ------------------------------------------------------------------
    # 4. Build uint8 RGB for saving / display
    # ------------------------------------------------------------------
    rgb = (np.clip(rgb_float.transpose(1, 2, 0), 0.0, 1.0) * 255).astype(np.uint8)
    print(f"  RGB uint8 shape : {rgb.shape}")

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    stem = filepath.stem

    # Save raw float32 Pauli components as .npy for downstream use
    pauli_stack = np.stack(
        [np.abs(components['double_bounce']),
         np.abs(components['volume']),
         np.abs(components['surface'])],
        axis=0,
    ).astype(np.float32)
    npy_path = out_dir / f'{stem}_pauli.npy'
    np.save(str(npy_path), pauli_stack)
    print(f"Saved Pauli stack → {npy_path}")

    # Save RGB as PNG via matplotlib (no PIL dependency)
    if _HAS_MPL and not args.no_plot:
        component_labels = [
            ('double_bounce', 'Double-bounce (R)', 'Reds'),
            ('volume',        'Volume (G)',         'Greens'),
            ('surface',       'Surface (B)',        'Blues'),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(
            f'Pauli Decomposition — {filepath.name}\n'
            f'freq={args.frequency}',
            fontsize=11,
        )

        for ax, (key, title, cmap), ch in zip(axes[:3], component_labels, range(3)):
            ax.imshow(rgb_float[ch], cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis('off')

        axes[3].imshow(rgb)
        axes[3].set_title('Pauli RGB composite')
        axes[3].axis('off')

        plt.tight_layout()
        fig_path = out_dir / f'{stem}_pauli_rgb.png'
        fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved figure       → {fig_path}")

        # Bare RGB PNG
        fig2, ax2 = plt.subplots(figsize=(10, 10))
        ax2.imshow(rgb)
        ax2.axis('off')
        rgb_path = out_dir / f'{stem}_pauli_rgb_clean.png'
        fig2.savefig(str(rgb_path), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig2)
        print(f"Saved RGB image    → {rgb_path}")
    else:
        if not _HAS_MPL:
            print("matplotlib not available — skipping PNG output.")

    print("Done.")


if __name__ == '__main__':
    main()
