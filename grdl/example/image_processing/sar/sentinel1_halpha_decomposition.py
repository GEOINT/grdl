# -*- coding: utf-8 -*-
"""
Sentinel-1 H/Alpha Decomposition — Colorized H-Alpha-Anisotropy RGB composite
from Sentinel-1 IW SLC dual-pol (VV/VH) data.

Reads a single IW swath + both polarizations as a CYX cube, runs the
DualPolHAlpha decomposition, and saves an RGB composite alongside an
optional annotated matplotlib figure.

H/Alpha RGB channel mapping:
  R = Span (total intensity, dB-stretched)
  G = Entropy (scattering randomness, 0=deterministic, 1=random)
  B = Alpha/90 (scattering mechanism angle, 0=surface, 90=double-bounce)

Usage
-----
  python sentinel1_halpha_decomposition.py <safe_dir>
  python sentinel1_halpha_decomposition.py <safe_dir> --swath IW2
  python sentinel1_halpha_decomposition.py <safe_dir> --swath IW1 --chip-size 2048
  python sentinel1_halpha_decomposition.py --help

Dependencies
------------
rasterio, matplotlib (optional, for display)

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
2026-04-16
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
from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader
from grdl.image_processing.decomposition.dual_pol_halpha import DualPolHAlpha


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'H/Alpha decomposition RGB from Sentinel-1 IW SLC dual-pol data.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'filepath',
        help='Path to a Sentinel-1 .SAFE directory.',
    )
    parser.add_argument(
        '--swath', default='IW1',
        help='IW swath to process (IW1, IW2, or IW3).',
    )
    parser.add_argument(
        '--co-pol', default='VV', dest='co_pol',
        help='Co-polarization channel (VV or HH).',
    )
    parser.add_argument(
        '--cross-pol', default='VH', dest='cross_pol',
        help='Cross-polarization channel (VH or HV).',
    )
    parser.add_argument(
        '--window-size', type=int, default=7, dest='window_size',
        help='Spatial averaging window for the coherency matrix (odd, >= 3).',
    )
    parser.add_argument(
        '--chip-size', type=int, default=None,
        help='Read only a centre chip of this many rows x cols. '
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
    safe_dir = Path(args.filepath)
    if not safe_dir.exists():
        parser.error(f"Path not found: {safe_dir}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pols = [args.co_pol.upper(), args.cross_pol.upper()]

    # ------------------------------------------------------------------
    # 1. Open reader in multi-pol CYX mode
    # ------------------------------------------------------------------
    print(f"Opening {safe_dir.name}  swath={args.swath}  pols={pols} ...")
    reader = Sentinel1SLCReader(
        safe_dir,
        swath=args.swath,
        polarizations=pols,        # returns CYX cube: (2, rows, cols)
    )

    print(f"  Available polarizations : {reader.metadata.channel_metadata and [cm.polarization for cm in reader.metadata.channel_metadata]}")
    print(f"  Image size              : {reader.metadata.rows} × {reader.metadata.cols} px")
    print(f"  Bursts                  : {reader.metadata.num_bursts}")
    print(f"  axis_order              : {reader.metadata.axis_order}")

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
        cube = reader.read_chip(r0, r1, c0, c1)   # (2, chip_rows, chip_cols)
    else:
        print("Reading full scene ...")
        cube = reader.read_full()                  # (2, rows, cols)

    metadata = reader.metadata
    reader.close()

    print(f"  Cube shape: {cube.shape}  dtype: {cube.dtype}")

    # ------------------------------------------------------------------
    # 3. H/Alpha decomposition
    # ------------------------------------------------------------------
    print(f"Running DualPolHAlpha (window_size={args.window_size}) ...")
    ha = DualPolHAlpha(window_size=args.window_size)

    # Channel 0 = co-pol (VV), channel 1 = cross-pol (VH)
    components, _ = ha.execute(metadata, cube)
    # components: dict with keys 'entropy', 'alpha', 'anisotropy', 'span'

    print(f"  Entropy   range : [{components['entropy'].min():.3f}, {components['entropy'].max():.3f}]")
    print(f"  Alpha     range : [{components['alpha'].min():.1f}°, {components['alpha'].max():.1f}°]")
    print(f"  Anisotropy range: [{components['anisotropy'].min():.3f}, {components['anisotropy'].max():.3f}]")

    # ------------------------------------------------------------------
    # 4. Build RGB composite  (R=span_dB, G=entropy, B=alpha/90)
    # ------------------------------------------------------------------
    rgb_float, rgb_meta = ha.to_rgb(components)
    # rgb_float: (3, rows, cols) float32 in [0, 1]
    rgb = (np.clip(rgb_float.transpose(1, 2, 0), 0.0, 1.0) * 255).astype(np.uint8)
    print(f"  RGB shape : {rgb.shape}")

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    # Use SAFE name without extension as stem
    stem = safe_dir.stem.replace('.SAFE', '') + f'_{args.swath}'

    # Save raw float32 components as .npy
    stack = np.stack(
        [components['entropy'],
         components['alpha'] / 90.0,
         components['anisotropy'],
         components['span'].astype(np.float32)],
        axis=0,
    ).astype(np.float32)
    npy_path = out_dir / f'{stem}_halpha.npy'
    np.save(str(npy_path), stack)
    print(f"Saved H/Alpha stack → {npy_path}")

    if _HAS_MPL and not args.no_plot:
        # ------ annotated 5-panel figure ------
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle(
            f'H/Alpha Decomposition — {safe_dir.name}\n'
            f'swath={args.swath}  co={args.co_pol}  cross={args.cross_pol}'
            f'  window={args.window_size}px',
            fontsize=10,
        )

        # Span (dB)
        span_db = 10.0 * np.log10(
            np.maximum(components['span'], np.finfo(float).tiny)
        )
        p2, p98 = np.percentile(span_db, [2, 98])
        axes[0].imshow(
            np.clip((span_db - p2) / (p98 - p2 + 1e-12), 0, 1),
            cmap='gray', vmin=0, vmax=1,
        )
        axes[0].set_title('Span (dB)')
        axes[0].axis('off')

        # Entropy
        axes[1].imshow(components['entropy'], cmap='hot', vmin=0, vmax=1)
        axes[1].set_title('Entropy H')
        axes[1].axis('off')

        # Alpha
        axes[2].imshow(components['alpha'], cmap='jet', vmin=0, vmax=90)
        axes[2].set_title('Alpha α (°)')
        axes[2].axis('off')

        # Anisotropy
        axes[3].imshow(components['anisotropy'], cmap='cool', vmin=0, vmax=1)
        axes[3].set_title('Anisotropy A')
        axes[3].axis('off')

        # H/Alpha RGB
        axes[4].imshow(rgb)
        axes[4].set_title('H/Alpha RGB  R=Span G=H B=α/90')
        axes[4].axis('off')

        plt.tight_layout()
        fig_path = out_dir / f'{stem}_halpha_rgb.png'
        fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved figure       → {fig_path}")

        # ------ bare RGB PNG ------
        fig2, ax2 = plt.subplots(figsize=(10, 10))
        ax2.imshow(rgb)
        ax2.axis('off')
        rgb_path = out_dir / f'{stem}_halpha_rgb_clean.png'
        fig2.savefig(str(rgb_path), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig2)
        print(f"Saved RGB image    → {rgb_path}")
    else:
        if not _HAS_MPL:
            print("matplotlib not available — skipping PNG output.")

    print("Done.")


if __name__ == '__main__':
    main()
