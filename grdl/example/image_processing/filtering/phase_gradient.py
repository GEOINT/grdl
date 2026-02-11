# -*- coding: utf-8 -*-
"""
Phase Gradient Example - Visualize phase gradients from complex SAR imagery.

Loads a SICD file, extracts a center chip, and applies ``ComplexLeeFilter``
to despeckle the complex data while preserving interferometric phase. The
despeckled complex chip then feeds both ``PhaseGradientFilter`` for phase
gradient estimation and ``StdDevFilter`` for texture analysis.

Demonstrates GRDL integration:
  - ``grdl.IO.SICDReader`` for metadata + pixel access
  - ``grdl.data_prep.ChipExtractor`` for chip planning (index-only)
  - ``grdl.image_processing.filters.ComplexLeeFilter`` for complex despeckle
  - ``grdl.image_processing.filters.PhaseGradientFilter`` for phase gradient
  - ``grdl.image_processing.filters.StdDevFilter`` for local texture
  - ``grdl.image_processing.intensity`` for dB conversion and contrast stretch

Usage:
  python phase_gradient.py <sicd_file>
  python phase_gradient.py <sicd_file> --chip-size 2048
  python phase_gradient.py <sicd_file> --kernel-size 7
  python phase_gradient.py --help

Dependencies
------------
matplotlib
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
2026-02-11

Modified
--------
2026-02-12
"""

# Standard library
import argparse
import sys
from pathlib import Path

# Third-party
import matplotlib
for _backend in ("QtAgg", "TkAgg", "MacOSX", "Agg"):
    try:
        matplotlib.use(_backend)
        break
    except ImportError:
        continue

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

# GRDL
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from grdl.IO import SICDReader
from grdl.data_prep import ChipExtractor
from grdl.image_processing.filters import ComplexLeeFilter, PhaseGradientFilter, StdDevFilter
from grdl.image_processing.intensity import ToDecibels, PercentileStretch


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compute and display phase gradients from a SICD "
                    "complex SAR image.",
    )
    parser.add_argument(
        "filepath",
        type=Path,
        help="Path to the SICD file (NITF or other SICD container).",
    )
    parser.add_argument(
        "--chip-size",
        type=int,
        default=2048*4,
        help="Side length of the center chip in pixels (default: 2048).",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=5,
        help="Smoothing window side length for phase gradient (odd, >= 3). "
             "Default: 5.",
    )
    parser.add_argument(
        "--plow",
        type=float,
        default=2.0,
        help="Lower percentile for amplitude contrast stretch (default: 2).",
    )
    parser.add_argument(
        "--phigh",
        type=float,
        default=98.0,
        help="Upper percentile for amplitude contrast stretch (default: 98).",
    )
    parser.add_argument(
        "--cmap-phase",
        type=str,
        default="RdBu_r",
        help="Colormap for phase gradient panels (default: RdBu_r).",
    )
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────


def phase_gradient_example(
    filepath: Path,
    chip_size: int = 2048*4,
    kernel_size: int = 5,
    plow: float = 2.0,
    phigh: float = 98.0,
    cmap_phase: str = "RdBu_r",
) -> None:
    """Load a SICD, extract a chip, compute phase gradients, and display.

    Parameters
    ----------
    filepath : Path
        Path to the SICD file.
    chip_size : int
        Side length of the center chip.
    kernel_size : int
        Smoothing window for the phase gradient filter.
    plow : float
        Lower percentile for amplitude contrast stretch.
    phigh : float
        Upper percentile for amplitude contrast stretch.
    cmap_phase : str
        Matplotlib colormap for phase gradient panels.
    """
    print(f"Opening: {filepath}")

    # ── Processors ────────────────────────────────────────────────────
    to_db = ToDecibels()
    stretch = PercentileStretch(plow=plow, phigh=phigh)
    clee = ComplexLeeFilter(kernel_size=7)
    pgf_row = PhaseGradientFilter(kernel_size=kernel_size, direction='row')
    pgf_col = PhaseGradientFilter(kernel_size=kernel_size, direction='col')
    pgf_mag = PhaseGradientFilter(kernel_size=kernel_size, direction='magnitude')
    std_filter = StdDevFilter(kernel_size=kernel_size)

    with SICDReader(filepath) as reader:
        meta = reader.metadata
        rows, cols = reader.get_shape()
        print(f"  Image size: {rows} x {cols}")

        # Plan center chip (index-only)
        extractor = ChipExtractor(nrows=rows, ncols=cols)
        region = extractor.chip_at_point(
            rows // 2, cols // 2,
            row_width=chip_size, col_width=chip_size,
        )

        chip_h = region.row_end - region.row_start
        chip_w = region.col_end - region.col_start
        print(f"  Center chip: [{region.row_start}:{region.row_end}, "
              f"{region.col_start}:{region.col_end}] ({chip_h} x {chip_w})")

        chip = reader.read_chip(
            region.row_start, region.row_end,
            region.col_start, region.col_end,
        )
        print(f"  Chip shape: {chip.shape}, dtype: {chip.dtype}")

    # ── Complex Lee despeckle (preserves phase) ──────────────────────
    print("  Applying ComplexLeeFilter (kernel_size=7)...")
    despeckled = clee.apply(chip)

    # ── Compute phase gradients (on despeckled complex chip) ──────────
    print(f"  Computing phase gradients (kernel_size={kernel_size})...")
    grad_row = pgf_row.apply(despeckled)
    grad_col = pgf_col.apply(despeckled)
    grad_mag = pgf_mag.apply(despeckled)

    print(f"  Row gradient range: [{grad_row.min():.4f}, {grad_row.max():.4f}] rad/px")
    print(f"  Col gradient range: [{grad_col.min():.4f}, {grad_col.max():.4f}] rad/px")
    print(f"  Magnitude range:    [{grad_mag.min():.4f}, {grad_mag.max():.4f}] rad/px")

    # ── Local std dev (texture) on despeckled dB amplitude ─────────────
    intensity = despeckled.real ** 2 + despeckled.imag ** 2
    amp_db = to_db.apply(intensity)
    print(f"  Computing local std dev (kernel_size={kernel_size})...")
    local_std = std_filter.apply(amp_db)
    print(f"  Std dev range: [{local_std.min():.4f}, {local_std.max():.4f}] dB")

    # ── Amplitude reference ───────────────────────────────────────────
    amplitude = stretch.apply(amp_db)

    # ── Build title ───────────────────────────────────────────────────
    ci = meta.collection_info
    title_parts = [filepath.name]
    if ci is not None and ci.collector_name:
        title_parts.append(ci.collector_name)
    file_title = "  |  ".join(title_parts)

    # ── Symmetric color limits for row/col gradients ──────────────────
    row_lim = np.percentile(np.abs(grad_row), 98)
    col_lim = np.percentile(np.abs(grad_col), 98)

    # ── Display (2 rows x 3 cols) ────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"Phase Gradient & Texture Analysis\n{file_title}", fontsize=12)

    # Row 1: Amplitude | Std Dev | Gradient Magnitude
    axes[0, 0].imshow(amplitude, cmap="gray", aspect="auto",
                      interpolation="nearest", vmin=0, vmax=1)
    axes[0, 0].set_title(f"Amplitude (dB stretch)\n{chip_h} x {chip_w} px")

    std_lim = np.percentile(local_std, 98)
    im_std = axes[0, 1].imshow(local_std, cmap="viridis", aspect="auto",
                               interpolation="nearest",
                               vmin=0, vmax=std_lim)
    axes[0, 1].set_title(f"Local Std Dev (k={kernel_size})\ndB")
    plt.colorbar(im_std, ax=axes[0, 1], fraction=0.046, pad=0.04)

    mag_lim = np.percentile(grad_mag, 98)
    im_mag = axes[0, 2].imshow(grad_mag, cmap="inferno", aspect="auto",
                               interpolation="nearest",
                               vmin=0, vmax=mag_lim)
    axes[0, 2].set_title("Gradient Magnitude\nrad/pixel")
    plt.colorbar(im_mag, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Row 2: Row Gradient | Col Gradient | (empty)
    im_row = axes[1, 0].imshow(grad_row, cmap=cmap_phase, aspect="auto",
                               interpolation="nearest",
                               vmin=-row_lim, vmax=row_lim)
    axes[1, 0].set_title("Row Gradient (azimuth)\nrad/pixel")
    plt.colorbar(im_row, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_col = axes[1, 1].imshow(grad_col, cmap=cmap_phase, aspect="auto",
                               interpolation="nearest",
                               vmin=-col_lim, vmax=col_lim)
    axes[1, 1].set_title("Column Gradient (range)\nrad/pixel")
    plt.colorbar(im_col, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Binary detection: phase gradient magnitude + local std dev > 9
    combined = grad_mag + local_std
    binary = (combined > 20).astype(np.uint8)
    pct = 100.0 * np.mean(binary)
    axes[1, 2].imshow(binary, cmap="gray", aspect="auto",
                      interpolation="nearest", vmin=0, vmax=1)
    axes[1, 2].set_title(f"Binary (grad_mag + std_dev > 9)\n{pct:.1f}% pixels")

    for ax in axes.flat:
        if ax.get_visible() and ax.axison:
            ax.set_xlabel("Column (range)")
            ax.set_ylabel("Row (azimuth)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    phase_gradient_example(
        args.filepath,
        chip_size=args.chip_size,
        kernel_size=args.kernel_size,
        plow=args.plow,
        phigh=args.phigh,
        cmap_phase=args.cmap_phase,
    )
