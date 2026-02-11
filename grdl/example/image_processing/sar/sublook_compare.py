# -*- coding: utf-8 -*-
"""
Sublook Comparison - Split a SICD image into 3 sub-aperture looks.

Loads a SICD file, uses ``ChipExtractor`` from ``data_prep`` to plan a
center chip, reads the chip via ``SICDReader``, decomposes it into
3 azimuth sub-apertures with zero overlap, and displays the
full-resolution chip alongside the three sub-looks for visual
comparison.

Demonstrates full GRDL integration:
  - ``grdl.IO.SICDReader`` for metadata + pixel access
  - ``grdl.data_prep.ChipExtractor`` for chip planning (index-only)
  - ``grdl.image_processing.sar.SublookDecomposition`` for processing

Usage:
  python sublook_compare.py <sicd_file>
  python sublook_compare.py <sicd_file> --chip-size 2048
  python sublook_compare.py <sicd_file> --plow 1 --phigh 99
  python sublook_compare.py --help

Dependencies
------------
matplotlib
sarkit or sarpy

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
2026-02-10

Modified
--------
2026-02-11
"""

# Standard library
import argparse
import sys
from pathlib import Path

# Third-party
import numpy as np

# Matplotlib -- set backend before importing pyplot
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt  # noqa: E402

# GRDL
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from grdl.IO import SICDReader
from grdl.data_prep import ChipExtractor
from grdl.image_processing.sar import SublookDecomposition


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Split a SICD image into 3 sub-aperture looks and "
                    "display side-by-side.",
    )
    parser.add_argument(
        "filepath",
        type=Path,
        help="Path to the SICD file (NITF or other SICD container).",
    )
    parser.add_argument(
        "--chip-size",
        type=int,
        default=5000,
        help="Side length of the center chip in pixels (default: 5000).",
    )
    parser.add_argument(
        "--plow",
        type=float,
        default=2.0,
        help="Lower percentile for contrast stretch (default: 2).",
    )
    parser.add_argument(
        "--phigh",
        type=float,
        default=98.0,
        help="Upper percentile for contrast stretch (default: 98).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="gray",
        help="Matplotlib colormap (default: gray).",
    )
    return parser.parse_args()


def percentile_stretch(
    mag: np.ndarray, plow: float, phigh: float
) -> np.ndarray:
    """Stretch magnitude to [0, 1] using percentile bounds.

    Parameters
    ----------
    mag : np.ndarray
        Magnitude array.
    plow : float
        Lower percentile.
    phigh : float
        Upper percentile.

    Returns
    -------
    np.ndarray
        Stretched array clipped to [0, 1], dtype float32.
    """
    vmin = np.percentile(mag, plow)
    vmax = np.percentile(mag, phigh)
    if vmax - vmin < np.finfo(np.float32).eps:
        return np.zeros_like(mag, dtype=np.float32)
    out = (mag - vmin) / (vmax - vmin)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def sublook_compare(
    filepath: Path,
    chip_size: int = 5000,
    plow: float = 2.0,
    phigh: float = 98.0,
    cmap: str = "gray",
) -> None:
    """Load a SICD, extract center chip, decompose, and plot.

    Parameters
    ----------
    filepath : Path
        Path to the SICD file.
    chip_size : int
        Side length of the center chip.
    plow : float
        Lower percentile for contrast stretch.
    phigh : float
        Upper percentile for contrast stretch.
    cmap : str
        Matplotlib colormap name.
    """
    print(f"Opening: {filepath}")

    with SICDReader(filepath) as reader:
        meta = reader.metadata
        rows, cols = reader.get_shape()
        print(f"  Image size: {rows} x {cols}")

        # Use ChipExtractor to plan a center chip (index-only, no pixel data)
        extractor = ChipExtractor(nrows=rows, ncols=cols)
        region = extractor.chip_at_point(
            rows // 2, cols // 2,
            row_width=chip_size, col_width=chip_size,
        )
        
        ### verbose IO.
        chip_h = region.row_end - region.row_start
        chip_w = region.col_end - region.col_start
        print(f"  Center chip: [{region.row_start}:{region.row_end}, "
              f"{region.col_start}:{region.col_end}] ({chip_h} x {chip_w})")

        # Read chip using ChipRegion bounds
        chip = reader.read_chip(
            region.row_start, region.row_end,
            region.col_start, region.col_end,
        )
        print(f"  Chip shape: {chip.shape}, dtype: {chip.dtype}")

        # Sublook decomposition
        print("  Decomposing into 3 azimuth sub-apertures (0% overlap)...")
        sublook = SublookDecomposition(
            meta, num_looks=3, dimension='azimuth', overlap=0.0
        )
        looks = sublook.decompose(chip)
        print(f"  Sublook stack: {looks.shape}")

    # ---- Compute magnitudes in dB ----
    chip_db = 20.0 * np.log10(np.abs(chip) + np.finfo(np.float32).tiny)
    look_dbs = [
        20.0 * np.log10(np.abs(looks[i]) + np.finfo(np.float32).tiny)
        for i in range(3)
    ]

    # ---- Stretch ----
    chip_stretched = percentile_stretch(chip_db, plow, phigh)
    look_stretched = [percentile_stretch(db, plow, phigh) for db in look_dbs]

    # ---- Build title ----
    ci = meta.collection_info
    title_parts = [filepath.name]
    if ci is not None and ci.collector_name:
        title_parts.append(ci.collector_name)
    file_title = "  |  ".join(title_parts)

    # ---- Plot ----
    fig = plt.figure(figsize=(18, 12))

    # Top: full-resolution center chip
    ax_full = fig.add_subplot(2, 1, 1)
    ax_full.imshow(chip_stretched, cmap=cmap, aspect="auto",
                   interpolation="nearest", vmin=0, vmax=1)
    ax_full.set_title(f"Full Aperture  ({chip_h} x {chip_w})\n{file_title}",
                      fontsize=11)
    ax_full.set_xlabel("Column (range)")
    ax_full.set_ylabel("Row (azimuth)")

    # Bottom: 3 sub-looks side by side
    look_labels = ["Sub-look 1 (fore)", "Sub-look 2 (mid)",
                    "Sub-look 3 (aft)"]
    for i in range(3):
        ax = fig.add_subplot(2, 3, 4 + i)
        ax.imshow(look_stretched[i], cmap=cmap, aspect="auto",
                  interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(look_labels[i], fontsize=10)
        ax.set_xlabel("Column (range)")
        if i == 0:
            ax.set_ylabel("Row (azimuth)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    sublook_compare(
        args.filepath,
        chip_size=args.chip_size,
        plow=args.plow,
        phigh=args.phigh,
        cmap=args.cmap,
    )
