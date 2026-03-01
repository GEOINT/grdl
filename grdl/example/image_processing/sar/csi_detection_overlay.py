# -*- coding: utf-8 -*-
"""
CSI Detection Overlay — Dominance detections on a CSI composite.

Loads a SICD file, extracts a center chip, computes sub-aperture
dominance-based detections and a CSI (Coherent Shape Index) RGB
composite, then displays the CSI with detection polygon outlines
overlaid in a single figure.

Demonstrates full GRDL integration:
  - ``grdl.IO.SICDReader`` for metadata + chip reading
  - ``grdl.data_prep.ChipExtractor`` for chip planning (index-only)
  - ``grdl.image_processing.sar.SublookDecomposition`` for sub-looks
  - ``grdl.image_processing.sar.dominance.compute_dominance`` for
    aperture dominance feature
  - ``grdl.image_processing.sar.CSIProcessor`` for RGB composite

Usage:
  python csi_detection_overlay.py <sicd_file>
  python csi_detection_overlay.py <sicd_file> --chip-size 8000
  python csi_detection_overlay.py <sicd_file> --sigma 2.5 --num-looks 5
  python csi_detection_overlay.py --help

Dependencies
------------
matplotlib
scipy
scikit-image
sarkit (or sarpy)

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
2026-02-26

Modified
--------
2026-02-26
"""

# Standard library
import argparse
import sys
from pathlib import Path

# Third-party
import numpy as np
from skimage.measure import find_contours, label, regionprops
from skimage.morphology import closing, footprint_rectangle, opening

# GRDL
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from grdl.IO.sar import SICDReader
from grdl.data_prep import ChipExtractor
from grdl.image_processing.sar import CSIProcessor, SublookDecomposition
from grdl.image_processing.sar.dominance import compute_dominance


# ── Argument parser ──────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Overlay sub-aperture dominance detections on a "
                    "CSI composite from a SICD image.",
    )
    p.add_argument("filepath", type=Path, help="Path to SICD file")
    p.add_argument("--chip-size", type=int, default=5000,
                   help="Center chip side length in pixels (default: 5000)")
    p.add_argument("--num-looks", type=int, default=7,
                   help="Number of sub-aperture looks (default: 7)")
    p.add_argument("--dimension", choices=("azimuth", "range"),
                   default="azimuth",
                   help="Frequency axis to split (default: azimuth)")
    p.add_argument("--sigma", type=float, default=3.0,
                   help="Detection threshold in std devs above mean "
                        "(default: 3.0)")
    p.add_argument("--smooth-win", type=int, default=7,
                   help="Spatial smoothing kernel size (default: 7)")
    p.add_argument("--dom-window", type=int, default=3,
                   help="Contiguous look window for dominance (default: 3)")
    p.add_argument("--morph-size", type=int, default=3,
                   help="Morphological kernel size (default: 3)")
    p.add_argument("--normalization", choices=("log", "percentile", "none"),
                   default="log",
                   help="CSI normalization method (default: log)")
    return p.parse_args()


# ── Visualization ────────────────────────────────────────────────────


def plot_csi_detections(
    csi_rgb: np.ndarray,
    labeled: np.ndarray,
    n_detections: int,
    *,
    title: str = "",
) -> None:
    """Display CSI composite with detection contour overlays.

    Parameters
    ----------
    csi_rgb : np.ndarray
        CSI RGB composite, shape ``(rows, cols, 3)``.
    labeled : np.ndarray
        Integer-labeled detection regions from ``skimage.measure.label``.
    n_detections : int
        Number of detected objects.
    title : str
        Figure title annotation (filename, collector, etc.).
    """
    import matplotlib
    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt  # noqa: E402

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(csi_rgb, aspect="auto", interpolation="nearest")

    contours = find_contours(labeled > 0, level=0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color="lime", linewidth=1.5)

    ax.set_title(
        f"CSI + Detections ({n_detections} objects)\n{title}",
        fontsize=12,
    )
    ax.set_xlabel("Column (range)")
    ax.set_ylabel("Row (azimuth)")
    plt.tight_layout()
    plt.show()


# ── Main processing ──────────────────────────────────────────────────


def csi_detection_overlay(
    filepath: Path,
    chip_size: int = 5000,
    num_looks: int = 7,
    dimension: str = "azimuth",
    sigma: float = 3.0,
    smooth_win: int = 7,
    dom_window: int = 3,
    morph_size: int = 3,
    normalization: str = "log",
) -> None:
    """Run detection + CSI pipeline and display overlay.

    Parameters
    ----------
    filepath : Path
        Path to SICD NITF file.
    chip_size : int
        Side length of center chip in pixels.
    num_looks : int
        Number of sub-aperture looks.
    dimension : str
        Frequency axis: ``'azimuth'`` or ``'range'``.
    sigma : float
        Detection threshold in standard deviations above the mean.
    smooth_win : int
        Spatial smoothing kernel size.
    dom_window : int
        Contiguous look window for dominance ratio.
    morph_size : int
        Morphological opening/closing kernel size.
    normalization : str
        CSI normalization method.
    """
    # ---- Read and chip ----
    print(f"Opening: {filepath.name}")
    with SICDReader(filepath) as reader:
        meta = reader.metadata
        rows, cols = reader.get_shape()
        print(f"  Image size: {rows} x {cols}")

        extractor = ChipExtractor(rows, cols)
        region = extractor.chip_at_point(
            rows // 2, cols // 2, chip_size, chip_size,
        )
        chip_h = region.row_end - region.row_start
        chip_w = region.col_end - region.col_start
        print(f"  Center chip: [{region.row_start}:{region.row_end}, "
              f"{region.col_start}:{region.col_end}] "
              f"({chip_h} x {chip_w})")

        chip = reader.read_chip(
            region.row_start, region.row_end,
            region.col_start, region.col_end,
        )

    # ---- Sub-look decomposition ----
    sublook = SublookDecomposition(
        meta, num_looks=num_looks, dimension=dimension,
    )
    looks = sublook.decompose(chip)
    print(f"  Sublook stack: {looks.shape}")

    # ---- Dominance ----
    print("\nComputing dominance...")
    dominance = compute_dominance(looks, window_size=smooth_win,
                                  dom_window=dom_window)

    # ---- Threshold (mean + N * sigma) ----
    dom_mu = np.mean(dominance)
    dom_std = np.std(dominance)
    dom_thresh = dom_mu + sigma * dom_std
    det_mask = dominance > dom_thresh
    print(f"  Threshold: {dom_thresh:.4f}  "
          f"(mu={dom_mu:.4f}, std={dom_std:.4f}, {sigma}\u03c3)")

    # ---- Morphological cleanup ----
    fp = footprint_rectangle((morph_size, morph_size))
    det_mask = opening(det_mask, footprint=fp)
    det_mask = closing(det_mask, footprint=fp)

    # ---- Label connected components ----
    labeled = label(det_mask)
    n_detections = labeled.max()
    print(f"  Detections: {n_detections}")

    # ---- Per-object summary ----
    props = regionprops(labeled)
    for p in props:
        print(f"    Object {p.label}: centroid=({p.centroid[0]:.0f}, "
              f"{p.centroid[1]:.0f}), area={p.area} px")

    # ---- CSI composite ----
    print("\nComputing CSI...")
    csi_proc = CSIProcessor(meta, dimension=dimension,
                            normalization=normalization)
    csi_rgb = csi_proc.apply(chip)
    print(f"  CSI shape: {csi_rgb.shape}")

    # ---- Build title ----
    ci = meta.collection_info
    title_parts = [filepath.name]
    if ci is not None and ci.collector_name:
        title_parts.append(ci.collector_name)
    title = "  |  ".join(title_parts)

    # ---- Plot ----
    plot_csi_detections(csi_rgb, labeled, n_detections, title=title)


# ── Entry point ──────────────────────────────────────────────────────


if __name__ == "__main__":
    args = parse_args()
    csi_detection_overlay(
        args.filepath,
        chip_size=args.chip_size,
        num_looks=args.num_looks,
        dimension=args.dimension,
        sigma=args.sigma,
        smooth_win=args.smooth_win,
        dom_window=args.dom_window,
        morph_size=args.morph_size,
        normalization=args.normalization,
    )
