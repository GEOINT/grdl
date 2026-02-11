# -*- coding: utf-8 -*-
"""
Sublook Comparison - Split a SICD image into sub-aperture looks.

Loads a SICD file, uses ``ChipExtractor`` from ``data_prep`` to plan a
center chip, reads the chip via ``SICDReader``, decomposes it into
azimuth sub-apertures, and displays them.

Two display modes:
  - **Static** (default): 3 non-overlapping sub-looks side-by-side.
  - **Movie** (``--movie``): Animated sweep through many overlapping
    sub-apertures, revealing scatterer motion across the aperture.

Demonstrates full GRDL integration:
  - ``grdl.IO.SICDReader`` for metadata + pixel access
  - ``grdl.data_prep.ChipExtractor`` for chip planning (index-only)
  - ``grdl.image_processing.sar.SublookDecomposition`` for processing
  - ``grdl.image_processing.intensity`` for dB and contrast stretch

Usage:
  python sublook_compare.py <sicd_file>
  python sublook_compare.py <sicd_file> --movie
  python sublook_compare.py <sicd_file> --movie --num-frames 30
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
from typing import List, Optional

# Third-party
import numpy as np

# GRDL
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from grdl.IO import SICDReader
from grdl.data_prep import ChipExtractor
from grdl.image_processing.sar import SublookDecomposition
from grdl.image_processing.intensity import ToDecibels, PercentileStretch


# ── Reusable visualization ───────────────────────────────────────────


def plot_sublook_comparison(
    chip_stretched: np.ndarray,
    looks_stretched: np.ndarray,
    *,
    title: str = "",
    chip_shape: Optional[tuple] = None,
    cmap: str = "gray",
    look_labels: Optional[List[str]] = None,
) -> None:
    """Display a full-aperture chip alongside sub-look images.

    This function is importable from both the manual ``sublook_compare``
    script and the ``grdl-runtime`` workflow example.

    Parameters
    ----------
    chip_stretched : np.ndarray
        2D display-ready full-aperture chip, values in [0, 1].
    looks_stretched : np.ndarray
        3D stack of display-ready sub-looks, shape ``(N, rows, cols)``,
        values in [0, 1].
    title : str
        Figure super-title (file name, collector, etc.).
    chip_shape : tuple, optional
        ``(rows, cols)`` of the original chip for annotation.
        Defaults to ``chip_stretched.shape``.
    cmap : str
        Matplotlib colormap.
    look_labels : list of str, optional
        Per-look axis titles.  Defaults to "Sub-look 1 (fore)" etc.
    """
    import matplotlib
    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt  # noqa: E402

    chip_shape = chip_shape or chip_stretched.shape
    num_looks = looks_stretched.shape[0]
    look_labels = look_labels or [
        f"Sub-look {i + 1}" for i in range(num_looks)
    ]
    if num_looks == 3 and look_labels == [f"Sub-look {i + 1}" for i in range(3)]:
        look_labels = ["Sub-look 1 (fore)", "Sub-look 2 (mid)",
                        "Sub-look 3 (aft)"]

    fig = plt.figure(figsize=(18, 12))

    ax_full = fig.add_subplot(2, 1, 1)
    ax_full.imshow(chip_stretched, cmap=cmap, aspect="auto",
                   interpolation="nearest", vmin=0, vmax=1)
    ax_full.set_title(
        f"Full Aperture  ({chip_shape[0]} x {chip_shape[1]})\n{title}",
        fontsize=11,
    )
    ax_full.set_xlabel("Column (range)")
    ax_full.set_ylabel("Row (azimuth)")

    for i in range(num_looks):
        ax = fig.add_subplot(2, num_looks, num_looks + 1 + i)
        ax.imshow(looks_stretched[i], cmap=cmap, aspect="auto",
                  interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(look_labels[i], fontsize=10)
        ax.set_xlabel("Column (range)")
        if i == 0:
            ax.set_ylabel("Row (azimuth)")

    plt.tight_layout()
    plt.show()


def animate_sublooks(
    looks_stretched: np.ndarray,
    *,
    title: str = "",
    cmap: str = "gray",
    interval_ms: int = 120,
    look_labels: Optional[List[str]] = None,
) -> None:
    """Animate a sweep through sub-aperture looks.

    Parameters
    ----------
    looks_stretched : np.ndarray
        3D stack of display-ready sub-looks, shape ``(N, rows, cols)``,
        values in [0, 1].
    title : str
        Figure super-title.
    cmap : str
        Matplotlib colormap.
    interval_ms : int
        Delay between frames in milliseconds.
    look_labels : list of str, optional
        Per-frame axis titles.  Defaults to "Frame 1 / N" etc.
    """
    import matplotlib
    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.animation import FuncAnimation  # noqa: E402

    num_frames = looks_stretched.shape[0]
    look_labels = look_labels or [
        f"Sub-aperture {i + 1} / {num_frames}" for i in range(num_frames)
    ]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        looks_stretched[0], cmap=cmap, aspect="auto",
        interpolation="nearest", vmin=0, vmax=1,
    )
    frame_title = ax.set_title(look_labels[0], fontsize=11)
    ax.set_xlabel("Column (range)")
    ax.set_ylabel("Row (azimuth)")
    if title:
        fig.suptitle(title, fontsize=12, y=0.98)

    def _update(frame: int) -> list:
        im.set_data(looks_stretched[frame])
        frame_title.set_text(look_labels[frame])
        return [im, frame_title]

    _anim = FuncAnimation(  # noqa: F841  prevent GC
        fig, _update, frames=num_frames,
        interval=interval_ms, blit=True, repeat=True,
    )

    plt.tight_layout()
    plt.show()


# ── CLI ──────────────────────────────────────────────────────────────


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
    parser.add_argument(
        "--movie",
        action="store_true",
        default=False,
        help="Animate a sweep through overlapping sub-apertures.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=20,
        help="Number of sub-aperture frames for movie mode (default: 20).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Playback speed in frames per second for movie (default: 8).",
    )
    return parser.parse_args()


def sublook_compare(
    filepath: Path,
    chip_size: int = 5000,
    plow: float = 2.0,
    phigh: float = 98.0,
    cmap: str = "gray",
    movie: bool = False,
    num_frames: int = 20,
    fps: int = 8,
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
    movie : bool
        If True, animate a sweep through overlapping sub-apertures
        instead of the static 3-panel comparison.
    num_frames : int
        Number of sub-aperture frames for movie mode.
    fps : int
        Playback speed in frames per second for movie mode.
    """
    print(f"Opening: {filepath}")

    # ── Reusable processors ──────────────────────────────────────────
    to_db = ToDecibels()
    stretch = PercentileStretch(plow=plow, phigh=phigh)

    with SICDReader(filepath) as reader:
        meta = reader.metadata
        rows, cols = reader.get_shape()
        print(f"  Image size: {rows} x {cols}")
        print(f'collection time: {meta.timeline.collect_start}, '
              f'duration: {meta.timeline.collect_duration}s')
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

        # Sublook decomposition — static or movie
        if movie:
            overlap = 1.0 - (1.0 / num_frames)
            print(f"  Decomposing into {num_frames} azimuth sub-apertures "
                  f"({overlap:.0%} overlap) for movie...")
            sublook = SublookDecomposition(
                meta, num_looks=num_frames, dimension='azimuth',
                overlap=overlap,
            )
        else:
            print("  Decomposing into 3 azimuth sub-apertures (0% overlap)...")
            sublook = SublookDecomposition(
                meta, num_looks=3, dimension='azimuth', overlap=0.0,
            )

        looks = sublook.decompose(chip)
        print(f"  Sublook stack: {looks.shape}")

    # ── Build title ──────────────────────────────────────────────────
    ci = meta.collection_info
    title_parts = [filepath.name]
    if ci is not None and ci.collector_name:
        title_parts.append(ci.collector_name)
    file_title = "  |  ".join(title_parts)

    # ── Display ──────────────────────────────────────────────────────
    if movie:
        looks_stretched = stretch.apply(to_db.apply(looks))
        print(f"  Animating {num_frames} frames at {fps} fps …")
        animate_sublooks(
            looks_stretched,
            title=file_title,
            cmap=cmap,
            interval_ms=int(1000 / fps),
        )
    else:
        chip_stretched = stretch.apply(to_db.apply(chip))
        looks_stretched = stretch.apply(to_db.apply(looks))
        plot_sublook_comparison(
            chip_stretched, looks_stretched,
            title=file_title,
            chip_shape=(chip_h, chip_w),
            cmap=cmap,
        )


if __name__ == "__main__":
    args = parse_args()
    sublook_compare(
        args.filepath,
        chip_size=args.chip_size,
        plow=args.plow,
        phigh=args.phigh,
        cmap=args.cmap,
        movie=args.movie,
        num_frames=args.num_frames,
        fps=args.fps,
    )
