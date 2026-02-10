# -*- coding: utf-8 -*-
"""
SICD Viewer - Display complex SAR imagery from SICD files.

Reads a SICD file and displays the magnitude image using matplotlib.
Displays linear magnitude (not dB) with percentile contrast stretch.
Shows metadata and geolocation information in the console.

Usage:
  python view_sicd.py <sicd_file>
  python view_sicd.py <sicd_file> --cmap gray
  python view_sicd.py <sicd_file> --plow 1 --phigh 99
  python view_sicd.py --help

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
2026-02-09

Modified
--------
2026-02-09
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from grdl.IO import SICDReader


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Display SICD complex SAR imagery (linear magnitude).",
    )
    parser.add_argument(
        "filepath",
        type=Path,
        help="Path to the SICD file (NITF or other SICD container).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="gray",
        help="Matplotlib colormap (default: gray).",
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
    return parser.parse_args()


def view_sicd(filepath: Path, cmap: str = "gray",
              plow: float = 2.0, phigh: float = 98.0) -> None:
    """Read and display a SICD image.

    Parameters
    ----------
    filepath : Path
        Path to the SICD file.
    cmap : str
        Matplotlib colormap name.
    plow : float
        Lower percentile for contrast stretch.
    phigh : float
        Upper percentile for contrast stretch.
    """
    print(f"Opening: {filepath}")

    with SICDReader(filepath) as reader:
        rows, cols = reader.get_shape()

        # Print metadata
        print(f"  Backend:        {reader.metadata.get('backend', '?')}")
        print(f"  Size:           {rows} x {cols}")
        print(f"  Pixel type:     {reader.metadata.get('pixel_type', '?')}")
        print(f"  Collector:      {reader.metadata.get('collector_name', '?')}")
        print(f"  Core name:      {reader.metadata.get('core_name', '?')}")
        print(f"  Classification: {reader.metadata.get('classification', '?')}")
        print(f"  Collect start:  {reader.metadata.get('collect_start', '?')}")
        duration = reader.metadata.get('collect_duration')
        if duration is not None:
            print(f"  Duration:       {duration:.3f} s")

        # Print geolocation from metadata
        scp = reader.metadata.get('scp_llh')
        if scp is not None:
            print(f"  SCP:            ({scp[0]:.6f}, {scp[1]:.6f}, {scp[2]:.1f} m)")
        print()

        # Read full complex image
        print(f"Reading full image ({rows} x {cols})...")
        complex_data = reader.read_full()
        print(f"  Shape: {complex_data.shape}, dtype: {complex_data.dtype}")

    # Compute linear magnitude
    magnitude = np.abs(complex_data)

    # Contrast stretch
    vmin = np.percentile(magnitude, plow)
    vmax = np.percentile(magnitude, phigh)
    print(f"  Magnitude range: [{magnitude.min():.4f}, {magnitude.max():.4f}]")
    print(f"  Display range:   [{vmin:.4f}, {vmax:.4f}] "
          f"(p{plow:.0f}-p{phigh:.0f})")
    print()

    # Build title
    title_parts = [filepath.name]
    collector = reader.metadata.get('collector_name')
    if collector:
        title_parts.append(collector)
    title = "  |  ".join(title_parts)

    # Display
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.imshow(
        magnitude, cmap=cmap, vmin=vmin, vmax=vmax,
        aspect="auto", interpolation="nearest",
    )
    ax.set_title(f"SICD Magnitude (linear)\n{title}", fontsize=10)
    ax.set_xlabel("Column (range)")
    ax.set_ylabel("Row (azimuth)")
    fig.colorbar(im, ax=ax, label="Magnitude", shrink=0.8)

    # Annotate SCP if available
    if scp is not None:
        scp_row, scp_col = rows // 2, cols // 2
        ax.plot(scp_col, scp_row, 'r*', markersize=12, markeredgecolor='black',
                markeredgewidth=0.5, zorder=5)
        ax.annotate(
            f"SCP\n({scp[0]:.4f}, {scp[1]:.4f})",
            xy=(scp_col, scp_row), xytext=(12, 0),
            textcoords="offset points", fontsize=8,
            color="red", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    view_sicd(args.filepath, cmap=args.cmap, plow=args.plow, phigh=args.phigh)
