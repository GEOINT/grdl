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
        nargs='?',
        type=Path,
        default=Path('default/path/to/file'),
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
        meta = reader.metadata
        rows, cols = reader.get_shape()

        # Print metadata
        print(f"  Backend:        {meta.backend or '?'}")
        print(f"  Size:           {rows} x {cols}")

        if meta.image_data is not None:
            print(f"  Pixel type:     {meta.image_data.pixel_type or '?'}")

        ci = meta.collection_info
        if ci is not None:
            print(f"  Collector:      {ci.collector_name or '?'}")
            print(f"  Core name:      {ci.core_name or '?'}")
            print(f"  Classification: {ci.classification or '?'}")

        tl = meta.timeline
        if tl is not None:
            print(f"  Collect start:  {tl.collect_start or '?'}")
            if tl.collect_duration is not None:
                print(f"  Duration:       {tl.collect_duration:.3f} s")

        # Print geolocation from metadata
        scp_llh = None
        if meta.geo_data is not None and meta.geo_data.scp is not None:
            scp_llh = meta.geo_data.scp.llh
        if scp_llh is not None:
            print(f"  SCP:            ({scp_llh.lat:.6f}, {scp_llh.lon:.6f}, "
                  f"{scp_llh.hae:.1f} m)")

        # Print SCPCOA (Scene Center Point Center of Aperture)
        scpcoa = meta.scpcoa
        if scpcoa is not None:
            print(f"  Side of track:  {scpcoa.side_of_track or '?'}")
            print(f"  SCP COA time:   {scpcoa.scp_time:.6f} s" if scpcoa.scp_time is not None else "  SCP COA time:   ?")
            if scpcoa.slant_range is not None:
                print(f"  Slant range:    {scpcoa.slant_range:,.1f} m")
            if scpcoa.ground_range is not None:
                print(f"  Ground range:   {scpcoa.ground_range:,.1f} m")
            if scpcoa.graze_ang is not None:
                print(f"  Graze angle:    {scpcoa.graze_ang:.4f} deg")
            if scpcoa.incidence_ang is not None:
                print(f"  Incidence angle:{scpcoa.incidence_ang:.4f} deg")
            if scpcoa.doppler_cone_ang is not None:
                print(f"  Doppler cone:   {scpcoa.doppler_cone_ang:.4f} deg")
            if scpcoa.twist_ang is not None:
                print(f"  Twist angle:    {scpcoa.twist_ang:.4f} deg")
            if scpcoa.slope_ang is not None:
                print(f"  Slope angle:    {scpcoa.slope_ang:.4f} deg")
            if scpcoa.azim_ang is not None:
                print(f"  Azimuth angle:  {scpcoa.azim_ang:.4f} deg")
            if scpcoa.layover_ang is not None:
                print(f"  Layover angle:  {scpcoa.layover_ang:.4f} deg")
            if scpcoa.arp_pos is not None:
                p = scpcoa.arp_pos
                print(f"  ARP position:   ({p.x:,.1f}, {p.y:,.1f}, {p.z:,.1f}) m ECF")
            if scpcoa.arp_vel is not None:
                v = scpcoa.arp_vel
                print(f"  ARP velocity:   ({v.x:.1f}, {v.y:.1f}, {v.z:.1f}) m/s")
            if scpcoa.arp_acc is not None:
                a = scpcoa.arp_acc
                print(f"  ARP accel:      ({a.x:.4f}, {a.y:.4f}, {a.z:.4f}) m/s^2")

        # Get SCP pixel location for annotation
        scp_pixel = None
        if meta.image_data is not None:
            scp_pixel = meta.image_data.scp_pixel
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
    if ci is not None and ci.collector_name:
        title_parts.append(ci.collector_name)
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
    if scp_llh is not None:
        scp_row = int(scp_pixel.row) if scp_pixel is not None else rows // 2
        scp_col = int(scp_pixel.col) if scp_pixel is not None else cols // 2
        ax.plot(scp_col, scp_row, 'r*', markersize=12, markeredgecolor='black',
                markeredgewidth=0.5, zorder=5)
        ax.annotate(
            f"SCP\n({scp_llh.lat:.4f}, {scp_llh.lon:.4f})",
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
