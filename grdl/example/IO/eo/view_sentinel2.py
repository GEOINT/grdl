# -*- coding: utf-8 -*-
"""
Sentinel-2 Viewer - Display multispectral imagery from Sentinel-2 products.

Reads a Sentinel-2 JP2 file and displays the image using matplotlib.
Displays single band with percentile contrast stretch, or RGB composite.
Shows metadata and geolocation information in the console.

Usage:
  python view_sentinel2.py <sentinel2_file>
  python view_sentinel2.py <sentinel2_file> --cmap gray
  python view_sentinel2.py <sentinel2_file> --plow 2 --phigh 98
  python view_sentinel2.py --help

Dependencies
------------
matplotlib
rasterio or glymur

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
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from grdl.IO.eo import Sentinel2Reader


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Display Sentinel-2 multispectral imagery.",
    )
    parser.add_argument(
        "filepath",
        type=Path,
        help="Path to the Sentinel-2 JP2 file (standalone or within SAFE archive).",
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
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for display (e.g., 0.5 for half size, default: 1.0).",
    )
    return parser.parse_args()


def view_sentinel2(filepath: Path, cmap: str = "gray",
                   plow: float = 2.0, phigh: float = 98.0,
                   scale: float = 1.0) -> None:
    """Read and display a Sentinel-2 image.

    Parameters
    ----------
    filepath : Path
        Path to the Sentinel-2 JP2 file.
    cmap : str
        Matplotlib colormap name.
    plow : float
        Lower percentile for contrast stretch.
    phigh : float
        Upper percentile for contrast stretch.
    scale : float
        Scale factor for display (1.0 = full resolution).
    """
    print(f"Opening: {filepath}")

    with Sentinel2Reader(filepath) as reader:
        meta = reader.metadata
        rows, cols = reader.get_shape()

        # Print metadata
        print(f"  Format:         {meta.format or '?'}")
        print(f"  Size:           {rows} x {cols}")
        print(f"  Data type:      {meta.dtype or '?'}")
        print(f"  Bands:          {meta.bands or '?'}")

        # Sentinel-2 specific metadata
        if meta.satellite:
            print(f"  Satellite:      {meta.satellite}")
        if meta.processing_level:
            print(f"  Product level:  {meta.processing_level}")
        if meta.band_id:
            print(f"  Band:           {meta.band_id}")
        if meta.resolution_tier:
            print(f"  Resolution:     {meta.resolution_tier}m")
        if meta.mgrs_tile_id:
            print(f"  MGRS Tile:      {meta.mgrs_tile_id}")
            if meta.utm_zone and meta.latitude_band:
                print(f"  UTM Zone:       {meta.utm_zone}{meta.latitude_band}")
        if meta.sensing_datetime:
            print(f"  Sensing time:   {meta.sensing_datetime}")
        if meta.wavelength_center:
            print(f"  Wavelength:     {meta.wavelength_center:.0f} nm", end="")
            if meta.wavelength_range:
                wl_min, wl_max = meta.wavelength_range
                print(f" ({wl_min:.0f}-{wl_max:.0f} nm)")
            else:
                print()
        if meta.crs:
            print(f"  CRS:            {meta.crs}")

        # Print bounds if available
        if meta.extras and 'bounds' in meta.extras:
            bounds = meta.extras['bounds']
            print(f"  Bounds:         ({bounds.left:.1f}, {bounds.bottom:.1f}, "
                  f"{bounds.right:.1f}, {bounds.top:.1f})")

        print()

        # Read full image
        print(f"Reading full image ({rows} x {cols})...")
        data = reader.read_full()
        print(f"  Shape: {data.shape}, dtype: {data.dtype}")

        # Handle scaling for display
        if scale != 1.0:
            from scipy.ndimage import zoom
            new_rows = int(rows * scale)
            new_cols = int(cols * scale)
            print(f"  Scaling to {new_rows} x {new_cols}...")
            data = zoom(data, scale, order=1)

    # Convert to float for processing
    data = data.astype(np.float32)

    # Mask nodata values (0 for Sentinel-2)
    if meta.nodata is not None:
        data[data == meta.nodata] = np.nan

    # Contrast stretch
    vmin = np.nanpercentile(data, plow)
    vmax = np.nanpercentile(data, phigh)
    print(f"  Value range:    [{np.nanmin(data):.0f}, {np.nanmax(data):.0f}]")
    print(f"  Display range:  [{vmin:.0f}, {vmax:.0f}] "
          f"(p{plow:.0f}-p{phigh:.0f})")
    print()

    # Build title
    title_parts = [filepath.name]
    if meta.satellite:
        title_parts.append(meta.satellite)
    if meta.band_id:
        title_parts.append(f"Band {meta.band_id}")
    if meta.processing_level:
        title_parts.append(meta.processing_level)
    title = "  |  ".join(title_parts)

    # Display
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.imshow(
        data, cmap=cmap, vmin=vmin, vmax=vmax,
        aspect="equal", interpolation="nearest",
    )
    ax.set_title(f"Sentinel-2 Imagery\n{title}", fontsize=10)
    ax.set_xlabel("Column (Easting)")
    ax.set_ylabel("Row (Northing)")

    # Format colorbar label with units
    cbar_label = "Digital Number"
    if meta.processing_level == 'L1C':
        cbar_label = "TOA Reflectance (DN)"
    elif meta.processing_level == 'L2A':
        cbar_label = "Surface Reflectance (DN)"

    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)

    # Annotate center point if MGRS tile is available
    if meta.mgrs_tile_id:
        center_row = data.shape[0] // 2
        center_col = data.shape[1] // 2
        ax.plot(center_col, center_row, 'r+', markersize=15,
                markeredgewidth=2, zorder=5)
        ax.annotate(
            f"Tile Center\n{meta.mgrs_tile_id}",
            xy=(center_col, center_row), xytext=(12, 0),
            textcoords="offset points", fontsize=8,
            color="red", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    view_sentinel2(args.filepath, cmap=args.cmap,
                   plow=args.plow, phigh=args.phigh,
                   scale=args.scale)
