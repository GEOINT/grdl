# -*- coding: utf-8 -*-
"""
Geolocation Overlay - Display a SAR image with pixel and geographic coordinates.

Loads a SAR image (SICD or BIOMASS), computes geolocation for a grid of
sample points across the image, and overlays each point with its
(row, col) and (lat, lon) coordinates. Clicking any marker opens its
location in Google Maps.

Supports two SAR formats:
  - **SICD** (default): Uses ``SICDReader`` + ``SICDGeolocation`` (sarpy)
  - **BIOMASS**: Uses ``BIOMASSL1Reader`` + ``GCPGeolocation`` (scipy)

Usage:
  python geolocation_overlay.py <sicd_file>
  python geolocation_overlay.py <biomass_dir> --format biomass
  python geolocation_overlay.py <sicd_file> --grid 5
  python geolocation_overlay.py <sicd_file> --chip-size 4096
  python geolocation_overlay.py --help

Dependencies
------------
matplotlib
sarkit or sarpy (SICD)
scipy (BIOMASS)

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
import webbrowser
from pathlib import Path

# Third-party
import numpy as np

# Matplotlib -- try interactive backends, fall back to Agg for --save
import matplotlib
for _backend in ("QtAgg", "TkAgg", "MacOSX", "Agg"):
    try:
        matplotlib.use(_backend)
        break
    except ImportError:
        continue
import matplotlib.pyplot as plt  # noqa: E402

# GRDL
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ── Helpers ───────────────────────────────────────────────────────────


def _google_maps_url(lat: float, lon: float) -> str:
    """Return a Google Maps URL centered on lat/lon."""
    return f"https://www.google.com/maps?q={lat},{lon}&t=k"


def _to_db(arr: np.ndarray) -> np.ndarray:
    """Convert complex or real SAR array to magnitude in dB."""
    return 20.0 * np.log10(np.abs(arr) + 1e-10)


# ── Loaders ───────────────────────────────────────────────────────────


def _load_sicd(filepath, chip_size):
    """Load a SICD image and build geolocation.

    Returns
    -------
    image : np.ndarray
        2D complex array (or magnitude).
    geo : SICDGeolocation
        Geolocation object.
    title : str
        Display title.
    """
    from grdl.IO import SICDReader
    from grdl.geolocation.sar.sicd import SICDGeolocation

    reader = SICDReader(filepath)
    meta = reader.metadata
    rows, cols = reader.get_shape()

    geo = SICDGeolocation.from_reader(reader)

    # Read a center chip if the image is very large
    if chip_size and (rows > chip_size or cols > chip_size):
        from grdl.data_prep import ChipExtractor
        extractor = ChipExtractor(
            image_height=rows, image_width=cols,
            chip_height=chip_size, chip_width=chip_size,
        )
        region = extractor.extract_chip(rows // 2, cols // 2)
        print(f"  Reading center chip: {region.height}x{region.width} "
              f"(from {rows}x{cols})")
        image = reader.read_chip(
            region.row_start, region.row_start + region.height,
            region.col_start, region.col_start + region.width,
        )
        # Adjust geo shape for chip coordinates
        chip_offset = (region.row_start, region.col_start)
    else:
        image = reader.read_full()
        chip_offset = (0, 0)

    # Build title
    title_parts = [filepath.name]
    ci = meta.collection_info
    if ci is not None and ci.collector_name:
        title_parts.append(ci.collector_name)
    title = "  |  ".join(title_parts)

    reader.close()
    return image, geo, title, chip_offset


def _load_biomass(filepath, chip_size):
    """Load a BIOMASS L1 SCS image and build geolocation.

    Returns
    -------
    image : np.ndarray
        2D complex array (HH band).
    geo : GCPGeolocation
        Geolocation object.
    title : str
        Display title.
    """
    from grdl.IO import BIOMASSL1Reader
    from grdl.geolocation.sar.gcp import GCPGeolocation

    reader = BIOMASSL1Reader(filepath)
    rows = reader.metadata['rows']
    cols = reader.metadata['cols']

    geo_info = {
        'gcps': reader.metadata['gcps'],
        'crs': reader.metadata.get('crs', 'WGS84'),
    }
    geo = GCPGeolocation.from_dict(geo_info, reader.metadata)

    # Read HH (band 0)
    if chip_size and (rows > chip_size or cols > chip_size):
        r0 = max(0, rows // 2 - chip_size // 2)
        c0 = max(0, cols // 2 - chip_size // 2)
        r1 = min(rows, r0 + chip_size)
        c1 = min(cols, c0 + chip_size)
        print(f"  Reading center chip: {r1 - r0}x{c1 - c0} "
              f"(from {rows}x{cols})")
        image = reader.read_chip(r0, r1, c0, c1, bands=[0])
        chip_offset = (r0, c0)
    else:
        image = reader.read_chip(0, rows, 0, cols, bands=[0])
        chip_offset = (0, 0)

    title = f"BIOMASS L1 SCS  |  {filepath.name}"
    reader.close()
    return image, geo, title, chip_offset


# ── Grid computation ──────────────────────────────────────────────────


def compute_grid_points(image_shape, geo, chip_offset, grid_size, margin_pct):
    """Compute a grid of sample points with pixel and geographic coordinates.

    Parameters
    ----------
    image_shape : tuple
        (rows, cols) of the displayed image.
    geo : Geolocation
        Geolocation object (operates in full-image coordinates).
    chip_offset : tuple
        (row_offset, col_offset) of the chip within the full image.
    grid_size : int
        Number of grid points per axis.
    margin_pct : float
        Percentage margin from image edges.

    Returns
    -------
    list of dict
        Each dict has keys: img_row, img_col, full_row, full_col,
        lat, lon, height.
    """
    img_rows, img_cols = image_shape
    mr = int(img_rows * margin_pct / 100.0)
    mc = int(img_cols * margin_pct / 100.0)

    sample_rows = np.linspace(mr, img_rows - 1 - mr, grid_size, dtype=int)
    sample_cols = np.linspace(mc, img_cols - 1 - mc, grid_size, dtype=int)

    r_off, c_off = chip_offset

    points = []
    for r in sample_rows:
        for c in sample_cols:
            full_r = float(r + r_off)
            full_c = float(c + c_off)
            try:
                lat, lon, h = geo.image_to_latlon(full_r, full_c)
                if not (np.isfinite(lat) and np.isfinite(lon)):
                    continue
                points.append({
                    'img_row': r, 'img_col': c,
                    'full_row': full_r, 'full_col': full_c,
                    'lat': lat, 'lon': lon, 'height': h,
                })
            except (ValueError, NotImplementedError):
                continue

    return points


# ── Console table ─────────────────────────────────────────────────────


def print_point_table(points):
    """Print a formatted table of sample points to the console.

    Parameters
    ----------
    points : list of dict
        Points from compute_grid_points.
    """
    print()
    print(f"{'#':>3}  {'Row':>7}  {'Col':>7}  {'Latitude':>12}  "
          f"{'Longitude':>12}  {'Height (m)':>10}")
    print("-" * 62)
    for i, p in enumerate(points):
        print(f"{i + 1:3d}  {p['full_row']:7.0f}  {p['full_col']:7.0f}  "
              f"{p['lat']:12.6f}  {p['lon']:12.6f}  {p['height']:10.1f}")
    print()


# ── Visualization ─────────────────────────────────────────────────────


def plot_geolocation_overlay(image, points, title, cmap, plow, phigh):
    """Display the SAR image with geolocation overlay markers.

    Each marker is annotated with both pixel (row, col) and geographic
    (lat, lon) coordinates. Clicking a marker opens Google Maps.

    Parameters
    ----------
    image : np.ndarray
        2D image array (complex or real).
    points : list of dict
        Grid points from compute_grid_points.
    title : str
        Figure title.
    cmap : str
        Matplotlib colormap.
    plow, phigh : float
        Percentile range for contrast stretch.
    """
    # Convert to dB magnitude
    db = _to_db(image)
    vmin = np.nanpercentile(db, plow)
    vmax = np.nanpercentile(db, phigh)

    rows, cols = image.shape[:2]
    aspect = rows / cols
    fig_w = 14
    fig_h = min(fig_w * aspect, 20)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    ax.imshow(db, cmap=cmap, vmin=vmin, vmax=vmax,
              aspect="auto", interpolation="nearest")

    # Plot markers
    marker_data = []
    px = [p['img_col'] for p in points]
    py = [p['img_row'] for p in points]

    sc = ax.scatter(
        px, py,
        marker='o', c='cyan', s=60, edgecolors='black', linewidths=0.8,
        zorder=5, picker=True, pickradius=8,
    )
    marker_data = [
        {'lat': p['lat'], 'lon': p['lon'],
         'label': f"({p['full_row']:.0f}, {p['full_col']:.0f})"}
        for p in points
    ]

    # Annotate each point with pixel and geographic coordinates
    for p in points:
        label = (
            f"({p['full_row']:.0f}, {p['full_col']:.0f})\n"
            f"{p['lat']:.4f}, {p['lon']:.4f}"
        )
        ax.annotate(
            label,
            xy=(p['img_col'], p['img_row']),
            xytext=(8, -4),
            textcoords="offset points",
            fontsize=6.5,
            color="cyan",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7),
        )

    ax.set_title(f"Geolocation Overlay\n{title}", fontsize=10)
    ax.set_xlabel("Column (range)")
    ax.set_ylabel("Row (azimuth)")

    # Click-to-Google-Maps handler
    def on_pick(event):
        if event.artist is not sc:
            return
        ind = event.ind[0]
        d = marker_data[ind]
        url = _google_maps_url(d['lat'], d['lon'])
        print(f"  {d['label']}: ({d['lat']:.6f}, {d['lon']:.6f})")
        print(f"  {url}")
        webbrowser.open(url)

    fig.canvas.mpl_connect("pick_event", on_pick)

    plt.tight_layout()
    print("Click any marker to open its location in Google Maps.\n")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Display a SAR image with geolocation coordinate overlay. "
            "Each sample point shows its (row, col) and (lat, lon)."
        ),
    )
    parser.add_argument(
        "filepath",
        type=Path,
        help="Path to SAR file (SICD NITF) or directory (BIOMASS product).",
    )
    parser.add_argument(
        "--format",
        choices=["sicd", "biomass"],
        default="sicd",
        help="SAR format: 'sicd' (default) or 'biomass'.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=4,
        help="Grid points per axis (default: 4, giving a 4x4 grid).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=10.0,
        help="Percentage margin from image edges (default: 10).",
    )
    parser.add_argument(
        "--chip-size",
        type=int,
        default=None,
        help="Max chip size in pixels. Reads a center crop for large images.",
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
        "--save",
        type=Path,
        default=None,
        help="Save figure to file instead of displaying interactively.",
    )
    return parser.parse_args()


def main():
    """Load SAR image, compute geolocation grid, and display overlay."""
    args = parse_args()

    print(f"Loading: {args.filepath}")
    print(f"  Format: {args.format}")

    # Load image and geolocation
    if args.format == "biomass":
        image, geo, title, chip_offset = _load_biomass(
            args.filepath, args.chip_size
        )
    else:
        image, geo, title, chip_offset = _load_sicd(
            args.filepath, args.chip_size
        )

    print(f"  Image shape: {image.shape}")
    print(f"  Chip offset: ({chip_offset[0]}, {chip_offset[1]})")

    # Compute grid of sample points
    img_shape = image.shape[:2]
    points = compute_grid_points(
        img_shape, geo, chip_offset, args.grid, args.margin
    )
    print(f"  Grid points: {len(points)} "
          f"({args.grid}x{args.grid}, {args.margin}% margin)")

    # Print table
    print_point_table(points)

    if not points:
        print("No valid geolocation points computed. Check the image and "
              "geolocation configuration.")
        sys.exit(1)

    # Display
    if args.save:
        # Non-interactive save
        matplotlib.use("Agg")
        plot_geolocation_overlay(
            image, points, title, args.cmap, args.plow, args.phigh
        )
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plot_geolocation_overlay(
            image, points, title, args.cmap, args.plow, args.phigh
        )


if __name__ == "__main__":
    main()
