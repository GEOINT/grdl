# -*- coding: utf-8 -*-
"""
SICD Orthorectification Example.

Loads a SICD complex SAR image, computes linear magnitude in slant range,
then orthorectifies to a regular WGS84 geographic grid using the universal
OrthoPipeline.  Displays original slant range alongside orthorectified output.

Key processing choice: magnitude is computed in slant range *before*
orthorectification.  Resampling complex SAR data causes phase cancellation
that destroys contrast and speckle texture.

Shows two panels:
  1. Original magnitude in slant range geometry
  2. Orthorectified magnitude on a geographic grid

Usage:
  python ortho_sicd.py <sicd_file>
  python ortho_sicd.py <sicd_file> --save [path]
  python ortho_sicd.py <sicd_file> --res 0.00005
  python ortho_sicd.py <sicd_file> --interp nearest
  python ortho_sicd.py <sicd_file> --dem /path/to/dted
  python ortho_sicd.py <sicd_file> --tile 2048
  python ortho_sicd.py <sicd_file> --roi              (center 25%)
  python ortho_sicd.py <sicd_file> --roi 36.0,36.05,-75.8,-75.75
  python ortho_sicd.py --help

Dependencies
------------
matplotlib
scipy
sarpy (or sarkit for I/O, sarpy for projection)

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
2026-02-17

Modified
--------
2026-02-17
"""

# Standard library
import sys
import time
from pathlib import Path

# Third-party
import numpy as np

# Matplotlib -- set backend before importing pyplot
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt  # noqa: E402

# GRDL
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from grdl.IO.sar import SICDReader
from grdl.geolocation.sar.sicd import SICDGeolocation
from grdl.image_processing.ortho import OrthoPipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def ortho_sicd(
    filepath: Path,
    save_path: Path = None,
    resolution: float = None,
    interpolation: str = 'nearest',
    dem_path: str = None,
    tile_size: int = None,
    roi_bounds: tuple = None,
) -> None:
    """Orthorectify a SICD image and display original vs. orthorectified.

    Parameters
    ----------
    filepath : Path
        Path to SICD NITF file.
    save_path : Path, optional
        If provided, save figure to this path.
    resolution : float, optional
        Output pixel size in degrees.  If None, auto-computed from SICD
        grid metadata (sample spacing + graze angle correction).
    interpolation : str, default='nearest'
        Resampling method: 'nearest', 'bilinear', or 'bicubic'.
    dem_path : str, optional
        Path to DTED data directory for terrain-corrected projection.
    tile_size : int, optional
        Process output in tiles of this size (pixels).  Reduces peak
        memory by computing the coordinate mapping per tile instead of
        for the full output grid at once.
    roi_bounds : tuple, optional
        Geographic ROI as (min_lat, max_lat, min_lon, max_lon).  If the
        sentinel value ``'center'`` is passed, a 25% center crop is
        computed from the image footprint.
    """
    print(f"Loading: {filepath.name}")

    with SICDReader(filepath) as reader:
        meta = reader.metadata
        rows, cols = meta.rows, meta.cols

        # Print metadata summary
        mode = 'unknown'
        if meta.collection_info and meta.collection_info.radar_mode:
            mode = meta.collection_info.radar_mode.mode_type or 'unknown'

        print(f"  Size:       {rows} x {cols}")
        print(f"  Mode:       {mode}")
        print(f"  Backend:    {meta.backend}")
        if meta.grid:
            print(f"  Row ss:     {meta.grid.row.ss:.4f} m")
            print(f"  Col ss:     {meta.grid.col.ss:.4f} m")
            print(f"  Plane:      {meta.grid.image_plane}")
        if meta.scpcoa:
            print(f"  Graze:      {meta.scpcoa.graze_ang:.2f} deg")
        if meta.geo_data and meta.geo_data.scp and meta.geo_data.scp.llh:
            scp = meta.geo_data.scp.llh
            print(f"  SCP:        ({scp.lat:.4f}, {scp.lon:.4f})")
        print()

        # ------------------------------------------------------------------
        # Create geolocation
        # ------------------------------------------------------------------
        geo = SICDGeolocation.from_reader(reader)

        # ------------------------------------------------------------------
        # Read complex data and compute magnitude in slant range
        # ------------------------------------------------------------------
        print("Reading complex data...")
        t0 = time.perf_counter()
        complex_data = reader.read_full()
        t_read = time.perf_counter() - t0
        print(f"  Read in {t_read:.2f} s ({complex_data.dtype})")

    # Compute linear magnitude while still in slant range
    print("Computing magnitude in slant range...")
    mag = np.abs(complex_data).astype(np.float32)
    del complex_data  # Free complex array

    # ------------------------------------------------------------------
    # Build and run pipeline
    # ------------------------------------------------------------------
    pipeline = (
        OrthoPipeline()
        .with_source_array(mag)
        .with_metadata(meta)
        .with_geolocation(geo)
        .with_interpolation(interpolation)
        .with_nodata(np.nan)
    )

    # DEM for terrain correction
    if dem_path is not None:
        from grdl.geolocation.elevation import DTEDElevation
        elev = DTEDElevation(dem_path)
        pipeline = pipeline.with_elevation(elev)
        print(f"DEM:          {dem_path}")

    # ROI: explicit bounds or center crop
    if roi_bounds == 'center':
        min_lon, min_lat, max_lon, max_lat = geo.get_bounds()
        lat_span = max_lat - min_lat
        lon_span = max_lon - min_lon
        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0
        half_lat = lat_span * 0.25
        half_lon = lon_span * 0.25
        roi_bounds = (
            center_lat - half_lat, center_lat + half_lat,
            center_lon - half_lon, center_lon + half_lon,
        )
        print(f"ROI:          center 25% of footprint")

    if roi_bounds is not None:
        pipeline = pipeline.with_roi(*roi_bounds)
        print(f"  Lat: [{roi_bounds[0]:.6f}, {roi_bounds[1]:.6f}]")
        print(f"  Lon: [{roi_bounds[2]:.6f}, {roi_bounds[3]:.6f}]")

    # Resolution: explicit or auto-computed
    if resolution is not None:
        pipeline = pipeline.with_resolution(resolution, resolution)
        print(f"Resolution:   {resolution:.6f} deg (user-specified)")
    else:
        print("Resolution:   auto (from SICD grid metadata)")

    # Tiled processing for memory efficiency
    if tile_size is not None:
        pipeline = pipeline.with_tile_size(tile_size)
        print(f"Tile size:    {tile_size}")

    print()
    print("Running ortho pipeline...")
    t0 = time.perf_counter()
    result = pipeline.run()
    t_pipe = time.perf_counter() - t0
    print(f"  Pipeline completed in {t_pipe:.2f} s")

    grid = result.output_grid
    print(f"  Output grid: {grid.rows} x {grid.cols}")
    print(f"  Lat: [{grid.min_lat:.4f}, {grid.max_lat:.4f}]")
    print(f"  Lon: [{grid.min_lon:.4f}, {grid.max_lon:.4f}]")
    print(f"  Pixel: {grid.pixel_size_lat:.6f} x {grid.pixel_size_lon:.6f} deg")

    n_valid = np.sum(np.isfinite(result.data))
    n_total = grid.rows * grid.cols
    print(f"  Valid: {n_valid:,} / {n_total:,} ({100 * n_valid / n_total:.1f}%)")
    print()

    # ------------------------------------------------------------------
    # Plot: 2 panels
    # ------------------------------------------------------------------
    slant_vmin = np.nanpercentile(mag, 2)
    slant_vmax = np.nanpercentile(mag, 98)

    ortho_data = result.data
    valid_mask = np.isfinite(ortho_data)
    if np.any(valid_mask):
        ortho_vmin = np.nanpercentile(ortho_data[valid_mask], 2)
        ortho_vmax = np.nanpercentile(ortho_data[valid_mask], 98)
    else:
        ortho_vmin, ortho_vmax = slant_vmin, slant_vmax

    fig, (ax_slant, ax_ortho) = plt.subplots(1, 2, figsize=(16, 8))

    # Panel 1: Original slant range
    ax_slant.imshow(
        mag, cmap='gray', vmin=slant_vmin, vmax=slant_vmax,
        aspect='auto', interpolation='nearest',
    )
    ax_slant.set_title("Original (Slant Range)\nMagnitude", fontsize=10)
    ax_slant.set_xlabel("Column (range)")
    ax_slant.set_ylabel("Row (azimuth)")

    # Panel 2: Orthorectified with geographic axes
    extent = [grid.min_lon, grid.max_lon, grid.min_lat, grid.max_lat]
    ax_ortho.imshow(
        ortho_data, cmap='gray', vmin=ortho_vmin, vmax=ortho_vmax,
        aspect='auto', interpolation='nearest', extent=extent,
    )
    ax_ortho.set_title("Orthorectified\nMagnitude", fontsize=10)
    ax_ortho.set_xlabel("Longitude (deg)")
    ax_ortho.set_ylabel("Latitude (deg)")

    title_parts = [f"SICD Orthorectification  |  {filepath.name}"]
    title_parts.append(
        f"{mode}  |  Grid {grid.rows}x{grid.cols} @ "
        f"{grid.pixel_size_lat:.6f}\u00b0  |  "
        f"{interpolation} interp"
    )
    if dem_path:
        title_parts[-1] += "  |  DEM-corrected"
    fig.suptitle("\n".join(title_parts), fontsize=11, fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    """Parse command-line arguments."""
    args = sys.argv[1:]
    filepath = None
    save_path = None
    resolution = None
    interpolation = 'nearest'
    dem_path = None
    tile_size = None
    roi_bounds = None

    i = 0
    while i < len(args):
        if args[i] == "--help":
            print("SICD Orthorectification Example")
            print()
            print("Usage:")
            print("  python ortho_sicd.py <sicd_file>")
            print("  python ortho_sicd.py <sicd_file> --save [path]")
            print("  python ortho_sicd.py <sicd_file> --res 0.00005")
            print("  python ortho_sicd.py <sicd_file> --interp nearest")
            print("  python ortho_sicd.py <sicd_file> --dem /path/to/dted")
            print()
            print("Options:")
            print("  --save [path]   Save figure (default: sicd_ortho.png)")
            print("  --res <deg>     Output resolution in degrees")
            print("  --interp <m>    Interpolation: nearest, bilinear, bicubic")
            print("  --dem <path>    DTED directory for terrain correction")
            print("  --tile <N>      Process output in NxN tiles (reduces memory)")
            print("  --roi [bounds]  ROI: omit for center 25%, or min_lat,max_lat,min_lon,max_lon")
            print()
            print("Processing notes:")
            print("  - Magnitude computed in slant range, then orthorectified")
            print("  - Default: nearest-neighbor to preserve SAR speckle texture")
            print("  - Resolution auto-computed from SICD grid metadata if not given")
            sys.exit(0)
        elif args[i] == "--save":
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                save_path = Path(args[i + 1])
                i += 2
            else:
                save_path = Path("sicd_ortho.png")
                i += 1
        elif args[i] == "--res":
            resolution = float(args[i + 1])
            i += 2
        elif args[i] == "--interp":
            interpolation = args[i + 1]
            i += 2
        elif args[i] == "--dem":
            dem_path = args[i + 1]
            i += 2
        elif args[i] == "--tile":
            tile_size = int(args[i + 1])
            i += 2
        elif args[i] == "--roi":
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                parts = args[i + 1].split(",")
                roi_bounds = tuple(float(x) for x in parts)
                i += 2
            else:
                roi_bounds = 'center'
                i += 1
        else:
            filepath = Path(args[i])
            i += 1

    return filepath, save_path, resolution, interpolation, dem_path, tile_size, roi_bounds


if __name__ == "__main__":
    fp, sv, res, interp, dem, tile, roi = _parse_args()

    if fp is None:
        print("Error: SICD file path required.")
        print("Usage: python ortho_sicd.py <sicd_file> [--save] [--res N] [--dem path]")
        sys.exit(1)

    ortho_sicd(fp, save_path=sv, resolution=res,
               interpolation=interp, dem_path=dem, tile_size=tile,
               roi_bounds=roi)
