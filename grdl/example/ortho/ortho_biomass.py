# -*- coding: utf-8 -*-
"""
BIOMASS Orthorectification Example.

Loads a BIOMASS L1A SCS product (slant range geometry), orthorectifies it
to a regular WGS84 geographic grid, and displays the result side-by-side
with the original slant range image.

Key processing choice: SAR-specific processing (magnitude extraction, Pauli
decomposition) is done in slant range *before* orthorectification.
Orthorectifying real-valued magnitude/dB avoids the phase cancellation
artifacts that occur when resampling complex SAR data.

Shows three panels:
  1. Original HH magnitude (dB) in slant range geometry
  2. Orthorectified HH magnitude (dB) on a geographic grid
  3. Orthorectified Pauli RGB (R=double-bounce, G=volume, B=surface)

Uses the PauliDecomposition class for a true quad-pol decomposition with
proper 1/sqrt(2) normalization and all four channels (HH, HV, VH, VV).

Usage:
  python ortho_biomass.py                 # Use most recent product
  python ortho_biomass.py <product_path>  # Specific product
  python ortho_biomass.py --save [path]   # Save figure to file
  python ortho_biomass.py --res 0.0002    # Custom resolution (degrees)
  python ortho_biomass.py --help          # Show usage

Dependencies
------------
matplotlib
rasterio
scipy

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
2026-01-30

Modified
--------
2026-01-30
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

from grdl.IO import BIOMASSL1Reader
from grdl.geolocation.sar.gcp import GCPGeolocation
from grdl.image_processing import Orthorectifier, OutputGrid, PauliDecomposition

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("/Volumes/PRO-G40/SAR_DATA/BIOMASS")

# Default output resolution in degrees.
# None = auto-compute from metadata pixel spacings (native resolution).
# Override with --res flag, e.g. --res 0.0002 (~22 m).
DEFAULT_RESOLUTION = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_product(data_dir: Path) -> Path:
    """Find the most recently modified BIOMASS product directory."""
    candidates = []
    for d in data_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("BIO_S"):
            continue
        inner = d / d.name
        if inner.is_dir() and (inner / "annotation").is_dir():
            candidates.append(inner)
        elif (d / "annotation").is_dir():
            candidates.append(d)

    if not candidates:
        raise FileNotFoundError(
            f"No BIOMASS products found in {data_dir}. "
            f"Run discover_and_download.py first."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _to_db(arr: np.ndarray) -> np.ndarray:
    """Convert complex SAR array to magnitude in dB."""
    return 20.0 * np.log10(np.abs(arr) + 1e-10)


def _normalize(arr: np.ndarray, valid: np.ndarray,
               plow: float = 2, phigh: float = 98) -> np.ndarray:
    """Percentile-stretch an array to [0, 1], using only valid pixels."""
    vals = arr[valid]
    if vals.size == 0:
        return np.zeros_like(arr)
    vmin = np.nanpercentile(vals, plow)
    vmax = np.nanpercentile(vals, phigh)
    out = (arr - vmin) / (vmax - vmin + 1e-10)
    return np.clip(out, 0, 1)


def _compute_resolution(reader: BIOMASSL1Reader, geo: GCPGeolocation):
    """
    Compute output grid resolution from the reader's pixel spacings.

    Uses the metadata range/azimuth pixel spacings and the image center
    latitude to convert meters to degrees.

    Returns
    -------
    Tuple[float, float]
        (pixel_size_lat, pixel_size_lon) in degrees.
    """
    range_m = reader.metadata.get('range_pixel_spacing', 0)
    azimuth_m = reader.metadata.get('azimuth_pixel_spacing', 0)

    if range_m <= 0 or azimuth_m <= 0:
        # Fallback: 0.0005 deg ~ 55 m
        return 0.0005, 0.0005

    # Use image center latitude for meters-to-degrees conversion
    center_lat, _, _ = geo.pixel_to_latlon(
        geo.shape[0] // 2, geo.shape[1] // 2
    )

    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(np.radians(center_lat))

    # Use the coarser of range/azimuth as the output pixel size
    spacing_m = max(range_m, azimuth_m)
    pixel_size_lat = spacing_m / meters_per_deg_lat
    pixel_size_lon = spacing_m / meters_per_deg_lon

    return pixel_size_lat, pixel_size_lon


# ---------------------------------------------------------------------------
# Orthorectify and display
# ---------------------------------------------------------------------------

def ortho_biomass(product_path: Path, save_path: Path = None,
                  resolution: float = None) -> None:
    """
    Orthorectify a BIOMASS product and display original vs. orthorectified.

    SAR-specific processing (magnitude, Pauli decomposition) is done in
    slant range *before* orthorectification. This avoids the phase
    cancellation artifacts that occur when resampling complex SAR data
    with bilinear or bicubic interpolation.

    Parameters
    ----------
    product_path : Path
        Path to the BIOMASS product directory.
    save_path : Path, optional
        If provided, save the figure to this path instead of displaying.
    resolution : float, optional
        Output pixel size in degrees. If None, auto-computed from the
        metadata pixel spacings (native resolution).
    """
    print(f"Loading: {product_path.name}")
    reader = BIOMASSL1Reader(product_path)
    geo_info = {
        'gcps': reader.metadata['gcps'],
        'crs': reader.metadata.get('crs', 'WGS84'),
    }
    geo = GCPGeolocation.from_dict(geo_info, reader.metadata)
    rows, cols = geo.shape
    pols = reader.metadata["polarizations"]

    range_m = reader.metadata.get('range_pixel_spacing', 0)
    azimuth_m = reader.metadata.get('azimuth_pixel_spacing', 0)

    print(f"  Size:    {rows} x {cols}")
    print(f"  Pols:    {pols}")
    print(f"  Orbit:   {reader.metadata.get('orbit_number', '?')}")
    print(f"  Date:    {reader.metadata.get('start_time', '?')}")
    print(f"  Spacing: {range_m:.2f} m (range) x {azimuth_m:.2f} m (azimuth)")
    print()

    # ------------------------------------------------------------------
    # Compute output grid resolution
    # ------------------------------------------------------------------
    if resolution is not None:
        pixel_size_lat = resolution
        pixel_size_lon = resolution
        print(f"Output resolution: {resolution:.6f} deg (user-specified)")
    elif DEFAULT_RESOLUTION is not None:
        pixel_size_lat = DEFAULT_RESOLUTION
        pixel_size_lon = DEFAULT_RESOLUTION
        print(f"Output resolution: {DEFAULT_RESOLUTION:.6f} deg (configured)")
    else:
        pixel_size_lat, pixel_size_lon = _compute_resolution(reader, geo)
        print(f"Output resolution: {pixel_size_lat:.6f} x {pixel_size_lon:.6f} "
              f"deg (auto from {max(range_m, azimuth_m):.1f} m spacing)")

    # ------------------------------------------------------------------
    # Build output grid from image footprint
    # ------------------------------------------------------------------
    grid = OutputGrid.from_geolocation(
        geo,
        pixel_size_lat=pixel_size_lat,
        pixel_size_lon=pixel_size_lon,
    )
    print(f"Output grid: {grid.rows} x {grid.cols}")
    print(f"  Lat: [{grid.min_lat:.4f}, {grid.max_lat:.4f}]")
    print(f"  Lon: [{grid.min_lon:.4f}, {grid.max_lon:.4f}]")
    print()

    # ------------------------------------------------------------------
    # Read polarizations and compute products IN SLANT RANGE
    # ------------------------------------------------------------------
    # Key insight: do SAR-specific processing (magnitude, Pauli) in the
    # native slant range geometry, then orthorectify the real-valued
    # results. Resampling complex data causes phase cancellation that
    # destroys contrast and speckle texture.
    # ------------------------------------------------------------------
    print(f"Reading all four polarizations ({rows} x {cols})...")
    t0 = time.perf_counter()
    shh = reader.read_chip(0, rows, 0, cols, bands=[0])
    shv = reader.read_chip(0, rows, 0, cols, bands=[1])
    svh = reader.read_chip(0, rows, 0, cols, bands=[2])
    svv = reader.read_chip(0, rows, 0, cols, bands=[3])
    t_read = time.perf_counter() - t0
    print(f"  Read in {t_read:.2f} s")

    # HH magnitude dB (for display)
    hh_slant_db = _to_db(shh)

    # True quad-pol Pauli decomposition in slant range (complex domain).
    # Phase relationships between channels drive the separation of
    # surface vs. double-bounce scattering. All four S-matrix elements
    # are used, with proper 1/sqrt(2) normalization.
    print("Computing Pauli decomposition in slant range...")
    pauli = PauliDecomposition()
    components = pauli.decompose(shh, shv, svh, svv)
    db_components = pauli.to_db(components)

    pauli_r_slant = db_components['double_bounce']
    pauli_g_slant = db_components['volume']
    pauli_b_slant = db_components['surface']

    # Free complex arrays -- no longer needed
    del shh, shv, svh, svv, components

    # ------------------------------------------------------------------
    # Create orthorectifier and compute mapping
    # ------------------------------------------------------------------
    # Nearest-neighbor preserves SAR speckle texture and point targets
    # without smoothing. Bilinear is better for smoother products.
    ortho = Orthorectifier(geo, grid, interpolation='nearest')

    print("Computing inverse mapping (output -> source pixel)...")
    t0 = time.perf_counter()
    ortho.compute_mapping()
    t_map = time.perf_counter() - t0
    print(f"  Mapping computed in {t_map:.2f} s")

    n_valid = np.sum(ortho._valid_mask)
    n_total = grid.rows * grid.cols
    print(f"  Valid pixels: {n_valid:,} / {n_total:,} "
          f"({100 * n_valid / n_total:.1f}%)")
    print()

    # ------------------------------------------------------------------
    # Orthorectify the real-valued products (not complex)
    # ------------------------------------------------------------------
    nodata = np.nan

    print("Orthorectifying HH dB...")
    t0 = time.perf_counter()
    hh_ortho_db = ortho.apply(hh_slant_db, nodata=nodata)
    t_hh = time.perf_counter() - t0
    print(f"  Done in {t_hh:.2f} s")

    print("Orthorectifying Pauli components...")
    t0 = time.perf_counter()
    pauli_r_ortho = ortho.apply(pauli_r_slant, nodata=nodata)
    pauli_g_ortho = ortho.apply(pauli_g_slant, nodata=nodata)
    pauli_b_ortho = ortho.apply(pauli_b_slant, nodata=nodata)
    t_pauli = time.perf_counter() - t0
    print(f"  Done in {t_pauli:.2f} s")
    print()

    # ------------------------------------------------------------------
    # Prepare display arrays
    # ------------------------------------------------------------------
    # Slant range HH dB contrast
    slant_vmin = np.nanpercentile(hh_slant_db, 2)
    slant_vmax = np.nanpercentile(hh_slant_db, 98)

    # Ortho HH dB contrast (only from valid pixels)
    valid_mask = np.isfinite(hh_ortho_db)
    if np.any(valid_mask):
        ortho_vmin = np.nanpercentile(hh_ortho_db[valid_mask], 2)
        ortho_vmax = np.nanpercentile(hh_ortho_db[valid_mask], 98)
    else:
        ortho_vmin, ortho_vmax = slant_vmin, slant_vmax

    # Ortho Pauli RGB
    pauli_rgb = np.dstack([
        _normalize(pauli_r_ortho, valid_mask),
        _normalize(pauli_g_ortho, valid_mask),
        _normalize(pauli_b_ortho, valid_mask),
    ])
    pauli_rgb[~valid_mask] = 0

    # ------------------------------------------------------------------
    # Output geolocation metadata
    # ------------------------------------------------------------------
    out_meta = ortho.get_output_geolocation_metadata()
    print("Output geolocation:")
    print(f"  CRS:    {out_meta['crs']}")
    print(f"  Bounds: {out_meta['bounds']}")
    print(f"  Size:   {out_meta['rows']} x {out_meta['cols']}")
    print()

    # ------------------------------------------------------------------
    # Plot: 3 panels
    # ------------------------------------------------------------------
    fig, (ax_slant, ax_ortho, ax_pauli) = plt.subplots(
        1, 3, figsize=(20, 8),
    )

    # Panel 1: Original slant range HH dB
    ax_slant.imshow(
        hh_slant_db, cmap="gray", vmin=slant_vmin, vmax=slant_vmax,
        aspect="auto", interpolation="nearest",
    )
    ax_slant.set_title("Original (Slant Range)\nHH Magnitude (dB)", fontsize=9)
    ax_slant.set_xlabel("Column (range)")
    ax_slant.set_ylabel("Row (azimuth)")

    # Panel 2: Orthorectified HH dB with geographic axes
    extent_ortho = [grid.min_lon, grid.max_lon, grid.min_lat, grid.max_lat]
    ax_ortho.imshow(
        hh_ortho_db, cmap="gray", vmin=ortho_vmin, vmax=ortho_vmax,
        aspect="auto", interpolation="nearest", extent=extent_ortho,
    )
    ax_ortho.set_title("Orthorectified\nHH Magnitude (dB)", fontsize=9)
    ax_ortho.set_xlabel("Longitude (deg)")
    ax_ortho.set_ylabel("Latitude (deg)")

    # Panel 3: Orthorectified Pauli RGB
    ax_pauli.imshow(
        pauli_rgb, aspect="auto", interpolation="nearest",
        extent=extent_ortho,
    )
    ax_pauli.set_title(
        "Orthorectified Pauli RGB (quad-pol)\n"
        "R=double-bounce  G=volume  B=surface",
        fontsize=9,
    )
    ax_pauli.set_xlabel("Longitude (deg)")
    ax_pauli.set_ylabel("Latitude (deg)")

    fig.suptitle(
        f"BIOMASS Orthorectification  |  {product_path.name}\n"
        f"Orbit {reader.metadata.get('orbit_number', '?')}  |  "
        f"{reader.metadata.get('start_time', '?')[:10]}  |  "
        f"Grid {grid.rows}x{grid.cols} @ "
        f"{pixel_size_lat:.6f}\u00b0 x {pixel_size_lon:.6f}\u00b0",
        fontsize=11, fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    reader.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    """Parse command-line arguments."""
    args = sys.argv[1:]
    save_path = None
    product_path = None
    resolution = None

    i = 0
    while i < len(args):
        if args[i] == "--help":
            print("BIOMASS Orthorectification Example")
            print()
            print("Usage:")
            print("  python ortho_biomass.py                  # Most recent product")
            print("  python ortho_biomass.py <product_path>   # Specific product")
            print("  python ortho_biomass.py --save [path]    # Save figure to file")
            print("  python ortho_biomass.py --res 0.0002     # Custom resolution (deg)")
            print()
            print(f"Data directory: {DATA_DIR}")
            print()
            print("Processing notes:")
            print("  - Magnitude/Pauli computed in slant range, then orthorectified")
            print("  - Nearest-neighbor resampling to preserve SAR speckle texture")
            print("  - Default resolution auto-computed from metadata pixel spacings")
            sys.exit(0)
        elif args[i] == "--save":
            save_path = Path(args[i + 1]) if i + 1 < len(args) else Path("biomass_ortho.png")
            i += 2
            continue
        elif args[i] == "--res":
            resolution = float(args[i + 1])
            i += 2
            continue
        else:
            product_path = Path(args[i])
            i += 1
            continue
        i += 1

    return product_path, save_path, resolution


if __name__ == "__main__":
    prod_path, sv_path, res = _parse_args()

    if prod_path is None:
        prod_path = find_latest_product(DATA_DIR)

    ortho_biomass(prod_path, save_path=sv_path, resolution=res)
