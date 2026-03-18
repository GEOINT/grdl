# -*- coding: utf-8 -*-
"""
BIOMASS Orthorectification Example — Accelerated Center Chip.

Loads a BIOMASS L1A SCS product (slant range geometry), extracts a center
chip using ``data_prep.ChipExtractor``, computes magnitude and Pauli
decomposition in slant range, then orthorectifies to a WGS-84 grid
using the accelerated ``OrthoPipeline``.

Key processing choices:
  - Center chip extracted via ``ChipExtractor`` (default 4096x4096).
  - SAR-specific processing (magnitude, Pauli) done in slant range
    *before* orthorectification — resampling complex data causes phase
    cancellation that destroys contrast and speckle texture.
  - Accelerated multi-backend resampling (numba/torch/scipy).

Shows three panels:
  1. Original HH magnitude (dB) in slant range geometry
  2. Orthorectified HH magnitude (dB) on a geographic grid
  3. Orthorectified Pauli RGB (R=double-bounce, G=volume, B=surface)

Usage:
  python ortho_biomass.py                 # Most recent product, 4k chip
  python ortho_biomass.py <product_path>
  python ortho_biomass.py --chip 2048
  python ortho_biomass.py --save [path]
  python ortho_biomass.py --res 0.0002
  python ortho_biomass.py --interp nearest
  python ortho_biomass.py --help

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
2026-03-08
"""

# Standard library
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

# Third-party
import numpy as np

# Matplotlib -- set backend before importing pyplot
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt  # noqa: E402

# GRDL
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from grdl.IO import BIOMASSL1Reader
from grdl.data_prep import ChipExtractor
from grdl.geolocation.sar.gcp import GCPGeolocation
from grdl.image_processing import PauliDecomposition
from grdl.image_processing.ortho import OrthoPipeline, detect_backend


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("/Volumes/PRO-G40/SAR_DATA/BIOMASS")


# ---------------------------------------------------------------------------
# Chip geolocation wrapper
# ---------------------------------------------------------------------------

class _ChipGeolocationWrapper:
    """Offset chip-local coords to full-image coords for geolocation."""

    def __init__(self, geo, row_offset, col_offset, chip_rows, chip_cols):
        self._geo = geo
        self._row_off = row_offset
        self._col_off = col_offset
        self.shape = (chip_rows, chip_cols)

    def image_to_latlon(self, row, col, height=0.0):
        return self._geo.image_to_latlon(
            row + self._row_off, col + self._col_off, height=height,
        )

    def latlon_to_image(self, lat, lon, height=0.0):
        r, c = self._geo.latlon_to_image(lat, lon, height=height)
        return r - self._row_off, c - self._col_off

    def get_bounds(self):
        corners_row = np.array([0.0, 0.0, self.shape[0], self.shape[0]])
        corners_col = np.array([0.0, self.shape[1], 0.0, self.shape[1]])
        lats, lons, _ = self.image_to_latlon(corners_row, corners_col)
        return (
            float(np.min(lons)), float(np.min(lats)),
            float(np.max(lons)), float(np.max(lats)),
        )

    def get_footprint(self):
        return self._geo.get_footprint()


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def ortho_biomass(
    product_path: Path,
    chip_size: int = 4096,
    save_path: Path = None,
    resolution: float = None,
    interpolation: str = 'nearest',
) -> None:
    """Orthorectify a BIOMASS center chip and display results.

    Parameters
    ----------
    product_path : Path
        Path to the BIOMASS product directory.
    chip_size : int
        Center chip size in pixels (default 4096).
    save_path : Path, optional
        If provided, save figure to this path.
    resolution : float, optional
        Output pixel size in degrees. If None, auto-computed from
        metadata pixel spacings.
    interpolation : str, default='nearest'
        Resampling method: 'nearest', 'bilinear', or 'bicubic'.
    """
    timings = {}
    backend = detect_backend()
    print(f"Loading: {product_path.name}")
    print(f"Backend: {backend}")
    print()

    # ------------------------------------------------------------------
    # Open reader and extract metadata
    # ------------------------------------------------------------------
    reader = BIOMASSL1Reader(product_path)
    geo_info = {
        'gcps': reader.metadata['gcps'],
        'crs': reader.metadata.get('crs', 'WGS84'),
    }
    geo_full = GCPGeolocation.from_dict(geo_info, reader.metadata)
    rows, cols = geo_full.shape
    pols = reader.metadata["polarizations"]

    range_m = reader.metadata.get('range_pixel_spacing', 0)
    azimuth_m = reader.metadata.get('azimuth_pixel_spacing', 0)

    print(f"  Full size:  {rows} x {cols}")
    print(f"  Pols:       {pols}")
    print(f"  Orbit:      {reader.metadata.get('orbit_number', '?')}")
    print(f"  Spacing:    {range_m:.2f} m (range) x "
          f"{azimuth_m:.2f} m (azimuth)")
    print()

    # ------------------------------------------------------------------
    # Plan center chip
    # ------------------------------------------------------------------
    extractor = ChipExtractor(nrows=rows, ncols=cols)
    region = extractor.chip_at_point(
        rows // 2, cols // 2,
        row_width=chip_size, col_width=chip_size,
    )
    chip_rows = region.row_end - region.row_start
    chip_cols = region.col_end - region.col_start
    print(f"  Chip:       [{region.row_start}:{region.row_end}, "
          f"{region.col_start}:{region.col_end}]  "
          f"({chip_rows} x {chip_cols})")

    # Wrap geolocation for chip
    geo = _ChipGeolocationWrapper(
        geo_full, region.row_start, region.col_start,
        chip_rows, chip_cols,
    )

    # ------------------------------------------------------------------
    # Read all four polarizations from center chip
    # ------------------------------------------------------------------
    print(f"Reading quad-pol center chip ({chip_rows} x {chip_cols})...")
    t0 = time.perf_counter()
    r0, r1 = region.row_start, region.row_end
    c0, c1 = region.col_start, region.col_end
    shh = reader.read_chip(r0, r1, c0, c1, bands=[0])
    shv = reader.read_chip(r0, r1, c0, c1, bands=[1])
    svh = reader.read_chip(r0, r1, c0, c1, bands=[2])
    svv = reader.read_chip(r0, r1, c0, c1, bands=[3])
    timings['read'] = time.perf_counter() - t0
    print(f"  Read in {timings['read']:.2f} s")

    # ------------------------------------------------------------------
    # SAR processing in slant range
    # ------------------------------------------------------------------
    # HH magnitude dB
    t0 = time.perf_counter()
    hh_slant_db = _to_db(shh).astype(np.float32)

    # Pauli decomposition (complex domain, all four channels)
    print("Computing Pauli decomposition in slant range...")
    pauli = PauliDecomposition()
    components = pauli.decompose(shh, shv, svh, svv)
    db_components = pauli.to_db(components)

    pauli_r_slant = db_components['double_bounce'].astype(np.float32)
    pauli_g_slant = db_components['volume'].astype(np.float32)
    pauli_b_slant = db_components['surface'].astype(np.float32)
    del shh, shv, svh, svv, components, db_components
    timings['sar_proc'] = time.perf_counter() - t0
    print(f"  Done in {timings['sar_proc']:.2f} s")

    reader.close()

    # ------------------------------------------------------------------
    # Build and run ortho pipeline — HH dB
    # ------------------------------------------------------------------
    pipeline = (
        OrthoPipeline()
        .with_source_array(hh_slant_db)
        .with_metadata(reader.metadata)
        .with_geolocation(geo)
        .with_interpolation(interpolation)
        .with_nodata(np.nan)
    )

    if resolution is not None:
        pipeline = pipeline.with_resolution(resolution, resolution)
        print(f"  Resolution: {resolution:.6f} deg (user)")
    else:
        print("  Resolution: auto (from BIOMASS spacing)")

    print()
    print("Running ortho pipeline (HH dB)...")
    t0 = time.perf_counter()
    result_hh = pipeline.run()
    timings['ortho_hh'] = time.perf_counter() - t0
    grid = result_hh.output_grid
    print(f"  {grid.rows}x{grid.cols} in {timings['ortho_hh']:.2f} s")

    # ------------------------------------------------------------------
    # Orthorectify Pauli components (reuse grid settings)
    # ------------------------------------------------------------------
    print("Orthorectifying Pauli components...")
    t0 = time.perf_counter()

    def _ortho_band(data):
        p = (
            OrthoPipeline()
            .with_source_array(data)
            .with_geolocation(geo)
            .with_interpolation(interpolation)
            .with_nodata(np.nan)
            .with_output_grid(grid)
        )
        return p.run().data

    pauli_r_ortho = _ortho_band(pauli_r_slant)
    pauli_g_ortho = _ortho_band(pauli_g_slant)
    pauli_b_ortho = _ortho_band(pauli_b_slant)
    timings['ortho_pauli'] = time.perf_counter() - t0
    print(f"  Done in {timings['ortho_pauli']:.2f} s")

    n_valid = np.sum(np.isfinite(result_hh.data))
    n_total = grid.rows * grid.cols
    print(f"  Valid:      {n_valid:,} / {n_total:,} "
          f"({100 * n_valid / n_total:.1f}%)")

    # ------------------------------------------------------------------
    # Timing summary
    # ------------------------------------------------------------------
    print()
    print("Timing summary:")
    total = 0.0
    for stage, dt in timings.items():
        print(f"  {stage:<16s} {dt:6.2f} s")
        total += dt
    print(f"  {'TOTAL':<16s} {total:6.2f} s")
    print()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    slant_vmin = np.nanpercentile(hh_slant_db, 2)
    slant_vmax = np.nanpercentile(hh_slant_db, 98)

    valid_mask = np.isfinite(result_hh.data)
    if np.any(valid_mask):
        ortho_vmin = np.nanpercentile(result_hh.data[valid_mask], 2)
        ortho_vmax = np.nanpercentile(result_hh.data[valid_mask], 98)
    else:
        ortho_vmin, ortho_vmax = slant_vmin, slant_vmax

    pauli_rgb = np.dstack([
        _normalize(pauli_r_ortho, valid_mask),
        _normalize(pauli_g_ortho, valid_mask),
        _normalize(pauli_b_ortho, valid_mask),
    ])
    pauli_rgb[~valid_mask] = 0

    fig, (ax_slant, ax_ortho, ax_pauli) = plt.subplots(
        1, 3, figsize=(21, 8),
    )

    # Panel 1: Slant range HH dB
    ax_slant.imshow(
        hh_slant_db, cmap="gray", vmin=slant_vmin, vmax=slant_vmax,
        aspect="auto", interpolation="nearest",
    )
    ax_slant.set_title(
        f"Slant Range (chip {chip_rows}x{chip_cols})\nHH Magnitude (dB)",
        fontsize=9,
    )
    ax_slant.set_xlabel("Column (range)")
    ax_slant.set_ylabel("Row (azimuth)")

    # Panel 2: Ortho HH dB
    extent = [grid.min_lon, grid.max_lon, grid.min_lat, grid.max_lat]
    ax_ortho.imshow(
        result_hh.data, cmap="gray", vmin=ortho_vmin, vmax=ortho_vmax,
        aspect="auto", interpolation="nearest", extent=extent,
    )
    ax_ortho.set_title("Orthorectified\nHH Magnitude (dB)", fontsize=9)
    ax_ortho.set_xlabel("Longitude (deg)")
    ax_ortho.set_ylabel("Latitude (deg)")

    # Panel 3: Ortho Pauli RGB
    ax_pauli.imshow(
        pauli_rgb, aspect="auto", interpolation="nearest", extent=extent,
    )
    ax_pauli.set_title(
        "Orthorectified Pauli RGB (quad-pol)\n"
        "R=double-bounce  G=volume  B=surface",
        fontsize=9,
    )
    ax_pauli.set_xlabel("Longitude (deg)")
    ax_pauli.set_ylabel("Latitude (deg)")

    fig.suptitle(
        f"BIOMASS Ortho  |  {product_path.name}\n"
        f"Orbit {reader.metadata.get('orbit_number', '?')}  |  "
        f"Chip {chip_rows}x{chip_cols}  |  "
        f"Grid {grid.rows}x{grid.cols} @ "
        f"{grid.pixel_size_lat:.6f}\u00b0  |  "
        f"{interpolation}  |  {backend}",
        fontsize=10, fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    """Parse command-line arguments."""
    args = sys.argv[1:]
    product_path = None
    chip_size = 4096
    save_path = None
    resolution = None
    interpolation = 'nearest'

    i = 0
    while i < len(args):
        if args[i] == "--help":
            print("BIOMASS Orthorectification — Accelerated Center Chip")
            print()
            print("Usage:")
            print("  python ortho_biomass.py                  # Most recent product")
            print("  python ortho_biomass.py <product_path>")
            print("  python ortho_biomass.py --chip 2048")
            print("  python ortho_biomass.py --save [path]")
            print("  python ortho_biomass.py --res 0.0002")
            print()
            print("Options:")
            print("  --chip <N>      Center chip size (default: 4096)")
            print("  --save [path]   Save figure (default: biomass_ortho.png)")
            print("  --res <deg>     Output resolution in degrees")
            print("  --interp <m>    Interpolation: nearest, bilinear, bicubic")
            print()
            print(f"Data directory: {DATA_DIR}")
            sys.exit(0)
        elif args[i] == "--chip":
            chip_size = int(args[i + 1])
            i += 2
        elif args[i] == "--save":
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                save_path = Path(args[i + 1])
                i += 2
            else:
                save_path = Path("biomass_ortho.png")
                i += 1
        elif args[i] == "--res":
            resolution = float(args[i + 1])
            i += 2
        elif args[i] == "--interp":
            interpolation = args[i + 1]
            i += 2
        else:
            product_path = Path(args[i])
            i += 1

    return product_path, chip_size, save_path, resolution, interpolation


if __name__ == "__main__":
    prod_path, cs, sv_path, res, interp = _parse_args()

    if prod_path is None:
        prod_path = find_latest_product(DATA_DIR)

    ortho_biomass(
        prod_path, chip_size=cs, save_path=sv_path,
        resolution=res, interpolation=interp,
    )
