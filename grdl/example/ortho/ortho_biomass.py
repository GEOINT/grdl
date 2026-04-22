# -*- coding: utf-8 -*-
"""
BIOMASS Orthorectification Example — Accelerated Center Chip.

Loads a BIOMASS L1A SCS product (slant range geometry), extracts a center
chip using ``data_prep.ChipExtractor``, computes magnitude and Pauli
decomposition in slant range, then orthorectifies to a WGS-84 grid
using the accelerated ``OrthoBuilder``.

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
plotly
rasterio
scipy

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

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
2026-03-27
"""

# Standard library
import sys
import time
from pathlib import Path
from typing import Optional

# Third-party
import numpy as np

# GRDL
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from grdl.IO import BIOMASSL1Reader
from grdl.data_prep import ChipExtractor
from grdl.geolocation.chip import ChipGeolocation
from grdl.geolocation.sar.gcp import GCPGeolocation
from grdl.image_processing import PauliDecomposition
from grdl.image_processing.ortho import orthorectify, detect_backend


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("/Volumes/PRO-G40/SAR_DATA/BIOMASS")


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
    geo = ChipGeolocation(
        geo_full,
        row_offset=region.row_start,
        col_offset=region.col_start,
        shape=(chip_rows, chip_cols),
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
    hh_kwargs = dict(
        geolocation=geo,
        source_array=hh_slant_db,
        metadata=reader.metadata,
        interpolation=interpolation,
        nodata=np.nan,
    )
    if resolution is not None:
        hh_kwargs['resolution'] = (resolution, resolution)
        print(f"  Resolution: {resolution:.6f} deg (user)")
    else:
        print("  Resolution: auto (from BIOMASS spacing)")

    print()
    print("Running ortho pipeline (HH dB)...")
    t0 = time.perf_counter()
    result_hh = orthorectify(**hh_kwargs)
    timings['ortho_hh'] = time.perf_counter() - t0
    grid = result_hh.output_grid
    print(f"  {grid.rows}x{grid.cols} in {timings['ortho_hh']:.2f} s")

    # ------------------------------------------------------------------
    # Orthorectify Pauli components (reuse grid settings)
    # ------------------------------------------------------------------
    print("Orthorectifying Pauli components...")
    t0 = time.perf_counter()

    def _ortho_band(data):
        return orthorectify(
            geolocation=geo,
            source_array=data,
            interpolation=interpolation,
            nodata=np.nan,
            output_grid=grid,
        ).data

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
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    slant_vmin = float(np.nanpercentile(hh_slant_db, 2))
    slant_vmax = float(np.nanpercentile(hh_slant_db, 98))

    valid_mask = np.isfinite(result_hh.data)
    if np.any(valid_mask):
        ortho_vmin = float(np.nanpercentile(result_hh.data[valid_mask], 2))
        ortho_vmax = float(np.nanpercentile(result_hh.data[valid_mask], 98))
    else:
        ortho_vmin, ortho_vmax = slant_vmin, slant_vmax

    # Build Pauli RGB as uint8 for go.Image
    pauli_rgb = np.dstack([
        _normalize(pauli_r_ortho, valid_mask),
        _normalize(pauli_g_ortho, valid_mask),
        _normalize(pauli_b_ortho, valid_mask),
    ])
    pauli_rgb[~valid_mask] = 0
    pauli_rgb_u8 = (pauli_rgb * 255).astype(np.uint8)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f"Slant Range (chip {chip_rows}x{chip_cols})<br>HH Magnitude (dB)",
            "Orthorectified<br>HH Magnitude (dB)",
            "Orthorectified Pauli RGB (quad-pol)<br>"
            "R=double-bounce  G=volume  B=surface",
        ],
    )

    # Panel 1: Slant range HH dB
    fig.add_trace(
        go.Heatmap(z=hh_slant_db, zmin=slant_vmin, zmax=slant_vmax,
                   colorscale='Gray', showscale=False),
        row=1, col=1,
    )
    fig.update_xaxes(title_text="Column (range)", row=1, col=1)
    fig.update_yaxes(title_text="Row (azimuth)", autorange='reversed',
                     row=1, col=1)

    # Panel 2: Ortho HH dB (north-up)
    fig.add_trace(
        go.Heatmap(z=np.flipud(result_hh.data),
                   x0=grid.min_lon, dx=grid.pixel_size_lon,
                   y0=grid.min_lat, dy=grid.pixel_size_lat,
                   zmin=ortho_vmin, zmax=ortho_vmax,
                   colorscale='Gray', showscale=False),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Longitude (deg)", row=1, col=2)
    fig.update_yaxes(title_text="Latitude (deg)", row=1, col=2)

    # Panel 3: Ortho Pauli RGB (north-up)
    # Use go.Image with geographic axis mapping
    fig.add_trace(
        go.Image(z=np.flipud(pauli_rgb_u8)),
        row=1, col=3,
    )
    fig.update_xaxes(title_text="Longitude (deg)", row=1, col=3)
    fig.update_yaxes(title_text="Latitude (deg)", row=1, col=3)

    title = (
        f"BIOMASS Ortho  |  {product_path.name}  |  "
        f"Orbit {reader.metadata.get('orbit_number', '?')}  |  "
        f"Chip {chip_rows}x{chip_cols}  |  "
        f"Grid {grid.rows}x{grid.cols} @ "
        f"{grid.pixel_size_lat:.6f}\u00b0  |  "
        f"{interpolation}  |  {backend}"
    )
    fig.update_layout(title_text=title, width=2100, height=700)

    if save_path:
        fig.write_image(str(save_path), scale=2)
        print(f"Saved to {save_path}")
    else:
        fig.show()


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
