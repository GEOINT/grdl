# -*- coding: utf-8 -*-
"""
Combined Orthorectification Example — SICD + SIDD with ENU.

Auto-detects whether the input is SICD (complex SAR) or SIDD (derived
product), extracts a center chip, and orthorectifies to both WGS-84
and ENU grids.  Uses accelerated resampling backends and
``data_prep.ChipExtractor`` for efficient chip planning.

Shows four panels:
  1. Source data (slant range / native geometry)
  2. Orthorectified WGS-84 (geographic degrees)
  3. Orthorectified ENU (local meters)
  4. Geospatial info panel (footprint, reference point, grid stats)

Usage:
  python ortho_combined.py <nitf_file>
  python ortho_combined.py <nitf_file> --chip 2048
  python ortho_combined.py <nitf_file> --dem /Volumes/PRO-G40/terrain/FABDEM
  python ortho_combined.py <nitf_file> --enu 1.0
  python ortho_combined.py --help

Dependencies
------------
plotly
scipy
sarkit (or sarpy)

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
2026-03-08

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

# GRDL
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from grdl.data_prep import ChipExtractor
from grdl.image_processing.ortho import OrthoBuilder, detect_backend


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

def _find_fabdem_tile(lat, lon, fabdem_root):
    """Find the FABDEM GeoTIFF tile covering the given lat/lon."""
    fabdem = Path(fabdem_root)
    if not fabdem.exists():
        return None
    lat_floor = int(np.floor(lat))
    lon_floor = int(np.floor(lon))
    ns = 'N' if lat_floor >= 0 else 'S'
    ew = 'E' if lon_floor >= 0 else 'W'
    tile_name = f"{ns}{abs(lat_floor):02d}{ew}{abs(lon_floor):03d}"
    for tif in fabdem.rglob(f"{tile_name}*.tif"):
        return str(tif)
    return None


def _detect_format(filepath: Path) -> str:
    """Detect whether a NITF file is SICD or SIDD."""
    # Try SICD first (complex SAR)
    try:
        from grdl.IO.sar import SICDReader
        with SICDReader(filepath) as r:
            if r.metadata.format == 'SICD':
                return 'SICD'
    except Exception:
        pass

    # Try SIDD (derived product)
    try:
        from grdl.IO.sar import SIDDReader
        with SIDDReader(filepath) as r:
            if r.metadata.format == 'SIDD':
                return 'SIDD'
    except Exception:
        pass

    raise ValueError(f"Cannot identify {filepath.name} as SICD or SIDD")


def _load_sicd(filepath, chip_size):
    """Load SICD: open reader, plan chip, read, compute magnitude."""
    from grdl.IO.sar import SICDReader
    from grdl.geolocation.sar.sicd import SICDGeolocation

    timings = {}
    with SICDReader(filepath) as reader:
        meta = reader.metadata
        rows, cols = meta.rows, meta.cols

        mode = 'unknown'
        if meta.collection_info and meta.collection_info.radar_mode:
            mode = meta.collection_info.radar_mode.mode_type or 'unknown'

        print(f"  Format:     SICD ({mode})")
        print(f"  Full size:  {rows} x {cols}")
        print(f"  Backend:    {meta.backend}")
        if meta.grid:
            print(f"  Row ss:     {meta.grid.row.ss:.4f} m")
            print(f"  Col ss:     {meta.grid.col.ss:.4f} m")
        if meta.geo_data and meta.geo_data.scp and meta.geo_data.scp.llh:
            scp = meta.geo_data.scp.llh
            print(f"  SCP:        ({scp.lat:.4f}, {scp.lon:.4f})")

        # Plan chip
        extractor = ChipExtractor(nrows=rows, ncols=cols)
        region = extractor.chip_at_point(
            rows // 2, cols // 2,
            row_width=chip_size, col_width=chip_size,
        )
        chip_rows = region.row_end - region.row_start
        chip_cols = region.col_end - region.col_start
        print(f"  Chip:       {chip_rows} x {chip_cols}")

        # Geolocation
        geo_full = SICDGeolocation.from_reader(reader)
        geo = _ChipGeolocationWrapper(
            geo_full, region.row_start, region.col_start,
            chip_rows, chip_cols,
        )

        # Read chip
        t0 = time.perf_counter()
        complex_chip = reader.read_chip(
            region.row_start, region.row_end,
            region.col_start, region.col_end,
        )
        timings['read'] = time.perf_counter() - t0

    # Compute magnitude in slant range
    t0 = time.perf_counter()
    source = np.abs(complex_chip).astype(np.float32)
    del complex_chip
    timings['magnitude'] = time.perf_counter() - t0

    return source, geo, meta, timings, 'SICD'


def _load_sidd(filepath, chip_size):
    """Load SIDD: open reader, plan chip, read."""
    from grdl.IO.sar import SIDDReader
    from grdl.geolocation.sar.sidd import SIDDGeolocation

    timings = {}
    with SIDDReader(filepath) as reader:
        meta = reader.metadata
        rows, cols = meta.rows, meta.cols

        print(f"  Format:     SIDD")
        print(f"  Full size:  {rows} x {cols}")
        print(f"  Dtype:      {meta.dtype}")
        print(f"  Backend:    {meta.backend}")
        if meta.measurement:
            print(f"  Projection: {meta.measurement.projection_type}")
        if meta.geo_data:
            gd = meta.geo_data
            if hasattr(gd, 'scp') and gd.scp is not None:
                llh = gd.scp.llh if hasattr(gd.scp, 'llh') else None
                if llh is not None:
                    print(f"  SCP:        ({llh.lat:.4f}, {llh.lon:.4f})")

        # Plan chip
        extractor = ChipExtractor(nrows=rows, ncols=cols)
        region = extractor.chip_at_point(
            rows // 2, cols // 2,
            row_width=chip_size, col_width=chip_size,
        )
        chip_rows = region.row_end - region.row_start
        chip_cols = region.col_end - region.col_start
        print(f"  Chip:       {chip_rows} x {chip_cols}")

        # Geolocation
        geo_full = SIDDGeolocation.from_reader(reader)
        geo = _ChipGeolocationWrapper(
            geo_full, region.row_start, region.col_start,
            chip_rows, chip_cols,
        )

        # Read chip
        t0 = time.perf_counter()
        chip_data = reader.read_chip(
            region.row_start, region.row_end,
            region.col_start, region.col_end,
        )
        timings['read'] = time.perf_counter() - t0

    source = chip_data.astype(np.float32)
    del chip_data
    return source, geo, meta, timings, 'SIDD'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def ortho_combined(
    filepath: Path,
    chip_size: int = 4096,
    save_path: Path = None,
    resolution: float = None,
    interpolation: str = 'nearest',
    dem_path: str = None,
    enu_pixel_m: float = 1.0,
) -> None:
    """Orthorectify a SAR image (SICD or SIDD) to WGS-84 and ENU.

    Parameters
    ----------
    filepath : Path
        Path to NITF file (SICD or SIDD).
    chip_size : int
        Center chip size in pixels (default 4096).
    save_path : Path, optional
        If provided, save figure to this path.
    resolution : float, optional
        Output pixel size in degrees.
    interpolation : str, default='nearest'
        Resampling method.
    dem_path : str, optional
        Path to FABDEM directory for terrain correction.
    enu_pixel_m : float, default=1.0
        ENU output pixel spacing in meters.
    """
    timings = {}
    backend = detect_backend()
    print(f"{'=' * 60}")
    print(f"Combined Ortho Demo  |  {filepath.name}")
    print(f"Accelerated backend: {backend}")
    print(f"{'=' * 60}")
    print()

    # ------------------------------------------------------------------
    # Detect format and load
    # ------------------------------------------------------------------
    fmt = _detect_format(filepath)
    if fmt == 'SICD':
        source, geo, meta, load_timings, label = _load_sicd(
            filepath, chip_size,
        )
    else:
        source, geo, meta, load_timings, label = _load_sidd(
            filepath, chip_size,
        )
    timings.update(load_timings)
    chip_rows, chip_cols = source.shape[:2]
    print()

    # ------------------------------------------------------------------
    # DEM
    # ------------------------------------------------------------------
    elev = None
    if dem_path is not None:
        dem_file = dem_path
        if Path(dem_path).is_dir():
            clat, clon, _ = geo.image_to_latlon(
                chip_rows / 2.0, chip_cols / 2.0,
            )
            tile = _find_fabdem_tile(float(clat), float(clon), dem_path)
            if tile:
                dem_file = tile
                print(f"  FABDEM:     {Path(tile).name}")
            else:
                dem_file = None

        if dem_file is not None:
            from grdl.geolocation.elevation.geotiff_dem import GeoTIFFDEM
            elev = GeoTIFFDEM(dem_file)

    # ------------------------------------------------------------------
    # WGS-84 ortho
    # ------------------------------------------------------------------
    wgs_pipeline = (
        OrthoBuilder()
        .with_source_array(source)
        .with_geolocation(geo)
        .with_interpolation(interpolation)
        .with_nodata(np.nan)
    )
    if elev is not None:
        wgs_pipeline = wgs_pipeline.with_elevation(elev)

    if resolution is not None:
        wgs_pipeline = wgs_pipeline.with_resolution(resolution, resolution)
    elif fmt == 'SICD':
        wgs_pipeline = wgs_pipeline.with_metadata(meta)
    else:
        # SIDD: derive from footprint
        min_lon, min_lat, max_lon, max_lat = geo.get_bounds()
        auto_lat = (max_lat - min_lat) / chip_rows
        auto_lon = (max_lon - min_lon) / chip_cols
        wgs_pipeline = wgs_pipeline.with_resolution(auto_lat, auto_lon)

    print("Running WGS-84 ortho...")
    t0 = time.perf_counter()
    result_wgs = wgs_pipeline.run()
    timings['ortho_wgs84'] = time.perf_counter() - t0
    grid = result_wgs.output_grid
    print(f"  {grid.rows}x{grid.cols} in {timings['ortho_wgs84']:.2f} s")

    # ------------------------------------------------------------------
    # ENU ortho
    # ------------------------------------------------------------------
    enu_pipeline = (
        OrthoBuilder()
        .with_source_array(source)
        .with_geolocation(geo)
        .with_interpolation(interpolation)
        .with_nodata(np.nan)
        .with_enu_grid(pixel_size_m=enu_pixel_m)
    )
    if elev is not None:
        enu_pipeline = enu_pipeline.with_elevation(elev)

    print(f"Running ENU ortho ({enu_pixel_m:.1f} m)...")
    t0 = time.perf_counter()
    result_enu = enu_pipeline.run()
    timings['ortho_enu'] = time.perf_counter() - t0
    eg = result_enu.output_grid
    print(f"  {eg.rows}x{eg.cols} in {timings['ortho_enu']:.2f} s")

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

    # ------------------------------------------------------------------
    # Plot: 3 image panels + info annotation
    # ------------------------------------------------------------------
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # Percentile stretch
    src_vmin = float(np.nanpercentile(source, 2))
    src_vmax = float(np.nanpercentile(source, 98))

    wgs_valid = np.isfinite(result_wgs.data)
    if np.any(wgs_valid):
        wgs_vmin = float(np.nanpercentile(result_wgs.data[wgs_valid], 2))
        wgs_vmax = float(np.nanpercentile(result_wgs.data[wgs_valid], 98))
    else:
        wgs_vmin, wgs_vmax = src_vmin, src_vmax

    enu_valid = np.isfinite(result_enu.data)
    if np.any(enu_valid):
        enu_vmin = float(np.nanpercentile(result_enu.data[enu_valid], 2))
        enu_vmax = float(np.nanpercentile(result_enu.data[enu_valid], 98))
    else:
        enu_vmin, enu_vmax = src_vmin, src_vmax

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f"{label} Source<br>(chip {chip_rows}x{chip_cols})",
            f"WGS-84 ({grid.rows}x{grid.cols})<br>"
            f"{grid.pixel_size_lat:.6f}\u00b0",
            f"ENU ({eg.rows}x{eg.cols})<br>{enu_pixel_m:.1f} m/px",
        ],
    )

    # Panel 1: Source
    fig.add_trace(
        go.Heatmap(z=source, zmin=src_vmin, zmax=src_vmax,
                   colorscale='Gray', showscale=False),
        row=1, col=1,
    )
    fig.update_xaxes(title_text="Column", row=1, col=1)
    fig.update_yaxes(title_text="Row", autorange='reversed', row=1, col=1)

    # Panel 2: WGS-84 (north-up)
    fig.add_trace(
        go.Heatmap(z=np.flipud(result_wgs.data),
                   x0=grid.min_lon, dx=grid.pixel_size_lon,
                   y0=grid.min_lat, dy=grid.pixel_size_lat,
                   zmin=wgs_vmin, zmax=wgs_vmax,
                   colorscale='Gray', showscale=False),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Longitude (deg)", row=1, col=2)
    fig.update_yaxes(title_text="Latitude (deg)", row=1, col=2)

    # Panel 3: ENU (north-up)
    fig.add_trace(
        go.Heatmap(z=np.flipud(result_enu.data),
                   x0=eg.min_east, dx=eg.pixel_size_east,
                   y0=eg.min_north, dy=eg.pixel_size_north,
                   zmin=enu_vmin, zmax=enu_vmax,
                   colorscale='Gray', showscale=False),
        row=1, col=3,
    )
    fig.update_xaxes(title_text="East (m)", row=1, col=3)
    fig.update_yaxes(title_text="North (m)", row=1, col=3)

    # Info annotation
    n_wgs = int(np.sum(wgs_valid))
    n_enu = int(np.sum(enu_valid))
    info_lines = [
        f"File: {filepath.name}  |  Format: {label}  |  "
        f"Backend: {backend}  |  Interp: {interpolation}  |  "
        f"DEM: {'yes' if elev else 'none'}",
        f"WGS-84: {grid.rows}x{grid.cols} "
        f"({grid.pixel_size_lat:.6f}\u00b0) "
        f"Valid: {100 * n_wgs / max(1, grid.rows * grid.cols):.1f}%  |  "
        f"ENU: {eg.rows}x{eg.cols} ({enu_pixel_m:.1f} m/px) "
        f"Valid: {100 * n_enu / max(1, eg.rows * eg.cols):.1f}%  |  "
        f"Total: {total:.2f} s",
    ]

    fig.update_layout(
        title_text=(
            f"Combined Ortho: {filepath.name}  |  {label}  |  {backend}"
        ),
        width=2100, height=700,
    )

    # Add info as annotation below plots
    fig.add_annotation(
        text="<br>".join(info_lines),
        xref="paper", yref="paper",
        x=0.0, y=-0.08, showarrow=False,
        font=dict(size=10, family="monospace"),
        align="left",
    )

    if save_path:
        fig.write_image(str(save_path), scale=2)
        print(f"\nSaved to {save_path}")
    else:
        fig.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    """Parse command-line arguments."""
    args = sys.argv[1:]
    filepath = None
    chip_size = 4096
    save_path = None
    resolution = None
    interpolation = 'nearest'
    dem_path = None
    enu_pixel_m = 1.0

    i = 0
    while i < len(args):
        if args[i] == "--help":
            print("Combined Ortho Demo — SICD + SIDD with WGS-84 + ENU")
            print()
            print("Usage:")
            print("  python ortho_combined.py <nitf_file>")
            print("  python ortho_combined.py <nitf_file> --chip 2048")
            print("  python ortho_combined.py <nitf_file> --dem /path/to/fabdem")
            print("  python ortho_combined.py <nitf_file> --enu 0.5")
            print()
            print("Options:")
            print("  --chip <N>      Center chip size (default: 4096)")
            print("  --save [path]   Save figure (default: ortho_combined.png)")
            print("  --res <deg>     Output resolution in degrees")
            print("  --interp <m>    Interpolation: nearest, bilinear, bicubic")
            print("  --dem <path>    FABDEM directory for terrain correction")
            print("  --enu <m>       ENU pixel size in meters (default: 1.0)")
            sys.exit(0)
        elif args[i] == "--chip":
            chip_size = int(args[i + 1])
            i += 2
        elif args[i] == "--save":
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                save_path = Path(args[i + 1])
                i += 2
            else:
                save_path = Path("ortho_combined.png")
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
        elif args[i] == "--enu":
            enu_pixel_m = float(args[i + 1])
            i += 2
        else:
            filepath = Path(args[i])
            i += 1

    return (filepath, chip_size, save_path, resolution,
            interpolation, dem_path, enu_pixel_m)


if __name__ == "__main__":
    (fp, cs, sv, res, interp, dem, enu) = _parse_args()

    if fp is None:
        print("Error: NITF file path required (SICD or SIDD).")
        print("Usage: python ortho_combined.py <nitf_file> [--chip N] [--dem path]")
        sys.exit(1)

    ortho_combined(
        fp, chip_size=cs, save_path=sv, resolution=res,
        interpolation=interp, dem_path=dem, enu_pixel_m=enu,
    )
