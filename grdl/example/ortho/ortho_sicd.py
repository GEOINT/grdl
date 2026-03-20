# -*- coding: utf-8 -*-
"""
SICD Orthorectification Example — Accelerated Center Chip.

Loads a SICD complex SAR image, extracts a center chip using
``data_prep.ChipExtractor``, computes linear magnitude in slant range,
then orthorectifies to a regular WGS84 geographic grid using the
accelerated ``OrthoBuilder``.  Also demonstrates ENU output mode.

Key processing choices:
  - Center chip extracted via ``ChipExtractor`` (default 4096x4096).
  - Magnitude computed in slant range *before* orthorectification
    (resampling complex data causes phase cancellation).
  - Geolocation wrapper offsets chip-local pixel coords to full-image
    coords so the existing ``SICDGeolocation`` works unmodified.
  - Accelerated multi-backend resampling (torch/numba/scipy).

Shows three panels:
  1. Original magnitude in slant range geometry (chip)
  2. Orthorectified magnitude on a WGS-84 geographic grid
  3. Orthorectified magnitude on an ENU grid (meters)

Usage:
  python ortho_sicd.py <sicd_file>
  python ortho_sicd.py <sicd_file> --chip 2048
  python ortho_sicd.py <sicd_file> --dem /path/to/fabdem
  python ortho_sicd.py <sicd_file> --save [path]
  python ortho_sicd.py <sicd_file> --res 0.00005
  python ortho_sicd.py <sicd_file> --interp nearest
  python ortho_sicd.py --help

Dependencies
------------
plotly
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

from grdl.IO.sar import SICDReader
from grdl.data_prep import ChipExtractor
from grdl.geolocation.sar.sicd import SICDGeolocation
from grdl.image_processing.ortho import OrthoBuilder, detect_backend


# ---------------------------------------------------------------------------
# Chip geolocation wrapper
# ---------------------------------------------------------------------------

class _ChipGeolocationWrapper:
    """Offset chip-local coords to full-image coords for geolocation.

    Wraps a real ``Geolocation`` so that pixel (0, 0) in the chip maps
    to (row_offset, col_offset) in the original image.

    Parameters
    ----------
    geo : SICDGeolocation
        Full-image geolocation.
    row_offset : int
        Row offset of chip origin in the full image.
    col_offset : int
        Column offset of chip origin in the full image.
    chip_rows : int
        Number of rows in the chip.
    chip_cols : int
        Number of columns in the chip.
    """

    def __init__(
        self,
        geo: SICDGeolocation,
        row_offset: int,
        col_offset: int,
        chip_rows: int,
        chip_cols: int,
    ) -> None:
        self._geo = geo
        self._row_off = row_offset
        self._col_off = col_offset
        self.shape = (chip_rows, chip_cols)

    def image_to_latlon(
        self,
        row: Union[float, np.ndarray],
        col: Union[float, np.ndarray],
        height: float = 0.0,
    ) -> Tuple:
        return self._geo.image_to_latlon(
            row + self._row_off, col + self._col_off, height=height,
        )

    def latlon_to_image(
        self,
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray],
        height: float = 0.0,
    ) -> Tuple:
        r, c = self._geo.latlon_to_image(lat, lon, height=height)
        return r - self._row_off, c - self._col_off

    def get_bounds(self) -> Tuple[float, float, float, float]:
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
# FABDEM tile finder
# ---------------------------------------------------------------------------

def _find_fabdem_tile(
    lat: float, lon: float, fabdem_root: str,
) -> Optional[str]:
    """Find the FABDEM GeoTIFF tile covering the given lat/lon.

    FABDEM tiles are stored as 1-deg GeoTIFFs in 10-deg block
    directories.  Returns the tile path, or None if not found.
    """
    fabdem = Path(fabdem_root)
    if not fabdem.exists():
        return None

    # Tile naming: N36W076_FABDEM_V1-2.tif  (lower-left corner)
    lat_floor = int(np.floor(lat))
    lon_floor = int(np.floor(lon))
    ns = 'N' if lat_floor >= 0 else 'S'
    ew = 'E' if lon_floor >= 0 else 'W'
    tile_name = f"{ns}{abs(lat_floor):02d}{ew}{abs(lon_floor):03d}"

    # Search recursively for a matching tile
    for tif in fabdem.rglob(f"{tile_name}*.tif"):
        return str(tif)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def ortho_sicd(
    filepath: Path,
    chip_size: int = 4096,
    save_path: Path = None,
    resolution: float = None,
    interpolation: str = 'nearest',
    dem_path: str = None,
    tile_size: int = None,
    enu_pixel_m: float = None,
) -> None:
    """Orthorectify a SICD center chip and display results.

    Parameters
    ----------
    filepath : Path
        Path to SICD NITF file.
    chip_size : int
        Center chip size in pixels (default 4096).
    save_path : Path, optional
        If provided, save figure to this path.
    resolution : float, optional
        Output pixel size in degrees.  If None, auto-computed from SICD
        grid metadata (sample spacing + graze angle correction).
    interpolation : str, default='nearest'
        Resampling method: 'nearest', 'bilinear', or 'bicubic'.
    dem_path : str, optional
        Path to FABDEM directory for terrain-corrected projection.
    tile_size : int, optional
        Process output in tiles of this size (pixels).
    enu_pixel_m : float, optional
        If set, also produce ENU output at this pixel spacing (meters).
    """
    timings = {}
    print(f"Loading: {filepath.name}")
    print(f"Backend: {detect_backend()}")
    print()

    # ------------------------------------------------------------------
    # Open and extract metadata
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    with SICDReader(filepath) as reader:
        meta = reader.metadata
        rows, cols = meta.rows, meta.cols

        mode = 'unknown'
        if meta.collection_info and meta.collection_info.radar_mode:
            mode = meta.collection_info.radar_mode.mode_type or 'unknown'

        print(f"  Full size:  {rows} x {cols}")
        print(f"  Mode:       {mode}")
        print(f"  Backend:    {meta.backend}")
        if meta.grid:
            print(f"  Row ss:     {meta.grid.row.ss:.4f} m")
            print(f"  Col ss:     {meta.grid.col.ss:.4f} m")
        if meta.scpcoa:
            print(f"  Graze:      {meta.scpcoa.graze_ang:.2f} deg")
        if meta.geo_data and meta.geo_data.scp and meta.geo_data.scp.llh:
            scp = meta.geo_data.scp.llh
            print(f"  SCP:        ({scp.lat:.4f}, {scp.lon:.4f})")
        print()

        # --------------------------------------------------------------
        # Plan center chip with ChipExtractor
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # Create geolocation (full image), then wrap for chip
        # --------------------------------------------------------------
        geo_full = SICDGeolocation.from_reader(reader)
        geo = _ChipGeolocationWrapper(
            geo_full, region.row_start, region.col_start,
            chip_rows, chip_cols,
        )

        # --------------------------------------------------------------
        # Read center chip
        # --------------------------------------------------------------
        print("Reading center chip...")
        t_read0 = time.perf_counter()
        complex_chip = reader.read_chip(
            region.row_start, region.row_end,
            region.col_start, region.col_end,
        )
        timings['read'] = time.perf_counter() - t_read0
        print(f"  Read in {timings['read']:.2f} s ({complex_chip.dtype})")

    # Compute linear magnitude in slant range
    print("Computing magnitude...")
    t_mag0 = time.perf_counter()
    mag = np.abs(complex_chip).astype(np.float32)
    del complex_chip
    timings['magnitude'] = time.perf_counter() - t_mag0

    # ------------------------------------------------------------------
    # Auto-discover FABDEM tile if dem_path is a directory
    # ------------------------------------------------------------------
    elev = None
    if dem_path is not None:
        dem_file = dem_path
        if Path(dem_path).is_dir():
            # Get scene center lat/lon
            clat, clon, _ = geo.image_to_latlon(
                chip_rows / 2.0, chip_cols / 2.0,
            )
            tile = _find_fabdem_tile(float(clat), float(clon), dem_path)
            if tile:
                dem_file = tile
                print(f"  FABDEM tile: {Path(tile).name}")
            else:
                print(f"  No FABDEM tile found for ({clat:.2f}, {clon:.2f})")
                dem_file = None

        if dem_file is not None:
            from grdl.geolocation.elevation.geotiff_dem import GeoTIFFDEM
            elev = GeoTIFFDEM(dem_file)
            print(f"  DEM:        {dem_file}")

    # ------------------------------------------------------------------
    # Build and run WGS-84 pipeline
    # ------------------------------------------------------------------
    pipeline = (
        OrthoBuilder()
        .with_source_array(mag)
        .with_metadata(meta)
        .with_geolocation(geo)
        .with_interpolation(interpolation)
        .with_nodata(np.nan)
    )

    if elev is not None:
        pipeline = pipeline.with_elevation(elev)

    if resolution is not None:
        pipeline = pipeline.with_resolution(resolution, resolution)
        print(f"  Resolution: {resolution:.6f} deg (user)")
    else:
        print("  Resolution: auto (from SICD grid metadata)")

    if tile_size is not None:
        pipeline = pipeline.with_tile_size(tile_size)
        print(f"  Tile size:  {tile_size}")

    print()
    print("Running WGS-84 ortho pipeline...")
    t_pipe0 = time.perf_counter()
    result = pipeline.run()
    timings['ortho_wgs84'] = time.perf_counter() - t_pipe0
    print(f"  Completed in {timings['ortho_wgs84']:.2f} s")

    grid = result.output_grid
    print(f"  Grid:       {grid.rows} x {grid.cols}")
    print(f"  Lat:        [{grid.min_lat:.4f}, {grid.max_lat:.4f}]")
    print(f"  Lon:        [{grid.min_lon:.4f}, {grid.max_lon:.4f}]")
    print(f"  Pixel:      {grid.pixel_size_lat:.6f} x "
          f"{grid.pixel_size_lon:.6f} deg")

    n_valid = np.sum(np.isfinite(result.data))
    n_total = grid.rows * grid.cols
    print(f"  Valid:      {n_valid:,} / {n_total:,} "
          f"({100 * n_valid / n_total:.1f}%)")

    # ------------------------------------------------------------------
    # ENU ortho (optional)
    # ------------------------------------------------------------------
    result_enu = None
    if enu_pixel_m is not None:
        print()
        print(f"Running ENU ortho pipeline ({enu_pixel_m:.1f} m)...")
        enu_pipeline = (
            OrthoBuilder()
            .with_source_array(mag)
            .with_metadata(meta)
            .with_geolocation(geo)
            .with_interpolation(interpolation)
            .with_nodata(np.nan)
            .with_enu_grid(pixel_size_m=enu_pixel_m)
        )
        if elev is not None:
            enu_pipeline = enu_pipeline.with_elevation(elev)

        t_enu0 = time.perf_counter()
        result_enu = enu_pipeline.run()
        timings['ortho_enu'] = time.perf_counter() - t_enu0
        eg = result_enu.output_grid
        print(f"  Completed in {timings['ortho_enu']:.2f} s")
        print(f"  Grid:       {eg.rows} x {eg.cols}")
        print(f"  E:          [{eg.min_east:.1f}, {eg.max_east:.1f}] m")
        print(f"  N:          [{eg.min_north:.1f}, {eg.max_north:.1f}] m")

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

    n_panels = 3 if result_enu is not None else 2

    # Percentile stretch
    slant_vmin = float(np.nanpercentile(mag, 2))
    slant_vmax = float(np.nanpercentile(mag, 98))
    valid_mask = np.isfinite(result.data)
    if np.any(valid_mask):
        ortho_vmin = float(np.nanpercentile(result.data[valid_mask], 2))
        ortho_vmax = float(np.nanpercentile(result.data[valid_mask], 98))
    else:
        ortho_vmin, ortho_vmax = slant_vmin, slant_vmax

    titles = [
        f"Slant Range (chip {chip_rows}x{chip_cols})<br>Magnitude",
        "Orthorectified (WGS-84)<br>Magnitude",
    ]
    if result_enu is not None:
        titles.append(
            f"Orthorectified (ENU {enu_pixel_m:.1f} m)<br>Magnitude")

    fig = make_subplots(rows=1, cols=n_panels, subplot_titles=titles)

    # Panel 1: Slant range chip
    fig.add_trace(
        go.Heatmap(z=mag, zmin=slant_vmin, zmax=slant_vmax,
                   colorscale='Gray', showscale=False),
        row=1, col=1,
    )
    fig.update_xaxes(title_text="Column (range)", row=1, col=1)
    fig.update_yaxes(title_text="Row (azimuth)", autorange='reversed',
                     row=1, col=1)

    # Panel 2: WGS-84 ortho (north-up)
    fig.add_trace(
        go.Heatmap(z=np.flipud(result.data),
                   x0=grid.min_lon, dx=grid.pixel_size_lon,
                   y0=grid.min_lat, dy=grid.pixel_size_lat,
                   zmin=ortho_vmin, zmax=ortho_vmax,
                   colorscale='Gray', showscale=False),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Longitude (deg)", row=1, col=2)
    fig.update_yaxes(title_text="Latitude (deg)", row=1, col=2)

    # Panel 3: ENU ortho
    if result_enu is not None:
        eg = result_enu.output_grid
        enu_valid = np.isfinite(result_enu.data)
        if np.any(enu_valid):
            enu_vmin = float(np.nanpercentile(result_enu.data[enu_valid], 2))
            enu_vmax = float(np.nanpercentile(result_enu.data[enu_valid], 98))
        else:
            enu_vmin, enu_vmax = ortho_vmin, ortho_vmax

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

    title = (
        f"SICD Ortho  |  {filepath.name}  |  {mode}  |  "
        f"Chip {chip_rows}x{chip_cols}  |  "
        f"Grid {grid.rows}x{grid.cols} @ "
        f"{grid.pixel_size_lat:.6f}\u00b0  |  "
        f"{interpolation}  |  {detect_backend()}"
    )
    if dem_path:
        title += "  |  DEM"
    fig.update_layout(title_text=title, width=700 * n_panels, height=700)

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
    filepath = None
    chip_size = 4096
    save_path = None
    resolution = None
    interpolation = 'nearest'
    dem_path = None
    tile_size = None
    enu_pixel_m = None

    i = 0
    while i < len(args):
        if args[i] == "--help":
            print("SICD Orthorectification — Accelerated Center Chip")
            print()
            print("Usage:")
            print("  python ortho_sicd.py <sicd_file>")
            print("  python ortho_sicd.py <sicd_file> --chip 2048")
            print("  python ortho_sicd.py <sicd_file> --dem /path/to/fabdem")
            print("  python ortho_sicd.py <sicd_file> --save [path]")
            print("  python ortho_sicd.py <sicd_file> --enu 1.0")
            print()
            print("Options:")
            print("  --chip <N>      Center chip size (default: 4096)")
            print("  --save [path]   Save figure (default: sicd_ortho.png)")
            print("  --res <deg>     Output resolution in degrees")
            print("  --interp <m>    Interpolation: nearest, bilinear, bicubic")
            print("  --dem <path>    FABDEM directory for terrain correction")
            print("  --tile <N>      Process output in NxN tiles")
            print("  --enu <m>       Also produce ENU output at this pixel size (meters)")
            sys.exit(0)
        elif args[i] == "--chip":
            chip_size = int(args[i + 1])
            i += 2
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
        elif args[i] == "--enu":
            enu_pixel_m = float(args[i + 1])
            i += 2
        else:
            filepath = Path(args[i])
            i += 1

    return (filepath, chip_size, save_path, resolution,
            interpolation, dem_path, tile_size, enu_pixel_m)


if __name__ == "__main__":
    (fp, cs, sv, res, interp,
     dem, tile, enu) = _parse_args()

    if fp is None:
        print("Error: SICD file path required.")
        print("Usage: python ortho_sicd.py <sicd_file> [--chip N] [--dem path]")
        sys.exit(1)

    ortho_sicd(
        fp, chip_size=cs, save_path=sv, resolution=res,
        interpolation=interp, dem_path=dem, tile_size=tile,
        enu_pixel_m=enu,
    )
