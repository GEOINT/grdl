# -*- coding: utf-8 -*-
"""
Chip and Orthorectify — Ground-extent chip extraction and orthorectification.

Loads a SAR image (SICD or SIDD), computes the pixel extent for a
user-specified ground patch (in meters) centered on a lat/lon point or
pixel coordinate, extracts the chip, and orthorectifies it to a local
ENU meter grid.

Demonstrates full GRDL integration:
  - ``grdl.IO.sar`` readers (SICDReader / SIDDReader)
  - ``grdl.data_prep.ChipExtractor`` for pixel-domain chip planning
  - ``grdl.geolocation`` for pixel ↔ ground transforms
  - ``grdl.geolocation.coordinates.enu_to_geodetic`` for meter → pixel
  - ``grdl.image_processing.ortho.OrthoBuilder`` with ENU grid output

Usage
-----
  python chip_ortho.py <sar_file>
  python chip_ortho.py <sar_file> --lat 34.05 --lon -118.25
  python chip_ortho.py <sar_file> --row 3000 --col 4000
  python chip_ortho.py <sar_file> --extent 1000 --pixel-size 0.5
  python chip_ortho.py --help

Dependencies
------------
plotly
sarpy or sarkit

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-03-18

Modified
--------
2026-03-27
"""

# Standard library
import argparse
import sys
from pathlib import Path

# Third-party
import numpy as np

# GRDL
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from grdl.data_prep import ChipExtractor
from grdl.geolocation.chip import ChipGeolocation
from grdl.geolocation.coordinates import enu_to_geodetic


# ── Format detection ─────────────────────────────────────────────────


def _detect_format(filepath: Path) -> str:
    """Detect SAR format from filename or content.

    Returns
    -------
    str
        ``'SICD'`` or ``'SIDD'``.
    """
    name = filepath.name.upper()
    if 'SIDD' in name:
        return 'SIDD'
    if 'SICD' in name:
        return 'SICD'
    # Default to SICD for .nitf files
    return 'SICD'


# ── Argument parser ──────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract a ground-extent chip from a SAR image "
                    "(SICD or SIDD) and orthorectify it to ENU meters.",
    )
    p.add_argument("filepath", type=Path,
                   help="Path to SICD or SIDD NITF file")

    center = p.add_argument_group("center point (choose one)")
    center.add_argument("--lat", type=float, default=None,
                        help="Center latitude (degrees)")
    center.add_argument("--lon", type=float, default=None,
                        help="Center longitude (degrees)")
    center.add_argument("--row", type=int, default=None,
                        help="Center row pixel")
    center.add_argument("--col", type=int, default=None,
                        help="Center col pixel")

    p.add_argument("--extent", type=float, default=500.0,
                   help="Ground extent in meters (default: 500)")
    p.add_argument("--pixel-size", type=float, default=1.0,
                   help="Ortho output pixel size in meters (default: 1.0)")
    p.add_argument("--interp", choices=("nearest", "bilinear", "bicubic"),
                   default="bilinear",
                   help="Resampling interpolation (default: bilinear)")
    p.add_argument("--format", choices=("SICD", "SIDD"), default=None,
                   help="Force format (default: auto-detect from filename)")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    filepath = args.filepath
    extent_m = args.extent
    half = extent_m / 2.0

    # Detect format
    fmt = args.format or _detect_format(filepath)
    print(f"Opening ({fmt}): {filepath.name}")

    # Open reader and geolocation
    if fmt == 'SICD':
        from grdl.IO.sar import SICDReader
        from grdl.geolocation.sar.sicd import SICDGeolocation
        reader = SICDReader(filepath)
        meta = reader.metadata
        geo = SICDGeolocation.from_reader(reader)
        is_complex = True
    else:
        from grdl.IO.sar import SIDDReader
        from grdl.geolocation.sar.sidd import SIDDGeolocation
        reader = SIDDReader(filepath)
        meta = reader.metadata
        geo = SIDDGeolocation(meta, refine=False)
        is_complex = False

    rows, cols = meta.rows, meta.cols
    print(f"  Image size: {rows} x {cols}")

    # ── Determine center point ───────────────────────────────
    if args.lat is not None and args.lon is not None:
        target_lat, target_lon = args.lat, args.lon
        cr, cc = geo.latlon_to_image(target_lat, target_lon)
        center_row = int(round(float(np.asarray(cr).ravel()[0])))
        center_col = int(round(float(np.asarray(cc).ravel()[0])))
        print(f"\n  Input lat/lon: ({target_lat:.6f}, {target_lon:.6f})")
        print(f"  Mapped pixel:  ({center_row}, {center_col})")
        # Check if point is within image bounds
        if (center_row < 0 or center_row >= rows
                or center_col < 0 or center_col >= cols):
            print(f"  ERROR: point is outside image bounds "
                  f"[0:{rows}, 0:{cols}]")
            margin = int(half / max(
                getattr(getattr(meta, 'grid', None),
                        'row', None) and meta.grid.row.ss or 1.0, 0.1))
            in_row = 0 <= center_row < rows
            in_col = 0 <= center_col < cols
            if not in_row and not in_col:
                print("  Both row and col are out of bounds. "
                      "Choose a point inside the image footprint.")
                reader.close()
                return
            # Clamp for partial coverage
            center_row = max(0, min(center_row, rows - 1))
            center_col = max(0, min(center_col, cols - 1))
            print(f"  Clamped to:    ({center_row}, {center_col}) "
                  f"— chip will be partial")
        center_lat, center_lon = target_lat, target_lon
        _, _, center_h = geo.image_to_latlon(
            float(center_row), float(center_col))
        center_h = float(center_h)
    else:
        center_row = args.row if args.row is not None else rows // 2
        center_col = args.col if args.col is not None else cols // 2
        # Validate pixel is in bounds
        if (center_row < 0 or center_row >= rows
                or center_col < 0 or center_col >= cols):
            print(f"\n  ERROR: pixel ({center_row}, {center_col}) is "
                  f"outside image bounds [0:{rows}, 0:{cols}]")
            reader.close()
            return
        center_lat, center_lon, center_h = geo.image_to_latlon(
            float(center_row), float(center_col))
        center_lat, center_lon, center_h = (
            float(center_lat), float(center_lon), float(center_h))
        print(f"\n  Center pixel: ({center_row}, {center_col})")

    print(f"  Center geo:   ({center_lat:.6f}, {center_lon:.6f}), "
          f"h={center_h:.1f} m")

    # ── Convert ground extent to pixel extent ────────────────
    east_off = np.array([-half, half, -half, half])
    north_off = np.array([-half, -half, half, half])
    corner_lats, corner_lons, _ = enu_to_geodetic(
        east_off, north_off, np.zeros(4),
        ref_lat=center_lat, ref_lon=center_lon, ref_alt=center_h,
    )
    corner_rows, corner_cols = geo.latlon_to_image(
        corner_lats, corner_lons, center_h)

    row_min = int(np.floor(np.min(corner_rows)))
    row_max = int(np.ceil(np.max(corner_rows)))
    col_min = int(np.floor(np.min(corner_cols)))
    col_max = int(np.ceil(np.max(corner_cols)))

    extractor = ChipExtractor(nrows=rows, ncols=cols)
    region = extractor.chip_at_point(
        center_row, center_col, row_max - row_min, col_max - col_min)

    chip_rows = region.row_end - region.row_start
    chip_cols = region.col_end - region.col_start
    print(f"  Ground extent: {extent_m:.0f} m x {extent_m:.0f} m")
    print(f"  Pixel extent:  {chip_rows} x {chip_cols}")

    # ── Read chip ────────────────────────────────────────────
    print("\nReading chip...")
    chip = reader.read_chip(
        region.row_start, region.row_end,
        region.col_start, region.col_end,
    )
    reader.close()

    # Convert to display array
    if is_complex:
        img = np.abs(chip).astype(np.float32)
    elif chip.ndim == 3 and chip.shape[0] == 3:
        img = (0.299 * chip[0].astype(np.float32)
               + 0.587 * chip[1].astype(np.float32)
               + 0.114 * chip[2].astype(np.float32))
    elif chip.ndim == 3:
        img = chip[0].astype(np.float32)
    else:
        img = chip.astype(np.float32)

    # ── Wrap geolocation ─────────────────────────────────────
    chip_geo = ChipGeolocation(
        geo,
        row_offset=region.row_start,
        col_offset=region.col_start,
        shape=(chip_rows, chip_cols),
    )

    # ── Orthorectify ─────────────────────────────────────────
    from grdl.image_processing.ortho import orthorectify
    from grdl.geolocation.elevation.constant import ConstantElevation

    chip_geo.elevation = ConstantElevation(height=center_h)

    print(f"Orthorectifying to ENU grid ({args.pixel_size:.1f} m)...")
    result = orthorectify(
        geolocation=chip_geo,
        source_array=img,
        metadata=meta,
        interpolation=args.interp,
        nodata=np.nan,
        enu_grid=dict(
            pixel_size_m=args.pixel_size,
            ref_lat=center_lat,
            ref_lon=center_lon,
            ref_alt=center_h,
        ),
    )

    ortho = result.data
    grid = result.output_grid
    print(f"  Output: {ortho.shape[0]} x {ortho.shape[1]} px, "
          f"{grid.pixel_size_east:.2f} m/px")

    valid = ~np.isnan(ortho)
    print(f"  Valid:  {100 * np.sum(valid) / ortho.size:.1f}%")

    # ── Plot ─────────────────────────────────────────────────
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    vmin = np.percentile(img[img > 0], 2) if np.any(img > 0) else 0
    vmax = np.percentile(img[img > 0], 99) if np.any(img > 0) else 1

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"{fmt} Chip  ({chip_rows} x {chip_cols} px)<br>"
            f"{extent_m:.0f} m x {extent_m:.0f} m ground",
            f"Orthorectified ENU  ({ortho.shape[0]} x "
            f"{ortho.shape[1]} px)<br>"
            f"{args.pixel_size:.1f} m pixels, {args.interp}",
        ],
    )

    # Left: chip
    fig.add_trace(
        go.Heatmap(z=img, zmin=vmin, zmax=vmax,
                   colorscale='Gray', showscale=False),
        row=1, col=1,
    )
    mr, mc = chip_geo.latlon_to_image(center_lat, center_lon, center_h)
    mr = float(np.asarray(mr).ravel()[0])
    mc = float(np.asarray(mc).ravel()[0])
    fig.add_trace(
        go.Scatter(x=[mc], y=[mr], mode='markers',
                   marker=dict(symbol='cross', size=16,
                               color='red', line=dict(width=3, color='red')),
                   showlegend=False),
        row=1, col=1,
    )
    fig.update_xaxes(title_text="Column", row=1, col=1)
    fig.update_yaxes(title_text="Row", autorange='reversed', row=1, col=1)

    # Right: ortho (north-up: flipud so z[0] = south at bottom)
    fig.add_trace(
        go.Heatmap(z=np.flipud(ortho), zmin=vmin, zmax=vmax,
                   x0=grid.min_east, dx=grid.pixel_size_east,
                   y0=grid.min_north, dy=grid.pixel_size_north,
                   colorscale='Gray', showscale=False),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=[0.0], y=[0.0], mode='markers',
                   marker=dict(symbol='cross', size=16,
                               color='red', line=dict(width=3, color='red')),
                   showlegend=False),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="East (m)", row=1, col=2)
    fig.update_yaxes(title_text="North (m)", row=1, col=2)

    fig.update_layout(
        title_text=(f"{filepath.name}  |  "
                    f"({center_lat:.4f}, {center_lon:.4f})"),
        width=1400, height=650,
    )
    fig.show()


if __name__ == "__main__":
    main()
