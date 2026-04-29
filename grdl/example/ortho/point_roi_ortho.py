# -*- coding: utf-8 -*-
"""
Point-ROI Orthorectification — NxM meter area centered on a lat/lon point.

Thin config + display shell over ``grdl.image_processing.ortho.orthorectify_point_roi``.
Supports any GRDL imagery format (SAR, EO NITF, GeoTIFF, …).

Primary configuration is via ``point_roi_ortho.yaml`` (next to this script).
CLI flags override YAML values when supplied.

Usage
-----
  # Edit point_roi_ortho.yaml, then:
  python point_roi_ortho.py

  # Or override individual values on the command line:
  python point_roi_ortho.py --lat 34.05 --lon -118.25
  python point_roi_ortho.py --config /path/to/custom.yaml
  python point_roi_ortho.py --help

Dependencies
------------
matplotlib
pyyaml

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
2026-04-25

Modified
--------
2026-04-25  Refactored to use orthorectify_point_roi() helper.
"""

# Standard library
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

# Third-party
import numpy as np

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

# GRDL
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from grdl.IO.generic import open_any
from grdl.image_processing.ortho import orthorectify_point_roi

# grdl.contrast operators for the matplotlib preview.
from grdl.contrast import (
    auto_select, Brighter, CLAHE, Darker, GammaCorrection, HighContrast,
    HistogramEqualization, LinearStretch, LogStretch, MangisDensity,
    NRLStretch, PercentileStretch,
)


_DEFAULT_CONFIG = Path(__file__).resolve().parent / "point_roi_ortho.yaml"

_DEFAULTS: dict = {
    "filepath":     None,
    "lat":          None,
    "lon":          None,
    "row":          None,
    "col":          None,
    "width":        500.0,
    "height":       500.0,
    "pixel_size":   None,
    "interp":       "lanczos",
    "dem":          None,
    "geoid":        None,
    "band":         0,
    "no_plot":      False,
    "save":         None,
    "complex_mode": "magnitude",
    "display_contrast": "auto",
}


_CONTRAST_CHOICES = (
    "auto",
    "percentile", "linear", "log",
    "mangis", "brighter", "darker", "high_contrast",
    "nrl", "gamma", "histogram", "clahe",
)


def _apply_display_contrast(
    name: str,
    src: np.ndarray,
    ortho: np.ndarray,
    ortho_finite: np.ndarray,
    metadata: object = None,
) -> tuple:
    """Run the chosen ``grdl.contrast`` operator over ``src`` and ``ortho``.

    Pre-computes shared stats over the pooled finite samples of both
    arrays so the two preview panels stay visually consistent (same
    brightness, same dynamic range).  Returns ``(src_out, ortho_out)`` —
    both ``float32`` in ``[0, 1]``, ready for ``imshow(vmin=0, vmax=1)``.

    When ``name == 'auto'``, ``grdl.contrast.auto_select(metadata)`` picks
    the operator from the reader's metadata type (SAR → ``'brighter'``,
    EO NITF → ``'gamma'``, MSI/unknown → ``'percentile'``).

    Stats threading by operator:

    - ``linear`` / ``log``  — pooled ``min`` / ``max``
    - ``percentile``        — pooled 2/99 percentiles via LinearStretch
    - ``mangis`` / ``brighter`` / ``darker`` / ``high_contrast``
                            — pooled mean amplitude (``data_mean=``)
    - ``nrl``               — pooled ``(min, max, p99)`` (``stats=``)
    - ``gamma``             — pooled 2/99 percentile normalisation, then
                              ``GammaCorrection(gamma=1.4)``
    - ``histogram`` / ``clahe`` — applied per array (no shared stats)
    """
    if name == "auto":
        name = auto_select(metadata)

    pooled = (
        np.concatenate([src.ravel(), ortho[ortho_finite].ravel()])
        if ortho_finite.any() else src.ravel()
    )

    if pooled.size == 0:
        zero = np.zeros_like(src, dtype=np.float32)
        return zero, np.zeros_like(ortho, dtype=np.float32)

    if name == "percentile":
        vmin = float(np.percentile(pooled, 2))
        vmax = float(np.percentile(pooled, 99))
        op = LinearStretch(min_value=vmin, max_value=vmax)
        return op.apply(src), op.apply(ortho)

    if name == "linear":
        vmin = float(np.min(pooled))
        vmax = float(np.max(pooled))
        op = LinearStretch(min_value=vmin, max_value=vmax)
        return op.apply(src), op.apply(ortho)

    if name == "log":
        vmin = float(np.min(pooled))
        vmax = float(np.max(pooled))
        op = LogStretch(min_value=vmin, max_value=vmax)
        return op.apply(src), op.apply(ortho)

    if name in ("mangis", "brighter", "darker", "high_contrast"):
        op_cls = {
            "mangis": MangisDensity,
            "brighter": Brighter,
            "darker": Darker,
            "high_contrast": HighContrast,
        }[name]
        data_mean = float(np.mean(pooled))
        op = op_cls()
        return (op.apply(src, data_mean=data_mean),
                op.apply(ortho, data_mean=data_mean))

    if name == "nrl":
        vmin = float(np.min(pooled))
        vmax = float(np.max(pooled))
        p99 = float(np.percentile(pooled, 99))
        if not (vmin <= p99 <= vmax):
            p99 = vmax
        op = NRLStretch()
        return (op.apply(src, stats=(vmin, vmax, p99)),
                op.apply(ortho, stats=(vmin, vmax, p99)))

    if name == "gamma":
        vmin = float(np.percentile(pooled, 2))
        vmax = float(np.percentile(pooled, 99))
        norm = LinearStretch(min_value=vmin, max_value=vmax)
        gamma = GammaCorrection(gamma=1.4)
        return gamma.apply(norm.apply(src)), gamma.apply(norm.apply(ortho))

    if name == "histogram":
        op = HistogramEqualization()
        return op.apply(src), op.apply(ortho)

    if name == "clahe":
        op = CLAHE(kernel_size=64, clip_limit=0.02)
        return op.apply(src), op.apply(ortho)

    # Fallback: PercentileStretch behaves per-array but is GRDL's stock stretch.
    op = PercentileStretch(plow=2.0, phigh=99.0)
    return op.apply(src), op.apply(ortho)


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Orthorectify an NxM meter ROI from any GRDL-supported format.\n"
            "Primary config is point_roi_ortho.yaml; CLI flags override it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG,
                   help=f"YAML config file (default: {_DEFAULT_CONFIG.name})")
    p.add_argument("--file", dest="filepath", type=Path, default=None)

    center = p.add_argument_group("center point (lat/lon OR row/col)")
    center.add_argument("--lat", type=float, default=None)
    center.add_argument("--lon", type=float, default=None)
    center.add_argument("--row", type=int,   default=None)
    center.add_argument("--col", type=int,   default=None)

    p.add_argument("--width",      type=float, default=None)
    p.add_argument("--height",     type=float, default=None)
    p.add_argument("--pixel-size", type=float, default=None)
    p.add_argument("--interp",
                   choices=("nearest", "bilinear", "bicubic", "lanczos"),
                   default=None)
    p.add_argument("--dem",   type=str, default=None)
    p.add_argument("--geoid", type=str, default=None)
    p.add_argument("--band",  type=int, default=None)
    p.add_argument("--complex-mode",
                   choices=("magnitude", "complex"), default=None)
    p.add_argument("--display-contrast",
                   choices=_CONTRAST_CHOICES, default=None,
                   help="grdl.contrast operator used for the matplotlib "
                        "preview (default: percentile)")
    p.add_argument("--no-plot",  action="store_true", default=None)
    p.add_argument("--save",     type=Path, default=None)
    return p.parse_args()


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    if not _HAS_YAML:
        print(f"WARNING: pyyaml not installed — cannot load {path.name}.")
        return {}
    with open(path) as f:
        return _yaml.safe_load(f) or {}


def _build_config(args: argparse.Namespace) -> SimpleNamespace:
    yaml_cfg = _load_yaml(args.config)
    merged   = dict(_DEFAULTS)

    for key, val in yaml_cfg.items():
        if key in merged and val is not None:
            merged[key] = val

    for key, val in vars(args).items():
        if key == "config":
            continue
        if val is not None:
            merged[key] = val

    if merged["filepath"] is not None:
        merged["filepath"] = Path(str(merged["filepath"]))
    if merged["save"] is not None:
        merged["save"] = Path(str(merged["save"]))
    for k in ("width", "height"):
        if merged[k] is not None:
            merged[k] = float(merged[k])
    if merged["pixel_size"] is not None:
        merged["pixel_size"] = float(merged["pixel_size"])
    if merged["band"] is not None:
        merged["band"] = int(merged["band"])
    merged["no_plot"] = bool(merged["no_plot"])

    return SimpleNamespace(**merged)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    cfg  = _build_config(args)

    if cfg.filepath is None:
        print(
            "ERROR: no image file specified.\n"
            f"  Edit {args.config} and set 'filepath', or pass --file <path>."
        )
        sys.exit(1)

    if not cfg.filepath.exists():
        print(f"ERROR: file not found: {cfg.filepath}")
        sys.exit(1)

    # ── Open image ───────────────────────────────────────────────────
    print(f"Opening: {cfg.filepath.name}")
    reader = open_any(cfg.filepath)
    meta   = reader.metadata
    print(f"  Format:     {meta.format}")
    print(f"  Image size: {int(meta.rows)} × {int(meta.cols)}")

    # ── Build elevation model (DEM → flat plane sampled at center) ───
    elevation = None
    if cfg.dem is not None:
        from grdl.geolocation.elevation import open_elevation
        center_hint = (cfg.lat, cfg.lon) if cfg.lat is not None else None
        elevation = open_elevation(
            cfg.dem,
            geoid_path=cfg.geoid,
            location=center_hint,
        )

    # ── Orthorectify ─────────────────────────────────────────────────
    try:
        result = orthorectify_point_roi(
            reader=reader,
            lat=cfg.lat,
            lon=cfg.lon,
            row=cfg.row,
            col=cfg.col,
            width_m=float(cfg.width),
            height_m=float(cfg.height),
            pixel_size_m=cfg.pixel_size,
            interpolation=cfg.interp,
            band=cfg.band,
            complex_mode=cfg.complex_mode,
            elevation=elevation,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        reader.close()
        sys.exit(1)

    reader.close()

    # ── Stats ────────────────────────────────────────────────────────
    grid      = result.grid
    ortho     = result.data
    ortho_disp = np.abs(ortho).astype(np.float32) if np.iscomplexobj(ortho) else ortho
    if ortho_disp.ndim == 3:
        ortho_disp = ortho_disp[min(cfg.band, ortho_disp.shape[0] - 1)]

    valid_mask = np.isfinite(ortho_disp)
    valid_pct  = 100.0 * np.sum(valid_mask) / ortho_disp.size

    print(f"\n  Center:     ({result.center_lat:.6f}°, {result.center_lon:.6f}°)")
    print(f"  Pixel:      ({result.center_row:.1f}, {result.center_col:.1f})")
    print(f"  ROI:        {cfg.width:.0f} × {cfg.height:.0f} m")
    print(f"  Pixel size: {grid.pixel_size_east:.3f} m")
    print(f"  Output:     {ortho_disp.shape[0]} × {ortho_disp.shape[1]} px")
    print(f"  Valid:      {valid_pct:.1f}%")
    print(f"  Bounds:     E [{grid.min_east:.0f}, {grid.max_east:.0f}] m  "
          f"N [{grid.min_north:.0f}, {grid.max_north:.0f}] m")

    # ── Save ─────────────────────────────────────────────────────────
    if cfg.save is not None:
        result.save_geotiff(cfg.save)
        print(f"  Saved: {cfg.save}")

    if cfg.no_plot:
        return

    # ── Plot ─────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    src_display = result.source_chip

    # Apply the chosen grdl.contrast operator with shared stats so both
    # panels look uniform.  Output is float32 in [0, 1] → imshow uses
    # vmin=0, vmax=1.
    resolved_contrast = (
        auto_select(meta) if cfg.display_contrast == "auto"
        else cfg.display_contrast
    )
    src_disp_norm, ortho_disp_norm = _apply_display_contrast(
        resolved_contrast, src_display, ortho_disp, valid_mask, meta,
    )

    fig, (ax_src, ax_ortho) = plt.subplots(1, 2, figsize=(16, 8))

    ax_src.imshow(src_disp_norm, cmap="gray", vmin=0.0, vmax=1.0,
                  origin="upper", interpolation="none")
    ax_src.plot(
        result.center_col - result.src_c0,
        result.center_row - result.src_r0,
        "r+", markersize=16, markeredgewidth=2,
    )
    ax_src.set_xlabel("Column (chip)")
    ax_src.set_ylabel("Row (chip)")
    ax_src.set_title(
        f"Original  {result.src_r1 - result.src_r0} × "
        f"{result.src_c1 - result.src_c0} px  "
        f"rows {result.src_r0}:{result.src_r1}  "
        f"cols {result.src_c0}:{result.src_c1}",
        fontsize=10,
    )

    enu_extent = [grid.min_east, grid.max_east, grid.min_north, grid.max_north]
    ax_ortho.imshow(np.flipud(ortho_disp_norm), cmap="gray",
                    vmin=0.0, vmax=1.0, extent=enu_extent,
                    origin="lower", interpolation="none")
    ax_ortho.plot(0.0, 0.0, "r+", markersize=16, markeredgewidth=2)
    ax_ortho.set_xlabel("East (m)")
    ax_ortho.set_ylabel("North (m)")
    ax_ortho.set_title(
        f"Orthorectified  {cfg.width:.0f}×{cfg.height:.0f} m  "
        f"{grid.pixel_size_east:.2f} m/px  {valid_pct:.1f}% valid",
        fontsize=10,
    )

    contrast_label = (
        f"{cfg.display_contrast}→{resolved_contrast}"
        if cfg.display_contrast == "auto" and resolved_contrast != "auto"
        else cfg.display_contrast
    )
    fig.suptitle(
        f"{cfg.filepath.name}  "
        f"({result.center_lat:.5f}°, {result.center_lon:.5f}°)  "
        f"contrast={contrast_label}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
