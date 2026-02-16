# -*- coding: utf-8 -*-
"""
IFP Example - Form a SICD image from CPHD phase history data.

End-to-end demonstration of the GRDL Image Formation Processor (IFP)
pipeline. Reads a CPHD file, computes collection geometry and polar
grid, runs the Polar Format Algorithm, and optionally saves the result
as a SICD NITF.

Demonstrates full GRDL integration:
  - ``grdl.IO.sar.CPHDReader`` for metadata + signal access
  - ``grdl.image_processing.sar.CollectionGeometry`` for coordinate systems
  - ``grdl.image_processing.sar.PolarGrid`` for k-space grid
  - ``grdl.image_processing.sar.PolarFormatAlgorithm`` for image formation
  - ``grdl.IO.sar.SICDWriter`` for NITF output (optional)

Pipeline stages are independently inspectable (tapout points):
  1. Metadata — inspect PVP, frequency params
  2. Collection geometry — plot platform track, angles
  3. Polar grid — plot k-space annulus, resolution
  4. Range interpolation — inspect keystone-formatted data
  5. Azimuth interpolation — inspect rectangular grid
  6. Compress — final complex SAR image

Usage:
  python ifp_example.py <cphd_file>
  python ifp_example.py <cphd_file> --output output.nitf
  python ifp_example.py <cphd_file> --grid-mode circumscribed
  python ifp_example.py <cphd_file> --range-oversample 2.0
  python ifp_example.py <cphd_file> --plot-geometry --plot-grid
  python ifp_example.py --help

Dependencies
------------
matplotlib (optional, for plots)
sarkit or sarpy

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
2026-02-12

Modified
--------
2026-02-12
"""

# Standard library
import argparse
import sys
import time
from pathlib import Path

# Third-party
import numpy as np

# GRDL
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from grdl.interpolation import PolyphaseInterpolator

from grdl.IO.sar import CPHDReader
from grdl.image_processing.sar import (
    CollectionGeometry,
    PolarGrid,
    PolarFormatAlgorithm,
)


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Form a complex SAR image from CPHD phase history data "
                    "using the Polar Format Algorithm.",
    )
    parser.add_argument(
        "filepath",
        type=Path,
        help="Path to the CPHD file.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output SICD NITF path. If omitted, displays the image only.",
    )
    parser.add_argument(
        "--grid-mode",
        type=str,
        default="inscribed",
        choices=["inscribed", "circumscribed"],
        help="Rectangular grid fitting mode (default: inscribed).",
    )
    parser.add_argument(
        "--range-oversample",
        type=float,
        default=1.0,
        help="Range k-space oversampling factor, 1.0 = Nyquist (default: 1.0).",
    )
    parser.add_argument(
        "--azimuth-oversample",
        type=float,
        default=1.0,
        help="Azimuth k-space oversampling factor, 1.0 = Nyquist (default: 1.0).",
    )
    parser.add_argument(
        "--slant",
        action="store_true",
        default=True,
        help="Use slant plane projection (default).",
    )
    parser.add_argument(
        "--ground",
        action="store_true",
        default=False,
        help="Use ground plane projection instead of slant.",
    )
    parser.add_argument(
        "--plot-geometry",
        action="store_true",
        default=False,
        help="Plot collection geometry (requires matplotlib + basemap).",
    )
    parser.add_argument(
        "--plot-grid",
        action="store_true",
        default=False,
        help="Plot polar grid k-space annulus (requires matplotlib).",
    )
    return parser.parse_args()


# ── Display ──────────────────────────────────────────────────────────


def display_image(
    image: np.ndarray,
    signal: np.ndarray,
    *,
    title: str = "",
) -> None:
    """Display rectangular-format image, PFA k-space, and PFA image.

    Three panels:

    1. **Rectangular format** — direct ``ifftshift → ifft2 → fftshift``
       of the raw CPHD phase history (no interpolation).  Shows the
       unfocused baseline before polar-to-rectangular resampling.
    2. **PFA k-space** — recovered from the formed image via
       ``fftshift → fft2 → ifftshift``.  Zero-padding from compression
       appears as zero borders around the data support.
    3. **PFA image** — fully formed complex SAR image.

    Parameters
    ----------
    image : np.ndarray
        Complex SAR image (rows = azimuth, cols = range), as returned
        by ``PolarFormatAlgorithm.compress``.
    signal : np.ndarray
        Raw CPHD phase history, shape ``(npulses, nsamples)``.
    title : str
        Figure title for the PFA image panel.
    """
    import matplotlib
    try:
        matplotlib.use("QtAgg")
    except ImportError:
        pass
    import matplotlib.pyplot as plt

    # Rectangular format: direct 2D IFFT of raw phase history
    rect_image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(signal)))

    # Recover k-space in the same (azimuth, range) layout as the image
    kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))

    fig, (ax_rect, ax_k, ax_img) = plt.subplots(1, 3, figsize=(24, 7))

    # ── Left: rectangular format (direct FFT) ──
    im_rect = ax_rect.imshow(
        np.abs(rect_image), cmap="gray", aspect="auto",
        interpolation="nearest",
    )
    ax_rect.set_title("Rectangular Format (direct IFFT)", fontsize=12)
    ax_rect.set_xlabel("Range (samples)")
    ax_rect.set_ylabel("Azimuth (samples)")
    fig.colorbar(im_rect, ax=ax_rect, shrink=0.8)

    # ── Center: PFA k-space magnitude ──
    im_k = ax_k.imshow(
        np.abs(kspace), cmap="viridis", aspect="auto",
        interpolation="nearest",
    )
    ax_k.set_title("PFA K-Space", fontsize=12)
    ax_k.set_xlabel("Range Spatial Frequency (samples)")
    ax_k.set_ylabel("Azimuth Spatial Frequency (samples)")
    fig.colorbar(im_k, ax=ax_k, shrink=0.8)

    # ── Right: PFA formed image ──
    im_img = ax_img.imshow(
        np.abs(image), cmap="gray", aspect="auto",
        interpolation="nearest",
    )
    ax_img.set_title(title or "PFA Image", fontsize=12)
    ax_img.set_xlabel("Range (samples)")
    ax_img.set_ylabel("Azimuth (samples)")
    fig.colorbar(im_img, ax=ax_img, shrink=0.8)

    plt.tight_layout()
    plt.show()


# ── Main pipeline ────────────────────────────────────────────────────


def run_ifp(
    filepath: Path,
    output: Path = None,
    grid_mode: str = "inscribed",
    range_oversample: float = 1.0,
    azimuth_oversample: float = 1.0,
    slant: bool = True,
    plot_geometry: bool = False,
    plot_grid: bool = False,
) -> np.ndarray:
    """Run the full IFP pipeline: CPHD -> geometry -> grid -> PFA -> image.

    Parameters
    ----------
    filepath : Path
        Path to the CPHD file.
    output : Path, optional
        If provided, write the formed image as a SICD NITF.
    grid_mode : str
        ``'inscribed'`` or ``'circumscribed'``.
    range_oversample : float
        Range k-space oversampling factor.
    azimuth_oversample : float
        Azimuth k-space oversampling factor.
    slant : bool
        If True, use slant plane.
    plot_geometry : bool
        If True, plot collection geometry.
    plot_grid : bool
        If True, plot polar grid.

    Returns
    -------
    np.ndarray
        Complex SAR image.
    """
    t0 = time.perf_counter()

    # ── Stage 0: Read CPHD metadata and signal ───────────────────
    print(f"Opening: {filepath}")
    with CPHDReader(filepath) as reader:
        meta = reader.metadata
        rows, cols = reader.get_shape()
        print(f"  Phase history: {rows} pulses x {cols} samples")

        if meta.collection_info is not None:
            ci = meta.collection_info
            print(f"  Collector: {ci.collector_name or 'unknown'}")
            print(f"  Mode: {ci.radar_mode or 'unknown'}")

        # Read signal data
        print("  Reading signal data...")
        signal = reader.read_full()
        print(f"  Signal shape: {signal.shape}, dtype: {signal.dtype}")

    t_read = time.perf_counter()
    print(f"  Read time: {t_read - t0:.2f}s")

    # ── Stage 1: Collection geometry ─────────────────────────────
    print("\nComputing collection geometry...")
    geo = CollectionGeometry(meta, slant=slant)
    print(f"  Image plane: {geo.image_plane}")
    print(f"  Side of track: {geo.side_of_track}")
    print(f"  Graze angle (COA): {np.degrees(geo.graz_ang_coa):.2f} deg")
    print(f"  Azimuth angle (COA): {np.degrees(geo.azim_ang_coa):.2f} deg")
    print(f"  Integration angle: {np.degrees(2 * geo.theta):.4f} deg")

    if plot_geometry:
        geo.plot()

    t_geo = time.perf_counter()
    print(f"  Geometry time: {t_geo - t_read:.2f}s")

    # ── Stage 2: Polar grid ──────────────────────────────────────
    print(f"\nComputing polar grid ({grid_mode})...")
    grid = PolarGrid(
        geo,
        grid_mode=grid_mode,
        range_oversample=range_oversample,
        azimuth_oversample=azimuth_oversample,
    )
    print(f"  kv bounds: [{grid.kv_bounds[0]:.2f}, {grid.kv_bounds[1]:.2f}]")
    print(f"  ku bounds: [{grid.ku_bounds[0]:.4f}, {grid.ku_bounds[1]:.4f}]")
    print(f"  Resampled grid: {grid.rec_n_pulses} x {grid.rec_n_samples}")
    print(f"  Range resolution: {grid.range_resolution:.3f} m")
    print(f"  Azimuth resolution: {grid.azimuth_resolution:.3f} m")

    if plot_grid:
        grid.plot()

    t_grid = time.perf_counter()
    print(f"  Grid time: {t_grid - t_geo:.2f}s")

    # ── Stage 3: PFA ─────────────────────────────────────────────
    gp = meta.global_params
    phase_sgn = gp.phase_sgn if gp is not None else -1
    print(f"  PhaseSGN: {phase_sgn:+d}")

    pfa = PolarFormatAlgorithm(
        grid=grid,
        interpolator=PolyphaseInterpolator( kernel_length=64, num_phases=256, prototype='kaiser' ),
        phase_sgn=phase_sgn,
    )

    # ── Diagnostics: check kv ranges ──
    kv_polar_0 = grid.get_kv_for_pulse(0)
    kv_min, kv_max = grid.kv_bounds
    kv_sampling = (kv_max - kv_min) / grid.rec_n_samples
    kv_uniform = np.arange(grid.rec_n_samples) * kv_sampling + kv_min
    print(f"\n  [DIAG] kv_polar[0] range: [{kv_polar_0[0]:.4f}, {kv_polar_0[-1]:.4f}]")
    print(f"  [DIAG] kv_uniform range:  [{kv_uniform[0]:.4f}, {kv_uniform[-1]:.4f}]")
    print(f"  [DIAG] kv_polar sorted ascending: {np.all(np.diff(kv_polar_0) > 0)}")
    print(f"  [DIAG] signal[0] max|val|: {np.max(np.abs(signal[0,:])):.6f}")
    print(f"  [DIAG] signal dtype: {signal.dtype}")

    print("\nRange interpolation...")
    range_interp = pfa.interpolate_range(signal, geo)
    t_range = time.perf_counter()
    print(f"  Output shape: {range_interp.shape}")
    print(f"  Range interp time: {t_range - t_grid:.2f}s")
    print(f"  [DIAG] range_interp max|val|: {np.max(np.abs(range_interp)):.6f}")
    print(f"  [DIAG] range_interp nonzero: {np.count_nonzero(range_interp)} / {range_interp.size}")

    print("Azimuth interpolation...")
    az_interp = pfa.interpolate_azimuth(range_interp, geo)
    t_az = time.perf_counter()
    print(f"  Output shape: {az_interp.shape}")
    print(f"  Azimuth interp time: {t_az - t_range:.2f}s")
    print(f"  [DIAG] az_interp max|val|: {np.max(np.abs(az_interp)):.6f}")
    print(f"  [DIAG] az_interp nonzero: {np.count_nonzero(az_interp)} / {az_interp.size}")

    xform = "FFT2" if phase_sgn >= 0 else "IFFT2"
    print(f"Compressing (2D {xform})...")
    image = pfa.compress(az_interp)
    t_compress = time.perf_counter()
    print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"  Compress time: {t_compress - t_az:.2f}s")
    print(f"  [DIAG] image max|val|: {np.max(np.abs(image)):.6f}")

    t_total = time.perf_counter() - t0
    print(f"\nTotal IFP time: {t_total:.2f}s")

    # ── Optional: Save SICD ──────────────────────────────────────
    if output is not None:
        print(f"\nWriting SICD to: {output}")
        from grdl.IO.sar import SICDWriter
        writer = SICDWriter(output)
        writer.write(image)
        print("  Done.")

    # ── Display ──────────────────────────────────────────────────
    title = filepath.name
    if meta.collection_info and meta.collection_info.collector_name:
        title += f"  |  {meta.collection_info.collector_name}"
    display_image(image, signal, title=title)

    return image


if __name__ == "__main__":
    args = parse_args()
    run_ifp(
        args.filepath,
        output=args.output,
        grid_mode=args.grid_mode,
        range_oversample=args.range_oversample,
        azimuth_oversample=args.azimuth_oversample,
        slant=not args.ground,
        plot_geometry=args.plot_geometry,
        plot_grid=args.plot_grid,
    )
