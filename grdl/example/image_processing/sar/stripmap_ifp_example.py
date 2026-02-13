# -*- coding: utf-8 -*-
"""
Stripmap IFP Example - Form a SAR image from stripmap CPHD data.

End-to-end demonstration of the GRDL Stripmap Image Formation
pipeline.  Reads a stripmap CPHD file, partitions the aperture into
overlapping sub-apertures, forms each with the Polar Format Algorithm,
and mosaics the results into a continuous strip image.

Demonstrates full GRDL integration:
  - ``grdl.IO.sar.CPHDReader`` for metadata + signal access
  - ``grdl.image_processing.sar.StripmapPFA`` for subaperture IFP
  - ``grdl.IO.sar.SICDWriter`` for NITF output (optional)

Usage:
  python stripmap_ifp_example.py <cphd_file>
  python stripmap_ifp_example.py <cphd_file> --output output.nitf
  python stripmap_ifp_example.py <cphd_file> --subaperture-pulses 4096
  python stripmap_ifp_example.py <cphd_file> --overlap 0.5
  python stripmap_ifp_example.py <cphd_file> --dump-metadata meta.txt
  python stripmap_ifp_example.py --help

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
2026-02-13

Modified
--------
2026-02-13
"""

# Standard library
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Third-party
import numpy as np
from numpy.linalg import norm

# GRDL
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from grdl.IO.sar import CPHDReader
from grdl.image_processing.sar import StripmapPFA


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Form a complex SAR image from stripmap CPHD data "
                    "using subaperture PFA.",
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
        help="Output SICD NITF path. If omitted, displays only.",
    )
    parser.add_argument(
        "--subaperture-pulses",
        type=int,
        default=None,
        help="Fixed sub-aperture length (pulses). Auto-sized if omitted.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap fraction between sub-apertures (default: 0.5).",
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
        help="Range k-space oversampling factor (default: 1.0).",
    )
    parser.add_argument(
        "--azimuth-oversample",
        type=float,
        default=1.0,
        help="Azimuth k-space oversampling factor (default: 1.0).",
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
        "--dump-metadata",
        type=Path,
        default=None,
        help="Path to dump CPHD metadata text file.",
    )
    return parser.parse_args()


# ── Display ──────────────────────────────────────────────────────────


def display_image(
    image: np.ndarray,
    *,
    title: str = "",
    db_range: float = 50.0,
) -> None:
    """Display the formed stripmap image in dB scale.

    Parameters
    ----------
    image : np.ndarray
        Complex SAR image.
    title : str
        Figure title.
    db_range : float
        Dynamic range in dB for display.
    """
    import matplotlib
    try:
        matplotlib.use("QtAgg")
    except ImportError:
        pass
    import matplotlib.pyplot as plt
    
    mag = np.abs(image)
    mag[mag == 0] = np.finfo(mag.dtype).tiny
    db = 20.0 * np.log10(mag)
    vmax = db.max()
    vmin = vmax - db_range

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    im = ax.imshow(
        db, cmap="gray", aspect="auto",
        interpolation="nearest",
        vmin=vmin, vmax=vmax,
    )
    ax.set_title(title or "Stripmap PFA Image", fontsize=13)
    ax.set_xlabel("Range (samples)")
    ax.set_ylabel("Azimuth (samples)")
    fig.colorbar(im, ax=ax, label="dB", shrink=0.8)
    plt.tight_layout()
    plt.show()


# ── Metadata summary ─────────────────────────────────────────────────


def print_metadata_summary(meta) -> None:
    """Print a compact metadata summary to stdout.

    Parameters
    ----------
    meta : CPHDMetadata
        Metadata from a CPHDReader.
    """
    pvp = meta.pvp
    gp = meta.global_params

    print(f"  Pulses: {pvp.num_vectors}")
    print(f"  Samples: {meta.cols}")

    if gp and gp.center_frequency:
        print(f"  Center freq: {gp.center_frequency / 1e9:.4f} GHz")
        print(f"  Bandwidth: {gp.bandwidth / 1e6:.2f} MHz")

    if pvp.srp_pos is not None:
        drift = norm(pvp.srp_pos[-1] - pvp.srp_pos[0])
        print(f"  SRP drift: {drift:.1f} m ({drift / 1000:.2f} km)")

    mid_times = 0.5 * (pvp.tx_time + pvp.rcv_time)
    dt = np.mean(np.diff(mid_times))
    print(f"  PRF: {1.0 / dt:.1f} Hz")
    print(f"  Time span: {mid_times[-1] - mid_times[0]:.3f} s")

    ci = meta.collection_info
    if ci and ci.radar_mode:
        print(f"  Mode: {ci.radar_mode}")
    if ci and ci.collector_name:
        print(f"  Collector: {ci.collector_name}")


# ── Main pipeline ────────────────────────────────────────────────────


def run_stripmap_ifp(
    filepath: Path,
    output: Optional[Path] = None,
    subaperture_pulses: Optional[int] = None,
    overlap: float = 0.5,
    grid_mode: str = "inscribed",
    range_oversample: float = 1.0,
    azimuth_oversample: float = 1.0,
    slant: bool = True,
    dump_metadata: Optional[Path] = None,
) -> np.ndarray:
    """Run the stripmap subaperture PFA pipeline.

    Parameters
    ----------
    filepath : Path
        Path to the CPHD file.
    output : Path, optional
        If provided, write the formed image as SICD NITF.
    subaperture_pulses : int, optional
        Fixed sub-aperture length. Auto-sized if None.
    overlap : float
        Overlap fraction between sub-apertures.
    grid_mode : str
        ``'inscribed'`` or ``'circumscribed'``.
    range_oversample : float
        Range oversampling factor.
    azimuth_oversample : float
        Azimuth oversampling factor.
    slant : bool
        If True, use slant plane.
    dump_metadata : Path, optional
        If provided, dump metadata to this file.

    Returns
    -------
    np.ndarray
        Complex SAR image (mosaiced strip).
    """
    t0 = time.perf_counter()

    # ── Read CPHD ──
    print(f"Opening: {filepath}")
    with CPHDReader(filepath) as reader:
        meta = reader.metadata

        print("\nMetadata summary:")
        print_metadata_summary(meta)

        # Optional metadata dump
        if dump_metadata is not None:
            from grdl.example.image_processing.sar.dump_cphd_metadata import (
                dump_metadata as _dump,
            )
            _dump(filepath, dump_metadata)

        # Read signal
        print("\nReading signal data...")
        signal = reader.read_full()
        print(f"  Signal shape: {signal.shape}, dtype: {signal.dtype}")

    t_read = time.perf_counter()
    print(f"  Read time: {t_read - t0:.2f}s")

    # ── Stripmap PFA ──
    print("\nConfiguring StripmapPFA...")
    from grdl.interpolation import PolyphaseInterpolator
    interpolator = PolyphaseInterpolator(
        kernel_length=64, num_phases=256, prototype='remez',
    )

    ifp = StripmapPFA(
        metadata=meta,
        interpolator=interpolator,
        weighting='taylor',
        grid_mode=grid_mode,
        range_oversample=range_oversample,
        azimuth_oversample=azimuth_oversample,
        subaperture_pulses=subaperture_pulses,
        overlap_fraction=overlap,
        slant=slant,
        verbose=True,
    )

    part = ifp.partitioner
    print(f"  Sub-aperture length: {part.sub_length} pulses")
    print(f"  Stride: {part.stride} pulses")
    print(f"  Num sub-apertures: {part.num_subapertures}")

    # Form image
    print("\nForming image...")
    t_form = time.perf_counter()
    image = ifp.form_image(signal, geometry=None)
    t_done = time.perf_counter()

    print(f"\nImage formed: {image.shape}, dtype: {image.dtype}")
    print(f"  Formation time: {t_done - t_form:.2f}s")
    print(f"  Total time: {t_done - t0:.2f}s")

    # ── Optional: Save SICD ──
    if output is not None:
        print(f"\nWriting SICD to: {output}")
        from grdl.IO.sar import SICDWriter
        writer = SICDWriter(output)
        writer.write(image)
        print("  Done.")

    # ── Display ──
    title = filepath.name
    ci = meta.collection_info
    if ci and ci.collector_name:
        title += f"  |  {ci.collector_name}"
    display_image(image, title=title)

    return image


if __name__ == "__main__":
    args = parse_args()
    run_stripmap_ifp(
        args.filepath,
        output=args.output,
        subaperture_pulses=args.subaperture_pulses,
        overlap=args.overlap,
        grid_mode=args.grid_mode,
        range_oversample=args.range_oversample,
        azimuth_oversample=args.azimuth_oversample,
        slant=not args.ground,
        dump_metadata=args.dump_metadata,
    )
