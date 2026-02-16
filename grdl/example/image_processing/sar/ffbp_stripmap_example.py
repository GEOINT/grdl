# -*- coding: utf-8 -*-
"""
FFBP Stripmap Example - Form a SAR image using Fast Factorized Back-Projection.

End-to-end demonstration of the GRDL Fast Factorized Back-Projection
pipeline.  Reads a stripmap CPHD file and forms a complex SAR image
via hierarchical back-projection: leaf formation, binary-tree merge,
and polar-to-rectangular conversion.

Demonstrates full GRDL integration and CPHD metadata utilisation:
  - ``grdl.IO.sar.CPHDReader`` for metadata + signal access
  - ``grdl.image_processing.sar.FastBackProjection`` for IFP
  - AmpSF normalisation and invalid pulse trimming
  - Configurable leaf size and angular sampling

Usage:
  python ffbp_stripmap_example.py <cphd_file>
  python ffbp_stripmap_example.py <cphd_file> --leaf-size 8
  python ffbp_stripmap_example.py <cphd_file> --n-angular 256
  python ffbp_stripmap_example.py <cphd_file> --range-weighting taylor
  python ffbp_stripmap_example.py --help

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
2026-02-16

Modified
--------
2026-02-16
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
from grdl.image_processing.sar import FastBackProjection


# -- CLI ------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Form a complex SAR image from stripmap CPHD data "
                    "using Fast Factorized Back-Projection.",
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        type=Path,
        default=Path(
            "/Volumes/PRO-G40/SAR_DATA/CPHD/"
            "CAPELLA_C02_SM_CPHD_HH_20210131030851_20210131030856.cphd"
        ),
        help="Path to the CPHD file (default: Capella C02 stripmap).",
    )
    parser.add_argument(
        "--range-weighting",
        type=str,
        default=None,
        choices=["uniform", "taylor", "hamming", "hanning"],
        help="Range weighting window (default: none).",
    )
    parser.add_argument(
        "--leaf-size",
        type=int,
        default=8,
        help="Number of pulses per leaf subaperture (default: 8).",
    )
    parser.add_argument(
        "--n-angular",
        type=int,
        default=128,
        help="Angular samples per tree node (default: 128).",
    )
    parser.add_argument(
        "--cross-range-spacing",
        type=float,
        default=None,
        help="Output cross-range pixel spacing in metres. "
             "If omitted, derived from Nyquist sampling.",
    )
    parser.add_argument(
        "--no-amp-sf",
        action="store_true",
        default=False,
        help="Disable AmpSF normalization.",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        default=False,
        help="Disable invalid pulse trimming.",
    )
    return parser.parse_args()


# -- Display --------------------------------------------------------


def display_image(
    image: np.ndarray,
    *,
    title: str = "",
    db_range: float = 50.0,
) -> None:
    """Display the formed image in dB scale.

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
    ax.set_title(title or "FFBP Stripmap Image", fontsize=13)
    ax.set_xlabel("Range (samples)")
    ax.set_ylabel("Cross-range (samples)")
    fig.colorbar(im, ax=ax, label="dB", shrink=0.8)
    plt.tight_layout()
    plt.show()


# -- Metadata summary -----------------------------------------------


def print_metadata_summary(meta) -> None:
    """Print a compact metadata summary to stdout.

    Parameters
    ----------
    meta : CPHDMetadata
        Metadata from a CPHDReader.
    """
    pvp = meta.pvp
    gp = meta.global_params
    ci = meta.collection_info

    print(f"  Pulses: {pvp.num_vectors}")
    print(f"  Samples: {meta.cols}")

    if ci is not None:
        if ci.collect_type:
            print(f"  Collect type: {ci.collect_type}")
        if ci.radar_mode:
            print(f"  Radar mode: {ci.radar_mode}")

    if gp is not None:
        if gp.domain_type:
            print(f"  Domain: {gp.domain_type}")
        print(f"  Phase SGN: {gp.phase_sgn:+d}")
        if gp.center_frequency:
            print(f"  Center freq: "
                  f"{gp.center_frequency / 1e9:.4f} GHz")
            print(f"  Bandwidth: {gp.bandwidth / 1e6:.2f} MHz")

    if pvp.srp_pos is not None:
        drift = norm(pvp.srp_pos[-1] - pvp.srp_pos[0])
        print(f"  SRP drift: {drift:.1f} m ({drift / 1000:.2f} km)")

    mid_times = 0.5 * (pvp.tx_time + pvp.rcv_time)
    dt = np.mean(np.diff(mid_times))
    print(f"  PRF: {1.0 / dt:.1f} Hz")
    print(f"  Time span: {mid_times[-1] - mid_times[0]:.3f} s")

    if pvp.amp_sf is not None:
        print(f"  AmpSF: min={pvp.amp_sf.min():.4f}, "
              f"max={pvp.amp_sf.max():.4f}, "
              f"mean={pvp.amp_sf.mean():.4f}")

    if pvp.signal is not None:
        n_valid = int(np.sum(pvp.signal > 0))
        print(f"  Valid pulses: {n_valid}/{pvp.num_vectors}")


# -- Main pipeline --------------------------------------------------


def run_ffbp(
    filepath: Path,
    range_weighting: Optional[str] = None,
    leaf_size: int = 8,
    n_angular: int = 128,
    cross_range_spacing: Optional[float] = None,
    apply_amp_sf: bool = True,
    trim_invalid: bool = True,
) -> np.ndarray:
    """Run the Fast Factorized Back-Projection pipeline.

    Parameters
    ----------
    filepath : Path
        Path to the CPHD file.
    range_weighting : str, optional
        Range weighting window.
    leaf_size : int
        Number of pulses per leaf subaperture.
    n_angular : int
        Angular samples per tree node.
    cross_range_spacing : float, optional
        Output cross-range pixel spacing in metres.
    apply_amp_sf : bool
        Apply per-pulse AmpSF normalization.
    trim_invalid : bool
        Trim invalid pulses.

    Returns
    -------
    np.ndarray
        Complex SAR image.
    """
    t0 = time.perf_counter()

    # -- Read CPHD --
    print(f"Opening: {filepath}")
    with CPHDReader(filepath) as reader:
        meta = reader.metadata

        print("\nMetadata summary:")
        print_metadata_summary(meta)

        print("\nReading signal data...")
        signal = reader.read_full()
        print(f"  Signal shape: {signal.shape}, dtype: {signal.dtype}")

    t_read = time.perf_counter()
    print(f"  Read time: {t_read - t0:.2f}s")

    # -- FFBP --
    print("\nConfiguring FastBackProjection...")
    from grdl.interpolation import PolyphaseInterpolator
    interpolator = PolyphaseInterpolator(
        kernel_length=8, num_phases=128, prototype='kaiser',
    )

    ffbp = FastBackProjection(
        metadata=meta,
        interpolator=interpolator,
        range_weighting=range_weighting,
        leaf_size=leaf_size,
        n_angular=n_angular,
        cross_range_spacing=cross_range_spacing,
        apply_amp_sf=apply_amp_sf,
        trim_invalid=trim_invalid,
        verbose=True,
    )

    # Print output grid metadata
    grid = ffbp.get_output_grid()
    print(f"\nOutput grid:")
    print(f"  Range resolution: {grid['range_resolution']:.3f} m")
    print(f"  Azimuth resolution: {grid['azimuth_resolution']:.3f} m")
    if 'graze_angle_deg' in grid:
        print(f"  Graze angle: {grid['graze_angle_deg']:.2f} deg")
    if 'side_of_track' in grid:
        print(f"  Side of track: {grid['side_of_track']}")
    if 'iarp_llh' in grid:
        llh = grid['iarp_llh']
        print(f"  IARP: {llh[0]:.4f}N, {llh[1]:.4f}E, "
              f"{llh[2]:.1f}m")

    # Form image
    print("\nForming image...")
    t_form = time.perf_counter()
    image = ffbp.form_image(signal, geometry=None)
    t_done = time.perf_counter()

    print(f"\nImage formed: {image.shape}, dtype: {image.dtype}")
    print(f"  Formation time: {t_done - t_form:.2f}s")
    print(f"  Total time: {t_done - t0:.2f}s")

    # -- Display --
    title = filepath.name
    ci = meta.collection_info
    if ci and ci.collector_name:
        title += f"  |  {ci.collector_name}  |  FFBP"
    display_image(image, title=title)

    return image


if __name__ == "__main__":
    args = parse_args()
    run_ffbp(
        args.filepath,
        range_weighting=args.range_weighting,
        leaf_size=args.leaf_size,
        n_angular=args.n_angular,
        cross_range_spacing=args.cross_range_spacing,
        apply_amp_sf=not args.no_amp_sf,
        trim_invalid=not args.no_trim,
    )
