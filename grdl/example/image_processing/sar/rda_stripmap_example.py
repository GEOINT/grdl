# -*- coding: utf-8 -*-
"""
RDA Stripmap Example - Form a SAR image using the Range-Doppler Algorithm.

End-to-end demonstration of the GRDL Range-Doppler Algorithm pipeline.
Reads a stripmap CPHD file and forms a complex SAR image via the
subaperture RDA: overlapping azimuth blocks with local rephasing,
per-block Doppler centroid estimation, baseband demodulation,
windowing, and Hann-weighted mosaicking.

Demonstrates full GRDL integration and CPHD metadata utilization:
  - ``grdl.IO.sar.CPHDReader`` for metadata + signal access
  - ``grdl.image_processing.sar.RangeDopplerAlgorithm`` for IFP
  - AmpSF normalization and invalid pulse trimming
  - Physics-driven subaperture auto-sizing (``--block-size auto``)
  - Antenna pattern compensation (``--antenna-compensation``)
  - Per-block Doppler centroid demodulation
  - Reference geometry and scene coordinates in output grid

Usage:
  python rda_stripmap_example.py <cphd_file>
  python rda_stripmap_example.py <cphd_file> --block-size auto
  python rda_stripmap_example.py <cphd_file> --block-size 8000
  python rda_stripmap_example.py <cphd_file> --output output.nitf
  python rda_stripmap_example.py <cphd_file> --range-weighting taylor
  python rda_stripmap_example.py <cphd_file> --antenna-compensation
  python rda_stripmap_example.py --help

Dependencies
------------
matplotlib (optional, for plots)
sarkit or sarpy

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
2026-02-13

Modified
--------
2026-02-15
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
from grdl.image_processing.sar import RangeDopplerAlgorithm


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
                    "using the Range-Doppler Algorithm.",
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
        "--output", "-o",
        type=Path,
        default=None,
        help="Output SICD NITF path. If omitted, displays only.",
    )
    parser.add_argument(
        "--range-weighting",
        type=str,
        default=None,
        choices=["uniform", "taylor", "hamming", "hanning"],
        help="Range weighting window (default: none).",
    )
    parser.add_argument(
        "--azimuth-weighting",
        type=str,
        default=None,
        choices=["uniform", "taylor", "hamming", "hanning"],
        help="Azimuth weighting window (default: none).",
    )
    parser.add_argument(
        "--block-size",
        type=str,
        default="0",
        help="Subaperture block size in pulses. "
             "Use 'auto' for physics-driven sizing (default), "
             "an integer, or '0' for single-reference mode.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap fraction between blocks (default: 0.5).",
    )
    parser.add_argument(
        "--antenna-compensation",
        action="store_true",
        default=False,
        help="Apply antenna pattern compensation.",
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
    ax.set_title(title or "RDA Stripmap Image", fontsize=13)
    ax.set_xlabel("Range (samples)")
    ax.set_ylabel("Azimuth (samples)")
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

    # New metadata fields
    if pvp.amp_sf is not None:
        print(f"  AmpSF: min={pvp.amp_sf.min():.4f}, "
              f"max={pvp.amp_sf.max():.4f}, "
              f"mean={pvp.amp_sf.mean():.4f}")

    if pvp.signal is not None:
        n_valid = int(np.sum(pvp.signal > 0))
        print(f"  Valid pulses: {n_valid}/{pvp.num_vectors}")

    tx = meta.tx_waveform
    if tx is not None:
        if tx.lfm_rate:
            print(f"  LFM rate: {tx.lfm_rate:.3e} Hz/s")
        if tx.pulse_length:
            print(f"  Pulse length: {tx.pulse_length * 1e6:.2f} us")

    if meta.antenna_pattern is not None:
        ant = meta.antenna_pattern
        if ant.gain_zero is not None:
            print(f"  Antenna gain: {ant.gain_zero:.1f} dB")
        if ant.gain_poly is not None:
            print(f"  Gain polynomial: "
                  f"{ant.gain_poly.shape[0]}x"
                  f"{ant.gain_poly.shape[1]}")

    rg = meta.reference_geometry
    if rg is not None:
        if rg.graze_angle_deg is not None:
            print(f"  Graze angle: {rg.graze_angle_deg:.2f} deg")
        if rg.side_of_track is not None:
            print(f"  Side of track: {rg.side_of_track}")

    sc = meta.scene_coordinates
    if sc is not None and sc.iarp_llh is not None:
        llh = sc.iarp_llh
        print(f"  IARP: {llh[0]:.4f}N, {llh[1]:.4f}E, "
              f"{llh[2]:.1f}m")


# -- Main pipeline --------------------------------------------------


def run_rda(
    filepath: Path,
    output: Optional[Path] = None,
    range_weighting: Optional[str] = None,
    azimuth_weighting: Optional[str] = None,
    block_size: str = "auto",
    overlap: float = 0.5,
    antenna_compensation: bool = False,
    apply_amp_sf: bool = True,
    trim_invalid: bool = True,
) -> np.ndarray:
    """Run the Range-Doppler Algorithm pipeline.

    Parameters
    ----------
    filepath : Path
        Path to the CPHD file.
    output : Path, optional
        If provided, write the formed image as SICD NITF.
    range_weighting : str, optional
        Range weighting window.
    azimuth_weighting : str, optional
        Azimuth weighting window.
    block_size : str
        Subaperture block size: 'auto', an integer string, or '0'.
    overlap : float
        Overlap fraction between subaperture blocks.
    antenna_compensation : bool
        Apply antenna pattern compensation.
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

    # -- Resolve block_size --
    if block_size == "auto":
        resolved_block_size = "auto"
    elif block_size == "0":
        resolved_block_size = None
    else:
        resolved_block_size = int(block_size)

    # -- RDA --
    print("\nConfiguring RangeDopplerAlgorithm...")
    from grdl.interpolation import PolyphaseInterpolator
    interpolator = PolyphaseInterpolator(
        kernel_length=8, num_phases=128, prototype='kaiser',
    )

    rda = RangeDopplerAlgorithm(
        metadata=meta,
        interpolator=interpolator,
        range_weighting=range_weighting,
        azimuth_weighting=azimuth_weighting,
        block_size=resolved_block_size,
        overlap=overlap,
        apply_amp_sf=apply_amp_sf,
        trim_invalid=trim_invalid,
        antenna_compensation=antenna_compensation,
        verbose=True,
    )

    # Print output grid metadata
    grid = rda.get_output_grid()
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
    image = rda.form_image(signal, geometry=None)
    t_done = time.perf_counter()

    print(f"\nImage formed: {image.shape}, dtype: {image.dtype}")
    print(f"  Formation time: {t_done - t_form:.2f}s")
    print(f"  Total time: {t_done - t0:.2f}s")

    # -- Optional: Save SICD --
    if output is not None:
        print(f"\nWriting SICD to: {output}")
        from grdl.IO.sar import SICDWriter
        writer = SICDWriter(output)
        writer.write(image)
        print("  Done.")

    # -- Display --
    title = filepath.name
    ci = meta.collection_info
    if ci and ci.collector_name:
        title += f"  |  {ci.collector_name}  |  RDA"
    display_image(image, title=title)

    return image


if __name__ == "__main__":
    args = parse_args()
    run_rda(
        args.filepath,
        output=args.output,
        range_weighting=args.range_weighting,
        azimuth_weighting=args.azimuth_weighting,
        block_size=args.block_size,
        overlap=args.overlap,
        antenna_compensation=args.antenna_compensation,
        apply_amp_sf=not args.no_amp_sf,
        trim_invalid=not args.no_trim,
    )
