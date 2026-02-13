# -*- coding: utf-8 -*-
"""
RDA Stripmap Example - Form a SAR image using the Range-Doppler Algorithm.

End-to-end demonstration of the GRDL Range-Doppler Algorithm pipeline.
Reads a stripmap CPHD file, processes the full aperture coherently via
range compression, RCMC, and azimuth compression.

Demonstrates full GRDL integration:
  - ``grdl.IO.sar.CPHDReader`` for metadata + signal access
  - ``grdl.image_processing.sar.RangeDopplerAlgorithm`` for full-aperture IFP

Usage:
  python rda_stripmap_example.py <cphd_file>
  python rda_stripmap_example.py <cphd_file> --output output.nitf
  python rda_stripmap_example.py <cphd_file> --range-weighting taylor
  python rda_stripmap_example.py --help

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


# -- Main pipeline --------------------------------------------------


def run_rda(
    filepath: Path,
    output: Optional[Path] = None,
    range_weighting: Optional[str] = None,
    azimuth_weighting: Optional[str] = None,
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
        verbose=True,
    )

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
    )
