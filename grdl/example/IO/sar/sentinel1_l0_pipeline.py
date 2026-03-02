# -*- coding: utf-8 -*-
"""
Sentinel-1 L0 Pipeline — End-to-end SAR processing from raw Level-0 data.

Demonstrates the full GRDL + sarkit pipeline:
  1. Read Sentinel-1 Level-0 SAFE → Write NGA CRSD
  2. Convert CRSD → CPHD (FX-domain, with range FFT)
  3. Form a SAR image via the Range-Doppler Algorithm
  4. Write the complex image as SICD NITF

Note: Sentinel-1 IW mode is a TOPS (burst) acquisition.  A standard
stripmap RDA does not perform burst-specific Doppler deramping, so the
resulting image will exhibit defocusing.  For a focused IW image, a
TOPS-aware processor would be needed.  This example demonstrates the
*data pipeline* rather than production-quality focusing.

Usage:
  python sentinel1_l0_pipeline.py <safe_dir>
  python sentinel1_l0_pipeline.py <safe_dir> --channel VV
  python sentinel1_l0_pipeline.py <safe_dir> --max-pulses 4096
  python sentinel1_l0_pipeline.py <safe_dir> --output-dir /tmp/output
  python sentinel1_l0_pipeline.py --help

Dependencies
------------
sentinel1decoder, sarkit, sarpy, scipy, lxml, numpy, matplotlib (optional)

Author
------
Jason Fritz, PhD
fritz-jason@zai.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-03-02

Modified
--------
2026-03-02
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
from grdl.IO.sar import Sentinel1L0ToCRSD, CRSDToCPHD, CPHDReader, SICDWriter
from grdl.image_processing.sar import RangeDopplerAlgorithm


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sentinel-1 L0 SAFE → CRSD → CPHD → SICD pipeline.",
    )
    parser.add_argument(
        "safe_path",
        type=Path,
        help="Path to the Sentinel-1 Level-0 .SAFE directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to parent of SAFE directory.",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Polarization channel to process (e.g., VV, VH). "
             "Defaults to first available.",
    )
    parser.add_argument(
        "--max-pulses",
        type=int,
        default=None,
        help="Limit processing to this many pulses (for memory/speed).",
    )
    parser.add_argument(
        "--block-size",
        type=str,
        default="auto",
        help="RDA subaperture block size: 'auto', integer, or '0'.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        default=False,
        help="Skip image display.",
    )
    return parser.parse_args()


# ── Stage 3: CPHD → Image (RDA) ─────────────────────────────────────


def cphd_to_sicd(
    cphd_path: Path,
    sicd_path: Path,
    block_size: str = "auto",
    max_pulses: Optional[int] = None,
) -> np.ndarray:
    """Form a SAR image from CPHD using the Range-Doppler Algorithm.

    Parameters
    ----------
    cphd_path : Path
        Input CPHD file.
    sicd_path : Path
        Output SICD NITF file.
    block_size : str
        Subaperture block size: 'auto', integer string, or '0'.
    max_pulses : int, optional
        Limit pulses (should match the CPHD file).

    Returns
    -------
    np.ndarray
        Complex SAR image.
    """
    print(f"\n{'='*60}")
    print("Stage 3–4: CPHD → Image → SICD")
    print(f"{'='*60}")

    t0 = time.perf_counter()

    # Read CPHD
    print(f"  Reading CPHD: {cphd_path.name}")
    with CPHDReader(cphd_path) as reader:
        meta = reader.metadata

        print(f"  Vectors: {meta.rows}, Samples: {meta.cols}")
        if meta.global_params:
            gp = meta.global_params
            print(f"  Domain: {gp.domain_type}, PhaseSGN: {gp.phase_sgn}")
            if gp.center_frequency:
                print(f"  Center freq: {gp.center_frequency / 1e9:.4f} GHz")
                print(f"  Bandwidth: {gp.bandwidth / 1e6:.2f} MHz")

        if meta.pvp:
            pvp = meta.pvp
            print(f"  PVP vectors: {pvp.num_vectors}")
            if pvp.srp_pos is not None:
                r_srp = norm(
                    pvp.srp_pos[0] - 0.5 * (pvp.tx_pos[0] + pvp.rcv_pos[0])
                )
                print(f"  Slant range: {r_srp / 1000:.1f} km")
            if pvp.amp_sf is not None:
                n_valid = int(np.sum(pvp.signal > 0)) if pvp.signal is not None else pvp.num_vectors
                print(f"  Valid pulses: {n_valid}/{pvp.num_vectors}")

        # Read signal
        print("  Reading signal data...")
        signal = reader.read_full()
        print(f"  Signal: {signal.shape}, dtype: {signal.dtype}")

    t_read = time.perf_counter()
    print(f"  Read time: {t_read - t0:.2f}s")

    # Configure RDA
    print("\n  Configuring RangeDopplerAlgorithm...")
    if block_size == "auto":
        resolved_block = "auto"
    elif block_size == "0":
        resolved_block = None
    else:
        resolved_block = int(block_size)

    from grdl.interpolation import PolyphaseInterpolator
    interpolator = PolyphaseInterpolator(
        kernel_length=8, num_phases=128, prototype='kaiser',
    )

    rda = RangeDopplerAlgorithm(
        metadata=meta,
        interpolator=interpolator,
        range_weighting='taylor',
        block_size=resolved_block,
        overlap=0.5,
        apply_amp_sf=True,
        trim_invalid=True,
        verbose=True,
    )

    # Output grid info
    grid = rda.get_output_grid()
    print(f"\n  Output grid:")
    print(f"    Range resolution: {grid['range_resolution']:.3f} m")
    print(f"    Azimuth resolution: {grid['azimuth_resolution']:.3f} m")

    # Form image
    print("\n  Forming image...")
    t_form = time.perf_counter()
    image = rda.form_image(signal, geometry=None)
    t_done = time.perf_counter()

    print(f"\n  Image: {image.shape}, dtype: {image.dtype}")
    print(f"  Formation time: {t_done - t_form:.2f}s")

    del signal  # Free memory

    # Write SICD via sarpy directly with minimal required metadata
    print(f"\n  Writing SICD: {sicd_path.name}")
    from sarpy.io.complex.sicd_elements.SICD import SICDType
    from sarpy.io.complex.sicd_elements.ImageData import (
        ImageDataType, FullImageType,
    )
    from sarpy.io.complex.sicd_elements.CollectionInfo import (
        CollectionInfoType, RadarModeType,
    )
    from sarpy.io.complex.sicd_elements.Timeline import TimelineType
    from sarpy.io.complex.sicd import SICDWriter as _SarpySICDWriter
    from datetime import datetime

    nr, nc = image.shape
    sicd_type = SICDType()
    sicd_type.ImageData = ImageDataType(
        PixelType='RE32F_IM32F',
        NumRows=nr,
        NumCols=nc,
        FullImage=FullImageType(NumRows=nr, NumCols=nc),
        SCPPixel=[nr // 2, nc // 2],
        ValidData=[[0, 0], [0, nc - 1], [nr - 1, nc - 1], [nr - 1, 0]],
    )
    sicd_type.CollectionInfo = CollectionInfoType(
        Classification='UNCLASSIFIED',
        CoreName=cphd_path.stem,
        RadarMode=RadarModeType(ModeType='DYNAMIC STRIPMAP', ModeID='IW'),
    )
    sicd_type.Timeline = TimelineType(
        CollectStart=datetime.utcnow(),
        CollectDuration=1.0,
    )

    sicd_writer = _SarpySICDWriter(
        str(sicd_path), sicd_type, check_existence=False,
    )
    sicd_writer.write(image.astype(np.complex64))
    size_mb = sicd_path.stat().st_size / (1024**2)
    print(f"  SICD written: {size_mb:.1f} MB")
    print(f"  Total stage 3–4 time: {time.perf_counter() - t0:.2f}s")

    return image


# ── Display ──────────────────────────────────────────────────────────


def display_image(image: np.ndarray, *, title: str = "", db_range: float = 50.0):
    """Display the formed image in dB scale."""
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
    ax.set_title(title or "SAR Image", fontsize=13)
    ax.set_xlabel("Range (samples)")
    ax.set_ylabel("Azimuth (samples)")
    fig.colorbar(im, ax=ax, label="dB", shrink=0.8)
    plt.tight_layout()
    plt.show()


# ── Main pipeline ────────────────────────────────────────────────────


def run_pipeline(
    safe_path: Path,
    output_dir: Optional[Path] = None,
    channel: Optional[str] = None,
    max_pulses: Optional[int] = None,
    block_size: str = "auto",
    no_display: bool = False,
) -> None:
    """Run the full L0 SAFE → CRSD → CPHD → SICD pipeline."""
    t_start = time.perf_counter()

    if output_dir is None:
        output_dir = safe_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive output filenames
    stem = safe_path.name
    if stem.upper().endswith(".SAFE"):
        stem = stem[:-5]  # Remove .SAFE
    crsd_path = output_dir / f"{stem}.crsd"
    cphd_path = output_dir / f"{stem}.cphd"
    sicd_path = output_dir / f"{stem}.sicd"

    # ── Stage 1: L0 SAFE → CRSD ──
    print(f"\n{'='*60}")
    print("Stage 1: Sentinel-1 L0 SAFE → CRSD")
    print(f"{'='*60}")
    print(f"  Input:  {safe_path}")
    print(f"  Output: {crsd_path}")

    t0 = time.perf_counter()
    converter = Sentinel1L0ToCRSD(
        safe_path=safe_path,
        output_path=crsd_path,
        channel=channel,
    )
    converter.convert()
    t1 = time.perf_counter()
    size_gb = crsd_path.stat().st_size / (1024**3)
    print(f"  CRSD written: {size_gb:.2f} GB in {t1 - t0:.1f}s")

    # ── Stage 2: CRSD → CPHD ──
    print(f"\n{'='*60}")
    print("Stage 2: CRSD → CPHD")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    crsd_to_cphd = CRSDToCPHD(
        crsd_path=crsd_path,
        output_path=cphd_path,
        channel=channel,
        max_pulses=max_pulses,
    )
    crsd_to_cphd.convert()
    t1 = time.perf_counter()
    size_gb = cphd_path.stat().st_size / (1024**3)
    print(f"  CPHD written: {size_gb:.2f} GB in {t1 - t0:.1f}s")

    # ── Stages 3–4: CPHD → Image → SICD ──
    image = cphd_to_sicd(
        cphd_path=cphd_path,
        sicd_path=sicd_path,
        block_size=block_size,
        max_pulses=max_pulses,
    )

    # ── Summary ──
    t_total = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print("Pipeline Complete")
    print(f"{'='*60}")
    print(f"  CRSD: {crsd_path}")
    print(f"  CPHD: {cphd_path}")
    print(f"  SICD: {sicd_path}")
    print(f"  Image: {image.shape}")
    print(f"  Total time: {t_total:.1f}s")

    # ── Display ──
    if not no_display:
        title = f"{safe_path.name}  |  RDA"
        display_image(image, title=title)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        safe_path=args.safe_path,
        output_dir=args.output_dir,
        channel=args.channel,
        max_pulses=args.max_pulses,
        block_size=args.block_size,
        no_display=args.no_display,
    )
