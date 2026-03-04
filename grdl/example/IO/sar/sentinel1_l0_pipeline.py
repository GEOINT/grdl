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
  python sentinel1_l0_pipeline.py <safe_dir> --info
  python sentinel1_l0_pipeline.py <safe_dir> --channel VV --swath 1
  python sentinel1_l0_pipeline.py <safe_dir> --max-pulses 4096 --loglevel INFO
  python sentinel1_l0_pipeline.py <safe_dir> --loglevel DEBUG --logfile run.log
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
2026-03-03
"""

# Standard library
import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# GRDL
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from grdl.IO.sar import Sentinel1L0ToCRSD, CRSDToCPHD, CPHDToSICD
from grdl.IO.sar.sentinel1_l0 import Sentinel1L0Reader

logger = logging.getLogger(__name__)


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
             "Defaults to all available.",
    )
    parser.add_argument(
        "--swath",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="IW sub-swath number (1, 2, or 3). Defaults to all.",
    )
    parser.add_argument(
        "--max-pulses",
        type=int,
        default=None,
        help="Limit processing to this many pulses (for memory/speed).",
    )
    parser.add_argument(
        "--burst",
        type=int,
        default=None,
        help="1-based burst index for TOPSAR/IW mode. "
             "Selects a single contiguous burst for image formation "
             "(avoids inter-burst gaps that corrupt the RDA). "
             "If not set, all vectors are processed.",
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
    parser.add_argument(
        "--info",
        action="store_true",
        default=False,
        help="Print product info (polarizations, swaths, pulses) and exit.",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level (default: WARNING).",
    )
    parser.add_argument(
        "--logfile",
        type=Path,
        default=None,
        help="Write log messages to this file in addition to stderr.",
    )
    return parser.parse_args()


# ── Logging setup ───────────────────────────────────────────────────


def setup_logging(loglevel: str, logfile: Optional[Path] = None) -> None:
    """Configure the root logger with console and optional file handlers."""
    level = getattr(logging, loglevel.upper(), logging.WARNING)
    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(logfile)))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


# ── Product info ────────────────────────────────────────────────────


def print_info(safe_path: Path) -> None:
    """Print product metadata and exit without processing."""
    from grdl.IO.models import Sentinel1L0Metadata

    reader = Sentinel1L0Reader(safe_path)
    meta: Sentinel1L0Metadata = reader.metadata  # type: ignore[assignment]
    pi = meta.product_info
    radar = meta.radar_params
    assert pi is not None, "No product info in manifest"
    assert radar is not None, "No radar params decoded"

    print(f"{'='*60}")
    print("Sentinel-1 Level-0 Product Info")
    print(f"{'='*60}")
    print(f"  SAFE:              {safe_path.name}")
    print(f"  Mission:           {pi.mission}")
    print(f"  Mode:              {pi.mode}")
    print(f"  Product type:      {pi.product_type}")
    print(f"  Polarization mode: {pi.polarization_mode}")
    print(f"  Start time:        {pi.start_time}")
    print(f"  Stop time:         {pi.stop_time}")
    print(f"  Absolute orbit:    {pi.absolute_orbit}")
    print(f"  Relative orbit:    {pi.relative_orbit}")
    print(f"  Orbit pass:        {pi.orbit_pass}")
    print(f"  Data take ID:      {meta.data_take_id}")

    print(f"\n  Radar Parameters")
    print(f"  {'─'*40}")
    print(f"  Center frequency:  {radar.center_frequency / 1e9:.4f} GHz")
    print(f"  TX bandwidth:      {radar.tx_bandwidth / 1e6:.2f} MHz")
    print(f"  Chirp rate:        {radar.chirp_rate:.6g} Hz/s")
    print(f"  Sampling rate:     {radar.sampling_rate / 1e6:.2f} MHz")
    print(f"  TX pulse length:   {radar.tx_pulse_length * 1e6:.2f} us")
    print(f"  PRI:               {radar.pri * 1e6:.2f} us")

    print(f"\n  Polarization Channels")
    print(f"  {'─'*40}")
    for pol, ch in meta.channels.items():
        print(f"  {pol}:")
        print(f"    TX/RX:           {ch.tx_pol}/{ch.rx_pol}")
        print(f"    Echo packets:    {ch.num_echo_packets}")
        print(f"    Max quads:       {ch.max_num_quads}  ({2*ch.max_num_quads} samples)")
        print(f"    Swath numbers:   {ch.swath_numbers}")
        print(f"    Measurement:     {Path(ch.measurement_file).name}")

        # Per-swath breakdown
        swath_info = reader.get_swath_info(pol)
        print(f"\n    Per-swath breakdown:")
        print(f"    {'Swath':<8} {'Channel':<12} {'Packets':>8} {'Max Quads':>10} {'Samples':>10}")
        print(f"    {'─'*52}")
        for sn in sorted(swath_info.keys()):
            si = swath_info[sn]
            print(
                f"    IW{si.swath_number:<5} {si.channel_key:<12} "
                f"{si.num_echo_packets:>8} {si.max_num_quads:>10} "
                f"{2*si.max_num_quads:>10}"
            )
        print()

    if meta.footprint:
        print(f"  Footprint")
        print(f"  {'─'*40}")
        for i, c in enumerate(meta.footprint):
            print(f"    [{i}]  lat={c.lat:.6f}  lon={c.lon:.6f}")

    reader.close()


# ── Display ──────────────────────────────────────────────────────────


def launch_viewer(filepath: Path) -> None:
    """Open a file in grdk-viewer (non-blocking)."""
    viewer = "grdk-viewer"
    logger.info("Launching %s %s", viewer, filepath.name)
    try:
        subprocess.Popen([viewer, str(filepath)])
    except FileNotFoundError:
        logger.warning(
            "%s not found on PATH; skipping display. "
            "Install grdk or open the file manually.",
            viewer,
        )


# ── Main pipeline ────────────────────────────────────────────────────


def run_pipeline(
    safe_path: Path,
    output_dir: Optional[Path] = None,
    channel: Optional[str] = None,
    swath: Optional[int] = None,
    max_pulses: Optional[int] = None,
    burst: Optional[int] = None,
    block_size: str = "auto",
    no_display: bool = False,
) -> None:
    """Run the full L0 SAFE → CRSD → CPHD → SICD pipeline.

    When *swath* and/or *channel* are specified, filenames include a
    swath+pol suffix, e.g. ``stem.IW1_VV.crsd``.
    """
    t_start = time.perf_counter()

    if output_dir is None:
        output_dir = safe_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive output filenames with swath+pol suffix
    stem = safe_path.name
    if stem.upper().endswith(".SAFE"):
        stem = stem[:-5]  # Remove .SAFE

    # Build suffix from swath/channel selection
    suffix_parts = []
    if swath is not None:
        suffix_parts.append(f"IW{swath}")
    if channel is not None:
        suffix_parts.append(channel)
    file_suffix = "_".join(suffix_parts) if suffix_parts else ""

    if file_suffix:
        crsd_path = output_dir / f"{stem}.{file_suffix}.crsd"
        cphd_path = output_dir / f"{stem}.{file_suffix}.cphd"
        sicd_path = output_dir / f"{stem}.{file_suffix}.sicd"
    else:
        crsd_path = output_dir / f"{stem}.crsd"
        cphd_path = output_dir / f"{stem}.cphd"
        sicd_path = output_dir / f"{stem}.sicd"

    # Determine CRSD channel ID for downstream steps
    # CRSD channels are keyed as "IW{n}_{pol}", e.g. "IW1_VV"
    crsd_channel_id = file_suffix if file_suffix else None

    # ── Stage 1: L0 SAFE → CRSD ──
    print(f"\n{'='*60}")
    print("Stage 1: Sentinel-1 L0 SAFE → CRSD")
    print(f"{'='*60}")
    print(f"  Input:  {safe_path}")
    print(f"  Output: {crsd_path}")

    logger.info("Stage 1: L0 SAFE → CRSD  input=%s  output=%s", safe_path, crsd_path)
    t0 = time.perf_counter()
    converter = Sentinel1L0ToCRSD(
        safe_path=safe_path,
        output_path=crsd_path,
        channel=channel,
        swath=swath,
    )
    converter.convert()
    t1 = time.perf_counter()
    size_gb = crsd_path.stat().st_size / (1024**3)
    logger.info("Stage 1 complete: %.2f GB in %.1fs", size_gb, t1 - t0)
    print(f"  CRSD written: {size_gb:.2f} GB in {t1 - t0:.1f}s")

    # ── Stage 2: CRSD → CPHD ──
    print(f"\n{'='*60}")
    print("Stage 2: CRSD → CPHD")
    print(f"{'='*60}")

    logger.info("Stage 2: CRSD → CPHD  input=%s  output=%s", crsd_path, cphd_path)
    t0 = time.perf_counter()
    crsd_to_cphd = CRSDToCPHD(
        crsd_path=crsd_path,
        output_path=cphd_path,
        channel=crsd_channel_id,
        max_pulses=max_pulses,
    )
    crsd_to_cphd.convert()
    t1 = time.perf_counter()
    size_gb = cphd_path.stat().st_size / (1024**3)
    logger.info("Stage 2 complete: %.2f GB in %.1fs", size_gb, t1 - t0)
    print(f"  CPHD written: {size_gb:.2f} GB in {t1 - t0:.1f}s")

    # ── Stages 3–4: CPHD → Image → SICD ──
    print(f"\n{'='*60}")
    print("Stage 3–4: CPHD → Image → SICD")
    print(f"{'='*60}")

    logger.info("Stage 3-4: CPHD → Image → SICD  input=%s  output=%s", cphd_path, sicd_path)
    t0 = time.perf_counter()
    cphd_to_sicd = CPHDToSICD(
        cphd_path=cphd_path,
        output_path=sicd_path,
        block_size=block_size,
        burst=burst,
        max_pulses=max_pulses,
    )
    cphd_to_sicd.convert()
    t1 = time.perf_counter()
    size_mb = sicd_path.stat().st_size / (1024**2)
    logger.info("Stage 3-4 complete: %.1f MB in %.1fs", size_mb, t1 - t0)
    print(f"  SICD written: {size_mb:.1f} MB in {t1 - t0:.1f}s")

    # ── Summary ──
    t_total = time.perf_counter() - t_start
    logger.info("Pipeline complete: %s  total=%.1fs", safe_path.name, t_total)
    print(f"\n{'='*60}")
    print("Pipeline Complete")
    print(f"{'='*60}")
    print(f"  CRSD: {crsd_path}")
    print(f"  CPHD: {cphd_path}")
    print(f"  SICD: {sicd_path}")
    print(f"  Image: {cphd_to_sicd.image.shape if cphd_to_sicd.image is not None else 'N/A'}")
    print(f"  Total time: {t_total:.1f}s")

    # ── Display ──
    if not no_display:
        launch_viewer(sicd_path)


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.loglevel, args.logfile)

    if args.info:
        print_info(args.safe_path)
        sys.exit(0)

    run_pipeline(
        safe_path=args.safe_path,
        output_dir=args.output_dir,
        channel=args.channel,
        swath=args.swath,
        max_pulses=args.max_pulses,
        burst=args.burst,
        block_size=args.block_size,
        no_display=args.no_display,
    )
