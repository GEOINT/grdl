# -*- coding: utf-8 -*-
"""
CRSD comparison tool — diff two CRSD files channel-by-channel.

Compares metadata, PVP (per-vector parameters), and signal data
between two CRSD files produced by different converters (e.g.
GRDL vs NGA/Valkyrie).  Reports channel counts, orbit position
deltas, timing offsets, signal power ratios, and sample-count
mismatches.

Usage
-----
  python compare_crsd.py FILE_A FILE_B
  python compare_crsd.py --help

When channel identifiers differ between the two files (common
when converters use different burst-counter schemes), channels
are matched by position order.

Dependencies
------------
sarkit

Author
------
Jason Fritz
43161141+stryder-vtx@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2025-05-22

Modified
--------
2025-05-26
"""

# Standard library
import argparse
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
from sarkit.crsd import Reader

# ---------------------------------------------------------------------------
# CRSD XML namespace
# ---------------------------------------------------------------------------
NS = {"c": "http://api.nsgreg.nga.mil/schema/crsd/1.0"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_channel_ids(path: Path) -> list:
    """Extract ordered channel identifiers from a CRSD file."""
    with open(path, "rb") as f:
        r = Reader(f)
        try:
            root = r.metadata.xmltree.getroot()
            params = root.findall(".//c:Channel/c:Parameters", NS)
            channel_ids = [p.find("c:Identifier", NS).text for p in params]
        finally:
            r.done()
        return channel_ids

def _rcvstart_to_sec(rs) -> float:
    """Convert a structured RcvStart (Int, Frac) to seconds."""
    return float(rs["Int"]) + float(rs["Frac"])


def _signal_power(sig: np.ndarray) -> float:
    """Mean power of a CI2 signal array (int8 real/imag struct)."""
    r = sig["real"].astype(np.float32)
    i = sig["imag"].astype(np.float32)
    return float(np.mean(r**2 + i**2))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(path_a: Path, path_b: Path, *, label_a: str = "A",
            label_b: str = "B", max_detail: int = 3) -> None:
    """Compare two CRSD files and print a summary.

    Parameters
    ----------
    path_a, path_b : Path
        Paths to the two CRSD files.
    label_a, label_b : str
        Short labels used in output (e.g. ``"GRDL"``, ``"NGA"``).
    max_detail : int
        Number of channels to show detailed PVP/signal analysis
        for.  Set to 0 for summary only, -1 for all channels.
    """
    # ── File sizes ──────────────────────────────────────────────
    sz_a = os.path.getsize(path_a)
    sz_b = os.path.getsize(path_b)
    print(f"File sizes: {label_a}={sz_a / 1e6:.1f} MB, "
          f"{label_b}={sz_b / 1e6:.1f} MB, "
          f"delta={abs(sz_a - sz_b) / 1e6:.1f} MB")

    # ── Channel IDs ─────────────────────────────────────────────
    ids_a = _get_channel_ids(path_a)
    ids_b = _get_channel_ids(path_b)
    print(f"\nChannels: {label_a}={len(ids_a)}, "
          f"{label_b}={len(ids_b)}")
    print(f"  {label_a} IDs: {ids_a[:4]} ... {ids_a[-2:]}")
    print(f"  {label_b} IDs: {ids_b[:4]} ... {ids_b[-2:]}")

    common = sorted(set(ids_a) & set(ids_b))
    only_a = sorted(set(ids_a) - set(ids_b))
    only_b = sorted(set(ids_b) - set(ids_a))
    print(f"  Common: {len(common)}, "
          f"{label_a}-only: {len(only_a)}, "
          f"{label_b}-only: {len(only_b)}")

    # Build ordered pairs
    if common:
        pairs = [(c, c) for c in common]
    elif len(ids_a) == len(ids_b):
        print("  IDs differ (different burst-counter scheme) "
              "— matching by position order")
        pairs = list(zip(ids_a, ids_b))
    else:
        print("  Channel count mismatch and no common IDs "
              "— cannot compare.")
        return

    # ── Detailed per-channel comparison ─────────────────────────
    detail_count = len(pairs) if max_detail < 0 else max_detail
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        ra = Reader(fa)
        rb = Reader(fb)

        for idx, (id_a, id_b) in enumerate(pairs[:detail_count]):
            print(f"\n{'─' * 60}")
            print(f"Channel {idx + 1}/{len(pairs)}: "
                  f"{label_a}={id_a} / {label_b}={id_b}")
            print("─" * 60)

            pvp_a = ra.read_pvps(id_a)
            pvp_b = rb.read_pvps(id_b)
            sig_a = ra.read_signal(id_a)
            sig_b = rb.read_signal(id_b)

            print(f"  PVP vectors: {label_a}={len(pvp_a)}, "
                  f"{label_b}={len(pvp_b)}")
            print(f"  Signal shape: {label_a}={sig_a.shape}, "
                  f"{label_b}={sig_b.shape}")

            # Position
            pos_a = np.array(
                [pvp_a["RcvPos"][i] for i in range(len(pvp_a))]
            )
            pos_b = np.array(
                [pvp_b["RcvPos"][i] for i in range(len(pvp_b))]
            )
            n = min(len(pos_a), len(pos_b))
            pos_diff = np.linalg.norm(
                pos_a[:n] - pos_b[:n], axis=1
            )
            print(f"  RcvPos diff (m): mean={pos_diff.mean():.3f}, "
                  f"max={pos_diff.max():.3f}")

            rad_a = np.linalg.norm(pos_a[0])
            rad_b = np.linalg.norm(pos_b[0])
            print(f"  Orbit radius: {label_a}={rad_a / 1000:.1f} km, "
                  f"{label_b}={rad_b / 1000:.1f} km")

            # Time
            t0_a = _rcvstart_to_sec(pvp_a["RcvStart"][0])
            t0_b = _rcvstart_to_sec(pvp_b["RcvStart"][0])
            dt = abs(t0_a - t0_b)
            print(f"  RcvStart[0]: {label_a}={t0_a:.6f}, "
                  f"{label_b}={t0_b:.6f}, diff={dt:.6f} s")

            # Power
            pwr_a = _signal_power(sig_a)
            pwr_b = _signal_power(sig_b)
            ratio = pwr_a / pwr_b if pwr_b > 0 else float("inf")
            print(f"  Mean power: {label_a}={pwr_a:.1f}, "
                  f"{label_b}={pwr_b:.1f}, ratio={ratio:.3f}")

    # ── Sample-count comparison across all channels ─────────────
    print(f"\n{'═' * 60}")
    print(f"Sample counts — all {len(pairs)} channels")
    print("═" * 60)
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        ra = Reader(fa)
        rb = Reader(fb)
        matches = 0
        diffs = 0
        for id_a, id_b in pairs:
            sa = ra.read_signal(id_a)
            sb = rb.read_signal(id_b)
            if sa.shape[1] == sb.shape[1]:
                matches += 1
                status = "OK"
            else:
                diffs += 1
                status = (f"DIFF ({sa.shape[1]} vs "
                          f"{sb.shape[1]})")
            print(f"  {id_a:30s}  {sa.shape}  vs  "
                  f"{sb.shape}  {status}")
        print(f"\n  {matches} match, {diffs} differ")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Compare two CRSD files channel-by-channel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Compare GRDL output against NGA reference:\n"
            "  python compare_crsd.py grdl.crsd nga.crsd "
            "--labels GRDL NGA\n\n"
            "  # Show detailed analysis for all channels:\n"
            "  python compare_crsd.py a.crsd b.crsd "
            "--max-detail -1\n\n"
            "  # Summary only (sample counts, no PVP detail):\n"
            "  python compare_crsd.py a.crsd b.crsd "
            "--max-detail 0"
        ),
    )
    parser.add_argument(
        "file_a", type=Path, help="First CRSD file"
    )
    parser.add_argument(
        "file_b", type=Path, help="Second CRSD file"
    )
    parser.add_argument(
        "--labels", nargs=2, default=["A", "B"],
        metavar=("LABEL_A", "LABEL_B"),
        help="Short labels for the two files (default: A B)",
    )
    parser.add_argument(
        "--max-detail", type=int, default=3,
        help=(
            "Number of channels to show detailed PVP/signal "
            "analysis for. 0 = summary only, -1 = all channels. "
            "(default: 3)"
        ),
    )
    args = parser.parse_args()

    for p in (args.file_a, args.file_b):
        if not p.exists():
            print(f"Error: {p} not found", file=sys.stderr)
            sys.exit(1)

    compare(
        args.file_a, args.file_b,
        label_a=args.labels[0],
        label_b=args.labels[1],
        max_detail=args.max_detail,
    )


if __name__ == "__main__":
    main()
