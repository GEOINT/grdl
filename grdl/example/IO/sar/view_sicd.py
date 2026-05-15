# -*- coding: utf-8 -*-
"""
SICD Viewer - Display complex SAR imagery from SICD files using grdk-viewer.

Reads a SICD file and displays the imagery in the interactive grdk-viewer
application. Shows metadata and geolocation information in the console.

Usage:
  python view_sicd.py <sicd_file>
  python view_sicd.py --help

Dependencies
------------
grdk

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
2026-02-09

Modified
--------
2026-05-14
"""

# Standard library
import argparse
import sys
from pathlib import Path

# Third-party
from grdk.viewers import show

# GRDL
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from grdl.IO import SICDReader
from grdl.geolocation import create_geolocation


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Display SICD complex SAR imagery in the grdk-viewer.",
    )
    parser.add_argument(
        "filepath",
        nargs='?',
        type=Path,
        default=Path('default/path/to/file'),
        help="Path to the SICD file (NITF or other SICD container).",
    )
    return parser.parse_args()


def view_sicd(filepath: Path) -> None:
    """Read and display a SICD image in the grdk-viewer.

    Parameters
    ----------
    filepath : Path
        Path to the SICD file.
    """
    print(f"Opening: {filepath}")

    with SICDReader(filepath) as reader:
        meta = reader.metadata
        rows, cols = reader.get_shape()

        # Print metadata
        print(f"  Backend:        {meta.backend or '?'}")
        print(f"  Size:           {rows} x {cols}")

        if meta.image_data is not None:
            print(f"  Pixel type:     {meta.image_data.pixel_type or '?'}")

        ci = meta.collection_info
        if ci is not None:
            print(f"  Collector:      {ci.collector_name or '?'}")
            print(f"  Core name:      {ci.core_name or '?'}")
            print(f"  Classification: {ci.classification or '?'}")

        tl = meta.timeline
        if tl is not None:
            print(f"  Collect start:  {tl.collect_start or '?'}")
            if tl.collect_duration is not None:
                print(f"  Duration:       {tl.collect_duration:.3f} s")

        scp_llh = None
        if meta.geo_data is not None and meta.geo_data.scp is not None:
            scp_llh = meta.geo_data.scp.llh
        if scp_llh is not None:
            print(f"  SCP:            ({scp_llh.lat:.6f}, {scp_llh.lon:.6f}, "
                  f"{scp_llh.hae:.1f} m)")

        scpcoa = meta.scpcoa
        if scpcoa is not None:
            print(f"  Side of track:  {scpcoa.side_of_track or '?'}")
            if scpcoa.scp_time is not None:
                print(f"  SCP COA time:   {scpcoa.scp_time:.6f} s")
            else:
                print("  SCP COA time:   ?")
            if scpcoa.slant_range is not None:
                print(f"  Slant range:    {scpcoa.slant_range:,.1f} m")
            if scpcoa.ground_range is not None:
                print(f"  Ground range:   {scpcoa.ground_range:,.1f} m")
            if scpcoa.graze_ang is not None:
                print(f"  Graze angle:    {scpcoa.graze_ang:.4f} deg")
            if scpcoa.incidence_ang is not None:
                print(f"  Incidence angle:{scpcoa.incidence_ang:.4f} deg")
            if scpcoa.doppler_cone_ang is not None:
                print(f"  Doppler cone:   {scpcoa.doppler_cone_ang:.4f} deg")
            if scpcoa.twist_ang is not None:
                print(f"  Twist angle:    {scpcoa.twist_ang:.4f} deg")
            if scpcoa.slope_ang is not None:
                print(f"  Slope angle:    {scpcoa.slope_ang:.4f} deg")
            if scpcoa.azim_ang is not None:
                print(f"  Azimuth angle:  {scpcoa.azim_ang:.4f} deg")
            if scpcoa.layover_ang is not None:
                print(f"  Layover angle:  {scpcoa.layover_ang:.4f} deg")
            if scpcoa.arp_pos is not None:
                p = scpcoa.arp_pos
                print(f"  ARP position:   ({p.x:,.1f}, {p.y:,.1f}, {p.z:,.1f}) m ECF")
            if scpcoa.arp_vel is not None:
                v = scpcoa.arp_vel
                print(f"  ARP velocity:   ({v.x:.1f}, {v.y:.1f}, {v.z:.1f}) m/s")
            if scpcoa.arp_acc is not None:
                a = scpcoa.arp_acc
                print(f"  ARP accel:      ({a.x:.4f}, {a.y:.4f}, {a.z:.4f}) m/s^2")

        print()

        # Build title from filename and collector
        title_parts = [filepath.name]
        if ci is not None and ci.collector_name:
            title_parts.append(ci.collector_name)
        title = "  |  ".join(title_parts)

        geo = create_geolocation(reader)
        show(reader, geolocation=geo, title=title, block=True)


if __name__ == "__main__":
    args = parse_args()
    view_sicd(args.filepath)
