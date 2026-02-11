# -*- coding: utf-8 -*-
"""
BIOMASS Product Viewer Example.

Loads a downloaded BIOMASS L1A SCS product and displays:
  1. HH magnitude (dB) with geolocation and calibration markers
  2. Pauli decomposition RGB (R=double-bounce, G=volume, B=surface)

Uses the PauliDecomposition class for a true quad-pol decomposition with
proper 1/sqrt(2) normalization and all four channels (HH, HV, VH, VV).

Markers are interactive -- click any marker to open its location in
Google Maps.  Uses the QtAgg backend for interactive windowed display.

This example uses the most recently downloaded product in the data
directory, or a specific product path passed as a command-line argument.

Dependencies
------------
matplotlib
rasterio

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
2026-01-30

Modified
--------
2026-01-30
"""

# Standard library
import json
import sys
import webbrowser
from pathlib import Path

# Third-party
import numpy as np

# Matplotlib -- set backend before importing pyplot
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt  # noqa: E402

# GRDL
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from grdl.IO import BIOMASSL1Reader
from grdl.geolocation.sar.gcp import GCPGeolocation
from grdl.image_processing import PauliDecomposition

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("/Volumes/PRO-G40/SAR_DATA/BIOMASS")
CALVAL_GEOJSON = Path(__file__).parent.parent.parent / "ground_truth" / "biomass_calibration_targets.geojson"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _google_maps_url(lat: float, lon: float) -> str:
    """Return a Google Maps URL centered on lat/lon."""
    return f"https://www.google.com/maps?q={lat},{lon}&t=k"


def load_calibration_targets(
    geojson_path: Path = CALVAL_GEOJSON,
) -> list:
    """Load calibration target locations from GeoJSON.

    Parameters
    ----------
    geojson_path : Path
        Path to the calibration_targets.geojson file.

    Returns
    -------
    list
        List of dicts with keys: name, lat, lon, type, description.
    """
    if not geojson_path.exists():
        return []

    with open(geojson_path, "r") as f:
        data = json.load(f)

    targets = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates", [])
        if len(coords) >= 2:
            targets.append({
                "name": props.get("name", "Unknown"),
                "lon": coords[0],
                "lat": coords[1],
                "type": props.get("type", ""),
                "description": props.get("description", ""),
            })
    return targets


def find_latest_product(data_dir: Path) -> Path:
    """Find the most recently modified BIOMASS product directory.

    Parameters
    ----------
    data_dir : Path
        Root directory to scan for BIOMASS products.

    Returns
    -------
    Path
        Path to the product directory (containing annotation/, measurement/).
    """
    candidates = []
    for d in data_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("BIO_S"):
            continue
        # Check for nested product directory (ZIP extraction creates a wrapper)
        inner = d / d.name
        if inner.is_dir() and (inner / "annotation").is_dir():
            candidates.append(inner)
        elif (d / "annotation").is_dir():
            candidates.append(d)

    if not candidates:
        raise FileNotFoundError(
            f"No BIOMASS products found in {data_dir}. "
            f"Run discover_and_download.py first."
        )

    # Most recently modified
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _to_db(arr: np.ndarray) -> np.ndarray:
    """Convert complex SAR array to magnitude in dB."""
    return 20.0 * np.log10(np.abs(arr) + 1e-10)


# ---------------------------------------------------------------------------
# View product
# ---------------------------------------------------------------------------

def view_product(product_path: Path) -> None:
    """Load and display a BIOMASS L1A product.

    Shows two panels:
      Left  -- HH magnitude (dB) with geolocation and cal/val markers
      Right -- Pauli decomposition RGB

    Parameters
    ----------
    product_path : Path
        Path to the BIOMASS product directory.
    """
    print(f"Loading: {product_path.name}")
    reader = BIOMASSL1Reader(product_path)
    geo_info = {
        'gcps': reader.metadata['gcps'],
        'crs': reader.metadata.get('crs', 'WGS84'),
    }
    geo = GCPGeolocation.from_dict(geo_info, reader.metadata)
    rows, cols = geo.shape
    pols = reader.metadata["polarizations"]

    print(f"  Size:  {rows} x {cols}")
    print(f"  Pols:  {pols}")
    print(f"  Orbit: {reader.metadata.get('orbit_number', '?')}")
    print(f"  Date:  {reader.metadata.get('start_time', '?')}")
    print()

    # ------------------------------------------------------------------
    # Read polarization channels
    # ------------------------------------------------------------------
    # Band indices: 0=HH, 1=HV, 2=VH, 3=VV
    print(f"Reading all four polarizations ({rows} x {cols})...")
    shh = reader.read_chip(0, rows, 0, cols, bands=[0])
    shv = reader.read_chip(0, rows, 0, cols, bands=[1])
    svh = reader.read_chip(0, rows, 0, cols, bands=[2])
    svv = reader.read_chip(0, rows, 0, cols, bands=[3])

    # HH magnitude in dB
    hh_db = _to_db(shh)
    vmin = np.nanpercentile(hh_db, 2)
    vmax = np.nanpercentile(hh_db, 98)

    # ------------------------------------------------------------------
    # True quad-pol Pauli decomposition (complex domain)
    # Uses all four S-matrix elements with 1/sqrt(2) normalization.
    # Phase relationships between channels drive the separation of
    # surface vs. double-bounce scattering.
    # ------------------------------------------------------------------
    print("Computing Pauli decomposition...")
    pauli = PauliDecomposition()
    pauli_rgb = pauli.to_rgb(pauli.decompose(shh, shv, svh, svv))

    del shh, shv, svh, svv

    # ------------------------------------------------------------------
    # Compute marker locations
    # ------------------------------------------------------------------
    # Center
    center_r, center_c = rows // 2, cols // 2
    center_lat, center_lon, _ = geo.image_to_latlon(center_r, center_c)

    # Corners (5% inset)
    mr = int(rows * 0.05)
    mc = int(cols * 0.05)
    corner_names = ["TL", "TR", "BL", "BR"]
    corner_rc = [
        (mr, mc),
        (mr, cols - 1 - mc),
        (rows - 1 - mr, mc),
        (rows - 1 - mr, cols - 1 - mc),
    ]
    corner_latlons = []
    for r, c in corner_rc:
        lat, lon, _ = geo.image_to_latlon(r, c)
        corner_latlons.append((lat, lon))

    # Calibration targets
    cal_targets = load_calibration_targets()
    cal_in_image = []
    for t in cal_targets:
        try:
            r, c = geo.latlon_to_image(t["lat"], t["lon"])
            if 0 <= r < rows and 0 <= c < cols:
                cal_in_image.append({**t, "row": r, "col": c})
        except Exception:
            pass

    if cal_in_image:
        print(f"Calibration targets in image: {len(cal_in_image)}")
        for t in cal_in_image:
            print(f"  {t['name']}  ({t['lat']:.4f}, {t['lon']:.4f})"
                  f"  -> pixel ({t['row']:.0f}, {t['col']:.0f})")
        print()

    # ------------------------------------------------------------------
    # Plot -- two panels
    # ------------------------------------------------------------------
    marker_registry = {}

    aspect = rows / cols
    fig_w = 16
    panel_h = min(fig_w * 0.5 * aspect * 0.6, 28)
    fig, (ax_hh, ax_pauli) = plt.subplots(
        1, 2, figsize=(fig_w, panel_h), sharex=True, sharey=True,
    )

    # --- Left panel: HH dB with markers ---
    ax_hh.imshow(hh_db, cmap="gray", vmin=vmin, vmax=vmax,
                 aspect="auto", interpolation="nearest")

    # Calibration targets (orange hexagons)
    if cal_in_image:
        sc_cal = ax_hh.scatter(
            [t["col"] for t in cal_in_image],
            [t["row"] for t in cal_in_image],
            marker="h", c="orange", s=150, edgecolors="black",
            linewidths=1.0, zorder=7, label="Cal/Val Targets",
            picker=True, pickradius=5,
        )
        marker_registry[sc_cal] = [
            {"lat": t["lat"], "lon": t["lon"], "label": t["name"]}
            for t in cal_in_image
        ]
        for t in cal_in_image:
            ax_hh.annotate(
                f"{t['name']}\n{t['lat']:.4f}, {t['lon']:.4f}",
                xy=(t["col"], t["row"]), xytext=(10, -5),
                textcoords="offset points", fontsize=7,
                color="orange", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7),
            )

    # Corners (yellow diamonds)
    sc_corners = ax_hh.scatter(
        [rc[1] for rc in corner_rc], [rc[0] for rc in corner_rc],
        marker="D", c="yellow", s=100, edgecolors="black", linewidths=0.8,
        zorder=5, label="Corners (5% inset)", picker=True, pickradius=5,
    )
    marker_registry[sc_corners] = [
        {"lat": ll[0], "lon": ll[1], "label": f"Corner {name}"}
        for name, ll in zip(corner_names, corner_latlons)
    ]
    for name, (r, c), (lat, lon) in zip(corner_names, corner_rc, corner_latlons):
        ax_hh.annotate(
            f"{name}\n{lat:.4f}, {lon:.4f}",
            xy=(c, r), xytext=(10, -5),
            textcoords="offset points", fontsize=7,
            color="yellow", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
        )

    # Center (red star)
    sc_center = ax_hh.scatter(
        [center_c], [center_r], marker="*", c="red", s=200,
        edgecolors="black", linewidths=0.5, zorder=6, label="Center",
        picker=True, pickradius=5,
    )
    marker_registry[sc_center] = [
        {"lat": center_lat, "lon": center_lon, "label": "Center"}
    ]
    ax_hh.annotate(
        f"Center\n{center_lat:.4f}, {center_lon:.4f}",
        xy=(center_c, center_r), xytext=(12, 0),
        textcoords="offset points", fontsize=7,
        color="red", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
    )

    ax_hh.set_title(f"HH Magnitude (dB)\n{product_path.name}", fontsize=9)
    ax_hh.set_xlabel("Column (range)")
    ax_hh.set_ylabel("Row (azimuth)")
    ax_hh.legend(loc="upper right", fontsize=7, framealpha=0.8)

    # --- Right panel: Pauli RGB ---
    ax_pauli.imshow(pauli_rgb, aspect="auto", interpolation="nearest")

    # Replicate cal-target markers on Pauli panel
    if cal_in_image:
        sc_cal_p = ax_pauli.scatter(
            [t["col"] for t in cal_in_image],
            [t["row"] for t in cal_in_image],
            marker="h", c="orange", s=150, edgecolors="white",
            linewidths=1.0, zorder=7, picker=True, pickradius=5,
        )
        marker_registry[sc_cal_p] = [
            {"lat": t["lat"], "lon": t["lon"], "label": t["name"]}
            for t in cal_in_image
        ]
        for t in cal_in_image:
            ax_pauli.annotate(
                t["name"],
                xy=(t["col"], t["row"]), xytext=(10, -5),
                textcoords="offset points", fontsize=7,
                color="orange", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7),
            )

    ax_pauli.set_title(
        "Pauli Decomposition (quad-pol)\n"
        "R=double-bounce  G=volume  B=surface",
        fontsize=9,
    )
    ax_pauli.set_xlabel("Column (range)")

    # --- Interactive click handler ---
    def on_pick(event):
        artist = event.artist
        if artist not in marker_registry:
            return
        ind = event.ind[0]
        data = marker_registry[artist][ind]
        url = _google_maps_url(data["lat"], data["lon"])
        print(f"  {data['label']}: ({data['lat']:.6f}, {data['lon']:.6f})")
        print(f"  {url}")
        webbrowser.open(url)

    fig.canvas.mpl_connect("pick_event", on_pick)

    fig.suptitle(
        f"BIOMASS L1A SCS  |  Orbit {reader.metadata.get('orbit_number', '?')}"
        f"  |  {reader.metadata.get('start_time', '?')[:10]}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    print("Click any marker to open its location in Google Maps.\n")
    plt.show()

    reader.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("BIOMASS Product Viewer")
        print()
        print("Usage:")
        print("  python view_product.py              # View most recent product")
        print("  python view_product.py <path>        # View specific product")
        print()
        print(f"Data directory: {DATA_DIR}")
    elif len(sys.argv) > 1:
        view_product(Path(sys.argv[1]))
    else:
        product = find_latest_product(DATA_DIR)
        view_product(product)
