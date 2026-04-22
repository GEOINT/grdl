# -*- coding: utf-8 -*-
"""
RapidEye NITF + RPC Geolocation Experiment.

Reads RapidEye Basic Analytic NITF imagery (separate per-band NTF files),
extracts metadata, parses RPC coefficients from both the embedded TRE and
the sidecar XML, and validates RPC geolocation via round-trip and footprint
checks.

RapidEye L1B "Basic Analytic" delivers each spectral band as a separate
NITF file (band1..band5) plus a sidecar _rpc.xml with the RPC00B model.
The individual NTF files may or may not embed RPC TREs -- this experiment
tests both paths.

Band assignments (5-band MSI, 6.5 m GSD pushbroom):
    1: Blue     (440 - 510 nm)
    2: Green    (520 - 590 nm)
    3: Red      (630 - 685 nm)
    4: Red Edge (690 - 730 nm)
    5: NIR      (760 - 850 nm)

Dependencies
------------
rasterio
matplotlib
scipy (for bicubic DEM interpolation)

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
2026-03-21

Modified
--------
2026-03-21
"""

# Standard library
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# Third-party
import numpy as np
import matplotlib.pyplot as plt


# GRDL
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from grdl.IO.eo.nitf import EONITFReader
from grdl.IO.models.eo_nitf import RPCCoefficients
from grdl.geolocation.eo.rpc import RPCGeolocation
from grdl.geolocation.elevation.tiled_geotiff_dem import TiledGeoTIFFDEM

# =====================================================================
# Configuration
# =====================================================================

DATA_DIR = Path.home() / "Data" / "MSI_Data" / "rapideye" / \
    "REScene_Basic_Analytic_nitf" / "2019-10-05T100157_RE4" / \
    "basic_analytic_nitf"

BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR"]

# FABDEM tiles covering the scene (lat ~43-44, lon ~4-5, southern France)
DEM_DIR = Path("/Volumes/PRO-G40/terrain/FABDEM/"
               "N40E000-N50E010_FABDEM_V1-2")

# Known footprint corners from metadata (lat, lon) for validation
# From the _metadata.xml geographicLocation block
KNOWN_CORNERS = {
    "topLeft":     (44.17616, 4.61565),
    "topRight":    (44.05639, 5.54855),
    "bottomRight": (43.38042, 5.33166),
    "bottomLeft":  (43.49903, 4.40902),
}

# Expected image dimensions from metadata XML
EXPECTED_ROWS = 5000
EXPECTED_COLS = 5000


def parse_rpc_xml(xml_path: Path) -> RPCCoefficients:
    """Parse RapidEye sidecar RPC XML into an RPCCoefficients object.

    Parameters
    ----------
    xml_path : Path
        Path to the *_rpc.xml file.

    Returns
    -------
    RPCCoefficients
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Handle namespace
    ns = {"re": "http://schemas.planet.com/products/productMetadataRpc"}

    def get_float(tag: str) -> float:
        el = root.find(f"re:{tag}", ns)
        return float(el.text.strip())

    def get_coefs(tag: str) -> np.ndarray:
        el = root.find(f"re:{tag}", ns)
        return np.array([float(x) for x in el.text.strip().split()])

    return RPCCoefficients(
        line_off=get_float("lineOff"),
        samp_off=get_float("sampleOff"),
        lat_off=get_float("latOff"),
        long_off=get_float("longOff"),
        height_off=get_float("heightOff"),
        line_scale=get_float("lineScale"),
        samp_scale=get_float("sampleScale"),
        lat_scale=get_float("latScale"),
        long_scale=get_float("longScale"),
        height_scale=get_float("heightScale"),
        line_num_coef=get_coefs("lineNumCoeff"),
        line_den_coef=get_coefs("lineDenCoeff"),
        samp_num_coef=get_coefs("sampleNumCoeff"),
        samp_den_coef=get_coefs("sampleDenCoeff"),
    )


def main() -> None:
    """Run the RapidEye NITF + RPC experiment."""
    print("=" * 70)
    print("RapidEye NITF + RPC Geolocation Experiment")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Discover band files
    # ------------------------------------------------------------------
    band_files = sorted(DATA_DIR.glob("*_band*.ntf"))
    print(f"\nData directory: {DATA_DIR}")
    print(f"Band files found: {len(band_files)}")
    for bf in band_files:
        print(f"  {bf.name}")

    if not band_files:
        print("ERROR: No band NTF files found.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Read each band with EONITFReader, inspect metadata
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SECTION 1: Reading individual band NITF files")
    print("-" * 70)

    readers = []
    for i, bf in enumerate(band_files):
        reader = EONITFReader(bf)
        readers.append(reader)
        meta = reader.metadata
        band_label = BAND_NAMES[i] if i < len(BAND_NAMES) else f"Band{i+1}"

        print(f"\n  Band {i+1} ({band_label}): {bf.name}")
        print(f"    Shape:  {meta.rows} x {meta.cols}")
        print(f"    Bands:  {meta.bands}")
        print(f"    Dtype:  {meta.dtype}")
        print(f"    CRS:    {meta.crs}")
        print(f"    IID1:   {meta.iid1}")
        print(f"    ICORDS: {meta.icords}")
        print(f"    ICAT:   {meta.icat}")
        print(f"    ABPP:   {meta.abpp}")
        print(f"    Has RPC (embedded): {reader.has_rpc}")
        print(f"    Has RSM (embedded): {reader.has_rsm}")

        # Verify expected dimensions
        assert meta.rows > 0, f"Band {i+1} has 0 rows"
        assert meta.cols > 0, f"Band {i+1} has 0 cols"

    # ------------------------------------------------------------------
    # 3. Read a small chip from each band, verify pixel statistics
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SECTION 2: Pixel data verification (center 256x256 chip)")
    print("-" * 70)

    r0 = readers[0].metadata.rows
    c0 = readers[0].metadata.cols
    chip_size = min(256, r0, c0)
    row_start = (r0 - chip_size) // 2
    col_start = (c0 - chip_size) // 2

    print(f"\n  Chip region: rows [{row_start}:{row_start+chip_size}], "
          f"cols [{col_start}:{col_start+chip_size}]")

    for i, reader in enumerate(readers):
        chip = reader.read_chip(
            row_start, row_start + chip_size,
            col_start, col_start + chip_size,
        )
        band_label = BAND_NAMES[i] if i < len(BAND_NAMES) else f"Band{i+1}"
        print(f"\n  Band {i+1} ({band_label}):")
        print(f"    Chip shape: {chip.shape}")
        print(f"    Dtype:      {chip.dtype}")
        print(f"    Min:        {chip.min()}")
        print(f"    Max:        {chip.max()}")
        print(f"    Mean:       {chip.mean():.2f}")
        print(f"    Std:        {chip.std():.2f}")
        print(f"    Non-zero:   {np.count_nonzero(chip)} / {chip.size} "
              f"({100*np.count_nonzero(chip)/chip.size:.1f}%)")

    # ------------------------------------------------------------------
    # 4. RPC extraction: embedded vs sidecar XML
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SECTION 3: RPC coefficient extraction")
    print("-" * 70)

    # Try embedded RPC from the first band
    embedded_rpc = readers[0].metadata.rpc

    # Parse sidecar XML
    rpc_xml_path = DATA_DIR / "2019-10-05T100157_RE4_1B_rpc.xml"
    print(f"\n  Sidecar RPC XML: {rpc_xml_path.name}")
    xml_rpc = parse_rpc_xml(rpc_xml_path)

    print(f"\n  RPC from sidecar XML:")
    print(f"    Line offset:   {xml_rpc.line_off}")
    print(f"    Sample offset: {xml_rpc.samp_off}")
    print(f"    Lat offset:    {xml_rpc.lat_off}°")
    print(f"    Lon offset:    {xml_rpc.long_off}°")
    print(f"    Height offset: {xml_rpc.height_off} m")
    print(f"    Line scale:    {xml_rpc.line_scale}")
    print(f"    Sample scale:  {xml_rpc.samp_scale}")
    print(f"    Lat scale:     {xml_rpc.lat_scale}°")
    print(f"    Lon scale:     {xml_rpc.long_scale}°")
    print(f"    Height scale:  {xml_rpc.height_scale} m")

    # Compare if embedded exists
    if embedded_rpc is not None:
        print(f"\n  Embedded RPC also found -- comparing:")
        lat_diff = abs(embedded_rpc.lat_off - xml_rpc.lat_off)
        lon_diff = abs(embedded_rpc.long_off - xml_rpc.long_off)
        line_diff = abs(embedded_rpc.line_off - xml_rpc.line_off)
        samp_diff = abs(embedded_rpc.samp_off - xml_rpc.samp_off)
        coef_diff = np.max(np.abs(
            embedded_rpc.line_num_coef - xml_rpc.line_num_coef))
        print(f"    Lat offset diff:        {lat_diff:.6e}")
        print(f"    Lon offset diff:        {lon_diff:.6e}")
        print(f"    Line offset diff:       {line_diff:.6e}")
        print(f"    Sample offset diff:     {samp_diff:.6e}")
        print(f"    Max line_num_coef diff: {coef_diff:.6e}")
        match = (lat_diff < 1e-6 and lon_diff < 1e-6 and
                 line_diff < 1e-3 and samp_diff < 1e-3)
        print(f"    Match: {'YES' if match else 'NO (using XML RPC)'}")
        rpc = embedded_rpc if match else xml_rpc
    else:
        print(f"\n  No embedded RPC in NTF -- using sidecar XML RPC.")
        rpc = xml_rpc

    # Select which reader to base geometry on (use band 1)
    ref_reader = readers[0]
    shape = ref_reader.get_shape()

    # ------------------------------------------------------------------
    # 5. Load FABDEM terrain model
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SECTION 4: FABDEM DEM loading (TiledGeoTIFFDEM)")
    print("-" * 70)

    dem = TiledGeoTIFFDEM(str(DEM_DIR), interpolation=3)
    print(f"\n  DEM directory: {DEM_DIR}")
    print(f"  Tiles indexed: {dem.tile_count}")
    print(f"  Coverage: {dem.coverage_bounds}")
    print(f"  Interpolation: bicubic (order=3)")

    # Sample DEM at scene center
    scene_center_lat = 43.78
    scene_center_lon = 4.98
    center_elev = dem.get_elevation(scene_center_lat, scene_center_lon)
    print(f"  Scene center elevation: {center_elev:.1f} m")

    # ------------------------------------------------------------------
    # 6. Create RPCGeolocation -- with and without DEM
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SECTION 5: RPC Geolocation -- h=0 vs DEM-corrected")
    print("-" * 70)

    geo_flat = RPCGeolocation(rpc=rpc, shape=shape)
    geo_dem = RPCGeolocation(rpc=rpc, shape=shape)
    geo_dem.elevation = dem  # attach DEM for iterative refinement

    # Compare corners: h=0 vs DEM-corrected
    corners = {
        "Top-Left":     (0, 0),
        "Top-Right":    (0, shape[1] - 1),
        "Center":       (shape[0] // 2, shape[1] // 2),
        "Bottom-Right": (shape[0] - 1, shape[1] - 1),
        "Bottom-Left":  (shape[0] - 1, 0),
    }

    print(f"\n  {'Point':>14s}  {'Flat Lat':>10s}  {'Flat Lon':>10s}  "
          f"{'DEM Lat':>10s}  {'DEM Lon':>10s}  {'DEM h(m)':>10s}  "
          f"{'dLat(deg)':>10s}  {'dLon(deg)':>10s}")
    print(f"  {'-'*14:>14s}  {'-'*10:>10s}  {'-'*10:>10s}  "
          f"{'-'*10:>10s}  {'-'*10:>10s}  {'-'*10:>10s}  "
          f"{'-'*10:>10s}  {'-'*10:>10s}")

    for name, (r, c) in corners.items():
        flat_lat, flat_lon, _ = geo_flat.image_to_latlon(r, c)
        dem_lat, dem_lon, dem_h = geo_dem.image_to_latlon(r, c)
        dlat = dem_lat - flat_lat
        dlon = dem_lon - flat_lon
        print(f"  {name:>14s}  {flat_lat:10.6f}  {flat_lon:10.6f}  "
              f"{dem_lat:10.6f}  {dem_lon:10.6f}  {dem_h:10.1f}  "
              f"{dlat:10.6f}  {dlon:10.6f}")

    # Use DEM-corrected geolocation going forward
    geo = geo_dem

    # ------------------------------------------------------------------
    # 7. Round-trip validation with DEM
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SECTION 6: Round-trip validation (DEM-corrected)")
    print("-" * 70)

    test_rows = np.array([0, 0, shape[0]//2, shape[0]-1, shape[0]-1],
                         dtype=np.float64)
    test_cols = np.array([0, shape[1]-1, shape[1]//2, 0, shape[1]-1],
                         dtype=np.float64)

    lats, lons, heights = geo.image_to_latlon(test_rows, test_cols)
    rt_rows, rt_cols = geo.latlon_to_image(lats, lons, heights)

    print(f"\n  {'Point':>6s}  {'Row':>8s}  {'Col':>8s}  "
          f"{'RT Row':>10s}  {'RT Col':>10s}  "
          f"{'Row Err':>10s}  {'Col Err':>10s}  {'DEM h':>8s}")
    print(f"  {'-'*6:>6s}  {'-'*8:>8s}  {'-'*8:>8s}  "
          f"{'-'*10:>10s}  {'-'*10:>10s}  "
          f"{'-'*10:>10s}  {'-'*10:>10s}  {'-'*8:>8s}")

    for i in range(len(test_rows)):
        r_err = abs(rt_rows[i] - test_rows[i])
        c_err = abs(rt_cols[i] - test_cols[i])
        print(f"  {i+1:6d}  {test_rows[i]:8.1f}  {test_cols[i]:8.1f}  "
              f"{rt_rows[i]:10.4f}  {rt_cols[i]:10.4f}  "
              f"{r_err:10.6f}  {c_err:10.6f}  {heights[i]:8.1f}")

    max_row_err = np.max(np.abs(rt_rows - test_rows))
    max_col_err = np.max(np.abs(rt_cols - test_cols))
    print(f"\n  Max round-trip error: row={max_row_err:.6f} px, "
          f"col={max_col_err:.6f} px")

    if max_row_err < 0.1 and max_col_err < 0.1:
        print("  PASS: Round-trip error < 0.1 pixels")
    else:
        print("  WARNING: Round-trip error exceeds 0.1 pixels")

    # ------------------------------------------------------------------
    # 8. Footprint validation against metadata corners
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SECTION 7: Footprint validation vs metadata corners")
    print("-" * 70)

    corner_map = {
        "topLeft":     (0, 0),
        "topRight":    (0, shape[1] - 1),
        "bottomRight": (shape[0] - 1, shape[1] - 1),
        "bottomLeft":  (shape[0] - 1, 0),
    }

    print(f"\n  {'Corner':>14s}  {'Meta Lat':>10s}  {'Meta Lon':>10s}  "
          f"{'DEM Lat':>10s}  {'DEM Lon':>10s}  {'dLat':>10s}  {'dLon':>10s}")
    print(f"  {'-'*14:>14s}  {'-'*10:>10s}  {'-'*10:>10s}  "
          f"{'-'*10:>10s}  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*10:>10s}")

    for cname, (r, c) in corner_map.items():
        meta_lat, meta_lon = KNOWN_CORNERS[cname]
        rpc_lat, rpc_lon, _ = geo.image_to_latlon(float(r), float(c))
        dlat = abs(rpc_lat - meta_lat)
        dlon = abs(rpc_lon - meta_lon)
        print(f"  {cname:>14s}  {meta_lat:10.5f}  {meta_lon:10.5f}  "
              f"{rpc_lat:10.5f}  {rpc_lon:10.5f}  {dlat:10.6f}  {dlon:10.6f}")

    # ------------------------------------------------------------------
    # 9. Vectorized performance: batch of random points
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SECTION 8: Vectorized batch geolocation (DEM-corrected)")
    print("-" * 70)

    rng = np.random.default_rng(42)
    n_pts = 10000
    rand_rows = rng.uniform(0, shape[0] - 1, n_pts)
    rand_cols = rng.uniform(0, shape[1] - 1, n_pts)

    t0 = time.perf_counter()
    batch_lats, batch_lons, batch_heights = geo.image_to_latlon(
        rand_rows, rand_cols)
    t_fwd = time.perf_counter() - t0

    t0 = time.perf_counter()
    batch_rt_rows, batch_rt_cols = geo.latlon_to_image(
        batch_lats, batch_lons, batch_heights)
    t_inv = time.perf_counter() - t0

    batch_row_err = np.max(np.abs(batch_rt_rows - rand_rows))
    batch_col_err = np.max(np.abs(batch_rt_cols - rand_cols))

    print(f"\n  {n_pts:,} random points (with DEM lookup per point):")
    print(f"    Forward  (image->latlon+DEM): {t_fwd*1000:.1f} ms "
          f"({n_pts/t_fwd:.0f} pts/sec)")
    print(f"    Inverse  (latlon->image):     {t_inv*1000:.1f} ms "
          f"({n_pts/t_inv:.0f} pts/sec)")
    print(f"    Max round-trip error:         row={batch_row_err:.6f} px, "
          f"col={batch_col_err:.6f} px")
    print(f"    Elevation range: [{batch_heights.min():.1f}, "
          f"{batch_heights.max():.1f}] m")
    print(f"    Lat range: [{batch_lats.min():.5f}, {batch_lats.max():.5f}]")
    print(f"    Lon range: [{batch_lons.min():.5f}, {batch_lons.max():.5f}]")

    # ------------------------------------------------------------------
    # 10. Visualization
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SECTION 9: Visualization")
    print("-" * 70)

    # Re-read full-resolution bands for display (downsample for speed)
    ds_factor = 4
    nrows, ncols = shape

    band_data = []
    for reader in readers:
        full = reader.read_chip(0, nrows, 0, ncols)
        band_data.append(full[::ds_factor, ::ds_factor].astype(np.float32))

    ds_rows, ds_cols = band_data[0].shape

    def stretch_band(b: np.ndarray) -> np.ndarray:
        """Percentile stretch a single band to [0, 1]."""
        mask = b > 0
        if not mask.any():
            return np.zeros_like(b)
        p2, p98 = np.percentile(b[mask], [2, 98])
        out = (b - p2) / (p98 - p2 + 1e-10)
        return np.clip(out, 0, 1)

    # Build composites
    tc_rgb = np.stack([stretch_band(band_data[2]),
                       stretch_band(band_data[1]),
                       stretch_band(band_data[0])], axis=-1)
    fc_rgb = np.stack([stretch_band(band_data[4]),
                       stretch_band(band_data[2]),
                       stretch_band(band_data[1])], axis=-1)
    re_rgb = np.stack([stretch_band(band_data[4]),
                       stretch_band(band_data[3]),
                       stretch_band(band_data[2])], axis=-1)

    # Compute geographic extent via DEM-corrected RPC
    corner_lats = []
    corner_lons = []
    corner_elevs = []
    for r, c in [(0, 0), (0, ncols-1), (nrows-1, ncols-1), (nrows-1, 0)]:
        clat, clon, ch = geo.image_to_latlon(float(r), float(c))
        corner_lats.append(clat)
        corner_lons.append(clon)
        corner_elevs.append(ch)
    lon_min, lon_max = min(corner_lons), max(corner_lons)
    lat_min, lat_max = min(corner_lats), max(corner_lats)
    extent = [lon_min, lon_max, lat_min, lat_max]

    # Pre-compute labeled marker positions (3x3 grid across image)
    marker_rows_px = np.array([0, 0, 0,
                               nrows//2, nrows//2, nrows//2,
                               nrows-1, nrows-1, nrows-1], dtype=np.float64)
    marker_cols_px = np.array([0, ncols//2, ncols-1,
                               0, ncols//2, ncols-1,
                               0, ncols//2, ncols-1], dtype=np.float64)
    m_lats, m_lons, m_heights = geo.image_to_latlon(
        marker_rows_px, marker_cols_px)

    def add_latlon_markers(ax, in_geo_coords: bool = False) -> None:
        """Add lat/lon/elev markers to an axes.

        Parameters
        ----------
        ax : matplotlib Axes
        in_geo_coords : bool
            If True, plot in lon/lat space. If False, plot in
            downsampled pixel space.
        """
        for k in range(len(marker_rows_px)):
            if in_geo_coords:
                mx, my = m_lons[k], m_lats[k]
            else:
                mx = marker_cols_px[k] / ds_factor
                my = marker_rows_px[k] / ds_factor
            ax.plot(mx, my, 'c.', markersize=5)
            label = (f"{m_lats[k]:.3f}, {m_lons[k]:.3f}"
                     f"\n{m_heights[k]:.0f}m")
            ax.annotate(
                label, (mx, my),
                textcoords="offset points", xytext=(5, 5),
                fontsize=6, color="cyan", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="black",
                          alpha=0.6, ec="none"),
            )

    # --- Figure 1: Individual bands with lat/lon markers ---
    fig1, axes1 = plt.subplots(1, 5, figsize=(22, 4))
    fig1.suptitle("RapidEye 5-Band Imagery (Individual Bands)", fontsize=14)

    band_cmaps = ["Blues", "Greens", "Reds", "RdPu", "YlGn"]
    for i, (ax, bd) in enumerate(zip(axes1, band_data)):
        p2, p98 = np.percentile(bd[bd > 0], [2, 98])
        ax.imshow(bd, cmap=band_cmaps[i], vmin=p2, vmax=p98)
        ax.set_title(f"Band {i+1}: {BAND_NAMES[i]}", fontsize=10)
        ax.axis("off")
    # Markers on center band only (avoid clutter)
    add_latlon_markers(axes1[2])
    fig1.tight_layout()

    # --- Figure 2: Composites with lat/lon markers ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle("RapidEye Band Composites", fontsize=14)

    for ax, img, title in zip(
        axes2,
        [tc_rgb, fc_rgb, re_rgb],
        ["True Color (R-G-B)", "False Color CIR (NIR-R-G)",
         "Red Edge (NIR-RE-R)"],
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        add_latlon_markers(ax)

    fig2.tight_layout()

    # --- Figure 3: Geolocation grid overlay with DEM heights ---
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))
    ax3.imshow(tc_rgb, extent=extent, aspect="auto", origin="upper")
    ax3.set_xlabel("Longitude (deg)")
    ax3.set_ylabel("Latitude (deg)")
    ax3.set_title("RapidEye True Color -- DEM-Corrected RPC Grid",
                  fontsize=13)

    # Geolocation grid
    grid_step = 500
    for gr in np.arange(0, nrows, grid_step, dtype=np.float64):
        cols_line = np.linspace(0, ncols - 1, 100)
        rows_line = np.full_like(cols_line, gr)
        glats, glons, _ = geo.image_to_latlon(rows_line, cols_line)
        ax3.plot(glons, glats, 'w-', linewidth=0.5, alpha=0.5)
    for gc in np.arange(0, ncols, grid_step, dtype=np.float64):
        rows_line = np.linspace(0, nrows - 1, 100)
        cols_line = np.full_like(rows_line, gc)
        glats, glons, _ = geo.image_to_latlon(rows_line, cols_line)
        ax3.plot(glons, glats, 'w-', linewidth=0.5, alpha=0.5)

    # Metadata footprint
    meta_lats = [v[0] for v in KNOWN_CORNERS.values()]
    meta_lons = [v[1] for v in KNOWN_CORNERS.values()]
    meta_lats.append(meta_lats[0])
    meta_lons.append(meta_lons[0])
    ax3.plot(meta_lons, meta_lats, 'r--', linewidth=1.5,
             label="Metadata footprint")
    ax3.plot([v[1] for v in KNOWN_CORNERS.values()],
             [v[0] for v in KNOWN_CORNERS.values()],
             'ro', markersize=6)

    # Lat/lon markers with elevation
    add_latlon_markers(ax3, in_geo_coords=True)

    # Center marker
    center_lat, center_lon, _ = geo.image_to_latlon(
        float(nrows // 2), float(ncols // 2))
    ax3.plot(center_lon, center_lat, 'y+', markersize=15,
             markeredgewidth=2, label="Image center")
    ax3.legend(loc="lower right")
    fig3.tight_layout()

    # --- Figure 4: DEM elevation map under the scene ---
    print("  Generating DEM elevation map...")
    dem_step = 8  # pixels between DEM samples (in ds image space)
    dem_r = np.arange(0, ds_rows, dem_step, dtype=np.float64) * ds_factor
    dem_c = np.arange(0, ds_cols, dem_step, dtype=np.float64) * ds_factor
    dem_cc, dem_rr = np.meshgrid(dem_c, dem_r)
    dem_rr_flat = dem_rr.ravel()
    dem_cc_flat = dem_cc.ravel()
    dem_lats, dem_lons, _ = geo_flat.image_to_latlon(
        dem_rr_flat, dem_cc_flat)
    dem_elevs = dem.get_elevation(dem_lats, dem_lons)
    dem_grid = dem_elevs.reshape(dem_rr.shape)

    fig4, axes4 = plt.subplots(1, 2, figsize=(16, 6))
    fig4.suptitle("Terrain Elevation (FABDEM)", fontsize=14)

    im_dem = axes4[0].imshow(
        dem_grid, cmap="terrain",
        extent=extent, aspect="auto", origin="upper")
    axes4[0].set_title("DEM Elevation Under Scene", fontsize=11)
    axes4[0].set_xlabel("Longitude (deg)")
    axes4[0].set_ylabel("Latitude (deg)")
    plt.colorbar(im_dem, ax=axes4[0], label="Elevation (m)",
                 fraction=0.046, pad=0.04)
    add_latlon_markers(axes4[0], in_geo_coords=True)

    # DEM + imagery blend
    # Normalize DEM to [0,1] for blending
    dem_norm = dem_grid.copy()
    valid_dem = ~np.isnan(dem_norm)
    if valid_dem.any():
        dmin, dmax = np.nanmin(dem_norm), np.nanmax(dem_norm)
        dem_norm = (dem_norm - dmin) / (dmax - dmin + 1e-10)
    dem_norm[~valid_dem] = 0

    # Resample DEM to match tc_rgb resolution
    from scipy.ndimage import zoom
    dem_resized = zoom(dem_norm, (
        ds_rows / dem_norm.shape[0],
        ds_cols / dem_norm.shape[1]),
        order=1)
    dem_resized = dem_resized[:ds_rows, :ds_cols]

    # Blend: imagery * 0.6 + terrain tint * 0.4
    from matplotlib.cm import terrain as terrain_cmap
    dem_tint = terrain_cmap(dem_resized)[..., :3]
    blended = tc_rgb * 0.6 + dem_tint * 0.4

    axes4[1].imshow(blended, extent=extent, aspect="auto", origin="upper")
    axes4[1].set_title("True Color + Terrain Blend", fontsize=11)
    axes4[1].set_xlabel("Longitude (deg)")
    axes4[1].set_ylabel("Latitude (deg)")
    add_latlon_markers(axes4[1], in_geo_coords=True)

    fig4.tight_layout()

    # --- Figure 5: NDVI ---
    red = band_data[2].astype(np.float64)
    nir = band_data[4].astype(np.float64)
    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)

    fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))
    fig5.suptitle("RapidEye Derived Products", fontsize=14)

    im_ndvi = axes5[0].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    axes5[0].set_title("NDVI (Band5-Band3)/(Band5+Band3)", fontsize=11)
    axes5[0].axis("off")
    plt.colorbar(im_ndvi, ax=axes5[0], fraction=0.046, pad=0.04)
    add_latlon_markers(axes5[0])

    valid_ndvi = ndvi[band_data[2] > 0].ravel()
    axes5[1].hist(valid_ndvi, bins=200, color="forestgreen", alpha=0.7,
                  edgecolor="none")
    axes5[1].axvline(x=0.3, color="red", linestyle="--", linewidth=1,
                     label="Vegetation threshold (0.3)")
    veg_pct = 100 * np.sum(valid_ndvi > 0.3) / len(valid_ndvi)
    axes5[1].set_title(f"NDVI Distribution ({veg_pct:.1f}% vegetation)",
                       fontsize=11)
    axes5[1].set_xlabel("NDVI")
    axes5[1].set_ylabel("Pixel Count")
    axes5[1].legend()

    fig5.tight_layout()

    plt.show()

    # ------------------------------------------------------------------
    # 11. Cleanup
    # ------------------------------------------------------------------
    for reader in readers:
        reader.close()
    dem.close()

    print("\n" + "=" * 70)
    print("Experiment complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
