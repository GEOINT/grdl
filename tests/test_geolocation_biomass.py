# -*- coding: utf-8 -*-
"""
BIOMASS Geolocation Tests - Coordinate transform tests with Google Maps verification.

Tests GCP-based geolocation for BIOMASS L1 SCS products. Converts pixel
coordinates to lat/lon and generates Google Maps URLs for visual verification.
Also tests round-trip accuracy (pixel -> latlon -> pixel).

Dependencies
------------
pytest
scipy
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

import pytest
from pathlib import Path
import sys

import numpy as np

try:
    import matplotlib
    for _backend in ['QtAgg', 'MacOSX', 'TkAgg', 'Agg']:
        try:
            matplotlib.use(_backend)
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.close(fig)
            break
        except (ImportError, RuntimeError):
            continue
    else:
        import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from grdl.IO import BIOMASSL1Reader
from grdl.geolocation import Geolocation


# Path to test data
TEST_DATA_PATH = Path(
    "/Volumes/PRO-G40/SAR_DATA/BIOMASS/"
    "BIO_S1_SCS__1S_20251121T045325_20251121T045346_T_G01_M01_C01_T003_F290_01_DJUPJI"
)

# Skip all tests if data is not available
pytestmark = pytest.mark.skipif(
    not TEST_DATA_PATH.exists(),
    reason=f"Test data not found at {TEST_DATA_PATH}"
)


def _google_maps_url(lat: float, lon: float, zoom: int = 15) -> str:
    """Generate a Google Maps URL for a lat/lon coordinate."""
    return f"https://www.google.com/maps/@{lat},{lon},{zoom}z"


def _google_maps_pin_url(lat: float, lon: float) -> str:
    """Generate a Google Maps URL with a dropped pin at lat/lon."""
    return f"https://www.google.com/maps/place/{lat},{lon}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reader():
    """Open BIOMASS reader for the module."""
    r = BIOMASSL1Reader(TEST_DATA_PATH)
    yield r
    r.close()


@pytest.fixture(scope="module")
def geo(reader):
    """Create geolocation object from reader."""
    return Geolocation.from_reader(reader)


# ---------------------------------------------------------------------------
# Basic geolocation setup tests
# ---------------------------------------------------------------------------

def test_geolocation_creation(reader):
    """Test that Geolocation.from_reader works with BIOMASS data."""
    geo = Geolocation.from_reader(reader)
    assert geo is not None
    assert geo.shape[0] == reader.metadata['rows']
    assert geo.shape[1] == reader.metadata['cols']


def test_gcp_count(reader):
    """Verify sufficient GCPs are loaded for interpolation."""
    gcps = reader.metadata.get('gcps', [])
    assert len(gcps) > 10, f"Only {len(gcps)} GCPs loaded, expected many more"
    print(f"\nGCP count: {len(gcps)}")


# ---------------------------------------------------------------------------
# Pixel-to-latlon tests with Google Maps URLs
# ---------------------------------------------------------------------------

def test_image_center(geo):
    """Convert image center pixel to lat/lon and print Google Maps URL."""
    center_row = geo.shape[0] // 2
    center_col = geo.shape[1] // 2

    lat, lon, height = geo.pixel_to_latlon(center_row, center_col)

    assert -90 <= lat <= 90
    assert -180 <= lon <= 180
    assert np.isfinite(height)

    print(f"\n--- Image Center ---")
    print(f"  Pixel: ({center_row}, {center_col})")
    print(f"  Lat/Lon: ({lat:.6f}, {lon:.6f})")
    print(f"  Height: {height:.1f} m")
    print(f"  Google Maps: {_google_maps_pin_url(lat, lon)}")


def test_image_corners(geo):
    """Convert image corners to lat/lon and print Google Maps URLs."""
    rows, cols = geo.shape
    # Inset corners slightly to stay within GCP convex hull
    margin_r = int(rows * 0.05)
    margin_c = int(cols * 0.05)

    corners = {
        'Top-Left':     (margin_r, margin_c),
        'Top-Right':    (margin_r, cols - 1 - margin_c),
        'Bottom-Left':  (rows - 1 - margin_r, margin_c),
        'Bottom-Right': (rows - 1 - margin_r, cols - 1 - margin_c),
    }

    print(f"\n--- Image Corners (5% inset) ---")
    for name, (r, c) in corners.items():
        lat, lon, height = geo.pixel_to_latlon(r, c)

        assert -90 <= lat <= 90, f"{name}: lat {lat} out of range"
        assert -180 <= lon <= 180, f"{name}: lon {lon} out of range"

        print(f"  {name}: pixel ({r}, {c}) -> ({lat:.6f}, {lon:.6f}) h={height:.1f}m")
        print(f"    {_google_maps_pin_url(lat, lon)}")


def test_grid_sample(geo):
    """Convert a grid of points across the image and print Google Maps URLs."""
    rows, cols = geo.shape
    margin_r = int(rows * 0.1)
    margin_c = int(cols * 0.1)

    # 3x3 grid across the image interior
    sample_rows = np.linspace(margin_r, rows - 1 - margin_r, 3, dtype=int)
    sample_cols = np.linspace(margin_c, cols - 1 - margin_c, 3, dtype=int)

    print(f"\n--- 3x3 Grid Sample (10% inset) ---")
    for r in sample_rows:
        for c in sample_cols:
            lat, lon, height = geo.pixel_to_latlon(float(r), float(c))

            assert -90 <= lat <= 90
            assert -180 <= lon <= 180

            print(f"  ({r:6d}, {c:6d}) -> ({lat:.6f}, {lon:.6f}) h={height:.1f}m")
            print(f"    {_google_maps_pin_url(lat, lon)}")


# ---------------------------------------------------------------------------
# Round-trip accuracy tests
# ---------------------------------------------------------------------------

def test_round_trip_center(geo):
    """Test pixel->latlon->pixel round-trip at image center."""
    center_row = float(geo.shape[0] // 2)
    center_col = float(geo.shape[1] // 2)

    lat, lon, height = geo.pixel_to_latlon(center_row, center_col)
    row_back, col_back = geo.latlon_to_pixel(lat, lon)

    row_err = abs(row_back - center_row)
    col_err = abs(col_back - center_col)

    print(f"\n--- Round-trip (center) ---")
    print(f"  Original:  ({center_row:.0f}, {center_col:.0f})")
    print(f"  Returned:  ({row_back:.2f}, {col_back:.2f})")
    print(f"  Error:     ({row_err:.4f} rows, {col_err:.4f} cols)")

    assert row_err < 1.0, f"Row round-trip error {row_err:.4f} exceeds 1 pixel"
    assert col_err < 1.0, f"Col round-trip error {col_err:.4f} exceeds 1 pixel"


def test_round_trip_grid(geo):
    """Test round-trip accuracy at multiple points across the image."""
    rows, cols = geo.shape
    margin_r = int(rows * 0.15)
    margin_c = int(cols * 0.15)

    sample_rows = np.linspace(margin_r, rows - 1 - margin_r, 5)
    sample_cols = np.linspace(margin_c, cols - 1 - margin_c, 5)

    row_errors = []
    col_errors = []

    for r in sample_rows:
        for c in sample_cols:
            lat, lon, _ = geo.pixel_to_latlon(float(r), float(c))
            r_back, c_back = geo.latlon_to_pixel(lat, lon)

            row_errors.append(abs(r_back - r))
            col_errors.append(abs(c_back - c))

    row_errors = np.array(row_errors)
    col_errors = np.array(col_errors)

    print(f"\n--- Round-trip grid (5x5, 15% inset) ---")
    print(f"  Row error: mean={np.mean(row_errors):.4f}, max={np.max(row_errors):.4f} pixels")
    print(f"  Col error: mean={np.mean(col_errors):.4f}, max={np.max(col_errors):.4f} pixels")

    assert np.max(row_errors) < 2.0, f"Max row error {np.max(row_errors):.4f} exceeds 2 pixels"
    assert np.max(col_errors) < 2.0, f"Max col error {np.max(col_errors):.4f} exceeds 2 pixels"


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------

def test_batch_pixel_to_latlon(geo):
    """Test vectorized batch pixel-to-latlon conversion."""
    rows, cols = geo.shape
    margin_r = int(rows * 0.1)
    margin_c = int(cols * 0.1)

    sample_rows = np.linspace(margin_r, rows - 1 - margin_r, 10)
    sample_cols = np.linspace(margin_c, cols - 1 - margin_c, 10)

    lats, lons, heights = geo.pixel_to_latlon_batch(sample_rows, sample_cols)

    assert lats.shape == (10,)
    assert lons.shape == (10,)
    assert heights.shape == (10,)

    # Verify against individual calls
    for i in range(len(sample_rows)):
        lat_single, lon_single, h_single = geo.pixel_to_latlon(
            float(sample_rows[i]), float(sample_cols[i])
        )
        assert abs(lats[i] - lat_single) < 1e-10
        assert abs(lons[i] - lon_single) < 1e-10


# ---------------------------------------------------------------------------
# Footprint and bounds
# ---------------------------------------------------------------------------

def test_footprint(geo):
    """Test footprint calculation and print bounds."""
    footprint = geo.get_footprint()

    assert footprint['type'] == 'Polygon'
    assert footprint['coordinates'] is not None
    assert footprint['bounds'] is not None

    min_lon, min_lat, max_lon, max_lat = footprint['bounds']

    print(f"\n--- Footprint ---")
    print(f"  Bounds: ({min_lat:.6f}, {min_lon:.6f}) to ({max_lat:.6f}, {max_lon:.6f})")
    print(f"  Lat span: {max_lat - min_lat:.6f} deg")
    print(f"  Lon span: {max_lon - min_lon:.6f} deg")
    print(f"  Perimeter points: {len(footprint['coordinates'])}")

    # Verify bounds URL (center of bounding box)
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    print(f"  Footprint center: {_google_maps_pin_url(center_lat, center_lon)}")


# ---------------------------------------------------------------------------
# Interpolation quality
# ---------------------------------------------------------------------------

def test_interpolation_error(geo):
    """Test leave-one-out cross-validation error estimate."""
    errors = geo.get_interpolation_error()

    print(f"\n--- Interpolation Error (leave-one-out) ---")
    print(f"  Mean error:  {errors['mean_error_m']:.2f} m")
    print(f"  RMS error:   {errors['rms_error_m']:.2f} m")
    print(f"  Max error:   {errors['max_error_m']:.2f} m")

    # With ~48 GCPs over a ~150km x 90km image, leave-one-out creates
    # ~20km gaps in the triangulation. RMS errors of 500-600m are expected.
    assert errors['rms_error_m'] < 1000.0, (
        f"RMS error {errors['rms_error_m']:.2f} m exceeds 1000 m threshold"
    )


# ---------------------------------------------------------------------------
# GCP verification against Google Maps
# ---------------------------------------------------------------------------

def test_gcp_locations(reader):
    """Print actual GCP locations for Google Maps verification."""
    gcps = reader.metadata.get('gcps', [])

    # Show a sample of GCPs spread across the image
    n_gcps = len(gcps)
    indices = np.linspace(0, n_gcps - 1, min(8, n_gcps), dtype=int)

    print(f"\n--- GCP Locations (sample of {len(indices)}/{n_gcps}) ---")
    for idx in indices:
        lon, lat, height, row, col = gcps[idx]
        print(f"  GCP[{idx:4d}]: pixel ({row:.0f}, {col:.0f}) -> ({lat:.6f}, {lon:.6f}) h={height:.1f}m")
        print(f"    {_google_maps_pin_url(lat, lon)}")


# ---------------------------------------------------------------------------
# Compare GCP lat/lon with interpolated lat/lon
# ---------------------------------------------------------------------------

def test_gcp_vs_interpolated(geo, reader):
    """Compare GCP ground truth positions with interpolated values."""
    gcps = reader.metadata.get('gcps', [])

    # Sample GCPs from interior (avoid edge where convex hull issues arise)
    rows_img, cols_img = geo.shape
    margin_r = int(rows_img * 0.1)
    margin_c = int(cols_img * 0.1)

    interior_gcps = [
        g for g in gcps
        if margin_r < g[3] < rows_img - margin_r and margin_c < g[4] < cols_img - margin_c
    ]

    # Sample up to 20 interior GCPs
    step = max(1, len(interior_gcps) // 20)
    sample = interior_gcps[::step][:20]

    errors_m = []
    print(f"\n--- GCP vs Interpolated ({len(sample)} interior points) ---")
    for lon_true, lat_true, h_true, row, col in sample:
        lat_interp, lon_interp, _ = geo.pixel_to_latlon(row, col)

        from grdl.geolocation.utils import geographic_distance
        err = geographic_distance(lat_true, lon_true, lat_interp, lon_interp)
        errors_m.append(err)

    errors_m = np.array(errors_m)
    print(f"  Mean error: {np.mean(errors_m):.2f} m")
    print(f"  Max error:  {np.max(errors_m):.2f} m")
    print(f"  Min error:  {np.min(errors_m):.2f} m")

    # GCPs used directly should have very low interpolation error
    assert np.mean(errors_m) < 5.0, (
        f"Mean GCP interpolation error {np.mean(errors_m):.2f} m exceeds 5 m"
    )


# ---------------------------------------------------------------------------
# Visualization: image with geolocation markers
# ---------------------------------------------------------------------------

def plot_geolocation_markers(save_path=None):
    """
    Load the BIOMASS image, plot HH dB magnitude, and overlay geolocation
    test markers (center, corners, grid, GCPs) with lat/lon labels.

    Markers are interactive: clicking any marker opens its location in
    Google Maps in the default browser.

    Parameters
    ----------
    save_path : str or Path, optional
        If provided, save the figure to this path instead of showing it.
    """
    import webbrowser

    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib is required for plotting. Install with: pip install matplotlib")
        return

    if not TEST_DATA_PATH.exists():
        print(f"Test data not found at {TEST_DATA_PATH}")
        return

    print("Opening BIOMASS reader...")
    reader = BIOMASSL1Reader(TEST_DATA_PATH)
    geo = Geolocation.from_reader(reader)
    rows, cols = geo.shape

    # --- Read HH polarization (band 0) as dB magnitude ---
    print(f"Reading HH polarization ({rows} x {cols})...")
    hh = reader.read_chip(0, rows, 0, cols, bands=[0])
    mag_db = 20 * np.log10(np.abs(hh) + 1e-10)

    # Clip dB range for contrast
    vmin = np.nanpercentile(mag_db, 2)
    vmax = np.nanpercentile(mag_db, 98)

    # --- Compute marker locations ---
    # Center
    center_r, center_c = rows // 2, cols // 2
    center_lat, center_lon, _ = geo.pixel_to_latlon(center_r, center_c)

    # Corners (5% inset)
    mr = int(rows * 0.05)
    mc = int(cols * 0.05)
    corner_names = ['TL', 'TR', 'BL', 'BR']
    corner_rc = [
        (mr, mc),
        (mr, cols - 1 - mc),
        (rows - 1 - mr, mc),
        (rows - 1 - mr, cols - 1 - mc),
    ]
    corner_latlons = []
    for r, c in corner_rc:
        lat, lon, _ = geo.pixel_to_latlon(r, c)
        corner_latlons.append((lat, lon))

    # 3x3 grid (10% inset)
    mr10 = int(rows * 0.1)
    mc10 = int(cols * 0.1)
    grid_rows = np.linspace(mr10, rows - 1 - mr10, 3, dtype=int)
    grid_cols = np.linspace(mc10, cols - 1 - mc10, 3, dtype=int)
    grid_points = []
    for r in grid_rows:
        for c in grid_cols:
            lat, lon, _ = geo.pixel_to_latlon(float(r), float(c))
            grid_points.append((r, c, lat, lon))

    # GCPs
    gcps = reader.metadata.get('gcps', [])

    # --- Plot ---
    # Registry mapping each scatter artist to its lat/lon data for clicks
    marker_registry = {}

    # Figure sized proportionally to the image aspect ratio
    aspect = rows / cols
    fig_w = 10
    fig_h = min(fig_w * aspect * 0.6, 28)  # cap height
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    ax.imshow(mag_db, cmap='gray', vmin=vmin, vmax=vmax,
              aspect='auto', interpolation='nearest')

    # GCPs (small green triangles)
    gcp_r = [g[3] for g in gcps]
    gcp_c = [g[4] for g in gcps]
    sc_gcp = ax.scatter(gcp_c, gcp_r, marker='^', c='lime', s=50,
                        edgecolors='black', linewidths=0.5, zorder=3,
                        label='GCPs', picker=True, pickradius=5)
    marker_registry[sc_gcp] = [
        {'lat': g[1], 'lon': g[0], 'label': f'GCP[{i}]'}
        for i, g in enumerate(gcps)
    ]

    # 3x3 grid (cyan circles)
    sc_grid = ax.scatter(
        [p[1] for p in grid_points], [p[0] for p in grid_points],
        marker='o', c='cyan', s=64, edgecolors='black', linewidths=0.5,
        zorder=4, label='Grid (10% inset)', picker=True, pickradius=5)
    marker_registry[sc_grid] = [
        {'lat': p[2], 'lon': p[3], 'label': f'Grid ({p[0]}, {p[1]})'}
        for p in grid_points
    ]
    for r, c, lat, lon in grid_points:
        ax.annotate(f'{lat:.4f}\n{lon:.4f}',
                    xy=(c, r), xytext=(8, 0),
                    textcoords='offset points', fontsize=6,
                    color='cyan', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))

    # Corners (yellow diamonds)
    sc_corners = ax.scatter(
        [rc[1] for rc in corner_rc], [rc[0] for rc in corner_rc],
        marker='D', c='yellow', s=100, edgecolors='black', linewidths=0.8,
        zorder=5, label='Corners (5% inset)', picker=True, pickradius=5)
    marker_registry[sc_corners] = [
        {'lat': ll[0], 'lon': ll[1], 'label': f'Corner {name}'}
        for name, ll in zip(corner_names, corner_latlons)
    ]
    for name, (r, c), (lat, lon) in zip(corner_names, corner_rc, corner_latlons):
        ax.annotate(f'{name}\n{lat:.4f}, {lon:.4f}',
                    xy=(c, r), xytext=(10, -5),
                    textcoords='offset points', fontsize=7,
                    color='yellow', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))

    # Center (red star)
    sc_center = ax.scatter(
        [center_c], [center_r], marker='*', c='red', s=200,
        edgecolors='black', linewidths=0.5, zorder=6, label='Center',
        picker=True, pickradius=5)
    marker_registry[sc_center] = [
        {'lat': center_lat, 'lon': center_lon, 'label': 'Center'}
    ]
    ax.annotate(f'Center\n{center_lat:.4f}, {center_lon:.4f}',
                xy=(center_c, center_r), xytext=(12, 0),
                textcoords='offset points', fontsize=7,
                color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))

    ax.set_title(f'BIOMASS L1 SCS - HH dB  ({rows}x{cols})\n'
                 f'{TEST_DATA_PATH.name}', fontsize=10)
    ax.set_xlabel('Column (range)')
    ax.set_ylabel('Row (azimuth)')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

    # --- Interactive click-to-Google-Maps handler ---
    def on_pick(event):
        artist = event.artist
        if artist not in marker_registry:
            return
        ind = event.ind[0]
        data = marker_registry[artist][ind]
        url = _google_maps_pin_url(data['lat'], data['lon'])
        print(f"  {data['label']}: ({data['lat']:.6f}, {data['lon']:.6f})")
        print(f"  {url}")
        webbrowser.open(url)

    fig.canvas.mpl_connect('pick_event', on_pick)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        print("\nClick any marker to open its location in Google Maps.")
        plt.show()

    reader.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--maps':
        # Run just the Google Maps verification tests with verbose output
        pytest.main([__file__, '-v', '-s', '-k', 'center or corners or grid_sample or gcp_locations or footprint'])
    elif len(sys.argv) > 1 and sys.argv[1] == '--test':
        pytest.main([__file__, '-v', '-s'])
    elif len(sys.argv) > 1 and sys.argv[1] == '--save':
        save = sys.argv[2] if len(sys.argv) > 2 else 'biomass_geolocation.png'
        plot_geolocation_markers(save_path=save)
    elif len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("BIOMASS Geolocation Tests")
        print()
        print("Usage:")
        print("  python tests/test_geolocation_biomass.py          # Plot image with markers (default)")
        print("  python tests/test_geolocation_biomass.py --save [path]  # Save plot to file")
        print("  python tests/test_geolocation_biomass.py --test   # Run all pytest tests")
        print("  python tests/test_geolocation_biomass.py --maps   # Google Maps URLs only")
    else:
        plot_geolocation_markers()