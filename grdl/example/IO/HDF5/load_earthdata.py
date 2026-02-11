# -*- coding: utf-8 -*-
"""
Earthdata Reader Demo - Load NASA HDF5 and GeoTIFF imagery with GRDL.

Demonstrates HDF5Reader and GeoTIFFReader on real NASA Earthdata
products downloaded from LAADS DAAC and LP DAAC:

  - VIIRS VNP46A1 nighttime lights (HDF5)
  - VIIRS VNP13A1 vegetation index (HDF5)
  - ASTER L1T thermal infrared bands (GeoTIFF)
  - ASTER GDEM v3 elevation model (GeoTIFF)

Shows dataset browsing, metadata extraction, chip reading, full
reads, and visualization for each product.

Usage:
  python load_earthdata.py
  python load_earthdata.py --data-root /path/to/imagery
  python load_earthdata.py --product VNP46A1
  python load_earthdata.py --help

Dependencies
------------
h5py
rasterio
matplotlib

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
2026-02-10

Modified
--------
2026-02-10
"""

# Standard library
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party
import numpy as np

# Matplotlib -- set backend before importing pyplot
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt  # noqa: E402

# GRDL
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from grdl.IO.hdf5 import HDF5Reader
from grdl.IO.geotiff import GeoTIFFReader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/Volumes/PRO-G40/Imagery_data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct_stretch(
    data: np.ndarray,
    plow: float = 2.0,
    phigh: float = 98.0,
) -> np.ndarray:
    """Percentile contrast stretch to [0, 1] float32.

    Parameters
    ----------
    data : np.ndarray
        Input array (any numeric dtype).
    plow : float
        Lower percentile.
    phigh : float
        Upper percentile.

    Returns
    -------
    np.ndarray
        Stretched array clipped to [0, 1], dtype float32.
    """
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    vmin = np.percentile(valid, plow)
    vmax = np.percentile(valid, phigh)
    if vmax <= vmin:
        return np.zeros_like(data, dtype=np.float32)
    stretched = (data.astype(np.float32) - vmin) / (vmax - vmin)
    return np.clip(stretched, 0.0, 1.0)


def _print_metadata(metadata: Dict, indent: int = 2) -> None:
    """Print a metadata dict with consistent formatting.

    Parameters
    ----------
    metadata : Dict
        Reader metadata dictionary.
    indent : int
        Number of leading spaces.
    """
    pad = " " * indent
    # Standard keys first
    for key in ('format', 'rows', 'cols', 'bands', 'dtype', 'dataset_path',
                'crs', 'nodata'):
        if key in metadata:
            print(f"{pad}{key:20s}: {metadata[key]}")

    # Extra attributes (skip standard keys)
    standard = {'format', 'rows', 'cols', 'bands', 'dtype', 'dataset_path',
                'crs', 'nodata', 'transform', 'bounds', 'resolution'}
    extras = {k: v for k, v in metadata.items() if k not in standard}
    if extras:
        print(f"{pad}{'--- attributes ---':20s}")
        for key, val in list(extras.items())[:10]:
            val_str = str(val)
            if len(val_str) > 60:
                val_str = val_str[:57] + "..."
            print(f"{pad}{key:20s}: {val_str}")
        if len(extras) > 10:
            print(f"{pad}  ... and {len(extras) - 10} more")


# ---------------------------------------------------------------------------
# Product loaders
# ---------------------------------------------------------------------------

def load_hdf5_product(
    product_dir: Path,
    product_name: str,
) -> List[Tuple[str, np.ndarray, Dict]]:
    """Load all HDF5 files in a product directory.

    For each file, lists available datasets, opens with auto-detect,
    reads the full image, and returns the data with metadata.

    Parameters
    ----------
    product_dir : Path
        Directory containing .h5 files.
    product_name : str
        Product name for display.

    Returns
    -------
    List[Tuple[str, np.ndarray, Dict]]
        List of (filename, data_array, metadata) tuples.
    """
    h5_files = sorted(product_dir.glob("*.h5"))
    if not h5_files:
        print(f"  No HDF5 files found in {product_dir}")
        return []

    results = []
    for h5_path in h5_files:
        print(f"\n  File: {h5_path.name}")
        size_mb = h5_path.stat().st_size / (1024 * 1024)
        print(f"  Size: {size_mb:.1f} MB")

        # Browse available datasets
        print(f"  Available datasets:")
        try:
            datasets = HDF5Reader.list_datasets(h5_path)
        except (OSError, ValueError) as e:
            print(f"    ERROR: Cannot open file ({e})")
            continue

        for ds_path, ds_shape, ds_dtype in datasets:
            print(f"    {ds_path}: {ds_shape} ({ds_dtype})")

        if not datasets:
            print("    (no 2D+ numeric datasets found)")
            continue

        # Open with auto-detect
        with HDF5Reader(h5_path) as reader:
            print(f"\n  Auto-selected: {reader.dataset_path}")
            print(f"  Metadata:")
            _print_metadata(reader.metadata, indent=4)

            shape = reader.get_shape()
            print(f"\n  Shape: {shape}")
            print(f"  Dtype: {reader.get_dtype()}")

            # Read a chip (first 256x256 or full if smaller)
            rows = reader.metadata['rows']
            cols = reader.metadata['cols']
            chip_rows = min(256, rows)
            chip_cols = min(256, cols)
            chip = reader.read_chip(0, chip_rows, 0, chip_cols)
            print(f"  Chip [0:{chip_rows}, 0:{chip_cols}]: "
                  f"shape={chip.shape}, range=[{chip.min()}, {chip.max()}]")

            # Read full image
            data = reader.read_full()
            print(f"  Full read: shape={data.shape}, "
                  f"range=[{data.min()}, {data.max()}]")

            results.append((h5_path.name, data, dict(reader.metadata)))

    return results


def load_geotiff_product(
    product_dir: Path,
    product_name: str,
    pattern: str = "*.tif",
) -> List[Tuple[str, np.ndarray, Dict]]:
    """Load all GeoTIFF files in a product directory.

    Parameters
    ----------
    product_dir : Path
        Directory containing .tif files.
    product_name : str
        Product name for display.
    pattern : str
        Glob pattern for TIF files.

    Returns
    -------
    List[Tuple[str, np.ndarray, Dict]]
        List of (filename, data_array, metadata) tuples.
    """
    tif_files = sorted(product_dir.glob(pattern))
    if not tif_files:
        print(f"  No GeoTIFF files found in {product_dir}")
        return []

    results = []
    for tif_path in tif_files:
        print(f"\n  File: {tif_path.name}")
        size_mb = tif_path.stat().st_size / (1024 * 1024)
        print(f"  Size: {size_mb:.1f} MB")

        with GeoTIFFReader(tif_path) as reader:
            print(f"  Metadata:")
            _print_metadata(reader.metadata, indent=4)

            shape = reader.get_shape()
            print(f"\n  Shape: {shape}")
            print(f"  Dtype: {reader.get_dtype()}")

            # Read full image
            data = reader.read_full()
            print(f"  Full read: shape={data.shape}, "
                  f"range=[{data.min()}, {data.max()}]")

            results.append((tif_path.name, data, dict(reader.metadata)))

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_hdf5_results(
    results: List[Tuple[str, np.ndarray, Dict]],
    product_name: str,
) -> Optional[plt.Figure]:
    """Create a multi-panel plot for HDF5 products.

    Parameters
    ----------
    results : list
        Output from load_hdf5_product.
    product_name : str
        Product name for the figure title.

    Returns
    -------
    Optional[plt.Figure]
        The matplotlib figure, or None if no results.
    """
    if not results:
        return None

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for i, (fname, data, meta) in enumerate(results):
        ax = axes[0, i]

        # For 3D data, show the first band
        if data.ndim == 3:
            display = data[0].astype(np.float32)
            band_label = "Band 0"
        else:
            display = data.astype(np.float32)
            band_label = ""

        # Replace fill values with NaN for cleaner display
        display = np.where(display == 0, np.nan, display)

        stretched = _pct_stretch(display)
        im = ax.imshow(stretched, cmap="viridis", interpolation="nearest")
        ax.set_title(f"{fname}\n{band_label}", fontsize=8)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        fig.colorbar(im, ax=ax, shrink=0.7, label="Stretched")

    fig.suptitle(f"{product_name} — HDF5Reader", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_geotiff_results(
    results: List[Tuple[str, np.ndarray, Dict]],
    product_name: str,
    cmap: str = "gray",
) -> Optional[plt.Figure]:
    """Create a multi-panel plot for GeoTIFF products.

    Parameters
    ----------
    results : list
        Output from load_geotiff_product.
    product_name : str
        Product name for the figure title.
    cmap : str
        Colormap to use.

    Returns
    -------
    Optional[plt.Figure]
        The matplotlib figure, or None if no results.
    """
    if not results:
        return None

    n = len(results)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
                             squeeze=False)

    for i, (fname, data, meta) in enumerate(results):
        r, c = divmod(i, cols)
        ax = axes[r, c]

        if data.ndim == 3:
            display = data[0].astype(np.float32)
        else:
            display = data.astype(np.float32)

        # Replace nodata with NaN
        nodata = meta.get('nodata')
        if nodata is not None:
            display = np.where(display == nodata, np.nan, display)

        stretched = _pct_stretch(display)
        im = ax.imshow(stretched, cmap=cmap, interpolation="nearest")
        ax.set_title(fname, fontsize=7)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    # Hide unused axes
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)

    fig.suptitle(f"{product_name} — GeoTIFFReader", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Load NASA Earthdata imagery with GRDL IO readers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python load_earthdata.py                     "
            "# Load all products\n"
            "  python load_earthdata.py --product VNP46A1   "
            "# Single product\n"
            "  python load_earthdata.py --no-plot            "
            "# Metadata only\n"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help=f"Root imagery directory (default: {DATA_ROOT})",
    )
    parser.add_argument(
        "--product", "-p",
        type=str,
        default=None,
        choices=["VNP46A1", "VNP13A1", "AST_L1T", "ASTGTM"],
        help="Load only this product (default: all)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip visualization, print metadata only",
    )
    return parser.parse_args()


def main() -> None:
    """Load and display NASA Earthdata imagery using GRDL readers."""
    args = parse_args()
    data_root = args.data_root
    figures = []

    # -- HDF5 products -------------------------------------------------------

    hdf5_products = {
        "VNP46A1": "VIIRS Nighttime Lights Daily",
        "VNP13A1": "VIIRS Vegetation Index 16-Day",
    }

    for short_name, description in hdf5_products.items():
        if args.product and args.product != short_name:
            continue

        product_dir = data_root / short_name
        if not product_dir.exists():
            print(f"\nSkipping {short_name}: {product_dir} not found")
            continue

        print(f"\n{'='*60}")
        print(f"{short_name} — {description}")
        print(f"Reader: HDF5Reader")
        print(f"{'='*60}")

        results = load_hdf5_product(product_dir, short_name)

        if not args.no_plot and results:
            fig = plot_hdf5_results(results, f"{short_name}: {description}")
            if fig:
                figures.append(fig)

    # -- GeoTIFF products ----------------------------------------------------

    geotiff_products = {
        "AST_L1T": ("ASTER L1T Thermal Infrared", "*.tif", "inferno"),
        "ASTGTM": ("ASTER Global DEM v3", "*_dem.tif", "terrain"),
    }

    for short_name, (description, pattern, cmap) in geotiff_products.items():
        if args.product and args.product != short_name:
            continue

        product_dir = data_root / short_name
        if not product_dir.exists():
            print(f"\nSkipping {short_name}: {product_dir} not found")
            continue

        print(f"\n{'='*60}")
        print(f"{short_name} — {description}")
        print(f"Reader: GeoTIFFReader")
        print(f"{'='*60}")

        results = load_geotiff_product(product_dir, short_name, pattern)

        if not args.no_plot and results:
            fig = plot_geotiff_results(results, f"{short_name}: {description}",
                                       cmap=cmap)
            if fig:
                figures.append(fig)

    # -- Show all figures ----------------------------------------------------

    print(f"\n{'='*60}")
    print(f"Done. Loaded imagery from {data_root}")
    print(f"{'='*60}")

    if figures:
        plt.show()


if __name__ == "__main__":
    main()
