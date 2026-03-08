# -*- coding: utf-8 -*-
"""
File Loading Experiment - Test open_any() across formats and modalities.

Walks a set of data directories containing SAR (SICD, SIDD, CPHD),
EO (Sentinel-2, GeoTIFF), and HDF5 products.  For each file, calls
``open_any()`` and reports which reader was selected, the metadata
extracted, and the modality classification.  Optionally reads a chip
and displays a summary figure.

This exercises the full reader cascade: specialized readers first,
then the GDAL fallback with modality classification.

Usage:
  python test_file_loading.py
  python test_file_loading.py --sar-root /path/to/SAR_DATA
  python test_file_loading.py --plot
  python test_file_loading.py --help

Dependencies
------------
matplotlib (optional, for --plot)
rasterio
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
2026-03-07

Modified
--------
2026-03-07
"""

# Standard library
import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import numpy as np

# GRDL
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from grdl.IO.generic import open_any, GDALFallbackReader
from grdl.IO.base import ImageReader


# ===================================================================
# Configuration
# ===================================================================

SAR_ROOT = Path("/Users/duanesmalley/SAR_DATA")
EO_ROOT = Path("/Volumes/PRO-G40/Imagery_data")

# Files to test — (label, path) tuples
# Populated dynamically from available directories


# ===================================================================
# Helpers
# ===================================================================

def _discover_files(
    sar_root: Path,
    eo_root: Path,
    max_per_format: int = 2,
) -> List[Tuple[str, Path]]:
    """Discover test files from known data directories.

    Parameters
    ----------
    sar_root : Path
        Root directory for SAR data.
    eo_root : Path
        Root directory for EO/multispectral data.
    max_per_format : int
        Maximum files to test per format category.

    Returns
    -------
    List[Tuple[str, Path]]
        List of (label, filepath) pairs.
    """
    files: List[Tuple[str, Path]] = []

    # SICD (complex SAR NITF)
    sicd_dir = sar_root / "SICD"
    if sicd_dir.exists():
        for f in sorted(sicd_dir.glob("*.nitf"))[:max_per_format]:
            files.append(("SICD", f))

    # SIDD (derived SAR NITF)
    sidd_dir = sar_root / "SIDD"
    if sidd_dir.exists():
        for f in sorted(sidd_dir.glob("*.nitf"))[:max_per_format]:
            files.append(("SIDD", f))

    # CPHD (compensated phase history)
    cphd_dir = sar_root / "CPHD"
    if cphd_dir.exists():
        for f in sorted(cphd_dir.glob("*.cphd"))[:max_per_format]:
            files.append(("CPHD", f))

    # GeoTIFF (ASTER DEM, thermal)
    for subdir in ("ASTGTM", "AST_L1T"):
        geotiff_dir = eo_root / subdir
        if geotiff_dir.exists():
            for f in sorted(geotiff_dir.glob("*.tif"))[:max_per_format]:
                files.append((subdir, f))

    # HDF5 (VIIRS)
    for subdir in ("VNP46A1", "VNP13A1"):
        hdf5_dir = eo_root / subdir
        if hdf5_dir.exists():
            for f in sorted(hdf5_dir.glob("*.h5"))[:max_per_format]:
                files.append((subdir, f))

    # Sentinel-2 SAFE (first JP2 from first SAFE directory)
    safe_dirs = sorted(eo_root.glob("S2*_MSIL*.SAFE"))
    for safe_dir in safe_dirs[:1]:
        jp2_files = sorted(safe_dir.rglob("*_B04_*.jp2"))
        if jp2_files:
            files.append(("Sentinel-2", jp2_files[0]))

    return files


def _size_str(filepath: Path) -> str:
    """Human-readable file size."""
    size = filepath.stat().st_size
    if size >= 1024 * 1024 * 1024:
        return f"{size / (1024**3):.1f} GB"
    if size >= 1024 * 1024:
        return f"{size / (1024**2):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def _print_separator(char: str = "=", width: int = 72) -> None:
    print(char * width)


def _print_metadata_summary(reader: ImageReader) -> None:
    """Print a compact metadata summary for any reader."""
    meta = reader.metadata
    print(f"  Format:         {meta.format}")
    print(f"  Reader:         {type(reader).__name__}")
    print(f"  Size:           {meta.rows} x {meta.cols}")
    print(f"  Dtype:          {meta.dtype}")

    if meta.bands is not None:
        print(f"  Bands:          {meta.bands}")
    if meta.crs is not None:
        print(f"  CRS:            {meta.crs}")
    if meta.nodata is not None:
        print(f"  NoData:         {meta.nodata}")

    # GDAL fallback classification
    if isinstance(reader, GDALFallbackReader):
        print(f"  Modality:       {reader.detected_modality or 'unknown'}")
        print(f"  Confidence:     {reader.classification_confidence}")
        print(f"  Clues:          {reader.classification_clues}")
        gdal_driver = meta.get('gdal_driver')
        if gdal_driver:
            print(f"  GDAL Driver:    {gdal_driver}")

    # Sensor-specific metadata (if present)
    if hasattr(meta, 'collection_info') and meta.collection_info is not None:
        ci = meta.collection_info
        if hasattr(ci, 'collector_name') and ci.collector_name:
            print(f"  Collector:      {ci.collector_name}")
        if hasattr(ci, 'core_name') and ci.core_name:
            print(f"  Core name:      {ci.core_name}")

    if hasattr(meta, 'geo_data') and meta.geo_data is not None:
        gd = meta.geo_data
        if hasattr(gd, 'scp') and gd.scp is not None:
            llh = gd.scp.llh if hasattr(gd.scp, 'llh') else None
            if llh is not None:
                print(f"  SCP:            ({llh.lat:.6f}, {llh.lon:.6f})")

    # Extras count
    extras_count = len(meta.extras) if meta.extras else 0
    if extras_count > 0:
        print(f"  Extras:         {extras_count} additional fields")


def _try_read_chip(
    reader: ImageReader,
    chip_size: int = 256,
) -> Optional[np.ndarray]:
    """Try to read a small chip from the center of the image.

    Parameters
    ----------
    reader : ImageReader
        Open reader instance.
    chip_size : int
        Target chip size (rows and cols).

    Returns
    -------
    np.ndarray or None
        Chip data, or None if read fails.
    """
    rows = reader.metadata.rows
    cols = reader.metadata.cols
    half = chip_size // 2

    r_center = rows // 2
    c_center = cols // 2
    r0 = max(0, r_center - half)
    c0 = max(0, c_center - half)
    r1 = min(rows, r0 + chip_size)
    c1 = min(cols, c0 + chip_size)

    try:
        chip = reader.read_chip(r0, r1, c0, c1)
        return chip
    except Exception as e:
        print(f"  Chip read failed: {e}")
        return None


# ===================================================================
# Test runner
# ===================================================================

class LoadResult:
    """Result from loading a single file."""

    def __init__(
        self,
        label: str,
        filepath: Path,
        reader_class: str,
        format_name: str,
        rows: int,
        cols: int,
        dtype: str,
        modality: Optional[str],
        load_time_ms: float,
        chip: Optional[np.ndarray],
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        self.label = label
        self.filepath = filepath
        self.reader_class = reader_class
        self.format_name = format_name
        self.rows = rows
        self.cols = cols
        self.dtype = dtype
        self.modality = modality
        self.load_time_ms = load_time_ms
        self.chip = chip
        self.success = success
        self.error = error


def run_load_test(
    label: str,
    filepath: Path,
    read_chip: bool = True,
    chip_size: int = 256,
) -> LoadResult:
    """Test loading a single file with open_any().

    Parameters
    ----------
    label : str
        Expected format label for reporting.
    filepath : Path
        Path to the file.
    read_chip : bool
        Whether to also read a chip.
    chip_size : int
        Chip size for test read.

    Returns
    -------
    LoadResult
        Results of the load attempt.
    """
    _print_separator("-", 72)
    print(f"[{label}] {filepath.name}")
    print(f"  Path: {filepath}")
    print(f"  Size: {_size_str(filepath)}")

    t0 = time.perf_counter()
    try:
        reader = open_any(filepath)
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  FAILED: {e}")
        return LoadResult(
            label=label, filepath=filepath, reader_class="N/A",
            format_name="N/A", rows=0, cols=0, dtype="N/A",
            modality=None, load_time_ms=elapsed, chip=None,
            success=False, error=str(e),
        )

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  Opened in {elapsed:.1f} ms")
    _print_metadata_summary(reader)

    chip = None
    if read_chip:
        chip = _try_read_chip(reader, chip_size)
        if chip is not None:
            print(f"  Chip:           shape={chip.shape}, "
                  f"dtype={chip.dtype}, "
                  f"range=[{np.nanmin(chip):.4g}, {np.nanmax(chip):.4g}]")

    modality = None
    if isinstance(reader, GDALFallbackReader):
        modality = reader.detected_modality
    elif reader.metadata.format in ('SICD', 'CPHD', 'CRSD', 'SIDD'):
        modality = 'SAR'
    elif reader.metadata.format == 'JPEG2000':
        modality = 'MSI'
    elif reader.metadata.format == 'HDF5':
        modality = 'EO/MS'

    result = LoadResult(
        label=label,
        filepath=filepath,
        reader_class=type(reader).__name__,
        format_name=reader.metadata.format,
        rows=reader.metadata.rows,
        cols=reader.metadata.cols,
        dtype=reader.metadata.dtype,
        modality=modality,
        load_time_ms=elapsed,
        chip=chip,
        success=True,
    )

    reader.close()
    return result


# ===================================================================
# Summary and visualization
# ===================================================================

def print_summary(results: List[LoadResult]) -> None:
    """Print a summary table of all load results.

    Parameters
    ----------
    results : List[LoadResult]
        Results from all load tests.
    """
    _print_separator("=", 72)
    print("SUMMARY")
    _print_separator("=", 72)

    passed = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    print(f"Total: {len(results)}  |  Passed: {passed}  |  Failed: {failed}")
    print()

    # Table header
    hdr = (f"{'Label':<12} {'Reader':<22} {'Format':<10} "
           f"{'Size':>12} {'Dtype':<10} {'Time':>8}")
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        status = "OK" if r.success else "FAIL"
        size_str = f"{r.rows}x{r.cols}" if r.success else "---"
        dtype_str = r.dtype if r.success else "---"
        time_str = f"{r.load_time_ms:.0f}ms"
        reader_str = r.reader_class if r.success else f"FAIL: {r.error[:18]}"
        print(f"{'['+status+']':<5} {r.label:<12} {reader_str:<22} "
              f"{r.format_name:<10} {size_str:>12} {dtype_str:<10} "
              f"{time_str:>8}")

    print()

    # Modality breakdown
    modalities: Dict[str, int] = {}
    for r in results:
        if r.success and r.modality:
            modalities[r.modality] = modalities.get(r.modality, 0) + 1
    if modalities:
        print("Modality breakdown:")
        for mod, count in sorted(modalities.items()):
            print(f"  {mod:<12}: {count}")
    print()


def plot_results(results: List[LoadResult]) -> None:
    """Display chip thumbnails for all successful loads.

    Parameters
    ----------
    results : List[LoadResult]
        Results from all load tests.
    """
    import matplotlib
    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt

    # Filter to results with chips
    with_chips = [r for r in results if r.success and r.chip is not None]
    if not with_chips:
        print("No chips to display.")
        return

    n = len(with_chips)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
                             squeeze=False)

    for i, r in enumerate(with_chips):
        row_idx, col_idx = divmod(i, cols)
        ax = axes[row_idx, col_idx]

        chip = r.chip
        # Handle complex data (SAR)
        if np.iscomplexobj(chip):
            display = np.abs(chip).astype(np.float32)
            cmap = "gray"
        elif chip.ndim == 3 and chip.shape[0] == 3:
            # RGB (bands, rows, cols) -> (rows, cols, bands)
            display = np.transpose(chip, (1, 2, 0)).astype(np.float32)
            cmap = None
        elif chip.ndim == 3:
            display = chip[0].astype(np.float32)
            cmap = "viridis"
        else:
            display = chip.astype(np.float32)
            cmap = "gray"

        # Percentile stretch
        valid = display[np.isfinite(display)]
        if valid.size > 0:
            vmin = np.percentile(valid, 2)
            vmax = np.percentile(valid, 98)
            if vmax > vmin:
                display = np.clip((display - vmin) / (vmax - vmin), 0, 1)

        if cmap:
            ax.imshow(display, cmap=cmap, interpolation="nearest")
        else:
            ax.imshow(display, interpolation="nearest")

        ax.set_title(
            f"{r.label}\n{r.reader_class} | {r.format_name}\n"
            f"{r.rows}x{r.cols} {r.dtype}",
            fontsize=7,
        )
        ax.set_xlabel(f"{r.load_time_ms:.0f} ms", fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused axes
    for i in range(n, rows * cols):
        row_idx, col_idx = divmod(i, cols)
        axes[row_idx, col_idx].set_visible(False)

    fig.suptitle("open_any() File Loading Experiment", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    plt.show()


# ===================================================================
# Main
# ===================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Test open_any() across multiple imagery formats. "
            "Discovers files in SAR and EO data directories, opens each "
            "with open_any(), reports the reader selected and metadata "
            "extracted, and optionally displays chip thumbnails."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python test_file_loading.py\n"
            "  python test_file_loading.py --plot\n"
            "  python test_file_loading.py --sar-root /data/SAR\n"
            "  python test_file_loading.py --max 1 --no-chip\n"
        ),
    )
    parser.add_argument(
        "--sar-root",
        type=Path,
        default=SAR_ROOT,
        help=f"Root directory for SAR data (default: {SAR_ROOT})",
    )
    parser.add_argument(
        "--eo-root",
        type=Path,
        default=EO_ROOT,
        help=f"Root directory for EO data (default: {EO_ROOT})",
    )
    parser.add_argument(
        "--max", "-n",
        type=int,
        default=2,
        dest="max_per_format",
        help="Maximum files per format category (default: 2)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display chip thumbnails for all loaded files.",
    )
    parser.add_argument(
        "--no-chip",
        action="store_true",
        help="Skip chip reads (metadata only).",
    )
    parser.add_argument(
        "--chip-size",
        type=int,
        default=256,
        help="Chip size in pixels (default: 256).",
    )
    parser.add_argument(
        "--file", "-f",
        type=Path,
        default=None,
        help="Test a single file instead of auto-discovery.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the file loading experiment."""
    args = parse_args()

    _print_separator("=", 72)
    print("FILE LOADING EXPERIMENT")
    print("Tests open_any() across imagery formats and modalities")
    _print_separator("=", 72)

    # Discover or use single file
    if args.file:
        if not args.file.exists():
            print(f"File not found: {args.file}")
            sys.exit(1)
        test_files = [("manual", args.file)]
    else:
        print(f"SAR root:  {args.sar_root}")
        print(f"EO root:   {args.eo_root}")
        print(f"Max/format: {args.max_per_format}")
        print()

        test_files = _discover_files(
            args.sar_root, args.eo_root,
            max_per_format=args.max_per_format,
        )

    if not test_files:
        print("No test files found. Check data paths.")
        sys.exit(1)

    print(f"Discovered {len(test_files)} files to test:")
    for label, fp in test_files:
        print(f"  [{label}] {fp.name}")
    print()

    # Run tests
    results: List[LoadResult] = []
    for label, filepath in test_files:
        result = run_load_test(
            label, filepath,
            read_chip=not args.no_chip,
            chip_size=args.chip_size,
        )
        results.append(result)
        print()

    # Summary
    print_summary(results)

    # Plot
    if args.plot:
        plot_results(results)


if __name__ == "__main__":
    main()
