# -*- coding: utf-8 -*-
"""
SICD orthorectification demo -- slant range vs. orthorectified.

A minimal, documentation-ready example that orthorectifies a SICD
complex SAR image over Savannah, Georgia using a FABDEM terrain model,
then writes two side-by-side PNG figures:

  - ``sicd_slant_range.png``     -- magnitude in native slant geometry
  - ``sicd_orthorectified.png``  -- magnitude on a WGS-84 ground grid

The Savannah scene has a graze angle of ~45 degrees, so the ortho
output is visibly compressed relative to the slant-range chip -- a
clear illustration of why slant-plane SAR imagery must be
orthorectified before it is geographically meaningful.

Dependencies
------------
matplotlib
sarpy (or sarkit)
rasterio (for the FABDEM GeoTIFF)

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
2026-04-19

Modified
--------
2026-04-19
"""

# Standard library
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import numpy as np

# GRDL
from grdl.IO.sar import SICDReader
from grdl.data_prep import ChipExtractor
from grdl.geolocation.chip import ChipGeolocation
from grdl.geolocation.elevation import open_elevation
from grdl.geolocation.sar.sicd import SICDGeolocation
from grdl.image_processing.ortho import orthorectify


# Default inputs -- override by calling run() with different arguments.
SICD_PATH = Path(
    '/Users/duanesmalley/SAR_DATA/SICD/'
    '2025-06-20-02-42-41_UMBRA-05_SICD.nitf'
)
DEM_PATH = Path('/Volumes/PRO-G40/terrain/FABDEM')
OUTPUT_DIR = Path(__file__).resolve().parents[3] / 'docs' / 'images'
CHIP_SIZE = 2048*4


def _stretch(image: np.ndarray, low: float = 2.0,
             high: float = 98.0) -> tuple:
    """Return (vmin, vmax) percentile clip for display."""
    finite = image[np.isfinite(image)]
    return (float(np.percentile(finite, low)),
            float(np.percentile(finite, high)))


def _save_figure(image: np.ndarray, title: str, xlabel: str,
                 ylabel: str, extent, output_path: Path) -> None:
    """Render a single magnitude image to a standalone PNG."""
    vmin, vmax = _stretch(image)
    fig, ax = plt.subplots(dpi=125)
    ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax,
              extent=extent, origin='upper',
              interpolation='bilinear', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {output_path}')


def run(sicd_path: Path = SICD_PATH,
        dem_path: Path = DEM_PATH,
        output_dir: Path = OUTPUT_DIR,
        chip_size: int = CHIP_SIZE) -> None:
    """Generate slant-range and orthorectified PNGs for documentation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Loading SICD: {sicd_path.name}')

    # 1. Open reader, plan a center chip, and read complex samples.
    with SICDReader(sicd_path) as reader:
        meta = reader.metadata
        rows, cols = meta.rows, meta.cols
        scp = meta.geo_data.scp.llh
        graze = meta.scpcoa.graze_ang
        print(f'  Full image : {rows} x {cols}')
        print(f'  SCP        : ({scp.lat:.4f}, {scp.lon:.4f})')
        print(f'  Graze      : {graze:.2f} deg')

        extractor = ChipExtractor(nrows=rows, ncols=cols)
        region = extractor.chip_at_point(
            rows // 2, cols // 2,
            row_width=chip_size, col_width=chip_size,
        )
        chip_rows = region.row_end - region.row_start
        chip_cols = region.col_end - region.col_start
        print(f'  Chip       : {chip_rows} x {chip_cols} '
              f'at [{region.row_start}, {region.col_start}]')

        # 2. Build geolocation for the full image, wrap for the chip,
        #    and attach a DEM so R/Rdot projection is terrain-corrected.
        geo_full = SICDGeolocation.from_reader(reader)
        geo = ChipGeolocation(
            geo_full,
            row_offset=region.row_start,
            col_offset=region.col_start,
            shape=(chip_rows, chip_cols),
        )
        geo.elevation = open_elevation(
            dem_path,
            location=(float(scp.lat), float(scp.lon)),
        )
        print(f'  DEM        : {type(geo.elevation).__name__}')

        # 3. Read the chip and drop to magnitude *before* resampling --
        #    resampling complex samples causes phase cancellation.
        complex_chip = reader.read_chip(
            region.row_start, region.row_end,
            region.col_start, region.col_end,
        )

    magnitude = np.abs(complex_chip).astype(np.float32)
    del complex_chip

    # 4. Orthorectify to a WGS-84 geographic grid.  Resolution and
    #    grid bounds are auto-computed from the SICD grid metadata.
    print('Running orthorectifier...')
    result = orthorectify(
        geolocation=geo,
        source_array=magnitude,
        metadata=meta,
        interpolation='bilinear',
        nodata=np.nan,
    )
    grid = result.output_grid
    print(f'  Output grid: {grid.rows} x {grid.cols}')
    print(f'  Lat range  : [{grid.min_lat:.4f}, {grid.max_lat:.4f}]')
    print(f'  Lon range  : [{grid.min_lon:.4f}, {grid.max_lon:.4f}]')

    # 5. Save two standalone PNGs.
    print('Writing PNGs...')
    _save_figure(
        magnitude,
        title=f'SICD slant range ({chip_rows}x{chip_cols} chip)',
        xlabel='Column (range)',
        ylabel='Row (azimuth)',
        extent=None,
        output_path=output_dir / 'sicd_slant_range.png',
    )
    _save_figure(
        np.flipud(result.data),
        title=f'Orthorectified (WGS-84, DEM-corrected, graze {graze:.1f}\u00b0)',
        xlabel='Longitude (deg)',
        ylabel='Latitude (deg)',
        extent=(grid.min_lon, grid.max_lon,
                grid.min_lat, grid.max_lat),
        output_path=output_dir / 'sicd_orthorectified.png',
    )
    print('Done.')


if __name__ == '__main__':
    run()
