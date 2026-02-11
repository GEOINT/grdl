# -*- coding: utf-8 -*-
"""
Sentinel-2 Reader - Read Sentinel-2 MSI JPEG2000 products.

Sensor-specific reader for Sentinel-2A/2B/2C multispectral imagery.
Wraps JP2Reader for pixel access and extracts Sentinel-2-specific
metadata from filename patterns and JPEG2000 tags into a typed
Sentinel2Metadata dataclass.

Supports Level-1C (TOA Reflectance) and Level-2A (Surface Reflectance)
products in both standalone JP2 format and SAFE archive structure.

Dependencies
------------
rasterio or glymur (via JP2Reader)

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
2026-02-11

Modified
--------
2026-02-11
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from grdl.IO.base import ImageReader
from grdl.IO.models import Sentinel2Metadata
from grdl.IO.jpeg2000 import JP2Reader


# Band wavelength lookup table (center, min, max) in nm
BAND_WAVELENGTHS = {
    'B01': (443, 432, 453),   # Coastal aerosol
    'B02': (490, 458, 523),   # Blue
    'B03': (560, 543, 578),   # Green
    'B04': (665, 650, 680),   # Red
    'B05': (705, 698, 713),   # Red edge 1
    'B06': (740, 733, 748),   # Red edge 2
    'B07': (783, 773, 793),   # Red edge 3
    'B08': (842, 785, 900),   # NIR
    'B8A': (865, 855, 875),   # NIR narrow
    'B09': (945, 935, 955),   # Water vapor
    'B10': (1375, 1360, 1390), # Cirrus
    'B11': (1610, 1565, 1655), # SWIR 1
    'B12': (2190, 2100, 2280), # SWIR 2
}


def _parse_sentinel2_filename(filepath: Path) -> dict:
    """Parse Sentinel-2 metadata from filename.

    Handles three naming conventions:

    1. Standalone band JP2:
       T10SEG_20240115T184719_B04_10m.jp2

    2. SAFE archive directory:
       S2A_MSIL2A_20240115T184719_N0510_R070_T10SEG_20240115T201234.SAFE

    3. Band file within SAFE structure:
       S2*.SAFE/GRANULE/.../IMG_DATA/R10m/T10SEG_20240115T184719_B04_10m.jp2

    Parameters
    ----------
    filepath : Path
        Path to Sentinel-2 file or directory.

    Returns
    -------
    dict
        Extracted metadata fields. Empty dict if parsing fails (no exceptions).
        Possible keys: satellite, product_type, sensing_datetime,
        baseline_processing, relative_orbit, mgrs_tile_id,
        product_discriminator, processing_level, band_id, resolution_tier
    """
    result = {}
    name = filepath.name
    parts = filepath.parts

    # =========================================================================
    # Pattern 2: SAFE Archive Directory Name (check first for comprehensive info)
    # =========================================================================
    # Format: S2{sat}_MSI{level}_{datetime}_{baseline}_R{orbit}_{tile}_{discrim}.SAFE
    # Example: S2A_MSIL2A_20240115T184719_N0510_R070_T10SEG_20240115T201234.SAFE

    # First, check if any parent directory is a SAFE archive
    for part in parts:
        if part.endswith('.SAFE'):
            match = re.match(
                r'(S2[ABC])_'                         # Satellite: S2A, S2B, or S2C
                r'(MSIL[12][CA])_'                    # Product type: MSIL1C or MSIL2A
                r'([0-9]{8}T[0-9]{6})_'               # Sensing datetime
                r'(N[0-9]{4})_'                       # Baseline: N + 4 digits
                r'R([0-9]{3})_'                       # Relative orbit: R + 3 digits
                r'(T[0-9]{2}[A-Z]{3})_'               # MGRS tile
                r'([0-9]{8}T[0-9]{6})'                # Product discriminator datetime
                r'(?:\.SAFE)?$',                      # Optional .SAFE extension at end
                part
            )
            if match:
                result['satellite'] = match.group(1)
                result['product_type'] = match.group(2)
                result['sensing_datetime'] = match.group(3)
                result['baseline_processing'] = match.group(4)
                result['relative_orbit'] = int(match.group(5))
                result['mgrs_tile_id'] = match.group(6)
                result['product_discriminator'] = match.group(7)
                result['processing_level'] = 'L1C' if 'L1C' in match.group(2) else 'L2A'
                break

    # =========================================================================
    # Pattern 1: Standalone Band File or Band within SAFE
    # =========================================================================
    # Format: T{tile}_{datetime}_{band}_{resolution}m.jp2
    # Example: T10SEG_20240115T184719_B04_10m.jp2
    band_match = re.match(
        r'(T[0-9]{2}[A-Z]{3})_'       # MGRS tile: T + 2 digits + 3 letters
        r'([0-9]{8}T[0-9]{6})_'       # Datetime: YYYYMMDDTHHMMSS
        r'(B[0-9]{1,2}A?)_'           # Band: B + 1-2 digits + optional A (for B8A)
        r'([0-9]{2})m',               # Resolution: 2 digits + 'm'
        name
    )
    if band_match:
        # Only set if not already from SAFE parsing
        if 'mgrs_tile_id' not in result:
            result['mgrs_tile_id'] = band_match.group(1)
        if 'sensing_datetime' not in result:
            result['sensing_datetime'] = band_match.group(2)
        # Always set band-specific fields
        result['band_id'] = band_match.group(3)
        result['resolution_tier'] = int(band_match.group(4))

    return result  # Combined SAFE + band metadata, or standalone band, or empty dict


def _parse_mgrs_tile(mgrs_id: str) -> Tuple[Optional[int], Optional[str]]:
    """Extract UTM zone and latitude band from MGRS tile ID.

    Parameters
    ----------
    mgrs_id : str
        MGRS tile identifier (e.g., 'T10SEG').

    Returns
    -------
    utm_zone : int or None
        UTM zone number (1-60).
    latitude_band : str or None
        MGRS latitude band letter.

    Examples
    --------
    >>> _parse_mgrs_tile('T10SEG')
    (10, 'S')
    """
    if not mgrs_id or len(mgrs_id) < 4:
        return None, None

    try:
        utm_zone = int(mgrs_id[1:3])
        latitude_band = mgrs_id[3]
        return utm_zone, latitude_band
    except (ValueError, IndexError):
        return None, None


class Sentinel2Reader(ImageReader):
    """Read Sentinel-2 MSI JPEG2000 products.

    Wraps JP2Reader for pixel access and extracts Sentinel-2-specific
    metadata from filename patterns and JPEG2000 tags. Supports both
    Level-1C (TOA Reflectance) and Level-2A (Surface Reflectance).

    Parameters
    ----------
    filepath : str or Path
        Path to Sentinel-2 JP2 file (standalone or within SAFE archive).
    backend : str, optional
        JP2 backend to use ('rasterio', 'glymur', or 'auto'). Default 'auto'.

    Attributes
    ----------
    filepath : Path
        Path to the JP2 file.
    metadata : Sentinel2Metadata
        Typed Sentinel-2 metadata with mission, band, and tile info.
    jp2_reader : JP2Reader
        Wrapped JP2Reader instance for pixel access.

    Raises
    ------
    ImportError
        If neither rasterio nor glymur is available.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not a valid Sentinel-2 JP2.

    Examples
    --------
    >>> from grdl.IO.eo import Sentinel2Reader
    >>> with Sentinel2Reader('T10SEG_20240115T184719_B04_10m.jp2') as reader:
    ...     print(reader.metadata.satellite)
    ...     print(reader.metadata.mgrs_tile_id)
    ...     chip = reader.read_chip(0, 1000, 0, 1000)
    'S2A'
    'T10SEG'

    >>> # Auto-detection via open_eo()
    >>> from grdl.IO.eo import open_eo
    >>> reader = open_eo('S2A_MSIL2A_*.SAFE/GRANULE/.../B04_10m.jp2')
    >>> type(reader)
    <class 'grdl.IO.eo.sentinel2.Sentinel2Reader'>
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        backend: str = 'auto',
    ) -> None:
        self._backend = backend
        self.jp2_reader: Optional[JP2Reader] = None
        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Load Sentinel-2 metadata from filename and JP2 file."""
        # Open with JP2Reader
        try:
            self.jp2_reader = JP2Reader(self.filepath, backend=self._backend)
        except Exception as e:
            raise ValueError(
                f"Failed to open Sentinel-2 JP2 file: {self.filepath}: {e}"
            ) from e

        # Extract base metadata from JP2Reader
        jp2_meta = self.jp2_reader.metadata
        rows = jp2_meta['rows']
        cols = jp2_meta['cols']
        dtype = jp2_meta['dtype']
        bands = jp2_meta.get('bands', 1)
        nodata = 0  # Sentinel-2 standard nodata value

        # Parse filename for Sentinel-2-specific fields
        parsed = _parse_sentinel2_filename(self.filepath)

        # Extract CRS from JP2Reader (rasterio provides this)
        crs_str = None
        if hasattr(self.jp2_reader, 'dataset') and self.jp2_reader.dataset:
            if hasattr(self.jp2_reader.dataset, 'crs') and self.jp2_reader.dataset.crs:
                crs_str = str(self.jp2_reader.dataset.crs)

        # Derive UTM zone and latitude band from MGRS tile
        utm_zone, latitude_band = None, None
        if 'mgrs_tile_id' in parsed:
            utm_zone, latitude_band = _parse_mgrs_tile(parsed['mgrs_tile_id'])

        # Band wavelength lookup
        wavelength_center, wavelength_range = None, None
        if 'band_id' in parsed and parsed['band_id'] in BAND_WAVELENGTHS:
            center, wl_min, wl_max = BAND_WAVELENGTHS[parsed['band_id']]
            wavelength_center = center
            wavelength_range = (wl_min, wl_max)

        # Format sensing datetime as ISO 8601
        sensing_dt = parsed.get('sensing_datetime')
        if sensing_dt and 'T' in sensing_dt:
            # Convert YYYYMMDDTHHMMSS -> YYYY-MM-DDTHH:MM:SS
            sensing_dt = f"{sensing_dt[:4]}-{sensing_dt[4:6]}-{sensing_dt[6:11]}:{sensing_dt[11:13]}:{sensing_dt[13:]}"

        # Collect extras from JP2
        extras = jp2_meta.get('extras', {}).copy()
        if hasattr(self.jp2_reader, 'dataset'):
            ds = self.jp2_reader.dataset
            if ds and hasattr(ds, 'transform'):
                extras['transform'] = ds.transform
            if ds and hasattr(ds, 'bounds'):
                extras['bounds'] = ds.bounds
            if ds and hasattr(ds, 'res'):
                extras['resolution'] = ds.res

        # Construct Sentinel2Metadata
        self.metadata = Sentinel2Metadata(
            format=f"Sentinel-2_{parsed.get('processing_level', 'L2A')}",
            rows=rows,
            cols=cols,
            dtype=dtype,
            bands=bands,
            crs=crs_str,
            nodata=nodata,
            extras=extras,
            satellite=parsed.get('satellite'),
            processing_level=parsed.get('processing_level'),
            product_type=parsed.get('product_type'),
            sensing_datetime=sensing_dt,
            mgrs_tile_id=parsed.get('mgrs_tile_id'),
            band_id=parsed.get('band_id'),
            resolution_tier=parsed.get('resolution_tier'),
            baseline_processing=parsed.get('baseline_processing'),
            relative_orbit=parsed.get('relative_orbit'),
            product_discriminator=parsed.get('product_discriminator'),
            utm_zone=utm_zone,
            latitude_band=latitude_band,
            wavelength_center=wavelength_center,
            wavelength_range=wavelength_range,
        )

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial chip from the Sentinel-2 JP2 file.

        Delegates to wrapped JP2Reader for efficient windowed reading.

        Parameters
        ----------
        row_start : int
            Starting row index (0-based).
        row_end : int
            Ending row index (exclusive).
        col_start : int
            Starting column index (0-based).
        col_end : int
            Ending column index (exclusive).
        bands : List[int], optional
            List of band indices to read (0-based). If None, reads all bands.

        Returns
        -------
        np.ndarray
            Array of shape (rows, cols) for single band or
            (bands, rows, cols) for multiple bands.
        """
        return self.jp2_reader.read_chip(
            row_start, row_end, col_start, col_end, bands=bands
        )

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """Read the entire Sentinel-2 image.

        Parameters
        ----------
        bands : List[int], optional
            List of band indices to read (0-based). If None, reads all bands.

        Returns
        -------
        np.ndarray
            Array of shape (rows, cols) for single band or
            (bands, rows, cols) for multiple bands.
        """
        return self.jp2_reader.read_full(bands=bands)

    def get_shape(self) -> Tuple[int, ...]:
        """Get image dimensions.

        Returns
        -------
        Tuple[int, ...]
            (rows, cols) for single band or (rows, cols, bands) for multi-band.
        """
        return self.jp2_reader.get_shape()

    def get_dtype(self) -> np.dtype:
        """Get the data type (uint16 for Sentinel-2).

        Returns
        -------
        np.dtype
            NumPy data type of the image.
        """
        return self.jp2_reader.get_dtype()

    def close(self) -> None:
        """Close the wrapped JP2Reader."""
        if self.jp2_reader is not None:
            self.jp2_reader.close()
