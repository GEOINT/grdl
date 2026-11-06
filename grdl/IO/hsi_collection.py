# -*- coding: utf-8 -*-
"""
HSI Collection Reader - Reader for multi-file single-band hyperspectral directories.

Aggregates directories of individual single-band GeoTIFFs (such as USGS NASA
EO-1 Hyperion or similar tiled HSI collections) into a unified virtual
hyperspectral cube implementing the ``ImageReader`` interface.

Dependencies
------------
rasterio

Author
------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-11-06
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

from grdl.exceptions import DependencyError
from grdl.IO.base import ImageReader
from grdl.IO.models import ChannelMetadata, ImageMetadata
from grdl.IO.geotiff import GeoTIFFReader

logger = logging.getLogger(__name__)


def _is_hsi_collection_dir(path: Path) -> bool:
    """Check if directory contains a single-band HSI image collection."""
    if not path.is_dir():
        return False
    count = 0
    for p in path.iterdir():
        if p.is_file() and p.suffix.lower() in ('.tif', '.tiff'):
            if re.search(r'(_B\d+|_band_?\d+)', p.name, re.IGNORECASE):
                count += 1
                if count >= 3:
                    return True
    return False


class HSICollectionReader(ImageReader):
    """Virtual cube reader for multi-file single-band HSI collections.

    Parameters
    ----------
    filepath : Union[str, Path]
        Directory containing the band files or path to metadata file (e.g., MTL.TXT).
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        if not _HAS_RASTERIO:
            raise DependencyError("rasterio is required to read HSI GeoTIFF collections.")

        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")

        if p.is_file():
            self.directory = p.parent
        else:
            self.directory = p

        super().__init__(self.directory)
        self._band_files: List[Path] = []
        self._readers: Dict[int, GeoTIFFReader] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Discover and sort band files, loading spatial metadata and MTL if present."""
        # Check for band GeoTIFF files (case-insensitive deduplicated)
        seen = set()
        tifs = []
        for p in self.directory.iterdir():
            if p.is_file() and p.suffix.lower() in ('.tif', '.tiff'):
                if re.search(r'(_B\d+|_band_?\d+)', p.name, re.IGNORECASE):
                    key = p.name.lower()
                    if key not in seen:
                        seen.add(key)
                        tifs.append(p)

        def _extract_band_num(f: Path) -> int:
            m = re.search(r'_B(\d+)', f.name, re.IGNORECASE) or re.search(r'_band_?(\d+)', f.name, re.IGNORECASE)
            return int(m.group(1)) if m else 999999

        tifs.sort(key=_extract_band_num)
        if not tifs:
            raise ValueError(f"No multi-band GeoTIFF files found in {self.directory}")

        self._band_files = tifs

        # Open reference band to obtain geometry
        ref_reader = GeoTIFFReader(self._band_files[0])
        ref_meta = ref_reader.metadata
        ref_shape = ref_reader.get_shape()

        # Parse MTL file if present
        mtl_files = list(self.directory.glob("*_MTL*.TXT")) + list(self.directory.glob("*_MTL*.txt"))
        mtl_tags: Dict[str, str] = {}
        if mtl_files:
            try:
                with open(mtl_files[0], 'r', encoding='utf-8', errors='ignore') as mf:
                    for line in mf:
                        if '=' in line:
                            k, v = line.strip().split('=', 1)
                            mtl_tags[k.strip()] = v.strip().strip('"')
            except Exception as e:
                logger.debug("Failed parsing MTL file: %s", e)

        # Build ChannelMetadata for each band
        channel_metadata: List[ChannelMetadata] = []
        for idx, bf in enumerate(self._band_files):
            b_num = _extract_band_num(bf)
            ch_extras: Dict[str, Any] = {'file_name': bf.name, 'band_number': b_num}
            channel_metadata.append(
                ChannelMetadata(
                    index=idx,
                    name=f"band_{b_num}",
                    role='measurement',
                    extras=ch_extras,
                )
            )

        self.metadata = ImageMetadata(
            format='GeoTIFF_Collection',
            rows=ref_meta.rows,
            cols=ref_meta.cols,
            bands=len(self._band_files),
            dtype=str(ref_meta.dtype),
            axis_order='CYX',
            crs=ref_meta.crs,
            nodata=ref_meta.nodata,
            transform=ref_meta.transform,
            bounds=ref_meta.bounds,
            pixel_resolution=ref_meta.pixel_resolution,
            channel_metadata=channel_metadata,
            extras=mtl_tags,
        )
        ref_reader.close()

    def _get_band_reader(self, band_idx: int) -> GeoTIFFReader:
        """Lazy loader for individual band readers."""
        if band_idx not in self._readers:
            if band_idx < 0 or band_idx >= len(self._band_files):
                raise IndexError(f"Band index {band_idx} out of range [0, {len(self._band_files)-1}]")
            self._readers[band_idx] = GeoTIFFReader(self._band_files[band_idx])
        return self._readers[band_idx]

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial window across selected or all bands."""
        if bands is None:
            target_bands = list(range(len(self._band_files)))
        else:
            target_bands = [b if b < len(self._band_files) else b - 1 for b in bands]

        slices = []
        for b in target_bands:
            r = self._get_band_reader(b)
            chip = r.read_chip(row_start, row_end, col_start, col_end)
            slices.append(chip)

        out = np.stack(slices, axis=0)
        return out[0] if len(target_bands) == 1 else out

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """Read full raster volume across selected bands."""
        return self.read_chip(0, self.metadata.rows, 0, self.metadata.cols, bands=bands)

    def read_band(self, band: int) -> np.ndarray:
        """Read a single 2D band."""
        r = self._get_band_reader(band)
        return r.read_full()

    def get_shape(self) -> Tuple[int, int, int]:
        return (self.metadata.rows, self.metadata.cols, len(self._band_files))

    def get_dtype(self) -> np.dtype:
        return np.dtype(self.metadata.dtype)

    def close(self) -> None:
        """Close all opened band readers."""
        for r in self._readers.values():
            try:
                r.close()
            except Exception:
                pass
        self._readers.clear()
