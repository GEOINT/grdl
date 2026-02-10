# -*- coding: utf-8 -*-
"""
GeoTIFF Reader - Read GeoTIFF and Cloud-Optimized GeoTIFF imagery.

Base data format reader for any GeoTIFF file regardless of modality
(EO, SAR GRD, MSI, etc.). Lives at the IO level so modality submodules
can use it without cross-submodule dependencies.

Dependencies
------------
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
2026-02-09

Modified
--------
2026-02-09
"""

# Standard library
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party
import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

# GRDL internal
from grdl.IO.base import ImageReader


class GeoTIFFReader(ImageReader):
    """Read GeoTIFF and Cloud-Optimized GeoTIFF imagery.

    Reads any GeoTIFF file including SAR GRD products, standard EO
    imagery, COGs, and multi-band rasters. Uses rasterio (GDAL) as
    the backend.

    Parameters
    ----------
    filepath : str or Path
        Path to the GeoTIFF file.

    Attributes
    ----------
    filepath : Path
        Path to the image file.
    metadata : Dict[str, Any]
        Standardized metadata dictionary.
    dataset : rasterio.DatasetReader
        Rasterio dataset object for direct access.

    Raises
    ------
    ImportError
        If rasterio is not installed.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be opened as a GeoTIFF.

    Examples
    --------
    >>> from grdl.IO.geotiff import GeoTIFFReader
    >>> with GeoTIFFReader('image.tif') as reader:
    ...     chip = reader.read_chip(0, 1024, 0, 1024)
    ...     print(reader.metadata['crs'])
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        if not _HAS_RASTERIO:
            raise ImportError(
                "rasterio is required for GeoTIFF reading. "
                "Install with: pip install rasterio"
            )
        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Load GeoTIFF metadata using rasterio."""
        try:
            self.dataset = rasterio.open(str(self.filepath))

            self.metadata = {
                'format': 'GeoTIFF',
                'rows': self.dataset.height,
                'cols': self.dataset.width,
                'bands': self.dataset.count,
                'dtype': str(self.dataset.dtypes[0]),
                'crs': str(self.dataset.crs),
                'transform': self.dataset.transform,
                'bounds': self.dataset.bounds,
                'resolution': self.dataset.res,
                'nodata': self.dataset.nodata,
            }

            if 'TIFFTAG_IMAGEDESCRIPTION' in self.dataset.tags():
                self.metadata['description'] = (
                    self.dataset.tags()['TIFFTAG_IMAGEDESCRIPTION']
                )

        except Exception as e:
            raise ValueError(f"Failed to load GeoTIFF metadata: {e}") from e

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial chip from the GeoTIFF.

        Parameters
        ----------
        row_start : int
            Starting row index (inclusive).
        row_end : int
            Ending row index (exclusive).
        col_start : int
            Starting column index (inclusive).
        col_end : int
            Ending column index (exclusive).
        bands : Optional[List[int]]
            Band indices to read (0-based). If None, read all bands.

        Returns
        -------
        np.ndarray
            Image chip with shape ``(rows, cols)`` for single band or
            ``(bands, rows, cols)`` for multi-band.

        Raises
        ------
        ValueError
            If indices are out of bounds.
        """
        if row_start < 0 or col_start < 0:
            raise ValueError("Start indices must be non-negative")
        if row_end > self.metadata['rows'] or col_end > self.metadata['cols']:
            raise ValueError("End indices exceed image dimensions")

        window = Window(
            col_start, row_start,
            col_end - col_start, row_end - row_start,
        )

        if bands is None:
            data = self.dataset.read(window=window)
        else:
            data = self.dataset.read([b + 1 for b in bands], window=window)

        if data.shape[0] == 1:
            return data[0]
        return data

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """Read the entire GeoTIFF image.

        Parameters
        ----------
        bands : Optional[List[int]]
            Band indices to read (0-based). If None, read all bands.

        Returns
        -------
        np.ndarray
            Full image data.
        """
        if bands is None:
            data = self.dataset.read()
        else:
            data = self.dataset.read([b + 1 for b in bands])

        if data.shape[0] == 1:
            return data[0]
        return data

    def get_shape(self) -> Tuple[int, ...]:
        """Get image dimensions.

        Returns
        -------
        Tuple[int, ...]
            ``(rows, cols)`` for single band or
            ``(rows, cols, bands)`` for multi-band.
        """
        if self.metadata['bands'] == 1:
            return (self.metadata['rows'], self.metadata['cols'])
        return (
            self.metadata['rows'],
            self.metadata['cols'],
            self.metadata['bands'],
        )

    def get_dtype(self) -> np.dtype:
        """Get the data type of the image.

        Returns
        -------
        np.dtype
        """
        return np.dtype(self.metadata['dtype'])

    def get_geolocation(self) -> Optional[Dict[str, Any]]:
        """Get geolocation information.

        Returns
        -------
        Dict[str, Any]
            CRS, affine transform, geographic bounds, and resolution.
        """
        return {
            'crs': self.metadata['crs'],
            'transform': self.metadata['transform'],
            'bounds': self.metadata['bounds'],
            'resolution': self.metadata['resolution'],
        }

    def close(self) -> None:
        """Close the rasterio dataset."""
        if hasattr(self, 'dataset') and self.dataset is not None:
            self.dataset.close()
