# -*- coding: utf-8 -*-
"""
SIDD Reader - Sensor Independent Derived Data format.

NGA standard for derived SAR products in NITF containers. Uses sarkit
as the backend (no sarpy fallback).

Dependencies
------------
sarkit

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

# GRDL internal
from grdl.IO.base import ImageReader
from grdl.IO.sar._backend import require_sarkit


class SIDDReader(ImageReader):
    """Read SIDD (Sensor Independent Derived Data) format.

    SIDD is the NGA standard for derived SAR products (processed
    imagery) in NITF containers. A single file may contain multiple
    product images. Requires sarkit (no sarpy fallback).

    Parameters
    ----------
    filepath : str or Path
        Path to the SIDD file (NITF container).
    image_index : int, optional
        Index of the product image to read (default 0). SIDD files
        can contain multiple product images.

    Attributes
    ----------
    filepath : Path
        Path to the SIDD file.
    metadata : Dict[str, Any]
        Standardized metadata dictionary.
    image_index : int
        Active product image index.

    Raises
    ------
    ImportError
        If sarkit is not installed.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not valid SIDD.

    Examples
    --------
    >>> from grdl.IO.sar import SIDDReader
    >>> with SIDDReader('derived.nitf') as reader:
    ...     chip = reader.read_chip(0, 512, 0, 512)
    ...     print(reader.metadata['num_images'])
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        image_index: int = 0,
    ) -> None:
        require_sarkit('SIDD')
        self.image_index = image_index
        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Load SIDD metadata using sarkit."""
        import sarkit.sidd

        try:
            self._file_handle = open(str(self.filepath), 'rb')
            self._reader = sarkit.sidd.NitfReader(self._file_handle)

            num_images = len(self._reader.metadata.images)

            if self.image_index >= num_images:
                raise ValueError(
                    f"image_index {self.image_index} out of range, "
                    f"file contains {num_images} product image(s)"
                )

            img_meta = self._reader.metadata.images[self.image_index]
            xml = img_meta.xmltree

            # Extract dimensions
            num_rows = xml.findtext(
                '{*}Measurement/{*}PixelFootprint/{*}Row'
            )
            num_cols = xml.findtext(
                '{*}Measurement/{*}PixelFootprint/{*}Col'
            )
            pixel_type = xml.findtext('{*}Display/{*}PixelType')

            # Map pixel type to numpy dtype
            dtype_map = {
                'MONO8I': 'uint8',
                'MONO8LU': 'uint8',
                'MONO16I': 'uint16',
                'RGB24I': 'uint8',
                'RGB8LU': 'uint8',
            }
            dtype_str = dtype_map.get(pixel_type, 'uint8')

            self.metadata = {
                'format': 'SIDD',
                'backend': 'sarkit',
                'rows': int(num_rows) if num_rows else 0,
                'cols': int(num_cols) if num_cols else 0,
                'dtype': dtype_str,
                'pixel_type': pixel_type,
                'num_images': num_images,
                'image_index': self.image_index,
            }

            # Classification
            classification = xml.findtext(
                '{*}ProductCreation/{*}Classification/{*}SecurityClassification'
            )
            if classification:
                self.metadata['classification'] = classification

            self._xmltree = xml

        except Exception as e:
            raise ValueError(f"Failed to load SIDD metadata: {e}") from e

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial chip from the SIDD product image.

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
            Ignored for SIDD.

        Returns
        -------
        np.ndarray
            Image chip data.

        Raises
        ------
        ValueError
            If indices are out of bounds.
        """
        if row_start < 0 or col_start < 0:
            raise ValueError("Start indices must be non-negative")
        if row_end > self.metadata['rows'] or col_end > self.metadata['cols']:
            raise ValueError("End indices exceed image dimensions")

        data, _ = self._reader.read_image_sub_image(
            self.image_index,
            start_row=row_start,
            start_col=col_start,
            stop_row=row_end,
            stop_col=col_end,
        )
        return data

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """Read the full SIDD product image.

        Parameters
        ----------
        bands : Optional[List[int]]
            Ignored for SIDD.

        Returns
        -------
        np.ndarray
            Full product image data.
        """
        return self._reader.read_image(self.image_index)

    def get_shape(self) -> Tuple[int, int]:
        """Get image dimensions.

        Returns
        -------
        Tuple[int, int]
            ``(rows, cols)``.
        """
        return (self.metadata['rows'], self.metadata['cols'])

    def get_dtype(self) -> np.dtype:
        """Get data type.

        Returns
        -------
        np.dtype
            Data type of the product image.
        """
        return np.dtype(self.metadata['dtype'])

    def get_geolocation(self) -> Optional[Dict[str, Any]]:
        """Get geolocation information.

        Returns
        -------
        Optional[Dict[str, Any]]
            Measurement pixel footprint and projection info.
        """
        return {
            'projection': 'SIDD derived product',
        }

    def close(self) -> None:
        """Close the reader and release resources."""
        if hasattr(self, '_reader') and self._reader is not None:
            self._reader.done()
        if hasattr(self, '_file_handle') and self._file_handle is not None:
            self._file_handle.close()
