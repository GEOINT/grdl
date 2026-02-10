# -*- coding: utf-8 -*-
"""
SICD Reader - Sensor Independent Complex Data format.

NGA standard for complex SAR imagery. Uses sarkit as the primary backend
with sarpy as fallback.

Dependencies
------------
sarkit (primary) or sarpy (fallback)

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
2026-02-10
"""

# Standard library
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.base import ImageReader
from grdl.IO.models import ImageMetadata
from grdl.IO.sar._backend import (
    _HAS_SARKIT,
    _HAS_SARPY,
    require_sar_backend,
)


class SICDReader(ImageReader):
    """Read SICD (Sensor Independent Complex Data) format.

    SICD is the NGA standard for complex SAR imagery in NITF containers.
    This reader uses sarkit as the primary backend with sarpy as fallback.

    Parameters
    ----------
    filepath : str or Path
        Path to the SICD file (NITF or other SICD container).

    Attributes
    ----------
    filepath : Path
        Path to the SICD file.
    metadata : Dict[str, Any]
        Standardized metadata dictionary.
    backend : str
        Active backend (``'sarkit'`` or ``'sarpy'``).

    Raises
    ------
    ImportError
        If neither sarkit nor sarpy is installed.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not a valid SICD file.

    Examples
    --------
    >>> from grdl.IO.sar import SICDReader
    >>> with SICDReader('image.nitf') as reader:
    ...     chip = reader.read_chip(0, 1000, 0, 1000)
    ...     magnitude = np.abs(chip)

    Notes
    -----
    SICD data is complex-valued (I/Q). Use ``abs()`` for magnitude images.
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        self.backend = require_sar_backend('SICD')
        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Load SICD metadata using the active backend."""
        if self.backend == 'sarkit':
            self._load_metadata_sarkit()
        else:
            self._load_metadata_sarpy()

    def _load_metadata_sarkit(self) -> None:
        """Load metadata via sarkit."""
        import sarkit.sicd

        try:
            self._file_handle = open(str(self.filepath), 'rb')
            self._reader = sarkit.sicd.NitfReader(self._file_handle)

            xml = self._reader.metadata.xmltree
            num_rows = int(xml.findtext('{*}ImageData/{*}NumRows'))
            num_cols = int(xml.findtext('{*}ImageData/{*}NumCols'))
            pixel_type = xml.findtext('{*}ImageData/{*}PixelType')

            extras: Dict[str, Any] = {
                'pixel_type': pixel_type,
                'backend': 'sarkit',
            }

            # Collection info
            collector = xml.findtext(
                '{*}CollectionInfo/{*}CollectorName'
            )
            core_name = xml.findtext(
                '{*}CollectionInfo/{*}CoreName'
            )
            classification = xml.findtext(
                '{*}CollectionInfo/{*}Classification'
            )
            if collector:
                extras['collector_name'] = collector
            if core_name:
                extras['core_name'] = core_name
            if classification:
                extras['classification'] = classification

            # Timeline
            collect_start = xml.findtext(
                '{*}Timeline/{*}CollectStart'
            )
            collect_duration = xml.findtext(
                '{*}Timeline/{*}CollectDuration'
            )
            if collect_start:
                extras['collect_start'] = collect_start
            if collect_duration:
                extras['collect_duration'] = float(collect_duration)

            # Geolocation
            scp_lat = xml.findtext(
                '{*}GeoData/{*}SCP/{*}LLH/{*}Lat'
            )
            scp_lon = xml.findtext(
                '{*}GeoData/{*}SCP/{*}LLH/{*}Lon'
            )
            scp_hae = xml.findtext(
                '{*}GeoData/{*}SCP/{*}LLH/{*}HAE'
            )
            if scp_lat is not None:
                extras['scp_llh'] = [
                    float(scp_lat), float(scp_lon), float(scp_hae),
                ]

            self.metadata = ImageMetadata(
                format='SICD',
                rows=num_rows,
                cols=num_cols,
                dtype='complex64',
                extras=extras,
            )

            # Store raw XML tree for advanced users
            self._xmltree = xml

        except Exception as e:
            raise ValueError(f"Failed to load SICD metadata: {e}") from e

    def _load_metadata_sarpy(self) -> None:
        """Load metadata via sarpy (fallback)."""
        from sarpy.io.complex.converter import open_complex

        try:
            self._reader = open_complex(str(self.filepath))
            self._sarpy_meta = self._reader.sicd_meta

            extras: Dict[str, Any] = {
                'backend': 'sarpy',
                'collector_name': (
                    self._sarpy_meta.CollectionInfo.CollectorName
                ),
                'core_name': self._sarpy_meta.CollectionInfo.CoreName,
                'classification': (
                    self._sarpy_meta.CollectionInfo.Classification
                ),
                'collect_start': str(
                    self._sarpy_meta.Timeline.CollectStart
                ),
                'collect_duration': (
                    self._sarpy_meta.Timeline.CollectDuration
                ),
                'pixel_type': self._sarpy_meta.ImageData.PixelType,
            }

            if (hasattr(self._sarpy_meta, 'GeoData')
                    and self._sarpy_meta.GeoData is not None):
                extras['scp_llh'] = (
                    self._sarpy_meta.GeoData.SCP.LLH.get_array()
                )
                if self._sarpy_meta.GeoData.ImageCorners is not None:
                    corners = self._sarpy_meta.GeoData.ImageCorners
                    extras['corner_coords'] = {
                        'icp1': corners.ICP1.get_array(),
                        'icp2': corners.ICP2.get_array(),
                        'icp3': corners.ICP3.get_array(),
                        'icp4': corners.ICP4.get_array(),
                    }

            self.metadata = ImageMetadata(
                format='SICD',
                rows=self._sarpy_meta.ImageData.NumRows,
                cols=self._sarpy_meta.ImageData.NumCols,
                dtype='complex64',
                extras=extras,
            )

        except Exception as e:
            raise ValueError(f"Failed to load SICD metadata: {e}") from e

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial chip from the SICD image.

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
            Ignored for SICD (single complex band).

        Returns
        -------
        np.ndarray
            Complex-valued image chip with shape ``(rows, cols)``.

        Raises
        ------
        ValueError
            If indices are out of bounds.
        """
        if row_start < 0 or col_start < 0:
            raise ValueError("Start indices must be non-negative")
        if row_end > self.metadata['rows'] or col_end > self.metadata['cols']:
            raise ValueError("End indices exceed image dimensions")

        if self.backend == 'sarkit':
            data, _ = self._reader.read_sub_image(
                row_start, col_start, row_end, col_end,
            )
            return data
        else:
            return self._reader[row_start:row_end, col_start:col_end]

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """Read the entire SICD image.

        Parameters
        ----------
        bands : Optional[List[int]]
            Ignored for SICD (single complex band).

        Returns
        -------
        np.ndarray
            Full complex-valued image.

        Warnings
        --------
        SICD files can be very large. Consider using ``read_chip()``
        instead.
        """
        if self.backend == 'sarkit':
            return self._reader.read_image()
        else:
            return self._reader[:, :]

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
            ``complex64`` for SICD data.
        """
        return np.dtype('complex64')

    def close(self) -> None:
        """Close the reader and release resources."""
        if self.backend == 'sarkit':
            if hasattr(self, '_reader') and self._reader is not None:
                self._reader.done()
            if hasattr(self, '_file_handle') and self._file_handle is not None:
                self._file_handle.close()
        else:
            if hasattr(self, '_reader') and self._reader is not None:
                self._reader.close()
