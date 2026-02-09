# -*- coding: utf-8 -*-
"""
CRSD Reader - Compensated Radar Signal Data format.

NGA standard for compensated radar signal data. Uses sarkit as the
backend (no sarpy fallback — sarpy does not fully support CRSD).

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


class CRSDReader(ImageReader):
    """Read CRSD (Compensated Radar Signal Data) format.

    CRSD is the NGA standard for compensated radar signal data,
    providing a standardized format for bistatic and multistatic SAR
    collections. Requires sarkit (no sarpy fallback).

    Parameters
    ----------
    filepath : str or Path
        Path to the CRSD file.

    Attributes
    ----------
    filepath : Path
        Path to the CRSD file.
    metadata : Dict[str, Any]
        Standardized metadata dictionary.

    Raises
    ------
    ImportError
        If sarkit is not installed.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not valid CRSD.

    Examples
    --------
    >>> from grdl.IO.sar import CRSDReader
    >>> with CRSDReader('data.crsd') as reader:
    ...     shape = reader.get_shape()
    ...     signal = reader.read_chip(0, 100, 0, 200)
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        require_sarkit('CRSD')
        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Load CRSD metadata using sarkit."""
        import sarkit.crsd

        try:
            self._file_handle = open(str(self.filepath), 'rb')
            self._reader = sarkit.crsd.Reader(self._file_handle)

            xml = self._reader.metadata.xmltree

            self.metadata = {
                'format': 'CRSD',
                'backend': 'sarkit',
            }

            # Extract channel information from Receive element
            channels = {}
            for ch_elem in xml.findall(
                '{*}Data/{*}Receive/{*}Channel'
            ):
                ch_id = ch_elem.findtext('{*}ChId')
                num_vectors = int(ch_elem.findtext('{*}NumVectors'))
                num_samples = int(ch_elem.findtext('{*}NumSamples'))
                channels[ch_id] = {
                    'num_vectors': num_vectors,
                    'num_samples': num_samples,
                }

            self.metadata['channels'] = channels
            self.metadata['num_channels'] = len(channels)

            # Collection info
            collector = xml.findtext(
                '{*}CollectionInfo/{*}CollectorName'
            )
            classification = xml.findtext(
                '{*}CollectionInfo/{*}Classification'
            )
            if collector:
                self.metadata['collector_name'] = collector
            if classification:
                self.metadata['classification'] = classification

            self._xmltree = xml

        except Exception as e:
            raise ValueError(f"Failed to load CRSD metadata: {e}") from e

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read radar signal data.

        Parameters
        ----------
        row_start : int
            Starting vector index.
        row_end : int
            Ending vector index.
        col_start : int
            Starting sample index.
        col_end : int
            Ending sample index.
        bands : Optional[List[int]]
            Channel indices to read. If None, reads the first channel.

        Returns
        -------
        np.ndarray
            Signal data.
        """
        channel = bands[0] if bands else 0
        ch_id = list(self.metadata['channels'].keys())[channel]
        data = self._reader.read_signal(ch_id)
        return data[row_start:row_end, col_start:col_end]

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """Read full signal data for a channel.

        Parameters
        ----------
        bands : Optional[List[int]]
            Channel indices to read. If None, reads the first channel.

        Returns
        -------
        np.ndarray
            Signal data for the specified channel.
        """
        channel = bands[0] if bands else 0
        ch_id = list(self.metadata['channels'].keys())[channel]
        return self._reader.read_signal(ch_id)

    def get_shape(self) -> Tuple[int, int]:
        """Get signal dimensions for first channel.

        Returns
        -------
        Tuple[int, int]
            ``(num_vectors, num_samples)`` for the first channel.
        """
        first_channel = list(self.metadata['channels'].values())[0]
        return (first_channel['num_vectors'], first_channel['num_samples'])

    def get_dtype(self) -> np.dtype:
        """Get data type.

        Returns
        -------
        np.dtype
            ``complex64`` for CRSD signal data.
        """
        return np.dtype('complex64')

    def get_geolocation(self) -> Optional[Dict[str, Any]]:
        """Get geolocation information.

        Returns
        -------
        Optional[Dict[str, Any]]
            Reference geometry information.
        """
        return {
            'projection': 'Radar signal space',
        }

    def close(self) -> None:
        """Close the reader and release resources."""
        if hasattr(self, '_reader') and self._reader is not None:
            self._reader.done()
        if hasattr(self, '_file_handle') and self._file_handle is not None:
            self._file_handle.close()
