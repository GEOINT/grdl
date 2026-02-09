# -*- coding: utf-8 -*-
"""
CPHD Reader - Compensated Phase History Data format.

NGA standard for phase history data (unfocused SAR). Uses sarkit as the
primary backend with sarpy as fallback.

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
2026-02-09
"""

# Standard library
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.base import ImageReader
from grdl.IO.sar._backend import (
    _HAS_SARKIT,
    _HAS_SARPY,
    require_sar_backend,
)


class CPHDReader(ImageReader):
    """Read CPHD (Compensated Phase History Data) format.

    CPHD is the NGA standard for phase history data. This reader uses
    sarkit as the primary backend with sarpy as fallback.

    Parameters
    ----------
    filepath : str or Path
        Path to the CPHD file.

    Attributes
    ----------
    filepath : Path
        Path to the CPHD file.
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
        If the file is not valid CPHD.

    Notes
    -----
    CPHD contains phase history data, not formed imagery. Use focusing
    algorithms to generate images from CPHD.

    Examples
    --------
    >>> from grdl.IO.sar import CPHDReader
    >>> with CPHDReader('data.cphd') as reader:
    ...     shape = reader.get_shape()
    ...     data = reader.read_chip(0, 100, 0, 200)
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        self.backend = require_sar_backend('CPHD')
        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Load CPHD metadata using the active backend."""
        if self.backend == 'sarkit':
            self._load_metadata_sarkit()
        else:
            self._load_metadata_sarpy()

    def _load_metadata_sarkit(self) -> None:
        """Load metadata via sarkit."""
        import sarkit.cphd

        try:
            self._file_handle = open(str(self.filepath), 'rb')
            self._reader = sarkit.cphd.Reader(self._file_handle)

            xml = self._reader.metadata.xmltree

            self.metadata = {
                'format': 'CPHD',
                'backend': 'sarkit',
            }

            # Extract channel information
            channels = {}
            for ch_elem in xml.findall('{*}Data/{*}Channel'):
                ch_id = ch_elem.findtext('{*}Identifier')
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
            core_name = xml.findtext(
                '{*}CollectionInfo/{*}CoreName'
            )
            classification = xml.findtext(
                '{*}CollectionInfo/{*}Classification'
            )
            if collector:
                self.metadata['collector_name'] = collector
            if core_name:
                self.metadata['core_name'] = core_name
            if classification:
                self.metadata['classification'] = classification

            # Store first channel ID for default operations
            self._default_channel = next(iter(channels))
            self._xmltree = xml

        except Exception as e:
            raise ValueError(f"Failed to load CPHD metadata: {e}") from e

    def _load_metadata_sarpy(self) -> None:
        """Load metadata via sarpy (fallback)."""
        from sarpy.io.phase_history.converter import open_phase_history

        try:
            self._reader = open_phase_history(str(self.filepath))
            self._sarpy_meta = self._reader.cphd_meta

            self.metadata = {
                'format': 'CPHD',
                'backend': 'sarpy',
                'num_channels': self._sarpy_meta.Data.NumCPHDChannels,
                'classification': (
                    self._sarpy_meta.CollectionInfo.Classification
                ),
                'collector_name': (
                    self._sarpy_meta.CollectionInfo.CollectorName
                ),
                'core_name': self._sarpy_meta.CollectionInfo.CoreName,
            }

            channels = {}
            for channel in self._sarpy_meta.Data.Channels:
                channels[channel.Identifier] = {
                    'num_vectors': channel.NumVectors,
                    'num_samples': channel.NumSamples,
                }
            self.metadata['channels'] = channels

        except Exception as e:
            raise ValueError(f"Failed to load CPHD metadata: {e}") from e

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read phase history vectors.

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
            Phase history data.
        """
        channel = bands[0] if bands else 0

        if self.backend == 'sarkit':
            ch_id = list(self.metadata['channels'].keys())[channel]
            return self._reader.read_signal(
                ch_id,
                start_vector=row_start,
                stop_vector=row_end,
            )[:, col_start:col_end]
        else:
            return self._reader.read_chip(
                index=channel,
                dim1_range=(row_start, row_end),
                dim2_range=(col_start, col_end),
            )

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """Read full phase history data.

        Parameters
        ----------
        bands : Optional[List[int]]
            Channel indices to read. If None, reads the first channel.

        Returns
        -------
        np.ndarray
            Phase history data for the specified channel.
        """
        channel = bands[0] if bands else 0

        if self.backend == 'sarkit':
            ch_id = list(self.metadata['channels'].keys())[channel]
            return self._reader.read_signal(ch_id)
        else:
            return self._reader.read(index=channel)

    def get_shape(self) -> Tuple[int, int]:
        """Get phase history dimensions for first channel.

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
            ``complex64`` for CPHD data.
        """
        return np.dtype('complex64')

    def get_geolocation(self) -> Optional[Dict[str, Any]]:
        """Get geolocation information.

        Returns
        -------
        Optional[Dict[str, Any]]
            Reference point and coordinate system info.
        """
        return {
            'projection': 'Phase history space',
        }

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
