# -*- coding: utf-8 -*-
"""
CRSD Writer - Write compensated radar signal data in NGA CRSD format.

Wraps ``sarkit.crsd.Writer`` to provide a GRDL-compatible interface
for writing CRSD files.  Accepts a pre-built ``lxml.etree.ElementTree``
describing the CRSD XML metadata.

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
2026-03-02

Modified
--------
2026-03-02
"""

from __future__ import annotations

# Standard library
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.base import ImageWriter
from grdl.IO.models import ImageMetadata
from grdl.IO.sar._backend import require_sarkit


class CRSDWriter(ImageWriter):
    """Write compensated radar signal data in NGA CRSD format.

    Wraps ``sarkit.crsd.Writer`` and accepts a pre-built CRSD XML
    tree for metadata.  Signal data, PVP, PPP, and support arrays are
    written through dedicated methods.

    Parameters
    ----------
    filepath : str or Path
        Output path for the ``.crsd`` file.
    xmltree : lxml.etree.ElementTree
        Fully populated CRSDsar XML metadata tree.
    metadata : ImageMetadata, optional
        Optional GRDL metadata (stored but not used for writing).

    Raises
    ------
    ImportError
        If sarkit is not installed.

    Examples
    --------
    >>> from grdl.IO.sar import CRSDWriter
    >>> writer = CRSDWriter('output.crsd', xmltree=crsd_xml)
    >>> writer.write_ppp('TX1', ppp_array)
    >>> writer.write_signal('VV', signal_array)
    >>> writer.write_pvp('VV', pvp_array)
    >>> writer.close()
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        xmltree: Any = None,
        metadata: Optional[ImageMetadata] = None,
    ) -> None:
        require_sarkit('CRSD')
        super().__init__(filepath, metadata)

        import sarkit.crsd

        self._xmltree = xmltree
        self._file_handle = None
        self._writer = None

        if xmltree is not None:
            self._open_writer(xmltree)

    def _open_writer(self, xmltree: Any) -> None:
        """Open the sarkit writer with the given XML tree."""
        import sarkit.crsd

        crsd_metadata = sarkit.crsd.Metadata(xmltree=xmltree)
        self._file_handle = open(str(self.filepath), 'wb')
        self._writer = sarkit.crsd.Writer(
            self._file_handle, crsd_metadata,
        )

    def write_support_array(
        self,
        array_id: str,
        data: np.ndarray,
    ) -> None:
        """Write a support array.

        Parameters
        ----------
        array_id : str
            Support array identifier matching the XML ``SAId``.
        data : np.ndarray
            Support array data.
        """
        if self._writer is None:
            raise RuntimeError("Writer not initialized (no xmltree)")
        self._writer.write_support_array(array_id, data)

    def write_ppp(
        self,
        tx_id: str,
        ppp_array: np.ndarray,
    ) -> None:
        """Write Per-Pulse Parameter array for a transmit sequence.

        Parameters
        ----------
        tx_id : str
            Transmit sequence identifier.
        ppp_array : np.ndarray
            Structured array with PPP dtype.
        """
        if self._writer is None:
            raise RuntimeError("Writer not initialized (no xmltree)")
        self._writer.write_ppp(tx_id, ppp_array)

    def write_signal(
        self,
        channel_id: str,
        signal_array: np.ndarray,
    ) -> None:
        """Write signal data for a receive channel.

        Parameters
        ----------
        channel_id : str
            Channel identifier (e.g. ``'VV'``, ``'VH'``).
        signal_array : np.ndarray
            Complex signal data, shape ``(num_vectors, num_samples)``.
        """
        if self._writer is None:
            raise RuntimeError("Writer not initialized (no xmltree)")
        self._writer.write_signal(channel_id, signal_array)

    def write_pvp(
        self,
        channel_id: str,
        pvp_array: np.ndarray,
    ) -> None:
        """Write Per-Vector Parameter array for a receive channel.

        Parameters
        ----------
        channel_id : str
            Channel identifier.
        pvp_array : np.ndarray
            Structured array with PVP dtype.
        """
        if self._writer is None:
            raise RuntimeError("Writer not initialized (no xmltree)")
        self._writer.write_pvp(channel_id, pvp_array)

    def write(
        self,
        data: np.ndarray,
        geolocation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write signal data (ImageWriter ABC).

        For CRSD, prefer using the channel-specific ``write_signal()``,
        ``write_pvp()``, and ``write_ppp()`` methods directly.  This
        method writes to the first declared channel.

        Parameters
        ----------
        data : np.ndarray
            Complex signal data, shape ``(num_vectors, num_samples)``.
        geolocation : dict, optional
            Ignored for CRSD.
        """
        if self._xmltree is None:
            raise RuntimeError("Writer not initialized (no xmltree)")

        # Find first channel ID from XML
        ch_elem = self._xmltree.find(
            './/{*}Data/{*}Receive/{*}Channel/{*}ChId'
        )
        if ch_elem is None or ch_elem.text is None:
            raise ValueError("No channel found in CRSD XML")

        self.write_signal(ch_elem.text, data)

    def write_chip(
        self,
        data: np.ndarray,
        row_start: int,
        col_start: int,
        geolocation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Not supported for CRSD format.

        Raises
        ------
        NotImplementedError
            Always raised.
        """
        raise NotImplementedError(
            "CRSD format does not support chip-level writes."
        )

    def close(self) -> None:
        """Finalize and close the CRSD file."""
        if self._writer is not None:
            self._writer.done()
            self._writer = None
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
