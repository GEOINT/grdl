# -*- coding: utf-8 -*-
"""
SICD Collection Reader - Multi-polarization SICD reader producing CYX cubes.

Wraps multiple single-polarization ``SICDReader`` instances (one per SICD
file) and presents them as a single ``ImageReader`` whose ``read_chip()``
and ``read_full()`` return a ``(C, rows, cols)`` channel-first complex
cube, with ``metadata.axis_order='CYX'`` and ``metadata.channel_metadata``
populated with per-channel polarization labels.

This normalises SICD multi-polarization collections — where each
polarization is stored in a separate NITF/SICD file — to the same
interface already used by ``NISARReader`` (``polarizations='all'``),
``TerraSARReader`` (``polarizations='all'``), and ``BIOMASSL1Reader``.

Usage
-----
::

    from grdl.IO.sar import open_sicd_collection

    with open_sicd_collection(['hh.nitf', 'hv.nitf', 'vv.nitf']) as reader:
        print(reader.metadata.axis_order)          # 'CYX'
        print(reader.metadata.bands)               # 3
        pols = reader.get_available_polarizations()  # ['HH', 'HV', 'VV']
        cube = reader.read_chip(0, 1024, 0, 1024)  # (3, 1024, 1024)

    # Downstream geolocation — each channel has its own SICD geometry:
    from grdl.geolocation.sar import SICDGeolocation
    geo = SICDGeolocation.from_reader(reader.get_reader_for('HH'))

Author
------
Jason Fritz
43161141+stryder-vtx@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-05-26

Modified
--------
2026-05-26
"""

# Standard library
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.base import ImageReader
from grdl.IO.models.base import ChannelMetadata
from grdl.IO.models.sicd import SICDCollectionMetadata, SICDMetadata
from grdl.IO.sar.sicd import SICDReader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Polarization inference
# ---------------------------------------------------------------------------

# Known compound tx_rcv_polarization values that cannot be collapsed to a
# two-letter code.
_PASSTHROUGH_POLS = frozenset({'OTHER', 'UNKNOWN', 'SEQUENCE'})


def _infer_polarization(sicd_meta: SICDMetadata) -> str:
    """Extract a two-letter polarization label from SICD metadata.

    Inspects ``radar_collection.rcv_channels[0].tx_rcv_polarization``
    first, then falls back to ``radar_collection.tx_polarization``.
    Returns ``'UNKNOWN'`` with a warning if neither is present.

    The NGA SICD standard encodes combined Tx/Rcv polarization in the
    form ``'Tx:Rcv'`` (e.g. ``'V:V'``, ``'H:V'``).  This helper strips
    the colon to produce the conventional two-letter codes used across
    GRDL (``'VV'``, ``'HV'``, etc.).

    Parameters
    ----------
    sicd_meta : SICDMetadata
        Metadata from a single ``SICDReader`` instance.

    Returns
    -------
    str
        Two-letter polarization code (``'HH'``, ``'HV'``, ``'VH'``,
        ``'VV'``, etc.) or the original value if it does not contain
        a colon separator (e.g. ``'OTHER'``).
    """
    # ---- Primary: rcv_channels -----------------------------------------
    rc = getattr(sicd_meta, 'radar_collection', None)
    rcv_channels = getattr(rc, 'rcv_channels', None) if rc else None

    if rcv_channels:
        if len(rcv_channels) > 1:
            logger.warning(
                "SICD file has %d rcv_channels; using channel index 0 for "
                "polarization inference.  Pass explicit polarization= labels "
                "to open_sicd_collection() to override.",
                len(rcv_channels),
            )
        pol_str = getattr(rcv_channels[0], 'tx_rcv_polarization', None)
        if pol_str and pol_str.upper() not in _PASSTHROUGH_POLS:
            if ':' in pol_str:
                # 'V:V' → 'VV', 'H:V' → 'HV', etc.
                parts = pol_str.split(':', maxsplit=1)
                return (parts[0] + parts[1]).upper()
            return pol_str.upper()

    # ---- Fallback: tx_polarization (single char, e.g. 'V') ------------
    tx_pol = getattr(rc, 'tx_polarization', None) if rc else None
    if tx_pol and tx_pol.upper() not in _PASSTHROUGH_POLS:
        logger.warning(
            "Could not determine full Tx:Rcv polarization; using "
            "tx_polarization=%r as label.  Accuracy not guaranteed.",
            tx_pol,
        )
        return tx_pol.upper()

    logger.warning(
        "Polarization cannot be determined from SICD metadata "
        "(radar_collection.rcv_channels and tx_polarization are absent "
        "or non-informative).  Labelling this channel 'UNKNOWN'.  "
        "Pass explicit polarization= labels to open_sicd_collection() "
        "to avoid this.",
    )
    return 'UNKNOWN'


# ---------------------------------------------------------------------------
# Collection reader
# ---------------------------------------------------------------------------

class SICDCollectionReader(ImageReader):
    """Multi-polarization SICD reader producing a CYX complex cube.

    Wraps N ``SICDReader`` instances — one per polarization channel — and
    presents them as a single reader whose ``read_chip()`` and
    ``read_full()`` return a ``(C, rows, cols)`` stacked NumPy array with
    ``dtype=complex64``.  Metadata is an ``SICDCollectionMetadata`` with
    ``axis_order='CYX'``, ``bands=N``, and a ``channel_metadata`` list
    carrying the per-channel polarization labels.

    All SICD files in the collection **must share the same image
    dimensions** (rows × cols).  A ``ValueError`` is raised at
    construction time if the dimensions differ.

    Parameters
    ----------
    filepaths : list of str or Path
        Ordered list of SICD file paths, one per polarization channel.
        The channel order in the output cube follows this list.
    polarizations : list of str, optional
        Explicit polarization labels (e.g. ``['HH', 'VV']``).  If
        supplied, ``len(polarizations)`` must equal ``len(filepaths)``
        and each label is used verbatim.  If ``None`` (default), the
        label for each file is inferred from its SICD metadata via
        :func:`_infer_polarization`.

    Raises
    ------
    ValueError
        If *filepaths* is empty, if an explicit *polarizations* list has
        the wrong length, or if the image dimensions are inconsistent
        across files.
    FileNotFoundError
        If any path in *filepaths* does not exist.

    Examples
    --------
    >>> reader = SICDCollectionReader(['hh.nitf', 'hv.nitf', 'vv.nitf'])
    >>> reader.metadata.axis_order
    'CYX'
    >>> reader.metadata.bands
    3
    >>> reader.get_available_polarizations()
    ['HH', 'HV', 'VV']
    >>> cube = reader.read_chip(0, 512, 0, 512)  # (3, 512, 512) complex64
    """

    def __init__(
        self,
        filepaths: List[Union[str, Path]],
        polarizations: Optional[List[str]] = None,
    ) -> None:
        if not filepaths:
            raise ValueError("filepaths must contain at least one path.")
        if polarizations is not None and len(polarizations) != len(filepaths):
            raise ValueError(
                f"polarizations length ({len(polarizations)}) must match "
                f"filepaths length ({len(filepaths)})."
            )
        self._filepaths = [Path(p) for p in filepaths]
        self._polarization_override = (
            [p.upper() for p in polarizations] if polarizations else None
        )
        self._readers: List[SICDReader] = []
        # ImageReader.__init__ calls _load_metadata(); pass a sentinel so the
        # parent path-existence check is satisfied by the first filepath.
        super().__init__(self._filepaths[0])

    def _load_metadata(self) -> None:
        """Open all sub-readers and build collection metadata."""
        # Open one SICDReader per filepath
        for path in self._filepaths:
            self._readers.append(SICDReader(path))

        # Validate consistent dimensions
        rows0 = self._readers[0].metadata.rows
        cols0 = self._readers[0].metadata.cols
        for i, reader in enumerate(self._readers[1:], start=1):
            r, c = reader.metadata.rows, reader.metadata.cols
            if r != rows0 or c != cols0:
                self.close()
                raise ValueError(
                    f"Image dimensions of file {i} ({r} × {c}) do not match "
                    f"file 0 ({rows0} × {cols0}).  All SICD files in a "
                    f"collection must have the same rows and cols."
                )

        # Resolve polarization labels
        if self._polarization_override is not None:
            pols = self._polarization_override
        else:
            pols = [
                _infer_polarization(r.metadata) for r in self._readers
            ]

        # Build per-channel metadata (mirrors NISARReader/TerraSARReader)
        channel_metadata = [
            ChannelMetadata(
                index=i,
                name=pol,
                role='measurement',
                polarization=pol,
                source_indices=[i],
            )
            for i, pol in enumerate(pols)
        ]

        self.metadata = SICDCollectionMetadata(
            format='SICD_COLLECTION',
            rows=rows0,
            cols=cols0,
            dtype='complex64',
            bands=len(self._readers),
            axis_order='CYX',
            channel_metadata=channel_metadata,
            per_file_metadata=[r.metadata for r in self._readers],
        )

        logger.info(
            "SICDCollectionReader: %d channels %s, %d × %d px",
            len(self._readers),
            pols,
            rows0,
            cols0,
        )

    # -----------------------------------------------------------------------
    # ImageReader interface
    # -----------------------------------------------------------------------

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial chip from all (or selected) polarization channels.

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
        bands : list of int, optional
            0-based channel indices to read.  If ``None``, all channels
            are read.

        Returns
        -------
        np.ndarray
            Complex-valued array with shape ``(C, rows, cols)``,
            ``dtype=complex64``.  ``C`` equals the number of channels
            requested.

        Raises
        ------
        ValueError
            If row/col indices are out of bounds or *bands* contains an
            invalid channel index.
        """
        if row_start < 0 or col_start < 0:
            raise ValueError("Start indices must be non-negative.")
        if row_end > self.metadata.rows or col_end > self.metadata.cols:
            raise ValueError(
                f"End indices ({row_end}, {col_end}) exceed image "
                f"dimensions ({self.metadata.rows}, {self.metadata.cols})."
            )

        n_channels = len(self._readers)
        if bands is None:
            indices = list(range(n_channels))
        else:
            invalid = [b for b in bands if b < 0 or b >= n_channels]
            if invalid:
                raise ValueError(
                    f"Band index/indices {invalid} out of range for collection "
                    f"with {n_channels} channel(s)."
                )
            indices = bands

        chips = [
            self._readers[i].read_chip(row_start, row_end, col_start, col_end)
            for i in indices
        ]
        return np.stack(chips, axis=0)   # (C, rows, cols)

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """Read the entire image for all (or selected) polarization channels.

        Parameters
        ----------
        bands : list of int, optional
            0-based channel indices.  If ``None``, all channels are read.

        Returns
        -------
        np.ndarray
            ``(C, rows, cols)`` complex64 array.

        Warnings
        --------
        SICD files can be very large.  Prefer ``read_chip()`` for large
        scenes.
        """
        return self.read_chip(
            0, self.metadata.rows, 0, self.metadata.cols, bands=bands,
        )

    def get_shape(self) -> Tuple[int, ...]:
        """Return spatial image dimensions.

        Returns
        -------
        tuple of int
            ``(rows, cols)`` — the spatial extent of each channel.  The
            channel count is available as ``metadata.bands``.
        """
        return (self.metadata.rows, self.metadata.cols)

    def get_dtype(self) -> np.dtype:
        """Return the data type of the imagery.

        Returns
        -------
        np.dtype
            ``complex64`` for all SICD data.
        """
        return np.dtype('complex64')

    def close(self) -> None:
        """Close all sub-readers and release file handles."""
        for reader in self._readers:
            try:
                reader.close()
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Convenience methods
    # -----------------------------------------------------------------------

    def get_available_polarizations(self) -> List[str]:
        """Return the polarization labels for each channel in order.

        Returns
        -------
        list of str
            Polarization labels (e.g. ``['HH', 'HV', 'VH', 'VV']``)
            corresponding to channel indices 0, 1, 2, 3.
        """
        return [cm.polarization for cm in self.metadata.channel_metadata]

    def get_reader_for(self, polarization: str) -> SICDReader:
        """Return the sub-reader whose channel matches *polarization*.

        Use this to obtain a single-channel ``SICDReader`` for downstream
        geolocation::

            geo = SICDGeolocation.from_reader(
                collection.get_reader_for('HH')
            )

        Parameters
        ----------
        polarization : str
            Polarization label to look up (case-insensitive).

        Returns
        -------
        SICDReader
            The ``SICDReader`` instance for that polarization channel.

        Raises
        ------
        KeyError
            If *polarization* is not present in this collection.
        """
        pol_upper = polarization.upper()
        for i, cm in enumerate(self.metadata.channel_metadata):
            if cm.polarization == pol_upper:
                return self._readers[i]
        available = self.get_available_polarizations()
        raise KeyError(
            f"Polarization {polarization!r} not found in collection.  "
            f"Available: {available}"
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def open_sicd_collection(
    filepaths: List[Union[str, Path]],
    polarizations: Optional[List[str]] = None,
) -> SICDCollectionReader:
    """Open multiple single-polarization SICD files as a CYX collection.

    Convenience wrapper around :class:`SICDCollectionReader`.  Validates
    that every path exists before constructing the reader.

    Parameters
    ----------
    filepaths : list of str or Path
        Ordered list of SICD file paths, one per polarization channel.
    polarizations : list of str, optional
        Explicit polarization labels.  If ``None``, labels are inferred
        from each file's SICD metadata.

    Returns
    -------
    SICDCollectionReader
        Opened reader.  Use as a context manager (``with`` statement) to
        guarantee file handles are released.

    Raises
    ------
    ValueError
        If *filepaths* is empty or a *polarizations* list length mismatch.
    FileNotFoundError
        If any path does not exist.

    Examples
    --------
    >>> from grdl.IO.sar import open_sicd_collection
    >>> with open_sicd_collection(
    ...     ['hh.nitf', 'hv.nitf', 'vh.nitf', 'vv.nitf'],
    ... ) as reader:
    ...     cube = reader.read_chip(0, 512, 0, 512)  # (4, 512, 512)
    ...     pols = reader.get_available_polarizations()
    """
    if not filepaths:
        raise ValueError("filepaths must contain at least one path.")
    missing = [p for p in filepaths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"The following SICD file(s) do not exist: {missing}"
        )
    return SICDCollectionReader(filepaths, polarizations=polarizations)
