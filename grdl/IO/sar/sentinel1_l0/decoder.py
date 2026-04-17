# -*- coding: utf-8 -*-
"""
Sentinel-1 L0 Packet Decoder Wrapper.

Wrapper around the optional ``sentinel1decoder`` package for
decoding raw ISP packets from Sentinel-1 Level 0 measurement
files.

Features
--------
- Graceful handling of missing dependency (module import succeeds
  even when ``sentinel1decoder`` is not installed; only construction
  raises)
- Automatic grouping by Number of Quads for uniform decode calls
- Progress callbacks for long operations

Install the optional dependency with::

    pip install grdl[s1_l0]

References
----------
- ``sentinel1decoder``: https://github.com/Rich-Hall/sentinel1decoder
- Sentinel-1 SAR Space Packet Protocol Data Unit
  (S1-IF-ASD-PL-0007)

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
2026-04-16

Modified
--------
2026-04-16
"""

# Standard library
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

# Third-party
import numpy as np

# GRDL internal
from grdl.exceptions import DependencyError

logger = logging.getLogger(__name__)


# =============================================================================
# Dependency check
# =============================================================================

try:
    import pandas as pd
    from sentinel1decoder import Level0Decoder as _Level0Decoder
    _HAS_S1_DECODER = True
except ImportError:
    _HAS_S1_DECODER = False
    pd = None
    _Level0Decoder = None


def check_decoder_available() -> bool:
    """Return whether ``sentinel1decoder`` is importable."""
    return _HAS_S1_DECODER


def require_decoder() -> None:
    """Raise :class:`grdl.exceptions.DependencyError` if the
    optional ``sentinel1decoder`` package is not installed."""
    if not _HAS_S1_DECODER:
        raise DependencyError(
            "sentinel1decoder is required for Sentinel-1 L0 "
            "packet decoding but is not installed. "
            "Install with: pip install grdl[s1_l0]"
        )


# =============================================================================
# Packet metadata
# =============================================================================


@dataclass
class PacketMetadata:
    """Decoded packet header metadata.

    Extracted from the ``sentinel1decoder`` metadata DataFrame.

    Parameters
    ----------
    packet_index : int
        Sequential packet index.
    coarse_time, fine_time : int
        ISP Coarse Time (seconds since GPS epoch) and Fine Time
        (16-bit fraction of a second).
    swath_number : int
        Raw swath number from the header.
    polarization : int
        Polarization code (0–3).
    num_quads : int
        Number of quads (sample pairs).
    baq_mode : int, optional
        BAQ compression mode.
    pri_code, swst_code : int, optional
        PRI / SWST codes from the header.
    ecc_number : int, optional
        ECC number for calibration.
    """

    packet_index: int
    coarse_time: int
    fine_time: int
    swath_number: int
    polarization: int
    num_quads: int
    baq_mode: int = 0
    pri_code: int = 0
    swst_code: int = 0
    ecc_number: int = 0


# =============================================================================
# Decoder wrapper
# =============================================================================


class Sentinel1Decoder:
    """Wrapper around ``sentinel1decoder`` for packet decoding.

    Provides a clean interface with automatic quad grouping and
    optional progress callbacks.

    Parameters
    ----------
    measurement_file : str or Path
        Path to ``.dat`` measurement file.
    enable_gpu : bool, optional
        Whether to enable GPU acceleration (reserved — currently
        no-op for ``sentinel1decoder``).

    Raises
    ------
    grdl.exceptions.DependencyError
        If ``sentinel1decoder`` is not installed.
    FileNotFoundError
        If the measurement file does not exist.

    Examples
    --------
    >>> decoder = Sentinel1Decoder('/path/to/measurement.dat')
    >>> meta_df = decoder.decode_metadata()
    >>> print(f'Found {len(meta_df)} packets')
    >>> iq_data = decoder.decode_all()
    """

    def __init__(
        self,
        measurement_file: Union[str, Path],
        enable_gpu: bool = False,
    ) -> None:
        require_decoder()

        self.measurement_file = Path(measurement_file)
        if not self.measurement_file.exists():
            raise FileNotFoundError(
                f"Measurement file not found: "
                f"{self.measurement_file}"
            )

        self.enable_gpu = enable_gpu
        self._decoder: Optional[_Level0Decoder] = None
        self._metadata_df: Optional["pd.DataFrame"] = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the underlying ``sentinel1decoder``."""
        try:
            self._decoder = _Level0Decoder(
                str(self.measurement_file)
            )
            logger.debug(
                f"Initialized decoder for "
                f"{self.measurement_file.name}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize decoder: {e}"
            ) from e

    @property
    def num_packets(self) -> int:
        """Total number of packets in the file."""
        if self._metadata_df is None:
            self.decode_metadata()
        return len(self._metadata_df)

    def decode_metadata(self) -> "pd.DataFrame":
        """Decode packet headers only (no payload).

        Returns
        -------
        pandas.DataFrame
            Packet metadata with columns such as ``Coarse Time``,
            ``Fine Time``, ``Swath Number``, ``Polarization``,
            ``Number of Quads``, ``BAQ Mode``, ``PRI Code``,
            ``SWST Code``, ``ECC Number``.
        """
        if self._metadata_df is not None:
            return self._metadata_df

        try:
            self._metadata_df = self._decoder.decode_metadata()
            logger.debug(
                f"Decoded metadata for "
                f"{len(self._metadata_df)} packets"
            )
            return self._metadata_df
        except Exception as e:
            raise RuntimeError(
                f"Failed to decode metadata: {e}"
            ) from e

    def get_packet_metadata(
        self, packet_index: int,
    ) -> Optional[PacketMetadata]:
        """Get metadata for a specific packet.

        Parameters
        ----------
        packet_index : int
            Packet index.

        Returns
        -------
        PacketMetadata or None
            Parsed metadata, or ``None`` if the index is invalid.
        """
        if self._metadata_df is None:
            self.decode_metadata()

        if (
            packet_index < 0
            or packet_index >= len(self._metadata_df)
        ):
            return None

        row = self._metadata_df.iloc[packet_index]
        return PacketMetadata(
            packet_index=packet_index,
            coarse_time=int(row.get("Coarse Time", 0)),
            fine_time=int(row.get("Fine Time", 0)),
            swath_number=int(row.get("Swath Number", 0)),
            polarization=int(row.get("Polarization", 0)),
            num_quads=int(row.get("Number of Quads", 0)),
            baq_mode=int(row.get("BAQ Mode", 0)),
            pri_code=int(row.get("PRI Code", 0)),
            swst_code=int(row.get("SWST Code", 0)),
            ecc_number=int(row.get("ECC Number", 0)),
        )

    def decode_packets(
        self,
        packet_df: Optional["pd.DataFrame"] = None,
        packet_indices: Optional[List[int]] = None,
        progress_callback: Optional[
            Callable[[int, int], None]
        ] = None,
    ) -> np.ndarray:
        """Decode packets to I/Q data.

        Automatically groups packets by Number of Quads to handle
        variable packet sizes (a ``sentinel1decoder``
        requirement).

        Parameters
        ----------
        packet_df : pandas.DataFrame, optional
            DataFrame of packets to decode.  If ``None``, uses
            all packets.
        packet_indices : list of int, optional
            Packet indices to decode.  Ignored if ``packet_df``
            is provided.
        progress_callback : callable, optional
            ``callback(current, total)`` for progress reporting.

        Returns
        -------
        numpy.ndarray
            Complex I/Q samples.
        """
        if self._metadata_df is None:
            self.decode_metadata()

        if packet_df is not None:
            df = packet_df
        elif packet_indices is not None:
            df = self._metadata_df.iloc[packet_indices]
        else:
            df = self._metadata_df

        if len(df) == 0:
            return np.array([], dtype=np.complex64)

        return self._decode_with_grouping(
            df, progress_callback
        )

    def _decode_with_grouping(
        self,
        df: "pd.DataFrame",
        progress_callback: Optional[
            Callable[[int, int], None]
        ] = None,
    ) -> np.ndarray:
        """Decode packets with automatic quad grouping."""
        if "Number of Quads" not in df.columns:
            try:
                iq_data = self._decoder.decode_packets(df)
                return iq_data.astype(np.complex64)
            except Exception as e:
                logger.error(f"Decode failed: {e}")
                return np.array([], dtype=np.complex64)

        unique_quads = df["Number of Quads"].unique()
        if len(unique_quads) == 1:
            try:
                iq_data = self._decoder.decode_packets(df)
                return iq_data.astype(np.complex64)
            except Exception as e:
                logger.error(f"Decode failed: {e}")
                return np.array([], dtype=np.complex64)

        results = []
        total = len(unique_quads)
        for idx, (num_quads, group_df) in enumerate(
            df.groupby("Number of Quads")
        ):
            try:
                iq_chunk = self._decoder.decode_packets(
                    group_df
                )
                results.append(iq_chunk)
                if progress_callback:
                    progress_callback(idx + 1, total)
            except Exception as e:
                logger.warning(
                    f"Failed to decode group with "
                    f"{num_quads} quads: {e}"
                )
                continue

        if not results:
            return np.array([], dtype=np.complex64)

        try:
            iq_data = np.vstack(results)
            return iq_data.astype(np.complex64)
        except ValueError:
            # Non-uniform shapes — flatten.
            iq_data = np.concatenate(
                [r.flatten() for r in results]
            )
            return iq_data.astype(np.complex64)

    def decode_all(
        self,
        progress_callback: Optional[
            Callable[[int, int], None]
        ] = None,
    ) -> np.ndarray:
        """Decode all packets in the file.

        Parameters
        ----------
        progress_callback : callable, optional
            ``callback(current, total)`` for progress reporting.

        Returns
        -------
        numpy.ndarray
            Complex I/Q samples.
        """
        return self.decode_packets(
            progress_callback=progress_callback,
        )

    def iter_packets(
        self, chunk_size: int = 1000,
    ) -> Iterator[Tuple[int, int, np.ndarray]]:
        """Iterate over packets in fixed-size chunks.

        Parameters
        ----------
        chunk_size : int, optional
            Packets per chunk.

        Yields
        ------
        (int, int, numpy.ndarray)
            ``(start_index, end_index, iq_data)`` for each chunk.
        """
        if self._metadata_df is None:
            self.decode_metadata()

        total = len(self._metadata_df)
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            df = self._metadata_df.iloc[start:end]
            try:
                iq_data = self._decode_with_grouping(df)
                yield start, end, iq_data
            except Exception as e:
                logger.warning(
                    f"Failed to decode chunk "
                    f"{start}:{end}: {e}"
                )
                yield (
                    start, end,
                    np.array([], dtype=np.complex64),
                )

    def get_swath_packets(
        self, swath_number: int,
    ) -> "pd.DataFrame":
        """Return packets for a specific swath.

        Parameters
        ----------
        swath_number : int
            Raw swath number.

        Returns
        -------
        pandas.DataFrame
            Packets for the given swath.
        """
        if self._metadata_df is None:
            self.decode_metadata()
        if "Swath Number" not in self._metadata_df.columns:
            return self._metadata_df
        return self._metadata_df[
            self._metadata_df["Swath Number"] == swath_number
        ]

    def get_polarization_packets(
        self, polarization: int,
    ) -> "pd.DataFrame":
        """Return packets for a specific polarization code.

        Parameters
        ----------
        polarization : int
            Polarization code (0–3).
        """
        if self._metadata_df is None:
            self.decode_metadata()
        if "Polarization" not in self._metadata_df.columns:
            return self._metadata_df
        return self._metadata_df[
            self._metadata_df["Polarization"] == polarization
        ]

    def get_burst_packets(
        self, start_index: int, end_index: int,
    ) -> "pd.DataFrame":
        """Return packets in a half-open index range.

        Parameters
        ----------
        start_index, end_index : int
            Starting (inclusive) and ending (exclusive) packet
            indices.
        """
        if self._metadata_df is None:
            self.decode_metadata()
        return self._metadata_df.iloc[start_index:end_index]

    def close(self) -> None:
        """Release decoder resources."""
        self._decoder = None
        self._metadata_df = None

    def __enter__(self) -> "Sentinel1Decoder":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# =============================================================================
# Convenience functions
# =============================================================================


def decode_measurement_file(
    measurement_file: Union[str, Path],
) -> np.ndarray:
    """Decode an entire measurement file.

    Parameters
    ----------
    measurement_file : str or Path
        Path to ``.dat`` measurement file.

    Returns
    -------
    numpy.ndarray
        Complex I/Q samples.
    """
    with Sentinel1Decoder(measurement_file) as decoder:
        return decoder.decode_all()


def get_packet_count(
    measurement_file: Union[str, Path],
) -> int:
    """Return the number of packets in a measurement file.

    Parameters
    ----------
    measurement_file : str or Path
        Path to ``.dat`` measurement file.

    Returns
    -------
    int
    """
    with Sentinel1Decoder(measurement_file) as decoder:
        return decoder.num_packets
