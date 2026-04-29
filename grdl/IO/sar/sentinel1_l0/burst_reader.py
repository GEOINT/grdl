# -*- coding: utf-8 -*-
"""
Sentinel-1 L0 Burst-Level Data Access.

Provides burst-level access to Sentinel-1 Level 0 data.  Bursts
are the fundamental unit of TOPS-mode acquisitions, containing a
set of azimuth lines acquired during a single beam-steering cycle.

Features
--------
- Fast burst seeking using index files
- Burst boundary detection from packet headers
- Automatic FDBAQ decompression (requires optional
  ``sentinel1decoder`` dependency)
- Parallel burst decoding across CPU cores

Burst Structure
---------------
For IW mode, each swath contains ~10-12 bursts per cycle.  Each
burst contains ~1400-1500 azimuth lines organized as packets
carrying I/Q samples.

References
----------
- Sentinel-1 Level-0 Product Format Specifications (S1PD.SP.00110)

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.sar.sentinel1_l0.binary_parser import (
    BurstIndexRecord,
    parse_burst_index_file,
)
from grdl.IO.sar.sentinel1_l0.constants import (
    BURST_GAP_THRESHOLD_US,
    BURST_LINE_FILTER_RATIO,
    GPS_LEAP_SECONDS,
)
from grdl.IO.sar.sentinel1_l0.decoder import (
    Sentinel1Decoder,
    check_decoder_available,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Parallel processing worker
# =============================================================================


def _decode_chunk_worker(
    chunk_df: "pd.DataFrame",  # noqa: F821
    meas_file: str,
) -> np.ndarray:
    """Worker function for parallel burst decoding.

    Runs in a separate process and creates its own decoder
    instance to decode a chunk of packets.

    Parameters
    ----------
    chunk_df : pandas.DataFrame
        Packet metadata for this chunk.
    meas_file : str
        Path to the measurement file.

    Returns
    -------
    numpy.ndarray
        Decoded I/Q data.
    """
    try:
        from sentinel1decoder import Level0Decoder
        decoder = Level0Decoder(meas_file)
        iq_data = decoder.decode_packets(chunk_df)
        return iq_data
    except Exception as e:
        logger.error(f"Chunk decode failed: {e}")
        raise


# =============================================================================
# Burst information
# =============================================================================


@dataclass
class BurstInfo:
    """Information about a single burst.

    Parameters
    ----------
    burst_index : int
        Index within swath (0-based).
    swath : int
        Raw swath number.
    polarization : int
        Polarization code (0–3).
    start_packet, end_packet : int
        Packet index range (half-open).
    num_lines, num_samples : int, optional
        Burst dimensions.
    byte_offset : int, optional
        Byte offset in the measurement file (from index file).
    reference_time : float, optional
        GPS seconds at burst start (from index).
    duration : float, optional
        Burst duration in seconds (from index).
    swst, pri : float, optional
        Sampling Window Start Time and Pulse Repetition
        Interval in seconds.
    rank : int, optional
        Number of PRI intervals before receive window.
    """

    burst_index: int
    swath: int
    polarization: int
    start_packet: int
    end_packet: int
    num_lines: int = 0
    num_samples: int = 0
    byte_offset: int = 0
    reference_time: float = 0.0
    duration: float = 0.0
    swst: float = 0.0
    pri: float = 0.0
    rank: int = 0

    @property
    def num_packets(self) -> int:
        """Number of packets in the burst."""
        return self.end_packet - self.start_packet

    @property
    def start_time(self) -> Optional[datetime]:
        """UTC datetime of burst start.

        Converts GPS seconds (``reference_time``) to UTC.  GPS
        epoch is January 6, 1980, 00:00:00 UTC; leap seconds
        are applied per :data:`constants.GPS_LEAP_SECONDS`.

        Returns
        -------
        datetime or None
            UTC datetime, or ``None`` if ``reference_time`` is
            not set.
        """
        if self.reference_time <= 0:
            return None
        gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
        gps_datetime = gps_epoch + timedelta(
            seconds=self.reference_time
        )
        return gps_datetime - timedelta(
            seconds=GPS_LEAP_SECONDS
        )


@dataclass
class SwathInfo:
    """Information about a swath.

    Parameters
    ----------
    swath : int
        Swath number.
    polarization : int
        Polarization code.
    num_bursts : int, optional
        Number of bursts.
    bursts : list of BurstInfo, optional
        Burst descriptors.
    start_packet, end_packet : int, optional
        Packet index range spanning the swath.
    """

    swath: int
    polarization: int
    num_bursts: int = 0
    bursts: List[BurstInfo] = field(default_factory=list)
    start_packet: int = 0
    end_packet: int = 0


# =============================================================================
# Burst reader
# =============================================================================


class BurstReader:
    """Reader for burst-level access to Sentinel-1 L0 data.

    Enumerates bursts, reads individual bursts, and reads entire
    swaths with automatic FDBAQ decompression.

    Parameters
    ----------
    measurement_file : str or Path
        Path to ``.dat`` measurement file.
    index_file : str or Path, optional
        Path to burst index file.  If ``None``, looks for
        ``*-index.dat`` alongside the measurement file.
    burst_gap_threshold_us : int, optional
        Override for burst-gap detection threshold
        (microseconds).  Defaults to
        :data:`constants.BURST_GAP_THRESHOLD_US`.
    burst_line_filter_ratio : float, optional
        Override for minimum burst-line ratio.  Defaults to
        :data:`constants.BURST_LINE_FILTER_RATIO`.

    Examples
    --------
    >>> reader = BurstReader('/path/to/measurement.dat')
    >>> bursts = reader.get_burst_info()
    >>> print(f'Found {len(bursts)} bursts')
    >>> iq_data = reader.read_burst(bursts[0])
    >>> swath_data = reader.read_swath(swath=1)
    """

    def __init__(
        self,
        measurement_file: Union[str, Path],
        index_file: Optional[Union[str, Path]] = None,
        burst_gap_threshold_us: Optional[int] = None,
        burst_line_filter_ratio: Optional[float] = None,
    ) -> None:
        self.measurement_file = Path(measurement_file)
        self._index_file: Optional[Path] = None
        self._decoder: Optional[Sentinel1Decoder] = None
        self._burst_index: Optional[
            List[BurstIndexRecord]
        ] = None
        self._burst_info: Optional[List[BurstInfo]] = None
        self._swath_info: Optional[Dict[int, SwathInfo]] = None
        self._burst_gap_threshold_us: int = (
            burst_gap_threshold_us
            if burst_gap_threshold_us is not None
            else BURST_GAP_THRESHOLD_US
        )
        self._burst_line_filter_ratio: float = (
            burst_line_filter_ratio
            if burst_line_filter_ratio is not None
            else BURST_LINE_FILTER_RATIO
        )

        if index_file is not None:
            self._index_file = Path(index_file)
        else:
            default_index = (
                self.measurement_file.parent
                / (self.measurement_file.stem + "-index.dat")
            )
            if default_index.exists():
                self._index_file = default_index

        self._initialize()

    def _initialize(self) -> None:
        """Initialize decoder and load index if available."""
        if check_decoder_available():
            self._decoder = Sentinel1Decoder(
                self.measurement_file
            )
            self._decoder.decode_metadata()

        if self._index_file and self._index_file.exists():
            self._burst_index = parse_burst_index_file(
                self._index_file
            )
            logger.debug(
                f"Loaded {len(self._burst_index)} burst "
                "index records"
            )

    @property
    def has_index(self) -> bool:
        """Whether a burst index file is loaded."""
        return (
            self._burst_index is not None
            and len(self._burst_index) > 0
        )

    @property
    def num_packets(self) -> int:
        """Total number of packets in the file."""
        if self._decoder:
            return self._decoder.num_packets
        return 0

    def get_burst_info(
        self, force_recompute: bool = False,
    ) -> List[BurstInfo]:
        """Get information about all bursts.

        Prefers packet-based detection when a decoder is
        available (the index file lacks swath information).

        Parameters
        ----------
        force_recompute : bool, optional
            Force recomputation even if cached.

        Returns
        -------
        list of BurstInfo
        """
        if (
            self._burst_info is not None
            and not force_recompute
        ):
            return self._burst_info

        if self._decoder:
            self._burst_info = (
                self._detect_bursts_from_packets()
            )
        elif self.has_index:
            self._burst_info = self._get_bursts_from_index()
        else:
            self._burst_info = []

        return self._burst_info

    def _get_bursts_from_index(self) -> List[BurstInfo]:
        """Get burst info purely from the index file."""
        bursts = []
        for record in self._burst_index:
            bursts.append(BurstInfo(
                burst_index=record.burst_index,
                swath=0,  # Unknown without packet headers.
                polarization=0,
                start_packet=record.start_packet,
                end_packet=record.start_packet + 1,
                byte_offset=record.byte_offset,
                reference_time=record.reference_time,
                duration=record.duration,
            ))
        return bursts

    def _detect_bursts_from_packets(
        self,
    ) -> List[BurstInfo]:
        """Detect burst boundaries from packet headers."""
        if not self._decoder:
            return []

        df = self._decoder._metadata_df
        if df is None or len(df) == 0:
            return []

        bursts: List[BurstInfo] = []

        # British spelling used by some versions of the decoder.
        pol_col = None
        if "Polarisation" in df.columns:
            pol_col = "Polarisation"
        elif "Polarization" in df.columns:
            pol_col = "Polarization"

        if pol_col and "Swath Number" in df.columns:
            for (swath, pol), group_df in df.groupby(
                ["Swath Number", pol_col], group_keys=False,
            ):
                # SwathNumber is np.int64 (int() works);
                # Polarisation is an enum-like with .value (int() raises).
                swath_int = int(getattr(swath, "value", swath))
                pol_int = int(getattr(pol, "value", pol))
                bursts.extend(
                    self._detect_burst_boundaries(
                        group_df, swath_int, pol_int
                    )
                )
        elif "Swath Number" in df.columns:
            for swath, group_df in df.groupby("Swath Number", group_keys=False):
                bursts.extend(
                    self._detect_burst_boundaries(
                        group_df, int(swath), 0
                    )
                )
        else:
            bursts = self._detect_burst_boundaries(df, 0, 0)

        bursts.sort(key=lambda b: b.start_packet)
        return bursts

    def _detect_burst_boundaries(
        self,
        df,
        swath: int,
        polarization: int,
    ) -> List[BurstInfo]:
        """Detect burst boundaries using inter-packet time gaps."""
        if len(df) < 2:
            return []

        bursts: List[BurstInfo] = []
        indices = df.index.tolist()

        coarse_times = df["Coarse Time"].values
        fine_times = df["Fine Time"].values

        # GPS time in microseconds.  Fine Time is decoded to
        # fractional seconds (raw / 65536) by the decoder.
        times = (coarse_times + fine_times) * 1e6

        has_timing = all(
            col in df.columns
            for col in ["SWST", "PRI", "Rank"]
        )
        if has_timing:
            swst_values = df["SWST"].values
            pri_values = df["PRI"].values
            rank_values = df["Rank"].values

        has_quads = "Number of Quads" in df.columns
        if has_quads:
            quad_values = df["Number of Quads"].values

        gap_threshold = self._burst_gap_threshold_us
        time_diffs = np.diff(times)
        gap_indices = np.where(time_diffs > gap_threshold)[0]

        def get_timing_params(
            start_idx: int, end_idx: int,
        ) -> Tuple[float, float, int, int]:
            """Mean SWST, PRI, mode Rank, and num_samples."""
            swst, pri, rank = 0.0, 0.0, 0
            if has_timing:
                end_sample = min(
                    start_idx + 10, len(swst_values)
                )
                swst = float(np.mean(
                    swst_values[start_idx:end_sample]
                ))
                pri = float(np.mean(
                    pri_values[start_idx:end_sample]
                ))
                rank = int(rank_values[start_idx])
            n_samples = 0
            if has_quads:
                burst_quads = quad_values[
                    start_idx:end_idx
                ]
                vals, counts = np.unique(
                    burst_quads, return_counts=True
                )
                n_samples = int(vals[np.argmax(counts)])
            return swst, pri, rank, n_samples

        start_idx = 0
        for burst_num, gap_idx in enumerate(gap_indices):
            end_idx = gap_idx + 1
            burst_ref_time = times[start_idx] / 1e6
            swst, pri, rank, n_samples = get_timing_params(
                start_idx, end_idx
            )
            bursts.append(BurstInfo(
                burst_index=burst_num,
                swath=swath,
                polarization=polarization,
                start_packet=int(indices[start_idx]),
                end_packet=int(indices[end_idx - 1]) + 1,
                num_lines=end_idx - start_idx,
                num_samples=n_samples,
                reference_time=burst_ref_time,
                swst=swst,
                pri=pri,
                rank=rank,
            ))
            start_idx = end_idx

        if start_idx < len(indices):
            burst_ref_time = times[start_idx] / 1e6
            swst, pri, rank, n_samples = get_timing_params(
                start_idx, len(indices)
            )
            bursts.append(BurstInfo(
                burst_index=len(gap_indices),
                swath=swath,
                polarization=polarization,
                start_packet=int(indices[start_idx]),
                end_packet=int(indices[-1]) + 1,
                num_lines=len(indices) - start_idx,
                num_samples=n_samples,
                reference_time=burst_ref_time,
                swst=swst,
                pri=pri,
                rank=rank,
            ))

        # Filter partial (edge) bursts below the median-ratio
        # threshold and re-index sequentially.
        if len(bursts) >= 3:
            line_counts = [b.num_lines for b in bursts]
            median_lines = np.median(line_counts)
            min_lines = int(
                median_lines * self._burst_line_filter_ratio
            )
            bursts = [
                b for b in bursts if b.num_lines >= min_lines
            ]

            reindexed = []
            for i, b in enumerate(bursts):
                reindexed.append(BurstInfo(
                    burst_index=i,
                    swath=b.swath,
                    polarization=b.polarization,
                    start_packet=b.start_packet,
                    end_packet=b.end_packet,
                    num_lines=b.num_lines,
                    num_samples=b.num_samples,
                    byte_offset=b.byte_offset,
                    reference_time=b.reference_time,
                    duration=b.duration,
                    swst=b.swst,
                    pri=b.pri,
                    rank=b.rank,
                ))
            bursts = reindexed

        return bursts

    def get_swath_info(
        self, force_recompute: bool = False,
    ) -> Dict[int, SwathInfo]:
        """Return per-swath summaries.

        Parameters
        ----------
        force_recompute : bool, optional
            Force recomputation.

        Returns
        -------
        dict
            Mapping swath number → :class:`SwathInfo`.
        """
        if (
            self._swath_info is not None
            and not force_recompute
        ):
            return self._swath_info

        bursts = self.get_burst_info()
        self._swath_info = {}

        for burst in bursts:
            if burst.swath not in self._swath_info:
                self._swath_info[burst.swath] = SwathInfo(
                    swath=burst.swath,
                    polarization=burst.polarization,
                )
            swath = self._swath_info[burst.swath]
            swath.bursts.append(burst)
            swath.num_bursts = len(swath.bursts)

            if (
                swath.start_packet == 0
                or burst.start_packet < swath.start_packet
            ):
                swath.start_packet = burst.start_packet
            if burst.end_packet > swath.end_packet:
                swath.end_packet = burst.end_packet

        return self._swath_info

    def get_burst_swst_array(
        self, burst: Union["BurstInfo", int],
    ) -> np.ndarray:
        """Per-packet SWST values for a burst.

        Returns the Sampling Window Start Time for every packet
        in the burst, enabling per-vector receive-start
        computation.

        Parameters
        ----------
        burst : BurstInfo or int
            Burst descriptor or burst index.

        Returns
        -------
        numpy.ndarray
            1-D float64 array of SWST values (seconds).
        """
        if isinstance(burst, int):
            bursts = self.get_burst_info()
            if burst >= len(bursts):
                raise IndexError(
                    f"Burst index {burst} >= {len(bursts)}"
                )
            burst = bursts[burst]

        if self._decoder is None:
            return np.full(
                burst.num_lines, burst.swst,
                dtype=np.float64,
            )

        df = self._decoder._metadata_df
        burst_df = df.loc[
            burst.start_packet:burst.end_packet - 1
        ]

        if "SWST" in burst_df.columns:
            return burst_df["SWST"].values.astype(
                np.float64
            )

        return np.full(
            burst.num_lines, burst.swst,
            dtype=np.float64,
        )

    def read_burst(
        self,
        burst: Union[BurstInfo, int],
        progress_callback: Optional[
            Callable[[int, int], None]
        ] = None,
    ) -> np.ndarray:
        """Read and decode a single burst.

        Parameters
        ----------
        burst : BurstInfo or int
            Burst descriptor or burst index.
        progress_callback : callable, optional
            ``callback(current, total)`` for progress.

        Returns
        -------
        numpy.ndarray
            Complex I/Q samples.

        Raises
        ------
        RuntimeError
            If the decoder is not available.
        """
        if not self._decoder:
            raise RuntimeError("Decoder not available")

        if isinstance(burst, int):
            bursts = self.get_burst_info()
            if burst < 0 or burst >= len(bursts):
                raise ValueError(
                    f"Invalid burst index: {burst}"
                )
            burst = bursts[burst]

        df = self._decoder._metadata_df.iloc[
            burst.start_packet:burst.end_packet
        ]
        return self._decoder.decode_packets(
            packet_df=df,
            progress_callback=progress_callback,
        )

    def read_burst_parallel(
        self,
        burst: Union[BurstInfo, int],
        num_workers: Optional[int] = None,
        chunk_size: int = 1024,
    ) -> np.ndarray:
        """Read and decode a burst using parallel processes.

        Parameters
        ----------
        burst : BurstInfo or int
            Burst descriptor or burst index.
        num_workers : int, optional
            Number of worker processes (default CPU count).
        chunk_size : int, optional
            Packets per chunk.

        Returns
        -------
        numpy.ndarray
            Complex I/Q samples.
        """
        from concurrent.futures import ProcessPoolExecutor
        import os

        if not self._decoder:
            raise RuntimeError("Decoder not available")

        if isinstance(burst, int):
            bursts = self.get_burst_info()
            if burst < 0 or burst >= len(bursts):
                raise ValueError(
                    f"Invalid burst index: {burst}"
                )
            burst = bursts[burst]

        burst_df = self._decoder._metadata_df.iloc[
            burst.start_packet:burst.end_packet
        ].copy()
        if burst_df.empty:
            return np.array([], dtype=np.complex64)

        if num_workers is None:
            num_workers = os.cpu_count() or 4

        num_packets = len(burst_df)
        meas_file = str(self.measurement_file)

        effective_chunk_size = min(
            chunk_size, max(1, num_packets // num_workers)
        )

        if num_packets < num_workers * 2 or num_workers <= 1:
            return self._decoder.decode_packets(
                packet_df=burst_df
            )

        chunks = []
        if "Number of Quads" in burst_df.columns:
            for _, group_df in burst_df.groupby(
                "Number of Quads"
            ):
                group_df = group_df.copy()
                for i in range(
                    0, len(group_df), effective_chunk_size
                ):
                    chunks.append(
                        group_df.iloc[
                            i:i + effective_chunk_size
                        ]
                    )
        else:
            for i in range(
                0, num_packets, effective_chunk_size
            ):
                chunks.append(
                    burst_df.iloc[
                        i:i + effective_chunk_size
                    ]
                )

        logger.debug(
            f"Parallel decoding: {len(chunks)} chunks across "
            f"{num_workers} workers"
        )

        try:
            with ProcessPoolExecutor(
                max_workers=num_workers
            ) as executor:
                futures = [
                    executor.submit(
                        _decode_chunk_worker,
                        chunk,
                        meas_file,
                    )
                    for chunk in chunks
                ]
                results = [f.result() for f in futures]

            if not results:
                return np.array([], dtype=np.complex64)

            widths = [
                r.shape[1] if r.ndim > 1 else r.shape[0]
                for r in results
            ]
            if len(set(widths)) == 1:
                iq_data = np.vstack(results)
            else:
                max_width = max(widths)
                padded = []
                for r in results:
                    if r.ndim == 1:
                        r = r.reshape(1, -1)
                    if r.shape[1] < max_width:
                        pad_w = max_width - r.shape[1]
                        r = np.pad(
                            r, ((0, 0), (0, pad_w)),
                            mode="constant",
                            constant_values=0,
                        )
                    padded.append(r)
                iq_data = np.vstack(padded)

            return iq_data.astype(np.complex64)

        except Exception as e:
            logger.error(f"Parallel decoding failed: {e}")
            return self._decoder.decode_packets(
                packet_df=burst_df
            )

    def read_swath(
        self,
        swath: int,
        progress_callback: Optional[
            Callable[[int, int], None]
        ] = None,
    ) -> np.ndarray:
        """Read and decode an entire swath.

        Parameters
        ----------
        swath : int
            Raw swath number.
        progress_callback : callable, optional
            ``callback(current, total)`` for progress.

        Returns
        -------
        numpy.ndarray
            Complex I/Q samples.
        """
        if not self._decoder:
            raise RuntimeError("Decoder not available")

        swath_info = self.get_swath_info()
        if swath not in swath_info:
            raise ValueError(f"Swath {swath} not found")

        info = swath_info[swath]
        df = self._decoder._metadata_df.iloc[
            info.start_packet:info.end_packet
        ]
        if "Swath Number" in df.columns:
            df = df[df["Swath Number"].apply(
                lambda x: getattr(x, "value", int(x)) == swath
            )]

        return self._decoder.decode_packets(
            packet_df=df,
            progress_callback=progress_callback,
        )

    def iter_bursts(
        self, swath: Optional[int] = None,
    ) -> Iterator[Tuple[BurstInfo, np.ndarray]]:
        """Iterate over bursts, yielding info and I/Q data.

        Parameters
        ----------
        swath : int, optional
            If provided, yield only bursts for that swath.

        Yields
        ------
        (BurstInfo, numpy.ndarray)
            ``(info, iq_data)`` for each burst.
        """
        bursts = self.get_burst_info()
        for burst in bursts:
            if swath is not None and burst.swath != swath:
                continue
            try:
                iq_data = self.read_burst(burst)
                yield burst, iq_data
            except Exception as e:
                logger.warning(
                    f"Failed to read burst "
                    f"{burst.burst_index}: {e}"
                )
                continue

    def close(self) -> None:
        """Release reader resources."""
        if self._decoder:
            self._decoder.close()
            self._decoder = None

    def __enter__(self) -> "BurstReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# =============================================================================
# Convenience functions
# =============================================================================


def count_bursts(
    measurement_file: Union[str, Path],
) -> int:
    """Count bursts in a measurement file.

    Parameters
    ----------
    measurement_file : str or Path
        Path to ``.dat`` file.

    Returns
    -------
    int
    """
    with BurstReader(measurement_file) as reader:
        return len(reader.get_burst_info())


def read_single_burst(
    measurement_file: Union[str, Path],
    burst_index: int,
) -> np.ndarray:
    """Read a single burst from a measurement file.

    Parameters
    ----------
    measurement_file : str or Path
        Path to ``.dat`` file.
    burst_index : int
        Burst index.

    Returns
    -------
    numpy.ndarray
        Complex I/Q samples.
    """
    with BurstReader(measurement_file) as reader:
        return reader.read_burst(burst_index)
