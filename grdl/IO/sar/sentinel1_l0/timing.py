# -*- coding: utf-8 -*-
"""
Sentinel-1 L0 Timing Calculations.

Timing utilities for Sentinel-1 Level 0 data:

- Reference time determination (epoch for relative-time math)
- Conversion between absolute UTC and reference-relative seconds
- Pulse transmit time sequences
- Receive window time calculations

Time Representation
-------------------
High-precision times are stored as ``(integer, fractional)`` pairs to
avoid float64 rounding over long time spans.  This mirrors the
representation used by standards that demand sub-microsecond
precision over multi-hour collections.

References
----------
- Sentinel-1 Level-0 Product Format Specifications (S1PD.SP.00110)

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
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.sar.sentinel1_l0.constants import IW_MODE_PARAMS

if TYPE_CHECKING:
    from grdl.IO.models.sentinel1_l0 import (
        S1L0OrbitStateVector,
        Sentinel1L0Metadata,
    )
    from grdl.IO.sar.sentinel1_l0.orbit import POEParser


@dataclass
class TimeComponents:
    """High-precision time as integer + fractional seconds.

    Parameters
    ----------
    integer : int
        Integer part of the time in seconds (may be negative).
    fractional : float
        Fractional part in the range ``[0, 1)``.
    """

    integer: int
    fractional: float

    @classmethod
    def from_seconds(cls, seconds: float) -> "TimeComponents":
        """Create ``TimeComponents`` from a float seconds value.

        Parameters
        ----------
        seconds : float
            Time in seconds (may be negative).

        Returns
        -------
        TimeComponents
            Properly split integer/fractional parts.
        """
        if seconds >= 0:
            int_part = int(seconds)
            frac_part = seconds - int_part
        else:
            # For negative times, ensure fractional part is positive.
            int_part = int(np.floor(seconds))
            frac_part = seconds - int_part
        return cls(integer=int_part, fractional=frac_part)

    @property
    def total_seconds(self) -> float:
        """Recombined total seconds."""
        return self.integer + self.fractional


class TimingCalculator:
    """Timing conversions relative to a reference epoch.

    Encapsulates conversions between absolute UTC datetimes and
    seconds relative to a chosen reference epoch.  Used for pulse
    time sequences, receive window timing, and time-bound queries.

    Parameters
    ----------
    t_ref : datetime
        Reference epoch (UTC).
    prf_hz : float, optional
        Pulse Repetition Frequency in Hz.
    pulse_duration_s : float, optional
        Transmit pulse duration in seconds.
    range_sampling_rate_hz : float, optional
        Range sampling rate in Hz.

    Examples
    --------
    >>> timing = TimingCalculator(t_ref=start_time, prf_hz=1717.0)
    >>> rel_time = timing.to_relative_seconds(some_absolute_time)
    >>> components = timing.to_time_components(some_absolute_time)
    """

    def __init__(
        self,
        t_ref: datetime,
        prf_hz: float = IW_MODE_PARAMS[
            "pulse_repetition_frequency_hz"
        ],
        pulse_duration_s: float = IW_MODE_PARAMS[
            "tx_pulse_length_s"
        ],
        range_sampling_rate_hz: float = IW_MODE_PARAMS[
            "range_sampling_rate_hz"
        ],
    ) -> None:
        self.t_ref = t_ref
        self.prf_hz = prf_hz
        self.pulse_duration_s = pulse_duration_s
        self.range_sampling_rate_hz = range_sampling_rate_hz

    @property
    def pri_s(self) -> float:
        """Pulse Repetition Interval in seconds."""
        if self.prf_hz > 0:
            return 1.0 / self.prf_hz
        return 0.0

    @property
    def sample_period_s(self) -> float:
        """Range sample period in seconds."""
        if self.range_sampling_rate_hz > 0:
            return 1.0 / self.range_sampling_rate_hz
        return 0.0

    def to_relative_seconds(self, time: datetime) -> float:
        """Convert absolute time to seconds relative to ``t_ref``.

        Parameters
        ----------
        time : datetime
            Absolute UTC datetime.

        Returns
        -------
        float
            Seconds relative to ``t_ref`` (negative if before).
        """
        delta = time - self.t_ref
        return delta.total_seconds()

    def to_time_components(self, time: datetime) -> TimeComponents:
        """Convert absolute time to integer+fractional seconds.

        Parameters
        ----------
        time : datetime
            Absolute UTC datetime.

        Returns
        -------
        TimeComponents
            ``(integer, fractional)`` relative to ``t_ref``.
        """
        seconds = self.to_relative_seconds(time)
        return TimeComponents.from_seconds(seconds)

    def from_relative_seconds(self, seconds: float) -> datetime:
        """Convert reference-relative seconds to absolute UTC.

        Parameters
        ----------
        seconds : float
            Seconds relative to ``t_ref``.

        Returns
        -------
        datetime
            Absolute UTC datetime.
        """
        return self.t_ref + timedelta(seconds=seconds)

    def compute_pulse_times(
        self,
        start_time: datetime,
        num_pulses: int,
        prf_hz: Optional[float] = None,
    ) -> np.ndarray:
        """Compute transmit times for a sequence of pulses.

        Parameters
        ----------
        start_time : datetime
            Time of first pulse.
        num_pulses : int
            Number of pulses.
        prf_hz : float, optional
            Override PRF.  Uses the instance default if ``None``.

        Returns
        -------
        numpy.ndarray
            Pulse times in reference-relative seconds.
        """
        if prf_hz is None:
            prf_hz = self.prf_hz

        pri = 1.0 / prf_hz if prf_hz > 0 else 0.0
        start_offset = self.to_relative_seconds(start_time)

        return start_offset + np.arange(num_pulses) * pri

    def compute_receive_times(
        self,
        pulse_times: np.ndarray,
        slant_range_delay_s: float = 0.0,
    ) -> np.ndarray:
        """Compute receive window start times from pulse times.

        Parameters
        ----------
        pulse_times : numpy.ndarray
            Pulse transmit times (reference-relative seconds).
        slant_range_delay_s : float, optional
            Delay from pulse to receive window start.

        Returns
        -------
        numpy.ndarray
            Receive start times (reference-relative seconds).
        """
        return pulse_times + slant_range_delay_s

    def compute_receive_duration(self, num_samples: int) -> float:
        """Compute receive window duration from number of samples.

        Parameters
        ----------
        num_samples : int
            Number of range samples.

        Returns
        -------
        float
            Duration of receive window in seconds.
        """
        return num_samples * self.sample_period_s

    def get_time_bounds(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> Tuple[float, float]:
        """Get time bounds as reference-relative seconds.

        Parameters
        ----------
        start_time, end_time : datetime
            UTC bounds.

        Returns
        -------
        (float, float)
            ``(start_seconds, end_seconds)`` relative to ``t_ref``.
        """
        return (
            self.to_relative_seconds(start_time),
            self.to_relative_seconds(end_time),
        )


class BurstTimingExtractor:
    """Extract timing information from Sentinel-1 burst data.

    Handles packet-level timing extraction and synchronization with
    orbit data.

    Parameters
    ----------
    timing : TimingCalculator
        Time-conversion helper.
    """

    def __init__(self, timing: TimingCalculator) -> None:
        self.timing = timing

    def extract_pulse_times_from_packets(
        self,
        sensing_times: np.ndarray,
        pri_offsets: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Extract pulse times from packet sensing times.

        Parameters
        ----------
        sensing_times : numpy.ndarray
            Packet sensing times (datetime elements).
        pri_offsets : numpy.ndarray, optional
            Optional PRI offset corrections.

        Returns
        -------
        numpy.ndarray
            Pulse times in reference-relative seconds.
        """
        times = np.array([
            self.timing.to_relative_seconds(t)
            for t in sensing_times
        ])
        if pri_offsets is not None:
            times = times + pri_offsets
        return times

    def compute_vector_rcv_times(
        self,
        tx_times: np.ndarray,
        rank: int,
        swst_samples: int,
        sample_rate_hz: float,
    ) -> np.ndarray:
        """Compute receive start times from transmit times.

        In Sentinel-1, the receive window starts after a delay
        determined by the rank (number of missed pulses) and SWST
        (Sampling Window Start Time).

        Parameters
        ----------
        tx_times : numpy.ndarray
            Transmit times (reference-relative seconds).
        rank : int
            Sentinel-1 rank (number of PRI delays).
        swst_samples : int
            Sampling window start time in samples.
        sample_rate_hz : float
            Sampling rate in Hz.

        Returns
        -------
        numpy.ndarray
            Receive start times (reference-relative seconds).
        """
        # SWST in seconds.
        swst_seconds = swst_samples / sample_rate_hz

        # Total delay includes rank * PRI plus SWST.
        if self.timing.prf_hz > 0:
            rank_delay = rank * (1.0 / self.timing.prf_hz)
        else:
            rank_delay = 0.0

        total_delay = rank_delay + swst_seconds

        return tx_times + total_delay


def determine_reference_time(
    start_time: datetime,
    orbit_vectors: Optional[List["S1L0OrbitStateVector"]] = None,
    poe_parser: Optional["POEParser"] = None,
) -> datetime:
    """Determine the reference epoch for relative-time math.

    Uses the product annotation start time (sensing start) truncated
    to whole seconds.

    Parameters
    ----------
    start_time : datetime
        Product start time (from annotation).
    orbit_vectors : list of S1L0OrbitStateVector, optional
        Unused; reserved for future orbit-based refinement.
    poe_parser : POEParser, optional
        Unused; reserved for future POE-based refinement.

    Returns
    -------
    datetime
        Reference time as UTC datetime (whole seconds).
    """
    return start_time.replace(microsecond=0)


def create_timing_calculator(
    metadata: "Sentinel1L0Metadata",
    poe_path: Optional[Path] = None,
) -> TimingCalculator:
    """Create a ``TimingCalculator`` from Sentinel-1 L0 metadata.

    Parameters
    ----------
    metadata : Sentinel1L0Metadata
        Parsed L0 metadata.
    poe_path : Path, optional
        Optional path to a POE file for precision timing.

    Returns
    -------
    TimingCalculator
        Configured timing helper.
    """
    from grdl.IO.sar.sentinel1_l0.orbit import POEParser

    poe_parser = None
    if poe_path and poe_path.exists():
        try:
            poe_parser = POEParser(str(poe_path))
            poe_parser.parse()
        except Exception:
            poe_parser = None

    t_ref = determine_reference_time(
        start_time=metadata.start_time,
        orbit_vectors=metadata.orbit_state_vectors,
        poe_parser=poe_parser,
    )

    rp = metadata.radar_parameters
    prf_hz = (
        rp.pulse_repetition_frequency_hz
        if rp else IW_MODE_PARAMS[
            "pulse_repetition_frequency_hz"
        ]
    )
    pulse_duration = (
        rp.tx_pulse_length_s
        if rp else IW_MODE_PARAMS["tx_pulse_length_s"]
    )
    sample_rate = (
        rp.range_sampling_rate_hz
        if rp else IW_MODE_PARAMS["range_sampling_rate_hz"]
    )

    return TimingCalculator(
        t_ref=t_ref,
        prf_hz=prf_hz,
        pulse_duration_s=pulse_duration,
        range_sampling_rate_hz=sample_rate,
    )
