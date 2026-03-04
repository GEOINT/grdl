# -*- coding: utf-8 -*-
"""
Sentinel-1 Level-0 Metadata - Typed metadata for Sentinel-1 IW Level-0 products.

Dataclasses representing the metadata contained in a Sentinel-1 Level-0
SAFE product, including ISP header parameters, derived radar parameters,
and product-level information from the XFDU manifest.

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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from grdl.IO.models.base import ImageMetadata


@dataclass
class S1L0ProductInfo:
    """Sentinel-1 Level-0 product-level metadata from manifest.safe.

    Parameters
    ----------
    mission : str, optional
        Mission identifier (``'S1A'``, ``'S1B'``, ``'S1C'``).
    mode : str, optional
        Acquisition mode (``'IW'``, ``'EW'``, ``'SM'``).
    product_type : str, optional
        Always ``'RAW'`` for Level-0.
    polarization_mode : str, optional
        Polarization class (``'DV'``, ``'DH'``, ``'SV'``, ``'SH'``).
    start_time : str, optional
        Acquisition start time (ISO 8601 UTC).
    stop_time : str, optional
        Acquisition stop time (ISO 8601 UTC).
    absolute_orbit : int, optional
        Absolute orbit number.
    relative_orbit : int, optional
        Relative orbit number within the cycle.
    orbit_pass : str, optional
        Orbit direction (``'ASCENDING'`` or ``'DESCENDING'``).
    """

    mission: Optional[str] = None
    mode: Optional[str] = None
    product_type: Optional[str] = None
    polarization_mode: Optional[str] = None
    start_time: Optional[str] = None
    stop_time: Optional[str] = None
    absolute_orbit: Optional[int] = None
    relative_orbit: Optional[int] = None
    orbit_pass: Optional[str] = None


@dataclass
class S1L0RadarParams:
    """Derived radar parameters from ISP secondary headers.

    Parameters
    ----------
    center_frequency : float
        Radar center frequency in Hz (5.405e9 for all S1).
    tx_bandwidth : float
        Transmit chirp bandwidth in Hz.
    chirp_rate : float
        Chirp rate (Hz/s).
    sampling_rate : float
        ADC sampling rate in Hz.
    tx_pulse_length : float
        Transmit pulse length in seconds.
    pri : float
        Pulse Repetition Interval in seconds.
    """

    center_frequency: float = 5.405e9
    tx_bandwidth: float = 0.0
    chirp_rate: float = 0.0
    sampling_rate: float = 0.0
    tx_pulse_length: float = 0.0
    pri: float = 0.0


@dataclass
class S1L0ChannelInfo:
    """Information about one polarization channel.

    Parameters
    ----------
    polarization : str
        Polarization label (``'VV'``, ``'VH'``, ``'HH'``, ``'HV'``).
    tx_pol : str
        Transmit polarization (``'V'`` or ``'H'``).
    rx_pol : str
        Receive polarization (``'V'`` or ``'H'``).
    measurement_file : str
        Path to the measurement .dat file.
    annotation_file : str, optional
        Path to the annotation .dat file.
    num_echo_packets : int
        Number of echo (signal_type=0) packets.
    max_num_quads : int
        Maximum number of quads across all echo packets.
    swath_numbers : list of int
        Unique swath numbers found in echo packets.
    """

    polarization: str = ""
    tx_pol: str = ""
    rx_pol: str = ""
    measurement_file: str = ""
    annotation_file: Optional[str] = None
    num_echo_packets: int = 0
    max_num_quads: int = 0
    swath_numbers: List[int] = field(default_factory=list)


@dataclass
class S1L0SwathInfo:
    """Information about one sub-swath within a polarization channel.

    Parameters
    ----------
    swath_number : int
        Logical sub-swath index (1, 2, or 3 for IW mode).
    raw_swath_number : int
        Raw ISP header swath number (e.g. 10, 11, 12 for IW mode).
    polarization : str
        Polarization label (``'VV'``, ``'VH'``, ``'HH'``, ``'HV'``).
    tx_pol : str
        Transmit polarization (``'V'`` or ``'H'``).
    rx_pol : str
        Receive polarization (``'V'`` or ``'H'``).
    num_echo_packets : int
        Number of echo packets belonging to this swath.
    max_num_quads : int
        Maximum number of quads across echo packets in this swath.
    channel_key : str
        Combined key for CRSD channel identification, e.g. ``'IW1_VV'``.
    """

    swath_number: int = 0
    raw_swath_number: int = 0
    polarization: str = ""
    tx_pol: str = ""
    rx_pol: str = ""
    num_echo_packets: int = 0
    max_num_quads: int = 0
    channel_key: str = ""


@dataclass
class S1L0FootprintCoord:
    """A single lat/lon coordinate in the scene footprint."""

    lat: float = 0.0
    lon: float = 0.0


@dataclass
class Sentinel1L0Metadata(ImageMetadata):
    """Top-level metadata for a Sentinel-1 Level-0 SAFE product.

    Parameters
    ----------
    product_info : S1L0ProductInfo, optional
        Product-level metadata from manifest.safe.
    radar_params : S1L0RadarParams, optional
        Derived radar parameters.
    channels : dict, optional
        Mapping from polarization label to ``S1L0ChannelInfo``.
    footprint : list of S1L0FootprintCoord, optional
        Scene footprint corner coordinates.
    data_take_id : int, optional
        Data take identifier from ISP headers.
    """

    product_info: Optional[S1L0ProductInfo] = None
    radar_params: Optional[S1L0RadarParams] = None
    channels: Dict[str, S1L0ChannelInfo] = field(default_factory=dict)
    footprint: List[S1L0FootprintCoord] = field(default_factory=list)
    data_take_id: Optional[int] = None
