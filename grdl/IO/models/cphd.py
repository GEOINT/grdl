# -*- coding: utf-8 -*-
"""
CPHD Metadata - Typed metadata for Compensated Phase History Data.

Nested dataclasses modeling the NGA CPHD standard as used by the IFP
(Image Formation Processor) pipeline. Covers channel descriptors,
per-vector parameters (PVP), transmit/receive waveform characteristics,
frequency band parameters, and collection information.

``CPHDMetadata`` extends ``ImageMetadata`` so that ``CPHDReader`` can
expose fully typed fields alongside the standard ``rows``, ``cols``,
``dtype`` interface.

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
2026-02-12

Modified
--------
2026-02-13
"""

# NOTE: Fields added 2026-02-13:
#   CPHDPVP: amp_sf, toa1, toa2
#   CPHDAntennaPattern, CPHDSceneCoordinates, CPHDReferenceGeometry,
#   CPHDDwellPolynomial — all optional on CPHDMetadata

# Standard library
from dataclasses import dataclass, field
from typing import List, Optional

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.models.base import ImageMetadata


# ===================================================================
# Channel descriptor
# ===================================================================

@dataclass
class CPHDChannel:
    """Per-channel descriptor from the CPHD Data section.

    Parameters
    ----------
    identifier : str
        Channel identifier string.
    num_vectors : int
        Number of phase history vectors (pulses).
    num_samples : int
        Number of samples per vector.
    signal_array_byte_offset : int, optional
        Byte offset of the signal array in the file.
    """

    identifier: str = ''
    num_vectors: int = 0
    num_samples: int = 0
    signal_array_byte_offset: Optional[int] = None


# ===================================================================
# Per-Vector Parameters (PVP)
# ===================================================================

@dataclass
class CPHDPVP:
    """Per-Vector Parameters — arrays indexed by pulse number.

    All position arrays have shape ``(N, 3)`` in ECF meters.
    All velocity arrays have shape ``(N, 3)`` in ECF m/s.
    Scalar-per-pulse arrays have shape ``(N,)``.

    Parameters
    ----------
    tx_time : np.ndarray
        Transmit time per pulse (seconds), shape ``(N,)``.
    tx_pos : np.ndarray
        Transmitter position ECF per pulse, shape ``(N, 3)``.
    rcv_time : np.ndarray
        Receive time per pulse (seconds), shape ``(N,)``.
    rcv_pos : np.ndarray
        Receiver position ECF per pulse, shape ``(N, 3)``.
    srp_pos : np.ndarray
        Scene Reference Point ECF per pulse, shape ``(N, 3)``.
    fx1 : np.ndarray
        Start frequency per pulse (Hz), shape ``(N,)``.
    fx2 : np.ndarray
        Stop frequency per pulse (Hz), shape ``(N,)``.
    tx_vel : np.ndarray, optional
        Transmitter velocity ECF per pulse, shape ``(N, 3)``.
    rcv_vel : np.ndarray, optional
        Receiver velocity ECF per pulse, shape ``(N, 3)``.
    sc0 : np.ndarray, optional
        Signal start index per pulse, shape ``(N,)``.
    scss : np.ndarray, optional
        Signal sample spacing per pulse (Hz), shape ``(N,)``.
    signal : np.ndarray, optional
        Signal validity indicator per pulse, shape ``(N,)``.
        Values > 0 indicate valid pulses.
    a_fdop : np.ndarray, optional
        Doppler centroid rate per pulse, shape ``(N,)``.
    a_frr1 : np.ndarray, optional
        Frequency rate of return 1 per pulse, shape ``(N,)``.
    a_frr2 : np.ndarray, optional
        Frequency rate of return 2 per pulse, shape ``(N,)``.
    amp_sf : np.ndarray, optional
        Amplitude scale factor per pulse, shape ``(N,)``.
        Multiply signal by this to normalize power across pulses.
    toa1 : np.ndarray, optional
        TOA start per pulse (seconds), shape ``(N,)``.
    toa2 : np.ndarray, optional
        TOA end per pulse (seconds), shape ``(N,)``.
    """

    tx_time: Optional[np.ndarray] = None
    tx_pos: Optional[np.ndarray] = None
    rcv_time: Optional[np.ndarray] = None
    rcv_pos: Optional[np.ndarray] = None
    srp_pos: Optional[np.ndarray] = None
    fx1: Optional[np.ndarray] = None
    fx2: Optional[np.ndarray] = None
    tx_vel: Optional[np.ndarray] = None
    rcv_vel: Optional[np.ndarray] = None
    sc0: Optional[np.ndarray] = None
    scss: Optional[np.ndarray] = None
    signal: Optional[np.ndarray] = None
    a_fdop: Optional[np.ndarray] = None
    a_frr1: Optional[np.ndarray] = None
    a_frr2: Optional[np.ndarray] = None
    amp_sf: Optional[np.ndarray] = None
    toa1: Optional[np.ndarray] = None
    toa2: Optional[np.ndarray] = None

    @property
    def num_vectors(self) -> int:
        """Number of pulse vectors in the PVP arrays."""
        if self.tx_time is not None:
            return len(self.tx_time)
        return 0

    @property
    def midpoint_time(self) -> float:
        """Midpoint time (average of tx and rcv time), in seconds."""
        if self.tx_time is not None and self.rcv_time is not None:
            return float(0.5 * np.mean(self.rcv_time + self.tx_time))
        return 0.0

    @property
    def first_valid_pulse(self) -> int:
        """Index of the first pulse with signal > 0.

        Returns 0 if no signal indicator is available or all pulses
        are valid.
        """
        if self.signal is not None:
            valid = np.where(self.signal > 0)[0]
            if len(valid) > 0:
                return int(valid[0])
        return 0

    def trim_to_valid(self) -> 'CPHDPVP':
        """Return a new CPHDPVP trimmed to valid pulses only.

        Uses the ``signal`` array to identify valid pulses (signal > 0).
        If no signal array exists, returns self unchanged.

        Returns
        -------
        CPHDPVP
            New instance with arrays sliced to valid pulse range.
        """
        start = self.first_valid_pulse
        if start == 0:
            return self

        n = self.num_vectors - start
        fields_1d = [
            'tx_time', 'rcv_time', 'fx1', 'fx2', 'sc0', 'scss',
            'signal', 'a_fdop', 'a_frr1', 'a_frr2',
            'amp_sf', 'toa1', 'toa2',
        ]
        fields_2d = ['tx_pos', 'tx_vel', 'rcv_pos', 'rcv_vel', 'srp_pos']

        kwargs = {}
        for name in fields_1d:
            arr = getattr(self, name)
            if arr is not None:
                kwargs[name] = arr[start:start + n]
            else:
                kwargs[name] = None

        for name in fields_2d:
            arr = getattr(self, name)
            if arr is not None:
                kwargs[name] = arr[start:start + n, :]
            else:
                kwargs[name] = None

        return CPHDPVP(**kwargs)

    def slice(self, start: int, end: int) -> 'CPHDPVP':
        """Return a new CPHDPVP with arrays sliced to ``[start:end]``.

        Parameters
        ----------
        start : int
            Starting pulse index (inclusive).
        end : int
            Ending pulse index (exclusive).

        Returns
        -------
        CPHDPVP
            New instance with arrays sliced to the given range.

        Raises
        ------
        ValueError
            If ``start >= end`` or indices are out of range.
        """
        n = self.num_vectors
        if start < 0 or end > n or start >= end:
            raise ValueError(
                f"Invalid slice [{start}:{end}] for PVP with "
                f"{n} vectors"
            )

        fields_1d = [
            'tx_time', 'rcv_time', 'fx1', 'fx2', 'sc0', 'scss',
            'signal', 'a_fdop', 'a_frr1', 'a_frr2',
            'amp_sf', 'toa1', 'toa2',
        ]
        fields_2d = ['tx_pos', 'tx_vel', 'rcv_pos', 'rcv_vel', 'srp_pos']

        kwargs = {}
        for name in fields_1d:
            arr = getattr(self, name)
            kwargs[name] = arr[start:end].copy() if arr is not None else None

        for name in fields_2d:
            arr = getattr(self, name)
            kwargs[name] = (
                arr[start:end, :].copy() if arr is not None else None
            )

        return CPHDPVP(**kwargs)


# ===================================================================
# Transmit waveform parameters
# ===================================================================

@dataclass
class CPHDTxWaveform:
    """Transmit waveform parameters from TxRcv/TxWFParameters.

    Parameters
    ----------
    lfm_rate : float, optional
        Linear FM chirp rate (Hz/s). Absolute value.
    pulse_length : float, optional
        Transmit pulse duration (seconds).
    identifier : str, optional
        Waveform identifier string.
    """

    lfm_rate: Optional[float] = None
    pulse_length: Optional[float] = None
    identifier: Optional[str] = None


# ===================================================================
# Receive parameters
# ===================================================================

@dataclass
class CPHDRcvParameters:
    """Receive parameters from TxRcv/RcvParameters.

    Parameters
    ----------
    window_length : float, optional
        Receive window duration (seconds).
    sample_rate : float, optional
        Receiver sample rate (Hz).
    identifier : str, optional
        Receiver identifier string.
    """

    window_length: Optional[float] = None
    sample_rate: Optional[float] = None
    identifier: Optional[str] = None


# ===================================================================
# Global / frequency band parameters
# ===================================================================

@dataclass
class CPHDGlobal:
    """Global parameters from the CPHD Global section.

    Parameters
    ----------
    domain_type : str, optional
        Signal domain: ``'FX'`` (frequency) or ``'TOA'`` (time of arrival).
    phase_sgn : int
        Phase sign convention from the CPHD standard (``+1`` or ``-1``).
        Applied when computing target signal phase:
        ``Phase(fx) = SGN * fx * TOA_TGT``.  Default ``-1``.
    fx_band_min : float, optional
        Minimum frequency of the collection band (Hz).
    fx_band_max : float, optional
        Maximum frequency of the collection band (Hz).
    toa_swath_min : float, optional
        Minimum TOA of the swath (seconds). For TOA domain.
    toa_swath_max : float, optional
        Maximum TOA of the swath (seconds). For TOA domain.
    """

    domain_type: Optional[str] = None
    phase_sgn: int = -1
    fx_band_min: Optional[float] = None
    fx_band_max: Optional[float] = None
    toa_swath_min: Optional[float] = None
    toa_swath_max: Optional[float] = None

    @property
    def bandwidth(self) -> Optional[float]:
        """Collection bandwidth in Hz, or None if not available."""
        if self.fx_band_min is not None and self.fx_band_max is not None:
            return self.fx_band_max - self.fx_band_min
        return None

    @property
    def center_frequency(self) -> Optional[float]:
        """Center frequency in Hz, or None if not available."""
        if self.fx_band_min is not None and self.fx_band_max is not None:
            return 0.5 * (self.fx_band_min + self.fx_band_max)
        return None


# ===================================================================
# Collection information
# ===================================================================

@dataclass
class CPHDCollectionInfo:
    """Collection information from the CollectionInfo section.

    Parameters
    ----------
    collector_name : str, optional
        Name of the collecting platform.
    core_name : str, optional
        Unique collection identifier.
    classification : str, optional
        Security classification string.
    collect_type : str, optional
        ``'MONOSTATIC'`` or ``'BISTATIC'``.
    radar_mode : str, optional
        Radar mode: ``'SPOTLIGHT'``, ``'STRIPMAP'``,
        ``'DYNAMIC STRIPMAP'``.
    radar_mode_id : str, optional
        Mode identifier string.
    """

    collector_name: Optional[str] = None
    core_name: Optional[str] = None
    classification: Optional[str] = None
    collect_type: Optional[str] = None
    radar_mode: Optional[str] = None
    radar_mode_id: Optional[str] = None


# ===================================================================
# Antenna pattern
# ===================================================================

@dataclass
class CPHDAntennaPattern:
    """Antenna gain/phase pattern from the CPHD Antenna section.

    The gain polynomial models 2D directional cosine variation::

        G(dcx, dcy) = GainZero + sum_ij GainPoly[i,j] * dcx^i * dcy^j

    Parameters
    ----------
    freq_zero : float, optional
        Reference frequency for the pattern (Hz).
    gain_zero : float, optional
        Boresight gain at freq_zero (dB).
    gain_poly : np.ndarray, optional
        2D gain polynomial coefficients, shape ``(M, N)``.
        Evaluated in directional cosine coordinates (dcx, dcy).
    eb_dcx_poly : np.ndarray, optional
        Electrical boresight DCX polynomial coefficients.
    eb_dcy_poly : np.ndarray, optional
        Electrical boresight DCY polynomial coefficients.
    acf_x_poly : np.ndarray, optional
        Antenna Coordinate Frame X-axis polynomial (ECF, order K x 3).
    acf_y_poly : np.ndarray, optional
        Antenna Coordinate Frame Y-axis polynomial (ECF, order K x 3).
    apc_offset : np.ndarray, optional
        Antenna Phase Center offset in ACF (3,).
    """

    freq_zero: Optional[float] = None
    gain_zero: Optional[float] = None
    gain_poly: Optional[np.ndarray] = None
    eb_dcx_poly: Optional[np.ndarray] = None
    eb_dcy_poly: Optional[np.ndarray] = None
    acf_x_poly: Optional[np.ndarray] = None
    acf_y_poly: Optional[np.ndarray] = None
    apc_offset: Optional[np.ndarray] = None


# ===================================================================
# Scene coordinates
# ===================================================================

@dataclass
class CPHDSceneCoordinates:
    """Scene coordinate information from the SceneCoordinates section.

    Parameters
    ----------
    earth_model : str, optional
        Earth model (e.g., ``'WGS_84'``).
    iarp_ecf : np.ndarray, optional
        Image Area Reference Point in ECF meters, shape ``(3,)``.
    iarp_llh : np.ndarray, optional
        Image Area Reference Point as ``[lat_deg, lon_deg, hae_m]``.
    image_area_x : tuple, optional
        Image area X extent in meters: ``(x_min, x_max)``.
    image_area_y : tuple, optional
        Image area Y extent in meters: ``(y_min, y_max)``.
    corner_points : np.ndarray, optional
        Image area corner coordinates, shape ``(4, 2)`` as
        ``[[lat, lon], ...]``.
    """

    earth_model: Optional[str] = None
    iarp_ecf: Optional[np.ndarray] = None
    iarp_llh: Optional[np.ndarray] = None
    image_area_x: Optional[tuple] = None
    image_area_y: Optional[tuple] = None
    corner_points: Optional[np.ndarray] = None


# ===================================================================
# Reference geometry
# ===================================================================

@dataclass
class CPHDReferenceGeometry:
    """Reference geometry at center of dwell from the CPHD file.

    Parameters
    ----------
    ref_time : float, optional
        Reference time in seconds from collection start.
    srp_ecf : np.ndarray, optional
        SRP position in ECF meters at reference time, shape ``(3,)``.
    srp_llh : np.ndarray, optional
        SRP lat/lon/HAE at reference time, shape ``(3,)``.
    side_of_track : str, optional
        ``'L'`` (left-looking) or ``'R'`` (right-looking).
    graze_angle_deg : float, optional
        Grazing angle at reference time (degrees).
    azimuth_angle_deg : float, optional
        Azimuth angle at reference time (degrees).
    twist_angle_deg : float, optional
        Twist angle at reference time (degrees).
    slope_angle_deg : float, optional
        Slope angle at reference time (degrees).
    layover_angle_deg : float, optional
        Layover angle at reference time (degrees).
    """

    ref_time: Optional[float] = None
    srp_ecf: Optional[np.ndarray] = None
    srp_llh: Optional[np.ndarray] = None
    side_of_track: Optional[str] = None
    graze_angle_deg: Optional[float] = None
    azimuth_angle_deg: Optional[float] = None
    twist_angle_deg: Optional[float] = None
    slope_angle_deg: Optional[float] = None
    layover_angle_deg: Optional[float] = None


# ===================================================================
# Dwell polynomial
# ===================================================================

@dataclass
class CPHDDwellPolynomial:
    """Dwell time polynomial from the CPHD Dwell section.

    Parameters
    ----------
    cod_time_poly : np.ndarray, optional
        Center of Dwell time polynomial coefficients.
    dwell_time_poly : np.ndarray, optional
        Dwell time polynomial coefficients.
    """

    cod_time_poly: Optional[np.ndarray] = None
    dwell_time_poly: Optional[np.ndarray] = None


# ===================================================================
# Top-level CPHDMetadata
# ===================================================================

@dataclass
class CPHDMetadata(ImageMetadata):
    """Typed metadata for CPHD (Compensated Phase History Data) files.

    Extends ``ImageMetadata`` with CPHD-specific typed fields for
    channels, per-vector parameters, waveform characteristics, frequency
    band, and collection information.

    The base ``rows``/``cols`` correspond to the first channel's
    ``num_vectors``/``num_samples``.

    Parameters
    ----------
    channels : List[CPHDChannel]
        Per-channel descriptors.
    pvp : CPHDPVP, optional
        Per-vector parameters for the active channel.
    global_params : CPHDGlobal, optional
        Global frequency/domain parameters.
    collection_info : CPHDCollectionInfo, optional
        Collection information.
    tx_waveform : CPHDTxWaveform, optional
        Transmit waveform parameters.
    rcv_parameters : CPHDRcvParameters, optional
        Receive parameters.
    antenna_pattern : CPHDAntennaPattern, optional
        Antenna gain/phase pattern data.
    scene_coordinates : CPHDSceneCoordinates, optional
        Scene coordinate information (IARP, image area bounds).
    reference_geometry : CPHDReferenceGeometry, optional
        Reference geometry at center of dwell.
    dwell : CPHDDwellPolynomial, optional
        Dwell time polynomials.
    num_channels : int
        Number of CPHD channels.

    Examples
    --------
    >>> from grdl.IO.sar import CPHDReader
    >>> with CPHDReader('data.cphd') as reader:
    ...     meta = reader.metadata  # CPHDMetadata
    ...     meta.pvp.tx_time        # (N,) transmit times
    ...     meta.pvp.srp_pos        # (N, 3) SRP positions
    ...     meta.global_params.bandwidth  # Hz
    """

    channels: List[CPHDChannel] = field(default_factory=list)
    pvp: Optional[CPHDPVP] = None
    global_params: Optional[CPHDGlobal] = None
    collection_info: Optional[CPHDCollectionInfo] = None
    tx_waveform: Optional[CPHDTxWaveform] = None
    rcv_parameters: Optional[CPHDRcvParameters] = None
    antenna_pattern: Optional[CPHDAntennaPattern] = None
    scene_coordinates: Optional[CPHDSceneCoordinates] = None
    reference_geometry: Optional[CPHDReferenceGeometry] = None
    dwell: Optional[CPHDDwellPolynomial] = None
    num_channels: int = 0


def create_subaperture_metadata(
    metadata: CPHDMetadata,
    start_pulse: int,
    end_pulse: int,
) -> CPHDMetadata:
    """Create a sub-aperture metadata slice for stripmap processing.

    Slices the PVP arrays and updates channel dimensions while
    preserving global parameters, waveform, and collection info.

    Parameters
    ----------
    metadata : CPHDMetadata
        Full CPHD metadata.
    start_pulse : int
        Starting pulse index (inclusive).
    end_pulse : int
        Ending pulse index (exclusive).

    Returns
    -------
    CPHDMetadata
        New metadata with sliced PVP and updated dimensions.

    Raises
    ------
    ValueError
        If metadata has no PVP or indices are invalid.
    """
    if metadata.pvp is None:
        raise ValueError("CPHDMetadata must have populated PVP arrays")

    sub_pvp = metadata.pvp.slice(start_pulse, end_pulse)
    n_pulses = end_pulse - start_pulse

    # Create updated channel descriptors with sub-aperture dimensions
    sub_channels = []
    for ch in metadata.channels:
        sub_channels.append(CPHDChannel(
            identifier=ch.identifier,
            num_vectors=n_pulses,
            num_samples=ch.num_samples,
            signal_array_byte_offset=ch.signal_array_byte_offset,
        ))

    return CPHDMetadata(
        format=metadata.format,
        rows=n_pulses,
        cols=metadata.cols,
        dtype=metadata.dtype,
        channels=sub_channels,
        pvp=sub_pvp,
        global_params=metadata.global_params,
        collection_info=metadata.collection_info,
        tx_waveform=metadata.tx_waveform,
        rcv_parameters=metadata.rcv_parameters,
        antenna_pattern=metadata.antenna_pattern,
        scene_coordinates=metadata.scene_coordinates,
        reference_geometry=metadata.reference_geometry,
        dwell=metadata.dwell,
        num_channels=metadata.num_channels,
        extras=metadata.extras,
    )
