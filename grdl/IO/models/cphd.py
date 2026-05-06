# -*- coding: utf-8 -*-
"""
CPHD Metadata - Typed metadata for Compensated Phase History Data.

Full coverage of NGA CPHD 1.1.0 (NGA.STND.0068-1) every required and
optional XML element. Top-level ``CPHDMetadata`` extends ``ImageMetadata``
and exposes one field per spec section: ``collection_info``, ``global_params``,
``scene_coordinates``, ``data``, ``channel_section``, ``pvp``,
``support_arrays``, ``dwell``, ``reference_geometry``, ``antenna``,
``tx_rcv``, ``error_parameters``, ``product_info``, ``geo_info``,
``match_info``.

Legacy single-instance accessors (``tx_waveform``, ``rcv_parameters``,
``antenna_pattern``, ``channels``, ``num_channels``) are preserved as
projections of the first element of the corresponding list, so existing
GRDL consumers (image_formation, contrast, discovery) continue to work.

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
2026-02-12

Modified
--------
2026-05-05
"""

# Standard library
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Tuple

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.models.base import ImageMetadata


# ===================================================================
# Reusable primitives
# ===================================================================

@dataclass
class CPHDParameter:
    """Free-form ``Parameter`` element with ``name`` attribute and string body."""

    name: str = ''
    value: str = ''


# ===================================================================
# 1. CollectionID  (CollectionIDType)
# ===================================================================

@dataclass
class CPHDCollectionInfo:
    """Collection information from the ``CollectionID`` section.

    Parameters
    ----------
    collector_name : str, optional
        Name of the collecting platform (required by spec, optional
        here for backward-compatibility).
    illuminator_name : str, optional
        Name of illuminator (bistatic collections).
    core_name : str, optional
        Unique collection identifier (required by spec).
    collect_type : str, optional
        ``'MONOSTATIC'`` or ``'BISTATIC'`` (required by spec).
    radar_mode : str, optional
        Radar mode: ``'SPOTLIGHT'``, ``'STRIPMAP'``, ``'DYNAMIC STRIPMAP'``.
    radar_mode_id : str, optional
        Mode identifier string.
    classification : str, optional
        Security classification banner (required by spec).
    release_info : str, optional
        Release authority info (required by spec).
    country_code : str, optional
        DIGEST GEC country code(s).
    parameters : List[CPHDParameter]
        Free-form name/value parameters.
    """

    collector_name: Optional[str] = None
    illuminator_name: Optional[str] = None
    core_name: Optional[str] = None
    collect_type: Optional[str] = None
    radar_mode: Optional[str] = None
    radar_mode_id: Optional[str] = None
    classification: Optional[str] = None
    release_info: Optional[str] = None
    country_code: Optional[str] = None
    parameters: List[CPHDParameter] = field(default_factory=list)


# ===================================================================
# 2. Global  (GlobalType)
# ===================================================================

@dataclass
class CPHDTimeline:
    """Collection timeline from ``Global/Timeline``.

    Parameters
    ----------
    collection_start : str, optional
        UTC dateTime when the collection started (ISO 8601 string).
    rcv_collection_start : str, optional
        UTC dateTime when receive collection started (bistatic).
    tx_time1 : float, optional
        Earliest TxTime over all vectors (s, relative to collection start).
    tx_time2 : float, optional
        Latest TxTime over all vectors (s, relative to collection start).
    """

    collection_start: Optional[str] = None
    rcv_collection_start: Optional[str] = None
    tx_time1: Optional[float] = None
    tx_time2: Optional[float] = None


@dataclass
class CPHDTropoParameters:
    """Tropospheric correction parameters from ``Global/TropoParameters``.

    Parameters
    ----------
    n0 : float, optional
        Refractivity at the reference height.
    ref_height : str, optional
        Reference height for ``n0``: ``'IARP'`` or ``'ZERO'``.
    """

    n0: Optional[float] = None
    ref_height: Optional[str] = None


@dataclass
class CPHDIonoParameters:
    """Ionospheric correction parameters from ``Global/IonoParameters``.

    Parameters
    ----------
    tecv : float, optional
        Total electron content (vertical), in TEC units.
    f2_height : float, optional
        F2 layer height (m).
    """

    tecv: Optional[float] = None
    f2_height: Optional[float] = None


@dataclass
class CPHDGlobal:
    """Global parameters from the CPHD ``Global`` section.

    Parameters
    ----------
    domain_type : str, optional
        Signal domain: ``'FX'`` (frequency) or ``'TOA'``.
    phase_sgn : int
        Phase sign convention from the CPHD standard (``+1`` or ``-1``).
        Default ``-1``.
    timeline : CPHDTimeline, optional
        Collection timeline.
    fx_band_min : float, optional
        Minimum FX frequency over all vectors (Hz).
    fx_band_max : float, optional
        Maximum FX frequency over all vectors (Hz).
    toa_swath_min : float, optional
        Minimum TOA over all vectors (s).
    toa_swath_max : float, optional
        Maximum TOA over all vectors (s).
    tropo_parameters : CPHDTropoParameters, optional
        Troposphere parameters.
    iono_parameters : CPHDIonoParameters, optional
        Ionosphere parameters.
    """

    domain_type: Optional[str] = None
    phase_sgn: int = -1
    timeline: Optional[CPHDTimeline] = None
    fx_band_min: Optional[float] = None
    fx_band_max: Optional[float] = None
    toa_swath_min: Optional[float] = None
    toa_swath_max: Optional[float] = None
    tropo_parameters: Optional[CPHDTropoParameters] = None
    iono_parameters: Optional[CPHDIonoParameters] = None

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
# 3. SceneCoordinates  (SceneCoordinatesType)
# ===================================================================

@dataclass
class CPHDPlanarSurface:
    """Planar reference surface from ``SceneCoordinates/ReferenceSurface/Planar``.

    Parameters
    ----------
    u_iax : np.ndarray, optional
        Unit vector IAX in ECF, shape ``(3,)``.
    u_iay : np.ndarray, optional
        Unit vector IAY in ECF, shape ``(3,)``.
    """

    u_iax: Optional[np.ndarray] = None
    u_iay: Optional[np.ndarray] = None


@dataclass
class CPHDHAESurface:
    """HAE (curved) reference surface from ``SceneCoordinates/ReferenceSurface/HAE``.

    Parameters
    ----------
    u_iax_ll : Tuple[float, float], optional
        Unit IAX as ``(lat_deg, lon_deg)``.
    u_iay_ll : Tuple[float, float], optional
        Unit IAY as ``(lat_deg, lon_deg)``.
    """

    u_iax_ll: Optional[Tuple[float, float]] = None
    u_iay_ll: Optional[Tuple[float, float]] = None


@dataclass
class CPHDReferenceSurface:
    """Reference surface (Planar XOR HAE)."""

    planar: Optional[CPHDPlanarSurface] = None
    hae: Optional[CPHDHAESurface] = None


@dataclass
class CPHDImageArea:
    """Image area / extended area bounding rectangle (and optional polygon)."""

    x1y1: Optional[Tuple[float, float]] = None
    x2y2: Optional[Tuple[float, float]] = None
    polygon: Optional[np.ndarray] = None  # shape (N, 2) of (X, Y)


@dataclass
class CPHDIASegment:
    """Image grid segment description."""

    identifier: str = ''
    start_line: int = 0
    start_sample: int = 0
    end_line: int = 0
    end_sample: int = 0
    segment_polygon: Optional[np.ndarray] = None  # shape (N, 2) of (line, sample)


@dataclass
class CPHDImageGrid:
    """Image grid description from ``SceneCoordinates/ImageGrid``."""

    identifier: Optional[str] = None
    iarp_location: Optional[Tuple[float, float]] = None  # (line, sample)
    line_spacing: Optional[float] = None
    first_line: Optional[int] = None
    num_lines: Optional[int] = None
    sample_spacing: Optional[float] = None
    first_sample: Optional[int] = None
    num_samples: Optional[int] = None
    segments: List[CPHDIASegment] = field(default_factory=list)


@dataclass
class CPHDSceneCoordinates:
    """Scene coordinate information from ``SceneCoordinates``.

    Combines required fields (earth model, IARP, reference surface, image
    area, four corner points) and optional structures (extended area,
    image grid).

    Parameters
    ----------
    earth_model : str, optional
        Earth model identifier (always ``'WGS_84'`` in CPHD 1.1.0).
    iarp_ecf : np.ndarray, optional
        IARP in ECF meters, shape ``(3,)``.
    iarp_llh : np.ndarray, optional
        IARP as ``[lat_deg, lon_deg, hae_m]``.
    reference_surface : CPHDReferenceSurface, optional
        Planar XOR HAE reference surface descriptor.
    image_area : CPHDImageArea, optional
        Bounding image area in IA coordinates.
    image_area_x : tuple, optional
        Convenience: ``(x_min, x_max)`` derived from ``image_area``.
    image_area_y : tuple, optional
        Convenience: ``(y_min, y_max)`` derived from ``image_area``.
    corner_points : np.ndarray, optional
        Four image-area corner lat/lon points, shape ``(4, 2)``.
    extended_area : CPHDImageArea, optional
        Optional extended area descriptor.
    image_grid : CPHDImageGrid, optional
        Optional image grid description.
    """

    earth_model: Optional[str] = None
    iarp_ecf: Optional[np.ndarray] = None
    iarp_llh: Optional[np.ndarray] = None
    reference_surface: Optional[CPHDReferenceSurface] = None
    image_area: Optional[CPHDImageArea] = None
    image_area_x: Optional[tuple] = None
    image_area_y: Optional[tuple] = None
    corner_points: Optional[np.ndarray] = None
    extended_area: Optional[CPHDImageArea] = None
    image_grid: Optional[CPHDImageGrid] = None


# ===================================================================
# 4. Data  (DataType)
# ===================================================================

@dataclass
class CPHDChannel:
    """Per-channel size descriptor from the ``Data`` section.

    This corresponds to the ``ChannelSizeType`` element. The full
    per-channel parameter set lives in :class:`CPHDChannelParameters`
    (under the ``Channel`` section).

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
    pvp_array_byte_offset : int, optional
        Byte offset of the PVP array in the file.
    compressed_signal_size : int, optional
        Bytes when the signal array is compressed.
    """

    identifier: str = ''
    num_vectors: int = 0
    num_samples: int = 0
    signal_array_byte_offset: Optional[int] = None
    pvp_array_byte_offset: Optional[int] = None
    compressed_signal_size: Optional[int] = None


@dataclass
class CPHDSupportArraySize:
    """Per-support-array size descriptor from the ``Data`` section."""

    identifier: str = ''
    num_rows: int = 0
    num_cols: int = 0
    bytes_per_element: int = 0
    array_byte_offset: int = 0


@dataclass
class CPHDData:
    """``Data`` section: signal-array binary layout for the CPHD file.

    Parameters
    ----------
    signal_array_format : str, optional
        Signal sample binary format: ``'CI2'``, ``'CI4'``, ``'CF8'``.
    num_bytes_pvp : int, optional
        Bytes per PVP record.
    num_cphd_channels : int, optional
        Number of CPHD signal channels.
    signal_compression_id : str, optional
        Optional compression-scheme identifier.
    channels : List[CPHDChannel]
        Per-channel size descriptors.
    num_support_arrays : int, optional
        Number of support arrays.
    support_arrays : List[CPHDSupportArraySize]
        Per-support-array size descriptors.
    """

    signal_array_format: Optional[str] = None
    num_bytes_pvp: Optional[int] = None
    num_cphd_channels: Optional[int] = None
    signal_compression_id: Optional[str] = None
    channels: List[CPHDChannel] = field(default_factory=list)
    num_support_arrays: Optional[int] = None
    support_arrays: List[CPHDSupportArraySize] = field(default_factory=list)


# ===================================================================
# 5. Channel  (ChannelType / ChannelParametersType)
# ===================================================================

@dataclass
class CPHDPolRef:
    """Polarization reference (AmpH, AmpV, PhaseV)."""

    amp_h: Optional[float] = None
    amp_v: Optional[float] = None
    phase_v: Optional[float] = None


@dataclass
class CPHDPolarization:
    """Channel polarization."""

    tx_pol: Optional[str] = None
    rcv_pol: Optional[str] = None
    tx_pol_ref: Optional[CPHDPolRef] = None
    rcv_pol_ref: Optional[CPHDPolRef] = None


@dataclass
class CPHDLFMEclipse:
    """LFM eclipse parameters under ``ChannelParameters/TOAExtended``."""

    fx_early_low: Optional[float] = None
    fx_early_high: Optional[float] = None
    fx_late_low: Optional[float] = None
    fx_late_high: Optional[float] = None


@dataclass
class CPHDTOAExtended:
    """Extended TOA description."""

    toa_ext_saved: Optional[float] = None
    lfm_eclipse: Optional[CPHDLFMEclipse] = None


@dataclass
class CPHDDwellTimesRef:
    """References to Dwell polynomials for one channel."""

    cod_id: Optional[str] = None
    dwell_id: Optional[str] = None
    dta_id: Optional[str] = None
    use_dta: Optional[bool] = None


@dataclass
class CPHDChannelAntennaRefs:
    """Per-channel antenna ID references."""

    tx_apc_id: Optional[str] = None
    tx_apat_id: Optional[str] = None
    rcv_apc_id: Optional[str] = None
    rcv_apat_id: Optional[str] = None


@dataclass
class CPHDChannelTxRcvRefs:
    """Per-channel TxRcv ID references."""

    tx_wf_ids: List[str] = field(default_factory=list)
    rcv_ids: List[str] = field(default_factory=list)


@dataclass
class CPHDFxNoiseProfilePoint:
    """One ``(Fx, PN)`` point of a noise profile."""

    fx: Optional[float] = None
    pn: Optional[float] = None


@dataclass
class CPHDNoiseLevel:
    """Per-channel noise level."""

    pn_ref: Optional[float] = None
    bn_ref: Optional[float] = None
    fx_noise_profile: List[CPHDFxNoiseProfilePoint] = field(default_factory=list)


@dataclass
class CPHDChannelParameters:
    """Full ``ChannelParametersType`` per channel."""

    identifier: Optional[str] = None
    ref_vector_index: Optional[int] = None
    fx_fixed: Optional[bool] = None
    toa_fixed: Optional[bool] = None
    srp_fixed: Optional[bool] = None
    signal_normal: Optional[bool] = None
    polarization: Optional[CPHDPolarization] = None
    fx_c: Optional[float] = None
    fx_bw: Optional[float] = None
    fx_bw_noise: Optional[float] = None
    toa_saved: Optional[float] = None
    toa_extended: Optional[CPHDTOAExtended] = None
    dwell_times: Optional[CPHDDwellTimesRef] = None
    image_area: Optional[CPHDImageArea] = None
    antenna: Optional[CPHDChannelAntennaRefs] = None
    tx_rcv: Optional[CPHDChannelTxRcvRefs] = None
    pt_ref: Optional[float] = None
    noise_level: Optional[CPHDNoiseLevel] = None


@dataclass
class CPHDChannelSection:
    """Top-level ``Channel`` section."""

    ref_ch_id: Optional[str] = None
    fx_fixed_cphd: Optional[bool] = None
    toa_fixed_cphd: Optional[bool] = None
    srp_fixed_cphd: Optional[bool] = None
    parameters: List[CPHDChannelParameters] = field(default_factory=list)
    added_parameters: List[CPHDParameter] = field(default_factory=list)


# ===================================================================
# 6. PVP  (PVPType)  — the per-vector parameter ARRAYS
# ===================================================================

@dataclass
class CPHDPVP:
    """Per-Vector Parameters arrays indexed by pulse number.

    All position arrays have shape ``(N, 3)`` in ECF meters.
    All velocity arrays have shape ``(N, 3)`` in ECF m/s.
    Scalar-per-pulse arrays have shape ``(N,)``.

    Required PVPs (per CPHD 1.1.0): ``tx_time``, ``tx_pos``, ``tx_vel``,
    ``rcv_time``, ``rcv_pos``, ``rcv_vel``, ``srp_pos``, ``a_fdop``,
    ``a_frr1``, ``a_frr2``, ``fx1``, ``fx2``, ``toa1``, ``toa2``,
    ``td_tropo_srp``, ``sc0``, ``scss``.
    Optional PVPs: ``amp_sf``, ``signal``, ``fxn1``, ``fxn2``, ``toae1``,
    ``toae2``, ``td_iono_srp``, ``tx_acx``/``tx_acy``/``tx_eb`` (TxAntenna),
    ``rcv_acx``/``rcv_acy``/``rcv_eb`` (RcvAntenna), and any user-defined
    PVPs in ``added_pvps``.
    """

    tx_time: Optional[np.ndarray] = None
    tx_pos: Optional[np.ndarray] = None
    tx_vel: Optional[np.ndarray] = None
    rcv_time: Optional[np.ndarray] = None
    rcv_pos: Optional[np.ndarray] = None
    rcv_vel: Optional[np.ndarray] = None
    srp_pos: Optional[np.ndarray] = None
    fx1: Optional[np.ndarray] = None
    fx2: Optional[np.ndarray] = None
    toa1: Optional[np.ndarray] = None
    toa2: Optional[np.ndarray] = None
    td_tropo_srp: Optional[np.ndarray] = None
    sc0: Optional[np.ndarray] = None
    scss: Optional[np.ndarray] = None
    a_fdop: Optional[np.ndarray] = None
    a_frr1: Optional[np.ndarray] = None
    a_frr2: Optional[np.ndarray] = None
    amp_sf: Optional[np.ndarray] = None
    signal: Optional[np.ndarray] = None
    fxn1: Optional[np.ndarray] = None
    fxn2: Optional[np.ndarray] = None
    toae1: Optional[np.ndarray] = None
    toae2: Optional[np.ndarray] = None
    td_iono_srp: Optional[np.ndarray] = None
    tx_acx: Optional[np.ndarray] = None
    tx_acy: Optional[np.ndarray] = None
    tx_eb: Optional[np.ndarray] = None
    rcv_acx: Optional[np.ndarray] = None
    rcv_acy: Optional[np.ndarray] = None
    rcv_eb: Optional[np.ndarray] = None
    added_pvps: Dict[str, np.ndarray] = field(default_factory=dict)

    _FIELDS_1D: ClassVar[Tuple[str, ...]] = (
        'tx_time', 'rcv_time', 'fx1', 'fx2', 'toa1', 'toa2',
        'td_tropo_srp', 'sc0', 'scss',
        'a_fdop', 'a_frr1', 'a_frr2',
        'amp_sf', 'signal',
        'fxn1', 'fxn2', 'toae1', 'toae2', 'td_iono_srp',
    )
    _FIELDS_2D: ClassVar[Tuple[str, ...]] = (
        'tx_pos', 'tx_vel', 'rcv_pos', 'rcv_vel', 'srp_pos',
        'tx_acx', 'tx_acy', 'tx_eb',
        'rcv_acx', 'rcv_acy', 'rcv_eb',
    )

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
        """Index of the first pulse with ``signal > 0``.

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
        """
        start = self.first_valid_pulse
        if start == 0:
            return self
        n = self.num_vectors - start
        return self.slice(start, start + n)

    def slice(self, start: int, end: int) -> 'CPHDPVP':
        """Return a new CPHDPVP with arrays sliced to ``[start:end]``."""
        n = self.num_vectors
        if start < 0 or end > n or start >= end:
            raise ValueError(
                f"Invalid slice [{start}:{end}] for PVP with "
                f"{n} vectors"
            )

        kwargs: Dict[str, Optional[np.ndarray]] = {}
        for name in self._FIELDS_1D:
            arr = getattr(self, name)
            kwargs[name] = arr[start:end].copy() if arr is not None else None
        for name in self._FIELDS_2D:
            arr = getattr(self, name)
            kwargs[name] = (
                arr[start:end, :].copy() if arr is not None else None
            )

        added = {
            k: (v[start:end].copy() if v.ndim == 1 else v[start:end, :].copy())
            for k, v in self.added_pvps.items()
            if v is not None
        }
        return CPHDPVP(added_pvps=added, **kwargs)


# ===================================================================
# 7. SupportArray  (SupportArrayType)
# ===================================================================

@dataclass
class _CPHDSupportArrayCore:
    """Core grid descriptor common to all SupportArray entries."""

    identifier: str = ''
    element_format: Optional[str] = None
    x0: Optional[float] = None
    y0: Optional[float] = None
    xss: Optional[float] = None
    yss: Optional[float] = None
    nodata: Optional[bytes] = None


@dataclass
class CPHDIAZArray(_CPHDSupportArrayCore):
    """``IAZArray`` support array (height grid)."""


@dataclass
class CPHDAntGainPhaseArray(_CPHDSupportArrayCore):
    """``AntGainPhase`` support array."""


@dataclass
class CPHDDwellTimeArray(_CPHDSupportArrayCore):
    """``DwellTimeArray`` support array."""


@dataclass
class CPHDAddedSupportArray(_CPHDSupportArrayCore):
    """User-defined support array."""

    x_units: Optional[str] = None
    y_units: Optional[str] = None
    z_units: Optional[str] = None
    parameters: List[CPHDParameter] = field(default_factory=list)


@dataclass
class CPHDSupportArrays:
    """Container for the four kinds of optional support arrays."""

    iaz_arrays: List[CPHDIAZArray] = field(default_factory=list)
    ant_gain_phase: List[CPHDAntGainPhaseArray] = field(default_factory=list)
    dwell_time_arrays: List[CPHDDwellTimeArray] = field(default_factory=list)
    added_support_arrays: List[CPHDAddedSupportArray] = field(
        default_factory=list,
    )


# ===================================================================
# 8. Dwell  (DwellType)
# ===================================================================

@dataclass
class CPHDCODTime:
    """One named center-of-dwell time polynomial (Poly2D in IAX, IAY)."""

    identifier: str = ''
    cod_time_poly: Optional[np.ndarray] = None  # 2D


@dataclass
class CPHDDwellTime:
    """One named dwell-duration polynomial (Poly2D in IAX, IAY)."""

    identifier: str = ''
    dwell_time_poly: Optional[np.ndarray] = None  # 2D


@dataclass
class CPHDDwell:
    """Top-level ``Dwell`` section.

    Holds lists of named CODTime and DwellTime polynomials. Provides
    ``cod_time_poly`` and ``dwell_time_poly`` properties for legacy
    callers expecting a single polynomial — these resolve to the first
    list entry's polynomial.
    """

    num_cod_times: Optional[int] = None
    cod_times: List[CPHDCODTime] = field(default_factory=list)
    num_dwell_times: Optional[int] = None
    dwell_times: List[CPHDDwellTime] = field(default_factory=list)

    @property
    def cod_time_poly(self) -> Optional[np.ndarray]:
        """First CODTime polynomial, or None."""
        if self.cod_times:
            return self.cod_times[0].cod_time_poly
        return None

    @property
    def dwell_time_poly(self) -> Optional[np.ndarray]:
        """First DwellTime polynomial, or None."""
        if self.dwell_times:
            return self.dwell_times[0].dwell_time_poly
        return None


# Legacy alias kept importable from grdl.IO.models.cphd
CPHDDwellPolynomial = CPHDDwell


# ===================================================================
# 9. ReferenceGeometry  (ReferenceGeometryType)
# ===================================================================

@dataclass
class CPHDMonostaticGeometry:
    """Full monostatic reference geometry."""

    arp_pos: Optional[np.ndarray] = None  # (3,)
    arp_vel: Optional[np.ndarray] = None  # (3,)
    side_of_track: Optional[str] = None
    slant_range: Optional[float] = None
    ground_range: Optional[float] = None
    doppler_cone_angle: Optional[float] = None
    graze_angle: Optional[float] = None
    incidence_angle: Optional[float] = None
    azimuth_angle: Optional[float] = None
    twist_angle: Optional[float] = None
    slope_angle: Optional[float] = None
    layover_angle: Optional[float] = None


@dataclass
class CPHDBistaticPlatform:
    """One platform's geometry inside a bistatic ReferenceGeometry."""

    time: Optional[float] = None
    pos: Optional[np.ndarray] = None  # (3,)
    vel: Optional[np.ndarray] = None  # (3,)
    side_of_track: Optional[str] = None
    slant_range: Optional[float] = None
    ground_range: Optional[float] = None
    doppler_cone_angle: Optional[float] = None
    graze_angle: Optional[float] = None
    incidence_angle: Optional[float] = None
    azimuth_angle: Optional[float] = None


@dataclass
class CPHDBistaticGeometry:
    """Full bistatic reference geometry."""

    azimuth_angle: Optional[float] = None
    azimuth_angle_rate: Optional[float] = None
    bistatic_angle: Optional[float] = None
    bistatic_angle_rate: Optional[float] = None
    graze_angle: Optional[float] = None
    twist_angle: Optional[float] = None
    slope_angle: Optional[float] = None
    layover_angle: Optional[float] = None
    tx_platform: Optional[CPHDBistaticPlatform] = None
    rcv_platform: Optional[CPHDBistaticPlatform] = None


@dataclass
class CPHDReferenceGeometry:
    """Reference geometry at center of dwell.

    Combines the shared SRP / reference-time fields with optional
    monostatic XOR bistatic detail. The flat ``*_deg`` legacy fields
    mirror ``monostatic`` for backward compatibility with existing GRDL
    consumers.
    """

    ref_time: Optional[float] = None
    srp_ecf: Optional[np.ndarray] = None  # (3,)
    srp_iac: Optional[np.ndarray] = None  # (3,)
    srp_llh: Optional[np.ndarray] = None  # (3,)
    srp_cod_time: Optional[float] = None
    srp_dwell_time: Optional[float] = None
    side_of_track: Optional[str] = None
    graze_angle_deg: Optional[float] = None
    azimuth_angle_deg: Optional[float] = None
    twist_angle_deg: Optional[float] = None
    slope_angle_deg: Optional[float] = None
    layover_angle_deg: Optional[float] = None
    monostatic: Optional[CPHDMonostaticGeometry] = None
    bistatic: Optional[CPHDBistaticGeometry] = None


# ===================================================================
# 10. Antenna  (AntennaType)
# ===================================================================

@dataclass
class CPHDAntCoordFrame:
    """``AntCoordFrame`` entry."""

    identifier: str = ''
    x_axis_poly: Optional[np.ndarray] = None  # (K, 3)
    y_axis_poly: Optional[np.ndarray] = None  # (K, 3)
    use_acf_pvp: Optional[bool] = None


@dataclass
class CPHDAntPhaseCenter:
    """``AntPhaseCenter`` entry."""

    identifier: str = ''
    acf_id: Optional[str] = None
    apc_xyz: Optional[np.ndarray] = None  # (3,)


@dataclass
class CPHDEBFreqShiftSF:
    """Electrical-boresight frequency-shift scale factors."""

    dcx_sf: Optional[float] = None
    dcy_sf: Optional[float] = None


@dataclass
class CPHDMLFreqDilationSF:
    """Mainlobe frequency-dilation scale factors."""

    dcx_sf: Optional[float] = None
    dcy_sf: Optional[float] = None


@dataclass
class CPHDAntPolRef:
    """Antenna polarization reference (AmpX, AmpY, PhaseY)."""

    amp_x: Optional[float] = None
    amp_y: Optional[float] = None
    phase_y: Optional[float] = None


@dataclass
class CPHDAntEB:
    """Electrical-boresight steering polynomials."""

    dcx_poly: Optional[np.ndarray] = None  # 1D
    dcy_poly: Optional[np.ndarray] = None  # 1D
    use_eb_pvp: Optional[bool] = None


@dataclass
class CPHDAntGainPhasePoly:
    """Gain/phase polynomials with an optional support-array reference."""

    gain_poly: Optional[np.ndarray] = None  # 2D
    phase_poly: Optional[np.ndarray] = None  # 2D
    ant_gp_id: Optional[str] = None


@dataclass
class CPHDAntGainPhaseFreqEntry:
    """Per-frequency entry of ``AntPattern/GainPhaseArray``."""

    freq: Optional[float] = None
    array_id: Optional[str] = None
    element_id: Optional[str] = None


@dataclass
class CPHDAntPatternFull:
    """Full ``AntPattern`` entry."""

    identifier: str = ''
    freq_zero: Optional[float] = None
    gain_zero: Optional[float] = None
    eb_freq_shift: Optional[bool] = None
    eb_freq_shift_sf: Optional[CPHDEBFreqShiftSF] = None
    ml_freq_dilation: Optional[bool] = None
    ml_freq_dilation_sf: Optional[CPHDMLFreqDilationSF] = None
    gain_bs_poly: Optional[np.ndarray] = None  # 1D
    ant_pol_ref: Optional[CPHDAntPolRef] = None
    eb: Optional[CPHDAntEB] = None
    array: Optional[CPHDAntGainPhasePoly] = None
    element: Optional[CPHDAntGainPhasePoly] = None
    gain_phase_arrays: List[CPHDAntGainPhaseFreqEntry] = field(
        default_factory=list,
    )


@dataclass
class CPHDAntenna:
    """Top-level ``Antenna`` section."""

    num_acfs: Optional[int] = None
    num_apcs: Optional[int] = None
    num_ant_pats: Optional[int] = None
    ant_coord_frames: List[CPHDAntCoordFrame] = field(default_factory=list)
    ant_phase_centers: List[CPHDAntPhaseCenter] = field(default_factory=list)
    ant_patterns: List[CPHDAntPatternFull] = field(default_factory=list)


@dataclass
class CPHDAntennaPattern:
    """Legacy first-pattern projection used by ``rda.py``.

    Populated by the reader from ``Antenna/AntPattern[0]`` and the first
    ``AntCoordFrame``. Kept for backward compatibility — the full
    multi-pattern hierarchy is available via :class:`CPHDAntenna`.
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
# 11. TxRcv  (TxRcvType)
# ===================================================================

@dataclass
class CPHDTxWaveform:
    """One ``TxWFParameters`` entry."""

    identifier: Optional[str] = None
    pulse_length: Optional[float] = None
    rf_bandwidth: Optional[float] = None
    freq_center: Optional[float] = None
    lfm_rate: Optional[float] = None
    polarization: Optional[str] = None
    power: Optional[float] = None


@dataclass
class CPHDRcvParameters:
    """One ``RcvParameters`` entry."""

    identifier: Optional[str] = None
    window_length: Optional[float] = None
    sample_rate: Optional[float] = None
    if_filter_bw: Optional[float] = None
    freq_center: Optional[float] = None
    lfm_rate: Optional[float] = None
    polarization: Optional[str] = None
    path_gain: Optional[float] = None


@dataclass
class CPHDTxRcv:
    """Top-level ``TxRcv`` section."""

    num_tx_wfs: Optional[int] = None
    tx_waveforms: List[CPHDTxWaveform] = field(default_factory=list)
    num_rcvs: Optional[int] = None
    rcv_parameters: List[CPHDRcvParameters] = field(default_factory=list)


# ===================================================================
# 12. ErrorParameters  (ErrorParametersType)
# ===================================================================

@dataclass
class CPHDErrorDecorrFunc:
    """``ErrorDecorrFuncType`` — decorrelation rate function."""

    corr_coef_zero: Optional[float] = None
    decorr_rate: Optional[float] = None


@dataclass
class CPHDPosVelCorrCoefs:
    """15 cross-correlation coefficients between (P1,P2,P3,V1,V2,V3)."""

    p1_p2: Optional[float] = None
    p1_p3: Optional[float] = None
    p1_v1: Optional[float] = None
    p1_v2: Optional[float] = None
    p1_v3: Optional[float] = None
    p2_p3: Optional[float] = None
    p2_v1: Optional[float] = None
    p2_v2: Optional[float] = None
    p2_v3: Optional[float] = None
    p3_v1: Optional[float] = None
    p3_v2: Optional[float] = None
    p3_v3: Optional[float] = None
    v1_v2: Optional[float] = None
    v1_v3: Optional[float] = None
    v2_v3: Optional[float] = None


@dataclass
class CPHDPosVelErr:
    """Position / velocity error parameters."""

    frame: Optional[str] = None
    p1: Optional[float] = None
    p2: Optional[float] = None
    p3: Optional[float] = None
    v1: Optional[float] = None
    v2: Optional[float] = None
    v3: Optional[float] = None
    corr_coefs: Optional[CPHDPosVelCorrCoefs] = None
    position_decorr: Optional[CPHDErrorDecorrFunc] = None


@dataclass
class CPHDMonoRadarSensorError:
    """Monostatic radar-sensor error parameters."""

    range_bias: Optional[float] = None
    clock_freq_sf: Optional[float] = None
    collection_start_time: Optional[float] = None
    range_bias_decorr: Optional[CPHDErrorDecorrFunc] = None


@dataclass
class CPHDTropoError:
    """Tropospheric error parameters."""

    tropo_range_vertical: Optional[float] = None
    tropo_range_slant: Optional[float] = None
    tropo_range_decorr: Optional[CPHDErrorDecorrFunc] = None


@dataclass
class CPHDIonoError:
    """Ionospheric error parameters."""

    iono_range_vertical: Optional[float] = None
    iono_range_rate_vertical: Optional[float] = None
    iono_rg_rg_rate_cc: Optional[float] = None
    iono_range_vert_decorr: Optional[CPHDErrorDecorrFunc] = None


@dataclass
class CPHDMonostaticError:
    """``ErrorParameters/Monostatic``."""

    pos_vel_err: Optional[CPHDPosVelErr] = None
    radar_sensor: Optional[CPHDMonoRadarSensorError] = None
    tropo_error: Optional[CPHDTropoError] = None
    iono_error: Optional[CPHDIonoError] = None
    added_parameters: List[CPHDParameter] = field(default_factory=list)


@dataclass
class CPHDBistaticRadarSensorError:
    """Bistatic radar-sensor error parameters."""

    delay_bias: Optional[float] = None
    clock_freq_sf: Optional[float] = None
    collection_start_time: Optional[float] = None


@dataclass
class CPHDBistaticPlatformError:
    """One platform's error block inside ErrorParameters/Bistatic."""

    pos_vel_err: Optional[CPHDPosVelErr] = None
    radar_sensor: Optional[CPHDBistaticRadarSensorError] = None


@dataclass
class CPHDBistaticError:
    """``ErrorParameters/Bistatic``."""

    tx_platform: Optional[CPHDBistaticPlatformError] = None
    rcv_platform: Optional[CPHDBistaticPlatformError] = None
    added_parameters: List[CPHDParameter] = field(default_factory=list)


@dataclass
class CPHDErrorParameters:
    """Top-level ``ErrorParameters`` section."""

    monostatic: Optional[CPHDMonostaticError] = None
    bistatic: Optional[CPHDBistaticError] = None


# ===================================================================
# 13. ProductInfo  (ProductInfoType)
# ===================================================================

@dataclass
class CPHDCreationInfo:
    """One ``CreationInfo`` event."""

    application: Optional[str] = None
    date_time: Optional[str] = None
    site: Optional[str] = None
    parameters: List[CPHDParameter] = field(default_factory=list)


@dataclass
class CPHDProductInfo:
    """``ProductInfo`` section."""

    profile: Optional[str] = None
    creation_info: List[CPHDCreationInfo] = field(default_factory=list)
    parameters: List[CPHDParameter] = field(default_factory=list)


# ===================================================================
# 14. GeoInfo  (GeoInfoType, recursive)
# ===================================================================

@dataclass
class CPHDGeoLine:
    """``LineType`` — sequence of lat/lon endpoints (>= 2)."""

    endpoints: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class CPHDGeoPolygon:
    """``LatLonPolygonType`` — sequence of lat/lon vertices (>= 3)."""

    vertices: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class CPHDGeoInfo:
    """``GeoInfo`` block (recursive)."""

    name: Optional[str] = None
    desc: List[CPHDParameter] = field(default_factory=list)
    points: List[Tuple[float, float]] = field(default_factory=list)
    lines: List[CPHDGeoLine] = field(default_factory=list)
    polygons: List[CPHDGeoPolygon] = field(default_factory=list)
    geo_infos: List['CPHDGeoInfo'] = field(default_factory=list)


# ===================================================================
# 15. MatchInfo  (MatchInfoType)
# ===================================================================

@dataclass
class CPHDMatchCollection:
    """One ``MatchCollection`` entry."""

    index: Optional[int] = None
    core_name: Optional[str] = None
    match_index: Optional[int] = None
    parameters: List[CPHDParameter] = field(default_factory=list)


@dataclass
class CPHDMatchType:
    """One ``MatchType`` entry."""

    index: Optional[int] = None
    type_id: Optional[str] = None
    current_index: Optional[int] = None
    num_match_collections: Optional[int] = None
    match_collections: List[CPHDMatchCollection] = field(default_factory=list)


@dataclass
class CPHDMatchInfo:
    """``MatchInfo`` section."""

    num_match_types: Optional[int] = None
    match_types: List[CPHDMatchType] = field(default_factory=list)


# ===================================================================
# Top-level CPHDMetadata
# ===================================================================

@dataclass
class CPHDMetadata(ImageMetadata):
    """Typed metadata for CPHD (Compensated Phase History Data) files.

    Extends ``ImageMetadata`` with one field per CPHD 1.1.0 top-level
    XML section. The base ``rows``/``cols`` correspond to the first
    channel's ``num_vectors``/``num_samples``.

    Required spec sections (always populated when reading a valid CPHD
    1.1.0 file): ``collection_info``, ``global_params``,
    ``scene_coordinates``, ``data``, ``channel_section``, ``pvp``,
    ``dwell``, ``reference_geometry``.

    Optional spec sections: ``support_arrays``, ``antenna``, ``tx_rcv``,
    ``error_parameters``, ``product_info``, ``geo_info``, ``match_info``.

    Backward-compat projections of single-instance items: ``channels``
    (mirrors ``data.channels``), ``num_channels``, ``tx_waveform``
    (first ``tx_rcv.tx_waveforms``), ``rcv_parameters`` (first
    ``tx_rcv.rcv_parameters``), ``antenna_pattern`` (first
    ``antenna.ant_patterns`` projected to legacy fields).

    Examples
    --------
    >>> from grdl.IO.sar import CPHDReader
    >>> with CPHDReader('data.cphd') as reader:
    ...     meta = reader.metadata
    ...     meta.pvp.tx_pos               # (N, 3) ECF transmit positions
    ...     meta.global_params.bandwidth  # Hz
    ...     meta.data.signal_array_format # 'CI2' / 'CI4' / 'CF8'
    ...     meta.channel_section.parameters[0].fx_c
    ...     meta.tx_rcv.tx_waveforms[0].rf_bandwidth
    """

    # Required spec sections
    collection_info: Optional[CPHDCollectionInfo] = None
    global_params: Optional[CPHDGlobal] = None
    scene_coordinates: Optional[CPHDSceneCoordinates] = None
    data: Optional[CPHDData] = None
    channel_section: Optional[CPHDChannelSection] = None
    pvp: Optional[CPHDPVP] = None
    dwell: Optional[CPHDDwell] = None
    reference_geometry: Optional[CPHDReferenceGeometry] = None

    # Optional spec sections
    support_arrays: Optional[CPHDSupportArrays] = None
    antenna: Optional[CPHDAntenna] = None
    tx_rcv: Optional[CPHDTxRcv] = None
    error_parameters: Optional[CPHDErrorParameters] = None
    product_info: Optional[CPHDProductInfo] = None
    geo_info: List[CPHDGeoInfo] = field(default_factory=list)
    match_info: Optional[CPHDMatchInfo] = None

    # Legacy / convenience projections
    channels: List[CPHDChannel] = field(default_factory=list)
    num_channels: int = 0
    tx_waveform: Optional[CPHDTxWaveform] = None
    rcv_parameters: Optional[CPHDRcvParameters] = None
    antenna_pattern: Optional[CPHDAntennaPattern] = None


# ===================================================================
# Subaperture metadata factory
# ===================================================================

def create_subaperture_metadata(
    metadata: CPHDMetadata,
    start_pulse: int,
    end_pulse: int,
) -> CPHDMetadata:
    """Create a sub-aperture metadata slice for stripmap processing.

    Slices the PVP arrays and updates channel dimensions while
    preserving every other CPHD section unchanged.

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
        New metadata with sliced PVP and updated channel dimensions.

    Raises
    ------
    ValueError
        If metadata has no PVP or indices are invalid.
    """
    if metadata.pvp is None:
        raise ValueError("CPHDMetadata must have populated PVP arrays")

    sub_pvp = metadata.pvp.slice(start_pulse, end_pulse)
    n_pulses = end_pulse - start_pulse

    sub_channels = [
        CPHDChannel(
            identifier=ch.identifier,
            num_vectors=n_pulses,
            num_samples=ch.num_samples,
            signal_array_byte_offset=ch.signal_array_byte_offset,
            pvp_array_byte_offset=ch.pvp_array_byte_offset,
            compressed_signal_size=ch.compressed_signal_size,
        )
        for ch in metadata.channels
    ]

    sub_data = None
    if metadata.data is not None:
        sub_data = CPHDData(
            signal_array_format=metadata.data.signal_array_format,
            num_bytes_pvp=metadata.data.num_bytes_pvp,
            num_cphd_channels=metadata.data.num_cphd_channels,
            signal_compression_id=metadata.data.signal_compression_id,
            channels=sub_channels,
            num_support_arrays=metadata.data.num_support_arrays,
            support_arrays=list(metadata.data.support_arrays),
        )

    return CPHDMetadata(
        format=metadata.format,
        rows=n_pulses,
        cols=metadata.cols,
        dtype=metadata.dtype,
        collection_info=metadata.collection_info,
        global_params=metadata.global_params,
        scene_coordinates=metadata.scene_coordinates,
        data=sub_data,
        channel_section=metadata.channel_section,
        pvp=sub_pvp,
        dwell=metadata.dwell,
        reference_geometry=metadata.reference_geometry,
        support_arrays=metadata.support_arrays,
        antenna=metadata.antenna,
        tx_rcv=metadata.tx_rcv,
        error_parameters=metadata.error_parameters,
        product_info=metadata.product_info,
        geo_info=list(metadata.geo_info),
        match_info=metadata.match_info,
        channels=sub_channels,
        num_channels=metadata.num_channels,
        tx_waveform=metadata.tx_waveform,
        rcv_parameters=metadata.rcv_parameters,
        antenna_pattern=metadata.antenna_pattern,
        extras=dict(metadata.extras) if metadata.extras else {},
    )
