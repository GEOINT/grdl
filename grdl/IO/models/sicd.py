# -*- coding: utf-8 -*-
"""
SICD Metadata - Complete typed metadata for SICD imagery.

Nested dataclasses mirroring all 17 sections of the NGA SICD standard
as implemented by sarpy/sarkit. Every field from the SICD XML schema
is represented as a typed attribute.

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
2026-02-10

Modified
--------
2026-02-10
"""

# Standard library
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.models.base import ImageMetadata
from grdl.IO.models.common import (
    XYZ,
    LatLon,
    LatLonHAE,
    RowCol,
    Poly1D,
    Poly2D,
    XYZPoly,
)


# ===================================================================
# Section 1: CollectionInfo
# ===================================================================

@dataclass
class SICDRadarMode:
    """Radar collection mode.

    Parameters
    ----------
    mode_type : str, optional
        Mode type: ``'SPOTLIGHT'``, ``'STRIPMAP'``,
        ``'DYNAMIC STRIPMAP'``.
    mode_id : str, optional
        Mode identifier string.
    """

    mode_type: Optional[str] = None
    mode_id: Optional[str] = None


@dataclass
class SICDCollectionInfo:
    """Collection information.

    Parameters
    ----------
    collector_name : str, optional
        Name of the collecting platform.
    illuminator_name : str, optional
        Name of the illuminating platform (bistatic only).
    core_name : str, optional
        Unique collection identifier.
    collect_type : str, optional
        ``'MONOSTATIC'`` or ``'BISTATIC'``.
    radar_mode : SICDRadarMode, optional
        Radar mode parameters.
    classification : str, optional
        Security classification string.
    country_codes : List[str], optional
        Country code list.
    """

    collector_name: Optional[str] = None
    illuminator_name: Optional[str] = None
    core_name: Optional[str] = None
    collect_type: Optional[str] = None
    radar_mode: Optional[SICDRadarMode] = None
    classification: Optional[str] = None
    country_codes: Optional[List[str]] = None


# ===================================================================
# Section 2: ImageCreation
# ===================================================================

@dataclass
class SICDImageCreation:
    """Image creation information.

    Parameters
    ----------
    application : str, optional
        Name of the application that created the image.
    date_time : str, optional
        Creation date/time in ISO 8601 format.
    site : str, optional
        Site where the image was created.
    profile : str, optional
        Processing profile identifier.
    """

    application: Optional[str] = None
    date_time: Optional[str] = None
    site: Optional[str] = None
    profile: Optional[str] = None


# ===================================================================
# Section 3: ImageData
# ===================================================================

@dataclass
class SICDFullImage:
    """Full image dimensions (before any sub-imaging).

    Parameters
    ----------
    num_rows : int
        Number of rows in the full image.
    num_cols : int
        Number of columns in the full image.
    """

    num_rows: int = 0
    num_cols: int = 0


@dataclass
class SICDImageData:
    """Image data parameters.

    Parameters
    ----------
    pixel_type : str, optional
        Pixel type: ``'RE32F_IM32F'``, ``'RE16I_IM16I'``,
        ``'AMP8I_PHS8I'``.
    num_rows : int
        Number of rows.
    num_cols : int
        Number of columns.
    first_row : int
        First row offset from full image.
    first_col : int
        First column offset from full image.
    full_image : SICDFullImage, optional
        Full image dimensions.
    scp_pixel : RowCol, optional
        Scene Center Point pixel location.
    amp_table : numpy.ndarray, optional
        Amplitude lookup table (256 entries for AMP8I_PHS8I).
    """

    pixel_type: Optional[str] = None
    num_rows: int = 0
    num_cols: int = 0
    first_row: int = 0
    first_col: int = 0
    full_image: Optional[SICDFullImage] = None
    scp_pixel: Optional[RowCol] = None
    amp_table: Optional[np.ndarray] = None


# ===================================================================
# Section 4: GeoData
# ===================================================================

@dataclass
class SICDSCP:
    """Scene Center Point in ECF and geodetic coordinates.

    Parameters
    ----------
    ecf : XYZ, optional
        ECF coordinates (meters).
    llh : LatLonHAE, optional
        Geodetic coordinates (degrees, meters).
    """

    ecf: Optional[XYZ] = None
    llh: Optional[LatLonHAE] = None


@dataclass
class SICDGeoData:
    """Geographic data.

    Parameters
    ----------
    earth_model : str
        Earth model identifier (default ``'WGS_84'``).
    scp : SICDSCP, optional
        Scene Center Point.
    image_corners : List[LatLon], optional
        Four image corner coordinates.
    valid_data : List[LatLon], optional
        Valid data polygon vertices.
    """

    earth_model: str = 'WGS_84'
    scp: Optional[SICDSCP] = None
    image_corners: Optional[List[LatLon]] = None
    valid_data: Optional[List[LatLon]] = None


# ===================================================================
# Section 5: Grid
# ===================================================================

@dataclass
class SICDWgtType:
    """Window/weighting type for spatial frequency domain processing.

    Parameters
    ----------
    window_name : str, optional
        Window name (e.g., ``'UNIFORM'``, ``'HAMMING'``, ``'TAYLOR'``).
    parameters : Dict[str, str], optional
        Window-specific parameters.
    """

    window_name: Optional[str] = None
    parameters: Optional[Dict[str, str]] = None


@dataclass
class SICDDirParam:
    """Direction parameters for Grid row or column.

    Parameters
    ----------
    uvect_ecf : XYZ, optional
        Unit vector in ECF.
    ss : float, optional
        Sample spacing (meters).
    imp_resp_wid : float, optional
        Impulse response width (meters).
    sgn : int, optional
        Sign (+1 or -1).
    imp_resp_bw : float, optional
        Impulse response bandwidth (cycles/meter).
    k_ctr : float, optional
        Center spatial frequency (cycles/meter).
    delta_k1 : float, optional
        Minimum spatial frequency offset.
    delta_k2 : float, optional
        Maximum spatial frequency offset.
    delta_k_coa_poly : Poly2D, optional
        Center of aperture spatial frequency polynomial.
    wgt_type : SICDWgtType, optional
        Weighting type.
    wgt_funct : numpy.ndarray, optional
        Weight function samples.
    """

    uvect_ecf: Optional[XYZ] = None
    ss: Optional[float] = None
    imp_resp_wid: Optional[float] = None
    sgn: Optional[int] = None
    imp_resp_bw: Optional[float] = None
    k_ctr: Optional[float] = None
    delta_k1: Optional[float] = None
    delta_k2: Optional[float] = None
    delta_k_coa_poly: Optional[Poly2D] = None
    wgt_type: Optional[SICDWgtType] = None
    wgt_funct: Optional[np.ndarray] = None


@dataclass
class SICDGrid:
    """Image sample grid parameters.

    Parameters
    ----------
    image_plane : str, optional
        ``'SLANT'`` or ``'GROUND'``.
    type : str, optional
        Grid type: ``'RGAZIM'``, ``'RGZERO'``, ``'XRGYCR'``,
        ``'XCTYAT'``, ``'PLANE'``.
    row : SICDDirParam, optional
        Row direction parameters.
    col : SICDDirParam, optional
        Column direction parameters.
    time_coa_poly : Poly2D, optional
        Center of aperture time polynomial.
    """

    image_plane: Optional[str] = None
    type: Optional[str] = None
    row: Optional[SICDDirParam] = None
    col: Optional[SICDDirParam] = None
    time_coa_poly: Optional[Poly2D] = None


# ===================================================================
# Section 6: Timeline
# ===================================================================

@dataclass
class SICDIPPSet:
    """Inter-Pulse Period set.

    Parameters
    ----------
    t_start : float
        Start time offset (seconds).
    t_end : float
        End time offset (seconds).
    ipp_start : int
        Start IPP index.
    ipp_end : int
        End IPP index.
    ipp_poly : Poly1D, optional
        IPP polynomial.
    index : int
        Set index.
    """

    t_start: float = 0.0
    t_end: float = 0.0
    ipp_start: int = 0
    ipp_end: int = 0
    ipp_poly: Optional[Poly1D] = None
    index: int = 0


@dataclass
class SICDTimeline:
    """Collection timeline.

    Parameters
    ----------
    collect_start : str, optional
        Collection start time (ISO 8601).
    collect_duration : float, optional
        Collection duration (seconds).
    ipp : List[SICDIPPSet], optional
        Inter-Pulse Period sets.
    """

    collect_start: Optional[str] = None
    collect_duration: Optional[float] = None
    ipp: Optional[List[SICDIPPSet]] = None


# ===================================================================
# Section 7: Position
# ===================================================================

@dataclass
class SICDPosition:
    """Platform position polynomials.

    Parameters
    ----------
    arp_poly : XYZPoly, optional
        Aperture Reference Point position polynomial.
    grp_poly : XYZPoly, optional
        Ground Reference Point polynomial.
    tx_apc_poly : XYZPoly, optional
        Transmit Aperture Phase Center polynomial.
    rcv_apc : List[XYZPoly], optional
        Receive Aperture Phase Center polynomials.
    """

    arp_poly: Optional[XYZPoly] = None
    grp_poly: Optional[XYZPoly] = None
    tx_apc_poly: Optional[XYZPoly] = None
    rcv_apc: Optional[List[XYZPoly]] = None


# ===================================================================
# Section 8: RadarCollection
# ===================================================================

@dataclass
class SICDTxFrequency:
    """Transmit frequency range.

    Parameters
    ----------
    min : float, optional
        Minimum frequency (Hz).
    max : float, optional
        Maximum frequency (Hz).
    """

    min: Optional[float] = None
    max: Optional[float] = None


@dataclass
class SICDWaveformParams:
    """Waveform parameters for a single waveform.

    Parameters
    ----------
    tx_pulse_length : float, optional
        Transmit pulse length (seconds).
    tx_rf_bandwidth : float, optional
        Transmit RF bandwidth (Hz).
    tx_freq_start : float, optional
        Transmit start frequency (Hz).
    tx_fm_rate : float, optional
        Transmit FM rate (Hz/s).
    rcv_window_length : float, optional
        Receive window length (seconds).
    adc_sample_rate : float, optional
        ADC sample rate (Hz).
    rcv_if_bandwidth : float, optional
        Receive IF bandwidth (Hz).
    rcv_freq_start : float, optional
        Receive demodulation start frequency (Hz).
    rcv_demod_type : str, optional
        ``'STRETCH'`` or ``'CHIRP'``.
    rcv_fm_rate : float, optional
        Receive FM rate (Hz/s).
    index : int
        Waveform index.
    """

    tx_pulse_length: Optional[float] = None
    tx_rf_bandwidth: Optional[float] = None
    tx_freq_start: Optional[float] = None
    tx_fm_rate: Optional[float] = None
    rcv_window_length: Optional[float] = None
    adc_sample_rate: Optional[float] = None
    rcv_if_bandwidth: Optional[float] = None
    rcv_freq_start: Optional[float] = None
    rcv_demod_type: Optional[str] = None
    rcv_fm_rate: Optional[float] = None
    index: int = 0


@dataclass
class SICDRcvChannel:
    """Receive channel parameters.

    Parameters
    ----------
    tx_rcv_polarization : str, optional
        Combined polarization (e.g., ``'V:V'``, ``'H:H'``).
    rcv_ape_index : int, optional
        Receive aperture index.
    index : int
        Channel index.
    """

    tx_rcv_polarization: Optional[str] = None
    rcv_ape_index: Optional[int] = None
    index: int = 0


@dataclass
class SICDAreaCorner:
    """Area corner point.

    Parameters
    ----------
    lat : float
        Latitude (degrees).
    lon : float
        Longitude (degrees).
    hae : float
        Height Above Ellipsoid (meters).
    index : int
        Corner index.
    """

    lat: float = 0.0
    lon: float = 0.0
    hae: float = 0.0
    index: int = 0


@dataclass
class SICDArea:
    """Scene area.

    Parameters
    ----------
    corner_points : List[SICDAreaCorner], optional
        Corner points defining the area.
    plane : Dict[str, Any], optional
        Area plane segmentation info.
    """

    corner_points: Optional[List[SICDAreaCorner]] = None
    plane: Optional[Dict[str, Any]] = None


@dataclass
class SICDRadarCollection:
    """Radar collection parameters.

    Parameters
    ----------
    tx_frequency : SICDTxFrequency, optional
        Transmit frequency range.
    ref_freq_index : int, optional
        Reference frequency index.
    waveform : List[SICDWaveformParams], optional
        Waveform parameter sets.
    tx_polarization : str, optional
        Transmit polarization (``'V'``, ``'H'``, ``'RHC'``, ``'LHC'``,
        ``'SEQUENCE'``).
    rcv_channels : List[SICDRcvChannel], optional
        Receive channels.
    area : SICDArea, optional
        Scene area.
    """

    tx_frequency: Optional[SICDTxFrequency] = None
    ref_freq_index: Optional[int] = None
    waveform: Optional[List[SICDWaveformParams]] = None
    tx_polarization: Optional[str] = None
    rcv_channels: Optional[List[SICDRcvChannel]] = None
    area: Optional[SICDArea] = None


# ===================================================================
# Section 9: ImageFormation
# ===================================================================

@dataclass
class SICDRcvChanProc:
    """Receive channel processing parameters.

    Parameters
    ----------
    num_chan_proc : int, optional
        Number of channels processed.
    prf_scale_factor : float, optional
        PRF scale factor.
    chan_indices : List[int], optional
        Processed channel indices.
    """

    num_chan_proc: Optional[int] = None
    prf_scale_factor: Optional[float] = None
    chan_indices: Optional[List[int]] = None


@dataclass
class SICDTxFrequencyProc:
    """Processed transmit frequency range.

    Parameters
    ----------
    min_proc : float, optional
        Minimum processed frequency (Hz).
    max_proc : float, optional
        Maximum processed frequency (Hz).
    """

    min_proc: Optional[float] = None
    max_proc: Optional[float] = None


@dataclass
class SICDProcessingStep:
    """Image formation processing step.

    Parameters
    ----------
    type : str, optional
        Processing type identifier.
    applied : bool, optional
        Whether this step was applied.
    parameters : Dict[str, str], optional
        Processing parameters.
    """

    type: Optional[str] = None
    applied: Optional[bool] = None
    parameters: Optional[Dict[str, str]] = None


@dataclass
class SICDImageFormation:
    """Image formation parameters.

    Parameters
    ----------
    rcv_chan_proc : SICDRcvChanProc, optional
        Receive channel processing.
    image_form_algo : str, optional
        Algorithm: ``'PFA'``, ``'RGAZCOMP'``, ``'RMA'``.
    t_start_proc : float, optional
        Processing start time offset (seconds).
    t_end_proc : float, optional
        Processing end time offset (seconds).
    tx_frequency_proc : SICDTxFrequencyProc, optional
        Processed frequency range.
    seg_id : str, optional
        Segment identifier.
    image_beam_comp : str, optional
        Beam compensation: ``'NO'``, ``'SV'``, ``'GLOBAL'``.
    az_autofocus : str, optional
        Azimuth autofocus: ``'NO'``, ``'GLOBAL'``, ``'SV'``.
    rg_autofocus : str, optional
        Range autofocus: ``'NO'``, ``'GLOBAL'``, ``'SV'``.
    processing : List[SICDProcessingStep], optional
        Processing steps applied.
    polarization_hv_angle_poly : Poly1D, optional
        HV angle polynomial.
    polarization_calibration : Dict[str, Any], optional
        Polarization calibration data.
    """

    rcv_chan_proc: Optional[SICDRcvChanProc] = None
    image_form_algo: Optional[str] = None
    t_start_proc: Optional[float] = None
    t_end_proc: Optional[float] = None
    tx_frequency_proc: Optional[SICDTxFrequencyProc] = None
    seg_id: Optional[str] = None
    image_beam_comp: Optional[str] = None
    az_autofocus: Optional[str] = None
    rg_autofocus: Optional[str] = None
    processing: Optional[List[SICDProcessingStep]] = None
    polarization_hv_angle_poly: Optional[Poly1D] = None
    polarization_calibration: Optional[Dict[str, Any]] = None


# ===================================================================
# Section 10: SCPCOA
# ===================================================================

@dataclass
class SICDSCPCOA:
    """Scene Center Point Center of Aperture parameters.

    Parameters
    ----------
    scp_time : float, optional
        SCP center of aperture time (seconds from collect start).
    arp_pos : XYZ, optional
        ARP position at SCP COA (ECF meters).
    arp_vel : XYZ, optional
        ARP velocity at SCP COA (m/s).
    arp_acc : XYZ, optional
        ARP acceleration at SCP COA (m/s^2).
    side_of_track : str, optional
        ``'L'`` (left) or ``'R'`` (right).
    slant_range : float, optional
        Slant range to SCP (meters).
    ground_range : float, optional
        Ground range to SCP (meters).
    doppler_cone_ang : float, optional
        Doppler cone angle (degrees).
    graze_ang : float, optional
        Grazing angle (degrees).
    incidence_ang : float, optional
        Incidence angle (degrees).
    twist_ang : float, optional
        Twist angle (degrees).
    slope_ang : float, optional
        Slope angle (degrees).
    azim_ang : float, optional
        Azimuth angle (degrees).
    layover_ang : float, optional
        Layover angle (degrees).
    """

    scp_time: Optional[float] = None
    arp_pos: Optional[XYZ] = None
    arp_vel: Optional[XYZ] = None
    arp_acc: Optional[XYZ] = None
    side_of_track: Optional[str] = None
    slant_range: Optional[float] = None
    ground_range: Optional[float] = None
    doppler_cone_ang: Optional[float] = None
    graze_ang: Optional[float] = None
    incidence_ang: Optional[float] = None
    twist_ang: Optional[float] = None
    slope_ang: Optional[float] = None
    azim_ang: Optional[float] = None
    layover_ang: Optional[float] = None


# ===================================================================
# Section 11: Radiometric
# ===================================================================

@dataclass
class SICDNoiseLevel:
    """Noise level parameters.

    Parameters
    ----------
    noise_level_type : str, optional
        ``'ABSOLUTE'`` or ``'RELATIVE'``.
    noise_poly : Poly2D, optional
        Noise level polynomial.
    """

    noise_level_type: Optional[str] = None
    noise_poly: Optional[Poly2D] = None


@dataclass
class SICDRadiometric:
    """Radiometric calibration parameters.

    Parameters
    ----------
    noise_level : SICDNoiseLevel, optional
        Noise level.
    rcs_sf_poly : Poly2D, optional
        RCS scale factor polynomial.
    sigma_zero_sf_poly : Poly2D, optional
        Sigma-zero scale factor polynomial.
    beta_zero_sf_poly : Poly2D, optional
        Beta-zero scale factor polynomial.
    gamma_zero_sf_poly : Poly2D, optional
        Gamma-zero scale factor polynomial.
    """

    noise_level: Optional[SICDNoiseLevel] = None
    rcs_sf_poly: Optional[Poly2D] = None
    sigma_zero_sf_poly: Optional[Poly2D] = None
    beta_zero_sf_poly: Optional[Poly2D] = None
    gamma_zero_sf_poly: Optional[Poly2D] = None


# ===================================================================
# Section 12: Antenna
# ===================================================================

@dataclass
class SICDEBType:
    """Electrical boresight parameters.

    Parameters
    ----------
    dcx_poly : Poly1D, optional
        DCX electrical boresight polynomial.
    dcy_poly : Poly1D, optional
        DCY electrical boresight polynomial.
    """

    dcx_poly: Optional[Poly1D] = None
    dcy_poly: Optional[Poly1D] = None


@dataclass
class SICDGainPhasePoly:
    """Gain and phase polynomial pair.

    Parameters
    ----------
    gain : Poly2D, optional
        Gain polynomial.
    phase : Poly2D, optional
        Phase polynomial.
    """

    gain: Optional[Poly2D] = None
    phase: Optional[Poly2D] = None


@dataclass
class SICDAntParam:
    """Antenna parameters (transmit, receive, or two-way).

    Parameters
    ----------
    x_axis_poly : XYZPoly, optional
        X-axis orientation polynomial.
    y_axis_poly : XYZPoly, optional
        Y-axis orientation polynomial.
    freq_zero : float, optional
        Reference frequency (Hz).
    eb : SICDEBType, optional
        Electrical boresight.
    array : SICDGainPhasePoly, optional
        Array gain/phase polynomial.
    elem : SICDGainPhasePoly, optional
        Element gain/phase polynomial.
    gain_bs_poly : Poly1D, optional
        Gain broadening/steering polynomial.
    eb_freq_shift : bool, optional
        Electrical boresight frequency shift flag.
    ml_freq_dilation : bool, optional
        Mainlobe frequency dilation flag.
    """

    x_axis_poly: Optional[XYZPoly] = None
    y_axis_poly: Optional[XYZPoly] = None
    freq_zero: Optional[float] = None
    eb: Optional[SICDEBType] = None
    array: Optional[SICDGainPhasePoly] = None
    elem: Optional[SICDGainPhasePoly] = None
    gain_bs_poly: Optional[Poly1D] = None
    eb_freq_shift: Optional[bool] = None
    ml_freq_dilation: Optional[bool] = None


@dataclass
class SICDAntenna:
    """Antenna parameters for all paths.

    Parameters
    ----------
    tx : SICDAntParam, optional
        Transmit antenna parameters.
    rcv : SICDAntParam, optional
        Receive antenna parameters.
    two_way : SICDAntParam, optional
        Two-way (combined) antenna parameters.
    """

    tx: Optional[SICDAntParam] = None
    rcv: Optional[SICDAntParam] = None
    two_way: Optional[SICDAntParam] = None


# ===================================================================
# Section 13: ErrorStatistics
# ===================================================================

@dataclass
class SICDCompositeSCPError:
    """Composite SCP error estimates.

    Parameters
    ----------
    rg : float, optional
        Range error (meters).
    az : float, optional
        Azimuth error (meters).
    rg_az : float, optional
        Range-azimuth cross-correlation.
    """

    rg: Optional[float] = None
    az: Optional[float] = None
    rg_az: Optional[float] = None


@dataclass
class SICDCorrCoefs:
    """Position/velocity correlation coefficients.

    All 15 unique pairwise correlations between 3 position
    and 3 velocity components.
    """

    p1p2: Optional[float] = None
    p1p3: Optional[float] = None
    p1v1: Optional[float] = None
    p1v2: Optional[float] = None
    p1v3: Optional[float] = None
    p2p3: Optional[float] = None
    p2v1: Optional[float] = None
    p2v2: Optional[float] = None
    p2v3: Optional[float] = None
    p3v1: Optional[float] = None
    p3v2: Optional[float] = None
    p3v3: Optional[float] = None
    v1v2: Optional[float] = None
    v1v3: Optional[float] = None
    v2v3: Optional[float] = None


@dataclass
class SICDErrorDecorrFunc:
    """Error decorrelation function.

    Parameters
    ----------
    corr_coef_zero : float, optional
        Correlation coefficient at zero lag.
    decorr_rate : float, optional
        Decorrelation rate.
    """

    corr_coef_zero: Optional[float] = None
    decorr_rate: Optional[float] = None


@dataclass
class SICDPosVelErr:
    """Position and velocity error parameters.

    Parameters
    ----------
    frame : str, optional
        Coordinate frame: ``'ECF'``, ``'RIC_ECF'``, ``'RIC_ECI'``.
    p1 : float, optional
        Position error component 1 (meters).
    p2 : float, optional
        Position error component 2 (meters).
    p3 : float, optional
        Position error component 3 (meters).
    v1 : float, optional
        Velocity error component 1 (m/s).
    v2 : float, optional
        Velocity error component 2 (m/s).
    v3 : float, optional
        Velocity error component 3 (m/s).
    corr_coefs : SICDCorrCoefs, optional
        Correlation coefficients.
    position_decorr : SICDErrorDecorrFunc, optional
        Position decorrelation function.
    """

    frame: Optional[str] = None
    p1: Optional[float] = None
    p2: Optional[float] = None
    p3: Optional[float] = None
    v1: Optional[float] = None
    v2: Optional[float] = None
    v3: Optional[float] = None
    corr_coefs: Optional[SICDCorrCoefs] = None
    position_decorr: Optional[SICDErrorDecorrFunc] = None


@dataclass
class SICDRadarSensorError:
    """Radar sensor error parameters.

    Parameters
    ----------
    range_bias : float, optional
        Range bias (meters).
    clock_freq_sf : float, optional
        Clock frequency scale factor error.
    transmit_freq_sf : float, optional
        Transmit frequency scale factor error.
    range_bias_decorr : SICDErrorDecorrFunc, optional
        Range bias decorrelation function.
    """

    range_bias: Optional[float] = None
    clock_freq_sf: Optional[float] = None
    transmit_freq_sf: Optional[float] = None
    range_bias_decorr: Optional[SICDErrorDecorrFunc] = None


@dataclass
class SICDErrorStatistics:
    """Error statistics.

    Parameters
    ----------
    composite_scp : SICDCompositeSCPError, optional
        Composite SCP error.
    monostatic : SICDPosVelErr, optional
        Monostatic position/velocity errors.
    radar_sensor : SICDRadarSensorError, optional
        Radar sensor errors.
    """

    composite_scp: Optional[SICDCompositeSCPError] = None
    monostatic: Optional[SICDPosVelErr] = None
    radar_sensor: Optional[SICDRadarSensorError] = None


# ===================================================================
# Section 14: MatchInfo
# ===================================================================

@dataclass
class SICDMatchCollection:
    """Match collection entry.

    Parameters
    ----------
    core_name : str, optional
        Matched collection core name.
    match_index : int, optional
        Match index.
    parameters : Dict[str, str], optional
        Match parameters.
    """

    core_name: Optional[str] = None
    match_index: Optional[int] = None
    parameters: Optional[Dict[str, str]] = None


@dataclass
class SICDMatchType:
    """Match type entry.

    Parameters
    ----------
    type_id : str, optional
        Match type identifier.
    current_index : int, optional
        Current collection index.
    match_collections : List[SICDMatchCollection], optional
        Matched collections.
    """

    type_id: Optional[str] = None
    current_index: Optional[int] = None
    match_collections: Optional[List[SICDMatchCollection]] = None


@dataclass
class SICDMatchInfo:
    """Match information.

    Parameters
    ----------
    match_types : List[SICDMatchType], optional
        Match type entries.
    """

    match_types: Optional[List[SICDMatchType]] = None


# ===================================================================
# Section 15: RgAzComp
# ===================================================================

@dataclass
class SICDRgAzComp:
    """Range-azimuth compression parameters.

    Parameters
    ----------
    az_sf : float, optional
        Azimuth scale factor.
    kaz_poly : Poly1D, optional
        Azimuth spatial frequency polynomial.
    """

    az_sf: Optional[float] = None
    kaz_poly: Optional[Poly1D] = None


# ===================================================================
# Section 16: PFA
# ===================================================================

@dataclass
class SICDSTDeskew:
    """Slow-time deskew parameters.

    Parameters
    ----------
    applied : bool, optional
        Whether deskew was applied.
    st_ds_phase_poly : Poly2D, optional
        Slow-time deskew phase polynomial.
    """

    applied: Optional[bool] = None
    st_ds_phase_poly: Optional[Poly2D] = None


@dataclass
class SICDPFA:
    """Polar Format Algorithm parameters.

    Parameters
    ----------
    fpn : XYZ, optional
        Focus Plane Normal (unit vector).
    ipn : XYZ, optional
        Image Plane Normal (unit vector).
    polar_ang_ref_time : float, optional
        Polar angle reference time (seconds).
    polar_ang_poly : Poly1D, optional
        Polar angle polynomial.
    spatial_freq_sf_poly : Poly1D, optional
        Spatial frequency scale factor polynomial.
    krg1 : float, optional
        Minimum range spatial frequency.
    krg2 : float, optional
        Maximum range spatial frequency.
    kaz1 : float, optional
        Minimum azimuth spatial frequency.
    kaz2 : float, optional
        Maximum azimuth spatial frequency.
    st_deskew : SICDSTDeskew, optional
        Slow-time deskew parameters.
    """

    fpn: Optional[XYZ] = None
    ipn: Optional[XYZ] = None
    polar_ang_ref_time: Optional[float] = None
    polar_ang_poly: Optional[Poly1D] = None
    spatial_freq_sf_poly: Optional[Poly1D] = None
    krg1: Optional[float] = None
    krg2: Optional[float] = None
    kaz1: Optional[float] = None
    kaz2: Optional[float] = None
    st_deskew: Optional[SICDSTDeskew] = None


# ===================================================================
# Section 17: RMA
# ===================================================================

@dataclass
class SICDRMRef:
    """Range migration reference parameters.

    Parameters
    ----------
    pos_ref : XYZ, optional
        Reference position (ECF meters).
    vel_ref : XYZ, optional
        Reference velocity (m/s).
    dop_cone_ang_ref : float, optional
        Reference Doppler cone angle (degrees).
    """

    pos_ref: Optional[XYZ] = None
    vel_ref: Optional[XYZ] = None
    dop_cone_ang_ref: Optional[float] = None


@dataclass
class SICDINCA:
    """INCA (INverse Chirp-scaling Algorithm) parameters.

    Parameters
    ----------
    time_ca_poly : Poly1D, optional
        Center of aperture time polynomial.
    r_ca_scp : float, optional
        Range at center of aperture for SCP (meters).
    freq_zero : float, optional
        Reference frequency (Hz).
    d_rate_sf_poly : Poly2D, optional
        Doppler rate scale factor polynomial.
    dop_centroid_poly : Poly2D, optional
        Doppler centroid polynomial.
    dop_centroid_coa : bool, optional
        Doppler centroid at COA flag.
    """

    time_ca_poly: Optional[Poly1D] = None
    r_ca_scp: Optional[float] = None
    freq_zero: Optional[float] = None
    d_rate_sf_poly: Optional[Poly2D] = None
    dop_centroid_poly: Optional[Poly2D] = None
    dop_centroid_coa: Optional[bool] = None


@dataclass
class SICDRMA:
    """Range Migration Algorithm parameters.

    Parameters
    ----------
    rm_ref : SICDRMRef, optional
        Range migration reference.
    inca : SICDINCA, optional
        INCA parameters.
    image_type : str, optional
        Image type: ``'RMAT'``, ``'RMCR'``, ``'INCA'``.
    """

    rm_ref: Optional[SICDRMRef] = None
    inca: Optional[SICDINCA] = None
    image_type: Optional[str] = None


# ===================================================================
# Top-level: SICDMetadata
# ===================================================================

@dataclass
class SICDMetadata(ImageMetadata):
    """Complete typed metadata for SICD imagery.

    Contains all 17 sections of the SICD standard as nested
    dataclasses, plus the backend identifier. Inherits from
    ``ImageMetadata`` for universal fields and dict-like access.

    Parameters
    ----------
    backend : str, optional
        Active reader backend (``'sarkit'`` or ``'sarpy'``).
    collection_info : SICDCollectionInfo, optional
    image_creation : SICDImageCreation, optional
    image_data : SICDImageData, optional
    geo_data : SICDGeoData, optional
    grid : SICDGrid, optional
    timeline : SICDTimeline, optional
    position : SICDPosition, optional
    radar_collection : SICDRadarCollection, optional
    image_formation : SICDImageFormation, optional
    scpcoa : SICDSCPCOA, optional
    radiometric : SICDRadiometric, optional
    antenna : SICDAntenna, optional
    error_statistics : SICDErrorStatistics, optional
    match_info : SICDMatchInfo, optional
    rg_az_comp : SICDRgAzComp, optional
    pfa : SICDPFA, optional
    rma : SICDRMA, optional

    Examples
    --------
    >>> meta = reader.metadata  # SICDMetadata
    >>> meta.scpcoa.graze_ang
    45.2
    >>> meta.grid.row.ss
    0.5
    >>> meta.collection_info.radar_mode.mode_type
    'SPOTLIGHT'
    """

    backend: Optional[str] = None
    collection_info: Optional[SICDCollectionInfo] = None
    image_creation: Optional[SICDImageCreation] = None
    image_data: Optional[SICDImageData] = None
    geo_data: Optional[SICDGeoData] = None
    grid: Optional[SICDGrid] = None
    timeline: Optional[SICDTimeline] = None
    position: Optional[SICDPosition] = None
    radar_collection: Optional[SICDRadarCollection] = None
    image_formation: Optional[SICDImageFormation] = None
    scpcoa: Optional[SICDSCPCOA] = None
    radiometric: Optional[SICDRadiometric] = None
    antenna: Optional[SICDAntenna] = None
    error_statistics: Optional[SICDErrorStatistics] = None
    match_info: Optional[SICDMatchInfo] = None
    rg_az_comp: Optional[SICDRgAzComp] = None
    pfa: Optional[SICDPFA] = None
    rma: Optional[SICDRMA] = None
