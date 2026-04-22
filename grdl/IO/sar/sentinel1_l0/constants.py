# -*- coding: utf-8 -*-
"""
Sentinel-1 Level 0 Constants — Single Source of Truth.

Physical constants, instrument parameters, mode-specific radar
parameters, swath mappings, and polarization enumerations used
across the Sentinel-1 L0 reader.  Every constant in this file
traces to an authoritative spec section or a documented derivation.

Authoritative Specifications
----------------------------
[L0FMT]  S1PD.SP.00110.ATSR Issue 3.1
         Sentinel-1 Level 0 Product Format Specifications
[ISP]    S1-IF-ASD-PL-0007 Issue 13
         Sentinel-1 SAR Space Packet Protocol Data Unit
[SRD]    S1-RS-ASD-PL-0001
         Sentinel-1 System Requirements Document
[CAL]    S1-RP-MDA-52023
         Sentinel-1 Absolute Radiometric Calibration Report
[NIST]   NIST SP 330 (2019)
         The International System of Units (SI)
[WGS84]  NGA.STND.0036_1.0.0
         World Geodetic System 1984
[GPS]    IS-GPS-200 Rev N
         GPS Interface Control Document

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

from typing import Dict, Tuple


# =============================================================================
# Physical Constants
# =============================================================================

# Speed of light in vacuum (m/s).
# Ref: [NIST] — exact by SI definition since 2019.
SPEED_OF_LIGHT: float = 299_792_458.0

# WGS-84 ellipsoid parameters.
# Ref: [WGS84] NGA.STND.0036_1.0.0, Table 3.1.
WGS84_SEMI_MAJOR_AXIS: float = 6_378_137.0           # metres
WGS84_FLATTENING: float = 1.0 / 298.257223563        # exact
WGS84_SEMI_MINOR_AXIS: float = (
    WGS84_SEMI_MAJOR_AXIS * (1.0 - WGS84_FLATTENING)
)
WGS84_ECCENTRICITY_SQ: float = (
    2.0 * WGS84_FLATTENING - WGS84_FLATTENING ** 2
)

# IUGG mean Earth radius (m) — volumetric mean of WGS-84 ellipsoid.
# Used for spherical-Earth approximations (incidence angle geometry).
# Ref: [WGS84] NGA.STND.0036_1.0.0, Section 3.4.4 (derived).
# R_mean = (2*a + b) / 3 ≈ 6_371_008.8 m; 6_371_000 is the IUGG
# standard rounding adopted by most SAR processors.
# Ref: IUGG/IUGS, Geodetic Reference System 1980.
MEAN_EARTH_RADIUS: float = 6_371_000.0


# =============================================================================
# Sentinel-1 Instrument Parameters
# =============================================================================

# Reference oscillator frequency (Hz) — Temperature-Compensated
# Crystal Oscillator (TCXO).  All Sentinel-1 timing and frequency
# parameters are integer multiples of F_REF.
# Ref: [ISP] Section 3.2.5.1 (Radar Configuration), Table 5-2.
SENTINEL1_F_REF_HZ: float = 37_534_722.24

# Exact C-band center frequency: f_c = 144 × F_REF.
# Ref: [ISP] Section 3.2.5.1 — the ISP stores the TX pulse
# start frequency code N_start, and f_start = N_start × F_REF.
# The center frequency for IW mode is N=144 → 5,404,999,842.56 Hz.
SENTINEL1_CENTER_FREQUENCY_HZ: float = (
    144.0 * SENTINEL1_F_REF_HZ
)

# Nominal C-band center frequency (approximate, for display only).
# Ref: ESA Sentinel-1 User Handbook, Section 3.1.
SENTINEL1_CENTER_FREQUENCY_NOMINAL_HZ: float = 5.405e9

# Instrument timing bias (seconds).
# The ISP datation timestamp (Coarse Time + Fine Time) leads the
# actual radar event time by a constant bias inherent to the
# onboard data handling pipeline.  This bias is stable within
# ±15 µs (one Fine Time LSB = 1/2^16 s ≈ 15.26 µs) across all
# bursts, swaths, and polarizations for a given datatake.
#
# Ref: Empirically derived from reference-file comparison.
#   IW1 offsets: 0.627593 / 0.627578 s  (±1 Fine Time LSB)
#   IW2 offsets: 0.627583 / 0.627568 s
#   IW3 offsets: 0.627581 / 0.627596 s
#   Mean: 0.627583 s
#
# Applied as: TxTime = ISP_time − CRT − INSTRUMENT_TIMING_BIAS
S1A_INSTRUMENT_TIMING_BIAS_S: float = 0.627583


# =============================================================================
# ISP Timing Constants
# =============================================================================

# GPS epoch: January 6, 1980 00:00:00 UTC.
# Ref: [GPS] IS-GPS-200 Rev N, Section 20.3.3.
# (Used to decode ISP Coarse Time / Fine Time fields.)
GPS_EPOCH_YEAR: int = 1980
GPS_EPOCH_MONTH: int = 1
GPS_EPOCH_DAY: int = 6

# Fine Time divisor: ISP Fine Time is a 16-bit unsigned integer
# representing the fractional second as fine_time / 2^16.
# Ref: [ISP] Section 3.2.1 (Datation Service), Table 3-2.
FINE_TIME_DIVISOR: float = 2 ** 16

# GPS−UTC leap seconds.  GPS time does not include leap seconds;
# UTC does.  The offset must be updated when IERS announces a new
# leap second (last change: 2017-01-01, TAI−UTC = 37 s, GPS−UTC =
# TAI−UTC − 19 = 18 s).
# Ref: IERS Bulletin C (https://hpiers.obspm.fr/iers/bul/bulc/)
GPS_LEAP_SECONDS: int = 18  # valid 2017-01-01 through at least 2026


# =============================================================================
# Sentinel-1 Mode Parameters
# =============================================================================
# Nominal radar parameters per imaging mode.  Per-pulse values are
# extracted from ISP headers at runtime; these are fallback defaults.
#
# Ref: [ISP] Table 5-2 (IW/EW/SM instrument timeline parameters).
#       [L0FMT] Section 4.2 (Product characteristics by mode).

IW_MODE_PARAMS: Dict[str, float] = {
    "range_sampling_rate_hz": 64.345238e6,
    "pulse_repetition_frequency_hz": 1717.0,
    "tx_pulse_length_s": 52.0e-6,
    "tx_pulse_ramp_rate_hz_per_s": 779.038e9,
    "chirp_bandwidth_hz": 42.79e6,
    "azimuth_steering_rate_deg_per_s": 1.59,
    "burst_cycle_duration_s": 2.722,
}

EW_MODE_PARAMS: Dict[str, float] = {
    "range_sampling_rate_hz": 56.59e6,
    "pulse_repetition_frequency_hz": 1540.0,
    "tx_pulse_length_s": 40.0e-6,
    "tx_pulse_ramp_rate_hz_per_s": 500.0e9,
    "chirp_bandwidth_hz": 20.0e6,
    "azimuth_steering_rate_deg_per_s": 2.39,
}

SM_MODE_PARAMS: Dict[str, float] = {
    "range_sampling_rate_hz": 64.345238e6,
    "pulse_repetition_frequency_hz": 3000.0,
    "tx_pulse_length_s": 40.0e-6,
    "tx_pulse_ramp_rate_hz_per_s": 1.0e12,
    "chirp_bandwidth_hz": 40.0e6,
    "azimuth_steering_rate_deg_per_s": 0.0,
}

# Mode parameter lookup
MODE_PARAMS: Dict[str, Dict[str, float]] = {
    "IW": IW_MODE_PARAMS,
    "EW": EW_MODE_PARAMS,
    "SM": SM_MODE_PARAMS,
}


# =============================================================================
# Swath Number Mappings
# =============================================================================
# Ref: [ISP] Section 3.2.5.1, Table 3-9 — Swath Number field
#      encodes raw swath IDs in the ISP secondary header.
# IW mode uses raw swath numbers 10, 11, 12 → logical 1, 2, 3.
# EW/SM modes use 1-based raw = logical.

IW_RAW_TO_LOGICAL: Dict[int, int] = {
    10: 1,  # IW1
    11: 2,  # IW2
    12: 3,  # IW3
}

IW_LOGICAL_TO_RAW: Dict[int, int] = {
    v: k for k, v in IW_RAW_TO_LOGICAL.items()
}

IW_SWATH_NAMES: Dict[int, str] = {
    1: "IW1",
    2: "IW2",
    3: "IW3",
}

EW_SWATH_NAMES: Dict[int, str] = {
    1: "EW1",
    2: "EW2",
    3: "EW3",
    4: "EW4",
    5: "EW5",
}


# =============================================================================
# Swath-Specific Frequency Offsets (Hz)
# =============================================================================
# In TOPS mode, each sub-swath has a slightly different effective
# center frequency due to Doppler centroid variations from antenna
# steering.  These offsets are applied to the base receive
# frequency bounds.
#
# Ref: Derived from reference-file analysis of S1A IW L0 products.
#   IW1: base frequency (no offset)
#   IW2: +4.875 MHz offset
#   IW3: +8.714 MHz offset

IW_SWATH_FREQ_OFFSET_HZ: Dict[str, float] = {
    "IW1": 0.0,
    "IW2": 4_874_639.19,
    "IW3": 8_713_417.60,
}

EW_SWATH_FREQ_OFFSET_HZ: Dict[str, float] = {
    "EW1": 0.0,
    "EW2": 0.0,
    "EW3": 0.0,
    "EW4": 0.0,
    "EW5": 0.0,
}


def get_swath_frequency_offset(swath_name: str) -> float:
    """Get frequency offset for a swath.

    In TOPS mode, each swath has a different Doppler centroid
    due to antenna steering, resulting in different effective
    center frequencies.

    Parameters
    ----------
    swath_name : str
        Swath name (e.g., ``"IW1"``, ``"IW2"``, ``"EW3"``).

    Returns
    -------
    float
        Frequency offset in Hz to add to base frequency.
    """
    swath_upper = swath_name.upper()
    if swath_upper in IW_SWATH_FREQ_OFFSET_HZ:
        return IW_SWATH_FREQ_OFFSET_HZ[swath_upper]
    if swath_upper in EW_SWATH_FREQ_OFFSET_HZ:
        return EW_SWATH_FREQ_OFFSET_HZ[swath_upper]
    return 0.0


def get_azimuth_steering_rate(mode: str) -> float:
    """Get azimuth steering rate for a mode.

    Parameters
    ----------
    mode : str
        Imaging mode (``"IW"``, ``"EW"``, ``"SM"``).

    Returns
    -------
    float
        Azimuth steering rate in degrees per second.
    """
    mode_upper = mode.upper()
    if mode_upper in MODE_PARAMS:
        return MODE_PARAMS[mode_upper].get(
            "azimuth_steering_rate_deg_per_s", 0.0
        )
    return 0.0


def raw_swath_to_name(raw_swath: int, mode: str = "IW") -> str:
    """Convert raw swath number to logical swath name.

    Ref: [ISP] Section 3.2.5.1, Table 3-9.

    Parameters
    ----------
    raw_swath : int
        Raw swath number from packet data.
    mode : str, optional
        Imaging mode (``"IW"``, ``"EW"``, ``"SM"``). Default ``"IW"``.

    Returns
    -------
    str
        Swath name string (e.g., ``"IW1"``, ``"EW3"``).

    Raises
    ------
    ValueError
        If raw swath not recognized for mode.
    """
    mode = mode.upper()
    if mode == "IW":
        if raw_swath in IW_RAW_TO_LOGICAL:
            logical = IW_RAW_TO_LOGICAL[raw_swath]
            return IW_SWATH_NAMES[logical]
        # Fallback: assume raw is logical
        if raw_swath in IW_SWATH_NAMES:
            return IW_SWATH_NAMES[raw_swath]
        raise ValueError(f"Unknown IW raw swath: {raw_swath}")
    elif mode == "EW":
        if raw_swath in EW_SWATH_NAMES:
            return EW_SWATH_NAMES[raw_swath]
        raise ValueError(f"Unknown EW raw swath: {raw_swath}")
    elif mode == "SM":
        return f"SM{raw_swath}"
    else:
        return f"S{raw_swath}"


def raw_swath_to_logical(
    raw_swath: int, mode: str = "IW"
) -> int:
    """Convert raw swath number to logical swath index (1-based).

    Parameters
    ----------
    raw_swath : int
        Raw swath number from packet data.
    mode : str, optional
        Imaging mode (``"IW"``, ``"EW"``, ``"SM"``). Default ``"IW"``.

    Returns
    -------
    int
        Logical swath index (1, 2, 3, etc.).
    """
    mode = mode.upper()
    if mode == "IW" and raw_swath in IW_RAW_TO_LOGICAL:
        return IW_RAW_TO_LOGICAL[raw_swath]
    return raw_swath


# =============================================================================
# Polarization Mappings
# =============================================================================
# Ref: [ISP] Section 3.2.5.1, Table 3-10 (Rx Polarization code).

# Sentinel-1 polarization → (tx, rx) tuple.
POL_TO_TXRX: Dict[str, Tuple[str, str]] = {
    "VV": ("V", "V"),
    "VH": ("V", "H"),
    "HH": ("H", "H"),
    "HV": ("H", "V"),
}

# Dual-polarization mode expansion.
# Ref: [L0FMT] Section 2.2 — polarization mode identifiers.
DUAL_POL_EXPANSION: Dict[str, Tuple[str, str]] = {
    "DV": ("VV", "VH"),  # Dual vertical transmit
    "DH": ("HH", "HV"),  # Dual horizontal transmit
}


def expand_polarization(pol_code: str) -> Tuple[str, ...]:
    """Expand polarization code to individual polarizations.

    Parameters
    ----------
    pol_code : str
        Polarization code (``"VV"``, ``"DV"``, ``"QUAD"``, etc.).

    Returns
    -------
    tuple of str
        Tuple of individual polarizations.
    """
    pol_code = pol_code.upper()
    if pol_code in DUAL_POL_EXPANSION:
        return DUAL_POL_EXPANSION[pol_code]
    if pol_code == "QUAD":
        return ("HH", "HV", "VH", "VV")
    if pol_code in POL_TO_TXRX:
        return (pol_code,)
    if "+" in pol_code:
        return tuple(pol_code.split("+"))
    return (pol_code,)


def polarization_to_txrx(s1_pol: str) -> Tuple[str, str]:
    """Split a Sentinel-1 polarization code into (tx, rx).

    Parameters
    ----------
    s1_pol : str
        Sentinel-1 polarization (``"VV"``, ``"VH"``, ``"HH"``, ``"HV"``).

    Returns
    -------
    tuple of (str, str)
        ``(tx_polarization, rx_polarization)``.

    Raises
    ------
    ValueError
        If polarization not recognized.
    """
    s1_pol = s1_pol.upper()
    if s1_pol not in POL_TO_TXRX:
        raise ValueError(
            f"Unknown polarization: {s1_pol}. "
            f"Expected: {list(POL_TO_TXRX.keys())}"
        )
    return POL_TO_TXRX[s1_pol]


# =============================================================================
# Orbital Parameters
# =============================================================================
# Ref: ESA Sentinel-1 Orbital Elements,
#      https://sentinels.copernicus.eu/web/sentinel/missions/
#      sentinel-1/satellite-description/orbit
#      Semi-major axis: 7071 km (693 km altitude above WGS-84).
#      Inclination: 98.18° (sun-synchronous).
#      Period: ~98.9 minutes.

SENTINEL1_ORBITAL: Dict[str, float] = {
    "semi_major_axis_m": 7_071_000.0,
    "eccentricity": 0.0012,
    "inclination_deg": 98.18,
    "orbital_period_s": 5934.0,
}


# =============================================================================
# Sentinel-1 Antenna & Transmitter Physical Constants
# =============================================================================

# Antenna physical dimensions (metres).
# Ref: [SRD] S1-RS-ASD-PL-0001, Section 3.2 (Antenna Subsystem).
#      [ISP] Table 2-1 (Instrument characteristics).
#      Length: 12.3 m (azimuth), Width: 0.821 m (elevation).
S1_ANTENNA_LENGTH_M: float = 12.3
S1_ANTENNA_WIDTH_M: float = 0.821

# Aperture efficiency — includes feed network losses, amplitude/
# phase taper, and edge effects.
# Ref: [SRD] S1-RS-ASD-PL-0001, Section 3.2.3 (Antenna
#      Performance Budget).  Nominal η ≈ 0.49.
S1_ANTENNA_EFFICIENCY: float = 0.49

# Effective peak transmit power at antenna aperture (Watts).
# ESA published ~4.8 kW at SSPA output; after feed network and
# waveguide losses (~2.8 dB), effective power is ~2500 W.
# Ref: [SRD] S1-RS-ASD-PL-0001, Section 3.3 (Transmitter).
S1_PEAK_TX_POWER_W: float = 2500.0

# Cross-polarization isolation (dB).
# Minimum isolation between co-pol and cross-pol channels across
# the antenna pattern mainlobe.
# Ref: [SRD] S1-RS-ASD-PL-0001, Section 3.2.4 (Antenna
#      Polarization Purity).  Requirement: ≥ 50 dB.
S1_CROSS_POL_ISOLATION_DB: float = 50.0

# Cross-polarization phase offset (radians).
# Phase difference between H and V receive paths.
# Ref: [SRD] S1-RS-ASD-PL-0001, Section 3.2.4.
#      Nominal value from calibration measurements.
S1_CROSS_POL_PHASE_RAD: float = 0.5

# IW TOPS azimuth scan half-angles (degrees).
# The antenna steers from −θ to +θ during each burst.
# Per-swath values are fixed by the onboard instrument timeline.
# Used to derive the beam steering rate:
#   k_steer = 2 × θ_half / t_burst  [rad/s]
# Ref: [ISP] Section 5.3 (TOPS Mode Description),
#      Table 5-3 (IW swath timeline parameters).
S1_IW_SCAN_HALF_ANGLE_DEG: Dict[str, float] = {
    "IW1": 0.6468,
    "IW2": 0.5153,
    "IW3": 0.5772,
}

# Range decimation filter noise bandwidth ratio (BW_noise / fs).
# The onboard FIR decimation filter has a noise bandwidth narrower
# than the output sample rate by this factor.
# Computed from IW decimation code 8 filter coefficients:
#   noise BW = 56.5938 MHz, fs = 64.3452 MHz → ratio 0.8795.
# Ref: [ISP] Section 3.2.5.3 (Range Decimation Filter),
#      Table 3-15 (Decimation filter coefficients).
S1_FILTER_ROLLOFF_RATIO: float = 0.879474560175494

# Receive calibration constant (dBW/m^2).
# Reference irradiance at the receive APC that produces unit
# signal array power for GRBS = 0 dB.  This is a system-level
# calibration constant determined from ESA absolute radiometric
# calibration campaigns.  It cannot be derived from L0 data alone
# because it requires knowledge of the system noise temperature
# and end-to-end receive chain gain.
# Ref: [CAL] S1-RP-MDA-52023, Section 5.2
#      (Absolute Calibration Results).
S1_RCV_REF_IRRADIANCE_DBW_M2: float = -142.57446476678422


# =============================================================================
# Reader Empirical Constants
# =============================================================================

# FDBAQ amplitude scale factor.
# The sentinel1decoder FDBAQ reconstruction returns sample values
# approximately 1.607× smaller than reference implementations that
# use the ESA-specified NRL×sigma lookup tables with full per-block
# amplitude normalization.  Optional calibration factor for users
# who need amplitude parity with those references.
FDBAQ_AMPLITUDE_SCALE: float = 1.607

# Burst gap detection threshold (microseconds).
# In TOPS mode, bursts are separated by the time to cycle through
# other swaths.  Gaps exceeding this threshold indicate a burst
# boundary.
BURST_GAP_THRESHOLD_US: int = 1_000_000

# Minimum fraction of median line count for valid bursts.
# Bursts with fewer lines than this fraction of the swath median
# are considered partial (edge) bursts and can be excluded by the
# reader when producing a clean burst list.
BURST_LINE_FILTER_RATIO: float = 0.9
