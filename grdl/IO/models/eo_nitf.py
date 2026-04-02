# -*- coding: utf-8 -*-
"""
EO NITF Metadata - Typed dataclasses for electro-optical NITF imagery.

Provides metadata models for EO NITF files including RPC (Rational
Polynomial Coefficients) and RSM (Replacement Sensor Model) geolocation
parameters extracted from NITF TREs (Tagged Record Extensions).

Dependencies
------------
rasterio (for ``RPCCoefficients.from_rasterio`` factory method)

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
2026-03-17

Modified
--------
2026-04-01
"""

# Standard library
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.models.base import ImageMetadata
from grdl.IO.models.common import XYZ


# ===================================================================
# RPC (Rational Polynomial Coefficients) — RPC00B TRE
# ===================================================================


@dataclass
class RPCCoefficients:
    """RPC00B rational polynomial coefficients for ground-to-image mapping.

    The RPC model maps normalized geodetic coordinates (P, L, H) to
    normalized image coordinates (rn, cn) via cubic rational polynomials::

        rn = Σ line_num_coef[i] · ρ_i(P,L,H) / Σ line_den_coef[i] · ρ_i(P,L,H)
        cn = Σ samp_num_coef[i] · ρ_i(P,L,H) / Σ samp_den_coef[i] · ρ_i(P,L,H)

    where ρ is a 20-term monomial vector in normalized (P, L, H).

    Normalization::

        P = (lat - lat_off) / lat_scale
        L = (lon - long_off) / long_scale
        H = (hae - height_off) / height_scale

    De-normalization::

        row = line_off + line_scale · rn
        col = samp_off + samp_scale · cn

    Parameters
    ----------
    line_off : float
        Row (line) offset for de-normalization.
    samp_off : float
        Column (sample) offset for de-normalization.
    lat_off : float
        Latitude offset (degrees).
    long_off : float
        Longitude offset (degrees).
    height_off : float
        Height offset (meters HAE).
    line_scale : float
        Row scale for de-normalization.
    samp_scale : float
        Column scale for de-normalization.
    lat_scale : float
        Latitude scale (degrees).
    long_scale : float
        Longitude scale (degrees).
    height_scale : float
        Height scale (meters).
    line_num_coef : np.ndarray
        Line numerator coefficients, shape ``(20,)``.
    line_den_coef : np.ndarray
        Line denominator coefficients, shape ``(20,)``.
    samp_num_coef : np.ndarray
        Sample numerator coefficients, shape ``(20,)``.
    samp_den_coef : np.ndarray
        Sample denominator coefficients, shape ``(20,)``.
    err_bias : float, optional
        RPC fitting error bias (meters CE90).
    err_rand : float, optional
        RPC fitting error random (meters CE90).
    """

    line_off: float = 0.0
    samp_off: float = 0.0
    lat_off: float = 0.0
    long_off: float = 0.0
    height_off: float = 0.0
    line_scale: float = 1.0
    samp_scale: float = 1.0
    lat_scale: float = 1.0
    long_scale: float = 1.0
    height_scale: float = 1.0
    line_num_coef: np.ndarray = field(
        default_factory=lambda: np.zeros(20))
    line_den_coef: np.ndarray = field(
        default_factory=lambda: np.zeros(20))
    samp_num_coef: np.ndarray = field(
        default_factory=lambda: np.zeros(20))
    samp_den_coef: np.ndarray = field(
        default_factory=lambda: np.zeros(20))
    err_bias: Optional[float] = None
    err_rand: Optional[float] = None

    @classmethod
    def from_rasterio(cls, rpcs: object) -> 'RPCCoefficients':
        """Create from a ``rasterio.rpc.RPC`` object.

        Parameters
        ----------
        rpcs : rasterio.rpc.RPC
            RPC object from ``dataset.rpcs``.

        Returns
        -------
        RPCCoefficients
        """
        return cls(
            line_off=float(rpcs.line_off),
            samp_off=float(rpcs.samp_off),
            lat_off=float(rpcs.lat_off),
            long_off=float(rpcs.long_off),
            height_off=float(rpcs.height_off),
            line_scale=float(rpcs.line_scale),
            samp_scale=float(rpcs.samp_scale),
            lat_scale=float(rpcs.lat_scale),
            long_scale=float(rpcs.long_scale),
            height_scale=float(rpcs.height_scale),
            line_num_coef=np.array(rpcs.line_num_coeff, dtype=np.float64),
            line_den_coef=np.array(rpcs.line_den_coeff, dtype=np.float64),
            samp_num_coef=np.array(rpcs.samp_num_coeff, dtype=np.float64),
            samp_den_coef=np.array(rpcs.samp_den_coeff, dtype=np.float64),
            err_bias=(float(rpcs.err_bias)
                      if rpcs.err_bias is not None else None),
            err_rand=(float(rpcs.err_rand)
                      if rpcs.err_rand is not None else None),
        )


# ===================================================================
# RSM (Replacement Sensor Model) — RSMPCA / RSMIDA TREs
# ===================================================================


@dataclass
class RSMIdentification:
    """RSM identification metadata from the RSMIDA TRE.

    Parameters
    ----------
    image_id : str, optional
        Image identifier (IID field, 80 chars).
    edition : str, optional
        RSM edition string (40 chars).
    sensor_id : str, optional
        Sensor identifier (SID, 40 chars).
    sensor_type_id : str, optional
        Sensor type identifier (STID, 40 chars).
    image_sensor_id : str, optional
        Image sensor identifier (ISID, 40 chars).
    collection_datetime : datetime, optional
        Image collection date/time from YEAR/MONTH/DAY/HOUR/MINUTE/SECOND.
    ground_domain_type : str, optional
        Ground domain coordinate system: ``'G'`` (geodetic),
        ``'C'`` (cartographic), ``'R'`` (relative).
    ground_ref_point : XYZ, optional
        Ground reference point (interpretation depends on
        ``ground_domain_type``).
    num_row_sections : int, optional
        Number of row sections (NRG).
    num_col_sections : int, optional
        Number of column sections (NCG).
    time_ref_row : float, optional
        Time reference row (TRG).
    time_ref_col : float, optional
        Time reference column (TCG).
    coord_origin : XYZ, optional
        Rectangular coordinate system origin (XUOR, YUOR, ZUOR).
    coord_unit_vectors : np.ndarray, optional
        Rectangular coordinate unit vectors, shape ``(3, 3)``.
        Rows are x, y, z unit vectors; columns are XR, YR, ZR.
    ground_domain_vertices : np.ndarray, optional
        Ground domain boundary vertices, shape ``(8, 3)``.
        Eight corner points defining the RSM validity region.
    full_image_rows : int, optional
        Full image row count (FULLR).
    full_image_cols : int, optional
        Full image column count (FULLC).
    min_row : int, optional
        Minimum valid row (MINR).
    max_row : int, optional
        Maximum valid row (MAXR).
    min_col : int, optional
        Minimum valid column (MINC).
    max_col : int, optional
        Maximum valid column (MAXC).
    """

    image_id: Optional[str] = None
    edition: Optional[str] = None
    sensor_id: Optional[str] = None
    sensor_type_id: Optional[str] = None
    image_sensor_id: Optional[str] = None
    collection_datetime: Optional[datetime] = None
    ground_domain_type: Optional[str] = None
    ground_ref_point: Optional[XYZ] = None
    num_row_sections: Optional[int] = None
    num_col_sections: Optional[int] = None
    time_ref_row: Optional[float] = None
    time_ref_col: Optional[float] = None
    coord_origin: Optional[XYZ] = None
    coord_unit_vectors: Optional[np.ndarray] = None
    ground_domain_vertices: Optional[np.ndarray] = None
    full_image_rows: Optional[int] = None
    full_image_cols: Optional[int] = None
    min_row: Optional[int] = None
    max_row: Optional[int] = None
    min_col: Optional[int] = None
    max_col: Optional[int] = None


@dataclass
class RSMCoefficients:
    """RSM polynomial coefficients from the RSMPCA TRE.

    The RSM model maps normalized ground coordinates (x, y, z) to
    normalized image coordinates (row_n, col_n) via rational
    polynomials with variable-order terms::

        row_n = Σ row_num_coefs[k]·x^i·y^j·z^k / Σ row_den_coefs[k]·x^i·y^j·z^k
        col_n = Σ col_num_coefs[k]·x^i·y^j·z^k / Σ col_den_coefs[k]·x^i·y^j·z^k

    De-normalization::

        row = row_off + row_norm_sf · row_n
        col = col_off + col_norm_sf · col_n

    Parameters
    ----------
    row_off : float
        Row offset for de-normalization.
    col_off : float
        Column offset for de-normalization.
    row_norm_sf : float
        Row normalization scale factor.
    col_norm_sf : float
        Column normalization scale factor.
    x_off : float
        Ground X normalization offset.
    y_off : float
        Ground Y normalization offset.
    z_off : float
        Ground Z normalization offset.
    x_norm_sf : float
        Ground X normalization scale factor.
    y_norm_sf : float
        Ground Y normalization scale factor.
    z_norm_sf : float
        Ground Z normalization scale factor.
    row_num_powers : np.ndarray
        Max polynomial powers for row numerator, shape ``(3,)``
        as ``[max_x, max_y, max_z]``.
    row_den_powers : np.ndarray
        Max polynomial powers for row denominator, shape ``(3,)``.
    col_num_powers : np.ndarray
        Max polynomial powers for col numerator, shape ``(3,)``.
    col_den_powers : np.ndarray
        Max polynomial powers for col denominator, shape ``(3,)``.
    row_num_coefs : np.ndarray
        Row numerator polynomial coefficients.
    row_den_coefs : np.ndarray
        Row denominator polynomial coefficients.
    col_num_coefs : np.ndarray
        Column numerator polynomial coefficients.
    col_den_coefs : np.ndarray
        Column denominator polynomial coefficients.
    rsn : int, optional
        Row section number (1-based).  Identifies which row in
        the segment grid this polynomial covers.
    csn : int, optional
        Column section number (1-based).  Identifies which column
        in the segment grid this polynomial covers.
    row_fit_error : float, optional
        Row fitting error proportion (RFEP).
    col_fit_error : float, optional
        Column fitting error proportion (CFEP).
    """

    row_off: float = 0.0
    col_off: float = 0.0
    row_norm_sf: float = 1.0
    col_norm_sf: float = 1.0
    x_off: float = 0.0
    y_off: float = 0.0
    z_off: float = 0.0
    x_norm_sf: float = 1.0
    y_norm_sf: float = 1.0
    z_norm_sf: float = 1.0
    row_num_powers: np.ndarray = field(
        default_factory=lambda: np.array([3, 3, 3]))
    row_den_powers: np.ndarray = field(
        default_factory=lambda: np.array([3, 3, 3]))
    col_num_powers: np.ndarray = field(
        default_factory=lambda: np.array([3, 3, 3]))
    col_den_powers: np.ndarray = field(
        default_factory=lambda: np.array([3, 3, 3]))
    row_num_coefs: np.ndarray = field(
        default_factory=lambda: np.zeros(1))
    row_den_coefs: np.ndarray = field(
        default_factory=lambda: np.array([1.0]))
    col_num_coefs: np.ndarray = field(
        default_factory=lambda: np.zeros(1))
    col_den_coefs: np.ndarray = field(
        default_factory=lambda: np.array([1.0]))
    rsn: Optional[int] = None
    csn: Optional[int] = None
    row_fit_error: Optional[float] = None
    col_fit_error: Optional[float] = None


# ===================================================================
# RSM Segment Grid — multi-segment RSM support
# ===================================================================


@dataclass
class RSMSegmentGrid:
    """Grid of RSM polynomial segments for multi-section imagery.

    When an RSM model is partitioned into multiple sections (NRG × NCG),
    each section has its own RSMPCA TRE with different polynomial
    coefficients.  This container holds all segments indexed by their
    ``(rsn, csn)`` section coordinates (1-based).

    Parameters
    ----------
    num_row_sections : int
        Number of row sections (NRG from RSMIDA).
    num_col_sections : int
        Number of column sections (NCG from RSMIDA).
    segments : Dict[Tuple[int, int], RSMCoefficients]
        Mapping from ``(row_section, col_section)`` to the
        polynomial coefficients for that section.
    """

    num_row_sections: int = 1
    num_col_sections: int = 1
    segments: Dict[Tuple[int, int], RSMCoefficients] = field(
        default_factory=dict)


# ===================================================================
# Geospatial Accuracy and Context TREs
# ===================================================================


@dataclass
class CSEXRAMetadata:
    """Compensated Sensor Error Extension metadata (CSEXRA TRE).

    Provides positioning accuracy and ground sample distance estimates
    computed by the sensor or processing system.

    Parameters
    ----------
    predicted_niirs : float, optional
        Predicted National Imagery Interpretability Rating Scale value.
    ce90 : float, optional
        Circular error at 90%% confidence (meters).
    le90 : float, optional
        Linear (vertical) error at 90%% confidence (meters).
    ground_gsd_row : float, optional
        Ground sample distance in row direction (meters).
    ground_gsd_col : float, optional
        Ground sample distance in column direction (meters).
    """

    predicted_niirs: Optional[float] = None
    ce90: Optional[float] = None
    le90: Optional[float] = None
    ground_gsd_row: Optional[float] = None
    ground_gsd_col: Optional[float] = None


@dataclass
class USE00AMetadata:
    """Exploitation Usability Extension metadata (USE00A TRE).

    Provides sensor geometry and illumination parameters useful for
    exploitation and quality assessment.

    Parameters
    ----------
    obliquity_angle : float, optional
        Sensor obliquity angle (degrees from nadir).
    sun_azimuth : float, optional
        Sun azimuth angle (degrees clockwise from north).
    sun_elevation : float, optional
        Sun elevation angle (degrees above horizon).
    mean_gsd : float, optional
        Mean ground sample distance (meters).
    predicted_niirs : float, optional
        Predicted NIIRS value.
    """

    obliquity_angle: Optional[float] = None
    sun_azimuth: Optional[float] = None
    sun_elevation: Optional[float] = None
    mean_gsd: Optional[float] = None
    predicted_niirs: Optional[float] = None


@dataclass
class ICHIPBMetadata:
    """Image Chip Block metadata (ICHIPB TRE).

    Defines the affine relationship between chipped image pixel
    coordinates and the full (original) image pixel coordinates.
    Critical for evaluating RPC/RSM polynomials on chipped imagery.

    The transform from chip to full-image coordinates is::

        full_row = fi_row_off + fi_row_scale * chip_row
        full_col = fi_col_off + fi_col_scale * chip_col

    Parameters
    ----------
    xfrm_flag : int, optional
        Transform type: 0 = no transform, 1 = identity, 2 = affine.
    scale_factor_r : float, optional
        Row scale factor (chip-to-original).
    scale_factor_c : float, optional
        Column scale factor (chip-to-original).
    anamorphic_corr : float, optional
        Anamorphic correction factor.
    fi_row_off : float, optional
        Full image row offset for chip origin.
    fi_col_off : float, optional
        Full image column offset for chip origin.
    fi_row_scale : float, optional
        Full image row scale (typically 1.0 unless resampled).
    fi_col_scale : float, optional
        Full image column scale (typically 1.0 unless resampled).
    op_row : float, optional
        Original product row at chip center.
    op_col : float, optional
        Original product column at chip center.
    full_image_rows : int, optional
        Row count of the original full image.
    full_image_cols : int, optional
        Column count of the original full image.
    """

    xfrm_flag: Optional[int] = None
    scale_factor_r: Optional[float] = None
    scale_factor_c: Optional[float] = None
    anamorphic_corr: Optional[float] = None
    fi_row_off: Optional[float] = None
    fi_col_off: Optional[float] = None
    fi_row_scale: Optional[float] = None
    fi_col_scale: Optional[float] = None
    op_row: Optional[float] = None
    op_col: Optional[float] = None
    full_image_rows: Optional[int] = None
    full_image_cols: Optional[int] = None


@dataclass
class BLOCKAMetadata:
    """Image Geographic Location metadata (BLOCKA TRE).

    Provides geographic corner coordinates for image blocks,
    useful for quick spatial indexing and footprint extraction.

    Corner strings are in DDMMSS.SSh format (h = N/S or E/W).

    Parameters
    ----------
    block_number : int, optional
        Block number (typically 1 for single-block images).
    frfc_loc : str, optional
        First row, first column lat/lon.
    frlc_loc : str, optional
        First row, last column lat/lon.
    lrfc_loc : str, optional
        Last row, first column lat/lon.
    lrlc_loc : str, optional
        Last row, last column lat/lon.
    """

    block_number: Optional[int] = None
    frfc_loc: Optional[str] = None
    frlc_loc: Optional[str] = None
    lrfc_loc: Optional[str] = None
    lrlc_loc: Optional[str] = None


@dataclass
class CollectionInfo:
    """Aggregated collection context from AIMIDB, STDIDC, and PIAIMC TREs.

    Merges fields from multiple TREs into a single collection-context
    object.  Priority: AIMIDB > STDIDC > PIAIMC for overlapping fields.

    Parameters
    ----------
    collection_datetime : datetime, optional
        Image collection date/time.
    mission_id : str, optional
        Mission identifier.
    sensor_mode : str, optional
        Sensor imaging mode (e.g., ``'PAN'``, ``'MULTI'``).
    cloud_cover : float, optional
        Cloud cover percentage (0-100).
    country_code : str, optional
        Country code of imaged area.
    """

    collection_datetime: Optional[datetime] = None
    mission_id: Optional[str] = None
    sensor_mode: Optional[str] = None
    cloud_cover: Optional[float] = None
    country_code: Optional[str] = None


@dataclass
class AccuracyInfo:
    """Aggregated geospatial accuracy from available TREs.

    Consolidates positioning accuracy from the best available source.
    Priority: CSEXRA > USE00A > RPC ``err_bias``.

    Parameters
    ----------
    ce90 : float, optional
        Circular error at 90%% confidence (meters).
    le90 : float, optional
        Linear (vertical) error at 90%% confidence (meters).
    mean_gsd : float, optional
        Mean ground sample distance (meters).
    source : str, optional
        TRE that provided the accuracy values
        (e.g., ``'CSEXRA'``, ``'USE00A'``, ``'RPC00B'``).
    """

    ce90: Optional[float] = None
    le90: Optional[float] = None
    mean_gsd: Optional[float] = None
    source: Optional[str] = None


# ===================================================================
# Top-level EO NITF Metadata
# ===================================================================


@dataclass
class EONITFMetadata(ImageMetadata):
    """Metadata for electro-optical NITF imagery.

    Extends :class:`~grdl.IO.models.base.ImageMetadata` with RPC and
    RSM geolocation models, geospatial accuracy, collection context,
    NITF header fields, and band information.

    Parameters
    ----------
    rpc : RPCCoefficients, optional
        RPC00B rational polynomial coefficients.
    rsm : RSMCoefficients, optional
        RSM polynomial coefficients (RSMPCA TRE).  For multi-segment
        files this is the first segment; use ``rsm_segments`` for the
        full grid.
    rsm_id : RSMIdentification, optional
        RSM identification metadata (RSMIDA TRE).
    rsm_segments : RSMSegmentGrid, optional
        Full grid of RSM polynomial segments for multi-section imagery.
    csexra : CSEXRAMetadata, optional
        Compensated sensor error (CE90, LE90, GSD) from CSEXRA TRE.
    use00a : USE00AMetadata, optional
        Exploitation usability (sun angles, GSD) from USE00A TRE.
    ichipb : ICHIPBMetadata, optional
        Image chip transform from ICHIPB TRE.
    blocka : BLOCKAMetadata, optional
        Geographic corner coordinates from BLOCKA TRE.
    collection_info : CollectionInfo, optional
        Aggregated collection context from AIMIDB/STDIDC/PIAIMC.
    accuracy : AccuracyInfo, optional
        Aggregated geospatial accuracy from best available source.
    iid1 : str, optional
        NITF Image Identifier 1 (IID1, 10 chars).
    iid2 : str, optional
        NITF Image Identifier 2 (IID2, 80 chars).
    icords : str, optional
        Image coordinate representation (``'G'`` = geographic,
        ``'D'`` = decimal degrees, ``'N'`` = UTM North, etc.).
    icat : str, optional
        Image category (``'VIS'``, ``'MS'``, ``'IR'``, etc.).
    abpp : int, optional
        Actual bits per pixel.
    idatim : str, optional
        Image date/time from NITF header (IDATIM, 14 chars).
    tgtid : str, optional
        Target identifier from NITF header (TGTID, 17 chars).
    isource : str, optional
        Image source from NITF header (ISOURCE, 42 chars).
    igeolo : str, optional
        Image geographic location from NITF header (IGEOLO, 60 chars).
    band_names : List[str], optional
        Band names (e.g., ``['R', 'G', 'B', 'NIR']``).
    wavelengths : List[float], optional
        Center wavelengths per band (micrometers).

    Examples
    --------
    >>> from grdl.IO.models.eo_nitf import EONITFMetadata
    >>> meta = EONITFMetadata(
    ...     format='NITF', rows=4096, cols=4096, dtype='uint16',
    ...     bands=4, icat='MS',
    ... )
    >>> meta.rpc is None
    True
    """

    rpc: Optional[RPCCoefficients] = None
    rsm: Optional[RSMCoefficients] = None
    rsm_id: Optional[RSMIdentification] = None
    rsm_segments: Optional[RSMSegmentGrid] = None
    csexra: Optional[CSEXRAMetadata] = None
    use00a: Optional[USE00AMetadata] = None
    ichipb: Optional[ICHIPBMetadata] = None
    blocka: Optional[BLOCKAMetadata] = None
    collection_info: Optional[CollectionInfo] = None
    accuracy: Optional[AccuracyInfo] = None
    iid1: Optional[str] = None
    iid2: Optional[str] = None
    icords: Optional[str] = None
    icat: Optional[str] = None
    abpp: Optional[int] = None
    idatim: Optional[str] = None
    tgtid: Optional[str] = None
    isource: Optional[str] = None
    igeolo: Optional[str] = None
    band_names: Optional[List[str]] = None
    wavelengths: Optional[List[float]] = None
