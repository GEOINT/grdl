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
170194430+DDSmalls@users.noreply.github.com

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
2026-06-09  Add ImageGroupInfo + heterogeneous-segment fields on
            ImageSegmentInfo (group_id, IMAG, ILOC/IALVL, placement).
2026-04-17  Add CSEPHA, RSMGGA dataclasses and RSMSegmentGrid helper.
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
from grdl.IO.models.eo_airborne import (
    ACFTBMetadata,
    MENSRAMetadata,
    MENSRBMetadata,
    SENSRBMetadata,
)
from grdl.IO.models.eo_band import BANDSAMetadata, BANDSBMetadata
from grdl.IO.models.rsm_error import (
    RSMAPAMetadata,
    RSMDCAMetadata,
    RSMECAMetadata,
    RSMPIAMetadata,
)


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
        Ground domain coordinate system per STDI-0002 App U §5.3:
        ``'G'`` (geodetic; x=lon-rad, y=lat-rad, z=hae-m),
        ``'H'`` (geocentric; same as G with WGS-84 specifics),
        ``'R'`` (rectangular; local Cartesian xyz in meters).
    ground_ref_point : XYZ, optional
        Ground reference point (GRPX, GRPY, GRPZ).  Units depend on
        ``ground_domain_type``: for ``'G'``/``'H'`` x=lon (radians),
        y=lat (radians), z=height-above-ellipsoid (meters); for
        ``'R'`` xyz are local rectangular coordinates (meters)
        relative to ``coord_origin``.
    num_row_sections : int, optional
        Number of row sections (NRG).
    num_col_sections : int, optional
        Number of column sections (NCG).
    time_ref_row : float, optional
        Time reference row (TRG).
    time_ref_col : float, optional
        Time reference column (TCG).
    coord_origin : XYZ, optional
        Rectangular coordinate system origin (XUOR, YUOR, ZUOR),
        WGS-84 ECEF meters.  Present only when
        ``ground_domain_type='R'``; ``None`` otherwise.
    coord_unit_vectors : np.ndarray, optional
        Rectangular coordinate unit-vector matrix, shape ``(3, 3)``.
        Each row is a local axis expressed in WGS-84 ECEF
        components: row 0 = local-X axis ``[XUXR, YUXR, ZUXR]``,
        row 1 = local-Y axis ``[XUYR, YUYR, ZUYR]``,
        row 2 = local-Z axis ``[XUZR, YUZR, ZUZR]``.
        All zeros when ``ground_domain_type != 'R'``.
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

    Image-coordinate convention follows STDI-0002 Vol 1 App U: the
    upper-left pixel center is at ``(0.5, 0.5)`` (pixel-corner origin).
    GRDL uses the same 0-based pixel-corner convention as GDAL/rasterio,
    so the values returned by RSM evaluation can be passed directly to
    pixel-array indexers without an extra offset.
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

    def segment_for_pixel(
        self,
        row: float,
        col: float,
    ) -> Optional[RSMCoefficients]:
        """Return the RSM segment whose validity region covers a pixel.

        Each RSMPCA segment declares its valid-row / valid-col window via
        ``(row_off ± |row_norm_sf|, col_off ± |col_norm_sf|)``.  This
        helper returns the first segment whose window contains the
        given (row, col).  When the pixel falls outside every segment
        (which can happen near image boundaries due to the
        normalization-window / partition-grid mismatch allowed by
        STDI-0002 §U.7), the spatially nearest segment is returned.

        Parameters
        ----------
        row, col : float
            Full-image pixel coordinates (not chip coordinates).

        Returns
        -------
        RSMCoefficients or None
            The selected segment, or ``None`` when the grid is empty.
        """
        if not self.segments:
            return None

        best_key: Optional[Tuple[int, int]] = None
        best_dist = float('inf')
        for key, seg in self.segments.items():
            row_half = abs(seg.row_norm_sf)
            col_half = abs(seg.col_norm_sf)
            d_row = row - seg.row_off
            d_col = col - seg.col_off
            inside = (abs(d_row) <= row_half) and (abs(d_col) <= col_half)
            dist = d_row * d_row + d_col * d_col
            if inside and dist < best_dist:
                best_dist = dist
                best_key = key

        if best_key is None:
            # No segment contains the point; pick the nearest by centre.
            # Reset the running minimum so the fallback search is
            # independent of the inside-test pass.
            best_dist = float('inf')
            for key, seg in self.segments.items():
                d_row = row - seg.row_off
                d_col = col - seg.col_off
                dist = d_row * d_row + d_col * d_col
                if dist < best_dist:
                    best_dist = dist
                    best_key = key

        return self.segments.get(best_key) if best_key is not None else None


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

    Per STDI-0002 Vol 1 Appendix B (ICHIPB v1.0/CN2 — 224 bytes), the
    ICHIPB SDE ties an output-product (chip) image back to its source
    full-image (FI) coordinate system.  All 22 spec fields are required;
    each is stored verbatim below.  Convenience accessors expose the
    common axis-aligned chip→full affine and a full 2D-affine fit for
    rotated chips (Annex A.3 / A.4).

    Coordinate system
    -----------------
    The chip-to-full-image affine in ``full = off + scale * chip`` form
    is exposed via :attr:`fi_row_off`, :attr:`fi_col_off`,
    :attr:`fi_row_scale`, :attr:`fi_col_scale` for the simple separable
    (axis-aligned) case used by RSM/RPC geolocators in this library.
    For rotated chips use :meth:`chip_to_full_affine` /
    :meth:`full_to_chip_affine` which solve the over-determined
    four-corner system.

    Required spec fields (stored verbatim)
    --------------------------------------
    xfrm_flag : int
        Non-linear transformation flag (Table B-2): ``0`` = the chip
        contains non-dewarped data (the OP/FI corner fields are valid),
        ``1`` = no transformation data provided (the remaining fields
        are zero-fill per spec; consumers must skip them).
    scale_factor : float
        Scale factor relative to R0 (original full-image resolution).
        Allowed values are the discrete RRDS codes ``1, 2, 4, 8, 16,
        32, 64, 128`` (R0 through R7) or the reciprocal of the image
        magnification when not scaled by powers of 2.  Per the spec
        the value must directly correlate with the IMAG field of the
        image segment subheader.
    anamorphic_corr : int
        Anamorphic correction indicator (binary): ``0`` = no
        anamorphic correction has been applied to the chip, ``1`` =
        correction applied.  When ``1``, the simple separable affine
        is no longer the correct mapping and callers should use the
        4-corner fit.
    scanblk_num : int
        Scan-block index ``0..99``; ``0`` if not applicable.  When
        chipping from imagery with multiple scan blocks, identifies
        which scan block the chip was extracted from so the correct
        per-scan-block SDEs can be selected.
    op_row_11, op_col_11 : float
        Output product row/col of the chip's upper-left corner
        (intelligent-pixel index ``(1,1)``).  Typically ``0.5``
        following pixel-corner-origin convention.
    op_row_12, op_col_12 : float
        Upper-right corner ``(1,2)``.
    op_row_21, op_col_21 : float
        Lower-left corner ``(2,1)``.
    op_row_22, op_col_22 : float
        Lower-right corner ``(2,2)``.
    fi_row_11, fi_col_11 : float
        Full-image row/col index for chip corner ``(1,1)``.
    fi_row_12, fi_col_12 : float
        Full-image index for chip corner ``(1,2)``.
    fi_row_21, fi_col_21 : float
        Full-image index for chip corner ``(2,1)``.
    fi_row_22, fi_col_22 : float
        Full-image index for chip corner ``(2,2)``.
    full_image_rows, full_image_cols : int
        Row / column count of the original full image.  ``0`` (or
        ``None``) carries the spec's "unknown" semantic — the
        chipping application did not have access to the full-image
        size; consumers must not presume zero area.

    Derived axial affine (convenience)
    ----------------------------------
    fi_row_off, fi_col_off : float
        Offset terms of the separable ``full = off + scale * chip``
        affine derived from the 11 / 22 corners.  ``None`` when
        ``xfrm_flag == 1`` (no data provided).
    fi_row_scale, fi_col_scale : float
        Scale terms of the separable affine.  ``None`` when
        ``xfrm_flag == 1``.

    Back-compat aliases
    -------------------
    scale_factor_r, scale_factor_c : float
        Both alias :attr:`scale_factor`.  ICHIPB declares one
        SCALE_FACTOR; pre-existing call sites that read row/col
        variants continue to work.
    op_row, op_col : float
        Alias :attr:`op_row_22` / :attr:`op_col_22` (the chip's
        lower-right corner — historical, retained for
        backward compatibility).
    """

    # Required spec fields ------------------------------------------------
    xfrm_flag: Optional[int] = None
    scale_factor: Optional[float] = None
    anamorphic_corr: Optional[int] = None
    scanblk_num: Optional[int] = None

    op_row_11: Optional[float] = None
    op_col_11: Optional[float] = None
    op_row_12: Optional[float] = None
    op_col_12: Optional[float] = None
    op_row_21: Optional[float] = None
    op_col_21: Optional[float] = None
    op_row_22: Optional[float] = None
    op_col_22: Optional[float] = None

    fi_row_11: Optional[float] = None
    fi_col_11: Optional[float] = None
    fi_row_12: Optional[float] = None
    fi_col_12: Optional[float] = None
    fi_row_21: Optional[float] = None
    fi_col_21: Optional[float] = None
    fi_row_22: Optional[float] = None
    fi_col_22: Optional[float] = None

    full_image_rows: Optional[int] = None
    full_image_cols: Optional[int] = None

    # Derived axial affine (set by the parsers) ---------------------------
    fi_row_off: Optional[float] = None
    fi_col_off: Optional[float] = None
    fi_row_scale: Optional[float] = None
    fi_col_scale: Optional[float] = None

    # Back-compat aliases (set by the parsers) ----------------------------
    scale_factor_r: Optional[float] = None
    scale_factor_c: Optional[float] = None
    op_row: Optional[float] = None
    op_col: Optional[float] = None

    @property
    def is_no_transform_provided(self) -> bool:
        """True when ``xfrm_flag == 1`` (spec: zero-fill remainder).

        Per STDI-0002 Vol 1 App B Table B-2 / B.7, ``XFRM_FLAG=01``
        means the chip carries data other than non-dewarped imagery
        and the OP/FI corner fields are populated with the designated
        zero-fill defaults.  Callers must not compute an affine in
        that case.
        """
        return self.xfrm_flag == 1

    @property
    def has_full_image_size(self) -> bool:
        """True when full-image dimensions are known.

        Per Table B-3 / B.8.2, the spec's default ``00000000`` for
        FI_ROW / FI_COL means the chipping application did not know
        the full-image extent.  Returns ``False`` in that case.
        """
        return bool(self.full_image_rows) and bool(self.full_image_cols)

    @property
    def has_anamorphic_correction(self) -> bool:
        """True iff ``anamorphic_corr == 1``."""
        return self.anamorphic_corr == 1

    def chip_to_full_affine(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Solve a 2D affine ``(M, b)`` mapping chip → full-image pixels.

        For non-rotated (axis-aligned) chips this reduces to a
        diagonal ``M`` and matches the separable
        ``fi_*_off``/``fi_*_scale`` fields.  For rotated chips
        (Annex A.3) or chips with anamorphic correction
        (``anamorphic_corr == 1``), the matrix carries the off-axis
        terms that the separable form cannot represent.

        Returns
        -------
        (M, b) : Tuple[np.ndarray, np.ndarray] or None
            ``M`` is shape ``(2, 2)``, ``b`` is shape ``(2,)``;
            ``[fi_row, fi_col] = M @ [op_row, op_col] + b``.
            Returns ``None`` when ``xfrm_flag == 1`` (no data
            provided) or any required corner is missing.
        """
        if self.is_no_transform_provided:
            return None
        op = self._stack_corners('op')
        fi = self._stack_corners('fi')
        if op is None or fi is None:
            return None
        # Least-squares solve of fi_x = M @ op + b for each output
        # axis: ``A @ params = fi_axis`` where A is [[op_r, op_c, 1]]
        # rows and params is (m_r, m_c, b).  Four corners over-
        # determine the 3-parameter affine; lstsq is robust.
        a = np.column_stack([op, np.ones(4)])
        m = np.zeros((2, 2), dtype=np.float64)
        b = np.zeros(2, dtype=np.float64)
        for axis in (0, 1):
            sol, *_ = np.linalg.lstsq(a, fi[:, axis], rcond=None)
            m[axis, :] = sol[:2]
            b[axis] = sol[2]
        return m, b

    def full_to_chip_affine(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Inverse of :meth:`chip_to_full_affine`.

        Returns ``(M_inv, b_inv)`` such that
        ``[op_row, op_col] = M_inv @ [fi_row, fi_col] + b_inv``.
        Returns ``None`` when no transform data is provided or the
        forward matrix is singular.
        """
        forward = self.chip_to_full_affine()
        if forward is None:
            return None
        m, b = forward
        det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
        if abs(det) < 1e-15:
            return None
        m_inv = np.linalg.inv(m)
        b_inv = -m_inv @ b
        return m_inv, b_inv

    def _stack_corners(
        self, kind: str,
    ) -> Optional[np.ndarray]:
        """Return shape ``(4, 2)`` corner array for ``'op'`` or ``'fi'``.

        Order: 11, 12, 21, 22.  ``None`` when any corner is missing.
        """
        prefix_r = f'{kind}_row'
        prefix_c = f'{kind}_col'
        rows = []
        cols = []
        for corner in ('11', '12', '21', '22'):
            r = getattr(self, f'{prefix_r}_{corner}')
            c = getattr(self, f'{prefix_c}_{corner}')
            if r is None or c is None:
                return None
            rows.append(r)
            cols.append(c)
        return np.column_stack([rows, cols]).astype(np.float64)


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
class CSEPHAMetadata:
    """Commercial Support Ephemeris metadata (CSEPHA TRE).

    Dense ECEF position samples of the sensor during the collection,
    per STDI-0002 Vol 1 Appendix D (Table D-3).  Combined with
    CSEXRA ``time_image_duration`` this reconstructs a position curve
    that a native R/Rdot geolocation chain can differentiate or
    interpolate (Lagrange) for higher accuracy than corner GCPs.

    Parameters
    ----------
    ephem_flag : str, optional
        ``'PREDICTED'``, ``'COLLECT-TIME'``, or ``'REFINED'``.
    dt_ephem : float, optional
        Time interval between vectors, seconds.
    date_ephem : str, optional
        Date of first vector, ``YYYYMMDD`` (UTC).
    t0_ephem : float, optional
        Time of first vector as seconds since ``date_ephem`` 00:00 UTC
        (converted from the raw ``HHMMSS.mmmmmm`` field).
    num_ephem : int, optional
        Number of ephemeris samples.
    position : np.ndarray, optional
        ECEF position samples in meters, shape ``(num_ephem, 3)``
        with columns ``[x, y, z]``.  Ordered chronologically (reversed
        when ``CSEXRA.time_image_duration`` is negative — callers must
        honour that flag to restore time-ordering).
    """

    ephem_flag: Optional[str] = None
    dt_ephem: Optional[float] = None
    date_ephem: Optional[str] = None
    t0_ephem: Optional[float] = None
    num_ephem: Optional[int] = None
    position: Optional[np.ndarray] = None


@dataclass
class RSMGGAGridPlane:
    """One plane of the RSM Ground-to-Image Grid (RSMGGA TRE).

    Each plane samples (row, col) at a regularly spaced ground grid at
    a fixed height-above-ellipsoid.  Stacking planes provides the
    volumetric lookup used for fast, inverse-polynomial-free ground→
    image projection (STDI-0002 Vol 1 Appendix U Table 12).

    Parameters
    ----------
    z_plane : float
        Plane height-above-ellipsoid, meters.
    xi : float
        X-coordinate of the first grid point in this plane (radians for
        geodetic ground domain, meters for rectangular).
    yi : float
        Y-coordinate of the first grid point in this plane.
    num_x : int
        Number of grid points along the X axis for this plane.
    num_y : int
        Number of grid points along the Y axis for this plane.
    rows : np.ndarray
        Image-row offsets from ``REFROW`` at each grid point,
        shape ``(num_x, num_y)``.  NaN when the spec places a
        "no measurement" (all-spaces) entry.
    cols : np.ndarray
        Image-column offsets from ``REFCOL`` at each grid point,
        shape ``(num_x, num_y)``.  NaN where unavailable.
    """

    z_plane: float
    xi: float
    yi: float
    num_x: int
    num_y: int
    rows: np.ndarray
    cols: np.ndarray


@dataclass
class RSMGGAMetadata:
    """RSM Ground-to-Image Grid metadata (RSMGGA TRE).

    Container for the volumetric ground→image grid produced alongside
    the RSMPCA polynomial section, used either as a seed for iterative
    inversion or as the primary projector when configured.  Fields
    follow STDI-0002 Vol 1 Appendix U Table 12.

    Parameters
    ----------
    iid : str, optional
        Image identifier.
    edition : str, optional
        RSM support data edition.
    ggrsn : int, optional
        Row section number this grid corresponds to (1-based).
    ggcsn : int, optional
        Column section number this grid corresponds to (1-based).
    row_fit_error : float, optional
        Reported row fit error (pixels).
    col_fit_error : float, optional
        Reported column fit error (pixels).
    interpolation_order : int, optional
        Interpolation order (0–3) declared by the producer.
    num_planes : int, optional
        Number of grid planes (``P``).
    delta_z : float, optional
        Height step between adjacent planes (meters).
    delta_x : float, optional
        Step along the X axis (radians or meters).
    delta_y : float, optional
        Step along the Y axis (radians or meters).
    ref_row : int, optional
        Reference image row (``REFROW``); plane-level ``rows`` arrays
        are offsets relative to this value.
    ref_col : int, optional
        Reference image column (``REFCOL``).
    planes : List[RSMGGAGridPlane], optional
        One entry per height plane in ascending index.
    """

    iid: Optional[str] = None
    edition: Optional[str] = None
    ggrsn: Optional[int] = None
    ggcsn: Optional[int] = None
    row_fit_error: Optional[float] = None
    col_fit_error: Optional[float] = None
    interpolation_order: Optional[int] = None
    num_planes: Optional[int] = None
    delta_z: Optional[float] = None
    delta_x: Optional[float] = None
    delta_y: Optional[float] = None
    ref_row: Optional[int] = None
    ref_col: Optional[int] = None
    planes: Optional[List[RSMGGAGridPlane]] = None


@dataclass
class ImageSegmentInfo:
    """Per-segment summary for unified multi-image NITF readers.

    Populated by :class:`grdl.IO.eo.nitf.EONITFReader` when a NITF file
    contains multiple physical image segments (GDAL subdatasets accessed
    via ``NITF_IM:N:/path/file.ntf``).  Each segment occupies a
    rectangular region of its group's full-image grid; the bbox here is
    in full-image pixel coordinates derived from that segment's ICHIPB
    TRE (per STDI-0002 Vol 1 App G), its ILOC/IALVL attachment chain
    (per MIL-STD-2500C), or sequential stacking as a last resort.

    Parameters
    ----------
    segment_index : int
        0-based subdataset order as reported by GDAL.
    uri : str
        Subdataset URI (e.g. ``"NITF_IM:1:/path/file.ntf"``).
    fi_row_lo, fi_row_hi : int
        Full-image row bbox covered by this segment, half-open
        ``[fi_row_lo, fi_row_hi)``.
    fi_col_lo, fi_col_hi : int
        Full-image column bbox, half-open.
    rows, cols : int
        Segment-local pixel dimensions (i.e. the dimensions GDAL
        reports for this subdataset).
    ichipb : ICHIPBMetadata, optional
        The segment's original ICHIPB TRE.  ``None`` when the segment
        carries no ICHIPB.
    scale_factor : float
        ICHIPB ``SCALE_FACTOR`` for this segment, or ``1.0`` when
        absent.  Segments with differing scale factors are placed in
        separate groups; the unified view covers one group only.
    group_id : int
        Index of the segment group this segment belongs to.  Segments
        are grouped by compatible imaging characteristics (band count,
        dtype, ICHIPB scale factor, IMAG magnification, ICAT).
    is_primary : bool
        ``True`` when this segment belongs to the primary group — the
        group the unified reader exposes for ``read_chip`` and that
        geolocation models operate in.
    bands : int
        Number of bands in this segment.
    dtype : str
        Pixel dtype string for this segment (e.g. ``'uint16'``).
    iid1 : str, optional
        NITF IID1 image identifier from the segment subheader.
    icat : str, optional
        NITF ICAT image category (``'VIS'``, ``'MS'``, ``'CLOUD'``...).
    irep : str, optional
        NITF IREP image representation (``'MONO'``, ``'MULTI'``...).
    imag : float
        Decoded NITF IMAG magnification (``'/2'`` → ``0.5``).  ``1.0``
        when absent — full resolution.
    idlvl, ialvl : int, optional
        NITF display / attachment level from the segment subheader.
    iloc_row, iloc_col : int, optional
        NITF ILOC offsets from the segment subheader (relative to the
        attached-to segment per MIL-STD-2500C).
    placement : str
        How the bbox was derived: ``'ichipb'``, ``'iloc'`` (attachment
        chain into the common coordinate system), or ``'stacked'``
        (sequential row stacking fallback).
    """

    segment_index: int
    uri: str
    fi_row_lo: int
    fi_row_hi: int
    fi_col_lo: int
    fi_col_hi: int
    rows: int
    cols: int
    ichipb: Optional[ICHIPBMetadata] = None
    scale_factor: float = 1.0
    group_id: int = 0
    is_primary: bool = True
    bands: int = 1
    dtype: str = ''
    iid1: Optional[str] = None
    icat: Optional[str] = None
    irep: Optional[str] = None
    imag: float = 1.0
    idlvl: Optional[int] = None
    ialvl: Optional[int] = None
    iloc_row: Optional[int] = None
    iloc_col: Optional[int] = None
    placement: str = 'ichipb'


@dataclass
class ImageGroupInfo:
    """Summary of one segment group in a multi-image NITF.

    A *group* is a set of image segments with compatible imaging
    characteristics (band count, dtype, ICHIPB scale factor, IMAG
    magnification, ICAT) that unify into a single full-image pixel
    grid.  A heterogeneous NITF (primary imagery plus overviews,
    cloud masks, or other support images) yields several groups; the
    reader unifies the *primary* group and leaves the rest accessible
    via ``image_index`` pinning.

    Parameters
    ----------
    group_id : int
        Index of this group (matches ``ImageSegmentInfo.group_id``).
    segment_indices : List[int]
        Subdataset indices of the member segments.
    rows, cols : int
        Unified full-image dimensions of this group (union of member
        bboxes).
    bands : int
        Band count shared by all member segments.
    dtype : str
        Pixel dtype shared by all member segments.
    scale_factor : float
        ICHIPB ``SCALE_FACTOR`` shared by all member segments.
    imag : float
        Decoded IMAG magnification shared by all member segments.
    icat : str, optional
        NITF ICAT category shared by all member segments.
    is_primary : bool
        Whether this is the group the unified reader exposes.
    placement : str
        Placement mode used for member bboxes: ``'ichipb'``,
        ``'iloc'``, or ``'stacked'``.
    has_geolocation : bool
        Whether any member segment carries RPC or RSM geolocation.
    """

    group_id: int
    segment_indices: List[int]
    rows: int
    cols: int
    bands: int
    dtype: str
    scale_factor: float = 1.0
    imag: float = 1.0
    icat: Optional[str] = None
    is_primary: bool = False
    placement: str = 'ichipb'
    has_geolocation: bool = False


@dataclass
class CSCRNAMetadata:
    """CSCRNA TRE -- image corner footprint with per-corner heights.

    Corner Footprint TRE per STDI-0002 (109-byte CEDATA): a predicted
    or measured geographic footprint of the image with WGS84 lat/lon
    and height at each corner.  Preferred over IGEOLO (which is
    truncated to arc-second-level precision) when present.

    Parameters
    ----------
    predicted : bool
        ``True`` when corners are predicted (``PREDICT_CORNERS='Y'``)
        rather than measured.
    corners : np.ndarray
        Corner lat/lon in degrees, shape ``(4, 2)`` ordered
        ``[UL, UR, LR, LL]`` with columns ``[lat, lon]``.
    heights : np.ndarray, optional
        Corner heights (meters, HAE), shape ``(4,)`` in the same
        corner order.  ``None`` when absent.
    """

    predicted: bool
    corners: np.ndarray
    heights: Optional[np.ndarray] = None

    @property
    def mean_height(self) -> Optional[float]:
        """Mean corner height in meters, or ``None`` when unavailable."""
        if self.heights is None:
            return None
        return float(np.mean(self.heights))


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
    csepha : CSEPHAMetadata, optional
        Dense ECEF ephemeris samples (CSEPHA TRE) — enables native
        R/Rdot-style projection from the raw sensor path.
    rsmgga : RSMGGAMetadata, optional
        RSM volumetric ground-to-image grid (RSMGGA TRE) — faster /
        non-iterative ground→image projection than RSMPCA alone.
    rsm_pia : RSMPIAMetadata, optional
        RSM polynomial identification (RSMPIA TRE) — low-order
        polynomials + section layout for multi-section RSM.
    rsm_dca : RSMDCAMetadata, optional
        RSM direct error covariance (RSMDCA/RSMDCB TRE); the
        ``variant`` field distinguishes A/B layouts.
    rsm_eca : RSMECAMetadata, optional
        RSM indirect error covariance (RSMECA/RSMECB TRE).
    rsm_apa : RSMAPAMetadata, optional
        RSM adjustable parameters (RSMAPA/RSMAPB TRE) — bundle
        adjustment hook.
    bandsb : BANDSBMetadata, optional
        Multispectral band characterization (BANDSB TRE); also feeds
        ``band_names`` / ``wavelengths``.
    bandsa : BANDSAMetadata, optional
        Band parameters (BANDSA TRE).
    sensrb : SENSRBMetadata, optional
        Airborne sensor model (SENSRB TRE).
    mensrb : MENSRBMetadata, optional
        Airborne mensuration data (MENSRB TRE).
    mensra : MENSRAMetadata, optional
        Airborne mensuration data, legacy variant (MENSRA TRE).
    acftb : ACFTBMetadata, optional
        Aircraft information (ACFTB TRE).
    cscrna : CSCRNAMetadata, optional
        Corner footprint with per-corner heights (CSCRNA TRE) —
        preferred over IGEOLO for corner-based fallback geolocation.
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
    image_segments : List[ImageSegmentInfo], optional
        Per-segment summary populated by readers that unify a
        multi-image NITF into a single full-image grid.  ``None`` for
        single-segment files and for readers pinned to one segment via
        ``image_index=N``.  When non-``None``, ``rows`` and ``cols``
        are the full-image dimensions, ``ichipb`` is ``None`` (the
        unified reader has already absorbed the chip-to-full
        transform), and each entry locates one segment in the unified
        grid for diagnostics.
    image_groups : List[ImageGroupInfo], optional
        Per-group summary for heterogeneous multi-image NITFs.  Each
        group is a set of compatible segments unifying into one pixel
        grid; the entry with ``is_primary=True`` is the group ``rows``
        / ``cols`` and ``read_chip`` operate in.  ``None`` for
        single-segment files and pinned readers.

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
    cscrna: Optional[CSCRNAMetadata] = None
    ichipb: Optional[ICHIPBMetadata] = None
    blocka: Optional[BLOCKAMetadata] = None
    csepha: Optional[CSEPHAMetadata] = None
    rsmgga: Optional[RSMGGAMetadata] = None
    rsm_pia: Optional[RSMPIAMetadata] = None
    rsm_dca: Optional[RSMDCAMetadata] = None
    rsm_eca: Optional[RSMECAMetadata] = None
    rsm_apa: Optional[RSMAPAMetadata] = None
    bandsb: Optional[BANDSBMetadata] = None
    bandsa: Optional[BANDSAMetadata] = None
    sensrb: Optional[SENSRBMetadata] = None
    mensrb: Optional[MENSRBMetadata] = None
    mensra: Optional[MENSRAMetadata] = None
    acftb: Optional[ACFTBMetadata] = None
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
    image_segments: Optional[List[ImageSegmentInfo]] = None
    image_groups: Optional[List[ImageGroupInfo]] = None
