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
2026-03-17
"""

# Standard library
from dataclasses import dataclass, field
from typing import List, Optional

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
    """

    image_id: Optional[str] = None
    edition: Optional[str] = None
    sensor_id: Optional[str] = None
    sensor_type_id: Optional[str] = None
    ground_domain_type: Optional[str] = None
    ground_ref_point: Optional[XYZ] = None
    num_row_sections: Optional[int] = None
    num_col_sections: Optional[int] = None


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


# ===================================================================
# Top-level EO NITF Metadata
# ===================================================================


@dataclass
class EONITFMetadata(ImageMetadata):
    """Metadata for electro-optical NITF imagery.

    Extends :class:`~grdl.IO.models.base.ImageMetadata` with RPC and
    RSM geolocation models, NITF header fields, and band information.

    Parameters
    ----------
    rpc : RPCCoefficients, optional
        RPC00B rational polynomial coefficients.
    rsm : RSMCoefficients, optional
        RSM polynomial coefficients (RSMPCA TRE).
    rsm_id : RSMIdentification, optional
        RSM identification metadata (RSMIDA TRE).
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
    iid1: Optional[str] = None
    iid2: Optional[str] = None
    icords: Optional[str] = None
    icat: Optional[str] = None
    abpp: Optional[int] = None
    band_names: Optional[List[str]] = None
    wavelengths: Optional[List[float]] = None
