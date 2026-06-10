# -*- coding: utf-8 -*-
"""
RSM error-model TRE metadata -- RSMPIA, RSMDCA, RSMECA, RSMAPA dataclasses.

Typed containers for the RSM (Replacement Sensor Model) error-model
Tagged Record Extensions defined in STDI-0002 Vol 1 Appendix U:

* ``RSMPIAMetadata`` -- Polynomial Identification (RSMPIA, Table 3).
* ``RSMDCAMetadata`` -- Direct Error Covariance (RSMDCA, Table 9) and
  its B-variant RSMDCB (Table 6 of the v5.0 appendix).
* ``RSMECAMetadata`` -- Indirect Error Covariance (RSMECA, Table 8)
  and its B-variant RSMECB (Table 10 of the v5.0 appendix).
* ``RSMAPAMetadata`` -- Adjustable Parameters (RSMAPA, Table 10) and
  its B-variant RSMAPB (Table 8 of the v5.0 appendix).

The B-variants are parsed into the same dataclasses with
``variant='B'``; the ``variant`` docstrings on each class state
honestly which fields are populated at which depth for each variant.
All classes carry an optional ``raw`` field retaining the full
original payload so no information is lost for downstream consumers
that need the parts this parser does not model.

Adjustable-parameter ordering
-----------------------------
The A-variant TREs reference a canonical set of 36 adjustable
parameters via thirty-six 2-byte index fields.  Throughout this
module the canonical order is::

    IR0 IRX IRY IRZ IRXX IRXY IRXZ IRYY IRYZ IRZZ   (row polynomial, 10)
    IC0 ICX ICY ICZ ICXX ICXY ICXZ ICYY ICYZ ICZZ   (col polynomial, 10)
    GXO GYO GZO GXR GYR GZR GS
    GXX GXY GXZ GYX GYY GYZ GZX GZY GZZ             (ground, 16)

An index value of 0 means "parameter not adjusted"; positive values
give the 1-based position of that parameter in the covariance /
parameter-value vectors.

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
2026-06-09

Modified
--------
2026-06-09
"""

# Standard library
from dataclasses import dataclass, field
from typing import List, Optional

# Third-party
import numpy as np


#: Names of the 10 low-order polynomial terms, in storage order.
RSM_POLY_TERM_ORDER = (
    '0', 'X', 'Y', 'Z', 'XX', 'XY', 'XZ', 'YY', 'YZ', 'ZZ',
)

#: Canonical order of the 36 adjustable-parameter index fields.
RSM_ADJUSTABLE_PARAM_ORDER = (
    'IR0', 'IRX', 'IRY', 'IRZ', 'IRXX', 'IRXY', 'IRXZ',
    'IRYY', 'IRYZ', 'IRZZ',
    'IC0', 'ICX', 'ICY', 'ICZ', 'ICXX', 'ICXY', 'ICXZ',
    'ICYY', 'ICYZ', 'ICZZ',
    'GXO', 'GYO', 'GZO', 'GXR', 'GYR', 'GZR', 'GS',
    'GXX', 'GXY', 'GXZ', 'GYX', 'GYY', 'GYZ', 'GZX', 'GZY', 'GZZ',
)


@dataclass
class RSMPIAMetadata:
    """RSMPIA -- RSM Polynomial Identification (STDI-0002 App U Table 3).

    RSMPIA CEDATA is fixed-length (591 bytes)::

        IID(80) EDITION(40)
        R0 RX RY RZ RXX RXY RXZ RYY RYZ RZZ   (10 x 21)
        C0 CX CY CZ CXX CXY CXZ CYY CYZ CZZ   (10 x 21)
        RNIS(3) CNIS(3) TNIS(3) RSSIZ(21) CSSIZ(21)

    The low-order polynomials map ground coordinates to the row /
    column used for image-section selection in multi-section RSMs.

    Parameters
    ----------
    image_id : str, optional
        Image identifier (IID).
    edition : str, optional
        RSM edition identifier (EDITION).
    row_poly : np.ndarray
        Low-order row polynomial coefficients, shape ``(10,)``
        float64, ordered ``[R0, RX, RY, RZ, RXX, RXY, RXZ, RYY,
        RYZ, RZZ]`` (see :data:`RSM_POLY_TERM_ORDER`).
    col_poly : np.ndarray
        Low-order column polynomial coefficients, shape ``(10,)``
        float64, ordered ``[C0, CX, ..., CZZ]``.
    num_row_sections : int, optional
        Number of image sections in the row direction (RNIS).
    num_col_sections : int, optional
        Number of image sections in the column direction (CNIS).
    num_total_sections : int, optional
        Total number of image sections (TNIS).
    row_section_size : float, optional
        Section size in rows (RSSIZ).
    col_section_size : float, optional
        Section size in columns (CSSIZ).
    """

    image_id: Optional[str] = None
    edition: Optional[str] = None
    row_poly: np.ndarray = field(
        default_factory=lambda: np.full(10, np.nan, dtype=np.float64))
    col_poly: np.ndarray = field(
        default_factory=lambda: np.full(10, np.nan, dtype=np.float64))
    num_row_sections: Optional[int] = None
    num_col_sections: Optional[int] = None
    num_total_sections: Optional[int] = None
    row_section_size: Optional[float] = None
    col_section_size: Optional[float] = None


@dataclass
class RSMDCAMetadata:
    """RSMDCA / RSMDCB -- RSM Direct Error Covariance.

    The A-variant (STDI-0002 App U Table 9) carries the full
    symmetric error covariance of the direct (image-specific)
    adjustable parameters of one or more images, stored as the
    upper triangle in 21-byte real fields.

    Variant depth
    -------------
    ``variant='A'`` (RSMDCA) is parsed at full depth: identification,
    per-image parameter counts, local coordinate frame, the 36
    adjustable-parameter indices, and the assembled symmetric
    ``covariance`` matrix.  When the ground-offset parameters
    (GXO/GYO/GZO) are active for this image, ``sigma_x`` /
    ``sigma_y`` / ``sigma_z`` are derived from the corresponding
    covariance diagonal entries (units of the local frame, meters).

    ``variant='B'`` (RSMDCB) is parsed at identification depth only:
    ``image_id``, ``edition``, ``trigger_id``, ``npar`` (from NROWCB,
    interpreted as this image's adjustable-parameter count -- the row
    dimension of each cross-covariance block), ``nimge``, and the
    per-image ``image_ids`` / ``image_npars`` (from IIDI / NCOLCB).
    The B-variant stores per-image *cross*-covariance blocks rather
    than one symmetric matrix, and its adjustable-parameter basis is
    APTYP-dependent, so ``covariance`` and the index arrays are left
    unpopulated; the full payload is retained in ``raw``.

    Parameters
    ----------
    image_id : str, optional
        Image identifier (IID).
    edition : str, optional
        RSM edition identifier (EDITION).
    trigger_id : str, optional
        Triangulation (trigger) identifier (TID).
    npar : int, optional
        Number of adjustable parameters for this image (NPAR;
        NROWCB for variant 'B').
    nimge : int, optional
        Number of images covered by the covariance (NIMGE).
    npar_total : int, optional
        Total number of adjustable parameters across all images
        (NPART).  Dimension of ``covariance``.
    image_ids : list of str
        Identifier of each covered image, in covariance order.
    image_npars : np.ndarray
        Per-image adjustable-parameter counts, shape ``(nimge,)``
        int64.
    local_origin : np.ndarray, optional
        Local rectangular coordinate origin (XUOL, YUOL, ZUOL),
        shape ``(3,)`` float64, meters ECEF.
    local_unit_vectors : np.ndarray, optional
        Local frame unit vectors, shape ``(3, 3)`` float64; rows are
        the local X, Y, Z axes expressed in ECEF.
    row_param_indices : np.ndarray
        Indices of the 10 row-polynomial adjustable parameters,
        shape ``(10,)`` int64 (0 = not adjusted).
    col_param_indices : np.ndarray
        Indices of the 10 column-polynomial adjustable parameters,
        shape ``(10,)`` int64.
    ground_param_indices : np.ndarray
        Indices of the 16 ground adjustable parameters in the order
        ``[GXO, GYO, GZO, GXR, GYR, GZR, GS, GXX, GXY, GXZ, GYX,
        GYY, GYZ, GZX, GZY, GZZ]``, shape ``(16,)`` int64.
    covariance : np.ndarray, optional
        Full symmetric error covariance, shape
        ``(npar_total, npar_total)`` float64, assembled from the
        upper-triangle DERCOV fields (row-major over ``i <= j``).
    sigma_x, sigma_y, sigma_z : float, optional
        One-sigma ground uncertainties (meters, local frame) taken
        from the covariance diagonal at the GXO/GYO/GZO parameter
        positions of this image, when those parameters are active.
    variant : str
        ``'A'`` for RSMDCA, ``'B'`` for RSMDCB.
    raw : str, optional
        Full original payload (CEDATA string or serialized XML).
    """

    image_id: Optional[str] = None
    edition: Optional[str] = None
    trigger_id: Optional[str] = None
    npar: Optional[int] = None
    nimge: Optional[int] = None
    npar_total: Optional[int] = None
    image_ids: List[str] = field(default_factory=list)
    image_npars: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64))
    local_origin: Optional[np.ndarray] = None
    local_unit_vectors: Optional[np.ndarray] = None
    row_param_indices: np.ndarray = field(
        default_factory=lambda: np.zeros(10, dtype=np.int64))
    col_param_indices: np.ndarray = field(
        default_factory=lambda: np.zeros(10, dtype=np.int64))
    ground_param_indices: np.ndarray = field(
        default_factory=lambda: np.zeros(16, dtype=np.int64))
    covariance: Optional[np.ndarray] = None
    sigma_x: Optional[float] = None
    sigma_y: Optional[float] = None
    sigma_z: Optional[float] = None
    variant: str = 'A'
    raw: Optional[str] = None


@dataclass
class RSMECAMetadata:
    """RSMECA / RSMECB -- RSM Indirect Error Covariance.

    The A-variant (STDI-0002 App U Table 8) carries the indirect
    error covariance of the original adjustable parameters,
    partitioned into independent subgroups, plus a mapping matrix to
    the image's active adjustable parameters and an optional
    unmodeled-error model in image space.

    RSMECA is highly variable; parsing is defensive.  On structural
    surprise the parser returns a dataclass with the identification
    header populated and the full payload retained in ``raw`` --
    it never raises.

    Variant depth
    -------------
    ``variant='A'`` (RSMECA): full identification, INCLIC/INCLUC
    flags, NPAR/NPARO/IGN/CVDATE, local frame, the 36 parameter
    indices, per-group covariance blocks (``group_covariances``),
    the block-diagonal original-space ``covariance`` (when the group
    sizes sum to NPARO), the ``mapping_matrix`` (MAP, NPAR x NPARO
    row-major), and the unmodeled image-space variances URR/URC/UCC.
    Correlation-segment data (TCDF/NCSEG/CORSEG/TAUSEG and the
    unmodeled correlation segments) are consumed positionally but
    not modeled; they remain available in ``raw``.
    ``sigma_x``/``sigma_y``/``sigma_z`` are derived from the
    adjusted-space covariance ``M C M^T`` at the GXO/GYO/GZO index
    positions when both the covariance and mapping parse cleanly.

    ``variant='B'`` (RSMECB): identification depth only --
    ``image_id``, ``edition``, ``trigger_id``, INCLIC/INCLUC, and
    (when INCLIC='Y') NPARO, IGN, CVDATE, NPAR.  When INCLIC='N'
    and INCLUC='Y' the unmodeled URR/URC/UCC are also read (they sit
    at a fixed offset in that case).  The APTYP-dependent parameter
    basis, group covariances, and MAP are not parsed; the full
    payload is retained in ``raw``.

    Parameters
    ----------
    image_id : str, optional
        Image identifier (IID).
    edition : str, optional
        RSM edition identifier (EDITION).
    trigger_id : str, optional
        Triangulation (trigger) identifier (TID).
    inclic : bool, optional
        True when indirect error covariance data is included
        (INCLIC = 'Y').
    incluc : bool, optional
        True when unmodeled error data is included (INCLUC = 'Y').
    npar : int, optional
        Number of active adjustable parameters (NPAR).
    nparo : int, optional
        Number of original adjustable parameters (NPARO).
    num_groups : int, optional
        Number of independent parameter subgroups (IGN).
    cov_date : str, optional
        Version date of the original covariance (CVDATE, YYYYMMDD).
    local_origin : np.ndarray, optional
        Local frame origin, shape ``(3,)`` float64, meters ECEF.
    local_unit_vectors : np.ndarray, optional
        Local frame unit vectors, shape ``(3, 3)`` float64.
    row_param_indices : np.ndarray
        Row-polynomial parameter indices, shape ``(10,)`` int64.
    col_param_indices : np.ndarray
        Column-polynomial parameter indices, shape ``(10,)`` int64.
    ground_param_indices : np.ndarray
        Ground parameter indices, shape ``(16,)`` int64 (same order
        as :class:`RSMDCAMetadata`).
    group_covariances : list of np.ndarray
        Per-subgroup symmetric covariance blocks, each shape
        ``(NUMOPG_g, NUMOPG_g)`` float64.
    covariance : np.ndarray, optional
        Block-diagonal original-space covariance, shape
        ``(nparo, nparo)`` float64; populated only when the group
        sizes sum exactly to ``nparo`` (the "simple" structure).
    mapping_matrix : np.ndarray, optional
        MAP matrix relating original to active parameters, shape
        ``(npar, nparo)`` float64, row-major.
    unmodeled_rr, unmodeled_rc, unmodeled_cc : float, optional
        Unmodeled image-space error variances URR / URC / UCC
        (pixels squared).
    sigma_x, sigma_y, sigma_z : float, optional
        One-sigma ground uncertainties (meters, local frame) from
        the adjusted covariance diagonal at the GXO/GYO/GZO
        positions, when derivable.
    variant : str
        ``'A'`` for RSMECA, ``'B'`` for RSMECB.
    raw : str, optional
        Full original payload (CEDATA string or serialized XML).
    """

    image_id: Optional[str] = None
    edition: Optional[str] = None
    trigger_id: Optional[str] = None
    inclic: Optional[bool] = None
    incluc: Optional[bool] = None
    npar: Optional[int] = None
    nparo: Optional[int] = None
    num_groups: Optional[int] = None
    cov_date: Optional[str] = None
    local_origin: Optional[np.ndarray] = None
    local_unit_vectors: Optional[np.ndarray] = None
    row_param_indices: np.ndarray = field(
        default_factory=lambda: np.zeros(10, dtype=np.int64))
    col_param_indices: np.ndarray = field(
        default_factory=lambda: np.zeros(10, dtype=np.int64))
    ground_param_indices: np.ndarray = field(
        default_factory=lambda: np.zeros(16, dtype=np.int64))
    group_covariances: List[np.ndarray] = field(default_factory=list)
    covariance: Optional[np.ndarray] = None
    mapping_matrix: Optional[np.ndarray] = None
    unmodeled_rr: Optional[float] = None
    unmodeled_rc: Optional[float] = None
    unmodeled_cc: Optional[float] = None
    sigma_x: Optional[float] = None
    sigma_y: Optional[float] = None
    sigma_z: Optional[float] = None
    variant: str = 'A'
    raw: Optional[str] = None


@dataclass
class RSMAPAMetadata:
    """RSMAPA / RSMAPB -- RSM Adjustable Parameters.

    The A-variant (STDI-0002 App U Table 10) lists the values of the
    active adjustable parameters drawn from the canonical 36-element
    set (see :data:`RSM_ADJUSTABLE_PARAM_ORDER`).

    Variant depth
    -------------
    ``variant='A'`` (RSMAPA) is parsed at full depth: identification,
    NPAR, local frame, the 36 parameter index fields
    (``param_indices``), and the NPAR parameter values
    (``param_values``, in index order).

    ``variant='B'`` (RSMAPB): identification, NPAR, the local frame
    when LOCTYP='R', and ``param_values`` taken from the trailing
    ``NPAR x 21`` bytes of the payload (PARVAL is always the final
    block in RSMAPB regardless of APTYP/APBASE structure).
    ``param_indices`` is not populated -- the B-variant parameter
    set is APTYP-dependent (image-space powers or ground-space
    GSAPID list) and does not map onto the canonical 36-element
    basis.  The full payload is retained in ``raw``.

    Parameters
    ----------
    image_id : str, optional
        Image identifier (IID).
    edition : str, optional
        RSM edition identifier (EDITION).
    trigger_id : str, optional
        Triangulation (trigger) identifier (TID).
    npar : int, optional
        Number of adjustable parameters (NPAR).
    local_origin : np.ndarray, optional
        Local frame origin, shape ``(3,)`` float64, meters ECEF.
    local_unit_vectors : np.ndarray, optional
        Local frame unit vectors, shape ``(3, 3)`` float64.
    param_indices : np.ndarray
        Index (1-based position in ``param_values``; 0 = not
        adjusted) for each of the 36 canonical adjustable
        parameters, shape ``(36,)`` int64, ordered per
        :data:`RSM_ADJUSTABLE_PARAM_ORDER`.
    param_values : np.ndarray
        Adjustable parameter values, shape ``(npar,)`` float64.
    variant : str
        ``'A'`` for RSMAPA, ``'B'`` for RSMAPB.
    raw : str, optional
        Full original payload (CEDATA string or serialized XML).
    """

    image_id: Optional[str] = None
    edition: Optional[str] = None
    trigger_id: Optional[str] = None
    npar: Optional[int] = None
    local_origin: Optional[np.ndarray] = None
    local_unit_vectors: Optional[np.ndarray] = None
    param_indices: np.ndarray = field(
        default_factory=lambda: np.zeros(36, dtype=np.int64))
    param_values: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64))
    variant: str = 'A'
    raw: Optional[str] = None
