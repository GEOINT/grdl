# -*- coding: utf-8 -*-
"""
EO NITF Chip Writer - Geolocation-preserving chip-out to NITF.

Writes a spatial chip of an EO NITF product to a new NITF file that
geolocates identically to the parent image.  The chip-to-full-image
mapping is carried in an ICHIPB TRE (serialized per STDI-0002 Vol 1
Appendix B, 224-byte CEDATA), composed with the parent's own ICHIPB
when the parent is itself a chipped product.

Geolocation models are propagated in the **full-image** coordinate
space — this is how chipped NITFs work in practice: downstream
consumers (including :class:`grdl.geolocation.eo.rpc.RPCGeolocation`
and :class:`grdl.geolocation.eo.rsm.RSMGeolocation`) evaluate the
sensor model in full-image pixels and apply the ICHIPB transform to
convert chip pixels to/from full-image pixels.

* RPC: the parent's RPC00B coefficients are written unchanged via
  GDAL's RPC metadata domain (``rasterio.open(..., rpcs=...)`` on the
  intermediate dataset).  The GDAL NITF driver writes the RPC00B TRE
  from RPC metadata (creation option ``RPC00B`` defaults to ``YES``).
* RSM: RSMIDA and one RSMPCA TRE are serialized back to their CEDATA
  layouts (the exact inverses of the parsers in
  :mod:`grdl.IO.eo.nitf`).  GDAL/rasterio cannot repeat a TRE name in
  a single output file (creation options and metadata-domain keys are
  unique), so for multi-section RSM grids the single RSMPCA section
  covering the chip center is written with a warning.  RSMGGA and the
  RSM error TREs (RSMECA/RSMECB) are not propagated.

Implementation note: rasterio exposes GDAL creation options as a
dict, so multiple ``TRE=...`` options cannot be passed directly.  The
writer therefore stages the chip in an in-memory GTiff carrying the
TREs in GDAL's ``TRE`` metadata domain plus the RPC metadata domain,
then converts to NITF via ``rasterio.shutil.copy`` (GDAL
``CreateCopy``), which transcribes both domains into the output NITF
(verified against GDAL 3.10 in this environment).

Dependencies
------------
rasterio

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
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Third-party
import numpy as np

try:
    import rasterio
    import rasterio.shutil
    from rasterio.io import MemoryFile
    from rasterio.rpc import RPC
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

# GRDL internal
from grdl.IO.base import ImageReader
from grdl.IO.models.eo_nitf import (
    ICHIPBMetadata,
    RPCCoefficients,
    RSMCoefficients,
    RSMIdentification,
)

__all__ = ['write_chip']


# ===================================================================
# Fixed-width field formatting helpers
# ===================================================================

def _real21(value: Optional[float]) -> str:
    """Format one 21-char RSM real (``+d.ddddddddddddddE+dd``).

    ``None`` produces a 21-space blank field (the spec's "no value").

    Parameters
    ----------
    value : float, optional
        Value to format.

    Returns
    -------
    str
        Exactly 21 characters.

    Raises
    ------
    ValueError
        If the value needs a 3-digit exponent and cannot fit.
    """
    if value is None:
        return ' ' * 21
    s = f'{float(value):+.14E}'
    if len(s) != 21:
        raise ValueError(
            f"Value {value!r} does not fit the 21-character RSM "
            "real field (exponent out of range)."
        )
    return s


def _text(value: Optional[str], width: int) -> str:
    """Left-justify ``value`` in a fixed-width BCS-A field."""
    return f'{(value or ""):<{width}.{width}}'


def _int_field(value: int, width: int) -> str:
    """Right-justify an integer in a fixed-width numeric field."""
    s = f'{int(value):{width}d}'
    if len(s) != width:
        raise ValueError(
            f"Value {value!r} does not fit a {width}-character field."
        )
    return s


# ===================================================================
# ICHIPB serialization (STDI-0002 Vol 1 App B — 224-byte CEDATA)
# ===================================================================

def _serialize_ichipb(
    chip_rows: int,
    chip_cols: int,
    fi_row_off: float,
    fi_col_off: float,
    fi_row_scale: float,
    fi_col_scale: float,
    full_image_rows: int,
    full_image_cols: int,
    scale_factor: float = 1.0,
) -> str:
    """Serialize the chip→full-image mapping to ICHIPB CEDATA.

    Field order (224 bytes total)::

        XFRM_FLAG(2) SCALE_FACTOR(10) ANAMRPH_CORR(2) SCANBLK_NUM(2)
        OP_ROW_11(12) OP_COL_11(12) OP_ROW_12(12) OP_COL_12(12)
        OP_ROW_21(12) OP_COL_21(12) OP_ROW_22(12) OP_COL_22(12)
        FI_ROW_11(12) FI_COL_11(12) FI_ROW_12(12) FI_COL_12(12)
        FI_ROW_21(12) FI_COL_21(12) FI_ROW_22(12) FI_COL_22(12)
        FI_ROW(8) FI_COL(8)

    The OP corners use the 0.5 / (size - 0.5) pixel-center convention
    and the FI corners encode ``full = fi_off + fi_scale * chip``.

    Parameters
    ----------
    chip_rows, chip_cols : int
        Chip dimensions in pixels.
    fi_row_off, fi_col_off : float
        Offset terms of the chip→full affine.
    fi_row_scale, fi_col_scale : float
        Scale terms of the chip→full affine.
    full_image_rows, full_image_cols : int
        Original full-image dimensions; ``0`` encodes the spec's
        "unknown" sentinel.
    scale_factor : float
        ICHIPB SCALE_FACTOR (RRDS code), inherited from the parent.

    Returns
    -------
    str
        Exactly 224 characters of CEDATA.
    """
    op_r1, op_r2 = 0.5, chip_rows - 0.5
    op_c1, op_c2 = 0.5, chip_cols - 0.5
    # Corner order: 11 (upper-left), 12 (upper-right),
    #               21 (lower-left), 22 (lower-right).
    op_corners = [
        (op_r1, op_c1), (op_r1, op_c2), (op_r2, op_c1), (op_r2, op_c2),
    ]
    fi_corners = [
        (fi_row_off + fi_row_scale * r, fi_col_off + fi_col_scale * c)
        for r, c in op_corners
    ]

    parts = [
        '00',                            # XFRM_FLAG (transform valid)
        f'{scale_factor:010.5f}',        # SCALE_FACTOR
        '00',                            # ANAMRPH_CORR
        '00',                            # SCANBLK_NUM
    ]
    for r, c in op_corners:
        parts.append(f'{r:012.3f}')
        parts.append(f'{c:012.3f}')
    for r, c in fi_corners:
        parts.append(f'{r:012.3f}')
        parts.append(f'{c:012.3f}')
    parts.append(_int_field(full_image_rows, 8).replace(' ', '0'))
    parts.append(_int_field(full_image_cols, 8).replace(' ', '0'))

    cedata = ''.join(parts)
    if len(cedata) != 224:
        raise ValueError(
            f"ICHIPB CEDATA serialized to {len(cedata)} bytes "
            "(expected 224)."
        )
    return cedata


# ===================================================================
# RSM serialization (STDI-0002 Vol 1 App U)
# ===================================================================

def _serialize_rsmpca(
    rsm: RSMCoefficients,
    image_id: str = 'GRDL',
    edition: str = '',
) -> str:
    """Serialize RSMCoefficients to RSMPCA CEDATA.

    Exact inverse of :func:`grdl.IO.eo.nitf._parse_rsmpca_tre`::

        IID(80) EDITION(40) RSN(3) CSN(3) RFEP(21) CFEP(21)
        RNRMO(21) CNRMO(21) XNRMO(21) YNRMO(21) ZNRMO(21)
        RNRMSF(21) CNRMSF(21) XNRMSF(21) YNRMSF(21) ZNRMSF(21)
        RNPWRX(1) RNPWRY(1) RNPWRZ(1) RNTRMS(3) RNPCF(21)×RNTRMS
        RDPWRX(1) RDPWRY(1) RDPWRZ(1) RDTRMS(3) RDPCF(21)×RDTRMS
        CNPWRX(1) CNPWRY(1) CNPWRZ(1) CNTRMS(3) CNPCF(21)×CNTRMS
        CDPWRX(1) CDPWRY(1) CDPWRZ(1) CDTRMS(3) CDPCF(21)×CDTRMS

    Parameters
    ----------
    rsm : RSMCoefficients
        Polynomial section to serialize.
    image_id : str
        IID field content.  Must be non-blank so the byte-offset
        parser's leading ``strip()`` cannot shift field alignment.
    edition : str
        EDITION field content.

    Returns
    -------
    str
        RSMPCA CEDATA string.
    """
    if not image_id.strip():
        image_id = 'GRDL'

    parts = [
        _text(image_id, 80),
        _text(edition, 40),
        _int_field(rsm.rsn or 1, 3).replace(' ', '0'),
        _int_field(rsm.csn or 1, 3).replace(' ', '0'),
        _real21(rsm.row_fit_error),
        _real21(rsm.col_fit_error),
        _real21(rsm.row_off),
        _real21(rsm.col_off),
        _real21(rsm.x_off),
        _real21(rsm.y_off),
        _real21(rsm.z_off),
        _real21(rsm.row_norm_sf),
        _real21(rsm.col_norm_sf),
        _real21(rsm.x_norm_sf),
        _real21(rsm.y_norm_sf),
        _real21(rsm.z_norm_sf),
    ]

    groups = [
        (rsm.row_num_powers, rsm.row_num_coefs),
        (rsm.row_den_powers, rsm.row_den_coefs),
        (rsm.col_num_powers, rsm.col_num_coefs),
        (rsm.col_den_powers, rsm.col_den_coefs),
    ]
    for powers, coefs in groups:
        powers = np.asarray(powers, dtype=int)
        coefs = np.asarray(coefs, dtype=np.float64)
        parts.append(f'{powers[0]:1d}{powers[1]:1d}{powers[2]:1d}')
        parts.append(_int_field(len(coefs), 3).replace(' ', '0'))
        for c in coefs:
            parts.append(_real21(float(c)))

    return ''.join(parts)


def _serialize_rsmida(rsm_id: RSMIdentification) -> str:
    """Serialize RSMIdentification to RSMIDA CEDATA.

    Exact inverse of :func:`grdl.IO.eo.nitf._parse_rsmida_tre`.  The
    illumination / sensor-trajectory tail (21 × 21-byte fields after
    the image extents) is blank-filled; total CEDATA length is 1376
    bytes for geodetic ground domains and 1628 bytes for rectangular
    (``GRNDD='R'``) domains, per STDI-0002 Vol 1 App U Table 2.

    Parameters
    ----------
    rsm_id : RSMIdentification
        Identification record to serialize.

    Returns
    -------
    str
        RSMIDA CEDATA string.
    """
    image_id = (rsm_id.image_id or '').strip() or 'GRDL'
    grndd = (rsm_id.ground_domain_type or 'G')[:1]

    parts = [
        _text(image_id, 80),
        _text(rsm_id.edition, 40),
        _text(rsm_id.image_sensor_id, 40),
        _text(rsm_id.sensor_id, 40),
        _text(rsm_id.sensor_type_id, 40),
    ]

    # --- Date/time: YEAR(4) MONTH(2) DAY(2) HOUR(2) MINUTE(2) SECOND(9)
    dt = rsm_id.collection_datetime
    if dt is not None:
        seconds = dt.second + dt.microsecond / 1e6
        parts.append(
            f'{dt.year:04d}{dt.month:02d}{dt.day:02d}'
            f'{dt.hour:02d}{dt.minute:02d}{seconds:09.6f}'
        )
    else:
        parts.append(' ' * 21)

    # --- Time-of-image model: NRG(8) NCG(8) TRG(21) TCG(21)
    parts.append(_int_field(rsm_id.num_row_sections or 1, 8))
    parts.append(_int_field(rsm_id.num_col_sections or 1, 8))
    parts.append(_real21(rsm_id.time_ref_row))
    parts.append(_real21(rsm_id.time_ref_col))

    # --- Ground coordinate system
    parts.append(grndd)

    if grndd == 'R':
        origin = rsm_id.coord_origin
        if origin is not None:
            parts.append(_real21(origin.x))
            parts.append(_real21(origin.y))
            parts.append(_real21(origin.z))
        else:
            parts.append(' ' * 63)
        uv = rsm_id.coord_unit_vectors
        uv = (np.zeros((3, 3)) if uv is None
              else np.asarray(uv, dtype=np.float64))
        for v in uv.ravel():
            parts.append(_real21(float(v)))

    # --- Ground domain vertices (8 × 3 × 21 bytes)
    verts = rsm_id.ground_domain_vertices
    verts = (np.zeros((8, 3)) if verts is None
             else np.asarray(verts, dtype=np.float64))
    for v in verts.ravel():
        parts.append(_real21(float(v)))

    # --- Ground reference point GRPX/GRPY/GRPZ
    grp = rsm_id.ground_ref_point
    parts.append(_real21(grp.x if grp is not None else 0.0))
    parts.append(_real21(grp.y if grp is not None else 0.0))
    parts.append(_real21(grp.z if grp is not None else 0.0))

    # --- Image extents: FULLR FULLC MINR MAXR MINC MAXC (6 × 8 bytes)
    fullr = rsm_id.full_image_rows
    fullc = rsm_id.full_image_cols
    if fullr and fullc:
        minr = rsm_id.min_row if rsm_id.min_row is not None else 0
        maxr = (rsm_id.max_row if rsm_id.max_row is not None
                else fullr - 1)
        minc = rsm_id.min_col if rsm_id.min_col is not None else 0
        maxc = (rsm_id.max_col if rsm_id.max_col is not None
                else fullc - 1)
        for v in (fullr, fullc, minr, maxr, minc, maxc):
            parts.append(_int_field(v, 8))
    else:
        parts.append(' ' * 48)

    # --- Illumination / trajectory tail (21 fields × 21 bytes), blank
    parts.append(' ' * 441)

    cedata = ''.join(parts)
    expected = 1628 if grndd == 'R' else 1376
    if len(cedata) != expected:
        raise ValueError(
            f"RSMIDA CEDATA serialized to {len(cedata)} bytes "
            f"(expected {expected} for GRNDD='{grndd}')."
        )
    return cedata


# ===================================================================
# RPC conversion
# ===================================================================

def _rpc_to_rasterio(rpc: RPCCoefficients) -> 'RPC':
    """Convert GRDL RPCCoefficients to a ``rasterio.rpc.RPC``.

    Parameters
    ----------
    rpc : RPCCoefficients
        Parent full-image RPC00B coefficients.

    Returns
    -------
    rasterio.rpc.RPC
    """
    return RPC(
        height_off=float(rpc.height_off),
        height_scale=float(rpc.height_scale),
        lat_off=float(rpc.lat_off),
        lat_scale=float(rpc.lat_scale),
        line_den_coeff=[float(v) for v in rpc.line_den_coef],
        line_num_coeff=[float(v) for v in rpc.line_num_coef],
        line_off=float(rpc.line_off),
        line_scale=float(rpc.line_scale),
        long_off=float(rpc.long_off),
        long_scale=float(rpc.long_scale),
        samp_den_coeff=[float(v) for v in rpc.samp_den_coef],
        samp_num_coeff=[float(v) for v in rpc.samp_num_coef],
        samp_off=float(rpc.samp_off),
        samp_scale=float(rpc.samp_scale),
        err_bias=float(rpc.err_bias) if rpc.err_bias is not None else 0.0,
        err_rand=float(rpc.err_rand) if rpc.err_rand is not None else 0.0,
    )


# ===================================================================
# Chip affine composition
# ===================================================================

def _compose_chip_affine(
    parent_ichipb: Optional[ICHIPBMetadata],
    row_start: int,
    col_start: int,
    full_rows_fallback: int,
    full_cols_fallback: int,
) -> Tuple[float, float, float, float, int, int, float]:
    """Compose the chip→full-image affine with the parent's ICHIPB.

    For pinned / single-segment chipped parents, ``metadata.ichipb``
    carries ``full = parent_off + parent_scale * parent_local`` and
    the new chip's local pixel ``c`` maps to parent-local
    ``start + c``, so the composed affine is
    ``full = (parent_off + parent_scale * start) + parent_scale * c``.

    For unified readers ``metadata.ichipb`` is ``None`` and
    ``(row_start, col_start)`` are already full-image coordinates, so
    the affine is a pure translation by the chip origin.

    Returns
    -------
    Tuple[float, float, float, float, int, int, float]
        ``(fi_row_off, fi_col_off, fi_row_scale, fi_col_scale,
        full_image_rows, full_image_cols, scale_factor)``.

    Raises
    ------
    ValueError
        If the parent ICHIPB carries no transform data
        (``XFRM_FLAG=01``) and cannot be composed.
    """
    if parent_ichipb is None:
        return (
            float(row_start), float(col_start), 1.0, 1.0,
            int(full_rows_fallback), int(full_cols_fallback), 1.0,
        )

    if (parent_ichipb.fi_row_off is None
            or parent_ichipb.fi_col_off is None
            or parent_ichipb.fi_row_scale is None
            or parent_ichipb.fi_col_scale is None):
        raise ValueError(
            "Parent ICHIPB carries no usable chip-to-full transform "
            "(XFRM_FLAG=01 or missing corner data); cannot compose "
            "the chip geolocation mapping."
        )

    p_ro = float(parent_ichipb.fi_row_off)
    p_co = float(parent_ichipb.fi_col_off)
    p_rs = float(parent_ichipb.fi_row_scale)
    p_cs = float(parent_ichipb.fi_col_scale)

    full_rows = int(parent_ichipb.full_image_rows or 0)
    full_cols = int(parent_ichipb.full_image_cols or 0)
    scale_factor = float(parent_ichipb.scale_factor or 1.0)

    return (
        p_ro + p_rs * row_start,
        p_co + p_cs * col_start,
        p_rs,
        p_cs,
        full_rows,
        full_cols,
        scale_factor,
    )


# ===================================================================
# write_chip — public entry point
# ===================================================================

def write_chip(
    reader: ImageReader,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    output_path: Union[str, Path],
    bands: Optional[List[int]] = None,
) -> Path:
    """Write a chip of an EO NITF to a new, geolocation-preserving NITF.

    Reads pixels via ``reader.read_chip`` and writes a NITF (GDAL
    NITF driver via rasterio) carrying the geolocation forward so the
    chip geolocates identically to the parent:

    * **ICHIPB** encodes the chip→full-image mapping, composed with
      the parent's own ``metadata.ichipb`` when present (pinned /
      single-segment chipped parents).  For unified multi-segment
      readers ``metadata.ichipb`` is ``None`` and the chip origin is
      already in full-image coordinates.
    * **RPC** (when ``reader.metadata.rpc`` is not ``None``) is
      written for the *full-image* coordinate space — ICHIPB handles
      the chip offset downstream, matching how chipped NITF products
      work in practice.
    * **RSM** (when ``reader.metadata.rsm`` is not ``None``) writes
      RSMIDA and one RSMPCA.  Multi-section grids collapse to the
      section covering the chip center (GDAL/rasterio cannot repeat
      a TRE name); RSMGGA and RSM error TREs are not propagated.

    Parameters
    ----------
    reader : ImageReader
        Open :class:`grdl.IO.eo.nitf.EONITFReader` (or compatible
        reader exposing ``metadata`` and ``read_chip``).
    row_start : int
        Starting row index (inclusive), in the reader's pixel space.
    row_end : int
        Ending row index (exclusive).
    col_start : int
        Starting column index (inclusive).
    col_end : int
        Ending column index (exclusive).
    output_path : str or Path
        Destination NITF path.
    bands : List[int], optional
        Band indices to copy (0-based).  ``None`` copies all bands.

    Returns
    -------
    Path
        Path to the written NITF file.

    Raises
    ------
    ImportError
        If rasterio is not installed.
    ValueError
        If the chip bounds are empty, negative, or exceed the image.

    Examples
    --------
    >>> from grdl.IO.eo.nitf import EONITFReader
    >>> from grdl.IO.eo.nitf_writer import write_chip
    >>> with EONITFReader('parent.ntf') as reader:
    ...     write_chip(
    ...         reader=reader,
    ...         row_start=1024, row_end=2048,
    ...         col_start=512, col_end=1536,
    ...         output_path='chip.ntf',
    ...     )
    """
    if not _HAS_RASTERIO:
        raise ImportError(
            "rasterio is required for EO NITF chip writing. "
            "Install with: pip install rasterio  "
            "or: conda install -c conda-forge rasterio"
        )

    meta = reader.metadata

    # --- Validate chip bounds (fail fast) ---
    row_start = int(row_start)
    row_end = int(row_end)
    col_start = int(col_start)
    col_end = int(col_end)
    if row_start < 0 or col_start < 0:
        raise ValueError(
            f"Chip start indices must be non-negative "
            f"(got row_start={row_start}, col_start={col_start})."
        )
    if row_end <= row_start or col_end <= col_start:
        raise ValueError(
            f"Chip bounds are empty: rows [{row_start}, {row_end}), "
            f"cols [{col_start}, {col_end})."
        )
    if row_end > meta.rows or col_end > meta.cols:
        raise ValueError(
            f"Chip bounds exceed image dimensions "
            f"({meta.rows} x {meta.cols}): rows [{row_start}, "
            f"{row_end}), cols [{col_start}, {col_end})."
        )

    # --- Read pixels ---
    data = reader.read_chip(row_start, row_end, col_start, col_end,
                            bands=bands)
    arr = data if data.ndim == 3 else data[np.newaxis, ...]
    n_bands, chip_rows, chip_cols = arr.shape

    # --- Compose chip→full affine and serialize ICHIPB ---
    (fi_row_off, fi_col_off, fi_row_scale, fi_col_scale,
     full_rows, full_cols, scale_factor) = _compose_chip_affine(
        parent_ichipb=getattr(meta, 'ichipb', None),
        row_start=row_start,
        col_start=col_start,
        full_rows_fallback=meta.rows,
        full_cols_fallback=meta.cols,
    )

    tres = {
        'ICHIPB': _serialize_ichipb(
            chip_rows=chip_rows,
            chip_cols=chip_cols,
            fi_row_off=fi_row_off,
            fi_col_off=fi_col_off,
            fi_row_scale=fi_row_scale,
            fi_col_scale=fi_col_scale,
            full_image_rows=full_rows,
            full_image_cols=full_cols,
            scale_factor=scale_factor,
        ),
    }

    # --- RSM TREs (full-image coordinate space, pass-through) ---
    rsm = getattr(meta, 'rsm', None)
    if rsm is not None:
        rsm_id = getattr(meta, 'rsm_id', None)
        if rsm_id is not None:
            tres['RSMIDA'] = _serialize_rsmida(rsm_id)

        grid = getattr(meta, 'rsm_segments', None)
        if grid is not None and grid.segments:
            segments = list(grid.segments.values())
        else:
            segments = [rsm]
        if len(segments) > 1:
            center_row = fi_row_off + fi_row_scale * (chip_rows / 2.0)
            center_col = fi_col_off + fi_col_scale * (chip_cols / 2.0)
            selected = grid.segment_for_pixel(center_row, center_col)
            selected = selected if selected is not None else segments[0]
            warnings.warn(
                f"Parent RSM has {len(segments)} polynomial sections; "
                "GDAL/rasterio cannot write repeated RSMPCA TREs, so "
                "only the section covering the chip center "
                f"(rsn={selected.rsn}, csn={selected.csn}) is "
                "propagated.",
                RuntimeWarning, stacklevel=2,
            )
        else:
            selected = segments[0]

        iid = (rsm_id.image_id if rsm_id is not None and rsm_id.image_id
               else 'GRDL')
        edition = (rsm_id.edition
                   if rsm_id is not None and rsm_id.edition else '')
        tres['RSMPCA'] = _serialize_rsmpca(
            selected, image_id=iid, edition=edition)

    # --- RPC (full-image coordinate space, pass-through) ---
    rpc = getattr(meta, 'rpc', None)
    rpcs = _rpc_to_rasterio(rpc) if rpc is not None else None

    # --- Stage in-memory GTiff, convert to NITF via CreateCopy ---
    output_path = Path(output_path)
    profile = {
        'driver': 'GTiff',
        'height': chip_rows,
        'width': chip_cols,
        'count': n_bands,
        'dtype': arr.dtype.name,
    }
    if rpcs is not None:
        profile['rpcs'] = rpcs

    with MemoryFile() as memfile:
        with memfile.open(**profile) as mem:
            mem.write(arr)
            mem.update_tags(ns='TRE', **tres)
        rasterio.shutil.copy(
            memfile.name, str(output_path), driver='NITF')

    return output_path
