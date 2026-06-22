# -*- coding: utf-8 -*-
"""
EO NITF Reader - Read electro-optical NITF imagery with RPC/RSM metadata.

Extends the base NITF reading capability with extraction of EO-specific
geolocation models: RPC (Rational Polynomial Coefficients) from the
RPC00B TRE, and RSM (Replacement Sensor Model) from RSMPCA/RSMIDA TREs.

Uses rasterio (GDAL NITF driver) for pixel access and metadata
extraction.  No sarpy or sarkit dependency.

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
2026-03-17

Modified
--------
2026-06-09  Heterogeneous multi-image NITF support: segment grouping,
            primary-group auto-selection, ILOC/IALVL placement; never
            fail loading on mixed-segment files.
2026-04-17  Parse CSEPHA and RSMGGA TREs (ephemeris + ground grid).
2026-04-01
"""

# Standard library
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party
import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

# GRDL internal
from grdl.IO.base import ImageReader
from grdl.IO.eo._tre_airborne import (
    parse_acftb_cedata,
    parse_mensra_cedata,
    parse_mensrb_cedata,
    parse_sensrb_cedata,
)
from grdl.IO.eo._tre_band import (
    parse_bandsa_cedata,
    parse_bandsb_cedata,
)
from grdl.IO.eo._tre_rsm_error import (
    parse_rsmapa_cedata,
    parse_rsmapb_cedata,
    parse_rsmdca_cedata,
    parse_rsmdcb_cedata,
    parse_rsmeca_cedata,
    parse_rsmecb_cedata,
    parse_rsmpia_cedata,
    summarize_accuracy,
)
from grdl.IO.models.eo_nitf import (
    AccuracyInfo,
    BLOCKAMetadata,
    CSCRNAMetadata,
    CSEPHAMetadata,
    CSEXRAMetadata,
    CollectionInfo,
    EONITFMetadata,
    ICHIPBMetadata,
    ImageGroupInfo,
    ImageSegmentInfo,
    RPCCoefficients,
    RSMCoefficients,
    RSMGGAGridPlane,
    RSMGGAMetadata,
    RSMIdentification,
    RSMSegmentGrid,
    USE00AMetadata,
)
from grdl.IO.models.common import XYZ
from grdl.IO.performance import (
    ReadConfig,
    _ensure_gdal_threads,
    _resolve_workers,
    apply_gdal_env,
    chunked_parallel_read,
    parallel_band_read,
)


def _normalize_remote_path(filepath: Union[str, Path]) -> Optional[str]:
    """Translate remote URIs to GDAL virtual-filesystem paths.

    Supports GDAL ``/vsi*`` paths verbatim, plus ``http(s)://`` (via
    ``/vsicurl/`` HTTP range requests) and ``s3://`` (via ``/vsis3/``).

    Parameters
    ----------
    filepath : str or Path
        Path or URI as given by the caller.

    Returns
    -------
    str or None
        GDAL-openable virtual path, or ``None`` when ``filepath`` is
        an ordinary local path.
    """
    s = str(filepath)
    if s.startswith('/vsi'):
        return s
    if s.startswith(('http://', 'https://')):
        return '/vsicurl/' + s
    if s.startswith('s3://'):
        return '/vsis3/' + s[len('s3://'):]
    return None


def _parse_rsmpca_tre(tre_value: str) -> Optional[RSMCoefficients]:
    """Parse an RSMPCA TRE CEDATA string into RSMCoefficients.

    Field layout per STDI-0002 Vol 1 Appendix U, Table 4 (RSMPCA)::

        IID(80) EDITION(40) RSN(3) CSN(3) RFEP(21) CFEP(21)
        RNRMO(21) CNRMO(21) XNRMO(21) YNRMO(21) ZNRMO(21)
        RNRMSF(21) CNRMSF(21) XNRMSF(21) YNRMSF(21) ZNRMSF(21)
        RNPWRX(1) RNPWRY(1) RNPWRZ(1) RNTRMS(3) RNPCF(21)×RNTRMS
        RDPWRX(1) RDPWRY(1) RDPWRZ(1) RDTRMS(3) RDPCF(21)×RDTRMS
        CNPWRX(1) CNPWRY(1) CNPWRZ(1) CNTRMS(3) CNPCF(21)×CNTRMS
        CDPWRX(1) CDPWRY(1) CDPWRZ(1) CDTRMS(3) CDPCF(21)×CDTRMS

    Parameters
    ----------
    tre_value : str
        Raw CEDATA string from GDAL TRE metadata.

    Returns
    -------
    RSMCoefficients or None
        Parsed coefficients, or None if parsing fails.
    """
    try:
        v = tre_value.strip()
        # Minimum RSMPCA size is 486 bytes (per spec CEL range)
        if len(v) < 486:
            return None

        pos = 0

        def read_str(n: int) -> str:
            nonlocal pos
            s = v[pos:pos + n].strip()
            pos += n
            return s

        def read_float(n: int = 21) -> float:
            nonlocal pos
            val = float(v[pos:pos + n].strip())
            pos += n
            return val

        def read_int(n: int = 1) -> int:
            nonlocal pos
            val = int(v[pos:pos + n].strip())
            pos += n
            return val

        # --- Image Information ---
        read_str(80)              # IID (not stored)
        read_str(40)              # EDITION (not stored)

        # --- Section identification ---
        rsn = read_int(3)         # RSN (row section number)
        csn = read_int(3)         # CSN (column section number)

        # --- Fitting errors ---
        rfep_str = read_str(21)   # RFEP
        cfep_str = read_str(21)   # CFEP
        rfep = float(rfep_str) if rfep_str else None
        cfep = float(cfep_str) if cfep_str else None

        # --- Normalization offsets (5 × 21 bytes) ---
        row_off = read_float()     # RNRMO
        col_off = read_float()     # CNRMO
        x_off = read_float()       # XNRMO
        y_off = read_float()       # YNRMO
        z_off = read_float()       # ZNRMO

        # --- Normalization scale factors (5 × 21 bytes) ---
        row_norm_sf = read_float()  # RNRMSF
        col_norm_sf = read_float()  # CNRMSF
        x_norm_sf = read_float()    # XNRMSF
        y_norm_sf = read_float()    # YNRMSF
        z_norm_sf = read_float()    # ZNRMSF

        # --- Row numerator polynomial ---
        rnpwrx = read_int()         # RNPWRX
        rnpwry = read_int()         # RNPWRY
        rnpwrz = read_int()         # RNPWRZ
        rntms = read_int(3)         # RNTRMS
        row_num_coefs = np.array(
            [read_float() for _ in range(rntms)])

        # --- Row denominator polynomial ---
        rdpwrx = read_int()
        rdpwry = read_int()
        rdpwrz = read_int()
        rdtms = read_int(3)
        row_den_coefs = np.array(
            [read_float() for _ in range(rdtms)])

        # --- Column numerator polynomial ---
        cnpwrx = read_int()
        cnpwry = read_int()
        cnpwrz = read_int()
        cntms = read_int(3)
        col_num_coefs = np.array(
            [read_float() for _ in range(cntms)])

        # --- Column denominator polynomial ---
        cdpwrx = read_int()
        cdpwry = read_int()
        cdpwrz = read_int()
        cdtms = read_int(3)
        col_den_coefs = np.array(
            [read_float() for _ in range(cdtms)])

        return RSMCoefficients(
            row_off=row_off,
            col_off=col_off,
            row_norm_sf=row_norm_sf,
            col_norm_sf=col_norm_sf,
            x_off=x_off,
            y_off=y_off,
            z_off=z_off,
            x_norm_sf=x_norm_sf,
            y_norm_sf=y_norm_sf,
            z_norm_sf=z_norm_sf,
            row_num_powers=np.array([rnpwrx, rnpwry, rnpwrz]),
            row_den_powers=np.array([rdpwrx, rdpwry, rdpwrz]),
            col_num_powers=np.array([cnpwrx, cnpwry, cnpwrz]),
            col_den_powers=np.array([cdpwrx, cdpwry, cdpwrz]),
            row_num_coefs=row_num_coefs,
            row_den_coefs=row_den_coefs,
            col_num_coefs=col_num_coefs,
            col_den_coefs=col_den_coefs,
            rsn=rsn,
            csn=csn,
            row_fit_error=rfep,
            col_fit_error=cfep,
        )
    except (ValueError, IndexError):
        return None


def _parse_rsmida_tre(tre_value: str) -> Optional[RSMIdentification]:
    """Parse an RSMIDA TRE CEDATA string into RSMIdentification.

    Field layout per STDI-0002 Vol 1 Appendix U, Table 2 (RSMIDA)::

        IID(80) EDITION(40) ISID(40) SID(40) STID(40)
        YEAR(4) MONTH(2) DAY(2) HOUR(2) MINUTE(2) SECOND(9)
        NRG(8) NCG(8) TRG(21) TCG(21) GRNDD(1)
        XUOR(21) YUOR(21) ZUOR(21)
        XUXR(21)..ZUZR(21)  [9 unit-vector components]
        V1X(21)..V8Z(21)    [24 ground domain vertices]
        GRPX(21) GRPY(21) GRPZ(21)
        FULLR(8) FULLC(8) MINR(8) MAXR(8) MINC(8) MAXC(8)
        [illumination and trajectory fields follow]

    Total CEDATA length: 1628 bytes.

    Parameters
    ----------
    tre_value : str
        Raw CEDATA string from GDAL TRE metadata.

    Returns
    -------
    RSMIdentification or None
        Parsed identification, or None if parsing fails.
    """
    try:
        v = tre_value.strip()
        # RSMIDA CEDATA is fixed at 1628 bytes
        if len(v) < 500:
            return None

        pos = 0

        def read_str(n: int) -> str:
            nonlocal pos
            s = v[pos:pos + n].strip()
            pos += n
            return s

        def read_float(n: int = 21) -> float:
            nonlocal pos
            raw = v[pos:pos + n].strip()
            pos += n
            if not raw:
                return 0.0
            return float(raw)

        def read_int(n: int) -> int:
            nonlocal pos
            raw = v[pos:pos + n].strip()
            pos += n
            if not raw:
                return 0
            return int(raw)

        # --- Image Information ---
        image_id = read_str(80)     # IID
        edition = read_str(40)      # EDITION

        # --- Sensor identification ---
        isid = read_str(40)         # ISID
        sensor_id = read_str(40)    # SID
        sensor_type_id = read_str(40)  # STID

        # --- Date/time ---
        year_s = read_str(4)        # YEAR
        month_s = read_str(2)       # MONTH
        day_s = read_str(2)         # DAY
        hour_s = read_str(2)        # HOUR
        minute_s = read_str(2)      # MINUTE
        second_s = read_str(9)      # SECOND

        collection_dt = None
        try:
            if year_s and month_s and day_s:
                sec_f = float(second_s) if second_s else 0.0
                sec_int = int(sec_f)
                micro = int((sec_f - sec_int) * 1_000_000)
                collection_dt = datetime(
                    int(year_s), int(month_s), int(day_s),
                    int(hour_s) if hour_s else 0,
                    int(minute_s) if minute_s else 0,
                    sec_int, micro,
                )
        except (ValueError, TypeError):
            pass

        # --- Time-of-image model ---
        nrg = read_int(8)           # NRG (8 bytes)
        ncg = read_int(8)           # NCG (8 bytes)
        trg_s = read_str(21)        # TRG (21 bytes)
        tcg_s = read_str(21)        # TCG (21 bytes)
        trg = float(trg_s) if trg_s else None
        tcg = float(tcg_s) if tcg_s else None

        # --- Ground coordinate system ---
        grndd = read_str(1)         # GRNDD

        # --- Rectangular coordinate origin and unit vectors ---
        # Per STDI-0002 App U Table 2, XUOR/YUOR/ZUOR (3 × 21) and the
        # XUXR..ZUZR unit-vector matrix (9 × 21) are present only when
        # GRNDD='R' (rectangular).  For GRNDD='G' (geodetic) these
        # fields are absent and the parser must skip them.
        coord_origin = None
        coord_unit_vectors = np.zeros((3, 3), dtype=np.float64)
        if grndd == 'R':
            xuor_s = read_str(21)       # XUOR
            yuor_s = read_str(21)       # YUOR
            zuor_s = read_str(21)       # ZUOR
            try:
                if xuor_s and yuor_s and zuor_s:
                    coord_origin = XYZ(
                        x=float(xuor_s), y=float(yuor_s), z=float(zuor_s))
            except ValueError:
                pass

            uv_vals = []
            for _ in range(9):
                uv_s = read_str(21)
                try:
                    uv_vals.append(float(uv_s) if uv_s else 0.0)
                except ValueError:
                    uv_vals.append(0.0)
            coord_unit_vectors = np.array(
                uv_vals, dtype=np.float64).reshape(3, 3)

        # --- Ground domain vertices (24 × 21 = 504 bytes) ---
        vert_vals = []
        for _ in range(24):
            vs = read_str(21)
            try:
                vert_vals.append(float(vs) if vs else 0.0)
            except ValueError:
                vert_vals.append(0.0)
        ground_domain_vertices = np.array(
            vert_vals, dtype=np.float64).reshape(8, 3)

        # --- Ground reference point (3 × 21) ---
        grpx = read_float(21)       # GRPX
        grpy = read_float(21)       # GRPY
        grpz = read_float(21)       # GRPZ

        # --- Image extent fields (6 × 8 bytes) ---
        fullr = None
        fullc = None
        minr = None
        maxr = None
        minc = None
        maxc = None
        try:
            if pos + 48 <= len(v):
                fullr_s = read_str(8)
                fullc_s = read_str(8)
                minr_s = read_str(8)
                maxr_s = read_str(8)
                minc_s = read_str(8)
                maxc_s = read_str(8)
                fullr = int(fullr_s) if fullr_s else None
                fullc = int(fullc_s) if fullc_s else None
                minr = int(minr_s) if minr_s else None
                maxr = int(maxr_s) if maxr_s else None
                minc = int(minc_s) if minc_s else None
                maxc = int(maxc_s) if maxc_s else None
        except (ValueError, IndexError):
            pass

        return RSMIdentification(
            image_id=image_id,
            edition=edition,
            sensor_id=sensor_id,
            sensor_type_id=sensor_type_id,
            image_sensor_id=isid if isid else None,
            collection_datetime=collection_dt,
            ground_domain_type=grndd,
            ground_ref_point=XYZ(x=grpx, y=grpy, z=grpz),
            num_row_sections=nrg,
            num_col_sections=ncg,
            time_ref_row=trg,
            time_ref_col=tcg,
            coord_origin=coord_origin,
            coord_unit_vectors=coord_unit_vectors,
            ground_domain_vertices=ground_domain_vertices,
            full_image_rows=fullr,
            full_image_cols=fullc,
            min_row=minr,
            max_row=maxr,
            min_col=minc,
            max_col=maxc,
        )
    except (ValueError, IndexError):
        return None


def _parse_csexra_tre(tre_value: str) -> Optional[CSEXRAMetadata]:
    """Parse a CSEXRA TRE CEDATA string.

    Compensated Sensor Error Extension Record provides CE90, LE90,
    and GSD accuracy estimates.

    Parameters
    ----------
    tre_value : str
        Raw CEDATA string from GDAL TRE metadata.

    Returns
    -------
    CSEXRAMetadata or None
        Parsed metadata, or None if parsing fails.
    """
    try:
        v = tre_value.strip()
        if len(v) < 120:
            return None

        pos = 0

        def read_str(n: int) -> str:
            nonlocal pos
            s = v[pos:pos + n].strip()
            pos += n
            return s

        def read_opt_float(n: int) -> Optional[float]:
            nonlocal pos
            raw = v[pos:pos + n].strip()
            pos += n
            if not raw or raw == '-':
                return None
            return float(raw)

        predicted_niirs = read_opt_float(3)  # PREDICTED_NIIRS
        read_str(3)                          # CSNR (not stored)
        read_str(1)                          # GROUND_COVER (not stored)
        read_str(3)                          # SNOW_DEPTH_CAT (not stored)
        read_str(7)                          # SUN_AZIMUTH (handled by USE00A)
        read_str(5)                          # SUN_ELEVATION
        read_str(7)                          # PREDICTED_GSD (deprecated)
        ground_gsd_row = read_opt_float(5)   # GROUND_GSD_ROW
        ground_gsd_col = read_opt_float(5)   # GROUND_GSD_COL
        read_str(5)                          # GSD_BETA_ANGLE
        read_str(2)                          # DYNAMIC_RANGE
        read_str(7)                          # LINE_FF_AVG
        read_str(7)                          # COL_FF_AVG
        read_str(7)                          # LINE_FF_STD
        read_str(7)                          # COL_FF_STD
        read_str(7)                          # LINE_CNT_AVG
        read_str(7)                          # COL_CNT_AVG
        read_str(7)                          # LINE_CNT_STD
        read_str(7)                          # COL_CNT_STD
        read_str(5)                          # PREDICTED_EDGE_RESP_ROW
        read_str(5)                          # PREDICTED_EDGE_RESP_COL
        read_str(5)                          # PREDICTED_MTF_EL
        read_str(5)                          # PREDICTED_MTF_AZ
        ce90 = read_opt_float(7)             # CE
        le90 = read_opt_float(7)             # LE

        return CSEXRAMetadata(
            predicted_niirs=predicted_niirs,
            ce90=ce90,
            le90=le90,
            ground_gsd_row=ground_gsd_row,
            ground_gsd_col=ground_gsd_col,
        )
    except (ValueError, IndexError):
        return None


def _parse_cscrna_tre(tre_value: str) -> Optional[CSCRNAMetadata]:
    """Parse a CSCRNA TRE CEDATA string.

    Corner Footprint TRE per STDI-0002 (109-byte CEDATA)::

        PREDICT_CORNERS(1)
        ULCNR_LAT(9) ULCNR_LONG(10) ULCNR_HT(8)
        URCNR_LAT(9) URCNR_LONG(10) URCNR_HT(8)
        LRCNR_LAT(9) LRCNR_LONG(10) LRCNR_HT(8)
        LLCNR_LAT(9) LLCNR_LONG(10) LLCNR_HT(8)

    Parameters
    ----------
    tre_value : str
        Raw CEDATA string from GDAL TRE metadata.

    Returns
    -------
    CSCRNAMetadata or None
        Parsed corner footprint, or None if parsing fails.
    """
    try:
        v = tre_value
        if len(v) < 109:
            return None

        pos = 0

        def read_str(n: int) -> str:
            nonlocal pos
            s = v[pos:pos + n].strip()
            pos += n
            return s

        predict = read_str(1) == 'Y'
        corners = np.empty((4, 2), dtype=np.float64)
        heights = np.empty(4, dtype=np.float64)
        for i in range(4):
            corners[i, 0] = float(read_str(9))    # *CNR_LAT
            corners[i, 1] = float(read_str(10))   # *CNR_LONG
            ht_s = read_str(8)                    # *CNR_HT
            heights[i] = float(ht_s) if ht_s else 0.0

        return CSCRNAMetadata(
            predicted=predict, corners=corners, heights=heights)
    except (ValueError, IndexError):
        return None


def _parse_use00a_tre(tre_value: str) -> Optional[USE00AMetadata]:
    """Parse a USE00A TRE CEDATA string.

    Exploitation Usability Extension provides sensor geometry,
    sun angles, and mean GSD.

    Parameters
    ----------
    tre_value : str
        Raw CEDATA string from GDAL TRE metadata.

    Returns
    -------
    USE00AMetadata or None
        Parsed metadata, or None if parsing fails.
    """
    try:
        v = tre_value.strip()
        if len(v) < 107:
            return None

        pos = 0

        def read_str(n: int) -> str:
            nonlocal pos
            s = v[pos:pos + n].strip()
            pos += n
            return s

        def read_opt_float(n: int) -> Optional[float]:
            nonlocal pos
            raw = v[pos:pos + n].strip()
            pos += n
            if not raw or raw == '-':
                return None
            return float(raw)

        obliquity_angle = read_opt_float(5)  # OBL_ANG
        read_str(5)                           # ROLL_ANG
        read_str(12)                          # PRIME_ID
        read_str(15)                          # PRIME_BE
        read_str(5)                           # ILETEFOV (N/RS)
        sun_azimuth = read_opt_float(7)       # SUN_AZ
        sun_elevation = read_opt_float(5)     # SUN_EL
        read_str(1)                           # DATA_RATE (not stored)
        read_str(5)                           # NRL_EL
        read_str(5)                           # NRL_AZ
        read_str(5)                           # SWATH_AZ
        mean_gsd = read_opt_float(5)          # MEAN_GSD
        predicted_niirs = read_opt_float(3)   # PREDICTED_NIIRS

        return USE00AMetadata(
            obliquity_angle=obliquity_angle,
            sun_azimuth=sun_azimuth,
            sun_elevation=sun_elevation,
            mean_gsd=mean_gsd,
            predicted_niirs=predicted_niirs,
        )
    except (ValueError, IndexError):
        return None


def _parse_ichipb_tre(tre_value: str) -> Optional[ICHIPBMetadata]:
    """Parse an ICHIPB TRE CEDATA string.

    Field layout per STDI-0002 Vol 1 Appendix B (ICHIPB v1.0/CN2,
    224 bytes total — see Tables B-2 and B-3)::

        XFRM_FLAG(2) SCALE_FACTOR(10) ANAMRPH_CORR(2) SCANBLK_NUM(2)
        OP_ROW_11(12) OP_COL_11(12) OP_ROW_12(12) OP_COL_12(12)
        OP_ROW_21(12) OP_COL_21(12) OP_ROW_22(12) OP_COL_22(12)
        FI_ROW_11(12) FI_COL_11(12) FI_ROW_12(12) FI_COL_12(12)
        FI_ROW_21(12) FI_COL_21(12) FI_ROW_22(12) FI_COL_22(12)
        FI_ROW(8) FI_COL(8)

    All 22 fields are required; this parser stores every spec field
    verbatim on :class:`ICHIPBMetadata` and additionally derives the
    separable axial affine ``full = fi_off + fi_scale * chip`` for
    consumers that don't need the rotated case.

    When ``XFRM_FLAG=01`` the spec mandates the OP/FI corner and
    extent fields are zero-fill; the derived affine fields are left
    ``None`` to flag the no-data case.  ``FI_ROW``/``FI_COL`` of
    ``00000000`` (per B.8.2 — chipping app didn't know the
    full-image size) are surfaced as ``None`` so consumers can
    distinguish "unknown" from "explicitly zero".
    """
    try:
        v = tre_value.strip()
        if len(v) < 224:
            return None

        pos = 0

        def read_float(n: int) -> float:
            nonlocal pos
            raw = v[pos:pos + n].strip()
            pos += n
            return float(raw)

        def read_int(n: int) -> int:
            nonlocal pos
            raw = v[pos:pos + n].strip()
            pos += n
            return int(raw)

        xfrm_flag = read_int(2)              # XFRM_FLAG
        scale_factor = read_float(10)        # SCALE_FACTOR
        anamorphic_corr = read_int(2)        # ANAMRPH_CORR
        scanblk_num = read_int(2)            # SCANBLK_NUM

        op_row_11 = read_float(12)
        op_col_11 = read_float(12)
        op_row_12 = read_float(12)
        op_col_12 = read_float(12)
        op_row_21 = read_float(12)
        op_col_21 = read_float(12)
        op_row_22 = read_float(12)
        op_col_22 = read_float(12)

        fi_row_11 = read_float(12)
        fi_col_11 = read_float(12)
        fi_row_12 = read_float(12)
        fi_col_12 = read_float(12)
        fi_row_21 = read_float(12)
        fi_col_21 = read_float(12)
        fi_row_22 = read_float(12)
        fi_col_22 = read_float(12)

        full_image_rows = read_int(8)
        full_image_cols = read_int(8)
        # 00000000 is the spec's "unknown" sentinel (B.8.2).
        if full_image_rows == 0:
            full_image_rows = None
        if full_image_cols == 0:
            full_image_cols = None

        # Axial-affine derivation; suppressed when XFRM_FLAG=01 since
        # the corners are zero-fill in that case.
        fi_row_off = fi_col_off = None
        fi_row_scale = fi_col_scale = None
        if xfrm_flag != 1:
            d_op_r = op_row_22 - op_row_11
            d_op_c = op_col_22 - op_col_11
            fi_row_scale = ((fi_row_22 - fi_row_11) / d_op_r
                            if d_op_r != 0 else 1.0)
            fi_col_scale = ((fi_col_22 - fi_col_11) / d_op_c
                            if d_op_c != 0 else 1.0)
            fi_row_off = fi_row_11 - fi_row_scale * op_row_11
            fi_col_off = fi_col_11 - fi_col_scale * op_col_11

        return ICHIPBMetadata(
            xfrm_flag=xfrm_flag,
            scale_factor=scale_factor,
            anamorphic_corr=anamorphic_corr,
            scanblk_num=scanblk_num,
            op_row_11=op_row_11, op_col_11=op_col_11,
            op_row_12=op_row_12, op_col_12=op_col_12,
            op_row_21=op_row_21, op_col_21=op_col_21,
            op_row_22=op_row_22, op_col_22=op_col_22,
            fi_row_11=fi_row_11, fi_col_11=fi_col_11,
            fi_row_12=fi_row_12, fi_col_12=fi_col_12,
            fi_row_21=fi_row_21, fi_col_21=fi_col_21,
            fi_row_22=fi_row_22, fi_col_22=fi_col_22,
            full_image_rows=full_image_rows,
            full_image_cols=full_image_cols,
            fi_row_off=fi_row_off,
            fi_col_off=fi_col_off,
            fi_row_scale=fi_row_scale,
            fi_col_scale=fi_col_scale,
            scale_factor_r=scale_factor,
            scale_factor_c=scale_factor,
            op_row=op_row_22,
            op_col=op_col_22,
        )
    except (ValueError, IndexError):
        return None


def _parse_blocka_tre(tre_value: str) -> Optional[BLOCKAMetadata]:
    """Parse a BLOCKA TRE CEDATA string.

    Image Geographic Location provides corner coordinates.

    Parameters
    ----------
    tre_value : str
        Raw CEDATA string from GDAL TRE metadata.

    Returns
    -------
    BLOCKAMetadata or None
        Parsed metadata, or None if parsing fails.
    """
    try:
        v = tre_value.strip()
        # BLOCKA layout per STDI-0002 Vol 1 App F:
        #   BLOCK_INSTANCE(2) N_GRAY(5) L_LINES(5)
        #   LAYOVER_ANGLE(3) SHADOW_ANGLE(3)
        #   FRFC_LOC(21) FRLC_LOC(21) LRFC_LOC(21) LRLC_LOC(21)
        #   reserved(5)
        # Total: 107 bytes
        if len(v) < 102:
            return None

        pos = 0

        def read_str(n: int) -> str:
            nonlocal pos
            s = v[pos:pos + n].strip()
            pos += n
            return s

        def read_int(n: int) -> int:
            nonlocal pos
            raw = v[pos:pos + n].strip()
            pos += n
            return int(raw)

        block_number = read_int(2)           # BLOCK_INSTANCE
        read_str(5)                          # N_GRAY
        read_str(5)                          # L_LINES
        read_str(3)                          # LAYOVER_ANGLE
        read_str(3)                          # SHADOW_ANGLE
        # Spec field order: FRFC, FRLC, LRFC, LRLC
        frfc_loc = read_str(21)
        frlc_loc = read_str(21)
        lrfc_loc = read_str(21)
        lrlc_loc = read_str(21)

        return BLOCKAMetadata(
            block_number=block_number,
            frfc_loc=frfc_loc if frfc_loc else None,
            frlc_loc=frlc_loc if frlc_loc else None,
            lrfc_loc=lrfc_loc if lrfc_loc else None,
            lrlc_loc=lrlc_loc if lrlc_loc else None,
        )
    except (ValueError, IndexError):
        return None


def _parse_aimidb_tre(tre_value: str) -> Optional[Dict[str, Any]]:
    """Parse an AIMIDB TRE CEDATA string.

    Additional Image ID provides collection date/time and mission info.

    Returns
    -------
    dict or None
        Keys: ``'collection_datetime'``, ``'mission_id'``,
        ``'country_code'``.
    """
    try:
        v = tre_value.strip()
        if len(v) < 89:
            return None

        pos = 0

        def read_str(n: int) -> str:
            nonlocal pos
            s = v[pos:pos + n].strip()
            pos += n
            return s

        # ACQUISITION_DATE: DDHHMMSSZMONYY (14 chars)
        acq_date_s = read_str(14)
        mission_id = read_str(12)            # MISSION_NO
        read_str(20)                         # MISSION_IDENTIFICATION
        read_str(6)                          # FLIGHT_NO
        read_str(3)                          # OP_NUM
        read_str(2)                          # CURRENT_SEGMENT
        read_str(2)                          # REPROCESS_NUM
        read_str(3)                          # REPLAY
        read_str(1)                          # START_TILE_COL
        read_str(5)                          # START_TILE_ROW
        read_str(2)                          # END_SEGMENT
        read_str(2)                          # END_TILE_COL
        read_str(5)                          # END_TILE_ROW
        country_code = read_str(2)           # COUNTRY

        collection_dt = None
        if acq_date_s and len(acq_date_s) >= 14:
            try:
                # Format: DDHHMMSSZMONYY
                day = int(acq_date_s[0:2])
                hour = int(acq_date_s[2:4])
                minute = int(acq_date_s[4:6])
                second = int(acq_date_s[6:8])
                mon_str = acq_date_s[9:12]
                year_2 = int(acq_date_s[12:14])
                months = {
                    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
                    'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
                    'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
                }
                month = months.get(mon_str.upper(), 1)
                year = 2000 + year_2 if year_2 < 80 else 1900 + year_2
                collection_dt = datetime(
                    year, month, day, hour, minute, second)
            except (ValueError, KeyError):
                pass

        return {
            'collection_datetime': collection_dt,
            'mission_id': mission_id if mission_id else None,
            'country_code': country_code if country_code else None,
        }
    except (ValueError, IndexError):
        return None


def _parse_stdidc_tre(tre_value: str) -> Optional[Dict[str, Any]]:
    """Parse a STDIDC TRE CEDATA string.

    Standard ID Extension provides acquisition date and mission number.

    Returns
    -------
    dict or None
        Keys: ``'collection_datetime'``, ``'mission_id'``,
        ``'country_code'``.
    """
    try:
        v = tre_value.strip()
        if len(v) < 89:
            return None

        pos = 0

        def read_str(n: int) -> str:
            nonlocal pos
            s = v[pos:pos + n].strip()
            pos += n
            return s

        acq_date_s = read_str(14)            # ACQUISITION_DATE
        mission_id = read_str(14)            # MISSION
        read_str(2)                          # PASS
        read_str(3)                          # OP_NUM
        read_str(2)                          # START_SEGMENT
        read_str(2)                          # REPROCESS_NUM
        read_str(3)                          # END_SEGMENT
        country_code = read_str(2)           # COUNTRY_CODE

        collection_dt = None
        if acq_date_s and len(acq_date_s) >= 14:
            try:
                collection_dt = datetime.strptime(
                    acq_date_s[:14], '%Y%m%d%H%M%S')
            except ValueError:
                pass

        return {
            'collection_datetime': collection_dt,
            'mission_id': mission_id if mission_id else None,
            'country_code': country_code if country_code else None,
        }
    except (ValueError, IndexError):
        return None


def _parse_piaimc_tre(tre_value: str) -> Optional[Dict[str, Any]]:
    """Parse a PIAIMC TRE CEDATA string.

    Profile for Imagery Access Image provides sensor mode, cloud cover,
    and band info.

    Returns
    -------
    dict or None
        Keys: ``'sensor_mode'``, ``'cloud_cover'``.
    """
    try:
        v = tre_value.strip()
        if len(v) < 73:
            return None

        pos = 0

        def read_str(n: int) -> str:
            nonlocal pos
            s = v[pos:pos + n].strip()
            pos += n
            return s

        def read_opt_float(n: int) -> Optional[float]:
            nonlocal pos
            raw = v[pos:pos + n].strip()
            pos += n
            if not raw or raw == '-':
                return None
            return float(raw)

        read_str(7)                          # CLOUDCVR
        read_str(1)                          # SRP_FLAG
        sensor_mode = read_str(12)           # SENSMODE
        read_str(18)                         # SENSNAME
        read_str(5)                          # SOURCE
        cloud_cover = read_opt_float(3)      # COMGEN (cloud cover %)
        # remaining fields not critical for geospatial context

        return {
            'sensor_mode': sensor_mode if sensor_mode else None,
            'cloud_cover': cloud_cover,
        }
    except (ValueError, IndexError):
        return None


def _parse_csepha_tre(tre_value: str) -> Optional[CSEPHAMetadata]:
    """Parse a CSEPHA TRE CEDATA string.

    Commercial Support Ephemeris Data -- ECEF position samples of the
    sensor during the collection.  Layout per STDI-0002 Vol 1
    Appendix D (Table D-3)::

        EPHEM_FLAG(12 A) DT_EPHEM(5 N) DATE_EPHEM(8 N) T0_EPHEM(13 N)
        NUM_EPHEM(3 N) [EPHEM_X(12 N) EPHEM_Y(12 N) EPHEM_Z(12 N)] * N

    ``EPHEM_X/Y/Z`` are signed decimals including the sign character in
    their 12-byte width.  CEL ranges from 77 to 36005 bytes.
    """
    try:
        v = tre_value
        # 41 bytes fixed header + at least 36 bytes for one sample
        if len(v) < 41 + 36:
            return None

        pos = 0

        def read_str(n: int) -> str:
            nonlocal pos
            s = v[pos:pos + n].strip()
            pos += n
            return s

        def read_signed_float(n: int) -> float:
            nonlocal pos
            raw = v[pos:pos + n].strip()
            pos += n
            return float(raw)

        ephem_flag = read_str(12) or None
        dt_ephem = float(read_str(5))
        date_ephem = read_str(8) or None
        t0_raw = read_str(13)
        # HHMMSS.mmmmmm → seconds since midnight
        hh = int(t0_raw[0:2])
        mm = int(t0_raw[2:4])
        ss = float(t0_raw[4:])
        t0_ephem = hh * 3600.0 + mm * 60.0 + ss

        num = int(read_str(3))
        if num < 1 or 41 + 36 * num > len(v):
            return None

        pos_array = np.empty((num, 3), dtype=np.float64)
        for i in range(num):
            pos_array[i, 0] = read_signed_float(12)
            pos_array[i, 1] = read_signed_float(12)
            pos_array[i, 2] = read_signed_float(12)

        return CSEPHAMetadata(
            ephem_flag=ephem_flag,
            dt_ephem=dt_ephem,
            date_ephem=date_ephem,
            t0_ephem=t0_ephem,
            num_ephem=num,
            position=pos_array,
        )
    except (ValueError, IndexError):
        return None


def _parse_rsmgga_tre(tre_value: str) -> Optional[RSMGGAMetadata]:
    """Parse an RSMGGA TRE CEDATA string.

    RSM Ground-to-Image Grid -- stacked ground→image grid planes.
    Layout per STDI-0002 Vol 1 Appendix U, Table 12::

        IID(80 A) EDITION(40 A) GGRSN(3 N) GGCSN(3 N)
        GGRFEP(21 A) GGCFEP(21 A) INTORD(1 A)
        NPLN(3 N) DELTAZ(21 A) DELTAX(21 A) DELTAY(21 A)
        ZPLN1(21 A) XIPLN1(21 A) YIPLN1(21 A)
        REFROW(9 A/signed) REFCOL(9 A/signed)
        TNUMRD(2 N) TNUMCD(2 N) FNUMRD(1 N) FNUMCD(1 N)
        [IXO(4 A/signed) IYO(4 A/signed)] × (NPLN - 1)
        per plane: NXPTS(3 N) NYPTS(3 N)
                   RCOORD(TNUMRD A) CCOORD(TNUMCD A) per grid point

    Grid points are matrix row-major over the X axis; all sci-notation
    real fields are ``±d.ddddddddddddddE±dd``.
    """
    try:
        v = tre_value
        # Fixed header is 324 bytes (after CETAG/CEL stripped by caller)
        if len(v) < 324:
            return None

        pos = 0

        def read_str(n: int) -> str:
            nonlocal pos
            s = v[pos:pos + n].strip()
            pos += n
            return s

        def read_optional_float(n: int = 21) -> Optional[float]:
            nonlocal pos
            raw = v[pos:pos + n]
            pos += n
            s = raw.strip()
            if not s:
                return None
            try:
                return float(s)
            except ValueError:
                return None

        def read_required_float(n: int = 21) -> float:
            nonlocal pos
            raw = v[pos:pos + n].strip()
            pos += n
            return float(raw)

        iid = read_str(80) or None
        edition = read_str(40) or None
        ggrsn = int(read_str(3))
        ggcsn = int(read_str(3))
        row_fit_error = read_optional_float(21)
        col_fit_error = read_optional_float(21)

        intord_raw = v[pos:pos + 1]
        pos += 1
        interpolation_order = (
            int(intord_raw) if intord_raw.strip() else None)

        num_planes = int(read_str(3))
        if num_planes < 2:
            return None

        delta_z = read_required_float(21)
        delta_x = read_required_float(21)
        delta_y = read_required_float(21)
        z1 = read_required_float(21)
        x1 = read_required_float(21)
        y1 = read_required_float(21)

        ref_row = int(read_str(9))
        ref_col = int(read_str(9))
        tnumrd = int(read_str(2))
        tnumcd = int(read_str(2))
        fnumrd = int(read_str(1))
        fnumcd = int(read_str(1))
        # Unused but keep cursor advance explicit for clarity
        _ = (fnumrd, fnumcd)

        if not (3 <= tnumrd <= 11 and 3 <= tnumcd <= 11):
            return None

        # Planes 2..P store integer offsets of their initial (x, y) in
        # multiples of (delta_x, delta_y) -- signed 4-byte fields.
        offsets: List[Tuple[int, int]] = [(0, 0)]
        for _ in range(num_planes - 1):
            ixo = int(read_str(4))
            iyo = int(read_str(4))
            offsets.append((ixo, iyo))

        planes: List[RSMGGAGridPlane] = []
        for p_idx in range(num_planes):
            nxpts = int(read_str(3))
            nypts = int(read_str(3))
            if nxpts < 2 or nypts < 2:
                return None

            rows_arr = np.full((nxpts, nypts), np.nan, dtype=np.float64)
            cols_arr = np.full((nxpts, nypts), np.nan, dtype=np.float64)

            # Grid points are matrix row-major: outer loop over X.
            for ix in range(nxpts):
                for iy in range(nypts):
                    r_raw = v[pos:pos + tnumrd]
                    pos += tnumrd
                    c_raw = v[pos:pos + tnumcd]
                    pos += tnumcd
                    r_s = r_raw.strip()
                    c_s = c_raw.strip()
                    if r_s:
                        rows_arr[ix, iy] = float(r_s)
                    if c_s:
                        cols_arr[ix, iy] = float(c_s)

            ixo, iyo = offsets[p_idx]
            planes.append(RSMGGAGridPlane(
                z_plane=z1 + p_idx * delta_z,
                xi=x1 + ixo * delta_x,
                yi=y1 + iyo * delta_y,
                num_x=nxpts,
                num_y=nypts,
                rows=rows_arr,
                cols=cols_arr,
            ))

        return RSMGGAMetadata(
            iid=iid,
            edition=edition,
            ggrsn=ggrsn,
            ggcsn=ggcsn,
            row_fit_error=row_fit_error,
            col_fit_error=col_fit_error,
            interpolation_order=interpolation_order,
            num_planes=num_planes,
            delta_z=delta_z,
            delta_x=delta_x,
            delta_y=delta_y,
            ref_row=ref_row,
            ref_col=ref_col,
            planes=planes,
        )
    except (ValueError, IndexError):
        return None


def _build_accuracy_info(
    csexra: Optional[CSEXRAMetadata],
    use00a: Optional[USE00AMetadata],
    rpc: Optional[RPCCoefficients],
    rsm_dca: Optional[Any] = None,
    rsm_eca: Optional[Any] = None,
) -> Optional[AccuracyInfo]:
    """Build aggregated accuracy from best available TRE source.

    Priority: RSM error covariance (RSMDCA/B then RSMECA/B, the
    rigorous sensor-model error propagation) > CSEXRA > USE00A >
    RPC err_bias.

    Parameters
    ----------
    csexra : CSEXRAMetadata or None
    use00a : USE00AMetadata or None
    rpc : RPCCoefficients or None
    rsm_dca : RSMDCAMetadata or None
        Direct error covariance (A or B variant).
    rsm_eca : RSMECAMetadata or None
        Indirect error covariance (A or B variant).

    Returns
    -------
    AccuracyInfo or None
    """
    # GSD context from whichever TRE carries it; attached to the
    # winning accuracy source.
    gsd = None
    if csexra is not None:
        if csexra.ground_gsd_row is not None and csexra.ground_gsd_col is not None:
            gsd = (csexra.ground_gsd_row + csexra.ground_gsd_col) / 2.0
        elif csexra.ground_gsd_row is not None:
            gsd = csexra.ground_gsd_row
        elif csexra.ground_gsd_col is not None:
            gsd = csexra.ground_gsd_col
    if gsd is None and use00a is not None:
        gsd = use00a.mean_gsd

    # RSM error covariance — rigorous CE90/LE90 from the sensor model.
    if rsm_dca is not None:
        acc = summarize_accuracy(rsm_dca, None)
        if acc is not None:
            variant = getattr(rsm_dca, 'variant', 'A') or 'A'
            return AccuracyInfo(
                ce90=acc[0], le90=acc[1],
                mean_gsd=gsd, source=f'RSMDC{variant}')
    if rsm_eca is not None:
        acc = summarize_accuracy(None, rsm_eca)
        if acc is not None:
            variant = getattr(rsm_eca, 'variant', 'A') or 'A'
            return AccuracyInfo(
                ce90=acc[0], le90=acc[1],
                mean_gsd=gsd, source=f'RSMEC{variant}')

    if csexra and (csexra.ce90 is not None or csexra.le90 is not None):
        return AccuracyInfo(
            ce90=csexra.ce90, le90=csexra.le90,
            mean_gsd=gsd, source='CSEXRA')

    if use00a and use00a.mean_gsd is not None:
        return AccuracyInfo(
            ce90=None, le90=None,
            mean_gsd=use00a.mean_gsd, source='USE00A')

    if rpc and rpc.err_bias is not None:
        return AccuracyInfo(
            ce90=rpc.err_bias, le90=None,
            mean_gsd=None, source='RPC00B')

    return None


def _build_collection_info(
    aimidb: Optional[Dict[str, Any]],
    stdidc: Optional[Dict[str, Any]],
    piaimc: Optional[Dict[str, Any]],
) -> Optional[CollectionInfo]:
    """Build aggregated collection context from available TREs.

    Priority: AIMIDB > STDIDC > PIAIMC for overlapping fields.

    Parameters
    ----------
    aimidb : dict or None
    stdidc : dict or None
    piaimc : dict or None

    Returns
    -------
    CollectionInfo or None
    """
    if not aimidb and not stdidc and not piaimc:
        return None

    dt = None
    mission = None
    country = None
    sensor_mode = None
    cloud_cover = None

    # AIMIDB first (highest priority)
    if aimidb:
        dt = aimidb.get('collection_datetime')
        mission = aimidb.get('mission_id')
        country = aimidb.get('country_code')

    # STDIDC fills gaps
    if stdidc:
        if dt is None:
            dt = stdidc.get('collection_datetime')
        if mission is None:
            mission = stdidc.get('mission_id')
        if country is None:
            country = stdidc.get('country_code')

    # PIAIMC fills remaining gaps
    if piaimc:
        sensor_mode = piaimc.get('sensor_mode')
        cloud_cover = piaimc.get('cloud_cover')

    if not any([dt, mission, sensor_mode, cloud_cover, country]):
        return None

    return CollectionInfo(
        collection_datetime=dt,
        mission_id=mission,
        sensor_mode=sensor_mode,
        cloud_cover=cloud_cover,
        country_code=country,
    )


# ===================================================================
# Multi-image NITF segment plumbing
# ===================================================================


@dataclass
class _OpenSegment:
    """Internal: one opened NITF image segment with its dataset handle.

    Owned by ``EONITFReader`` in unified mode; closed in
    ``EONITFReader.close()``.  ``info`` is the public diagnostic record
    surfaced via ``metadata.image_segments``.  ``tres`` caches this
    segment's parsed TRE bundle (one parse per segment, reused by
    metadata aggregation); ``tre_source`` records which parser
    produced it (``'xml:TRE'``, ``'manual'``, or ``'none'``).
    """

    info: ImageSegmentInfo
    dataset: Any   # rasterio.DatasetReader; typed Any to avoid hard dep
    tres: Optional[Dict[str, Any]] = None
    tre_source: str = 'none'


def _decode_imag(imag_str: Optional[str]) -> float:
    """Decode the NITF IMAG (image magnification) subheader field.

    Per MIL-STD-2500C the 4-character IMAG field is either a decimal
    magnification (``'1.0 '``) or a reciprocal (``'/2  '`` meaning
    one-half resolution, used for overview/reduced-resolution image
    segments).  Returns ``1.0`` for blank or unparseable values.

    Parameters
    ----------
    imag_str : str, optional
        Raw IMAG value from the segment subheader tags.

    Returns
    -------
    float
        Magnification factor (``1.0`` = full resolution, ``0.5`` =
        half resolution, ...).
    """
    if not imag_str:
        return 1.0
    s = imag_str.strip()
    if not s:
        return 1.0
    try:
        if s.startswith('/'):
            denom = float(s[1:])
            return 1.0 / denom if denom != 0 else 1.0
        val = float(s)
        return val if val > 0 else 1.0
    except ValueError:
        return 1.0


def _int_tag(tags: Dict[str, Any], *names: str) -> Optional[int]:
    """Return the first parseable integer among ``tags[name]`` candidates."""
    for name in names:
        raw = tags.get(name)
        if raw is None:
            continue
        try:
            return int(str(raw).strip())
        except (ValueError, TypeError):
            continue
    return None


def _segment_bbox_from_ichipb(
    ichipb: Optional[ICHIPBMetadata],
    rows: int,
    cols: int,
    fallback_row_off: int,
    fallback_col_off: int,
) -> Tuple[int, int, int, int]:
    """Return ``(fi_row_lo, fi_row_hi, fi_col_lo, fi_col_hi)`` for one segment.

    Uses the ICHIPB chip→full affine on the four chip corners when
    ICHIPB is present (per STDI-0002 Vol 1 App G the chip-corner
    indices are 11/12/21/22 for the rectangle corners).  Falls back
    to ``(fallback_row_off, fallback_row_off + rows)`` and
    ``(fallback_col_off, fallback_col_off + cols)`` when ICHIPB is
    absent — used to stack segments sequentially in row-major order
    when the file omits ICHIPB.

    Parameters
    ----------
    ichipb : ICHIPBMetadata, optional
        Per-segment ICHIPB.  When ``None`` triggers the fallback path.
    rows, cols : int
        Segment-local pixel dimensions.
    fallback_row_off, fallback_col_off : int
        Origin to use when ``ichipb`` is ``None``.

    Returns
    -------
    Tuple[int, int, int, int]
        Half-open full-image bbox.
    """
    if ichipb is None or ichipb.fi_row_scale is None:
        warnings.warn(
            "Segment without ICHIPB; assuming sequential row stacking "
            f"at row offset {fallback_row_off}.  Provide ICHIPB on the "
            "image segment for accurate full-image placement.",
            RuntimeWarning, stacklevel=3,
        )
        return (
            int(fallback_row_off),
            int(fallback_row_off + rows),
            int(fallback_col_off),
            int(fallback_col_off + cols),
        )

    # full = fi_off + fi_scale * chip; sample the four chip corners.
    fi_row_off = float(ichipb.fi_row_off or 0.0)
    fi_col_off = float(ichipb.fi_col_off or 0.0)
    fi_row_scale = float(ichipb.fi_row_scale or 1.0)
    fi_col_scale = float(ichipb.fi_col_scale or 1.0)

    chip_corners = [
        (0.5, 0.5),
        (rows - 0.5, 0.5),
        (0.5, cols - 0.5),
        (rows - 0.5, cols - 0.5),
    ]
    full_rows_corners = [fi_row_off + fi_row_scale * r for r, _ in chip_corners]
    full_cols_corners = [fi_col_off + fi_col_scale * c for _, c in chip_corners]

    return (
        int(np.floor(min(full_rows_corners))),
        int(np.ceil(max(full_rows_corners))),
        int(np.floor(min(full_cols_corners))),
        int(np.ceil(max(full_cols_corners))),
    )


class EONITFReader(ImageReader):
    """Read electro-optical NITF imagery with RPC/RSM geolocation.

    Opens EO NITF files via rasterio (GDAL NITF driver) and extracts
    RPC00B and RSM TRE metadata in addition to standard imagery.

    Multi-image NITFs are unified automatically: segments are grouped
    by compatible imaging characteristics, the primary imagery group
    is auto-discovered, and ``read_chip`` / geolocation models operate
    in that group's full-image coordinate space.  Heterogeneous files
    (overviews, cloud masks, support images alongside the primary
    imagery) load without error; every segment stays discoverable via
    ``metadata.image_segments`` / ``metadata.image_groups`` and
    readable via ``image_index`` pinning.

    Parameters
    ----------
    filepath : str or Path
        Path to the EO NITF file.

    Attributes
    ----------
    filepath : Path
        Path to the NITF file.
    metadata : EONITFMetadata
        Typed metadata including RPC/RSM geolocation models.
    dataset : rasterio.DatasetReader
        Rasterio dataset object.
    has_rpc : bool
        Whether RPC coefficients are available.
    has_rsm : bool
        Whether RSM coefficients are available.

    Raises
    ------
    ImportError
        If rasterio is not installed.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be opened as NITF.

    Examples
    --------
    >>> from grdl.IO.eo.nitf import EONITFReader
    >>> with EONITFReader('worldview.ntf') as reader:
    ...     print(f"RPC: {reader.has_rpc}, RSM: {reader.has_rsm}")
    ...     chip = reader.read_chip(0, 512, 0, 512)
    ...     if reader.has_rpc:
    ...         from grdl.geolocation.eo.rpc import RPCGeolocation
    ...         geo = RPCGeolocation.from_reader(reader)
    ...         lat, lon, h = geo.image_to_latlon(256, 256)
    """

    #: Which TRE parser produced the metadata: ``'xml:TRE'``,
    #: ``'manual'`` (legacy fallback), or ``'none'`` if no TREs were
    #: recognized.  Useful for diagnostics when geolocation looks off.
    tre_source: str = 'none'

    def __init__(
        self,
        filepath: Union[str, Path],
        read_config: Optional[ReadConfig] = None,
        use_xml_tre: bool = True,
        image_index: Optional[int] = None,
    ) -> None:
        """Open an EO NITF file.

        Parameters
        ----------
        filepath : str or Path
            Path to the EO NITF file.
        read_config : ReadConfig, optional
            Pixel-read configuration.
        use_xml_tre : bool, default True
            When ``True`` (default), TRE metadata is parsed from
            GDAL's ``xml:TRE`` namespace via
            :mod:`grdl.IO.eo._tre_xml` (field-by-name parsing --
            order-independent, no byte-offset guessing).  If GDAL
            recognizes no TREs in this namespace (vendor-specific
            files), the reader transparently falls back to the
            legacy byte-offset parser.  Set to ``False`` to force
            the legacy path -- useful for A/B comparison.
        image_index : int, optional
            When ``None`` (default), multi-image NITF files are
            transparently unified: segments are grouped by compatible
            imaging characteristics (bands, dtype, ICHIPB scale
            factor, IMAG magnification, ICAT), the *primary* group
            (full-resolution, geolocated, largest) is auto-selected,
            and ``read_chip`` is routed across that group's segments
            in full-image coordinates.  ``metadata.rows``/``cols``
            are the primary group's full-image dims and
            ``metadata.ichipb`` is set to ``None`` (the unified
            reader has absorbed the chip-to-full transform).
            Heterogeneous files (overviews, cloud masks, support
            images) never fail to load — non-primary segments are
            listed in ``metadata.image_segments`` /
            ``metadata.image_groups`` and remain readable by pinning.
            When an integer, opens only the segment at that
            subdataset index (``NITF_IM:N:``) and behaves as a
            single-segment reader.  Useful for diagnostics and for
            exploring non-primary segments.
        """
        if not _HAS_RASTERIO:
            raise ImportError(
                "rasterio is required for EO NITF reading. "
                "Install with: pip install rasterio  "
                "or: conda install -c conda-forge rasterio"
            )
        self.read_config = read_config or ReadConfig(parallel=True)
        self.use_xml_tre = bool(use_xml_tre)
        self._image_index = image_index
        self._segments: List[_OpenSegment] = []
        self._primary_segments: List[_OpenSegment] = []
        self._group_infos: List[ImageGroupInfo] = []
        apply_gdal_env(self.read_config)

        remote = _normalize_remote_path(filepath)
        if remote is not None:
            # Remote NITF via GDAL virtual filesystems (/vsicurl/,
            # /vsis3/, or any /vsi* path passed verbatim).  Bypass the
            # base class: Path() would mangle '//' in URLs and the
            # existence check does not apply to network resources.
            # GDAL_DISABLE_READDIR_ON_OPEN avoids a directory-listing
            # round trip per open; user environment wins if already set.
            os.environ.setdefault(
                'GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
            self.filepath = remote  # GDAL-style string, not a Path
            self._load_metadata()
        else:
            super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Open the file, dispatch single/pinned/unified, build metadata.

        Discovery order:

        1. ``image_index`` is set: pin to that one subdataset.
        2. The file has no GDAL subdatasets: single-image NITF — use
           the existing path.
        3. The file has subdatasets: open each, validate uniformity,
           build a unified full-image view.
        """
        try:
            parent = rasterio.open(str(self.filepath))
        except Exception as e:
            raise ValueError(
                f"Failed to open EO NITF file: {e}") from e

        try:
            if self._image_index is not None:
                self._open_pinned(parent, self._image_index)
                self._build_metadata_single()
                return

            subs = list(getattr(parent, 'subdatasets', None) or [])
            if not subs:
                # Single-image NITF — existing behavior verbatim.
                self.dataset = parent
                self._build_metadata_single()
                return

            # Multi-image NITF: discover, group, select, unify.
            parent.close()
            self._open_all_segments(subs)
            self._build_metadata_unified()
        except Exception:
            for seg in getattr(self, '_segments', []):
                try:
                    seg.dataset.close()
                except Exception:
                    pass
            self._segments = []
            self._primary_segments = []
            try:
                parent.close()
            except Exception:
                pass
            raise

    def _open_pinned(self, parent, idx: int) -> None:
        """Open a single subdataset pinned by ``image_index``.

        Closes the parent dataset and opens ``NITF_IM:idx:filepath``.
        Raises ``ValueError`` when the file has no subdatasets and
        ``idx != 0``, or when ``idx`` is out of range.
        """
        subs = list(getattr(parent, 'subdatasets', None) or [])
        if not subs:
            if idx != 0:
                parent.close()
                raise ValueError(
                    f"image_index={idx} requested but file has no "
                    f"NITF subdatasets (single-image NITF)."
                )
            self.dataset = parent
            return
        if idx < 0 or idx >= len(subs):
            parent.close()
            raise ValueError(
                f"image_index={idx} out of range for file with "
                f"{len(subs)} image segments (valid: 0..{len(subs)-1})."
            )
        uri = subs[idx]
        parent.close()
        self.dataset = rasterio.open(uri)

    def _open_all_segments(self, subs: List[str]) -> None:
        """Open every subdataset, group compatible segments, pick primary.

        Discovery pipeline (never fails on heterogeneous files):

        1. Open each subdataset; parse its TRE bundle once (cached on
           the ``_OpenSegment``) and read subheader tags (IID1, ICAT,
           IREP, IMAG, IDLVL, IALVL, ILOC).
        2. Group segments by compatible imaging characteristics:
           ``(bands, dtype, SCALE_FACTOR, IMAG, ICAT)``.  Overviews,
           cloud masks, and other support images land in their own
           groups instead of raising.
        3. Resolve each group's full-image placement: ICHIPB affine
           when every member carries one; else the ILOC/IALVL
           attachment chain into the common coordinate system (per
           MIL-STD-2500C); else sequential row stacking (warns).
        4. Select the primary group — prefers full resolution
           (SCALE_FACTOR≈1, IMAG≈1), then RPC/RSM geolocation
           presence, then total pixel area, then lowest segment index.

        Populates ``self._segments`` (all segments, sorted by group
        then bbox), ``self._primary_segments`` (unified-read routing
        list), and ``self._group_infos``.  ``self.dataset`` aliases
        the first primary segment for back-compat attribute access.
        """
        def _open_one(idx_uri: Tuple[int, str]) -> Optional[Dict[str, Any]]:
            idx, uri = idx_uri
            try:
                ds = rasterio.open(uri)
            except Exception as exc:  # never fail the whole file
                warnings.warn(
                    f"Skipping unreadable image segment {idx} "
                    f"({uri!r}): {exc}",
                    RuntimeWarning, stacklevel=2,
                )
                return None
            tres, source = self._parse_segment_tres(ds)
            ichipb = tres.get('ICHIPB')
            tags = {}
            try:
                tags = ds.tags() or {}
            except (AttributeError, TypeError):
                pass
            scale = (ichipb.scale_factor_r
                     if ichipb is not None and ichipb.scale_factor_r
                     else 1.0)
            has_rpc = False
            try:
                has_rpc = ds.rpcs is not None
            except (AttributeError, TypeError):
                pass
            return {
                'index': idx,
                'uri': uri,
                'ds': ds,
                'tres': tres,
                'tre_source': source,
                'ichipb': ichipb,
                'scale': float(scale),
                'rows': ds.height,
                'cols': ds.width,
                'bands': ds.count,
                'dtype': str(ds.dtypes[0]),
                'iid1': tags.get('NITF_IID1', tags.get('IID1')) or None,
                'icat': tags.get('NITF_ICAT', tags.get('ICAT')) or None,
                'irep': tags.get('NITF_IREP', tags.get('IREP')) or None,
                'imag': _decode_imag(
                    tags.get('NITF_IMAG', tags.get('IMAG'))),
                'idlvl': _int_tag(tags, 'NITF_IDLVL', 'IDLVL'),
                'ialvl': _int_tag(tags, 'NITF_IALVL', 'IALVL'),
                'iloc_row': _int_tag(tags, 'NITF_ILOC_ROW', 'ILOC_ROW'),
                'iloc_col': _int_tag(
                    tags, 'NITF_ILOC_COLUMN', 'ILOC_COLUMN'),
                'has_geo': has_rpc
                or bool(tres.get('RSMPCA'))
                or tres.get('RSMIDA') is not None
                or bool(tres.get('RSMGGA')),
            }

        # Open + parse segments concurrently when configured: each
        # subdataset open re-reads the NITF header (and TRE parse is
        # pure CPU), so many-segment files pay N round trips serially.
        # rasterio handles are independent objects; opening them on
        # worker threads is safe.  Order is restored by index.
        if self.read_config.parallel and len(subs) > 1:
            workers = min(len(subs), _resolve_workers(self.read_config))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                records = list(pool.map(_open_one, enumerate(subs)))
        else:
            records = [_open_one(item) for item in enumerate(subs)]
        records = [r for r in records if r is not None]
        if not records:
            raise ValueError(
                f"No readable image segments in multi-image NITF "
                f"{self.filepath!s} ({len(subs)} subdatasets, all "
                "failed to open)."
            )

        # Attachment-chain CCS offsets are file-wide (a segment may be
        # attached to a segment in another group), so resolve before
        # grouping.
        self._resolve_ccs_offsets(records)

        # Group by compatible imaging characteristics, in order of
        # first appearance so group ids are deterministic.
        groups: List[List[Dict[str, Any]]] = []
        key_to_gid: Dict[Tuple, int] = {}
        for rec in records:
            key = (rec['bands'], rec['dtype'],
                   round(rec['scale'], 6), round(rec['imag'], 6),
                   rec['icat'] or '')
            gid = key_to_gid.get(key)
            if gid is None:
                gid = len(groups)
                key_to_gid[key] = gid
                groups.append([])
            rec['group_id'] = gid
            groups[gid].append(rec)

        for members in groups:
            self._place_group(members)

        primary_gid = self._select_primary_group(groups)

        self._group_infos = []
        for gid, members in enumerate(groups):
            self._group_infos.append(ImageGroupInfo(
                group_id=gid,
                segment_indices=[m['index'] for m in members],
                rows=max(m['fi_bbox'][1] for m in members),
                cols=max(m['fi_bbox'][3] for m in members),
                bands=members[0]['bands'],
                dtype=members[0]['dtype'],
                scale_factor=members[0]['scale'],
                imag=members[0]['imag'],
                icat=members[0]['icat'],
                is_primary=(gid == primary_gid),
                placement=members[0]['placement'],
                has_geolocation=any(m['has_geo'] for m in members),
            ))

        for rec in records:
            bbox = rec['fi_bbox']
            info = ImageSegmentInfo(
                segment_index=rec['index'],
                uri=rec['uri'],
                fi_row_lo=bbox[0],
                fi_row_hi=bbox[1],
                fi_col_lo=bbox[2],
                fi_col_hi=bbox[3],
                rows=rec['rows'],
                cols=rec['cols'],
                ichipb=rec['ichipb'],
                scale_factor=rec['scale'],
                group_id=rec['group_id'],
                is_primary=(rec['group_id'] == primary_gid),
                bands=rec['bands'],
                dtype=rec['dtype'],
                iid1=rec['iid1'],
                icat=rec['icat'],
                irep=rec['irep'],
                imag=rec['imag'],
                idlvl=rec['idlvl'],
                ialvl=rec['ialvl'],
                iloc_row=rec['iloc_row'],
                iloc_col=rec['iloc_col'],
                placement=rec['placement'],
            )
            self._segments.append(_OpenSegment(
                info=info, dataset=rec['ds'],
                tres=rec['tres'], tre_source=rec['tre_source'],
            ))

        self._segments.sort(
            key=lambda s: (s.info.group_id, s.info.fi_row_lo,
                           s.info.fi_col_lo, s.info.segment_index))
        self._primary_segments = [
            s for s in self._segments if s.info.is_primary]
        self.dataset = self._primary_segments[0].dataset

    def _parse_segment_tres(self, ds) -> Tuple[Dict[str, Any], str]:
        """Parse one segment's TRE bundle via xml:TRE, else legacy bytes.

        Returns
        -------
        Tuple[Dict[str, Any], str]
            ``(parsed, source)`` where ``source`` is ``'xml:TRE'``,
            ``'manual'``, or ``'none'``.
        """
        if self.use_xml_tre:
            try:
                from grdl.IO.eo._tre_xml import parse_all_tres
                parsed = parse_all_tres(ds) or {}
            except Exception:
                parsed = {}
            if any(parsed.values()):
                return parsed, 'xml:TRE'
        parsed = self._tres_manual_for_dataset(ds)
        if any(parsed.values()):
            return parsed, 'manual'
        return parsed, 'none'

    @staticmethod
    def _resolve_ccs_offsets(records: List[Dict[str, Any]]) -> None:
        """Resolve each segment's common-coordinate-system offset.

        Per MIL-STD-2500C, a segment's ILOC is an offset relative to
        the origin of the segment whose display level (IDLVL) equals
        this segment's attachment level (IALVL); ``IALVL=0`` means the
        offset is absolute in the CCS.  Walks the attachment chain
        (with cycle protection) and stores the absolute offset on
        ``rec['ccs']`` as ``(row, col)``, or ``None`` when the segment
        carries no ILOC information.
        """
        by_dlvl = {
            r['idlvl']: r for r in records if r['idlvl'] is not None}
        for rec in records:
            if rec['iloc_row'] is None or rec['iloc_col'] is None:
                rec['ccs'] = None
                continue
            row = rec['iloc_row']
            col = rec['iloc_col']
            seen = {rec['idlvl']}
            cur = rec
            while cur['ialvl'] not in (None, 0):
                parent = by_dlvl.get(cur['ialvl'])
                if (parent is None or parent is rec
                        or parent['idlvl'] in seen
                        or parent['iloc_row'] is None):
                    break
                row += parent['iloc_row']
                col += parent['iloc_col'] or 0
                seen.add(parent['idlvl'])
                cur = parent
            rec['ccs'] = (row, col)

    @staticmethod
    def _place_group(members: List[Dict[str, Any]]) -> None:
        """Compute each member's full-image bbox and placement mode.

        Priority of placement authorities:

        1. ``'ichipb'`` — every member carries an ICHIPB with a
           derivable chip→full affine (STDI-0002 App B).  Bboxes are
           absolute in the original full image.
        2. ``'iloc'`` — every member resolved an ILOC/IALVL offset
           into the CCS (MIL-STD-2500C).  Offsets are normalized so
           the group's grid starts at (0, 0).
        3. ``'stacked'`` — sequential row stacking, honoring any
           per-member ICHIPB.  Warns per ICHIPB-less member (existing
           behavior).

        Stores ``rec['fi_bbox']`` (half-open) and ``rec['placement']``.
        """
        all_ichipb = all(
            m['ichipb'] is not None and m['ichipb'].fi_row_scale
            for m in members)
        all_iloc = all(m['ccs'] is not None for m in members)

        if all_ichipb:
            for m in members:
                m['fi_bbox'] = _segment_bbox_from_ichipb(
                    m['ichipb'], m['rows'], m['cols'], 0, 0)
                m['placement'] = 'ichipb'
            return

        if all_iloc:
            min_row = min(m['ccs'][0] for m in members)
            min_col = min(m['ccs'][1] for m in members)
            for m in members:
                r0 = m['ccs'][0] - min_row
                c0 = m['ccs'][1] - min_col
                m['fi_bbox'] = (
                    int(r0), int(r0 + m['rows']),
                    int(c0), int(c0 + m['cols']),
                )
                m['placement'] = 'iloc'
            return

        running_row_off = 0
        for m in members:
            m['fi_bbox'] = _segment_bbox_from_ichipb(
                m['ichipb'], m['rows'], m['cols'],
                running_row_off, 0,
            )
            m['placement'] = (
                'ichipb' if m['ichipb'] is not None
                and m['ichipb'].fi_row_scale else 'stacked')
            running_row_off = m['fi_bbox'][1]

    @staticmethod
    def _select_primary_group(
        groups: List[List[Dict[str, Any]]],
    ) -> int:
        """Pick the primary group for the unified full-image view.

        Score, in priority order: full resolution (SCALE_FACTOR≈1 and
        IMAG≈1), RPC/RSM geolocation presence, total native pixel
        area, then lowest first-segment index for determinism.
        """
        def score(gid_members: Tuple[int, List[Dict[str, Any]]]):
            _, members = gid_members
            full_res = int(
                abs(members[0]['scale'] - 1.0) <= 1e-6
                and abs(members[0]['imag'] - 1.0) <= 1e-6)
            has_geo = int(any(m['has_geo'] for m in members))
            area = sum(m['rows'] * m['cols'] for m in members)
            first_idx = min(m['index'] for m in members)
            return (full_res, has_geo, area, -first_idx)

        return max(enumerate(groups), key=score)[0]

    def _build_metadata_single(self) -> None:
        """Build ``EONITFMetadata`` from a single open ``self.dataset``.

        This is the original ``_load_metadata`` body, factored out so
        the multi-image dispatcher can reuse it for the single-segment
        and pinned paths verbatim.
        """
        rpc = None
        try:
            rpcs_obj = self.dataset.rpcs
            if rpcs_obj is not None:
                rpc = RPCCoefficients.from_rasterio(rpcs_obj)
        except (AttributeError, TypeError):
            pass

        parsed, self.tre_source = self._parse_segment_tres(self.dataset)
        rsm_segments_list = list(parsed.get('RSMPCA') or [])
        rsm_id = parsed.get('RSMIDA')
        csexra = parsed.get('CSEXRA')
        cscrna = parsed.get('CSCRNA')
        use00a = parsed.get('USE00A')
        ichipb = parsed.get('ICHIPB')
        blocka = parsed.get('BLOCKA')
        csepha = parsed.get('CSEPHA')
        rsmgga_list = list(parsed.get('RSMGGA') or [])
        aimidb = parsed.get('AIMIDB')
        stdidc = parsed.get('STDIDC')
        piaimc = parsed.get('PIAIMC')
        extras = self._extract_extra_tres(parsed)

        rsm = None
        rsm_segments = None
        if rsm_segments_list:
            rsm = rsm_segments_list[0]
            nrg = rsm_id.num_row_sections if rsm_id else 1
            ncg = rsm_id.num_col_sections if rsm_id else 1
            seg_dict: Dict[Tuple[int, int], RSMCoefficients] = {}
            for seg in rsm_segments_list:
                seg_dict[(seg.rsn or 1, seg.csn or 1)] = seg
            rsm_segments = RSMSegmentGrid(
                num_row_sections=nrg or 1,
                num_col_sections=ncg or 1,
                segments=seg_dict,
            )

        accuracy = _build_accuracy_info(
            csexra, use00a, rpc,
            rsm_dca=extras['rsm_dca'], rsm_eca=extras['rsm_eca'])
        collection_info = _build_collection_info(aimidb, stdidc, piaimc)
        header = self._extract_header_tags(self.dataset)

        self.metadata = EONITFMetadata(
            format='NITF',
            rows=self.dataset.height,
            cols=self.dataset.width,
            bands=self.dataset.count,
            dtype=str(self.dataset.dtypes[0]),
            crs=str(self.dataset.crs) if self.dataset.crs else None,
            nodata=self.dataset.nodata,
            rpc=rpc,
            rsm=rsm,
            rsm_id=rsm_id,
            rsm_segments=rsm_segments,
            csexra=csexra,
            cscrna=cscrna,
            use00a=use00a,
            ichipb=ichipb,
            blocka=blocka,
            csepha=csepha,
            rsmgga=rsmgga_list[0] if rsmgga_list else None,
            collection_info=collection_info,
            accuracy=accuracy,
            **extras,
            **header,
        )

    @staticmethod
    def _extract_extra_tres(parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Map a TRE-name-keyed bundle to extended metadata kwargs.

        Folds B-variant RSM error TREs into the A-variant fields
        (the dataclasses carry a ``variant`` flag) and derives
        ``band_names`` / ``wavelengths`` from BANDSB when present.
        """
        bandsb = parsed.get('BANDSB')
        band_names = None
        wavelengths = None
        if bandsb is not None:
            try:
                band_names = list(bandsb.band_names) or None
                wavelengths = list(bandsb.wavelengths_um) or None
            except (AttributeError, TypeError):
                pass
        return {
            'rsm_pia': parsed.get('RSMPIA'),
            'rsm_dca': parsed.get('RSMDCA') or parsed.get('RSMDCB'),
            'rsm_eca': parsed.get('RSMECA') or parsed.get('RSMECB'),
            'rsm_apa': parsed.get('RSMAPA') or parsed.get('RSMAPB'),
            'bandsb': bandsb,
            'bandsa': parsed.get('BANDSA'),
            'sensrb': parsed.get('SENSRB'),
            'mensrb': parsed.get('MENSRB'),
            'mensra': parsed.get('MENSRA'),
            'acftb': parsed.get('ACFTB'),
            'band_names': band_names,
            'wavelengths': wavelengths,
        }

    #: TRE names merged generically (first non-None across segments)
    #: by :meth:`_aggregate_tres` in unified mode.
    _EXTRA_TRE_KEYS = (
        'RSMPIA', 'RSMDCA', 'RSMECA', 'RSMAPA',
        'RSMDCB', 'RSMECB', 'RSMAPB',
        'BANDSB', 'BANDSA',
        'SENSRB', 'MENSRB', 'MENSRA', 'ACFTB',
    )

    @staticmethod
    def _extract_header_tags(dataset) -> Dict[str, Any]:
        """Pull NITF header fields (IID1, ICORDS, IDATIM, ...) from tags."""
        tags = dataset.tags() or {}
        abpp_str = tags.get('NITF_ABPP', tags.get('ABPP'))
        return {
            'iid1': tags.get('NITF_IID1', tags.get('IID1')),
            'iid2': tags.get('NITF_IID2', tags.get('IID2')),
            'icords': tags.get('NITF_ICORDS', tags.get('ICORDS')),
            'icat': tags.get('NITF_ICAT', tags.get('ICAT')),
            'abpp': int(abpp_str) if abpp_str else None,
            'idatim': tags.get('NITF_IDATIM', tags.get('IDATIM')),
            'tgtid': tags.get('NITF_TGTID', tags.get('TGTID')),
            'isource': tags.get('NITF_ISOURCE', tags.get('ISOURCE')),
            'igeolo': tags.get('NITF_IGEOLO', tags.get('IGEOLO')),
        }

    def _build_metadata_unified(self) -> None:
        """Build full-image ``EONITFMetadata`` for the primary group.

        - Aggregates RSMPCA/RSMGGA across the *primary group's*
          segments only (a non-primary overview or mask may carry TREs
          for a different coordinate space), deduplicated by
          ``(rsn, csn)`` / ``(ggrsn, ggcsn)``.
        - Single-instance TREs (RSMIDA, CSEXRA, USE00A, BLOCKA, CSEPHA,
          AIMIDB, STDIDC, PIAIMC) take the first non-``None`` across
          primary segments.
        - Resolves full-image dimensions in priority RSMIDA.FULLR/FULLC
          → max ICHIPB.FI_ROW/FI_COL → union of segment fi_row_hi/col_hi,
          over the primary group.
        - Sets ``metadata.ichipb=None`` so geolocators don't double-apply
          the chip-to-full transform — the unified reader is the full
          image.
        - ``metadata.image_segments`` lists *all* segments (primary and
          not) and ``metadata.image_groups`` summarizes every group, so
          heterogeneous content stays discoverable.
        """
        agg = self._aggregate_tres()
        rsm_segments_list = agg['rsm_segments_list']
        rsm_id = agg['rsm_id']
        csexra = agg['csexra']
        cscrna = agg['cscrna']
        use00a = agg['use00a']
        blocka = agg['blocka']
        csepha = agg['csepha']
        rsmgga_list = agg['rsmgga_list']
        aimidb = agg['aimidb']
        stdidc = agg['stdidc']
        piaimc = agg['piaimc']
        extras = self._extract_extra_tres(agg['extra'])

        rsm = rsm_segments_list[0] if rsm_segments_list else None
        rsm_segments = None
        if rsm_segments_list:
            nrg = rsm_id.num_row_sections if rsm_id else 1
            ncg = rsm_id.num_col_sections if rsm_id else 1
            seg_dict: Dict[Tuple[int, int], RSMCoefficients] = {}
            for seg in rsm_segments_list:
                seg_dict[(seg.rsn or 1, seg.csn or 1)] = seg
            rsm_segments = RSMSegmentGrid(
                num_row_sections=nrg or 1,
                num_col_sections=ncg or 1,
                segments=seg_dict,
            )

        full_rows, full_cols = self._resolve_full_dims(
            rsm_id, [s.info for s in self._primary_segments])

        # RPC from the first primary segment (typically present once).
        rpc = None
        try:
            rpcs_obj = self._primary_segments[0].dataset.rpcs
            if rpcs_obj is not None:
                rpc = RPCCoefficients.from_rasterio(rpcs_obj)
        except (AttributeError, TypeError):
            pass

        accuracy = _build_accuracy_info(
            csexra, use00a, rpc,
            rsm_dca=extras['rsm_dca'], rsm_eca=extras['rsm_eca'])
        collection_info = _build_collection_info(aimidb, stdidc, piaimc)
        header = self._extract_header_tags(
            self._primary_segments[0].dataset)

        ds0 = self._primary_segments[0].dataset
        self.metadata = EONITFMetadata(
            format='NITF',
            rows=full_rows,
            cols=full_cols,
            bands=ds0.count,
            dtype=str(ds0.dtypes[0]),
            crs=str(ds0.crs) if ds0.crs else None,
            nodata=ds0.nodata,
            rpc=rpc,
            rsm=rsm,
            rsm_id=rsm_id,
            rsm_segments=rsm_segments,
            csexra=csexra,
            cscrna=cscrna,
            use00a=use00a,
            ichipb=None,
            blocka=blocka,
            csepha=csepha,
            rsmgga=rsmgga_list[0] if rsmgga_list else None,
            collection_info=collection_info,
            accuracy=accuracy,
            image_segments=[s.info for s in self._segments],
            image_groups=list(self._group_infos),
            **extras,
            **header,
        )

    def _aggregate_tres(self) -> Dict[str, Any]:
        """Merge the primary group's cached per-segment TRE bundles.

        Scoped to ``self._primary_segments`` — TREs on non-primary
        segments (overviews, masks, other products) describe *their*
        coordinate space and must not leak into the unified metadata.
        Multi-instance TREs (RSMPCA, RSMGGA) are deduplicated by their
        section keys.  Single-instance TREs take the first non-``None``
        seen.  Sets ``self.tre_source`` to the parser used.
        """
        rsm_segments_list: List[RSMCoefficients] = []
        rsmgga_list: List[RSMGGAMetadata] = []
        seen_rsmpca: set = set()
        seen_rsmgga: set = set()
        rsm_id = csexra = cscrna = use00a = blocka = csepha = None
        aimidb = stdidc = piaimc = None
        extra: Dict[str, Any] = {k: None for k in self._EXTRA_TRE_KEYS}
        any_xml = False
        any_manual = False

        for s in self._primary_segments:
            # Parsed once during _open_all_segments and cached.
            parsed = s.tres or {}
            if s.tre_source == 'xml:TRE':
                any_xml = True
            elif s.tre_source == 'manual':
                any_manual = True

            for r in parsed.get('RSMPCA', []) or []:
                key = (r.rsn or 1, r.csn or 1)
                if key not in seen_rsmpca:
                    seen_rsmpca.add(key)
                    rsm_segments_list.append(r)
            for g in parsed.get('RSMGGA', []) or []:
                key = (g.ggrsn or 1, g.ggcsn or 1)
                if key not in seen_rsmgga:
                    seen_rsmgga.add(key)
                    rsmgga_list.append(g)
            rsm_id = rsm_id or parsed.get('RSMIDA')
            csexra = csexra or parsed.get('CSEXRA')
            cscrna = cscrna or parsed.get('CSCRNA')
            use00a = use00a or parsed.get('USE00A')
            blocka = blocka or parsed.get('BLOCKA')
            csepha = csepha or parsed.get('CSEPHA')
            aimidb = aimidb or parsed.get('AIMIDB')
            stdidc = stdidc or parsed.get('STDIDC')
            piaimc = piaimc or parsed.get('PIAIMC')
            for k in self._EXTRA_TRE_KEYS:
                if extra[k] is None:
                    extra[k] = parsed.get(k)

        if any_xml:
            self.tre_source = 'xml:TRE'
        elif any_manual:
            self.tre_source = 'manual'
        else:
            self.tre_source = 'none'

        return {
            'rsm_segments_list': rsm_segments_list,
            'rsm_id': rsm_id,
            'extra': extra,
            'csexra': csexra,
            'cscrna': cscrna,
            'use00a': use00a,
            'blocka': blocka,
            'csepha': csepha,
            'rsmgga_list': rsmgga_list,
            'aimidb': aimidb,
            'stdidc': stdidc,
            'piaimc': piaimc,
        }

    def _tres_manual_for_dataset(self, ds) -> Dict[str, Any]:
        """Legacy byte-offset parsing for one dataset's TRE namespaces."""
        out: Dict[str, Any] = {
            'RSMPCA': [], 'RSMGGA': [],
            'RSMIDA': None, 'CSEXRA': None, 'CSCRNA': None,
            'USE00A': None,
            'ICHIPB': None, 'BLOCKA': None, 'CSEPHA': None,
            'AIMIDB': None, 'STDIDC': None, 'PIAIMC': None,
            'RSMPIA': None, 'RSMDCA': None, 'RSMECA': None,
            'RSMAPA': None, 'RSMDCB': None, 'RSMECB': None,
            'RSMAPB': None,
            'BANDSB': None, 'BANDSA': None,
            'SENSRB': None, 'MENSRB': None, 'MENSRA': None,
            'ACFTB': None,
        }
        single_parsers = {
            'RSMPIA': parse_rsmpia_cedata,
            'RSMDCA': parse_rsmdca_cedata,
            'RSMECA': parse_rsmeca_cedata,
            'RSMAPA': parse_rsmapa_cedata,
            'RSMDCB': parse_rsmdcb_cedata,
            'RSMECB': parse_rsmecb_cedata,
            'RSMAPB': parse_rsmapb_cedata,
            'BANDSB': parse_bandsb_cedata,
            'BANDSA': parse_bandsa_cedata,
            'SENSRB': parse_sensrb_cedata,
            'MENSRB': parse_mensrb_cedata,
            'MENSRA': parse_mensra_cedata,
            'ACFTB': parse_acftb_cedata,
        }
        try:
            for ns in ds.tag_namespaces():
                tags = ds.tags(ns=ns)
                if not tags:
                    continue
                for key, val in tags.items():
                    ku = key.upper()
                    if 'RSMPCA' in ku:
                        seg = _parse_rsmpca_tre(val)
                        if seg is not None:
                            out['RSMPCA'].append(seg)
                    elif 'RSMIDA' in ku and out['RSMIDA'] is None:
                        out['RSMIDA'] = _parse_rsmida_tre(val)
                    elif 'CSEXRA' in ku and out['CSEXRA'] is None:
                        out['CSEXRA'] = _parse_csexra_tre(val)
                    elif 'CSCRNA' in ku and out['CSCRNA'] is None:
                        out['CSCRNA'] = _parse_cscrna_tre(val)
                    elif 'USE00A' in ku and out['USE00A'] is None:
                        out['USE00A'] = _parse_use00a_tre(val)
                    elif 'ICHIPB' in ku and out['ICHIPB'] is None:
                        out['ICHIPB'] = _parse_ichipb_tre(val)
                    elif 'BLOCKA' in ku and out['BLOCKA'] is None:
                        out['BLOCKA'] = _parse_blocka_tre(val)
                    elif 'CSEPHA' in ku and out['CSEPHA'] is None:
                        out['CSEPHA'] = _parse_csepha_tre(val)
                    elif 'RSMGGA' in ku:
                        gga = _parse_rsmgga_tre(val)
                        if gga is not None:
                            out['RSMGGA'].append(gga)
                    elif 'AIMIDB' in ku and out['AIMIDB'] is None:
                        out['AIMIDB'] = _parse_aimidb_tre(val)
                    elif 'STDIDC' in ku and out['STDIDC'] is None:
                        out['STDIDC'] = _parse_stdidc_tre(val)
                    elif 'PIAIMC' in ku and out['PIAIMC'] is None:
                        out['PIAIMC'] = _parse_piaimc_tre(val)
                    else:
                        for name, parser in single_parsers.items():
                            if name in ku and out[name] is None:
                                out[name] = parser(val)
                                break
        except (AttributeError, TypeError):
            pass
        return out

    @staticmethod
    def _resolve_full_dims(
        rsm_id: Optional[RSMIdentification],
        infos: List[ImageSegmentInfo],
    ) -> Tuple[int, int]:
        """Pick full-image dims from RSMIDA → ICHIPB.FI_ROW/COL → bbox union.

        Warns when authorities disagree by more than 1 pixel.
        """
        candidates: List[Tuple[str, int, int]] = []
        if rsm_id is not None and rsm_id.full_image_rows and rsm_id.full_image_cols:
            candidates.append(
                ('RSMIDA', int(rsm_id.full_image_rows),
                 int(rsm_id.full_image_cols)))
        ichipb_rows = max(
            (i.ichipb.full_image_rows for i in infos
             if i.ichipb is not None and i.ichipb.full_image_rows),
            default=0,
        )
        ichipb_cols = max(
            (i.ichipb.full_image_cols for i in infos
             if i.ichipb is not None and i.ichipb.full_image_cols),
            default=0,
        )
        if ichipb_rows and ichipb_cols:
            candidates.append(('ICHIPB', int(ichipb_rows), int(ichipb_cols)))
        union_rows = max((i.fi_row_hi for i in infos), default=0)
        union_cols = max((i.fi_col_hi for i in infos), default=0)
        candidates.append(('union', int(union_rows), int(union_cols)))

        chosen = candidates[0]
        for src, r, c in candidates[1:]:
            if abs(r - chosen[1]) > 1 or abs(c - chosen[2]) > 1:
                warnings.warn(
                    f"Full-image dimension disagreement: {chosen[0]} "
                    f"reports ({chosen[1]}, {chosen[2]}); {src} reports "
                    f"({r}, {c}).  Using {chosen[0]}.",
                    RuntimeWarning, stacklevel=4,
                )
        return chosen[1], chosen[2]

    @property
    def has_rpc(self) -> bool:
        """Whether RPC coefficients are available."""
        return self.metadata.rpc is not None

    @property
    def has_rsm(self) -> bool:
        """Whether RSM coefficients are available."""
        return self.metadata.rsm is not None

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
        decimation: int = 1,
    ) -> np.ndarray:
        """Read a spatial chip from the EO NITF file.

        Coordinates are full-image pixel indices.  For multi-image NITFs
        opened in unified mode (default), the request is routed across
        whichever *primary-group* image segments overlap the bbox;
        pixels falling in gaps between segments are filled with
        ``dataset.nodata`` (else ``0``).  Non-primary segments
        (overviews, masks) are not stitched — reopen with
        ``image_index=N`` to read them.  For single-image and pinned
        (``image_index``) modes the request goes directly to the
        underlying dataset.

        Parameters
        ----------
        row_start : int
            Starting row index (inclusive).
        row_end : int
            Ending row index (exclusive).
        col_start : int
            Starting column index (inclusive).
        col_end : int
            Ending column index (exclusive).
        bands : Optional[List[int]]
            Band indices to read (0-based). If None, read all bands.
        decimation : int
            Read every ``decimation``-th pixel (1 = full resolution).
            Coordinates stay full-resolution full-image indices; the
            returned chip has ``ceil(n / decimation)`` rows/cols.
            Served, in priority order, from: a matching reduced-
            resolution image segment group (NITF ``IMAG`` overviews —
            discovered automatically at open), a decimated GDAL read
            (``out_shape``; exploits embedded overviews and JPEG2000
            resolution levels) when one dataset covers the request, or
            a full-resolution read sliced ``[::decimation]``.

        Returns
        -------
        np.ndarray
            Image chip with shape ``(rows, cols)`` for single band or
            ``(bands, rows, cols)`` for multi-band.
        """
        if row_start < 0 or col_start < 0:
            raise ValueError("Start indices must be non-negative")
        if row_end > self.metadata.rows or col_end > self.metadata.cols:
            raise ValueError("End indices exceed image dimensions")
        decimation = int(decimation)
        if decimation < 1:
            raise ValueError(
                f"decimation must be >= 1, got {decimation}")

        if decimation > 1:
            return self._read_chip_decimated(
                row_start, row_end, col_start, col_end, bands,
                decimation)
        if self._image_index is not None or not self._segments:
            return self._read_chip_single(
                row_start, row_end, col_start, col_end, bands)
        return self._read_chip_unified(
            row_start, row_end, col_start, col_end, bands)

    def _read_chip_single(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]],
    ) -> np.ndarray:
        """Single-dataset read path (original behavior)."""
        window = Window(
            col_start, row_start,
            col_end - col_start, row_end - row_start,
        )
        data = self._read_window(self.dataset, window, bands)
        if data.shape[0] == 1:
            return data[0]
        return data

    def _read_window(self, ds, window: 'Window',
                     bands: Optional[List[int]]) -> np.ndarray:
        """Read a window from one rasterio dataset honouring read_config.

        Returns ``(bands, rows, cols)`` regardless of band count;
        callers squeeze the band axis when appropriate.
        """
        cfg = self.read_config
        if bands is None:
            band_indices = list(range(1, ds.count + 1))
        else:
            band_indices = [b + 1 for b in bands]

        if cfg.parallel:
            _ensure_gdal_threads(cfg)
            workers = _resolve_workers(cfg)
            n_pixels = int(window.width) * int(window.height)
            if n_pixels >= cfg.chunk_threshold:
                return chunked_parallel_read(ds, window, band_indices, workers)
            if len(band_indices) > 1:
                return parallel_band_read(ds, window, band_indices, workers)
            return ds.read(band_indices, window=window)
        return ds.read(band_indices, window=window)

    def _read_chip_unified(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]],
    ) -> np.ndarray:
        """Stitch a chip from primary-group segments overlapping the bbox.

        Routes only across the primary group — non-primary segments
        (overviews, masks) live in different pixel grids and are
        readable via ``image_index`` pinning instead.
        """
        return self._read_chip_group(
            self._primary_segments,
            row_start, row_end, col_start, col_end, bands)

    def _read_chip_group(
        self,
        segments: List[_OpenSegment],
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]],
    ) -> np.ndarray:
        """Stitch a chip from the given segments' shared pixel grid.

        Coordinates are indices in the *group's* grid (the primary
        full-image grid for unified reads; the reduced-resolution grid
        when serving decimated reads from an overview group).  Pixels
        not covered by any segment are filled with nodata (else 0) —
        no bounds requirement on the request bbox.

        For ``SCALE_FACTOR=1`` (the v1 supported case) the inverse
        ICHIPB collapses to integer translation, so each per-segment
        read is a plain ``Window`` in that segment's local pixels.

        When the request spans multiple segments and
        ``read_config.parallel`` is set, per-segment windows are read
        concurrently on a thread pool (GDAL releases the GIL during
        I/O and decode); each window writes a disjoint region of the
        output, so no synchronization is needed.
        """
        n_rows = row_end - row_start
        n_cols = col_end - col_start
        n_bands = (self.dataset.count if bands is None else len(bands))

        # Allocate output filled with nodata (else 0).
        nodata = self.metadata.nodata
        fill = nodata if nodata is not None else 0
        dtype = np.dtype(self.metadata.dtype)
        out = np.full((n_bands, n_rows, n_cols), fill, dtype=dtype)

        jobs = self._plan_segment_jobs(
            segments, row_start, row_end, col_start, col_end)

        if len(jobs) > 1 and self.read_config.parallel:
            # Fan segments across threads.  Inner reads stay
            # single-threaded (plain ds.read) — the segment pool is
            # the parallelism; nesting chunked reads would
            # oversubscribe cores.
            _ensure_gdal_threads(self.read_config)
            workers = min(len(jobs), _resolve_workers(self.read_config))
            if bands is None:
                band_indices = list(range(1, n_bands + 1))
            else:
                band_indices = [b + 1 for b in bands]

            def _read_one(job):
                ds, window, sl = job
                out[:, sl[0], sl[1]] = ds.read(
                    band_indices, window=window)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(_read_one, jobs))
        else:
            for ds, window, sl in jobs:
                out[:, sl[0], sl[1]] = self._read_window(
                    ds, window, bands)

        if n_bands == 1:
            return out[0]
        return out

    @staticmethod
    def _plan_segment_jobs(
        segments: List[_OpenSegment],
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> List[Tuple[Any, 'Window', Tuple[slice, slice]]]:
        """Plan per-segment read jobs for a group-grid bbox request.

        Returns one ``(dataset, window, (row_slice, col_slice))`` per
        segment overlapping the request: the segment-local ``Window``
        to read and the output-array slices it fills.  Shared by pixel
        reads (:meth:`_read_chip_group`) and validity-mask reads
        (:meth:`read_mask`).
        """
        jobs: List[Tuple[Any, 'Window', Tuple[slice, slice]]] = []
        for seg in segments:
            info = seg.info
            ov_r0 = max(row_start, info.fi_row_lo)
            ov_r1 = min(row_end, info.fi_row_hi)
            ov_c0 = max(col_start, info.fi_col_lo)
            ov_c1 = min(col_end, info.fi_col_hi)
            if ov_r0 >= ov_r1 or ov_c0 >= ov_c1:
                continue

            # Map full-image overlap into segment-local pixels via the
            # inverse of full = fi_off + fi_scale * chip.
            ichipb = info.ichipb
            if ichipb is not None and ichipb.fi_row_scale:
                fi_row_off = float(ichipb.fi_row_off or 0.0)
                fi_col_off = float(ichipb.fi_col_off or 0.0)
                fi_row_scale = float(ichipb.fi_row_scale or 1.0)
                fi_col_scale = float(ichipb.fi_col_scale or 1.0)
                seg_r0 = int(round((ov_r0 - fi_row_off) / fi_row_scale))
                seg_r1 = int(round((ov_r1 - fi_row_off) / fi_row_scale))
                seg_c0 = int(round((ov_c0 - fi_col_off) / fi_col_scale))
                seg_c1 = int(round((ov_c1 - fi_col_off) / fi_col_scale))
            else:
                seg_r0 = ov_r0 - info.fi_row_lo
                seg_r1 = ov_r1 - info.fi_row_lo
                seg_c0 = ov_c0 - info.fi_col_lo
                seg_c1 = ov_c1 - info.fi_col_lo

            # Clamp to segment dims (defensive against off-by-one rounding).
            seg_r0 = max(0, min(info.rows, seg_r0))
            seg_r1 = max(0, min(info.rows, seg_r1))
            seg_c0 = max(0, min(info.cols, seg_c0))
            seg_c1 = max(0, min(info.cols, seg_c1))
            if seg_r0 >= seg_r1 or seg_c0 >= seg_c1:
                continue

            window = Window(
                seg_c0, seg_r0,
                seg_c1 - seg_c0, seg_r1 - seg_r0,
            )
            out_r0 = ov_r0 - row_start
            out_c0 = ov_c0 - col_start
            out_slices = (
                slice(out_r0, out_r0 + (seg_r1 - seg_r0)),
                slice(out_c0, out_c0 + (seg_c1 - seg_c0)),
            )
            jobs.append((seg.dataset, window, out_slices))
        return jobs

    def read_mask(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        decimation: int = 1,
    ) -> np.ndarray:
        """Read the validity mask for a chip region.

        Surfaces GDAL's dataset mask, which reflects NITF pad-pixel /
        blocked-image masks (``IC=NM``, ``M3``, ``M8`` mask tables)
        and the nodata value.  In unified multi-segment mode, pixels
        in gaps between segments are invalid (``0``).

        Parameters
        ----------
        row_start, row_end, col_start, col_end : int
            Full-image bbox, same convention as :meth:`read_chip`.
        decimation : int
            Subsample stride applied to the mask (1 = full).

        Returns
        -------
        np.ndarray
            ``uint8`` mask, shape ``(rows, cols)``; ``255`` = valid,
            ``0`` = invalid or uncovered.
        """
        if row_start < 0 or col_start < 0:
            raise ValueError("Start indices must be non-negative")
        if row_end > self.metadata.rows or col_end > self.metadata.cols:
            raise ValueError("End indices exceed image dimensions")
        decimation = int(decimation)
        if decimation < 1:
            raise ValueError(
                f"decimation must be >= 1, got {decimation}")

        n_rows = row_end - row_start
        n_cols = col_end - col_start

        def _segment_mask(ds, window: 'Window') -> np.ndarray:
            try:
                return ds.read_masks(1, window=window)
            except (AttributeError, TypeError, NotImplementedError):
                # No mask support — covered pixels are valid.
                return np.full(
                    (int(window.height), int(window.width)),
                    255, dtype=np.uint8)

        if self._image_index is not None or not self._segments:
            window = Window(col_start, row_start, n_cols, n_rows)
            mask = _segment_mask(self.dataset, window)
        else:
            mask = np.zeros((n_rows, n_cols), dtype=np.uint8)
            jobs = self._plan_segment_jobs(
                self._primary_segments,
                row_start, row_end, col_start, col_end)
            for ds, window, sl in jobs:
                mask[sl[0], sl[1]] = _segment_mask(ds, window)

        if decimation > 1:
            mask = mask[::decimation, ::decimation]
        return mask

    def get_lut(self, band: int = 0) -> Optional[np.ndarray]:
        """Return the NITF look-up table for a band, if any.

        NITF ``IREP=LU`` (and palette/pseudo-color) segments carry a
        LUT mapping stored pixel indices to display values; GDAL
        exposes it as a color table.

        Parameters
        ----------
        band : int
            0-based band index.  Default 0.

        Returns
        -------
        np.ndarray or None
            ``uint8`` array of shape ``(N, 3)`` (RGB rows indexed by
            stored pixel value), or ``None`` when the band carries no
            LUT.
        """
        try:
            cmap = self.dataset.colormap(band + 1)
        except (AttributeError, ValueError, TypeError):
            return None
        if not cmap:
            return None
        n = max(cmap.keys()) + 1
        lut = np.zeros((n, 3), dtype=np.uint8)
        for idx, rgba in cmap.items():
            lut[idx] = rgba[:3]
        return lut

    def normalize_abpp(self, chip: np.ndarray) -> np.ndarray:
        """Scale a chip to ``[0, 1]`` float32 honoring ABPP bit depth.

        NITF stores the *actual* bits per pixel (ABPP) separately from
        the container width (NBPP) — e.g. 11-bit or 12-bit sensor data
        in a uint16 container.  Naive scaling by the dtype maximum
        makes such imagery appear 16–32× too dark; this helper divides
        by ``2**ABPP - 1`` instead, falling back to the dtype width
        when ABPP is absent.  Floating-point chips are returned as
        float32 unchanged.

        Parameters
        ----------
        chip : np.ndarray
            Pixel data from :meth:`read_chip`.

        Returns
        -------
        np.ndarray
            float32 array scaled to ``[0, 1]`` (integer inputs).
        """
        if np.issubdtype(chip.dtype, np.floating):
            return chip.astype(np.float32, copy=False)
        abpp = self.metadata.abpp
        if abpp is None or abpp <= 0:
            abpp = np.dtype(self.metadata.dtype).itemsize * 8
        scale = float(2 ** abpp - 1)
        return (chip.astype(np.float32) / scale).clip(0.0, 1.0)

    def _find_overview_group(self, decimation: int) -> Optional[ImageGroupInfo]:
        """Find a reduced-resolution group matching ``1/decimation``.

        A usable overview group has the same band count, dtype, and
        ICAT as the primary group, full ICHIPB scale, and an IMAG
        magnification within 0.1% of ``1/decimation``.
        """
        if not self._group_infos:
            return None
        primary = next(
            (g for g in self._group_infos if g.is_primary), None)
        if primary is None:
            return None
        target = 1.0 / float(decimation)
        for g in self._group_infos:
            if g.is_primary:
                continue
            if (g.bands == primary.bands
                    and g.dtype == primary.dtype
                    and (g.icat or '') == (primary.icat or '')
                    and abs(g.scale_factor - 1.0) <= 1e-6
                    and abs(g.imag - target) <= max(1e-6, 1e-3 * target)):
                return g
        return None

    def _read_chip_decimated(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]],
        decimation: int,
    ) -> np.ndarray:
        """Serve a decimated chip via the cheapest available source.

        Priority:

        1. A matching reduced-resolution segment group (NITF ``IMAG``
           overviews) — reads ``1/decimation``-scale pixels directly,
           never touching full-resolution data.
        2. A decimated GDAL read (``out_shape``) when a single dataset
           covers the request — GDAL serves it from embedded overviews
           or JPEG2000 resolution levels when present.
        3. Full-resolution read sliced ``[::decimation]`` (correct but
           no I/O savings) — the universal fallback.
        """
        d = decimation
        out_rows = -(-(row_end - row_start) // d)   # ceil division
        out_cols = -(-(col_end - col_start) // d)

        # 1. Overview group routing (unified mode only).
        if self._image_index is None and self._segments:
            grp = self._find_overview_group(d)
            if grp is not None:
                segs = [s for s in self._segments
                        if s.info.group_id == grp.group_id]
                g_r0 = int(round(row_start / d))
                g_c0 = int(round(col_start / d))
                return self._read_chip_group(
                    segs,
                    g_r0, g_r0 + out_rows,
                    g_c0, g_c0 + out_cols,
                    bands,
                )

        # 2. Single covering dataset → decimated GDAL read.
        ds_direct = None
        window = None
        if self._image_index is not None or not self._segments:
            ds_direct = self.dataset
            window = Window(
                col_start, row_start,
                col_end - col_start, row_end - row_start,
            )
        elif len(self._primary_segments) == 1:
            seg = self._primary_segments[0]
            info = seg.info
            ichipb = info.ichipb
            plain_offset = (
                ichipb is None
                or not ichipb.fi_row_scale
                or (abs(float(ichipb.fi_row_scale) - 1.0) <= 1e-9
                    and abs(float(ichipb.fi_col_scale) - 1.0) <= 1e-9)
            )
            if (plain_offset
                    and info.fi_row_lo <= row_start
                    and info.fi_row_hi >= row_end
                    and info.fi_col_lo <= col_start
                    and info.fi_col_hi >= col_end):
                ds_direct = seg.dataset
                window = Window(
                    col_start - info.fi_col_lo,
                    row_start - info.fi_row_lo,
                    col_end - col_start,
                    row_end - row_start,
                )
        if ds_direct is not None:
            if bands is None:
                band_indices = list(range(1, ds_direct.count + 1))
            else:
                band_indices = [b + 1 for b in bands]
            try:
                data = ds_direct.read(
                    band_indices, window=window,
                    out_shape=(len(band_indices), out_rows, out_cols),
                )
                if data.shape[0] == 1:
                    return data[0]
                return data
            except TypeError:
                # Dataset (or test double) without out_shape support —
                # fall through to the slicing fallback.
                pass

        # 3. Full-resolution read, sliced.
        if self._image_index is not None or not self._segments:
            full = self._read_chip_single(
                row_start, row_end, col_start, col_end, bands)
        else:
            full = self._read_chip_unified(
                row_start, row_end, col_start, col_end, bands)
        return full[..., ::d, ::d]

    def get_shape(self) -> Tuple[int, ...]:
        """Get image dimensions.

        Returns
        -------
        Tuple[int, ...]
            ``(rows, cols)``.
        """
        return (self.metadata.rows, self.metadata.cols)

    def get_dtype(self) -> np.dtype:
        """Get data type.

        Returns
        -------
        np.dtype
            Pixel data type.
        """
        return np.dtype(self.metadata.dtype)

    def close(self) -> None:
        """Close every owned rasterio dataset (single or per-segment)."""
        if hasattr(self, 'dataset') and self.dataset is not None:
            try:
                self.dataset.close()
            except Exception:
                pass
            self.dataset = None
        for seg in getattr(self, '_segments', []):
            try:
                seg.dataset.close()
            except Exception:
                pass
        self._segments = []
        self._primary_segments = []

    @property
    def image_segments(self) -> Optional[List[ImageSegmentInfo]]:
        """Per-segment summaries when this reader is unifying a multi-image NITF.

        ``None`` for single-image files and pinned (``image_index``)
        readers.  Same data as ``self.metadata.image_segments``;
        provided as a property for ergonomic introspection.
        """
        return getattr(self.metadata, 'image_segments', None)

    @property
    def image_groups(self) -> Optional[List[ImageGroupInfo]]:
        """Segment-group summaries for heterogeneous multi-image NITFs.

        One entry per group of compatible segments; the entry with
        ``is_primary=True`` is the group ``read_chip`` and the
        full-image metadata operate in.  Non-primary groups (overviews,
        cloud masks, support imagery) are readable by reopening with
        ``image_index=<segment_index>``.  ``None`` for single-image
        files and pinned readers.
        """
        return getattr(self.metadata, 'image_groups', None)
