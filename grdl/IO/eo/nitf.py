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
2026-04-17  Parse CSEPHA and RSMGGA TREs (ephemeris + ground grid).
2026-04-01
"""

# Standard library
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
from grdl.IO.models.eo_nitf import (
    AccuracyInfo,
    BLOCKAMetadata,
    CSEPHAMetadata,
    CSEXRAMetadata,
    CollectionInfo,
    EONITFMetadata,
    ICHIPBMetadata,
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
    chunked_parallel_read,
    parallel_band_read,
)


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

        # --- Rectangular coordinate origin (3 × 21) ---
        xuor_s = read_str(21)       # XUOR
        yuor_s = read_str(21)       # YUOR
        zuor_s = read_str(21)       # ZUOR
        coord_origin = None
        try:
            if xuor_s and yuor_s and zuor_s:
                coord_origin = XYZ(
                    x=float(xuor_s), y=float(yuor_s), z=float(zuor_s))
        except ValueError:
            pass

        # --- Rectangular unit vectors (9 × 21) ---
        uv_vals = []
        for _ in range(9):
            uv_s = read_str(21)
            try:
                uv_vals.append(float(uv_s) if uv_s else 0.0)
            except ValueError:
                uv_vals.append(0.0)
        coord_unit_vectors = np.array(uv_vals, dtype=np.float64).reshape(3, 3)

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

    Image Chip Block defines the affine transform from chip pixel
    coordinates to original full-image coordinates.

    Parameters
    ----------
    tre_value : str
        Raw CEDATA string from GDAL TRE metadata.

    Returns
    -------
    ICHIPBMetadata or None
        Parsed metadata, or None if parsing fails.
    """
    try:
        v = tre_value.strip()
        if len(v) < 216:
            return None

        pos = 0

        def read_str(n: int) -> str:
            nonlocal pos
            s = v[pos:pos + n].strip()
            pos += n
            return s

        def read_float(n: int = 12) -> float:
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
        scale_factor_r = read_float(10)      # SCALE_FACTOR_1 (row)
        scale_factor_c = read_float(10)      # SCALE_FACTOR_2 (col)
        anamorphic_corr = read_float(10)     # ANAMORPHIC_CORR

        # Chip-to-full-image mapping (12 bytes each)
        read_float(12)                       # SCANBLK_NUM
        fi_col_off = read_float(12)          # OP_COL_11
        fi_row_off = read_float(12)          # OP_ROW_11
        read_float(12)                       # OP_COL_12
        read_float(12)                       # OP_ROW_12
        read_float(12)                       # OP_COL_21
        read_float(12)                       # OP_ROW_21
        op_col = read_float(12)              # OP_COL_22
        op_row = read_float(12)              # OP_ROW_22

        # Full-image row/col at (0,0) and (1,1)
        fi_row_scale = read_float(12)        # FI_ROW_11
        fi_col_scale = read_float(12)        # FI_COL_11
        read_float(12)                       # FI_ROW_12
        read_float(12)                       # FI_COL_12
        read_float(12)                       # FI_ROW_21
        read_float(12)                       # FI_COL_21
        read_float(12)                       # FI_ROW_22
        read_float(12)                       # FI_COL_22

        full_image_rows = None
        full_image_cols = None
        try:
            if pos + 16 <= len(v):
                full_image_rows = read_int(8)  # FI_ROW
                full_image_cols = read_int(8)  # FI_COL
        except (ValueError, IndexError):
            pass

        return ICHIPBMetadata(
            xfrm_flag=xfrm_flag,
            scale_factor_r=scale_factor_r,
            scale_factor_c=scale_factor_c,
            anamorphic_corr=anamorphic_corr,
            fi_row_off=fi_row_off,
            fi_col_off=fi_col_off,
            fi_row_scale=fi_row_scale,
            fi_col_scale=fi_col_scale,
            op_row=op_row,
            op_col=op_col,
            full_image_rows=full_image_rows,
            full_image_cols=full_image_cols,
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
        if len(v) < 117:
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
        read_str(2)                          # LAYOVER_ANGLE
        read_str(2)                          # SHADOW_ANGLE
        read_str(16)                         # reserved
        frfc_loc = read_str(21)              # FRLC_LOC
        frlc_loc = read_str(21)              # FRFC_LOC
        lrfc_loc = read_str(21)              # LRLC_LOC
        lrlc_loc = read_str(21)              # LRFC_LOC

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
) -> Optional[AccuracyInfo]:
    """Build aggregated accuracy from best available TRE source.

    Priority: CSEXRA > USE00A > RPC err_bias.

    Parameters
    ----------
    csexra : CSEXRAMetadata or None
    use00a : USE00AMetadata or None
    rpc : RPCCoefficients or None

    Returns
    -------
    AccuracyInfo or None
    """
    if csexra and (csexra.ce90 is not None or csexra.le90 is not None):
        gsd = None
        if csexra.ground_gsd_row is not None and csexra.ground_gsd_col is not None:
            gsd = (csexra.ground_gsd_row + csexra.ground_gsd_col) / 2.0
        elif csexra.ground_gsd_row is not None:
            gsd = csexra.ground_gsd_row
        elif csexra.ground_gsd_col is not None:
            gsd = csexra.ground_gsd_col
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


class EONITFReader(ImageReader):
    """Read electro-optical NITF imagery with RPC/RSM geolocation.

    Opens EO NITF files via rasterio (GDAL NITF driver) and extracts
    RPC00B and RSM TRE metadata in addition to standard imagery.

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

    def __init__(
        self,
        filepath: Union[str, Path],
        read_config: Optional[ReadConfig] = None,
        use_xml_tre: bool = True,
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
        """
        if not _HAS_RASTERIO:
            raise ImportError(
                "rasterio is required for EO NITF reading. "
                "Install with: pip install rasterio  "
                "or: conda install -c conda-forge rasterio"
            )
        self.read_config = read_config or ReadConfig(parallel=True)
        self.use_xml_tre = bool(use_xml_tre)
        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Load EO NITF metadata including RPC, RSM, and context TREs."""
        try:
            self.dataset = rasterio.open(str(self.filepath))
        except Exception as e:
            raise ValueError(
                f"Failed to open EO NITF file: {e}") from e

        # Extract RPC coefficients
        rpc = None
        try:
            rpcs_obj = self.dataset.rpcs
            if rpcs_obj is not None:
                rpc = RPCCoefficients.from_rasterio(rpcs_obj)
        except (AttributeError, TypeError):
            pass

        # Extract TREs.  Default path is GDAL's xml:TRE (parses fields
        # by name).  Fall back to the legacy byte-offset parser when
        # GDAL recognized no TREs in the xml:TRE namespace (rare --
        # happens for vendor-specific TREs not in GDAL's spec table).
        # ``use_xml_tre=False`` forces the legacy path for comparison.
        tre_bundle = None
        self.tre_source = 'manual'
        if self.use_xml_tre:
            tre_bundle = self._load_tres_xml()
            if any(x for x in tre_bundle):
                self.tre_source = 'xml:TRE'
            else:
                tre_bundle = None  # nothing recognized -- try manual
        if tre_bundle is None:
            tre_bundle = self._load_tres_manual()
        (rsm_segments_list, rsm_id, csexra, use00a, ichipb,
         blocka, csepha, rsmgga_list,
         aimidb, stdidc, piaimc) = tre_bundle

        # Build RSM segment grid and backward-compatible single RSM
        rsm = None
        rsm_segments = None
        if rsm_segments_list:
            rsm = rsm_segments_list[0]
            nrg = rsm_id.num_row_sections if rsm_id else 1
            ncg = rsm_id.num_col_sections if rsm_id else 1
            seg_dict: Dict[Tuple[int, int], RSMCoefficients] = {}
            for seg in rsm_segments_list:
                seg_key = (seg.rsn or 1, seg.csn or 1)
                seg_dict[seg_key] = seg
            rsm_segments = RSMSegmentGrid(
                num_row_sections=nrg or 1,
                num_col_sections=ncg or 1,
                segments=seg_dict,
            )

        # Build aggregated metadata
        accuracy = _build_accuracy_info(csexra, use00a, rpc)
        collection_info = _build_collection_info(aimidb, stdidc, piaimc)

        # Extract NITF header fields from tags
        tags = self.dataset.tags() or {}
        iid1 = tags.get('NITF_IID1', tags.get('IID1'))
        iid2 = tags.get('NITF_IID2', tags.get('IID2'))
        icords = tags.get('NITF_ICORDS', tags.get('ICORDS'))
        icat = tags.get('NITF_ICAT', tags.get('ICAT'))
        abpp_str = tags.get('NITF_ABPP', tags.get('ABPP'))
        abpp = int(abpp_str) if abpp_str else None
        idatim = tags.get('NITF_IDATIM', tags.get('IDATIM'))
        tgtid = tags.get('NITF_TGTID', tags.get('TGTID'))
        isource = tags.get('NITF_ISOURCE', tags.get('ISOURCE'))
        igeolo = tags.get('NITF_IGEOLO', tags.get('IGEOLO'))

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
            use00a=use00a,
            ichipb=ichipb,
            blocka=blocka,
            csepha=csepha,
            rsmgga=rsmgga_list[0] if rsmgga_list else None,
            collection_info=collection_info,
            accuracy=accuracy,
            iid1=iid1,
            iid2=iid2,
            icords=icords,
            icat=icat,
            abpp=abpp,
            idatim=idatim,
            tgtid=tgtid,
            isource=isource,
            igeolo=igeolo,
        )

    def _load_tres_manual(
        self,
    ) -> Tuple[
        List[RSMCoefficients],
        Optional[RSMIdentification],
        Optional[CSEXRAMetadata],
        Optional[USE00AMetadata],
        Optional[ICHIPBMetadata],
        Optional[BLOCKAMetadata],
        Optional[CSEPHAMetadata],
        List[RSMGGAMetadata],
        Optional[Dict[str, Any]],
        Optional[Dict[str, Any]],
        Optional[Dict[str, Any]],
    ]:
        """Legacy path: parse TREs from the raw 'TRE' namespace.

        Walks every metadata namespace and dispatches CEDATA strings
        to the byte-offset parsers in this module.  Retained for
        backward compatibility and as a fallback comparison target
        for the xml:TRE path.
        """
        rsm_segments_list: List[RSMCoefficients] = []
        rsm_id = None
        csexra = None
        use00a = None
        ichipb = None
        blocka = None
        csepha = None
        rsmgga_list: List[RSMGGAMetadata] = []
        aimidb = None
        stdidc = None
        piaimc = None

        try:
            namespaces = self.dataset.tag_namespaces()
            for ns in namespaces:
                tags = self.dataset.tags(ns=ns)
                if not tags:
                    continue
                for key, val in tags.items():
                    key_upper = key.upper()
                    if 'RSMPCA' in key_upper:
                        seg = _parse_rsmpca_tre(val)
                        if seg is not None:
                            rsm_segments_list.append(seg)
                    elif 'RSMIDA' in key_upper and rsm_id is None:
                        rsm_id = _parse_rsmida_tre(val)
                    elif 'CSEXRA' in key_upper and csexra is None:
                        csexra = _parse_csexra_tre(val)
                    elif 'USE00A' in key_upper and use00a is None:
                        use00a = _parse_use00a_tre(val)
                    elif 'ICHIPB' in key_upper and ichipb is None:
                        ichipb = _parse_ichipb_tre(val)
                    elif 'BLOCKA' in key_upper and blocka is None:
                        blocka = _parse_blocka_tre(val)
                    elif 'CSEPHA' in key_upper and csepha is None:
                        csepha = _parse_csepha_tre(val)
                    elif 'RSMGGA' in key_upper:
                        gga = _parse_rsmgga_tre(val)
                        if gga is not None:
                            rsmgga_list.append(gga)
                    elif 'AIMIDB' in key_upper and aimidb is None:
                        aimidb = _parse_aimidb_tre(val)
                    elif 'STDIDC' in key_upper and stdidc is None:
                        stdidc = _parse_stdidc_tre(val)
                    elif 'PIAIMC' in key_upper and piaimc is None:
                        piaimc = _parse_piaimc_tre(val)
        except (AttributeError, TypeError):
            pass

        return (rsm_segments_list, rsm_id, csexra, use00a, ichipb,
                blocka, csepha, rsmgga_list, aimidb, stdidc, piaimc)

    def _load_tres_xml(
        self,
    ) -> Tuple[
        List[RSMCoefficients],
        Optional[RSMIdentification],
        Optional[CSEXRAMetadata],
        Optional[USE00AMetadata],
        Optional[ICHIPBMetadata],
        Optional[BLOCKAMetadata],
        Optional[CSEPHAMetadata],
        List[RSMGGAMetadata],
        Optional[Dict[str, Any]],
        Optional[Dict[str, Any]],
        Optional[Dict[str, Any]],
    ]:
        """New path: parse TREs from GDAL's xml:TRE namespace.

        Delegates to :func:`grdl.IO.eo._tre_xml.parse_all_tres`, which
        walks the parsed-XML tree GDAL emits.  Field access by name
        rather than byte offset eliminates the off-by-N bugs in the
        legacy parser, especially for chipped and multi-segment files.
        """
        # Imported lazily so the legacy byte-offset path keeps working
        # when this module's xml dependency tree changes.
        from grdl.IO.eo._tre_xml import parse_all_tres

        parsed = parse_all_tres(self.dataset)

        rsm_segments_list = list(parsed.get('RSMPCA', []))
        rsm_id = parsed.get('RSMIDA')
        csexra = parsed.get('CSEXRA')
        use00a = parsed.get('USE00A')
        ichipb = parsed.get('ICHIPB')
        blocka = parsed.get('BLOCKA')
        csepha = parsed.get('CSEPHA')
        rsmgga_list = list(parsed.get('RSMGGA', []))
        aimidb = parsed.get('AIMIDB')
        stdidc = parsed.get('STDIDC')
        piaimc = parsed.get('PIAIMC')

        return (rsm_segments_list, rsm_id, csexra, use00a, ichipb,
                blocka, csepha, rsmgga_list, aimidb, stdidc, piaimc)

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
    ) -> np.ndarray:
        """Read a spatial chip from the EO NITF file.

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

        window = Window(
            col_start, row_start,
            col_end - col_start, row_end - row_start,
        )

        cfg = self.read_config
        if cfg.parallel:
            _ensure_gdal_threads(cfg)
            workers = _resolve_workers(cfg)
            n_pixels = (row_end - row_start) * (col_end - col_start)

            if bands is None:
                band_indices = list(range(1, self.dataset.count + 1))
            else:
                band_indices = [b + 1 for b in bands]

            # Chunked parallel for large windows
            if n_pixels >= cfg.chunk_threshold:
                data = chunked_parallel_read(
                    self.dataset, window, band_indices, workers)
            elif len(band_indices) > 1:
                # Parallel band read for multi-band
                data = parallel_band_read(
                    self.dataset, window, band_indices, workers)
            else:
                data = self.dataset.read(band_indices, window=window)
        else:
            if bands is None:
                data = self.dataset.read(window=window)
            else:
                data = self.dataset.read(
                    [b + 1 for b in bands], window=window)

        if data.shape[0] == 1:
            return data[0]
        return data

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
        """Close the rasterio dataset."""
        if hasattr(self, 'dataset') and self.dataset is not None:
            self.dataset.close()
            self.dataset = None
