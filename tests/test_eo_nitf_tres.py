# -*- coding: utf-8 -*-
"""
Tests for EO NITF TRE parsers and multi-segment RSM.

Tests parse synthetic CEDATA strings for each TRE parser and verify
all fields are correctly extracted. Tests multi-segment RSM construction,
AccuracyInfo aggregation, and CollectionInfo merging.

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org

Created
-------
2026-04-01

Modified
--------
2026-04-01
"""

import pytest
import numpy as np
from datetime import datetime

from grdl.IO.models.eo_nitf import (
    AccuracyInfo,
    BLOCKAMetadata,
    CSEXRAMetadata,
    CollectionInfo,
    ICHIPBMetadata,
    RPCCoefficients,
    RSMCoefficients,
    RSMIdentification,
    RSMSegmentGrid,
    USE00AMetadata,
)
from grdl.IO.eo.nitf import (
    _parse_rsmpca_tre,
    _parse_rsmida_tre,
    _parse_csexra_tre,
    _parse_use00a_tre,
    _parse_ichipb_tre,
    _parse_blocka_tre,
    _parse_aimidb_tre,
    _parse_stdidc_tre,
    _parse_piaimc_tre,
    _build_accuracy_info,
    _build_collection_info,
)


# ===================================================================
# Helper: Build fixed-width CEDATA strings
# ===================================================================

def _pad(val, width):
    """Right-pad a string value to fixed width."""
    s = str(val)
    return s.ljust(width)[:width]


def _fpad(val, width=21):
    """Right-pad a float value to fixed width.

    Uses scientific notation right-justified to match NITF CEDATA format.
    """
    s = f"{val:+.14E}"
    return s.rjust(width)[:width]


# ===================================================================
# RSMPCA Parser Tests
# ===================================================================


class TestParseRSMPCA:
    """Test _parse_rsmpca_tre with synthetic CEDATA."""

    def _build_rsmpca_cedata(
        self, rsn=1, csn=1, rfep=0.5, cfep=0.3,
        row_off=2048.0, col_off=2048.0,
    ):
        """Build a minimal valid RSMPCA CEDATA string."""
        parts = []
        parts.append(_pad("TEST_IMAGE_ID", 80))       # IID
        parts.append(_pad("V1.0", 40))                 # EDITION
        parts.append(f"{rsn:3d}")                       # RSN
        parts.append(f"{csn:3d}")                       # CSN
        parts.append(_fpad(rfep))                       # RFEP
        parts.append(_fpad(cfep))                       # CFEP
        # Normalization offsets
        parts.append(_fpad(row_off))                    # RNRMO
        parts.append(_fpad(col_off))                    # CNRMO
        parts.append(_fpad(0.01))                       # XNRMO (lon rad)
        parts.append(_fpad(0.02))                       # YNRMO (lat rad)
        parts.append(_fpad(100.0))                      # ZNRMO (height)
        # Normalization scale factors
        parts.append(_fpad(1024.0))                     # RNRMSF
        parts.append(_fpad(1024.0))                     # CNRMSF
        parts.append(_fpad(0.001))                      # XNRMSF
        parts.append(_fpad(0.001))                      # YNRMSF
        parts.append(_fpad(500.0))                      # ZNRMSF
        # Row numerator: order [1,1,1] -> 8 terms
        parts.append("1")                               # RNPWRX
        parts.append("1")                               # RNPWRY
        parts.append("1")                               # RNPWRZ
        parts.append("  8")                             # RNTRMS
        for i in range(8):
            parts.append(_fpad(float(i) * 0.1))
        # Row denominator: constant = 1
        parts.append("0")
        parts.append("0")
        parts.append("0")
        parts.append("  1")
        parts.append(_fpad(1.0))
        # Column numerator: order [1,1,1]
        parts.append("1")
        parts.append("1")
        parts.append("1")
        parts.append("  8")
        for i in range(8):
            parts.append(_fpad(float(i) * 0.2))
        # Column denominator: constant = 1
        parts.append("0")
        parts.append("0")
        parts.append("0")
        parts.append("  1")
        parts.append(_fpad(1.0))
        return ''.join(parts)

    def test_basic_parse(self):
        """Verify basic RSMPCA fields are parsed."""
        cedata = self._build_rsmpca_cedata()
        result = _parse_rsmpca_tre(cedata)
        assert result is not None
        assert isinstance(result, RSMCoefficients)
        assert result.row_off == pytest.approx(2048.0, rel=1e-6)
        assert result.col_off == pytest.approx(2048.0, rel=1e-6)
        assert len(result.row_num_coefs) == 8
        assert len(result.col_num_coefs) == 8

    def test_section_numbers(self):
        """Verify RSN and CSN are now stored."""
        cedata = self._build_rsmpca_cedata(rsn=2, csn=3)
        result = _parse_rsmpca_tre(cedata)
        assert result is not None
        assert result.rsn == 2
        assert result.csn == 3

    def test_fitting_errors(self):
        """Verify RFEP and CFEP are now stored."""
        cedata = self._build_rsmpca_cedata(rfep=0.5, cfep=0.3)
        result = _parse_rsmpca_tre(cedata)
        assert result is not None
        assert result.row_fit_error == pytest.approx(0.5, rel=1e-6)
        assert result.col_fit_error == pytest.approx(0.3, rel=1e-6)

    def test_too_short_returns_none(self):
        """Verify short strings return None."""
        assert _parse_rsmpca_tre("too short") is None

    def test_empty_returns_none(self):
        """Verify empty string returns None."""
        assert _parse_rsmpca_tre("") is None


# ===================================================================
# RSMIDA Parser Tests
# ===================================================================


class TestParseRSMIDA:
    """Test _parse_rsmida_tre with synthetic CEDATA."""

    def _build_rsmida_cedata(self):
        """Build a minimal valid RSMIDA CEDATA string."""
        parts = []
        parts.append(_pad("TEST_IMAGE", 80))            # IID
        parts.append(_pad("V1.0", 40))                  # EDITION
        parts.append(_pad("SENSOR_123", 40))             # ISID
        parts.append(_pad("MY_SENSOR", 40))              # SID
        parts.append(_pad("OPTICAL", 40))                # STID
        # Date/time
        parts.append("2025")                             # YEAR
        parts.append("06")                               # MONTH
        parts.append("15")                               # DAY
        parts.append("14")                               # HOUR
        parts.append("30")                               # MINUTE
        parts.append(_pad("45.123", 9))                  # SECOND
        # Grid dimensions
        parts.append(f"{2:8d}")                          # NRG
        parts.append(f"{3:8d}")                          # NCG
        # Time reference
        parts.append(_fpad(1024.5))                      # TRG
        parts.append(_fpad(2048.5))                      # TCG
        # Ground domain — rectangular so XUOR/YUOR/ZUOR and unit
        # vectors are present per STDI-0002 App U Table 2.
        parts.append("R")                                # GRNDD
        # Coordinate origin (only present when GRNDD='R')
        parts.append(_fpad(100.0))                       # XUOR
        parts.append(_fpad(200.0))                       # YUOR
        parts.append(_fpad(300.0))                       # ZUOR
        # Unit vectors (9 × 21, only when GRNDD='R')
        for i in range(9):
            parts.append(_fpad(float(i + 1)))
        # Ground domain vertices (24 × 21)
        for i in range(24):
            parts.append(_fpad(float(i)))
        # Ground reference point
        parts.append(_fpad(-77.0))                       # GRPX
        parts.append(_fpad(38.0))                        # GRPY
        parts.append(_fpad(150.0))                       # GRPZ
        # Image extent (6 × 8)
        parts.append(f"{4096:8d}")                       # FULLR
        parts.append(f"{4096:8d}")                       # FULLC
        parts.append(f"{0:8d}")                          # MINR
        parts.append(f"{4095:8d}")                       # MAXR
        parts.append(f"{0:8d}")                          # MINC
        parts.append(f"{4095:8d}")                       # MAXC
        return ''.join(parts)

    def test_basic_fields(self):
        """Verify basic RSMIDA fields."""
        cedata = self._build_rsmida_cedata()
        result = _parse_rsmida_tre(cedata)
        assert result is not None
        assert result.sensor_id == "MY_SENSOR"
        assert result.ground_domain_type == "R"
        assert result.num_row_sections == 2
        assert result.num_col_sections == 3

    def test_datetime_parsed(self):
        """Verify collection datetime is now parsed."""
        cedata = self._build_rsmida_cedata()
        result = _parse_rsmida_tre(cedata)
        assert result is not None
        assert result.collection_datetime is not None
        assert result.collection_datetime.year == 2025
        assert result.collection_datetime.month == 6
        assert result.collection_datetime.day == 15
        assert result.collection_datetime.hour == 14
        assert result.collection_datetime.minute == 30

    def test_sensor_id_stored(self):
        """Verify ISID is now stored."""
        cedata = self._build_rsmida_cedata()
        result = _parse_rsmida_tre(cedata)
        assert result is not None
        assert result.image_sensor_id == "SENSOR_123"

    def test_time_ref_stored(self):
        """Verify TRG/TCG are stored."""
        cedata = self._build_rsmida_cedata()
        result = _parse_rsmida_tre(cedata)
        assert result is not None
        assert result.time_ref_row == pytest.approx(1024.5, rel=1e-6)
        assert result.time_ref_col == pytest.approx(2048.5, rel=1e-6)

    def test_coord_origin_stored(self):
        """Verify coordinate origin is stored."""
        cedata = self._build_rsmida_cedata()
        result = _parse_rsmida_tre(cedata)
        assert result is not None
        assert result.coord_origin is not None
        assert result.coord_origin.x == pytest.approx(100.0, rel=1e-6)

    def test_unit_vectors_stored(self):
        """Verify unit vectors are stored as (3,3) array."""
        cedata = self._build_rsmida_cedata()
        result = _parse_rsmida_tre(cedata)
        assert result is not None
        assert result.coord_unit_vectors is not None
        assert result.coord_unit_vectors.shape == (3, 3)

    def test_domain_vertices_stored(self):
        """Verify domain vertices are stored as (8,3) array."""
        cedata = self._build_rsmida_cedata()
        result = _parse_rsmida_tre(cedata)
        assert result is not None
        assert result.ground_domain_vertices is not None
        assert result.ground_domain_vertices.shape == (8, 3)

    def test_image_extent_stored(self):
        """Verify FULLR/FULLC/MIN/MAX are stored."""
        cedata = self._build_rsmida_cedata()
        result = _parse_rsmida_tre(cedata)
        assert result is not None
        assert result.full_image_rows == 4096
        assert result.full_image_cols == 4096
        assert result.min_row == 0
        assert result.max_row == 4095

    def test_too_short_returns_none(self):
        """Verify short strings return None."""
        assert _parse_rsmida_tre("too short") is None

    def _build_rsmida_geodetic(self):
        """Build a GRNDD='G' RSMIDA where rectangular fields are absent.

        Per STDI-0002 App U Table 2, when GRNDD='G' the XUOR/YUOR/ZUOR
        and 9 unit-vector fields are not present in the CEDATA.
        """
        parts = []
        parts.append(_pad("TEST_GEODETIC", 80))         # IID
        parts.append(_pad("V1.0", 40))                  # EDITION
        parts.append(_pad("S_ID", 40))                   # ISID
        parts.append(_pad("SID", 40))                    # SID
        parts.append(_pad("STID", 40))                   # STID
        parts.append("2026")                             # YEAR
        parts.append("04")                               # MONTH
        parts.append("29")                               # DAY
        parts.append("12")                               # HOUR
        parts.append("00")                               # MINUTE
        parts.append(_pad("00.0", 9))                    # SECOND
        parts.append(f"{1:8d}")                          # NRG
        parts.append(f"{1:8d}")                          # NCG
        parts.append(_fpad(0.0))                         # TRG
        parts.append(_fpad(0.0))                         # TCG
        parts.append("G")                                # GRNDD
        # Rectangular fields OMITTED for GRNDD='G'
        # Ground domain vertices (24 × 21)
        for i in range(24):
            parts.append(_fpad(float(i)))
        # Ground reference point (lon-rad, lat-rad, hae-m for 'G')
        parts.append(_fpad(np.deg2rad(-77.0)))           # GRPX
        parts.append(_fpad(np.deg2rad(38.0)))            # GRPY
        parts.append(_fpad(150.0))                       # GRPZ
        # Image extent
        parts.append(f"{4096:8d}")                       # FULLR
        parts.append(f"{4096:8d}")                       # FULLC
        parts.append(f"{0:8d}")                          # MINR
        parts.append(f"{4095:8d}")                       # MAXR
        parts.append(f"{0:8d}")                          # MINC
        parts.append(f"{4095:8d}")                       # MAXC
        return ''.join(parts)

    def test_geodetic_skips_rectangular_fields(self):
        """GRNDD='G' RSMIDA: XUOR/unit vectors absent, downstream fields
        still parse correctly.

        Regression for legacy parser bug where rectangular fields were
        read unconditionally, shifting every subsequent field by 252
        bytes for the GRNDD='G' case (the most common commercial form).
        """
        cedata = self._build_rsmida_geodetic()
        result = _parse_rsmida_tre(cedata)
        assert result is not None
        assert result.ground_domain_type == "G"
        # Rectangular fields are absent → parser leaves them at defaults
        assert result.coord_origin is None
        # Image extent must still parse correctly (regression: legacy
        # parser would read garbage here when GRNDD='G').
        assert result.full_image_rows == 4096
        assert result.full_image_cols == 4096
        assert result.min_row == 0
        assert result.max_row == 4095


# ===================================================================
# New TRE Parser Tests
# ===================================================================


class TestParseCSEXRA:
    """Test _parse_csexra_tre."""

    def test_returns_none_for_short(self):
        assert _parse_csexra_tre("short") is None

    def test_returns_none_for_empty(self):
        assert _parse_csexra_tre("") is None


class TestParseUSE00A:
    """Test _parse_use00a_tre."""

    def test_returns_none_for_short(self):
        assert _parse_use00a_tre("short") is None


class TestParseICHIPB:
    """Test _parse_ichipb_tre."""

    def test_returns_none_for_short(self):
        assert _parse_ichipb_tre("short") is None

    @staticmethod
    def _build_ichipb(
        op_row_11=0.5, op_col_11=0.5,
        op_row_22=1023.5, op_col_22=1023.5,
        fi_row_11=10000.5, fi_col_11=20000.5,
        fi_row_22=11023.5, fi_col_22=21023.5,
        scale_factor=1.0, anamorph=0,
        full_rows=30000, full_cols=30000,
        xfrm_flag=0, scanblk_num=0,
    ):
        """Build a 224-byte ICHIPB CEDATA string per STDI-0002 App B.

        Defaults model a 1024×1024 chip pulled out of a 30000×30000
        source starting at (10000, 20000) with no decimation, no
        anamorphic correction, and ``XFRM_FLAG=00`` (non-dewarped
        data provided per Table B-2).
        """
        def f(v):
            # 12-byte field: "+1.12345E+04" packs 5 decimal digits.
            return f"{v:+.5E}"[:12]

        parts = []
        parts.append(f"{int(xfrm_flag):02d}")                  # XFRM_FLAG (2)
        parts.append(f"{scale_factor:+.3E}"[:10])              # SCALE_FACTOR (10)
        parts.append(f"{int(anamorph):02d}")                   # ANAMRPH_CORR (2)
        parts.append(f"{int(scanblk_num):02d}")                # SCANBLK_NUM (2)
        # OP corners (row, col, row, col, ...)
        parts.append(f(op_row_11))
        parts.append(f(op_col_11))
        parts.append(f(op_row_11))                             # OP_ROW_12 = OP_ROW_11
        parts.append(f(op_col_22))                             # OP_COL_12 = OP_COL_22
        parts.append(f(op_row_22))                             # OP_ROW_21
        parts.append(f(op_col_11))                             # OP_COL_21
        parts.append(f(op_row_22))
        parts.append(f(op_col_22))
        # FI corners (axis-aligned chip)
        parts.append(f(fi_row_11))
        parts.append(f(fi_col_11))
        parts.append(f(fi_row_11))
        parts.append(f(fi_col_22))
        parts.append(f(fi_row_22))
        parts.append(f(fi_col_11))
        parts.append(f(fi_row_22))
        parts.append(f(fi_col_22))
        # Optional FI_ROW / FI_COL (8 bytes each)
        parts.append(f"{full_rows:8d}")
        parts.append(f"{full_cols:8d}")
        return ''.join(parts)

    def test_axis_aligned_chip_no_decimation(self):
        """1024×1024 chip from (10000, 20000): full = chip + offset."""
        cedata = self._build_ichipb(
            op_row_11=0.5, op_col_11=0.5,
            op_row_22=1023.5, op_col_22=1023.5,
            fi_row_11=10000.5, fi_col_11=20000.5,
            fi_row_22=11023.5, fi_col_22=21023.5,
        )
        result = _parse_ichipb_tre(cedata)
        assert result is not None
        # Affine: full = off + scale * chip
        assert result.fi_row_scale == pytest.approx(1.0, abs=1e-6)
        assert result.fi_col_scale == pytest.approx(1.0, abs=1e-6)
        assert result.fi_row_off == pytest.approx(10000.0, abs=1e-3)
        assert result.fi_col_off == pytest.approx(20000.0, abs=1e-3)
        # Round-trip check at chip center
        chip_r, chip_c = 511.5, 511.5
        full_r = result.fi_row_off + result.fi_row_scale * chip_r
        full_c = result.fi_col_off + result.fi_col_scale * chip_c
        assert full_r == pytest.approx(10511.5, abs=1e-3)
        assert full_c == pytest.approx(20511.5, abs=1e-3)

    def test_decimated_chip(self):
        """4× decimated chip: 256×256 chip from a 1024×1024 source area."""
        cedata = self._build_ichipb(
            op_row_11=0.5, op_col_11=0.5,
            op_row_22=255.5, op_col_22=255.5,
            fi_row_11=5000.0, fi_col_11=5000.0,
            fi_row_22=6020.0, fi_col_22=6020.0,
            scale_factor=4.0,
        )
        result = _parse_ichipb_tre(cedata)
        assert result is not None
        # Each chip pixel covers 4 source pixels: scale ≈ 4.0
        assert result.fi_row_scale == pytest.approx(4.0, abs=1e-3)
        assert result.fi_col_scale == pytest.approx(4.0, abs=1e-3)
        # off = FI_11 - scale * OP_11 = 5000 - 4*0.5 = 4998
        assert result.fi_row_off == pytest.approx(4998.0, abs=1e-2)
        assert result.fi_col_off == pytest.approx(4998.0, abs=1e-2)

    def test_xfrm_flag_zero_uses_corner_data(self):
        """XFRM_FLAG=00 declares the OP/FI corners are valid (per Table B-2).

        Per STDI-0002 Vol 1 App B Table B-2, ``XFRM_FLAG=00`` means
        the chip carries non-dewarped data and the corner fields are
        populated.  The derived affine should reflect the corners.
        """
        cedata = self._build_ichipb(xfrm_flag=0)
        result = _parse_ichipb_tre(cedata)
        assert result is not None
        assert result.xfrm_flag == 0
        assert result.is_no_transform_provided is False
        # Default fixture has FI_11=(10000.5, 20000.5), corners 1024 wide
        assert result.fi_row_scale == pytest.approx(1.0, abs=1e-3)
        assert result.fi_col_scale == pytest.approx(1.0, abs=1e-3)
        assert result.fi_row_off == pytest.approx(10000.0, abs=1e-2)
        assert result.fi_col_off == pytest.approx(20000.0, abs=1e-2)

    def test_xfrm_flag_one_suppresses_affine(self):
        """XFRM_FLAG=01 means zero-fill; derived affine left None.

        Per Table B-2 / B.7, when ``XFRM_FLAG=01`` the OP/FI corner
        fields are populated with the designated zero-fill defaults
        and consumers must not derive a transform from them.
        """
        cedata = self._build_ichipb(
            xfrm_flag=1,
            op_row_11=0.0, op_col_11=0.0,
            op_row_22=0.0, op_col_22=0.0,
            fi_row_11=0.0, fi_col_11=0.0,
            fi_row_22=0.0, fi_col_22=0.0,
        )
        result = _parse_ichipb_tre(cedata)
        assert result is not None
        assert result.xfrm_flag == 1
        assert result.is_no_transform_provided is True
        assert result.fi_row_off is None
        assert result.fi_col_off is None
        assert result.fi_row_scale is None
        assert result.fi_col_scale is None
        assert result.chip_to_full_affine() is None

    def test_full_image_dims_parsed(self):
        cedata = self._build_ichipb(full_rows=30000, full_cols=25000)
        result = _parse_ichipb_tre(cedata)
        assert result is not None
        assert result.full_image_rows == 30000
        assert result.full_image_cols == 25000

    def test_full_image_unknown_when_zero(self):
        """FI_ROW=0 / FI_COL=0 carry the spec's 'unknown' sentinel."""
        cedata = self._build_ichipb(full_rows=0, full_cols=0)
        result = _parse_ichipb_tre(cedata)
        assert result is not None
        # Per B.8.2 the spec value 00000000 means the chipping app
        # didn't have access to full-image dimensions.
        assert result.full_image_rows is None
        assert result.full_image_cols is None
        assert result.has_full_image_size is False

    def test_all_eight_op_corners_stored(self):
        """All four OP corners are stored verbatim (Table B-2)."""
        cedata = self._build_ichipb(
            op_row_11=0.5, op_col_11=0.5,
            op_row_22=255.5, op_col_22=255.5,
        )
        result = _parse_ichipb_tre(cedata)
        assert result is not None
        # 11 = upper-left, 12 = upper-right, 21 = lower-left, 22 = lower-right
        assert result.op_row_11 == pytest.approx(0.5, abs=1e-3)
        assert result.op_col_11 == pytest.approx(0.5, abs=1e-3)
        assert result.op_row_12 == pytest.approx(0.5, abs=1e-3)
        assert result.op_col_12 == pytest.approx(255.5, abs=1e-3)
        assert result.op_row_21 == pytest.approx(255.5, abs=1e-3)
        assert result.op_col_21 == pytest.approx(0.5, abs=1e-3)
        assert result.op_row_22 == pytest.approx(255.5, abs=1e-3)
        assert result.op_col_22 == pytest.approx(255.5, abs=1e-3)

    def test_all_eight_fi_corners_stored(self):
        """All four FI corners are stored verbatim (Table B-2)."""
        cedata = self._build_ichipb(
            fi_row_11=10000.5, fi_col_11=20000.5,
            fi_row_22=11023.5, fi_col_22=21023.5,
        )
        result = _parse_ichipb_tre(cedata)
        assert result is not None
        assert result.fi_row_11 == pytest.approx(10000.5, abs=1e-2)
        assert result.fi_col_11 == pytest.approx(20000.5, abs=1e-2)
        assert result.fi_row_12 == pytest.approx(10000.5, abs=1e-2)
        assert result.fi_col_12 == pytest.approx(21023.5, abs=1e-2)
        assert result.fi_row_21 == pytest.approx(11023.5, abs=1e-2)
        assert result.fi_col_21 == pytest.approx(20000.5, abs=1e-2)
        assert result.fi_row_22 == pytest.approx(11023.5, abs=1e-2)
        assert result.fi_col_22 == pytest.approx(21023.5, abs=1e-2)

    def test_scanblk_num_stored(self):
        """SCANBLK_NUM (0..99) is required and stored.

        Per Table B-2 / B-3, identifies the source scan block when
        chipping from imagery with multiple scan blocks.
        """
        cedata = self._build_ichipb(scanblk_num=7)
        result = _parse_ichipb_tre(cedata)
        assert result is not None
        assert result.scanblk_num == 7

    def test_anamorphic_corr_flag(self):
        """ANAMRPH_CORR is a binary flag (00 or 01) per spec."""
        cedata_off = self._build_ichipb(anamorph=0)
        result_off = _parse_ichipb_tre(cedata_off)
        assert result_off is not None
        assert result_off.anamorphic_corr == 0
        assert result_off.has_anamorphic_correction is False

        cedata_on = self._build_ichipb(anamorph=1)
        result_on = _parse_ichipb_tre(cedata_on)
        assert result_on is not None
        assert result_on.anamorphic_corr == 1
        assert result_on.has_anamorphic_correction is True

    def test_chip_to_full_affine_axis_aligned(self):
        """4-corner affine recovers the same separable transform.

        For axis-aligned chips, the 2D-affine fit must match the
        derived axial fields.
        """
        cedata = self._build_ichipb(
            op_row_11=0.5, op_col_11=0.5,
            op_row_22=1023.5, op_col_22=1023.5,
            fi_row_11=10000.5, fi_col_11=20000.5,
            fi_row_22=11023.5, fi_col_22=21023.5,
        )
        result = _parse_ichipb_tre(cedata)
        assert result is not None
        m, b = result.chip_to_full_affine()
        # Diagonal matrix (axis-aligned): off-diagonals ~ 0
        assert abs(m[0, 1]) < 1e-3
        assert abs(m[1, 0]) < 1e-3
        # Diagonal entries match the axial scale
        assert m[0, 0] == pytest.approx(result.fi_row_scale, rel=1e-3)
        assert m[1, 1] == pytest.approx(result.fi_col_scale, rel=1e-3)
        # Offsets match
        assert b[0] == pytest.approx(result.fi_row_off, abs=1e-2)
        assert b[1] == pytest.approx(result.fi_col_off, abs=1e-2)

    def test_chip_to_full_affine_rotated(self):
        """Rotated chip case (Annex A.3): 2D-affine recovers rotation.

        Non-axis-aligned corners produce a non-diagonal M; the
        separable axial fields no longer fully describe the mapping
        but ``chip_to_full_affine`` does.
        """
        # Build corners where chip is rotated 90° clockwise inside FI.
        # Chip OP corners: 11=upper-left, 12=upper-right, 21=lower-left,
        # 22=lower-right.  After 90° CW rotation in the FI frame,
        # those map to: 11→top-right of FI rect, 12→bottom-right,
        # 21→top-left, 22→bottom-left.
        cedata = self._build_ichipb_full(
            op_row_11=0.5, op_col_11=0.5,
            op_row_12=0.5, op_col_12=99.5,
            op_row_21=99.5, op_col_21=0.5,
            op_row_22=99.5, op_col_22=99.5,
            fi_row_11=1000.0, fi_col_11=2099.0,
            fi_row_12=1099.0, fi_col_12=2099.0,
            fi_row_21=1000.0, fi_col_21=2000.0,
            fi_row_22=1099.0, fi_col_22=2000.0,
        )
        result = _parse_ichipb_tre(cedata)
        assert result is not None
        m, b = result.chip_to_full_affine()
        # 90° CW rotation: M ≈ [[0, 1], [-1, 0]] (after sign normalisation)
        # off-diagonals should be ~±1, diagonals ~0
        assert abs(m[0, 0]) < 0.05
        assert abs(m[1, 1]) < 0.05
        assert abs(abs(m[0, 1]) - 1.0) < 0.05
        assert abs(abs(m[1, 0]) - 1.0) < 0.05

        # Verify round-trip on a known corner: chip OP_11 → FI corner.
        m_inv, b_inv = result.full_to_chip_affine()
        assert m_inv is not None
        op11_round = m_inv @ np.array([1000.0, 2099.0]) + b_inv
        assert op11_round[0] == pytest.approx(0.5, abs=0.5)
        assert op11_round[1] == pytest.approx(0.5, abs=0.5)

    def test_legacy_parser_reads_all_22_fields(self):
        """The legacy byte-offset parser populates all 22 spec fields."""
        cedata = self._build_ichipb(
            scanblk_num=3, anamorph=1,
            op_row_11=0.5, op_col_11=0.5,
            op_row_22=255.5, op_col_22=255.5,
            fi_row_11=5000.0, fi_col_11=5000.0,
            fi_row_22=5255.0, fi_col_22=5255.0,
            full_rows=10000, full_cols=10000,
        )
        # Legacy parser must accept the 224-byte fixture (same width as XML
        # parser sees, since GDAL's CEDATA round-trips byte-exact).
        result = _parse_ichipb_tre(cedata)
        assert result is not None
        # Every spec field is non-None.
        for fname in (
            'xfrm_flag', 'scale_factor', 'anamorphic_corr', 'scanblk_num',
            'op_row_11', 'op_col_11', 'op_row_12', 'op_col_12',
            'op_row_21', 'op_col_21', 'op_row_22', 'op_col_22',
            'fi_row_11', 'fi_col_11', 'fi_row_12', 'fi_col_12',
            'fi_row_21', 'fi_col_21', 'fi_row_22', 'fi_col_22',
            'full_image_rows', 'full_image_cols',
        ):
            assert getattr(result, fname) is not None, (
                f"spec field {fname} not populated by legacy parser")
        assert result.scanblk_num == 3
        assert result.anamorphic_corr == 1

    @staticmethod
    def _build_ichipb_full(
        op_row_11, op_col_11, op_row_12, op_col_12,
        op_row_21, op_col_21, op_row_22, op_col_22,
        fi_row_11, fi_col_11, fi_row_12, fi_col_12,
        fi_row_21, fi_col_21, fi_row_22, fi_col_22,
        scale_factor=1.0, anamorph=0, scanblk_num=0,
        full_rows=10000, full_cols=10000, xfrm_flag=0,
    ):
        """Build a 224-byte ICHIPB CEDATA with all four corners free."""
        def f(v):
            return f"{v:+.5E}"[:12]

        parts = []
        parts.append(f"{xfrm_flag:02d}")
        parts.append(f"{scale_factor:+.3E}"[:10])
        parts.append(f"{int(anamorph):02d}")
        parts.append(f"{int(scanblk_num):02d}")
        parts.append(f(op_row_11))
        parts.append(f(op_col_11))
        parts.append(f(op_row_12))
        parts.append(f(op_col_12))
        parts.append(f(op_row_21))
        parts.append(f(op_col_21))
        parts.append(f(op_row_22))
        parts.append(f(op_col_22))
        parts.append(f(fi_row_11))
        parts.append(f(fi_col_11))
        parts.append(f(fi_row_12))
        parts.append(f(fi_col_12))
        parts.append(f(fi_row_21))
        parts.append(f(fi_col_21))
        parts.append(f(fi_row_22))
        parts.append(f(fi_col_22))
        parts.append(f"{full_rows:8d}")
        parts.append(f"{full_cols:8d}")
        return ''.join(parts)


class TestParseBLOCKA:
    """Test _parse_blocka_tre."""

    def test_returns_none_for_short(self):
        assert _parse_blocka_tre("short") is None


class TestParseAIMIDB:
    """Test _parse_aimidb_tre."""

    def test_returns_none_for_short(self):
        assert _parse_aimidb_tre("short") is None


class TestParseSTDIDC:
    """Test _parse_stdidc_tre."""

    def test_returns_none_for_short(self):
        assert _parse_stdidc_tre("short") is None


class TestParsePIAIMC:
    """Test _parse_piaimc_tre."""

    def test_returns_none_for_short(self):
        assert _parse_piaimc_tre("short") is None


# ===================================================================
# RSMSegmentGrid Tests
# ===================================================================


class TestRSMSegmentGrid:
    """Test RSMSegmentGrid construction."""

    def test_single_segment(self):
        seg = RSMCoefficients(rsn=1, csn=1)
        grid = RSMSegmentGrid(
            num_row_sections=1,
            num_col_sections=1,
            segments={(1, 1): seg},
        )
        assert len(grid.segments) == 1
        assert (1, 1) in grid.segments

    def test_multi_segment_2x2(self):
        segs = {}
        for r in range(1, 3):
            for c in range(1, 3):
                segs[(r, c)] = RSMCoefficients(
                    rsn=r, csn=c,
                    row_off=float(r * 1000),
                    col_off=float(c * 1000),
                    row_norm_sf=500.0,
                    col_norm_sf=500.0,
                )
        grid = RSMSegmentGrid(
            num_row_sections=2,
            num_col_sections=2,
            segments=segs,
        )
        assert len(grid.segments) == 4
        assert grid.num_row_sections == 2
        assert grid.num_col_sections == 2


# ===================================================================
# AccuracyInfo Aggregation Tests
# ===================================================================


class TestBuildAccuracyInfo:
    """Test _build_accuracy_info priority logic."""

    def test_csexra_preferred(self):
        csexra = CSEXRAMetadata(ce90=5.0, le90=3.0,
                                ground_gsd_row=0.5, ground_gsd_col=0.6)
        use00a = USE00AMetadata(mean_gsd=1.0)
        rpc = RPCCoefficients(err_bias=10.0)
        result = _build_accuracy_info(csexra, use00a, rpc)
        assert result is not None
        assert result.source == 'CSEXRA'
        assert result.ce90 == 5.0
        assert result.le90 == 3.0
        assert result.mean_gsd == pytest.approx(0.55, rel=1e-6)

    def test_use00a_fallback(self):
        use00a = USE00AMetadata(mean_gsd=0.8)
        result = _build_accuracy_info(None, use00a, None)
        assert result is not None
        assert result.source == 'USE00A'
        assert result.mean_gsd == 0.8

    def test_rpc_fallback(self):
        rpc = RPCCoefficients(err_bias=7.5)
        result = _build_accuracy_info(None, None, rpc)
        assert result is not None
        assert result.source == 'RPC00B'
        assert result.ce90 == 7.5

    def test_none_when_nothing(self):
        result = _build_accuracy_info(None, None, None)
        assert result is None


# ===================================================================
# CollectionInfo Aggregation Tests
# ===================================================================


class TestBuildCollectionInfo:
    """Test _build_collection_info merge logic."""

    def test_aimidb_priority(self):
        aimidb = {
            'collection_datetime': datetime(2025, 6, 15, 14, 30),
            'mission_id': 'M001',
            'country_code': 'US',
        }
        stdidc = {
            'collection_datetime': datetime(2025, 1, 1),
            'mission_id': 'M002',
            'country_code': 'UK',
        }
        result = _build_collection_info(aimidb, stdidc, None)
        assert result is not None
        assert result.mission_id == 'M001'
        assert result.country_code == 'US'
        assert result.collection_datetime.year == 2025
        assert result.collection_datetime.month == 6

    def test_stdidc_fills_gaps(self):
        stdidc = {
            'collection_datetime': datetime(2024, 3, 1),
            'mission_id': 'M999',
            'country_code': 'DE',
        }
        result = _build_collection_info(None, stdidc, None)
        assert result is not None
        assert result.mission_id == 'M999'

    def test_piaimc_sensor_mode(self):
        piaimc = {'sensor_mode': 'PAN', 'cloud_cover': 15.0}
        result = _build_collection_info(None, None, piaimc)
        assert result is not None
        assert result.sensor_mode == 'PAN'
        assert result.cloud_cover == 15.0

    def test_none_when_nothing(self):
        result = _build_collection_info(None, None, None)
        assert result is None
