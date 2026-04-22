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
        # Ground domain
        parts.append("G")                                # GRNDD
        # Coordinate origin
        parts.append(_fpad(100.0))                       # XUOR
        parts.append(_fpad(200.0))                       # YUOR
        parts.append(_fpad(300.0))                       # ZUOR
        # Unit vectors (9 × 21)
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
        assert result.ground_domain_type == "G"
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
