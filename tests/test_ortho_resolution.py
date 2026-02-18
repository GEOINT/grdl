# -*- coding: utf-8 -*-
"""
Resolution Computation Tests - Tests for ortho output resolution strategies.

Dependencies
------------
pytest

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
2026-02-17

Modified
--------
2026-02-17
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from grdl.image_processing.ortho.resolution import (
    compute_output_resolution,
    _resolution_from_sicd,
    _resolution_from_biomass,
    _meters_to_degrees,
    _get_center_latitude,
)


# ---------------------------------------------------------------------------
# Minimal SICD metadata stubs
# ---------------------------------------------------------------------------

@dataclass
class StubLatLonHAE:
    lat: float = 34.0
    lon: float = -118.0
    hae: float = 0.0


@dataclass
class StubSCP:
    llh: Optional[StubLatLonHAE] = None


@dataclass
class StubGeoData:
    scp: Optional[StubSCP] = None


@dataclass
class StubDirParam:
    ss: Optional[float] = None
    imp_resp_wid: Optional[float] = None


@dataclass
class StubGrid:
    image_plane: Optional[str] = None
    type: Optional[str] = None
    row: Optional[StubDirParam] = None
    col: Optional[StubDirParam] = None


@dataclass
class StubScpCoa:
    graze_ang: Optional[float] = None


class StubSICDMetadata:
    """Mimics SICDMetadata enough for resolution tests."""

    def __init__(self, grid=None, scpcoa=None, geo_data=None):
        self.grid = grid
        self.scpcoa = scpcoa
        self.geo_data = geo_data
        # Required for isinstance check
        self.__class__.__module__ = 'grdl.IO.models.sicd'
        self.__class__.__qualname__ = 'SICDMetadata'


# Patch isinstance to accept our stub
import grdl.IO.models.sicd as _sicd_mod
_OrigSICDMeta = _sicd_mod.SICDMetadata


# ---------------------------------------------------------------------------
# Tests: _meters_to_degrees
# ---------------------------------------------------------------------------

class TestMetersToDegreesHelper:

    def test_equator(self):
        """At equator, lat and lon spacing are equal."""
        lat_deg, lon_deg = _meters_to_degrees(111.32, 0.0)
        assert abs(lat_deg - lon_deg) < 1e-6

    def test_high_latitude(self):
        """At 60 degrees, lon spacing should be ~2x lat spacing."""
        lat_deg, lon_deg = _meters_to_degrees(100.0, 60.0)
        assert lon_deg > lat_deg * 1.8
        assert lon_deg < lat_deg * 2.2


# ---------------------------------------------------------------------------
# Tests: SICD resolution
# ---------------------------------------------------------------------------

class TestResolutionSICD:

    def _make_sicd_meta(
        self, row_ss=0.5, col_ss=0.5, row_irw=None, col_irw=None,
        image_plane='SLANT', graze_ang=45.0, scp_lat=34.0,
    ):
        """Build a minimal SICD metadata stub."""
        grid = StubGrid(
            image_plane=image_plane,
            row=StubDirParam(ss=row_ss, imp_resp_wid=row_irw),
            col=StubDirParam(ss=col_ss, imp_resp_wid=col_irw),
        )
        scpcoa = StubScpCoa(graze_ang=graze_ang)
        geo_data = StubGeoData(
            scp=StubSCP(llh=StubLatLonHAE(lat=scp_lat))
        )
        return _OrigSICDMeta(
            format='SICD', rows=100, cols=100, dtype='complex64',
            grid=grid, scpcoa=scpcoa, geo_data=geo_data,
        )

    def test_slant_plane_graze_correction(self):
        """Slant-plane data should be corrected by graze angle."""
        meta = self._make_sicd_meta(
            row_ss=0.5, col_ss=0.5, image_plane='SLANT', graze_ang=45.0
        )
        lat_deg, lon_deg = _resolution_from_sicd(meta, None, 1.0)
        # At 45 deg graze, range ground resolution = 0.5/sin(45) ~ 0.707 m
        # Column stays 0.5 m, so max is 0.707 m
        expected_m = 0.5 / np.sin(np.radians(45.0))
        expected_lat = expected_m / 111320.0
        assert abs(lat_deg - expected_lat) < 1e-10

    def test_ground_plane_no_graze(self):
        """Ground-plane data should not apply graze correction."""
        meta = self._make_sicd_meta(
            row_ss=1.0, col_ss=1.0, image_plane='GROUND', graze_ang=45.0
        )
        lat_deg, lon_deg = _resolution_from_sicd(meta, None, 1.0)
        expected_lat = 1.0 / 111320.0
        assert abs(lat_deg - expected_lat) < 1e-10

    def test_imp_resp_wid_preferred(self):
        """imp_resp_wid should be preferred over ss when both present."""
        meta = self._make_sicd_meta(
            row_ss=0.5, col_ss=0.5, row_irw=1.0, col_irw=1.0,
            image_plane='GROUND',
        )
        lat_deg, _ = _resolution_from_sicd(meta, None, 1.0)
        # Should use 1.0 m (imp_resp_wid) not 0.5 m (ss)
        expected_lat = 1.0 / 111320.0
        assert abs(lat_deg - expected_lat) < 1e-10

    def test_missing_grid_raises(self):
        """Should raise ValueError when grid is None."""
        meta = _OrigSICDMeta(
            format='SICD', rows=100, cols=100, dtype='complex64',
            grid=None,
        )
        with pytest.raises(ValueError, match="missing grid"):
            _resolution_from_sicd(meta, None, 1.0)

    def test_scale_factor(self):
        """Scale factor should multiply the output resolution."""
        meta = self._make_sicd_meta(
            row_ss=1.0, col_ss=1.0, image_plane='GROUND'
        )
        lat1, _ = _resolution_from_sicd(meta, None, 1.0)
        lat2, _ = _resolution_from_sicd(meta, None, 2.0)
        assert abs(lat2 - 2.0 * lat1) < 1e-12


# ---------------------------------------------------------------------------
# Tests: BIOMASS resolution
# ---------------------------------------------------------------------------

class TestResolutionBIOMASS:

    def test_biomass_spacing(self):
        """Uses max of range and azimuth spacing."""
        meta = {
            'range_pixel_spacing': 10.0,
            'azimuth_pixel_spacing': 20.0,
        }
        lat_deg, _ = _resolution_from_biomass(meta, None, 1.0)
        expected = 20.0 / 111320.0
        assert abs(lat_deg - expected) < 1e-10

    def test_biomass_missing_spacing_raises(self):
        """ValueError when spacing is zero or missing."""
        meta = {'range_pixel_spacing': 0, 'azimuth_pixel_spacing': 0}
        with pytest.raises(ValueError, match="BIOMASS"):
            _resolution_from_biomass(meta, None, 1.0)


# ---------------------------------------------------------------------------
# Tests: dispatch
# ---------------------------------------------------------------------------

class TestResolutionDispatch:

    def test_sicd_metadata_dispatches(self):
        """compute_output_resolution recognizes SICDMetadata."""
        meta = _OrigSICDMeta(
            format='SICD', rows=100, cols=100, dtype='complex64',
            grid=StubGrid(
                image_plane='GROUND',
                row=StubDirParam(ss=1.0),
                col=StubDirParam(ss=1.0),
            ),
            geo_data=StubGeoData(
                scp=StubSCP(llh=StubLatLonHAE(lat=0.0))
            ),
        )
        lat, lon = compute_output_resolution(meta)
        assert lat > 0 and lon > 0

    def test_biomass_dict_dispatches(self):
        """compute_output_resolution recognizes BIOMASS dict."""
        meta = {
            'range_pixel_spacing': 10.0,
            'azimuth_pixel_spacing': 10.0,
        }
        lat, lon = compute_output_resolution(meta)
        assert lat > 0 and lon > 0

    def test_unknown_metadata_raises(self):
        """Unknown metadata type should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot determine"):
            compute_output_resolution(object())
