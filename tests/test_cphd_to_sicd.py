# -*- coding: utf-8 -*-
"""
Tests for cphd_to_sicd metadata builder and CPHDToSICDConverter pipeline.

Covers:
- ``build_sicd_metadata`` produces a correctly typed ``SICDMetadata`` with
  all mandatory sections populated.
- PFA-specific sections (``pfa``, ``Grid.TimeCOAPoly``) are present when
  ``image_form_algo='PFA'``.
- ``pfa`` section is absent for non-PFA algorithms.
- Nyquist sample-spacing correction: when ``rec_n_samples`` / ``rec_n_pulses``
  differ from the formed-image pixel count the SS values are rescaled.
- ``CPHDToSICDConverter`` constructor validates the ``algorithm`` arg.
- ``CPHDToSICDConverter._select_algorithm`` honours the explicit override and
  the mode-based default.
- ``CPHDToSICDConverter`` is accessible from the top-level ``grdl.IO.sar``
  namespace.

All data is synthetic — no real CPHD files needed.

Author
------
Ava Courtney
courtney-ava@zai.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-05-20

Modified
--------
2026-05-20
"""

import numpy as np
import pytest
from numpy.linalg import norm

from grdl.IO.models.cphd import (
    CPHDChannel,
    CPHDCollectionInfo,
    CPHDGlobal,
    CPHDMetadata,
    CPHDPVP,
)
from grdl.IO.sar.cphd_to_sicd import build_sicd_metadata
from grdl.IO.sar.cphd_to_sicd_pipeline import CPHDToSICDConverter
from grdl.image_processing.sar.image_formation import (
    CollectionGeometry,
    PolarFormatAlgorithm,
    PolarGrid,
)


# ===================================================================
# Shared synthetic helpers (duplicated here so this file is
# self-contained; the canonical version lives in
# test_image_processing_sar_ifp.py)
# ===================================================================

_C = 299792458.0


def _synthetic_pvp(
    npulses: int = 64,
    nsamples: int = 128,
    center_freq_ghz: float = 10.0,
    bandwidth_mhz: float = 600.0,
    altitude_km: float = 10.0,
    slant_range_km: float = 15.0,
    aperture_deg: float = 3.0,
) -> CPHDPVP:
    fc = center_freq_ghz * 1e9
    bw = bandwidth_mhz * 1e6
    alt = altitude_km * 1e3
    sr = slant_range_km * 1e3

    lat_rad = np.radians(40.0)
    lon_rad = np.radians(-75.0)
    a_wgs = 6378137.0
    f_wgs = 1.0 / 298.257223563
    e2 = 2 * f_wgs - f_wgs**2
    N_val = a_wgs / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    srp_ecf = np.array([
        N_val * np.cos(lat_rad) * np.cos(lon_rad),
        N_val * np.cos(lat_rad) * np.sin(lon_rad),
        N_val * (1 - e2) * np.sin(lat_rad),
    ])

    half_angle = np.radians(aperture_deg / 2)
    angles = np.linspace(-half_angle, half_angle, npulses)

    u_up = srp_ecf / norm(srp_ecf)
    u_east = np.cross(np.array([0., 0., 1.]), u_up)
    u_east = u_east / norm(u_east)
    u_north = np.cross(u_up, u_east)

    ground_range = np.sqrt(sr**2 - alt**2)
    arp = np.array([
        srp_ecf + ground_range * np.cos(a) * u_north
        + ground_range * np.sin(a) * u_east
        + alt * u_up
        for a in angles
    ])

    speed = 200.0
    vel = np.array([
        speed * (-np.sin(a) * u_north + np.cos(a) * u_east)
        for a in angles
    ])

    arc_length = sr * 2 * half_angle
    total_time = arc_length / speed
    tx_time = np.linspace(0.0, total_time, npulses)
    rcv_time = tx_time + 2 * sr / _C

    srp_pos = np.tile(srp_ecf, (npulses, 1))
    fx1 = np.full(npulses, fc - bw / 2)
    fx2 = np.full(npulses, fc + bw / 2)
    fxss = np.full(npulses, bw / nsamples)

    return CPHDPVP(
        tx_time=tx_time,
        tx_pos=arp,
        tx_vel=vel,
        rcv_time=rcv_time,
        rcv_pos=arp,
        rcv_vel=vel,
        srp_pos=srp_pos,
        fx1=fx1,
        fx2=fx2,
        sc0=fx1.copy(),
        scss=fxss,
    )


def _synthetic_metadata(npulses: int = 64, nsamples: int = 128,
                         radar_mode: str = 'SPOTLIGHT') -> CPHDMetadata:
    pvp = _synthetic_pvp(npulses=npulses, nsamples=nsamples)
    fc = 10.0e9
    bw = 600.0e6
    return CPHDMetadata(
        format='CPHD',
        rows=npulses,
        cols=nsamples,
        dtype='complex64',
        channels=[CPHDChannel(
            identifier='CHANNEL001',
            num_vectors=npulses,
            num_samples=nsamples,
        )],
        pvp=pvp,
        global_params=CPHDGlobal(
            domain_type='FX',
            fx_band_min=fc - bw / 2,
            fx_band_max=fc + bw / 2,
        ),
        collection_info=CPHDCollectionInfo(
            collector_name='SYNTHETIC',
            core_name='TEST001',
            collect_type='MONOSTATIC',
            radar_mode=radar_mode,
        ),
        num_channels=1,
    )


def _build_geo_and_grid_params(
    meta: CPHDMetadata,
) -> tuple:
    """Return (CollectionGeometry, grid_params, image_shape)."""
    geo = CollectionGeometry(meta, slant=True)
    grid = PolarGrid(geo, grid_mode='INSCRIBED')
    # Tiny synthetic image (doesn't need real formation)
    nrows = grid.rec_n_samples
    ncols = grid.rec_n_pulses
    grid_params = grid.get_output_grid()
    return geo, grid_params, (nrows, ncols)


# ===================================================================
# build_sicd_metadata — section presence tests
# ===================================================================

class TestBuildSICDMetadata:

    def setup_method(self):
        self.meta = _synthetic_metadata()
        self.geo, self.grid_params, self.shape = _build_geo_and_grid_params(
            self.meta
        )

    def test_returns_sicd_metadata(self):
        from grdl.IO.models import SICDMetadata
        result = build_sicd_metadata(
            self.meta, self.geo, self.shape,
            grid_params=self.grid_params,
        )
        assert isinstance(result, SICDMetadata)

    def test_collection_info_populated(self):
        result = build_sicd_metadata(
            self.meta, self.geo, self.shape,
            grid_params=self.grid_params,
        )
        assert result.collection_info is not None
        assert result.collection_info.collector_name == 'SYNTHETIC'

    def test_image_data_shape(self):
        rows, cols = self.shape
        result = build_sicd_metadata(
            self.meta, self.geo, self.shape,
            grid_params=self.grid_params,
        )
        assert result.image_data.num_rows == rows
        assert result.image_data.num_cols == cols

    def test_geo_data_has_scp(self):
        result = build_sicd_metadata(
            self.meta, self.geo, self.shape,
            grid_params=self.grid_params,
        )
        assert result.geo_data is not None
        assert result.geo_data.scp is not None
        scp = result.geo_data.scp
        assert scp.llh.lat == pytest.approx(40.0, abs=1.0)
        assert scp.llh.lon == pytest.approx(-75.0, abs=1.0)

    def test_grid_has_time_coa_poly(self):
        result = build_sicd_metadata(
            self.meta, self.geo, self.shape,
            grid_params=self.grid_params,
        )
        assert result.grid is not None
        assert result.grid.time_coa_poly is not None
        # Constant polynomial: coefficient array has at least one element
        assert result.grid.time_coa_poly.coefs.size >= 1

    def test_scpcoa_populated(self):
        result = build_sicd_metadata(
            self.meta, self.geo, self.shape,
            grid_params=self.grid_params,
        )
        assert result.scpcoa is not None
        assert result.scpcoa.slant_range > 0

    def test_position_has_arp_poly(self):
        result = build_sicd_metadata(
            self.meta, self.geo, self.shape,
            grid_params=self.grid_params,
        )
        assert result.position is not None
        assert result.position.arp_poly is not None

    def test_timeline_populated(self):
        result = build_sicd_metadata(
            self.meta, self.geo, self.shape,
            grid_params=self.grid_params,
        )
        assert result.timeline is not None
        assert result.timeline.collect_duration > 0

    def test_pfa_section_present_for_pfa(self):
        result = build_sicd_metadata(
            self.meta, self.geo, self.shape,
            image_form_algo='PFA',
            grid_params=self.grid_params,
        )
        assert result.pfa is not None, "PFA section should be populated for PFA algo"

    def test_pfa_section_absent_for_rgazcomp(self):
        result = build_sicd_metadata(
            self.meta, self.geo, self.shape,
            image_form_algo='RGAZCOMP',
            grid_params=self.grid_params,
        )
        assert result.pfa is None, "PFA section should be None for RGAZCOMP"

    def test_pfa_has_polar_ang_poly(self):
        result = build_sicd_metadata(
            self.meta, self.geo, self.shape,
            image_form_algo='PFA',
            grid_params=self.grid_params,
        )
        assert result.pfa.polar_ang_poly is not None
        assert result.pfa.polar_ang_poly.coefs.size == 6  # degree-5 fit

    def test_pfa_has_spatial_freq_sf_poly(self):
        result = build_sicd_metadata(
            self.meta, self.geo, self.shape,
            image_form_algo='PFA',
            grid_params=self.grid_params,
        )
        assert result.pfa.spatial_freq_sf_poly is not None
        assert result.pfa.spatial_freq_sf_poly.coefs.size == 4  # degree-3 fit

    def test_invalid_algo_raises(self):
        with pytest.raises(ValueError, match="image_form_algo"):
            build_sicd_metadata(
                self.meta, self.geo, self.shape,
                image_form_algo='BADVALUE',
            )


class TestNyquistSSCorrection:
    """Verify that Grid SS is rescaled when image is larger than Nyquist grid."""

    def test_ss_rescaled_when_oversampled(self):
        """When image rows > rec_n_samples, rg_ss should be smaller."""
        meta = _synthetic_metadata()
        geo, grid_params, (nrows, ncols) = _build_geo_and_grid_params(meta)

        # Simulate an oversampled FFT: double the pixel count
        oversampled_shape = (nrows * 2, ncols * 2)

        result = build_sicd_metadata(
            meta, geo, oversampled_shape,
            image_form_algo='PFA',
            grid_params=grid_params,
        )

        nyquist_result = build_sicd_metadata(
            meta, geo, (nrows, ncols),
            image_form_algo='PFA',
            grid_params=grid_params,
        )

        # Oversampled image should have finer per-pixel spacing
        assert result.grid.row.ss < nyquist_result.grid.row.ss
        assert result.grid.col.ss < nyquist_result.grid.col.ss


# ===================================================================
# CPHDToSICDConverter — constructor and algorithm selection tests
# ===================================================================

class TestCPHDToSICDConverterConstruction:

    def test_valid_algorithm_accepted(self):
        for algo in ('PFA', 'RDA', 'FFBP', 'StripmapPFA'):
            c = CPHDToSICDConverter('in.cphd', 'out.nitf', algorithm=algo)
            assert c.algorithm == algo

    def test_none_algorithm_accepted(self):
        c = CPHDToSICDConverter('in.cphd', 'out.nitf')
        assert c.algorithm is None

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="algorithm"):
            CPHDToSICDConverter('in.cphd', 'out.nitf', algorithm='GARBAGE')

    def test_default_options(self):
        c = CPHDToSICDConverter('in.cphd', 'out.nitf')
        assert c.grid_mode == 'inscribed'
        assert c.range_oversample == 1.0
        assert c.azimuth_oversample == 1.0
        assert c.weighting == 'uniform'
        assert c.phase_sgn == -1
        assert c.slant is True

    def test_paths_stored_as_path_objects(self):
        from pathlib import Path
        c = CPHDToSICDConverter('a.cphd', 'b.nitf')
        assert isinstance(c.input_path, Path)
        assert isinstance(c.output_path, Path)


class TestCPHDToSICDAlgorithmSelection:

    def _make_meta(self, radar_mode: str) -> CPHDMetadata:
        return _synthetic_metadata(radar_mode=radar_mode)

    def test_explicit_overrides_metadata(self):
        c = CPHDToSICDConverter('a.cphd', 'b.nitf', algorithm='RDA')
        meta = self._make_meta('SPOTLIGHT')  # would default to PFA
        assert c._select_algorithm(meta) == 'RDA'

    def test_spotlight_defaults_to_pfa(self):
        c = CPHDToSICDConverter('a.cphd', 'b.nitf')
        assert c._select_algorithm(self._make_meta('SPOTLIGHT')) == 'PFA'

    def test_stripmap_defaults_to_rda(self):
        c = CPHDToSICDConverter('a.cphd', 'b.nitf')
        assert c._select_algorithm(self._make_meta('STRIPMAP')) == 'RDA'

    def test_dynamic_stripmap_defaults_to_rda(self):
        c = CPHDToSICDConverter('a.cphd', 'b.nitf')
        assert c._select_algorithm(self._make_meta('DYNAMIC STRIPMAP')) == 'RDA'

    def test_unknown_mode_falls_back_to_pfa(self):
        c = CPHDToSICDConverter('a.cphd', 'b.nitf')
        assert c._select_algorithm(self._make_meta('WEIRD_MODE')) == 'PFA'


# ===================================================================
# grdl.IO.sar namespace export test
# ===================================================================

class TestSARNamespaceExport:

    def test_cphd_to_sicd_converter_importable(self):
        from grdl.IO.sar import CPHDToSICDConverter as C
        assert C is CPHDToSICDConverter

    def test_build_sicd_metadata_importable(self):
        from grdl.IO.sar import build_sicd_metadata as bsm
        assert callable(bsm)

    def test_cphd_to_sicd_converter_in_all(self):
        import grdl.IO.sar as sar_mod
        assert 'CPHDToSICDConverter' in sar_mod.__all__
