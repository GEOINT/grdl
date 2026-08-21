# -*- coding: utf-8 -*-
"""
Compact-pol synthesis metadata tests.

Verifies that synthesized compact-pol SICD metadata built from NISAR RSLC
inputs includes the geometry needed for geolocation.
"""

import numpy as np
import pytest

from grdl.IO.models.nisar import (
    NISARGeolocationGrid,
    NISARIdentification,
    NISARMetadata,
    NISAROrbit,
    NISARSwathParameters,
)
from grdl.IO.sar.compact_pol_synthesis import _build_compact_pol_sicd_metadata
from grdl.geolocation.sar.sicd import SICDGeolocation


def _make_nisar_metadata(rows: int = 11, cols: int = 13) -> NISARMetadata:
    """Create a small NISAR RSLC metadata object with consistent geometry."""
    n_az = 4
    n_rg = 5
    n_h = 3

    az = np.linspace(0.0, 1.0, n_az)[:, None]
    rg = np.linspace(0.0, 1.0, n_rg)[None, :]
    lat_grid = 65.0 + 0.25 * az + 0.02 * rg
    lon_grid = -147.0 + 0.02 * az + 0.35 * rg
    inc_grid = 35.0 + 2.0 * az + 0.5 * rg

    reference_epoch = 'seconds since 2026-01-05T00:00:00'
    start_sec = 21560.0
    end_sec = 21570.0
    orbit_time = np.linspace(start_sec - 20.0, end_sec + 20.0, 9)
    orbit_position = np.column_stack([
        7.0e6 + 12.0 * (orbit_time - start_sec),
        1.5e6 + 45.0 * (orbit_time - start_sec),
        2.0e6 - 8.0 * (orbit_time - start_sec),
    ])
    orbit_velocity = np.gradient(orbit_position, orbit_time, axis=0)

    return NISARMetadata(
        format='NISAR',
        dtype='complex64',
        rows=rows,
        cols=cols,
        identification=NISARIdentification(
            look_direction='left',
            zero_doppler_start_time='2026-01-05T05:59:20Z',
            zero_doppler_end_time='2026-01-05T05:59:30Z',
            mission_id='NISAR',
        ),
        orbit=NISAROrbit(
            time=orbit_time,
            position=orbit_position,
            velocity=orbit_velocity,
            reference_epoch=reference_epoch,
        ),
        swath_parameters=NISARSwathParameters(
            processed_center_frequency=1.229e9,
            processed_range_bandwidth=20.0e6,
            scene_center_along_track_spacing=4.0,
            slant_range_spacing=6.0,
            slant_range=np.linspace(800000.0, 800072.0, cols),
            zero_doppler_time=np.linspace(start_sec, end_sec, rows),
            zero_doppler_time_reference_epoch=reference_epoch,
        ),
        geolocation_grid=NISARGeolocationGrid(
            coordinate_x=np.repeat(lon_grid[None, :, :], n_h, axis=0),
            coordinate_y=np.repeat(lat_grid[None, :, :], n_h, axis=0),
            epsg=4326,
            slant_range=np.linspace(800000.0, 800072.0, n_rg),
            zero_doppler_time=np.linspace(start_sec, end_sec, n_az),
            height_above_ellipsoid=np.array([-100.0, 0.0, 100.0]),
            incidence_angle=np.repeat(
                inc_grid[None, :, :].astype(np.float32), n_h, axis=0,
            ),
            elevation_angle=np.repeat(
                (90.0 - inc_grid)[None, :, :].astype(np.float32), n_h, axis=0,
            ),
        ),
    )


def test_build_compact_pol_sicd_metadata_from_nisar_is_geolocatable():
    """NISAR-based synthesis should populate the SICD geometry sections."""
    meta = _build_compact_pol_sicd_metadata(
        _make_nisar_metadata(), 'H', rows=11, cols=13, transmit='R',
    )

    assert meta.geo_data is not None
    assert meta.geo_data.scp is not None
    assert meta.geo_data.scp.llh is not None
    assert meta.geo_data.scp.ecf is not None
    assert meta.geo_data.image_corners is not None
    assert len(meta.geo_data.image_corners) == 4

    assert meta.grid is not None
    assert meta.grid.type == 'PLANE'
    assert meta.grid.row is not None
    assert meta.grid.col is not None
    assert meta.grid.time_coa_poly is not None

    assert meta.position is not None
    assert meta.position.arp_poly is not None

    assert meta.scpcoa is not None
    assert meta.scpcoa.side_of_track == 'L'
    assert meta.scpcoa.arp_pos is not None
    assert meta.scpcoa.arp_vel is not None
    assert meta.scpcoa.slant_range == pytest.approx(800036.0)
    assert meta.scpcoa.incidence_ang == pytest.approx(36.25)

    assert meta.timeline is not None
    assert meta.timeline.collect_start == '2026-01-05T05:59:20Z'
    assert meta.timeline.collect_duration == pytest.approx(10.0)
    assert meta.radar_collection is not None
    assert meta.radar_collection.tx_frequency is not None

    geo = SICDGeolocation(meta, backend='native')
    assert geo._coa_proj is not None

    center = geo.image_to_latlon(5.0, 6.0)
    assert center.shape == (3,)
    assert center[0] == pytest.approx(meta.geo_data.scp.llh.lat, abs=0.2)
    assert center[1] == pytest.approx(meta.geo_data.scp.llh.lon, abs=0.2)
