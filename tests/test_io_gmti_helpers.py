# -*- coding: utf-8 -*-
"""
STANAG 4607 helper function tests.

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
2026-04-29

Modified
--------
2026-04-29
"""

import math

import pytest

from grdl.exceptions import DependencyError
from grdl.IO.gmti import (
    STANAG4607Reader,
    STANAG4607Writer,
    dwell_footprint_polygon,
    filter_target_reports,
    ground_relative_velocity,
    summarize,
)
from grdl.IO.models.stanag4607 import (
    DwellSegment,
    Packet,
    PacketHeader,
    STANAG4607Metadata,
    TargetReport,
)


def _build_dwell(num_targets=3):
    targets = [
        TargetReport(report_index=i,
                     target_lat=45.0 + 0.001 * i,
                     target_lon=-122.0 - 0.001 * i,
                     target_velocity_los=-500 + 200 * i,
                     target_snr=20 + i,
                     target_classification=10 + (i % 2))
        for i in range(num_targets)
    ]
    return DwellSegment(
        revisit_index=1, dwell_index=0,
        last_dwell_of_revisit=1, target_report_count=num_targets,
        dwell_time_ms=1000,
        sensor_pos_lat=45.5, sensor_pos_lon=-122.5,
        sensor_pos_alt=10000,
        sensor_track=90.0, sensor_speed=200_000,
        dwell_center_lat=45.0, dwell_center_lon=-122.0,
        dwell_range_half_extent=5, dwell_angle_half_extent=2.0,
        mdv=2,
        target_reports=targets,
    )


class TestFilterTargetReports:
    def test_snr_filter(self):
        d = _build_dwell(5)
        out = filter_target_reports(d.target_reports, snr_min=22)
        # snr values 20, 21, 22, 23, 24 -> >=22 leaves 3
        assert len(out) == 3
        for r in out:
            assert r.target_snr >= 22

    def test_classification_filter(self):
        d = _build_dwell(5)
        out = filter_target_reports(d.target_reports, classification=10)
        # classifications: 10, 11, 10, 11, 10 -> three with class 10
        assert len(out) == 3

    def test_mdv_filter_drops_slow_targets(self):
        d = _build_dwell(5)
        # mdv_min=2 (m/s) drops targets with |LOS velocity| < 2 m/s.
        # Velocities (cm/s): -500, -300, -100, 100, 300 -> m/s -5, -3, -1, 1, 3
        # |v| >= 2: -5, -3, 3 -> three remain
        out = filter_target_reports(d.target_reports, mdv_min=2.0)
        assert len(out) == 3

    def test_bbox_filter(self):
        d = _build_dwell(5)
        # Targets cluster around (45.0..45.004, -122.0..-122.004). Use a
        # bbox that excludes the last two targets.
        bbox = (-122.003, 44.999, -121.999, 45.0021)
        out = filter_target_reports(d.target_reports, bbox=bbox)
        assert len(out) == 3

    def test_empty_filter_returns_all(self):
        d = _build_dwell(3)
        out = filter_target_reports(d.target_reports)
        assert len(out) == 3

    def test_does_not_mutate_input(self):
        d = _build_dwell(3)
        original = list(d.target_reports)
        filter_target_reports(d.target_reports, snr_min=99)
        assert d.target_reports == original


class TestDwellFootprintPolygon:
    pytest.importorskip('shapely')

    def test_returns_polygon(self):
        d = _build_dwell(0)
        poly = dwell_footprint_polygon(d)
        assert poly.geom_type == 'Polygon'
        assert poly.is_valid
        # Centroid should be near the dwell center
        cx, cy = poly.centroid.x, poly.centroid.y
        assert abs(cy - 45.0) < 0.1
        # Center longitude is signed; -122 should be near the centroid
        assert abs(cx - (-122.0)) < 0.1 or abs(cx - 238.0) < 0.1

    def test_missing_extent_raises(self):
        d = DwellSegment(dwell_center_lat=45.0, dwell_center_lon=-122.0)
        # Half-extents are None — should raise ValueError, not silently
        # produce a degenerate polygon.
        with pytest.raises(ValueError):
            dwell_footprint_polygon(d)


class TestGroundRelativeVelocity:
    def test_returns_los_when_track_unknown(self):
        d = _build_dwell(1)
        d.sensor_track = None
        v = ground_relative_velocity(d.target_reports[0], d)
        # Falls back to LOS m/s = -500 cm/s / 100 = -5
        assert abs(v - (-5.0)) < 1e-6

    def test_corrects_for_platform_velocity(self):
        d = _build_dwell(1)
        # Platform moving east at 200 m/s; target slightly east of sensor
        # so platform contributes a closing rate that subtracts from
        # the target's measured LOS velocity. Just verify the value
        # changed from the LOS-only value.
        v = ground_relative_velocity(d.target_reports[0], d)
        v_los_only = float(d.target_reports[0].target_velocity_los) / 100.0
        assert abs(v - v_los_only) > 1e-3


class TestSummarize:
    def test_keys_and_counts(self, tmp_path):
        ph = PacketHeader(version_id='40', platform_id='X',
                          mission_id=7, job_id=11)
        d = _build_dwell(3)
        meta = STANAG4607Metadata(
            edition=4, packets=[Packet(header=ph, segments=[d])],
        )
        path = tmp_path / 's.4607'
        STANAG4607Writer(path, metadata=meta, edition=4).write()

        with STANAG4607Reader(path) as r:
            s = summarize(r)

        assert s['edition'] == 4
        assert s['num_packets'] == 1
        assert s['num_dwells'] == 1
        assert s['num_target_reports'] == 3
        assert s['mission_id'] == 7
        assert s['job_id'] == 11
        assert s['platform_id'] == 'X'
        assert s['time_bounds_ms'] == (1000, 1000)
        # geographic_bounds should report signed longitude (≈ -122)
        bounds = s['geographic_bounds']
        assert bounds is not None
        min_lon, _, max_lon, _ = bounds
        assert min_lon < 0 and max_lon < 0
