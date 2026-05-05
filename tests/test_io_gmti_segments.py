# -*- coding: utf-8 -*-
"""
STANAG 4607 segment codec tests.

Per-segment serialize → parse round-trip tests for the binary codec
in ``grdl.IO.gmti._segments``. Covers packet headers, mission, dwell
(with and without target reports), job definition, free text, and
platform location segments.

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

import pytest

from grdl.IO.gmti import _segments as S
from grdl.IO.models.stanag4607 import (
    DwellSegment,
    FreeTextSegment,
    JobDefinitionSegment,
    MissionSegment,
    PacketHeader,
    PlatformLocationSegment,
    TargetReport,
)


def _approx(a, b, tol=1e-5):
    return abs(float(a) - float(b)) < tol


class TestPacketHeader:
    def test_size_is_32_bytes(self):
        ph = PacketHeader()
        assert len(S.serialize_packet_header(ph)) == 32

    def test_roundtrip_identity_fields(self):
        ph = PacketHeader(
            version_id='40', packet_size=128, nationality='US',
            classification=5, class_system='XN', code=42,
            exercise_indicator=1, platform_id='RSI-001',
            mission_id=12345, job_id=67890,
        )
        buf = S.serialize_packet_header(ph)
        ph2, end = S.parse_packet_header(buf)
        assert end == 32
        assert ph2.version_id == '40'
        assert ph2.packet_size == 128
        assert ph2.nationality == 'US'
        assert ph2.classification == 5
        assert ph2.class_system == 'XN'
        assert ph2.code == 42
        assert ph2.exercise_indicator == 1
        assert ph2.platform_id == 'RSI-001'
        assert ph2.mission_id == 12345
        assert ph2.job_id == 67890


class TestMissionSegment:
    def test_roundtrip_through_segment_dispatch(self):
        ms = MissionSegment(
            mission_plan='OP_NORTHSTAR', flight_plan='FP-Alpha',
            platform_type=3, platform_configuration='C-130J',
            reference_time_year=2026, reference_time_month=4,
            reference_time_day=29,
        )
        buf = S.serialize_segment(ms)
        # First byte should be the SEG_MISSION type code
        assert buf[0] == S.SEG_MISSION
        seg, end = S.parse_segment(buf, 0)
        assert end == len(buf)
        assert isinstance(seg, MissionSegment)
        assert seg.mission_plan == 'OP_NORTHSTAR'
        assert seg.flight_plan == 'FP-Alpha'
        assert seg.platform_type == 3
        assert seg.platform_configuration == 'C-130J'
        assert seg.reference_time_year == 2026
        assert seg.reference_time_month == 4
        assert seg.reference_time_day == 29


class TestTargetReport:
    def test_roundtrip_field_for_field(self):
        tr = TargetReport(
            report_index=7, target_lat=45.5, target_lon=-122.5,
            target_height=120, target_velocity_los=-1234,
            target_wrap_velocity=20000, target_snr=27,
            target_classification=18, target_class_probability=85,
            slant_range_std=33, cross_range_std=11, height_std=4,
            velocity_los_std=12, truth_tag_application=1,
            truth_tag_entity=99, target_rcs=-3,
        )
        buf = S.serialize_target_report(tr)
        tr2, off = S.parse_target_report(buf, 0)
        assert off == len(buf)
        assert tr2.report_index == 7
        assert _approx(tr2.target_lat, 45.5)
        # BA32 is unsigned [0, 360); -122.5 wraps to 237.5
        assert _approx(tr2.target_lon, 237.5) or _approx(tr2.target_lon, -122.5)
        assert tr2.target_height == 120
        assert tr2.target_velocity_los == -1234
        assert tr2.target_wrap_velocity == 20000
        assert tr2.target_snr == 27
        assert tr2.target_classification == 18
        assert tr2.target_class_probability == 85
        assert tr2.slant_range_std == 33
        assert tr2.cross_range_std == 11
        assert tr2.height_std == 4
        assert tr2.velocity_los_std == 12
        assert tr2.truth_tag_application == 1
        assert tr2.truth_tag_entity == 99
        assert tr2.target_rcs == -3


class TestDwellSegment:
    def test_empty_dwell_roundtrips(self):
        d = DwellSegment(
            revisit_index=1, dwell_index=1, last_dwell_of_revisit=1,
            target_report_count=0, dwell_time_ms=500,
            sensor_pos_lat=45.0, sensor_pos_lon=-100.0,
            sensor_pos_alt=10000,
        )
        buf = S.serialize_segment(d)
        seg, end = S.parse_segment(buf, 0)
        assert end == len(buf)
        assert isinstance(seg, DwellSegment)
        assert seg.dwell_index == 1
        assert seg.target_report_count == 0
        assert len(seg.target_reports) == 0

    def test_dwell_with_targets_roundtrips_and_count_is_recomputed(self):
        targets = [
            TargetReport(report_index=i, target_lat=45.0 + 0.001 * i,
                         target_lon=-100.0 - 0.001 * i, target_snr=20 + i)
            for i in range(3)
        ]
        # Set target_report_count to a wrong value — the writer should
        # rewrite it from len(target_reports).
        d = DwellSegment(
            revisit_index=1, dwell_index=2, last_dwell_of_revisit=0,
            target_report_count=99, dwell_time_ms=500,
            sensor_pos_lat=45.0, sensor_pos_lon=-100.0,
            sensor_pos_alt=10000,
            dwell_center_lat=45.0, dwell_center_lon=-100.0,
            dwell_range_half_extent=5, dwell_angle_half_extent=2.0,
            mdv=2,
            target_reports=targets,
        )
        buf = S.serialize_segment(d)
        seg, end = S.parse_segment(buf, 0)
        assert seg.target_report_count == 3
        assert len(seg.target_reports) == 3
        assert [t.report_index for t in seg.target_reports] == [0, 1, 2]

    def test_existence_mask_filled_to_all_ones(self):
        d = DwellSegment(
            revisit_index=1, dwell_index=1, last_dwell_of_revisit=0,
            target_report_count=0, dwell_time_ms=0,
            sensor_pos_lat=0.0, sensor_pos_lon=0.0, sensor_pos_alt=0,
        )
        buf = S.serialize_segment(d)
        seg, _ = S.parse_segment(buf, 0)
        assert seg.existence_mask == 0xFFFFFFFFFFFFFFFF


class TestJobDefinition:
    def test_roundtrip(self):
        j = JobDefinitionSegment(
            job_id=42, sensor_id_type=2, sensor_id_model='RDR-X',
            target_filtering_flag=0, priority=10,
            bounding_a_lat=45.0, bounding_a_lon=-100.0,
            bounding_b_lat=45.0, bounding_b_lon=-99.0,
            bounding_c_lat=44.0, bounding_c_lon=-99.0,
            bounding_d_lat=44.0, bounding_d_lon=-100.0,
            radar_mode=3, nominal_revisit_interval=60,
            nominal_uncertainty_along_track=10,
            nominal_uncertainty_cross_track=10,
            nominal_uncertainty_altitude=5,
            nominal_uncertainty_track_heading=2,
            nominal_uncertainty_speed=100,
            nominal_sensor_slant_range_std=100,
            nominal_sensor_cross_range_std=0.001,
            nominal_sensor_los_velocity_std=20,
            nominal_sensor_mdv=2,
            nominal_sensor_detection_probability=85,
            nominal_sensor_false_alarm_density=1,
            terrain_elevation_model=1, geoid_model=1,
        )
        buf = S.serialize_segment(j)
        seg, end = S.parse_segment(buf, 0)
        assert end == len(buf)
        assert isinstance(seg, JobDefinitionSegment)
        assert seg.job_id == 42
        assert seg.sensor_id_model == 'RDR-X'
        assert seg.priority == 10
        assert seg.radar_mode == 3
        assert _approx(seg.bounding_a_lat, 45.0)
        # cross_range_std is a float; allow a wider tolerance
        assert abs(seg.nominal_sensor_cross_range_std - 0.001) < 1e-6


class TestFreeText:
    def test_roundtrip_short_text(self):
        f = FreeTextSegment(
            originator='OP1', recipient='OPS',
            text='hello world',
        )
        buf = S.serialize_segment(f)
        seg, end = S.parse_segment(buf, 0)
        assert end == len(buf)
        assert isinstance(seg, FreeTextSegment)
        assert seg.originator == 'OP1'
        assert seg.recipient == 'OPS'
        assert seg.text == 'hello world'

    def test_empty_text_roundtrips(self):
        f = FreeTextSegment(originator='A', recipient='B', text='')
        buf = S.serialize_segment(f)
        seg, _ = S.parse_segment(buf, 0)
        assert seg.text == ''


class TestPlatformLocation:
    def test_roundtrip(self):
        p = PlatformLocationSegment(
            location_time_ms=12345, platform_lat=45.123,
            platform_lon=-100.456, platform_alt=10000,
            platform_track=270.0, platform_speed=200_000,
            platform_vertical_velocity=-3,
        )
        buf = S.serialize_segment(p)
        seg, end = S.parse_segment(buf, 0)
        assert end == len(buf)
        assert seg.location_time_ms == 12345
        assert _approx(seg.platform_lat, 45.123)
        # platform_speed encoded as B32 mm/s — exact match expected
        assert seg.platform_speed == 200_000


class TestUnsupportedSegmentType:
    def test_returns_none_and_advances_offset(self):
        # Build a fake "HRR" segment (type 3) of size 10 and confirm
        # the parser skips it without raising.
        import struct
        seg_type = S.SEG_HRR
        seg_size = S.SEGMENT_HEADER_SIZE + 5
        buf = struct.pack('>BI', seg_type, seg_size) + b'XXXXX'
        seg, end = S.parse_segment(buf, 0)
        assert seg is None
        assert end == seg_size
