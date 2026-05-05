# -*- coding: utf-8 -*-
"""
STANAG 4607 reader/writer round-trip tests.

End-to-end tests: build a synthetic mission, write it through
``STANAG4607Writer``, read it back through ``STANAG4607Reader``,
and assert structural equality. Includes multi-packet, multi-dwell,
and accessor / context-manager checks.

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

from grdl.exceptions import ValidationError
from grdl.IO.gmti import (
    STANAG4607Reader,
    STANAG4607Writer,
    open_gmti,
)
from grdl.IO.models.stanag4607 import (
    DwellSegment,
    MissionSegment,
    Packet,
    PacketHeader,
    STANAG4607Metadata,
    TargetReport,
)


def _build_synthetic_metadata(num_packets=1, num_dwells_per=1, num_targets=2):
    """Build a STANAG4607Metadata with the requested layout."""
    packets = []
    target_idx = 0
    for p in range(num_packets):
        ph = PacketHeader(
            version_id='40', nationality='US', classification=5,
            class_system='XN', platform_id=f'PLAT-{p}',
            mission_id=1000 + p, job_id=42,
        )
        segments = []
        if p == 0:
            segments.append(MissionSegment(
                mission_plan='SYNTH', flight_plan='F1',
                platform_type=1, platform_configuration='C-130J',
                reference_time_year=2026, reference_time_month=4,
                reference_time_day=29,
            ))
        for d in range(num_dwells_per):
            targets = []
            for t in range(num_targets):
                targets.append(TargetReport(
                    report_index=target_idx,
                    target_lat=45.0 + 0.001 * target_idx,
                    target_lon=-100.0 - 0.001 * target_idx,
                    target_velocity_los=-500 + 100 * target_idx,
                    target_snr=20 + target_idx,
                    target_classification=10,
                    target_class_probability=80,
                ))
                target_idx += 1
            segments.append(DwellSegment(
                revisit_index=1, dwell_index=d,
                last_dwell_of_revisit=int(d == num_dwells_per - 1),
                target_report_count=len(targets),
                dwell_time_ms=1000 * (p * num_dwells_per + d),
                sensor_pos_lat=45.5, sensor_pos_lon=-100.5,
                sensor_pos_alt=10000,
                sensor_track=90.0, sensor_speed=200_000,
                dwell_center_lat=45.0, dwell_center_lon=-100.0,
                dwell_range_half_extent=5,
                dwell_angle_half_extent=2.0,
                mdv=2,
                target_reports=targets,
            ))
        packets.append(Packet(header=ph, segments=segments))
    return STANAG4607Metadata(edition=4, packets=packets)


class TestRoundTrip:
    def test_single_packet_single_dwell_two_targets(self, tmp_path):
        meta = _build_synthetic_metadata(1, 1, 2)
        path = tmp_path / 'mission.4607'

        STANAG4607Writer(path, metadata=meta, edition=4).write()

        with STANAG4607Reader(path) as r:
            assert r.edition == 4
            assert r.num_packets() == 1
            assert r.num_dwells() == 1
            assert r.num_target_reports() == 2

    def test_multi_packet_multi_dwell(self, tmp_path):
        meta = _build_synthetic_metadata(3, 2, 4)
        path = tmp_path / 'multi.4607'

        STANAG4607Writer(path, metadata=meta, edition=4).write()

        with STANAG4607Reader(path) as r:
            assert r.num_packets() == 3
            assert r.num_dwells() == 6
            assert r.num_target_reports() == 24

    def test_open_gmti_factory(self, tmp_path):
        meta = _build_synthetic_metadata(1, 1, 1)
        path = tmp_path / 'factory.4607'
        STANAG4607Writer(path, metadata=meta, edition=4).write()

        with open_gmti(path) as r:
            assert isinstance(r, STANAG4607Reader)
            assert r.num_target_reports() == 1

    def test_target_field_preservation(self, tmp_path):
        meta = _build_synthetic_metadata(1, 1, 3)
        path = tmp_path / 'fields.4607'
        STANAG4607Writer(path, metadata=meta, edition=4).write()

        with STANAG4607Reader(path) as r:
            targets = list(r.iter_target_reports())
            assert len(targets) == 3
            for i, (dwell, target) in enumerate(targets):
                assert target.report_index == i
                assert target.target_snr == 20 + i
                # cm/s, exact integer round-trip
                assert target.target_velocity_los == -500 + 100 * i

    def test_mission_segment_roundtrips(self, tmp_path):
        meta = _build_synthetic_metadata(1, 1, 0)
        path = tmp_path / 'mission_only.4607'
        STANAG4607Writer(path, metadata=meta, edition=4).write()

        with STANAG4607Reader(path) as r:
            missions = r.metadata.missions
            assert len(missions) == 1
            assert missions[0].mission_plan == 'SYNTH'
            assert missions[0].reference_time_year == 2026

    def test_packet_header_identity_preserved(self, tmp_path):
        meta = _build_synthetic_metadata(2, 1, 1)
        path = tmp_path / 'headers.4607'
        STANAG4607Writer(path, metadata=meta, edition=4).write()

        with STANAG4607Reader(path) as r:
            headers = [p.header for p in r.metadata.packets]
            assert [h.platform_id for h in headers] == ['PLAT-0', 'PLAT-1']
            assert [h.mission_id for h in headers] == [1000, 1001]
            assert all(h.job_id == 42 for h in headers)


class TestEditionField:
    @pytest.mark.parametrize('edition', [2, 3, 4])
    def test_edition_in_round_trip(self, tmp_path, edition):
        meta = _build_synthetic_metadata(1, 1, 1)
        path = tmp_path / f'ed{edition}.4607'
        STANAG4607Writer(path, metadata=meta, edition=edition).write()

        with STANAG4607Reader(path) as r:
            assert r.edition == edition

    def test_invalid_edition_rejected(self, tmp_path):
        meta = _build_synthetic_metadata(1, 1, 1)
        path = tmp_path / 'bad.4607'
        with pytest.raises(ValidationError):
            STANAG4607Writer(path, metadata=meta, edition=5)


class TestErrorHandling:
    def test_missing_file_raises(self, tmp_path):
        path = tmp_path / 'does-not-exist.4607'
        with pytest.raises(FileNotFoundError):
            STANAG4607Reader(path)

    def test_truncated_file_raises(self, tmp_path):
        path = tmp_path / 'truncated.4607'
        path.write_bytes(b'\x00' * 8)
        with pytest.raises(ValidationError):
            STANAG4607Reader(path)

    def test_writer_rejects_wrong_metadata_type(self, tmp_path):
        with pytest.raises(ValidationError):
            STANAG4607Writer(tmp_path / 'x.4607', metadata={})


class TestContextManager:
    def test_with_statement(self, tmp_path):
        meta = _build_synthetic_metadata(1, 1, 1)
        path = tmp_path / 'cm.4607'
        STANAG4607Writer(path, metadata=meta, edition=4).write()

        with STANAG4607Reader(path) as r:
            assert r.num_packets() == 1
        # close() is a no-op, but exiting the with should not error
