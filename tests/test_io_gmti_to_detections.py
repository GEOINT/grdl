# -*- coding: utf-8 -*-
"""
STANAG 4607 → DetectionSet bridge tests.

Validates the ``STANAG4607Reader.to_detection_set()`` conversion:
each target report becomes one ``Detection``, ``geo_geometry`` is a
shapely ``Point`` in signed lon/lat, and ``properties`` are populated
from the ``gmti.*`` field domain plus ``physical.velocity_radial``.

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

shapely = pytest.importorskip('shapely')

from grdl.IO.gmti import STANAG4607Reader, STANAG4607Writer
from grdl.IO.models.stanag4607 import (
    DwellSegment,
    Packet,
    PacketHeader,
    STANAG4607Metadata,
    TargetReport,
)


@pytest.fixture
def synthetic_path(tmp_path):
    """Write a tiny synthetic 4607 file with two targets and yield its path."""
    ph = PacketHeader(version_id='40', nationality='US', classification=5,
                      class_system='XN', platform_id='SYNTH',
                      mission_id=7, job_id=1)
    targets = [
        TargetReport(report_index=0, target_lat=45.0, target_lon=-122.0,
                     target_velocity_los=-500, target_snr=25,
                     target_classification=10),
        TargetReport(report_index=1, target_lat=45.001, target_lon=-122.001,
                     target_velocity_los=300, target_snr=15,
                     target_classification=12),
    ]
    dwell = DwellSegment(revisit_index=1, dwell_index=0,
                         last_dwell_of_revisit=1, target_report_count=2,
                         dwell_time_ms=1500,
                         sensor_pos_lat=45.5, sensor_pos_lon=-122.5,
                         sensor_pos_alt=10000,
                         dwell_center_lat=45.0, dwell_center_lon=-122.0,
                         dwell_range_half_extent=5,
                         dwell_angle_half_extent=2.0,
                         mdv=2, target_reports=targets)
    meta = STANAG4607Metadata(
        edition=4, packets=[Packet(header=ph, segments=[dwell])],
    )
    path = tmp_path / 'targets.4607'
    STANAG4607Writer(path, metadata=meta, edition=4).write()
    return path


class TestToDetectionSet:
    def test_one_detection_per_target(self, synthetic_path):
        with STANAG4607Reader(synthetic_path) as r:
            ds = r.to_detection_set()
        assert len(ds) == 2

    def test_detector_metadata(self, synthetic_path):
        with STANAG4607Reader(synthetic_path) as r:
            ds = r.to_detection_set()
        assert ds.detector_name == 'STANAG4607Reader'
        assert ds.detector_version == '1.0.0'
        assert ds.metadata['edition'] == 4
        assert ds.metadata['num_dwells'] == 1

    def test_geo_geometry_is_signed_point(self, synthetic_path):
        from shapely.geometry import Point

        with STANAG4607Reader(synthetic_path) as r:
            ds = r.to_detection_set()

        first = ds[0]
        assert first.pixel_geometry is None
        assert isinstance(first.geo_geometry, Point)
        # Longitude should be in [-180, 180] (signed) — not BA32 wrapped
        assert -123 < first.geo_geometry.x < -121
        assert 44 < first.geo_geometry.y < 46

    def test_gmti_properties_present(self, synthetic_path):
        with STANAG4607Reader(synthetic_path) as r:
            ds = r.to_detection_set()

        props = ds[0].properties
        # gmti.* domain fields
        for key in (
            'gmti.report_index', 'gmti.dwell_index', 'gmti.snr_db',
            'gmti.target_classification', 'gmti.target_class_probability',
            'gmti.platform_lat', 'gmti.platform_lon',
        ):
            assert key in props, f"missing key {key}"

        # Reused physical.velocity_radial domain (m/s, not cm/s)
        assert 'physical.velocity_radial' in props
        assert abs(props['physical.velocity_radial'] - (-5.0)) < 1e-3

    def test_confidence_normalized(self, synthetic_path):
        with STANAG4607Reader(synthetic_path) as r:
            ds = r.to_detection_set(snr_normalization=40.0)

        # SNR=25 dB / 40 dB → 0.625
        assert ds[0].confidence is not None
        assert abs(ds[0].confidence - 0.625) < 1e-3
        # SNR=15 dB → 0.375
        assert abs(ds[1].confidence - 0.375) < 1e-3

    def test_geojson_export_works(self, synthetic_path):
        with STANAG4607Reader(synthetic_path) as r:
            ds = r.to_detection_set()
        gj = ds.to_geojson()
        assert gj['type'] == 'FeatureCollection'
        assert len(gj['features']) == 2
        first = gj['features'][0]
        assert first['type'] == 'Feature'
        assert first['geometry']['type'] == 'Point'

    def test_gmti_fields_in_data_dictionary(self):
        """The ``gmti.*`` field domain should be registered."""
        from grdl.image_processing.detection.fields import (
            DATA_DICTIONARY, Fields,
        )
        assert 'gmti.snr_db' in DATA_DICTIONARY
        assert 'gmti.report_index' in DATA_DICTIONARY
        assert 'gmti.platform_lat' in DATA_DICTIONARY
        # Fields accessor works
        assert Fields.gmti.SNR_DB == 'gmti.snr_db'
