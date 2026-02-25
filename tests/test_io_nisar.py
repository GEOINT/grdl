# -*- coding: utf-8 -*-
"""
Tests for NISAR RSLC/GSLC reader and metadata.

All tests use synthetic HDF5 data -- no real NISAR products required.

Author
------
Jason Fritz
jpfritz@zai.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-25

Modified
--------
2026-02-25
"""

# Standard library
from pathlib import Path

# Third-party
import numpy as np
import pytest

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

# GRDL
from grdl.IO.models.nisar import (
    NISARMetadata,
    NISARIdentification,
    NISAROrbit,
    NISARAttitude,
    NISARSwathParameters,
    NISARGridParameters,
    NISARGeolocationGrid,
    NISARCalibration,
    NISARProcessingInfo,
)

requires_h5py = pytest.mark.skipif(
    not _HAS_H5PY, reason="h5py not installed"
)

# ===================================================================
# Constants for synthetic data
# ===================================================================

_RSLC_ROWS = 50
_RSLC_COLS = 40
_GSLC_ROWS = 60
_GSLC_COLS = 45
_RNG = np.random.default_rng(42)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def synthetic_nisar_rslc(tmp_path):
    """Create a minimal synthetic NISAR RSLC HDF5 file.

    Returns (filepath, expected_hh_data, expected_hv_data).
    """
    if not _HAS_H5PY:
        pytest.skip("h5py not installed")

    filepath = tmp_path / 'NISAR_L1_RSLC_TEST.h5'
    rng = np.random.default_rng(42)

    slc_hh = (
        rng.standard_normal((_RSLC_ROWS, _RSLC_COLS))
        + 1j * rng.standard_normal((_RSLC_ROWS, _RSLC_COLS))
    ).astype(np.complex64)

    slc_hv = (
        rng.standard_normal((_RSLC_ROWS, _RSLC_COLS))
        + 1j * rng.standard_normal((_RSLC_ROWS, _RSLC_COLS))
    ).astype(np.complex64)

    with h5py.File(str(filepath), 'w') as f:
        # Root attributes
        f.attrs['Conventions'] = 'CF-1.7'
        f.attrs['mission_name'] = 'NISAR'
        f.attrs['title'] = 'NISAR L1 RSLC Product'

        # Identification
        id_grp = f.create_group('science/LSAR/identification')
        id_grp.create_dataset('missionId', data=b'NISAR')
        id_grp.create_dataset('productType', data=b'RSLC')
        id_grp.create_dataset('radarBand', data=b'L')
        id_grp.create_dataset('lookDirection', data=b'right')
        id_grp.create_dataset('orbitPassDirection', data=b'ascending')
        id_grp.create_dataset('absoluteOrbitNumber', data=np.uint32(1234))
        id_grp.create_dataset('trackNumber', data=np.uint8(30))
        id_grp.create_dataset('frameNumber', data=np.uint16(19))
        id_grp.create_dataset('isGeocoded', data=b'False')
        id_grp.create_dataset('processingType', data=b'CUSTOM')
        id_grp.create_dataset(
            'zeroDopplerStartTime',
            data=b'2008-10-12T06:09:12.000000000',
        )
        id_grp.create_dataset(
            'zeroDopplerEndTime',
            data=b'2008-10-12T06:09:25.000000000',
        )
        id_grp.create_dataset(
            'boundingPolygon',
            data=b'POLYGON ((-118.2 34.3, -117.5 34.5, -117.7 35.2, '
                 b'-118.4 35.1, -118.2 34.3))',
        )
        id_grp.create_dataset(
            'granuleId',
            data=b'NISAR_L1_PR_RSLC_001_030_A_019_2000_TEST',
        )
        id_grp.create_dataset('instrumentName', data=b'PALSAR')
        id_grp.create_dataset(
            'processingDateTime', data=b'2024-06-05T17:23:18',
        )
        id_grp.create_dataset('productVersion', data=b'0.1.0')

        # RSLC product group
        rslc = f.create_group('science/LSAR/RSLC')

        # Swath data - frequency A
        freq_a = rslc.create_group('swaths/frequencyA')
        freq_a.create_dataset(
            'listOfPolarizations', data=[b'HH', b'HV']
        )
        freq_a.create_dataset(
            'acquiredCenterFrequency', data=1.27e9
        )
        freq_a.create_dataset(
            'acquiredRangeBandwidth', data=28e6
        )
        freq_a.create_dataset(
            'processedCenterFrequency', data=1.27e9
        )
        freq_a.create_dataset(
            'processedRangeBandwidth', data=28e6
        )
        freq_a.create_dataset(
            'processedAzimuthBandwidth', data=1260.0
        )
        freq_a.create_dataset('slantRangeSpacing', data=4.684)
        freq_a.create_dataset('nominalAcquisitionPRF', data=1862.2)
        freq_a.create_dataset(
            'sceneCenterAlongTrackSpacing', data=4.503
        )
        freq_a.create_dataset(
            'sceneCenterGroundRangeSpacing', data=11.472
        )
        freq_a.create_dataset('numberOfSubSwaths', data=np.uint8(1))
        freq_a.create_dataset('zeroDopplerTimeSpacing', data=0.000658)

        # SLC imagery
        freq_a.create_dataset('HH', data=slc_hh)
        freq_a.create_dataset('HV', data=slc_hv)

        # Zero-Doppler time
        zdt = np.linspace(22152.0, 22165.0, _RSLC_ROWS)
        ds_zdt = freq_a.create_dataset('zeroDopplerTime', data=zdt)
        ds_zdt.attrs['units'] = b'seconds since 2008-10-12 00:00:00'

        # Slant range
        sr = np.linspace(743587.0, 772527.0, _RSLC_COLS)
        freq_a.create_dataset('slantRange', data=sr)

        # Swath-level zero Doppler time (group already exists)
        swaths_grp = rslc['swaths']
        ds_swath_zdt = swaths_grp.create_dataset(
            'zeroDopplerTime', data=zdt
        )
        ds_swath_zdt.attrs['units'] = b'seconds since 2008-10-12 00:00:00'
        swaths_grp.create_dataset(
            'zeroDopplerTimeSpacing', data=0.000658
        )

        # Orbit
        orbit_grp = rslc.create_group('metadata/orbit')
        orbit_pos = rng.standard_normal((8, 3)) * 1e6
        orbit_vel = rng.standard_normal((8, 3)) * 1e3
        orbit_time = np.linspace(21960.0, 22380.0, 8)
        orbit_grp.create_dataset('position', data=orbit_pos)
        orbit_grp.create_dataset('velocity', data=orbit_vel)
        ds_ot = orbit_grp.create_dataset('time', data=orbit_time)
        ds_ot.attrs['units'] = b'seconds since 2008-10-12 00:00:00'
        orbit_grp.create_dataset('orbitType', data=b'DOE')
        orbit_grp.create_dataset('interpMethod', data=b'Hermite')

        # Attitude
        att_grp = rslc.create_group('metadata/attitude')
        att_grp.create_dataset(
            'quaternions', data=rng.random((22, 4))
        )
        att_grp.create_dataset(
            'eulerAngles', data=rng.random((22, 3))
        )
        att_time = np.linspace(21960.0, 22380.0, 22)
        ds_at = att_grp.create_dataset('time', data=att_time)
        ds_at.attrs['units'] = b'seconds since 2008-10-12 00:00:00'
        att_grp.create_dataset('attitudeType', data=b'quaternion')

        # Geolocation grid
        geo_grp = rslc.create_group('metadata/geolocationGrid')
        geo_grp.create_dataset(
            'coordinateX',
            data=rng.uniform(-120.0, -117.0, (5, 10, 8)),
        )
        geo_grp.create_dataset(
            'coordinateY',
            data=rng.uniform(34.0, 36.0, (5, 10, 8)),
        )
        geo_grp.create_dataset('epsg', data=np.int32(4326))
        geo_grp.create_dataset(
            'slantRange',
            data=np.linspace(741000.0, 775000.0, 8),
        )
        geo_grp.create_dataset(
            'zeroDopplerTime',
            data=np.linspace(22151.0, 22165.0, 10),
        )
        geo_grp.create_dataset(
            'heightAboveEllipsoid',
            data=np.linspace(-500.0, 9000.0, 5),
        )
        geo_grp.create_dataset(
            'incidenceAngle',
            data=rng.uniform(20.0, 30.0, (5, 10, 8)).astype(np.float32),
        )
        geo_grp.create_dataset(
            'elevationAngle',
            data=rng.uniform(18.0, 26.0, (5, 10, 8)).astype(np.float32),
        )

        # Calibration
        cal_geom = rslc.create_group(
            'metadata/calibrationInformation/geometry'
        )
        cal_geom.create_dataset(
            'sigma0', data=rng.random((10, 8)).astype(np.float32)
        )
        cal_geom.create_dataset(
            'beta0', data=rng.random((10, 8)).astype(np.float32)
        )
        cal_geom.create_dataset(
            'gamma0', data=rng.random((10, 8)).astype(np.float32)
        )

        # Processing information
        algo_grp = rslc.create_group(
            'metadata/processingInformation/algorithms'
        )
        algo_grp.create_dataset(
            'softwareVersion', data=b'isce3 v0.14.0'
        )
        algo_grp.create_dataset(
            'rangeCompression', data=b'chirp scaling'
        )

    return filepath, slc_hh, slc_hv


@pytest.fixture
def synthetic_nisar_gslc(tmp_path):
    """Create a minimal synthetic NISAR GSLC HDF5 file.

    Returns (filepath, expected_hh_data).
    """
    if not _HAS_H5PY:
        pytest.skip("h5py not installed")

    filepath = tmp_path / 'NISAR_L2_GSLC_TEST.h5'
    rng = np.random.default_rng(99)

    slc_hh = (
        rng.standard_normal((_GSLC_ROWS, _GSLC_COLS))
        + 1j * rng.standard_normal((_GSLC_ROWS, _GSLC_COLS))
    ).astype(np.complex64)

    mask = np.ones((_GSLC_ROWS, _GSLC_COLS), dtype=np.uint8)
    mask[:5, :] = 0  # Some invalid rows

    with h5py.File(str(filepath), 'w') as f:
        f.attrs['Conventions'] = 'CF-1.7'
        f.attrs['mission_name'] = 'NISAR'
        f.attrs['title'] = 'NISAR L2 GSLC Product'

        # Identification
        id_grp = f.create_group('science/LSAR/identification')
        id_grp.create_dataset('missionId', data=b'NISAR')
        id_grp.create_dataset('productType', data=b'GSLC')
        id_grp.create_dataset('radarBand', data=b'L')
        id_grp.create_dataset('isGeocoded', data=b'True')
        id_grp.create_dataset('orbitPassDirection', data=b'ascending')
        id_grp.create_dataset('lookDirection', data=b'right')

        # GSLC product group
        gslc = f.create_group('science/LSAR/GSLC')

        # Grid data - frequency A
        freq_a = gslc.create_group('grids/frequencyA')
        freq_a.create_dataset(
            'listOfPolarizations', data=[b'HH']
        )
        freq_a.create_dataset('HH', data=slc_hh)
        freq_a.create_dataset('mask', data=mask)

        x_coords = np.linspace(500000.0, 505000.0, _GSLC_COLS)
        y_coords = np.linspace(3700000.0, 3706000.0, _GSLC_ROWS)
        freq_a.create_dataset('xCoordinates', data=x_coords)
        freq_a.create_dataset('yCoordinates', data=y_coords)
        freq_a.create_dataset('xCoordinateSpacing', data=5.0)
        freq_a.create_dataset('yCoordinateSpacing', data=-10.0)
        freq_a.create_dataset('projection', data=np.uint32(32611))
        freq_a.create_dataset('centerFrequency', data=1.27e9)
        freq_a.create_dataset('rangeBandwidth', data=28e6)
        freq_a.create_dataset('slantRangeSpacing', data=4.684)

        # Orbit (minimal)
        orbit_grp = gslc.create_group('metadata/orbit')
        orbit_grp.create_dataset(
            'position', data=rng.standard_normal((4, 3)) * 1e6
        )
        orbit_grp.create_dataset(
            'velocity', data=rng.standard_normal((4, 3)) * 1e3
        )
        ds_t = orbit_grp.create_dataset(
            'time', data=np.linspace(0.0, 30.0, 4)
        )
        ds_t.attrs['units'] = b'seconds since 2008-10-12 00:00:00'

        # Calibration
        cal_geom = gslc.create_group(
            'metadata/calibrationInformation/geometry'
        )
        cal_geom.create_dataset(
            'sigma0', data=rng.random((10, 8)).astype(np.float32)
        )

    return filepath, slc_hh, mask


# ===================================================================
# Metadata dataclass tests
# ===================================================================

class TestNISARMetadata:
    """Test NISARMetadata dataclass construction and dict-like access."""

    def test_basic_construction(self):
        meta = NISARMetadata(
            format='NISAR_RSLC',
            rows=100,
            cols=200,
            dtype='complex64',
            product_type='RSLC',
            radar_band='LSAR',
        )
        assert meta.format == 'NISAR_RSLC'
        assert meta.rows == 100
        assert meta.cols == 200
        assert meta.dtype == 'complex64'
        assert meta.product_type == 'RSLC'
        assert meta.radar_band == 'LSAR'

    def test_dict_like_access(self):
        meta = NISARMetadata(
            format='NISAR_GSLC',
            rows=50,
            cols=60,
            dtype='complex64',
            product_type='GSLC',
            crs='EPSG:32611',
        )
        assert meta['format'] == 'NISAR_GSLC'
        assert meta['rows'] == 50
        assert meta['product_type'] == 'GSLC'
        assert 'crs' in meta
        assert meta.get('nodata', -1) == -1

    def test_identification_fields(self):
        ident = NISARIdentification(
            mission_id='NISAR',
            product_type='RSLC',
            radar_band='L',
            look_direction='right',
            orbit_pass_direction='ascending',
            absolute_orbit_number=1234,
            track_number=30,
            frame_number=19,
            is_geocoded=False,
        )
        assert ident.mission_id == 'NISAR'
        assert ident.absolute_orbit_number == 1234
        assert ident.is_geocoded is False

    def test_orbit_fields(self):
        orbit = NISAROrbit(
            time=np.array([0.0, 10.0, 20.0]),
            position=np.zeros((3, 3)),
            velocity=np.ones((3, 3)),
            reference_epoch='seconds since 2025-01-01 00:00:00',
            orbit_type='DOE',
        )
        assert orbit.time.shape == (3,)
        assert orbit.position.shape == (3, 3)
        assert orbit.reference_epoch == 'seconds since 2025-01-01 00:00:00'

    def test_swath_parameters_fields(self):
        swath = NISARSwathParameters(
            acquired_center_frequency=1.27e9,
            slant_range_spacing=4.684,
            polarizations=['HH', 'HV'],
        )
        assert swath.acquired_center_frequency == 1.27e9
        assert swath.polarizations == ['HH', 'HV']

    def test_grid_parameters_fields(self):
        grid = NISARGridParameters(
            x_coordinate_spacing=5.0,
            y_coordinate_spacing=-10.0,
            epsg=32611,
        )
        assert grid.epsg == 32611
        assert grid.y_coordinate_spacing == -10.0

    def test_full_metadata_with_all_sections(self):
        meta = NISARMetadata(
            format='NISAR_RSLC',
            rows=100,
            cols=200,
            dtype='complex64',
            product_type='RSLC',
            radar_band='LSAR',
            frequency='A',
            polarization='HH',
            identification=NISARIdentification(mission_id='NISAR'),
            orbit=NISAROrbit(orbit_type='DOE'),
            attitude=NISARAttitude(attitude_type='quaternion'),
            swath_parameters=NISARSwathParameters(
                slant_range_spacing=4.684,
            ),
            calibration=NISARCalibration(),
            processing_info=NISARProcessingInfo(
                software_version='isce3 v0.14.0',
            ),
        )
        assert meta.identification.mission_id == 'NISAR'
        assert meta.orbit.orbit_type == 'DOE'
        assert meta.swath_parameters.slant_range_spacing == 4.684
        assert meta.processing_info.software_version == 'isce3 v0.14.0'


# ===================================================================
# RSLC reader tests
# ===================================================================

@requires_h5py
class TestNISARReaderRSLC:
    """Test NISARReader with synthetic RSLC data."""

    def test_opens_and_populates_metadata(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            assert reader.metadata is not None
            assert isinstance(reader.metadata, NISARMetadata)

    def test_product_type_detected(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            assert reader.get_product_type() == 'RSLC'
            assert reader.metadata.product_type == 'RSLC'
            assert reader.metadata.format == 'NISAR_RSLC'

    def test_radar_band_detected(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            assert reader.get_radar_band() == 'LSAR'
            assert reader.metadata.radar_band == 'LSAR'

    def test_frequency_auto_detected(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            assert reader.metadata.frequency == 'A'
            assert reader.get_available_frequencies() == ['A']

    def test_polarization_auto_detected(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            # First available is HH
            assert reader.metadata.polarization == 'HH'
            assert reader.get_available_polarizations() == ['HH', 'HV']

    def test_explicit_polarization_selection(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, slc_hv = synthetic_nisar_rslc
        with NISARReader(filepath, polarization='HV') as reader:
            assert reader.metadata.polarization == 'HV'
            chip = reader.read_chip(0, 5, 0, 5)
            np.testing.assert_array_equal(chip, slc_hv[:5, :5])

    def test_identification_parsed(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            ident = reader.metadata.identification
            assert ident is not None
            assert ident.mission_id == 'NISAR'
            assert ident.product_type == 'RSLC'
            assert ident.radar_band == 'L'
            assert ident.look_direction == 'right'
            assert ident.orbit_pass_direction == 'ascending'
            assert ident.absolute_orbit_number == 1234
            assert ident.track_number == 30
            assert ident.frame_number == 19
            assert ident.is_geocoded is False
            assert ident.granule_id is not None
            assert 'NISAR' in ident.granule_id

    def test_orbit_parsed(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            orbit = reader.metadata.orbit
            assert orbit is not None
            assert orbit.position.shape == (8, 3)
            assert orbit.velocity.shape == (8, 3)
            assert orbit.time.shape == (8,)
            assert orbit.orbit_type == 'DOE'
            assert orbit.interp_method == 'Hermite'
            assert 'seconds since' in orbit.reference_epoch

    def test_attitude_parsed(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            att = reader.metadata.attitude
            assert att is not None
            assert att.quaternions.shape == (22, 4)
            assert att.euler_angles.shape == (22, 3)
            assert att.attitude_type == 'quaternion'

    def test_swath_parameters_parsed(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            sp = reader.metadata.swath_parameters
            assert sp is not None
            assert sp.acquired_center_frequency == pytest.approx(1.27e9)
            assert sp.acquired_range_bandwidth == pytest.approx(28e6)
            assert sp.slant_range_spacing == pytest.approx(4.684)
            assert sp.polarizations == ['HH', 'HV']
            assert sp.slant_range is not None
            assert sp.slant_range.shape == (_RSLC_COLS,)
            assert sp.zero_doppler_time is not None
            assert 'seconds since' in sp.zero_doppler_time_reference_epoch

    def test_geolocation_grid_parsed(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            geo = reader.metadata.geolocation_grid
            assert geo is not None
            assert geo.epsg == 4326
            assert geo.coordinate_x is not None
            assert geo.coordinate_y is not None
            assert geo.height_above_ellipsoid is not None
            assert geo.incidence_angle is not None

    def test_calibration_parsed(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            cal = reader.metadata.calibration
            assert cal is not None
            assert cal.sigma0 is not None
            assert cal.beta0 is not None
            assert cal.gamma0 is not None

    def test_processing_info_parsed(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            pi = reader.metadata.processing_info
            assert pi is not None
            assert 'isce3' in pi.software_version
            assert pi.algorithms is not None
            assert 'rangeCompression' in pi.algorithms

    def test_read_chip_returns_complex64(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            chip = reader.read_chip(0, 10, 0, 10)
            assert chip.dtype == np.complex64

    def test_read_chip_correct_shape(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            chip = reader.read_chip(5, 15, 10, 25)
            assert chip.shape == (10, 15)

    def test_read_chip_correct_values(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, slc_hh, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            chip = reader.read_chip(0, 10, 0, 10)
            np.testing.assert_array_equal(chip, slc_hh[:10, :10])

    def test_read_full(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, slc_hh, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            full = reader.read_full()
            assert full.shape == (_RSLC_ROWS, _RSLC_COLS)
            np.testing.assert_array_equal(full, slc_hh)

    def test_get_shape(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            assert reader.get_shape() == (_RSLC_ROWS, _RSLC_COLS)

    def test_get_dtype(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            assert reader.get_dtype() == np.dtype('complex64')

    def test_context_manager(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        reader = NISARReader(filepath)
        assert reader._file is not None
        reader.close()
        assert reader._file is None

    def test_grid_params_none_for_rslc(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            assert reader.metadata.grid_parameters is None

    def test_read_mask_returns_none_for_rslc(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            assert reader.read_mask() is None

    def test_no_crs_for_rslc(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            assert reader.metadata.crs is None


# ===================================================================
# GSLC reader tests
# ===================================================================

@requires_h5py
class TestNISARReaderGSLC:
    """Test NISARReader with synthetic GSLC data."""

    def test_opens_gslc(self, synthetic_nisar_gslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_gslc
        with NISARReader(filepath) as reader:
            assert reader.metadata is not None
            assert reader.get_product_type() == 'GSLC'

    def test_product_type_detected(self, synthetic_nisar_gslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_gslc
        with NISARReader(filepath) as reader:
            assert reader.metadata.format == 'NISAR_GSLC'
            assert reader.metadata.product_type == 'GSLC'

    def test_crs_set_for_gslc(self, synthetic_nisar_gslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_gslc
        with NISARReader(filepath) as reader:
            assert reader.metadata.crs == 'EPSG:32611'

    def test_grid_parameters_parsed(self, synthetic_nisar_gslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_gslc
        with NISARReader(filepath) as reader:
            gp = reader.metadata.grid_parameters
            assert gp is not None
            assert gp.epsg == 32611
            assert gp.x_coordinate_spacing == pytest.approx(5.0)
            assert gp.y_coordinate_spacing == pytest.approx(-10.0)
            assert gp.x_coordinates is not None
            assert gp.x_coordinates.shape == (_GSLC_COLS,)
            assert gp.y_coordinates is not None
            assert gp.y_coordinates.shape == (_GSLC_ROWS,)
            assert gp.center_frequency == pytest.approx(1.27e9)
            assert gp.polarizations == ['HH']

    def test_read_chip_gslc(self, synthetic_nisar_gslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, slc_hh, _ = synthetic_nisar_gslc
        with NISARReader(filepath) as reader:
            chip = reader.read_chip(0, 10, 0, 10)
            assert chip.dtype == np.complex64
            assert chip.shape == (10, 10)
            np.testing.assert_array_equal(chip, slc_hh[:10, :10])

    def test_read_mask(self, synthetic_nisar_gslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, expected_mask = synthetic_nisar_gslc
        with NISARReader(filepath) as reader:
            mask = reader.read_mask()
            assert mask is not None
            assert mask.shape == (_GSLC_ROWS, _GSLC_COLS)
            assert mask.dtype == np.uint8
            np.testing.assert_array_equal(mask, expected_mask)

    def test_swath_params_none_for_gslc(self, synthetic_nisar_gslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_gslc
        with NISARReader(filepath) as reader:
            assert reader.metadata.swath_parameters is None
            assert reader.metadata.geolocation_grid is None

    def test_get_shape_gslc(self, synthetic_nisar_gslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_gslc
        with NISARReader(filepath) as reader:
            assert reader.get_shape() == (_GSLC_ROWS, _GSLC_COLS)


# ===================================================================
# Error handling tests
# ===================================================================

@requires_h5py
class TestNISARReaderErrors:
    """Test error conditions for NISARReader."""

    def test_missing_file_raises(self, tmp_path):
        from grdl.IO.sar.nisar import NISARReader
        with pytest.raises(FileNotFoundError):
            NISARReader(tmp_path / 'nonexistent.h5')

    def test_invalid_hdf5_raises(self, tmp_path):
        from grdl.IO.sar.nisar import NISARReader
        bad_file = tmp_path / 'bad.h5'
        bad_file.write_bytes(b'not an hdf5 file')
        with pytest.raises((ValueError, OSError)):
            NISARReader(bad_file)

    def test_no_science_group_raises(self, tmp_path):
        from grdl.IO.sar.nisar import NISARReader
        filepath = tmp_path / 'empty.h5'
        with h5py.File(str(filepath), 'w') as f:
            f.create_group('other')
        with pytest.raises((ValueError, KeyError)):
            NISARReader(filepath)

    def test_invalid_frequency_raises(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with pytest.raises(ValueError, match="Frequency"):
            NISARReader(filepath, frequency='B')

    def test_invalid_polarization_raises(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with pytest.raises(ValueError, match="Polarization"):
            NISARReader(filepath, polarization='VV')

    def test_chip_negative_indices(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            with pytest.raises(ValueError, match="non-negative"):
                reader.read_chip(-1, 10, 0, 10)

    def test_chip_exceeds_dimensions(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import NISARReader
        filepath, _, _ = synthetic_nisar_rslc
        with NISARReader(filepath) as reader:
            with pytest.raises(ValueError, match="exceed"):
                reader.read_chip(0, 10, 0, _RSLC_COLS + 10)


# ===================================================================
# Factory function tests
# ===================================================================

@requires_h5py
class TestOpenNisar:
    """Test open_nisar() factory function."""

    def test_open_nisar_rslc(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import open_nisar
        filepath, _, _ = synthetic_nisar_rslc
        reader = open_nisar(filepath)
        assert reader.get_product_type() == 'RSLC'
        reader.close()

    def test_open_nisar_gslc(self, synthetic_nisar_gslc):
        from grdl.IO.sar.nisar import open_nisar
        filepath, _, _ = synthetic_nisar_gslc
        reader = open_nisar(filepath)
        assert reader.get_product_type() == 'GSLC'
        reader.close()

    def test_open_nisar_with_explicit_params(self, synthetic_nisar_rslc):
        from grdl.IO.sar.nisar import open_nisar
        filepath, _, slc_hv = synthetic_nisar_rslc
        with open_nisar(filepath, frequency='A', polarization='HV') as r:
            assert r.metadata.frequency == 'A'
            assert r.metadata.polarization == 'HV'
            chip = r.read_chip(0, 5, 0, 5)
            np.testing.assert_array_equal(chip, slc_hv[:5, :5])


# ===================================================================
# ABC compliance test
# ===================================================================

class TestNISARReaderIsImageReader:
    """Verify NISARReader is a proper ImageReader subclass."""

    def test_is_subclass_of_image_reader(self):
        from grdl.IO.base import ImageReader
        from grdl.IO.sar.nisar import NISARReader
        assert issubclass(NISARReader, ImageReader)
