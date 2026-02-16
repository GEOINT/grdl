# -*- coding: utf-8 -*-
"""
Tests for Sentinel-1 IW SLC reader and metadata.

All tests use synthetic data — no real SAFE products required.

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
2026-02-16

Modified
--------
2026-02-16
"""

# Standard library
import shutil
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

# Third-party
import numpy as np
import pytest

try:
    import rasterio
    from rasterio.transform import from_bounds
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

# GRDL
from grdl.IO.models.sentinel1_slc import (
    Sentinel1SLCMetadata,
    S1SLCProductInfo,
    S1SLCSwathInfo,
    S1SLCBurst,
    S1SLCOrbitStateVector,
    S1SLCGeoGridPoint,
    S1SLCDopplerCentroid,
    S1SLCDopplerFmRate,
    S1SLCCalibrationVector,
    S1SLCNoiseRangeVector,
    S1SLCNoiseAzimuthVector,
)
from grdl.IO.models.common import XYZ


# ===================================================================
# Synthetic SAFE builder
# ===================================================================

# Small dimensions for tests
_LINES_PER_BURST = 20
_SAMPLES_PER_BURST = 30
_NUM_BURSTS = 2
_TOTAL_LINES = _LINES_PER_BURST * _NUM_BURSTS
_TOTAL_SAMPLES = _SAMPLES_PER_BURST


def _build_annotation_xml() -> str:
    """Build a minimal but valid Sentinel-1 annotation XML string."""
    # Build burst entries
    burst_entries = []
    for i in range(_NUM_BURSTS):
        fvs = ' '.join(['5'] * _LINES_PER_BURST)
        lvs = ' '.join([str(_SAMPLES_PER_BURST - 6)] * _LINES_PER_BURST)
        burst_entries.append(textwrap.dedent(f"""\
            <burst>
              <azimuthTime>2026-02-01T14:24:{32 + i}.000000</azimuthTime>
              <azimuthAnxTime>{1234.0 + i * 2.8:.6f}</azimuthAnxTime>
              <sensingTime>2026-02-01T14:24:{32 + i}.000000</sensingTime>
              <byteOffset>{i * _LINES_PER_BURST * _SAMPLES_PER_BURST * 4}</byteOffset>
              <firstValidSample>{fvs}</firstValidSample>
              <lastValidSample>{lvs}</lastValidSample>
            </burst>"""))

    bursts_xml = '\n            '.join(burst_entries)

    # Build geolocation grid (4 corner points)
    grid_points = []
    for line, pixel, lat, lon in [
        (0, 0, 30.0, 50.0),
        (0, _TOTAL_SAMPLES - 1, 30.0, 50.5),
        (_TOTAL_LINES - 1, 0, 30.5, 50.0),
        (_TOTAL_LINES - 1, _TOTAL_SAMPLES - 1, 30.5, 50.5),
    ]:
        grid_points.append(textwrap.dedent(f"""\
            <geolocationGridPoint>
              <azimuthTime>2026-02-01T14:24:32.000000</azimuthTime>
              <slantRangeTime>5.33e-03</slantRangeTime>
              <line>{line}</line>
              <pixel>{pixel}</pixel>
              <latitude>{lat}</latitude>
              <longitude>{lon}</longitude>
              <height>100.0</height>
              <incidenceAngle>34.0</incidenceAngle>
              <elevationAngle>30.0</elevationAngle>
            </geolocationGridPoint>"""))

    grid_xml = '\n            '.join(grid_points)

    return textwrap.dedent(f"""\
    <?xml version="1.0" encoding="UTF-8"?>
    <product>
      <adsHeader>
        <missionId>S1A</missionId>
        <productType>SLC</productType>
        <polarisation>VV</polarisation>
        <mode>IW</mode>
        <swath>IW1</swath>
        <startTime>2026-02-01T14:24:32.000000</startTime>
        <stopTime>2026-02-01T14:24:57.000000</stopTime>
        <absoluteOrbitNumber>63027</absoluteOrbitNumber>
        <missionDataTakeId>518336</missionDataTakeId>
        <imageNumber>004</imageNumber>
      </adsHeader>
      <generalAnnotation>
        <productInformation>
          <pass>Ascending</pass>
          <rangeSamplingRate>64345238.12571428</rangeSamplingRate>
          <radarFrequency>5405000454.33435</radarFrequency>
          <azimuthSteeringRate>1.590368784</azimuthSteeringRate>
        </productInformation>
        <orbitList count="3">
          <orbit>
            <time>2026-02-01T14:24:23.000000</time>
            <frame>Earth Fixed</frame>
            <position><x>4164056.4</x><y>5282058.8</y><z>2191522.7</z></position>
            <velocity><x>-162.8</x><y>-2812.7</y><z>7055.8</z></velocity>
          </orbit>
          <orbit>
            <time>2026-02-01T14:24:33.000000</time>
            <frame>Earth Fixed</frame>
            <position><x>4162173.4</x><y>5253637.2</y><z>2262000.0</z></position>
            <velocity><x>-200.0</x><y>-2900.0</y><z>7000.0</z></velocity>
          </orbit>
          <orbit>
            <time>2026-02-01T14:24:43.000000</time>
            <frame>Earth Fixed</frame>
            <position><x>4160000.0</x><y>5225000.0</y><z>2332000.0</z></position>
            <velocity><x>-230.0</x><y>-2950.0</y><z>6950.0</z></velocity>
          </orbit>
        </orbitList>
        <azimuthFmRateList count="1">
          <azimuthFmRate>
            <azimuthTime>2026-02-01T14:24:32.000000</azimuthTime>
            <t0>5.33e-03</t0>
            <azimuthFmRatePolynomial count="3">-2000.0 400000.0 -5.0e+07</azimuthFmRatePolynomial>
          </azimuthFmRate>
        </azimuthFmRateList>
      </generalAnnotation>
      <imageAnnotation>
        <imageInformation>
          <numberOfLines>{_TOTAL_LINES}</numberOfLines>
          <numberOfSamples>{_TOTAL_SAMPLES}</numberOfSamples>
          <rangePixelSpacing>2.329562</rangePixelSpacing>
          <azimuthPixelSpacing>13.995907</azimuthPixelSpacing>
          <azimuthTimeInterval>2.055556e-03</azimuthTimeInterval>
          <slantRangeTime>5.330568e-03</slantRangeTime>
          <incidenceAngleMidSwath>34.007</incidenceAngleMidSwath>
          <azimuthFrequency>486.486</azimuthFrequency>
        </imageInformation>
      </imageAnnotation>
      <dopplerCentroid>
        <dcEstimateList count="1">
          <dcEstimate>
            <azimuthTime>2026-02-01T14:24:32.000000</azimuthTime>
            <t0>5.33e-03</t0>
            <dataDcPolynomial count="3">10.5 -200.0 50000.0</dataDcPolynomial>
          </dcEstimate>
        </dcEstimateList>
      </dopplerCentroid>
      <swathTiming>
        <linesPerBurst>{_LINES_PER_BURST}</linesPerBurst>
        <samplesPerBurst>{_SAMPLES_PER_BURST}</samplesPerBurst>
        <burstList count="{_NUM_BURSTS}">
            {bursts_xml}
        </burstList>
      </swathTiming>
      <geolocationGrid>
        <geolocationGridPointList count="4">
            {grid_xml}
        </geolocationGridPointList>
      </geolocationGrid>
    </product>
    """).lstrip()


def _build_calibration_xml() -> str:
    """Build a minimal calibration XML."""
    pixels = ' '.join([str(i * 10) for i in range(3)])
    sigma = ' '.join(['100.0'] * 3)
    beta = ' '.join(['90.0'] * 3)
    gamma = ' '.join(['110.0'] * 3)
    dn = ' '.join(['1.0'] * 3)

    return textwrap.dedent(f"""\
    <?xml version="1.0" encoding="UTF-8"?>
    <calibration>
      <adsHeader>
        <missionId>S1A</missionId>
        <productType>SLC</productType>
        <polarisation>VV</polarisation>
        <mode>IW</mode>
        <swath>IW1</swath>
      </adsHeader>
      <calibrationInformation>
        <absoluteCalibrationConstant>1.0</absoluteCalibrationConstant>
      </calibrationInformation>
      <calibrationVectorList count="1">
        <calibrationVector>
          <azimuthTime>2026-02-01T14:24:32.000000</azimuthTime>
          <line>0</line>
          <pixel count="3">{pixels}</pixel>
          <sigmaNought count="3">{sigma}</sigmaNought>
          <betaNought count="3">{beta}</betaNought>
          <gamma count="3">{gamma}</gamma>
          <dn count="3">{dn}</dn>
        </calibrationVector>
      </calibrationVectorList>
    </calibration>
    """).lstrip()


def _build_noise_xml() -> str:
    """Build a minimal noise XML."""
    pixels = ' '.join([str(i * 10) for i in range(3)])
    noise_lut = ' '.join(['0.001'] * 3)
    az_lines = ' '.join([str(i) for i in range(_TOTAL_LINES)])
    az_lut = ' '.join(['0.002'] * _TOTAL_LINES)

    return textwrap.dedent(f"""\
    <?xml version="1.0" encoding="UTF-8"?>
    <noise>
      <adsHeader>
        <missionId>S1A</missionId>
        <productType>SLC</productType>
        <polarisation>VV</polarisation>
        <mode>IW</mode>
        <swath>IW1</swath>
      </adsHeader>
      <noiseRangeVectorList count="1">
        <noiseRangeVector>
          <azimuthTime>2026-02-01T14:24:32.000000</azimuthTime>
          <line>0</line>
          <pixel count="3">{pixels}</pixel>
          <noiseRangeLut count="3">{noise_lut}</noiseRangeLut>
        </noiseRangeVector>
      </noiseRangeVectorList>
      <noiseAzimuthVectorList count="1">
        <noiseAzimuthVector>
          <swath>IW1</swath>
          <firstAzimuthLine>0</firstAzimuthLine>
          <firstRangeSample>0</firstRangeSample>
          <lastAzimuthLine>{_TOTAL_LINES - 1}</lastAzimuthLine>
          <lastRangeSample>{_TOTAL_SAMPLES - 1}</lastRangeSample>
          <line count="{_TOTAL_LINES}">{az_lines}</line>
          <noiseAzimuthLut count="{_TOTAL_LINES}">{az_lut}</noiseAzimuthLut>
        </noiseAzimuthVector>
      </noiseAzimuthVectorList>
    </noise>
    """).lstrip()


@pytest.fixture
def synthetic_safe(tmp_path):
    """Create a synthetic SAFE directory for testing.

    Yields the path to the .SAFE directory.
    """
    if not _HAS_RASTERIO:
        pytest.skip("rasterio required for Sentinel-1 SLC tests")

    safe_dir = tmp_path / 'S1A_IW_SLC__1SDV_20260201_TEST.SAFE'
    ann_dir = safe_dir / 'annotation'
    cal_dir = ann_dir / 'calibration'
    meas_dir = safe_dir / 'measurement'

    safe_dir.mkdir()
    ann_dir.mkdir()
    cal_dir.mkdir()
    meas_dir.mkdir()

    # manifest.safe (minimal)
    (safe_dir / 'manifest.safe').write_text(
        '<?xml version="1.0"?><xfdu:XFDU/>'
    )

    # Annotation XML
    ann_name = 's1a-iw1-slc-vv-20260201t142432-20260201t142457-063027-07e8c0-004.xml'
    (ann_dir / ann_name).write_text(_build_annotation_xml())

    # Calibration XML
    cal_name = f'calibration-{ann_name}'
    (cal_dir / cal_name).write_text(_build_calibration_xml())

    # Noise XML
    noise_name = f'noise-{ann_name}'
    (cal_dir / noise_name).write_text(_build_noise_xml())

    # Measurement TIFF — write complex int16 data as 2-band int16
    tiff_name = ann_name.replace('.xml', '.tiff')
    tiff_path = meas_dir / tiff_name

    # Generate synthetic complex data
    rng = np.random.default_rng(42)
    real_part = rng.integers(-1000, 1000, (_TOTAL_LINES, _TOTAL_SAMPLES),
                             dtype=np.int16)
    imag_part = rng.integers(-1000, 1000, (_TOTAL_LINES, _TOTAL_SAMPLES),
                             dtype=np.int16)
    data = np.stack([real_part, imag_part], axis=0)  # (2, rows, cols)

    transform = from_bounds(50.0, 30.0, 50.5, 30.5,
                            _TOTAL_SAMPLES, _TOTAL_LINES)

    with rasterio.open(
        str(tiff_path), 'w',
        driver='GTiff',
        height=_TOTAL_LINES,
        width=_TOTAL_SAMPLES,
        count=2,
        dtype='int16',
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(data)

    yield safe_dir


# ===================================================================
# Metadata dataclass tests
# ===================================================================

class TestSentinel1SLCMetadata:
    """Test metadata dataclass construction and access."""

    def test_basic_construction(self):
        """Sentinel1SLCMetadata can be constructed with required fields."""
        meta = Sentinel1SLCMetadata(
            format='Sentinel-1_IW_SLC',
            rows=100,
            cols=200,
            dtype='complex64',
        )
        assert meta.rows == 100
        assert meta.cols == 200
        assert meta.format == 'Sentinel-1_IW_SLC'
        assert meta.num_bursts == 0

    def test_dict_like_access(self):
        """Metadata supports dict-like access."""
        meta = Sentinel1SLCMetadata(
            format='Sentinel-1_IW_SLC',
            rows=100,
            cols=200,
            dtype='complex64',
        )
        assert meta['rows'] == 100
        assert meta['format'] == 'Sentinel-1_IW_SLC'

    def test_product_info(self):
        """S1SLCProductInfo fields are accessible."""
        info = S1SLCProductInfo(
            mission='S1A',
            mode='IW',
            product_type='SLC',
            absolute_orbit=63027,
            orbit_pass='ASCENDING',
        )
        assert info.mission == 'S1A'
        assert info.absolute_orbit == 63027

    def test_burst(self):
        """S1SLCBurst stores valid sample arrays."""
        fvs = np.array([5, 5, -1, 5], dtype=np.int32)
        lvs = np.array([20, 20, -1, 20], dtype=np.int32)
        burst = S1SLCBurst(
            index=0,
            first_valid_sample=fvs,
            last_valid_sample=lvs,
            first_line=0,
            last_line=4,
            lines_per_burst=4,
            samples_per_burst=25,
        )
        assert burst.lines_per_burst == 4
        assert burst.first_valid_sample[2] == -1

    def test_geo_grid_point(self):
        """S1SLCGeoGridPoint stores coordinates."""
        pt = S1SLCGeoGridPoint(
            line=0.0, pixel=0.0,
            latitude=30.0, longitude=50.0, height=100.0,
        )
        assert pt.latitude == 30.0
        assert pt.longitude == 50.0

    def test_orbit_state_vector(self):
        """S1SLCOrbitStateVector stores position and velocity."""
        osv = S1SLCOrbitStateVector(
            time='2026-02-01T14:24:33.000000',
            position=XYZ(x=4164056.4, y=5282058.8, z=2191522.7),
            velocity=XYZ(x=-162.8, y=-2812.7, z=7055.8),
        )
        assert osv.position.x == pytest.approx(4164056.4)

    def test_calibration_vector(self):
        """S1SLCCalibrationVector stores numpy arrays."""
        cv = S1SLCCalibrationVector(
            pixel=np.array([0, 10, 20], dtype=np.int32),
            sigma_nought=np.array([100.0, 100.0, 100.0]),
        )
        assert cv.pixel.shape == (3,)
        assert cv.sigma_nought[0] == 100.0

    def test_doppler_centroid(self):
        """S1SLCDopplerCentroid stores polynomial coefficients."""
        dc = S1SLCDopplerCentroid(
            t0=5.33e-3,
            coefficients=np.array([10.5, -200.0, 50000.0]),
        )
        assert dc.coefficients.shape == (3,)

    def test_full_metadata_with_all_sections(self):
        """Sentinel1SLCMetadata with all sections populated."""
        meta = Sentinel1SLCMetadata(
            format='Sentinel-1_IW_SLC',
            rows=40, cols=30, dtype='complex64',
            product_info=S1SLCProductInfo(mission='S1A'),
            swath_info=S1SLCSwathInfo(swath='IW1', polarization='VV',
                                      lines=40, samples=30),
            bursts=[S1SLCBurst(index=0), S1SLCBurst(index=1)],
            orbit_state_vectors=[S1SLCOrbitStateVector(time='t0')],
            geolocation_grid=[S1SLCGeoGridPoint(latitude=30.0)],
            doppler_centroids=[S1SLCDopplerCentroid(t0=0.005)],
            doppler_fm_rates=[S1SLCDopplerFmRate(t0=0.005)],
            calibration_vectors=[S1SLCCalibrationVector(line=0)],
            noise_range_vectors=[S1SLCNoiseRangeVector(line=0)],
            noise_azimuth_vectors=[S1SLCNoiseAzimuthVector(swath='IW1')],
            num_bursts=2,
            lines_per_burst=20,
            samples_per_burst=30,
        )
        assert meta.num_bursts == 2
        assert meta.product_info.mission == 'S1A'
        assert len(meta.bursts) == 2


# ===================================================================
# Reader tests
# ===================================================================

@pytest.mark.skipif(not _HAS_RASTERIO,
                    reason="rasterio required for Sentinel-1 SLC")
class TestSentinel1SLCReader:
    """Test reader with synthetic SAFE data."""

    def test_reader_loads(self, synthetic_safe):
        """Reader opens synthetic SAFE and populates metadata."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            meta = reader.metadata
            assert meta.format == 'Sentinel-1_IW_SLC'
            assert meta.rows == _TOTAL_LINES
            assert meta.cols == _TOTAL_SAMPLES
            assert meta.dtype == 'complex64'

    def test_product_info_parsed(self, synthetic_safe):
        """Product info is extracted from adsHeader."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            pi = reader.metadata.product_info
            assert pi.mission == 'S1A'
            assert pi.mode == 'IW'
            assert pi.product_type == 'SLC'
            assert pi.absolute_orbit == 63027

    def test_swath_info_parsed(self, synthetic_safe):
        """Swath geometry is extracted from imageAnnotation."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            si = reader.metadata.swath_info
            assert si.swath == 'IW1'
            assert si.polarization == 'VV'
            assert si.range_pixel_spacing == pytest.approx(2.329562)
            assert si.azimuth_pixel_spacing == pytest.approx(13.995907)
            assert si.radar_frequency == pytest.approx(5.405e9, rel=1e-3)

    def test_bursts_parsed(self, synthetic_safe):
        """Burst list is correctly extracted."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            meta = reader.metadata
            assert meta.num_bursts == _NUM_BURSTS
            assert meta.lines_per_burst == _LINES_PER_BURST
            assert meta.samples_per_burst == _SAMPLES_PER_BURST
            assert len(meta.bursts) == _NUM_BURSTS

            b0 = meta.bursts[0]
            assert b0.index == 0
            assert b0.first_line == 0
            assert b0.last_line == _LINES_PER_BURST
            assert b0.first_valid_sample is not None
            assert b0.first_valid_sample.shape == (_LINES_PER_BURST,)

    def test_orbit_parsed(self, synthetic_safe):
        """Orbit state vectors are extracted."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            osvs = reader.metadata.orbit_state_vectors
            assert len(osvs) == 3
            assert osvs[0].position.x == pytest.approx(4164056.4)

    def test_geolocation_grid_parsed(self, synthetic_safe):
        """Geolocation grid points are extracted."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            grid = reader.metadata.geolocation_grid
            assert len(grid) == 4
            assert grid[0].latitude == pytest.approx(30.0)
            assert grid[0].longitude == pytest.approx(50.0)

    def test_doppler_centroids_parsed(self, synthetic_safe):
        """Doppler centroid estimates are extracted."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            dcs = reader.metadata.doppler_centroids
            assert len(dcs) == 1
            assert dcs[0].t0 == pytest.approx(5.33e-3)
            assert dcs[0].coefficients.shape == (3,)

    def test_fm_rates_parsed(self, synthetic_safe):
        """Azimuth FM rates are extracted."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            fms = reader.metadata.doppler_fm_rates
            assert len(fms) == 1
            assert fms[0].coefficients[0] == pytest.approx(-2000.0)

    def test_calibration_parsed(self, synthetic_safe):
        """Calibration vectors are extracted."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            cvs = reader.metadata.calibration_vectors
            assert len(cvs) == 1
            assert cvs[0].sigma_nought is not None
            assert cvs[0].sigma_nought[0] == pytest.approx(100.0)

    def test_noise_parsed(self, synthetic_safe):
        """Noise vectors are extracted."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            nr = reader.metadata.noise_range_vectors
            na = reader.metadata.noise_azimuth_vectors
            assert len(nr) == 1
            assert len(na) == 1
            assert na[0].swath == 'IW1'

    def test_read_chip(self, synthetic_safe):
        """read_chip returns complex64 data with correct shape."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            chip = reader.read_chip(0, 10, 0, 15)
            assert chip.shape == (10, 15)
            assert chip.dtype == np.complex64
            # Should have non-zero values (synthetic data)
            assert np.any(chip != 0)

    def test_read_chip_full(self, synthetic_safe):
        """read_chip can read entire image."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            full = reader.read_chip(
                0, _TOTAL_LINES, 0, _TOTAL_SAMPLES
            )
            assert full.shape == (_TOTAL_LINES, _TOTAL_SAMPLES)

    def test_read_burst(self, synthetic_safe):
        """read_burst returns a burst-sized complex array."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            burst_data = reader.read_burst(0, apply_valid_mask=False)
            assert burst_data.shape == (
                _LINES_PER_BURST, _TOTAL_SAMPLES
            )
            assert burst_data.dtype == np.complex64

    def test_read_burst_with_masking(self, synthetic_safe):
        """read_burst zeros invalid border samples."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            burst_data = reader.read_burst(0, apply_valid_mask=True)
            # First 5 samples should be zeroed
            assert np.all(burst_data[:, :5] == 0)
            # Last few samples should be zeroed
            assert np.all(burst_data[:, _SAMPLES_PER_BURST - 5:] == 0)

    def test_get_shape(self, synthetic_safe):
        """get_shape returns (rows, cols)."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            assert reader.get_shape() == (_TOTAL_LINES, _TOTAL_SAMPLES)

    def test_get_dtype(self, synthetic_safe):
        """get_dtype returns complex64."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            assert reader.get_dtype() == np.dtype('complex64')

    def test_get_burst_count(self, synthetic_safe):
        """get_burst_count returns correct number."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            assert reader.get_burst_count() == _NUM_BURSTS

    def test_available_swaths_and_pols(self, synthetic_safe):
        """Discovery methods return available swaths and polarizations."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            assert 'IW1' in reader.get_available_swaths()
            assert 'VV' in reader.get_available_polarizations()

    def test_context_manager(self, synthetic_safe):
        """Context manager opens and closes cleanly."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        reader = Sentinel1SLCReader(synthetic_safe)
        assert reader.metadata.rows == _TOTAL_LINES
        reader.close()


# ===================================================================
# Error handling tests
# ===================================================================

@pytest.mark.skipif(not _HAS_RASTERIO,
                    reason="rasterio required for Sentinel-1 SLC")
class TestSentinel1SLCReaderErrors:
    """Test error conditions."""

    def test_missing_directory(self, tmp_path):
        """FileNotFoundError for non-existent path."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with pytest.raises((FileNotFoundError, ValueError)):
            Sentinel1SLCReader(tmp_path / 'nonexistent.SAFE')

    def test_invalid_safe_no_manifest(self, tmp_path):
        """ValueError for directory without manifest.safe."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        fake_dir = tmp_path / 'fake.SAFE'
        fake_dir.mkdir()
        with pytest.raises(ValueError, match="manifest.safe"):
            Sentinel1SLCReader(fake_dir)

    def test_invalid_swath(self, synthetic_safe):
        """ValueError for unavailable swath."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with pytest.raises(ValueError, match="No annotation XML found"):
            Sentinel1SLCReader(synthetic_safe, swath='IW3')

    def test_invalid_polarization(self, synthetic_safe):
        """ValueError for unavailable polarization."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with pytest.raises(ValueError, match="No annotation XML found"):
            Sentinel1SLCReader(synthetic_safe, polarization='HH')

    def test_chip_negative_indices(self, synthetic_safe):
        """ValueError for negative start indices."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            with pytest.raises(ValueError, match="non-negative"):
                reader.read_chip(-1, 10, 0, 10)

    def test_chip_exceeds_dimensions(self, synthetic_safe):
        """ValueError for indices beyond image dimensions."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            with pytest.raises(ValueError, match="exceed"):
                reader.read_chip(0, _TOTAL_LINES + 1, 0, 10)

    def test_burst_index_out_of_range(self, synthetic_safe):
        """IndexError for invalid burst index."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        with Sentinel1SLCReader(synthetic_safe) as reader:
            with pytest.raises(IndexError):
                reader.read_burst(99)


# ===================================================================
# _to_complex conversion tests
# ===================================================================

class TestToComplex:
    """Test the _to_complex static method."""

    def test_already_complex(self):
        """Pass-through for complex input."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        data = np.array([[1 + 2j, 3 + 4j]], dtype=np.complex128)
        result = Sentinel1SLCReader._to_complex(data)
        assert result.dtype == np.complex64
        assert result[0, 0] == pytest.approx(1 + 2j)

    def test_two_band_int16(self):
        """Convert 2-band int16 (I, Q) to complex64."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        real = np.array([[10, 20]], dtype=np.int16)
        imag = np.array([[30, 40]], dtype=np.int16)
        data = np.stack([real, imag], axis=0)  # shape (2, 1, 2)
        result = Sentinel1SLCReader._to_complex(data)
        assert result.dtype == np.complex64
        assert result[0, 0] == pytest.approx(10 + 30j)
        assert result[0, 1] == pytest.approx(20 + 40j)

    def test_structured_array(self):
        """Convert structured dtype with real/imag fields."""
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        dt = np.dtype([('real', np.int16), ('imag', np.int16)])
        data = np.array([(5, 10), (15, 20)], dtype=dt)
        result = Sentinel1SLCReader._to_complex(data)
        assert result.dtype == np.complex64
        assert result[0] == pytest.approx(5 + 10j)
