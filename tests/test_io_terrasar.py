# -*- coding: utf-8 -*-
"""
Tests for TerraSAR-X / TanDEM-X reader and metadata.

All tests use synthetic data -- no real TSX/TDX products required.

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
2026-02-19

Modified
--------
2026-02-19
"""

# Standard library
import struct
import textwrap
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
from grdl.IO.models.terrasar import (
    TerraSARMetadata,
    TSXProductInfo,
    TSXSceneInfo,
    TSXImageInfo,
    TSXRadarParams,
    TSXOrbitStateVector,
    TSXGeoGridPoint,
    TSXCalibration,
    TSXDopplerInfo,
    TSXProcessingInfo,
)
from grdl.IO.models.common import XYZ, LatLonHAE


# ===================================================================
# Synthetic product constants
# ===================================================================

_NUM_ROWS = 32
_NUM_COLS = 48
_SAMPLE_SIZE = 2  # int16
_ANNOTATION_SIZE_PER_LINE = 8  # bytes of per-line annotation
_RTNB = _ANNOTATION_SIZE_PER_LINE + _NUM_COLS * 2 * _SAMPLE_SIZE
_HEADER_LINES = 4  # standard COSAR header occupies 4 range lines
_TNL = _HEADER_LINES + _NUM_ROWS


# ===================================================================
# Synthetic XML builders
# ===================================================================

def _build_main_annotation_xml(
    product_type: str = 'SSC',
    data_format: str = 'COSAR',
) -> str:
    """Build a minimal TerraSAR-X annotation XML string."""
    sample_type = 'COMPLEX' if product_type == 'SSC' else 'DETECTED'
    projection = 'SLANTRANGE' if product_type == 'SSC' else 'MAP'

    return textwrap.dedent(f"""\
    <?xml version="1.0" encoding="UTF-8"?>
    <level1Product>
      <generalHeader>
        <mission>TerraSAR-X</mission>
        <satellite>TSX-1</satellite>
        <generationTime>2026-02-19T10:00:00.000000Z</generationTime>
      </generalHeader>
      <productInfo>
        <productVariantInfo>
          <productType>{product_type}</productType>
        </productVariantInfo>
        <acquisitionInfo>
          <imagingMode>SM</imagingMode>
          <lookDirection>RIGHT</lookDirection>
          <polarisationMode>SINGLE</polarisationMode>
          <polarisationList>
            <polLayer>HH</polLayer>
          </polarisationList>
        </acquisitionInfo>
        <missionInfo>
          <orbitDirection>ASCENDING</orbitDirection>
          <absOrbit>12345</absOrbit>
        </missionInfo>
        <sceneInfo>
          <start>
            <timeUTC>2026-02-19T06:12:30.123456Z</timeUTC>
          </start>
          <stop>
            <timeUTC>2026-02-19T06:12:32.654321Z</timeUTC>
          </stop>
          <sceneCenterCoord>
            <lat>48.15</lat>
            <lon>11.58</lon>
            <incidenceAngle>35.5</incidenceAngle>
          </sceneCenterCoord>
          <sceneCornerCoord>
            <lat>48.10</lat><lon>11.50</lon><height>500.0</height>
          </sceneCornerCoord>
          <sceneCornerCoord>
            <lat>48.10</lat><lon>11.66</lon><height>500.0</height>
          </sceneCornerCoord>
          <sceneCornerCoord>
            <lat>48.20</lat><lon>11.66</lon><height>500.0</height>
          </sceneCornerCoord>
          <sceneCornerCoord>
            <lat>48.20</lat><lon>11.50</lon><height>500.0</height>
          </sceneCornerCoord>
          <rangeNearIncidenceAngle>33.0</rangeNearIncidenceAngle>
          <rangeFarIncidenceAngle>38.0</rangeFarIncidenceAngle>
          <headingAngle>350.5</headingAngle>
        </sceneInfo>
        <imageDataInfo>
          <imageDataType>{sample_type}</imageDataType>
          <imageDataFormat>{data_format}</imageDataFormat>
          <imageRaster>
            <numberOfRows>{_NUM_ROWS}</numberOfRows>
            <numberOfColumns>{_NUM_COLS}</numberOfColumns>
            <rowSpacing>1.8</rowSpacing>
            <columnSpacing>0.9</columnSpacing>
            <bitsPerSample>16</bitsPerSample>
          </imageRaster>
        </imageDataInfo>
      </productInfo>
      <productSpecific>
        <complexImageInfo>
          <commonPRF>3800.0</commonPRF>
          <projection>{projection}</projection>
        </complexImageInfo>
      </productSpecific>
      <instrument>
        <radarParameters>
          <centerFrequency>9.65e+09</centerFrequency>
          <totalBandwidth>150000000.0</totalBandwidth>
          <chirpDuration>3.72e-05</chirpDuration>
          <adcSamplingRate>164800000.0</adcSamplingRate>
        </radarParameters>
      </instrument>
      <platform>
        <orbit>
          <stateVec>
            <timeUTC>2026-02-19T06:12:28.000000Z</timeUTC>
            <posX>4164056.4</posX><posY>5282058.8</posY><posZ>2191522.7</posZ>
            <velX>-162.8</velX><velY>-2812.7</velY><velZ>7055.8</velZ>
          </stateVec>
          <stateVec>
            <timeUTC>2026-02-19T06:12:38.000000Z</timeUTC>
            <posX>4162173.4</posX><posY>5253637.2</posY><posZ>2262000.0</posZ>
            <velX>-200.0</velX><velY>-2900.0</velY><velZ>7000.0</velZ>
          </stateVec>
          <stateVec>
            <timeUTC>2026-02-19T06:12:48.000000Z</timeUTC>
            <posX>4160000.0</posX><posY>5225000.0</posY><posZ>2332000.0</posZ>
            <velX>-230.0</velX><velY>-2950.0</velY><velZ>6950.0</velZ>
          </stateVec>
        </orbit>
      </platform>
      <processing>
        <doppler>
          <dopplerCentroid>
            <referenceTime>5.33e-03</referenceTime>
            <dopplerCentroidPolynomial>10.5 -200.0 50000.0</dopplerCentroidPolynomial>
          </dopplerCentroid>
        </doppler>
        <processingInfo>
          <processorVersion>4.12</processorVersion>
          <processingLevel>L1B</processingLevel>
          <rangeLooks>1</rangeLooks>
          <azimuthLooks>1</azimuthLooks>
        </processingInfo>
      </processing>
    </level1Product>
    """).lstrip()


def _build_georef_xml() -> str:
    """Build a minimal GEOREF.xml with 4 corner tie points."""
    return textwrap.dedent(f"""\
    <?xml version="1.0" encoding="UTF-8"?>
    <geoReference>
      <geolocationGrid>
        <gridPoint>
          <lin>0</lin><pix>0</pix>
          <lat>48.10</lat><lon>11.50</lon>
          <height>500.0</height>
          <incidenceAngle>33.0</incidenceAngle>
        </gridPoint>
        <gridPoint>
          <lin>0</lin><pix>{_NUM_COLS - 1}</pix>
          <lat>48.10</lat><lon>11.66</lon>
          <height>500.0</height>
          <incidenceAngle>35.0</incidenceAngle>
        </gridPoint>
        <gridPoint>
          <lin>{_NUM_ROWS - 1}</lin><pix>0</pix>
          <lat>48.20</lat><lon>11.50</lon>
          <height>500.0</height>
          <incidenceAngle>34.0</incidenceAngle>
        </gridPoint>
        <gridPoint>
          <lin>{_NUM_ROWS - 1}</lin><pix>{_NUM_COLS - 1}</pix>
          <lat>48.20</lat><lon>11.66</lon>
          <height>500.0</height>
          <incidenceAngle>38.0</incidenceAngle>
        </gridPoint>
      </geolocationGrid>
    </geoReference>
    """).lstrip()


def _build_caldata_xml() -> str:
    """Build a minimal CALDATA.xml with a calibration constant."""
    return textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <calibration>
      <calibrationConstant>
        <calFactor>3.56e+04</calFactor>
      </calibrationConstant>
      <noiseEquivalentBetaNought>-25.3</noiseEquivalentBetaNought>
      <calibrationType>CALIBRATED</calibrationType>
    </calibration>
    """).lstrip()


def _build_cosar_binary(
    num_rows: int = _NUM_ROWS,
    num_cols: int = _NUM_COLS,
    sample_size: int = _SAMPLE_SIZE,
    annotation_size: int = _ANNOTATION_SIZE_PER_LINE,
) -> bytes:
    """Build a synthetic COSAR binary file.

    COSAR format: big-endian, 32-byte burst header (8 x uint32),
    then ``header_lines`` padding lines, then ``num_rows`` data lines.
    Each data line has ``annotation_size`` bytes of padding followed by
    ``num_cols`` interleaved int16 I/Q pairs.  I values are set to
    the column index, Q values to the row index, for easy verification.

    Parameters
    ----------
    num_rows : int
        Number of azimuth lines (data lines).
    num_cols : int
        Number of range samples per line.
    sample_size : int
        Bytes per sample component (2 for int16).
    annotation_size : int
        Padding bytes per data line before sample data.

    Returns
    -------
    bytes
        Complete COSAR binary content.
    """
    rtnb = annotation_size + num_cols * 2 * sample_size
    header_lines = _HEADER_LINES
    tnl = header_lines + num_rows
    bib = tnl * rtnb

    # Burst header: 8 x uint32 big-endian
    # BIB, RSRI, RSNB, AS, BI, RTNB, TNL, "CSAR"
    header = struct.pack(
        '>7I4s',
        bib,              # BIB
        0,                # RSRI
        num_cols,         # RSNB
        num_rows,         # AS
        1,                # BI
        rtnb,             # RTNB
        tnl,              # TNL
        b'CSAR',          # magic
    )

    # Pad burst header to fill header_lines * rtnb bytes
    header_pad = b'\x00' * (header_lines * rtnb - len(header))

    # Data lines (big-endian int16 I/Q)
    lines = []
    for row in range(num_rows):
        annotation = b'\x00' * annotation_size

        iq = np.empty((num_cols, 2), dtype='>i2')  # big-endian
        iq[:, 0] = np.arange(num_cols, dtype=np.int16)  # I = col
        iq[:, 1] = np.full(num_cols, row, dtype=np.int16)  # Q = row

        lines.append(annotation + iq.tobytes())

    return header + header_pad + b''.join(lines)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def synthetic_tsx_ssc(tmp_path):
    """Create a synthetic TerraSAR-X SSC product directory.

    Yields the path to the product directory.
    """
    product_dir = tmp_path / 'TSX1_SAR__SSC______SM_S_SRA_20260219_TEST'
    product_dir.mkdir()

    # Main annotation XML (name matches directory)
    xml_name = product_dir.name + '.xml'
    (product_dir / xml_name).write_text(
        _build_main_annotation_xml(
            product_type='SSC', data_format='COSAR',
        )
    )

    # IMAGEDATA/
    imagedata_dir = product_dir / 'IMAGEDATA'
    imagedata_dir.mkdir()
    cos_name = 'IMAGE_HH_SRA_strip_001.cos'
    (imagedata_dir / cos_name).write_bytes(
        _build_cosar_binary()
    )

    # ANNOTATION/
    ann_dir = product_dir / 'ANNOTATION'
    ann_dir.mkdir()
    (ann_dir / 'GEOREF.xml').write_text(_build_georef_xml())

    cal_dir = ann_dir / 'CALIBRATION'
    cal_dir.mkdir()
    (cal_dir / 'CALDATA.xml').write_text(_build_caldata_xml())

    yield product_dir


@pytest.fixture
def synthetic_tsx_mgd(tmp_path):
    """Create a synthetic TerraSAR-X MGD product directory.

    Requires rasterio.  Yields the path to the product directory.
    """
    if not _HAS_RASTERIO:
        pytest.skip("rasterio required for TerraSAR-X MGD tests")

    product_dir = tmp_path / 'TSX1_SAR__MGD______SM_D_SRA_20260219_TEST'
    product_dir.mkdir()

    xml_name = product_dir.name + '.xml'
    (product_dir / xml_name).write_text(
        _build_main_annotation_xml(
            product_type='MGD', data_format='GEOTIFF',
        )
    )

    imagedata_dir = product_dir / 'IMAGEDATA'
    imagedata_dir.mkdir()
    tiff_path = imagedata_dir / 'IMAGE_HH_SRA_strip_001.tif'

    rng = np.random.default_rng(42)
    data = rng.random((_NUM_ROWS, _NUM_COLS)).astype(np.float32) * 1000

    with rasterio.open(
        str(tiff_path), 'w', driver='GTiff',
        height=_NUM_ROWS, width=_NUM_COLS,
        count=1, dtype='float32',
    ) as dst:
        dst.write(data, 1)

    ann_dir = product_dir / 'ANNOTATION'
    ann_dir.mkdir()
    (ann_dir / 'GEOREF.xml').write_text(_build_georef_xml())
    cal_dir = ann_dir / 'CALIBRATION'
    cal_dir.mkdir()
    (cal_dir / 'CALDATA.xml').write_text(_build_caldata_xml())

    yield product_dir


# ===================================================================
# Metadata dataclass tests
# ===================================================================

class TestTerraSARMetadata:
    """Test metadata dataclass construction and access."""

    def test_basic_construction(self):
        """TerraSARMetadata can be constructed with required fields."""
        meta = TerraSARMetadata(
            format='TerraSAR-X_SSC',
            rows=100,
            cols=200,
            dtype='complex64',
        )
        assert meta.rows == 100
        assert meta.cols == 200
        assert meta.format == 'TerraSAR-X_SSC'

    def test_dict_like_access(self):
        """Metadata supports dict-like access."""
        meta = TerraSARMetadata(
            format='TerraSAR-X_SSC',
            rows=100,
            cols=200,
            dtype='complex64',
        )
        assert meta['rows'] == 100
        assert meta['format'] == 'TerraSAR-X_SSC'

    def test_product_info(self):
        """TSXProductInfo fields are accessible."""
        info = TSXProductInfo(
            mission='TerraSAR-X',
            satellite='TSX-1',
            product_type='SSC',
            imaging_mode='SM',
            absolute_orbit=12345,
        )
        assert info.satellite == 'TSX-1'
        assert info.absolute_orbit == 12345

    def test_scene_info(self):
        """TSXSceneInfo fields are accessible."""
        si = TSXSceneInfo(
            center_lat=48.15,
            center_lon=11.58,
            incidence_angle_near=33.0,
            incidence_angle_far=38.0,
        )
        assert si.center_lat == pytest.approx(48.15)
        assert si.incidence_angle_near == pytest.approx(33.0)

    def test_image_info(self):
        """TSXImageInfo fields are accessible."""
        ii = TSXImageInfo(
            num_rows=1000,
            num_cols=2000,
            row_spacing=1.8,
            col_spacing=0.9,
            sample_type='COMPLEX',
            data_format='COSAR',
        )
        assert ii.num_rows == 1000
        assert ii.data_format == 'COSAR'

    def test_radar_params(self):
        """TSXRadarParams fields are accessible."""
        rp = TSXRadarParams(
            center_frequency=9.65e9,
            prf=3800.0,
            range_bandwidth=150e6,
        )
        assert rp.center_frequency == pytest.approx(9.65e9)
        assert rp.prf == pytest.approx(3800.0)

    def test_orbit_state_vector(self):
        """TSXOrbitStateVector stores position and velocity."""
        osv = TSXOrbitStateVector(
            time_utc='2026-02-19T06:12:28.000000Z',
            position=XYZ(x=4164056.4, y=5282058.8, z=2191522.7),
            velocity=XYZ(x=-162.8, y=-2812.7, z=7055.8),
        )
        assert osv.position.x == pytest.approx(4164056.4)

    def test_geo_grid_point(self):
        """TSXGeoGridPoint stores coordinates."""
        pt = TSXGeoGridPoint(
            line=0.0, pixel=0.0,
            latitude=48.10, longitude=11.50, height=500.0,
        )
        assert pt.latitude == pytest.approx(48.10)
        assert pt.longitude == pytest.approx(11.50)

    def test_calibration(self):
        """TSXCalibration stores calibration constant."""
        cal = TSXCalibration(
            calibration_constant=3.56e4,
            noise_equivalent_beta_nought=-25.3,
        )
        assert cal.calibration_constant == pytest.approx(3.56e4)

    def test_doppler_info(self):
        """TSXDopplerInfo stores polynomial coefficients."""
        di = TSXDopplerInfo(
            doppler_centroid_coefficients=np.array(
                [10.5, -200.0, 50000.0]
            ),
            reference_time=5.33e-3,
        )
        assert di.doppler_centroid_coefficients.shape == (3,)

    def test_processing_info(self):
        """TSXProcessingInfo stores processor parameters."""
        pi = TSXProcessingInfo(
            processor_version='4.12',
            range_looks=1,
            azimuth_looks=1,
        )
        assert pi.processor_version == '4.12'

    def test_full_metadata_with_all_sections(self):
        """TerraSARMetadata with all sections populated."""
        meta = TerraSARMetadata(
            format='TerraSAR-X_SSC',
            rows=100, cols=200, dtype='complex64',
            product_info=TSXProductInfo(satellite='TSX-1'),
            scene_info=TSXSceneInfo(center_lat=48.15),
            image_info=TSXImageInfo(num_rows=100, num_cols=200),
            radar_params=TSXRadarParams(center_frequency=9.65e9),
            orbit_state_vectors=[TSXOrbitStateVector(time_utc='t0')],
            geolocation_grid=[TSXGeoGridPoint(latitude=48.10)],
            calibration=TSXCalibration(calibration_constant=3.56e4),
            doppler_info=TSXDopplerInfo(reference_time=5.33e-3),
            processing_info=TSXProcessingInfo(processor_version='4.12'),
        )
        assert meta.product_info.satellite == 'TSX-1'
        assert len(meta.orbit_state_vectors) == 1
        assert meta.calibration.calibration_constant == pytest.approx(
            3.56e4
        )


# ===================================================================
# COSAR binary reading tests
# ===================================================================

class TestCOSARBinaryReading:
    """Test low-level COSAR parsing functions."""

    def test_read_cosar_header(self, tmp_path):
        """Verify header extraction from synthetic COSAR binary."""
        from grdl.IO.sar.terrasar import _read_cosar_header

        cos_path = tmp_path / 'test.cos'
        cos_path.write_bytes(_build_cosar_binary())

        hdr = _read_cosar_header(cos_path)
        assert hdr.rsnb == _NUM_COLS
        assert hdr.azimuth_lines == _NUM_ROWS
        assert hdr.rtnb == _RTNB
        assert hdr.tnl == _TNL
        assert hdr.header_lines == _HEADER_LINES
        assert hdr.sample_size == _SAMPLE_SIZE

    def test_read_cosar_chip_known_values(self, tmp_path):
        """Read known I/Q values and verify complex conversion."""
        from grdl.IO.sar.terrasar import (
            _read_cosar_header, _read_cosar_chip,
        )

        cos_path = tmp_path / 'test.cos'
        cos_path.write_bytes(_build_cosar_binary())

        hdr = _read_cosar_header(cos_path)
        chip = _read_cosar_chip(cos_path, 0, 4, 0, 6, hdr)

        assert chip.shape == (4, 6)
        assert chip.dtype == np.complex64

        # I = col_index, Q = row_index
        # chip[row=2, col=3] should be (3 + 2j)
        assert chip[2, 3] == pytest.approx(3 + 2j)
        assert chip[0, 0] == pytest.approx(0 + 0j)
        assert chip[1, 5] == pytest.approx(5 + 1j)

    def test_read_cosar_chip_subsection(self, tmp_path):
        """Read a sub-chip from the middle of the image."""
        from grdl.IO.sar.terrasar import (
            _read_cosar_header, _read_cosar_chip,
        )

        cos_path = tmp_path / 'test.cos'
        cos_path.write_bytes(_build_cosar_binary())

        hdr = _read_cosar_header(cos_path)
        # Read rows 10-14, cols 20-25
        chip = _read_cosar_chip(cos_path, 10, 14, 20, 25, hdr)

        assert chip.shape == (4, 5)
        # chip[0, 0] → row=10, col=20 → I=20, Q=10
        assert chip[0, 0] == pytest.approx(20 + 10j)
        # chip[3, 4] → row=13, col=24 → I=24, Q=13
        assert chip[3, 4] == pytest.approx(24 + 13j)

    def test_read_cosar_full_image(self, tmp_path):
        """Read entire COSAR image."""
        from grdl.IO.sar.terrasar import (
            _read_cosar_header, _read_cosar_chip,
        )

        cos_path = tmp_path / 'test.cos'
        cos_path.write_bytes(_build_cosar_binary())

        hdr = _read_cosar_header(cos_path)
        full = _read_cosar_chip(
            cos_path, 0, hdr.azimuth_lines, 0, hdr.rsnb, hdr,
        )

        assert full.shape == (_NUM_ROWS, _NUM_COLS)
        assert full.dtype == np.complex64

    def test_invalid_cosar_header(self, tmp_path):
        """ValueError for truncated COSAR file."""
        from grdl.IO.sar.terrasar import _read_cosar_header

        cos_path = tmp_path / 'truncated.cos'
        cos_path.write_bytes(b'\x00' * 8)  # Too short

        with pytest.raises(ValueError, match="too small"):
            _read_cosar_header(cos_path)

    def test_invalid_cosar_magic(self, tmp_path):
        """ValueError for wrong COSAR magic bytes."""
        from grdl.IO.sar.terrasar import _read_cosar_header

        # Build 32 bytes with wrong magic
        bad_header = struct.pack(
            '>7I4s',
            1000, 0, 10, 10, 1, 100, 14, b'XXXX',
        )
        cos_path = tmp_path / 'bad_magic.cos'
        cos_path.write_bytes(bad_header)

        with pytest.raises(ValueError, match="magic"):
            _read_cosar_header(cos_path)


# ===================================================================
# SSC reader tests
# ===================================================================

class TestTerraSARSSCReader:
    """Test reader with synthetic SSC (COSAR) data."""

    def test_reader_loads(self, synthetic_tsx_ssc):
        """Reader opens synthetic product and populates metadata."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_ssc) as reader:
            meta = reader.metadata
            assert meta.format == 'TerraSAR-X_SSC'
            assert meta.rows == _NUM_ROWS
            assert meta.cols == _NUM_COLS
            assert meta.dtype == 'complex64'

    def test_product_info_parsed(self, synthetic_tsx_ssc):
        """Product info is extracted from main annotation."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_ssc) as reader:
            pi = reader.metadata.product_info
            assert pi.mission == 'TerraSAR-X'
            assert pi.satellite == 'TSX-1'
            assert pi.product_type == 'SSC'
            assert pi.imaging_mode == 'SM'
            assert pi.look_direction == 'RIGHT'
            assert pi.orbit_direction == 'ASCENDING'
            assert pi.absolute_orbit == 12345

    def test_scene_info_parsed(self, synthetic_tsx_ssc):
        """Scene info is extracted from main annotation."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_ssc) as reader:
            si = reader.metadata.scene_info
            assert si.center_lat == pytest.approx(48.15)
            assert si.center_lon == pytest.approx(11.58)
            assert si.incidence_angle_near == pytest.approx(33.0)
            assert si.incidence_angle_far == pytest.approx(38.0)
            assert si.heading_angle == pytest.approx(350.5)

    def test_scene_corners_parsed(self, synthetic_tsx_ssc):
        """Scene corner coordinates are extracted."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_ssc) as reader:
            corners = reader.metadata.scene_info.scene_extent
            assert corners is not None
            assert len(corners) == 4
            assert corners[0].lat == pytest.approx(48.10)
            assert corners[0].lon == pytest.approx(11.50)

    def test_image_info_parsed(self, synthetic_tsx_ssc):
        """Image info is extracted."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_ssc) as reader:
            ii = reader.metadata.image_info
            assert ii.sample_type == 'COMPLEX'
            assert ii.data_format == 'COSAR'
            assert ii.row_spacing == pytest.approx(1.8)
            assert ii.col_spacing == pytest.approx(0.9)

    def test_radar_params_parsed(self, synthetic_tsx_ssc):
        """Radar parameters are extracted."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_ssc) as reader:
            rp = reader.metadata.radar_params
            assert rp.center_frequency == pytest.approx(9.65e9)
            assert rp.prf == pytest.approx(3800.0)
            assert rp.range_bandwidth == pytest.approx(150e6)

    def test_orbit_parsed(self, synthetic_tsx_ssc):
        """Orbit state vectors are extracted."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_ssc) as reader:
            osvs = reader.metadata.orbit_state_vectors
            assert len(osvs) == 3
            assert osvs[0].position.x == pytest.approx(4164056.4)
            assert osvs[0].velocity.z == pytest.approx(7055.8)

    def test_geolocation_grid_parsed(self, synthetic_tsx_ssc):
        """Geolocation grid points are extracted from GEOREF.xml."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_ssc) as reader:
            grid = reader.metadata.geolocation_grid
            assert len(grid) == 4
            assert grid[0].latitude == pytest.approx(48.10)
            assert grid[0].longitude == pytest.approx(11.50)
            assert grid[0].incidence_angle == pytest.approx(33.0)

    def test_calibration_parsed(self, synthetic_tsx_ssc):
        """Calibration is extracted from CALDATA.xml."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_ssc) as reader:
            cal = reader.metadata.calibration
            assert cal is not None
            assert cal.calibration_constant == pytest.approx(3.56e4)
            assert cal.noise_equivalent_beta_nought == pytest.approx(
                -25.3
            )

    def test_doppler_info_parsed(self, synthetic_tsx_ssc):
        """Doppler centroid is extracted."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_ssc) as reader:
            di = reader.metadata.doppler_info
            assert di.reference_time == pytest.approx(5.33e-3)
            assert di.doppler_centroid_coefficients is not None
            assert di.doppler_centroid_coefficients.shape == (3,)

    def test_processing_info_parsed(self, synthetic_tsx_ssc):
        """Processing info is extracted."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_ssc) as reader:
            pi = reader.metadata.processing_info
            assert pi.processor_version == '4.12'
            assert pi.range_looks == 1
            assert pi.azimuth_looks == 1

    def test_read_chip(self, synthetic_tsx_ssc):
        """read_chip returns complex64 data with correct shape."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_ssc) as reader:
            chip = reader.read_chip(0, 10, 0, 15)
            assert chip.shape == (10, 15)
            assert chip.dtype == np.complex64
            # Verify known values: I=col, Q=row
            assert chip[2, 3] == pytest.approx(3 + 2j)

    def test_read_chip_full(self, synthetic_tsx_ssc):
        """read_chip can read entire image."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_ssc) as reader:
            full = reader.read_chip(0, _NUM_ROWS, 0, _NUM_COLS)
            assert full.shape == (_NUM_ROWS, _NUM_COLS)

    def test_get_shape(self, synthetic_tsx_ssc):
        """get_shape returns (rows, cols)."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_ssc) as reader:
            assert reader.get_shape() == (_NUM_ROWS, _NUM_COLS)

    def test_get_dtype(self, synthetic_tsx_ssc):
        """get_dtype returns complex64 for SSC."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_ssc) as reader:
            assert reader.get_dtype() == np.dtype('complex64')

    def test_available_polarizations(self, synthetic_tsx_ssc):
        """get_available_polarizations returns parsed list."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_ssc) as reader:
            pols = reader.get_available_polarizations()
            assert pols == ['HH']

    def test_context_manager(self, synthetic_tsx_ssc):
        """Context manager opens and closes cleanly."""
        from grdl.IO.sar.terrasar import TerraSARReader

        reader = TerraSARReader(synthetic_tsx_ssc)
        assert reader.metadata.rows == _NUM_ROWS
        reader.close()

    def test_open_from_xml(self, synthetic_tsx_ssc):
        """Reader can open from the main XML file path."""
        from grdl.IO.sar.terrasar import TerraSARReader

        xml_file = list(synthetic_tsx_ssc.glob('*.xml'))[0]
        with TerraSARReader(xml_file) as reader:
            assert reader.metadata.rows == _NUM_ROWS

    def test_open_from_cosar(self, synthetic_tsx_ssc):
        """Reader can open from a .cos file path."""
        from grdl.IO.sar.terrasar import TerraSARReader

        cos_file = list(
            (synthetic_tsx_ssc / 'IMAGEDATA').glob('*.cos')
        )[0]
        with TerraSARReader(cos_file) as reader:
            assert reader.metadata.rows == _NUM_ROWS


# ===================================================================
# MGD reader tests
# ===================================================================

@pytest.mark.skipif(not _HAS_RASTERIO,
                    reason="rasterio required for TerraSAR-X MGD")
class TestTerraSARMGDReader:
    """Test reader with synthetic MGD (GeoTIFF) data."""

    def test_reader_loads(self, synthetic_tsx_mgd):
        """Reader opens synthetic MGD and populates metadata."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_mgd) as reader:
            meta = reader.metadata
            assert meta.format == 'TerraSAR-X_MGD'
            assert meta.rows == _NUM_ROWS
            assert meta.cols == _NUM_COLS

    def test_product_type_detected(self, synthetic_tsx_mgd):
        """Product type is MGD with DETECTED sample type."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_mgd) as reader:
            pi = reader.metadata.product_info
            assert pi.product_type == 'MGD'
            ii = reader.metadata.image_info
            assert ii.sample_type == 'DETECTED'
            assert ii.data_format == 'GEOTIFF'

    def test_read_chip(self, synthetic_tsx_mgd):
        """read_chip returns float32 data with correct shape."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_mgd) as reader:
            chip = reader.read_chip(0, 10, 0, 15)
            assert chip.shape == (10, 15)
            assert chip.dtype == np.float32
            assert np.any(chip != 0)

    def test_read_chip_full(self, synthetic_tsx_mgd):
        """read_chip can read entire MGD image."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_mgd) as reader:
            full = reader.read_chip(0, _NUM_ROWS, 0, _NUM_COLS)
            assert full.shape == (_NUM_ROWS, _NUM_COLS)

    def test_get_dtype(self, synthetic_tsx_mgd):
        """get_dtype returns float32 for detected product."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_mgd) as reader:
            assert reader.get_dtype() == np.dtype('float32')


# ===================================================================
# Error handling tests
# ===================================================================

class TestTerraSARReaderErrors:
    """Test error conditions."""

    def test_missing_directory(self, tmp_path):
        """FileNotFoundError for non-existent path."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with pytest.raises((FileNotFoundError, ValueError)):
            TerraSARReader(tmp_path / 'nonexistent_dir')

    def test_invalid_product_no_xml(self, tmp_path):
        """ValueError for directory without main XML."""
        from grdl.IO.sar.terrasar import TerraSARReader

        fake_dir = tmp_path / 'TSX1_SAR__SSC_fake'
        fake_dir.mkdir()
        with pytest.raises(ValueError, match="No TerraSAR-X"):
            TerraSARReader(fake_dir)

    def test_invalid_polarization(self, synthetic_tsx_ssc):
        """ValueError for unavailable polarization."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with pytest.raises(ValueError, match="No image data file"):
            TerraSARReader(synthetic_tsx_ssc, polarization='VV')

    def test_chip_negative_indices(self, synthetic_tsx_ssc):
        """ValueError for negative start indices."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_ssc) as reader:
            with pytest.raises(ValueError, match="non-negative"):
                reader.read_chip(-1, 10, 0, 10)

    def test_chip_exceeds_dimensions(self, synthetic_tsx_ssc):
        """ValueError for indices beyond image dimensions."""
        from grdl.IO.sar.terrasar import TerraSARReader

        with TerraSARReader(synthetic_tsx_ssc) as reader:
            with pytest.raises(ValueError, match="exceed"):
                reader.read_chip(0, _NUM_ROWS + 1, 0, 10)


# ===================================================================
# Convenience function tests
# ===================================================================

class TestOpenTerrasar:
    """Test open_terrasar convenience function."""

    def test_open_terrasar_ssc(self, synthetic_tsx_ssc):
        """open_terrasar returns TerraSARReader."""
        from grdl.IO.sar.terrasar import open_terrasar

        with open_terrasar(synthetic_tsx_ssc) as reader:
            assert reader.metadata.format == 'TerraSAR-X_SSC'
