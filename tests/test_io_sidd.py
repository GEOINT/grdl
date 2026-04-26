# -*- coding: utf-8 -*-
"""
SIDD IO Tests - Reader, writer, and metadata model tests.

Tests SIDD reader (sarkit/sarpy backends), writer, and metadata
dataclasses using synthetic data and mocks.

Dependencies
------------
pytest

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
2026-03-07

Modified
--------
2026-03-07
"""

import pytest
import numpy as np
from unittest import mock


# ===================================================================
# SIDDMetadata model tests
# ===================================================================

class TestSIDDMetadata:
    """Tests for SIDD metadata dataclasses."""

    def test_basic_metadata(self):
        """SIDDMetadata stores basic fields."""
        from grdl.IO.models.sidd import SIDDMetadata

        meta = SIDDMetadata(
            format='SIDD',
            rows=1024,
            cols=2048,
            dtype='uint8',
        )
        assert meta['format'] == 'SIDD'
        assert meta['rows'] == 1024
        assert meta['cols'] == 2048
        assert meta['dtype'] == 'uint8'

    def test_product_creation(self):
        """SIDDProductCreation stores processor information."""
        from grdl.IO.models.sidd import (
            SIDDProductCreation, SIDDProcessorInformation, SIDDClassification,
        )

        pc = SIDDProductCreation(
            processor_information=SIDDProcessorInformation(
                application='TestApp',
                processing_date_time='2026-01-15T12:00:00Z',
                site='TestSite',
            ),
            classification=SIDDClassification(
                classification='U',
                owner_producer='USA',
            ),
            product_name='Test Product',
            product_class='TEST',
        )
        assert pc.processor_information.application == 'TestApp'
        assert pc.classification.classification == 'U'
        assert pc.product_name == 'Test Product'

    def test_display(self):
        """SIDDDisplay stores pixel type and DRA info."""
        from grdl.IO.models.sidd import (
            SIDDDisplay, SIDDDynamicRangeAdjustment, SIDDDRAParameters,
        )

        disp = SIDDDisplay(
            pixel_type='MONO8I',
            num_bands=1,
            dynamic_range_adjustment=SIDDDynamicRangeAdjustment(
                algorithm_type='AUTO',
                band_stats_source=1,
                dra_parameters=SIDDDRAParameters(
                    pmin=0.02, pmax=0.98,
                ),
            ),
        )
        assert disp.pixel_type == 'MONO8I'
        assert disp.num_bands == 1
        assert disp.dynamic_range_adjustment.algorithm_type == 'AUTO'
        assert disp.dynamic_range_adjustment.dra_parameters.pmin == 0.02

    def test_geo_data(self):
        """SIDDGeoData stores image corners."""
        from grdl.IO.models.sidd import SIDDGeoData
        from grdl.IO.models.common import LatLon

        geo = SIDDGeoData(
            earth_model='WGS_84',
            image_corners=[
                LatLon(lat=40.0, lon=-80.0),
                LatLon(lat=40.0, lon=-79.0),
                LatLon(lat=39.0, lon=-79.0),
                LatLon(lat=39.0, lon=-80.0),
            ],
        )
        assert geo.earth_model == 'WGS_84'
        assert len(geo.image_corners) == 4
        assert geo.image_corners[0].lat == 40.0

    def test_measurement(self):
        """SIDDMeasurement stores projection parameters."""
        from grdl.IO.models.sidd import (
            SIDDMeasurement, SIDDPlaneProjection, SIDDReferencePoint,
            SIDDProductPlane,
        )
        from grdl.IO.models.common import XYZ, RowCol

        meas = SIDDMeasurement(
            projection_type='PlaneProjection',
            plane_projection=SIDDPlaneProjection(
                reference_point=SIDDReferencePoint(
                    ecef=XYZ(x=1e6, y=2e6, z=3e6),
                    point=RowCol(row=512.0, col=1024.0),
                    name='SCP',
                ),
                sample_spacing=RowCol(row=0.5, col=0.5),
                product_plane=SIDDProductPlane(
                    row_unit_vector=XYZ(x=1.0, y=0.0, z=0.0),
                    col_unit_vector=XYZ(x=0.0, y=1.0, z=0.0),
                ),
            ),
            pixel_footprint=RowCol(row=1024.0, col=2048.0),
        )
        assert meas.projection_type == 'PlaneProjection'
        assert meas.plane_projection.reference_point.name == 'SCP'
        assert meas.plane_projection.sample_spacing.row == 0.5

    def test_exploitation_features(self):
        """SIDDExploitationFeatures stores collection info."""
        from grdl.IO.models.sidd import (
            SIDDExploitationFeatures, SIDDCollectionInfo, SIDDRadarMode,
            SIDDCollectionGeometry, SIDDCollectionPhenomenology,
            SIDDAngleMagnitude, SIDDTxRcvPolarization,
        )

        ef = SIDDExploitationFeatures(
            collections=[SIDDCollectionInfo(
                sensor_name='SAR-X',
                radar_mode=SIDDRadarMode(
                    mode_type='SPOTLIGHT',
                    mode_id='SP1',
                ),
                collection_date_time='2026-01-15T12:00:00Z',
                resolution_range=0.5,
                resolution_azimuth=0.5,
                polarizations=[SIDDTxRcvPolarization(
                    tx_polarization='V',
                    rcv_polarization='V',
                )],
                geometry=SIDDCollectionGeometry(
                    azimuth=45.0,
                    graze=30.0,
                ),
                phenomenology=SIDDCollectionPhenomenology(
                    shadow=SIDDAngleMagnitude(angle=180.0, magnitude=1.0),
                ),
            )],
        )
        assert ef.collections[0].sensor_name == 'SAR-X'
        assert ef.collections[0].radar_mode.mode_type == 'SPOTLIGHT'
        assert ef.collections[0].geometry.azimuth == 45.0

    def test_downstream_reprocessing(self):
        """SIDDDownstreamReprocessing stores chip and events."""
        from grdl.IO.models.sidd import (
            SIDDDownstreamReprocessing, SIDDGeometricChip,
            SIDDProcessingEvent,
        )
        from grdl.IO.models.common import RowCol

        dr = SIDDDownstreamReprocessing(
            geometric_chip=SIDDGeometricChip(
                chip_size=RowCol(row=256.0, col=256.0),
            ),
            processing_events=[SIDDProcessingEvent(
                application_name='ChipTool',
                applied_date_time='2026-01-15T12:00:00Z',
            )],
        )
        assert dr.geometric_chip.chip_size.row == 256.0
        assert dr.processing_events[0].application_name == 'ChipTool'

    def test_compression(self):
        """SIDDCompression stores J2K parameters."""
        from grdl.IO.models.sidd import SIDDCompression, SIDDJPEG2000Subtype

        comp = SIDDCompression(
            original=SIDDJPEG2000Subtype(
                num_wavelet_levels=5,
                num_bands=1,
            ),
        )
        assert comp.original.num_wavelet_levels == 5

    def test_digital_elevation_data(self):
        """SIDDDigitalElevationData stores DEM parameters."""
        from grdl.IO.models.sidd import (
            SIDDDigitalElevationData, SIDDGeographicCoordinates,
            SIDDGeopositioning,
        )
        from grdl.IO.models.common import LatLon

        ded = SIDDDigitalElevationData(
            geographic_coordinates=SIDDGeographicCoordinates(
                longitude_density=3600.0,
                latitude_density=3600.0,
                reference_origin=LatLon(lat=40.0, lon=-80.0),
            ),
            geopositioning=SIDDGeopositioning(
                coordinate_system_type='GCS',
                geodetic_datum='World Geodetic System 1984',
                reference_ellipsoid='World Geodetic System 1984',
                vertical_datum='Mean Sea Level',
            ),
            null_value=-32767,
        )
        assert ded.geographic_coordinates.longitude_density == 3600.0
        assert ded.geopositioning.coordinate_system_type == 'GCS'
        assert ded.null_value == -32767

    def test_product_processing(self):
        """SIDDProductProcessing stores processing modules."""
        from grdl.IO.models.sidd import (
            SIDDProductProcessing, SIDDProcessingModule,
        )

        pp = SIDDProductProcessing(
            processing_modules=[SIDDProcessingModule(
                module_name='Ortho',
                name='OrthoModule',
                parameters={'method': 'bilinear'},
            )],
        )
        assert pp.processing_modules[0].module_name == 'Ortho'
        assert pp.processing_modules[0].parameters['method'] == 'bilinear'

    def test_annotations(self):
        """SIDDAnnotations stores annotation entries."""
        from grdl.IO.models.sidd import SIDDAnnotations, SIDDAnnotation

        ann = SIDDAnnotations(
            annotations=[SIDDAnnotation(
                identifier='target-1',
                spatial_reference_system='WGS84',
            )],
        )
        assert ann.annotations[0].identifier == 'target-1'

    def test_shared_sicd_types(self):
        """SIDDMetadata reuses SICD types for shared sections."""
        from grdl.IO.models.sidd import SIDDMetadata
        from grdl.IO.models.sicd import (
            SICDErrorStatistics, SICDCompositeSCPError,
            SICDRadiometric, SICDNoiseLevel,
            SICDMatchInfo, SICDMatchType,
        )

        meta = SIDDMetadata(
            format='SIDD', rows=100, cols=100, dtype='uint8',
            error_statistics=SICDErrorStatistics(
                composite_scp=SICDCompositeSCPError(
                    rg=1.0, az=1.0, rg_az=0.5,
                ),
            ),
            radiometric=SICDRadiometric(
                noise_level=SICDNoiseLevel(
                    noise_level_type='ABSOLUTE',
                ),
            ),
            match_info=SICDMatchInfo(
                match_types=[SICDMatchType(type_id='COHERENT')],
            ),
        )
        assert meta.error_statistics.composite_scp.rg == 1.0
        assert meta.radiometric.noise_level.noise_level_type == 'ABSOLUTE'
        assert meta.match_info.match_types[0].type_id == 'COHERENT'

    def test_full_metadata_dict_access(self):
        """SIDDMetadata supports dict-like access."""
        from grdl.IO.models.sidd import SIDDMetadata, SIDDDisplay

        meta = SIDDMetadata(
            format='SIDD', rows=512, cols=512, dtype='uint8',
            display=SIDDDisplay(pixel_type='MONO8I', num_bands=1),
            backend='sarpy',
            num_images=2,
            image_index=0,
        )
        assert meta['backend'] == 'sarpy'
        assert meta['num_images'] == 2
        assert meta.display.pixel_type == 'MONO8I'


# ===================================================================
# SIDDReader tests
# ===================================================================

class TestSIDDReader:
    """Tests for SIDDReader."""

    def test_file_not_found(self):
        """FileNotFoundError for non-existent file."""
        from grdl.IO.sar.sidd import SIDDReader
        with pytest.raises(FileNotFoundError):
            SIDDReader('/nonexistent/file.nitf')

    def test_requires_backend(self):
        """SIDDReader raises ImportError when no backend is available."""
        from grdl.IO.sar.sidd import SIDDReader
        from grdl.IO.sar import _backend

        with mock.patch.object(_backend, '_HAS_SARKIT', False):
            with mock.patch.object(_backend, '_HAS_SARPY', False):
                with pytest.raises(ImportError, match='sarkit'):
                    SIDDReader('/some/file.nitf')

    def test_backend_selection_prefers_sarkit(self):
        """SIDDReader prefers sarkit when both are available."""
        from grdl.IO.sar import _backend

        # When both are available, sarkit should be preferred
        with mock.patch.object(_backend, '_HAS_SARKIT', True):
            with mock.patch.object(_backend, '_HAS_SARPY', True):
                result = _backend.require_sar_backend('SIDD')
                assert result == 'sarkit'

    def test_backend_selection_falls_back_to_sarpy(self):
        """SIDDReader falls back to sarpy when sarkit missing."""
        from grdl.IO.sar import _backend

        with mock.patch.object(_backend, '_HAS_SARKIT', False):
            with mock.patch.object(_backend, '_HAS_SARPY', True):
                result = _backend.require_sar_backend('SIDD')
                assert result == 'sarpy'


# ===================================================================
# XML extraction tests (sarkit backend)
# ===================================================================

class TestSIDDXMLExtraction:
    """Tests for SIDD XML metadata extraction helpers."""

    @staticmethod
    def _make_sidd_xml() -> str:
        """Build a minimal SIDD v3 XML for testing."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<SIDD xmlns="urn:SIDD:3.0.0">
  <ProductCreation>
    <ProcessorInformation>
      <Application>TestApp</Application>
      <ProcessingDateTime>2026-01-15T12:00:00Z</ProcessingDateTime>
      <Site>TestSite</Site>
    </ProcessorInformation>
    <Classification classification="U" ownerProducer="USA"/>
    <ProductName>TestProduct</ProductName>
    <ProductClass>TEST</ProductClass>
  </ProductCreation>
  <Display>
    <PixelType>MONO8I</PixelType>
    <NumBands>1</NumBands>
    <NonInteractiveProcessing band="1">
      <ProductGenerationOptions>
        <DataRemapping><LUTName>TestLUT</LUTName>
        <Predefined><DatabaseName>BILINEAR</DatabaseName></Predefined>
        </DataRemapping>
      </ProductGenerationOptions>
      <RRDS><DownsamplingMethod>DECIMATE</DownsamplingMethod></RRDS>
    </NonInteractiveProcessing>
    <InteractiveProcessing band="1">
      <GeometricTransform>
        <Scaling>
          <AntiAlias><FilterName>AA</FilterName>
            <FilterKernel><Predefined><DatabaseName>BILINEAR</DatabaseName></Predefined></FilterKernel>
            <Operation>CONVOLUTION</Operation>
          </AntiAlias>
          <Interpolation><FilterName>Interp</FilterName>
            <FilterKernel><Predefined><DatabaseName>BILINEAR</DatabaseName></Predefined></FilterKernel>
            <Operation>CONVOLUTION</Operation>
          </Interpolation>
        </Scaling>
        <Orientation><ShadowDirection>DOWN</ShadowDirection></Orientation>
      </GeometricTransform>
      <SharpnessEnhancement>
        <ModularTransferFunctionCompensation>
          <FilterName>MTFC</FilterName>
          <FilterKernel><Predefined><DatabaseName>BILINEAR</DatabaseName></Predefined></FilterKernel>
          <Operation>CONVOLUTION</Operation>
        </ModularTransferFunctionCompensation>
      </SharpnessEnhancement>
      <DynamicRangeAdjustment>
        <AlgorithmType>AUTO</AlgorithmType>
        <BandStatsSource>1</BandStatsSource>
        <DRAParameters>
          <Pmin>0.02</Pmin>
          <Pmax>0.98</Pmax>
          <EminModifier>0.0</EminModifier>
          <EmaxModifier>1.0</EmaxModifier>
        </DRAParameters>
      </DynamicRangeAdjustment>
    </InteractiveProcessing>
  </Display>
  <GeoData>
    <EarthModel>WGS_84</EarthModel>
    <ImageCorners>
      <ICP><Lat>40.0</Lat><Lon>-80.0</Lon></ICP>
      <FRFC><Lat>40.0</Lat><Lon>-79.0</Lon></FRFC>
      <FRLC><Lat>39.0</Lat><Lon>-79.0</Lon></FRLC>
      <LRLC><Lat>39.0</Lat><Lon>-80.0</Lon></LRLC>
    </ImageCorners>
  </GeoData>
  <Measurement>
    <PlaneProjection>
      <ReferencePoint name="SCP">
        <ECF><X>1000000</X><Y>2000000</Y><Z>3000000</Z></ECF>
        <Point><Row>512</Row><Col>1024</Col></Point>
      </ReferencePoint>
      <SampleSpacing><Row>0.5</Row><Col>0.5</Col></SampleSpacing>
      <TimeCOAPoly order1="0" order2="0">
        <Coef exponent1="0" exponent2="0">1.0</Coef>
      </TimeCOAPoly>
      <ProductPlane>
        <RowUnitVector><X>1</X><Y>0</Y><Z>0</Z></RowUnitVector>
        <ColUnitVector><X>0</X><Y>1</Y><Z>0</Z></ColUnitVector>
      </ProductPlane>
    </PlaneProjection>
    <PixelFootprint><Row>1024</Row><Col>2048</Col></PixelFootprint>
    <ARPFlag>REALTIME</ARPFlag>
  </Measurement>
  <ExploitationFeatures>
    <Collection identifier="COL001">
      <Information>
        <SensorName>TestSAR</SensorName>
        <RadarMode><ModeType>SPOTLIGHT</ModeType></RadarMode>
        <CollectionDateTime>2026-01-15T12:00:00Z</CollectionDateTime>
        <Resolution><Range>0.5</Range><Azimuth>0.5</Azimuth></Resolution>
      </Information>
      <Geometry>
        <Azimuth>45.0</Azimuth>
        <Graze>30.0</Graze>
        <Tilt>5.0</Tilt>
      </Geometry>
      <Phenomenology>
        <Shadow><Angle>180</Angle><Magnitude>1.0</Magnitude></Shadow>
        <Layover><Angle>0</Angle><Magnitude>0.5</Magnitude></Layover>
      </Phenomenology>
    </Collection>
    <Product>
      <Resolution><Row>0.5</Row><Col>0.5</Col></Resolution>
      <Ellipticity>0.9</Ellipticity>
      <North>0.0</North>
    </Product>
  </ExploitationFeatures>
  <DownstreamReprocessing>
    <GeometricChip>
      <ChipSize><Row>256</Row><Col>256</Col></ChipSize>
      <OriginalUpperLeftCoordinate><Row>100</Row><Col>200</Col></OriginalUpperLeftCoordinate>
      <OriginalUpperRightCoordinate><Row>100</Row><Col>456</Col></OriginalUpperRightCoordinate>
      <OriginalLowerLeftCoordinate><Row>356</Row><Col>200</Col></OriginalLowerLeftCoordinate>
      <OriginalLowerRightCoordinate><Row>356</Row><Col>456</Col></OriginalLowerRightCoordinate>
    </GeometricChip>
    <ProcessingEvents>
      <ProcessingEvent>
        <ApplicationName>ChipTool</ApplicationName>
        <AppliedDateTime>2026-01-15T12:00:00Z</AppliedDateTime>
      </ProcessingEvent>
    </ProcessingEvents>
  </DownstreamReprocessing>
  <Compression>
    <J2K>
      <Original><NumWaveletLevels>5</NumWaveletLevels><NumBands>1</NumBands></Original>
    </J2K>
  </Compression>
  <DigitalElevationData>
    <GeographicCoordinates>
      <LongitudeDensity>3600</LongitudeDensity>
      <LatitudeDensity>3600</LatitudeDensity>
      <ReferenceOrigin><Lat>40.0</Lat><Lon>-80.0</Lon></ReferenceOrigin>
    </GeographicCoordinates>
    <Geopositioning>
      <CoordinateSystemType>GCS</CoordinateSystemType>
      <GeodeticDatum>WGS84</GeodeticDatum>
      <ReferenceEllipsoid>WGS84</ReferenceEllipsoid>
      <VerticalDatum>MSL</VerticalDatum>
    </Geopositioning>
    <NullValue>-32767</NullValue>
  </DigitalElevationData>
  <ProductProcessing>
    <ProcessingModule name="OrthoMod">
      <ModuleName>Ortho</ModuleName>
    </ProcessingModule>
  </ProductProcessing>
  <ErrorStatistics>
    <CompositeSCP><Rg>1.0</Rg><Az>1.0</Az><RgAz>0.5</RgAz></CompositeSCP>
  </ErrorStatistics>
  <Radiometric>
    <NoiseLevel>
      <NoiseLevelType>ABSOLUTE</NoiseLevelType>
    </NoiseLevel>
  </Radiometric>
</SIDD>"""

    def _parse_xml(self):
        """Parse the test XML string."""
        import xml.etree.ElementTree as ET
        return ET.fromstring(self._make_sidd_xml())

    def test_product_creation_extraction(self):
        """Extract ProductCreation from XML."""
        from grdl.IO.sar.sidd import _extract_product_creation
        xml = self._parse_xml()
        pc = _extract_product_creation(xml)
        assert pc is not None
        assert pc.processor_information.application == 'TestApp'
        assert pc.classification.classification == 'U'
        assert pc.product_name == 'TestProduct'

    def test_display_extraction(self):
        """Extract Display from XML."""
        from grdl.IO.sar.sidd import _extract_display
        xml = self._parse_xml()
        disp = _extract_display(xml)
        assert disp is not None
        assert disp.pixel_type == 'MONO8I'
        assert disp.num_bands == 1
        assert disp.dynamic_range_adjustment.algorithm_type == 'AUTO'
        assert disp.dynamic_range_adjustment.dra_parameters.pmin == 0.02

    def test_geo_data_extraction(self):
        """Extract GeoData from XML."""
        from grdl.IO.sar.sidd import _extract_geo_data
        xml = self._parse_xml()
        geo = _extract_geo_data(xml)
        assert geo is not None
        assert geo.earth_model == 'WGS_84'
        assert len(geo.image_corners) == 4
        assert geo.image_corners[0].lat == 40.0

    def test_measurement_extraction(self):
        """Extract Measurement from XML."""
        from grdl.IO.sar.sidd import _extract_measurement
        xml = self._parse_xml()
        meas = _extract_measurement(xml)
        assert meas is not None
        assert meas.projection_type == 'PlaneProjection'
        assert meas.plane_projection.reference_point.name == 'SCP'
        assert meas.plane_projection.sample_spacing.row == 0.5
        assert meas.pixel_footprint.row == 1024.0
        assert meas.arp_flag == 'REALTIME'

    def test_exploitation_features_extraction(self):
        """Extract ExploitationFeatures from XML."""
        from grdl.IO.sar.sidd import _extract_exploitation_features
        xml = self._parse_xml()
        ef = _extract_exploitation_features(xml)
        assert ef is not None
        assert len(ef.collections) == 1
        assert ef.collections[0].sensor_name == 'TestSAR'
        assert ef.collections[0].radar_mode.mode_type == 'SPOTLIGHT'
        assert ef.collections[0].geometry.azimuth == 45.0
        assert ef.collections[0].phenomenology.shadow.angle == 180.0
        assert ef.products[0].resolution.row == 0.5
        assert ef.products[0].ellipticity == 0.9

    def test_downstream_reprocessing_extraction(self):
        """Extract DownstreamReprocessing from XML."""
        from grdl.IO.sar.sidd import _extract_downstream_reprocessing
        xml = self._parse_xml()
        dr = _extract_downstream_reprocessing(xml)
        assert dr is not None
        assert dr.geometric_chip.chip_size.row == 256.0
        assert dr.processing_events[0].application_name == 'ChipTool'

    def test_compression_extraction(self):
        """Extract Compression from XML."""
        from grdl.IO.sar.sidd import _extract_compression
        xml = self._parse_xml()
        comp = _extract_compression(xml)
        assert comp is not None
        assert comp.original.num_wavelet_levels == 5
        assert comp.original.num_bands == 1

    def test_digital_elevation_data_extraction(self):
        """Extract DigitalElevationData from XML."""
        from grdl.IO.sar.sidd import _extract_digital_elevation_data
        xml = self._parse_xml()
        ded = _extract_digital_elevation_data(xml)
        assert ded is not None
        assert ded.geographic_coordinates.longitude_density == 3600.0
        assert ded.geopositioning.coordinate_system_type == 'GCS'
        assert ded.null_value == -32767

    def test_product_processing_extraction(self):
        """Extract ProductProcessing from XML."""
        from grdl.IO.sar.sidd import _extract_product_processing
        xml = self._parse_xml()
        pp = _extract_product_processing(xml)
        assert pp is not None
        assert pp.processing_modules[0].module_name == 'Ortho'
        assert pp.processing_modules[0].name == 'OrthoMod'

    def test_error_statistics_extraction(self):
        """Extract ErrorStatistics from XML."""
        from grdl.IO.sar.sidd import _extract_error_statistics_xml
        xml = self._parse_xml()
        es = _extract_error_statistics_xml(xml)
        assert es is not None
        assert es.composite_scp.rg == 1.0
        assert es.composite_scp.az == 1.0
        assert es.composite_scp.rg_az == 0.5

    def test_radiometric_extraction(self):
        """Extract Radiometric from XML."""
        from grdl.IO.sar.sidd import _extract_radiometric_xml
        xml = self._parse_xml()
        rad = _extract_radiometric_xml(xml)
        assert rad is not None
        assert rad.noise_level.noise_level_type == 'ABSOLUTE'

    def test_missing_sections_return_none(self):
        """Missing XML sections return None."""
        import xml.etree.ElementTree as ET
        from grdl.IO.sar.sidd import (
            _extract_product_creation,
            _extract_compression,
            _extract_digital_elevation_data,
            _extract_product_processing,
            _extract_annotations,
            _extract_error_statistics_xml,
            _extract_radiometric_xml,
            _extract_match_info_xml,
        )

        empty = ET.fromstring('<SIDD xmlns="urn:SIDD:3.0.0"/>')
        assert _extract_product_creation(empty) is None
        assert _extract_compression(empty) is None
        assert _extract_digital_elevation_data(empty) is None
        assert _extract_product_processing(empty) is None
        assert _extract_annotations(empty) is None
        assert _extract_error_statistics_xml(empty) is None
        assert _extract_radiometric_xml(empty) is None
        assert _extract_match_info_xml(empty) is None


# ===================================================================
# SIDDWriter tests
# ===================================================================

class TestSIDDWriter:
    """Tests for SIDDWriter."""

    def test_import_error_without_sarpy(self):
        """SIDDWriter raises ImportError when sarpy is missing."""
        import grdl.IO.sar.sidd_writer as sw

        with mock.patch.object(sw, '_HAS_SARPY_SIDD', False):
            with pytest.raises(ImportError, match='sarpy'):
                sw.SIDDWriter('/some/output.nitf')

    def test_write_chip_not_supported(self):
        """SIDDWriter.write_chip raises NotImplementedError."""
        try:
            from grdl.IO.sar.sidd_writer import SIDDWriter
        except ImportError:
            pytest.skip("sarpy not installed")

        writer = SIDDWriter.__new__(SIDDWriter)
        writer.filepath = '/tmp/test.nitf'
        with pytest.raises(NotImplementedError, match='chip-level'):
            writer.write_chip(np.zeros((10, 10), dtype=np.uint8), 0, 0)

    def test_write_validates_dimensions(self):
        """SIDDWriter.write rejects 1D arrays."""
        try:
            from grdl.IO.sar.sidd_writer import SIDDWriter
        except ImportError:
            pytest.skip("sarpy not installed")

        writer = SIDDWriter.__new__(SIDDWriter)
        writer.filepath = '/tmp/test.nitf'
        writer._sarpy_meta = mock.MagicMock()
        with pytest.raises(ValueError, match='2D or 3D'):
            writer.write(np.zeros(100, dtype=np.uint8))


# ===================================================================
# Pixel dtype mapping tests
# ===================================================================

class TestPixelDtypeMap:
    """Tests for pixel type to dtype mapping."""

    def test_all_pixel_types(self):
        """All standard SIDD pixel types map correctly."""
        from grdl.IO.sar.sidd import _PIXEL_DTYPE_MAP

        assert _PIXEL_DTYPE_MAP['MONO8I'] == 'uint8'
        assert _PIXEL_DTYPE_MAP['MONO8LU'] == 'uint8'
        assert _PIXEL_DTYPE_MAP['MONO16I'] == 'uint16'
        assert _PIXEL_DTYPE_MAP['RGB24I'] == 'uint8'
        assert _PIXEL_DTYPE_MAP['RGB8LU'] == 'uint8'

    def test_unknown_pixel_type_defaults(self):
        """Unknown pixel types default to uint8."""
        from grdl.IO.sar.sidd import _PIXEL_DTYPE_MAP
        assert _PIXEL_DTYPE_MAP.get('UNKNOWN', 'uint8') == 'uint8'


# ===================================================================
# J2K compression support (NGA.STND.0025-2 Appendix B / BPJ2K01.00)
# ===================================================================

class TestJ2KCompression:
    """Tests for JPEG 2000 (IC=C8 / M8) SIDD decode path."""

    def test_ic_constants(self):
        """IC constants match SIDD NITF FFDD Table 2-3."""
        from grdl.IO.sar.sidd import _IC_UNCOMPRESSED, _IC_J2K
        assert _IC_UNCOMPRESSED == 'NC'
        assert _IC_J2K == {'C8', 'M8'}

    def test_decode_j2k_segment_round_trip(self, tmp_path):
        """_decode_j2k_segment reconstructs a glymur-encoded codestream.

        Encodes a synthetic uint8 image as a raw J2K codestream,
        embeds it at a non-zero offset in a larger file (mimicking a
        NITF image segment), and verifies the decoder recovers the
        original pixels.
        """
        glymur = pytest.importorskip('glymur')
        from grdl.IO.sar.sidd import _decode_j2k_segment

        rng = np.random.default_rng(42)
        source = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)

        j2k_path = tmp_path / "synthetic.j2k"
        glymur.Jp2k(str(j2k_path), data=source)
        codestream = j2k_path.read_bytes()

        # Embed after a dummy prefix so offset != 0
        prefix = b'\x00' * 128
        wrapper = tmp_path / "wrapped.bin"
        wrapper.write_bytes(prefix + codestream)

        with wrapper.open('rb') as fh:
            decoded = _decode_j2k_segment(fh, len(prefix), len(codestream))

        assert decoded.shape == source.shape
        assert decoded.dtype == source.dtype
        np.testing.assert_array_equal(decoded, source)

    def test_decode_j2k_without_glymur_raises(self, monkeypatch, tmp_path):
        """_decode_j2k_segment raises ImportError when glymur is missing."""
        from grdl.IO.sar import sidd as sidd_mod
        from grdl.IO.sar.sidd import _decode_j2k_segment

        monkeypatch.setattr(sidd_mod, '_HAS_GLYMUR', False)
        fh = (tmp_path / "empty.bin").open('wb')
        fh.write(b'\x00')
        fh.close()
        with (tmp_path / "empty.bin").open('rb') as rh:
            with pytest.raises(ImportError, match='sarpy or glymur'):
                _decode_j2k_segment(rh, 0, 1)

    def test_product_image_segment_ic_uncompressed(self):
        """_product_image_segment_ic reports NC for uncompressed SIDD."""
        from grdl.IO.sar.sidd import _product_image_segment_ic

        class _Field:
            def __init__(self, v): self.value = v

        class _Sub(dict):
            def __getitem__(self, k): return super().__getitem__(k)

        sub = _Sub({'IC': _Field('NC  ')})

        class _Data:
            def __init__(self, o, s): self._o, self.size = o, s
            def get_offset(self): return self._o

        seg = {'subheader': sub, 'Data': _Data(1024, 2048)}

        class _Reader:
            jbp = {'ImageSegments': [seg]}

        with mock.patch(
            'sarkit.sidd._io.product_image_segment_mapping',
            return_value={'SIDD001': [0]},
        ):
            ic, offset, size = _product_image_segment_ic(_Reader(), 0)
        assert ic == 'NC'
        assert offset == 1024
        assert size == 2048

    def test_product_image_segment_ic_j2k(self):
        """_product_image_segment_ic reports C8 for compressed SIDD."""
        from grdl.IO.sar.sidd import _product_image_segment_ic

        class _Field:
            def __init__(self, v): self.value = v

        sub = {'IC': _Field('C8')}

        class _Data:
            def __init__(self, o, s): self._o, self.size = o, s
            def get_offset(self): return self._o

        seg = {'subheader': sub, 'Data': _Data(4096, 131072)}

        class _Reader:
            jbp = {'ImageSegments': [seg]}

        with mock.patch(
            'sarkit.sidd._io.product_image_segment_mapping',
            return_value={'SIDD001': [0]},
        ):
            ic, offset, size = _product_image_segment_ic(_Reader(), 0)
        assert ic == 'C8'
        assert offset == 4096
        assert size == 131072
