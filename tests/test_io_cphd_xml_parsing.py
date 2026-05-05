# -*- coding: utf-8 -*-
"""
XML-tree parsing tests for the CPHD 1.1.0 sarkit backend.

Each ``CPHDReader._parse_*_sarkit`` helper is a pure function over a
parsed ``xmltree`` element — no signal-block I/O required. These tests
construct a synthetic CPHD 1.1.0 XML instance covering every required
section plus several optional branches and verify each parser populates
its dataclass correctly.

Author
------
Duane Smalley, PhD

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-05-05
"""

# Standard library
from xml.etree import ElementTree as ET

# Third-party
import numpy as np
import pytest

# GRDL internal
from grdl.IO.sar.cphd import CPHDReader


# ===================================================================
# Synthetic CPHD 1.1.0 XML
# ===================================================================

CPHD_NS = 'http://api.nsgreg.nga.mil/schema/cphd/1.1.0'

_CPHD_XML = f"""\
<CPHD xmlns="{CPHD_NS}">
  <CollectionID>
    <CollectorName>UnitTestSensor</CollectorName>
    <IlluminatorName>UnitTestIlluminator</IlluminatorName>
    <CoreName>RUN-12345</CoreName>
    <CollectType>BISTATIC</CollectType>
    <RadarMode>
      <ModeType>SPOTLIGHT</ModeType>
      <ModeID>SPOT-A</ModeID>
    </RadarMode>
    <Classification>UNCLASSIFIED</Classification>
    <ReleaseInfo>UNRESTRICTED</ReleaseInfo>
    <CountryCode>US</CountryCode>
    <Parameter name="Source">unit-test</Parameter>
  </CollectionID>

  <Global>
    <DomainType>FX</DomainType>
    <SGN>-1</SGN>
    <Timeline>
      <CollectionStart>2025-04-15T12:00:00.000000Z</CollectionStart>
      <RcvCollectionStart>2025-04-15T12:00:00.005000Z</RcvCollectionStart>
      <TxTime1>0.0</TxTime1>
      <TxTime2>1.5</TxTime2>
    </Timeline>
    <FxBand>
      <FxMin>9.0e9</FxMin>
      <FxMax>10.0e9</FxMax>
    </FxBand>
    <TOASwath>
      <TOAMin>0.0</TOAMin>
      <TOAMax>1.0e-5</TOAMax>
    </TOASwath>
    <TropoParameters>
      <N0>320.0</N0>
      <RefHeight>IARP</RefHeight>
    </TropoParameters>
    <IonoParameters>
      <TECV>12.5</TECV>
      <F2Height>350000.0</F2Height>
    </IonoParameters>
  </Global>

  <SceneCoordinates>
    <EarthModel>WGS_84</EarthModel>
    <IARP>
      <ECF>
        <X>6378137.0</X>
        <Y>0.0</Y>
        <Z>0.0</Z>
      </ECF>
      <LLH>
        <Lat>0.0</Lat>
        <Lon>0.0</Lon>
        <HAE>100.0</HAE>
      </LLH>
    </IARP>
    <ReferenceSurface>
      <Planar>
        <uIAX><X>1.0</X><Y>0.0</Y><Z>0.0</Z></uIAX>
        <uIAY><X>0.0</X><Y>1.0</Y><Z>0.0</Z></uIAY>
      </Planar>
    </ReferenceSurface>
    <ImageArea>
      <X1Y1><X>-100.0</X><Y>-50.0</Y></X1Y1>
      <X2Y2><X>100.0</X><Y>50.0</Y></X2Y2>
    </ImageArea>
    <ImageAreaCornerPoints>
      <IACP index="1"><Lat>0.001</Lat><Lon>-0.001</Lon></IACP>
      <IACP index="2"><Lat>0.001</Lat><Lon>0.001</Lon></IACP>
      <IACP index="3"><Lat>-0.001</Lat><Lon>0.001</Lon></IACP>
      <IACP index="4"><Lat>-0.001</Lat><Lon>-0.001</Lon></IACP>
    </ImageAreaCornerPoints>
    <ImageGrid>
      <Identifier>Grid1</Identifier>
      <IARPLocation><Line>500.0</Line><Sample>1000.0</Sample></IARPLocation>
      <IAXExtent>
        <LineSpacing>0.5</LineSpacing>
        <FirstLine>0</FirstLine>
        <NumLines>1024</NumLines>
      </IAXExtent>
      <IAYExtent>
        <SampleSpacing>0.4</SampleSpacing>
        <FirstSample>0</FirstSample>
        <NumSamples>2048</NumSamples>
      </IAYExtent>
    </ImageGrid>
  </SceneCoordinates>

  <Data>
    <SignalArrayFormat>CF8</SignalArrayFormat>
    <NumBytesPVP>240</NumBytesPVP>
    <NumCPHDChannels>1</NumCPHDChannels>
    <SignalCompressionID>None</SignalCompressionID>
    <Channel>
      <Identifier>Ch1</Identifier>
      <NumVectors>16</NumVectors>
      <NumSamples>32</NumSamples>
      <SignalArrayByteOffset>0</SignalArrayByteOffset>
      <PVPArrayByteOffset>2048</PVPArrayByteOffset>
    </Channel>
    <NumSupportArrays>1</NumSupportArrays>
    <SupportArray>
      <Identifier>SA1</Identifier>
      <NumRows>10</NumRows>
      <NumCols>10</NumCols>
      <BytesPerElement>4</BytesPerElement>
      <ArrayByteOffset>1024</ArrayByteOffset>
    </SupportArray>
  </Data>

  <Channel>
    <RefChId>Ch1</RefChId>
    <FXFixedCPHD>true</FXFixedCPHD>
    <TOAFixedCPHD>true</TOAFixedCPHD>
    <SRPFixedCPHD>true</SRPFixedCPHD>
    <Parameters>
      <Identifier>Ch1</Identifier>
      <RefVectorIndex>0</RefVectorIndex>
      <FXFixed>true</FXFixed>
      <TOAFixed>true</TOAFixed>
      <SRPFixed>true</SRPFixed>
      <SignalNormal>true</SignalNormal>
      <Polarization>
        <TxPol>V</TxPol>
        <RcvPol>V</RcvPol>
      </Polarization>
      <FxC>9.5e9</FxC>
      <FxBW>1.0e9</FxBW>
      <FxBWNoise>1.0e9</FxBWNoise>
      <TOASaved>1.0e-5</TOASaved>
      <TOAExtended>
        <TOAExtSaved>1.5e-5</TOAExtSaved>
        <LFMEclipse>
          <FxEarlyLow>9.0e9</FxEarlyLow>
          <FxEarlyHigh>9.1e9</FxEarlyHigh>
          <FxLateLow>9.9e9</FxLateLow>
          <FxLateHigh>10.0e9</FxLateHigh>
        </LFMEclipse>
      </TOAExtended>
      <DwellTimes>
        <CODId>COD1</CODId>
        <DwellId>DT1</DwellId>
      </DwellTimes>
      <Antenna>
        <TxAPCId>APC1</TxAPCId>
        <TxAPATId>AP1</TxAPATId>
        <RcvAPCId>APC1</RcvAPCId>
        <RcvAPATId>AP1</RcvAPATId>
      </Antenna>
      <TxRcv>
        <TxWFId>WF1</TxWFId>
        <RcvId>Rcv1</RcvId>
      </TxRcv>
      <TgtRefLevel>
        <PTRef>1.5</PTRef>
      </TgtRefLevel>
      <NoiseLevel>
        <PNRef>0.01</PNRef>
        <BNRef>0.5</BNRef>
        <FxNoiseProfile>
          <Point><Fx>9.0e9</Fx><PN>0.011</PN></Point>
          <Point><Fx>10.0e9</Fx><PN>0.012</PN></Point>
        </FxNoiseProfile>
      </NoiseLevel>
    </Parameters>
  </Channel>

  <PVP>
    <TxTime><Offset>0</Offset><Size>1</Size><Format>F8</Format></TxTime>
    <TxPos><Offset>1</Offset><Size>3</Size><Format>X=F8;Y=F8;Z=F8;</Format></TxPos>
  </PVP>

  <Dwell>
    <NumCODTimes>1</NumCODTimes>
    <CODTime>
      <Identifier>COD1</Identifier>
      <CODTimePoly order1="1" order2="1">
        <Coef exponent1="0" exponent2="0">0.5</Coef>
        <Coef exponent1="1" exponent2="0">0.0</Coef>
        <Coef exponent1="0" exponent2="1">0.0</Coef>
        <Coef exponent1="1" exponent2="1">0.0</Coef>
      </CODTimePoly>
    </CODTime>
    <NumDwellTimes>1</NumDwellTimes>
    <DwellTime>
      <Identifier>DT1</Identifier>
      <DwellTimePoly order1="0" order2="0">
        <Coef exponent1="0" exponent2="0">0.001</Coef>
      </DwellTimePoly>
    </DwellTime>
  </Dwell>

  <ReferenceGeometry>
    <SRP>
      <ECF><X>6378137.0</X><Y>0.0</Y><Z>0.0</Z></ECF>
      <IAC><X>0.0</X><Y>0.0</Y><Z>0.0</Z></IAC>
    </SRP>
    <ReferenceTime>0.5</ReferenceTime>
    <SRPCODTime>0.5</SRPCODTime>
    <SRPDwellTime>0.001</SRPDwellTime>
    <Bistatic>
      <AzimuthAngle>110.0</AzimuthAngle>
      <AzimuthAngleRate>0.5</AzimuthAngleRate>
      <BistaticAngle>20.0</BistaticAngle>
      <BistaticAngleRate>0.1</BistaticAngleRate>
      <GrazeAngle>30.0</GrazeAngle>
      <TwistAngle>5.0</TwistAngle>
      <SlopeAngle>32.0</SlopeAngle>
      <LayoverAngle>200.0</LayoverAngle>
      <TxPlatform>
        <Time>0.5</Time>
        <Pos><X>7000000.0</X><Y>0.0</Y><Z>0.0</Z></Pos>
        <Vel><X>0.0</X><Y>7000.0</Y><Z>0.0</Z></Vel>
        <SideOfTrack>R</SideOfTrack>
        <SlantRange>650000.0</SlantRange>
        <GroundRange>570000.0</GroundRange>
        <DopplerConeAngle>85.0</DopplerConeAngle>
        <GrazeAngle>30.0</GrazeAngle>
        <IncidenceAngle>60.0</IncidenceAngle>
        <AzimuthAngle>110.0</AzimuthAngle>
      </TxPlatform>
      <RcvPlatform>
        <Time>0.5</Time>
        <Pos><X>7100000.0</X><Y>0.0</Y><Z>0.0</Z></Pos>
        <Vel><X>0.0</X><Y>7100.0</Y><Z>0.0</Z></Vel>
        <SideOfTrack>R</SideOfTrack>
        <SlantRange>660000.0</SlantRange>
        <GroundRange>580000.0</GroundRange>
        <DopplerConeAngle>86.0</DopplerConeAngle>
        <GrazeAngle>31.0</GrazeAngle>
        <IncidenceAngle>59.0</IncidenceAngle>
        <AzimuthAngle>112.0</AzimuthAngle>
      </RcvPlatform>
    </Bistatic>
  </ReferenceGeometry>

  <Antenna>
    <NumACFs>1</NumACFs>
    <NumAPCs>1</NumAPCs>
    <NumAntPats>1</NumAntPats>
    <AntCoordFrame>
      <Identifier>ACF1</Identifier>
      <XAxisPoly>
        <X><Coef exponent1="0">1.0</Coef></X>
        <Y><Coef exponent1="0">0.0</Coef></Y>
        <Z><Coef exponent1="0">0.0</Coef></Z>
      </XAxisPoly>
      <YAxisPoly>
        <X><Coef exponent1="0">0.0</Coef></X>
        <Y><Coef exponent1="0">1.0</Coef></Y>
        <Z><Coef exponent1="0">0.0</Coef></Z>
      </YAxisPoly>
    </AntCoordFrame>
    <AntPhaseCenter>
      <Identifier>APC1</Identifier>
      <ACFId>ACF1</ACFId>
      <APCXYZ><X>0.0</X><Y>0.0</Y><Z>0.0</Z></APCXYZ>
    </AntPhaseCenter>
    <AntPattern>
      <Identifier>AP1</Identifier>
      <FreqZero>9.5e9</FreqZero>
      <GainZero>30.0</GainZero>
      <EBFreqShift>false</EBFreqShift>
      <MLFreqDilation>false</MLFreqDilation>
      <EB>
        <DCXPoly><Coef exponent1="0">0.0</Coef></DCXPoly>
        <DCYPoly><Coef exponent1="0">0.0</Coef></DCYPoly>
      </EB>
      <Array>
        <GainPoly order1="0" order2="0">
          <Coef exponent1="0" exponent2="0">0.0</Coef>
        </GainPoly>
        <PhasePoly order1="0" order2="0">
          <Coef exponent1="0" exponent2="0">0.0</Coef>
        </PhasePoly>
      </Array>
      <Element>
        <GainPoly order1="0" order2="0">
          <Coef exponent1="0" exponent2="0">0.0</Coef>
        </GainPoly>
        <PhasePoly order1="0" order2="0">
          <Coef exponent1="0" exponent2="0">0.0</Coef>
        </PhasePoly>
      </Element>
    </AntPattern>
  </Antenna>

  <TxRcv>
    <NumTxWFs>1</NumTxWFs>
    <TxWFParameters>
      <Identifier>WF1</Identifier>
      <PulseLength>1.0e-5</PulseLength>
      <RFBandwidth>1.0e9</RFBandwidth>
      <FreqCenter>9.5e9</FreqCenter>
      <LFMRate>1.0e14</LFMRate>
      <Polarization>V</Polarization>
      <Power>1000.0</Power>
    </TxWFParameters>
    <NumRcvs>1</NumRcvs>
    <RcvParameters>
      <Identifier>Rcv1</Identifier>
      <WindowLength>2.0e-5</WindowLength>
      <SampleRate>2.0e9</SampleRate>
      <IFFilterBW>1.0e9</IFFilterBW>
      <FreqCenter>9.5e9</FreqCenter>
      <Polarization>V</Polarization>
      <PathGain>10.0</PathGain>
    </RcvParameters>
  </TxRcv>

  <ErrorParameters>
    <Monostatic>
      <PosVelErr>
        <Frame>ECF</Frame>
        <P1>1.0</P1><P2>1.0</P2><P3>1.0</P3>
        <V1>0.1</V1><V2>0.1</V2><V3>0.1</V3>
      </PosVelErr>
      <RadarSensor>
        <RangeBias>0.5</RangeBias>
        <ClockFreqSF>1.0e-9</ClockFreqSF>
      </RadarSensor>
      <TropoError>
        <TropoRangeVertical>0.2</TropoRangeVertical>
      </TropoError>
      <IonoError>
        <IonoRangeVertical>0.1</IonoRangeVertical>
      </IonoError>
    </Monostatic>
  </ErrorParameters>

  <ProductInfo>
    <Profile>UnitTestProfile</Profile>
    <CreationInfo>
      <Application>pytest</Application>
      <DateTime>2026-05-05T00:00:00.000000Z</DateTime>
      <Site>Local</Site>
    </CreationInfo>
    <Parameter name="Note">synthetic</Parameter>
  </ProductInfo>

  <GeoInfo name="AOI">
    <Desc name="Region">TestRegion</Desc>
    <Point><Lat>0.0</Lat><Lon>0.0</Lon></Point>
  </GeoInfo>

  <MatchInfo>
    <NumMatchTypes>1</NumMatchTypes>
    <MatchType index="1">
      <TypeID>STEREO</TypeID>
      <NumMatchCollections>1</NumMatchCollections>
      <MatchCollection index="1">
        <CoreName>OTHER-RUN</CoreName>
      </MatchCollection>
    </MatchType>
  </MatchInfo>
</CPHD>
"""


@pytest.fixture(scope='module')
def xml_root() -> ET.Element:
    return ET.fromstring(_CPHD_XML)


@pytest.fixture(scope='module')
def parser() -> CPHDReader:
    """A bare CPHDReader instance for calling the parser methods.

    Bypasses ``__init__`` since the parser methods are pure functions
    over the XML tree and do not require an open backend.
    """
    return CPHDReader.__new__(CPHDReader)


# ===================================================================
# Section parsers
# ===================================================================

def test_collection_info(parser, xml_root) -> None:
    ci = parser._parse_collection_info_sarkit(xml_root)
    assert ci is not None
    assert ci.collector_name == 'UnitTestSensor'
    assert ci.illuminator_name == 'UnitTestIlluminator'
    assert ci.core_name == 'RUN-12345'
    assert ci.collect_type == 'BISTATIC'
    assert ci.radar_mode == 'SPOTLIGHT'
    assert ci.radar_mode_id == 'SPOT-A'
    assert ci.classification == 'UNCLASSIFIED'
    assert ci.release_info == 'UNRESTRICTED'
    assert ci.country_code == 'US'
    assert len(ci.parameters) == 1
    assert ci.parameters[0].name == 'Source'
    assert ci.parameters[0].value == 'unit-test'


def test_global(parser, xml_root) -> None:
    g = parser._parse_global_sarkit(xml_root)
    assert g is not None
    assert g.domain_type == 'FX'
    assert g.phase_sgn == -1
    assert g.fx_band_min == 9e9
    assert g.fx_band_max == 10e9
    assert g.bandwidth == 1e9
    assert g.center_frequency == 9.5e9
    assert g.toa_swath_min == 0.0
    assert g.toa_swath_max == 1e-5
    assert g.timeline.collection_start.startswith('2025-04-15')
    assert g.timeline.tx_time1 == 0.0
    assert g.timeline.tx_time2 == 1.5
    assert g.tropo_parameters.n0 == 320.0
    assert g.tropo_parameters.ref_height == 'IARP'
    assert g.iono_parameters.tecv == 12.5
    assert g.iono_parameters.f2_height == 350000.0


def test_scene_coordinates(parser, xml_root) -> None:
    sc = parser._parse_scene_coordinates_sarkit(xml_root)
    assert sc is not None
    assert sc.earth_model == 'WGS_84'
    assert np.allclose(sc.iarp_ecf, [6378137.0, 0.0, 0.0])
    assert np.allclose(sc.iarp_llh, [0.0, 0.0, 100.0])
    assert sc.reference_surface.planar is not None
    assert np.allclose(sc.reference_surface.planar.u_iax, [1.0, 0.0, 0.0])
    assert sc.image_area.x1y1 == (-100.0, -50.0)
    assert sc.image_area.x2y2 == (100.0, 50.0)
    assert sc.image_area_x == (-100.0, 100.0)
    assert sc.corner_points.shape == (4, 2)
    assert sc.image_grid.identifier == 'Grid1'
    assert sc.image_grid.line_spacing == 0.5
    assert sc.image_grid.num_lines == 1024
    assert sc.image_grid.num_samples == 2048


def test_data_section(parser, xml_root) -> None:
    d = parser._parse_data_sarkit(xml_root)
    assert d is not None
    assert d.signal_array_format == 'CF8'
    assert d.num_bytes_pvp == 240
    assert d.num_cphd_channels == 1
    assert len(d.channels) == 1
    ch = d.channels[0]
    assert ch.identifier == 'Ch1'
    assert ch.num_vectors == 16
    assert ch.num_samples == 32
    assert ch.signal_array_byte_offset == 0
    assert ch.pvp_array_byte_offset == 2048
    assert d.num_support_arrays == 1
    assert len(d.support_arrays) == 1
    assert d.support_arrays[0].identifier == 'SA1'
    assert d.support_arrays[0].num_rows == 10


def test_channel_section(parser, xml_root) -> None:
    cs = parser._parse_channel_section_sarkit(xml_root)
    assert cs is not None
    assert cs.ref_ch_id == 'Ch1'
    assert cs.fx_fixed_cphd is True
    assert cs.toa_fixed_cphd is True
    assert cs.srp_fixed_cphd is True
    assert len(cs.parameters) == 1
    p = cs.parameters[0]
    assert p.identifier == 'Ch1'
    assert p.ref_vector_index == 0
    assert p.fx_c == 9.5e9
    assert p.fx_bw == 1e9
    assert p.toa_saved == 1e-5
    assert p.polarization.tx_pol == 'V'
    assert p.polarization.rcv_pol == 'V'
    assert p.toa_extended.toa_ext_saved == 1.5e-5
    assert p.toa_extended.lfm_eclipse.fx_early_low == 9e9
    assert p.dwell_times.cod_id == 'COD1'
    assert p.dwell_times.dwell_id == 'DT1'
    assert p.antenna.tx_apc_id == 'APC1'
    assert p.tx_rcv.tx_wf_ids == ['WF1']
    assert p.tx_rcv.rcv_ids == ['Rcv1']
    assert p.pt_ref == 1.5
    assert p.noise_level.pn_ref == 0.01
    assert len(p.noise_level.fx_noise_profile) == 2


def test_dwell_uses_poly2d(parser, xml_root) -> None:
    """Regression: Dwell polys are 2D (the previous parser used Poly1D)."""
    d = parser._parse_dwell_sarkit(xml_root)
    assert d is not None
    assert d.num_cod_times == 1
    assert d.num_dwell_times == 1
    assert d.cod_times[0].identifier == 'COD1'
    assert d.cod_times[0].cod_time_poly.shape == (2, 2)
    assert d.cod_times[0].cod_time_poly[0, 0] == 0.5
    assert d.dwell_times[0].dwell_time_poly.shape == (1, 1)
    assert d.dwell_times[0].dwell_time_poly[0, 0] == 0.001
    # Legacy property
    assert d.cod_time_poly is d.cod_times[0].cod_time_poly


def test_reference_geometry_bistatic(parser, xml_root) -> None:
    rg = parser._parse_reference_geometry_sarkit(xml_root)
    assert rg is not None
    assert rg.ref_time == 0.5
    assert rg.srp_cod_time == 0.5
    assert rg.srp_dwell_time == 0.001
    assert np.allclose(rg.srp_ecf, [6378137.0, 0.0, 0.0])
    assert np.allclose(rg.srp_iac, [0.0, 0.0, 0.0])
    assert rg.bistatic is not None
    assert rg.bistatic.bistatic_angle == 20.0
    assert rg.bistatic.azimuth_angle_rate == 0.5
    assert rg.bistatic.tx_platform is not None
    assert rg.bistatic.tx_platform.slant_range == 650000.0
    assert rg.bistatic.rcv_platform is not None
    assert rg.bistatic.rcv_platform.slant_range == 660000.0
    # Flat legacy fields populate from bistatic when monostatic is absent
    assert rg.azimuth_angle_deg == 110.0
    assert rg.graze_angle_deg == 30.0


def test_antenna(parser, xml_root) -> None:
    ant = parser._parse_antenna_sarkit(xml_root)
    assert ant is not None
    assert ant.num_acfs == 1
    assert ant.num_apcs == 1
    assert ant.num_ant_pats == 1
    assert len(ant.ant_coord_frames) == 1
    assert ant.ant_coord_frames[0].identifier == 'ACF1'
    assert ant.ant_coord_frames[0].x_axis_poly.shape == (1, 3)
    assert np.allclose(
        ant.ant_coord_frames[0].x_axis_poly[0], [1.0, 0.0, 0.0],
    )
    assert len(ant.ant_phase_centers) == 1
    assert ant.ant_phase_centers[0].acf_id == 'ACF1'
    assert len(ant.ant_patterns) == 1
    pat = ant.ant_patterns[0]
    assert pat.identifier == 'AP1'
    assert pat.freq_zero == 9.5e9
    assert pat.gain_zero == 30.0
    assert pat.eb is not None
    assert pat.eb.dcx_poly is not None
    assert pat.array is not None
    assert pat.array.gain_poly is not None
    assert pat.element is not None


def test_legacy_antenna_pattern_projection(parser, xml_root) -> None:
    """Legacy ``CPHDAntennaPattern`` is populated for ``rda.py`` consumers."""
    ant = parser._parse_antenna_sarkit(xml_root)
    legacy = parser._project_antenna_pattern(ant)
    assert legacy is not None
    assert legacy.gain_zero == 30.0
    assert legacy.gain_poly is not None
    assert legacy.acf_x_poly is not None
    assert legacy.acf_y_poly is not None


def test_tx_rcv(parser, xml_root) -> None:
    tr = parser._parse_tx_rcv_sarkit(xml_root)
    assert tr is not None
    assert tr.num_tx_wfs == 1
    assert tr.num_rcvs == 1
    wf = tr.tx_waveforms[0]
    assert wf.identifier == 'WF1'
    assert wf.pulse_length == 1e-5
    assert wf.rf_bandwidth == 1e9
    assert wf.freq_center == 9.5e9
    assert wf.lfm_rate == 1e14
    assert wf.polarization == 'V'
    assert wf.power == 1000.0
    rcv = tr.rcv_parameters[0]
    assert rcv.identifier == 'Rcv1'
    assert rcv.window_length == 2e-5
    assert rcv.sample_rate == 2e9
    assert rcv.if_filter_bw == 1e9
    assert rcv.path_gain == 10.0


def test_error_parameters_monostatic(parser, xml_root) -> None:
    ep = parser._parse_error_parameters_sarkit(xml_root)
    assert ep is not None
    assert ep.monostatic is not None
    assert ep.monostatic.pos_vel_err.frame == 'ECF'
    assert ep.monostatic.pos_vel_err.p1 == 1.0
    assert ep.monostatic.radar_sensor.range_bias == 0.5
    assert ep.monostatic.radar_sensor.clock_freq_sf == 1e-9
    assert ep.monostatic.tropo_error.tropo_range_vertical == 0.2
    assert ep.monostatic.iono_error.iono_range_vertical == 0.1


def test_product_info(parser, xml_root) -> None:
    pi = parser._parse_product_info_sarkit(xml_root)
    assert pi is not None
    assert pi.profile == 'UnitTestProfile'
    assert len(pi.creation_info) == 1
    assert pi.creation_info[0].application == 'pytest'
    assert pi.creation_info[0].site == 'Local'
    assert len(pi.parameters) == 1
    assert pi.parameters[0].name == 'Note'


def test_geo_info_recursive(parser, xml_root) -> None:
    gis = [
        parser._parse_geo_info_sarkit(g)
        for g in xml_root.findall('{*}GeoInfo')
    ]
    assert len(gis) == 1
    gi = gis[0]
    assert gi.name == 'AOI'
    assert len(gi.points) == 1
    assert gi.points[0] == (0.0, 0.0)


def test_match_info(parser, xml_root) -> None:
    mi = parser._parse_match_info_sarkit(xml_root)
    assert mi is not None
    assert mi.num_match_types == 1
    assert len(mi.match_types) == 1
    mt = mi.match_types[0]
    assert mt.index == 1
    assert mt.type_id == 'STEREO'
    assert mt.num_match_collections == 1
    assert mt.match_collections[0].core_name == 'OTHER-RUN'
    assert mt.match_collections[0].index == 1


def test_optional_section_absent_returns_none(parser) -> None:
    """Sections absent from the XML produce ``None``."""
    minimal = ET.fromstring(f'<CPHD xmlns="{CPHD_NS}"></CPHD>')
    assert parser._parse_collection_info_sarkit(minimal) is None
    assert parser._parse_global_sarkit(minimal) is None
    assert parser._parse_scene_coordinates_sarkit(minimal) is None
    assert parser._parse_data_sarkit(minimal) is None
    assert parser._parse_channel_section_sarkit(minimal) is None
    assert parser._parse_dwell_sarkit(minimal) is None
    assert parser._parse_reference_geometry_sarkit(minimal) is None
    assert parser._parse_antenna_sarkit(minimal) is None
    assert parser._parse_tx_rcv_sarkit(minimal) is None
    assert parser._parse_error_parameters_sarkit(minimal) is None
    assert parser._parse_product_info_sarkit(minimal) is None
    assert parser._parse_match_info_sarkit(minimal) is None
    assert parser._parse_support_arrays_sarkit(minimal) is None
