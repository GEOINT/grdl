# -*- coding: utf-8 -*-
"""
Tests for SICD metadata coverage.

The SICD reader has two independent extractor sets -- one over the raw
XML (sarkit backend) and one over sarpy's object model -- and fields
have historically been declared on the dataclasses but never populated
by either.  These tests pin the sections that were previously dropped,
and add a coverage check that fails when any leaf value in a SICD XML
document does not reach ``SICDMetadata``.

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
2026-09-03

Modified
--------
2026-09-03
"""

# Standard library
import dataclasses as dc
import xml.etree.ElementTree as ET

# Third-party
import numpy as np
import pytest

# GRDL internal
from grdl.IO.sar.sicd import (
    _extract_collection_info_xml,
    _extract_error_statistics_xml,
    _extract_geo_data_xml,
    _extract_grid_xml,
    _extract_image_data_xml,
    _extract_image_formation_xml,
    _extract_position_xml,
    _extract_radar_collection_xml,
    _extract_rma_xml,
)


# ===================================================================
# A SICD document exercising every previously-dropped field
# ===================================================================

SICD_XML = """<?xml version="1.0"?>
<SICD xmlns="urn:SICD:1.3.0">
  <CollectionInfo>
    <CollectorName>TESTSAT</CollectorName>
    <CoreName>TEST_COLLECT</CoreName>
    <CollectType>MONOSTATIC</CollectType>
    <RadarMode><ModeType>SPOTLIGHT</ModeType></RadarMode>
    <Classification>UNCLASSIFIED</Classification>
    <Parameter name="collect_uuid">abc-123</Parameter>
    <Parameter name="ephemeris">POST_PASS_SMOOTHED</Parameter>
  </CollectionInfo>
  <ImageData>
    <PixelType>AMP8I_PHS8I</PixelType>
    <AmpTable size="3">
      <Amplitude index="2">30.5</Amplitude>
      <Amplitude index="0">10.5</Amplitude>
      <Amplitude index="1">20.5</Amplitude>
    </AmpTable>
    <NumRows>100</NumRows>
    <NumCols>200</NumCols>
    <FirstRow>0</FirstRow>
    <FirstCol>0</FirstCol>
    <FullImage><NumRows>100</NumRows><NumCols>200</NumCols></FullImage>
    <SCPPixel><Row>50</Row><Col>100</Col></SCPPixel>
  </ImageData>
  <GeoData>
    <EarthModel>WGS_84</EarthModel>
    <GeoInfo name="Airfield">
      <Desc name="source">manual</Desc>
      <Polygon size="3">
        <Vertex index="1"><Lat>33.1</Lat><Lon>-117.1</Lon></Vertex>
        <Vertex index="2"><Lat>33.2</Lat><Lon>-117.2</Lon></Vertex>
        <Vertex index="3"><Lat>33.3</Lat><Lon>-117.3</Lon></Vertex>
      </Polygon>
      <GeoInfo name="Runway">
        <Line size="2">
          <Endpoint index="1"><Lat>33.15</Lat><Lon>-117.15</Lon></Endpoint>
          <Endpoint index="2"><Lat>33.16</Lat><Lon>-117.16</Lon></Endpoint>
        </Line>
      </GeoInfo>
    </GeoInfo>
    <GeoInfo name="Tower">
      <Point><Lat>33.5</Lat><Lon>-117.5</Lon></Point>
    </GeoInfo>
  </GeoData>
  <Grid>
    <ImagePlane>SLANT</ImagePlane>
    <Type>RGAZIM</Type>
    <Row>
      <SS>0.5</SS>
      <ImpRespWid>0.45</ImpRespWid>
      <Sgn>-1</Sgn>
      <ImpRespBW>2.0</ImpRespBW>
      <KCtr>1.0</KCtr>
      <WgtType><WindowName>TAYLOR</WindowName></WgtType>
      <WgtFunct size="3">
        <Wgt index="3">0.3</Wgt>
        <Wgt index="1">0.1</Wgt>
        <Wgt index="2">0.2</Wgt>
      </WgtFunct>
    </Row>
    <Col>
      <SS>0.6</SS>
      <ImpRespWid>0.9</ImpRespWid>
      <Sgn>-1</Sgn>
      <ImpRespBW>1.0</ImpRespBW>
      <KCtr>0.0</KCtr>
    </Col>
  </Grid>
  <Position>
    <ARPPoly>
      <X><Coef exponent1="0">1.0</Coef></X>
      <Y><Coef exponent1="0">2.0</Coef></Y>
      <Z><Coef exponent1="0">3.0</Coef></Z>
    </ARPPoly>
    <RcvAPC>
      <RcvAPCPoly index="2">
        <X><Coef exponent1="0">20.0</Coef></X>
        <Y><Coef exponent1="0">21.0</Coef></Y>
        <Z><Coef exponent1="0">22.0</Coef></Z>
      </RcvAPCPoly>
      <RcvAPCPoly index="1">
        <X><Coef exponent1="0">10.0</Coef></X>
        <Y><Coef exponent1="0">11.0</Coef></Y>
        <Z><Coef exponent1="0">12.0</Coef></Z>
      </RcvAPCPoly>
    </RcvAPC>
  </Position>
  <RadarCollection>
    <TxFrequency><Min>9.0e9</Min><Max>9.6e9</Max></TxFrequency>
    <TxPolarization>SEQUENCE</TxPolarization>
    <TxSequence size="2">
      <TxStep index="2">
        <WFIndex>2</WFIndex><TxPolarization>V</TxPolarization>
      </TxStep>
      <TxStep index="1">
        <WFIndex>1</WFIndex><TxPolarization>H</TxPolarization>
      </TxStep>
    </TxSequence>
    <Area>
      <Plane>
        <RefPt name="ORP">
          <ECF><X>1.0</X><Y>2.0</Y><Z>3.0</Z></ECF>
          <Line>12.5</Line>
          <Sample>34.5</Sample>
        </RefPt>
        <XDir>
          <UVectECF><X>1.0</X><Y>0.0</Y><Z>0.0</Z></UVectECF>
          <LineSpacing>1.5</LineSpacing>
          <NumLines>500</NumLines>
          <FirstLine>0</FirstLine>
        </XDir>
        <YDir>
          <UVectECF><X>0.0</X><Y>1.0</Y><Z>0.0</Z></UVectECF>
          <SampleSpacing>2.5</SampleSpacing>
          <NumSamples>600</NumSamples>
          <FirstSample>0</FirstSample>
        </YDir>
        <SegmentList size="1">
          <Segment index="1">
            <StartLine>0</StartLine>
            <StartSample>0</StartSample>
            <EndLine>499</EndLine>
            <EndSample>599</EndSample>
            <Identifier>SEG1</Identifier>
          </Segment>
        </SegmentList>
        <Orientation>UP</Orientation>
      </Plane>
    </Area>
    <Parameter name="mode_id">SPOT_A</Parameter>
  </RadarCollection>
  <ImageFormation>
    <RcvChanProc><NumChanProc>1</NumChanProc></RcvChanProc>
    <ImageFormAlgo>PFA</ImageFormAlgo>
    <STBeamComp>GLOBAL</STBeamComp>
    <ImageBeamComp>SV</ImageBeamComp>
    <PolarizationCalibration>
      <DistortCorrectionApplied>true</DistortCorrectionApplied>
      <Distortion>
        <CalibrationDate>2024-01-01T00:00:00Z</CalibrationDate>
        <A>1.5</A>
        <F1><Real>0.1</Real><Imag>0.2</Imag></F1>
        <Q1><Real>0.3</Real><Imag>0.4</Imag></Q1>
        <Q2><Real>0.5</Real><Imag>0.6</Imag></Q2>
        <F2><Real>0.7</Real><Imag>0.8</Imag></F2>
        <Q3><Real>0.9</Real><Imag>1.0</Imag></Q3>
        <Q4><Real>1.1</Real><Imag>1.2</Imag></Q4>
        <GainErrorA>0.01</GainErrorA>
        <GainErrorF1>0.02</GainErrorF1>
        <GainErrorF2>0.03</GainErrorF2>
        <PhaseErrorF1>0.04</PhaseErrorF1>
        <PhaseErrorF2>0.05</PhaseErrorF2>
      </Distortion>
    </PolarizationCalibration>
  </ImageFormation>
  <ErrorStatistics>
    <Components>
      <PosVelErr>
        <Frame>RIC_ECI</Frame>
        <P1>5</P1><P2>5</P2><P3>5</P3>
        <V1>0.015</V1><V2>0.015</V2><V3>0.015</V3>
      </PosVelErr>
      <RadarSensor>
        <RangeBias>149.9</RangeBias>
        <ClockFreqSF>1e-9</ClockFreqSF>
      </RadarSensor>
      <TropoError>
        <TropoRangeVertical>2.5</TropoRangeVertical>
        <TropoRangeSlant>3.5</TropoRangeSlant>
        <TropoRangeDecorr>
          <CorrCoefZero>0.9</CorrCoefZero><DecorrRate>0.001</DecorrRate>
        </TropoRangeDecorr>
      </TropoError>
      <IonoError>
        <IonoRangeVertical>4.5</IonoRangeVertical>
        <IonoRangeRateVertical>0.05</IonoRangeRateVertical>
        <IonoRgRgRateCC>0.75</IonoRgRgRateCC>
      </IonoError>
    </Components>
    <Unmodeled>
      <Xrow>1.25</Xrow>
      <Ycol>2.25</Ycol>
      <XrowYcol>0.5</XrowYcol>
      <UnmodeledDecorr>
        <Xrow>
          <CorrCoefZero>0.8</CorrCoefZero><DecorrRate>0.002</DecorrRate>
        </Xrow>
        <Ycol>
          <CorrCoefZero>0.7</CorrCoefZero><DecorrRate>0.003</DecorrRate>
        </Ycol>
      </UnmodeledDecorr>
    </Unmodeled>
    <AdditionalParms>
      <Parameter name="note">extra</Parameter>
    </AdditionalParms>
  </ErrorStatistics>
  <RMA>
    <RMAlgoType>OMEGA_K</RMAlgoType>
    <ImageType>INCA</ImageType>
  </RMA>
</SICD>
"""


@pytest.fixture(scope='module')
def xml():
    return ET.fromstring(SICD_XML)


# ===================================================================
# Previously-dropped fields
# ===================================================================

class TestCollectionInfo:
    def test_parameters_captured(self, xml):
        ci = _extract_collection_info_xml(xml)
        assert ci.parameters == {
            'collect_uuid': 'abc-123',
            'ephemeris': 'POST_PASS_SMOOTHED',
        }


class TestImageData:
    def test_amp_table_captured_in_index_order(self, xml):
        idata = _extract_image_data_xml(xml)
        assert idata.amp_table is not None
        np.testing.assert_allclose(
            idata.amp_table, [10.5, 20.5, 30.5],
        )


class TestGeoInfo:
    def test_top_level_features(self, xml):
        geo = _extract_geo_data_xml(xml)
        assert geo.geo_info is not None
        assert [g.name for g in geo.geo_info] == ['Airfield', 'Tower']

    def test_polygon_in_index_order(self, xml):
        airfield = _extract_geo_data_xml(xml).geo_info[0]
        assert len(airfield.polygon) == 3
        assert airfield.polygon[0].lat == pytest.approx(33.1)
        assert airfield.polygon[2].lat == pytest.approx(33.3)

    def test_descriptions(self, xml):
        airfield = _extract_geo_data_xml(xml).geo_info[0]
        assert airfield.descriptions == {'source': 'manual'}

    def test_nested_feature_and_line(self, xml):
        airfield = _extract_geo_data_xml(xml).geo_info[0]
        assert len(airfield.geo_info) == 1
        runway = airfield.geo_info[0]
        assert runway.name == 'Runway'
        assert len(runway.line) == 2
        assert runway.line[0].lon == pytest.approx(-117.15)

    def test_point_feature(self, xml):
        tower = _extract_geo_data_xml(xml).geo_info[1]
        assert tower.point.lat == pytest.approx(33.5)
        assert tower.polygon is None and tower.line is None


class TestGridWeighting:
    def test_wgt_funct_captured_in_index_order(self, xml):
        grid = _extract_grid_xml(xml)
        np.testing.assert_allclose(
            grid.row.wgt_funct, [0.1, 0.2, 0.3],
        )

    def test_absent_wgt_funct_is_none(self, xml):
        grid = _extract_grid_xml(xml)
        assert grid.col.wgt_funct is None


class TestPosition:
    def test_rcv_apc_captured_in_index_order(self, xml):
        pos = _extract_position_xml(xml)
        assert pos.rcv_apc is not None and len(pos.rcv_apc) == 2
        assert pos.rcv_apc[0].x.coefs[0] == pytest.approx(10.0)
        assert pos.rcv_apc[1].x.coefs[0] == pytest.approx(20.0)


class TestRadarCollection:
    def test_tx_sequence_in_index_order(self, xml):
        rc = _extract_radar_collection_xml(xml)
        assert [s.wf_index for s in rc.tx_sequence] == [1, 2]
        assert [s.tx_polarization for s in rc.tx_sequence] == ['H', 'V']

    def test_parameters(self, xml):
        rc = _extract_radar_collection_xml(xml)
        assert rc.parameters == {'mode_id': 'SPOT_A'}

    def test_area_plane_reference_point(self, xml):
        plane = _extract_radar_collection_xml(xml).area.plane
        assert plane.ref_pt.name == 'ORP'
        assert plane.ref_pt.line == pytest.approx(12.5)
        assert plane.ref_pt.ecf.z == pytest.approx(3.0)

    def test_area_plane_directions(self, xml):
        plane = _extract_radar_collection_xml(xml).area.plane
        assert plane.x_dir.line_spacing == pytest.approx(1.5)
        assert plane.x_dir.num_lines == 500
        assert plane.y_dir.sample_spacing == pytest.approx(2.5)
        assert plane.y_dir.num_samples == 600

    def test_area_plane_segments_and_orientation(self, xml):
        plane = _extract_radar_collection_xml(xml).area.plane
        assert len(plane.segments) == 1
        assert plane.segments[0].identifier == 'SEG1'
        assert plane.segments[0].end_sample == 599
        assert plane.orientation == 'UP'


class TestImageFormation:
    def test_st_beam_comp(self, xml):
        imf = _extract_image_formation_xml(xml)
        assert imf.st_beam_comp == 'GLOBAL'
        assert imf.image_beam_comp == 'SV'

    def test_distortion_correction_flag(self, xml):
        pc = _extract_image_formation_xml(xml).polarization_calibration
        assert pc.distort_correction_applied is True

    def test_distortion_complex_terms(self, xml):
        d = _extract_image_formation_xml(
            xml,
        ).polarization_calibration.distortion
        assert d.a == pytest.approx(1.5)
        assert d.f1 == pytest.approx(complex(0.1, 0.2))
        assert d.q4 == pytest.approx(complex(1.1, 1.2))
        assert d.calibration_date == '2024-01-01T00:00:00Z'

    def test_distortion_error_terms(self, xml):
        d = _extract_image_formation_xml(
            xml,
        ).polarization_calibration.distortion
        assert d.gain_error_a == pytest.approx(0.01)
        assert d.phase_error_f2 == pytest.approx(0.05)


class TestErrorStatistics:
    def test_radar_sensor_under_components(self, xml):
        err = _extract_error_statistics_xml(xml)
        assert err.radar_sensor is not None
        assert err.radar_sensor.range_bias == pytest.approx(149.9)
        assert err.radar_sensor.clock_freq_sf == pytest.approx(1e-9)

    def test_tropo_error(self, xml):
        tropo = _extract_error_statistics_xml(xml).tropo_error
        assert tropo.tropo_range_vertical == pytest.approx(2.5)
        assert tropo.tropo_range_slant == pytest.approx(3.5)
        assert tropo.tropo_range_decorr.decorr_rate == pytest.approx(0.001)

    def test_iono_error(self, xml):
        iono = _extract_error_statistics_xml(xml).iono_error
        assert iono.iono_range_vertical == pytest.approx(4.5)
        assert iono.iono_rg_rg_rate_cc == pytest.approx(0.75)
        assert iono.iono_range_vert_decorr is None

    def test_unmodeled(self, xml):
        unm = _extract_error_statistics_xml(xml).unmodeled
        assert unm.xrow == pytest.approx(1.25)
        assert unm.xrow_ycol == pytest.approx(0.5)
        assert unm.unmodeled_decorr.ycol.corr_coef_zero == pytest.approx(0.7)

    def test_additional_parms(self, xml):
        err = _extract_error_statistics_xml(xml)
        assert err.additional_parms == {'note': 'extra'}


class TestRMA:
    def test_algorithm_type(self, xml):
        rma = _extract_rma_xml(xml)
        assert rma.rm_algo_type == 'OMEGA_K'
        assert rma.image_type == 'INCA'


# ===================================================================
# Whole-document coverage
# ===================================================================

def _model_values(obj, seen=None, out=None, depth=0):
    """Collect every scalar reachable from a parsed metadata tree."""
    if out is None:
        out = {'nums': [], 'strs': set()}
    if seen is None:
        seen = set()
    if obj is None or depth > 12 or id(obj) in seen:
        return out
    seen.add(id(obj))

    if isinstance(obj, bool):
        out['strs'].add(str(obj).lower())
    elif isinstance(obj, complex):
        out['nums'].extend([obj.real, obj.imag])
    elif isinstance(obj, (int, float, np.integer, np.floating)):
        out['nums'].append(float(obj))
    elif isinstance(obj, str):
        out['strs'].add(obj.strip())
    elif isinstance(obj, np.ndarray):
        out['nums'].extend(float(v) for v in obj.ravel())
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out['strs'].add(str(k))
            _model_values(v, seen, out, depth + 1)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _model_values(v, seen, out, depth + 1)
    elif dc.is_dataclass(obj):
        for f in dc.fields(obj):
            _model_values(getattr(obj, f.name, None), seen, out, depth + 1)
    return out


def _leaves(elem, path=''):
    tag = elem.tag.split('}')[-1]
    here = f'{path}/{tag}' if path else tag
    kids = list(elem)
    if not kids:
        yield here, (elem.text or '').strip()
    else:
        for k in kids:
            yield from _leaves(k, here)


# Derived counts are recoverable from the parsed list lengths, so the
# model deliberately does not duplicate them.
DERIVED = ('NumMatchTypes', 'NumMatchCollections', 'NumChanProc')


def test_every_leaf_value_reaches_the_model(xml):
    """No leaf value in the document is silently dropped.

    This is the check that catches a field declared on the dataclass
    but never populated by the extractor -- the failure mode that hid
    RcvAPC, WgtFunct, the distortion matrix and the collection
    parameters.
    """
    from grdl.IO.sar.sicd import (
        _extract_image_creation_xml, _extract_scpcoa_xml,
        _extract_timeline_xml, _extract_radiometric_xml,
        _extract_antenna_xml, _extract_match_info_xml,
        _extract_rg_az_comp_xml, _extract_pfa_xml,
    )
    from grdl.IO.models.sicd import SICDMetadata

    meta = SICDMetadata(
        format='SICD', rows=100, cols=200, dtype='complex64',
        collection_info=_extract_collection_info_xml(xml),
        image_creation=_extract_image_creation_xml(xml),
        image_data=_extract_image_data_xml(xml),
        geo_data=_extract_geo_data_xml(xml),
        grid=_extract_grid_xml(xml),
        timeline=_extract_timeline_xml(xml),
        position=_extract_position_xml(xml),
        radar_collection=_extract_radar_collection_xml(xml),
        image_formation=_extract_image_formation_xml(xml),
        scpcoa=_extract_scpcoa_xml(xml),
        radiometric=_extract_radiometric_xml(xml),
        antenna=_extract_antenna_xml(xml),
        error_statistics=_extract_error_statistics_xml(xml),
        match_info=_extract_match_info_xml(xml),
        rg_az_comp=_extract_rg_az_comp_xml(xml),
        pfa=_extract_pfa_xml(xml),
        rma=_extract_rma_xml(xml),
    )

    vals = _model_values(meta)
    nums = np.array(sorted(vals['nums'])) if vals['nums'] else np.array([])
    strs = vals['strs']

    def covered(text):
        if not text or text in strs:
            return True
        try:
            v = float(text)
        except ValueError:
            return False
        if nums.size == 0:
            return False
        i = np.searchsorted(nums, v)
        return any(
            abs(nums[j] - v) <= 1e-9 * max(1.0, abs(v))
            for j in (i - 1, i, i + 1)
            if 0 <= j < nums.size
        )

    missing = [
        (path, text) for path, text in _leaves(xml)
        if not covered(text) and path.split('/')[-1] not in DERIVED
    ]
    assert not missing, (
        "SICD leaf values not reaching SICDMetadata:\n"
        + "\n".join(f"  {p} = {t!r}" for p, t in missing)
    )
