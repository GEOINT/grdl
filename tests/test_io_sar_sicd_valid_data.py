# -*- coding: utf-8 -*-
"""
Tests for SICD ImageCorners and ValidData XML parsing.

Exercises the GeoData ImageCorners (``<ICP index="n:LABEL">``) ordering,
the GeoData ValidData lat/lon polygon, and the ImageData ValidData
pixel-space polygon. Uses synthetic SICD XML fragments -- no real files.

Author
------
Duane Smalley
duane.d.smalley@gmail.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-06-07

Modified
--------
2026-06-07
"""

# Standard library
import xml.etree.ElementTree as ET

# Third-party
import pytest

# GRDL internal
from grdl.IO.sar.sicd import (
    _extract_geo_data_xml,
    _extract_image_data_xml,
)


GEO_XML = """
<SICD>
  <GeoData>
    <EarthModel>WGS_84</EarthModel>
    <ImageCorners>
      <ICP index="2:FRLC"><Lat>2.0</Lat><Lon>-20.0</Lon></ICP>
      <ICP index="1:FRFC"><Lat>1.0</Lat><Lon>-10.0</Lon></ICP>
      <ICP index="4:LRFC"><Lat>4.0</Lat><Lon>-40.0</Lon></ICP>
      <ICP index="3:LRLC"><Lat>3.0</Lat><Lon>-30.0</Lon></ICP>
    </ImageCorners>
    <ValidData size="3">
      <Vertex index="2"><Lat>12.0</Lat><Lon>-120.0</Lon></Vertex>
      <Vertex index="1"><Lat>11.0</Lat><Lon>-110.0</Lon></Vertex>
      <Vertex index="3"><Lat>13.0</Lat><Lon>-130.0</Lon></Vertex>
    </ValidData>
  </GeoData>
</SICD>
"""

IMAGE_XML = """
<SICD>
  <ImageData>
    <PixelType>RE32F_IM32F</PixelType>
    <NumRows>100</NumRows>
    <NumCols>200</NumCols>
    <FirstRow>0</FirstRow>
    <FirstCol>0</FirstCol>
    <SCPPixel><Row>50</Row><Col>100</Col></SCPPixel>
    <ValidData size="4">
      <Vertex index="1"><Row>0</Row><Col>0</Col></Vertex>
      <Vertex index="2"><Row>0</Row><Col>199</Col></Vertex>
      <Vertex index="3"><Row>99</Row><Col>199</Col></Vertex>
      <Vertex index="4"><Row>99</Row><Col>0</Col></Vertex>
    </ValidData>
  </ImageData>
</SICD>
"""

IMAGE_XML_NO_VALID = """
<SICD>
  <ImageData>
    <NumRows>10</NumRows>
    <NumCols>10</NumCols>
  </ImageData>
</SICD>
"""


class TestImageCorners:
    def test_corners_parsed_and_ordered_by_index(self):
        geo = _extract_geo_data_xml(ET.fromstring(GEO_XML))
        assert geo is not None
        corners = geo.image_corners
        assert corners is not None
        assert len(corners) == 4
        # Returned in index order 1,2,3,4 regardless of XML order.
        assert [c.lat for c in corners] == [1.0, 2.0, 3.0, 4.0]
        assert [c.lon for c in corners] == [-10.0, -20.0, -30.0, -40.0]

    def test_missing_corners_returns_none(self):
        xml = ET.fromstring('<SICD><GeoData/></SICD>')
        geo = _extract_geo_data_xml(xml)
        assert geo is not None
        assert geo.image_corners is None


class TestGeoValidData:
    def test_geo_valid_data_parsed_and_ordered(self):
        geo = _extract_geo_data_xml(ET.fromstring(GEO_XML))
        assert geo.valid_data is not None
        assert len(geo.valid_data) == 3
        assert [v.lat for v in geo.valid_data] == [11.0, 12.0, 13.0]
        assert [v.lon for v in geo.valid_data] == [-110.0, -120.0, -130.0]

    def test_missing_geo_valid_data_returns_none(self):
        geo = _extract_geo_data_xml(ET.fromstring('<SICD><GeoData/></SICD>'))
        assert geo.valid_data is None


class TestImageValidData:
    def test_pixel_valid_data_parsed_and_ordered(self):
        idata = _extract_image_data_xml(ET.fromstring(IMAGE_XML))
        assert idata is not None
        assert idata.valid_data is not None
        assert len(idata.valid_data) == 4
        assert [v.row for v in idata.valid_data] == [0.0, 0.0, 99.0, 99.0]
        assert [v.col for v in idata.valid_data] == [0.0, 199.0, 199.0, 0.0]

    def test_pixel_valid_data_absent_returns_none(self):
        idata = _extract_image_data_xml(ET.fromstring(IMAGE_XML_NO_VALID))
        assert idata is not None
        assert idata.valid_data is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
