# -*- coding: utf-8 -*-
"""Tests for grdl.vector.io — VectorReader, VectorWriter."""

import json
import tempfile
from pathlib import Path

import pytest
from shapely.geometry import Point, box

from grdl.vector.models import Feature, FieldSchema, FeatureSet
from grdl.vector.io import VectorReader, VectorWriter


def _make_test_features():
    """Create a simple FeatureSet for I/O tests."""
    return FeatureSet(
        features=[
            Feature(
                geometry=Point(1.0, 2.0),
                properties={'name': 'alpha', 'value': 10},
                feature_id='f1',
            ),
            Feature(
                geometry=box(3.0, 4.0, 5.0, 6.0),
                properties={'name': 'beta', 'value': 20},
                feature_id='f2',
            ),
        ],
        crs='EPSG:4326',
        schema=[
            FieldSchema('name', 'str'),
            FieldSchema('value', 'int'),
        ],
        metadata={'author': 'test'},
    )


class TestVectorReaderCanRead:
    def test_geojson(self):
        assert VectorReader.can_read('data.geojson') is True
        assert VectorReader.can_read('data.json') is True

    def test_shapefile(self):
        assert VectorReader.can_read('data.shp') is True

    def test_unknown(self):
        assert VectorReader.can_read('data.xyz') is False
        assert VectorReader.can_read('data.tif') is False


class TestGeoJSONRoundTrip:
    def test_write_read_geojson(self, tmp_path):
        """Write a FeatureSet to GeoJSON and read it back."""
        fs = _make_test_features()
        out_path = tmp_path / 'test.geojson'

        VectorWriter.write(fs, out_path)
        assert out_path.exists()

        # Verify it's valid JSON
        with open(out_path) as f:
            data = json.load(f)
        assert data['type'] == 'FeatureCollection'
        assert len(data['features']) == 2

        # Read back
        fs2 = VectorReader.read(out_path)
        assert len(fs2) == 2
        assert fs2.crs == 'EPSG:4326'
        assert fs2[0].properties['name'] == 'alpha'
        assert fs2[1].properties['name'] == 'beta'

    def test_write_read_preserves_geometry(self, tmp_path):
        """Geometry should survive GeoJSON serialization."""
        fs = _make_test_features()
        out_path = tmp_path / 'geom.geojson'
        VectorWriter.write(fs, out_path)
        fs2 = VectorReader.read(out_path)

        # Point
        assert fs2[0].geometry.geom_type == 'Point'
        assert fs2[0].geometry.x == pytest.approx(1.0)
        # Polygon
        assert fs2[1].geometry.geom_type == 'Polygon'
        assert fs2[1].geometry.area == pytest.approx(4.0)

    def test_write_read_with_crs_override(self, tmp_path):
        """CRS override on read should work."""
        fs = _make_test_features()
        out_path = tmp_path / 'crs.geojson'
        VectorWriter.write(fs, out_path)
        fs2 = VectorReader.read(out_path, crs='EPSG:32618')
        assert fs2.crs == 'EPSG:32618'

    def test_write_creates_directories(self, tmp_path):
        """Writer should create parent directories."""
        fs = _make_test_features()
        out_path = tmp_path / 'sub' / 'dir' / 'output.geojson'
        VectorWriter.write(fs, out_path)
        assert out_path.exists()

    def test_json_extension(self, tmp_path):
        """Files with .json extension should work."""
        fs = _make_test_features()
        out_path = tmp_path / 'data.json'
        VectorWriter.write(fs, out_path)
        fs2 = VectorReader.read(out_path)
        assert len(fs2) == 2


class TestVectorReaderErrors:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            VectorReader.read('/nonexistent/path.geojson')

    def test_unsupported_format(self, tmp_path):
        p = tmp_path / 'data.xyz'
        p.write_text('{}')
        with pytest.raises(ValueError, match='Unsupported'):
            VectorReader.read(p)


class TestVectorWriterErrors:
    def test_unsupported_format(self, tmp_path):
        fs = _make_test_features()
        with pytest.raises(ValueError, match='Unsupported'):
            VectorWriter.write(fs, tmp_path / 'out.xyz')


class TestEmptyFeatureSet:
    def test_write_read_empty(self, tmp_path):
        """Empty FeatureSet should round-trip."""
        fs = FeatureSet(features=[], crs='EPSG:4326')
        out_path = tmp_path / 'empty.geojson'
        VectorWriter.write(fs, out_path)
        fs2 = VectorReader.read(out_path)
        assert len(fs2) == 0
        assert fs2.crs == 'EPSG:4326'
