# -*- coding: utf-8 -*-
"""Tests for grdl.vector.coords coordinate fast path."""

import numpy as np
import pytest

from grdl.vector.coords import (
    CoordSet,
    coords_from_geojson,
    read_coords,
    PART_POINT,
    PART_EXTERIOR,
    PART_HOLE,
    PART_LINE,
)


def _fc(*geoms):
    return {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {},
                      "geometry": g} for g in geoms],
    }


def test_points_flat_array():
    cs = coords_from_geojson(_fc(
        {"type": "Point", "coordinates": [10.0, 20.0]},
        {"type": "Point", "coordinates": [11.0, 21.0]},
    ))
    assert cs.n_features == 2
    assert cs.n_parts == 2
    assert cs.xy.shape == (2, 2)
    # GeoJSON axis order [lon, lat] preserved in xy
    np.testing.assert_array_equal(cs.xy[0], [10.0, 20.0])
    # latlon flips to [lat, lon] for geolocation
    np.testing.assert_array_equal(cs.latlon[0], [20.0, 10.0])
    assert list(cs.part_types) == [PART_POINT, PART_POINT]


def test_polygon_rings_and_holes():
    poly = {"type": "Polygon", "coordinates": [
        [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],       # exterior
        [[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.2]],  # hole
    ]}
    cs = coords_from_geojson(_fc(poly))
    assert cs.n_parts == 2
    assert list(cs.part_types) == [PART_EXTERIOR, PART_HOLE]
    # both rings map to feature 0
    np.testing.assert_array_equal(cs.feature_index, [0, 0])
    assert cs.part(0).shape == (5, 2)
    assert cs.part(1).shape == (4, 2)


def test_polygon_drop_holes():
    poly = {"type": "Polygon", "coordinates": [
        [[0, 0], [1, 0], [1, 1], [0, 0]],
        [[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.2]],
    ]}
    cs = coords_from_geojson(_fc(poly), holes=False)
    assert cs.n_parts == 1
    assert list(cs.part_types) == [PART_EXTERIOR]


def test_mixed_and_multipolygon():
    cs = coords_from_geojson(_fc(
        {"type": "Point", "coordinates": [5.0, 6.0]},
        {"type": "LineString", "coordinates": [[0, 0], [1, 1], [2, 2]]},
        {"type": "MultiPolygon", "coordinates": [
            [[[0, 0], [1, 0], [1, 1], [0, 0]]],
            [[[5, 5], [6, 5], [6, 6], [5, 5]]],
        ]},
    ))
    assert cs.n_features == 3
    # point + line + 2 exterior rings
    assert list(cs.part_types) == [PART_POINT, PART_LINE,
                                   PART_EXTERIOR, PART_EXTERIOR]
    assert list(cs.feature_index) == [0, 1, 2, 2]


def test_iter_parts_views_match_offsets():
    cs = coords_from_geojson(_fc(
        {"type": "Point", "coordinates": [1.0, 2.0]},
        {"type": "LineString", "coordinates": [[3, 4], [5, 6]]},
    ))
    parts = list(cs.iter_parts())
    assert len(parts) == 2
    fi, ptype, verts = parts[1]
    assert fi == 1 and ptype == PART_LINE
    np.testing.assert_array_equal(verts, [[3, 4], [5, 6]])


def test_unsupported_geometry_raises():
    with pytest.raises(ValueError, match="Unsupported geometry"):
        coords_from_geojson(_fc({"type": "Bogus", "coordinates": []}))


def test_latlon_requires_geographic_space():
    cs = coords_from_geojson(_fc(
        {"type": "Point", "coordinates": [1.0, 2.0]}))
    pixel = CoordSet(
        xy=cs.xy.copy(), offsets=cs.offsets.copy(),
        part_types=cs.part_types.copy(), feature_index=cs.feature_index.copy(),
        n_features=cs.n_features, space="pixel")
    with pytest.raises(ValueError, match="space='lonlat'"):
        _ = pixel.latlon
    with pytest.raises(ValueError, match="space='lonlat'"):
        pixel.to_image(geolocation=None)


def test_empty_collection():
    cs = coords_from_geojson({"type": "FeatureCollection", "features": []})
    assert cs.n_features == 0
    assert cs.n_parts == 0
    assert cs.xy.shape == (0, 2)


def test_read_coords_missing_file():
    with pytest.raises(FileNotFoundError):
        read_coords("/tmp/does_not_exist_12345.geojson")
