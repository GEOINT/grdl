# -*- coding: utf-8 -*-
"""Tests for grdl.vector.conversion — RasterToPoints, Rasterize."""

import numpy as np
import pytest
from shapely.geometry import box

from grdl.vector.models import Feature, FeatureSet
from grdl.vector.conversion import RasterToPoints, Rasterize


class TestRasterToPoints:
    def test_basic(self):
        """Extract points from a small raster."""
        data = np.array([
            [0, 0, 5],
            [0, 10, 0],
            [3, 0, 0],
        ], dtype=np.float64)
        converter = RasterToPoints(threshold=1.0)
        result = converter.convert(data)
        assert isinstance(result, FeatureSet)
        # Pixels >= 1.0: (2,0)=5, (1,1)=10, (0,2)=3
        assert len(result) == 3
        # Check properties
        values = result.get_property_array('value')
        assert sorted(values) == [3.0, 5.0, 10.0]

    def test_3d_input(self):
        """Extract from a specific band of a 3D array."""
        data = np.zeros((2, 4, 4), dtype=np.float64)
        data[1, 2, 3] = 100.0
        converter = RasterToPoints(threshold=50.0, band=1)
        result = converter.convert(data)
        assert len(result) == 1
        assert result[0].properties['row'] == 2
        assert result[0].properties['col'] == 3

    def test_sample_step(self):
        """Sub-sampling should reduce output count."""
        data = np.ones((10, 10), dtype=np.float64)
        full = RasterToPoints(threshold=0.5).convert(data)
        sampled = RasterToPoints(threshold=0.5, sample_step=2).convert(data)
        assert len(sampled) < len(full)
        # 10x10 with step 2 -> 5x5 = 25
        assert len(sampled) == 25

    def test_threshold_zero(self):
        """Threshold 0 includes zeros."""
        data = np.zeros((3, 3), dtype=np.float64)
        result = RasterToPoints(threshold=0.0).convert(data)
        assert len(result) == 9


class TestRasterize:
    def test_basic(self):
        """Burn a box polygon into a raster."""
        # Create a box covering pixels (2,2) to (4,4) exclusive
        feat = Feature(
            geometry=box(2, 2, 5, 5),
            properties={'val': 7.0},
            feature_id='box1',
        )
        fs = FeatureSet(features=[feat])
        rasterizer = Rasterize(burn_value=1.0)
        result = rasterizer.convert(fs, shape=(8, 8))
        assert result.shape == (8, 8)
        # Center of box should be burned
        assert result[3, 3] == 1.0
        # Outside box should be fill
        assert result[0, 0] == 0.0

    def test_value_field(self):
        """Use a property field for burn values."""
        feat = Feature(
            geometry=box(1, 1, 4, 4),
            properties={'intensity': 42.0},
            feature_id='f1',
        )
        fs = FeatureSet(features=[feat])
        rasterizer = Rasterize(value_field='intensity', fill_value=-1.0)
        result = rasterizer.convert(fs, shape=(6, 6))
        assert result[2, 2] == 42.0
        assert result[0, 0] == -1.0

    def test_fill_value(self):
        """Empty feature set produces fill-only array."""
        fs = FeatureSet(features=[])
        rasterizer = Rasterize(fill_value=99.0)
        result = rasterizer.convert(fs, shape=(4, 4))
        assert np.all(result == 99.0)

    def test_round_trip_approximate(self):
        """RasterToPoints then Rasterize gives approximate round trip."""
        # Create a simple raster with a bright region
        data = np.zeros((10, 10), dtype=np.float64)
        data[3:7, 3:7] = 5.0

        # Raster -> Points
        converter = RasterToPoints(threshold=1.0)
        points = converter.convert(data)
        assert len(points) == 16  # 4x4 block

        # Points -> Raster (points are individual pixels)
        # Buffer points slightly so rasterize can hit pixel centers
        from grdl.vector.spatial import BufferOperator
        buffered = BufferOperator(distance=0.6, resolution=4).process(points)
        rasterizer = Rasterize(burn_value=5.0)
        result = rasterizer.convert(buffered, shape=(10, 10))
        # Original bright pixels should be recovered
        assert result[4, 4] == 5.0
