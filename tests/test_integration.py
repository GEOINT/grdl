# -*- coding: utf-8 -*-
"""
Cross-Module Integration Tests.

Tests end-to-end pipelines that chain multiple grdl modules together:
IO → data_prep → image_processing → detection → geolocation.

These are the tests that were completely missing from the suite.

Author
------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-10

Modified
--------
2026-02-10
"""

import numpy as np
import pytest

from grdl.data_prep import ChipExtractor, Normalizer, Tiler


# ---------------------------------------------------------------------------
# ChipExtractor → Normalizer pipeline
# ---------------------------------------------------------------------------

class TestChipNormalizePipeline:
    """Test chip extraction followed by normalization."""

    def test_chip_extract_then_normalize_each(self):
        """Extract chips from image, normalize each independently."""
        rng = np.random.RandomState(42)
        image = rng.rand(100, 100) * 500

        ext = ChipExtractor(nrows=100, ncols=100)
        norm = Normalizer(method='minmax')

        # Extract 4 non-overlapping chips
        chips = ext.chip_positions(row_width=50, col_width=50)

        results = []
        for chip in chips:
            data = image[chip.row_start:chip.row_end,
                         chip.col_start:chip.col_end]
            normalized = norm.normalize(data)
            results.append(normalized)

        assert len(results) == 4
        for r in results:
            assert r.shape == (50, 50)
            assert r.min() >= -0.01
            assert r.max() <= 1.01

    def test_fit_on_full_transform_on_chips(self):
        """Fit normalizer on full image, transform individual chips."""
        rng = np.random.RandomState(7)
        image = rng.rand(80, 80) * 200

        norm = Normalizer(method='zscore')
        norm.fit(image)

        ext = ChipExtractor(nrows=80, ncols=80)
        chips = ext.chip_positions(row_width=40, col_width=40)

        for chip in chips:
            data = image[chip.row_start:chip.row_end,
                         chip.col_start:chip.col_end]
            result = norm.transform(data)
            assert result.shape == (40, 40)
            assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# Tiler → Normalizer pipeline
# ---------------------------------------------------------------------------

class TestTilerNormalizePipeline:
    """Test tiling followed by normalization."""

    def test_tile_then_normalize_percentile(self):
        """Tile image with overlap, normalize each tile with percentile."""
        rng = np.random.RandomState(42)
        image = rng.rand(200, 200) * 1000 + 100

        # Add some outlier pixels
        image[0, 0] = 99999.0
        image[199, 199] = -9999.0

        tiler = Tiler(nrows=200, ncols=200, tile_size=64, stride=48)
        norm = Normalizer(method='percentile', percentile_low=2.0,
                          percentile_high=98.0)

        tiles = tiler.tile_positions()
        assert len(tiles) > 4  # should have overlap tiles

        for tile in tiles:
            data = image[tile.row_start:tile.row_end,
                         tile.col_start:tile.col_end]
            result = norm.normalize(data)
            assert result.shape == (64, 64)
            # Percentile norm clips outliers → result in [0, 1]
            assert result.min() >= -0.01
            assert result.max() <= 1.01


# ---------------------------------------------------------------------------
# Detection pipeline
# ---------------------------------------------------------------------------

class TestDetectionPipeline:
    """Test image → detect → geo-register → GeoJSON pipeline."""

    def test_detect_and_export_geojson(self):
        """Full pipeline: image → detector → geo-register → GeoJSON."""
        from grdl.image_processing.detection.models import (
            Detection, DetectionSet, Geometry, OutputField, OutputSchema,
        )
        from grdl.image_processing.versioning import processor_version
        from grdl.image_processing.detection.base import ImageDetector

        # Build a simple threshold detector
        @processor_version('1.0.0')
        class BrightPixelDetector(ImageDetector):
            def __init__(self, threshold=0.5):
                self.threshold = threshold

            @property
            def output_schema(self):
                return OutputSchema(fields=(
                    OutputField('intensity', 'float', 'Pixel value'),
                ))

            def detect(self, source, geolocation=None, **kwargs):
                rows, cols = np.where(source > self.threshold)
                detections = [
                    Detection(
                        Geometry.point(float(r), float(c)),
                        {'intensity': float(source[r, c])},
                        confidence=float(source[r, c]),
                    )
                    for r, c in zip(rows, cols)
                ]
                ds = DetectionSet(
                    detections=detections,
                    detector_name='BrightPixelDetector',
                    detector_version='1.0.0',
                    output_schema=self.output_schema,
                )
                if geolocation is not None:
                    self._geo_register_detections(ds, geolocation)
                return ds

        # Create synthetic image with known bright spots
        image = np.zeros((50, 50))
        image[10, 20] = 0.9
        image[30, 40] = 0.8
        image[5, 5] = 0.3  # below threshold

        detector = BrightPixelDetector(threshold=0.5)
        result = detector.detect(image)

        assert len(result) == 2
        geojson = result.to_geojson()
        assert geojson['type'] == 'FeatureCollection'
        assert len(geojson['features']) == 2
        assert geojson['properties']['detector_name'] == 'BrightPixelDetector'

    def test_detect_with_geo_registration(self):
        """Detections should get geographic coordinates from geolocation."""
        from grdl.image_processing.detection.models import (
            Detection, DetectionSet, Geometry, OutputField, OutputSchema,
        )
        from grdl.image_processing.versioning import processor_version
        from grdl.image_processing.detection.base import ImageDetector

        @processor_version('1.0.0')
        class SimpleDetector(ImageDetector):
            @property
            def output_schema(self):
                return OutputSchema(fields=(
                    OutputField('val', 'float', 'v'),
                ))

            def detect(self, source, geolocation=None, **kwargs):
                ds = DetectionSet(
                    [Detection(Geometry.point(0.0, 0.0), {'val': 1.0})],
                    'SD', '1.0', self.output_schema,
                )
                if geolocation is not None:
                    self._geo_register_detections(ds, geolocation)
                return ds

        class MockGeo:
            def image_to_latlon(self, row, col, height=0.0):
                if isinstance(row, np.ndarray):
                    return row * 0.01 + 40.0, col * 0.01 - 74.0, np.zeros_like(row)
                return float(row) * 0.01 + 40.0, float(col) * 0.01 - 74.0, 0.0

        detector = SimpleDetector()
        image = np.ones((10, 10))
        result = detector.detect(image, geolocation=MockGeo())

        assert len(result) == 1
        d = result[0]
        assert d.geometry.geographic_coordinates is not None
        lat, lon = d.geometry.geographic_coordinates
        assert lat == pytest.approx(40.0)
        assert lon == pytest.approx(-74.0)


# ---------------------------------------------------------------------------
# Co-Registration pipeline
# ---------------------------------------------------------------------------

class TestCoRegistrationPipeline:
    """Test image → estimate → apply alignment pipeline."""

    def test_affine_identity_round_trip(self):
        """Estimate + apply on same image should preserve pixel values."""
        from grdl.coregistration.affine import AffineCoRegistration

        rng = np.random.RandomState(42)
        image = rng.rand(100, 100) * 200

        # Identity control points
        pts = np.array([
            [10, 10], [10, 90], [90, 10], [90, 90], [50, 50],
        ], dtype=np.float64)

        coreg = AffineCoRegistration(pts, pts)
        result = coreg.estimate(image, image)

        assert result.residual_rms < 0.01

        warped = coreg.apply(image, result)
        assert warped.shape == image.shape
        # Center region should be nearly identical (edges may differ due to interpolation)
        np.testing.assert_allclose(
            warped[20:80, 20:80], image[20:80, 20:80], atol=1.0
        )

    def test_translation_recovery_pipeline(self):
        """Full pipeline: create shifted image → estimate → verify shift."""
        from grdl.coregistration.affine import AffineCoRegistration

        # Fixed points and shifted points (shift = 5 rows, 3 cols)
        fixed_pts = np.array([
            [10, 10], [10, 80], [80, 10], [80, 80],
        ], dtype=np.float64)
        moving_pts = fixed_pts + np.array([5.0, 3.0])

        coreg = AffineCoRegistration(fixed_pts, moving_pts)
        image = np.random.RandomState(42).rand(100, 100)
        result = coreg.estimate(image, image)

        # Transform should encode the shift
        tx = result.transform_matrix[0, 2]
        ty = result.transform_matrix[1, 2]
        assert tx == pytest.approx(-5.0, abs=0.1)
        assert ty == pytest.approx(-3.0, abs=0.1)


# ---------------------------------------------------------------------------
# Full IO → tile → normalize (with GeoTIFF if available)
# ---------------------------------------------------------------------------

class TestIOPipeline:
    """Test IO → data_prep pipeline (requires rasterio)."""

    def test_geotiff_to_tiles_to_normalized(self, tmp_path):
        """Write GeoTIFF → read → tile → normalize → verify."""
        try:
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            pytest.skip("rasterio not installed")

        from grdl.IO.geotiff import GeoTIFFReader

        # Write synthetic GeoTIFF
        filepath = tmp_path / "test.tif"
        rng = np.random.RandomState(42)
        data = (rng.rand(100, 100) * 500).astype(np.float32)

        transform = from_bounds(-74.1, 40.6, -73.9, 40.8, 100, 100)
        with rasterio.open(
            str(filepath), 'w', driver='GTiff',
            height=100, width=100, count=1, dtype='float32',
            transform=transform, crs='EPSG:4326',
        ) as ds:
            ds.write(data, 1)

        # Read → tile → normalize
        with GeoTIFFReader(filepath) as reader:
            full = reader.read_full()
            assert full.shape == (100, 100)

        tiler = Tiler(nrows=100, ncols=100, tile_size=50, stride=50)
        norm = Normalizer(method='minmax')

        tiles = tiler.tile_positions()
        for tile in tiles:
            chip = full[tile.row_start:tile.row_end,
                        tile.col_start:tile.col_end]
            result = norm.normalize(chip)
            assert result.min() >= -0.01
            assert result.max() <= 1.01
