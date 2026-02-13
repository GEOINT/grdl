# -*- coding: utf-8 -*-
"""
Execute Protocol Tests.

Tests for the universal ``execute(metadata, source, **kwargs)`` protocol
on ``ImageProcessor`` and its ABC subclasses (``ImageTransform``,
``ImageDetector``, ``PolarimetricDecomposition``, ``SublookDecomposition``).

All tests use synthetic data — no real imagery or model weights required.

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
2026-02-12
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
from shapely.geometry import box

from grdl.IO.models.base import ImageMetadata
from grdl.image_processing.base import ImageProcessor, ImageTransform
from grdl.image_processing.detection.base import ImageDetector
from grdl.image_processing.detection.models import Detection, DetectionSet
from grdl.image_processing.decomposition.base import PolarimetricDecomposition


# ---------------------------------------------------------------------------
# Concrete test doubles
# ---------------------------------------------------------------------------

class DoubleTransform(ImageTransform):
    """Transform that doubles its input."""

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        return (source * 2).astype(np.float64)


class CroppingTransform(ImageTransform):
    """Transform that changes spatial dimensions and drops to 2D."""

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        # Take first band and crop — produces 2D output
        return source[:5, :8, 0]


class MetadataInspectingTransform(ImageTransform):
    """Transform that records self.metadata during apply()."""

    seen_metadata = None

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        self.seen_metadata = self.metadata
        return source


class StubDetector(ImageDetector):
    """Detector that returns a fixed DetectionSet."""

    seen_metadata = None

    def detect(
        self,
        source: np.ndarray,
        geolocation=None,
        **kwargs: Any,
    ) -> DetectionSet:
        self.seen_metadata = self.metadata
        self.last_geolocation = geolocation
        det = Detection(
            pixel_geometry=box(0, 0, 10, 10),
            properties={'test': True},
            confidence=0.9,
        )
        return DetectionSet(
            detections=[det],
            detector_name='StubDetector',
            detector_version='1.0.0',
        )

    @property
    def output_fields(self) -> Tuple[str, ...]:
        return ()


class StubDecomposition(PolarimetricDecomposition):
    """Decomposition returning two named complex components."""

    def decompose(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: np.ndarray,
        svv: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        return {
            'alpha': shh + svv,
            'beta': shh - svv,
        }

    @property
    def component_names(self) -> Tuple[str, ...]:
        return ('alpha', 'beta')

    def to_rgb(self, components, representation='db',
               percentile_low=2.0, percentile_high=98.0) -> np.ndarray:
        return np.zeros((10, 10, 3), dtype=np.float32)


class BareProcessor(ImageProcessor):
    """ImageProcessor subclass with only apply() — tests fallback probing."""

    def apply(self, source, **kwargs):
        return source + 1


class EmptyProcessor(ImageProcessor):
    """ImageProcessor subclass with no action method — tests error path."""
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def meta():
    return ImageMetadata(
        format='test', rows=10, cols=12, dtype='float32',
        bands=3, crs='EPSG:4326',
    )


@pytest.fixture
def source_3band():
    return np.random.rand(10, 12, 3).astype(np.float32)


@pytest.fixture
def source_2d():
    return np.random.rand(10, 12).astype(np.float32)


# ---------------------------------------------------------------------------
# ImageTransform execute() tests
# ---------------------------------------------------------------------------

class TestTransformExecute:

    def test_delegates_to_apply(self, meta, source_3band):
        """execute() calls apply() and returns (ndarray, metadata)."""
        t = DoubleTransform()
        result, out_meta = t.execute(meta, source_3band)
        np.testing.assert_allclose(result, source_3band * 2)
        assert isinstance(out_meta, ImageMetadata)

    def test_metadata_dtype_updated(self, meta, source_3band):
        """Returned metadata reflects the output dtype."""
        t = DoubleTransform()
        _, out_meta = t.execute(meta, source_3band)
        assert out_meta.dtype == 'float64'

    def test_metadata_shape_updated(self, meta, source_3band):
        """Returned metadata reflects the output rows/cols/bands."""
        t = CroppingTransform()
        result, out_meta = t.execute(meta, source_3band)
        assert out_meta.rows == 5
        assert out_meta.cols == 8
        # 2D result → bands=1
        assert out_meta.bands == 1

    def test_metadata_bands_3d(self, meta, source_3band):
        """3D output → bands from shape[2]."""
        t = DoubleTransform()  # preserves shape
        _, out_meta = t.execute(meta, source_3band)
        assert out_meta.bands == 3

    def test_original_metadata_unchanged(self, meta, source_3band):
        """Input metadata is not mutated."""
        t = DoubleTransform()
        t.execute(meta, source_3band)
        assert meta.dtype == 'float32'
        assert meta.rows == 10
        assert meta.cols == 12

    def test_metadata_property_during_apply(self, meta, source_3band):
        """self.metadata is set before apply() runs."""
        t = MetadataInspectingTransform()
        assert t.metadata is None  # before execute()
        t.execute(meta, source_3band)
        assert t.seen_metadata is meta
        assert t.metadata is meta  # still set after execute()


# ---------------------------------------------------------------------------
# ImageDetector execute() tests
# ---------------------------------------------------------------------------

class TestDetectorExecute:

    def test_delegates_to_detect(self, meta, source_3band):
        """execute() calls detect() and returns DetectionSet."""
        d = StubDetector()
        result, out_meta = d.execute(meta, source_3band)
        assert isinstance(result, DetectionSet)
        assert len(result) == 1

    def test_metadata_unchanged(self, meta, source_3band):
        """Detector returns input metadata unchanged."""
        d = StubDetector()
        _, out_meta = d.execute(meta, source_3band)
        assert out_meta is meta

    def test_geolocation_flows_through_kwargs(self, meta, source_3band):
        """geolocation kwarg reaches detect()."""
        d = StubDetector()
        sentinel = object()
        d.execute(meta, source_3band, geolocation=sentinel)
        assert d.last_geolocation is sentinel

    def test_metadata_property_during_detect(self, meta, source_3band):
        """self.metadata is accessible inside detect()."""
        d = StubDetector()
        d.execute(meta, source_3band)
        assert d.seen_metadata is meta
        assert d.seen_metadata.crs == 'EPSG:4326'


# ---------------------------------------------------------------------------
# PolarimetricDecomposition execute() tests
# ---------------------------------------------------------------------------

class TestDecompositionExecute:

    def test_delegates_to_decompose_via_kwargs(self, meta):
        """execute() delegates to decompose() when channels are in kwargs."""
        shape = (10, 12)
        shh = np.ones(shape, dtype=np.complex64)
        svv = np.ones(shape, dtype=np.complex64) * 2
        shv = np.zeros(shape, dtype=np.complex64)
        svh = np.zeros(shape, dtype=np.complex64)

        dec = StubDecomposition()
        result, out_meta = dec.execute(
            meta, np.empty((10, 12)),
            shh=shh, shv=shv, svh=svh, svv=svv,
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {'alpha', 'beta'}
        np.testing.assert_allclose(result['alpha'], shh + svv)

    def test_delegates_to_decompose_via_bands(self, meta):
        """execute() extracts channels from 4-band source array."""
        source = np.random.rand(10, 12, 4).astype(np.complex64)
        dec = StubDecomposition()
        result, out_meta = dec.execute(meta, source)
        assert isinstance(result, dict)
        assert len(result) == 2

    def test_metadata_bands_updated(self, meta):
        """Returned metadata bands = number of components."""
        source = np.random.rand(10, 12, 4).astype(np.complex64)
        dec = StubDecomposition()
        _, out_meta = dec.execute(meta, source)
        assert out_meta.bands == 2  # alpha, beta


# ---------------------------------------------------------------------------
# ImageProcessor fallback probing
# ---------------------------------------------------------------------------

class TestFallbackProbing:

    def test_bare_processor_probes_apply(self, meta, source_2d):
        """ImageProcessor with only apply() gets probed by default execute()."""
        p = BareProcessor()
        result, out_meta = p.execute(meta, source_2d)
        np.testing.assert_allclose(result, source_2d + 1)
        assert out_meta is meta  # default execute doesn't update metadata

    def test_no_method_raises(self, meta, source_2d):
        """Processor with no action method raises NotImplementedError."""
        p = EmptyProcessor()
        with pytest.raises(NotImplementedError, match="has no execute"):
            p.execute(meta, source_2d)
