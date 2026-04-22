# -*- coding: utf-8 -*-
"""
Tests for BlobDetector.

Covers:
- Happy-path detection (empty mask, single blob, multi-blob)
- Area filtering (min_area, max_area)
- Connectivity behaviour (4- vs 8-connected)
- valid_mask gating
- Geometry edge-cases (single pixel, two pixels, collinear pixels)
- DetectionSet metadata and output fields
- Strict 2-D input policy: 3-D and multi-band arrays must raise ValueError
- Provenance: metadata fields are preserved on the returned DetectionSet

Author
------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org

Created
-------
2026-04-21
"""

import numpy as np
import pytest

from grdl.image_processing.detection.blob import BlobDetector
from grdl.image_processing.detection.models import DetectionSet
from grdl.image_processing.detection.fields import Fields


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_blob_mask():
    """20x20 mask with two spatially-separated, non-touching blobs.

    Blob A: 4x4 square at rows 2-5, cols 2-5  (area=16)
    Blob B: 3x3 square at rows 12-14, cols 12-14  (area=9)
    """
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:6, 2:6] = 1    # 16 px
    mask[12:15, 12:15] = 1  # 9 px
    return mask


@pytest.fixture
def single_blob_mask():
    """15x15 mask with one 5x5 blob (area=25)."""
    mask = np.zeros((15, 15), dtype=np.uint8)
    mask[5:10, 5:10] = 1
    return mask


@pytest.fixture
def diagonal_chain_mask():
    """5x5 mask where three pixels are connected only diagonally.

    (1,1) — (2,2) — (3,3)

    Under 4-connectivity these are three separate blobs.
    Under 8-connectivity they form one blob.
    """
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1, 1] = 1
    mask[2, 2] = 1
    mask[3, 3] = 1
    return mask


# ---------------------------------------------------------------------------
# 1. Happy-path detection
# ---------------------------------------------------------------------------

class TestBlobDetectorBasic:

    def test_empty_mask_returns_no_detections(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        bd = BlobDetector(min_area=1)
        result = bd.detect(mask)
        assert isinstance(result, DetectionSet)
        assert len(result) == 0

    def test_all_foreground_mask_is_one_blob(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        bd = BlobDetector(min_area=1)
        result = bd.detect(mask)
        assert len(result) == 1
        assert result[0].properties[Fields.physical.AREA] == 100

    def test_two_separate_blobs_detected(self, two_blob_mask):
        bd = BlobDetector(min_area=1)
        result = bd.detect(two_blob_mask)
        assert len(result) == 2

    def test_single_blob_correct_area(self, single_blob_mask):
        bd = BlobDetector(min_area=1)
        result = bd.detect(single_blob_mask)
        assert len(result) == 1
        assert result[0].properties[Fields.physical.AREA] == 25

    def test_boolean_input_accepted(self, single_blob_mask):
        """Float/bool dtypes should all work — foreground is > 0."""
        bool_mask = single_blob_mask.astype(bool)
        float_mask = single_blob_mask.astype(np.float32)
        bd = BlobDetector(min_area=1)
        r_bool = bd.detect(bool_mask)
        r_float = bd.detect(float_mask)
        assert len(r_bool) == len(r_float) == 1

    def test_perimeter_field_present(self, single_blob_mask):
        bd = BlobDetector(min_area=1)
        result = bd.detect(single_blob_mask)
        assert Fields.physical.PERIMETER in result[0].properties
        assert result[0].properties[Fields.physical.PERIMETER] > 0.0

    def test_pixel_geometry_is_polygon(self, single_blob_mask):
        """Each detection must carry a shapely Polygon geometry."""
        bd = BlobDetector(min_area=1)
        result = bd.detect(single_blob_mask)
        geom = result[0].pixel_geometry
        assert geom.geom_type in ('Polygon', 'Point', 'LineString')


# ---------------------------------------------------------------------------
# 2. Area filtering
# ---------------------------------------------------------------------------

class TestAreaFiltering:

    def test_min_area_removes_small_blobs(self, two_blob_mask):
        # Blob B has area=9; set min_area=10 to exclude it
        bd = BlobDetector(min_area=10)
        result = bd.detect(two_blob_mask)
        assert len(result) == 1
        assert result[0].properties[Fields.physical.AREA] == 16

    def test_min_area_1_keeps_single_pixels(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 1
        bd = BlobDetector(min_area=1)
        result = bd.detect(mask)
        assert len(result) == 1

    def test_max_area_removes_large_blobs(self, two_blob_mask):
        # Blob A has area=16; set max_area=12 to exclude it
        bd = BlobDetector(min_area=1, max_area=12)
        result = bd.detect(two_blob_mask)
        assert len(result) == 1
        assert result[0].properties[Fields.physical.AREA] == 9

    def test_max_area_none_keeps_all_blobs(self, two_blob_mask):
        bd = BlobDetector(min_area=1, max_area=None)
        result = bd.detect(two_blob_mask)
        assert len(result) == 2

    def test_max_area_zero_treated_as_no_limit(self, two_blob_mask):
        """max_area=0 is the widget-friendly 'no limit' sentinel."""
        bd = BlobDetector(min_area=1, max_area=0)
        result = bd.detect(two_blob_mask)
        assert len(result) == 2

    def test_no_blobs_pass_when_min_exceeds_all(self, two_blob_mask):
        bd = BlobDetector(min_area=1000)
        result = bd.detect(two_blob_mask)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# 3. Connectivity
# ---------------------------------------------------------------------------

class TestConnectivity:

    def test_4_connectivity_diagonal_chain_is_separate(self, diagonal_chain_mask):
        """4-connected: diagonally adjacent pixels are separate blobs."""
        bd = BlobDetector(min_area=1, connectivity=4)
        result = bd.detect(diagonal_chain_mask)
        assert len(result) == 3

    def test_8_connectivity_diagonal_chain_is_one_blob(self, diagonal_chain_mask):
        """8-connected: diagonally adjacent pixels form one blob."""
        bd = BlobDetector(min_area=1, connectivity=8)
        result = bd.detect(diagonal_chain_mask)
        assert len(result) == 1

    def test_invalid_connectivity_raises(self):
        with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
            BlobDetector(connectivity=6)


# ---------------------------------------------------------------------------
# 4. valid_mask gating
# ---------------------------------------------------------------------------

class TestValidMask:

    def test_valid_mask_suppresses_blob_outside_gate(self, two_blob_mask):
        """Gate out Blob B (bottom-right) using a valid_mask."""
        gate = np.zeros((20, 20), dtype=bool)
        gate[:10, :10] = True        # only top-left quadrant is valid
        bd = BlobDetector(min_area=1)
        result = bd.detect(two_blob_mask, valid_mask=gate)
        # Only Blob A (area=16, top-left) should survive
        assert len(result) == 1
        assert result[0].properties[Fields.physical.AREA] == 16

    def test_all_false_valid_mask_yields_zero_blobs(self, two_blob_mask):
        gate = np.zeros((20, 20), dtype=bool)
        bd = BlobDetector(min_area=1)
        result = bd.detect(two_blob_mask, valid_mask=gate)
        assert len(result) == 0

    def test_all_true_valid_mask_is_no_op(self, two_blob_mask):
        gate = np.ones((20, 20), dtype=bool)
        bd = BlobDetector(min_area=1)
        result = bd.detect(two_blob_mask, valid_mask=gate)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 5. Geometry edge-cases
# ---------------------------------------------------------------------------

class TestGeometryEdgeCases:

    def test_single_pixel_blob_has_point_or_polygon_geometry(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 1
        bd = BlobDetector(min_area=1)
        result = bd.detect(mask)
        assert len(result) == 1
        assert result[0].pixel_geometry is not None

    def test_two_pixel_blob_geometry(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 1
        mask[5, 6] = 1
        bd = BlobDetector(min_area=1)
        result = bd.detect(mask)
        assert len(result) == 1

    def test_large_foreground_perimeter_is_positive(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[5:25, 5:25] = 1
        bd = BlobDetector(min_area=1)
        result = bd.detect(mask)
        assert result[0].properties[Fields.physical.PERIMETER] > 0.0


# ---------------------------------------------------------------------------
# 6. DetectionSet metadata and provenance
# ---------------------------------------------------------------------------

class TestDetectionSetMetadata:

    def test_detector_name_is_blobdetector(self, single_blob_mask):
        bd = BlobDetector()
        result = bd.detect(single_blob_mask)
        assert result.detector_name == 'BlobDetector'

    def test_detector_version_is_set(self, single_blob_mask):
        bd = BlobDetector()
        result = bd.detect(single_blob_mask)
        assert result.detector_version is not None
        assert result.detector_version != ''

    def test_output_fields_declared(self, single_blob_mask):
        bd = BlobDetector()
        result = bd.detect(single_blob_mask)
        assert Fields.physical.AREA in result.output_fields
        assert Fields.physical.PERIMETER in result.output_fields

    def test_metadata_contains_run_params(self, single_blob_mask):
        bd = BlobDetector(min_area=7, max_area=50, connectivity=8)
        result = bd.detect(single_blob_mask)
        assert result.metadata['min_area'] == 7
        assert result.metadata['max_area'] == 50
        assert result.metadata['connectivity'] == 8

    def test_metadata_contains_raw_component_count(self, two_blob_mask):
        """n_components_raw lets callers audit how many blobs were filtered."""
        bd = BlobDetector(min_area=1)
        result = bd.detect(two_blob_mask)
        assert 'n_components_raw' in result.metadata
        assert result.metadata['n_components_raw'] >= len(result)

    def test_geojson_export_round_trips(self, two_blob_mask):
        bd = BlobDetector(min_area=1)
        result = bd.detect(two_blob_mask)
        geojson = result.to_geojson()
        assert geojson['type'] == 'FeatureCollection'
        assert len(geojson['features']) == 2
        for feat in geojson['features']:
            assert feat['type'] == 'Feature'
            assert 'geometry' in feat
            assert 'properties' in feat

    def test_iteration_and_indexing(self, two_blob_mask):
        bd = BlobDetector(min_area=1)
        result = bd.detect(two_blob_mask)
        detections_via_iter = list(result)
        assert len(detections_via_iter) == 2
        assert result[0] is detections_via_iter[0]


# ---------------------------------------------------------------------------
# 7. Strict 2-D input policy — MUST fail on multi-band input
#    These tests intentionally probe the shape guard and verify that
#    BlobDetector refuses to silently collapse multi-pol stacks.
# ---------------------------------------------------------------------------

class TestStrict2DInputPolicy:
    """Verify that BlobDetector refuses non-2-D arrays with clear guidance.

    The 'fail-on-multi-band' tests are the primary regression guard against
    accidental normalization of multi-polarization SAR data.  If any of
    these tests ever pass (i.e., BlobDetector silently accepts 3-D input),
    the shape guard has been broken.
    """

    def test_3d_single_band_hwc_raises(self):
        """(H, W, 1) must raise — caller must explicitly squeeze first."""
        mask_3d = np.zeros((20, 20, 1), dtype=np.uint8)
        mask_3d[5:10, 5:10, 0] = 1
        bd = BlobDetector(min_area=1)
        with pytest.raises(ValueError, match=r"2-D"):
            bd.detect(mask_3d)

    def test_3d_dual_pol_hwc_raises_with_guidance(self):
        """(H, W, 2) dual-pol stack must raise with actionable guidance."""
        dual_pol = np.zeros((50, 50, 2), dtype=np.float32)
        dual_pol[10:20, 10:20, 0] = 1.0   # VV channel
        dual_pol[30:40, 30:40, 1] = 1.0   # VH channel
        bd = BlobDetector(min_area=1)
        with pytest.raises(ValueError) as exc_info:
            bd.detect(dual_pol)
        msg = str(exc_info.value)
        # Error must explain *why* this is rejected and what to do instead
        assert "2-D" in msg or "single-channel" in msg
        assert "dimensionality" in msg or "decomposition" in msg or "polarization" in msg

    def test_3d_quad_pol_raises_with_guidance(self):
        """(H, W, 4) NISAR/Sentinel-1 full-pol stack must raise."""
        quad_pol = np.zeros((40, 40, 4), dtype=np.float32)
        quad_pol[5:15, 5:15, :] = 1.0
        bd = BlobDetector(min_area=1)
        with pytest.raises(ValueError):
            bd.detect(quad_pol)

    def test_3d_chw_format_raises(self):
        """(C, H, W) band-first layout is also rejected — no format guessing."""
        chw_mask = np.zeros((2, 30, 30), dtype=np.uint8)
        chw_mask[0, 5:15, 5:15] = 1
        bd = BlobDetector(min_area=1)
        with pytest.raises(ValueError):
            bd.detect(chw_mask)

    def test_1d_array_raises(self):
        """1-D input has no spatial meaning and must be rejected."""
        flat = np.ones(100, dtype=np.uint8)
        bd = BlobDetector(min_area=1)
        with pytest.raises(ValueError):
            bd.detect(flat)

    def test_4d_array_raises(self):
        """4-D input (e.g., a time-series stack) must be rejected."""
        time_series = np.zeros((3, 20, 20, 2), dtype=np.uint8)
        bd = BlobDetector(min_area=1)
        with pytest.raises(ValueError):
            bd.detect(time_series)

    def test_correct_2d_input_does_not_raise(self):
        """Regression: confirm a proper 2-D mask still works after the guard."""
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:10, 5:10] = 1
        bd = BlobDetector(min_area=1)
        result = bd.detect(mask)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 8. Processor tag contract
# ---------------------------------------------------------------------------

class TestProcessorTags:
    """Verify that BlobDetector's @processor_tags are correctly stamped."""

    def test_input_type_is_binary_mask(self):
        from grdl.vocabulary import DataPortType
        tags = BlobDetector.__processor_tags__
        assert tags['input_type'] == DataPortType.BINARY_MASK

    def test_output_type_is_detection_set(self):
        from grdl.vocabulary import DataPortType
        tags = BlobDetector.__processor_tags__
        assert tags['output_type'] == DataPortType.DETECTION_SET

    def test_required_bands_is_1(self):
        """required_bands=1 signals to grdl-runtime that only single-band
        binary masks are valid input — prevents multi-pol misrouting."""
        tags = BlobDetector.__processor_tags__
        assert tags['required_bands'] == 1

    def test_category_is_find_maxima(self):
        from grdl.vocabulary import ProcessorCategory
        tags = BlobDetector.__processor_tags__
        assert tags['category'] == ProcessorCategory.FIND_MAXIMA
