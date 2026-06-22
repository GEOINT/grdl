# -*- coding: utf-8 -*-
"""
Tests for grdl.data_prep.streaming_stats.

Covers the StreamingStats accumulator (exactness vs numpy, merge
associativity, histogram percentiles), value transforms, valid-pixel
selection, the build_valid_mask rasterizer, compute_image_statistics over a
lightweight in-memory reader, and Normalizer.fit_streaming. Uses synthetic
arrays only -- no real imagery files.

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
2026-06-09
"""

# Standard library
from types import SimpleNamespace

# Third-party
import numpy as np
import pytest

# GRDL internal
from grdl.data_prep import (
    Normalizer,
    StatsResult,
    StreamingStats,
    Tiler,
    build_valid_mask,
    compute_image_statistics,
)
from grdl.data_prep.streaming_stats import VALUE_TRANSFORMS
from grdl.exceptions import ProcessorError, ValidationError


# ---------------------------------------------------------------------------
# Lightweight in-memory reader (serial path; no file/open_any needed)
# ---------------------------------------------------------------------------

class _ArrayReader:
    """Minimal ImageReader stand-in backed by a numpy array."""

    def __init__(self, arr, valid_data=None, first_row=0, first_col=0):
        self._arr = arr
        self.filepath = None  # forces serial path in compute_image_statistics
        image_data = SimpleNamespace(
            valid_data=valid_data, first_row=first_row, first_col=first_col,
        )
        self.metadata = SimpleNamespace(
            image_data=image_data, geo_data=SimpleNamespace(valid_data=None),
        )

    def get_shape(self):
        return self._arr.shape

    def read_chip(self, r0, r1, c0, c1):
        return self._arr[r0:r1, c0:c1]

    def close(self):
        pass


def _tiles(arr, tile):
    return Tiler(arr.shape[0], arr.shape[1], tile_size=tile).partition_positions()


# ---------------------------------------------------------------------------
# StreamingStats core
# ---------------------------------------------------------------------------

class TestStreamingStatsExactness:
    def test_matches_numpy_mean_std_minmax(self):
        rng = np.random.default_rng(1)
        data = rng.normal(5.0, 3.0, size=(500, 400))
        acc = StreamingStats()
        for r in _tiles(data, 128):
            acc.update(data[r.row_start:r.row_end, r.col_start:r.col_end])
        res = acc.result()
        assert res.count == data.size
        assert res.mean == pytest.approx(data.mean(), rel=1e-10)
        assert res.std == pytest.approx(data.std(), rel=1e-10)
        assert res.minimum == data.min()
        assert res.maximum == data.max()

    def test_merge_is_associative(self):
        rng = np.random.default_rng(2)
        data = rng.normal(size=(300, 300))
        single = StreamingStats()
        single.update(data)

        a = StreamingStats()
        a.update(data[:150])
        b = StreamingStats()
        b.update(data[150:])
        a.merge(b)

        assert a.result().count == single.result().count
        assert a.result().mean == pytest.approx(single.result().mean, rel=1e-12)
        assert a.result().std == pytest.approx(single.result().std, rel=1e-12)

    def test_l2_norm_exact(self):
        rng = np.random.default_rng(3)
        data = rng.lognormal(size=(200, 200))
        acc = StreamingStats()
        acc.update(data)
        assert acc.result().l2_norm == pytest.approx(
            np.linalg.norm(data), rel=1e-10
        )

    def test_nonfinite_dropped(self):
        data = np.array([1.0, 2.0, np.nan, np.inf, -np.inf, 3.0])
        acc = StreamingStats()
        acc.update(data)
        res = acc.result()
        assert res.count == 3
        assert res.mean == pytest.approx(2.0)

    def test_empty_update_is_noop(self):
        acc = StreamingStats()
        acc.update(np.array([]))
        res = acc.result()
        assert res.count == 0
        assert np.isnan(res.mean)


class TestStreamingStatsHistogram:
    def test_log_percentiles_close_to_numpy(self):
        rng = np.random.default_rng(4)
        data = rng.lognormal(mean=0.0, sigma=1.5, size=(600, 600))
        acc = StreamingStats(
            percentiles=[1, 50, 95, 99],
            hist_range=(data.min(), data.max()),
            n_bins=65536, hist_spacing='log',
        )
        for r in _tiles(data, 200):
            acc.update(data[r.row_start:r.row_end, r.col_start:r.col_end])
        res = acc.result()
        for q in (1, 50, 95, 99):
            assert res.percentiles[q] == pytest.approx(
                np.percentile(data, q), rel=1e-3
            )

    def test_linear_spacing(self):
        rng = np.random.default_rng(5)
        data = rng.uniform(0.0, 10.0, size=(400, 400))
        acc = StreamingStats(
            percentiles=[50], hist_range=(0.0, 10.0),
            n_bins=4096, hist_spacing='linear',
        )
        acc.update(data)
        assert acc.result().percentiles[50] == pytest.approx(
            np.percentile(data, 50), rel=1e-2
        )

    def test_bad_spacing_raises(self):
        with pytest.raises(ValidationError):
            StreamingStats(hist_range=(1.0, 2.0), hist_spacing='quadratic')

    def test_float32_percentiles_close_to_numpy(self):
        rng = np.random.default_rng(9)
        data = rng.lognormal(mean=0.0, sigma=1.5, size=(600, 600))
        acc = StreamingStats(percentiles=[1, 50, 95, 99],
                             hist_spacing='float32')
        for r in _tiles(data, 200):
            acc.update(data[r.row_start:r.row_end, r.col_start:r.col_end])
        res = acc.result()
        for q in (1, 50, 95, 99):
            assert res.percentiles[q] == pytest.approx(
                np.percentile(data, q), rel=1e-2
            )

    def test_float32_handles_negative_values(self):
        # dB-like data spanning zero exercises the total-order key path.
        rng = np.random.default_rng(10)
        data = rng.normal(loc=-30.0, scale=20.0, size=100_000)
        acc = StreamingStats(percentiles=[5, 50, 95], hist_spacing='float32')
        acc.update(data)
        res = acc.result()
        for q in (5, 50, 95):
            ref = np.percentile(data, q)
            assert res.percentiles[q] == pytest.approx(ref, abs=0.2)

    def test_float32_merge_matches_single_pass(self):
        rng = np.random.default_rng(11)
        data = rng.lognormal(size=20_000)
        single = StreamingStats(percentiles=[50], hist_spacing='float32')
        single.update(data)
        a = StreamingStats(percentiles=[50], hist_spacing='float32')
        a.update(data[:7000])
        b = StreamingStats(percentiles=[50], hist_spacing='float32')
        b.update(data[7000:])
        a.merge(b)
        assert a.result().percentiles[50] == single.result().percentiles[50]
        assert a.result().mean == pytest.approx(single.result().mean)

    def test_float32_zero_and_negative_zero(self):
        acc = StreamingStats(percentiles=[50], hist_spacing='float32')
        acc.update(np.array([-0.0, 0.0, 0.0, 0.0]))
        res = acc.result()
        assert res.count == 4
        assert res.percentiles[50] == pytest.approx(0.0, abs=1e-30)

    def test_float32_bad_n_bins_raises(self):
        with pytest.raises(ValidationError):
            StreamingStats(percentiles=[50], n_bins=1000,
                           hist_spacing='float32')


# ---------------------------------------------------------------------------
# Value transforms
# ---------------------------------------------------------------------------

class TestValueTransforms:
    def test_auto_magnitude_for_complex(self):
        z = np.array([[3 + 4j, 0 + 0j]])
        out = VALUE_TRANSFORMS['auto'](z)
        assert np.allclose(out, [[5.0, 0.0]])

    def test_auto_passthrough_for_real(self):
        a = np.array([[1.0, 2.0]])
        assert np.array_equal(VALUE_TRANSFORMS['auto'](a), a)

    def test_power(self):
        z = np.array([3 + 4j])
        assert VALUE_TRANSFORMS['power'](z)[0] == pytest.approx(25.0)

    def test_decibel(self):
        a = np.array([10.0])
        assert VALUE_TRANSFORMS['decibel'](a)[0] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# compute_image_statistics (serial, via in-memory reader)
# ---------------------------------------------------------------------------

class TestComputeImageStatistics:
    def test_matches_full_load_nonzero(self):
        rng = np.random.default_rng(6)
        data = rng.lognormal(size=(512, 512))
        data[:10, :] = 0.0  # zero-fill band
        reader = _ArrayReader(data)
        res = compute_image_statistics(
            reader, tile=128, transform='identity',
            mask='nonzero_finite', percentiles=[50],
        )
        ref = data[data != 0]
        assert res.count == ref.size
        assert res.mean == pytest.approx(ref.mean(), rel=1e-10)
        assert res.std == pytest.approx(ref.std(), rel=1e-10)
        assert res.percentiles[50] == pytest.approx(
            np.percentile(ref, 50), rel=1e-3
        )

    def test_decibel_nonzero_mask_excludes_zero_fill(self):
        # Zero-fill pixels must not leak into dB stats as floored -200 dB
        # outliers: the nonzero test runs on raw pixels, pre-transform.
        rng = np.random.default_rng(12)
        data = rng.lognormal(size=(128, 128))
        data[:16, :] = 0.0
        res = compute_image_statistics(
            _ArrayReader(data), tile=32, transform='decibel',
            mask='nonzero_finite',
        )
        ref = 20.0 * np.log10(data[data != 0])
        assert res.count == ref.size
        assert res.mean == pytest.approx(ref.mean(), rel=1e-6)
        assert res.minimum == pytest.approx(ref.min(), rel=1e-6)

    def test_none_includes_zeros(self):
        data = np.ones((100, 100))
        data[0, 0] = 0.0
        res = compute_image_statistics(
            _ArrayReader(data), tile=32, transform='identity', mask='none',
        )
        assert res.count == data.size

    def test_complex_auto_transform(self):
        data = (np.ones((64, 64)) * (3 + 4j)).astype(np.complex64)
        res = compute_image_statistics(
            _ArrayReader(data), tile=16, transform='auto', mask='none',
        )
        assert res.mean == pytest.approx(5.0, rel=1e-5)

    def test_bad_transform_raises(self):
        with pytest.raises(ValidationError):
            compute_image_statistics(_ArrayReader(np.ones((8, 8))),
                                     transform='nope')

    def test_bad_mask_raises(self):
        with pytest.raises(ValidationError):
            compute_image_statistics(_ArrayReader(np.ones((8, 8))),
                                     mask='nope')

    def test_mad_matches_numpy(self):
        rng = np.random.default_rng(21)
        data = (rng.standard_normal((600, 700)).astype(np.float32) * 5.0 + 20.0)
        data.ravel()[:200] = 5000.0  # outliers
        flat = data.ravel().astype(np.float64)
        med = np.median(flat)
        mad = np.median(np.abs(flat - med))
        res = compute_image_statistics(
            _ArrayReader(data), tile=128, transform='identity', mad=True,
        )
        assert res.median == pytest.approx(med, abs=0.05)
        # float32 histogram: ~0.55% relative bin width at 65536 bins
        assert res.mad == pytest.approx(mad, rel=0.01)
        assert res.mad_std == pytest.approx(1.4826 * mad, rel=0.01)

    def test_mad_does_not_leak_internal_median_percentile(self):
        data = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
        res = compute_image_statistics(
            _ArrayReader(data), tile=32, transform='identity', mad=True,
        )
        assert 50.0 not in res.percentiles  # internal p50 stripped

    def test_mad_preserves_requested_percentiles(self):
        data = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
        res = compute_image_statistics(
            _ArrayReader(data), tile=32, transform='identity', mad=True,
            percentiles=[50.0, 90.0],
        )
        assert 50.0 in res.percentiles and 90.0 in res.percentiles
        assert np.isfinite(res.mad)

    def test_no_mad_leaves_median_and_mad_nan(self):
        res = compute_image_statistics(
            _ArrayReader(np.ones((32, 32))), tile=16, transform='identity',
        )
        assert not np.isfinite(res.median)
        assert not np.isfinite(res.mad)


# ---------------------------------------------------------------------------
# build_valid_mask
# ---------------------------------------------------------------------------

class TestBuildValidMask:
    def test_pixel_polygon_rasterized(self):
        verts = [SimpleNamespace(row=10, col=10),
                 SimpleNamespace(row=10, col=40),
                 SimpleNamespace(row=40, col=40),
                 SimpleNamespace(row=40, col=10)]
        reader = _ArrayReader(np.ones((64, 64)), valid_data=verts)
        mask = build_valid_mask(reader)
        assert mask.shape == (64, 64)
        assert mask[25, 25]
        assert not mask[5, 5]

    def test_none_when_no_polygon(self):
        assert build_valid_mask(_ArrayReader(np.ones((16, 16)))) is None

    def test_metadata_mask_used_in_stats(self):
        data = np.ones((64, 64)) * 7.0
        verts = [SimpleNamespace(row=0, col=0),
                 SimpleNamespace(row=0, col=31),
                 SimpleNamespace(row=31, col=31),
                 SimpleNamespace(row=31, col=0)]
        reader = _ArrayReader(data, valid_data=verts)
        res = compute_image_statistics(
            reader, tile=16, transform='identity', mask='metadata',
        )
        # Roughly a quarter of the image (32x32 of 64x64), all value 7.
        assert res.mean == pytest.approx(7.0)
        assert 32 * 32 * 0.9 < res.count <= 33 * 33

    def test_tiles_outside_polygon_not_read(self):
        data = np.ones((64, 64))
        verts = [SimpleNamespace(row=0, col=0),
                 SimpleNamespace(row=0, col=31),
                 SimpleNamespace(row=31, col=31),
                 SimpleNamespace(row=31, col=0)]
        reader = _ArrayReader(data, valid_data=verts)
        reads = []
        orig = reader.read_chip
        reader.read_chip = lambda *a: (reads.append(a), orig(*a))[1]
        compute_image_statistics(
            reader, tile=16, transform='identity', mask='metadata',
        )
        # Only the 4 tiles overlapping the 32x32 polygon get read (of 16).
        assert len(reads) == 4


# ---------------------------------------------------------------------------
# Normalizer.fit_streaming
# ---------------------------------------------------------------------------

class TestNormalizerFitStreaming:
    def test_zscore_matches_fit(self):
        rng = np.random.default_rng(7)
        data = rng.normal(3.0, 2.0, size=(256, 256))
        ref = Normalizer(method='zscore').fit(data)
        stream = Normalizer(method='zscore').fit_streaming(
            _ArrayReader(data), tile=64, transform='identity', mask='none',
        )
        assert stream._mean == pytest.approx(ref._mean, rel=1e-10)
        assert stream._std == pytest.approx(ref._std, rel=1e-10)
        # transform produces same output
        chip = data[:32, :32]
        assert np.allclose(stream.transform(chip), ref.transform(chip))

    def test_percentile_method_populates_bounds(self):
        rng = np.random.default_rng(8)
        data = rng.lognormal(size=(256, 256))
        norm = Normalizer(method='percentile', percentile_low=5.0,
                          percentile_high=95.0)
        norm.fit_streaming(_ArrayReader(data), tile=64, transform='identity')
        assert norm.is_fitted
        assert norm._pct_low_val == pytest.approx(
            np.percentile(data, 5.0), rel=1e-2
        )
        assert norm._pct_high_val == pytest.approx(
            np.percentile(data, 95.0), rel=1e-2
        )

    def test_mad_matches_fit(self):
        rng = np.random.default_rng(9)
        data = (rng.standard_normal((256, 256)) * 4.0 + 10.0)
        data.ravel()[:50] = 1e6  # outliers
        ref = Normalizer(method='mad').fit(data)
        stream = Normalizer(method='mad').fit_streaming(
            _ArrayReader(data), tile=64, transform='identity', mask='none',
        )
        assert stream.is_fitted
        assert stream._median == pytest.approx(ref._median, abs=0.05)
        assert stream._mad == pytest.approx(ref._mad, rel=0.01)
        chip = data[:32, :32]
        assert np.allclose(stream.transform(chip), ref.transform(chip),
                           rtol=0.01)

    def test_no_valid_pixels_raises(self):
        data = np.zeros((32, 32))
        with pytest.raises(ProcessorError):
            Normalizer(method='zscore').fit_streaming(
                _ArrayReader(data), mask='nonzero_finite',
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
