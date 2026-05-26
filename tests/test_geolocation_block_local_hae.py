# -*- coding: utf-8 -*-
"""
Tests for block-local gref in :func:`image_to_ground_hae`.

The R/Rdot iteration was historically driven by a single global ground
reference plane.  For wide swaths that plane is a poor fit at the
edges, costing iterations and accuracy.  The refactor partitions
``im_points`` into spatial blocks and gives each block its own
``gref``.  These tests cover:

- The spatial partition helper produces the right block structure
- Single-block-fast-path keeps existing behaviour for small batches
- Multi-block path agrees with the single-block path on a flat-DEM
  surface (no terrain → block ``gref`` and global ``gref`` converge
  to the same answer within tolerance)
- The per-call DEM cache memoizes and amortizes repeat queries
- Coordinate quantization preserves sub-meter accuracy

Dependencies
------------
pytest

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-05-26

Modified
--------
2026-05-26
"""

import numpy as np
import pytest

from grdl.geolocation.projection import (
    _HAE_BLOCK_PIXELS,
    _HAE_BLOCK_SIZE,
    _cached_dem_query,
    _partition_blocks,
    _quantize_latlon,
)


# ── Partition helper ────────────────────────────────────────────────


class TestPartitionBlocks:

    def test_small_batch_returns_single_block(self):
        pts = np.column_stack([np.arange(100), np.arange(100)])
        blocks = _partition_blocks(pts, _HAE_BLOCK_PIXELS, _HAE_BLOCK_SIZE)
        assert len(blocks) == 1
        np.testing.assert_array_equal(blocks[0], np.arange(100))

    def test_threshold_is_inclusive(self):
        """At exactly ``_HAE_BLOCK_SIZE`` the fast path is taken."""
        n = _HAE_BLOCK_SIZE
        pts = np.column_stack([np.arange(n), np.zeros(n)])
        blocks = _partition_blocks(pts, _HAE_BLOCK_PIXELS, _HAE_BLOCK_SIZE)
        assert len(blocks) == 1

    def test_wide_row_swath_splits_into_blocks(self):
        """20 000 points along one row should split by ``_HAE_BLOCK_PIXELS``."""
        n = 20000
        pts = np.column_stack([np.arange(n), np.zeros(n)])
        blocks = _partition_blocks(pts, _HAE_BLOCK_PIXELS, _HAE_BLOCK_SIZE)
        assert len(blocks) > 1
        # Each block points back into the original array.
        recovered = np.concatenate(blocks)
        assert recovered.shape[0] == n
        assert set(recovered.tolist()) == set(range(n))

    def test_oversized_cell_subsplits(self):
        """A single spatial cell with >block_size points subdivides."""
        # 8 192 points all in the same (row, col) bucket → must split.
        n = _HAE_BLOCK_SIZE * 2
        pts = np.column_stack([
            np.full(n, 100.0),    # all in row bucket 0
            np.full(n, 100.0),    # all in col bucket 0
        ])
        blocks = _partition_blocks(pts, _HAE_BLOCK_PIXELS, _HAE_BLOCK_SIZE)
        assert len(blocks) == 2
        assert blocks[0].size == _HAE_BLOCK_SIZE
        assert blocks[1].size == _HAE_BLOCK_SIZE

    def test_adjacent_pixels_share_a_block(self):
        """The whole point of spatial bucketing: keep neighbours together."""
        pts = np.column_stack([
            np.arange(0, _HAE_BLOCK_PIXELS, 1),
            np.arange(0, _HAE_BLOCK_PIXELS, 1),
        ])
        blocks = _partition_blocks(pts, _HAE_BLOCK_PIXELS, _HAE_BLOCK_SIZE)
        # All in one (row=0, col=0) bucket; one block.
        assert len(blocks) == 1


# ── lat/lon quantization ────────────────────────────────────────────


class TestQuantizeLatlon:

    def test_returns_integer_arrays(self):
        lats = np.array([34.123456, 34.123457])
        lons = np.array([116.0, 116.000001])
        qlat, qlon = _quantize_latlon(lats, lons)
        assert qlat.dtype == np.int64
        assert qlon.dtype == np.int64

    def test_quantization_step_under_meter(self):
        """1e-5 deg ≈ 1.1 m at equator — fine enough for DEM sampling."""
        # Two points 0.5 m apart should map to the same quantized key.
        lats = np.array([34.0, 34.0 + 4e-6])  # ≈ 0.4 m
        lons = np.array([116.0, 116.0])
        qlat, _ = _quantize_latlon(lats, lons)
        assert qlat[0] == qlat[1]


# ── DEM cache ────────────────────────────────────────────────────────


class _CountingDEM:
    """Test double tracking how many points are actually queried."""

    def __init__(self, height: float = 100.0) -> None:
        self.height = height
        self.calls = 0
        self.point_count = 0

    def get_elevation(self, lats, lons):
        lats = np.atleast_1d(lats)
        self.calls += 1
        self.point_count += lats.shape[0]
        return np.full(lats.shape[0], self.height, dtype=np.float64)


class TestCachedDEMQuery:

    def test_first_call_populates_cache(self):
        dem = _CountingDEM(height=250.0)
        cache: dict = {}
        lats = np.array([34.1, 34.2, 34.3])
        lons = np.array([116.1, 116.2, 116.3])
        out = _cached_dem_query(dem, lats, lons, cache)
        np.testing.assert_allclose(out, 250.0)
        assert dem.point_count == 3
        assert len(cache) == 3

    def test_repeat_call_uses_cache(self):
        """Re-querying the same (lat, lon) should not re-hit the DEM."""
        dem = _CountingDEM(height=250.0)
        cache: dict = {}
        lats = np.array([34.1, 34.2, 34.3])
        lons = np.array([116.1, 116.2, 116.3])
        _ = _cached_dem_query(dem, lats, lons, cache)
        assert dem.point_count == 3
        _ = _cached_dem_query(dem, lats, lons, cache)
        # Second call must not have added any new points.
        assert dem.point_count == 3

    def test_partial_hit(self):
        """Mixed hits/misses should only query the misses."""
        dem = _CountingDEM(height=250.0)
        cache: dict = {}
        lats1 = np.array([34.1, 34.2])
        lons1 = np.array([116.1, 116.2])
        _cached_dem_query(dem, lats1, lons1, cache)
        assert dem.point_count == 2

        # Now overlap: 34.1 hits cache, 34.5 is a miss.
        lats2 = np.array([34.1, 34.5])
        lons2 = np.array([116.1, 116.5])
        _cached_dem_query(dem, lats2, lons2, cache)
        # Only one new point queried.
        assert dem.point_count == 3
        # Result must still be correct.
        out = _cached_dem_query(dem, lats2, lons2, cache)
        np.testing.assert_allclose(out, 250.0)


# ── End-to-end: block-local convergence on synthetic geometry ──────


@pytest.fixture
def plane_coa():
    """Reuse the synthetic geometry from test_geolocation_projection."""
    from grdl.IO.models.common import Poly1D, Poly2D, XYZPoly
    from grdl.geolocation.coordinates import WGS84_A
    from grdl.geolocation.projection import COAProjection, _plane_projector

    scp = np.array([WGS84_A, 0.0, 0.0])
    arp_pos = np.array([WGS84_A + 600e3, 0.0, 600e3])
    arp_vel = np.array([0.0, 7500.0, 0.0])
    u_row = (scp - arp_pos) / np.linalg.norm(scp - arp_pos)
    u_col = arp_vel / np.linalg.norm(arp_vel)

    time_coa_poly = Poly2D(coefs=np.array([[2.5]]))
    arp_poly = XYZPoly(
        x=Poly1D(coefs=np.array([
            arp_pos[0] - arp_vel[0] * 2.5, arp_vel[0],
        ])),
        y=Poly1D(coefs=np.array([
            arp_pos[1] - arp_vel[1] * 2.5, arp_vel[1],
        ])),
        z=Poly1D(coefs=np.array([
            arp_pos[2] - arp_vel[2] * 2.5, arp_vel[2],
        ])),
    )
    projector = _plane_projector(
        scp_ecf=scp, u_row=u_row, u_col=u_col, row_ss=1.0, col_ss=1.0,
    )
    coa = COAProjection(
        time_coa_poly=time_coa_poly, arp_poly=arp_poly,
        method_projection=projector,
        scp_pixel=(500.0, 500.0),
        row_ss=1.0, col_ss=1.0,
    )
    return coa, scp


class TestBlockLocalConvergence:

    def test_single_block_matches_legacy_behaviour(self, plane_coa):
        """For N ≤ block size the single-block fast path is taken; this
        is the previous code path exactly, so SCP must still project
        near SCP at HAE=0."""
        from grdl.geolocation.coordinates import ecef_to_geodetic
        from grdl.geolocation.projection import image_to_ground_hae

        coa, scp = plane_coa
        im_points = np.array([[500.0, 500.0]])
        gpp = image_to_ground_hae(
            coa, im_points, hae=0.0, scp_ecf=scp,
            max_iter=15, tol=1.0,
        )
        geo = ecef_to_geodetic(gpp)
        assert abs(geo[0, 0]) < 5.0
        assert abs(geo[0, 1]) < 5.0
        assert abs(geo[0, 2]) < 100.0

    def test_multi_block_path_runs(self, plane_coa):
        """A batch larger than _HAE_BLOCK_SIZE must produce a finite
        ECF for every point and the right output shape."""
        from grdl.geolocation.projection import image_to_ground_hae

        coa, scp = plane_coa
        n = _HAE_BLOCK_SIZE + 100
        rows = np.linspace(200.0, 800.0, n)
        cols = np.linspace(200.0, 800.0, n)
        im_points = np.column_stack([rows, cols])
        gpp = image_to_ground_hae(
            coa, im_points, hae=0.0, scp_ecf=scp,
            max_iter=15, tol=1.0,
        )
        assert gpp.shape == (n, 3)
        assert np.all(np.isfinite(gpp))

    def test_block_local_matches_single_block_on_uniform_dem(
        self, plane_coa,
    ):
        """With a constant-elevation DEM (no terrain) the block-local
        and single-block solutions must agree to within the convergence
        tolerance; both project to the same flat surface."""
        from grdl.geolocation.projection import image_to_ground_hae

        coa, scp = plane_coa
        # Two batches: one small (fast path), one large (block path).
        rows_small = np.linspace(490.0, 510.0, 4)
        cols_small = np.linspace(490.0, 510.0, 4)
        small = np.column_stack([rows_small, cols_small])
        rows_big = np.tile(rows_small, _HAE_BLOCK_SIZE + 4)[
            : _HAE_BLOCK_SIZE + 4
        ]
        cols_big = np.tile(cols_small, _HAE_BLOCK_SIZE + 4)[
            : _HAE_BLOCK_SIZE + 4
        ]
        big = np.column_stack([rows_big, cols_big])

        elev = _CountingDEM(height=42.0)

        small_gpp = image_to_ground_hae(
            coa, small, hae=42.0, scp_ecf=scp,
            elevation_model=elev, nan_fill_height=42.0,
            max_iter=15, tol=1.0,
        )
        # Drop the dem call counter before the second run.
        elev2 = _CountingDEM(height=42.0)
        big_gpp = image_to_ground_hae(
            coa, big, hae=42.0, scp_ecf=scp,
            elevation_model=elev2, nan_fill_height=42.0,
            max_iter=15, tol=1.0,
        )

        # Every small point should appear (modulo duplicates) inside
        # big_gpp — same image pixels project to the same ground.
        for i in range(small.shape[0]):
            dists = np.linalg.norm(big_gpp - small_gpp[i], axis=1)
            assert np.min(dists) < 5.0, (
                f"Block-local result for pixel {i} drifted "
                f"{np.min(dists):.2f} m from the single-block result"
            )

    def test_dem_cache_amortizes_iterations(self, plane_coa):
        """When the same pixel cluster iterates several times, the
        per-call cache must prevent N×iter duplicate DEM queries."""
        from grdl.geolocation.projection import image_to_ground_hae

        coa, scp = plane_coa
        rows = np.full(64, 500.0)
        cols = np.full(64, 500.0)
        im_points = np.column_stack([rows, cols])
        elev = _CountingDEM(height=100.0)
        _ = image_to_ground_hae(
            coa, im_points, hae=100.0, scp_ecf=scp,
            elevation_model=elev, nan_fill_height=100.0,
            max_iter=10, tol=1.0,
        )
        # 64 quantized points → at most 64 unique DEM samples
        # regardless of iteration count.
        assert elev.point_count <= 64, (
            f"DEM cache failed: {elev.point_count} samples for 64 points"
        )
