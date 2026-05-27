# -*- coding: utf-8 -*-
"""
Tests for TiledGeoDTED — bbox-driven DTED elevation backend.

Covers construction without rglob, bbox-driven indexing, sampling
parity with the legacy DTEDElevation on identical fixtures,
cross-tile interpolation, and pickle safety (file handles dropped).

Dependencies
------------
pytest
rasterio

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

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import Affine as RioAffine

from grdl.geolocation.elevation.tiled_geotiff_dted import (
    TiledGeoDTED,
    _candidate_paths,
    _hemi_lat,
    _hemi_lon,
)
from grdl.geolocation.elevation.dted import DTEDElevation


# ── Fixtures ────────────────────────────────────────────────────────


def _make_dted_tile(
    path: Path,
    lon_floor: int,
    lat_floor: int,
    size: int = 121,
    value_fn=None,
) -> None:
    """Write a synthetic DTED tile at ``path``.

    The transform is north-up with row 0 at ``lat_floor + 1`` (the
    northern edge), matching real DTED layout.  ``value_fn(lat, lon)``
    sets each cell's elevation; defaults to a planar ramp so cross-tile
    continuity is easy to verify.
    """
    if value_fn is None:
        def value_fn(lat, lon):
            return 1000.0 + 10.0 * lon + 5.0 * lat
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = RioAffine(
        1.0 / (size - 1), 0.0, float(lon_floor),
        0.0, -1.0 / (size - 1), float(lat_floor + 1),
    )
    rows = np.arange(size)
    cols = np.arange(size)
    cc, rr = np.meshgrid(cols, rows)
    lons = lon_floor + cc / (size - 1)
    lats = (lat_floor + 1) - rr / (size - 1)
    data = value_fn(lats, lons).astype(np.int16)
    with rasterio.open(
        str(path), "w",
        driver="GTiff",
        height=size, width=size, count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=-32767,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def dted_two_tile_archive(tmp_path):
    """Build a DTED archive with two adjacent tiles for cross-tile tests.

    Tiles cover ``(116° E, 34° N)`` and ``(117° E, 34° N)`` so the
    east edge of the first abuts the west edge of the second.  A few
    decoy directories and files are added so a bbox-driven probe
    must skip them.
    """
    _make_dted_tile(tmp_path / "e116" / "n34.dt2", 116, 34)
    _make_dted_tile(tmp_path / "e117" / "n34.dt2", 117, 34)
    # Decoys: a tile far outside the bbox we'll probe.
    _make_dted_tile(tmp_path / "w073" / "n40.dt2", -73, 40)
    return tmp_path


# ── Construction / indexing ─────────────────────────────────────────


class TestConstruction:

    def test_rejects_missing_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TiledGeoDTED(str(tmp_path / "does_not_exist"))

    def test_rejects_file_input(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("not a directory")
        with pytest.raises(ValueError):
            TiledGeoDTED(str(f))

    def test_bbox_probe_indexes_only_overlapping_tiles(
        self, dted_two_tile_archive,
    ):
        """Bbox-driven discovery must skip the w073/n40 decoy."""
        elev = TiledGeoDTED(
            str(dted_two_tile_archive),
            bbox=(116.0, 34.0, 117.5, 35.0),
        )
        # The bbox covers e116/n34 directly and brushes e117/n34;
        # the 1-cell halo also keeps the southern (lat 33) and northern
        # (lat 35) rows in scope but no such files exist on disk.
        keys = set(elev._tile_index.keys())
        assert (116, 34) in keys
        assert (117, 34) in keys
        assert (-73, 40) not in keys

    def test_no_bbox_scan_indexes_archive(self, dted_two_tile_archive):
        """Without a bbox, all tiles in the archive are indexed."""
        elev = TiledGeoDTED(str(dted_two_tile_archive))
        keys = set(elev._tile_index.keys())
        assert (116, 34) in keys
        assert (117, 34) in keys
        assert (-73, 40) in keys

    def test_no_recursive_rglob_used(self, tmp_path, monkeypatch):
        """``Path.rglob`` must never be called when a bbox is supplied.

        Recursive rglob across thousands of archive tiles is the
        symptom the new backend fixes — guard against future
        regressions by trapping the call.
        """
        _make_dted_tile(tmp_path / "e116" / "n34.dt2", 116, 34)

        from pathlib import Path as _Path
        called = {"count": 0}
        original = _Path.rglob

        def trap(self, pattern):
            called["count"] += 1
            return iter([])

        monkeypatch.setattr(_Path, "rglob", trap)
        TiledGeoDTED(str(tmp_path), bbox=(116.0, 34.0, 116.5, 34.5))
        # Restore for cleanup
        monkeypatch.setattr(_Path, "rglob", original)
        assert called["count"] == 0


# ── Sampling parity with legacy DTEDElevation ───────────────────────


class TestSamplingParity:

    def test_bilinear_matches_legacy(self, dted_two_tile_archive):
        legacy = DTEDElevation(
            str(dted_two_tile_archive), interpolation=1,
        )
        tiled = TiledGeoDTED(
            str(dted_two_tile_archive), interpolation=1,
            bbox=(116.0, 34.0, 117.0, 35.0),
        )
        lats = np.array([34.25, 34.5, 34.75])
        lons = np.array([116.25, 116.5, 116.75])
        h_legacy = legacy.get_elevation(lats, lons)
        h_tiled = tiled.get_elevation(lats, lons)
        np.testing.assert_allclose(h_tiled, h_legacy, atol=1e-6)

    def test_bicubic_matches_legacy(self, dted_two_tile_archive):
        legacy = DTEDElevation(
            str(dted_two_tile_archive), interpolation=3,
        )
        tiled = TiledGeoDTED(
            str(dted_two_tile_archive), interpolation=3,
            bbox=(116.0, 34.0, 117.0, 35.0),
        )
        lats = np.array([34.25, 34.5, 34.75])
        lons = np.array([116.25, 116.5, 116.75])
        h_legacy = legacy.get_elevation(lats, lons)
        h_tiled = tiled.get_elevation(lats, lons)
        np.testing.assert_allclose(h_tiled, h_legacy, atol=1e-3)

    def test_cross_tile_sample(self, dted_two_tile_archive):
        """A point just inside tile e116 near its east edge interpolates
        identically across the two backends, even at the boundary."""
        legacy = DTEDElevation(
            str(dted_two_tile_archive), interpolation=3,
        )
        tiled = TiledGeoDTED(
            str(dted_two_tile_archive), interpolation=3,
            bbox=(116.0, 34.0, 117.5, 35.0),
        )
        lats = np.array([34.5, 34.5, 34.5])
        lons = np.array([116.995, 117.001, 117.005])
        h_legacy = legacy.get_elevation(lats, lons)
        h_tiled = tiled.get_elevation(lats, lons)
        # Boundary continuity is what matters: no NaN, finite values,
        # parity within spline tolerance.
        assert np.all(np.isfinite(h_tiled))
        np.testing.assert_allclose(h_tiled, h_legacy, atol=1.0)

    def test_outside_coverage_is_nan(self, dted_two_tile_archive):
        tiled = TiledGeoDTED(
            str(dted_two_tile_archive),
            bbox=(116.0, 34.0, 117.0, 35.0),
        )
        # Outside the bbox and outside any indexed tile.
        h = tiled.get_elevation(40.0, -73.5)
        assert np.isnan(h)


# ── Pickle safety ────────────────────────────────────────────────────


class TestPickleSafety:

    def test_pickle_round_trip(self, dted_two_tile_archive):
        elev = TiledGeoDTED(
            str(dted_two_tile_archive),
            bbox=(116.0, 34.0, 117.0, 35.0),
        )
        # Warm an open dataset so we are sure __getstate__ closes it.
        _ = elev.get_elevation(34.5, 116.5)
        assert len(elev._open_tiles) > 0
        payload = pickle.dumps(elev)
        restored = pickle.loads(payload)
        # File handles dropped before pickle.
        assert len(restored._open_tiles) == 0
        # Tile index survived intact.
        assert restored.tile_count == elev.tile_count
        # And the restored model can still sample.
        h = restored.get_elevation(34.5, 116.5)
        assert np.isfinite(h)

    def test_pickle_payload_small(self, dted_two_tile_archive):
        """Pickled payload should be metadata-only, well under any
        whole-tile raster size."""
        elev = TiledGeoDTED(
            str(dted_two_tile_archive),
            bbox=(116.0, 34.0, 117.0, 35.0),
        )
        # Force-load one tile into the LRU.
        elev.get_elevation(34.5, 116.5)
        payload = pickle.dumps(elev)
        # Two tiles × few hundred bytes of metadata + bbox + interp
        # should easily fit in well under 64 KB.
        assert len(payload) < 65_536


# ── Helpers ──────────────────────────────────────────────────────────


class TestCandidatePaths:

    def test_hemi_helpers(self):
        assert _hemi_lon(116) == 'e'
        assert _hemi_lon(-73) == 'w'
        assert _hemi_lat(34) == 'n'
        assert _hemi_lat(-12) == 's'

    def test_candidate_paths_cover_case_variants(self, tmp_path):
        # 12 candidates per cell × 3 extensions, with case variants on
        # both the directory and filename, in dt2→dt1→dt0 order.
        cands = list(_candidate_paths(tmp_path, 116, 34))
        names = [p.name for _ext, p in cands]
        dirs = [p.parent.name for _ext, p in cands]
        # dt2 comes first.
        assert any(n.lower() == "n34.dt2" for n in names[:4])
        # All ordered before dt0 appears.
        first_dt0 = next(
            i for i, n in enumerate(names) if n.lower().endswith(".dt0")
        )
        first_dt1 = next(
            i for i, n in enumerate(names) if n.lower().endswith(".dt1")
        )
        first_dt2 = next(
            i for i, n in enumerate(names) if n.lower().endswith(".dt2")
        )
        assert first_dt2 < first_dt1 < first_dt0
        assert "e116" in dirs
        assert "E116" in dirs
