# -*- coding: utf-8 -*-
"""
Tests for DTEDElevation nested-archive discovery and bbox probing.

``DTEDElevation`` supports two on-disk layouts: the standard
``<root>/<lon>/<lat>.dt?`` and the nested archive
``<root>/dted/dted{2,1,0}/<lon>/<lat>.dt?``. When a scene ``bbox`` is
supplied, discovery probes candidate paths for each overlapping cell in
resolution-priority order; otherwise a recursive scan handles the
standard layout. These tests cover both modes plus the layout-agnostic
``open_elevation`` probe and ``_try_dted`` loader.

Dependencies
------------
pytest
rasterio

Author
------
Duane Smalley
duane.d.smalley@gmail.com

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-06-08

Modified
--------
2026-06-08
"""

# Standard library
from pathlib import Path

# Third-party
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import Affine as RioAffine

# GRDL internal
from grdl.geolocation.elevation.dted import (
    DTEDElevation,
    _CaseInsensitiveDirCache,
    _dted_candidate_paths,
    _find_dted_file,
    _lat_base_name,
    _lon_dir_name,
)
from grdl.geolocation.elevation.open_elevation import (
    _has_dted_files,
    _try_dted,
)


# ── Fixtures ────────────────────────────────────────────────────────


def _make_dted_tile(
    path: Path,
    lon_floor: int,
    lat_floor: int,
    size: int = 121,
    value_fn=None,
) -> None:
    """Write a synthetic north-up DTED tile at ``path``.

    Row 0 sits at ``lat_floor + 1`` (northern edge), matching real DTED.
    ``value_fn(lat, lon)`` sets each cell's elevation; the default is a
    planar ramp so values are easy to predict.
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
def nested_archive(tmp_path):
    """Nested DTED archive: <root>/dted/dted2/e116/n34.dt2 (+ neighbors)."""
    base = tmp_path / "dted" / "dted2"
    _make_dted_tile(base / "e116" / "n34.dt2", 116, 34)
    _make_dted_tile(base / "e117" / "n34.dt2", 117, 34)
    return tmp_path


# ── Candidate-path helper ───────────────────────────────────────────


class TestCandidatePaths:

    def test_name_helpers(self):
        assert _lon_dir_name(116) == "e116"
        assert _lon_dir_name(-74) == "w074"
        assert _lat_base_name(34) == "n34"
        assert _lat_base_name(-12) == "s12"

    def test_priority_order(self, tmp_path):
        """Nested dted2/dted1/dted0 then standard, .dt2 before .dt0."""
        cands = list(_dted_candidate_paths(tmp_path, 116, 34))
        rel = [str(p.relative_to(tmp_path)) for p in cands]
        # First four are the .dt2 candidates: nested then standard.
        assert rel[0] == str(Path("dted/dted2/e116/n34.dt2"))
        assert rel[1] == str(Path("dted/dted1/e116/n34.dt2"))
        assert rel[2] == str(Path("dted/dted0/e116/n34.dt2"))
        assert rel[3] == str(Path("e116/n34.dt2"))
        # .dt2 group strictly precedes the .dt1 and .dt0 groups.
        first_dt1 = next(i for i, r in enumerate(rel) if r.endswith(".dt1"))
        first_dt0 = next(i for i, r in enumerate(rel) if r.endswith(".dt0"))
        assert 3 < first_dt1 < first_dt0


# ── bbox discovery of nested layout ─────────────────────────────────


class TestBboxDiscovery:

    def test_indexes_nested_tiles(self, nested_archive):
        elev = DTEDElevation(
            str(nested_archive), bbox=(116.0, 34.0, 117.5, 34.5),
        )
        keys = set(elev._tile_index)
        assert (116, 34) in keys
        assert (117, 34) in keys

    def test_samples_nested_tile(self, nested_archive):
        elev = DTEDElevation(
            str(nested_archive), interpolation=1,
            bbox=(116.0, 34.0, 117.0, 35.0),
        )
        h = elev.get_elevation(34.5, 116.5)
        # Planar ramp: 1000 + 10*lon + 5*lat
        assert np.isfinite(h)
        np.testing.assert_allclose(h, 1000.0 + 10 * 116.5 + 5 * 34.5, atol=1.0)

    def test_highest_resolution_wins(self, tmp_path):
        """A .dt2 in dted2/ supersedes a .dt1 in dted1/ for the same cell."""
        _make_dted_tile(
            tmp_path / "dted" / "dted2" / "e116" / "n34.dt2", 116, 34,
        )
        _make_dted_tile(
            tmp_path / "dted" / "dted1" / "e116" / "n34.dt1", 116, 34,
        )
        elev = DTEDElevation(
            str(tmp_path), bbox=(116.2, 34.2, 116.8, 34.8),
        )
        assert elev._tile_index[(116, 34)].suffix == ".dt2"

    def test_falls_back_to_lower_resolution(self, tmp_path):
        """Only .dt1 present → that cell selects the .dt1 file."""
        _make_dted_tile(
            tmp_path / "dted" / "dted1" / "e116" / "n34.dt1", 116, 34,
        )
        elev = DTEDElevation(
            str(tmp_path), bbox=(116.2, 34.2, 116.8, 34.8),
        )
        assert elev._tile_index[(116, 34)].suffix == ".dt1"

    def test_standard_layout_still_discovered_with_bbox(self, tmp_path):
        """bbox discovery also finds the plain <root>/<lon>/<lat>.dt? form."""
        _make_dted_tile(tmp_path / "e116" / "n34.dt2", 116, 34)
        elev = DTEDElevation(
            str(tmp_path), bbox=(116.2, 34.2, 116.8, 34.8),
        )
        assert (116, 34) in elev._tile_index
        assert elev._tile_index[(116, 34)].suffix == ".dt2"

    def test_no_rglob_when_bbox(self, nested_archive, monkeypatch):
        """A bbox must avoid the recursive rglob entirely."""
        called = {"n": 0}
        original = Path.rglob

        def trap(self, pattern):
            called["n"] += 1
            return iter([])

        monkeypatch.setattr(Path, "rglob", trap)
        DTEDElevation(str(nested_archive), bbox=(116.0, 34.0, 116.5, 34.5))
        monkeypatch.setattr(Path, "rglob", original)
        assert called["n"] == 0


# ── recursive fallback unchanged ────────────────────────────────────


class TestRecursiveFallback:

    def test_standard_layout_no_bbox(self, tmp_path):
        _make_dted_tile(tmp_path / "e116" / "n34.dt2", 116, 34)
        elev = DTEDElevation(str(tmp_path))
        assert (116, 34) in elev._tile_index

    def test_nested_layout_found_recursively(self, nested_archive):
        """rglob descends into the nested dirs even without a bbox."""
        elev = DTEDElevation(str(nested_archive))
        assert (116, 34) in elev._tile_index
        assert (117, 34) in elev._tile_index


# ── caches ───────────────────────────────────────────────────────────


class TestCaches:

    def test_caches_present_and_clearable(self, nested_archive):
        elev = DTEDElevation(
            str(nested_archive), interpolation=3,
            bbox=(116.0, 34.0, 117.0, 35.0),
        )
        # Query near an edge so the padded-tile path is exercised.
        elev.get_elevation(34.999, 116.999)
        assert elev._inv_transform_cache
        elev.clear_cache()
        assert not elev._inv_transform_cache
        assert not elev._padded_tile_cache
        assert not elev._tile_cache


# ── open_elevation integration ──────────────────────────────────────


class TestOpenElevationProbe:

    def test_has_dted_files_bbox_nested(self, nested_archive):
        assert _has_dted_files(
            nested_archive, bbox=(116.0, 34.0, 117.0, 35.0)
        )

    def test_has_dted_files_no_bbox_nested(self, nested_archive):
        assert _has_dted_files(nested_archive)

    def test_has_dted_files_no_bbox_standard(self, tmp_path):
        _make_dted_tile(tmp_path / "e116" / "n34.dt2", 116, 34)
        assert _has_dted_files(tmp_path)

    def test_has_dted_files_empty(self, tmp_path):
        assert not _has_dted_files(tmp_path)
        assert not _has_dted_files(tmp_path, bbox=(116.0, 34.0, 117.0, 35.0))

    def test_try_dted_loads_nested(self, nested_archive):
        # _try_dted prefers TiledGeoDTED but may fall back to
        # DTEDElevation; either is a valid DTED backend with the nested
        # tile indexed and a finite sample.
        from grdl.geolocation.elevation.tiled_geotiff_dted import (
            TiledGeoDTED,
        )
        model = _try_dted(
            nested_archive, None, location=(34.5, 116.5),
            interpolation=1, bbox=(116.0, 34.0, 117.0, 35.0),
        )
        assert isinstance(model, (TiledGeoDTED, DTEDElevation))
        assert model.tile_count >= 1
        assert np.isfinite(model.get_elevation(34.5, 116.5))

    def test_try_dted_none_when_empty(self, tmp_path):
        model = _try_dted(
            tmp_path, None, location=None,
            bbox=(116.0, 34.0, 117.0, 35.0),
        )
        assert model is None

    def test_try_dted_none_when_no_coverage_at_location(self, nested_archive):
        """Files exist but the location is far outside coverage → None."""
        model = _try_dted(
            nested_archive, None, location=(0.0, 0.0),
            interpolation=1, bbox=(116.0, 34.0, 117.0, 35.0),
        )
        assert model is None


# ── OS independence (case-insensitive resolution) ───────────────────


class TestCaseInsensitiveDiscovery:
    """Discovery must work regardless of on-disk casing, on any OS.

    The resolver matches canonical lowercase query components against the
    real directory entries, so an uppercase archive
    (``E116/N34.DT2``, ``DTED/DTED2/...``) is found even on a
    case-sensitive filesystem. These tests assert against the *actual*
    on-disk names returned by the resolver, which proves case-insensitive
    matching independently of the test host's filesystem semantics.
    """

    def test_dir_cache_resolves_uppercase(self, tmp_path):
        (tmp_path / "E116").mkdir()
        (tmp_path / "E116" / "N34.DT2").write_bytes(b"x")
        cache = _CaseInsensitiveDirCache()
        resolved = cache.resolve(tmp_path, ("e116", "n34.dt2"))
        assert resolved is not None
        # The resolver returns the real stored names, not the query case.
        assert resolved.name == "N34.DT2"
        assert resolved.parent.name == "E116"

    def test_dir_cache_missing_returns_none(self, tmp_path):
        cache = _CaseInsensitiveDirCache()
        assert cache.resolve(tmp_path, ("e116", "n34.dt2")) is None

    def test_find_uppercase_standard(self, tmp_path):
        (tmp_path / "E116").mkdir()
        (tmp_path / "E116" / "N34.DT2").write_bytes(b"x")
        cache = _CaseInsensitiveDirCache()
        found = _find_dted_file(tmp_path, 116, 34, cache)
        assert found is not None
        assert found.name == "N34.DT2"

    def test_find_uppercase_nested(self, tmp_path):
        nested = tmp_path / "DTED" / "DTED2" / "E116"
        nested.mkdir(parents=True)
        (nested / "N34.DT2").write_bytes(b"x")
        cache = _CaseInsensitiveDirCache()
        found = _find_dted_file(tmp_path, 116, 34, cache)
        assert found is not None
        assert found.parent.parent.name == "DTED2"

    def test_dir_cache_scans_each_dir_once(self, tmp_path, monkeypatch):
        """A bbox-style sweep must not re-scan shared parent directories."""
        import os as _os
        # Two tiles under the same lon dir so the parent is shared.
        (tmp_path / "e116").mkdir()
        (tmp_path / "e116" / "n34.dt2").write_bytes(b"x")
        (tmp_path / "e116" / "n35.dt2").write_bytes(b"x")

        scanned = []
        original = _os.scandir

        def counting_scandir(path, *a, **k):
            scanned.append(str(path))
            return original(path, *a, **k)

        monkeypatch.setattr(_os, "scandir", counting_scandir)
        cache = _CaseInsensitiveDirCache()
        _find_dted_file(tmp_path, 116, 34, cache)
        _find_dted_file(tmp_path, 116, 35, cache)
        # Each unique directory listed at most once despite two lookups.
        assert len(scanned) == len(set(scanned))

    def test_dtedelevation_indexes_uppercase_nested_with_bbox(self, tmp_path):
        nested = tmp_path / "DTED" / "DTED2" / "E116"
        _make_dted_tile(nested / "N34.DT2", 116, 34)
        elev = DTEDElevation(
            str(tmp_path), interpolation=1,
            bbox=(116.2, 34.2, 116.8, 34.8),
        )
        assert (116, 34) in elev._tile_index
        h = elev.get_elevation(34.5, 116.5)
        assert np.isfinite(h)
