# -*- coding: utf-8 -*-
"""
Tests for footprint-scoped DEM discovery and lazy elevation loading.

A ``Geolocation`` handed a ``dem_path`` no longer indexes the archive at
construction time.  The model is built on first access to
``.elevation``, and tile discovery is restricted to
:meth:`Geolocation.dem_bbox` -- the scene extent taken from sensor
metadata where it is carried, and from the projected perimeter
otherwise.  These tests cover the bbox derivation, the laziness, the
setter that cancels it, ``ChipGeolocation`` delegation, and the
``open_elevation_for`` entry point.

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
2026-08-31

Modified
--------
2026-08-31
"""

# Standard library
from typing import Optional, Tuple

# Third-party
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import Affine as RioAffine

# GRDL internal
from grdl.geolocation.base import Geolocation
from grdl.geolocation.chip import ChipGeolocation
from grdl.geolocation.elevation.constant import ConstantElevation
from grdl.geolocation.elevation.open_elevation import (
    open_elevation,
    open_elevation_for,
)


# ── Fixtures ────────────────────────────────────────────────────────


def _make_dted_tile(path, lon_floor: int, lat_floor: int, size: int = 21):
    """Write a small synthetic north-up DTED tile at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = RioAffine(
        1.0 / (size - 1), 0.0, float(lon_floor),
        0.0, -1.0 / (size - 1), float(lat_floor + 1),
    )
    data = np.full((size, size), 100 + lat_floor, dtype=np.int16)
    with rasterio.open(
        str(path), "w", driver="GTiff", height=size, width=size, count=1,
        dtype=data.dtype, crs="EPSG:4326", transform=transform,
        nodata=-32767,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def wide_archive(tmp_path):
    """A DTED archive spanning far more cells than any one scene."""
    for lon in range(110, 121):
        for lat in range(30, 41):
            _make_dted_tile(
                tmp_path / f"e{lon:03d}" / f"n{lat:02d}.dt2", lon, lat,
            )
    return tmp_path


class LinearGeolocation(Geolocation):
    """Minimal geolocation: a plate-carree ramp over the image grid.

    Carries no metadata corners, so it exercises the projected-perimeter
    branch of :meth:`Geolocation.dem_bbox`.
    """

    def __init__(self, origin=(34.0, 116.0), step=0.001, **kwargs):
        self._origin = origin
        self._step = step
        super().__init__(shape=(100, 100), crs='WGS84', **kwargs)

    def _image_to_latlon_array(self, rows, cols, height=0.0):
        lats = self._origin[0] + self._step * rows
        lons = self._origin[1] + self._step * cols
        heights = np.full(rows.shape, float(np.mean(height)))
        return lats, lons, heights

    def _latlon_to_image_array(self, lats, lons, height=0.0):
        rows = (lats - self._origin[0]) / self._step
        cols = (lons - self._origin[1]) / self._step
        return rows, cols


class CornerGeo(LinearGeolocation):
    """Same ramp, but advertising metadata corners far from the ramp.

    The disagreement is deliberate: it proves ``dem_bbox`` prefers
    metadata over the projected footprint.
    """

    def _metadata_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        return (5.0, 45.0, 6.0, 46.0)


# ── dem_bbox derivation ─────────────────────────────────────────────


class TestDemBbox:

    def test_bbox_from_projected_footprint(self):
        geo = LinearGeolocation()
        min_lon, min_lat, max_lon, max_lat = geo.dem_bbox(pad_deg=0.0)
        assert min_lat == pytest.approx(34.0, abs=1e-9)
        assert max_lat == pytest.approx(34.099, abs=1e-6)
        assert min_lon == pytest.approx(116.0, abs=1e-9)
        assert max_lon == pytest.approx(116.099, abs=1e-6)

    def test_pad_widens_every_side(self):
        geo = LinearGeolocation()
        tight = geo.dem_bbox(pad_deg=0.0)
        padded = geo.dem_bbox(pad_deg=0.25)
        assert padded[0] == pytest.approx(tight[0] - 0.25)
        assert padded[1] == pytest.approx(tight[1] - 0.25)
        assert padded[2] == pytest.approx(tight[2] + 0.25)
        assert padded[3] == pytest.approx(tight[3] + 0.25)

    def test_metadata_bbox_takes_precedence(self):
        geo = CornerGeo()
        assert geo.dem_bbox(pad_deg=0.0) == (5.0, 45.0, 6.0, 46.0)

    def test_pad_clamps_latitude_to_poles(self):
        geo = LinearGeolocation(origin=(89.9, 0.0), step=0.001)
        _, min_lat, _, max_lat = geo.dem_bbox(pad_deg=1.0)
        assert max_lat == 90.0
        assert min_lat == pytest.approx(88.9, abs=1e-6)

    def test_no_geolocation_yields_no_bbox(self):
        from grdl.geolocation.base import NoGeolocation
        assert NoGeolocation(shape=(10, 10)).dem_bbox() is None


# ── Lazy elevation ──────────────────────────────────────────────────


class TestLazyElevation:

    def test_construction_does_not_index_tiles(self, wide_archive, monkeypatch):
        """dem_path must not touch the archive until elevation is used."""
        import grdl.geolocation.base as base_mod
        calls = []
        real = base_mod._build_elevation_model

        def spy(*args, **kwargs):
            calls.append(kwargs.get('bbox'))
            return real(*args, **kwargs)

        monkeypatch.setattr(base_mod, '_build_elevation_model', spy)

        geo = LinearGeolocation(dem_path=str(wide_archive))
        assert calls == []

        assert geo.elevation is not None
        assert len(calls) == 1
        assert calls[0] is not None

    def test_only_footprint_tiles_are_indexed(self, wide_archive):
        """A one-cell scene indexes its cell plus the halo, not 121."""
        geo = LinearGeolocation(dem_path=str(wide_archive))
        model = geo.elevation
        # One 0.1-degree scene: its own cell plus the interpolation halo.
        assert model.tile_count <= 16
        # Unscoped discovery over the same archive sees everything.
        assert open_elevation(str(wide_archive)).tile_count == 121

    def test_elevation_is_built_once(self, wide_archive):
        geo = LinearGeolocation(dem_path=str(wide_archive))
        assert geo.elevation is geo.elevation

    def test_queries_return_terrain(self, wide_archive):
        geo = LinearGeolocation(dem_path=str(wide_archive))
        assert geo.elevation.get_elevation(34.5, 116.5) == pytest.approx(134.0)

    def test_setter_cancels_pending_build(self, wide_archive):
        geo = LinearGeolocation(dem_path=str(wide_archive))
        geo.elevation = ConstantElevation(height=42.0)
        assert isinstance(geo.elevation, ConstantElevation)
        assert geo.elevation.get_elevation(34.5, 116.5) == 42.0

    def test_elevation_model_passed_as_dem_path(self):
        model = ConstantElevation(height=7.0)
        geo = LinearGeolocation(dem_path=model)
        assert geo.elevation is model

    def test_no_dem_path_leaves_elevation_none(self):
        assert LinearGeolocation().elevation is None

    def test_bbox_derivation_sees_no_elevation(self, wide_archive):
        """The footprint projection must not recurse into the DEM build."""
        seen = []

        class Probe(LinearGeolocation):
            def _footprint_bbox(self):
                seen.append(self.elevation)
                return super()._footprint_bbox()

        geo = Probe(dem_path=str(wide_archive))
        assert geo.elevation is not None
        assert seen == [None]


# ── ChipGeolocation delegation ──────────────────────────────────────


class TestChipDelegation:

    def test_chip_does_not_force_parent_build(self, wide_archive):
        parent = LinearGeolocation(dem_path=str(wide_archive))
        ChipGeolocation(parent, 10, 10, (20, 20))
        assert parent._elevation_pending is True

    def test_chip_sees_parent_elevation(self, wide_archive):
        parent = LinearGeolocation(dem_path=str(wide_archive))
        chip = ChipGeolocation(parent, 10, 10, (20, 20))
        assert chip.elevation is parent.elevation

    def test_chip_sees_dem_attached_after_creation(self):
        parent = LinearGeolocation()
        chip = ChipGeolocation(parent, 10, 10, (20, 20))
        assert chip.elevation is None
        parent.elevation = ConstantElevation(height=5.0)
        assert chip.elevation is parent.elevation

    def test_chip_assignment_reaches_parent(self):
        """A DEM set on the chip must land on the parent.

        The chip has no projection engine of its own: both transforms
        delegate.  A model pinned to the chip alone would be reported
        by the property yet ignored by every projection.
        """
        parent = LinearGeolocation()
        parent.elevation = ConstantElevation(height=5.0)
        chip = ChipGeolocation(parent, 10, 10, (20, 20))
        chip.elevation = ConstantElevation(height=9.0)
        assert chip.elevation.get_elevation(0.0, 0.0) == 9.0
        assert parent.elevation.get_elevation(0.0, 0.0) == 9.0

    def test_chip_dem_reaches_an_internal_dem_projection(self):
        """The DEM must reach a parent that resolves terrain itself.

        SICD and SIDD set ``_handles_dem_internally``, so the base
        class skips its own refinement loop and the parent's transform
        reads the parent's ``elevation``.  This is the path on which a
        chip-pinned model used to be dropped silently, orthorectifying
        against the ellipsoid instead of the terrain.
        """
        class InternalDemGeo(LinearGeolocation):
            _handles_dem_internally = True

            def _image_to_latlon_array(self, rows, cols, height=0.0):
                lats, lons, heights = super()._image_to_latlon_array(
                    rows, cols, height,
                )
                if self.elevation is not None:
                    heights = np.broadcast_to(
                        np.asarray(
                            self.elevation.get_elevation(lats, lons),
                            dtype=float,
                        ),
                        lats.shape,
                    ).astype(float)
                return lats, lons, heights

        parent = InternalDemGeo()
        chip = ChipGeolocation(parent, 10, 10, (20, 20))
        chip.elevation = ConstantElevation(height=1234.0)

        coords = chip.image_to_latlon(np.array([[5.0, 5.0]]))
        assert coords[0, 2] == pytest.approx(1234.0)


# ── open_elevation_for ──────────────────────────────────────────────


class TestOpenElevationFor:

    def test_scopes_discovery_to_geolocation(self, wide_archive):
        geo = LinearGeolocation()
        model = open_elevation_for(geo, str(wide_archive))
        assert model.tile_count <= 16
        assert model.get_elevation(34.5, 116.5) == pytest.approx(134.0)

    def test_pad_widens_the_indexed_set(self, wide_archive):
        geo = LinearGeolocation()
        tight = open_elevation_for(geo, str(wide_archive), pad_deg=0.0)
        wide = open_elevation_for(geo, str(wide_archive), pad_deg=2.0)
        assert wide.tile_count > tight.tile_count

    def test_verify_coverage_falls_back_off_archive(self, wide_archive):
        geo = LinearGeolocation(origin=(0.0, 0.0))
        model = open_elevation_for(
            geo, str(wide_archive), fallback_height=12.0,
            verify_coverage=True,
        )
        assert isinstance(model, ConstantElevation)
        assert model.get_elevation(0.0, 0.0) == 12.0

    def test_rejects_unusable_source(self, wide_archive):
        with pytest.raises(TypeError):
            open_elevation_for(object(), str(wide_archive))


# ── Sensor metadata corner sources ──────────────────────────────────


class TestMetadataBbox:
    """Sensor subclasses take the extent straight from their metadata."""

    def test_sicd_uses_image_corners(self):
        from grdl.IO.models.common import LatLon
        from grdl.geolocation.sar.sicd import SICDGeolocation

        corners = [
            LatLon(lat=34.0, lon=116.0), LatLon(lat=34.2, lon=116.4),
            LatLon(lat=33.8, lon=116.5), LatLon(lat=33.7, lon=116.1),
        ]
        geo = SICDGeolocation.__new__(SICDGeolocation)
        geo.metadata = type('M', (), {
            'geo_data': type('G', (), {
                'image_corners': corners, 'valid_data': None,
            })(),
        })()
        assert geo._metadata_bbox() == (116.0, 33.7, 116.5, 34.2)

    def test_sicd_falls_back_to_valid_data(self):
        from grdl.IO.models.common import LatLon
        from grdl.geolocation.sar.sicd import SICDGeolocation

        geo = SICDGeolocation.__new__(SICDGeolocation)
        geo.metadata = type('M', (), {
            'geo_data': type('G', (), {
                'image_corners': None,
                'valid_data': [LatLon(lat=1.0, lon=2.0),
                               LatLon(lat=3.0, lon=4.0)],
            })(),
        })()
        assert geo._metadata_bbox() == (2.0, 1.0, 4.0, 3.0)

    def test_sicd_without_corners_returns_none(self):
        from grdl.geolocation.sar.sicd import SICDGeolocation

        geo = SICDGeolocation.__new__(SICDGeolocation)
        geo.metadata = type('M', (), {'geo_data': None})()
        assert geo._metadata_bbox() is None

    def test_rpc_uses_normalization_domain(self):
        from grdl.IO.models.eo_nitf import RPCCoefficients
        from grdl.geolocation.eo.rpc import RPCGeolocation

        rpc = RPCCoefficients(
            lat_off=34.0, long_off=116.0,
            lat_scale=0.1, long_scale=0.2,
        )
        geo = RPCGeolocation(rpc, shape=(100, 100))
        assert geo._metadata_bbox() == pytest.approx(
            (115.8, 33.9, 116.2, 34.1))

    def test_rpc_degenerate_scale_returns_none(self):
        from grdl.IO.models.eo_nitf import RPCCoefficients
        from grdl.geolocation.eo.rpc import RPCGeolocation

        rpc = RPCCoefficients(lat_off=34.0, long_off=116.0,
                              lat_scale=0.0, long_scale=0.2)
        geo = RPCGeolocation(rpc, shape=(100, 100))
        assert geo._metadata_bbox() is None


# ── Misplaced positional DEM arguments ──────────────────────────────


# All nine geolocation classes, with the leading positional arguments
# each one needs before dem_path.  Every entry must reject a DEM path
# handed to it positionally.
_GEOLOCATION_SIGNATURES = [
    ('grdl.geolocation.sar.sicd', 'SICDGeolocation', 1),
    ('grdl.geolocation.sar.sidd', 'SIDDGeolocation', 1),
    ('grdl.geolocation.sar.nisar', 'NISARGeolocation', 1),
    ('grdl.geolocation.sar.sentinel1_slc', 'Sentinel1SLCGeolocation', 1),
    ('grdl.geolocation.eo.rpc', 'RPCGeolocation', 3),
    ('grdl.geolocation.eo.rsm', 'RSMGeolocation', 5),
    ('grdl.geolocation.eo.affine', 'AffineGeolocation', 3),
    ('grdl.geolocation.eo.corner', 'CornerGeolocation', 4),
]


def _load(module_name, class_name):
    import importlib
    return getattr(importlib.import_module(module_name), class_name)


class TestDemArgsAreKeywordOnly:
    """``dem_path``/``geoid_path`` must never bind positionally.

    Before the sweep the classes disagreed: NISAR and Sentinel-1 took
    ``(metadata, dem_path, geoid_path)``, while SICD took
    ``(metadata, raw_meta, backend)`` and SIDD took
    ``(metadata, refine, dem_path)``.  A DEM path handed to SICD
    positionally bound to ``backend``, left the terrain model
    unattached, and projected every height onto the ellipsoid without
    raising.
    """

    @pytest.mark.parametrize(
        'module_name,class_name,n_leading', _GEOLOCATION_SIGNATURES,
        ids=[c for _, c, _ in _GEOLOCATION_SIGNATURES],
    )
    def test_signature_marks_dem_args_keyword_only(
        self, module_name, class_name, n_leading,
    ):
        import inspect
        cls = _load(module_name, class_name)
        params = inspect.signature(cls.__init__).parameters
        for name in ('dem_path', 'geoid_path', 'interpolation'):
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, (
                f'{class_name}.{name} is still positional'
            )

    @pytest.mark.parametrize(
        'module_name,class_name,n_leading', _GEOLOCATION_SIGNATURES,
        ids=[c for _, c, _ in _GEOLOCATION_SIGNATURES],
    )
    def test_from_reader_marks_dem_args_keyword_only(
        self, module_name, class_name, n_leading,
    ):
        import inspect
        cls = _load(module_name, class_name)
        params = inspect.signature(cls.from_reader).parameters
        for name in ('dem_path', 'geoid_path', 'interpolation'):
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, (
                f'{class_name}.from_reader.{name} is still positional'
            )

    @pytest.mark.parametrize(
        'module_name,class_name,n_leading', _GEOLOCATION_SIGNATURES,
        ids=[c for _, c, _ in _GEOLOCATION_SIGNATURES],
    )
    def test_positional_dem_path_is_rejected(
        self, module_name, class_name, n_leading, wide_archive,
    ):
        """One argument past the last positional slot must not be taken."""
        cls = _load(module_name, class_name)
        args = [object()] * n_leading + [str(wide_archive), '/geoid.pgm']
        with pytest.raises((TypeError, ValueError)):
            cls(*args)

    def test_sicd_names_the_fix_in_its_error(self, wide_archive):
        """SICD absorbs 6 positionals, so the backend guard is the catch."""
        from grdl.geolocation.sar.sicd import SICDGeolocation

        with pytest.raises(ValueError, match='keyword argument'):
            SICDGeolocation(object(), str(wide_archive), '/geoid.pgm')

    def test_sicd_rejects_unknown_backend(self):
        from grdl.geolocation.sar.sicd import _select_backend

        with pytest.raises(ValueError, match='Unknown SICD projection'):
            _select_backend('sarpy2')

    def test_sicd_accepts_the_real_backends(self):
        from grdl.geolocation.sar.sicd import _select_backend

        for name in ('native', 'sarpy', 'sarkit'):
            assert _select_backend(name) == name

    def test_sidd_names_the_fix_in_its_error(self, wide_archive):
        """A lone positional DEM path lands on `refine`; the guard names it."""
        from grdl.geolocation.sar.sidd import SIDDGeolocation

        with pytest.raises(TypeError, match='keyword argument'):
            SIDDGeolocation(object(), str(wide_archive))

    def test_sidd_still_accepts_refine_positionally(self):
        from grdl.geolocation.sar.sidd import SIDDGeolocation

        # Reaches the metadata validation, i.e. `refine` cleared the guard.
        with pytest.raises((ValueError, AttributeError)):
            SIDDGeolocation(object(), False)

    def test_keyword_form_still_works(self, wide_archive):
        from grdl.geolocation.eo.rpc import RPCGeolocation
        from grdl.IO.models.eo_nitf import RPCCoefficients

        rpc = RPCCoefficients(lat_off=34.5, long_off=116.5,
                              lat_scale=0.05, long_scale=0.05)
        geo = RPCGeolocation(rpc, shape=(100, 100),
                             dem_path=str(wide_archive))
        assert geo.elevation.tile_count <= 16
