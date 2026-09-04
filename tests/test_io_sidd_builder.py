# -*- coding: utf-8 -*-
"""
Tests for SIDD product construction and writing.

Covers the grid-to-projection mapping in
``grdl.IO.sar.sidd_builder``, the valid-data polygon projection, the
GRDL-to-sarpy metadata conversion in ``grdl.IO.sar.sidd_writer``, and
the full write/read round trip through ``SIDDReader`` and
``SIDDGeolocation``.

All fixtures are synthetic: a small analytic geolocation stands in for
a real SICD, so the suite needs no sample imagery.

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-09-03

Modified
--------
2026-09-03
"""

# Third-party
import numpy as np
import pytest

# GRDL internal
from grdl.exceptions import ValidationError
from grdl.geolocation.coordinates import geodetic_to_ecef
from grdl.IO.models.common import (
    LatLonHAE, Poly1D, Poly2D, RowCol, XYZ, XYZPoly,
)
from grdl.IO.models.sicd import (
    SICDCollectionInfo,
    SICDDirParam,
    SICDGeoData,
    SICDGrid,
    SICDImageData,
    SICDImageFormation,
    SICDMetadata,
    SICDPosition,
    SICDRadarMode,
    SICDSCPCOA,
    SICDSCP,
    SICDTimeline,
)
from grdl.image_processing.ortho.enu_grid import ENUGrid
from grdl.image_processing.ortho.ortho import GeographicGrid
from grdl.image_processing.ortho.rotated_enu_grid import RotatedENUGrid
from grdl.image_processing.ortho.utm_grid import UTMGrid
from grdl.IO.sar.sidd_builder import (
    _clip_to_rect,
    _order_clockwise,
    _simplify_convex,
    build_sidd_metadata,
    infer_pixel_type,
    to_display_samples,
)

sarpy = pytest.importorskip('sarpy', reason='sarpy required for SIDD')

REF_LAT, REF_LON = 33.5, -117.2


# ===================================================================
# Fixtures
# ===================================================================

class _AnalyticGeolocation:
    """Minimal geolocation: a linear pixel-to-degrees mapping.

    Stands in for a SICD geolocation so the builder can be exercised
    without sample imagery.  Implements only the two array methods the
    builder calls, in the stacked form the public API uses.
    """

    def __init__(self, rows: int = 400, cols: int = 600,
                 scale: float = 1e-5) -> None:
        self.rows = rows
        self.cols = cols
        self.scale = scale
        self.elevation = None

    def image_to_latlon(self, pixels: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pixels, dtype=np.float64))
        lat = REF_LAT - (p[:, 0] - self.rows / 2) * self.scale
        lon = REF_LON + (p[:, 1] - self.cols / 2) * self.scale
        return np.column_stack([lat, lon, np.zeros_like(lat)])

    def latlon_to_image(self, latlon: np.ndarray) -> np.ndarray:
        a = np.atleast_2d(np.asarray(latlon, dtype=np.float64))
        row = (REF_LAT - a[:, 0]) / self.scale + self.rows / 2
        col = (a[:, 1] - REF_LON) / self.scale + self.cols / 2
        return np.column_stack([row, col])


def _sicd_metadata(with_valid_data: bool = True) -> SICDMetadata:
    """Build synthetic SICD metadata with every section the builder uses."""
    scp_ecf = geodetic_to_ecef(np.array([REF_LAT, REF_LON, 0.0]))
    # Sensor 700 km up and 300 km to the side of the scene centre.
    arp = scp_ecf + np.array([300_000.0, 0.0, 700_000.0])

    valid = None
    if with_valid_data:
        valid = [
            RowCol(row=20.0, col=30.0),
            RowCol(row=20.0, col=560.0),
            RowCol(row=370.0, col=560.0),
            RowCol(row=370.0, col=30.0),
        ]

    return SICDMetadata(
        format='SICD', rows=400, cols=600, dtype='complex64',
        collection_info=SICDCollectionInfo(
            collector_name='TESTSAT',
            core_name='TEST_COLLECT_001',
            radar_mode=SICDRadarMode(mode_type='SPOTLIGHT'),
        ),
        image_data=SICDImageData(
            num_rows=400, num_cols=600,
            first_row=0, first_col=0,
            scp_pixel=RowCol(row=200.0, col=300.0),
            valid_data=valid,
        ),
        geo_data=SICDGeoData(
            scp=SICDSCP(
                ecf=XYZ.from_array(scp_ecf),
                llh=LatLonHAE(lat=REF_LAT, lon=REF_LON, hae=0.0),
            ),
        ),
        grid=SICDGrid(
            time_coa_poly=Poly2D(coefs=np.array([[1.5]])),
            row=SICDDirParam(ss=0.5, imp_resp_bw=2.0, imp_resp_wid=0.45),
            col=SICDDirParam(ss=0.5, imp_resp_bw=1.0, imp_resp_wid=0.9),
        ),
        timeline=SICDTimeline(
            collect_start='2024-11-11T19:07:13.000000Z',
            collect_duration=1.67,
        ),
        position=SICDPosition(
            arp_poly=XYZPoly(
                x=Poly1D(coefs=np.array([arp[0], 100.0])),
                y=Poly1D(coefs=np.array([arp[1], 7000.0])),
                z=Poly1D(coefs=np.array([arp[2], -50.0])),
            ),
        ),
        scpcoa=SICDSCPCOA(
            scp_time=1.5,
            arp_pos=XYZ.from_array(arp),
            arp_vel=XYZ(x=100.0, y=7000.0, z=-50.0),
        ),
        image_formation=SICDImageFormation(
            tx_rcv_polarization_proc='H:H',
        ),
    )


def _enu_grid(pixel: float = 2.0, half: float = 300.0) -> ENUGrid:
    return ENUGrid(
        ref_lat=REF_LAT, ref_lon=REF_LON, ref_alt=0.0,
        min_east=-half, max_east=half,
        min_north=-half, max_north=half,
        pixel_size_east=pixel, pixel_size_north=pixel,
    )


def _grid_error_metres(grid, geo, samples: int = 5) -> float:
    """Max ground distance between a grid and a geolocation, in metres."""
    rr, cc = np.meshgrid(
        np.linspace(0, grid.rows - 1, samples),
        np.linspace(0, grid.cols - 1, samples),
        indexing='ij',
    )
    want_lat, want_lon = grid.image_to_latlon(rr.ravel(), cc.ravel())
    got = geo.image_to_latlon(np.column_stack([rr.ravel(), cc.ravel()]))
    dlat = (got[:, 0] - np.asarray(want_lat)) * 111320.0
    dlon = (
        (got[:, 1] - np.asarray(want_lon))
        * 111320.0 * np.cos(np.radians(REF_LAT))
    )
    return float(np.max(np.hypot(dlat, dlon)))


# ===================================================================
# Pixel type helpers
# ===================================================================

class TestPixelTypes:
    """Tests for pixel type inference and display sample conversion."""

    def test_infer_mono8(self):
        assert infer_pixel_type(np.zeros((4, 4), np.uint8)) == 'MONO8I'

    def test_infer_mono16(self):
        assert infer_pixel_type(np.zeros((4, 4), np.uint16)) == 'MONO16I'

    def test_infer_rgb_band_first(self):
        assert infer_pixel_type(np.zeros((3, 4, 5), np.uint8)) == 'RGB24I'

    def test_infer_rgb_band_last(self):
        assert infer_pixel_type(np.zeros((4, 5, 3), np.uint8)) == 'RGB24I'

    def test_infer_rejects_1d(self):
        with pytest.raises(ValidationError, match='2D or 3D'):
            infer_pixel_type(np.zeros(10, np.uint8))

    def test_infer_rejects_two_band(self):
        with pytest.raises(ValidationError, match='three bands'):
            infer_pixel_type(np.zeros((2, 4, 5), np.uint8))

    def test_display_samples_scale_to_uint8(self):
        out = to_display_samples(
            np.array([[0.0, 0.5, 1.0]], np.float32), 'MONO8I',
        )
        assert out.dtype == np.uint8
        assert out.tolist() == [[0, 127, 255]]

    def test_display_samples_scale_to_uint16(self):
        out = to_display_samples(
            np.array([[0.0, 1.0]], np.float32), 'MONO16I',
        )
        assert out.tolist() == [[0, 65535]]

    def test_display_samples_nan_becomes_nodata(self):
        out = to_display_samples(
            np.array([[np.nan, 1.0]], np.float32), 'MONO8I', nodata=7,
        )
        assert out.tolist() == [[7, 255]]

    def test_display_samples_passes_integers_through(self):
        src = np.arange(4, dtype=np.uint8).reshape(2, 2)
        assert to_display_samples(src, 'MONO8I') is src

    def test_display_samples_rejects_wrong_integer_dtype(self):
        with pytest.raises(ValidationError, match='requires dtype uint8'):
            to_display_samples(np.zeros((2, 2), np.uint16), 'MONO8I')

    def test_display_samples_rejects_bad_pixel_type(self):
        with pytest.raises(ValidationError, match='Unsupported pixel_type'):
            to_display_samples(np.zeros((2, 2), np.float32), 'MONO4I')


# ===================================================================
# Polygon helpers
# ===================================================================

class TestPolygonHelpers:
    """Tests for the valid-data polygon geometry helpers."""

    def test_clip_keeps_interior_polygon(self):
        poly = np.array([[1.0, 1.0], [1.0, 5.0], [5.0, 5.0], [5.0, 1.0]])
        out = _clip_to_rect(poly, 10, 10)
        assert out.shape[0] == 4
        assert out[:, 0].min() >= 1.0 - 1e-9

    def test_clip_trims_to_rectangle(self):
        poly = np.array([
            [-10.0, -10.0], [-10.0, 20.0], [20.0, 20.0], [20.0, -10.0],
        ])
        out = _clip_to_rect(poly, 10, 10)
        assert out[:, 0].min() >= -1e-9
        assert out[:, 0].max() <= 9.0 + 1e-9
        assert out[:, 1].max() <= 9.0 + 1e-9

    def test_clip_returns_empty_when_disjoint(self):
        poly = np.array([
            [100.0, 100.0], [100.0, 110.0], [110.0, 110.0], [110.0, 100.0],
        ])
        assert _clip_to_rect(poly, 10, 10).shape[0] == 0

    def test_order_clockwise_starts_at_min_row(self):
        poly = np.array([[5.0, 5.0], [0.0, 5.0], [0.0, 0.0], [5.0, 0.0]])
        out = _order_clockwise(poly)
        assert out[0].tolist() == [0.0, 0.0]

    def test_order_clockwise_is_clockwise_in_image(self):
        # Counterclockwise on screen; must come back clockwise.
        poly = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
        out = _order_clockwise(poly)
        x, y = out[:, 1], out[:, 0]
        area = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        assert area > 0

    def test_simplify_drops_collinear_vertices(self):
        # A square with extra points along one edge.
        poly = np.array([
            [0.0, 0.0], [0.0, 3.0], [0.0, 6.0], [0.0, 9.0],
            [9.0, 9.0], [9.0, 0.0],
        ])
        out = _simplify_convex(poly, tol=0.5)
        assert out.shape[0] == 4

    def test_simplify_keeps_real_corners(self):
        poly = np.array([[0.0, 0.0], [0.0, 9.0], [9.0, 9.0], [9.0, 0.0]])
        assert _simplify_convex(poly, tol=0.5).shape[0] == 4

    def test_simplify_never_below_three(self):
        poly = np.array([[0.0, 0.0], [0.0, 1e-6], [1e-6, 1e-6]])
        assert _simplify_convex(poly, tol=10.0).shape[0] == 3


# ===================================================================
# Grid to projection mapping
# ===================================================================

class TestProjectionSelection:
    """Each grid type maps to the right SIDD projection."""

    def test_enu_grid_gives_plane_projection(self):
        grid = _enu_grid()
        meta = build_sidd_metadata(grid, (grid.rows, grid.cols))
        assert meta.measurement.projection_type == 'PlaneProjection'
        assert meta.measurement.plane_projection is not None

    def test_rotated_enu_gives_plane_projection(self):
        grid = RotatedENUGrid(
            ref_lat=REF_LAT, ref_lon=REF_LON, ref_alt=0.0, angle=0.6,
            min_u=-200.0, max_u=200.0, min_v=-150.0, max_v=150.0,
            pixel_size=2.0,
        )
        meta = build_sidd_metadata(grid, (grid.rows, grid.cols))
        assert meta.measurement.projection_type == 'PlaneProjection'

    def test_geographic_grid_gives_geographic_projection(self):
        grid = GeographicGrid(
            REF_LAT - 0.01, REF_LAT + 0.01,
            REF_LON - 0.01, REF_LON + 0.01, 1e-4, 1e-4,
        )
        meta = build_sidd_metadata(grid, (grid.rows, grid.cols))
        assert meta.measurement.projection_type == 'GeographicProjection'
        assert meta.measurement.geographic_projection is not None

    def test_utm_grid_falls_back_to_polynomial(self):
        grid = UTMGrid(
            zone=11, north=True,
            min_easting=480_000.0, max_easting=481_000.0,
            min_northing=3_707_000.0, max_northing=3_708_000.0,
            pixel_size=5.0,
        )
        meta = build_sidd_metadata(grid, (grid.rows, grid.cols))
        assert meta.measurement.projection_type == 'PolynomialProjection'
        poly = meta.measurement.polynomial_projection
        assert poly.row_col_to_lat is not None
        assert poly.lat_lon_to_col is not None

    def test_explicit_projection_override(self):
        grid = _enu_grid()
        meta = build_sidd_metadata(
            grid, (grid.rows, grid.cols),
            projection='PolynomialProjection',
        )
        assert meta.measurement.projection_type == 'PolynomialProjection'

    def test_unknown_projection_rejected(self):
        grid = _enu_grid()
        with pytest.raises(ValidationError, match='Unknown projection'):
            build_sidd_metadata(
                grid, (grid.rows, grid.cols), projection='Mercator',
            )

    def test_enu_sample_spacing_matches_grid(self):
        grid = _enu_grid(pixel=3.0)
        meta = build_sidd_metadata(grid, (grid.rows, grid.cols))
        ss = meta.measurement.plane_projection.sample_spacing
        assert ss.row == pytest.approx(3.0)
        assert ss.col == pytest.approx(3.0)

    def test_geographic_sample_spacing_in_arcseconds(self):
        grid = GeographicGrid(
            REF_LAT - 0.01, REF_LAT + 0.01,
            REF_LON - 0.01, REF_LON + 0.01, 1e-4, 2e-4,
        )
        meta = build_sidd_metadata(grid, (grid.rows, grid.cols))
        ss = meta.measurement.geographic_projection.sample_spacing
        assert ss.row == pytest.approx(0.36)     # 1e-4 deg = 0.36"
        assert ss.col == pytest.approx(0.72)

    def test_plane_unit_vectors_are_orthonormal(self):
        grid = _enu_grid()
        meta = build_sidd_metadata(grid, (grid.rows, grid.cols))
        plane = meta.measurement.plane_projection.product_plane
        row = plane.row_unit_vector.to_array()
        col = plane.col_unit_vector.to_array()
        assert np.linalg.norm(row) == pytest.approx(1.0)
        assert np.linalg.norm(col) == pytest.approx(1.0)
        assert row.dot(col) == pytest.approx(0.0, abs=1e-12)


# ===================================================================
# Shape handling
# ===================================================================

class TestShapeResolution:
    """The product shape must agree with the grid."""

    def test_two_dimensional_shape(self):
        grid = _enu_grid()
        meta = build_sidd_metadata(grid, (grid.rows, grid.cols))
        assert meta.measurement.pixel_footprint.row == grid.rows

    def test_band_first_shape(self):
        grid = _enu_grid()
        meta = build_sidd_metadata(
            grid, (3, grid.rows, grid.cols), pixel_type='RGB24I',
        )
        assert meta.measurement.pixel_footprint.col == grid.cols

    def test_band_last_shape(self):
        grid = _enu_grid()
        meta = build_sidd_metadata(
            grid, (grid.rows, grid.cols, 3), pixel_type='RGB24I',
        )
        assert meta.measurement.pixel_footprint.col == grid.cols

    def test_mismatched_shape_rejected(self):
        grid = _enu_grid()
        with pytest.raises(ValidationError, match='disagrees with the grid'):
            build_sidd_metadata(grid, (10, 10))

    def test_bad_rank_rejected(self):
        grid = _enu_grid()
        with pytest.raises(ValidationError, match='2 or 3 entries'):
            build_sidd_metadata(grid, (1, 2, 3, 4))

    def test_bad_pixel_type_rejected(self):
        grid = _enu_grid()
        with pytest.raises(ValidationError, match='Unsupported pixel_type'):
            build_sidd_metadata(
                grid, (grid.rows, grid.cols), pixel_type='MONO4I',
            )


# ===================================================================
# Valid data polygon
# ===================================================================

class TestValidData:
    """Valid-data polygon projection, clipping and ordering."""

    def test_defaults_to_full_rectangle_without_source(self):
        grid = _enu_grid()
        meta = build_sidd_metadata(grid, (grid.rows, grid.cols))
        verts = meta.measurement.valid_data
        assert len(verts) == 4
        assert verts[0].row == 0.0 and verts[0].col == 0.0

    def test_projects_source_polygon(self):
        grid = _enu_grid(pixel=1.0, half=400.0)
        geo = _AnalyticGeolocation()
        meta = build_sidd_metadata(
            grid, (grid.rows, grid.cols),
            source_metadata=_sicd_metadata(with_valid_data=True),
            geolocation=geo,
        )
        verts = meta.measurement.valid_data
        # The source polygon is inset from the image edge, so the
        # projected result must be smaller than the full grid.
        rows = [v.row for v in verts]
        cols = [v.col for v in verts]
        assert len(verts) >= 3
        assert max(rows) < grid.rows - 1
        assert max(cols) < grid.cols - 1

    def test_polygon_is_clipped_to_product(self):
        grid = _enu_grid(pixel=4.0, half=100.0)
        geo = _AnalyticGeolocation()
        meta = build_sidd_metadata(
            grid, (grid.rows, grid.cols),
            source_metadata=_sicd_metadata(with_valid_data=False),
            geolocation=geo,
        )
        for v in meta.measurement.valid_data:
            assert -1e-6 <= v.row <= grid.rows - 1 + 1e-6
            assert -1e-6 <= v.col <= grid.cols - 1 + 1e-6

    def test_polygon_is_clockwise_from_min_row(self):
        grid = _enu_grid(pixel=1.0, half=400.0)
        geo = _AnalyticGeolocation()
        meta = build_sidd_metadata(
            grid, (grid.rows, grid.cols),
            source_metadata=_sicd_metadata(),
            geolocation=geo,
        )
        pts = np.array(
            [[v.row, v.col] for v in meta.measurement.valid_data],
        )
        assert pts[0, 0] == pytest.approx(pts[:, 0].min())
        x, y = pts[:, 1], pts[:, 0]
        area = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        assert area > 0

    def test_explicit_polygon_is_used(self):
        grid = _enu_grid()
        meta = build_sidd_metadata(
            grid, (grid.rows, grid.cols),
            valid_data=[(0, 0), (0, 10), (10, 10), (10, 0)],
        )
        assert len(meta.measurement.valid_data) == 4
        assert meta.measurement.valid_data[2].row == 10.0

    def test_geo_valid_data_matches_pixel_polygon(self):
        grid = _enu_grid(pixel=1.0, half=400.0)
        geo = _AnalyticGeolocation()
        meta = build_sidd_metadata(
            grid, (grid.rows, grid.cols),
            source_metadata=_sicd_metadata(),
            geolocation=geo,
        )
        assert len(meta.geo_data.valid_data) == len(
            meta.measurement.valid_data,
        )


# ===================================================================
# Section population
# ===================================================================

class TestSections:
    """Every section derivable from the source is populated."""

    @pytest.fixture
    def full_metadata(self):
        grid = _enu_grid(pixel=1.0, half=400.0)
        return grid, build_sidd_metadata(
            grid, (grid.rows, grid.cols),
            source_metadata=_sicd_metadata(),
            geolocation=_AnalyticGeolocation(),
            interpolation='bilinear',
        )

    def test_required_sections_present(self, full_metadata):
        _, meta = full_metadata
        for name in ('product_creation', 'display', 'geo_data',
                     'measurement', 'exploitation_features'):
            assert getattr(meta, name) is not None, name

    def test_arp_poly_carried_across(self, full_metadata):
        _, meta = full_metadata
        assert meta.measurement.arp_poly is not None
        assert meta.measurement.arp_flag == 'REALTIME'

    def test_collection_info_from_source(self, full_metadata):
        _, meta = full_metadata
        coll = meta.exploitation_features.collections[0]
        assert coll.sensor_name == 'TESTSAT'
        assert coll.radar_mode.mode_type == 'SPOTLIGHT'
        assert coll.collection_date_time.startswith('2024-11-11')
        assert coll.collection_duration == pytest.approx(1.67)

    def test_polarization_split(self, full_metadata):
        _, meta = full_metadata
        pol = meta.exploitation_features.collections[0].polarizations[0]
        assert pol.tx_polarization == 'H'
        assert pol.rcv_polarization == 'H'

    def test_geometry_angles_in_range(self, full_metadata):
        _, meta = full_metadata
        g = meta.exploitation_features.collections[0].geometry
        assert 0.0 <= g.azimuth < 360.0
        assert 0.0 <= g.graze <= 90.0
        assert 0.0 <= g.slope <= 180.0
        assert 0.0 <= g.doppler_cone <= 180.0 if hasattr(
            g, 'doppler_cone') else True
        assert g.squint is not None

    def test_phenomenology_present(self, full_metadata):
        _, meta = full_metadata
        ph = meta.exploitation_features.collections[0].phenomenology
        assert ph.shadow is not None and ph.layover is not None
        assert 0.0 <= ph.shadow.angle < 360.0

    def test_product_resolution_and_north(self, full_metadata):
        _, meta = full_metadata
        prod = meta.exploitation_features.products[0]
        assert prod.resolution.row > 0
        assert prod.ellipticity >= 1.0
        assert 0.0 <= prod.north < 360.0

    def test_input_roi_within_source(self, full_metadata):
        _, meta = full_metadata
        roi = meta.exploitation_features.collections[0].input_roi
        assert roi is not None
        assert roi.upper_left.row >= 0
        assert roi.size.row <= 400

    def test_product_processing_records_grid(self, full_metadata):
        _, meta = full_metadata
        params = meta.product_processing.processing_modules[0].parameters
        assert params['GridType'] == 'ENUGrid'
        assert params['Projection'] == 'PlaneProjection'
        assert params['Interpolation'] == 'bilinear'

    def test_time_coa_carried_for_constant_poly(self, full_metadata):
        _, meta = full_metadata
        coa = meta.measurement.plane_projection.time_coa_poly
        assert coa.coefs.shape == (1, 1)
        assert coa.coefs[0, 0] == pytest.approx(1.5)

    def test_dem_section_omitted_without_posting(self, full_metadata):
        _, meta = full_metadata
        assert meta.digital_elevation_data is None

    def test_dem_section_built_with_posting(self):
        grid = _enu_grid()
        geo = _AnalyticGeolocation()
        geo.elevation = object()          # any attached terrain model
        meta = build_sidd_metadata(
            grid, (grid.rows, grid.cols),
            source_metadata=_sicd_metadata(),
            geolocation=geo,
            dem_posts_per_degree=(3600.0, 3600.0),
        )
        ded = meta.digital_elevation_data
        assert ded is not None
        assert ded.geographic_coordinates.latitude_density == 3600.0
        assert ded.geopositioning.coordinate_system_type == 'GGS'

    def test_no_source_still_builds(self):
        grid = _enu_grid()
        meta = build_sidd_metadata(grid, (grid.rows, grid.cols))
        assert meta.measurement.arp_poly is None
        assert meta.exploitation_features.collections[0].sensor_name \
            == 'UNKNOWN'


# ===================================================================
# Write and read round trip
# ===================================================================

class TestRoundTrip:
    """Write a SIDD and read the geolocation and pixels back."""

    def _write(self, tmp_path, grid, data, pixel_type, **kwargs):
        from grdl.IO.sar.sidd_writer import SIDDWriter
        meta = build_sidd_metadata(
            grid, data.shape, pixel_type=pixel_type,
            source_metadata=_sicd_metadata(),
            geolocation=_AnalyticGeolocation(),
            **kwargs,
        )
        path = tmp_path / f'{pixel_type}.nitf'
        SIDDWriter(str(path), metadata=meta).write(data)
        return path

    def test_mono8_round_trip(self, tmp_path):
        from grdl.IO.sar import SIDDReader
        grid = _enu_grid(pixel=4.0, half=200.0)
        data = np.random.randint(
            0, 256, (grid.rows, grid.cols),
        ).astype(np.uint8)
        path = self._write(tmp_path, grid, data, 'MONO8I')

        reader = SIDDReader(str(path))
        try:
            assert reader.metadata.display.pixel_type == 'MONO8I'
            assert np.array_equal(reader.read_full(), data)
        finally:
            reader.close()

    def test_mono16_round_trip(self, tmp_path):
        from grdl.IO.sar import SIDDReader
        grid = _enu_grid(pixel=8.0, half=200.0)
        data = np.random.randint(
            0, 65536, (grid.rows, grid.cols),
        ).astype(np.uint16)
        path = self._write(tmp_path, grid, data, 'MONO16I')

        reader = SIDDReader(str(path))
        try:
            assert reader.metadata.display.pixel_type == 'MONO16I'
            assert np.array_equal(reader.read_full(), data)
        finally:
            reader.close()

    def test_rgb_round_trip_band_first(self, tmp_path):
        from grdl.IO.sar import SIDDReader
        grid = _enu_grid(pixel=8.0, half=200.0)
        data = np.random.randint(
            0, 256, (3, grid.rows, grid.cols),
        ).astype(np.uint8)
        path = self._write(tmp_path, grid, data, 'RGB24I')

        reader = SIDDReader(str(path))
        try:
            assert reader.metadata.display.num_bands == 3
            back = reader.read_full()
            assert back.shape == (grid.rows, grid.cols, 3)
            assert np.array_equal(back, np.moveaxis(data, 0, -1))
        finally:
            reader.close()

    def test_rgb_round_trip_band_last(self, tmp_path):
        from grdl.IO.sar import SIDDReader
        grid = _enu_grid(pixel=8.0, half=200.0)
        data = np.random.randint(
            0, 256, (grid.rows, grid.cols, 3),
        ).astype(np.uint8)
        path = self._write(tmp_path, grid, data, 'RGB24I')

        reader = SIDDReader(str(path))
        try:
            assert np.array_equal(reader.read_full(), data)
        finally:
            reader.close()

    @pytest.mark.parametrize('grid_factory,expected', [
        (lambda: _enu_grid(pixel=4.0, half=200.0), 'PlaneProjection'),
        (
            lambda: RotatedENUGrid(
                ref_lat=REF_LAT, ref_lon=REF_LON, ref_alt=0.0, angle=0.7,
                min_u=-200.0, max_u=200.0, min_v=-200.0, max_v=200.0,
                pixel_size=4.0,
            ),
            'PlaneProjection',
        ),
        (
            lambda: GeographicGrid(
                REF_LAT - 0.01, REF_LAT + 0.01,
                REF_LON - 0.01, REF_LON + 0.01, 1e-4, 1e-4,
            ),
            'GeographicProjection',
        ),
    ])
    def test_geolocation_round_trip(self, tmp_path, grid_factory, expected):
        """The written geolocation reproduces the grid to under a metre."""
        from grdl.IO.sar import SIDDReader
        from grdl.geolocation.sar.sidd import SIDDGeolocation

        grid = grid_factory()
        data = np.zeros((grid.rows, grid.cols), np.uint8)
        path = self._write(tmp_path, grid, data, 'MONO8I')

        reader = SIDDReader(str(path))
        try:
            assert reader.metadata.measurement.projection_type == expected
            geo = SIDDGeolocation.from_reader(reader)
            assert _grid_error_metres(grid, geo) < 1.0
        finally:
            reader.close()

    def test_valid_data_round_trips(self, tmp_path):
        from grdl.IO.sar import SIDDReader
        grid = _enu_grid(pixel=2.0, half=300.0)
        data = np.zeros((grid.rows, grid.cols), np.uint8)
        path = self._write(tmp_path, grid, data, 'MONO8I')

        reader = SIDDReader(str(path))
        try:
            verts = reader.metadata.measurement.valid_data
            assert verts is not None and len(verts) >= 3
        finally:
            reader.close()

    def test_optional_sections_round_trip(self, tmp_path):
        from grdl.IO.sar import SIDDReader
        grid = _enu_grid(pixel=4.0, half=200.0)
        data = np.zeros((grid.rows, grid.cols), np.uint8)
        path = self._write(tmp_path, grid, data, 'MONO8I')

        reader = SIDDReader(str(path))
        try:
            meta = reader.metadata
            assert meta.product_processing is not None
            params = meta.product_processing.processing_modules[0].parameters
            assert params['GridType'] == 'ENUGrid'
            coll = meta.exploitation_features.collections[0]
            assert coll.sensor_name == 'TESTSAT'
            assert coll.input_roi is not None
        finally:
            reader.close()


# ===================================================================
# Writer validation
# ===================================================================

class TestWriterValidation:
    """The writer rejects arrays that disagree with the metadata."""

    def _writer(self, tmp_path, pixel_type='MONO8I'):
        from grdl.IO.sar.sidd_writer import SIDDWriter
        grid = _enu_grid(pixel=20.0, half=200.0)
        meta = build_sidd_metadata(
            grid, (grid.rows, grid.cols), pixel_type=pixel_type,
            source_metadata=_sicd_metadata(),
            geolocation=_AnalyticGeolocation(),
        )
        return grid, SIDDWriter(str(tmp_path / 'out.nitf'), metadata=meta)

    def test_rejects_wrong_shape(self, tmp_path):
        grid, writer = self._writer(tmp_path)
        with pytest.raises(ValidationError, match='pixel footprint'):
            writer.write(np.zeros((grid.rows + 1, grid.cols), np.uint8))

    def test_rejects_wrong_dtype(self, tmp_path):
        grid, writer = self._writer(tmp_path)
        with pytest.raises(ValidationError, match='requires dtype uint8'):
            writer.write(np.zeros((grid.rows, grid.cols), np.uint16))

    def test_rejects_two_dimensional_for_rgb(self, tmp_path):
        grid, writer = self._writer(tmp_path, 'RGB24I')
        with pytest.raises(ValidationError, match='3 bands'):
            writer.write(np.zeros((grid.rows, grid.cols), np.uint8))

    def test_chip_write_unsupported(self, tmp_path):
        _, writer = self._writer(tmp_path)
        with pytest.raises(NotImplementedError, match='chip-level'):
            writer.write_chip(np.zeros((2, 2), np.uint8), 0, 0)
