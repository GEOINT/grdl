# -*- coding: utf-8 -*-
"""
Tests for the collect-aligned rotated ENU output grid.

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
2026-08-31

Modified
--------
2026-08-31
"""

# Third-party
import numpy as np
import pytest

# GRDL internal
from grdl.image_processing.ortho import OutputGridProtocol, RotatedENUGrid


@pytest.fixture
def grid() -> RotatedENUGrid:
    """A modestly rotated 200 x 300 grid at 5 m spacing."""
    return RotatedENUGrid(
        ref_lat=19.4, ref_lon=-155.3, ref_alt=0.0,
        angle=np.radians(30.0),
        min_u=-750.0, max_u=750.0,
        min_v=-500.0, max_v=500.0,
        pixel_size=5.0,
    )


def test_satisfies_output_grid_protocol(grid: RotatedENUGrid) -> None:
    """The grid is usable anywhere Orthorectifier accepts a grid."""
    assert isinstance(grid, OutputGridProtocol)


def test_dimensions_follow_bounds_and_spacing(grid: RotatedENUGrid) -> None:
    """rows/cols derive from the rotated-axis extents."""
    assert grid.rows == 200
    assert grid.cols == 300


def test_pixel_roundtrip_is_exact(grid: RotatedENUGrid) -> None:
    """image_to_latlon and latlon_to_image invert each other."""
    rows = np.array([0.5, 37.5, 199.5])
    cols = np.array([0.5, 111.5, 299.5])
    lats, lons = grid.image_to_latlon(rows, cols)
    back_rows, back_cols = grid.latlon_to_image(lats, lons)
    assert np.allclose(back_rows, rows, atol=1e-4)
    assert np.allclose(back_cols, cols, atol=1e-4)


def test_scalar_and_array_forms_agree(grid: RotatedENUGrid) -> None:
    """Scalar calls match the array form element-wise."""
    lat, lon = grid.image_to_latlon(10.5, 20.5)
    lats, lons = grid.image_to_latlon(np.array([10.5]), np.array([20.5]))
    assert lat == pytest.approx(float(lats[0]))
    assert lon == pytest.approx(float(lons[0]))


def test_chunked_transform_matches_whole_array(
    grid: RotatedENUGrid, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunking the coordinate transform is bitwise identical.

    The chunk size bounds peak memory during mapping, so it must not
    change results.
    """
    import grdl.image_processing.ortho.rotated_enu_grid as module

    rows = np.linspace(0.5, 199.5, 5000)
    cols = np.linspace(0.5, 299.5, 5000)
    whole_lat, whole_lon = grid.image_to_latlon(rows, cols)

    monkeypatch.setattr(module, '_CHUNK', 97)
    chunked_lat, chunked_lon = grid.image_to_latlon(rows, cols)

    assert np.array_equal(whole_lat, chunked_lat)
    assert np.array_equal(whole_lon, chunked_lon)


def test_sub_grid_is_a_consistent_window(grid: RotatedENUGrid) -> None:
    """A sub-grid maps its pixels to the same ground as the parent."""
    sub = grid.sub_grid(40, 60, 140, 260)
    assert (sub.rows, sub.cols) == (100, 200)

    parent_lat, parent_lon = grid.image_to_latlon(40.5 + 7, 60.5 + 11)
    sub_lat, sub_lon = sub.image_to_latlon(7.5, 11.5)
    assert sub_lat == pytest.approx(parent_lat, abs=1e-9)
    assert sub_lon == pytest.approx(parent_lon, abs=1e-9)


def test_north_vector_points_north(grid: RotatedENUGrid) -> None:
    """north_vector agrees with where north actually is on the raster."""
    d_col, d_row = grid.north_vector()

    center_lat, center_lon = grid.image_to_latlon(100.5, 150.5)
    north_row, north_col = grid.latlon_to_image(
        center_lat + 0.001, center_lon,
    )
    measured = np.array([north_col - 150.5, north_row - 100.5])
    measured /= np.linalg.norm(measured)
    assert np.allclose(measured, [d_col, d_row], atol=1e-3)


def test_rejects_inverted_bounds() -> None:
    """Bad extents fail at construction, not at first use."""
    with pytest.raises(ValueError, match='max_u'):
        RotatedENUGrid(
            ref_lat=0.0, ref_lon=0.0, ref_alt=0.0, angle=0.0,
            min_u=10.0, max_u=0.0, min_v=0.0, max_v=10.0,
            pixel_size=1.0,
        )


def test_rejects_non_positive_pixel_size() -> None:
    """A zero or negative spacing fails at construction."""
    with pytest.raises(ValueError, match='pixel_size'):
        RotatedENUGrid(
            ref_lat=0.0, ref_lon=0.0, ref_alt=0.0, angle=0.0,
            min_u=0.0, max_u=10.0, min_v=0.0, max_v=10.0,
            pixel_size=0.0,
        )


def test_fit_to_polygon_encloses_its_input() -> None:
    """The fitted grid covers every vertex it was built from."""
    lats = np.array([19.400, 19.410, 19.420, 19.410])
    lons = np.array([-155.300, -155.280, -155.300, -155.320])
    grid = RotatedENUGrid.fit_to_polygon(lats, lons, pixel_size=5.0)

    rows, cols = grid.latlon_to_image(lats, lons)
    assert np.all(rows >= -1.0) and np.all(rows <= grid.rows + 1.0)
    assert np.all(cols >= -1.0) and np.all(cols <= grid.cols + 1.0)
