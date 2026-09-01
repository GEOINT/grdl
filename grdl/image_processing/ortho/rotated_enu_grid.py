# -*- coding: utf-8 -*-
"""
Collect-aligned ENU output grid -- ground projection without the fill.

A north-up output grid frames a rotated collect inside an axis-aligned
rectangle, so roughly half the raster is nodata and the product is
correspondingly larger and slower.  Orthorectification is a *ground
projection*, though: removing slant-range distortion says nothing about
which way the output raster's rows must point.  Rotating the grid to the
collect keeps the projection identical and drops the fill to near zero.

``RotatedENUGrid`` is a local East-North-Up grid in meters with an
in-plane rotation, satisfying ``OutputGridProtocol``.  Build it with
``fit_to_collect`` to align the raster with the source image's own
geometry, which is almost always what you want: the ortho then reads
like the collect it came from, and a bounding rectangle in that frame is
close to tight.  ``fit_to_polygon`` fits the minimum-area rectangle
instead, which packs marginally better but is free to land 90 or 180
degrees away from the collect, so the image can come out rotated or
flipped relative to the source.

Dependencies
------------
scipy

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

# Standard library
from typing import Tuple, Union

# Third-party
import numpy as np
from scipy.spatial import ConvexHull

# GRDL internal
from grdl.geolocation.coordinates import enu_to_geodetic, geodetic_to_enu

# Points converted per chunk in the coordinate transforms.  The whole-array
# form costs ~120 bytes per point in simultaneous temporaries, which is the
# dominant per-tile term in the orthorectifier's mapping stage.
#
# Kept small because Orthorectifier._compute_mapping_parallel runs one strip
# per worker thread and every strip is in flight at once, so this budget is
# multiplied by the thread count (cpu_count - 1).  Measured on a 2048x2048
# tile with 13 workers: 2M points cost 2.54 GB of transient, 250k cost
# nothing measurable, for byte-identical output and ~0.1 s more wall time.
_CHUNK = 250_000


def ground_axes(geolocation, step: float = 200.0) -> Tuple[
        np.ndarray, np.ndarray]:
    """Ground directions of the source image's row and column axes.

    Parameters
    ----------
    geolocation : Geolocation
        Source image geolocation.
    step : float, default=200.0
        Baseline in pixels over which the directions are measured.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Unit ``(east, north)`` vectors for increasing row and
        increasing column.
    """
    rows, cols = geolocation.shape
    center_row, center_col = rows / 2.0, cols / 2.0
    probes = geolocation.image_to_latlon(np.array([
        [center_row, center_col],
        [center_row + step, center_col],
        [center_row, center_col + step],
    ], dtype=np.float64))

    ref = np.array([probes[0, 0], probes[0, 1], 0.0])
    enu = geodetic_to_enu(
        np.column_stack([probes[:, 0], probes[:, 1], np.zeros(3)]), ref,
    )[:, :2]

    row_dir = enu[1] - enu[0]
    col_dir = enu[2] - enu[0]
    return (row_dir / np.linalg.norm(row_dir),
            col_dir / np.linalg.norm(col_dir))


def orientation_preserved(geolocation) -> bool:
    """Whether slant-to-ground keeps the displayed image orientation.

    False means the collect is mirrored relative to the ground, so no
    rotation can make an ortho match the slant image's handedness --
    matching it would require mirroring the geography.

    Parameters
    ----------
    geolocation : Geolocation
        Source image geolocation.

    Returns
    -------
    bool
    """
    row_dir, col_dir = ground_axes(geolocation)
    cross = col_dir[0] * row_dir[1] - col_dir[1] * row_dir[0]
    return bool(-cross > 0.0)


class RotatedENUGrid:
    """Local ENU output grid rotated in the ground plane.

    Identical to ``ENUGrid`` except that the raster axes are rotated by
    ``angle`` within the local tangent plane.  Column increases along
    ``(cos angle, sin angle)`` in ENU; row increases opposite the
    perpendicular, matching the usual image convention of row 0 at the
    top.

    Attributes
    ----------
    ref_lat, ref_lon, ref_alt : float
        Tangent-plane reference point (degrees, degrees, meters HAE).
    angle : float
        Rotation of the column axis from East, radians, CCW positive.
    min_u, max_u, min_v, max_v : float
        Bounds along the rotated axes, in meters.
    pixel_size : float
        Ground sample distance, meters, equal on both axes.
    rows, cols : int
        Raster dimensions.
    """

    def __init__(
        self,
        ref_lat: float,
        ref_lon: float,
        ref_alt: float,
        angle: float,
        min_u: float,
        max_u: float,
        min_v: float,
        max_v: float,
        pixel_size: float,
    ) -> None:
        """Initialize the rotated grid.

        Parameters
        ----------
        ref_lat, ref_lon, ref_alt : float
            Tangent-plane reference point.
        angle : float
            Column-axis rotation from East, radians.
        min_u, max_u : float
            Bounds along the column axis, meters.
        min_v, max_v : float
            Bounds along the row axis, meters.
        pixel_size : float
            Ground sample distance in meters.

        Raises
        ------
        ValueError
            If bounds are inverted or ``pixel_size`` is not positive.
        """
        if max_u <= min_u:
            raise ValueError(f"max_u ({max_u}) must exceed min_u ({min_u})")
        if max_v <= min_v:
            raise ValueError(f"max_v ({max_v}) must exceed min_v ({min_v})")
        if pixel_size <= 0:
            raise ValueError(
                f"pixel_size must be positive, got {pixel_size}")

        self.ref_lat = float(ref_lat)
        self.ref_lon = float(ref_lon)
        self.ref_alt = float(ref_alt)
        self.angle = float(angle)
        self.min_u = float(min_u)
        self.max_u = float(max_u)
        self.min_v = float(min_v)
        self.max_v = float(max_v)
        self.pixel_size = float(pixel_size)

        self.rows = int(np.ceil((max_v - min_v) / pixel_size))
        self.cols = int(np.ceil((max_u - min_u) / pixel_size))

    @property
    def _ref(self) -> np.ndarray:
        """Reference point as ``[lat, lon, alt]``."""
        return np.array([self.ref_lat, self.ref_lon, self.ref_alt])

    @classmethod
    def _from_angle(
        cls,
        angle: float,
        lats: np.ndarray,
        lons: np.ndarray,
        pixel_size: float,
        ref_alt: float,
        margin_m: float,
    ) -> 'RotatedENUGrid':
        """Build a grid at a fixed angle, bounded by a ground polygon.

        Parameters
        ----------
        angle : float
            Column-axis rotation from East, radians.
        lats, lons : np.ndarray
            Polygon vertices in degrees.
        pixel_size : float
            Ground sample distance, meters.
        ref_alt : float
            Tangent-plane altitude, meters HAE.
        margin_m : float
            Margin on all sides, meters.

        Returns
        -------
        RotatedENUGrid
        """
        ref_lat = float(np.mean(lats))
        ref_lon = float(np.mean(lons))
        enu = geodetic_to_enu(
            np.column_stack([lats, lons, np.full(lats.size, ref_alt)]),
            np.array([ref_lat, ref_lon, ref_alt]),
        )
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        u = enu[:, 0] * cos_a + enu[:, 1] * sin_a
        v = -enu[:, 0] * sin_a + enu[:, 1] * cos_a

        return cls(
            ref_lat=ref_lat, ref_lon=ref_lon, ref_alt=ref_alt, angle=angle,
            min_u=float(u.min()) - margin_m,
            max_u=float(u.max()) + margin_m,
            min_v=float(v.min()) - margin_m,
            max_v=float(v.max()) + margin_m,
            pixel_size=pixel_size,
        )

    @classmethod
    def fit_to_collect(
        cls,
        geolocation,
        lats: np.ndarray,
        lons: np.ndarray,
        pixel_size: float,
        ref_alt: float = 0.0,
        margin_m: float = 0.0,
    ) -> 'RotatedENUGrid':
        """Build a grid aligned to the source image's own geometry.

        The grid's column axis is set to the ground direction of the
        source image's increasing-column (azimuth) axis, so the ortho
        comes out in the same orientation as the collect rather than
        rotated or flipped relative to it.

        Prefer this over ``fit_to_polygon``: the minimum-area rectangle
        is free to pick any of four orientations and routinely lands 90
        or 180 degrees away from the collect, which reads as a flipped
        image.  The extra fill this costs is typically well under 1%.

        The slant-to-ground map preserves display orientation whenever
        the projected row and column axes keep their handedness, which
        is the normal case; ``orientation_preserved`` reports it so a
        caller can warn instead of silently mirroring.

        Parameters
        ----------
        geolocation : Geolocation
            Source image geolocation.
        lats, lons : np.ndarray
            Ground polygon to bound the grid, degrees.
        pixel_size : float
            Ground sample distance, meters.
        ref_alt : float, default=0.0
            Tangent-plane altitude, meters HAE.
        margin_m : float, default=0.0
            Margin on all sides, meters.

        Returns
        -------
        RotatedENUGrid
        """
        _, col_dir = ground_axes(geolocation)
        angle = float(np.arctan2(col_dir[1], col_dir[0]))
        return cls._from_angle(
            angle, lats, lons, pixel_size, ref_alt, margin_m,
        )

    @classmethod
    def fit_to_polygon(
        cls,
        lats: np.ndarray,
        lons: np.ndarray,
        pixel_size: float,
        ref_alt: float = 0.0,
        margin_m: float = 0.0,
    ) -> 'RotatedENUGrid':
        """Build the smallest rotated grid enclosing a ground polygon.

        The rotation is chosen by minimum-area bounding rectangle over
        the polygon's convex hull, so a collect that is rectangular on
        the ground fills the raster with almost no fill pixels.

        Parameters
        ----------
        lats, lons : np.ndarray
            Polygon vertices in degrees, shape ``(N,)``.
        pixel_size : float
            Ground sample distance in meters.
        ref_alt : float, default=0.0
            Tangent-plane altitude, meters HAE.
        margin_m : float, default=0.0
            Extra margin on all four sides, meters.

        Returns
        -------
        RotatedENUGrid
        """
        ref_lat = float(np.mean(lats))
        ref_lon = float(np.mean(lons))
        ref = np.array([ref_lat, ref_lon, ref_alt])

        enu = geodetic_to_enu(
            np.column_stack([lats, lons, np.full(lats.size, ref_alt)]), ref,
        )
        pts = enu[:, :2]

        hull = ConvexHull(pts)
        verts = pts[hull.vertices]

        best = None
        for i in range(verts.shape[0]):
            edge = verts[(i + 1) % verts.shape[0]] - verts[i]
            angle = float(np.arctan2(edge[1], edge[0]))
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            u = pts[:, 0] * cos_a + pts[:, 1] * sin_a
            v = -pts[:, 0] * sin_a + pts[:, 1] * cos_a
            area = (u.max() - u.min()) * (v.max() - v.min())
            if best is None or area < best[0]:
                best = (area, angle, u.min(), u.max(), v.min(), v.max())

        # A rectangle has four equivalent orientations; pick the one
        # that leaves north closest to up.  North in raster coords is
        # (sin a, -cos a), so the representative wanted is in
        # (-pi/4, pi/4].  Note this ignores the collect's own geometry
        # and so can land 90 or 180 degrees from it -- see
        # ``fit_to_collect``.
        angle = (best[1] + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0
        return cls._from_angle(
            angle, lats, lons, pixel_size, ref_alt, margin_m,
        )

    def image_to_latlon(
        self,
        row: Union[float, np.ndarray],
        col: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Convert grid pixel coordinates to lat/lon.

        Parameters
        ----------
        row, col : float or np.ndarray
            Pixel coordinates (0-based).

        Returns
        -------
        Tuple[float or np.ndarray, float or np.ndarray]
            ``(latitude, longitude)`` in degrees.
        """
        scalar = np.ndim(row) == 0 and np.ndim(col) == 0
        rows = np.atleast_1d(np.asarray(row, dtype=np.float64)).ravel()
        cols = np.atleast_1d(np.asarray(col, dtype=np.float64)).ravel()
        n = max(rows.size, cols.size)

        cos_a, sin_a = np.cos(self.angle), np.sin(self.angle)
        lats = np.empty(n, dtype=np.float64)
        lons = np.empty(n, dtype=np.float64)

        # Chunked so the peak is O(_CHUNK), not O(n).  enu_to_geodetic is
        # strictly elementwise per row, so this is bitwise identical to
        # converting everything at once -- but the whole-array form holds
        # u, v, east, north, the column_stack and the (n, 3) result all
        # at once, ~120 bytes per point, which dominates the per-tile
        # ortho footprint.
        for start in range(0, n, _CHUNK):
            stop = min(start + _CHUNK, n)
            sl = slice(start, stop)
            u = self.min_u + cols[sl] * self.pixel_size
            v = self.max_v - rows[sl] * self.pixel_size

            enu = np.empty((stop - start, 3), dtype=np.float64)
            enu[:, 0] = u * cos_a - v * sin_a
            enu[:, 1] = u * sin_a + v * cos_a
            enu[:, 2] = 0.0

            geo = enu_to_geodetic(enu, self._ref)
            lats[sl] = geo[:, 0]
            lons[sl] = geo[:, 1]

        if scalar:
            return float(lats[0]), float(lons[0])
        return lats, lons

    def latlon_to_image(
        self,
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Convert lat/lon to grid pixel coordinates.

        Parameters
        ----------
        lat, lon : float or np.ndarray
            Geographic coordinates in degrees.

        Returns
        -------
        Tuple[float or np.ndarray, float or np.ndarray]
            ``(row, col)`` pixel coordinates.
        """
        scalar = np.ndim(lat) == 0 and np.ndim(lon) == 0
        lat_a = np.atleast_1d(np.asarray(lat, dtype=np.float64)).ravel()
        lon_a = np.atleast_1d(np.asarray(lon, dtype=np.float64)).ravel()
        n = max(lat_a.size, lon_a.size)

        cos_a, sin_a = np.cos(self.angle), np.sin(self.angle)
        row = np.empty(n, dtype=np.float64)
        col = np.empty(n, dtype=np.float64)

        # Chunked for the same reason as image_to_latlon; phase 2 calls
        # this once per output tile through GridGeolocation.
        for start in range(0, n, _CHUNK):
            stop = min(start + _CHUNK, n)
            sl = slice(start, stop)

            pts = np.empty((stop - start, 3), dtype=np.float64)
            pts[:, 0] = lat_a[sl]
            pts[:, 1] = lon_a[sl]
            pts[:, 2] = self.ref_alt

            enu = geodetic_to_enu(pts, self._ref)
            east, north = enu[:, 0], enu[:, 1]

            row[sl] = (self.max_v - (-east * sin_a + north * cos_a)) \
                / self.pixel_size
            col[sl] = ((east * cos_a + north * sin_a) - self.min_u) \
                / self.pixel_size

        if scalar:
            return float(row[0]), float(col[0])
        return row, col

    def sub_grid(
        self,
        row_start: int,
        col_start: int,
        row_end: int,
        col_end: int,
    ) -> 'RotatedENUGrid':
        """Extract a sub-grid covering a rectangular tile.

        Parameters
        ----------
        row_start, col_start : int
            Top-left corner of the tile (inclusive).
        row_end, col_end : int
            Bottom-right corner of the tile (exclusive).

        Returns
        -------
        RotatedENUGrid
        """
        from grdl.image_processing.ortho import validate_sub_grid_indices

        validate_sub_grid_indices(
            self.rows, self.cols, row_start, col_start, row_end, col_end,
        )

        sub = RotatedENUGrid(
            ref_lat=self.ref_lat, ref_lon=self.ref_lon,
            ref_alt=self.ref_alt, angle=self.angle,
            min_u=self.min_u + col_start * self.pixel_size,
            max_u=self.min_u + col_end * self.pixel_size,
            min_v=self.max_v - row_end * self.pixel_size,
            max_v=self.max_v - row_start * self.pixel_size,
            pixel_size=self.pixel_size,
        )
        sub.rows = row_end - row_start
        sub.cols = col_end - col_start
        return sub

    def north_vector(self) -> Tuple[float, float]:
        """Direction of true north in raster ``(d_col, d_row)``.

        North is no longer up in a rotated grid, so callers annotating
        the raster need this to draw an orientation arrow.

        Returns
        -------
        Tuple[float, float]
            Unit vector ``(d_col, d_row)`` pointing to true north.
        """
        return float(np.sin(self.angle)), float(-np.cos(self.angle))

    def __repr__(self) -> str:
        return (
            f'RotatedENUGrid(ref=({self.ref_lat:.4f}, {self.ref_lon:.4f}), '
            f'angle={np.degrees(self.angle):.1f} deg, '
            f'{self.rows}x{self.cols} @ {self.pixel_size:.2f} m)'
        )


__all__ = [
    'RotatedENUGrid',
    'ground_axes',
    'orientation_preserved',
]
