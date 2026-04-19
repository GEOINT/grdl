# -*- coding: utf-8 -*-
"""
ENU Output Grid - Regular output grid in East-North-Up meters.

Provides ``ENUGrid``, an output grid specification for orthorectification
in local ENU (East-North-Up) coordinates centered on a WGS-84 reference
point.  Compatible with ``Orthorectifier`` as a drop-in alternative to
``GeographicGrid``: both provide ``rows``, ``cols``, ``image_to_latlon()``,
``latlon_to_image()``, and ``sub_grid()``.

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
2026-03-08

Modified
--------
2026-03-22
"""

import logging
from typing import Tuple, Union, TYPE_CHECKING

import numpy as np

from grdl.geolocation.coordinates import (
    geodetic_to_enu,
    enu_to_geodetic,
)
from grdl.image_processing.ortho.ortho import validate_sub_grid_indices

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from grdl.geolocation.base import Geolocation


class ENUGrid:
    """Output grid specification in ENU meters.

    Defines a regular grid in local East-North-Up coordinates centered
    on a WGS-84 reference point.  Grid axes are aligned with geographic
    East and North.  Pixel sizes are in meters.

    Row 0 is the north edge (max_north), row increases southward.
    Column 0 is the west edge (min_east), column increases eastward.
    This matches the ``GeographicGrid`` convention.

    Attributes
    ----------
    ref_lat : float
        Reference point latitude (degrees).
    ref_lon : float
        Reference point longitude (degrees).
    ref_alt : float
        Reference point altitude (meters HAE).
    min_east : float
        Western bound in meters.
    max_east : float
        Eastern bound in meters.
    min_north : float
        Southern bound in meters.
    max_north : float
        Northern bound in meters.
    pixel_size_east : float
        East spacing per pixel (meters).
    pixel_size_north : float
        North spacing per pixel (meters).
    rows : int
        Number of output rows.
    cols : int
        Number of output columns.

    Examples
    --------
    Create a 2km x 2km grid at 1m resolution centered on a point:

    >>> grid = ENUGrid(
    ...     ref_lat=36.0, ref_lon=-75.5, ref_alt=0.0,
    ...     min_east=-1000, max_east=1000,
    ...     min_north=-1000, max_north=1000,
    ...     pixel_size_east=1.0, pixel_size_north=1.0,
    ... )
    >>> print(f"{grid.rows} x {grid.cols}")
    2000 x 2000
    """

    def __init__(
        self,
        ref_lat: float,
        ref_lon: float,
        ref_alt: float,
        min_east: float,
        max_east: float,
        min_north: float,
        max_north: float,
        pixel_size_east: float,
        pixel_size_north: float,
    ) -> None:
        """Initialize ENU output grid.

        Parameters
        ----------
        ref_lat : float
            Reference point latitude (degrees).
        ref_lon : float
            Reference point longitude (degrees).
        ref_alt : float
            Reference point altitude (meters HAE).
        min_east : float
            Western bound in meters.
        max_east : float
            Eastern bound in meters.
        min_north : float
            Southern bound in meters.
        max_north : float
            Northern bound in meters.
        pixel_size_east : float
            East spacing per pixel (meters). Must be positive.
        pixel_size_north : float
            North spacing per pixel (meters). Must be positive.

        Raises
        ------
        ValueError
            If bounds are invalid or pixel sizes are not positive.
        """
        if max_east <= min_east:
            raise ValueError(
                f"max_east ({max_east}) must be greater than "
                f"min_east ({min_east})"
            )
        if max_north <= min_north:
            raise ValueError(
                f"max_north ({max_north}) must be greater than "
                f"min_north ({min_north})"
            )
        if pixel_size_east <= 0:
            raise ValueError(
                f"pixel_size_east must be positive, got {pixel_size_east}"
            )
        if pixel_size_north <= 0:
            raise ValueError(
                f"pixel_size_north must be positive, got {pixel_size_north}"
            )

        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.ref_alt = ref_alt
        self.min_east = min_east
        self.max_east = max_east
        self.min_north = min_north
        self.max_north = max_north
        self.pixel_size_east = pixel_size_east
        self.pixel_size_north = pixel_size_north

        self.rows = int(np.ceil(
            (max_north - min_north) / pixel_size_north
        ))
        self.cols = int(np.ceil(
            (max_east - min_east) / pixel_size_east
        ))

    @classmethod
    def from_geolocation(
        cls,
        geolocation: 'Geolocation',
        pixel_size_m: float,
        ref_lat: float = None,
        ref_lon: float = None,
        ref_alt: float = 0.0,
        margin_m: float = 0.0,
    ) -> 'ENUGrid':
        """Create an ENU grid covering a geolocation's footprint.

        If ``ref_lat`` / ``ref_lon`` are not provided, the image center
        is used as the reference point.

        Parameters
        ----------
        geolocation : Geolocation
            Source geolocation for computing bounds.
        pixel_size_m : float
            Pixel spacing in meters (used for both east and north).
        ref_lat : float, optional
            Reference latitude. Defaults to image center.
        ref_lon : float, optional
            Reference longitude. Defaults to image center.
        ref_alt : float
            Reference altitude in meters HAE.
        margin_m : float
            Margin to add around bounds in meters.

        Returns
        -------
        ENUGrid
            Grid covering the geolocation footprint in ENU meters.
        """
        rows, cols = geolocation.shape[:2]

        # Auto-compute reference point from image center
        if ref_lat is None or ref_lon is None:
            _center = geolocation.image_to_latlon(rows / 2.0, cols / 2.0)
            if ref_lat is None:
                ref_lat = float(_center[0])
            if ref_lon is None:
                ref_lon = float(_center[1])
            logger.debug(
                "Auto-selected reference point: lat=%.6f, lon=%.6f",
                ref_lat, ref_lon,
            )

        # Get footprint corners and convert to ENU
        min_lon, min_lat, max_lon, max_lat = geolocation.get_bounds()

        corner_lats = np.array([min_lat, min_lat, max_lat, max_lat])
        corner_lons = np.array([min_lon, max_lon, min_lon, max_lon])
        corner_heights = np.full(4, ref_alt)

        enu = geodetic_to_enu(
            np.column_stack([corner_lats, corner_lons, corner_heights]),
            np.array([ref_lat, ref_lon, ref_alt]),
        )
        east, north = enu[:, 0], enu[:, 1]

        return cls(
            ref_lat=ref_lat,
            ref_lon=ref_lon,
            ref_alt=ref_alt,
            min_east=float(east.min()) - margin_m,
            max_east=float(east.max()) + margin_m,
            min_north=float(north.min()) - margin_m,
            max_north=float(north.max()) + margin_m,
            pixel_size_east=pixel_size_m,
            pixel_size_north=pixel_size_m,
        )

    def image_to_latlon(
        self,
        row: Union[float, np.ndarray],
        col: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Convert grid pixel coordinates to lat/lon.

        Parameters
        ----------
        row : float or np.ndarray
            Row coordinate(s) (0-based).
        col : float or np.ndarray
            Column coordinate(s) (0-based).

        Returns
        -------
        Tuple[float or np.ndarray, float or np.ndarray]
            (latitude, longitude) in degrees.
        """
        north = self.max_north - np.asarray(row) * self.pixel_size_north
        east = self.min_east + np.asarray(col) * self.pixel_size_east
        up = np.zeros_like(east)

        scalar = np.ndim(east) == 0
        east_flat = np.atleast_1d(east)
        north_flat = np.atleast_1d(north)
        up_flat = np.atleast_1d(up)

        geo = enu_to_geodetic(
            np.column_stack([east_flat, north_flat, up_flat]),
            np.array([self.ref_lat, self.ref_lon, self.ref_alt]),
        )
        lats, lons = geo[:, 0], geo[:, 1]

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
        lat : float or np.ndarray
            Latitude(s) in degrees.
        lon : float or np.ndarray
            Longitude(s) in degrees.

        Returns
        -------
        Tuple[float or np.ndarray, float or np.ndarray]
            (row, col) pixel coordinates.
        """
        lat_arr = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        lon_arr = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        h_arr = np.full_like(lat_arr, self.ref_alt)

        enu = geodetic_to_enu(
            np.column_stack([lat_arr, lon_arr, h_arr]),
            np.array([self.ref_lat, self.ref_lon, self.ref_alt]),
        )
        east, north = enu[:, 0], enu[:, 1]

        row = (self.max_north - north) / self.pixel_size_north
        col = (east - self.min_east) / self.pixel_size_east

        scalar = np.ndim(lat) == 0 and np.ndim(lon) == 0
        if scalar:
            return float(row[0]), float(col[0])
        return row, col

    def sub_grid(
        self,
        row_start: int,
        col_start: int,
        row_end: int,
        col_end: int,
    ) -> 'ENUGrid':
        """Extract a sub-grid covering a rectangular tile.

        Parameters
        ----------
        row_start : int
            First row (inclusive).
        col_start : int
            First column (inclusive).
        row_end : int
            Last row (exclusive).
        col_end : int
            Last column (exclusive).

        Returns
        -------
        ENUGrid
            Sub-grid with ENU bounds matching the tile region.

        Raises
        ------
        ValueError
            If indices are out of range or produce an empty region.
        """
        validate_sub_grid_indices(
            self.rows, self.cols, row_start, col_start, row_end, col_end,
        )

        tile_rows = row_end - row_start
        tile_cols = col_end - col_start

        tile_max_north = (
            self.max_north - row_start * self.pixel_size_north
        )
        tile_min_north = tile_max_north - tile_rows * self.pixel_size_north
        tile_min_east = self.min_east + col_start * self.pixel_size_east
        tile_max_east = tile_min_east + tile_cols * self.pixel_size_east

        sub = ENUGrid(
            ref_lat=self.ref_lat,
            ref_lon=self.ref_lon,
            ref_alt=self.ref_alt,
            min_east=tile_min_east,
            max_east=tile_max_east,
            min_north=tile_min_north,
            max_north=tile_max_north,
            pixel_size_east=self.pixel_size_east,
            pixel_size_north=self.pixel_size_north,
        )
        sub.rows = tile_rows
        sub.cols = tile_cols
        return sub

    def __repr__(self) -> str:
        return (
            f"ENUGrid(ref=({self.ref_lat:.4f}, {self.ref_lon:.4f}), "
            f"east=[{self.min_east:.1f}, {self.max_east:.1f}]m, "
            f"north=[{self.min_north:.1f}, {self.max_north:.1f}]m, "
            f"size={self.rows}x{self.cols}, "
            f"res=({self.pixel_size_east:.1f}, {self.pixel_size_north:.1f})m)"
        )
