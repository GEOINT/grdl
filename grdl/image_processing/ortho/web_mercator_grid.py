# -*- coding: utf-8 -*-
"""
Web Mercator Output Grid - Regular output grid in EPSG:3857 coordinates.

Provides ``WebMercatorGrid``, an output grid specification for
orthorectification in Web Mercator (EPSG:3857) projected coordinates.
Compatible with ``Orthorectifier`` as a drop-in alternative to
``GeographicGrid``, ``ENUGrid``, and ``UTMGrid``: provides ``rows``,
``cols``, ``image_to_latlon()``, ``latlon_to_image()``, and
``sub_grid()``.

Web Mercator is the projection used by Leaflet, Google Maps, and
OpenStreetMap tile servers, making this grid ideal for generating
imagery that overlays directly on web basemaps.

Dependencies
------------
pyproj

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
2026-03-21

Modified
--------
2026-03-21
"""

import logging
from typing import Tuple, Union, TYPE_CHECKING

import numpy as np

try:
    from pyproj import Transformer
except ImportError as _e:
    raise ImportError(
        "WebMercatorGrid requires pyproj.  Install with: "
        "pip install pyproj  (or: conda install -c conda-forge pyproj)"
    ) from _e

from grdl.image_processing.ortho.ortho import validate_sub_grid_indices

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from grdl.geolocation.base import Geolocation


class WebMercatorGrid:
    """Output grid specification in Web Mercator (EPSG:3857) coordinates.

    Defines a regular grid in Web Mercator x/y meters.  Grid axes are
    aligned with the Mercator easting (columns) and northing (rows).
    Pixel sizes are in projected meters (which vary with latitude in
    ground distance, but are uniform in the Mercator projection).

    Row 0 is the north edge (max_y), row increases southward.
    Column 0 is the west edge (min_x), column increases eastward.
    This matches the ``GeographicGrid`` and ``ENUGrid`` conventions.

    Attributes
    ----------
    min_x : float
        Western bound in EPSG:3857 meters.
    max_x : float
        Eastern bound in EPSG:3857 meters.
    min_y : float
        Southern bound in EPSG:3857 meters.
    max_y : float
        Northern bound in EPSG:3857 meters.
    pixel_size : float
        Pixel spacing in EPSG:3857 meters.
    rows : int
        Number of output rows.
    cols : int
        Number of output columns.

    Examples
    --------
    >>> grid = WebMercatorGrid.from_bounds_latlon(
    ...     min_lat=-26.1, max_lat=-26.0,
    ...     min_lon=29.4, max_lon=29.5,
    ...     pixel_size=10.0,
    ... )
    >>> print(f"{grid.rows} x {grid.cols}")
    """

    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        pixel_size: float,
    ) -> None:
        if pixel_size <= 0:
            raise ValueError(f"pixel_size must be positive, got {pixel_size}")
        if max_x <= min_x:
            raise ValueError("max_x must exceed min_x")
        if max_y <= min_y:
            raise ValueError("max_y must exceed min_y")

        self.min_x = float(min_x)
        self.max_x = float(max_x)
        self.min_y = float(min_y)
        self.max_y = float(max_y)
        self.pixel_size = float(pixel_size)

        self.rows = int(round((self.max_y - self.min_y) / self.pixel_size))
        self.cols = int(round((self.max_x - self.min_x) / self.pixel_size))

        if self.rows <= 0 or self.cols <= 0:
            raise ValueError(
                f"Grid dimensions must be positive: got {self.rows} x "
                f"{self.cols} (check bounds vs pixel_size)."
            )

        # pyproj transformers (cached, thread-safe)
        self._to_wgs84 = Transformer.from_crs(
            "EPSG:3857", "EPSG:4326", always_xy=True)
        self._from_wgs84 = Transformer.from_crs(
            "EPSG:4326", "EPSG:3857", always_xy=True)

        logger.debug(
            "WebMercatorGrid EPSG:3857  %d x %d  @ %.1f m",
            self.rows, self.cols, pixel_size,
        )

    @classmethod
    def from_bounds_latlon(
        cls,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        pixel_size: float,
    ) -> 'WebMercatorGrid':
        """Build a Web Mercator grid from WGS-84 lat/lon bounds.

        Parameters
        ----------
        min_lat, max_lat : float
            Latitude bounds in degrees.
        min_lon, max_lon : float
            Longitude bounds in degrees.
        pixel_size : float
            Pixel spacing in EPSG:3857 meters.

        Returns
        -------
        WebMercatorGrid
            Grid covering the specified geographic bounds.
        """
        to_merc = Transformer.from_crs(
            "EPSG:4326", "EPSG:3857", always_xy=True)

        x_min, y_min = to_merc.transform(min_lon, min_lat)
        x_max, y_max = to_merc.transform(max_lon, max_lat)

        # Snap to pixel grid
        x_min = np.floor(x_min / pixel_size) * pixel_size
        x_max = np.ceil(x_max / pixel_size) * pixel_size
        y_min = np.floor(y_min / pixel_size) * pixel_size
        y_max = np.ceil(y_max / pixel_size) * pixel_size

        return cls(
            min_x=x_min, max_x=x_max,
            min_y=y_min, max_y=y_max,
            pixel_size=pixel_size,
        )

    @classmethod
    def from_geolocation(
        cls,
        geolocation: 'Geolocation',
        pixel_size: float,
        margin_m: float = 0.0,
    ) -> 'WebMercatorGrid':
        """Build a Web Mercator grid from a geolocation's footprint.

        Parameters
        ----------
        geolocation : Geolocation
            Source image geolocation (any subclass).
        pixel_size : float
            Pixel spacing in EPSG:3857 meters.
        margin_m : float, default=0.0
            Extra margin in Mercator meters beyond the footprint.

        Returns
        -------
        WebMercatorGrid
            Grid covering the image footprint.
        """
        bounds = geolocation.get_bounds()  # (min_lon, min_lat, max_lon, max_lat)

        to_merc = Transformer.from_crs(
            "EPSG:4326", "EPSG:3857", always_xy=True)
        corner_lons = [bounds[0], bounds[2], bounds[0], bounds[2]]
        corner_lats = [bounds[1], bounds[1], bounds[3], bounds[3]]
        xs, ys = to_merc.transform(corner_lons, corner_lats)

        min_x = float(np.min(xs)) - margin_m
        max_x = float(np.max(xs)) + margin_m
        min_y = float(np.min(ys)) - margin_m
        max_y = float(np.max(ys)) + margin_m

        # Snap to pixel grid
        min_x = np.floor(min_x / pixel_size) * pixel_size
        max_x = np.ceil(max_x / pixel_size) * pixel_size
        min_y = np.floor(min_y / pixel_size) * pixel_size
        max_y = np.ceil(max_y / pixel_size) * pixel_size

        return cls(
            min_x=min_x, max_x=max_x,
            min_y=min_y, max_y=max_y,
            pixel_size=pixel_size,
        )

    def image_to_latlon(
        self,
        row: Union[float, np.ndarray],
        col: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Convert grid pixel coordinates to WGS-84 lat/lon.

        Parameters
        ----------
        row, col : float or np.ndarray
            Grid pixel coordinates (row 0 = north edge).

        Returns
        -------
        Tuple
            (lat, lon) in WGS-84 degrees.
        """
        scalar = np.ndim(row) == 0 and np.ndim(col) == 0
        row_arr = np.asarray(row, dtype=np.float64)
        col_arr = np.asarray(col, dtype=np.float64)

        x = self.min_x + col_arr * self.pixel_size
        y = self.max_y - row_arr * self.pixel_size

        lon, lat = self._to_wgs84.transform(x, y)

        if scalar:
            return float(lat), float(lon)
        return np.asarray(lat), np.asarray(lon)

    def latlon_to_image(
        self,
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Convert WGS-84 lat/lon to grid pixel coordinates.

        Parameters
        ----------
        lat, lon : float or np.ndarray
            WGS-84 coordinates in degrees.

        Returns
        -------
        Tuple
            (row, col) in grid pixel coordinates.
        """
        scalar = np.ndim(lat) == 0 and np.ndim(lon) == 0
        lat_arr = np.asarray(lat, dtype=np.float64)
        lon_arr = np.asarray(lon, dtype=np.float64)

        x, y = self._from_wgs84.transform(lon_arr, lat_arr)

        row = (self.max_y - y) / self.pixel_size
        col = (x - self.min_x) / self.pixel_size

        if scalar:
            return float(row), float(col)
        return np.asarray(row), np.asarray(col)

    def sub_grid(
        self,
        row_start: int,
        col_start: int,
        row_end: int,
        col_end: int,
    ) -> 'WebMercatorGrid':
        """Extract a sub-grid for tiled processing.

        Parameters
        ----------
        row_start, col_start : int
            Top-left corner of the tile (inclusive).
        row_end, col_end : int
            Bottom-right corner of the tile (exclusive).

        Returns
        -------
        WebMercatorGrid
            Sub-grid covering the specified tile region.
        """
        validate_sub_grid_indices(
            self.rows, self.cols, row_start, col_start, row_end, col_end)

        sub_min_x = self.min_x + col_start * self.pixel_size
        sub_max_x = self.min_x + col_end * self.pixel_size
        sub_min_y = self.max_y - row_end * self.pixel_size
        sub_max_y = self.max_y - row_start * self.pixel_size

        return WebMercatorGrid(
            min_x=sub_min_x, max_x=sub_max_x,
            min_y=sub_min_y, max_y=sub_max_y,
            pixel_size=self.pixel_size,
        )

    def __repr__(self) -> str:
        return (
            f"WebMercatorGrid(EPSG:3857, "
            f"x=[{self.min_x:.0f}, {self.max_x:.0f}], "
            f"y=[{self.min_y:.0f}, {self.max_y:.0f}], "
            f"{self.rows}x{self.cols} @ {self.pixel_size:.1f}m)"
        )
