# -*- coding: utf-8 -*-
"""
UTM Output Grid - Regular output grid in UTM projected coordinates.

Provides ``UTMGrid``, an output grid specification for orthorectification
in Universal Transverse Mercator (UTM) coordinates.  Compatible with
``Orthorectifier`` as a drop-in alternative to ``GeographicGrid`` and
``ENUGrid``: provides ``rows``, ``cols``, ``image_to_latlon()``,
``latlon_to_image()``, and ``sub_grid()``.

Dependencies
------------
pyproj

Author
------
Duane Smalley, PhD
duane.d.smalley@gmail.com

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
        "UTMGrid requires pyproj.  Install with: "
        "pip install pyproj  (or: conda install -c conda-forge pyproj)"
    ) from _e

from grdl.image_processing.ortho.ortho import validate_sub_grid_indices

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from grdl.geolocation.base import Geolocation


def _utm_epsg(zone: int, north: bool) -> int:
    """Return the EPSG code for a UTM zone."""
    return 32600 + zone if north else 32700 + zone


def _auto_utm_zone(lon: float) -> int:
    """Compute the UTM zone number from a longitude."""
    return int((lon + 180.0) / 6.0) + 1


class UTMGrid:
    """Output grid specification in UTM projected coordinates.

    Defines a regular grid in UTM easting/northing meters for a
    specific zone and hemisphere.  Grid axes are aligned with UTM
    easting (columns) and northing (rows).  Pixel sizes are in meters.

    Row 0 is the north edge (max_northing), row increases southward.
    Column 0 is the west edge (min_easting), column increases eastward.
    This matches the ``GeographicGrid`` and ``ENUGrid`` conventions.

    Attributes
    ----------
    zone : int
        UTM zone number (1-60).
    north : bool
        True for northern hemisphere, False for southern.
    epsg : int
        EPSG code for this UTM zone (e.g. 32756 for zone 56 south).
    min_easting : float
        Western bound in UTM meters.
    max_easting : float
        Eastern bound in UTM meters.
    min_northing : float
        Southern bound in UTM meters.
    max_northing : float
        Northern bound in UTM meters.
    pixel_size : float
        Pixel spacing in meters (same for easting and northing).
    rows : int
        Number of output rows.
    cols : int
        Number of output columns.

    Examples
    --------
    >>> grid = UTMGrid(
    ...     zone=56, north=False,
    ...     min_easting=300000, max_easting=310000,
    ...     min_northing=7100000, max_northing=7110000,
    ...     pixel_size=10.0,
    ... )
    >>> print(f"{grid.rows} x {grid.cols}")
    1000 x 1000
    """

    def __init__(
        self,
        zone: int,
        north: bool,
        min_easting: float,
        max_easting: float,
        min_northing: float,
        max_northing: float,
        pixel_size: float,
    ) -> None:
        if not 1 <= zone <= 60:
            raise ValueError(f"UTM zone must be 1-60, got {zone}")
        if pixel_size <= 0:
            raise ValueError(f"pixel_size must be positive, got {pixel_size}")
        if max_easting <= min_easting:
            raise ValueError("max_easting must exceed min_easting")
        if max_northing <= min_northing:
            raise ValueError("max_northing must exceed min_northing")

        self.zone = zone
        self.north = north
        self.epsg = _utm_epsg(zone, north)
        self.min_easting = float(min_easting)
        self.max_easting = float(max_easting)
        self.min_northing = float(min_northing)
        self.max_northing = float(max_northing)
        self.pixel_size = float(pixel_size)

        self.rows = int(round(
            (self.max_northing - self.min_northing) / self.pixel_size))
        self.cols = int(round(
            (self.max_easting - self.min_easting) / self.pixel_size))

        if self.rows <= 0 or self.cols <= 0:
            raise ValueError(
                f"Grid dimensions must be positive: got {self.rows} x "
                f"{self.cols} (check bounds vs pixel_size)."
            )

        # pyproj transformers (cached, thread-safe)
        self._to_wgs84 = Transformer.from_crs(
            f"EPSG:{self.epsg}", "EPSG:4326", always_xy=True)
        self._from_wgs84 = Transformer.from_crs(
            "EPSG:4326", f"EPSG:{self.epsg}", always_xy=True)

        logger.debug(
            "UTMGrid zone=%d%s EPSG:%d  %d x %d  @ %.1f m",
            zone, 'N' if north else 'S', self.epsg,
            self.rows, self.cols, pixel_size,
        )

    @classmethod
    def from_geolocation(
        cls,
        geolocation: 'Geolocation',
        pixel_size_m: float,
        margin_m: float = 0.0,
    ) -> 'UTMGrid':
        """Build a UTM grid from a geolocation object's footprint.

        Auto-detects the UTM zone and hemisphere from the image center.

        Parameters
        ----------
        geolocation : Geolocation
            Source image geolocation (any subclass).
        pixel_size_m : float
            Output pixel spacing in meters.
        margin_m : float, default=0.0
            Extra margin in meters beyond the image footprint.

        Returns
        -------
        UTMGrid
            Grid covering the image footprint in UTM coordinates.
        """
        rows, cols = geolocation.shape
        _center = geolocation.image_to_latlon(
            float(rows // 2), float(cols // 2))
        center_lat = float(_center[0])
        center_lon = float(_center[1])

        zone = _auto_utm_zone(center_lon)
        north = center_lat >= 0

        # Get footprint bounds in WGS-84
        bounds = geolocation.get_bounds()  # (min_lon, min_lat, max_lon, max_lat)

        # Convert corners to UTM
        to_utm = Transformer.from_crs(
            "EPSG:4326", f"EPSG:{_utm_epsg(zone, north)}", always_xy=True)
        corner_lons = [bounds[0], bounds[2], bounds[0], bounds[2]]
        corner_lats = [bounds[1], bounds[1], bounds[3], bounds[3]]
        eastings, northings = to_utm.transform(corner_lons, corner_lats)

        min_e = float(np.min(eastings)) - margin_m
        max_e = float(np.max(eastings)) + margin_m
        min_n = float(np.min(northings)) - margin_m
        max_n = float(np.max(northings)) + margin_m

        # Snap to pixel grid
        min_e = np.floor(min_e / pixel_size_m) * pixel_size_m
        max_e = np.ceil(max_e / pixel_size_m) * pixel_size_m
        min_n = np.floor(min_n / pixel_size_m) * pixel_size_m
        max_n = np.ceil(max_n / pixel_size_m) * pixel_size_m

        return cls(
            zone=zone, north=north,
            min_easting=min_e, max_easting=max_e,
            min_northing=min_n, max_northing=max_n,
            pixel_size=pixel_size_m,
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

        easting = self.min_easting + col_arr * self.pixel_size
        northing = self.max_northing - row_arr * self.pixel_size

        lon, lat = self._to_wgs84.transform(easting, northing)

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

        easting, northing = self._from_wgs84.transform(lon_arr, lat_arr)

        row = (self.max_northing - northing) / self.pixel_size
        col = (easting - self.min_easting) / self.pixel_size

        if scalar:
            return float(row), float(col)
        return np.asarray(row), np.asarray(col)

    def sub_grid(
        self,
        row_start: int,
        col_start: int,
        row_end: int,
        col_end: int,
    ) -> 'UTMGrid':
        """Extract a sub-grid for tiled processing.

        Parameters
        ----------
        row_start, col_start : int
            Top-left corner of the tile (inclusive).
        row_end, col_end : int
            Bottom-right corner of the tile (exclusive).

        Returns
        -------
        UTMGrid
            Sub-grid covering the specified tile region.
        """
        validate_sub_grid_indices(
            self.rows, self.cols, row_start, col_start, row_end, col_end)

        sub_min_e = self.min_easting + col_start * self.pixel_size
        sub_max_e = self.min_easting + col_end * self.pixel_size
        sub_min_n = self.max_northing - row_end * self.pixel_size
        sub_max_n = self.max_northing - row_start * self.pixel_size

        return UTMGrid(
            zone=self.zone, north=self.north,
            min_easting=sub_min_e, max_easting=sub_max_e,
            min_northing=sub_min_n, max_northing=sub_max_n,
            pixel_size=self.pixel_size,
        )

    def __repr__(self) -> str:
        return (
            f"UTMGrid(zone={self.zone}{'N' if self.north else 'S'} "
            f"EPSG:{self.epsg}, "
            f"E=[{self.min_easting:.0f}, {self.max_easting:.0f}], "
            f"N=[{self.min_northing:.0f}, {self.max_northing:.0f}], "
            f"{self.rows}x{self.cols} @ {self.pixel_size:.1f}m)"
        )
