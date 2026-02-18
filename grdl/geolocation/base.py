# -*- coding: utf-8 -*-
"""
Geolocation Base Classes - Abstract interfaces for coordinate transformations.

Defines abstract base classes for transforming between image pixel coordinates
and geographic coordinates (latitude/longitude). Concrete implementations handle
different coordinate systems (geocoded raster, slant range SAR, etc.).

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
2026-01-30

Modified
--------
2026-02-17
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union, Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from grdl.IO.base import ImageReader


def _is_scalar(val: Any) -> bool:
    """Check if a value is a scalar (not array-like)."""
    if isinstance(val, np.ndarray):
        return val.ndim == 0
    return isinstance(val, (int, float, np.integer, np.floating))


def _to_array(val: Any) -> np.ndarray:
    """Convert scalar, list, or array to 1D numpy array of float64."""
    arr = np.asarray(val, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


class Geolocation(ABC):
    """
    Abstract base class for geolocation transformations.

    Provides interface for transforming between image pixel coordinates and
    geographic coordinates. Concrete implementations handle different coordinate
    systems and geometries (geocoded raster, SAR slant range, etc.).

    ``image_to_latlon`` and ``latlon_to_image`` accept three input forms:

    - **Scalar:** ``geo.image_to_latlon(500, 1000)``
    - **Separate arrays:** ``geo.image_to_latlon(rows_array, cols_array)``
    - **Stacked (2, N) array:** ``geo.image_to_latlon(points_2xN)``

    Coordinate Conventions
    ----------------------
    - **Image coordinates:** (row, col) with (0, 0) at top-left corner
    - **Geographic coordinates:** (lat, lon, height) in WGS84 (EPSG:4326)
    - **Height:** Above WGS84 ellipsoid (not geoid)

    Notes
    -----
    Subclasses implement ``_image_to_latlon_array`` and ``_latlon_to_image_array``
    which operate on 1D numpy arrays. The public methods handle scalar/list/array
    dispatch automatically.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        crs: str = 'WGS84',
        dem_path: Optional[Union[str, Any]] = None,
        geoid_path: Optional[Union[str, Any]] = None,
    ):
        """
        Initialize geolocation.

        Parameters
        ----------
        shape : Tuple[int, int]
            Image shape (rows, cols).
        crs : str, default='WGS84'
            Coordinate reference system.
        dem_path : str or Path, optional
            Path to DEM/DTED data folder. When provided, terrain-corrected
            heights are used in ``image_to_latlon`` instead of the constant
            ``height`` parameter.
        geoid_path : str or Path, optional
            Path to geoid correction file (EGM96/EGM2008). Used with
            ``dem_path`` to convert DEM heights (MSL) to HAE.
        """
        self.shape = shape
        self.crs = crs
        self.elevation = None

        if dem_path is not None:
            self.elevation = _build_elevation_model(dem_path, geoid_path)

    @abstractmethod
    def _image_to_latlon_array(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform pixel coordinate arrays to geographic coordinate arrays.

        Subclasses implement this single vectorized method. The public
        ``image_to_latlon`` dispatches to it for both scalar and array inputs.

        Parameters
        ----------
        rows : np.ndarray
            Row coordinates (1D array, float64).
        cols : np.ndarray
            Column coordinates (1D array, float64).
        height : float, default=0.0
            Height above WGS84 ellipsoid (meters).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (lats, lons, heights) arrays in WGS84 coordinates.
        """
        pass

    @abstractmethod
    def _latlon_to_image_array(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform geographic coordinate arrays to pixel coordinate arrays.

        Subclasses implement this single vectorized method. The public
        ``latlon_to_image`` dispatches to it for both scalar and array inputs.

        Parameters
        ----------
        lats : np.ndarray
            Latitudes in degrees North (1D array, float64).
        lons : np.ndarray
            Longitudes in degrees East (1D array, float64).
        height : float or np.ndarray, default=0.0
            Height above WGS84 ellipsoid (meters). Scalar applies a
            constant height to all points. An array of shape ``(N,)``
            provides per-point heights for terrain-corrected projection.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (rows, cols) pixel coordinate arrays.
        """
        pass

    def image_to_latlon(
        self,
        row_or_points: Union[float, list, np.ndarray],
        col: Optional[Union[float, list, np.ndarray]] = None,
        height: float = 0.0
    ) -> Union[Tuple[float, float, float],
               Tuple[np.ndarray, np.ndarray, np.ndarray],
               np.ndarray]:
        """
        Transform image coordinates to geographic coordinates.

        Accepts three input forms:

        - **Scalar:** ``image_to_latlon(row, col)`` returns ``(lat, lon, height)`` floats.
        - **Separate arrays:** ``image_to_latlon(rows, cols)`` returns
          ``(lats, lons, heights)`` tuple of arrays.
        - **Stacked array:** ``image_to_latlon(points_2xN)`` returns ``(3, N)``
          ndarray with rows ``[lats, lons, heights]``.

        Parameters
        ----------
        row_or_points : float, list, np.ndarray
            Row coordinate(s) when ``col`` is provided, or a ``(2, N)`` ndarray
            of stacked ``[rows; cols]`` when ``col`` is None.
        col : float, list, or np.ndarray, optional
            Column coordinate(s). Omit to pass a ``(2, N)`` stacked array as
            the first argument.
        height : float, default=0.0
            Height above WGS84 ellipsoid (meters). Ignored when a DEM
            elevation model is configured (``dem_path`` in constructor).

        Returns
        -------
        Tuple[float, float, float]
            ``(lat, lon, height)`` when scalar inputs are given.
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(lats, lons, heights)`` when separate array/list inputs are given.
        np.ndarray
            Shape ``(3, N)`` when a ``(2, N)`` stacked array is given.

        Raises
        ------
        ValueError
            If pixel coordinates are out of image bounds or input shape
            is invalid.
        NotImplementedError
            If geolocation is not available.

        Examples
        --------
        Single pixel:

        >>> lat, lon, h = geo.image_to_latlon(500, 1000)

        Separate arrays:

        >>> lats, lons, heights = geo.image_to_latlon(
        ...     [100, 200, 300], [400, 500, 600]
        ... )

        Stacked (2, N) array:

        >>> points = np.array([[100, 200, 300],   # rows
        ...                    [400, 500, 600]])   # cols
        >>> result = geo.image_to_latlon(points)   # (3, N)
        """
        if col is None:
            # (2, N) ndarray input
            pts = np.asarray(row_or_points, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] != 2:
                raise ValueError(
                    f"Expected (2, N) array, got shape {pts.shape}"
                )
            rows_arr = pts[0]
            cols_arr = pts[1]
            lats, lons, heights = self._image_to_latlon_array(
                rows_arr, cols_arr, height
            )
            if self.elevation is not None and height == 0.0:
                heights = self.elevation._get_elevation_array(lats, lons)
            return np.vstack([lats, lons, heights])
        elif _is_scalar(row_or_points) and _is_scalar(col):
            # Scalar input
            rows_arr = _to_array(row_or_points)
            cols_arr = _to_array(col)
            lats, lons, heights = self._image_to_latlon_array(
                rows_arr, cols_arr, height
            )
            if self.elevation is not None and height == 0.0:
                heights = self.elevation._get_elevation_array(lats, lons)
            return (float(lats[0]), float(lons[0]), float(heights[0]))
        else:
            # Separate array/list inputs
            rows_arr = _to_array(row_or_points)
            cols_arr = _to_array(col)
            lats, lons, heights = self._image_to_latlon_array(
                rows_arr, cols_arr, height
            )
            if self.elevation is not None and height == 0.0:
                heights = self.elevation._get_elevation_array(lats, lons)
            return lats, lons, heights

    def latlon_to_image(
        self,
        lat_or_points: Union[float, list, np.ndarray],
        lon: Optional[Union[float, list, np.ndarray]] = None,
        height: Union[float, np.ndarray] = 0.0
    ) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Transform geographic coordinates to image coordinates.

        Accepts three input forms:

        - **Scalar:** ``latlon_to_image(lat, lon)`` returns ``(row, col)`` floats.
        - **Separate arrays:** ``latlon_to_image(lats, lons)`` returns
          ``(rows, cols)`` tuple of arrays.
        - **Stacked array:** ``latlon_to_image(points_2xN)`` returns ``(2, N)``
          ndarray with rows ``[rows, cols]``.

        Parameters
        ----------
        lat_or_points : float, list, np.ndarray
            Latitude(s) when ``lon`` is provided, or a ``(2, N)`` ndarray
            of stacked ``[lats; lons]`` when ``lon`` is None.
        lon : float, list, or np.ndarray, optional
            Longitude(s). Omit to pass a ``(2, N)`` stacked array as
            the first argument.
        height : float or np.ndarray, default=0.0
            Height above WGS84 ellipsoid (meters). Scalar applies a
            constant height to all points. An array of shape ``(N,)``
            provides per-point heights for terrain-corrected projection.

        Returns
        -------
        Tuple[float, float]
            ``(row, col)`` when scalar inputs are given.
        Tuple[np.ndarray, np.ndarray]
            ``(rows, cols)`` when separate array/list inputs are given.
        np.ndarray
            Shape ``(2, N)`` when a ``(2, N)`` stacked array is given.

        Raises
        ------
        ValueError
            If geographic coordinates are outside image footprint or input
            shape is invalid.
        NotImplementedError
            If geolocation is not available.

        Examples
        --------
        Single point:

        >>> row, col = geo.latlon_to_image(-31.05, 116.19)

        Separate arrays:

        >>> rows, cols = geo.latlon_to_image(
        ...     [-31.0, -31.1, -31.2], [116.1, 116.2, 116.3]
        ... )

        Stacked (2, N) array:

        >>> coords = np.array([[-31.0, -31.1, -31.2],   # lats
        ...                    [116.1, 116.2, 116.3]])   # lons
        >>> result = geo.latlon_to_image(coords)          # (2, N)
        """
        if lon is None:
            # (2, N) ndarray input
            pts = np.asarray(lat_or_points, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] != 2:
                raise ValueError(
                    f"Expected (2, N) array, got shape {pts.shape}"
                )
            lats_arr = pts[0]
            lons_arr = pts[1]
            rows, cols = self._latlon_to_image_array(
                lats_arr, lons_arr, height
            )
            return np.vstack([rows, cols])
        elif _is_scalar(lat_or_points) and _is_scalar(lon):
            # Scalar input
            lats_arr = _to_array(lat_or_points)
            lons_arr = _to_array(lon)
            rows, cols = self._latlon_to_image_array(
                lats_arr, lons_arr, height
            )
            return (float(rows[0]), float(cols[0]))
        else:
            # Separate array/list inputs
            lats_arr = _to_array(lat_or_points)
            lons_arr = _to_array(lon)
            rows, cols = self._latlon_to_image_array(
                lats_arr, lons_arr, height
            )
            return rows, cols

    def get_footprint(self) -> Dict[str, Any]:
        """
        Calculate image footprint as geographic polygon and bounding box.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
            - 'type': 'Polygon' or 'None'
            - 'coordinates': List of (lon, lat) tuples forming perimeter polygon
            - 'bounds': (min_lon, min_lat, max_lon, max_lat) bounding box

        Notes
        -----
        Default implementation samples perimeter points using
        sample_image_perimeter(). Subclasses can override for more
        efficient implementations.
        """
        from grdl.geolocation.utils import sample_image_perimeter

        try:
            sample_rows, sample_cols = sample_image_perimeter(
                self.shape, samples_per_edge=10
            )

            lats, lons, _ = self.image_to_latlon(sample_rows, sample_cols)

            # Filter out any NaN values from outside coverage area
            valid = ~(np.isnan(lats) | np.isnan(lons))
            if not np.any(valid):
                return {
                    'type': 'None',
                    'coordinates': None,
                    'bounds': None
                }

            valid_lats = lats[valid]
            valid_lons = lons[valid]

            # Perimeter coordinates as (lon, lat) tuples
            perimeter_coords = list(zip(
                valid_lons.tolist(), valid_lats.tolist()
            ))

            # Calculate bounds from all valid sample points
            min_lon, max_lon = float(np.min(valid_lons)), float(np.max(valid_lons))
            min_lat, max_lat = float(np.min(valid_lats)), float(np.max(valid_lats))

            return {
                'type': 'Polygon',
                'coordinates': perimeter_coords,
                'bounds': (min_lon, min_lat, max_lon, max_lat)
            }

        except (ValueError, NotImplementedError):
            return {
                'type': 'None',
                'coordinates': None,
                'bounds': None
            }

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get bounding box of image footprint.

        Returns
        -------
        Tuple[float, float, float, float]
            (min_lon, min_lat, max_lon, max_lat) in degrees

        Raises
        ------
        NotImplementedError
            If geolocation is not available
        """
        footprint = self.get_footprint()
        bounds = footprint.get('bounds')

        if bounds is None:
            raise NotImplementedError("Geolocation not available for this imagery")

        return bounds


class NoGeolocation(Geolocation):
    """
    Fallback geolocation class for imagery without geolocation information.

    Raises NotImplementedError for all transformation operations.
    """

    def __init__(self, shape: Tuple[int, int], crs: str = 'WGS84'):
        """
        Initialize no-geolocation fallback.

        Parameters
        ----------
        shape : Tuple[int, int]
            Image shape (rows, cols)
        crs : str, default='WGS84'
            Coordinate reference system (not used)
        """
        super().__init__(shape, crs)

    def _image_to_latlon_array(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Raise NotImplementedError - no geolocation available."""
        raise NotImplementedError(
            "This imagery has no geolocation information. "
            "Cannot transform image coordinates to lat/lon."
        )

    def _latlon_to_image_array(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Raise NotImplementedError - no geolocation available."""
        raise NotImplementedError(
            "This imagery has no geolocation information. "
            "Cannot transform lat/lon to image coordinates."
        )

    def get_footprint(self) -> Dict[str, Any]:
        """Return empty footprint."""
        return {
            'type': 'None',
            'coordinates': None,
            'bounds': None
        }


def _build_elevation_model(
    dem_path: Union[str, Any],
    geoid_path: Optional[Union[str, Any]] = None,
) -> Any:
    """
    Build an ElevationModel from DEM path and optional geoid path.

    Attempts to auto-detect DEM type (DTED folder vs GeoTIFF file).

    Parameters
    ----------
    dem_path : str or Path
        Path to DEM/DTED data.
    geoid_path : str or Path, optional
        Path to geoid correction file.

    Returns
    -------
    ElevationModel
        Configured elevation model.

    Raises
    ------
    FileNotFoundError
        If dem_path does not exist.
    ImportError
        If required dependencies are not installed.
    """
    from pathlib import Path

    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM path does not exist: {dem_path}")

    if dem_path.is_dir():
        from grdl.geolocation.elevation.dted import DTEDElevation
        return DTEDElevation(dem_path, geoid_path)
    else:
        from grdl.geolocation.elevation.geotiff_dem import GeoTIFFDEM
        return GeoTIFFDEM(dem_path, geoid_path)
