# -*- coding: utf-8 -*-
"""
Geolocation Base Classes - Abstract interfaces for coordinate transformations.

Defines abstract base classes for transforming between image pixel coordinates
and geographic coordinates (latitude/longitude). Concrete implementations handle
different coordinate systems (geocoded raster, slant range SAR, etc.).

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
2026-01-30

Modified
--------
2026-03-31  Add _resolve_height and _fill_nan_heights for consistent
            height/NaN handling across all subclasses.  Add interpolation
            parameter to __init__ and _build_elevation_model.
2026-03-22  Refactor public API to (N, M) stacked ndarray convention.
            Add _handles_dem_internally bypass, _latlon_to_image_with_dem,
            delegate _build_elevation_model to open_elevation.
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
    - **Stacked (N, 2) array:** ``geo.image_to_latlon(points_Nx2)``

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

    Subclasses whose sensor model already performs per-point DEM iteration
    inside ``_image_to_latlon_array`` (e.g. SICD/SIDD R/Rdot via
    ``image_to_ground_hae``) should set ``_handles_dem_internally = True``
    so that the base-class wrapper does not add a redundant outer loop.
    """

    _handles_dem_internally: bool = False
    """If True, ``_image_to_latlon_array`` already queries ``self.elevation``
    internally (e.g. via the R/Rdot projection engine).  The base-class
    ``_image_to_latlon_with_dem`` wrapper will bypass its own DEM iteration
    to avoid double lookups.  Subclasses that handle DEM in their own
    projection loop should set this to True.  Subclasses that also handle
    DEM in ``_latlon_to_image_array`` should also set this flag so that
    ``_latlon_to_image_with_dem`` bypasses as well."""

    def __init__(
        self,
        shape: Tuple[int, int],
        crs: str = 'WGS84',
        dem_path: Optional[Union[str, Any]] = None,
        geoid_path: Optional[Union[str, Any]] = None,
        interpolation: int = 3,
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
        interpolation : int, default=3
            Spline interpolation order for DEM sampling (1=bilinear,
            3=bicubic, 5=quintic).  Only used when ``dem_path`` is
            provided.
        """
        self.shape = shape
        self.crs = crs
        self.elevation = None

        if dem_path is not None:
            self.elevation = _build_elevation_model(
                dem_path, geoid_path, interpolation=interpolation)

    @property
    def default_hae(self) -> float:
        """Default height above ellipsoid (meters) for this imagery.

        Returns the scene reference height used when no explicit height
        or DEM is provided.  Subclasses should override to return their
        sensor-model-specific reference height (SCP HAE, reference
        point height, average GCP height, etc.).

        Returns
        -------
        float
            Height above WGS-84 ellipsoid in meters.
        """
        return 0.0

    def _resolve_height(self, height: Union[float, np.ndarray]) -> float:
        """Resolve a single representative height for projection.

        Returns the explicit height if non-zero, otherwise falls back
        to ``default_hae`` (which subclasses override to return their
        sensor-specific reference height: SCP HAE, reference point,
        etc.).

        All geolocation subclasses should call this instead of
        implementing their own ``height if height != 0.0 else ...``
        logic, so that height resolution is consistent everywhere.

        Parameters
        ----------
        height : float or np.ndarray
            Height value from the caller.  Scalar ``0.0`` triggers
            fallback to ``default_hae``.  An array triggers fallback
            when all elements are zero.

        Returns
        -------
        float
            A single representative height in meters HAE.
        """
        if np.ndim(height) > 0:
            arr = np.asarray(height, dtype=np.float64)
            if np.any(arr != 0.0):
                return float(np.mean(arr))
            return self.default_hae
        return float(height) if height != 0.0 else self.default_hae

    def _fill_nan_heights(
        self,
        dem_h: np.ndarray,
        fallback_height: Union[float, np.ndarray] = 0.0,
    ) -> np.ndarray:
        """Fill NaN gaps in DEM-queried heights with a fallback value.

        Uses ``fallback_height`` when it is non-zero, otherwise falls
        back to ``default_hae``.  This ensures all geolocation paths
        handle missing DEM coverage identically.

        Parameters
        ----------
        dem_h : np.ndarray
            Heights from a DEM query, shape ``(N,)``.  May contain NaN
            where the DEM has no coverage.  **Modified in-place.**
        fallback_height : float or np.ndarray, default=0.0
            Caller's explicit height.  When scalar and non-zero, used
            directly as the fill value.  When zero or array, fills with
            ``default_hae``.

        Returns
        -------
        np.ndarray
            The same ``dem_h`` array with NaN values replaced.
        """
        nan_mask = np.isnan(dem_h)
        if not np.any(nan_mask):
            return dem_h
        if np.ndim(fallback_height) > 0:
            fill = self.default_hae
        else:
            fill = (float(fallback_height) if fallback_height != 0.0
                    else self.default_hae)
        dem_h[nan_mask] = fill
        return dem_h

    @abstractmethod
    def _image_to_latlon_array(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
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
        height : float or np.ndarray, default=0.0
            Height above WGS84 ellipsoid (meters).  Scalar applies a
            constant height to all points.  An array of shape ``(N,)``
            provides per-point heights for terrain-corrected projection.

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

    def _image_to_latlon_with_dem(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: float = 0.0,
        max_iter: int = 5,
        tol: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project pixels to ground with per-point iterative DEM refinement.

        When an elevation model is configured, iterates per point:
        project at current height → DEM lookup at that (lat, lon) →
        re-project at DEM height → converge.  Each point tracks its
        own height independently, correctly handling terrain variation
        across the query.

        When no elevation model is configured, falls through to the
        subclass ``_image_to_latlon_array`` with the constant ``height``.

        Parameters
        ----------
        rows, cols : np.ndarray
            Pixel coordinates (1D arrays of length N).
        height : float
            Initial height above WGS-84 (meters).  Used as the starting
            guess for all points when DEM is configured.
        max_iter : int
            Maximum DEM refinement iterations (default 5).
        tol : float
            Height convergence tolerance in meters (default 0.5).
            Iteration stops for a point once its height change is below
            this threshold.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (lats, lons, heights) in WGS-84.
        """
        if self.elevation is None or self._handles_dem_internally:
            return self._image_to_latlon_array(rows, cols, height)

        n = len(rows)
        initial_h = self._resolve_height(height)
        h_arr = np.full(n, initial_h)
        converged = np.zeros(n, dtype=bool)

        for _ in range(max_iter):
            lats, lons, heights_out = self._image_to_latlon_array(
                rows, cols, h_arr)

            # Per-point DEM lookup
            dem_h = np.asarray(
                self.elevation.get_elevation(lats, lons), dtype=np.float64)
            if dem_h.ndim == 0:
                dem_h = np.full(n, float(dem_h))

            self._fill_nan_heights(dem_h, height)

            # Check per-point convergence
            delta = np.abs(dem_h - h_arr)
            newly_converged = delta < tol
            converged |= newly_converged

            h_arr = dem_h

            if np.all(converged):
                break

        # Final projection at converged per-point heights
        lats, lons, _ = self._image_to_latlon_array(rows, cols, h_arr)
        return lats, lons, h_arr

    def _latlon_to_image_with_dem(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project geographic coordinates to image with DEM height lookup.

        When an elevation model is configured (and the subclass does not
        handle DEM internally), looks up the DEM height at each (lat, lon)
        and passes per-point heights to the subclass.  No iteration is
        needed because the caller already provides the ground position.

        Parameters
        ----------
        lats, lons : np.ndarray
            Geographic coordinates (1D arrays of length N).
        height : float or np.ndarray
            Height above WGS-84 (meters).  Ignored when DEM is configured.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (rows, cols) pixel coordinate arrays.
        """
        # Caller pre-sampled per-point heights — skip DEM lookup entirely.
        if np.ndim(height) > 0:
            return self._latlon_to_image_array(lats, lons, height)

        if self.elevation is None or self._handles_dem_internally:
            return self._latlon_to_image_array(lats, lons, height)

        # Single DEM lookup — no iteration needed (lat/lon are known)
        dem_h = np.asarray(
            self.elevation.get_elevation(lats, lons), dtype=np.float64)
        if dem_h.ndim == 0:
            dem_h = np.full(len(lats), float(dem_h))

        self._fill_nan_heights(dem_h, height)

        return self._latlon_to_image_array(lats, lons, dem_h)

    def image_to_latlon(
        self,
        row_or_points: Union[float, np.ndarray],
        col: Optional[float] = None,
        height: float = 0.0,
    ) -> np.ndarray:
        """
        Transform image coordinates to geographic coordinates.

        Parameters
        ----------
        row_or_points : float or np.ndarray
            Scalar row when ``col`` is also provided, or an ``(N, 2)``
            ndarray with columns ``[row, col]``.
        col : float, optional
            Scalar column.  Required when ``row_or_points`` is a scalar.
        height : float, default=0.0
            Height above WGS84 ellipsoid (meters). Ignored when a DEM
            elevation model is configured (``dem_path`` in constructor).

        Returns
        -------
        np.ndarray
            Shape ``(3,)`` for scalar input: ``[lat, lon, height]``.
            Shape ``(N, 3)`` for array input: columns are
            ``[lat, lon, height]``.

        Raises
        ------
        ValueError
            If input shape is invalid.
        NotImplementedError
            If geolocation is not available.

        Examples
        --------
        Single pixel (returns shape ``(3,)``):

        >>> result = geo.image_to_latlon(500, 1000)
        >>> lat, lon, h = result  # unpacking works

        Stacked ``(N, 2)`` array (returns shape ``(N, 3)``):

        >>> pixels = np.array([[100, 400],
        ...                    [200, 500],
        ...                    [300, 600]])
        >>> result = geo.image_to_latlon(pixels)
        >>> result[2]      # third point: [lat, lon, h]
        >>> result[:, 0]   # all latitudes
        """
        if col is not None:
            # Scalar shorthand: image_to_latlon(row, col)
            scalar = _is_scalar(row_or_points) and _is_scalar(col)
            rows_arr = _to_array(row_or_points)
            cols_arr = _to_array(col)
            lats, lons, heights = self._image_to_latlon_with_dem(
                rows_arr, cols_arr, height)
            result = np.column_stack([lats, lons, heights])
            return result[0] if scalar else result
        else:
            # Stacked (N, 2) array input
            pts = np.asarray(row_or_points, dtype=np.float64)
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            if pts.ndim != 2 or pts.shape[1] < 2:
                raise ValueError(
                    f"Expected (N, 2) array, got shape {pts.shape}")
            rows_arr = pts[:, 0]
            cols_arr = pts[:, 1]
            lats, lons, heights = self._image_to_latlon_with_dem(
                rows_arr, cols_arr, height)
            return np.column_stack([lats, lons, heights])

    def latlon_to_image(
        self,
        lat_or_points: Union[float, np.ndarray],
        lon: Optional[float] = None,
        height: Union[float, np.ndarray] = 0.0,
    ) -> np.ndarray:
        """
        Transform geographic coordinates to image coordinates.

        Parameters
        ----------
        lat_or_points : float or np.ndarray
            Scalar latitude when ``lon`` is also provided, or an ndarray
            of shape ``(N, 2)`` with columns ``[lat, lon]``, or
            ``(N, 3)`` with columns ``[lat, lon, height]`` (height
            column overrides the ``height`` parameter).
        lon : float, optional
            Scalar longitude.  Required when ``lat_or_points`` is a scalar.
        height : float or np.ndarray, default=0.0
            Height above WGS84 ellipsoid (meters). Ignored when
            ``lat_or_points`` has 3 columns or when a DEM is configured.

        Returns
        -------
        np.ndarray
            Shape ``(2,)`` for scalar input: ``[row, col]``.
            Shape ``(N, 2)`` for array input: columns are ``[row, col]``.

        Raises
        ------
        ValueError
            If input shape is invalid.
        NotImplementedError
            If geolocation is not available.

        Examples
        --------
        Single point (returns shape ``(2,)``):

        >>> result = geo.latlon_to_image(-31.05, 116.19)
        >>> row, col = result  # unpacking works

        Stacked ``(N, 2)`` without height:

        >>> coords = np.array([[-31.0, 116.1],
        ...                    [-31.1, 116.2]])
        >>> result = geo.latlon_to_image(coords)  # (N, 2)

        Stacked ``(N, 3)`` with embedded height:

        >>> coords = np.array([[-31.0, 116.1, 500.0],
        ...                    [-31.1, 116.2, 520.0]])
        >>> result = geo.latlon_to_image(coords)  # (N, 2)

        Round-trip with ``image_to_latlon`` output:

        >>> geo_pts = geo.image_to_latlon(pixels)   # (N, 3)
        >>> px_back = geo.latlon_to_image(geo_pts)   # (N, 2)
        """
        if lon is not None:
            # Scalar shorthand: latlon_to_image(lat, lon)
            scalar = _is_scalar(lat_or_points) and _is_scalar(lon)
            lats_arr = _to_array(lat_or_points)
            lons_arr = _to_array(lon)
            rows, cols = self._latlon_to_image_with_dem(
                lats_arr, lons_arr, height)
            result = np.column_stack([rows, cols])
            return result[0] if scalar else result
        else:
            # Stacked (N, 2) or (N, 3) array input
            pts = np.asarray(lat_or_points, dtype=np.float64)
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            if pts.ndim != 2 or pts.shape[1] < 2:
                raise ValueError(
                    f"Expected (N, 2) or (N, 3) array, got shape {pts.shape}")
            lats_arr = pts[:, 0]
            lons_arr = pts[:, 1]
            h = pts[:, 2] if pts.shape[1] >= 3 else height
            rows, cols = self._latlon_to_image_with_dem(
                lats_arr, lons_arr, h)
            return np.column_stack([rows, cols])

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

            result = self.image_to_latlon(
                np.column_stack([sample_rows, sample_cols]))
            lats = result[:, 0]
            lons = result[:, 1]

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
        height: Union[float, np.ndarray] = 0.0,
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
    interpolation: int = 3,
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
    interpolation : int, default=3
        Spline interpolation order for DEM sampling (1=bilinear,
        3=bicubic, 5=quintic).  Passed through to ``open_elevation``.

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
    from grdl.geolocation.elevation.open_elevation import open_elevation

    return open_elevation(
        str(dem_path),
        geoid_path=str(geoid_path) if geoid_path else None,
        interpolation=interpolation,
    )
