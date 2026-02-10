# -*- coding: utf-8 -*-
"""
GCP-Based Geolocation - Coordinate transformations using Ground Control Points.

Implements geolocation for imagery with Ground Control Points (GCPs) using
interpolation. Primary use case is SAR slant range imagery (e.g., BIOMASS L1 SCS)
where GCPs provide the only geolocation information.

Dependencies
------------
scipy

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
2026-01-30
"""

from typing import Dict, List, Tuple, Any

import numpy as np

try:
    from scipy.interpolate import LinearNDInterpolator
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    LinearNDInterpolator = None

from grdl.geolocation.base import Geolocation
from grdl.geolocation.utils import check_pixel_bounds


class GCPGeolocation(Geolocation):
    """
    Geolocation using Ground Control Points (GCPs) with interpolation.

    Uses 2D interpolation to transform between pixel and geographic coordinates
    based on a set of ground control points. Suitable for SAR slant range imagery
    (BIOMASS L1 SCS) where GCPs are the primary geolocation source.

    The interpolation uses scipy.interpolate.LinearNDInterpolator which provides:
    - Fast evaluation
    - Delaunay triangulation for irregular point distributions
    - Extrapolation handling (returns NaN outside convex hull)

    Attributes
    ----------
    gcps : List[Tuple[float, float, float, float, float]]
        Ground control points as (lon, lat, height, row, col) tuples
    shape : Tuple[int, int]
        Image shape (rows, cols)
    crs : str
        Coordinate reference system
    n_gcps : int
        Number of GCPs used for interpolation

    Notes
    -----
    Interpolation accuracy depends on GCP density and distribution. For best
    results, GCPs should be well-distributed across the image. Accuracy typically
    degrades near image edges and in regions far from GCPs.
    """

    def __init__(
        self,
        gcps: List[Tuple[float, float, float, float, float]],
        shape: Tuple[int, int],
        crs: str = 'WGS84'
    ):
        """
        Initialize GCP-based geolocation.

        Parameters
        ----------
        gcps : List[Tuple[float, float, float, float, float]]
            Ground control points as (lon, lat, height, row, col) tuples
            - lon: longitude in degrees East
            - lat: latitude in degrees North
            - height: height above WGS84 ellipsoid in meters
            - row: pixel row coordinate (0-based)
            - col: pixel column coordinate (0-based)
        shape : Tuple[int, int]
            Image shape (rows, cols)
        crs : str, default='WGS84'
            Coordinate reference system

        Raises
        ------
        ValueError
            If less than 4 GCPs provided (minimum for 2D interpolation)
        ImportError
            If scipy is not installed
        """
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for GCP-based geolocation. "
                "Install with: pip install scipy>=1.7.0"
            )

        if len(gcps) < 4:
            raise ValueError(
                f"At least 4 GCPs required for interpolation, got {len(gcps)}"
            )

        super().__init__(shape, crs)

        self.gcps = gcps
        self.n_gcps = len(gcps)

        # Build interpolation models
        self._build_interpolator()

    def _build_interpolator(self) -> None:
        """
        Build 2D interpolation models for forward and inverse transforms.

        Creates LinearNDInterpolator objects for:
        - Forward transform: (row, col) → (lat, lon, height)
        - Inverse transform: (lat, lon) → (row, col)

        Notes
        -----
        Uses Delaunay triangulation internally. Points outside the convex
        hull of GCPs will return NaN.
        """
        # Extract GCP arrays using vectorized numpy operations
        # Tuple format: (lon, lat, height, row, col)
        gcps_array = np.array(self.gcps)  # Shape: (n_gcps, 5)
        lons = gcps_array[:, 0]
        lats = gcps_array[:, 1]
        heights = gcps_array[:, 2]
        rows = gcps_array[:, 3]
        cols = gcps_array[:, 4]

        # Forward: (row, col) → (lat, lon, height)
        pixel_points = np.column_stack([rows, cols])
        self._lat_interp = LinearNDInterpolator(pixel_points, lats, fill_value=np.nan)
        self._lon_interp = LinearNDInterpolator(pixel_points, lons, fill_value=np.nan)
        self._height_interp = LinearNDInterpolator(pixel_points, heights, fill_value=np.nan)

        # Inverse: (lat, lon) → (row, col)
        geo_points = np.column_stack([lats, lons])
        self._row_interp = LinearNDInterpolator(geo_points, rows, fill_value=np.nan)
        self._col_interp = LinearNDInterpolator(geo_points, cols, fill_value=np.nan)

    def _pixel_to_latlon_array(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform pixel coordinate arrays to geographic coordinate arrays.

        Vectorized implementation using scipy LinearNDInterpolator.

        Parameters
        ----------
        rows : np.ndarray
            Row coordinates (1D array, float64).
        cols : np.ndarray
            Column coordinates (1D array, float64).
        height : float, default=0.0
            Height parameter (not used in GCP interpolation, heights come
            from the GCP data itself).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (lats, lons, heights) arrays in WGS84 coordinates.
            Points outside the GCP convex hull return NaN.
        """
        points = np.column_stack([rows, cols])

        lats = self._lat_interp(points)
        lons = self._lon_interp(points)
        heights = self._height_interp(points)

        return lats, lons, heights

    def _latlon_to_pixel_array(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform geographic coordinate arrays to pixel coordinate arrays.

        Vectorized implementation using scipy LinearNDInterpolator.

        Parameters
        ----------
        lats : np.ndarray
            Latitudes in degrees North (1D array, float64).
        lons : np.ndarray
            Longitudes in degrees East (1D array, float64).
        height : float, default=0.0
            Height parameter (not used in 2D interpolation).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (rows, cols) pixel coordinate arrays.
            Points outside the GCP convex hull return NaN.
        """
        geo_points = np.column_stack([lats, lons])

        rows = self._row_interp(geo_points)
        cols = self._col_interp(geo_points)

        return rows, cols

    def get_interpolation_error(self) -> Dict[str, float]:
        """
        Estimate interpolation error by cross-validation on GCPs.

        Uses leave-one-out approach: interpolate each GCP using all others,
        compare with true GCP location.

        Returns
        -------
        Dict[str, float]
            Error metrics:
            - 'mean_error_m': Mean position error in meters
            - 'rms_error_m': RMS position error in meters
            - 'max_error_m': Maximum position error in meters

        Notes
        -----
        This provides an estimate of interpolation accuracy. Actual errors
        may be higher in regions far from GCPs or near image edges.
        """
        from grdl.geolocation.utils import geographic_distance

        errors_m = []

        # Leave-one-out cross-validation
        for i in range(self.n_gcps):
            # True GCP location
            # Tuple format: (lon, lat, height, row, col)
            true_lat, true_lon = self.gcps[i][1], self.gcps[i][0]
            true_row, true_col = self.gcps[i][3], self.gcps[i][4]

            # Build interpolator without this GCP
            gcps_subset = self.gcps[:i] + self.gcps[i+1:]
            temp_geo = GCPGeolocation(gcps_subset, self.shape, self.crs)

            try:
                # Interpolate using other GCPs
                interp_lat, interp_lon, _ = temp_geo.pixel_to_latlon(true_row, true_col)

                # Calculate error in meters
                error_m = geographic_distance(true_lat, true_lon, interp_lat, interp_lon)
                errors_m.append(error_m)

            except ValueError:
                # Skip if outside convex hull
                continue

        if not errors_m:
            return {
                'mean_error_m': np.nan,
                'rms_error_m': np.nan,
                'max_error_m': np.nan
            }

        errors_array = np.array(errors_m)

        return {
            'mean_error_m': float(np.mean(errors_array)),
            'rms_error_m': float(np.sqrt(np.mean(errors_array**2))),
            'max_error_m': float(np.max(errors_array))
        }

    @classmethod
    def from_dict(cls, geo_info: Dict[str, Any], reader_metadata: Dict[str, Any]) -> 'GCPGeolocation':
        """
        Create GCPGeolocation from reader metadata dictionary.

        Parameters
        ----------
        geo_info : Dict[str, Any]
            Geolocation info dict. Must contain 'gcps' key.
        reader_metadata : Dict[str, Any]
            Reader metadata with 'rows' and 'cols'

        Returns
        -------
        GCPGeolocation
            Initialized geolocation object

        Raises
        ------
        KeyError
            If required keys are missing
        """
        gcps = geo_info['gcps']
        shape = (reader_metadata['rows'], reader_metadata['cols'])
        crs = geo_info.get('crs', 'WGS84')

        return cls(gcps, shape, crs)