# -*- coding: utf-8 -*-
"""
Elevation Model Base Class - Abstract interface for terrain elevation lookup.

Defines the abstract base class for all elevation models (DEM, DTED, constant).
Concrete implementations handle different elevation data formats and sources.
The public ``get_elevation`` method supports the same scalar/array dispatch
pattern used throughout the geolocation module.

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
2026-02-11

Modified
--------
2026-02-11
"""

# Standard library
from abc import ABC, abstractmethod
from typing import Optional, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.base import _is_scalar, _to_array


class ElevationModel(ABC):
    """Abstract base class for terrain elevation lookup.

    Provides a unified interface for querying terrain heights from
    DEM/DTED data. Concrete subclasses implement ``_get_elevation_array``
    for vectorized height lookup. The public ``get_elevation`` method
    handles scalar/array dispatch and optional geoid correction.

    When a ``geoid_path`` is provided, the model automatically applies
    geoid undulation correction to convert MSL heights to HAE heights.

    Parameters
    ----------
    dem_path : str or None, optional
        Path to DEM/DTED data source. Interpretation depends on the
        concrete subclass (file path, directory path, etc.).
    geoid_path : str or None, optional
        Path to a geoid model file (e.g., EGM96 .pgm grid). When provided,
        a ``GeoidCorrection`` instance is created and undulation values are
        added to MSL heights to produce HAE heights.

    Attributes
    ----------
    dem_path : str or None
        Path to DEM data source.
    geoid_path : str or None
        Path to geoid model file.

    Notes
    -----
    Subclasses implement ``_get_elevation_array`` which operates on 1D
    numpy arrays. The public method handles scalar/list/array dispatch
    automatically.

    Coordinate Conventions
    ----------------------
    - **Heights without geoid:** Mean Sea Level (MSL) as stored in the DEM.
    - **Heights with geoid:** Height Above Ellipsoid (HAE) = MSL + undulation.
    - **Latitude:** Degrees North, range [-90, 90].
    - **Longitude:** Degrees East, range [-180, 180].
    """

    def __init__(
        self,
        dem_path: Optional[str] = None,
        geoid_path: Optional[str] = None,
    ) -> None:
        """Initialize elevation model.

        Parameters
        ----------
        dem_path : str or None, optional
            Path to DEM/DTED data source.
        geoid_path : str or None, optional
            Path to geoid model file for MSL-to-HAE correction.
        """
        self.dem_path = dem_path
        self.geoid_path = geoid_path
        self._geoid = None

        if geoid_path is not None:
            from grdl.geolocation.elevation.geoid import GeoidCorrection
            self._geoid = GeoidCorrection(geoid_path)

    @abstractmethod
    def _get_elevation_array(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> np.ndarray:
        """Look up terrain elevation for arrays of coordinates.

        Subclasses implement this single vectorized method. Returns heights
        in meters -- MSL if no geoid correction, raw DEM values otherwise.

        Parameters
        ----------
        lats : np.ndarray
            Latitudes in degrees North. Shape ``(N,)``, dtype float64.
        lons : np.ndarray
            Longitudes in degrees East. Shape ``(N,)``, dtype float64.

        Returns
        -------
        np.ndarray
            Elevation values in meters. Shape ``(N,)``. NaN for points
            outside coverage area.
        """
        ...

    def get_elevation(
        self,
        lat_or_points: Union[float, list, np.ndarray],
        lon: Optional[Union[float, list, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """Query terrain elevation for one or more geographic locations.

        Accepts three input forms:

        - **Scalar:** ``get_elevation(lat, lon)`` returns a single float.
        - **Stacked (2, N) array:** ``get_elevation(points_2xN)`` returns
          an ``(N,)`` ndarray.
        - **Separate arrays:** ``get_elevation(lats_arr, lons_arr)`` returns
          an ndarray.

        When a geoid model is configured, MSL heights from the DEM are
        automatically converted to HAE:
        ``height_hae = height_msl + geoid_undulation``.

        Parameters
        ----------
        lat_or_points : float, list, or np.ndarray
            Latitude(s) when ``lon`` is provided, or a ``(2, N)`` ndarray
            of stacked ``[lats; lons]`` when ``lon`` is None.
        lon : float, list, or np.ndarray, optional
            Longitude(s). Omit to pass a ``(2, N)`` stacked array as the
            first argument.

        Returns
        -------
        float
            When scalar inputs are given.
        np.ndarray
            When array inputs are given. Shape ``(N,)``.

        Raises
        ------
        ValueError
            If a ``(2, N)`` array is expected but the shape is wrong.

        Examples
        --------
        Single point:

        >>> height = elev.get_elevation(34.05, -118.25)

        Separate arrays:

        >>> heights = elev.get_elevation(
        ...     np.array([34.0, 35.0]), np.array([-118.0, -117.0])
        ... )

        Stacked (2, N) array:

        >>> pts = np.array([[34.0, 35.0], [-118.0, -117.0]])
        >>> heights = elev.get_elevation(pts)
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
            heights = self._get_elevation_array(lats_arr, lons_arr)
            if self._geoid is not None:
                heights = heights + self._geoid.get_undulation(
                    lats_arr, lons_arr
                )
            return heights
        elif _is_scalar(lat_or_points) and _is_scalar(lon):
            # Scalar input
            lats_arr = _to_array(lat_or_points)
            lons_arr = _to_array(lon)
            heights = self._get_elevation_array(lats_arr, lons_arr)
            if self._geoid is not None:
                heights = heights + self._geoid.get_undulation(
                    lats_arr, lons_arr
                )
            return float(heights[0])
        else:
            # Separate array/list inputs
            lats_arr = _to_array(lat_or_points)
            lons_arr = _to_array(lon)
            heights = self._get_elevation_array(lats_arr, lons_arr)
            if self._geoid is not None:
                heights = heights + self._geoid.get_undulation(
                    lats_arr, lons_arr
                )
            return heights
