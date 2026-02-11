# -*- coding: utf-8 -*-
"""
Constant Elevation Model - Returns a fixed height for all locations.

Provides a trivial ElevationModel implementation that returns the same
height value for every query point. Useful as a default fallback when no
DEM data is available, or for testing and flat-terrain scenarios.

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

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.elevation.base import ElevationModel


class ConstantElevation(ElevationModel):
    """Elevation model that returns a fixed height for all locations.

    This is the simplest ElevationModel implementation. It ignores
    geographic coordinates and returns the same constant height for
    every query point.

    Parameters
    ----------
    height : float, optional
        Constant height in meters to return for all queries. Default is 0.0.

    Examples
    --------
    >>> from grdl.geolocation.elevation.constant import ConstantElevation
    >>> elev = ConstantElevation(height=100.0)
    >>> elev.get_elevation(34.0, -118.0)
    100.0
    >>> import numpy as np
    >>> elev.get_elevation(np.array([34.0, 35.0]), np.array([-118.0, -117.0]))
    array([100., 100.])
    """

    def __init__(self, height: float = 0.0) -> None:
        """Initialize constant elevation model.

        Parameters
        ----------
        height : float, optional
            Constant height in meters. Default is 0.0.
        """
        super().__init__(dem_path=None, geoid_path=None)
        self._height = float(height)

    def _get_elevation_array(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> np.ndarray:
        """Return constant height for all query points.

        Parameters
        ----------
        lats : np.ndarray
            Latitude values (ignored). Shape ``(N,)``.
        lons : np.ndarray
            Longitude values (ignored). Shape ``(N,)``.

        Returns
        -------
        np.ndarray
            Array of shape ``(N,)`` filled with the constant height.
        """
        return np.full(lats.shape, self._height, dtype=np.float64)
