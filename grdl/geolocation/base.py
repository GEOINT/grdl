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
2026-01-30
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from grdl.IO.base import ImageReader


class Geolocation(ABC):
    """
    Abstract base class for geolocation transformations.

    Provides interface for transforming between image pixel coordinates and
    geographic coordinates. Concrete implementations handle different coordinate
    systems and geometries (geocoded raster, SAR slant range, etc.).

    Coordinate Conventions
    ----------------------
    - **Image coordinates:** (row, col) with (0, 0) at top-left corner
    - **Geographic coordinates:** (lat, lon, height) in WGS84 (EPSG:4326)
    - **Height:** Above WGS84 ellipsoid (not geoid)

    Notes
    -----
    All implementations should transform to/from WGS84 geographic coordinates
    for consistency, even if the imagery uses a different native CRS.
    """

    def __init__(self, shape: Tuple[int, int], crs: str = 'WGS84'):
        """
        Initialize geolocation.

        Parameters
        ----------
        shape : Tuple[int, int]
            Image shape (rows, cols)
        crs : str, default='WGS84'
            Coordinate reference system
        """
        self.shape = shape
        self.crs = crs

    @abstractmethod
    def pixel_to_latlon(
        self,
        row: float,
        col: float,
        height: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Transform pixel coordinates to geographic coordinates.

        Parameters
        ----------
        row : float
            Row coordinate (0-based, top-down)
        col : float
            Column coordinate (0-based, left-right)
        height : float, default=0.0
            Height above WGS84 ellipsoid (meters)

        Returns
        -------
        Tuple[float, float, float]
            (latitude, longitude, height) in WGS84 coordinates
            - latitude: degrees North (-90 to +90)
            - longitude: degrees East (-180 to +180)
            - height: meters above WGS84 ellipsoid

        Raises
        ------
        ValueError
            If pixel coordinates are out of image bounds
        NotImplementedError
            If geolocation is not available
        """
        pass

    @abstractmethod
    def latlon_to_pixel(
        self,
        lat: float,
        lon: float,
        height: float = 0.0
    ) -> Tuple[float, float]:
        """
        Transform geographic coordinates to pixel coordinates.

        Parameters
        ----------
        lat : float
            Latitude in degrees North (-90 to +90)
        lon : float
            Longitude in degrees East (-180 to +180)
        height : float, default=0.0
            Height above WGS84 ellipsoid (meters)

        Returns
        -------
        Tuple[float, float]
            (row, col) pixel coordinates
            - row: row coordinate (0-based, may be fractional)
            - col: column coordinate (0-based, may be fractional)

        Raises
        ------
        ValueError
            If geographic coordinates are outside image footprint
        NotImplementedError
            If geolocation is not available
        """
        pass

    def pixel_to_latlon_batch(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform multiple pixel coordinates to geographic coordinates.

        Parameters
        ----------
        rows : np.ndarray
            Row coordinates (1D array)
        cols : np.ndarray
            Column coordinates (1D array)
        height : float, default=0.0
            Height above WGS84 ellipsoid (meters)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (lats, lons, heights) arrays in WGS84 coordinates

        Notes
        -----
        Default implementation loops over individual coordinates.
        Subclasses should override for vectorized operations if available.
        """
        lats = np.zeros_like(rows, dtype=float)
        lons = np.zeros_like(rows, dtype=float)
        heights = np.zeros_like(rows, dtype=float)

        for i in range(len(rows)):
            lats[i], lons[i], heights[i] = self.pixel_to_latlon(rows[i], cols[i], height)

        return lats, lons, heights

    def latlon_to_pixel_batch(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform multiple geographic coordinates to pixel coordinates.

        Parameters
        ----------
        lats : np.ndarray
            Latitudes in degrees North (1D array)
        lons : np.ndarray
            Longitudes in degrees East (1D array)
        height : float, default=0.0
            Height above WGS84 ellipsoid (meters)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (rows, cols) pixel coordinate arrays

        Notes
        -----
        Default implementation loops over individual coordinates.
        Subclasses should override for vectorized operations if available.
        """
        rows = np.zeros_like(lats, dtype=float)
        cols = np.zeros_like(lats, dtype=float)

        for i in range(len(lats)):
            rows[i], cols[i] = self.latlon_to_pixel(lats[i], lons[i], height)

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

            lats, lons, _ = self.pixel_to_latlon_batch(sample_rows, sample_cols)

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

    @classmethod
    def from_reader(cls, reader: 'ImageReader') -> 'Geolocation':
        """
        Factory method to create geolocation object from imagery reader.

        Parameters
        ----------
        reader : ImageReader
            Imagery reader with geolocation metadata

        Returns
        -------
        Geolocation
            Appropriate geolocation subclass based on reader geometry

        Notes
        -----
        Factory detects geometry type from reader metadata and returns:
        - GCPGeolocation: for SAR slant range with GCPs (BIOMASS L1)
        - SICDGeolocation: for SICD imagery with projection model
        - SARGeolocation/EOGeolocation: for geocoded imagery with affine transform
        - NoGeolocation: fallback when no geolocation available
        """
        geo_info = reader.get_geolocation()

        if geo_info is None:
            return NoGeolocation(reader.get_shape()[:2])

        # Detect geometry type from projection and available metadata
        projection = geo_info.get('projection', '').lower()

        # Determine modality (SAR vs EO) from reader module
        reader_module = reader.__class__.__module__.lower()
        is_sar = ('sar' in reader_module or
                  'biomass' in reader_module or
                  'sicd' in reader_module or
                  'cphd' in reader_module)

        # Route to appropriate implementation
        if 'slant' in projection and 'gcps' in geo_info:
            # SAR slant range with GCPs (BIOMASS)
            from grdl.geolocation.sar.gcp import GCPGeolocation
            return GCPGeolocation.from_dict(geo_info, reader.metadata)

        elif 'sicd' in projection or 'sicd' in reader_module:
            # SICD with SARPY projection model (not yet implemented)
            raise NotImplementedError(
                "SICD geolocation is not yet implemented. "
                "See grdl/geolocation/sar/ for available implementations."
            )

        elif 'transform' in geo_info or 'affine' in str(type(geo_info.get('transform', ''))):
            # Affine transform geolocation (not yet implemented)
            raise NotImplementedError(
                "Affine transform geolocation is not yet implemented. "
                "See grdl/geolocation/sar/ for available implementations."
            )

        else:
            # No geolocation available
            return NoGeolocation(reader.get_shape()[:2])


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

    def pixel_to_latlon(
        self,
        row: float,
        col: float,
        height: float = 0.0
    ) -> Tuple[float, float, float]:
        """Raise NotImplementedError - no geolocation available."""
        raise NotImplementedError(
            "This imagery has no geolocation information. "
            "Cannot transform pixel coordinates to lat/lon."
        )

    def latlon_to_pixel(
        self,
        lat: float,
        lon: float,
        height: float = 0.0
    ) -> Tuple[float, float]:
        """Raise NotImplementedError - no geolocation available."""
        raise NotImplementedError(
            "This imagery has no geolocation information. "
            "Cannot transform lat/lon to pixel coordinates."
        )

    def get_footprint(self) -> Dict[str, Any]:
        """Return empty footprint."""
        return {
            'type': 'None',
            'coordinates': None,
            'bounds': None
        }