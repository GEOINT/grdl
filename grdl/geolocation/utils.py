# -*- coding: utf-8 -*-
"""
Geolocation Utilities - Helper functions for coordinate transformations.

Utility functions for geolocation operations including footprint calculation,
bounds computation, and coordinate system conversions.

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


def calculate_footprint(
    corner_coords: List[Tuple[float, float]]
) -> Dict[str, Any]:
    """
    Calculate footprint polygon and bounding box from corner coordinates.

    Parameters
    ----------
    corner_coords : List[Tuple[float, float]]
        List of (lon, lat) tuples for polygon corners

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'type': 'Polygon'
        - 'coordinates': List of (lon, lat) tuples
        - 'bounds': (min_lon, min_lat, max_lon, max_lat)
    """
    if not corner_coords or len(corner_coords) < 3:
        return {
            'type': 'None',
            'coordinates': None,
            'bounds': None
        }

    # Convert to numpy array for vectorized operations
    coords_array = np.array(corner_coords)
    lons = coords_array[:, 0]
    lats = coords_array[:, 1]

    # Calculate bounds using vectorized min/max
    min_lon, max_lon = float(np.min(lons)), float(np.max(lons))
    min_lat, max_lat = float(np.min(lats)), float(np.max(lats))

    return {
        'type': 'Polygon',
        'coordinates': corner_coords,
        'bounds': (min_lon, min_lat, max_lon, max_lat)
    }


def bounds_from_corners(
    corners: List[Tuple[float, float]]
) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box from corner coordinates.

    Parameters
    ----------
    corners : List[Tuple[float, float]]
        List of (lon, lat) tuples

    Returns
    -------
    Tuple[float, float, float, float]
        (min_lon, min_lat, max_lon, max_lat) bounding box
    """
    if not corners:
        raise ValueError("No corner coordinates provided")

    # Convert to numpy array for vectorized operations
    corners_array = np.array(corners)
    lons = corners_array[:, 0]
    lats = corners_array[:, 1]

    return (float(np.min(lons)), float(np.min(lats)),
            float(np.max(lons)), float(np.max(lats)))


def check_pixel_bounds(
    row: float,
    col: float,
    shape: Tuple[int, int],
    tolerance: float = 0.5
) -> None:
    """
    Check if pixel coordinates are within image bounds.

    Parameters
    ----------
    row : float
        Row coordinate
    col : float
        Column coordinate
    shape : Tuple[int, int]
        Image shape (rows, cols)
    tolerance : float, default=0.5
        Tolerance for out-of-bounds check (pixels)

    Raises
    ------
    ValueError
        If coordinates are outside image bounds (with tolerance)
    """
    rows, cols = shape

    if row < -tolerance or row >= rows + tolerance:
        raise ValueError(
            f"Row coordinate {row} is outside image bounds [0, {rows-1}]"
        )

    if col < -tolerance or col >= cols + tolerance:
        raise ValueError(
            f"Column coordinate {col} is outside image bounds [0, {cols-1}]"
        )


def sample_image_perimeter(
    shape: Tuple[int, int],
    samples_per_edge: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample points along image perimeter for footprint calculation.

    Parameters
    ----------
    shape : Tuple[int, int]
        Image shape (rows, cols)
    samples_per_edge : int, default=10
        Number of sample points per edge

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (rows, cols) arrays of sample coordinates along perimeter
    """
    rows, cols = shape

    # Top edge (row=0, col varies)
    top_rows = np.zeros(samples_per_edge)
    top_cols = np.linspace(0, cols-1, samples_per_edge)

    # Right edge (col=cols-1, row varies)
    right_rows = np.linspace(0, rows-1, samples_per_edge)
    right_cols = np.full(samples_per_edge, cols-1)

    # Bottom edge (row=rows-1, col varies)
    bottom_rows = np.full(samples_per_edge, rows-1)
    bottom_cols = np.linspace(cols-1, 0, samples_per_edge)

    # Left edge (col=0, row varies)
    left_rows = np.linspace(rows-1, 0, samples_per_edge)
    left_cols = np.zeros(samples_per_edge)

    # Concatenate all edges
    all_rows = np.concatenate([top_rows, right_rows, bottom_rows, left_rows])
    all_cols = np.concatenate([top_cols, right_cols, bottom_cols, left_cols])

    return all_rows, all_cols


def interpolation_error_metrics(
    true_values: np.ndarray,
    interpolated_values: np.ndarray
) -> Dict[str, float]:
    """
    Calculate error metrics for interpolation accuracy.

    Parameters
    ----------
    true_values : np.ndarray
        Ground truth values
    interpolated_values : np.ndarray
        Interpolated values

    Returns
    -------
    Dict[str, float]
        Dictionary with error metrics:
        - 'mean_error': Mean absolute error
        - 'rms_error': Root mean square error
        - 'max_error': Maximum absolute error
        - 'std_error': Standard deviation of errors
    """
    errors = np.abs(true_values - interpolated_values)

    return {
        'mean_error': float(np.mean(errors)),
        'rms_error': float(np.sqrt(np.mean(errors**2))),
        'max_error': float(np.max(errors)),
        'std_error': float(np.std(errors))
    }


def geographic_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculate great circle distance between two geographic coordinates.

    Uses the Haversine formula to calculate distance on a sphere.

    Parameters
    ----------
    lat1, lon1 : float
        First point (latitude, longitude) in degrees
    lat2, lon2 : float
        Second point (latitude, longitude) in degrees

    Returns
    -------
    float
        Distance in meters

    Notes
    -----
    Assumes Earth radius of 6371 km. Accuracy decreases for very short
    distances or near the poles. For batch operations on arrays, use
    geographic_distance_batch() for better performance.
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Earth radius in meters
    r = 6371000

    return float(c * r)


def geographic_distance_batch(
    lats1: np.ndarray,
    lons1: np.ndarray,
    lats2: np.ndarray,
    lons2: np.ndarray
) -> np.ndarray:
    """
    Calculate great circle distances between arrays of geographic coordinates.

    Vectorized implementation of Haversine formula for batch operations.

    Parameters
    ----------
    lats1, lons1 : np.ndarray
        First points (latitude, longitude) in degrees
    lats2, lons2 : np.ndarray
        Second points (latitude, longitude) in degrees

    Returns
    -------
    np.ndarray
        Distances in meters (same shape as input arrays)

    Notes
    -----
    All input arrays must have the same shape. Vectorized operations provide
    significant speedup over looping with geographic_distance().
    """
    # Convert to radians (vectorized)
    lats1_rad = np.radians(lats1)
    lons1_rad = np.radians(lons1)
    lats2_rad = np.radians(lats2)
    lons2_rad = np.radians(lons2)

    # Haversine formula (vectorized)
    dlat = lats2_rad - lats1_rad
    dlon = lons2_rad - lons1_rad

    a = np.sin(dlat/2)**2 + np.cos(lats1_rad) * np.cos(lats2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Earth radius in meters
    r = 6371000

    return c * r