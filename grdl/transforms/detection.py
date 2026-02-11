# -*- coding: utf-8 -*-
"""
Detection Transforms - Apply co-registration transforms to detection geometries.

Provides functions to transform detection coordinates (points, bounding
boxes, polygons) from a moving image's pixel space to a fixed image's
pixel space using a co-registration transform.  This enables a
detect-then-transform workflow that preserves original image fidelity
by avoiding raster interpolation.

This module bridges the coregistration and detection domains without
creating a cross-dependency between them.

Coordinate Conventions
----------------------
- Detection pixel geometries use shapely convention: ``(x, y) = (col, row)``.
- ``apply_transform_to_points`` uses ``(row, col)`` convention.
- This module handles the conversion internally.

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
from shapely.geometry import Point, Polygon, box

# GRDL internal
from grdl.coregistration.base import RegistrationResult
from grdl.coregistration.utils import apply_transform_to_points
from grdl.image_processing.detection.models import (
    Detection,
    DetectionSet,
)


def transform_pixel_geometry(
    pixel_geometry,
    result: RegistrationResult,
    inverse: bool = False,
    bbox_mode: str = 'refit',
):
    """Transform a shapely pixel geometry using a registration result.

    Applies the forward (moving -> fixed) or inverse (fixed -> moving)
    transform to a shapely geometry in pixel space.

    Parameters
    ----------
    pixel_geometry : shapely.geometry.base.BaseGeometry
        Pixel-space geometry.  Convention: ``(x, y) = (col, row)``.
    result : RegistrationResult
        Registration result containing the transform matrix.
    inverse : bool
        If False (default), apply the forward transform
        (moving -> fixed).  If True, apply the inverse transform
        (fixed -> moving).
    bbox_mode : str
        Strategy for rectangular (box) Polygon geometries:

        - ``'refit'`` (default): Transform all corners, then
          compute the minimal axis-aligned bounding box.
        - ``'polygon'``: Transform all corners and return as a
          Polygon (preserves exact transformed shape).

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        New geometry with transformed coordinates.

    Raises
    ------
    ValueError
        If ``bbox_mode`` is not ``'refit'`` or ``'polygon'``.
    """
    if bbox_mode not in ('refit', 'polygon'):
        raise ValueError(
            f"bbox_mode must be 'refit' or 'polygon', got {bbox_mode!r}"
        )

    matrix = (
        result.inverse_transform_matrix if inverse
        else result.transform_matrix
    )

    geom_type = pixel_geometry.geom_type

    if geom_type == 'Point':
        return _transform_point(pixel_geometry, matrix)
    elif geom_type == 'Polygon':
        return _transform_polygon(pixel_geometry, matrix, bbox_mode)
    else:
        raise ValueError(
            f"Unsupported geometry type: {geom_type!r}. "
            f"Expected 'Point' or 'Polygon'."
        )


def transform_detection(
    detection: Detection,
    result: RegistrationResult,
    inverse: bool = False,
    bbox_mode: str = 'refit',
) -> Detection:
    """Transform a Detection's pixel geometry using a registration result.

    Returns a new Detection with transformed pixel coordinates.  All
    other attributes (properties, confidence, geo_geometry) are preserved.

    Parameters
    ----------
    detection : Detection
        Source detection to transform.
    result : RegistrationResult
        Registration result containing the transform matrix.
    inverse : bool
        If False (default), forward transform.  If True, inverse.
    bbox_mode : str
        Box handling strategy.  See ``transform_pixel_geometry``.

    Returns
    -------
    Detection
        New Detection with transformed pixel geometry.
    """
    new_pixel_geom = transform_pixel_geometry(
        detection.pixel_geometry, result,
        inverse=inverse, bbox_mode=bbox_mode,
    )
    return Detection(
        pixel_geometry=new_pixel_geom,
        properties=dict(detection.properties),
        confidence=detection.confidence,
        geo_geometry=detection.geo_geometry,
    )


def transform_detection_set(
    detection_set: DetectionSet,
    result: RegistrationResult,
    inverse: bool = False,
    bbox_mode: str = 'refit',
) -> DetectionSet:
    """Transform all detections in a DetectionSet.

    Returns a new DetectionSet with all detection pixel coordinates
    transformed.  Metadata is preserved with an added note about the
    applied transform.

    Parameters
    ----------
    detection_set : DetectionSet
        Source detection set to transform.
    result : RegistrationResult
        Registration result containing the transform matrix.
    inverse : bool
        If False (default), forward transform.  If True, inverse.
    bbox_mode : str
        Box handling strategy.  See ``transform_pixel_geometry``.

    Returns
    -------
    DetectionSet
        New DetectionSet with transformed detection geometries.
    """
    transformed = [
        transform_detection(d, result, inverse=inverse, bbox_mode=bbox_mode)
        for d in detection_set
    ]
    metadata = dict(detection_set.metadata)
    metadata['coordinate_transform_applied'] = True
    metadata['transform_direction'] = 'inverse' if inverse else 'forward'

    return DetectionSet(
        detections=transformed,
        detector_name=detection_set.detector_name,
        detector_version=detection_set.detector_version,
        output_fields=detection_set.output_fields,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _shapely_to_rowcol(coords: np.ndarray) -> np.ndarray:
    """Convert shapely (col, row) coords to (row, col) for transforms.

    Parameters
    ----------
    coords : np.ndarray
        Shape (N, 2) in (col, row) order (shapely convention).

    Returns
    -------
    np.ndarray
        Shape (N, 2) in (row, col) order (transform convention).
    """
    return coords[:, ::-1].copy()


def _rowcol_to_shapely(coords: np.ndarray) -> np.ndarray:
    """Convert (row, col) transform coords to shapely (col, row).

    Parameters
    ----------
    coords : np.ndarray
        Shape (N, 2) in (row, col) order.

    Returns
    -------
    np.ndarray
        Shape (N, 2) in (col, row) order (shapely convention).
    """
    return coords[:, ::-1].copy()


def _transform_point(geom, matrix: np.ndarray):
    """Transform a Point geometry."""
    # shapely Point: (x, y) = (col, row)
    rowcol = np.array([[geom.y, geom.x]], dtype=np.float64)
    transformed = apply_transform_to_points(rowcol, matrix)
    new_row, new_col = transformed[0]
    return Point(new_col, new_row)


def _transform_polygon(geom, matrix: np.ndarray, bbox_mode: str):
    """Transform a Polygon geometry.

    If the polygon is an axis-aligned rectangle and bbox_mode is 'refit',
    the result is a new axis-aligned bounding box enclosing the
    transformed corners.
    """
    # Get exterior ring coordinates (shapely closes the ring)
    coords_xy = np.array(geom.exterior.coords)  # (N+1, 2), (col, row)
    # Drop closing vertex for transform
    coords_xy = coords_xy[:-1]

    # Convert to (row, col) for transform
    coords_rc = _shapely_to_rowcol(coords_xy)
    transformed_rc = apply_transform_to_points(coords_rc, matrix)

    if bbox_mode == 'refit' and _is_axis_aligned_rect(coords_xy):
        # Re-fit axis-aligned bounding box
        transformed_xy = _rowcol_to_shapely(transformed_rc)
        min_col = float(np.min(transformed_xy[:, 0]))
        min_row = float(np.min(transformed_xy[:, 1]))
        max_col = float(np.max(transformed_xy[:, 0]))
        max_row = float(np.max(transformed_xy[:, 1]))
        return box(min_col, min_row, max_col, max_row)

    # Return as general polygon
    transformed_xy = _rowcol_to_shapely(transformed_rc)
    ring = list(map(tuple, transformed_xy))
    ring.append(ring[0])  # close the ring
    return Polygon(ring)


def _is_axis_aligned_rect(coords_xy: np.ndarray) -> bool:
    """Check if coordinates form an axis-aligned rectangle.

    Parameters
    ----------
    coords_xy : np.ndarray
        Vertices in (col, row) order, shape (N, 2), without closing vertex.

    Returns
    -------
    bool
        True if the 4 vertices form an axis-aligned rectangle.
    """
    if len(coords_xy) != 4:
        return False
    xs = coords_xy[:, 0]
    ys = coords_xy[:, 1]
    return len(set(np.round(xs, 10))) == 2 and len(set(np.round(ys, 10))) == 2
