# -*- coding: utf-8 -*-
"""
grdl.shapes -- Geographic shapes as first-class analysis primitives.

Shapes (circles, ellipses, polygons, arcs) defined in lat/lon and
designed to project accurately into any GRDL geolocation (SICD R/Rdot,
RPC/RSM, Affine basemap) with per-vertex DEM sampling. Shapes can:

- Produce boolean pixel-space masks (:meth:`GeographicShape.rasterize`).
- Overlay on native imagery (:func:`overlay_shape`, :func:`burn_shape`).
- Cue detectors to a region of interest (:func:`cued_detect`).
- Combine via Gaussian error propagation or set operations
  (:func:`convolve_ellipses`, :func:`combine_evidence`,
  :func:`minkowski_sum`, :func:`union_shapes`, :func:`intersect_shapes`).

Accuracy-first design: perimeter generation uses Karney's geodesic
algorithm via :class:`pyproj.Geod`; adaptive refinement guarantees
sub-pixel fidelity to the true projected curve even under steep SAR
foreshortening or sloped terrain; combined ellipses propagate exact
covariance arithmetic.

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org

Created
-------
2026-04-18

Modified
--------
2026-04-18
"""

from grdl.shapes.arc import Arc
from grdl.shapes.backend import (
    ComputeBackend,
    detect_backend,
    set_backend,
    cupy_available,
    numba_available,
    cucim_available,
)
from grdl.shapes.base import (
    GeographicShape,
    to_pixels_batch,
    rasterize_batch,
)
from grdl.shapes.circle import Circle
from grdl.shapes.combine import (
    combine_evidence,
    convolve_ellipses,
    intersect_shapes,
    minkowski_sum,
    union_shapes,
)
from grdl.shapes.cueing import cued_detect
from grdl.shapes.display import burn_shape, overlay_shape, overlay_shapes
from grdl.shapes.ellipse import Ellipse, GeodesicEllipse
from grdl.shapes.polygon import GeoPolygon
from grdl.shapes.rasterize import rasterize_polygon


# Bind batch ops onto the backend module for discoverability
from grdl.shapes import backend as _backend_module
_backend_module.to_pixels_batch = to_pixels_batch
_backend_module.rasterize_batch = rasterize_batch


__all__ = [
    # Core ABC
    'GeographicShape',
    # Concrete shapes
    'Circle',
    'Ellipse',
    'GeodesicEllipse',
    'GeoPolygon',
    'Arc',
    # Combination / error propagation
    'convolve_ellipses',
    'combine_evidence',
    'minkowski_sum',
    'union_shapes',
    'intersect_shapes',
    # Display
    'overlay_shape',
    'overlay_shapes',
    'burn_shape',
    # Detection cueing
    'cued_detect',
    # Batch operations
    'to_pixels_batch',
    'rasterize_batch',
    # Rasterization primitive
    'rasterize_polygon',
    # Backend
    'ComputeBackend',
    'detect_backend',
    'set_backend',
    'cupy_available',
    'numba_available',
    'cucim_available',
]
