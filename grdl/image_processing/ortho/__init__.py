# -*- coding: utf-8 -*-
"""
Orthorectification Sub-module - Geometric reprojection to geographic grids.

Provides orthorectification for imagery in native acquisition geometry
(SAR slant range, oblique EO, etc.) to regular ground-projected grids
in either WGS-84 geographic coordinates or local ENU (East-North-Up)
meters.

``orthorectify()`` is the recommended entry point — a keyword-argument
function for configuring source data, geolocation, resolution (explicit
or auto-computed from metadata), geographic ROI restriction, ENU output
mode, and memory-efficient tiled processing.  DEM terrain correction is
handled by attaching the DEM to the geolocation object
(``geo.elevation = dem``) before calling ``orthorectify()``.
``OrthoBuilder`` provides the underlying fluent builder for advanced use.

Resampling is accelerated via a multi-backend dispatch chain:
numba (JIT parallel) > torch GPU > torch CPU > scipy parallel > scipy.
The ``resample()`` function and ``detect_backend()`` are exposed for
direct use outside the pipeline.

Key Functions / Classes
-----------------------
orthorectify
    Keyword-argument function (recommended).  Wraps ``OrthoBuilder``
    with all parameters as kwargs.
OrthoBuilder
    Fluent builder for advanced use (partial configuration, reuse).
    Accepts source arrays or readers, resolves output grid, and
    dispatches to full-grid or tiled execution.  DEM terrain correction
    is configured on the geolocation object, not on the builder.
OrthoResult
    Output container holding orthorectified data, output grid, and
    geolocation metadata for downstream writers.
Orthorectifier
    Low-level inverse-geolocation mapper + resampler.  Computes the
    source-pixel coordinate mapping for any ``OutputGridProtocol`` grid
    and applies it via the accelerated resampling backend.  The mapping
    core (``_compute_strip``) is shared between sequential and parallel
    paths; grids >1M pixels are automatically parallelised across threads.
OutputGridProtocol
    ``@runtime_checkable`` ``Protocol`` defining the grid contract:
    ``rows``, ``cols``, ``image_to_latlon()``, ``latlon_to_image()``,
    ``sub_grid()``.  ``GeographicGrid``, ``ENUGrid``, ``UTMGrid``, and
    ``WebMercatorGrid`` satisfy it.  Custom grids are accepted by
    ``Orthorectifier`` if they implement this protocol.
GeographicGrid
    WGS-84 geographic grid specification (bounds, pixel sizes, row/col
    counts).  Satisfies ``OutputGridProtocol``.  Supports
    ``from_geolocation()`` construction and ``sub_grid()`` extraction
    for tiled processing.
ENUGrid
    Local East-North-Up grid specification in meters, centered on a
    WGS-84 reference point.  Satisfies ``OutputGridProtocol`` —
    provides ``image_to_latlon()``, ``latlon_to_image()``, and
    ``sub_grid()``.
validate_sub_grid_indices
    Shared bounds-checking helper used by ``GeographicGrid.sub_grid()``
    and ``ENUGrid.sub_grid()`` to validate tile indices.
compute_output_resolution
    Auto-compute output pixel size in degrees from sensor metadata.
    Dispatches on metadata type: SICD, BIOMASS, Sentinel-1 SLC, NISAR,
    GeoTIFF.
resample
    Unified accelerated resampling function.  Auto-dispatches to the
    fastest available backend (numba, torch, scipy).
detect_backend
    Return the name of the best available resampling backend.
orthorectify_point_roi
    Single-call helper: flat-plane ENU ortho centered on a lat/lon or
    pixel point, for any GRDL-supported modality.  Handles geolocation
    auto-detection, chip extraction, complex-mode preprocessing, and
    coverage validation internally.
PointRoiResult
    Result container returned by ``orthorectify_point_roi``.  Carries
    the ortho array, the source chip, center coordinates, and chip
    bounds.

Dependencies
------------
scipy
numba (optional — JIT parallel acceleration)
torch (optional — GPU/CPU acceleration)

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
2026-03-27
"""

from grdl.image_processing.ortho.ortho import (
    Orthorectifier,
    GeographicGrid,
    OutputGrid,
    OutputGridProtocol,
    validate_sub_grid_indices,
)
from grdl.image_processing.ortho.enu_grid import ENUGrid
from grdl.image_processing.ortho.ortho_builder import (
    OrthoBuilder,
    OrthoResult,
    orthorectify,
)
from grdl.image_processing.ortho.resolution import compute_output_resolution
from grdl.image_processing.ortho.accelerated import resample, detect_backend
from grdl.image_processing.ortho.roi import PointRoiResult, orthorectify_point_roi

try:
    from grdl.image_processing.ortho.utm_grid import UTMGrid
except ImportError:
    pass

try:
    from grdl.image_processing.ortho.web_mercator_grid import WebMercatorGrid
except ImportError:
    pass

__all__ = [
    'Orthorectifier',
    'GeographicGrid',
    'OutputGrid',
    'OutputGridProtocol',
    'validate_sub_grid_indices',
    'ENUGrid',
    'UTMGrid',
    'WebMercatorGrid',
    'OrthoBuilder',
    'OrthoResult',
    'orthorectify',
    'compute_output_resolution',
    'resample',
    'detect_backend',
    'PointRoiResult',
    'orthorectify_point_roi',
]
