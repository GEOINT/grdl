# -*- coding: utf-8 -*-
"""
Orthorectification Sub-module - Geometric reprojection to geographic grids.

Provides orthorectification for imagery in native acquisition geometry
(SAR slant range, oblique EO, etc.) to regular ground-projected grids.

``OrthoPipeline`` is the recommended entry point.  It provides a fluent
builder API for configuring source data, geolocation, resolution (explicit
or auto-computed from metadata), DEM terrain correction, geographic ROI
restriction, and memory-efficient tiled processing.

Key Classes
-----------
OrthoPipeline
    Builder-pattern orchestrator.  Accepts source arrays or readers,
    resolves output grid (from geolocation footprint, explicit bounds, or
    ROI), and dispatches to full-grid or tiled execution.
OrthoResult
    Output container holding orthorectified data, output grid, and
    geolocation metadata for downstream writers.
Orthorectifier
    Low-level inverse-geolocation mapper + resampler.  Computes the
    source-pixel coordinate mapping for an ``OutputGrid`` and applies it
    via ``scipy.ndimage.map_coordinates``.  Used internally by the
    pipeline; can also be used directly for custom workflows.
OutputGrid
    Geographic grid specification (bounds, pixel sizes, row/col counts).
    Supports ``from_geolocation()`` construction and ``sub_grid()``
    extraction for tiled processing.
compute_output_resolution
    Auto-compute output pixel size in degrees from sensor metadata.
    Dispatches on metadata type: SICD (sample spacing + graze angle),
    BIOMASS (range/azimuth pixel spacing).

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
2026-02-18
"""

from grdl.image_processing.ortho.ortho import Orthorectifier, OutputGrid
from grdl.image_processing.ortho.ortho_pipeline import OrthoPipeline, OrthoResult
from grdl.image_processing.ortho.resolution import compute_output_resolution

__all__ = [
    'Orthorectifier',
    'OutputGrid',
    'OrthoPipeline',
    'OrthoResult',
    'compute_output_resolution',
]
