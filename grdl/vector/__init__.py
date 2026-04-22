# -*- coding: utf-8 -*-
"""
Vector Sub-module - Generic geo-registered vector feature data.

Provides the ``FeatureSet`` data model for collections of geo-registered
features (points, lines, polygons) with properties, the ``VectorProcessor``
ABC for processors that operate on feature sets, and spatial operators
for common vector operations.

Key Classes
-----------
Data models:
    ``Feature``, ``FieldSchema``, ``FeatureSet``

Processing:
    ``VectorProcessor`` (ABC)

Spatial operators:
    ``BufferOperator``, ``IntersectionOperator``, ``UnionOperator``,
    ``DissolveOperator``, ``SpatialJoinOperator``, ``ClipOperator``,
    ``CentroidOperator``, ``ConvexHullOperator``

Raster-vector conversion:
    ``RasterToPoints``, ``Rasterize``

I/O:
    ``VectorReader``, ``VectorWriter``

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
2026-03-25
"""

from grdl.vector.models import Feature, FieldSchema, FeatureSet
from grdl.vector.base import VectorProcessor
from grdl.vector.spatial import (
    BufferOperator,
    IntersectionOperator,
    UnionOperator,
    DissolveOperator,
    SpatialJoinOperator,
    ClipOperator,
    CentroidOperator,
    ConvexHullOperator,
)
from grdl.vector.conversion import RasterToPoints, Rasterize
from grdl.vector.io import VectorReader, VectorWriter

__all__ = [
    # Data models
    'Feature',
    'FieldSchema',
    'FeatureSet',
    # Processing
    'VectorProcessor',
    # Spatial operators
    'BufferOperator',
    'IntersectionOperator',
    'UnionOperator',
    'DissolveOperator',
    'SpatialJoinOperator',
    'ClipOperator',
    'CentroidOperator',
    'ConvexHullOperator',
    # Conversion
    'RasterToPoints',
    'Rasterize',
    # I/O
    'VectorReader',
    'VectorWriter',
]
