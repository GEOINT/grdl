# -*- coding: utf-8 -*-
"""
Detection Sub-module - Sparse geo-registered vector detections.

Provides the ``ImageDetector`` ABC for processors that produce sparse
vector detections (points, bounding boxes, polygons) and the data models
for representing detection output.

Key Classes
-----------
- ImageDetector: ABC for sparse vector detectors
- Detection: Single geo-registered detection
- DetectionSet: Collection of detections from a detector run
- Geometry: Spatial location in pixel and geographic space
- OutputField: Declaration of a single output property
- OutputSchema: Self-declared output format for a detector

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
2026-02-06

Modified
--------
2026-02-06
"""

from grdl.image_processing.detection.base import ImageDetector
from grdl.image_processing.detection.models import (
    Detection,
    DetectionSet,
    Geometry,
    OutputField,
    OutputSchema,
)

__all__ = [
    'ImageDetector',
    'Detection',
    'DetectionSet',
    'Geometry',
    'OutputField',
    'OutputSchema',
]
