# -*- coding: utf-8 -*-
"""
Detection Sub-module - Sparse geo-registered vector detections.

Provides the ``ImageDetector`` ABC for processors that produce sparse
vector detections (points, bounding boxes, polygons) and the data models
for representing detection output.  Geometry is represented using
``shapely.geometry`` objects directly on ``Detection``.

Key Classes
-----------
- ImageDetector: ABC for sparse vector detectors
- Detection: Single geo-registered detection with shapely geometry
- DetectionSet: Collection of detections from a detector run
- FieldDefinition: Definition of a data dictionary field
- Fields: Accessor class for standard field name constants
- DATA_DICTIONARY: Registry of standardized field definitions

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
2026-02-11
"""

from grdl.image_processing.detection.base import ImageDetector
from grdl.image_processing.detection.models import (
    Detection,
    DetectionSet,
)
from grdl.image_processing.detection.fields import (
    DATA_DICTIONARY,
    FieldDefinition,
    Fields,
    is_dictionary_field,
    list_fields,
    lookup_field,
)

__all__ = [
    'ImageDetector',
    'Detection',
    'DetectionSet',
    'FieldDefinition',
    'Fields',
    'DATA_DICTIONARY',
    'lookup_field',
    'is_dictionary_field',
    'list_fields',
]
