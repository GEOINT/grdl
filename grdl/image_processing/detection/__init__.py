# -*- coding: utf-8 -*-
"""
Detection Sub-module - Sparse geo-registered vector detections.

Provides the ``ImageDetector`` ABC for processors that produce sparse
vector detections (points, bounding boxes, polygons) and the data models
for representing detection output.  Geometry is represented using
``shapely.geometry`` objects directly on ``Detection``.

Includes the ``cfar/`` sub-package with four CFAR detector variants that
share a common template-method base (``CFARDetector``).

Key Classes
-----------
Base and data models:
    ``ImageDetector`` (ABC), ``Detection``, ``DetectionSet``,
    ``FieldDefinition``, ``Fields``, ``DATA_DICTIONARY``

CFAR detectors:
    ``CFARDetector`` (template-method ABC),
    ``CACFARDetector`` (cell-averaging),
    ``GOCFARDetector`` (greatest-of),
    ``SOCFARDetector`` (smallest-of),
    ``OSCFARDetector`` (ordered-statistics)

When to Use What
----------------
- **Homogeneous clutter:** ``CACFARDetector`` — mean of all training
  cells. Optimal false-alarm control in uniform backgrounds.

- **Clutter edges:** ``GOCFARDetector`` — max of quadrant means.
  Suppresses false alarms at clutter boundaries at the cost of
  reduced sensitivity.

- **Weak targets near edges:** ``SOCFARDetector`` — min of quadrant
  means.  More sensitive than GO-CFAR but higher false-alarm rate
  at transitions.

- **Interfering targets in guard region:** ``OSCFARDetector`` — order
  statistic (e.g. median). Robust when nearby strong targets
  contaminate the training window.

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
2026-02-06

Modified
--------
2026-02-17
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
from grdl.image_processing.detection.cfar import (
    CFARDetector,
    CACFARDetector,
    GOCFARDetector,
    SOCFARDetector,
    OSCFARDetector,
)
from grdl.image_processing.detection.blob import BlobDetector

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
    # CFAR detectors
    'CFARDetector',
    'CACFARDetector',
    'GOCFARDetector',
    'SOCFARDetector',
    'OSCFARDetector',
    # Blob (connected-component) detector
    'BlobDetector',
]
