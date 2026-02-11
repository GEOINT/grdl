# -*- coding: utf-8 -*-
"""
Transforms - Apply spatial transforms to vector detection data.

Provides bridge functions that apply co-registration (and future ortho /
geolocation) transforms to detection geometries in vector space, avoiding
lossy raster warping.

Key Functions
-------------
- transform_pixel_geometry: Transform a single shapely pixel geometry.
- transform_detection: Transform a Detection's pixel geometry.
- transform_detection_set: Transform all detections in a DetectionSet.

Usage
-----
Detect on the original (unwarped) image, then transform coordinates:

    >>> from grdl.coregistration import FeatureMatchCoRegistration
    >>> from grdl.transforms import transform_detection_set
    >>> result = coreg.estimate(fixed_image, moving_image)
    >>> detections = detector.detect(moving_image)
    >>> aligned = transform_detection_set(detections, result)

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

from grdl.transforms.detection import (
    transform_pixel_geometry,
    transform_detection,
    transform_detection_set,
)

__all__ = [
    'transform_pixel_geometry',
    'transform_detection',
    'transform_detection_set',
]
