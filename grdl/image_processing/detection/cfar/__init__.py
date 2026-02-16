# -*- coding: utf-8 -*-
"""
CFAR Detectors - Constant False Alarm Rate detection for SAR imagery.

Provides a family of CFAR detectors sharing a common template-method
pipeline.  Each variant differs only in how it estimates local
background statistics from the training window.

Classes
-------
- CFARDetector: Abstract base (template method)
- CACFARDetector: Cell-Averaging — homogeneous clutter
- GOCFARDetector: Greatest-Of — clutter edge suppression
- SOCFARDetector: Smallest-Of — clutter edge sensitivity
- OSCFARDetector: Ordered-Statistics — robust to interfering targets

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
2026-02-16

Modified
--------
2026-02-16
"""

from grdl.image_processing.detection.cfar._base import CFARDetector
from grdl.image_processing.detection.cfar.ca_cfar import CACFARDetector
from grdl.image_processing.detection.cfar.go_cfar import GOCFARDetector
from grdl.image_processing.detection.cfar.so_cfar import SOCFARDetector
from grdl.image_processing.detection.cfar.os_cfar import OSCFARDetector

__all__ = [
    'CFARDetector',
    'CACFARDetector',
    'GOCFARDetector',
    'SOCFARDetector',
    'OSCFARDetector',
]
