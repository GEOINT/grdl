# -*- coding: utf-8 -*-
"""
EO Backend Detection - Detect available EO reading libraries.

Probes for rasterio and glymur at import time. Provides flags that
EO readers use to check library availability.

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
2026-02-10

Modified
--------
2026-02-10
"""

_HAS_RASTERIO = False
_HAS_GLYMUR = False

try:
    import rasterio  # noqa: F401
    _HAS_RASTERIO = True
except ImportError:
    pass

try:
    import glymur  # noqa: F401
    _HAS_GLYMUR = True
except ImportError:
    pass
