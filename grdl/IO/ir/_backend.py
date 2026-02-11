# -*- coding: utf-8 -*-
"""
IR Backend Detection - Detect available IR/thermal reading libraries.

Probes for rasterio and h5py at import time. Provides flags that
IR readers use to check library availability.

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
_HAS_H5PY = False

try:
    import rasterio  # noqa: F401
    _HAS_RASTERIO = True
except ImportError:
    pass

try:
    import h5py  # noqa: F401
    _HAS_H5PY = True
except ImportError:
    pass
