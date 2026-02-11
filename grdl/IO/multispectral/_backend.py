# -*- coding: utf-8 -*-
"""
Multispectral Backend Detection - Detect available MS reading libraries.

Probes for h5py, xarray, and spectral at import time. Provides flags
that multispectral readers use to check library availability.

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

_HAS_H5PY = False
_HAS_XARRAY = False
_HAS_SPECTRAL = False

try:
    import h5py  # noqa: F401
    _HAS_H5PY = True
except ImportError:
    pass

try:
    import xarray  # noqa: F401
    _HAS_XARRAY = True
except ImportError:
    pass

try:
    import spectral  # noqa: F401
    _HAS_SPECTRAL = True
except ImportError:
    pass
