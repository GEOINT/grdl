# -*- coding: utf-8 -*-
"""
Sentinel-1 Level 0 Reader subpackage.

Reads raw Sentinel-1 Level 0 SAFE products — SAFE directory
validation, annotation XML, burst-indexed packet parsing, FDBAQ
decompression (optional via ``pip install grdl[s1_l0]``),
orbit/attitude interpolation, and typed metadata.

Public API
----------
- :class:`Sentinel1L0Reader` — inherits
  :class:`grdl.IO.base.ImageReader`
- :class:`ReaderConfig` — reader configuration options
- :func:`open_safe_product` — factory convenience

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
2026-04-16

Modified
--------
2026-04-16
"""

from grdl.IO.sar.sentinel1_l0.reader import (
    ReaderConfig,
    Sentinel1L0Reader,
    open_safe_product,
)
from grdl.IO.sar.sentinel1_l0.crsd_writer import Sentinel1L0ToCRSD
from grdl.IO.sar.sentinel1_l0.orbit import (
    OrbitResolver,
    download_orbit_file,
    ORBIT_SOURCE_AUTO,
    ORBIT_SOURCE_DOWNLOAD,
    ORBIT_SOURCE_FILE,
    ORBIT_SOURCE_ANNOTATION,
)

__all__ = [
    "ReaderConfig",
    "Sentinel1L0Reader",
    "open_safe_product",
    "Sentinel1L0ToCRSD",
    "OrbitResolver",
    "download_orbit_file",
    "ORBIT_SOURCE_AUTO",
    "ORBIT_SOURCE_DOWNLOAD",
    "ORBIT_SOURCE_FILE",
    "ORBIT_SOURCE_ANNOTATION",
]
