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

try:
    from grdl.IO.sar.sentinel1_l0.crsd_converter import (
        Sentinel1L0ToCRSD,
        convert_s1_l0_to_crsd,
    )
except ImportError as _crsd_import_error:
    _CRSD_DEPENDENCY_ERROR = (
        "CRSD conversion dependencies are not installed. "
        "Install the optional dependencies required by "
        "'grdl.IO.sar.sentinel1_l0.crsd_converter' to use "
        "'Sentinel1L0ToCRSD' or 'convert_s1_l0_to_crsd'."
    )

    def Sentinel1L0ToCRSD(*args, **kwargs):
        raise ImportError(_CRSD_DEPENDENCY_ERROR) from _crsd_import_error

    def convert_s1_l0_to_crsd(*args, **kwargs):
        raise ImportError(_CRSD_DEPENDENCY_ERROR) from _crsd_import_error

__all__ = [
    "ReaderConfig",
    "Sentinel1L0Reader",
    "open_safe_product",
    "Sentinel1L0ToCRSD",
    "convert_s1_l0_to_crsd",
]
