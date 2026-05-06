# -*- coding: utf-8 -*-
"""
Centralized optional ``torch`` import.

Library modules that use PyTorch import ``torch`` and ``HAS_TORCH``
from here instead of doing their own ``try/except``. This guarantees
``import grdl`` never crashes when torch is missing or fails to load
its DLLs, and emits the diagnostic warning exactly once.

On Windows + conda envs, ``import torch`` raises ``OSError`` (WinError
127 on shm.dll) when numpy/MKL was imported first — this is caught
and reported as a warning rather than propagated.

Author
------
Duane Smalley
duane.d.smalley@gmail.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-05-02

Modified
--------
2026-05-02
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

torch: Optional[object] = None
HAS_TORCH: bool = False

try:
    import torch as _torch
    torch = _torch
    HAS_TORCH = True
except ImportError:
    pass
except OSError as _err:
    logger.warning(
        "PyTorch backend disabled (DLL load failed: %s). "
        "On Windows + conda envs this typically means numpy/MKL was "
        "imported before torch. Import torch before numpy/grdl to "
        "enable GPU acceleration, or use the numba/scipy backends.",
        _err,
    )
