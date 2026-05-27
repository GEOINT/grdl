# -*- coding: utf-8 -*-
"""
Numba JIT warmup for geolocation hot paths.

Forces eager compilation of the Numba kernels in
:mod:`grdl.geolocation._numba_projection` and
:mod:`grdl.geolocation.elevation._numba_dted` so the first real call from
a worker process does not pay the compile cost.  Because each kernel is
decorated with ``cache=True``, the compiled object goes into
``__pycache__`` and subsequent process startups load from disk.

Warmup is automatic on import of the kernel modules when
``GRDL_JIT_WARMUP`` is unset or set to a truthy value.  Set
``GRDL_JIT_WARMUP=0`` to skip it (useful for short scripts where the
compile cost would dwarf the run).

Dependencies
------------
numba (optional)

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-05-26

Modified
--------
2026-05-26
"""

# Standard library
import logging
import os
from typing import Callable, Iterable, Tuple

# Third-party
import numpy as np

logger = logging.getLogger(__name__)


def warmup_enabled() -> bool:
    """True when JIT warmup should run at module import."""
    return os.environ.get("GRDL_JIT_WARMUP", "1") not in ("0", "false", "False")


def warmup_projection_kernels() -> None:
    """Eagerly compile R/Rdot + WGS-84 numba kernels.

    Safe no-op when numba is not installed or warmup is disabled.
    """
    if not warmup_enabled():
        return
    try:
        from grdl.geolocation._numba_projection import (
            _HAS_NUMBA,
            _image_to_ground_plane_kernel,
            _wgs84_norm_batch,
        )
    except ImportError:
        return
    if not _HAS_NUMBA:
        return

    n = 8
    r = np.ones(n, dtype=np.float64) * 1.0e6
    rdot = np.zeros(n, dtype=np.float64)
    arp = np.tile(np.array([0.0, 0.0, 7.0e6]), (n, 1))
    varp = np.tile(np.array([7000.0, 0.0, 0.0]), (n, 1))
    gref = np.tile(np.array([6.378e6, 0.0, 0.0]), (n, 1))
    u_z = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    result = np.full((n, 3), np.nan, dtype=np.float64)
    try:
        _image_to_ground_plane_kernel(r, rdot, arp, varp, gref, u_z, result)
    except Exception as exc:  # pragma: no cover
        logger.debug("Projection kernel warmup failed: %s", exc)

    ecf = np.tile(np.array([6.378e6, 0.0, 0.0]), (n, 1))
    out = np.empty_like(ecf)
    try:
        _wgs84_norm_batch(ecf, out)
    except Exception as exc:  # pragma: no cover
        logger.debug("WGS84 norm kernel warmup failed: %s", exc)


def warmup_dted_kernels() -> None:
    """Eagerly compile DTED bilinear numba kernel.

    Safe no-op when numba or the DTED kernel module is unavailable.
    """
    if not warmup_enabled():
        return
    try:
        from grdl.geolocation.elevation._numba_dted import (
            _HAS_NUMBA,
            _dted_bilinear_kernel,
        )
    except ImportError:
        return
    if not _HAS_NUMBA:
        return

    data = np.ones((8, 8), dtype=np.float64)
    rows = np.linspace(0.5, 6.5, 8, dtype=np.float64)
    cols = np.linspace(0.5, 6.5, 8, dtype=np.float64)
    out = np.empty(8, dtype=np.float64)
    try:
        _dted_bilinear_kernel(data, rows, cols, out)
    except Exception as exc:  # pragma: no cover
        logger.debug("DTED kernel warmup failed: %s", exc)


def warmup_all() -> None:
    """Compile every registered numba kernel.

    Call from worker initializers to pay the JIT cost once.
    """
    warmup_projection_kernels()
    warmup_dted_kernels()


__all__ = [
    "warmup_enabled",
    "warmup_projection_kernels",
    "warmup_dted_kernels",
    "warmup_all",
]
