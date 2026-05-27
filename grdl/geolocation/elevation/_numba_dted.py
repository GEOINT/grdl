# -*- coding: utf-8 -*-
"""
Numba-Accelerated DTED Bilinear Sampler.

Provides a JIT-compiled bilinear interpolation kernel for sampling DEM
rasters at fractional pixel coordinates.  The kernel runs in parallel
over N query points and bounds-clamps every read so missing-data NaN
masks survive intact.  Higher-order paths (bicubic, quintic) stay on
``scipy.ndimage.map_coordinates`` because the spline filter is already
hand-tuned C; only the bilinear hot path benefits from JIT.

Drop-in pattern matches :mod:`grdl.geolocation._numba_projection`:
the public wrapper returns ``None`` when the numba kernel is not
beneficial (numba missing or N below threshold), letting callers fall
back to the existing NumPy implementation.

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
import math
from typing import Optional

# Third-party
import numpy as np

logger = logging.getLogger(__name__)


# ── Numba availability ───────────────────────────────────────────────

_HAS_NUMBA = False
try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    pass


# ── Kernel ───────────────────────────────────────────────────────────

if _HAS_NUMBA:

    @numba.njit(parallel=True, cache=True)
    def _dted_bilinear_kernel(
        data: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        out: np.ndarray,
    ) -> None:
        """Bilinear sample of ``data`` at fractional (rows, cols).

        Out-of-bounds samples become NaN.  NaNs in ``data`` propagate
        through the 4-point interpolation so DEM voids stay marked.
        """
        n = rows.shape[0]
        nrows = data.shape[0]
        ncols = data.shape[1]
        for i in numba.prange(n):
            r = rows[i]
            c = cols[i]
            if not (0.0 <= r <= nrows - 1) or not (0.0 <= c <= ncols - 1):
                out[i] = np.nan
                continue
            r0 = int(math.floor(r))
            c0 = int(math.floor(c))
            r1 = r0 + 1 if r0 + 1 < nrows else r0
            c1 = c0 + 1 if c0 + 1 < ncols else c0
            dr = r - r0
            dc = c - c0
            z00 = data[r0, c0]
            z01 = data[r0, c1]
            z10 = data[r1, c0]
            z11 = data[r1, c1]
            out[i] = (
                z00 * (1.0 - dc) * (1.0 - dr)
                + z01 * dc * (1.0 - dr)
                + z10 * (1.0 - dc) * dr
                + z11 * dc * dr
            )


# ── JIT warmup ───────────────────────────────────────────────────────

if _HAS_NUMBA:
    try:
        from grdl.geolocation._numba_warmup import warmup_dted_kernels
        warmup_dted_kernels()
    except Exception:  # pragma: no cover
        pass


# ── Public wrapper ───────────────────────────────────────────────────

# Same lowered threshold as ``_numba_projection``: small batches still
# pay setup cost, but per-tile DEM-query chunks of a few hundred points
# now take the JIT path.
_NUMBA_THRESHOLD = 16


def dted_bilinear_fast(
    data: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> Optional[np.ndarray]:
    """Bilinear DEM sample with numba acceleration.

    Parameters
    ----------
    data : np.ndarray
        2D DEM array, float64.
    rows, cols : np.ndarray
        1D fractional pixel coordinates, length N.

    Returns
    -------
    np.ndarray or None
        Sampled heights, shape ``(N,)``.  Returns ``None`` when numba
        is unavailable or N is below the dispatch threshold, signalling
        the caller to fall back to NumPy bilinear.
    """
    if not _HAS_NUMBA:
        return None
    n = rows.shape[0]
    if n < _NUMBA_THRESHOLD:
        return None
    data = np.ascontiguousarray(data, dtype=np.float64)
    rows = np.ascontiguousarray(rows, dtype=np.float64)
    cols = np.ascontiguousarray(cols, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    _dted_bilinear_kernel(data, rows, cols, out)
    return out


__all__ = [
    "dted_bilinear_fast",
]
