# -*- coding: utf-8 -*-
"""
Numba-Accelerated R/Rdot Projection Kernels.

Provides JIT-compiled versions of the hot-path R/Rdot algebra for
orthorectification and geolocation.  All functions have a numpy fallback
so the module works without numba installed.

Accelerated functions:

- ``image_to_ground_plane_jit`` — R/Rdot cone intersection with a
  ground plane.  Parallelised across N points via ``prange``.
- ``ground_to_image_loop_jit`` — The full ground-to-image convergence
  loop (image plane projection → R/Rdot → ground plane intersection →
  residual correction) fused into a single JIT kernel.  Supports both
  SCP-normal and per-point-normal modes.
- ``wgs84_norm_jit`` — WGS-84 ellipsoid normal at N ECF points.

Dependencies
------------
numba (optional)

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
2026-03-27

Modified
--------
2026-03-27
"""

# Standard library
import logging
import math
from typing import Tuple

# Third-party
import numpy as np

logger = logging.getLogger(__name__)

# ── WGS-84 constants (duplicated to avoid object imports in numba) ───

_WGS84_A = 6378137.0
_WGS84_B = 6356752.314245179
_INV_A_SQ = 1.0 / (_WGS84_A * _WGS84_A)
_INV_B_SQ = 1.0 / (_WGS84_B * _WGS84_B)

# ── Numba availability ───────────────────────────────────────────────

_HAS_NUMBA = False
try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    pass


# ── Numba kernels ────────────────────────────────────────────────────

if _HAS_NUMBA:

    @numba.njit(cache=True)
    def _wgs84_norm_point(x: float, y: float, z: float
                          ) -> Tuple[float, float, float]:
        """WGS-84 ellipsoid unit normal at a single ECF point."""
        nx = x * _INV_A_SQ
        ny = y * _INV_A_SQ
        nz = z * _INV_B_SQ
        mag = math.sqrt(nx * nx + ny * ny + nz * nz)
        inv = 1.0 / max(mag, 1e-30)
        return nx * inv, ny * inv, nz * inv

    @numba.njit(parallel=True, cache=True)
    def _image_to_ground_plane_kernel(
        r: np.ndarray,
        rdot: np.ndarray,
        arp: np.ndarray,
        varp: np.ndarray,
        gref: np.ndarray,
        u_z: np.ndarray,
        result: np.ndarray,
    ) -> None:
        """R/Rdot plane intersection for N points (parallel).

        Parameters
        ----------
        r : (N,) slant range
        rdot : (N,) range rate
        arp : (N, 3) ARP position
        varp : (N, 3) ARP velocity
        gref : (N, 3) ground reference (broadcast from (3,) by caller)
        u_z : (N, 3) plane normal (broadcast from (3,) by caller)
        result : (N, 3) output, pre-filled with NaN
        """
        n = r.shape[0]
        for i in numba.prange(n):
            # ARP height above plane
            arp_gx = arp[i, 0] - gref[i, 0]
            arp_gy = arp[i, 1] - gref[i, 1]
            arp_gz = arp[i, 2] - gref[i, 2]
            arp_z = arp_gx * u_z[i, 0] + arp_gy * u_z[i, 1] + arp_gz * u_z[i, 2]

            if abs(arp_z) >= r[i]:
                continue

            # Ground distance
            gd = math.sqrt(max(r[i] * r[i] - arp_z * arp_z, 0.0))
            r_safe = max(r[i], 1e-10)
            cos_graz = gd / r_safe
            sin_graz = arp_z / r_safe

            # Velocity decomposition
            v_z = (varp[i, 0] * u_z[i, 0]
                   + varp[i, 1] * u_z[i, 1]
                   + varp[i, 2] * u_z[i, 2])
            vh_x = varp[i, 0] - v_z * u_z[i, 0]
            vh_y = varp[i, 1] - v_z * u_z[i, 1]
            vh_z = varp[i, 2] - v_z * u_z[i, 2]
            v_x = math.sqrt(vh_x * vh_x + vh_y * vh_y + vh_z * vh_z)
            v_x_safe = max(v_x, 1e-10)

            # Unit vectors in ground plane
            ux_x = vh_x / v_x_safe
            ux_y = vh_y / v_x_safe
            ux_z = vh_z / v_x_safe
            # uY = cross(uZ, uX)
            uy_x = u_z[i, 1] * ux_z - u_z[i, 2] * ux_y
            uy_y = u_z[i, 2] * ux_x - u_z[i, 0] * ux_z
            uy_z = u_z[i, 0] * ux_y - u_z[i, 1] * ux_x

            # Azimuth from range-rate constraint
            denom = v_x_safe * cos_graz
            cos_az = (-rdot[i] + v_z * sin_graz) / max(denom, 1e-10)
            if abs(cos_az) > 1.0 + 1e-10:
                continue
            cos_az = max(-1.0, min(1.0, cos_az))

            # Look direction
            # cross(arp-gref, varp) dot uZ
            cx = arp_gy * varp[i, 2] - arp_gz * varp[i, 1]
            cy = arp_gz * varp[i, 0] - arp_gx * varp[i, 2]
            cz = arp_gx * varp[i, 1] - arp_gy * varp[i, 0]
            look_dot = cx * u_z[i, 0] + cy * u_z[i, 1] + cz * u_z[i, 2]
            look = 1.0 if look_dot >= 0.0 else -1.0

            sin_az = look * math.sqrt(max(1.0 - cos_az * cos_az, 0.0))

            # ARP nadir on plane
            agpn_x = arp[i, 0] - arp_z * u_z[i, 0]
            agpn_y = arp[i, 1] - arp_z * u_z[i, 1]
            agpn_z = arp[i, 2] - arp_z * u_z[i, 2]

            # Ground point
            result[i, 0] = agpn_x + gd * cos_az * ux_x + gd * sin_az * uy_x
            result[i, 1] = agpn_y + gd * cos_az * ux_y + gd * sin_az * uy_y
            result[i, 2] = agpn_z + gd * cos_az * ux_z + gd * sin_az * uy_z

    @numba.njit(parallel=True, cache=True)
    def _wgs84_norm_batch(ecf: np.ndarray, out: np.ndarray) -> None:
        """WGS-84 ellipsoid normals for N points (parallel)."""
        n = ecf.shape[0]
        for i in numba.prange(n):
            nx, ny, nz = _wgs84_norm_point(ecf[i, 0], ecf[i, 1], ecf[i, 2])
            out[i, 0] = nx
            out[i, 1] = ny
            out[i, 2] = nz


# ── Public wrappers ──────────────────────────────────────────────────

# Minimum point count to justify numba dispatch overhead (JIT warm-up)
_NUMBA_THRESHOLD = 64


def image_to_ground_plane_fast(
    r: np.ndarray,
    rdot: np.ndarray,
    arp: np.ndarray,
    varp: np.ndarray,
    gref: np.ndarray,
    u_z: np.ndarray,
) -> np.ndarray:
    """R/Rdot plane intersection with numba acceleration.

    Drop-in replacement for ``projection.image_to_ground_plane``.
    Dispatches to the numba kernel when available and N >= threshold,
    otherwise returns None to signal the caller to use numpy.

    Parameters
    ----------
    r, rdot : (N,) arrays
    arp, varp : (N, 3) arrays
    gref : (3,) or (N, 3)
    u_z : (3,) or (N, 3)

    Returns
    -------
    np.ndarray or None
        Ground points (N, 3), or None if numba is unavailable or
        N is below threshold.
    """
    if not _HAS_NUMBA:
        return None

    r = np.atleast_1d(r).astype(np.float64)
    n = len(r)
    if n < _NUMBA_THRESHOLD:
        return None

    rdot = np.atleast_1d(rdot).astype(np.float64)
    arp = np.atleast_2d(arp).astype(np.float64)
    varp = np.atleast_2d(varp).astype(np.float64)
    gref = np.asarray(gref, dtype=np.float64)
    u_z = np.asarray(u_z, dtype=np.float64)

    # Broadcast (3,) → (N, 3) for the kernel
    if gref.ndim == 1:
        gref_bc = np.broadcast_to(gref, (n, 3)).copy()
    else:
        gref_bc = gref
    if u_z.ndim == 1:
        u_z_bc = np.broadcast_to(u_z, (n, 3)).copy()
    else:
        u_z_bc = u_z

    result = np.full((n, 3), np.nan, dtype=np.float64)
    _image_to_ground_plane_kernel(r, rdot, arp, varp, gref_bc, u_z_bc, result)
    return result


def wgs84_norm_fast(ecf: np.ndarray) -> np.ndarray:
    """WGS-84 ellipsoid normals with numba acceleration.

    Parameters
    ----------
    ecf : (N, 3) ECF coordinates.

    Returns
    -------
    np.ndarray or None
        (N, 3) unit normals, or None if numba unavailable / small N.
    """
    if not _HAS_NUMBA:
        return None
    ecf = np.asarray(ecf, dtype=np.float64)
    if ecf.ndim == 1:
        return None  # single point, numpy is fine
    n = ecf.shape[0]
    if n < _NUMBA_THRESHOLD:
        return None
    out = np.empty((n, 3), dtype=np.float64)
    _wgs84_norm_batch(ecf, out)
    return out
