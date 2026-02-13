# -*- coding: utf-8 -*-
"""
Lagrange Interpolation - Maximally flat FIR fractional delay filter.

Implements the classical Lagrange polynomial interpolation as described
in Laakso et al. (1996) "Splitting the Unit Delay", Eq. 42. The
Lagrange interpolator has maximally flat frequency response at DC,
explicit closed-form coefficient formulas, and produces a smooth
magnitude response whose maximum never exceeds unity (important for
feedback applications). The approximation bandwidth grows slowly with
filter order.

The Lagrange interpolation weight for the n-th sample point is:

    h(n) = prod_{k=0, k!=n}^{N} (D - k) / (n - k)

where N is the filter order (L = N + 1 taps) and D is the fractional
delay.

When numba is available, the interpolation loop is parallelized
across output points using ``numba.prange`` for significant speedup
on multi-core systems.

Dependencies
------------
numba (optional, for parallel acceleration)

Reference
---------
T. I. Laakso, V. Valimaki, M. Karjalainen, and U. K. Laine,
"Splitting the Unit Delay — Tools for fractional delay filter design,"
IEEE Signal Processing Magazine, vol. 13, no. 1, pp. 30-60, Jan. 1996.

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
2026-02-12

Modified
--------
2026-02-12
"""

# Third-party
import numpy as np

# GRDL internal
from grdl.interpolation.base import KernelInterpolator

# Optional numba acceleration
try:
    import numba as nb
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ── Numba-accelerated kernel ────────────────────────────────────────────

if _HAS_NUMBA:

    @nb.njit(parallel=True, cache=True)
    def _lagrange_parallel(x_old, y_old, x_new, kernel_length, half):
        """Parallel Lagrange interpolation.

        Parameters
        ----------
        x_old : ndarray, shape (N,)
        y_old : ndarray, shape (N,), real or complex
        x_new : ndarray, shape (M,)
        kernel_length : int
        half : int

        Returns
        -------
        ndarray, shape (M,)
        """
        n = x_old.shape[0]
        m = x_new.shape[0]
        kl = kernel_length

        result = np.empty(m, dtype=y_old.dtype)

        for i in nb.prange(m):
            xi = x_new[i]

            # Out of bounds → zero
            if xi < x_old[0] or xi > x_old[n - 1]:
                result[i] = 0
                continue

            # Binary search for insertion index
            lo_s = 0
            hi_s = n
            while lo_s < hi_s:
                mid = (lo_s + hi_s) >> 1
                if x_old[mid] < xi:
                    lo_s = mid + 1
                else:
                    hi_s = mid
            idx = lo_s

            # Neighbor window
            start = idx - half + 1
            if start < 0:
                start = 0
            if start + kl > n:
                start = n - kl
            if start < 0:
                start = 0
            end = start + kl
            if end > n:
                end = n
            npts = end - start

            # Compute Lagrange weights and accumulate result
            v_sum = y_old[start] * 0  # type-preserving zero
            w_sum = 0.0

            for j in range(npts):
                xj = x_old[start + j]
                # Lagrange basis polynomial l_j(xi)
                wj = 1.0
                for k in range(npts):
                    if k != j:
                        xk = x_old[start + k]
                        denom = xj - xk
                        if abs(denom) < 1e-15:
                            continue
                        wj *= (xi - xk) / denom

                v_sum += wj * y_old[start + j]
                w_sum += wj

            # Normalize to preserve DC
            if abs(w_sum) > 1e-15:
                v_sum /= w_sum

            result[i] = v_sum

        return result


# ── Class ────────────────────────────────────────────────────────────────


class LagrangeInterpolator(KernelInterpolator):
    """Lagrange polynomial interpolator.

    Maximally flat FIR fractional delay filter design via classical
    Lagrange interpolation. The filter coefficients are obtained by
    fitting an Nth-order polynomial through N + 1 data points.

    The weight for the j-th neighbor of an output point is the
    Lagrange basis polynomial evaluated at that point:

        l_j(x) = prod_{k != j} (x - x_k) / (x_j - x_k)

    When numba is installed, the loop over output points is
    parallelized via ``numba.prange`` for significant speedup on
    multi-core systems.

    Parameters
    ----------
    order : int
        Polynomial order N. Uses L = N + 1 input samples per output
        point. Common values:

        - 1 — linear interpolation, 2 taps
        - 3 — cubic, 4 taps (good general-purpose default)
        - 5 — quintic, 6 taps
        - 7 — septic, 8 taps (high quality)

        Even-order filters (odd L) have better magnitude response
        but worse phase delay. Odd-order filters (even L) have
        better phase delay but a zero at Nyquist. Default is 3.

    Examples
    --------
    >>> interp = LagrangeInterpolator(order=3)
    >>> y_new = interp(x_old, y_old, x_new)
    """

    def __init__(self, order: int = 3) -> None:
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        self.order = order
        super().__init__(kernel_length=order + 1)

    def __call__(
        self,
        x_old: np.ndarray,
        y_old: np.ndarray,
        x_new: np.ndarray,
    ) -> np.ndarray:
        """Interpolate using Lagrange polynomial weights.

        Uses numba-parallelized loop when available, otherwise
        falls back to vectorized numpy.

        Parameters
        ----------
        x_old : np.ndarray
            Original sample coordinates, shape ``(N,)``.
        y_old : np.ndarray
            Original sample values, shape ``(N,)``.
        x_new : np.ndarray
            Target sample coordinates, shape ``(M,)``.

        Returns
        -------
        np.ndarray
            Interpolated values, shape ``(M,)``.
        """
        if _HAS_NUMBA:
            # Handle descending x_old
            if len(x_old) > 1 and x_old[-1] < x_old[0]:
                x_old = np.ascontiguousarray(x_old[::-1])
                y_old = np.ascontiguousarray(y_old[::-1])
            # Numba requires native byte order
            if not x_old.dtype.isnative:
                x_old = x_old.astype(x_old.dtype.newbyteorder('='))
            if not y_old.dtype.isnative:
                y_old = y_old.astype(y_old.dtype.newbyteorder('='))
            if not x_new.dtype.isnative:
                x_new = x_new.astype(x_new.dtype.newbyteorder('='))
            return _lagrange_parallel(
                x_old, y_old, x_new,
                self._kernel_length, self._half,
            )
        return super().__call__(x_old, y_old, x_new)

    def _compute_weights(self, dx: np.ndarray) -> np.ndarray:
        """Compute Lagrange basis polynomial weights (numpy fallback).

        The weight for neighbor j is:

            l_j = prod_{k != j} dx[:, k] / (dx[:, k] - dx[:, j])

        The local spacing normalization from the base class cancels
        in the ratio, making this formula valid for both uniform and
        non-uniform input grids. The outer loop is over L (kernel
        taps, typically 2-8) while the inner operations are vectorized
        across all M output points.

        Parameters
        ----------
        dx : np.ndarray
            Normalized distances from output points to their
            neighbors, shape ``(M, L)``.

        Returns
        -------
        np.ndarray
            Lagrange weights, shape ``(M, L)``.
        """
        M, L = dx.shape
        weights = np.ones((M, L))

        for j in range(L):
            for k in range(L):
                if k != j:
                    diff = dx[:, k] - dx[:, j]
                    # Guard against coincident neighbors (degenerate
                    # geometry).  When two neighbors occupy the same
                    # position the basis polynomial is undefined; we
                    # leave the weight unchanged (product term = 1).
                    safe_diff = np.where(np.abs(diff) < 1e-15, 1.0, diff)
                    weights[:, j] *= dx[:, k] / safe_diff

        return weights


def lagrange_interpolator(
    order: int = 3,
) -> LagrangeInterpolator:
    """Create a Lagrange polynomial interpolator.

    Convenience factory function. See :class:`LagrangeInterpolator`
    for full documentation.

    Parameters
    ----------
    order : int
        Polynomial order N (L = N + 1 taps). Default is 3 (cubic).

    Returns
    -------
    LagrangeInterpolator
        Callable interpolator.
    """
    return LagrangeInterpolator(order=order)
