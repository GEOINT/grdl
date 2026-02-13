# -*- coding: utf-8 -*-
"""
Farrow Structure Interpolation - Polynomial coefficient approximation.

Implements the Farrow structure for efficient continuously-variable
fractional delay filtering, as described in Laakso et al. (1996)
"Splitting the Unit Delay", Eqs. 59-63. The key idea is to pre-compute
polynomial approximations of filter coefficients as a function of the
fractional delay d:

    h_d(n) = sum_{m=0}^{P} c_m(n) * d^m,  n = 0, 1, ..., N

At runtime, evaluating the polynomials via Horner's method is cheaper
than recomputing Lagrange (or other) weights from scratch. The Farrow
structure is especially efficient when the fractional delay varies
per output point (as in PFA k-space resampling).

The default prototype is Lagrange interpolation, for which the
polynomial fit is exact when poly_order >= filter_order. Lower
poly_order values give an efficient approximation.

When numba is available, the interpolation loop is parallelized
across output points using ``numba.prange`` for significant speedup
on multi-core systems.

Dependencies
------------
numba (optional, for parallel acceleration)

Reference
---------
C. W. Farrow, "A continuously variable digital delay element," in
Proc. IEEE ISCAS-88, vol. 3, pp. 2641-2645, 1988.

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
from grdl.interpolation.base import Interpolator

# Optional numba acceleration
try:
    import numba as nb
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ── Prototype coefficient computation ───────────────────────────────────


def _lagrange_coeffs(d: float, L: int) -> np.ndarray:
    """Compute Lagrange FIR filter coefficients for fractional delay.

    From Laakso et al. Eq. 42:
        h(n) = prod_{k=0, k!=n}^{N} (D - k) / (n - k)

    where N = L - 1 and D is the total delay centered in the filter
    (Eq. 21).

    Parameters
    ----------
    d : float
        Fractional delay in [0, 1).
    L : int
        Filter length (number of taps).

    Returns
    -------
    np.ndarray
        Filter coefficients, shape ``(L,)``.
    """
    N = L - 1
    # Place the delay at the center of the filter (Eq. 21)
    if L % 2 == 0:
        D = N / 2.0 + d - 0.5
    else:
        D = (N - 1) / 2.0 + d

    h = np.ones(L)
    for n in range(L):
        for k in range(L):
            if k != n:
                h[n] *= (D - k) / (n - k)
    return h


# ── Numba-accelerated kernel ────────────────────────────────────────────


if _HAS_NUMBA:

    @nb.njit(parallel=True, cache=True)
    def _farrow_parallel(x_old, y_old, x_new, poly_coeffs,
                         kernel_length, half, poly_order):
        """Parallel Farrow structure interpolation.

        Parameters
        ----------
        x_old : ndarray, shape (N_in,)
        y_old : ndarray, shape (N_in,), real or complex
        x_new : ndarray, shape (M,)
        poly_coeffs : ndarray, shape (L, P+1), ascending power order
        kernel_length : int
        half : int
        poly_order : int

        Returns
        -------
        ndarray, shape (M,)
        """
        n_in = x_old.shape[0]
        m = x_new.shape[0]
        kl = kernel_length
        P = poly_order

        result = np.empty(m, dtype=y_old.dtype)

        for i in nb.prange(m):
            xi = x_new[i]

            # Out of bounds → zero
            if xi < x_old[0] or xi > x_old[n_in - 1]:
                result[i] = 0
                continue

            # Binary search for insertion index
            lo_s = 0
            hi_s = n_in
            while lo_s < hi_s:
                mid_s = (lo_s + hi_s) >> 1
                if x_old[mid_s] < xi:
                    lo_s = mid_s + 1
                else:
                    hi_s = mid_s
            idx = lo_s

            # Neighbor window — shift so x_new falls between the
            # center-left (sample half-1) and center-right (sample half)
            # of the kernel, matching the Lagrange coefficient centering.
            start = idx - half
            if start < 0:
                start = 0
            if start + kl > n_in:
                start = n_in - kl
            if start < 0:
                start = 0
            end = start + kl
            if end > n_in:
                end = n_in
            npts = end - start

            # Local spacing
            if npts > 1:
                dx_local = (x_old[end - 1] - x_old[start]) / (npts - 1)
            else:
                dx_local = 1.0
            if dx_local < 1e-30:
                dx_local = 1.0

            # Fractional delay: distance from center-left neighbor
            center_idx = half - 1
            if center_idx >= npts:
                center_idx = npts - 1
            d = (xi - x_old[start + center_idx]) / dx_local
            if d < 0.0:
                d = 0.0
            if d > 1.0:
                d = 1.0

            # Evaluate Horner polynomials and accumulate
            v_sum = y_old[start] * 0  # type-preserving zero
            w_sum = 0.0

            for j in range(npts):
                # Horner evaluation: c_P*d + c_{P-1}, ..., *d + c_0
                wj = poly_coeffs[j, P]
                for mm in range(P - 1, -1, -1):
                    wj = wj * d + poly_coeffs[j, mm]
                v_sum += wj * y_old[start + j]
                w_sum += wj

            # Normalize
            if abs(w_sum) > 1e-15:
                v_sum /= w_sum

            result[i] = v_sum

        return result


# ── Class ────────────────────────────────────────────────────────────────


class FarrowInterpolator(Interpolator):
    """Farrow structure interpolator for fractional delay approximation.

    Pre-computes polynomial approximations of filter coefficients
    from a set of prototype Lagrange filters, then evaluates the
    polynomials at runtime using Horner's method. This is efficient
    when many output points each require a different fractional delay
    (common in PFA k-space resampling).

    The filter coefficient for tap n at fractional delay d is:

        h_d(n) = c_0(n) + c_1(n) * d + c_2(n) * d^2 + ... + c_P(n) * d^P

    When numba is installed, the loop over output points is
    parallelized via ``numba.prange`` for significant speedup on
    multi-core systems.

    Parameters
    ----------
    filter_order : int
        FIR filter order N. Uses L = N + 1 taps per output point.
        Default is 3 (4 taps).
    poly_order : int
        Order P of the polynomial approximation of each filter
        coefficient. Higher P gives more accurate approximation.
        When P >= N, the approximation is exact for Lagrange
        prototypes. Default is 3.

    Examples
    --------
    >>> interp = FarrowInterpolator(filter_order=3, poly_order=3)
    >>> y_new = interp(x_old, y_old, x_new)

    For PFA k-space resampling (higher order):

    >>> interp = FarrowInterpolator(filter_order=5, poly_order=4)
    """

    def __init__(
        self,
        filter_order: int = 3,
        poly_order: int = 3,
    ) -> None:
        if filter_order < 1:
            raise ValueError(
                f"filter_order must be >= 1, got {filter_order}"
            )
        if poly_order < 1:
            raise ValueError(
                f"poly_order must be >= 1, got {poly_order}"
            )
        self._filter_order = filter_order
        self._poly_order = poly_order
        self._kernel_length = filter_order + 1
        self._half = self._kernel_length // 2
        self._build_coefficients()

    def _build_coefficients(self) -> None:
        """Pre-compute polynomial coefficients from Lagrange prototypes.

        Evaluates Lagrange filters at Q + 1 uniformly-spaced fractional
        delay values and fits polynomials to each tap's coefficient
        trajectory via least squares.
        """
        L = self._kernel_length
        P = self._poly_order

        # Sample enough prototype filters for a robust polynomial fit
        Q = max(4 * P, 20)
        d_vals = np.linspace(0.0, 1.0, Q + 1)

        # Evaluate Lagrange prototype at each fractional delay
        # Vectorized construction of the prototype matrix
        h_matrix = np.empty((Q + 1, L))
        for q, d in enumerate(d_vals):
            h_matrix[q, :] = _lagrange_coeffs(d, L)

        # Fit polynomial of order P to each tap using least squares
        # Construct Vandermonde matrix once for all taps
        vander = np.vander(d_vals, N=P + 1, increasing=True)  # (Q+1, P+1)
        # Solve all taps at once: vander @ coeffs.T = h_matrix
        # coeffs shape: (L, P+1)
        self._poly_coeffs, _, _, _ = np.linalg.lstsq(
            vander, h_matrix, rcond=None,
        )
        # _poly_coeffs shape is (P+1, L) from lstsq, transpose to (L, P+1)
        self._poly_coeffs = self._poly_coeffs.T

    def __call__(
        self,
        x_old: np.ndarray,
        y_old: np.ndarray,
        x_new: np.ndarray,
    ) -> np.ndarray:
        """Interpolate using Farrow polynomial coefficient evaluation.

        Uses numba-parallelized loop when available, otherwise
        falls back to vectorized numpy.

        Parameters
        ----------
        x_old : np.ndarray
            Original sample coordinates, shape ``(N_in,)``. Must be
            monotonic (increasing or decreasing).
        y_old : np.ndarray
            Original sample values (real or complex), shape ``(N_in,)``.
        x_new : np.ndarray
            Target sample coordinates, shape ``(M,)``.

        Returns
        -------
        np.ndarray
            Interpolated values at ``x_new``, shape ``(M,)``.
            Points outside the range of ``x_old`` are filled with 0.
        """
        # Handle descending x_old
        if len(x_old) > 1 and x_old[-1] < x_old[0]:
            x_old = x_old[::-1].copy()
            y_old = y_old[::-1].copy()

        if _HAS_NUMBA:
            # Numba requires native byte order and contiguous arrays
            if not x_old.dtype.isnative:
                x_old = x_old.astype(x_old.dtype.newbyteorder('='))
            if not y_old.dtype.isnative:
                y_old = y_old.astype(y_old.dtype.newbyteorder('='))
            if not x_new.dtype.isnative:
                x_new = x_new.astype(x_new.dtype.newbyteorder('='))
            x_old = np.ascontiguousarray(x_old)
            y_old = np.ascontiguousarray(y_old)
            x_new = np.ascontiguousarray(x_new)
            return _farrow_parallel(
                x_old, y_old, x_new, self._poly_coeffs,
                self._kernel_length, self._half, self._poly_order,
            )

        return self._numpy_call(x_old, y_old, x_new)

    def _numpy_call(
        self,
        x_old: np.ndarray,
        y_old: np.ndarray,
        x_new: np.ndarray,
    ) -> np.ndarray:
        """Vectorized numpy fallback."""
        n_in = len(x_old)
        kl = self._kernel_length
        half = self._half

        # Find insertion indices
        idx = np.searchsorted(x_old, x_new)

        # Build neighbor index matrix: (M, kernel_length)
        # Shift so x_new falls between center-left (half-1) and
        # center-right (half), matching the polynomial centering.
        offsets = np.arange(-half, half)
        neighbor_idx = idx[:, np.newaxis] + offsets[np.newaxis, :]
        neighbor_idx_clipped = np.clip(neighbor_idx, 0, n_in - 1)

        # Gather neighbor coordinates and values
        x_neighbors = x_old[neighbor_idx_clipped]
        y_neighbors = y_old[neighbor_idx_clipped]

        # Local spacing
        x_span = x_neighbors[:, -1] - x_neighbors[:, 0]
        dx_local = x_span / (kl - 1)
        dx_local = np.where(dx_local < 1e-30, 1.0, dx_local)

        # Fractional delay: distance from center-left neighbor
        center_idx = half - 1
        d = (x_new - x_neighbors[:, center_idx]) / dx_local
        d = np.clip(d, 0.0, 1.0)

        # Evaluate polynomials via Horner's method (vectorized over M)
        # poly_coeffs shape: (L, P+1) with ascending powers
        P = self._poly_order
        coeffs = self._poly_coeffs  # (L, P+1)

        # Horner: start from highest power, work down
        # weights shape: (M, L) — broadcast d (M,) with coeffs (L,)
        weights = np.broadcast_to(
            coeffs[:, P], (len(x_new), kl),
        ).copy()
        for m in range(P - 1, -1, -1):
            weights = weights * d[:, np.newaxis] + coeffs[:, m]

        # Normalize rows
        row_sums = np.sum(weights, axis=1, keepdims=True)
        row_sums = np.where(np.abs(row_sums) < 1e-15, 1.0, row_sums)
        weights /= row_sums

        # Weighted sum
        result = np.sum(weights * y_neighbors, axis=1)

        # Zero out-of-bounds points
        oob = (x_new < x_old[0]) | (x_new > x_old[-1])
        result[oob] = 0.0

        return result


def farrow_interpolator(
    filter_order: int = 3,
    poly_order: int = 3,
) -> FarrowInterpolator:
    """Create a Farrow structure interpolator.

    Convenience factory function. See :class:`FarrowInterpolator`
    for full documentation.

    Parameters
    ----------
    filter_order : int
        FIR filter order N (L = N + 1 taps). Default is 3.
    poly_order : int
        Polynomial approximation order P. Default is 3.

    Returns
    -------
    FarrowInterpolator
        Callable interpolator.
    """
    return FarrowInterpolator(
        filter_order=filter_order,
        poly_order=poly_order,
    )
