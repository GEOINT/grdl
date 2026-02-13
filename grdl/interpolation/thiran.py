# -*- coding: utf-8 -*-
"""
Thiran Allpass Fractional Delay Filter.

Implements the Thiran allpass filter for fractional delay approximation,
as described in Laakso et al. (1996) "Splitting the Unit Delay",
Eq. 86 (originally from Thiran, 1971). The Thiran filter is an IIR
allpass with maximally flat group delay at DC and closed-form
coefficients:

    a_k = (-1)^k * C(N,k) * prod_{n=0}^{N} (D - N + n) / (D - N + k + n)

where N is the filter order, D is the total desired delay, and C(N,k)
is the binomial coefficient.

The allpass transfer function is:

    A(z) = z^{-N} * D(z^{-1}) / D(z)

where D(z) is the denominator polynomial. The filter has exactly unity
magnitude response and approximates constant group delay D across the
frequency band, with best accuracy at low frequencies.

This module provides:

- ``ThiranDelayFilter`` — IIR allpass fractional delay filter applied
  to entire 1D signals. Unlike the FIR interpolators (Lagrange, Kaiser
  sinc), this operates on uniformly-sampled signals and shifts them by
  a fractional number of samples.

- ``thiran_delay`` — convenience function for one-shot delay application.

When numba is available, the sample-by-sample IIR recursion is
JIT-compiled for performance on long signals.

Dependencies
------------
numba (optional, for JIT-compiled IIR recursion)

Reference
---------
J.-P. Thiran, "Recursive digital filters with maximally flat group
delay," IEEE Trans. Circuit Theory, vol. CT-18, no. 6, pp. 659-664,
Nov. 1971.

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

# Standard library
from math import comb

# Third-party
import numpy as np

# Optional numba acceleration
try:
    import numba as nb
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ── Coefficient computation ─────────────────────────────────────────────


def _thiran_coefficients(delay: float, order: int) -> np.ndarray:
    """Compute Thiran allpass filter coefficients.

    From Laakso et al. Eq. 86:

        a_k = (-1)^k * C(N,k) * prod_{n=0}^{N} (D-N+n) / (D-N+k+n)

    Parameters
    ----------
    delay : float
        Total desired delay D in samples. Must satisfy D >= N - 0.5
        for stability (Thiran, 1971).
    order : int
        Filter order N. Common values: 1, 2, 3.

    Returns
    -------
    np.ndarray
        Denominator coefficients [a_0, a_1, ..., a_N] where a_0 = 1.

    Raises
    ------
    ValueError
        If delay is too small for the given order.
    """
    N = order
    D = delay

    if D < N - 0.5:
        raise ValueError(
            f"Delay {D} too small for order {N}: "
            f"must be >= {N - 0.5} for stability."
        )

    a = np.ones(N + 1)
    for k in range(1, N + 1):
        binom = comb(N, k)
        product = 1.0
        for n in range(N + 1):
            product *= (D - N + n) / (D - N + k + n)
        a[k] = ((-1) ** k) * binom * product

    return a


# ── Numba-accelerated IIR recursion ─────────────────────────────────────


if _HAS_NUMBA:

    @nb.njit(cache=True)
    def _allpass_filter_nb(x, a):
        """Apply Nth-order allpass filter using direct form I.

        A(z) = z^{-N} D(z^{-1}) / D(z)

        y[n] = a_N * x[n] + a_{N-1} * x[n-1] + ... + a_0 * x[n-N]
               - a_1 * y[n-1] - a_2 * y[n-2] - ... - a_N * y[n-N]

        Parameters
        ----------
        x : ndarray, shape (M,)
            Input signal (real or complex).
        a : ndarray, shape (N+1,)
            Denominator coefficients [a_0=1, a_1, ..., a_N].

        Returns
        -------
        ndarray, shape (M,)
            Delayed output signal.
        """
        N = len(a) - 1
        M = len(x)
        y = np.empty(M, dtype=x.dtype)

        # Numerator is reversed denominator: b = a[::-1]
        # Direct form I: y[n] = sum_k b[k]*x[n-k] - sum_k a[k]*y[n-k]
        for n in range(M):
            acc = x[n] * 0  # type-preserving zero
            # Numerator: b[k] = a[N-k]
            for k in range(N + 1):
                idx = n - k
                if idx >= 0:
                    acc += a[N - k] * x[idx]
            # Denominator: -a[k] * y[n-k] for k=1..N
            for k in range(1, N + 1):
                idx = n - k
                if idx >= 0:
                    acc -= a[k] * y[idx]
            y[n] = acc

        return y


# ── Numpy fallback ──────────────────────────────────────────────────────


def _allpass_filter_np(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Apply Nth-order allpass filter (numpy fallback).

    Parameters
    ----------
    x : np.ndarray
        Input signal, shape ``(M,)``.
    a : np.ndarray
        Denominator coefficients [a_0=1, a_1, ..., a_N].

    Returns
    -------
    np.ndarray
        Delayed output signal, shape ``(M,)``.
    """
    N = len(a) - 1
    M = len(x)
    y = np.zeros(M, dtype=x.dtype)

    for n in range(M):
        acc = x[n] * 0
        # Numerator: b[k] = a[N-k]
        for k in range(N + 1):
            idx = n - k
            if idx >= 0:
                acc += a[N - k] * x[idx]
        # Denominator
        for k in range(1, N + 1):
            idx = n - k
            if idx >= 0:
                acc -= a[k] * y[idx]
        y[n] = acc

    return y


# ── Class ────────────────────────────────────────────────────────────────


class ThiranDelayFilter:
    """Thiran allpass fractional delay filter.

    Applies a fractional sample delay to a 1D signal using an IIR
    allpass filter with maximally flat group delay at DC. The filter
    has exactly unity magnitude response across all frequencies and
    approximates constant group delay D, with best accuracy at low
    frequencies.

    Unlike the FIR ``Interpolator`` subclasses (Lagrange, Lanczos,
    Kaiser sinc), this filter operates on **uniformly-sampled**
    signals and shifts the entire signal by a specified fractional
    number of samples. It is useful for:

    - Motion compensation in SAR processing
    - Fine timing adjustment in matched filtering
    - Sub-sample registration between signal channels

    The filter coefficients are computed in closed form from Eq. 86
    of Laakso et al. (1996) and are guaranteed stable for D >= N - 0.5.

    Parameters
    ----------
    delay : float
        Desired delay in samples. The integer part is implemented
        as a simple shift; the fractional part uses the allpass
        filter. Must be >= order - 0.5 for stability.
    order : int
        Filter order N. Higher order gives wider bandwidth at the
        cost of more computation. Common values:

        - 1 — first-order, simplest (equivalent to first-order Thiran)
        - 2 — second-order, good for most applications
        - 3 — third-order, high quality (default)

    Raises
    ------
    ValueError
        If delay < order - 0.5 (instability region).

    Examples
    --------
    Delay a signal by 0.3 samples:

    >>> filt = ThiranDelayFilter(delay=0.3, order=3)
    >>> y_delayed = filt(signal)

    Delay by 5.7 samples (5 integer + 0.7 fractional):

    >>> filt = ThiranDelayFilter(delay=5.7, order=3)
    >>> y_delayed = filt(signal)
    """

    def __init__(self, delay: float, order: int = 3) -> None:
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        if delay < 0:
            raise ValueError(f"delay must be >= 0, got {delay}")

        self._delay = delay
        self._order = order

        # Split delay into integer shift + allpass.
        # The allpass of order N with parameter D introduces D samples
        # of group delay at DC.  We set D = N + frac so stability
        # (D >= N - 0.5) is always satisfied.  The integer shift then
        # compensates: int_shift = floor(delay) - N, and the total
        # delay is int_shift + D = floor(delay) - N + N + frac = delay.
        int_part = int(delay)
        frac = delay - int_part

        # If fractional part is negligible, no filter needed
        if abs(frac) < 1e-10:
            self._coeffs = None
            self._int_delay = int_part
        else:
            D = order + frac  # allpass group delay (always > N - 0.5)
            self._coeffs = _thiran_coefficients(D, order)
            self._int_delay = int_part - order
            # When delay < order, the integer shift is negative.
            # In that case, use the allpass alone with D = delay.
            if self._int_delay < 0:
                if delay < order - 0.5:
                    raise ValueError(
                        f"Delay {delay} too small for order {order}: "
                        f"must be >= {order - 0.5} for stability. "
                        f"Use order <= {int(delay + 0.5)}."
                    )
                D = delay
                self._coeffs = _thiran_coefficients(D, order)
                self._int_delay = 0

    @property
    def delay(self) -> float:
        """Total delay in samples."""
        return self._delay

    @property
    def order(self) -> int:
        """Filter order."""
        return self._order

    @property
    def coefficients(self) -> np.ndarray:
        """Denominator coefficients [a_0=1, a_1, ..., a_N].

        Returns ``None`` if the fractional part is negligible.
        """
        return self._coeffs

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Apply the fractional delay to a 1D signal.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (real or complex), shape ``(M,)``.

        Returns
        -------
        np.ndarray
            Delayed signal, shape ``(M,)``. The first ``int_delay``
            samples will be zero (causal shift).
        """
        result = signal

        # Apply allpass filter for fractional part
        if self._coeffs is not None:
            if _HAS_NUMBA:
                result = _allpass_filter_nb(
                    np.ascontiguousarray(result),
                    np.ascontiguousarray(self._coeffs),
                )
            else:
                result = _allpass_filter_np(result, self._coeffs)

        # Apply integer delay via shift
        if self._int_delay > 0:
            shifted = np.zeros_like(result)
            shifted[self._int_delay:] = result[:-self._int_delay]
            result = shifted

        return result


def thiran_delay(
    signal: np.ndarray,
    delay: float,
    order: int = 3,
) -> np.ndarray:
    """Apply a Thiran allpass fractional delay to a signal.

    Convenience function that creates a :class:`ThiranDelayFilter`
    and applies it. For repeated use with the same delay, prefer
    creating the filter object directly to avoid recomputing
    coefficients.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (real or complex), shape ``(M,)``.
    delay : float
        Desired delay in samples. Must be >= order - 0.5.
    order : int
        Filter order. Default is 3.

    Returns
    -------
    np.ndarray
        Delayed signal, shape ``(M,)``.
    """
    filt = ThiranDelayFilter(delay=delay, order=order)
    return filt(signal)
