# -*- coding: utf-8 -*-
"""
Windowed Sinc Interpolation - Kaiser-windowed sinc for bandlimited
reconstruction.

Uses a Kaiser window to truncate the ideal sinc kernel, providing
continuously adjustable sidelobe control via the ``beta`` parameter.
Standard in SAR signal processing for k-space resampling.

When numba is available, the interpolation loop is parallelized
across output points using ``numba.prange`` for significant speedup
on multi-core systems.

Dependencies
------------
numba (optional, for parallel acceleration)

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
import math

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

    @nb.njit(inline='always')
    def _sinc(x):
        """Normalized sinc: sin(pi*x) / (pi*x)."""
        if abs(x) < 1e-15:
            return 1.0
        px = math.pi * x
        return math.sin(px) / px

    @nb.njit(inline='always')
    def _bessel_i0(x):
        """Modified Bessel function I0(x) via series expansion."""
        val = 1.0
        term = 1.0
        for k in range(1, 30):
            half_x_over_k = x / (2.0 * k)
            term *= half_x_over_k * half_x_over_k
            val += term
            if term < 1e-16 * val:
                break
        return val

    @nb.njit(parallel=True, cache=True)
    def _kaiser_sinc_parallel(x_old, y_old, x_new, half, beta,
                              kernel_length, oversample, dx_output):
        """Parallel Kaiser-windowed sinc interpolation.

        Parameters
        ----------
        x_old : ndarray, shape (N,)
        y_old : ndarray, shape (N,), real or complex
        x_new : ndarray, shape (M,)
        half : int
        beta : float
        kernel_length : int
        oversample : float
            Anti-alias bandwidth margin. 1.0 = cutoff at Nyquist,
            >1.0 narrows passband to 0.5/oversample of Nyquist.
        dx_output : float
            Output grid spacing. Used with ``dx_local`` (input spacing)
            to adapt the sinc cutoff: ``max(dx_local, dx_output)``
            ensures anti-aliasing when the output is coarser than input.

        Returns
        -------
        ndarray, shape (M,)
        """
        n = x_old.shape[0]
        m = x_new.shape[0]
        i0_beta = _bessel_i0(beta)

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

            # Neighbor window centered at idx
            start = idx - half + 1
            if start < 0:
                start = 0
            if start + kernel_length > n:
                start = n - kernel_length
            if start < 0:
                start = 0
            end = start + kernel_length
            if end > n:
                end = n

            # Local input spacing from neighborhood span
            npts = end - start
            if npts > 1:
                dx_local = (x_old[end - 1] - x_old[start]) / (npts - 1)
            else:
                dx_local = 1.0
            if dx_local < 1e-30:
                dx_local = 1.0

            # Anti-alias: normalize sinc by the coarser of
            # input/output spacing so cutoff tracks output Nyquist
            # when downsampling.
            dx_norm = dx_local
            if dx_output > dx_local:
                dx_norm = dx_output

            # Accumulate weighted sum
            w_sum = 0.0
            v_sum = y_old[start] * 0  # type-preserving zero

            for k in range(start, end):
                dist = xi - x_old[k]

                # Sinc referenced to max(input, output) spacing
                w = _sinc(dist / dx_norm * oversample)

                # Kaiser window on input spacing (full kernel support)
                u = dist / (dx_local * half)
                u_sq = u * u
                if u_sq < 1.0:
                    arg = beta * math.sqrt(1.0 - u_sq)
                    w *= _bessel_i0(arg) / i0_beta
                else:
                    w = 0.0

                w_sum += w
                v_sum += w * y_old[k]

            if abs(w_sum) > 1e-15:
                v_sum /= w_sum

            result[i] = v_sum

        return result


# ── Class ────────────────────────────────────────────────────────────────


class KaiserSincInterpolator(KernelInterpolator):
    """Kaiser-windowed sinc interpolator.

    Kernel: ``sinc(x / dx_norm * oversample) * kaiser(x / dx_input)``
    where ``dx_norm = max(dx_input, dx_output)`` adapts the sinc
    cutoff to the coarser of the two grids. When the output is
    coarser than input (downsampling), the passband automatically
    lowers to prevent aliasing. When numba is installed, the loop
    is parallelized across output points for significant speedup.

    Parameters
    ----------
    kernel_length : int
        Number of input samples used per output point. Must be >= 2
        and even. Default is 8.
    beta : float
        Kaiser window shape parameter. Higher values give lower
        sidelobes but wider main lobe. Typical values:

        - 3.0 — fast, moderate sidelobes (~-25 dB)
        - 5.0 — good balance (~-50 dB sidelobes), default
        - 8.0 — high fidelity (~-80 dB sidelobes)

        Default is 5.0.
    oversample : float
        Additional anti-alias bandwidth margin applied on top of
        the adaptive input/output spacing normalization. Scales
        the sinc argument to lower the passband to
        ``Nyquist / oversample``.

        - 1.0 — cutoff at Nyquist of the coarser grid, default
        - 1.25 — passband at 80% Nyquist (standard for PFA)

        Use with larger ``kernel_length`` (16–64) to give the wider
        kernel sufficient support.

    Examples
    --------
    >>> interp = KaiserSincInterpolator(kernel_length=8, beta=5.0)
    >>> y_new = interp(x_old, y_old, x_new)

    Anti-aliased for PFA k-space resampling:

    >>> interp = KaiserSincInterpolator(
    ...     kernel_length=64, beta=5.0, oversample=1.25,
    ... )
    """

    def __init__(
        self,
        kernel_length: int = 8,
        beta: float = 5.0,
        oversample: float = 1.0,
    ) -> None:
        if kernel_length % 2 != 0:
            raise ValueError(
                f"kernel_length must be even, got {kernel_length}"
            )
        if oversample < 1.0:
            raise ValueError(
                f"oversample must be >= 1.0, got {oversample}"
            )
        self.beta = beta
        self.oversample = oversample
        super().__init__(kernel_length=kernel_length)

    def __call__(
        self,
        x_old: np.ndarray,
        y_old: np.ndarray,
        x_new: np.ndarray,
    ) -> np.ndarray:
        """Interpolate using Kaiser-windowed sinc kernel.

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
            # Handle descending x_old (e.g. azimuth ku positive→negative)
            if len(x_old) > 1 and x_old[-1] < x_old[0]:
                x_old = np.ascontiguousarray(x_old[::-1])
                y_old = np.ascontiguousarray(y_old[::-1])
            # Numba requires native byte order arrays
            if not x_old.dtype.isnative:
                x_old = x_old.astype(x_old.dtype.newbyteorder('='))
            if not y_old.dtype.isnative:
                y_old = y_old.astype(y_old.dtype.newbyteorder('='))
            if not x_new.dtype.isnative:
                x_new = x_new.astype(x_new.dtype.newbyteorder('='))
            # Output spacing for adaptive anti-aliasing
            if len(x_new) > 1:
                dx_output = abs(x_new[-1] - x_new[0]) / (len(x_new) - 1)
            else:
                dx_output = 0.0
            return _kaiser_sinc_parallel(
                x_old, y_old, x_new,
                self._half, self.beta, self._kernel_length,
                self.oversample, dx_output,
            )
        return super().__call__(x_old, y_old, x_new)

    def _compute_weights(self, dx: np.ndarray) -> np.ndarray:
        """Compute Kaiser-windowed sinc kernel weights (numpy fallback)."""
        half = self._half
        beta = self.beta

        # Sinc scaled by oversample to lower passband cutoff
        sinc_vals = np.sinc(dx * self.oversample)

        # Kaiser window on unscaled distance (full kernel span)
        u = dx / half
        u_sq = u * u
        inside = u_sq < 1.0
        arg = np.zeros_like(u_sq)
        arg[inside] = beta * np.sqrt(1.0 - u_sq[inside])
        kaiser_vals = np.i0(arg) / np.i0(beta)
        kaiser_vals[~inside] = 0.0

        return sinc_vals * kaiser_vals


def windowed_sinc_interpolator(
    kernel_length: int = 8,
    beta: float = 5.0,
    oversample: float = 1.0,
) -> KaiserSincInterpolator:
    """Create a Kaiser-windowed sinc interpolator.

    Convenience factory function. See :class:`KaiserSincInterpolator`
    for full documentation.

    Parameters
    ----------
    kernel_length : int
        Number of input samples per output point. Must be >= 2 and
        even. Default is 8.
    beta : float
        Kaiser window shape parameter. Default is 5.0.
    oversample : float
        Anti-alias bandwidth factor. Default is 1.0 (no
        anti-aliasing). Use 1.25 for PFA k-space resampling.

    Returns
    -------
    KaiserSincInterpolator
        Callable interpolator.
    """
    return KaiserSincInterpolator(
        kernel_length=kernel_length, beta=beta, oversample=oversample,
    )
