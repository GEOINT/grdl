# -*- coding: utf-8 -*-
"""
Polyphase FIR Interpolation - Pre-computed filter bank for fast resampling.

Implements a polyphase interpolation structure where a bank of L
FIR filters is pre-computed at L uniformly-spaced fractional delay
values.  At runtime, the two nearest phases are looked up and
linearly interpolated, avoiding all per-point weight computation.
This makes it the fastest FIR interpolator when many output points
each require a different fractional delay (the common case in PFA
k-space resampling).

Two prototype filter designs are supported:

- **Kaiser-windowed sinc** (``prototype='kaiser'``) — the same
  kernel used by ``KaiserSincInterpolator``, evaluated per-phase.
  Simple, well-characterized, good sidelobe control via ``beta``.
- **Remez / Parks-McClellan** (``prototype='remez'``) — optimal
  equiripple lowpass prototype of length ``L * K``, decomposed into
  L polyphase branches.  Better stopband rejection than Kaiser for
  the same kernel length (Chebyshev-optimal minimax error).

When numba is available, the interpolation loop is parallelized
across output points using ``numba.prange``.

Dependencies
------------
scipy (required for Remez prototype)
numba (optional, for parallel acceleration)

Reference
---------
R. E. Crochiere and L. R. Rabiner, *Multirate Digital Signal
Processing*, Prentice-Hall, 1983.

T. I. Laakso, V. Valimaki, M. Karjalainen, and U. K. Laine,
"Splitting the Unit Delay — Tools for fractional delay filter design,"
IEEE Signal Processing Magazine, vol. 13, no. 1, pp. 30-60, Jan. 1996.

W. G. Carrara, R. S. Goodman, and R. M. Majewski, *Spotlight
Synthetic Aperture Radar: Signal Processing Algorithms*, Artech
House, 1995.

T. J. Parks and J. H. McClellan, "Chebyshev Approximation for
Nonrecursive Digital Filters with Linear Phase," IEEE Transactions
on Circuit Theory, vol. CT-19, no. 2, pp. 189-194, Mar. 1972.

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
from grdl.interpolation.base import Interpolator

# Optional numba acceleration
try:
    import numba as nb
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ── Filter bank construction ──────────────────────────────────────────


def _build_filter_bank(
    kernel_length: int,
    num_phases: int,
    beta: float,
) -> np.ndarray:
    """Pre-compute the polyphase filter bank.

    For each of the *L* phases, evaluates a Kaiser-windowed sinc
    kernel at the corresponding fractional delay ``d = p / L``.
    The kernel is centered so that ``d = 0`` peaks at the
    center-left tap (index ``K // 2 - 1``), matching the neighbor
    window convention used by the other GRDL interpolators.

    Parameters
    ----------
    kernel_length : int
        Number of taps K per polyphase branch.
    num_phases : int
        Number of phases L (fractional delay quantization levels).
    beta : float
        Kaiser window shape parameter.

    Returns
    -------
    np.ndarray, shape ``(num_phases, kernel_length)``
        Filter bank.  Row *p* contains the normalized FIR weights
        for fractional delay ``p / num_phases``.
    """
    K = kernel_length
    L = num_phases
    half = K // 2
    i0_beta = float(np.i0(beta))

    bank = np.empty((L, K))

    for p in range(L):
        d = p / L  # fractional delay in [0, 1)
        for k in range(K):
            # Distance from tap k to the interpolation point.
            # center-left tap is at index (half - 1); at d=0 the
            # interpolation point sits on that tap.
            x = k - (half - 1) - d

            # Sinc (normalized: sinc(x) = sin(pi*x)/(pi*x))
            s = np.sinc(x)

            # Kaiser window: nonzero only inside |u| < 1
            u = x / half
            u_sq = u * u
            if u_sq < 1.0:
                w = float(np.i0(beta * math.sqrt(1.0 - u_sq))) / i0_beta
            else:
                w = 0.0

            bank[p, k] = s * w

        # Normalize to preserve DC
        row_sum = np.sum(bank[p, :])
        if abs(row_sum) > 1e-15:
            bank[p, :] /= row_sum

    return bank


def _build_filter_bank_remez(
    kernel_length: int,
    num_phases: int,
    transition_width: float,
) -> np.ndarray:
    """Pre-compute polyphase filter bank via short Remez + FFT resample.

    Designs an optimal equiripple (Parks-McClellan) lowpass FIR
    prototype at a short tap count where Remez converges reliably,
    then spectrally resamples to length ``L * K + 1`` via FFT.  The
    full-length prototype is directly decomposed into L polyphase
    branches of K taps each.

    The short-Remez + FFT-resample strategy (Carrara et al., 1995)
    avoids Remez convergence failures at large N while preserving
    the optimal equiripple stopband.  The band specification uses
    the ``oversample`` factor derived from ``transition_width``::

        oversample = 1 + transition_width / 2
        PASSBAND   = 0.5 / oversample
        STOPBAND   = 1 - PASSBAND

    which sets the fraction of the Nyquist band that is preserved
    through the interpolation.

    Parameters
    ----------
    kernel_length : int
        Number of taps K per polyphase branch.
    num_phases : int
        Number of phases L (fractional delay quantization levels).
    transition_width : float
        Transition bandwidth as a fraction of the passband cutoff
        frequency.  A value of 0.5 means the transition band is
        50 %% as wide as the passband (equivalent to
        ``oversample = 1.25``).

    Returns
    -------
    np.ndarray, shape ``(num_phases, kernel_length)``
        Filter bank.  Row *p* contains the normalized FIR weights
        for fractional delay ``p / num_phases``.

    Raises
    ------
    ImportError
        If scipy is not installed.
    RuntimeError
        If Remez fails to converge for all candidate tap counts.
    """
    from scipy.signal import remez as _remez, resample as _resample

    K = kernel_length
    L = num_phases

    # Full prototype length (one extra tap for symmetric
    # decomposition into L+1 phases, of which we keep L).
    full_filter_length = L * K + 1

    # Oversample factor from transition_width.
    #   transition_width = 0.5  →  oversample = 1.25
    oversample = 1.0 + transition_width / 2.0
    PASSBAND = 0.5 / oversample
    STOPBAND = 1.0 - PASSBAND

    # Band edges for the short Remez filter.  The prototype
    # operates at rate L * fs, so the per-phase passband/stopband
    # are PASSBAND/L and STOPBAND/L.  The short filter is designed
    # at a reduced length (n_taps), so the band edges are scaled
    # by (full_filter_length / n_taps) to produce the same spectral
    # shape after FFT resampling.
    full_passband = PASSBAND / L
    full_stopband = STOPBAND / L

    # Try increasing short filter lengths until Remez converges.
    # Minimum viable n_taps ≈ 1.2 * K to keep stopband edge < 0.5.
    n_taps_candidates = (101, 201, 301, 401)
    prototype = None

    for n_taps in n_taps_candidates:
        if n_taps >= full_filter_length:
            n_taps = full_filter_length if full_filter_length % 2 == 1 \
                else full_filter_length - 1

        scale_factor = full_filter_length / n_taps
        fp_short = full_passband * scale_factor
        fs_short = full_stopband * scale_factor

        if fs_short >= 0.5:
            continue

        try:
            kernel = _remez(
                n_taps,
                [0, fp_short, fs_short, 0.5],
                desired=[1, 0],
                weight=[0.05, 1.0],
            )
        except ValueError:
            continue

        # FFT-resample to full prototype length.
        # fftshift centers the symmetric kernel; ifftshift restores
        # causal ordering after spectral interpolation.
        if n_taps < full_filter_length:
            prototype = np.fft.ifftshift(
                _resample(np.fft.fftshift(kernel), full_filter_length),
            )
        else:
            prototype = kernel[:full_filter_length]
        break

    if prototype is None:
        raise RuntimeError(
            f"Remez failed to converge for K={K}, L={L}. "
            f"Try increasing kernel_length or transition_width."
        )

    # Normalize prototype to unit DC gain.
    prototype /= np.sum(prototype)

    # Polyphase decomposition via the Carrara/smalleyd indexing.
    # Each of the L+1 phases picks every L-th sample from the
    # prototype, offset by the phase index.
    #
    #   col_indices = [L, 2L, 3L, ..., K*L]     (K columns)
    #   row_indices = [0, -1, -2, ..., -L]       (L+1 rows)
    #   bank[p, k]  = prototype[col_indices[k] + row_indices[p]]
    #
    # Row 0 → fractional delay 0 (peak at tap K//2 - 1 = center-left).
    # Row p → fractional delay p/L.
    col_indices = (np.arange(K) + 1) * L
    row_indices = np.arange(L + 1) * -1
    filter_indices = (
        row_indices[:, np.newaxis] + col_indices[np.newaxis, :]
    )
    bank_full = prototype[filter_indices]  # shape (L+1, K)

    # Row-normalize to preserve DC for each phase.
    row_sums = bank_full.sum(axis=1, keepdims=True)
    row_sums = np.where(np.abs(row_sums) < 1e-15, 1.0, row_sums)
    bank_full = bank_full / row_sums

    # Keep rows 0..L-1 (fractional delays 0 to (L-1)/L).
    # Row L is the unit-delay phase (same as integer shift).
    return bank_full[:L, :]


# ── Numba-accelerated kernel ──────────────────────────────────────────


if _HAS_NUMBA:

    @nb.njit(parallel=True, cache=True)
    def _polyphase_parallel(
        x_old, y_old, x_new, bank, kernel_length, half, num_phases,
    ):
        """Parallel polyphase interpolation with phase interpolation.

        Parameters
        ----------
        x_old : ndarray, shape (N_in,)
        y_old : ndarray, shape (N_in,), real or complex
        x_new : ndarray, shape (M,)
        bank : ndarray, shape (L, K)
            Pre-computed filter bank.
        kernel_length : int
        half : int
        num_phases : int

        Returns
        -------
        ndarray, shape (M,)
        """
        n_in = x_old.shape[0]
        m = x_new.shape[0]
        kl = kernel_length
        L = num_phases

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

            # Neighbor window — same centering as FarrowInterpolator:
            # x_new falls between center-left (half-1) and center-right
            # (half) in the window.
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

            # Local input spacing
            if npts > 1:
                dx_local = (x_old[end - 1] - x_old[start]) / (npts - 1)
            else:
                dx_local = 1.0
            if dx_local < 1e-30:
                dx_local = 1.0

            # Fractional delay from center-left neighbor
            center_idx = half - 1
            if center_idx >= npts:
                center_idx = npts - 1
            d = (xi - x_old[start + center_idx]) / dx_local
            if d < 0.0:
                d = 0.0
            if d > 1.0:
                d = 1.0

            # Phase selection with linear interpolation between phases
            phase_f = d * (L - 1)
            p_lo = int(phase_f)
            if p_lo >= L - 1:
                p_lo = L - 2
            p_hi = p_lo + 1
            alpha = phase_f - p_lo

            # Apply FIR from each phase and blend
            v_lo = y_old[start] * 0  # type-preserving zero
            v_hi = y_old[start] * 0
            for k in range(npts):
                v_lo += bank[p_lo, k] * y_old[start + k]
                v_hi += bank[p_hi, k] * y_old[start + k]

            result[i] = (1.0 - alpha) * v_lo + alpha * v_hi

        return result


# ── Class ─────────────────────────────────────────────────────────────


class PolyphaseInterpolator(Interpolator):
    """Polyphase FIR interpolator with pre-computed filter bank.

    Pre-computes a bank of *L* FIR filters at uniformly-spaced
    fractional delays.  At runtime, selects the two nearest phases
    and linearly interpolates, giving fast high-quality interpolation
    with no per-point weight computation.

    This is the fastest FIR interpolator for PFA k-space resampling
    where every output point has a different fractional delay, because
    all kernel weights are pre-computed.  The accuracy is controlled
    by ``num_phases`` (finer quantization) and ``kernel_length``
    (wider kernel).

    Two prototype designs are available:

    - ``'kaiser'`` — Kaiser-windowed sinc evaluated per-phase.
      Simple, well-characterized sidelobe control via ``beta``.
    - ``'remez'`` — Optimal equiripple lowpass prototype (length
      ``L * K``) designed via the Parks-McClellan algorithm, then
      decomposed into L polyphase branches.  Better stopband
      rejection than Kaiser for the same kernel length.

    When numba is installed, the loop over output points is
    parallelized via ``numba.prange`` for significant speedup on
    multi-core systems.

    Parameters
    ----------
    kernel_length : int
        Number of taps K per polyphase branch.  Must be >= 2 and
        even.  Default is 8.
    num_phases : int
        Number of polyphase branches L (fractional delay quantization
        levels).  Higher L gives finer delay resolution at the cost
        of memory (``L * K * 8`` bytes).  Default is 128.
    beta : float
        Kaiser window shape parameter.  Controls sidelobe level of
        each polyphase branch.  Only used when ``prototype='kaiser'``.
        Default is 5.0.
    prototype : str
        Filter bank design method.  ``'kaiser'`` for Kaiser-windowed
        sinc (default), ``'remez'`` for Parks-McClellan equiripple.
    transition_width : float
        Transition bandwidth as a fraction of the passband cutoff.
        Only used when ``prototype='remez'``.  A value of 0.5 means
        the transition band is 50 %% as wide as the passband.
        Default is 0.5.

    Examples
    --------
    Kaiser prototype (default):

    >>> interp = PolyphaseInterpolator(kernel_length=8, num_phases=128)
    >>> y_new = interp(x_old, y_old, x_new)

    Remez prototype for better stopband rejection:

    >>> interp = PolyphaseInterpolator(
    ...     kernel_length=32, num_phases=256, prototype='remez',
    ... )

    High-quality Kaiser for PFA:

    >>> interp = PolyphaseInterpolator(
    ...     kernel_length=64, num_phases=256, beta=5.0,
    ... )
    """

    _PROTOTYPES = ('kaiser', 'remez')

    def __init__(
        self,
        kernel_length: int = 8,
        num_phases: int = 128,
        beta: float = 5.0,
        prototype: str = 'kaiser',
        transition_width: float = 0.5,
    ) -> None:
        if kernel_length < 2:
            raise ValueError(
                f"kernel_length must be >= 2, got {kernel_length}"
            )
        if kernel_length % 2 != 0:
            raise ValueError(
                f"kernel_length must be even, got {kernel_length}"
            )
        if num_phases < 2:
            raise ValueError(
                f"num_phases must be >= 2, got {num_phases}"
            )
        if prototype not in self._PROTOTYPES:
            raise ValueError(
                f"prototype must be one of {self._PROTOTYPES}, "
                f"got '{prototype}'"
            )
        self._kernel_length = kernel_length
        self._half = kernel_length // 2
        self._num_phases = num_phases
        self.beta = beta
        self._prototype = prototype
        self._transition_width = transition_width

        # Pre-compute filter bank (L × K)
        if prototype == 'kaiser':
            self._bank = _build_filter_bank(
                kernel_length, num_phases, beta,
            )
        else:  # 'remez'
            self._bank = _build_filter_bank_remez(
                kernel_length, num_phases, transition_width,
            )

    @property
    def kernel_length(self) -> int:
        """Number of taps per polyphase branch."""
        return self._kernel_length

    @property
    def num_phases(self) -> int:
        """Number of polyphase branches."""
        return self._num_phases

    @property
    def prototype(self) -> str:
        """Filter bank design method ('kaiser' or 'remez')."""
        return self._prototype

    def __call__(
        self,
        x_old: np.ndarray,
        y_old: np.ndarray,
        x_new: np.ndarray,
    ) -> np.ndarray:
        """Interpolate using polyphase filter bank lookup.

        Uses numba-parallelized loop when available, otherwise
        falls back to vectorized numpy.

        Parameters
        ----------
        x_old : np.ndarray
            Original sample coordinates, shape ``(N,)``.  Must be
            monotonic (increasing or decreasing).
        y_old : np.ndarray
            Original sample values (real or complex), shape ``(N,)``.
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
            return _polyphase_parallel(
                x_old, y_old, x_new, self._bank,
                self._kernel_length, self._half, self._num_phases,
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
        L = self._num_phases

        # Find insertion indices
        idx = np.searchsorted(x_old, x_new)

        # Build neighbor index matrix — same centering as Farrow
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

        # Fractional delay from center-left neighbor
        center_idx = half - 1
        d = (x_new - x_neighbors[:, center_idx]) / dx_local
        d = np.clip(d, 0.0, 1.0)

        # Phase selection with linear interpolation
        phase_f = d * (L - 1)
        p_lo = np.floor(phase_f).astype(int)
        p_lo = np.clip(p_lo, 0, L - 2)
        p_hi = p_lo + 1
        alpha = phase_f - p_lo  # (M,)

        # Gather filter bank rows for each output point: (M, K)
        weights_lo = self._bank[p_lo, :]
        weights_hi = self._bank[p_hi, :]

        # Blend phases
        weights = (1.0 - alpha[:, np.newaxis]) * weights_lo \
            + alpha[:, np.newaxis] * weights_hi

        # Weighted sum
        result = np.sum(weights * y_neighbors, axis=1)

        # Zero out-of-bounds points
        oob = (x_new < x_old[0]) | (x_new > x_old[-1])
        result[oob] = 0.0

        return result


def polyphase_interpolator(
    kernel_length: int = 8,
    num_phases: int = 128,
    beta: float = 5.0,
    prototype: str = 'kaiser',
    transition_width: float = 0.5,
) -> PolyphaseInterpolator:
    """Create a polyphase FIR interpolator.

    Convenience factory function.  See :class:`PolyphaseInterpolator`
    for full documentation.

    Parameters
    ----------
    kernel_length : int
        Number of taps K per polyphase branch.  Must be >= 2 and
        even.  Default is 8.
    num_phases : int
        Number of polyphase branches L.  Default is 128.
    beta : float
        Kaiser window shape parameter (``prototype='kaiser'`` only).
        Default is 5.0.
    prototype : str
        ``'kaiser'`` (default) or ``'remez'``.
    transition_width : float
        Transition bandwidth fraction (``prototype='remez'`` only).
        Default is 0.5.

    Returns
    -------
    PolyphaseInterpolator
        Callable interpolator.
    """
    return PolyphaseInterpolator(
        kernel_length=kernel_length,
        num_phases=num_phases,
        beta=beta,
        prototype=prototype,
        transition_width=transition_width,
    )
