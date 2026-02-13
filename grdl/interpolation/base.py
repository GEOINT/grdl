# -*- coding: utf-8 -*-
"""
Interpolation Base Classes - ABCs for 1D interpolation.

Defines the ``Interpolator`` ABC (callable interface) and
``KernelInterpolator`` (template for kernel-based methods that share
neighbor gathering, local spacing computation, weight normalization,
and out-of-bounds fill).

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
from abc import ABC, abstractmethod

# Third-party
import numpy as np


class Interpolator(ABC):
    """Abstract base class for 1D interpolation.

    All interpolators are callable with signature
    ``(x_old, y_old, x_new) -> y_new``.

    Parameters
    ----------
    x_old : np.ndarray
        Original sample coordinates, shape ``(N,)``. Must be
        monotonic (increasing or decreasing).
    y_old : np.ndarray
        Original sample values (real or complex), shape ``(N,)``.
    x_new : np.ndarray
        Target sample coordinates, shape ``(M,)``.

    Returns
    -------
    np.ndarray
        Interpolated values at ``x_new``, shape ``(M,)``.
    """

    @abstractmethod
    def __call__(
        self,
        x_old: np.ndarray,
        y_old: np.ndarray,
        x_new: np.ndarray,
    ) -> np.ndarray:
        """Interpolate ``y_old`` at ``x_new``."""
        ...


class KernelInterpolator(Interpolator):
    """Base class for kernel-based interpolators.

    Handles the common boilerplate: neighbor gathering via
    ``np.searchsorted``, per-output-point local spacing computation,
    weight normalization, and out-of-bounds zero fill. Subclasses
    only implement :meth:`_compute_weights`.

    Parameters
    ----------
    kernel_length : int
        Number of input samples used per output point. Must be >= 2.
    """

    def __init__(self, kernel_length: int) -> None:
        if kernel_length < 2:
            raise ValueError(
                f"kernel_length must be >= 2, got {kernel_length}"
            )
        self._kernel_length = kernel_length
        self._half = kernel_length // 2

    @abstractmethod
    def _compute_weights(self, dx: np.ndarray) -> np.ndarray:
        """Compute kernel weights from normalized distances.

        Parameters
        ----------
        dx : np.ndarray
            Normalized distances from output points to their neighbors,
            shape ``(M, kernel_length)``. Values are in sample-spacing
            units.

        Returns
        -------
        np.ndarray
            Kernel weights, same shape as ``dx``.
        """
        ...

    def __call__(
        self,
        x_old: np.ndarray,
        y_old: np.ndarray,
        x_new: np.ndarray,
    ) -> np.ndarray:
        """Interpolate using the kernel.

        Parameters
        ----------
        x_old : np.ndarray
            Original sample coordinates, shape ``(N,)``. Must be
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
        # Handle descending x_old (e.g. azimuth ku positive→negative)
        if len(x_old) > 1 and x_old[-1] < x_old[0]:
            x_old = x_old[::-1]
            y_old = y_old[::-1]

        n = len(x_old)
        kl = self._kernel_length
        half = self._half

        # Find insertion indices
        idx = np.searchsorted(x_old, x_new)

        # Build neighbor index matrix: (M, kernel_length)
        offsets = np.arange(-half + 1, half + 1)  # length = kernel_length
        neighbor_idx = idx[:, np.newaxis] + offsets[np.newaxis, :]

        # Clip to valid range
        neighbor_idx_clipped = np.clip(neighbor_idx, 0, n - 1)

        # Gather neighbor coordinates and values: (M, kernel_length)
        x_neighbors = x_old[neighbor_idx_clipped]
        y_neighbors = y_old[neighbor_idx_clipped]

        # Per-output-point local spacing
        x_span = x_neighbors[:, -1] - x_neighbors[:, 0]
        dx_local = x_span / (kl - 1)
        dx_local = np.where(dx_local < 1e-30, 1.0, dx_local)

        # Normalized distances
        dx = (x_new[:, np.newaxis] - x_neighbors) / dx_local[:, np.newaxis]

        # Subclass-defined kernel weights
        weights = self._compute_weights(dx)

        # Normalize rows to preserve DC level
        row_sums = np.sum(weights, axis=1, keepdims=True)
        row_sums = np.where(np.abs(row_sums) < 1e-15, 1.0, row_sums)
        weights /= row_sums

        # Apply weights
        result = np.sum(weights * y_neighbors, axis=1)

        # Zero out-of-bounds points (x_old already ascending after flip)
        oob = (x_new < x_old[0]) | (x_new > x_old[-1])
        result[oob] = 0.0

        return result
