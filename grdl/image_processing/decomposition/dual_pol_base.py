# -*- coding: utf-8 -*-
"""
Dual-Pol Decomposition Base Classes - Shared interface and helpers.

Provides ``DualPolDecompositionBase``, an abstract base class for
algorithms that operate on dual-polarization SAR data (co-pol + cross-pol).

This class standardizes the dual-pol interface and execution plumbing while
remaining compatible with the universal 4-channel decomposition API:

- Subclasses implement ``decompose_dual(s_co, s_cross)``.
- The inherited ``decompose(shh, shv, svh, svv)`` shim maps
  ``shh -> s_co`` and ``shv -> s_cross``.

The helper methods centralize common computations used across multiple
algorithms (C2/T2 construction and Stokes normalization).

Notes
-----
Compact-pol decomposition support will follow the same design via a dedicated
compact-pol base class.
"""

# Standard library
import dataclasses
from abc import abstractmethod
from typing import Any, Dict, Tuple, TYPE_CHECKING

# Third-party
import numpy as np
from scipy.ndimage import uniform_filter

# GRDL internal
from grdl.image_processing.decomposition.base import PolarimetricDecomposition

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata


class DualPolDecompositionBase(PolarimetricDecomposition):
    """Abstract base for dual-pol decomposition algorithms."""

    def decompose(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: np.ndarray,
        svv: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Bridge quad-pol interface to dual-pol decomposition.

        Parameters
        ----------
        shh : np.ndarray
            Co-pol channel.
        shv : np.ndarray
            Cross-pol channel.
        svh : np.ndarray
            Ignored for dual-pol decomposition.
        svv : np.ndarray
            Ignored for dual-pol decomposition.

        Returns
        -------
        Dict[str, np.ndarray]
            Decomposition outputs from ``decompose_dual``.
        """
        del svh, svv
        return self.decompose_dual(shh, shv)

    @abstractmethod
    def decompose_dual(
        self,
        s_co: np.ndarray,
        s_cross: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Decompose dual-pol data into algorithm-specific components."""
        ...

    def execute(
        self,
        metadata: 'ImageMetadata',
        source: np.ndarray,
        **kwargs: Any,
    ) -> tuple:
        """Execute via universal protocol using dual-pol channel extraction."""
        self._metadata = metadata
        s_co = kwargs.pop('s_co', None)
        s_cross = kwargs.pop('s_cross', None)
        if s_co is None and source.ndim == 3:
            axis_order = getattr(metadata, 'axis_order', None)
            if axis_order == 'CYX' and source.shape[0] >= 2:
                s_co = source[0]
                s_cross = source[1]
            elif axis_order == 'YXC' and source.shape[-1] >= 2:
                s_co = source[..., 0]
                s_cross = source[..., 1]
            else:
                channel_metadata = getattr(metadata, 'channel_metadata', None)
                bands = getattr(metadata, 'bands', None)
                n_channels = len(channel_metadata) if channel_metadata else bands
                if n_channels == 2 and source.shape[0] == 2 and source.shape[-1] != 2:
                    s_co = source[0]
                    s_cross = source[1]
                elif source.shape[-1] >= 2:
                    s_co = source[..., 0]
                    s_cross = source[..., 1]

        components = self.decompose_dual(s_co, s_cross)
        updated = dataclasses.replace(metadata, bands=len(components))
        return components, updated

    @staticmethod
    def _validate_dual_inputs(s_co: np.ndarray, s_cross: np.ndarray) -> None:
        """Validate dual-pol inputs."""
        for name, arr in [('s_co', s_co), ('s_cross', s_cross)]:
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"{name} must be a numpy ndarray, got {type(arr).__name__}"
                )
            if not np.iscomplexobj(arr):
                raise TypeError(
                    f"{name} must be complex-valued, got {arr.dtype}"
                )
            if arr.ndim != 2:
                raise ValueError(
                    f"{name} must be 2D (rows, cols), got {arr.ndim}D"
                )
        if s_co.shape != s_cross.shape:
            raise ValueError(
                f"Shape mismatch: s_co {s_co.shape} vs s_cross {s_cross.shape}"
            )

    @staticmethod
    def _compute_c2(
        s_co: np.ndarray,
        s_cross: np.ndarray,
        window_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute averaged dual-pol covariance matrix elements (C2)."""
        c11 = uniform_filter(np.abs(s_co) ** 2, size=window_size)
        c22 = uniform_filter(np.abs(s_cross) ** 2, size=window_size)
        c12_real = uniform_filter(np.real(s_co * np.conj(s_cross)), size=window_size)
        c12_imag = uniform_filter(np.imag(s_co * np.conj(s_cross)), size=window_size)
        c12 = c12_real + 1j * c12_imag
        return c11, c12, c22

    @staticmethod
    def _compute_t2(
        s_co: np.ndarray,
        s_cross: np.ndarray,
        window_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute averaged dual-pol coherency matrix elements (T2)."""
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        k1 = (s_co + s_cross) * inv_sqrt2
        k2 = (s_co - s_cross) * inv_sqrt2

        t11 = uniform_filter(np.abs(k1) ** 2, size=window_size)
        t22 = uniform_filter(np.abs(k2) ** 2, size=window_size)
        t12_real = uniform_filter(np.real(k1 * np.conj(k2)), size=window_size)
        t12_imag = uniform_filter(np.imag(k1 * np.conj(k2)), size=window_size)
        t12 = t12_real + 1j * t12_imag
        return t11, t12, t22

    @staticmethod
    def _normalized_stokes(
        c11: np.ndarray,
        c12: np.ndarray,
        c22: np.ndarray,
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute Stokes terms and percentile-normalized |S1|, |S2|, |S3|."""
        s0 = c11 + c22
        s1 = c11 - c22
        s2 = 2.0 * np.real(c12)
        s3 = 2.0 * np.imag(c12)

        s1_abs = np.abs(s1)
        s2_abs = np.abs(s2)
        s3_abs = np.abs(s3)

        def _clip_norm(arr: np.ndarray) -> np.ndarray:
            finite = np.isfinite(arr)
            if not np.any(finite):
                return np.zeros_like(arr, dtype=np.float64)
            vals = arr[finite]
            lo = np.percentile(vals, percentile_low)
            hi = np.percentile(vals, percentile_high)
            clipped = np.clip(arr, lo, hi)
            max_v = np.nanmax(clipped)
            if not np.isfinite(max_v) or max_v <= np.finfo(np.float64).tiny:
                return np.zeros_like(arr, dtype=np.float64)
            return clipped / max_v

        s1_norm = _clip_norm(s1_abs)
        s2_norm = _clip_norm(s2_abs)
        s3_norm = _clip_norm(s3_abs)

        return s0, s1, s2, s3, s1_norm, s2_norm, s3_norm
