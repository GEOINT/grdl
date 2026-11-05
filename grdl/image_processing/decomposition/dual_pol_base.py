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
from typing import TYPE_CHECKING, Any, cast

# Third-party
import numpy as np

from grdl.image_processing.decomposition.base import PolarimetricDecomposition
from grdl.image_processing.decomposition.pol_matrix import (
    build_target_vector_c,
    build_target_vector_t,
    compute_dual_elements,
    stokes_from_c2,
)

# GRDL internal
from grdl.IO.models.base import ChannelMetadata
from grdl.IO.models.polarization import channel_axis, resolve_raw_indices

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
    ) -> dict[str, np.ndarray]:
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
    ) -> dict[str, np.ndarray]:
        """Decompose dual-pol data into algorithm-specific components."""
        ...

    def execute(
        self,
        metadata: "ImageMetadata",
        source: np.ndarray,
        **kwargs: Any,
    ) -> tuple:
        """Execute via universal protocol using dual-pol channel extraction."""
        self._metadata = metadata
        explicit = any(name in kwargs for name in ("s_co", "s_cross"))
        s_co = kwargs.pop("s_co", None)
        s_cross = kwargs.pop("s_cross", None)
        source_indices = []
        if explicit:
            if s_co is None or s_cross is None:
                raise ValueError("Explicit dual-pol inputs require s_co and s_cross")
        else:
            axis = channel_axis(source.shape, metadata)
            cube = source if axis == 0 else source.transpose(2, 0, 1)
            source_indices = list(resolve_raw_indices(cube.shape[0], metadata, "dual"))
            s_co, s_cross = (cube[index] for index in source_indices)

        components = self.decompose_dual(s_co, s_cross)
        first = next(iter(components.values()))
        updated = dataclasses.replace(
            metadata,
            bands=len(components),
            rows=first.shape[0],
            cols=first.shape[1],
            dtype=str(np.result_type(*(value.dtype for value in components.values()))),
            axis_order="CYX",
            channel_metadata=[
                ChannelMetadata(
                    index=index,
                    name=name,
                    role="decomposition",
                    source_indices=list(source_indices),
                )
                for index, name in enumerate(components)
            ],
        )
        return components, updated

    @staticmethod
    def _validate_dual_inputs(s_co: np.ndarray, s_cross: np.ndarray) -> None:
        """Validate dual-pol inputs."""
        for name, arr in [("s_co", s_co), ("s_cross", s_cross)]:
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} must be a numpy ndarray, got {type(arr).__name__}")
            if not np.iscomplexobj(arr):
                raise TypeError(f"{name} must be complex-valued, got {arr.dtype}")
            if arr.ndim != 2:
                raise ValueError(f"{name} must be 2D (rows, cols), got {arr.ndim}D")
        if s_co.shape != s_cross.shape:
            raise ValueError(f"Shape mismatch: s_co {s_co.shape} vs s_cross {s_cross.shape}")

    @staticmethod
    def _compute_c2(
        s_co: np.ndarray,
        s_cross: np.ndarray,
        window_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute averaged dual-pol covariance matrix elements (C2)."""
        return compute_dual_elements(
            build_target_vector_c([s_co, s_cross]),
            window_size,
        )

    @staticmethod
    def _compute_t2(
        s_co: np.ndarray,
        s_cross: np.ndarray,
        window_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute averaged dual-pol coherency matrix elements (T2)."""
        return compute_dual_elements(
            build_target_vector_t([s_co, s_cross]),
            window_size,
        )

    @staticmethod
    def _normalized_stokes(
        c11: np.ndarray,
        c12: np.ndarray,
        c22: np.ndarray,
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Canonical Stokes and percentile-normalized magnitudes (not physical DoP)."""
        s0, s1, s2, s3 = stokes_from_c2(c11, c12, c22)

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
            return cast(np.ndarray, clipped / max_v)

        s1_norm = _clip_norm(s1_abs)
        s2_norm = _clip_norm(s2_abs)
        s3_norm = _clip_norm(s3_abs)

        return s0, s1, s2, s3, s1_norm, s2_norm, s3_norm
