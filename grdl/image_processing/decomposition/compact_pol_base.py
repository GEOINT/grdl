# -*- coding: utf-8 -*-
"""Compact-pol decomposition base classes.

Provides ``CompactPolDecompositionBase`` for algorithms that operate on
compact-polarimetric 2x2 covariance matrices [C2]. The base follows the same
inheritance pattern as other polarimetric decomposition families so subclasses
inherit shared helpers such as ``_percentile_stretch`` while standardizing
input extraction for C2-style notebook and processor workflows.
"""

# Standard library
import dataclasses
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# Third-party
import numpy as np
from scipy.ndimage import uniform_filter

# GRDL internal
from grdl.image_processing.decomposition.base import PolarimetricDecomposition

if TYPE_CHECKING:
    from grdl.IO.models.base import ChannelMetadata, ImageMetadata


class CompactPolDecompositionBase(PolarimetricDecomposition):
    """Abstract base for compact-pol decomposition algorithms."""

    def decompose(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: np.ndarray,
        svv: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Bridge the universal 4-channel interface to compact-pol C2 input."""
        return self.decompose_compact(shh, shv, svh, svv)

    def execute(
        self,
        metadata: 'ImageMetadata',
        source: np.ndarray,
        **kwargs: Any,
    ) -> tuple:
        """Execute the decomposition via the universal protocol.

        Compact-pol inputs may be supplied as keyword arguments
        (``c11``, ``c12``, ``c21``, ``c22``) or extracted from a
        4-band source cube storing ``[C11, C12_real, C12_imag, C22]``.
        """
        self._metadata = metadata

        c11 = kwargs.pop('c11', None)
        c12 = kwargs.pop('c12', None)
        c21 = kwargs.pop('c21', None)
        c22 = kwargs.pop('c22', None)
        c12_real = kwargs.pop('c12_real', None)
        c12_imag = kwargs.pop('c12_imag', None)

        if c11 is None and source.ndim == 3:
            axis_order = getattr(metadata, 'axis_order', None)
            if axis_order == 'CYX' and source.shape[0] >= 4:
                c11 = source[0]
                c12_real = source[1]
                c12_imag = source[2]
                c22 = source[3]
            elif axis_order == 'YXC' and source.shape[-1] >= 4:
                c11 = source[..., 0]
                c12_real = source[..., 1]
                c12_imag = source[..., 2]
                c22 = source[..., 3]
            elif source.shape[0] >= 4:
                c11 = source[0]
                c12_real = source[1]
                c12_imag = source[2]
                c22 = source[3]
            elif source.shape[-1] >= 4:
                c11 = source[..., 0]
                c12_real = source[..., 1]
                c12_imag = source[..., 2]
                c22 = source[..., 3]

        if c12 is None and c12_real is not None and c12_imag is not None:
            c12 = np.asarray(c12_real) + 1j * np.asarray(c12_imag)
        if c21 is None and c12 is not None:
            c21 = np.conj(c12)

        if c11 is None or c12 is None or c21 is None or c22 is None:
            raise ValueError(
                'Compact-pol inputs must provide c11, c12, c21, and c22 ' \
                'or a 4-band source cube with C11/C12_real/C12_imag/C22.'
            )

        components = self.decompose_compact(c11, c12, c21, c22)
        updated = dataclasses.replace(
            metadata,
            bands=len(components),
            axis_order='CYX',
            channel_metadata=self._build_component_metadata(metadata),
        )
        return components, updated

    def _build_component_metadata(
        self,
        metadata: 'ImageMetadata',
    ) -> list['ChannelMetadata']:
        """Build per-component metadata for decomposition outputs."""
        from grdl.IO.models.base import ChannelMetadata

        return [
            ChannelMetadata(
                index=i,
                name=name,
                role='decomposition',
                source_indices=[],
            )
            for i, name in enumerate(self.component_names)
        ]

    @staticmethod
    def _validate_c2_inputs(
        c11: np.ndarray,
        c12: np.ndarray,
        c21: np.ndarray,
        c22: np.ndarray,
    ) -> None:
        """Validate compact-pol covariance matrix inputs."""
        channels = {'c11': c11, 'c12': c12, 'c21': c21, 'c22': c22}

        for name, arr in channels.items():
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} must be a numpy ndarray, got {type(arr).__name__}")
            if arr.ndim != 2:
                raise ValueError(f"{name} must be 2D (rows, cols), got {arr.ndim}D")

        shape = c11.shape
        for name, arr in channels.items():
            if arr.shape != shape:
                raise ValueError(
                    f"All compact-pol channels must have the same shape. "
                    f"c11 has shape {shape}, but {name} has shape {arr.shape}"
                )

    def _boxcar_complex(self, arr: np.ndarray, window_size: int) -> np.ndarray:
        """Apply boxcar averaging to real or complex arrays."""
        if window_size <= 1:
            return np.asarray(arr)

        arr = np.asarray(arr)
        if np.iscomplexobj(arr):
            real = uniform_filter(np.real(arr), size=window_size)
            imag = uniform_filter(np.imag(arr), size=window_size)
            return real + 1j * imag
        return uniform_filter(arr, size=window_size)

    def _stokes_from_c2(
        self,
        c11: np.ndarray,
        c12: np.ndarray,
        c21: np.ndarray,
        c22: np.ndarray,
        chi: float,
        window_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute averaged Stokes parameters from compact-pol C2."""
        c11s = self._boxcar_complex(c11, window_size)
        c12s = self._boxcar_complex(c12, window_size)
        c21s = self._boxcar_complex(c21, window_size)
        c22s = self._boxcar_complex(c22, window_size)

        s0 = np.real(c11s + c22s)
        s1 = np.real(c11s - c22s)
        s2 = np.real(c12s + c21s)
        s3 = np.where(chi >= 0, 1j * (c12s - c21s), -1j * (c12s - c21s))
        s3 = np.real(s3)
        return s0, s1, s2, s3

    @abstractmethod
    def decompose_compact(
        self,
        c11: np.ndarray,
        c12: np.ndarray,
        c21: np.ndarray,
        c22: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Decompose compact-pol covariance matrix into named outputs."""
        ...

    @property
    @abstractmethod
    def component_names(self) -> Tuple[str, ...]:
        """Names of decomposition outputs."""
        ...

    @abstractmethod
    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
        channels: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        """Create an RGB composite from decomposition outputs."""
        ...