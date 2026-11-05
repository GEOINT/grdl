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
from typing import TYPE_CHECKING, Any

# Third-party
import numpy as np

from grdl.image_processing.decomposition.base import PolarimetricDecomposition
from grdl.image_processing.decomposition.pol_matrix import stokes_from_c2
from grdl.image_processing.filters.linear import boxcar_mean

# GRDL internal
from grdl.IO.models.base import ChannelMetadata
from grdl.IO.models.polarization import channel_axis, resolve_packed_c2_indices

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
    ) -> dict[str, np.ndarray]:
        """Bridge the universal 4-channel interface to compact-pol C2 input."""
        return self.decompose_compact(shh, shv, svh, svv)

    def execute(
        self,
        metadata: "ImageMetadata",
        source: np.ndarray,
        **kwargs: Any,
    ) -> tuple:
        """Execute the decomposition via the universal protocol.

        Compact-pol inputs may be supplied as keyword arguments
        (``c11``, ``c12``, ``c21``, ``c22``) or extracted from a
        4-band source cube storing ``[C11, C12_real, C12_imag, C22]``.
        """
        self._metadata = metadata

        explicit = any(
            name in kwargs for name in ("c11", "c12", "c21", "c22", "c12_real", "c12_imag")
        )
        c11 = kwargs.pop("c11", None)
        c12 = kwargs.pop("c12", None)
        c21 = kwargs.pop("c21", None)
        c22 = kwargs.pop("c22", None)
        c12_real = kwargs.pop("c12_real", None)
        c12_imag = kwargs.pop("c12_imag", None)

        source_indices = []
        if not explicit:
            axis = channel_axis(source.shape, metadata)
            cube = source if axis == 0 else source.transpose(2, 0, 1)
            source_indices = list(resolve_packed_c2_indices(cube.shape[0], metadata))
            c11, c12_real, c12_imag, c22 = (cube[index] for index in source_indices)
        elif (
            c11 is None
            or c22 is None
            or (c12 is None and (c12_real is None or c12_imag is None))
            or ((c12_real is None) != (c12_imag is None))
        ):
            raise ValueError("Explicit compact-pol inputs require c11, c22 and complete c12")

        if c12 is None and c12_real is not None and c12_imag is not None:
            c12 = np.asarray(c12_real) + 1j * np.asarray(c12_imag)
        if c21 is None and c12 is not None:
            c21 = np.conj(c12)

        if c11 is None or c12 is None or c21 is None or c22 is None:
            raise ValueError(
                "Compact-pol inputs must provide c11, c12, c21, and c22 "
                "or a 4-band source cube with C11/C12_real/C12_imag/C22."
            )

        components = self.decompose_compact(c11, c12, c21, c22)
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

    def _build_component_metadata(
        self,
        metadata: "ImageMetadata",
    ) -> list["ChannelMetadata"]:
        """Build per-component metadata for decomposition outputs."""
        from grdl.IO.models.base import ChannelMetadata

        return [
            ChannelMetadata(
                index=i,
                name=name,
                role="decomposition",
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
        channels = {"c11": c11, "c12": c12, "c21": c21, "c22": c22}

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

    def _stokes_from_c2(
        self,
        c11: np.ndarray,
        c12: np.ndarray,
        c21: np.ndarray,
        c22: np.ndarray,
        window_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Canonical averaged Stokes, independent of transmit handedness."""
        c11s = boxcar_mean(c11, window_size, mode="reflect")
        c12s = boxcar_mean(c12, window_size, mode="reflect")
        c21s = boxcar_mean(c21, window_size, mode="reflect")
        c22s = boxcar_mean(c22, window_size, mode="reflect")

        return stokes_from_c2(c11s, c12s, c22s, c21s)

    @abstractmethod
    def decompose_compact(
        self,
        c11: np.ndarray,
        c12: np.ndarray,
        c21: np.ndarray,
        c22: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Decompose compact-pol covariance matrix into named outputs."""
        ...

    @property
    @abstractmethod
    def component_names(self) -> tuple[str, ...]:
        """Names of decomposition outputs."""
        ...

    @abstractmethod
    def to_rgb(
        self,
        components: dict[str, np.ndarray],
        representation: str = "db",
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
        channels: list[str] | None = None,
    ) -> tuple[np.ndarray, "ImageMetadata"]:
        """Create an RGB composite from decomposition outputs."""
        ...
