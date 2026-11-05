"""Compact-pol degree of polarization (DOP) from C2."""

from typing import TYPE_CHECKING, Annotated

import numpy as np

from grdl.image_processing.decomposition.compact_pol_base import CompactPolDecompositionBase
from grdl.image_processing.decomposition.pol_matrix import c2_degree_of_polarization
from grdl.image_processing.filters.linear import boxcar_mean
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.vocabulary import ImageModality

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata


@processor_version("1.1.0")
@processor_tags(modalities=[ImageModality.SAR])
class CompactPolDegreeOfPolarization(CompactPolDecompositionBase):
    """C2 DoP preserving input precision; undefined zero-power DoP returns 0."""

    window_size: Annotated[int, Range(min=1, max=31), Desc("Boxcar averaging window size")] = 7
    chi: Annotated[
        float, Range(min=-90.0, max=90.0), Desc("Ellipticity angle of transmitted wave")
    ] = 45.0
    psi: Annotated[
        float, Range(min=-90.0, max=90.0), Desc("Orientation angle of transmitted wave")
    ] = 0.0

    @property
    def component_names(self) -> tuple[str, ...]:
        return ("dop",)

    def decompose_compact(self, c11, c12, c21, c22) -> dict[str, np.ndarray]:
        self._validate_c2_inputs(c11, c12, c21, c22)
        c11s = boxcar_mean(c11, self.window_size, mode="reflect")
        c12s = boxcar_mean(c12, self.window_size, mode="reflect")
        c21s = boxcar_mean(c21, self.window_size, mode="reflect")
        c22s = boxcar_mean(c22, self.window_size, mode="reflect")

        return {"dop": c2_degree_of_polarization(c11s, c12s, c22s, c21s)}

    def to_rgb(
        self,
        components: dict[str, np.ndarray],
        representation: str = "db",
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
        channels: list[str] | None = None,
    ) -> tuple[np.ndarray, "ImageMetadata"]:
        del representation
        from grdl.IO.models.base import ChannelMetadata, ImageMetadata

        dop = np.clip(components["dop"], 0.0, 1.0)
        dop = self._percentile_stretch(dop, percentile_low, percentile_high)
        rgb = np.stack([dop, dop, dop], axis=0)
        meta = ImageMetadata(
            format="CpDOP_RGB",
            rows=dop.shape[0],
            cols=dop.shape[1],
            bands=3,
            dtype="float32",
            axis_order="CYX",
            channel_metadata=[
                ChannelMetadata(index=0, name="dop", role="rgb_red"),
                ChannelMetadata(index=1, name="dop", role="rgb_green"),
                ChannelMetadata(index=2, name="dop", role="rgb_blue"),
            ],
        )
        return rgb, meta
