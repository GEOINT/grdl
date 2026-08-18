# -*- coding: utf-8 -*-
"""Dual-pol degree of polarization (DoP) from C2.

References
----------
Barakat, R. (1977). "Degree of polarization and the principal
idempotents," Optica Acta, 24(9), pp.1093-1096.
"""

from typing import Annotated, Dict, Tuple, TYPE_CHECKING

import numpy as np

from grdl.image_processing.decomposition.dual_pol_base import DualPolDecompositionBase
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.image_processing.params import Range, Desc
from grdl.vocabulary import ImageModality

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata


@processor_version('1.0.0')
@processor_tags(modalities=[ImageModality.SAR])
class DegreeOfPolarizationDP(DualPolDecompositionBase):
    """Barakat degree of polarization for dual-pol covariance matrix C2."""

    window_size: Annotated[int, Range(min=1, max=31),
                           Desc('Boxcar averaging window size')] = 7

    @property
    def component_names(self) -> Tuple[str, ...]:
        return ('dop',)

    def decompose_dual(self, s_co: np.ndarray, s_cross: np.ndarray) -> Dict[str, np.ndarray]:
        self._validate_dual_inputs(s_co, s_cross)
        c11, c12, c22 = self._compute_c2(s_co, s_cross, self.window_size)

        det_c2 = c11 * c22 - np.abs(c12) ** 2
        trace_c2 = c11 + c22
        eps = np.finfo(np.float64).tiny

        with np.errstate(divide='ignore', invalid='ignore'):
            radicand = 1.0 - 4.0 * det_c2 / (trace_c2 ** 2 + eps)
        dop = np.sqrt(np.clip(np.real(radicand), 0.0, 1.0))

        return {'dop': dop}

    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        del representation, percentile_low, percentile_high
        from grdl.IO.models.base import ImageMetadata, ChannelMetadata

        dop = np.clip(components['dop'], 0.0, 1.0).astype(np.float32)
        rgb = np.stack([dop, dop, dop], axis=0)

        meta = ImageMetadata(
            format='DoPDP_RGB',
            rows=dop.shape[0],
            cols=dop.shape[1],
            bands=3,
            dtype='float32',
            axis_order='CYX',
            channel_metadata=[
                ChannelMetadata(index=0, name='dop', role='rgb_red'),
                ChannelMetadata(index=1, name='dop', role='rgb_green'),
                ChannelMetadata(index=2, name='dop', role='rgb_blue'),
            ],
        )
        return rgb, meta
