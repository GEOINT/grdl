# -*- coding: utf-8 -*-
"""Compact-pol degree of polarization (DOP) from C2."""

from typing import Annotated, Dict, Tuple, List, Optional, TYPE_CHECKING

import numpy as np

from grdl.image_processing.decomposition.compact_pol_base import CompactPolDecompositionBase
from grdl.image_processing.decomposition.compact_pol_utils import safe_divide
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.image_processing.params import Range, Desc
from grdl.vocabulary import ImageModality

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata


@processor_version('1.0.0')
@processor_tags(modalities=[ImageModality.SAR])
class CompactPolDegreeOfPolarization(CompactPolDecompositionBase):
    """Barakat degree of polarization for compact-pol C2."""

    window_size: Annotated[int, Range(min=1, max=31), Desc('Boxcar averaging window size')] = 7
    chi: Annotated[float, Range(min=-90.0, max=90.0), Desc('Ellipticity angle of transmitted wave')] = 45.0
    psi: Annotated[float, Range(min=-90.0, max=90.0), Desc('Orientation angle of transmitted wave')] = 0.0

    @property
    def component_names(self) -> Tuple[str, ...]:
        return ('dop',)

    def decompose_compact(self, c11, c12, c21, c22) -> Dict[str, np.ndarray]:
        self._validate_c2_inputs(c11, c12, c21, c22)
        c11s = self._boxcar_complex(c11, self.window_size)
        c12s = self._boxcar_complex(c12, self.window_size)
        c21s = self._boxcar_complex(c21, self.window_size)
        c22s = self._boxcar_complex(c22, self.window_size)

        det = c11s * c22s - c12s * c21s
        trace = c11s + c22s
        dop = np.sqrt(np.clip(np.real(1.0 - 4.0 * safe_divide(det, trace ** 2)), 0.0, 1.0))
        return {'dop': np.real(dop)}

    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
        channels: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        del representation
        from grdl.IO.models.base import ImageMetadata, ChannelMetadata

        dop = np.clip(components['dop'], 0.0, 1.0)
        dop = self._percentile_stretch(dop, percentile_low, percentile_high)
        rgb = np.stack([dop, dop, dop], axis=0)
        meta = ImageMetadata(
            format='CpDOP_RGB',
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
