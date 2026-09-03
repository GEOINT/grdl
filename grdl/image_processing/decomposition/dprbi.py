# -*- coding: utf-8 -*-
"""Dual-pol Radar Built-up Index (DpRBI).

References
----------
Bhogapurapu, N., Dey, S., Bhattacharya, A., Mandal, D.,
Lopez-Sanchez, J.M., McNairn, H., Lopez-Martinez, C., and Rao, Y.S.
(2021), "Dual-polarimetric descriptors from Sentinel-1 GRD SAR data
for crop growth assessment," ISPRS Journal of Photogrammetry and
Remote Sensing, 178, pp.20-35. doi:10.1016/j.isprsjprs.2021.05.013.

Verma, A., Bhattacharya, A., Dey, S., and Marino, A. (2024),
"Enhanced Target Characterization with Dual-Pol Sentinel-1 SAR Data,"
in IGARSS 2024 - 2024 IEEE International Geoscience and Remote
Sensing Symposium, pp.11461-11464.
doi:10.1109/IGARSS53475.2024.10642506.

Notes
-----
This implementation follows the open-formulation pathway (Stokes-domain
normalization and Euclidean index construction described in the
dual-polarimetric descriptor literature and validated against inlined
reference equations in GRDL tests.
"""

from typing import Annotated, Dict, Tuple, List, Optional, TYPE_CHECKING

import numpy as np

from grdl.image_processing.decomposition.dual_pol_base import DualPolDecompositionBase
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.image_processing.params import Range, Desc
from grdl.vocabulary import ImageModality

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata


@processor_version('1.0.0')
@processor_tags(modalities=[ImageModality.SAR])
class DualPolRadarBuiltUpIndex(DualPolDecompositionBase):
    """Dual-pol Radar Built-up Index from normalized Stokes terms."""

    window_size: Annotated[int, Range(min=1, max=31),
                           Desc('Boxcar averaging window size')] = 7
    percentile_low: float = 2.0
    percentile_high: float = 98.0

    @property
    def component_names(self) -> Tuple[str, ...]:
        return ('dprbi',)

    def decompose_dual(self, s_co: np.ndarray, s_cross: np.ndarray) -> Dict[str, np.ndarray]:
        self._validate_dual_inputs(s_co, s_cross)
        c11, c12, c22 = self._compute_c2(s_co, s_cross, self.window_size)
        _, _, _, _, s1_norm, s2_norm, s3_norm = self._normalized_stokes(
            c11,
            c12,
            c22,
            percentile_low=self.percentile_low,
            percentile_high=self.percentile_high,
        )

        dprbi = np.sqrt(s1_norm ** 2 + s2_norm ** 2 + s3_norm ** 2) / np.sqrt(3.0)
        return {'dprbi': dprbi}

    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
        color_mode: str = 'standard',
        channels: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        del representation
        from grdl.IO.models.base import ImageMetadata, ChannelMetadata

        arr = np.clip(components['dprbi'], 0.0, 1.0)
        arr = self._percentile_stretch(arr, percentile_low, percentile_high)
        rgb = self._bands_to_rgb([arr, arr, arr], color_mode=color_mode, channel_keys=[list(components.keys())[0]] * 3)
        meta = ImageMetadata(
            format='DpRBI_RGB',
            rows=arr.shape[0],
            cols=arr.shape[1],
            bands=3,
            dtype='float32',
            axis_order='CYX',
            channel_metadata=[
                ChannelMetadata(index=0, name='dprbi', role='rgb_red'),
                ChannelMetadata(index=1, name='dprbi', role='rgb_green'),
                ChannelMetadata(index=2, name='dprbi', role='rgb_blue'),
            ],
        )
        return rgb, meta
