# -*- coding: utf-8 -*-
"""Dual-pol Radar Surface Index (DpRSI).

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
This implementation follows open-literature formulations using
Stokes-derived entropy and NESZ-aware gating for low-SNR behavior.
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
class DualPolRadarSurfaceIndex(DualPolDecompositionBase):
    """Dual-pol Radar Surface Index from entropy and Stokes normalization."""

    window_size: Annotated[int, Range(min=1, max=31),
                           Desc('Boxcar averaging window size')] = 7
    nesz_db: float = -16.0
    percentile_low: float = 2.0
    percentile_high: float = 98.0

    @property
    def component_names(self) -> Tuple[str, ...]:
        return ('dprsi',)

    def decompose_dual(self, s_co: np.ndarray, s_cross: np.ndarray) -> Dict[str, np.ndarray]:
        self._validate_dual_inputs(s_co, s_cross)
        c11, c12, c22 = self._compute_c2(s_co, s_cross, self.window_size)
        s0, s1, s2, s3, s1_norm, _, _ = self._normalized_stokes(
            c11,
            c12,
            c22,
            percentile_low=self.percentile_low,
            percentile_high=self.percentile_high,
        )

        tpp = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)
        l1 = (s0 + tpp) / 2.0
        l2 = (s0 - tpp) / 2.0
        lsum = l1 + l2
        eps = np.finfo(np.float64).tiny

        with np.errstate(divide='ignore', invalid='ignore'):
            p1 = l1 / np.where(np.abs(lsum) > eps, lsum, eps)
            p2 = l2 / np.where(np.abs(lsum) > eps, lsum, eps)
            ent = -(p1 * np.log2(p1) + p2 * np.log2(p2))
        ent = np.real(np.nan_to_num(ent, nan=0.0, posinf=0.0, neginf=0.0))

        root_term = np.sqrt(np.clip(1.0 - s1_norm ** 2, 0.0, 1.0))
        dprsi_valid = (1.0 - ent) * root_term
        dprsi_noise = root_term

        c11_db = 10.0 * np.log10(np.maximum(np.real(c11), np.finfo(np.float64).tiny))
        dprsi = np.where(c11_db > self.nesz_db, dprsi_valid, dprsi_noise)

        return {'dprsi': np.real(dprsi)}

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

        arr = np.clip(components['dprsi'], 0.0, 1.0)
        arr = self._percentile_stretch(arr, percentile_low, percentile_high)
        rgb = self._bands_to_rgb([arr, arr, arr], color_mode=color_mode, channel_keys=[list(components.keys())[0]] * 3)
        meta = ImageMetadata(
            format='DpRSI_RGB',
            rows=arr.shape[0],
            cols=arr.shape[1],
            bands=3,
            dtype='float32',
            axis_order='CYX',
            channel_metadata=[
                ChannelMetadata(index=0, name='dprsi', role='rgb_red'),
                ChannelMetadata(index=1, name='dprsi', role='rgb_green'),
                ChannelMetadata(index=2, name='dprsi', role='rgb_blue'),
            ],
        )
        return rgb, meta
