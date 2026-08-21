# -*- coding: utf-8 -*-
"""Dual-pol model-free 3-component decomposition (MF3CD).

References
----------
Dey, S., Bhattacharya, A., Ratha, D., Mandal, D., and Frery, A.C.
(2020), "Target characterization and scattering power decomposition
for full and compact polarimetric SAR data," IEEE Geoscience and
Remote Sensing Letters, 18(6), pp.1048-1052.

Dey, S., Bhattacharya, A., Ratha, D., Mandal, D., McNairn, H.,
Lopez-Sanchez, J.M., and Rao, Y.S. (2021), "Model-free four component
scattering power decomposition for polarimetric SAR data," IEEE Journal
of Selected Topics in Applied Earth Observations and Remote Sensing, 14,
pp.3887-3898.
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
class ModelFree3CD(DualPolDecompositionBase):
    """Model-free dual-pol 3-component decomposition from T2."""

    window_size: Annotated[int, Range(min=1, max=31),
                           Desc('Boxcar averaging window size')] = 7

    @property
    def component_names(self) -> Tuple[str, ...]:
        return ('surface', 'double_bounce', 'volume', 'theta_dp')

    def decompose_dual(self, s_co: np.ndarray, s_cross: np.ndarray) -> Dict[str, np.ndarray]:
        self._validate_dual_inputs(s_co, s_cross)
        t11, t12, t22 = self._compute_t2(s_co, s_cross, self.window_size)

        det_t2 = t11 * t22 - np.abs(t12) ** 2
        trace_t2 = t11 + t22
        eps = np.finfo(np.float64).tiny

        with np.errstate(divide='ignore', invalid='ignore'):
            m1 = np.sqrt(np.clip(np.real(1.0 - 4.0 * (det_t2 / (trace_t2 ** 2 + eps))), 0.0, 1.0))

        h = np.real(t11 - t22)
        g = np.real(t22)
        span = np.real(trace_t2)

        denom = np.real(t11) * g + m1 ** 2 * span ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            val = (m1 * span * h) / np.where(np.abs(denom) > eps, denom, eps)
            theta = np.arctan(val)

        theta_dp = np.degrees(theta)

        ps = m1 * span * (1.0 + np.sin(2.0 * theta)) / 2.0
        pd = m1 * span * (1.0 - np.sin(2.0 * theta)) / 2.0
        pv = span * (1.0 - m1)

        return {
            'surface': np.real(ps),
            'double_bounce': np.real(pd),
            'volume': np.real(pv),
            'theta_dp': np.real(theta_dp),
        }

    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        from grdl.IO.models.base import ImageMetadata, ChannelMetadata

        r = self._percentile_stretch(
            self._apply_power_representation(components['double_bounce'], representation),
            percentile_low,
            percentile_high,
        )
        g = self._percentile_stretch(
            self._apply_power_representation(components['volume'], representation),
            percentile_low,
            percentile_high,
        )
        b = self._percentile_stretch(
            self._apply_power_representation(components['surface'], representation),
            percentile_low,
            percentile_high,
        )
        rgb = np.stack([r, g, b], axis=0)

        meta = ImageMetadata(
            format='MF3CD_RGB',
            rows=r.shape[0],
            cols=r.shape[1],
            bands=3,
            dtype='float32',
            axis_order='CYX',
            channel_metadata=[
                ChannelMetadata(index=0, name='double_bounce', role='rgb_red'),
                ChannelMetadata(index=1, name='volume', role='rgb_green'),
                ChannelMetadata(index=2, name='surface', role='rgb_blue'),
            ],
        )
        return rgb, meta
