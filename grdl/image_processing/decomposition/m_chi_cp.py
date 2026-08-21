# -*- coding: utf-8 -*-
"""Compact-pol m-chi decomposition."""

from typing import Annotated, Dict, Tuple, TYPE_CHECKING

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
class CompactPolMChi(CompactPolDecompositionBase):
    """M-chi decomposition for compact-pol C2."""

    window_size: Annotated[int, Range(min=1, max=31), Desc('Boxcar averaging window size')] = 7
    chi: Annotated[float, Range(min=-90.0, max=90.0), Desc('Ellipticity angle of transmitted wave')] = 45.0
    psi: Annotated[float, Range(min=-90.0, max=90.0), Desc('Orientation angle of transmitted wave')] = 0.0

    @property
    def component_names(self) -> Tuple[str, ...]:
        return ('surface', 'double_bounce', 'volume', 'm_cp', 'chi_cp')

    def decompose_compact(self, c11, c12, c21, c22) -> Dict[str, np.ndarray]:
        self._validate_c2_inputs(c11, c12, c21, c22)
        s0, s1, s2, s3 = self._stokes_from_c2(c11, c12, c21, c22, self.chi, self.window_size)

        s0_abs = np.abs(s0)
        m = safe_divide(np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2), s0_abs)
        chi_rad = 0.5 * np.arcsin(np.clip(safe_divide(-s3, m * s0_abs), -1.0, 1.0))

        surface = np.sqrt(np.clip((m * s0_abs * (1.0 - np.sin(2.0 * chi_rad))) / 2.0, 0.0, None))
        double_bounce = np.sqrt(np.clip((m * s0_abs * (1.0 + np.sin(2.0 * chi_rad))) / 2.0, 0.0, None))
        volume = np.sqrt(np.clip(s0_abs * (1.0 - m), 0.0, None))

        return {
            'surface': np.real(np.nan_to_num(surface)),
            'double_bounce': np.real(np.nan_to_num(double_bounce)),
            'volume': np.real(np.nan_to_num(volume)),
            'm_cp': np.real(np.nan_to_num(m)),
            'chi_cp': np.degrees(np.real(chi_rad)),
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
            format='CpMChi_RGB',
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
