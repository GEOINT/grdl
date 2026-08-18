# -*- coding: utf-8 -*-
"""Dual-pol Shannon entropy decomposition from C2.

References
----------
Morio, J., Gini, F., Refregier, P., and Goudail, F. (2009),
"A Shannon entropy interpretation of the Shannon capacity,"
IEEE Signal Processing Letters, 16(3), pp.193-196.

Pottier, E. and Lee, J.S. (2009), "Unsupervised classification
scheme of POLSAR images based on the complex Wishart distribution
and H/A/Alpha polarimetric decomposition theorem."
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
class ShannonEntropyDP(DualPolDecompositionBase):
    """Shannon entropy components (total, intensity, polarimetric) for dual-pol."""

    window_size: Annotated[int, Range(min=1, max=31),
                           Desc('Boxcar averaging window size')] = 7

    @property
    def component_names(self) -> Tuple[str, ...]:
        return ('H_total', 'H_intensity', 'H_polarimetric')

    def decompose_dual(self, s_co: np.ndarray, s_cross: np.ndarray) -> Dict[str, np.ndarray]:
        self._validate_dual_inputs(s_co, s_cross)
        c11, c12, c22 = self._compute_c2(s_co, s_cross, self.window_size)

        det_c2 = c11 * c22 - np.abs(c12) ** 2
        i_term = np.real(c11 + c22)
        eps = 1e-8

        with np.errstate(divide='ignore', invalid='ignore'):
            dop_measure = 1.0 - 4.0 * det_c2 / (i_term ** 2 + eps)
        dop_measure = np.real(dop_measure)

        with np.errstate(divide='ignore', invalid='ignore'):
            hsp = np.where((1.0 - dop_measure) < eps,
                           np.nan,
                           np.log(np.abs(1.0 - dop_measure)))
        hsp[~np.isfinite(hsp)] = np.nan

        with np.errstate(divide='ignore', invalid='ignore'):
            hsi = 2.0 * np.log(np.e * np.pi * i_term / 2.0)
        hsi[~np.isfinite(hsi)] = np.nan

        hs = np.nansum(np.dstack((hsp, hsi)), axis=2)

        return {
            'H_total': np.real(hs),
            'H_intensity': np.real(hsi),
            'H_polarimetric': np.real(hsp),
        }

    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        del representation
        from grdl.IO.models.base import ImageMetadata, ChannelMetadata

        r = self._percentile_stretch(components['H_total'], percentile_low, percentile_high)
        g = self._percentile_stretch(components['H_intensity'], percentile_low, percentile_high)
        b = self._percentile_stretch(components['H_polarimetric'], percentile_low, percentile_high)
        rgb = np.stack([r, g, b], axis=0)

        meta = ImageMetadata(
            format='ShannonDP_RGB',
            rows=r.shape[0],
            cols=r.shape[1],
            bands=3,
            dtype='float32',
            axis_order='CYX',
            channel_metadata=[
                ChannelMetadata(index=0, name='H_total', role='rgb_red'),
                ChannelMetadata(index=1, name='H_intensity', role='rgb_green'),
                ChannelMetadata(index=2, name='H_polarimetric', role='rgb_blue'),
            ],
        )
        return rgb, meta
