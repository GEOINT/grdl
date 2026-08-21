# -*- coding: utf-8 -*-
"""
Dual-Pol H/Alpha Decomposition - Eigenvalue decomposition of the 2x2
coherency matrix for dual-polarization SAR data.

References
----------
Cloude, S.R. and Pottier, E. (1997), "An entropy based classification
for land applications of polarimetric SAR," IEEE Trans. Geoscience and
Remote Sensing, 35(1), pp.68-78.

Lee, J.S. and Pottier, E. (2009), Polarimetric Radar Imaging:
From Basics to Applications. CRC Press.
"""

from typing import Annotated, Dict, Tuple, TYPE_CHECKING

import numpy as np

from grdl.image_processing.decomposition.dual_pol_base import DualPolDecompositionBase
from grdl.image_processing.decomposition.h_a_alpha_base import HAalphaBase
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.image_processing.params import Range, Desc
from grdl.vocabulary import ImageModality

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata


@processor_version('1.0.0')
@processor_tags(modalities=[ImageModality.SAR])
class DualPolHAlpha(DualPolDecompositionBase, HAalphaBase):
    """Dual-pol H/Alpha eigenvalue decomposition."""

    window_size: Annotated[int, Range(min=3, max=31),
                           Desc('Boxcar averaging window size')] = 7

    @property
    def component_names(self) -> Tuple[str, str, str, str]:
        return ('entropy', 'alpha', 'anisotropy', 'span')

    def decompose_dual(
        self,
        s_co: np.ndarray,
        s_cross: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        self._validate_dual_inputs(s_co, s_cross)

        c11, c12, c22 = self._compute_c2(s_co, s_cross, self.window_size)
        c12_mag2 = np.abs(c12) ** 2

        trace = c11 + c22
        det = c11 * c22 - c12_mag2
        disc = np.sqrt(np.maximum(trace ** 2 - 4.0 * det, 0.0))
        lam1 = (trace + disc) * 0.5
        lam2 = (trace - disc) * 0.5

        np.maximum(lam1, 0.0, out=lam1)
        np.maximum(lam2, 0.0, out=lam2)

        span = lam1 + lam2
        safe_span = np.where(span > 0.0, span, 1.0)
        p1 = lam1 / safe_span
        p2 = lam2 / safe_span

        entropy = np.zeros_like(span)
        mask1 = p1 > 0.0
        mask2 = p2 > 0.0
        entropy[mask1] -= p1[mask1] * np.log2(p1[mask1])
        entropy[mask2] -= p2[mask2] * np.log2(p2[mask2])
        np.clip(entropy, 0.0, 1.0, out=entropy)

        c12_abs = np.sqrt(c12_mag2)
        norm1 = np.sqrt(c12_mag2 + (lam1 - c11) ** 2)
        norm2 = np.sqrt(c12_mag2 + (lam2 - c11) ** 2)

        safe_norm1 = np.where(norm1 > 0.0, norm1, 1.0)
        safe_norm2 = np.where(norm2 > 0.0, norm2, 1.0)

        alpha1 = np.arccos(np.clip(c12_abs / safe_norm1, 0.0, 1.0))
        alpha2 = np.arccos(np.clip(c12_abs / safe_norm2, 0.0, 1.0))

        alpha1 = np.where(norm1 > 0.0, alpha1, 0.0)
        alpha2 = np.where(norm2 > 0.0, alpha2, 0.0)

        alpha = np.degrees(p1 * alpha1 + p2 * alpha2)

        anisotropy = np.where(span > 0.0, disc / safe_span, 0.0)
        np.clip(anisotropy, 0.0, 1.0, out=anisotropy)

        return {
            'entropy': entropy,
            'alpha': alpha,
            'anisotropy': anisotropy,
            'span': span,
        }

    @classmethod
    def rgb_channel_metadata(
        cls,
        alpha_low_deg: float = 10.0,
        alpha_high_deg: float = 80.0,
    ) -> list:
        from grdl.IO.models.base import ChannelMetadata

        return [
            ChannelMetadata(
                index=0,
                name='entropy',
                role='decomposition',
                extras={
                    'halpha_component': 'entropy',
                    'formula': 'H in [0, 1]',
                    'display': 'Red',
                },
            ),
            ChannelMetadata(
                index=1,
                name='alpha_norm',
                role='decomposition',
                extras={
                    'halpha_component': 'alpha',
                    'formula': (
                        f'clip((alpha-{alpha_low_deg:g})/'
                        f'({alpha_high_deg:g}-{alpha_low_deg:g}), 0, 1)'
                    ),
                    'display': 'Green',
                },
            ),
            ChannelMetadata(
                index=2,
                name='span_db',
                role='decomposition',
                extras={
                    'halpha_component': 'span',
                    'formula': '10*log10(span), 2-98% stretch',
                    'display': 'Blue',
                },
            ),
        ]

    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
        alpha_low_deg: float = 10.0,
        alpha_high_deg: float = 80.0,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        del representation
        from grdl.IO.models.base import ImageMetadata

        required = {'entropy', 'alpha', 'span'}
        missing = required - set(components.keys())
        if missing:
            raise ValueError(
                f"Missing component keys: {missing}. Expected keys: {required}"
            )

        if alpha_high_deg <= alpha_low_deg:
            raise ValueError(
                "alpha_high_deg must be greater than alpha_low_deg for RGB scaling."
            )

        r = np.clip(components['entropy'], 0.0, 1.0).astype(np.float32)
        g = np.clip(
            (components['alpha'] - alpha_low_deg) / (alpha_high_deg - alpha_low_deg),
            0.0,
            1.0,
        ).astype(np.float32)

        span = components['span']
        span_db = 10.0 * np.log10(np.maximum(span, np.finfo(np.float64).tiny))
        b = self._percentile_stretch(span_db, percentile_low, percentile_high)

        rgb = np.stack([r, g, b], axis=0)
        metadata = ImageMetadata(
            format='HAlphaRGB',
            rows=int(rgb.shape[1]),
            cols=int(rgb.shape[2]),
            dtype=str(rgb.dtype),
            bands=3,
            axis_order='CYX',
            channel_metadata=self.rgb_channel_metadata(alpha_low_deg, alpha_high_deg),
        )
        return rgb, metadata

    def __repr__(self) -> str:
        return f"DualPolHAlpha(window_size={self.window_size})"
