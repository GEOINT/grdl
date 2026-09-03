# -*- coding: utf-8 -*-
"""Compact-pol m-chi decomposition."""

from typing import Annotated, Dict, List, Optional, TYPE_CHECKING, Tuple

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

    _RGB_CHANNELS = [
        ('double_bounce', 'rgb_red'),
        ('volume',        'rgb_green'),
        ('surface',       'rgb_blue'),
    ]

    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
        color_mode: str = 'standard',
        channels: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        """Create an RGB composite (R=double_bounce, G=volume, B=surface).

        Parameters
        ----------
        components : Dict[str, np.ndarray]
            Output of the decomposition.
        representation : str
            ``'db'`` (default), ``'power'``, or ``'magnitude'``.
        percentile_low : float
            Lower percentile for stretch. Default 2.0.
        percentile_high : float
            Upper percentile for stretch. Default 98.0.
        channels : list of str, optional
            Override which 3 component keys map to R, G, B (in that order).
            Defaults to ``['double_bounce', 'volume', 'surface']``.

        Returns
        -------
        tuple[np.ndarray, ImageMetadata]
            ``(rgb, metadata)`` — rgb shape ``(3, rows, cols)``, float32.
        """
        return self._build_power_rgb(
            components, self._RGB_CHANNELS, 'CpMChiRGB',
            representation, percentile_low, percentile_high, color_mode=color_mode, channels=channels,
        )
