# -*- coding: utf-8 -*-
"""Compact-pol improved S-Omega decomposition."""

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
class CompactPolSOmega(CompactPolDecompositionBase):
    """Improved S-Omega decomposition for compact-pol C2."""

    window_size: Annotated[int, Range(min=1, max=31), Desc('Boxcar averaging window size')] = 7
    chi: Annotated[float, Range(min=-90.0, max=90.0), Desc('Ellipticity angle of transmitted wave')] = 45.0
    psi: Annotated[float, Range(min=-90.0, max=90.0), Desc('Orientation angle of transmitted wave')] = 0.0

    @property
    def component_names(self) -> Tuple[str, ...]:
        return ('surface', 'double_bounce', 'volume')

    def decompose_compact(self, c11, c12, c21, c22) -> Dict[str, np.ndarray]:
        self._validate_c2_inputs(c11, c12, c21, c22)
        s0, s1, s2, s3 = self._stokes_from_c2(c11, c12, c21, c22, self.chi, self.window_size)

        s0_abs = np.abs(s0)
        sc = (s0_abs - s3) / 2.0
        oc = (s0_abs + s3) / 2.0
        cpr = safe_divide(sc, oc)

        dop = safe_divide(np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2), s0_abs)
        psi_angle = 0.5 * np.degrees(np.arctan2(s2, s1))
        docp = np.clip(safe_divide(-s3, dop * s0_abs), -1.0, 1.0)
        chi_angle = 0.5 * np.degrees(np.arcsin(docp))

        chi_r = np.radians(self.chi)
        psi_r = np.radians(self.psi)
        chi_img = np.radians(chi_angle)
        psi_img = np.radians(psi_angle)

        x1 = np.cos(2 * chi_r) * np.cos(2 * psi_r) * np.cos(2 * chi_img) * np.cos(2 * psi_img)
        x2 = np.cos(2 * chi_r) * np.sin(2 * psi_r) * np.cos(2 * chi_img) * np.sin(2 * psi_img)
        x3 = np.abs(np.sin(2 * chi_r) * np.sin(2 * chi_img))
        prec = dop * (1.0 + x1 + x2 + x3)
        prec1 = (1.0 - dop) + dop * (1.0 + x1 + x2 + x3)
        omega = safe_divide(prec, prec1)

        surface = np.zeros_like(s0_abs, dtype=np.float64)
        double_bounce = np.zeros_like(s0_abs, dtype=np.float64)

        g1 = cpr > 1.0
        l1 = cpr < 1.0
        e1 = ~(g1 | l1)

        surface[g1] = omega[g1] * (1.0 - omega[g1]) * oc[g1]
        double_bounce[g1] = omega[g1] * s0_abs[g1] - omega[g1] * (1.0 - omega[g1]) * oc[g1]

        surface[l1] = omega[l1] * s0_abs[l1] - omega[l1] * (1.0 - omega[l1]) * sc[l1]
        double_bounce[l1] = omega[l1] * (1.0 - omega[l1]) * sc[l1]

        surface[e1] = omega[e1] * oc[e1]
        double_bounce[e1] = omega[e1] * sc[e1]

        volume = np.clip((1.0 - omega) * s0_abs, 0.0, None)

        return {
            'surface': np.real(np.nan_to_num(surface)),
            'double_bounce': np.real(np.nan_to_num(double_bounce)),
            'volume': np.real(np.nan_to_num(volume)),
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
            components, self._RGB_CHANNELS, 'CpSOmegaRGB',
            representation, percentile_low, percentile_high, channels=channels,
        )
