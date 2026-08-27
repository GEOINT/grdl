# -*- coding: utf-8 -*-
"""Compact-pol model-free 3-component decomposition (MF3CC)."""

from typing import Annotated, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from grdl.image_processing.decomposition.compact_pol_base import CompactPolDecompositionBase
from grdl.image_processing.decomposition.compact_pol_utils import safe_divide
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.image_processing.params import Range, Desc
from grdl.vocabulary import ImageModality

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata


@processor_version('1.0.1')
@processor_tags(modalities=[ImageModality.SAR])
class CompactPolModelFree3C(CompactPolDecompositionBase):
    """Model-free 3-component compact-pol decomposition."""

    window_size: Annotated[int, Range(min=1, max=31), Desc('Boxcar averaging window size')] = 7
    chi: Annotated[float, Range(min=-90.0, max=90.0), Desc('Ellipticity angle of transmitted wave')] = 45.0
    psi: Annotated[float, Range(min=-90.0, max=90.0), Desc('Orientation angle of transmitted wave')] = 0.0

    @property
    def component_names(self) -> Tuple[str, ...]:
        return ('surface', 'double_bounce', 'volume', 'theta_cp')

    def decompose_compact(self, c11, c12, c21, c22) -> Dict[str, np.ndarray]:
        self._validate_c2_inputs(c11, c12, c21, c22)
        # Apply spatial averaging up front so that det, trace, and m are all
        # computed from the same windowed data.  Previously det/trace used the
        # raw (unwindowed) inputs while s0/s3 came from _stokes_from_c2 which
        # applies boxcar internally — that mismatch caused m ≈ 1 (single-pixel
        # C2 is rank-1 → det ≈ 0) and therefore pv ≈ 0.
        c11 = self._boxcar_complex(c11, self.window_size)
        c12 = self._boxcar_complex(c12, self.window_size)
        c21 = self._boxcar_complex(c21, self.window_size)
        c22 = self._boxcar_complex(c22, self.window_size)
        s0, _, _, s3 = self._stokes_from_c2(c11, c12, c21, c22, self.chi, 1)

        det = c11 * c22 - c12 * c21
        trace = c11 + c22
        m = np.sqrt(np.clip(np.real(1.0 - 4.0 * safe_divide(det, trace ** 2)), 0.0, 1.0))

        c2_trace = np.real(trace)
        h = ((s0 + s3) / 2.0) - ((s0 - s3) / 2.0)
        sc = (s0 - s3) / 2.0
        oc = (s0 + s3) / 2.0
        denom = sc * oc + (m ** 2) * (s0 ** 2)
        theta = np.arctan(safe_divide(m * s0 * h, denom))

        ps = np.clip(m * c2_trace * (1.0 + np.sin(2.0 * theta)) / 2.0, 0.0, None)
        pd = np.clip(m * c2_trace * (1.0 - np.sin(2.0 * theta)) / 2.0, 0.0, None)
        pv = np.clip(c2_trace * (1.0 - m), 0.0, None)

        return {
            'surface': np.real(np.nan_to_num(ps)),
            'double_bounce': np.real(np.nan_to_num(pd)),
            'volume': np.real(np.nan_to_num(pv)),
            'theta_cp': np.degrees(np.real(theta)),
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
        """Create an RGB composite from MF3CC components.

        Standard convention (same as Freeman-Durden):

        - **Red**: Double-bounce (Pd)
        - **Green**: Volume (Pv)
        - **Blue**: Surface (Ps)

        Parameters
        ----------
        components : Dict[str, np.ndarray]
            Output of ``decompose_compact()``.
        representation : str
            ``'db'`` (default), ``'power'``, or ``'magnitude'``.
        percentile_low : float
            Lower percentile for stretch. Default 2.0.
        percentile_high : float
            Upper percentile for stretch. Default 98.0.
        channels : list of str, optional
            Override which 3 component keys map to R, G, B (in that order).
            E.g. ``channels=['surface', 'volume', 'double_bounce']``.
            Defaults to ``['double_bounce', 'volume', 'surface']``.

        Returns
        -------
        tuple[np.ndarray, ImageMetadata]
            ``(rgb, metadata)`` — rgb shape ``(3, rows, cols)``, float32.
        """
        return self._build_power_rgb(
            components, self._RGB_CHANNELS, 'CpMF3CCRGB',
            representation, percentile_low, percentile_high, channels=channels,
        )

