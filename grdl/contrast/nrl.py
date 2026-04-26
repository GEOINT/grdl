# -*- coding: utf-8 -*-
"""
Noise Relative Level (NRL) contrast stretch.

Direct port of ``sarpy.visualization.remap.NRL``: linear up to a
``changeover`` value (typically the 99th percentile of amplitude),
then ``log2`` from there to the maximum.  The two-segment curve keeps
faint detail crisp while compressing bright outliers.

Output is ``float32`` in ``[0, 1]``.

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-04-26

Modified
--------
2026-04-26
"""

# Standard library
import logging
from typing import Annotated, Any, Optional, Tuple

# Third-party
import numpy as np

# GRDL internal
from grdl.contrast.base import linear_map, nan_safe_stats
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC

logger = logging.getLogger(__name__)


@processor_version('1.0.0')
@processor_tags(
    modalities=[IM.SAR],
    category=PC.ENHANCE,
    description='Linear-to-log knee remap (sarpy NRL).',
)
class NRLStretch(ImageTransform):
    """Linear-then-logarithmic SAR contrast remap.

    For ``x <= changeover``: linear stretch to ``[0, knee]``.
    For ``x > changeover``:  ``log2`` stretch to ``[knee, 1]``.

    The default ``knee=0.8`` (i.e. 80% of full output range) and
    ``percentile=99`` reproduce sarpy's defaults exactly.

    Parameters
    ----------
    knee : float, default=0.8
        Output value where the linear segment ends and the log segment
        begins.  Strictly between 0 and 1.
    percentile : float, default=99.0
        When ``stats`` is not supplied, this percentile of the amplitude
        is used as the linear→log changeover.
    """

    knee: Annotated[
        float, Range(min=0.001, max=0.999),
        Desc('Knee location in [0, 1] output'),
    ] = 0.8
    percentile: Annotated[
        float, Range(min=0.0, max=100.0),
        Desc('Percentile for linear→log changeover'),
    ] = 99.0

    def __init__(
        self,
        knee: float = 0.8,
        percentile: float = 99.0,
    ) -> None:
        if not (0.0 < knee < 1.0):
            raise ValueError(f"knee must be in (0, 1), got {knee}")
        if not (0.0 < percentile < 100.0):
            raise ValueError(
                f"percentile must be in (0, 100), got {percentile}"
            )
        self.knee = knee
        self.percentile = percentile

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply NRL remap.

        Parameters
        ----------
        source : np.ndarray
            Complex or real amplitude data.
        **kwargs
            ``stats`` — optional ``(min, max, changeover)`` triple
            pre-computed on a representative sample for tile-consistent
            display.  When omitted, the triple is computed on this call's
            input using ``percentile``.

        Returns
        -------
        np.ndarray
            ``float32`` in ``[0, 1]``.
        """
        amplitude = np.abs(source) if np.iscomplexobj(source) else source
        amplitude = np.asarray(amplitude)
        params = self._resolve_params(kwargs)

        stats: Optional[Tuple[float, float, float]] = kwargs.get('stats')
        if stats is None:
            stats = nan_safe_stats(amplitude, params['percentile'])
        amp_min, amp_max, changeover = stats
        if not (amp_min <= changeover <= amp_max):
            raise ValueError(
                f"Inconsistent stats: ({amp_min}, {amp_max}, {changeover})"
            )

        out = np.empty(amplitude.shape, dtype=np.float64)
        if amp_min == amp_max:
            out[:] = 0.0
            return out.astype(np.float32)

        knee = params['knee']
        linear_region = (amplitude <= changeover)

        if changeover > amp_min:
            out[linear_region] = np.clip(
                knee * linear_map(amplitude[linear_region], amp_min, changeover),
                0.0, 1.0,
            )
        else:
            logger.warning(
                "NRL: amplitude is approximately constant; output may "
                "look strange."
            )
            out[linear_region] = 0.0

        if changeover == amp_max:
            out[~linear_region] = knee
        else:
            high = np.clip(amplitude[~linear_region], changeover, amp_max)
            log_in = (high - changeover) / (amp_max - changeover) + 1.0
            out[~linear_region] = np.log2(log_in) * (1.0 - knee) + knee

        return np.clip(out, 0.0, 1.0).astype(np.float32)
