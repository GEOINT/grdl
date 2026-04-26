# -*- coding: utf-8 -*-
"""
Display contrast adjustment for GRDL imagery.

Multi-modal view-time contrast operators for SAR, MSI, EO, PAN, and HSI
imagery.  Every operator is an ``ImageTransform`` subclass with the
standard GRDL ``@processor_version`` / ``@processor_tags`` metadata —
discoverable via the same machinery that powers other ``image_processing``
processors.

All operators share a uniform contract:

- Input: real or complex ``np.ndarray``.  Complex inputs have their
  magnitude taken automatically.
- Output: ``np.ndarray`` of dtype ``float32`` in ``[0, 1]`` — directly
  consumable by ``matplotlib.pyplot.imshow(cmap=...)``.
- Cross-tile consistency: pre-computed stats (``data_mean``,
  ``min_value``/``max_value``, ``stats=(min, max, changeover)``) can be
  passed as ``apply()`` kwargs to keep brightness uniform across chips
  of a larger scene.

For uint8 / uint16 quantization at save time, use
:func:`grdl.contrast.base.clip_cast`.

Operator catalog
----------------
SAR (sarpy.visualization.remap ports)
    :class:`MangisDensity`, :class:`Brighter`, :class:`Darker`,
    :class:`HighContrast`, :class:`GDM`, :class:`PEDF`,
    :class:`NRLStretch`, :class:`LogStretch`, :class:`ToDecibels`

Generic (any modality)
    :class:`LinearStretch`, :class:`PercentileStretch`,
    :class:`GammaCorrection`, :class:`SigmoidStretch`,
    :class:`HistogramEqualization`, :class:`CLAHE`

Examples
--------
SAR display with cross-tile consistency:

>>> from grdl.contrast import MangisDensity
>>> import numpy as np
>>>
>>> remap = MangisDensity()
>>> mean  = float(np.mean(np.abs(full_image)))
>>> tile_a_disp = remap.apply(tile_a, data_mean=mean)
>>> tile_b_disp = remap.apply(tile_b, data_mean=mean)

Composed multi-step stretch (no Pipeline required):

>>> from grdl.contrast import PercentileStretch, GammaCorrection
>>> stretched = PercentileStretch(plow=2.0, phigh=98.0).apply(eo_image)
>>> bright    = GammaCorrection(gamma=1.4).apply(stretched)

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

from grdl.contrast.auto import auto_select
from grdl.contrast.base import clip_cast, linear_map, nan_safe_stats
from grdl.contrast.linear import LinearStretch
from grdl.contrast.logarithmic import LogStretch
from grdl.contrast.density import (
    MangisDensity, Brighter, Darker, HighContrast, GDM, PEDF,
)
from grdl.contrast.nrl import NRLStretch
from grdl.contrast.gamma import GammaCorrection
from grdl.contrast.sigmoid import SigmoidStretch
from grdl.contrast.histogram import HistogramEqualization, CLAHE
from grdl.contrast.percentile import PercentileStretch
from grdl.contrast.decibel import ToDecibels

__all__ = [
    # Helpers
    'auto_select',
    'clip_cast',
    'linear_map',
    'nan_safe_stats',
    # SAR (sarpy ports)
    'MangisDensity',
    'Brighter',
    'Darker',
    'HighContrast',
    'GDM',
    'PEDF',
    'NRLStretch',
    'LogStretch',
    'ToDecibels',
    # Generic
    'LinearStretch',
    'PercentileStretch',
    'GammaCorrection',
    'SigmoidStretch',
    'HistogramEqualization',
    'CLAHE',
]
