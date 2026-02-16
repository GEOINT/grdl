# -*- coding: utf-8 -*-
"""
CA-CFAR Detector - Cell-Averaging CFAR for homogeneous clutter.

Implements the classic Cell-Averaging CFAR (CA-CFAR) detector.  The
background is estimated as the mean of all training cells in an annular
ring around the cell under test (guard band excluded).  This is the
simplest and fastest CFAR variant, optimal for homogeneous clutter.

In heterogeneous environments (clutter edges, multiple targets in the
training window), consider ``GOCFARDetector``, ``SOCFARDetector``, or
``OSCFARDetector``.

Dependencies
------------
scipy

Author
------
Duane Smalley, PhD
duane.d.smalley@gmail.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-16

Modified
--------
2026-02-16
"""

# Standard library
from typing import Tuple

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.detection.cfar._base import (
    CFARDetector,
    _annular_stats,
)
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.vocabulary import DetectionType, ImageModality, ProcessorCategory


@processor_version('1.0.0')
@processor_tags(
    modalities=[ImageModality.SAR],
    category=ProcessorCategory.FIND_MAXIMA,
    description='Cell-Averaging CFAR detector for homogeneous clutter',
    detection_types=[DetectionType.PHENOMENON_SIGNATURE],
)
class CACFARDetector(CFARDetector):
    """Cell-Averaging CFAR (CA-CFAR) detector.

    Estimates the local background as the mean of training cells in an
    annular ring around the cell under test.  The annular mean and
    standard deviation are computed via uniform_filter subtraction,
    giving O(N) complexity regardless of window size.

    Best suited for scenes with **homogeneous clutter** where the
    background statistics are locally stationary.

    Parameters
    ----------
    guard_cells : int
        Guard band half-width in pixels.  Default 3.
    training_cells : int
        Training annulus half-width in pixels.  Default 12.
    pfa : float
        Probability of false alarm.  Default 1e-3.
    min_pixels : int
        Minimum cluster area in pixels.  Default 9.
    assumption : str
        Statistical model: ``'gaussian'`` or ``'exponential'``.
        Default ``'gaussian'``.

    Examples
    --------
    >>> from grdl.image_processing.detection.cfar import CACFARDetector
    >>> detector = CACFARDetector(pfa=1e-3, min_pixels=9)
    >>> detections = detector.detect(db_image)
    >>> len(detections)
    42
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _estimate_background(
        self,
        image: np.ndarray,
        guard_cells: int,
        training_cells: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate background via annular ring mean and std.

        Parameters
        ----------
        image : np.ndarray
            2D float64 image.
        guard_cells : int
            Guard band half-width.
        training_cells : int
            Training annulus half-width.

        Returns
        -------
        bg_mean : np.ndarray
            Annular ring mean.
        bg_std : np.ndarray
            Annular ring standard deviation.
        """
        return _annular_stats(image, guard_cells, training_cells)
