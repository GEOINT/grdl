# -*- coding: utf-8 -*-
"""
SO-CFAR Detector - Smallest-Of CFAR for clutter edge sensitivity.

Implements the Smallest-Of CFAR (SO-CFAR) detector.  The background
level is the minimum of four quadrant means surrounding the cell under
test.  By selecting the smallest local estimate, SO-CFAR lowers the
threshold at clutter edges, increasing sensitivity to targets near
clutter transitions at the cost of a higher false alarm rate.

Trade-off: SO-CFAR is more aggressive than CA-CFAR — better detection
at clutter boundaries, but more false alarms in transition zones.

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
    _quadrant_means,
)
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.vocabulary import DetectionType, ImageModality, ProcessorCategory


@processor_version('1.0.0')
@processor_tags(
    modalities=[ImageModality.SAR],
    category=ProcessorCategory.FIND_MAXIMA,
    description='Smallest-Of CFAR detector for clutter edge sensitivity',
    detection_types=[DetectionType.PHENOMENON_SIGNATURE],
)
class SOCFARDetector(CFARDetector):
    """Smallest-Of CFAR (SO-CFAR) detector.

    Splits the training annulus into four corner quadrants and takes the
    **minimum** of their means as the background level.  This lowers the
    detection threshold at clutter edges, improving sensitivity to
    targets near clutter transitions.

    The standard deviation is still computed from the full annular ring
    (via ``_annular_stats``) for use with the Gaussian threshold model.

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
    >>> from grdl.image_processing.detection.cfar import SOCFARDetector
    >>> detector = SOCFARDetector(pfa=1e-3)
    >>> detections = detector.detect(db_image)
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _estimate_background(
        self,
        image: np.ndarray,
        guard_cells: int,
        training_cells: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate background as min of four quadrant means.

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
            Minimum of four quadrant means.
        bg_std : np.ndarray
            Full annular ring standard deviation.
        """
        _, bg_std = _annular_stats(image, guard_cells, training_cells)
        quads = _quadrant_means(image, guard_cells, training_cells)
        bg_mean = np.minimum.reduce(quads)
        return bg_mean, bg_std
