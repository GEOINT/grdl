# -*- coding: utf-8 -*-
"""
GO-CFAR Detector - Greatest-Of CFAR for clutter edge suppression.

Implements the Greatest-Of CFAR (GO-CFAR) detector.  The background
level is the maximum of four quadrant means surrounding the cell under
test.  By selecting the largest local estimate, GO-CFAR raises the
threshold at clutter edges, suppressing false alarms where CA-CFAR
would see a low-clutter quadrant and declare a false detection.

Trade-off: GO-CFAR is more conservative than CA-CFAR — fewer false
alarms at clutter boundaries, but may miss weak targets near edges.

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
    description='Greatest-Of CFAR detector for clutter edge suppression',
    detection_types=[DetectionType.PHENOMENON_SIGNATURE],
)
class GOCFARDetector(CFARDetector):
    """Greatest-Of CFAR (GO-CFAR) detector.

    Splits the training annulus into four corner quadrants and takes the
    **maximum** of their means as the background level.  This raises the
    detection threshold at clutter edges, reducing false alarms where
    CA-CFAR would average low-clutter and high-clutter regions together.

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
    >>> from grdl.image_processing.detection.cfar import GOCFARDetector
    >>> detector = GOCFARDetector(pfa=1e-3)
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
        """Estimate background as max of four quadrant means.

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
            Maximum of four quadrant means.
        bg_std : np.ndarray
            Full annular ring standard deviation.
        """
        _, bg_std = _annular_stats(image, guard_cells, training_cells)
        quads = _quadrant_means(image, guard_cells, training_cells)
        bg_mean = np.maximum.reduce(quads)
        return bg_mean, bg_std
