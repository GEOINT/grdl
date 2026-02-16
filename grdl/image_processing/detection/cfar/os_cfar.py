# -*- coding: utf-8 -*-
"""
OS-CFAR Detector - Ordered-Statistics CFAR robust to interfering targets.

Implements the Ordered-Statistics CFAR (OS-CFAR) detector.  The
background level is the *k*-th percentile of training cells, rather
than the mean.  This makes OS-CFAR robust to interfering targets in
the training window: a few bright targets do not pull up the background
estimate as they would with CA-CFAR.

Trade-off: OS-CFAR uses ``scipy.ndimage.percentile_filter`` which is
O(N * k²) per pixel (k = window side length), slower than the O(N)
uniform_filter used by CA/GO/SO-CFAR.  For a 5000x5000 image with
``training_cells=12`` (window 25x25), expect roughly 4-8x slower than
CA-CFAR.

The guard band exclusion is approximated by adjusting the percentile:
``k_adj = percentile * outer_count / train_count``, clamped to
[1.0, 99.9].  This approximation is accurate when the guard/training
ratio is below ~0.3.

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
from typing import Annotated, Tuple

# Third-party
import numpy as np
from scipy.ndimage import percentile_filter

# GRDL internal
from grdl.image_processing.detection.cfar._base import (
    CFARDetector,
    _annular_stats,
)
from grdl.image_processing.detection.cfar._validation import (
    validate_assumption,
    validate_cfar_window,
    validate_pfa,
)
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.vocabulary import DetectionType, ImageModality, ProcessorCategory


@processor_version('1.0.0')
@processor_tags(
    modalities=[ImageModality.SAR],
    category=ProcessorCategory.FIND_MAXIMA,
    description='Ordered-Statistics CFAR detector robust to interfering targets',
    detection_types=[DetectionType.PHENOMENON_SIGNATURE],
)
class OSCFARDetector(CFARDetector):
    """Ordered-Statistics CFAR (OS-CFAR) detector.

    Uses the *k*-th percentile of training cells as the background
    estimate instead of the mean.  This makes the detector robust to
    **interfering targets** in the training window: a few bright
    pixels do not inflate the background level and mask weaker targets.

    The ``percentile`` parameter controls the rank statistic:

    - Lower values (e.g., 50) → more aggressive, lower threshold
    - Higher values (e.g., 90) → more conservative, higher threshold
    - The default of 75 balances robustness and sensitivity

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
    percentile : float
        Percentile rank (1-99) for background estimation from the
        training window.  Default 75.0.

    Examples
    --------
    >>> from grdl.image_processing.detection.cfar import OSCFARDetector
    >>> detector = OSCFARDetector(percentile=75.0, pfa=1e-3)
    >>> detections = detector.detect(db_image)
    """

    percentile: Annotated[float, Range(min=1.0, max=99.0),
                          Desc('Percentile rank for background estimate')] = 75.0

    def __init__(
        self,
        guard_cells: int = 3,
        training_cells: int = 12,
        pfa: float = 1e-3,
        min_pixels: int = 9,
        assumption: str = 'gaussian',
        percentile: float = 75.0,
    ) -> None:
        validate_cfar_window(guard_cells, training_cells)
        validate_pfa(pfa)
        validate_assumption(assumption)
        self.guard_cells = guard_cells
        self.training_cells = training_cells
        self.pfa = pfa
        self.min_pixels = min_pixels
        self.assumption = assumption
        self.percentile = percentile

    def _estimate_background(
        self,
        image: np.ndarray,
        guard_cells: int,
        training_cells: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate background via k-th percentile of training window.

        Uses ``scipy.ndimage.percentile_filter`` on the full outer
        window.  The guard band exclusion is approximated by adjusting
        the percentile upward to account for the guard cells.

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
            k-th percentile of training window (used as background level).
        bg_std : np.ndarray
            Full annular ring standard deviation (from ``_annular_stats``).
        """
        params = self._resolve_params({})
        pct = params['percentile']

        outer_size = 2 * training_cells + 1
        guard_size = 2 * guard_cells + 1
        outer_count = outer_size * outer_size
        train_count = outer_count - guard_size * guard_size

        # Adjust percentile for guard band exclusion
        k_adj = float(np.clip(pct * outer_count / train_count, 1.0, 99.9))

        bg_mean = percentile_filter(
            image, percentile=k_adj, size=outer_size, mode='reflect',
        )
        _, bg_std = _annular_stats(image, guard_cells, training_cells)

        return bg_mean, bg_std
