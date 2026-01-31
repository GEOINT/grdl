# -*- coding: utf-8 -*-
"""
Image Processing Base Classes - Abstract interfaces for image transforms.

Defines abstract base classes for image transforms including geometric
transforms (orthorectification, reprojection) and radiometric transforms
(filtering, enhancement, normalization). Concrete implementations handle
different transform types in sub-modules.

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
2026-01-30

Modified
--------
2026-01-30
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ImageTransform(ABC):
    """
    Abstract base class for image transforms.

    Provides interface for transforms that take a source image array and
    produce a transformed output array. Covers both geometric transforms
    (orthorectification, reprojection) and radiometric transforms
    (filtering, enhancement).

    Subclasses implement ``apply`` which operates on numpy arrays.
    """

    @abstractmethod
    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Apply the transform to a source image array.

        Parameters
        ----------
        source : np.ndarray
            Input image. Shape depends on the specific transform:
            (rows, cols) for single-band, (bands, rows, cols) or
            (rows, cols, bands) for multi-band.

        Returns
        -------
        np.ndarray
            Transformed image.
        """
        ...
