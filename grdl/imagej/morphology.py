# -*- coding: utf-8 -*-
"""
Binary Morphological Operations - Port of ImageJ's Binary plugin.

Implements the four fundamental binary morphological operations
(Erode, Dilate, Open, Close) and derived operations (Top-Hat,
Black-Hat, Morphological Gradient). Operates on binary or continuous
images using structuring elements of configurable shape and size.

Particularly useful for:
- Post-classification cleanup of land cover maps from MSI/HSI
- Removing small noise regions after thresholding SAR imagery
- Filling holes in segmented targets from PAN imagery
- Separating touching objects in thermal imagery
- Extracting morphological features (ridges, boundaries) for detection
- Cloud mask refinement in satellite EO imagery

Attribution
-----------
ImageJ implementation: Wayne Rasband (NIH), Gabriel Landini.
Source: ``ij/plugin/filter/Binary.java`` and
``ij/process/BinaryProcessor.java`` in ImageJ 1.54j.
ImageJ 1.x source is in the public domain.

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
2026-02-06

Modified
--------
2026-02-06
"""

# Standard library
from typing import Any, Optional

# Third-party
import numpy as np
from scipy.ndimage import (
    binary_erosion,
    binary_dilation,
    binary_opening,
    binary_closing,
    minimum_filter,
    maximum_filter,
    generate_binary_structure,
)

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version


MORPHOLOGY_OPERATIONS = (
    'erode', 'dilate', 'open', 'close',
    'tophat', 'blackhat', 'gradient',
)

KERNEL_SHAPES = ('square', 'cross', 'disk')


def _make_structuring_element(shape: str, radius: int) -> np.ndarray:
    """Build a 2D structuring element.

    Parameters
    ----------
    shape : str
        ``'square'``, ``'cross'``, or ``'disk'``.
    radius : int
        Half-size. Element is ``(2*radius+1) x (2*radius+1)``.

    Returns
    -------
    np.ndarray
        2D bool array.
    """
    size = 2 * radius + 1

    if shape == 'square':
        return np.ones((size, size), dtype=bool)

    elif shape == 'cross':
        se = np.zeros((size, size), dtype=bool)
        se[radius, :] = True
        se[:, radius] = True
        return se

    elif shape == 'disk':
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        return (x * x + y * y) <= radius * radius

    raise ValueError(f"Unknown shape: {shape}")


@processor_version('1.54j')
class MorphologicalFilter(ImageTransform):
    """Binary and grayscale morphological operations, ported from ImageJ 1.54j.

    Applies morphological operations using a configurable structuring
    element. Works on both binary images (0/1) and continuous grayscale
    images (grayscale morphology via min/max filters).

    Parameters
    ----------
    operation : str
        Morphological operation. One of:

        - ``'erode'``: Shrink bright regions (minimum filter).
        - ``'dilate'``: Expand bright regions (maximum filter).
        - ``'open'``: Erode then dilate. Removes small bright noise.
        - ``'close'``: Dilate then erode. Fills small dark holes.
        - ``'tophat'``: ``image - open(image)``. Extracts small bright
          structures removed by opening.
        - ``'blackhat'``: ``close(image) - image``. Extracts small dark
          structures filled by closing.
        - ``'gradient'``: ``dilate(image) - erode(image)``. Outlines
          boundaries of bright regions.

    radius : int
        Half-size of the structuring element. Default 1 (3x3 element),
        matching ImageJ's default.
    kernel_shape : str
        Shape of the structuring element: ``'square'`` (default),
        ``'cross'``, or ``'disk'``. ImageJ default is square (8-connected).
    iterations : int
        Number of times to apply the operation. Default 1.
        Multiple iterations with radius=1 approximate larger radii
        but with different boundary behavior.

    Notes
    -----
    Port of ``ij/plugin/filter/Binary.java`` from ImageJ 1.54j (public
    domain). Original implementation by Wayne Rasband and Gabriel Landini.

    For binary images (containing only 0 and 1 or True/False), the
    operation uses ``scipy.ndimage.binary_*`` functions. For continuous
    images, it uses grayscale morphology via min/max filters.

    Examples
    --------
    Clean up a binary classification mask:

    >>> from grdl.imagej import MorphologicalFilter
    >>> opener = MorphologicalFilter(operation='open', radius=1)
    >>> clean_mask = opener.apply(noisy_binary_mask)

    Extract edges from a binary region:

    >>> grad = MorphologicalFilter(operation='gradient', radius=1)
    >>> edges = grad.apply(segmented_region)
    """

    __imagej_source__ = 'ij/plugin/filter/Binary.java'
    __imagej_version__ = '1.54j'

    def __init__(
        self,
        operation: str = 'erode',
        radius: int = 1,
        kernel_shape: str = 'square',
        iterations: int = 1,
    ) -> None:
        op_lower = operation.lower()
        if op_lower not in MORPHOLOGY_OPERATIONS:
            raise ValueError(
                f"Unknown operation '{operation}'. "
                f"Must be one of {MORPHOLOGY_OPERATIONS}"
            )
        if kernel_shape.lower() not in KERNEL_SHAPES:
            raise ValueError(
                f"Unknown kernel_shape '{kernel_shape}'. "
                f"Must be one of {KERNEL_SHAPES}"
            )
        if radius < 1:
            raise ValueError(f"radius must be >= 1, got {radius}")
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")

        self.operation = op_lower
        self.radius = radius
        self.kernel_shape = kernel_shape.lower()
        self.iterations = iterations

    def _is_binary(self, image: np.ndarray) -> bool:
        """Check if image is binary (only 0 and 1 values)."""
        unique = np.unique(image)
        return len(unique) <= 2 and all(v in (0.0, 1.0) for v in unique)

    def _erode(self, image: np.ndarray, se: np.ndarray) -> np.ndarray:
        if self._is_binary(image):
            result = image.astype(bool)
            for _ in range(self.iterations):
                result = binary_erosion(result, structure=se,
                                        border_value=True)
            return result.astype(np.float64)
        else:
            size = 2 * self.radius + 1
            result = image.copy()
            for _ in range(self.iterations):
                result = minimum_filter(result, size=size, mode='nearest')
            return result

    def _dilate(self, image: np.ndarray, se: np.ndarray) -> np.ndarray:
        if self._is_binary(image):
            result = image.astype(bool)
            for _ in range(self.iterations):
                result = binary_dilation(result, structure=se,
                                         border_value=False)
            return result.astype(np.float64)
        else:
            size = 2 * self.radius + 1
            result = image.copy()
            for _ in range(self.iterations):
                result = maximum_filter(result, size=size, mode='nearest')
            return result

    def _open(self, image: np.ndarray, se: np.ndarray) -> np.ndarray:
        if self._is_binary(image):
            result = image.astype(bool)
            for _ in range(self.iterations):
                result = binary_opening(result, structure=se)
            return result.astype(np.float64)
        else:
            result = image.copy()
            size = 2 * self.radius + 1
            for _ in range(self.iterations):
                result = minimum_filter(result, size=size, mode='nearest')
                result = maximum_filter(result, size=size, mode='nearest')
            return result

    def _close(self, image: np.ndarray, se: np.ndarray) -> np.ndarray:
        if self._is_binary(image):
            result = image.astype(bool)
            for _ in range(self.iterations):
                result = binary_closing(result, structure=se)
            return result.astype(np.float64)
        else:
            result = image.copy()
            size = 2 * self.radius + 1
            for _ in range(self.iterations):
                result = maximum_filter(result, size=size, mode='nearest')
                result = minimum_filter(result, size=size, mode='nearest')
            return result

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply morphological operation to a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``. Binary images
            (0/1) use binary morphology; continuous images use
            grayscale morphology.

        Returns
        -------
        np.ndarray
            Processed image, dtype float64, same shape as input.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(
                f"Expected 2D image, got shape {source.shape}"
            )

        image = source.astype(np.float64)
        se = _make_structuring_element(self.kernel_shape, self.radius)

        if self.operation == 'erode':
            return self._erode(image, se)

        elif self.operation == 'dilate':
            return self._dilate(image, se)

        elif self.operation == 'open':
            return self._open(image, se)

        elif self.operation == 'close':
            return self._close(image, se)

        elif self.operation == 'tophat':
            opened = self._open(image, se)
            return image - opened

        elif self.operation == 'blackhat':
            closed = self._close(image, se)
            return closed - image

        elif self.operation == 'gradient':
            dilated = self._dilate(image, se)
            eroded = self._erode(image, se)
            return dilated - eroded

        raise ValueError(f"Unknown operation: {self.operation}")
