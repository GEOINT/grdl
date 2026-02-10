# -*- coding: utf-8 -*-
"""
Image Calculator - Port of ImageJ's Process > Image Calculator.

Performs pixel-wise arithmetic and logical operations between two
images. This is one of ImageJ's most frequently used tools, essential
for change detection, ratio imaging, masking, and multi-band
combination workflows.

Particularly useful for:
- SAR change detection (difference or ratio of two-date amplitude images)
- NDVI and other spectral indices from MSI bands (divide, subtract)
- Applying binary masks to any imagery (multiply, AND)
- Multi-temporal compositing of PAN/EO scenes (add, average, max)
- Thermal anomaly detection (subtract baseline from current)
- Band ratioing for mineral mapping in HSI
- Logical combination of multiple detection masks (OR, AND, XOR)

Attribution
-----------
ImageJ implementation: Wayne Rasband (NIH).
Source: ``ij/plugin/ImageCalculator.java`` in ImageJ 1.54j.
ImageJ 1.x source is in the public domain.

Author
------
Jason Fritz

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-09

Modified
--------
2026-02-09
"""

# Standard library
from typing import Any

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version, processor_tags


CALC_OPERATIONS = (
    'add', 'subtract', 'multiply', 'divide',
    'and', 'or', 'xor',
    'min', 'max', 'average', 'difference', 'ratio',
)


@processor_tags(modalities=['SAR', 'PAN', 'EO', 'MSI', 'HSI', 'thermal'],
                category='math')
@processor_version('1.54j')
class ImageCalculator(ImageTransform):
    """Pixel-wise image arithmetic and logic, ported from ImageJ 1.54j.

    Combines two images using a specified operation. Both images must
    have the same shape.

    Supports complex-valued input (SLC/SICD SAR data) for arithmetic
    operations. Bitwise (``and``, ``or``, ``xor``) and ordering
    (``min``, ``max``) operations are not defined for complex data
    and will raise ``ValueError``.

    Parameters
    ----------
    operation : str
        The operation to perform. One of:

        - ``'add'``: ``image1 + image2``
        - ``'subtract'``: ``image1 - image2``
        - ``'multiply'``: ``image1 * image2``
        - ``'divide'``: ``image1 / image2`` (division by zero → 0)
        - ``'and'``: Bitwise AND (real-valued only)
        - ``'or'``: Bitwise OR (real-valued only)
        - ``'xor'``: Bitwise XOR (real-valued only)
        - ``'min'``: ``min(image1, image2)`` (real-valued only)
        - ``'max'``: ``max(image1, image2)`` (real-valued only)
        - ``'average'``: ``(image1 + image2) / 2``
        - ``'difference'``: ``|image1 - image2|`` (complex magnitude
          for complex input)
        - ``'ratio'``: ``image1 / image2`` (same as divide)

    Notes
    -----
    Port of ``ij/plugin/ImageCalculator.java`` from ImageJ 1.54j
    (public domain). Original by Wayne Rasband.

    Unlike most ImageTransform subclasses that take a single image,
    this class requires two images. The second image is passed as
    the keyword argument ``image2`` in the ``apply()`` call.

    For complex-valued input, ``'difference'`` returns the magnitude
    of the complex difference (a real-valued result). All other
    supported arithmetic operations preserve the complex dtype.

    Examples
    --------
    Compute NDVI from NIR and Red bands:

    >>> from grdl.imagej import ImageCalculator
    >>> sub = ImageCalculator(operation='subtract')
    >>> add = ImageCalculator(operation='add')
    >>> ndvi = sub.apply(nir, image2=red)
    >>> ndvi_sum = add.apply(nir, image2=red)
    >>> ndvi_final = ndvi / (ndvi_sum + 1e-10)

    SAR change detection (ratio):

    >>> ratio = ImageCalculator(operation='ratio')
    >>> change_map = ratio.apply(sar_date2, image2=sar_date1)

    Complex SLC coherent subtraction:

    >>> sub = ImageCalculator(operation='subtract')
    >>> residual = sub.apply(slc_pass2, image2=slc_pass1)
    """

    __imagej_source__ = 'ij/plugin/ImageCalculator.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = True

    def __init__(self, operation: str = 'add') -> None:
        op_lower = operation.lower()
        if op_lower not in CALC_OPERATIONS:
            raise ValueError(
                f"Unknown operation '{operation}'. "
                f"Must be one of {CALC_OPERATIONS}"
            )
        self.operation = op_lower

    # Operations that have no meaning for complex-valued data
    _REAL_ONLY_OPS = frozenset({'and', 'or', 'xor', 'min', 'max'})

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply the image calculator operation.

        Parameters
        ----------
        source : np.ndarray
            First 2D image (image1). Shape ``(rows, cols)``. Supports
            both real-valued and complex-valued (SLC/SICD) input.
        image2 : np.ndarray
            Second 2D image, passed as keyword argument. Must have
            the same shape as ``source``.

        Returns
        -------
        np.ndarray
            Result image, same shape as input. dtype is complex128
            when either input is complex (except ``'difference'``
            which always returns float64). float64 otherwise.

        Raises
        ------
        ValueError
            If source is not 2D, if image2 is missing, if shapes
            do not match, or if a real-only operation is used with
            complex input.
        """
        if source.ndim != 2:
            raise ValueError(
                f"Expected 2D image, got shape {source.shape}"
            )

        if 'image2' not in kwargs:
            raise ValueError(
                "ImageCalculator requires 'image2' keyword argument"
            )

        image2 = kwargs['image2']
        if not isinstance(image2, np.ndarray):
            raise ValueError("image2 must be a numpy ndarray")
        if image2.shape != source.shape:
            raise ValueError(
                f"Shape mismatch: source {source.shape} vs "
                f"image2 {image2.shape}"
            )

        is_complex = np.iscomplexobj(source) or np.iscomplexobj(image2)

        if is_complex and self.operation in self._REAL_ONLY_OPS:
            raise ValueError(
                f"Operation '{self.operation}' is not defined for "
                f"complex-valued data. Use 'add', 'subtract', "
                f"'multiply', 'divide', 'average', 'difference', "
                f"or 'ratio' instead."
            )

        if is_complex:
            img1 = source.astype(np.complex128)
            img2 = image2.astype(np.complex128)
        else:
            img1 = source.astype(np.float64)
            img2 = image2.astype(np.float64)

        if self.operation == 'add':
            return img1 + img2

        elif self.operation == 'subtract':
            return img1 - img2

        elif self.operation == 'multiply':
            return img1 * img2

        elif self.operation in ('divide', 'ratio'):
            result = np.zeros_like(img1)
            nonzero = img2 != 0
            result[nonzero] = img1[nonzero] / img2[nonzero]
            return result

        elif self.operation == 'and':
            return (img1.astype(np.int64) & img2.astype(np.int64)).astype(
                np.float64
            )

        elif self.operation == 'or':
            return (img1.astype(np.int64) | img2.astype(np.int64)).astype(
                np.float64
            )

        elif self.operation == 'xor':
            return (img1.astype(np.int64) ^ img2.astype(np.int64)).astype(
                np.float64
            )

        elif self.operation == 'min':
            return np.minimum(img1, img2)

        elif self.operation == 'max':
            return np.maximum(img1, img2)

        elif self.operation == 'average':
            return (img1 + img2) / 2.0

        elif self.operation == 'difference':
            # For complex data, return magnitude of complex difference
            return np.abs(img1 - img2)

        raise ValueError(f"Unknown operation: {self.operation}")
