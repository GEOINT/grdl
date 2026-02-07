# -*- coding: utf-8 -*-
"""
Edge Detection Filters - Port of ImageJ's Find Edges and related plugins.

Implements gradient-based edge detection: Sobel, Prewitt, Roberts Cross,
Laplacian of Gaussian (LoG), and Scharr operators. These compute spatial
derivatives that highlight edges, boundaries, and abrupt intensity changes.

Particularly useful for:
- Coastline and boundary extraction from PAN/EO imagery
- Road and linear feature detection in satellite imagery
- Edge-based feature descriptors for SAR target detection
- Boundary detection in thermal plume imagery
- Band-ratio edge maps for geological feature mapping in MSI/HSI
- Ship wake detection in SAR ocean imagery

Attribution
-----------
ImageJ's ``Process > Find Edges`` uses a 3x3 Sobel filter.
Source: ``ij/process/ByteProcessor.java`` (``findEdges()`` method)
and ``ij/process/FloatProcessor.java`` in ImageJ 1.54j.
ImageJ 1.x source is in the public domain.

The Scharr operator is from: H. Scharr, "Optimal operators in digital
image processing", PhD thesis, University of Heidelberg, 2000.

Dependencies
------------
scipy

Author
------
Steven Siebert

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
from typing import Any

# Third-party
import numpy as np
from scipy.ndimage import convolve, gaussian_filter, laplace

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version, processor_tags


EDGE_METHODS = ('sobel', 'prewitt', 'roberts', 'log', 'scharr')

# Sobel kernels (ImageJ default for Find Edges)
_SOBEL_X = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=np.float64)
_SOBEL_Y = np.array([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]], dtype=np.float64)

# Prewitt kernels
_PREWITT_X = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]], dtype=np.float64)
_PREWITT_Y = np.array([[-1, -1, -1],
                        [ 0,  0,  0],
                        [ 1,  1,  1]], dtype=np.float64)

# Roberts Cross kernels (2x2, padded to 3x3 for consistent convolution)
_ROBERTS_X = np.array([[0,  0, 0],
                        [0,  1, 0],
                        [0,  0, -1]], dtype=np.float64)
_ROBERTS_Y = np.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]], dtype=np.float64)

# Scharr kernels (optimized rotational symmetry)
_SCHARR_X = np.array([[ -3, 0,  3],
                       [-10, 0, 10],
                       [ -3, 0,  3]], dtype=np.float64)
_SCHARR_Y = np.array([[ -3, -10, -3],
                       [  0,   0,  0],
                       [  3,  10,  3]], dtype=np.float64)


@processor_tags(modalities=['SAR', 'PAN', 'EO', 'MSI', 'HSI', 'thermal'], category='edges')
@processor_version('1.54j')
class EdgeDetector(ImageTransform):
    """Gradient-based edge detection, ported from ImageJ 1.54j.

    Computes the edge magnitude image using spatial derivative kernels.
    The output is the gradient magnitude: ``sqrt(Gx^2 + Gy^2)``.

    Parameters
    ----------
    method : str
        Edge detection operator. One of:

        - ``'sobel'``: 3x3 Sobel operator. ImageJ's default ``Find Edges``.
          Weights center row/column more heavily for noise robustness.

        - ``'prewitt'``: 3x3 Prewitt operator. Equal-weight derivative
          approximation.

        - ``'roberts'``: 2x2 Roberts Cross operator. Sensitive to
          diagonal edges. Fastest but noisiest.

        - ``'log'``: Laplacian of Gaussian. Gaussian smoothing (sigma)
          followed by Laplacian (second derivative). Detects edges as
          zero-crossings. Returns absolute value of LoG response.

        - ``'scharr'``: 3x3 Scharr operator. Optimized for rotational
          symmetry -- more isotropic than Sobel.

    sigma : float
        Gaussian sigma for LoG method. Ignored for other methods.
        Default 1.4 (matches common usage for ~3-pixel-wide edges).

    Notes
    -----
    Port of ``findEdges()`` from ``ij/process/FloatProcessor.java`` in
    ImageJ 1.54j (public domain). ImageJ uses the Sobel operator
    for its built-in ``Process > Find Edges`` command.

    All methods return the gradient magnitude (always non-negative).
    For signed gradients (directional derivatives), use the individual
    kernels directly via scipy.ndimage.convolve.

    Examples
    --------
    Standard Sobel edge detection (same as ImageJ Find Edges):

    >>> from grdl.imagej import EdgeDetector
    >>> ed = EdgeDetector(method='sobel')
    >>> edges = ed.apply(pan_image)

    LoG edge detection with larger sigma for coarser edges:

    >>> ed = EdgeDetector(method='log', sigma=2.0)
    >>> edges = ed.apply(thermal_image)
    """

    __imagej_source__ = 'ij/process/FloatProcessor.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = False

    def __init__(
        self,
        method: str = 'sobel',
        sigma: float = 1.4,
    ) -> None:
        method_lower = method.lower()
        if method_lower not in EDGE_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Must be one of {EDGE_METHODS}"
            )
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")

        self.method = method_lower
        self.sigma = sigma

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply edge detection to a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Edge magnitude image, dtype float64, values >= 0.
            Same shape as input.

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

        if self.method == 'sobel':
            gx = convolve(image, _SOBEL_X, mode='nearest')
            gy = convolve(image, _SOBEL_Y, mode='nearest')
            return np.sqrt(gx * gx + gy * gy)

        elif self.method == 'prewitt':
            gx = convolve(image, _PREWITT_X, mode='nearest')
            gy = convolve(image, _PREWITT_Y, mode='nearest')
            return np.sqrt(gx * gx + gy * gy)

        elif self.method == 'roberts':
            gx = convolve(image, _ROBERTS_X, mode='nearest')
            gy = convolve(image, _ROBERTS_Y, mode='nearest')
            return np.sqrt(gx * gx + gy * gy)

        elif self.method == 'log':
            smoothed = gaussian_filter(image, sigma=self.sigma,
                                       mode='nearest')
            return np.abs(laplace(smoothed, mode='nearest'))

        elif self.method == 'scharr':
            gx = convolve(image, _SCHARR_X, mode='nearest')
            gy = convolve(image, _SCHARR_Y, mode='nearest')
            return np.sqrt(gx * gx + gy * gy)

        raise ValueError(f"Unknown method: {self.method}")
