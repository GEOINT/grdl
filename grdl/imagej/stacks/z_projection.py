# -*- coding: utf-8 -*-
"""
Z-Projection - Port of ImageJ's ZProjector plugin.

Projects a multi-band or multi-temporal image stack along the band/time
axis using various statistical methods (max, mean, min, sum, standard
deviation, median). This reduces a 3D stack to a single 2D image.

Particularly useful for:
- Multi-temporal compositing of PAN/EO imagery (max for cloud-free mosaics)
- Mean compositing of SAR time series for speckle reduction
- Min projection for background estimation in thermal time series
- StdDev projection for change detection (high variance = changed pixels)
- Median compositing for robust noise removal across MSI time stacks

Attribution
-----------
ImageJ implementation: Patrick Kelly and Wayne Rasband (NIH).
Source: ``ij/plugin/ZProjector.java`` in ImageJ 1.54j.
ImageJ 1.x source is in the public domain.

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

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version, processor_tags, TunableParameterSpec


PROJECTION_METHODS = (
    'average', 'max', 'min', 'sum', 'std', 'median',
)


@processor_tags(modalities=['SAR', 'PAN', 'EO', 'MSI', 'HSI', 'thermal'], category='stacks')
@processor_version('1.54j')
class ZProjection(ImageTransform):
    """Z-Projection of image stacks, ported from ImageJ 1.54j.

    Projects a 3D image stack ``(bands, rows, cols)`` along axis 0
    (the band/slice/time dimension) to produce a single 2D image.

    Parameters
    ----------
    method : str
        Projection method. One of:

        - ``'average'``: Mean intensity per pixel across all slices.
        - ``'max'``: Maximum intensity projection (MIP).
        - ``'min'``: Minimum intensity projection.
        - ``'sum'``: Sum of all slices.
        - ``'std'``: Standard deviation across slices.
        - ``'median'``: Median value across slices.

        ImageJ default is ``'max'``.

    start_slice : int or None
        First slice index (0-based, inclusive). None means start at 0.
    stop_slice : int or None
        Last slice index (0-based, exclusive). None means use all slices.

    Notes
    -----
    Port of ``ij/plugin/ZProjector.java`` from ImageJ 1.54j (public
    domain). Original implementation by Patrick Kelly and Wayne Rasband.

    Unlike the rest of the ``ImageTransform`` family which operates on
    2D images, ``ZProjection`` expects a 3D input ``(bands, rows, cols)``
    and returns a 2D output ``(rows, cols)``. This matches ImageJ's stack
    convention where the first axis is the slice/z dimension.

    Examples
    --------
    Maximum intensity projection of a SAR time series:

    >>> from grdl.imagej import ZProjection
    >>> zp = ZProjection(method='max')
    >>> mip = zp.apply(sar_stack)  # (T, rows, cols) -> (rows, cols)

    Temporal mean for speckle reduction:

    >>> zp = ZProjection(method='average')
    >>> mean_image = zp.apply(sar_stack)

    Standard deviation for change detection:

    >>> zp = ZProjection(method='std')
    >>> change_map = zp.apply(sar_stack)
    """

    __imagej_source__ = 'ij/plugin/ZProjector.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = True

    def __init__(
        self,
        method: str = 'max',
        start_slice: int = None,
        stop_slice: int = None,
    ) -> None:
        method_lower = method.lower()
        if method_lower not in PROJECTION_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Must be one of {PROJECTION_METHODS}"
            )
        self.method = method_lower
        self.start_slice = start_slice
        self.stop_slice = stop_slice

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Z-projection to a 3D image stack.

        Parameters
        ----------
        source : np.ndarray
            3D image stack. Shape ``(slices, rows, cols)`` where the
            first axis is the band/time/z dimension.

        Returns
        -------
        np.ndarray
            2D projected image, shape ``(rows, cols)``, dtype float64.

        Raises
        ------
        ValueError
            If source is not 3D.
        """
        if source.ndim != 3:
            raise ValueError(
                f"Expected 3D stack (slices, rows, cols), got shape {source.shape}"
            )

        stack = source.astype(np.float64)

        # Slice range
        start = self.start_slice if self.start_slice is not None else 0
        stop = self.stop_slice if self.stop_slice is not None else stack.shape[0]
        stack = stack[start:stop]

        if stack.shape[0] == 0:
            raise ValueError("Empty stack after slicing")

        if self.method == 'average':
            return np.mean(stack, axis=0)
        elif self.method == 'max':
            return np.max(stack, axis=0)
        elif self.method == 'min':
            return np.min(stack, axis=0)
        elif self.method == 'sum':
            return np.sum(stack, axis=0)
        elif self.method == 'std':
            return np.std(stack, axis=0, ddof=0)
        elif self.method == 'median':
            return np.median(stack, axis=0)
        else:
            raise ValueError(f"Unknown method: {self.method}")
