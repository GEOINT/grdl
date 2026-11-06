"""
Linear Spatial Filters - Mean (box) and Gaussian smoothing filters.

Provides separable linear convolution filters backed by scipy's C-optimized
implementations. Both filters run in O(N) per pixel regardless of kernel
size due to separable decomposition into two 1D passes.

- ``MeanFilter``: Uniform averaging via ``scipy.ndimage.uniform_filter``
- ``GaussianFilter``: Gaussian smoothing via ``scipy.ndimage.gaussian_filter``

Dependencies
------------
scipy

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-11

Modified
--------
2026-02-11
"""

# Standard library
from numbers import Integral
from typing import Annotated, Any, cast

# Third-party
import numpy as np
from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter
from scipy.ndimage import uniform_filter as _scipy_uniform_filter

try:
    import cupy as cp
    import cupyx.scipy.ndimage as _cupyx_ndimage

    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False
    cp = None
    _cupyx_ndimage = None

# GRDL internal
from grdl.image_processing.base import BandwiseTransformMixin, ImageTransform
from grdl.image_processing.filters._validation import (
    validate_kernel_size,
    validate_mode,
)
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.vocabulary import ProcessorCategory


def boxcar_mean(source: np.ndarray, window_size: int, mode: str = "reflect") -> np.ndarray:
    """Average a 2D array, preserving dtype, backend and the source values.

    This is the shared normalized boxcar kernel used by ``MeanFilter``.
    Complex averaging filters real and imaginary components independently.
    A unit window returns the input without copying; callers must not mutate
    the result unless they intentionally want to mutate that input.
    Processor-specific dtype promotion and window restrictions belong to
    the caller. Boundary modes follow scipy.ndimage semantics.

    For an odd window, tile callers must read ``window_size // 2`` extra
    pixels on each side and crop the filtered result to the tile core.
    The kernel neither reads halo pixels nor stitches tiles. Local padding
    cannot reproduce whole-image ``wrap`` boundaries on clipped edge tiles.
    """
    if source.ndim != 2:
        raise ValueError(f"Boxcar averaging requires a 2D array, got {source.shape}")
    if isinstance(window_size, bool) or not isinstance(window_size, Integral) or window_size < 1:
        raise ValueError("Boxcar window_size must be a positive integer")
    if mode not in ("reflect", "constant", "nearest", "wrap", "mirror"):
        raise ValueError(f"Unsupported boxcar boundary mode {mode!r}")
    if window_size == 1:
        return source
    xp: Any = np
    average: Any = _scipy_uniform_filter
    if cp is not None and _cupyx_ndimage is not None and isinstance(source, cp.ndarray):
        xp = cp
        average = _cupyx_ndimage.uniform_filter
    result = xp.empty_like(source)
    if xp.iscomplexobj(source):
        average(xp.real(source), size=window_size, mode=mode, output=xp.real(result))
        average(xp.imag(source), size=window_size, mode=mode, output=xp.imag(result))
    else:
        average(source, size=window_size, mode=mode, output=result)
    return result


@processor_version("1.0.0")
@processor_tags(category=ProcessorCategory.FILTERS)
class MeanFilter(BandwiseTransformMixin, ImageTransform):
    """Spatial mean (box) filter using separable uniform convolution.

    Replaces each pixel with the arithmetic mean of its local neighborhood.
    Backed by ``scipy.ndimage.uniform_filter`` which decomposes into two
    1D passes, making it O(N) per pixel regardless of kernel size.

    Parameters
    ----------
    kernel_size : int
        Square kernel side length in pixels. Must be odd and >= 3.
        Default is 3.
    mode : str
        Boundary handling mode. One of ``'reflect'``, ``'constant'``,
        ``'nearest'``, ``'wrap'``. Default is ``'reflect'``.

    Examples
    --------
    >>> from grdl.image_processing.filters import MeanFilter
    >>> f = MeanFilter(kernel_size=5)
    >>> smoothed = f.apply(image)
    """

    __gpu_compatible__ = True

    kernel_size: Annotated[int, Range(min=3, max=101), Desc("Square kernel side length (odd)")] = 3
    mode: Annotated[
        str, Options("reflect", "constant", "nearest", "wrap"), Desc("Boundary handling mode")
    ] = "reflect"

    def __init__(self, kernel_size: int = 3, mode: str = "reflect") -> None:
        validate_kernel_size(kernel_size)
        validate_mode(mode)
        self.kernel_size = kernel_size
        self.mode = mode

    @property
    def tile_halo(self) -> tuple[int, int] | None:
        """Finite boxcar support, or whole-image processing for wrap mode.

        Halo-aware callers read neighboring pixels and crop to the core.
        Periodic padding requires opposite image edges, not local padding.
        Runtime overrides require planning with the effective parameters.
        """
        if self.mode == "wrap":
            return None
        half = self.kernel_size // 2
        return (half, half)

    def _apply_2d(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply mean filter to a single 2D band.

        Parameters
        ----------
        source : np.ndarray
            2D image array, shape ``(rows, cols)``. Accepts cupy arrays.

        Returns
        -------
        np.ndarray
            Smoothed image, same shape. Complex inputs retain their precision;
            real inputs return float32 or float64.
        """
        params = self._resolve_params(kwargs)
        ks = params["kernel_size"]
        mode = params["mode"]
        validate_kernel_size(ks)
        validate_mode(mode)

        xp: Any = np
        if cp is not None and isinstance(source, cp.ndarray):
            xp = cp

        if xp.iscomplexobj(source):
            working = source
        elif source.dtype == xp.float64:
            working = source.astype(xp.float64, copy=False)
        else:
            working = source.astype(xp.float32, copy=False)
        return boxcar_mean(working, ks, mode=mode)


@processor_version("1.0.0")
@processor_tags(category=ProcessorCategory.FILTERS)
class GaussianFilter(BandwiseTransformMixin, ImageTransform):
    """Gaussian smoothing filter using separable convolution.

    Applies a Gaussian blur with the specified standard deviation. Backed
    by ``scipy.ndimage.gaussian_filter`` which decomposes into two 1D
    passes, making it O(N) per pixel regardless of sigma.

    Parameters
    ----------
    sigma : float
        Gaussian standard deviation in pixels. Default is 1.0.
    truncate : float
        Truncate the filter at this many standard deviations. Controls
        the effective kernel extent. Default is 4.0.
    mode : str
        Boundary handling mode. One of ``'reflect'``, ``'constant'``,
        ``'nearest'``, ``'wrap'``. Default is ``'reflect'``.

    Examples
    --------
    >>> from grdl.image_processing.filters import GaussianFilter
    >>> f = GaussianFilter(sigma=2.0)
    >>> smoothed = f.apply(image)
    """

    __gpu_compatible__ = True

    sigma: Annotated[float, Range(min=0.1, max=100.0), Desc("Gaussian standard deviation")] = 1.0
    truncate: Annotated[
        float, Range(min=1.0, max=10.0), Desc("Truncate filter at this many sigmas")
    ] = 4.0
    mode: Annotated[
        str, Options("reflect", "constant", "nearest", "wrap"), Desc("Boundary handling mode")
    ] = "reflect"

    def __init__(
        self,
        sigma: float = 1.0,
        truncate: float = 4.0,
        mode: str = "reflect",
    ) -> None:
        validate_mode(mode)
        self.sigma = sigma
        self.truncate = truncate
        self.mode = mode

    @property
    def tile_halo(self) -> tuple[int, int] | None:
        """Gaussian support radius, or global processing for periodic edges."""
        if self.mode == "wrap":
            return None
        radius = int(self.truncate * self.sigma + 0.5)
        return (radius, radius)

    def _apply_2d(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Gaussian filter to a single 2D band.

        Parameters
        ----------
        source : np.ndarray
            2D image array, shape ``(rows, cols)``. Accepts cupy arrays.

        Returns
        -------
        np.ndarray
            Smoothed image, same shape. Complex inputs retain their precision;
            real inputs return float32 or float64.
        """
        params = self._resolve_params(kwargs)
        mode = params["mode"]
        validate_mode(mode)

        xp: Any = np
        average: Any = _scipy_gaussian_filter
        if cp is not None and _cupyx_ndimage is not None and isinstance(source, cp.ndarray):
            xp = cp
            average = _cupyx_ndimage.gaussian_filter

        if xp.iscomplexobj(source):
            working = source
        elif source.dtype == xp.float64:
            working = source.astype(xp.float64, copy=False)
        else:
            working = source.astype(xp.float32, copy=False)
        result = average(
            working,
            sigma=params["sigma"],
            truncate=params["truncate"],
            mode=mode,
        )
        return cast(np.ndarray, result)
