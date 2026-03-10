# -*- coding: utf-8 -*-
"""
Sub-aperture dominance features for SAR target discrimination.

Provides feature maps computed from a sub-aperture look stack
``(n_looks, rows, cols)``.  These operate on the power distribution
across looks to identify pixels where energy is concentrated in a
contiguous block of apertures (aspect-dependent targets) rather than
spread uniformly (clutter).

Physical basis:

- Man-made targets scatter strongly from a limited range of aspect
  angles, concentrating power in a few contiguous sub-apertures.
- Natural clutter scatters diffusely, spreading power evenly across
  all looks.

Two features are provided:

``DominanceFeatures``
    ImageTransform wrapper that delegates sub-aperture splitting to
    :class:`~grdl.image_processing.sar.SublookDecomposition` and
    returns a ``(2, rows, cols)`` stack of ``[dominance, entropy]``.

``compute_dominance``
    Pure function — sliding-window power dominance ratio.

``compute_sublook_entropy``
    Pure function — Shannon entropy of power distribution.

Dependencies
------------
scipy

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-26

Modified
--------
2026-03-09
"""

# Standard library
from typing import Annotated, Any, TYPE_CHECKING

# Third-party
import numpy as np
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
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.image_processing.sar.sublook import SublookDecomposition
from grdl.vocabulary import ImageModality, ProcessorCategory
from grdl.IO.models import SICDMetadata

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata


# ===================================================================
# Pure functions
# ===================================================================

def compute_dominance(
    sublooks: np.ndarray,
    window_size: int = 7,
    dom_window: int = 3,
) -> np.ndarray:
    """Incoherent sliding-window power dominance.

    For each pixel, smooths per-look power with a spatial uniform
    filter, then slides a window of ``dom_window`` contiguous looks
    and reports the ratio of the best window's total power to the
    overall total.

    .. math::

        D(x,y) = \\max_k \\frac{\\sum_{i=k}^{k+W-1} P_i}{\\sum_i P_i}

    Parameters
    ----------
    sublooks : np.ndarray or cupy.ndarray
        Complex sub-aperture stack, shape ``(n_looks, rows, cols)``.
        If a cupy array is passed, all computation is performed on the
        GPU using ``cupyx.scipy.ndimage.uniform_filter``.  The return
        value is a cupy array when the input is cupy, numpy otherwise.
    window_size : int
        Spatial smoothing kernel size (uniform_filter side length).
        Default is 7.
    dom_window : int
        Number of contiguous looks in the sliding window.
        Default is 3.

    Returns
    -------
    np.ndarray or cupy.ndarray
        Dominance ratio, shape ``(rows, cols)``, values in
        ``[dom_window / n_looks, 1.0]``.  High values indicate power
        concentrated in a contiguous aperture block.  Return type
        matches the input array type.

    Raises
    ------
    ValueError
        If ``sublooks`` is not 3-D or ``dom_window`` exceeds the
        number of looks.
    """
    if sublooks.ndim != 3:
        raise ValueError(
            f"sublooks must be 3-D (n_looks, rows, cols), got {sublooks.ndim}-D"
        )
    num_looks = sublooks.shape[0]
    if dom_window > num_looks:
        raise ValueError(
            f"dom_window ({dom_window}) cannot exceed n_looks ({num_looks})"
        )

    _is_gpu = _HAS_CUPY and isinstance(sublooks, cp.ndarray)
    xp = cp if _is_gpu else np
    ndi_uniform = _cupyx_ndimage.uniform_filter if _is_gpu else _scipy_uniform_filter

    eps = xp.finfo(xp.float64).tiny

    power = xp.abs(sublooks) ** 2
    smooth_power = xp.stack([
        ndi_uniform(power[i], size=window_size)
        for i in range(num_looks)
    ])
    total_power = smooth_power.sum(axis=0) + eps

    n_windows = num_looks - dom_window + 1
    window_sums = xp.stack([
        smooth_power[i:i + dom_window].sum(axis=0)
        for i in range(n_windows)
    ])

    return window_sums.max(axis=0) / total_power


def compute_sublook_entropy(
    sublooks: np.ndarray,
    window_size: int = 7,
) -> np.ndarray:
    """Shannon entropy of power distribution across looks.

    .. math::

        H(x,y) = -\\sum_i p_i \\log(p_i)

    where :math:`p_i = P_i / \\sum_j P_j` is the normalised power in
    look *i* after spatial smoothing.

    Low entropy means power is concentrated in few looks (target-like).
    Maximum entropy is ``log(n_looks)``, reached when power is uniformly
    spread (clutter).

    Parameters
    ----------
    sublooks : np.ndarray or cupy.ndarray
        Complex sub-aperture stack, shape ``(n_looks, rows, cols)``.
        If a cupy array is passed, all computation is performed on the
        GPU using ``cupyx.scipy.ndimage.uniform_filter``.  The return
        value is a cupy array when the input is cupy, numpy otherwise.
    window_size : int
        Spatial smoothing kernel size (uniform_filter side length).
        Default is 7.

    Returns
    -------
    np.ndarray or cupy.ndarray
        Entropy array, shape ``(rows, cols)``, values in
        ``[0, log(n_looks)]``.  Lower values indicate more concentrated
        power.  Return type matches the input array type.

    Raises
    ------
    ValueError
        If ``sublooks`` is not 3-D.
    """
    if sublooks.ndim != 3:
        raise ValueError(
            f"sublooks must be 3-D (n_looks, rows, cols), got {sublooks.ndim}-D"
        )

    _is_gpu = _HAS_CUPY and isinstance(sublooks, cp.ndarray)
    xp = cp if _is_gpu else np
    ndi_uniform = _cupyx_ndimage.uniform_filter if _is_gpu else _scipy_uniform_filter

    eps = xp.finfo(xp.float64).tiny
    num_looks = sublooks.shape[0]

    power = xp.abs(sublooks) ** 2
    smooth_power = xp.stack([
        ndi_uniform(power[i], size=window_size)
        for i in range(num_looks)
    ])
    total_power = smooth_power.sum(axis=0) + eps

    p = smooth_power / total_power
    p = xp.clip(p, eps, None)

    return -(p * xp.log(p)).sum(axis=0)


# ===================================================================
# ImageTransform wrapper
# ===================================================================

@processor_version('1.0.0')
@processor_tags(
    modalities=[ImageModality.SAR],
    category=ProcessorCategory.ANALYZE,
    description='Sub-aperture dominance and entropy feature maps',
)
class DominanceFeatures(ImageTransform):
    """Compute dominance and entropy feature maps from complex SAR imagery.

    Delegates sub-aperture splitting to
    :class:`~grdl.image_processing.sar.SublookDecomposition`, then
    computes per-pixel dominance ratio and Shannon entropy across the
    resulting look stack.

    Returns a ``(2, rows, cols)`` array where ``result[0]`` is the
    dominance map and ``result[1]`` is the entropy map.

    Parameters
    ----------
    metadata : SICDMetadata
        SICD metadata from the reader.
    num_looks : int
        Number of sub-aperture looks.  Default is 7.
    dimension : str
        Frequency-domain axis to split: ``'azimuth'`` or ``'range'``.
        Default is ``'azimuth'``.
    window_size : int
        Spatial smoothing kernel size for power averaging.
        Default is 7.
    dom_window : int
        Number of contiguous looks in the dominance sliding window.
        Default is 3.

    Examples
    --------
    >>> from grdl.IO.sar import SICDReader
    >>> from grdl.image_processing.sar import DominanceFeatures
    >>>
    >>> with SICDReader('image.nitf') as reader:
    ...     image = reader.read_full()
    ...     dom = DominanceFeatures(reader.metadata, num_looks=7)
    ...     features = dom.apply(image)  # (2, rows, cols)
    ...     dominance = features[0]
    ...     entropy = features[1]
    """

    # -- Annotated tunable parameters --
    num_looks: Annotated[int, Range(min=2, max=64),
                         Desc('Number of sub-aperture looks')] = 7
    dimension: Annotated[str, Desc('Frequency axis to split')] = 'azimuth'
    window_size: Annotated[int, Range(min=1, max=101),
                           Desc('Spatial smoothing kernel size')] = 7
    dom_window: Annotated[int, Range(min=1, max=32),
                          Desc('Contiguous look window for dominance')] = 3

    def __init__(
        self,
        metadata: SICDMetadata,
        num_looks: int = 7,
        dimension: str = 'azimuth',
        window_size: int = 7,
        dom_window: int = 3,
    ) -> None:
        if dimension not in ('azimuth', 'range'):
            raise ValueError(
                f"dimension must be 'azimuth' or 'range', got {dimension!r}"
            )
        if dom_window > num_looks:
            raise ValueError(
                f"dom_window ({dom_window}) cannot exceed "
                f"num_looks ({num_looks})"
            )

        self._sublook = SublookDecomposition(
            metadata, num_looks=num_looks, dimension=dimension,
        )
        self.num_looks = num_looks
        self.dimension = dimension
        self.window_size = window_size
        self.dom_window = dom_window

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute dominance and entropy from complex SAR imagery.

        Parameters
        ----------
        source : np.ndarray
            Complex SAR image, shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Feature stack, shape ``(2, rows, cols)`` — ``[dominance,
            entropy]``.
        """
        params = self._resolve_params(kwargs)
        window_size = params['window_size']
        dom_window = params['dom_window']

        looks = self._sublook.decompose(source)
        dominance = compute_dominance(looks, window_size, dom_window)
        entropy = compute_sublook_entropy(looks, window_size)

        xp = cp if (_HAS_CUPY and isinstance(dominance, cp.ndarray)) else np
        return xp.stack([dominance, entropy], axis=0)
