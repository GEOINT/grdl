# -*- coding: utf-8 -*-
"""
CLAHE - Contrast Limited Adaptive Histogram Equalization, ported from Fiji.

Implements the CLAHE algorithm for adaptive contrast enhancement. Unlike
global histogram equalization, CLAHE operates on local tiles and limits
contrast amplification to prevent noise enhancement. Tiles are blended
using bilinear interpolation for smooth results.

Particularly useful for:
- Enhancing contrast in SAR amplitude imagery with strong dynamic range
- Improving visibility in low-light PAN (nighttime) imagery
- Normalizing thermal imagery with spatially varying contrast
- Pre-processing MSI/HSI bands before visual inspection or classification
- Enhancing detail in shadowed regions of EO imagery

Attribution
-----------
Algorithm: Karel Zuiderveld, "Contrast Limited Adaptive Histogram
Equalization", Graphics Gems IV, Academic Press, 1994, pp. 474-485.

Fiji implementation: Stephan Saalfeld (Max Planck Institute of Molecular
Cell Biology and Genetics, Dresden). Source: ``mpicbg/ij/clahe/CLAHE.java``
(Fiji, GPL-2). This is an independent NumPy reimplementation following the
published algorithm, not a derivative of the GPL source.

Dependencies
------------
(none beyond numpy)

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
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


def _clip_histogram(hist: np.ndarray, clip_limit: int) -> np.ndarray:
    """Clip histogram and redistribute excess counts.

    Counts exceeding ``clip_limit`` are removed and redistributed
    uniformly across all bins. The redistribution is iterative:
    any bins that exceed the limit after redistribution are clipped
    again, matching the Zuiderveld algorithm.

    Parameters
    ----------
    hist : np.ndarray
        1D histogram array (int or float).
    clip_limit : int
        Maximum count per bin.

    Returns
    -------
    np.ndarray
        Clipped histogram with total count preserved.
    """
    hist = hist.astype(np.float64).copy()
    n_bins = len(hist)

    excess = np.sum(np.maximum(hist - clip_limit, 0))
    hist = np.minimum(hist, clip_limit)

    # Distribute excess uniformly
    per_bin = excess / n_bins
    hist += per_bin

    # Iterative re-clipping (typically converges in 1-2 passes)
    for _ in range(10):
        excess = np.sum(np.maximum(hist - clip_limit, 0))
        if excess < 0.5:
            break
        hist = np.minimum(hist, float(clip_limit))
        per_bin = excess / n_bins
        hist += per_bin

    return hist


def _compute_cdf(hist: np.ndarray) -> np.ndarray:
    """Compute normalized CDF from histogram for equalization mapping.

    Parameters
    ----------
    hist : np.ndarray
        Histogram (possibly clipped).

    Returns
    -------
    np.ndarray
        CDF normalized to [0, 1], same length as hist.
    """
    cdf = np.cumsum(hist)
    total = cdf[-1]
    if total > 0:
        cdf = cdf / total
    return cdf


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR], category=PC.ENHANCE)
@processor_version('0.5.0')
class CLAHE(ImageTransform):
    """Contrast Limited Adaptive Histogram Equalization, ported from Fiji.

    Divides the image into tiles, computes a clipped histogram for each
    tile, and maps pixel intensities via the resulting CDF. Adjacent
    tiles are blended using bilinear interpolation for smooth transitions.

    Parameters
    ----------
    block_size : int
        Tile side length in pixels. ImageJ/Fiji default is 127.
    n_bins : int
        Number of histogram bins. Fiji default is 256.
    max_slope : float
        Maximum slope of the CDF, which controls the clip limit.
        Higher values allow more contrast amplification. Fiji default
        is 3.0. A value of 1.0 produces standard (unclipped) AHE.

    Notes
    -----
    Independent reimplementation of the Zuiderveld (1994) algorithm
    following the Fiji plugin by Stephan Saalfeld
    (``mpicbg/ij/clahe/CLAHE.java``, Fiji, GPL-2). The algorithm
    itself is published in Graphics Gems IV and is not restricted.

    Examples
    --------
    >>> from grdl.imagej import CLAHE
    >>> clahe = CLAHE(block_size=127, n_bins=256, max_slope=3.0)
    >>> enhanced = clahe.apply(thermal_image)
    """

    __imagej_source__ = 'mpicbg/ij/clahe/CLAHE.java'
    __imagej_version__ = '0.5.0'
    __gpu_compatible__ = True

    def __init__(
        self,
        block_size: int = 127,
        n_bins: int = 256,
        max_slope: float = 3.0,
    ) -> None:
        if block_size < 2:
            raise ValueError(f"block_size must be >= 2, got {block_size}")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        if max_slope < 1.0:
            raise ValueError(f"max_slope must be >= 1.0, got {max_slope}")

        self.block_size = block_size
        self.n_bins = n_bins
        self.max_slope = max_slope

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply CLAHE to a 2D image (vectorized).

        Uses vectorized NumPy operations for the CDF lookup and
        bilinear interpolation steps for good performance on large
        images.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Any numeric dtype. Values are internally
            normalized to [0, 1] for processing. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Enhanced image, dtype float64, values in [0, 1].
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

        vmin = image.min()
        vmax = image.max()
        if vmax - vmin < 1e-15:
            return np.zeros_like(image, dtype=np.float64)
        image = (image - vmin) / (vmax - vmin)

        rows, cols = image.shape
        bs = self.block_size
        n_bins = self.n_bins

        n_tiles_r = max(2, (rows + bs - 1) // bs)
        n_tiles_c = max(2, (cols + bs - 1) // bs)

        tile_centers_r = np.linspace(
            bs // 2, rows - 1 - bs // 2, n_tiles_r
        ).astype(int)
        tile_centers_c = np.linspace(
            bs // 2, cols - 1 - bs // 2, n_tiles_c
        ).astype(int)

        block_pixels = bs * bs
        clip_limit = max(1, int(self.max_slope * block_pixels / n_bins))

        # Compute CDFs for all tiles
        cdfs = np.zeros((n_tiles_r, n_tiles_c, n_bins), dtype=np.float64)

        for ti, cr in enumerate(tile_centers_r):
            for tj, cc in enumerate(tile_centers_c):
                r0 = max(0, cr - bs // 2)
                r1 = min(rows, cr + bs // 2 + 1)
                c0 = max(0, cc - bs // 2)
                c1 = min(cols, cc + bs // 2 + 1)

                tile = image[r0:r1, c0:c1]
                bin_idx = np.clip(
                    (tile * (n_bins - 1)).astype(int), 0, n_bins - 1
                )
                hist = np.bincount(bin_idx.ravel(), minlength=n_bins).astype(
                    np.float64
                )
                hist = _clip_histogram(hist, clip_limit)
                cdfs[ti, tj] = _compute_cdf(hist)

        # Vectorized mapping: compute tile indices and weights for all pixels
        row_coords = np.arange(rows, dtype=np.float64)
        col_coords = np.arange(cols, dtype=np.float64)

        ti_float = np.interp(row_coords, tile_centers_r,
                             np.arange(n_tiles_r, dtype=np.float64))
        tj_float = np.interp(col_coords, tile_centers_c,
                             np.arange(n_tiles_c, dtype=np.float64))

        ti0 = np.clip(np.floor(ti_float).astype(int), 0, n_tiles_r - 2)
        tj0 = np.clip(np.floor(tj_float).astype(int), 0, n_tiles_c - 2)
        ti1 = ti0 + 1
        tj1 = tj0 + 1

        wr = (ti_float - ti0).astype(np.float64)
        wc = (tj_float - tj0).astype(np.float64)

        # Quantize all pixels to bin indices
        bin_image = np.clip(
            (image * (n_bins - 1)).astype(int), 0, n_bins - 1
        )

        # Fully vectorized CDF lookup via advanced indexing
        # Build 2D index grids for tile corners
        ti0_2d = ti0[:, np.newaxis] * np.ones(cols, dtype=int)[np.newaxis, :]
        ti1_2d = ti1[:, np.newaxis] * np.ones(cols, dtype=int)[np.newaxis, :]
        tj0_2d = np.ones(rows, dtype=int)[:, np.newaxis] * tj0[np.newaxis, :]
        tj1_2d = np.ones(rows, dtype=int)[:, np.newaxis] * tj1[np.newaxis, :]

        v00 = cdfs[ti0_2d, tj0_2d, bin_image]
        v01 = cdfs[ti0_2d, tj1_2d, bin_image]
        v10 = cdfs[ti1_2d, tj0_2d, bin_image]
        v11 = cdfs[ti1_2d, tj1_2d, bin_image]

        # Bilinear interpolation weights (broadcast to 2D)
        wr_2d = wr[:, np.newaxis]
        wc_2d = wc[np.newaxis, :]

        output = (
            v00 * (1 - wr_2d) * (1 - wc_2d) +
            v01 * (1 - wr_2d) * wc_2d +
            v10 * wr_2d * (1 - wc_2d) +
            v11 * wr_2d * wc_2d
        )

        return output

    def apply_reference(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Pixel-by-pixel CLAHE for validation and testing.

        Produces identical results to ``apply()`` but uses explicit
        Python loops. Retained for correctness verification against
        the vectorized implementation.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Enhanced image, dtype float64, values in [0, 1].
        """
        if source.ndim != 2:
            raise ValueError(
                f"Expected 2D image, got shape {source.shape}"
            )

        image = source.astype(np.float64)

        vmin = image.min()
        vmax = image.max()
        if vmax - vmin < 1e-15:
            return np.zeros_like(image, dtype=np.float64)
        image = (image - vmin) / (vmax - vmin)

        rows, cols = image.shape
        bs = self.block_size
        n_bins = self.n_bins

        n_tiles_r = max(1, (rows + bs - 1) // bs)
        n_tiles_c = max(1, (cols + bs - 1) // bs)

        tile_centers_r = np.linspace(
            bs // 2, rows - 1 - bs // 2, n_tiles_r
        ).astype(int)
        tile_centers_c = np.linspace(
            bs // 2, cols - 1 - bs // 2, n_tiles_c
        ).astype(int)

        if len(tile_centers_r) < 1:
            tile_centers_r = np.array([rows // 2])
        if len(tile_centers_c) < 1:
            tile_centers_c = np.array([cols // 2])

        block_pixels = bs * bs
        clip_limit = max(1, int(self.max_slope * block_pixels / n_bins))

        cdfs = np.zeros((len(tile_centers_r), len(tile_centers_c), n_bins),
                        dtype=np.float64)

        for ti, cr in enumerate(tile_centers_r):
            for tj, cc in enumerate(tile_centers_c):
                r0 = max(0, cr - bs // 2)
                r1 = min(rows, cr + bs // 2 + 1)
                c0 = max(0, cc - bs // 2)
                c1 = min(cols, cc + bs // 2 + 1)

                tile = image[r0:r1, c0:c1]
                bin_idx = np.clip(
                    (tile * (n_bins - 1)).astype(int), 0, n_bins - 1
                )
                hist = np.bincount(bin_idx.ravel(), minlength=n_bins).astype(
                    np.float64
                )
                hist = _clip_histogram(hist, clip_limit)
                cdfs[ti, tj] = _compute_cdf(hist)

        output = np.zeros_like(image)

        for r in range(rows):
            for c in range(cols):
                ti = np.searchsorted(tile_centers_r, r, side='right') - 1
                tj = np.searchsorted(tile_centers_c, c, side='right') - 1
                ti = max(0, min(ti, len(tile_centers_r) - 2))
                tj = max(0, min(tj, len(tile_centers_c) - 2))

                ti1 = min(ti + 1, len(tile_centers_r) - 1)
                tj1 = min(tj + 1, len(tile_centers_c) - 1)

                cr0 = tile_centers_r[ti]
                cr1 = tile_centers_r[ti1]
                cc0 = tile_centers_c[tj]
                cc1 = tile_centers_c[tj1]

                wr = (r - cr0) / (cr1 - cr0) if cr1 > cr0 else 0.0
                wc = (c - cc0) / (cc1 - cc0) if cc1 > cc0 else 0.0
                wr = max(0.0, min(1.0, wr))
                wc = max(0.0, min(1.0, wc))

                bin_val = min(int(image[r, c] * (n_bins - 1)), n_bins - 1)

                v00 = cdfs[ti, tj, bin_val]
                v01 = cdfs[ti, tj1, bin_val]
                v10 = cdfs[ti1, tj, bin_val]
                v11 = cdfs[ti1, tj1, bin_val]

                output[r, c] = (
                    v00 * (1 - wr) * (1 - wc) +
                    v01 * (1 - wr) * wc +
                    v10 * wr * (1 - wc) +
                    v11 * wr * wc
                )

        return output
