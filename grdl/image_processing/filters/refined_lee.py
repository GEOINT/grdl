# -*- coding: utf-8 -*-
"""
Refined Lee Filter - Edge-preserving polarimetric speckle filter.

Implements the Refined Lee filter for polarimetric SAR covariance or
coherency matrices. Unlike scalar Lee filters that process channels
independently, this filter uses the polarimetric span to determine
local edge orientation, then applies directional MMSE filtering to all
matrix elements jointly. This preserves polarimetric information and
edge structure while reducing speckle.

Algorithm (Lee et al., 1999)
----------------------------
1. Compute span = trace(matrix) for edge detection.
2. For each pixel, evaluate 4 directional gradients on span sub-windows.
3. Select the direction of maximum gradient → choose one of 8 half-plane
   directional masks (the side with lower gradient = more homogeneous).
4. Within the selected mask, compute MMSE coefficient from span statistics:
       cv² = Var(span) / E[span]²
       coeff = clamp((cv² - σ²) / (cv² * (1 + σ²)), 0, 1)
   where σ² = 1/ENL (noise variance for single-look data).
5. Apply to each matrix element: out = mean_dir + coeff * (pixel - mean_dir)
   where mean_dir is the mean over the directional mask pixels.

References
----------
Lee, J.-S., Grunes, M.R., and de Grandi, G. (1999), "Polarimetric SAR
speckle filtering and its implication for classification," IEEE Trans.
Geoscience and Remote Sensing, 37(5), pp.2363-2373.

Author
------
Jason Fritz, PhD
43161141+stryder-vtx@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-06-25

Modified
--------
2026-06-25
"""

# Standard library
import logging
from typing import Any

# Third-party
import numpy as np
from scipy.ndimage import uniform_filter

# GRDL internal
from grdl.image_processing.filters.sar_base import SARFilter
from grdl.image_processing.filters._validation import validate_kernel_size
from grdl.image_processing.decomposition.pol_matrix import CovarianceMatrix, CoherencyMatrix
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.exceptions import ValidationError
from grdl.vocabulary import ImageModality, ProcessorCategory

logger = logging.getLogger(__name__)

# Window parameters: kernel_size → (Nwindow_size, Deplct)
# Nwindow_size = sub-window side for gradient computation
# Deplct = displacement between sub-window centers
_WINDOW_PARAMS = {
    3: (1, 1), 5: (3, 1), 7: (3, 2), 9: (5, 2), 11: (5, 3),
    13: (5, 4), 15: (7, 4), 17: (7, 5), 19: (7, 6), 21: (9, 6),
    23: (9, 7), 25: (9, 8), 27: (11, 8), 29: (11, 9), 31: (11, 10),
}


def _build_masks(window_size: int) -> np.ndarray:
    """Build 8 directional half-plane masks for the given window size.

    Returns shape (8, window_size, window_size) float32 mask array with values {0, 1}.

    Mask indices:
        0: right half (cols >= center)
        1: upper-right diagonal (col >= row)
        2: upper half (rows <= center)
        3: upper-left anti-diagonal (col < window_size - row)
        4: left half (cols <= center)
        5: lower-left diagonal (col <= row)
        6: lower half (rows >= center)
        7: lower-right anti-diagonal (col >= window_size - 1 - row)
    """
    ws = window_size
    center = ws // 2
    masks = np.zeros((8, ws, ws), dtype=np.float32)

    row_idx = np.arange(ws)[:, None]
    col_idx = np.arange(ws)[None, :]

    # Right half
    masks[0] = (col_idx >= center)
    # Upper-right diagonal
    masks[1] = (col_idx >= row_idx)
    # Upper half
    masks[2] = (row_idx <= center)
    # Upper-left anti-diagonal
    masks[3] = (col_idx < ws - row_idx)
    # Left half
    masks[4] = (col_idx <= center)
    # Lower-left diagonal
    masks[5] = (col_idx <= row_idx)
    # Lower half
    masks[6] = (row_idx >= center)
    # Lower-right anti-diagonal
    masks[7] = (col_idx >= ws - 1 - row_idx)

    return masks


def _compute_directional_gradients(span: np.ndarray, kernel_size: int) -> np.ndarray:
    """Compute per-pixel optimal directional mask index (0..7).

    Uses sub-window averaged span to compute 4 directional gradients,
    then selects the direction of maximum gradient magnitude. The sign
    determines which half-plane (the more homogeneous side) to use.

    Parameters
    ----------
    span : np.ndarray
        Span image, shape (rows, cols). Already padded.
    kernel_size : int
        Filter window size.

    Returns
    -------
    np.ndarray
        Integer array (rows, cols) with values in [0, 7] indicating
        which directional mask to use for each output pixel.
    """
    nwindow_size, deplct = _WINDOW_PARAMS[kernel_size]
    half = kernel_size // 2

    rows_out = span.shape[0] - kernel_size
    cols_out = span.shape[1] - kernel_size

    # Build 3x3 sub-window averages using uniform_filter on displaced grids
    # Each sub-window is nwindow_size × nwindow_size pixels
    subwin = np.zeros((3, 3, rows_out, cols_out), dtype=np.float64)

    for ki in range(3):
        for li in range(3):
            # Top-left corner of this sub-window for each output pixel
            r_start = ki * deplct
            c_start = li * deplct
            # Extract the sub-window region and average
            for kk in range(nwindow_size):
                for ll in range(nwindow_size):
                    subwin[ki, li] += span[
                        r_start + kk: r_start + kk + rows_out,
                        c_start + ll: c_start + ll + cols_out,
                    ]
            subwin[ki, li] /= (nwindow_size * nwindow_size)

    # 4 directional gradients (Sobel-like on the 3x3 sub-window grid)
    # Direction 0: horizontal (left vs right)
    d0 = (-subwin[0, 0] + subwin[0, 2]
           - subwin[1, 0] + subwin[1, 2]
           - subwin[2, 0] + subwin[2, 2])
    # Direction 1: 45° diagonal
    d1 = (subwin[0, 1] + subwin[0, 2]
           - subwin[1, 0] + subwin[1, 2]
           - subwin[2, 0] - subwin[2, 1])
    # Direction 2: vertical (top vs bottom)
    d2 = (subwin[0, 0] + subwin[0, 1] + subwin[0, 2]
           - subwin[2, 0] - subwin[2, 1] - subwin[2, 2])
    # Direction 3: 135° diagonal
    d3 = (subwin[0, 0] + subwin[0, 1] + subwin[1, 0]
           - subwin[1, 2] - subwin[2, 1] - subwin[2, 2])

    # Stack and find direction of maximum |gradient|
    gradients = np.stack([d0, d1, d2, d3], axis=0)  # (4, rows, cols)
    abs_grad = np.abs(gradients)
    max_dir = np.argmax(abs_grad, axis=0)  # (rows, cols), values 0..3

    # If gradient is positive, use mask index + 4 (opposite half-plane)
    # Gather the gradient value at the max direction
    r_idx, c_idx = np.meshgrid(
        np.arange(rows_out), np.arange(cols_out), indexing='ij'
    )
    grad_at_max = gradients[max_dir, r_idx, c_idx]

    # mask_index: 0..3 if grad <= 0, 4..7 if grad > 0
    mask_index = np.where(grad_at_max > 0, max_dir + 4, max_dir)

    return mask_index.astype(np.int32)


def _compute_masked_stats(
    span_padded: np.ndarray,
    masks: np.ndarray,
    mask_index: np.ndarray,
    kernel_size: int,
    nlook: float,
) -> np.ndarray:
    """Compute per-pixel MMSE coefficient using directional mask statistics.

    Parameters
    ----------
    span_padded : np.ndarray
        Padded span, shape (rows+ks, cols+ks).
    masks : np.ndarray
        Directional masks, shape (8, ks, ks).
    mask_index : np.ndarray
        Per-pixel mask selection, shape (rows, cols), values 0..7.
    kernel_size : int
        Filter window size.
    nlook : float
        Number of looks (ENL).

    Returns
    -------
    np.ndarray
        MMSE coefficient per pixel, shape (rows, cols), values in [0, 1].
    """
    half = kernel_size // 2
    rows, cols = mask_index.shape
    sigma2 = 1.0 / nlook

    # Pre-compute sum and sum-of-squares for each mask direction
    # For each of the 8 masks, compute the masked mean and variance
    # using a gathering approach

    # Build all 8 directional means and variances
    mask_sum = np.zeros((8, rows, cols), dtype=np.float64)
    mask_sum2 = np.zeros((8, rows, cols), dtype=np.float64)
    mask_count = masks.sum(axis=(1, 2))  # (8,) number of pixels per mask

    for m in range(8):
        # Get positions where this mask is 1
        mk_rows, mk_cols = np.where(masks[m] > 0)
        for kr, kc in zip(mk_rows, mk_cols):
            patch = span_padded[kr:kr + rows, kc:kc + cols]
            mask_sum[m] += patch
            mask_sum2[m] += patch * patch

    # For each pixel, select its mask's statistics
    r_idx, c_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    npts = mask_count[mask_index]  # (rows, cols)
    m_span = mask_sum[mask_index, r_idx, c_idx] / npts
    m_span2 = mask_sum2[mask_index, r_idx, c_idx] / npts

    # Coefficient of variation
    var_span = m_span2 - m_span * m_span
    np.maximum(var_span, 0.0, out=var_span)
    cv2 = var_span / (m_span * m_span + 1e-10)

    # MMSE coefficient: (cv² - σ²) / (cv² * (1 + σ²))
    coeff = (cv2 - sigma2) / (cv2 * (1.0 + sigma2) + 1e-10)
    np.clip(coeff, 0.0, 1.0, out=coeff)

    return coeff.astype(np.float64)


def _compute_masked_means(
    element_padded: np.ndarray,
    masks: np.ndarray,
    mask_index: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    """Compute per-pixel directional mean for one matrix element.

    Parameters
    ----------
    element_padded : np.ndarray
        Padded matrix element, shape (rows+ks, cols+ks). May be complex.
    masks : np.ndarray
        Directional masks, shape (8, ks, ks).
    mask_index : np.ndarray
        Per-pixel mask selection, shape (rows, cols).
    kernel_size : int
        Filter window size.

    Returns
    -------
    np.ndarray
        Directional mean per pixel, shape (rows, cols). Same dtype as input.
    """
    rows, cols = mask_index.shape
    mask_count = masks.sum(axis=(1, 2))  # (8,)

    # Accumulate sum for each direction
    is_complex = np.iscomplexobj(element_padded)
    dtype = np.complex128 if is_complex else np.float64
    dir_sum = np.zeros((8, rows, cols), dtype=dtype)

    for m in range(8):
        mk_rows, mk_cols = np.where(masks[m] > 0)
        for kr, kc in zip(mk_rows, mk_cols):
            dir_sum[m] += element_padded[kr:kr + rows, kc:kc + cols]

    # Gather per-pixel
    r_idx, c_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    npts = mask_count[mask_index]
    mean = dir_sum[mask_index, r_idx, c_idx] / npts

    return mean


@processor_version('1.0.0')
@processor_tags(category=ProcessorCategory.FILTERS,
                modalities=[ImageModality.SAR])
class RefinedLeeFilter(SARFilter):
    """Refined Lee polarimetric speckle filter.

    Edge-preserving filter for polarimetric covariance or coherency
    matrices. Uses the total power (span) to determine local edge
    orientation via directional gradients, then applies MMSE filtering
    within half-plane directional masks to all matrix elements jointly.

    This filter operates on the full polarimetric matrix (C3, T3, C2, etc.)
    rather than individual channels, preserving inter-channel correlations
    and polarimetric information.

    Parameters
    ----------
    kernel_size : int
        Square kernel side length in pixels. Must be odd and in [3, 31].
        Default is 7.
    nlook : float
        Number of looks. Controls the noise variance σ² = 1/nlook.
        For single-look complex (SLC) data, use nlook=1.
        Default is 1.0.

    References
    ----------
    Lee, J.-S., Grunes, M.R., and de Grandi, G. (1999), "Polarimetric SAR
    speckle filtering and its implication for classification," IEEE Trans.
    Geoscience and Remote Sensing, 37(5), pp.2363-2373.

    Examples
    --------
    Filter a pre-computed C3 matrix:

    >>> from grdl.image_processing.filters import RefinedLeeFilter
    >>> rlf = RefinedLeeFilter(kernel_size=7)
    >>> c3_filtered = rlf.filter_matrix(c3)  # (3, 3, rows, cols)

    Filter from raw SLC channels and feed to Freeman-Durden:

    >>> c3_filt = rlf.filter_channels(shh, shv, svh, svv, matrix_type='C3')
    >>> from grdl.image_processing.decomposition import FreemanDurden3C
    >>> fd = FreemanDurden3C()
    >>> result = fd.decompose_from_c3(c3_filt)
    """

    def __init__(
        self,
        kernel_size: int = 7,
        nlook: float = 1.0,
    ) -> None:
        if kernel_size not in _WINDOW_PARAMS:
            raise ValidationError(
                f"kernel_size must be odd and in [3, 31], got {kernel_size}"
            )
        super().__init__(kernel_size=kernel_size, enl=nlook)
        self.nlook = nlook

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Refined Lee filter to a polarimetric matrix.

        This is the ``ImageTransform`` interface. For most uses, prefer
        ``filter_matrix()`` or ``filter_channels()`` which have clearer
        semantics.

        Parameters
        ----------
        source : np.ndarray
            Polarimetric matrix, shape (N, N, rows, cols) where N is
            the matrix dimension (2, 3, or 4). Complex-valued.

        Returns
        -------
        np.ndarray
            Filtered matrix, same shape and dtype.
        """
        return self.filter_matrix(source)

    def filter_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Filter a polarimetric covariance or coherency matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Shape (N, N, rows, cols) where N ∈ {2, 3, 4}.
            The matrix should already be spatially averaged (boxcar or
            otherwise) if desired before edge-directed filtering.

        Returns
        -------
        np.ndarray
            Filtered matrix, same shape. Complex-valued.
        """
        if matrix.ndim != 4:
            raise ValidationError(
                f"Expected 4D matrix (N, N, rows, cols), got shape {matrix.shape}"
            )
        n = matrix.shape[0]
        if matrix.shape[1] != n:
            raise ValidationError(
                f"Matrix must be square in first two dims, got {matrix.shape[:2]}"
            )
        if n not in (2, 3, 4):
            raise ValidationError(
                f"Matrix dimension must be 2, 3, or 4, got {n}"
            )

        rows, cols = matrix.shape[2], matrix.shape[3]
        ks = self.kernel_size
        half = ks // 2

        # Compute span (trace of diagonal)
        span = np.zeros((rows, cols), dtype=np.float64)
        for i in range(n):
            span += np.real(matrix[i, i])

        # Pad span with zeros
        span_padded = np.pad(span, ((half, half + 1), (half, half + 1)),
                             mode='constant', constant_values=0)

        # Build directional masks
        masks = _build_masks(ks)

        # Compute per-pixel directional mask index
        mask_index = _compute_directional_gradients(span_padded, ks)

        # Compute MMSE coefficient from span statistics
        coeff = _compute_masked_stats(
            span_padded, masks, mask_index, ks, self.nlook
        )

        # Apply filter to each matrix element
        out = np.zeros_like(matrix)
        for i in range(n):
            for j in range(n):
                element = matrix[i, j]
                # Pad element
                elem_padded = np.pad(
                    element, ((half, half + 1), (half, half + 1)),
                    mode='constant', constant_values=0,
                )
                # Compute directional mean
                dir_mean = _compute_masked_means(
                    elem_padded, masks, mask_index, ks
                )
                # MMSE: out = mean + coeff * (center - mean)
                center = element
                if np.iscomplexobj(element):
                    out[i, j] = dir_mean + coeff * (center - dir_mean)
                else:
                    out[i, j] = np.real(dir_mean) + coeff * (center - np.real(dir_mean))

        return out

    def filter_channels(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: np.ndarray,
        svv: np.ndarray,
        matrix_type: str = 'C3',
    ) -> np.ndarray:
        """Build a polarimetric matrix from SLC channels and filter it.

        Convenience method that constructs the covariance [C3] or
        coherency [T3] matrix from quad-pol SLC data (with boxcar
        spatial averaging), then applies the Refined Lee filter.

        Parameters
        ----------
        shh, shv, svh, svv : np.ndarray
            Complex SLC channels, each shape (rows, cols).
        matrix_type : str
            'C3' for covariance matrix or 'T3' for coherency matrix.
            Default is 'C3'.

        Returns
        -------
        np.ndarray
            Filtered matrix, shape (3, 3, rows, cols).
        """

        channels = np.stack([shh, shv, svh, svv], axis=0)

        if matrix_type.upper() == 'C3':
            matrix = CovarianceMatrix(window_size=self.kernel_size).compute(channels)
        elif matrix_type.upper() == 'T3':
            matrix = CoherencyMatrix(window_size=self.kernel_size).compute(channels)
        else:
            raise ValidationError(
                f"matrix_type must be 'C3' or 'T3', got '{matrix_type}'"
            )

        return self.filter_matrix(matrix)
