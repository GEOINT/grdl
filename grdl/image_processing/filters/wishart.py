# -*- coding: utf-8 -*-
"""
Complex Wishart Filter - Joint speckle filtering for polarimetric SAR data.

The Wishart filter smooths covariance or coherency matrix images from dual-pol
or quad-pol synthetic aperture radar (SAR) data while retaining edge structure
and inter-channel polarimetric information. Unlike scalar filters that process
intensity images or independent bands, it applies the same spatial weights to
all elements of the polarimetric matrix at each pixel.

The complex Wishart distribution is the multivariate generalization of the
gamma distribution for Hermitian positive-definite matrices. It provides a
statistical model for multilook sample covariance matrices of complex Gaussian
scattering vectors in homogeneous regions. Wishart-based matrix comparisons
therefore provide a measure of similarity for polarimetric SAR processing.

This implementation computes matrix distances over a local spatial window,
derives an adaptive threshold from their median and median absolute deviation,
and averages neighboring matrices with exponentially decaying weights. Neighbors
beyond the threshold receive zero weight. Where accumulated weight is below
min_weight_sum, the original matrix is retained. The weights are data-adaptive;
the retained enl argument is informational and does not affect this calculation.

Input Data Formats
------------------
Preformed C2/T2/C3/T3/C4/T4 matrices use canonical CCYX arrays with shape
(N, N, rows, cols), where N is 2, 3, or 4. A matrix_type descriptor validates
dimension but does not convert or establish the physical covariance/coherency
basis. Complex CYX scattering-channel cubes can be formed into C2/T2 or
reciprocal C3/T3 matrices by selecting input_kind='channels' and an explicit
matrix_type. C4/T4 formation from raw channels is not implemented.

Real packed CYX arrays with 4, 9, or 16 bands are also accepted at the storage
boundary. Each matrix row contributes its diagonal followed by real/imaginary
pairs for upper-triangle elements; C2/T2 order is [M11, Re(M12), Im(M12), M22].
Packed matrix elements must not be interpreted as raw scattering channels.

Channel formation, packed storage conversion, input validation, and execution
metadata are owned by the shared polarimetric filter contract. The kernel uses
the full matrix directly and preserves its floating/complex precision.

References
----------
[1] Lee, J.S., Grunes, M.R., Ainsworth, T.L., Du, L.J., Schuler, D.L., and Cloude, S.R.
    "Unsupervised classification using polarimetric decomposition and the complex
    Wishart classifier." IEEE Transactions on Geoscience and Remote Sensing, 1999.
[2] Vasile, G., Trouvé, E., Lee, J.S., and Buzuloiu, V. "Intensity-driven
    adaptive-neighborhood technique for polarimetric and interferometric SAR
    parameters estimation." IEEE Transactions on Geoscience and Remote Sensing, 2006.
[3] Conradsen, K., Nielsen, A.A., Skriver, H., and Schou, J. "A test statistic in
    the complex Wishart distribution and its application to change detection in
    polarimetric SAR data." IEEE Transactions on Geoscience and Remote Sensing, 2003.

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
2026-08-25

Modified
--------
2026-10-30
"""

import logging
from typing import Annotated, Any
import warnings

import numpy as np

from grdl.image_processing.decomposition.pol_matrix import validate_matrix_type
from grdl.image_processing.filters.base import PolarimetricFilter
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.vocabulary import ImageModality, ProcessorCategory


logger = logging.getLogger(__name__)


@processor_version('2.0.0')
@processor_tags(category=ProcessorCategory.FILTERS, modalities=[ImageModality.SAR])
class WishartFilter(PolarimetricFilter):
    """Adaptive Wishart-distance averaging of C2/T2/C3/T3/C4/T4 matrices.

    Smooth neighboring covariance or coherency matrices according to their
    statistical similarity, applying common weights to every matrix element
    rather than filtering polarization channels independently. A local
    median/MAD distance threshold suppresses dissimilar neighbors to reduce
    smoothing across edges. Matrix inversions use trace-scaled regularization
    to accommodate singular or poorly conditioned inputs.

    Use ``apply(matrix)`` for a formed CCYX matrix or
    ``apply(cube, input_kind='channels', matrix_type='T3')`` for CYX SLC data.
    Only C2/T2 and reciprocal C3/T3 can be formed from raw channels. Matrix
    basis descriptors never trigger conversion of a preformed matrix.

    Parameters
    ----------
    kernel_size : int
        Odd spatial window size in [3, 31], default 7.
    enl : float
        Deprecated informational setting, default 0.0. It does not affect
        the weights, and zero does not request automatic ENL estimation.
    matrix_type : str
        Optional default input basis descriptor, not an algorithm parameter.
        ``'auto'`` leaves the basis unspecified for preformed/packed input.
        Raw channels require C2/T2/C3/T3, supplied here or to apply().
    sigma_range : float
        Multiplier for the robust distance spread in the adaptive threshold,
        in [0.5, 10], default 3. Smaller values exclude more dissimilar
        neighbors; larger values allow more spatial smoothing.
    min_weight_sum : float
        Minimum accumulated weight in [0, 1], default 0.1. Pixels below this
        threshold retain their original matrix instead of a weighted average.

    Notes
    -----
    Legacy real packed CYX arrays remain supported at the input/output boundary.
    The filtering kernel always operates directly on a CCYX tensor.
    Construction and runtime overrides use the same Annotated constraints.
    ENL does not affect the weights or require a scene-wide estimation pass.
    Matrix input validation checks structure and finite values, not physical
    Hermiticity or positive definiteness. Common nonnegative weights preserve
    Hermitian structure for Hermitian inputs up to numerical precision.
    Larger windows and matrix dimensions increase computational cost; the
    distance buffer contains kernel_size**2 values per image pixel.

    Examples
    --------
    >>> filtering = WishartFilter(kernel_size=3)
    >>> filtered = filtering.apply(t3, input_kind='matrix', matrix_type='T3')
    >>> filtered = filtering.apply(cube, input_kind='channels', matrix_type='T3')
    """

    _legacy_packed = True

    sigma_range: Annotated[
        float, Range(min=0.5, max=10.0), Desc('Adaptive similarity threshold'),
    ] = 3.0
    min_weight_sum: Annotated[
        float, Range(min=0.0, max=1.0), Desc('Minimum weight sum to apply filter'),
    ] = 0.1

    def __init__(
        self,
        kernel_size: int = 7,
        enl: float = 0.0,
        matrix_type: str = 'auto',
        sigma_range: float = 3.0,
        min_weight_sum: float = 0.1,
    ) -> None:
        """Initialize an adaptive Wishart filter.

        Parameters
        ----------
        kernel_size : int
            Odd spatial window width and height. The supported range is
            3 through 31; the default is 7. A window of width ``k`` uses
            ``k**2`` neighbors for every output pixel.
        enl : float
            Deprecated informational equivalent number of looks. It is
            retained for compatibility but does not enter the distance,
            threshold, or weight equations. Nonzero values emit a
            ``DeprecationWarning``.
        matrix_type : str
            Default matrix basis descriptor. ``'auto'`` leaves the basis
            unspecified for preformed or packed matrices. Raw channels must
            explicitly select a supported C2/T2/C3/T3 basis.
        sigma_range : float
            Nonnegative multiplier for the robust distance spread in the
            adaptive threshold. The configured range is [0.5, 10.0].
        min_weight_sum : float
            Minimum accumulated neighbor weight required to replace the
            center matrix. The configured range is [0.0, 1.0]. If the sum
            is smaller, the original center matrix is retained.
        """
        self.kernel_size = kernel_size
        self.enl = enl
        self.matrix_type = validate_matrix_type(matrix_type)
        self.sigma_range = sigma_range
        self.min_weight_sum = min_weight_sum
        self._resolve_params({})
        if enl != 0:
            warnings.warn(
                'Wishart enl is informational and deprecated; it does not affect filtering',
                DeprecationWarning, stacklevel=2,
            )

    @property
    def tile_halo(self) -> tuple[int, int]:
        """Wishart weights have finite support and require no global estimate."""
        return (self.kernel_size // 2, self.kernel_size // 2)

    def _filter_matrix(self, matrix: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        """Apply adaptive Wishart weighting directly to a CCYX tensor.

        Parameters
        ----------
        matrix : numpy.ndarray
            Matrix image with shape ``(n, n, rows, cols)``. The matrix is
            padded by reflection at the spatial boundary.
        params : dict[str, Any]
            Resolved filter parameters containing ``kernel_size``,
            ``sigma_range``, and ``min_weight_sum``.

        Returns
        -------
        numpy.ndarray
            Filtered matrix image with the same shape and dtype as ``matrix``.

        Notes
        -----
        For each center matrix ``C`` and neighboring matrix ``N``, the
        symmetric distance ``D(C, N)`` is computed first. Let ``m`` be the
        median of the local distances and ``MAD`` their median absolute
        deviation. The robust spread and acceptance threshold are

        ``sigma = 1.4826 * MAD`` when ``MAD > 0``, otherwise
        ``sigma = max(m, 1e-6)``,

        ``threshold = m + sigma_range * sigma``.

        Accepted neighbors receive weight

        ``w = exp(-D / (sigma + 1e-10))`` when ``D <= threshold``;
        otherwise ``w = 0``.

        The output is the weighted matrix average

        ``S = sum(w * N) / sum(w)``.

        When ``sum(w) < min_weight_sum``, the center matrix ``C`` is used
        instead. The same weights apply to every matrix element, preserving
        the joint polarimetric structure rather than filtering channels
        independently.
        """
        dimension, _, rows, cols = matrix.shape
        kernel_size = params['kernel_size']
        logger.info(
            'WishartFilter: processing %dx%d matrix image (%dx%d pixels) '
            'with kernel_size=%d, sigma_range=%g, min_weight_sum=%g, dtype=%s',
            dimension, dimension, rows, cols, kernel_size,
            params['sigma_range'], params['min_weight_sum'], matrix.dtype,
        )
        half = kernel_size // 2
        pixel_count = rows * cols
        padded = np.pad(
            matrix, ((0, 0), (0, 0), (half, half), (half, half)), mode='reflect',
        )
        center = matrix.transpose(2, 3, 0, 1).reshape(pixel_count, dimension, dimension)
        center_inv, center_logdet = self._batch_inv_logdet(center)
        distances = np.empty((kernel_size * kernel_size, pixel_count), dtype=np.float64)
        for row_offset in range(kernel_size):
            for col_offset in range(kernel_size):
                neighbor = padded[
                    :, :, row_offset:row_offset + rows, col_offset:col_offset + cols,
                ].transpose(2, 3, 0, 1).reshape(pixel_count, dimension, dimension)
                distances[row_offset * kernel_size + col_offset] = self._batch_wishart_distance(
                    center, center_inv, center_logdet, neighbor,
                )
        median = np.median(distances, axis=0)
        mad = np.median(np.abs(distances - median[np.newaxis, :]), axis=0)
        sigma = np.where(mad > 0, 1.4826 * mad, median.clip(min=1e-6))
        threshold = median + params['sigma_range'] * sigma
        weight_sum = np.zeros(pixel_count, dtype=np.float64)
        accumulated = np.zeros((pixel_count, dimension, dimension), dtype=np.complex128)
        for row_offset in range(kernel_size):
            for col_offset in range(kernel_size):
                distance = distances[row_offset * kernel_size + col_offset]
                weight = np.where(
                    distance > threshold, 0.0, np.exp(-distance / (sigma + 1e-10)),
                )
                neighbor = padded[
                    :, :, row_offset:row_offset + rows, col_offset:col_offset + cols,
                ].transpose(2, 3, 0, 1).reshape(pixel_count, dimension, dimension)
                weight_sum += weight
                accumulated += weight[:, np.newaxis, np.newaxis] * neighbor
        low_weight = weight_sum < params['min_weight_sum']
        safe_weight = np.where(low_weight, 1.0, weight_sum)
        result = accumulated / safe_weight[:, np.newaxis, np.newaxis]
        result[low_weight] = center[low_weight]
        result = result.reshape(rows, cols, dimension, dimension).transpose(2, 3, 0, 1)
        if not np.iscomplexobj(matrix):
            result = result.real
        return result.astype(matrix.dtype, copy=False)

    @staticmethod
    def _regularise(batch: np.ndarray) -> np.ndarray:
        """Add trace-scaled Tikhonov regularisation to avoid singular matrices.

        Parameters
        ----------
        batch : numpy.ndarray
            Batch of square matrices with shape ``(..., n, n)``.

        Returns
        -------
        numpy.ndarray
            Complex128 matrices with a small diagonal term added.

        Notes
        -----
        Single-look coherency/covariance matrices are rank-1 and singular by
        construction. For each matrix ``M``, this adds

        ``M_reg = M + (1e-6 * max(real(trace(M)), 1e-10) / n) * I``.

        The trace scale keeps the perturbation relative to the matrix energy,
        while the lower bound prevents zero matrices from producing a zero
        regularizer. This keeps eigenvalues positive while staying
        small relative to the matrix scale.
        """
        # Rank-one SLC matrices suffer cofactor cancellation in complex64;
        # zero-matrix regularized determinants also underflow in that dtype.
        batch = batch.astype(np.complex128, copy=False)
        dimension = batch.shape[-1]
        trace = np.real(np.einsum('mii->m', batch)).clip(min=1e-10)
        scale = trace / dimension * 1e-6
        identity = np.eye(dimension, dtype=batch.dtype)
        return batch + scale[:, np.newaxis, np.newaxis] * identity[np.newaxis]

    @staticmethod
    def _batch_inv_logdet(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return inverses and log-determinants of regularized matrices.

        Uses analytic closed-form formulas for 2×2 and 3×3 matrices (pure
        element-wise numpy, no LAPACK) and falls back to ``np.linalg.inv`` for
        4×4.  Trace-scaled regularisation is applied first so that near-singular
        (e.g. single-look) matrices are handled gracefully.

        Parameters
        ----------
        batch : numpy.ndarray
            Batch of square matrices with shape ``(..., n, n)``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The regularized matrix inverses and the natural logarithm of the
            absolute determinant for each matrix. Two- and three-dimensional
            matrices use the explicit analytic helpers; other dimensions use
            NumPy's batched inverse and sign-aware determinant routines.
        """
        regularized = WishartFilter._regularise(batch)
        dimension = regularized.shape[-1]
        if dimension == 2:
            return WishartFilter._inv_logdet_2x2(regularized)
        if dimension == 3:
            return WishartFilter._inv_logdet_3x3(regularized)
        inverse = np.linalg.inv(regularized)
        sign, logdet = np.linalg.slogdet(regularized)
        return inverse, np.where(sign != 0, logdet, -23.0)

    @staticmethod
    def _inv_logdet_2x2(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute analytic inverses and log-determinants for 2x2 matrices.

        Parameters
        ----------
        batch : numpy.ndarray
            Complex matrix batch with shape ``(m, 2, 2)``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The inverse and ``log(abs(det(M)))`` for each matrix, using

            ``det(M) = M00 * M11 - M01 * M10``

            and ``M_inv = det(M)**-1 * [[M11, -M01], [-M10, M00]]``.
        """
        first = batch[:, 0, 0]
        upper = batch[:, 0, 1]
        lower = batch[:, 1, 0]
        last = batch[:, 1, 1]
        determinant = first * last - upper * lower
        inverse = np.empty_like(batch)
        inv_det = 1.0 / determinant
        inverse[:, 0, 0] = last * inv_det
        inverse[:, 0, 1] = -upper * inv_det
        inverse[:, 1, 0] = -lower * inv_det
        inverse[:, 1, 1] = first * inv_det
        return inverse, np.log(np.abs(determinant))

    @staticmethod
    def _inv_logdet_3x3(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute cofactor-based inverses and log-determinants for 3x3 matrices.

        Parameters
        ----------
        batch : numpy.ndarray
            Complex matrix batch with shape ``(m, 3, 3)``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The inverse and ``log(abs(det(M)))`` for each matrix. The inverse
            is ``adj(M) / det(M)``, where ``adj(M)`` is the transpose of the
            signed cofactor matrix. The determinant is expanded across the
            first row as ``sum(M[0, j] * cofactor[0, j])``.
        """
        first = batch[:, 0, 0]
        second = batch[:, 0, 1]
        third = batch[:, 0, 2]
        fourth = batch[:, 1, 0]
        fifth = batch[:, 1, 1]
        sixth = batch[:, 1, 2]
        seventh = batch[:, 2, 0]
        eighth = batch[:, 2, 1]
        ninth = batch[:, 2, 2]

        # Cofactors (= minors with sign, no transpose yet)
        cofactor_00 = fifth * ninth - sixth * eighth
        cofactor_01 = -(fourth * ninth - sixth * seventh)
        cofactor_02 = fourth * eighth - fifth * seventh
        cofactor_10 = -(second * ninth - third * eighth)
        cofactor_11 = first * ninth - third * seventh
        cofactor_12 = -(first * eighth - second * seventh)
        cofactor_20 = second * sixth - third * fifth
        cofactor_21 = -(first * sixth - third * fourth)
        cofactor_22 = first * fifth - second * fourth

        determinant = first * cofactor_00 + second * cofactor_01 + third * cofactor_02
        inv_det = 1.0 / determinant

        inverse = np.empty_like(batch)
        # adjugate = cofactor^T → swap row/col indices
        inverse[:, 0, 0] = cofactor_00 * inv_det
        inverse[:, 0, 1] = cofactor_10 * inv_det
        inverse[:, 0, 2] = cofactor_20 * inv_det
        inverse[:, 1, 0] = cofactor_01 * inv_det
        inverse[:, 1, 1] = cofactor_11 * inv_det
        inverse[:, 1, 2] = cofactor_21 * inv_det
        inverse[:, 2, 0] = cofactor_02 * inv_det
        inverse[:, 2, 1] = cofactor_12 * inv_det
        inverse[:, 2, 2] = cofactor_22 * inv_det
        return inverse, np.log(np.abs(determinant))

    @staticmethod
    def _batch_wishart_distance(
        center: np.ndarray,
        center_inv: np.ndarray,
        center_logdet: np.ndarray,
        neighbor: np.ndarray,
    ) -> np.ndarray:
        """Compute the symmetric Wishart distance for matrix pairs.

        Parameters
        ----------
        center : numpy.ndarray
            Center matrices with shape ``(m, n, n)``.
        center_inv : numpy.ndarray
            Inverses of the regularized center matrices.
        center_logdet : numpy.ndarray
            Log absolute determinants of the regularized center matrices.
        neighbor : numpy.ndarray
            Neighbor matrices with the same shape as ``center``.

        Returns
        -------
        numpy.ndarray
            One nonnegative distance per matrix pair.

        Notes
        -----
        Uses the full symmetric formula (identical to the reference per-pixel
        algorithm) but operates on entire image batches at once.
        The neighbor is regularized before its inverse and determinant are
        computed. For center matrix ``C`` and neighbor ``N`` of dimension
        ``n``, the implemented symmetric statistic is

            d(C, N) = ln|det(C)| + ln|det(N)| + tr(C⁻¹N) + tr(N⁻¹C) − 2p

        The neighbor inverse is computed analytically (no LAPACK for 2×2/3×3),
        so this is fast even though both inverses are required.

        The absolute value keeps numerical roundoff from producing negative
        distances while preserving the existing symmetric formulation.
        """
        neighbor_reg = WishartFilter._regularise(neighbor)
        neighbor_inv, neighbor_logdet = WishartFilter._batch_inv_logdet(neighbor_reg)

        # tr(C⁻¹N) + tr(N⁻¹C) via batch einsum: tr(A@B) = ΣᵢⱼAᵢⱼBⱼᵢ
        trace_cn = np.einsum('mij,mji->m', center_inv, neighbor_reg).real
        trace_nc = np.einsum('mij,mji->m', neighbor_inv, center).real
        distance = center_logdet + neighbor_logdet + trace_cn + trace_nc - 2 * center.shape[-1]
        
        return np.abs(distance)


__all__ = ['WishartFilter']
