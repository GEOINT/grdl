# -*- coding: utf-8 -*-
"""
Complex Wishart Filter - Multi-dimensional speckle filter for polarimetric SAR data.

The Wishart filter is specifically designed to smooth covariance or coherency matrix
images from dual-pol or quad-pol synthetic aperture radar (SAR) data while preserving
edges and polarimetric information. Unlike scalar filters that operate on intensity
images, this filter operates on the full covariance matrix at each pixel.

The complex Wishart distribution is the multivariate generalization of the gamma
distribution for Hermitian positive-definite matrices. For polarimetric SAR data,
each pixel's covariance matrix follows a scaled complex Wishart distribution.

The filter performs weighted averaging of neighboring covariance matrices, where
weights are computed based on the Wishart distance (statistical similarity) between
the center pixel and each neighbor. This approach preserves edges while effectively
reducing speckle in homogeneous regions.

Supported Matrix Formats
------------------------
- C2 (dual-pol): 2×2 covariance matrix with 4 bands [C11, C12_real, C12_imag, C22]
- C3 (quad-pol): 3×3 covariance matrix (e.g., compact-pol)
- C4 (full quad-pol): 4×4 covariance matrix (e.g., full-pol)
- T3: 3×3 coherency matrix
- T4: 4×4 coherency matrix

Input Data Format
-----------------
The input array should have shape (bands, rows, cols) where:
- For C2: bands = 4, representing [C11, C12_real, C12_imag, C22]
- For C3: bands = 9, representing the upper triangle + diagonal in row-major order
- For C4: bands = 16, representing the full 4×4 Hermitian matrix
- For T3/T4: similar to C3/C4 but for coherency matrices

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
2026-08-25
"""

# Standard library
import logging
from typing import Annotated, Any, Optional

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.filters.sar_base import SARFilter
from grdl.image_processing.decomposition.pol_matrix import CovarianceMatrix, CoherencyMatrix
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.exceptions import ValidationError
from grdl.vocabulary import ImageModality, ProcessorCategory

logger = logging.getLogger(__name__)


@processor_version('1.0.0')
@processor_tags(category=ProcessorCategory.FILTERS,
                modalities=[ImageModality.SAR])
class WishartFilter(SARFilter):
    """Complex Wishart adaptive filter for polarimetric SAR covariance matrices.

    This filter operates on multi-dimensional covariance or coherency matrix images
    from polarimetric SAR data. It performs weighted averaging of neighboring
    covariance matrices based on their statistical similarity (Wishart distance),
    effectively reducing speckle while preserving edges and polarimetric information.

    The filter automatically detects the matrix dimension (C2, C3, C4, T3, T4) from
    the input band count and reconstructs the full Hermitian matrix at each pixel
    for distance computation.

    Parameters
    ----------
    kernel_size : int
        Square kernel side length in pixels. Must be odd and >= 3.
        Default is 7. Larger kernels provide more speckle reduction but may
        blur edges.
    enl : float
        Equivalent Number of Looks. Controls the noise threshold and weighting.
        If 0.0, automatically estimated from the image statistics.
        Default is 0.0 (auto-estimate). Higher ENL values assume less speckle.
    matrix_type : str
        Type of matrix representation: 'auto', 'C2', 'T2', 'C3', 'C4', 'T3', 'T4'.
        Default is 'auto' (detect from band count).
        - 'C2': 2×2 covariance matrix (4 bands)
        - 'T2': 2×2 coherency matrix (4 bands)
        - 'C3': 3×3 covariance matrix (9 bands)
        - 'C4': 4×4 covariance matrix (16 bands)
        - 'T3': 3×3 coherency matrix (9 bands)
        - 'T4': 4×4 coherency matrix (16 bands)
    sigma_range : float
        Adaptive threshold for similarity weighting in units of standard deviations.
        Default is 3.0. Pixels with Wishart distance beyond sigma_range * sigma
        receive zero weight. Smaller values preserve edges better but reduce
        smoothing; larger values increase smoothing but may blur edges.
    min_weight_sum : float
        Minimum sum of weights required to perform filtering at a pixel.
        If the weight sum is below this threshold, the original pixel is returned.
        Default is 0.1. Prevents filtering in very heterogeneous regions.

    Examples
    --------
    >>> from grdl.image_processing.filters import WishartFilter
    >>> import numpy as np
    >>> # Create a C2 covariance matrix image (4 bands: C11, C12_real, C12_imag, C22)
    >>> covmat = np.random.rand(4, 512, 512).astype(np.float32)
    >>> # Ensure the matrix is positive semi-definite
    >>> covmat[0] = np.abs(covmat[0]) + 0.1  # C11 > 0
    >>> covmat[3] = np.abs(covmat[3]) + 0.1  # C22 > 0
    >>> filt = WishartFilter(kernel_size=7, enl=0.0)
    >>> smoothed = filt.apply(covmat)
    >>> print(smoothed.shape)  # (4, 512, 512)

    Notes
    -----
    The Wishart filter is computationally intensive for large kernels and high-dimensional
    matrices. For real-time applications, consider using smaller kernel sizes (5×5 or 7×7)
    and/or downsampled data.

    The filter preserves the Hermitian structure of the covariance/coherency matrices
    and ensures the output matrices remain positive semi-definite.
    """

    matrix_type: Annotated[
        str,
        Options(['auto', 'C2', 'T2', 'C3', 'C4', 'T3', 'T4']),
        Desc('Matrix type (auto=detect from bands)')
    ] = 'auto'

    sigma_range: Annotated[
        float,
        Range(min=0.5, max=10.0),
        Desc('Adaptive threshold for similarity (std devs)')
    ] = 3.0

    min_weight_sum: Annotated[
        float,
        Range(min=0.0, max=1.0),
        Desc('Minimum weight sum to apply filter')
    ] = 0.1

    def __init__(
        self,
        kernel_size: int = 7,
        enl: float = 0.0,
        matrix_type: str = 'auto',
        sigma_range: float = 3.0,
        min_weight_sum: float = 0.1,
    ) -> None:
        super().__init__(kernel_size=kernel_size, enl=enl)
        self.matrix_type = matrix_type.upper() if matrix_type != 'auto' else 'auto'
        self.sigma_range = sigma_range
        self.min_weight_sum = min_weight_sum

    def apply(self, source: np.ndarray, metadata: Optional[object] = None, **kwargs: Any) -> np.ndarray:
        """Apply Wishart filter to polarimetric covariance/coherency data.

        Parameters
        ----------
        source : np.ndarray
            Input covariance/coherency data.
            Supported input layouts:
            - Packed band stack: ``(bands, rows, cols)`` where bands are
              [C11, Re(C12), Im(C12), ...] in the Wishart packed format.
            - Matrix tensor: ``(N, N, rows, cols)`` (complex), compatible with
              ``CovarianceMatrix.compute`` and ``CoherencyMatrix.compute`` outputs.
        metadata : object, optional
            Image metadata (not currently used by this filter).
        **kwargs
            Additional keyword arguments (not currently used).

        Returns
        -------
        np.ndarray
            Filtered image with the same shape as input.

        Raises
        ------
        ValueError
            If the input shape is not a supported matrix layout.
        """
        if source.ndim == 4:
            return self.filter_matrix(source)
        if source.ndim != 3:
            raise ValueError(
                f'WishartFilter requires 3D packed bands or 4D matrix tensor, got shape {source.shape}'
            )
        return self._filter_band_stack(source)

    def filter_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Filter a covariance/coherency matrix tensor.

        Parameters
        ----------
        matrix : np.ndarray
            Complex matrix tensor of shape ``(N, N, rows, cols)``, where
            ``N`` must be 2, 3, or 4.

        Returns
        -------
        np.ndarray
            Filtered matrix tensor with the same shape and dtype as input.
        """
        if matrix.ndim != 4:
            raise ValidationError(
                f'Expected 4D matrix (N, N, rows, cols), got shape {matrix.shape}'
            )
        n = matrix.shape[0]
        if matrix.shape[1] != n:
            raise ValidationError(
                f'Matrix must be square in first two dims, got {matrix.shape[:2]}'
            )
        if n not in (2, 3, 4):
            raise ValidationError(f'Matrix dimension must be 2, 3, or 4, got {n}')

        packed = self._matrix_cube_to_bands(matrix)
        filtered_packed = self._filter_band_stack(packed)
        return self._bands_to_matrix_cube(filtered_packed, n).astype(matrix.dtype, copy=False)

    def filter_channels(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: Optional[np.ndarray] = None,
        svv: Optional[np.ndarray] = None,
        matrix_type: str = 'C3',
    ) -> np.ndarray:
        """Build a matrix from SLC channels and apply Wishart filtering.

        Parameters
        ----------
        shh, shv, svh, svv : np.ndarray
            Complex SLC channels.
            - For ``C2``/``T2`` pass ``shh`` and ``shv``.
            - For ``C3``/``T3`` pass ``shh``, ``shv``, ``svh``, ``svv``.
        matrix_type : str
            One of ``'C2'``, ``'T2'``, ``'C3'``, ``'T3'``.

        Returns
        -------
        np.ndarray
            Filtered covariance/coherency matrix, shape ``(N, N, rows, cols)``.
        """
        kind = matrix_type.upper()
        if kind in ('C2', 'T2'):
            channels = np.stack([shh, shv], axis=0)
        elif kind in ('C3', 'T3'):
            if svh is None or svv is None:
                raise ValidationError(
                    f"matrix_type='{kind}' requires shh, shv, svh, and svv channels"
                )
            channels = np.stack([shh, shv, svh, svv], axis=0)
        else:
            raise ValidationError(
                f"matrix_type must be one of 'C2', 'T2', 'C3', or 'T3', got '{matrix_type}'"
            )

        if kind.startswith('C'):
            matrix = CovarianceMatrix(window_size=1).compute(channels)
        else:
            matrix = CoherencyMatrix(window_size=1).compute(channels)

        return self.filter_matrix(matrix)

    def _filter_band_stack(self, image: np.ndarray) -> np.ndarray:
        """Apply Wishart filtering to packed band representation."""

        n_bands, n_rows, n_cols = image.shape
        matrix_dim = self._determine_matrix_dimension(n_bands)
        
        logger.info(
            f'WishartFilter: processing {matrix_dim}×{matrix_dim} matrix image '
            f'({n_bands} bands, {n_rows}×{n_cols} pixels) with kernel_size={self.kernel_size}'
        )

        # Determine or estimate ENL
        enl = self.enl
        if enl <= 0.0:
            enl = self._estimate_enl_from_matrix(image, matrix_dim)
            logger.info(f'WishartFilter: auto-estimated ENL = {enl:.2f}')

        # Apply the Wishart filter
        filtered = self._apply_wishart_filter(image, matrix_dim, enl)

        return filtered

    def _determine_matrix_dimension(self, n_bands: int) -> int:
        """Determine the matrix dimension from band count.

        Parameters
        ----------
        n_bands : int
            Number of bands in the input image.

        Returns
        -------
        int
            Matrix dimension (2, 3, or 4).

        Raises
        ------
        ValueError
            If the band count doesn't correspond to a valid matrix dimension.
        """
        if self.matrix_type != 'auto':
            expected_bands = {
                'C2': 4, 'T2': 4, 'C3': 9, 'C4': 16, 'T3': 9, 'T4': 16
            }
            if n_bands != expected_bands[self.matrix_type]:
                raise ValueError(
                    f'matrix_type={self.matrix_type} expects {expected_bands[self.matrix_type]} '
                    f'bands, got {n_bands}'
                )
            return {'C2': 2, 'T2': 2, 'C3': 3, 'C4': 4, 'T3': 3, 'T4': 4}[self.matrix_type]

        # Auto-detect from band count
        if n_bands == 4:
            return 2  # C2
        elif n_bands == 9:
            return 3  # C3 or T3
        elif n_bands == 16:
            return 4  # C4 or T4
        else:
            raise ValueError(
                f'Cannot auto-detect matrix dimension from {n_bands} bands. '
                f'Expected 4 (C2), 9 (C3/T3), or 16 (C4/T4) bands. '
                f'Specify matrix_type explicitly if using a custom format.'
            )

    def _estimate_enl_from_matrix(self, image: np.ndarray, matrix_dim: int) -> float:
        """Estimate ENL from covariance matrix image statistics.

        Uses the intensity channel (first diagonal element) to estimate ENL.

        Parameters
        ----------
        image : np.ndarray
            Input covariance matrix image (bands, rows, cols).
        matrix_dim : int
            Matrix dimension (2, 3, or 4).

        Returns
        -------
        float
            Estimated ENL.
        """
        # Use the first diagonal element (C11 or T11) as intensity proxy
        intensity = image[0]
        
        # Compute coefficient of variation squared
        mean_int = np.mean(intensity)
        var_int = np.var(intensity)
        
        if mean_int > 0 and var_int > 0:
            ci2 = var_int / (mean_int ** 2)
            if ci2 > 0:
                # For fully developed speckle: ENL ≈ 1/Ci²
                enl_est = 1.0 / ci2
                # Clip to reasonable range
                return np.clip(enl_est, 1.0, 100.0)
        
        # Default fallback
        return 4.0

    def _apply_wishart_filter(
        self,
        image: np.ndarray,
        matrix_dim: int,
        enl: float
    ) -> np.ndarray:
        """Apply the Wishart filter to the covariance matrix image.

        Fully vectorized two-pass implementation.  The first pass collects all
        K×K per-pixel Wishart divergences simultaneously using batch numpy linear
        algebra.  A per-pixel adaptive threshold is then computed from the median
        and MAD of the divergence distribution across the kernel window (identical
        to the reference algorithm but fully vectorized).  The second pass
        accumulates the exponentially-weighted matrix sums.

        Parameters
        ----------
        image : np.ndarray
            Input covariance matrix image (bands, rows, cols).
        matrix_dim : int
            Matrix dimension (2, 3, or 4).
        enl : float
            Equivalent Number of Looks (currently informational; the adaptive
            per-pixel threshold is derived from the distance distribution).

        Returns
        -------
        np.ndarray
            Filtered covariance matrix image.
        """
        n_bands, n_rows, n_cols = image.shape
        half_k = self.kernel_size // 2
        n = matrix_dim
        RC = n_rows * n_cols
        K = self.kernel_size

        # Build complex (n, n, R, C) cube from packed bands once
        center_cube = self._bands_to_matrix_cube(image, matrix_dim)

        # Pad for border handling
        padded_cube = np.pad(
            center_cube,
            ((0, 0), (0, 0), (half_k, half_k), (half_k, half_k)),
            mode='reflect'
        )

        # center_batch: (RC, n, n)
        center_batch = center_cube.transpose(2, 3, 0, 1).reshape(RC, n, n)

        # Precompute center matrix inverses and log-determinants once
        center_inv, center_logdet = self._batch_inv_logdet(center_batch)

        # ── Pass 1: compute all K² per-offset distance maps ──────────────────
        # all_dists shape: (K*K, RC)
        all_dists = np.empty((K * K, RC), dtype=np.float64)
        for ki in range(K):
            for kj in range(K):
                nb_cube = padded_cube[:, :, ki:ki + n_rows, kj:kj + n_cols]
                nb_batch = nb_cube.transpose(2, 3, 0, 1).reshape(RC, n, n)
                all_dists[ki * K + kj] = self._batch_wishart_distance(
                    center_batch, center_inv, center_logdet, nb_batch
                )

        # ── Adaptive per-pixel threshold (same logic as reference algorithm) ─
        # Compute median and MAD across kernel positions for every pixel
        median_dist = np.median(all_dists, axis=0)                          # (RC,)
        mad = np.median(np.abs(all_dists - median_dist[np.newaxis, :]), axis=0)
        sigma = np.where(mad > 0, 1.4826 * mad, median_dist.clip(min=1e-6))
        threshold = median_dist + self.sigma_range * sigma                  # (RC,)

        # ── Pass 2: weighted accumulation ────────────────────────────────────
        weight_sum = np.zeros(RC, dtype=np.float64)
        weighted_acc = np.zeros((RC, n, n), dtype=np.complex128)

        for ki in range(K):
            for kj in range(K):
                dist = all_dists[ki * K + kj]                               # (RC,)
                w = np.where(
                    dist > threshold, 0.0,
                    np.exp(-dist / (sigma + 1e-10))
                )

                nb_cube = padded_cube[:, :, ki:ki + n_rows, kj:kj + n_cols]
                nb_batch = nb_cube.transpose(2, 3, 0, 1).reshape(RC, n, n)

                weight_sum += w
                weighted_acc += w[:, np.newaxis, np.newaxis] * nb_batch

        # Normalise; fall back to original pixel where weight sum is too low
        low_weight = weight_sum < self.min_weight_sum
        safe_ws = np.where(low_weight, 1.0, weight_sum)
        result_batch = weighted_acc / safe_ws[:, np.newaxis, np.newaxis]
        result_batch[low_weight] = center_batch[low_weight]

        # (RC, n, n) → (n, n, R, C) and repack to bands
        result_cube = result_batch.reshape(n_rows, n_cols, n, n).transpose(2, 3, 0, 1)
        result_cube = result_cube.astype(center_cube.dtype, copy=False)
        return self._matrix_cube_to_bands(result_cube)

    @staticmethod
    def _regularise(batch: np.ndarray) -> np.ndarray:
        """Add trace-scaled Tikhonov regularisation to avoid singular matrices.

        Single-look coherency/covariance matrices are rank-1 and singular by
        construction. Regularises each matrix by adding ``eps * trace(M) / n``
        times the identity, which keeps eigenvalues positive while staying
        small relative to the matrix scale.
        """
        n = batch.shape[-1]
        # Per-matrix trace: shape (M,), clamped to avoid zero
        trace = np.real(np.einsum('mii->m', batch)).clip(min=1e-10)
        scale = trace / n * 1e-6  # (M,)
        eye_n = np.eye(n, dtype=batch.dtype)
        return batch + scale[:, np.newaxis, np.newaxis] * eye_n[np.newaxis]

    @staticmethod
    def _batch_inv_logdet(
        batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch inverse and log-determinant for Hermitian positive-definite matrices.

        Uses analytic closed-form formulas for 2×2 and 3×3 matrices (pure
        element-wise numpy, no LAPACK) and falls back to ``np.linalg.inv`` for
        4×4.  Trace-scaled regularisation is applied first so that near-singular
        (e.g. single-look) matrices are handled gracefully.

        Parameters
        ----------
        batch : np.ndarray
            Shape (M, n, n) complex array of positive-definite matrices.

        Returns
        -------
        inv_batch : np.ndarray
            Shape (M, n, n) — matrix inverses.
        logdet : np.ndarray
            Shape (M,) — natural log of |det|.
        """
        batch_reg = WishartFilter._regularise(batch)
        n = batch_reg.shape[-1]
        if n == 2:
            return WishartFilter._inv_logdet_2x2(batch_reg)
        if n == 3:
            return WishartFilter._inv_logdet_3x3(batch_reg)
        # 4×4: fall back to LAPACK
        inv_batch = np.linalg.inv(batch_reg)
        sign, logdet = np.linalg.slogdet(batch_reg)
        logdet = np.where(sign != 0, logdet, -23.0)
        return inv_batch, logdet

    @staticmethod
    def _inv_logdet_2x2(
        batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Analytic batch inverse and logdet for 2×2 complex matrices."""
        a = batch[:, 0, 0]; b = batch[:, 0, 1]
        c = batch[:, 1, 0]; d = batch[:, 1, 1]
        det = a * d - b * c
        inv = np.empty_like(batch)
        inv_det = 1.0 / det
        inv[:, 0, 0] = d * inv_det
        inv[:, 0, 1] = -b * inv_det
        inv[:, 1, 0] = -c * inv_det
        inv[:, 1, 1] = a * inv_det
        logdet = np.log(np.abs(det))
        return inv, logdet.real

    @staticmethod
    def _inv_logdet_3x3(
        batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Analytic batch inverse and logdet for 3×3 complex matrices.

        Computes cofactor matrix → adjugate → inverse without any LAPACK calls,
        making it significantly faster than ``np.linalg.inv`` for large batches
        of small matrices.
        """
        a = batch[:, 0, 0]; b = batch[:, 0, 1]; c = batch[:, 0, 2]
        d = batch[:, 1, 0]; e = batch[:, 1, 1]; f = batch[:, 1, 2]
        g = batch[:, 2, 0]; h = batch[:, 2, 1]; k = batch[:, 2, 2]

        # Cofactors (= minors with sign, no transpose yet)
        A =  e * k - f * h;  B = -(d * k - f * g);  C =  d * h - e * g
        D = -(b * k - c * h); E =  a * k - c * g;  F = -(a * h - b * g)
        G =  b * f - c * e;  H = -(a * f - c * d);  K =  a * e - b * d

        det = a * A + b * B + c * C
        inv_det = 1.0 / det

        inv = np.empty_like(batch)
        # adjugate = cofactor^T → swap row/col indices
        inv[:, 0, 0] = A * inv_det;  inv[:, 0, 1] = D * inv_det;  inv[:, 0, 2] = G * inv_det
        inv[:, 1, 0] = B * inv_det;  inv[:, 1, 1] = E * inv_det;  inv[:, 1, 2] = H * inv_det
        inv[:, 2, 0] = C * inv_det;  inv[:, 2, 1] = F * inv_det;  inv[:, 2, 2] = K * inv_det

        logdet = np.log(np.abs(det))
        return inv, logdet.real

    @staticmethod
    def _batch_wishart_distance(
        center: np.ndarray,
        center_inv: np.ndarray,
        center_logdet: np.ndarray,
        neighbor: np.ndarray,
    ) -> np.ndarray:
        """Symmetric Wishart distance for all pixels simultaneously.

        Uses the full symmetric formula (identical to the reference per-pixel
        algorithm) but operates on entire image batches at once:

            d(C, N) = ln|det(C)| + ln|det(N)| + tr(C⁻¹N) + tr(N⁻¹C) − 2p

        The neighbor inverse is computed analytically (no LAPACK for 2×2/3×3),
        so this is fast even though both inverses are required.

        Parameters
        ----------
        center : np.ndarray
            Shape (M, n, n) center matrices.
        center_inv : np.ndarray
            Shape (M, n, n) inverses of center matrices.
        center_logdet : np.ndarray
            Shape (M,) log-determinants of center matrices.
        neighbor : np.ndarray
            Shape (M, n, n) neighbor matrices.

        Returns
        -------
        np.ndarray
            Shape (M,) non-negative Wishart distances.
        """
        neighbor_reg = WishartFilter._regularise(neighbor)
        nb_inv, nb_logdet = WishartFilter._batch_inv_logdet(neighbor_reg)

        # tr(C⁻¹N) + tr(N⁻¹C) via batch einsum: tr(A@B) = ΣᵢⱼAᵢⱼBⱼᵢ
        trace_cn = np.einsum('mij,mji->m', center_inv, neighbor_reg).real
        trace_nc = np.einsum('mij,mji->m', nb_inv, center).real

        dist = center_logdet + nb_logdet + trace_cn + trace_nc - 2 * center.shape[-1]
        return np.abs(dist)

    @staticmethod
    def _matrix_cube_to_bands(matrix: np.ndarray) -> np.ndarray:
        """Convert matrix tensor (N,N,Y,X) into packed Wishart band format."""
        n = matrix.shape[0]
        rows, cols = matrix.shape[2], matrix.shape[3]
        if n == 2:
            return np.stack(
                [
                    np.real(matrix[0, 0]).astype(np.float32),
                    np.real(matrix[0, 1]).astype(np.float32),
                    np.imag(matrix[0, 1]).astype(np.float32),
                    np.real(matrix[1, 1]).astype(np.float32),
                ],
                axis=0,
            )
        if n == 3:
            return np.stack(
                [
                    np.real(matrix[0, 0]).astype(np.float32),
                    np.real(matrix[0, 1]).astype(np.float32),
                    np.imag(matrix[0, 1]).astype(np.float32),
                    np.real(matrix[0, 2]).astype(np.float32),
                    np.imag(matrix[0, 2]).astype(np.float32),
                    np.real(matrix[1, 1]).astype(np.float32),
                    np.real(matrix[1, 2]).astype(np.float32),
                    np.imag(matrix[1, 2]).astype(np.float32),
                    np.real(matrix[2, 2]).astype(np.float32),
                ],
                axis=0,
            )
        if n == 4:
            bands = np.zeros((16, rows, cols), dtype=np.float32)
            idx = 0
            for row in range(4):
                bands[idx] = np.real(matrix[row, row]).astype(np.float32)
                idx += 1
                for col in range(row + 1, 4):
                    bands[idx] = np.real(matrix[row, col]).astype(np.float32)
                    bands[idx + 1] = np.imag(matrix[row, col]).astype(np.float32)
                    idx += 2
            return bands
        raise ValidationError(f'Unsupported matrix dimension {n} for Wishart packing')

    @staticmethod
    def _bands_to_matrix_cube(bands: np.ndarray, matrix_dim: int) -> np.ndarray:
        """Convert packed Wishart band format to matrix tensor (N,N,Y,X)."""
        rows, cols = bands.shape[1], bands.shape[2]
        out = np.zeros((matrix_dim, matrix_dim, rows, cols), dtype=np.complex64)
        if matrix_dim == 2:
            c12 = bands[1] + 1j * bands[2]
            out[0, 0] = bands[0]
            out[0, 1] = c12
            out[1, 0] = np.conj(c12)
            out[1, 1] = bands[3]
            return out
        if matrix_dim == 3:
            c12 = bands[1] + 1j * bands[2]
            c13 = bands[3] + 1j * bands[4]
            c23 = bands[6] + 1j * bands[7]
            out[0, 0] = bands[0]
            out[0, 1] = c12
            out[1, 0] = np.conj(c12)
            out[0, 2] = c13
            out[2, 0] = np.conj(c13)
            out[1, 1] = bands[5]
            out[1, 2] = c23
            out[2, 1] = np.conj(c23)
            out[2, 2] = bands[8]
            return out
        if matrix_dim == 4:
            idx = 0
            for row in range(4):
                out[row, row] = bands[idx]
                idx += 1
                for col in range(row + 1, 4):
                    val = bands[idx] + 1j * bands[idx + 1]
                    out[row, col] = val
                    out[col, row] = np.conj(val)
                    idx += 2
            return out
        raise ValidationError(f'Unsupported matrix dimension {matrix_dim}')


__all__ = ['WishartFilter']
