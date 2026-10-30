# -*- coding: utf-8 -*-
"""
IDAN Filter - Intensity-Driven Adaptive-Neighborhood speckle filter.

Implements the IDAN filter for polarimetric SAR covariance or coherency
matrices. Unlike fixed-window filters (Lee, Refined Lee), IDAN grows a
per-pixel adaptive neighborhood by including neighbors whose intensity
falls within a similarity band of the seed-window mean. The adaptive
neighborhood is used to compute MMSE (Lee-style) statistics and filter
all matrix elements jointly, preserving inter-channel correlations.

Algorithm (Vasile et al., 2006)
--------------------------------
For each output pixel (r, c):

1. **Seed statistics**: compute mean intensity (span) over a small
   ``kernel_size × kernel_size`` seed window centred on (r, c).

2. **Neighborhood growing**: starting from the seed, iteratively admit
   connected 4-neighbors whose intensity ``I`` satisfies
   ``|I − μ_seed| < similarity_threshold × μ_seed``.
   Growing halts when either ``max_pixels`` is reached or no more
   admissible neighbors remain.

3. **MMSE coefficient**:
       σ_n² = 1 / ENL
       Ci²  = Var(span_neighborhood) / E[span_neighborhood]²
       W    = clamp((Ci² − σ_n²) / (Ci² × (1 + σ_n²)), 0, 1)

4. **Filter**: apply to every matrix element ``Z[i,j]``:
       out[i,j] = mean_Z_neighborhood + W × (Z[i,j] − mean_Z_neighborhood)

The filter preserves bright point targets (high Ci² → W → 1) and
strongly smooths homogeneous areas (low Ci² → W → 0), while the
adaptive neighborhood avoids crossing strong edges.

Acceleration
------------
When ``numba`` is installed (``pip install grdl[polsar]``), the inner BFS
loop and pixel loop are JIT-compiled for a ~50–100× speedup over pure
Python.  With ``parallel=True`` and ``prange``, independent rows are
processed simultaneously for a further ~4–8× gain on multi-core systems.

Without numba, the filter falls back to the pure Python implementation
with a one-time ``RuntimeWarning``.

References
----------
Vasile, G., Trovè, E., Lee, J.-S., and Buzuloiu, V. (2006).
    "Intensity-Driven Adaptive-Neighborhood Technique for Polarimetric
    and Interferometric SAR Parameters Estimation," IEEE Transactions
    on Geoscience and Remote Sensing, 44(6), 1609–1621.

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
2026-08-20

Modified
--------
2026-08-20
"""

# Standard library
import logging
import warnings
from typing import Annotated, Any, List, Tuple

# Third-party
import numpy as np

# Optional numba JIT
try:
    import numba as _numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

# GRDL internal
from grdl.image_processing.filters.base import PolarimetricFilter
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.exceptions import ValidationError
from grdl.image_processing.params import Desc, Range
from grdl.vocabulary import ImageModality, ProcessorCategory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numba-accelerated implementation (preferred)
# ---------------------------------------------------------------------------

if _HAS_NUMBA:

    @_numba.njit(cache=True)
    def _idan_bfs_nb(
        span,        # (rows, cols) float64 — read only
        r, c,        # center pixel
        seed_mean,   # float
        sim_thr,     # float
        max_pixels,  # int
        sigma2,      # float = 1/ENL
        visited,     # (rows, cols) bool — reset in-place each call
        vis_r,       # (max_vis,) int32  — tracks visited coords for reset
        vis_c,
        q_r,         # (max_vis,) int32  — BFS queue
        q_c,
        nbr_r,       # (max_pixels,) int32 — output neighborhood
        nbr_c,
    ):
        """BFS neighborhood growing + MMSE coefficient (global visited array).

        Uses a single shared visited array (shape rows×cols) that is reset
        in-place at the end of each call.  Best for the serial path because
        the same array is reused across all pixel calls without re-allocation.

        Returns (n_nbr, coeff).
        """
        rows, cols = span.shape
        tol = sim_thr * max(seed_mean, 1.0e-10)
        lo  = seed_mean - tol
        hi  = seed_mean + tol
        max_vis = vis_r.shape[0]

        q_head = 0; q_tail = 0; n_nbr = 0; n_vis = 0

        visited[r, c] = True
        vis_r[0] = r;  vis_c[0] = c;  n_vis = 1
        q_r[0]   = r;  q_c[0]   = c;  q_tail = 1
        nbr_r[0] = r;  nbr_c[0] = c;  n_nbr  = 1

        while q_head < q_tail and n_nbr < max_pixels:
            pr = q_r[q_head];  pc = q_c[q_head];  q_head += 1

            for k in range(4):
                if   k == 0:  nr = pr - 1;  nc = pc
                elif k == 1:  nr = pr + 1;  nc = pc
                elif k == 2:  nr = pr;      nc = pc - 1
                else:         nr = pr;      nc = pc + 1

                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                if visited[nr, nc]:
                    continue

                visited[nr, nc] = True
                if n_vis < max_vis:
                    vis_r[n_vis] = nr;  vis_c[n_vis] = nc;  n_vis += 1

                if lo <= span[nr, nc] <= hi:
                    nbr_r[n_nbr] = nr;  nbr_c[n_nbr] = nc;  n_nbr += 1
                    if n_nbr < max_pixels:
                        q_r[q_tail] = nr;  q_c[q_tail] = nc;  q_tail += 1
                    else:
                        break

        for k in range(n_vis):
            visited[vis_r[k], vis_c[k]] = False

        if n_nbr < 2:
            return n_nbr, 1.0

        m = 0.0
        for k in range(n_nbr):
            m += span[nbr_r[k], nbr_c[k]]
        m /= n_nbr

        var = 0.0
        for k in range(n_nbr):
            d = span[nbr_r[k], nbr_c[k]] - m
            var += d * d
        var /= n_nbr

        cv2 = var / (m * m + 1.0e-20)
        coeff = (cv2 - sigma2) / (cv2 * (1.0 + sigma2) + 1.0e-20)
        if coeff < 0.0: coeff = 0.0
        elif coeff > 1.0: coeff = 1.0

        return n_nbr, coeff

    @_numba.njit(cache=True)
    def _idan_bfs_local_nb(
        span,
        r, c,
        seed_mean,
        sim_thr,
        max_pixels,
        sigma2,
        vis_local,   # (2*max_pixels+1, 2*max_pixels+1) bool — LOCAL coords, reset in-place
        vis_r,       # (max_vis,) int32 — global coords of visited, for reset
        vis_c,
        q_r,         # (max_vis,) int32 — BFS queue
        q_c,
        nbr_r,       # (max_pixels,) int32 — output neighborhood (global coords)
        nbr_c,
    ):
        """BFS neighborhood growing + MMSE coefficient (compact local visited array).

        Uses a small ``(2*max_pixels+1)²`` boolean array in LOCAL pixel
        coordinates.  Since BFS can never advance more than ``max_pixels``
        Manhattan steps from the center, this array covers all reachable
        pixels while fitting comfortably in L1/L2 cache — ideal for the
        parallel path where many threads run concurrently.

        Returns (n_nbr, coeff).
        """
        rows, cols = span.shape
        local_size = vis_local.shape[0]   # == 2*max_pixels + 1
        local_half = max_pixels           # offset to convert global→local
        tol = sim_thr * max(seed_mean, 1.0e-10)
        lo  = seed_mean - tol
        hi  = seed_mean + tol
        max_vis = vis_r.shape[0]

        q_head = 0; q_tail = 0; n_nbr = 0; n_vis = 0

        vis_local[local_half, local_half] = True
        vis_r[0] = r;  vis_c[0] = c;  n_vis = 1
        q_r[0]   = r;  q_c[0]   = c;  q_tail = 1
        nbr_r[0] = r;  nbr_c[0] = c;  n_nbr  = 1

        while q_head < q_tail and n_nbr < max_pixels:
            pr = q_r[q_head];  pc = q_c[q_head];  q_head += 1

            for k in range(4):
                if   k == 0:  nr = pr - 1;  nc = pc
                elif k == 1:  nr = pr + 1;  nc = pc
                elif k == 2:  nr = pr;      nc = pc - 1
                else:         nr = pr;      nc = pc + 1

                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue

                # Local offset — pixels outside BFS radius are skipped
                lr = nr - r + local_half
                lc = nc - c + local_half
                if lr < 0 or lr >= local_size or lc < 0 or lc >= local_size:
                    continue

                if vis_local[lr, lc]:
                    continue

                vis_local[lr, lc] = True
                if n_vis < max_vis:
                    vis_r[n_vis] = nr;  vis_c[n_vis] = nc;  n_vis += 1

                if lo <= span[nr, nc] <= hi:
                    nbr_r[n_nbr] = nr;  nbr_c[n_nbr] = nc;  n_nbr += 1
                    if n_nbr < max_pixels:
                        q_r[q_tail] = nr;  q_c[q_tail] = nc;  q_tail += 1
                    else:
                        break

        # Reset local visited array using tracked coords
        for k in range(n_vis):
            lr = vis_r[k] - r + local_half
            lc = vis_c[k] - c + local_half
            vis_local[lr, lc] = False

        if n_nbr < 2:
            return n_nbr, 1.0

        m = 0.0
        for k in range(n_nbr):
            m += span[nbr_r[k], nbr_c[k]]
        m /= n_nbr

        var = 0.0
        for k in range(n_nbr):
            d = span[nbr_r[k], nbr_c[k]] - m
            var += d * d
        var /= n_nbr

        cv2 = var / (m * m + 1.0e-20)
        coeff = (cv2 - sigma2) / (cv2 * (1.0 + sigma2) + 1.0e-20)
        if coeff < 0.0: coeff = 0.0
        elif coeff > 1.0: coeff = 1.0

        return n_nbr, coeff

    @_numba.njit(cache=True)
    def _idan_filter_nb(matrix, span, span_padded, kernel_size, sigma2, sim_thr, max_pixels):
        """Serial JIT-compiled IDAN loop (~50–100× faster than pure Python).

        Uses a single large visited array reset in-place per pixel (no
        per-pixel allocation overhead).
        """
        n, _, rows, cols = matrix.shape
        out     = np.empty_like(matrix)
        max_vis = max_pixels * 8

        visited = np.zeros((rows, cols), dtype=np.bool_)
        vis_r   = np.empty(max_vis,    dtype=np.int32)
        vis_c   = np.empty(max_vis,    dtype=np.int32)
        q_r     = np.empty(max_vis,    dtype=np.int32)
        q_c     = np.empty(max_vis,    dtype=np.int32)
        nbr_r   = np.empty(max_pixels, dtype=np.int32)
        nbr_c   = np.empty(max_pixels, dtype=np.int32)

        for r in range(rows):
            for c in range(cols):
                s = 0.0
                for kr in range(kernel_size):
                    for kc in range(kernel_size):
                        s += span_padded[r + kr, c + kc]
                seed_mean = s / (kernel_size * kernel_size)

                n_nbr, coeff = _idan_bfs_nb(
                    span, r, c, seed_mean, sim_thr, max_pixels, sigma2,
                    visited, vis_r, vis_c, q_r, q_c, nbr_r, nbr_c,
                )

                for i in range(n):
                    for j in range(n):
                        m_re = 0.0;  m_im = 0.0
                        for k in range(n_nbr):
                            v = matrix[i, j, nbr_r[k], nbr_c[k]]
                            m_re += v.real
                            m_im += v.imag
                        m_re /= n_nbr;  m_im /= n_nbr
                        cv = matrix[i, j, r, c]
                        out[i, j, r, c] = (
                            (m_re + coeff * (cv.real - m_re))
                            + 1j * (m_im + coeff * (cv.imag - m_im))
                        )
        return out

    @_numba.njit(parallel=True, cache=True)
    def _idan_filter_nb_parallel(matrix, span, span_padded, kernel_size, sigma2, sim_thr, max_pixels):
        """Parallel JIT-compiled IDAN loop.

        Each row runs on its own thread.  The compact local visited array
        ``(2*max_pixels+1)²`` is small enough (~10 KB for max_pixels=50)
        to fit in L1/L2 cache per thread, avoiding the cache contention
        that kills performance when sharing a large ``rows×cols`` array.
        Thread writes to ``out`` are row-disjoint so no locking is needed.
        """
        n, _, rows, cols = matrix.shape
        out        = np.empty_like(matrix)
        max_vis    = max_pixels * 8
        local_size = 2 * max_pixels + 1

        for r in _numba.prange(rows):
            # Per-iteration allocations — only (local_size)² + small buffers,
            # so peak memory is num_threads × ~10 KB regardless of image size.
            vis_local = np.zeros((local_size, local_size), dtype=np.bool_)
            vis_r     = np.empty(max_vis,    dtype=np.int32)
            vis_c     = np.empty(max_vis,    dtype=np.int32)
            q_r       = np.empty(max_vis,    dtype=np.int32)
            q_c       = np.empty(max_vis,    dtype=np.int32)
            nbr_r     = np.empty(max_pixels, dtype=np.int32)
            nbr_c     = np.empty(max_pixels, dtype=np.int32)

            for c in range(cols):
                s = 0.0
                for kr in range(kernel_size):
                    for kc in range(kernel_size):
                        s += span_padded[r + kr, c + kc]
                seed_mean = s / (kernel_size * kernel_size)

                n_nbr, coeff = _idan_bfs_local_nb(
                    span, r, c, seed_mean, sim_thr, max_pixels, sigma2,
                    vis_local, vis_r, vis_c, q_r, q_c, nbr_r, nbr_c,
                )

                for i in range(n):
                    for j in range(n):
                        m_re = 0.0;  m_im = 0.0
                        for k in range(n_nbr):
                            v = matrix[i, j, nbr_r[k], nbr_c[k]]
                            m_re += v.real
                            m_im += v.imag
                        m_re /= n_nbr;  m_im /= n_nbr
                        cv = matrix[i, j, r, c]
                        out[i, j, r, c] = (
                            (m_re + coeff * (cv.real - m_re))
                            + 1j * (m_im + coeff * (cv.imag - m_im))
                        )
        return out


# ---------------------------------------------------------------------------
# Pure-Python fallback (used when numba is not installed)
# ---------------------------------------------------------------------------

def _grow_neighborhood(
    span: np.ndarray,
    r: int,
    c: int,
    seed_mean: float,
    similarity_threshold: float,
    max_pixels: int,
) -> List[Tuple[int, int]]:
    """BFS neighborhood growing — pure-Python fallback."""
    rows, cols = span.shape
    tol = similarity_threshold * max(seed_mean, 1e-10)
    lo  = seed_mean - tol
    hi  = seed_mean + tol

    visited = np.zeros((rows, cols), dtype=bool)
    visited[r, c] = True
    neighborhood: List[Tuple[int, int]] = [(r, c)]
    queue:        List[Tuple[int, int]] = [(r, c)]

    while queue and len(neighborhood) < max_pixels:
        pr, pc = queue.pop(0)
        for nr, nc in ((pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                visited[nr, nc] = True
                val = float(span[nr, nc])
                if lo <= val <= hi:
                    neighborhood.append((nr, nc))
                    queue.append((nr, nc))
                    if len(neighborhood) >= max_pixels:
                        break

    return neighborhood


def _idan_filter_matrix_impl(
    matrix: np.ndarray,
    kernel_size: int,
    similarity_threshold: float,
    max_pixels: int,
    enl: float,
) -> np.ndarray:
    """Core IDAN filtering loop.

    Dispatches to the numba JIT implementation when available, otherwise
    falls back to a pure-Python loop with a one-time warning.
    """
    n, _, rows, cols = matrix.shape
    half   = kernel_size // 2
    sigma2 = 1.0 / enl

    span = np.zeros((rows, cols), dtype=np.float64)
    for i in range(n):
        span += np.real(matrix[i, i])

    span_padded = np.pad(span, half, mode='reflect')

    if _HAS_NUMBA:
        return _idan_filter_nb_parallel(
            matrix.astype(np.complex128),
            span, span_padded,
            kernel_size, sigma2, similarity_threshold, max_pixels,
        )

    # --- Pure-Python fallback ---
    warnings.warn(
        "numba is not installed — IDANFilter running in pure Python, which is "
        "very slow for large images.  Install numba: pip install grdl[polsar]",
        RuntimeWarning,
        stacklevel=4,
    )

    out = np.empty_like(matrix)

    for r in range(rows):
        for c in range(cols):
            seed_patch = span_padded[r:r + kernel_size, c:c + kernel_size]
            seed_mean  = float(np.mean(seed_patch))

            pixels = _grow_neighborhood(
                span, r, c, seed_mean, similarity_threshold, max_pixels
            )

            if len(pixels) < 2:
                coeff = 1.0
            else:
                nbr_span = np.array([span[pr, pc] for pr, pc in pixels],
                                    dtype=np.float64)
                m_span  = nbr_span.mean()
                cv2     = nbr_span.var() / (m_span * m_span + 1e-20)
                coeff   = (cv2 - sigma2) / (cv2 * (1.0 + sigma2) + 1e-20)
                coeff   = max(0.0, min(1.0, coeff))

            rows_idx = [pr for pr, pc in pixels]
            cols_idx = [pc for pr, pc in pixels]

            for i in range(n):
                for j in range(n):
                    elem     = matrix[i, j]
                    mean_val = elem[rows_idx, cols_idx].mean()
                    out[i, j, r, c] = mean_val + coeff * (elem[r, c] - mean_val)

    return out


# ---------------------------------------------------------------------------
# Public filter class
# ---------------------------------------------------------------------------

@processor_version('1.0.0')
@processor_tags(category=ProcessorCategory.FILTERS,
                modalities=[ImageModality.SAR])
class IDANFilter(PolarimetricFilter):
    """Intensity-Driven Adaptive-Neighborhood (IDAN) speckle filter.

    Reduces speckle by filtering each pixel within its own adaptive
    neighborhood rather than a fixed window. The neighborhood is grown
    from a seed window by including 4-connected pixels whose intensity
    falls within a similarity band around the seed mean. This avoids
    crossing edges while still gathering enough samples for reliable
    MMSE estimation.

    For polarimetric data (covariance or coherency matrix) the same
    adaptive neighborhood is used for every matrix element, preserving
    inter-channel correlations. Use ``apply(matrix, input_kind='matrix')``
    for preformed C2/T2/C3/T3/C4/T4 tensors. Use ``input_kind='channels'``
    and an explicit C2/T2/C3/T3 matrix_type to form a matrix from CYX SLC
    channels. Other YX/CYX input defaults to independent image filtering;
    a complex cube alone does not select joint polarimetric processing.
    Both paths honor per-call parameter overrides without instance mutation.

    Parameters
    ----------
    kernel_size : int
        Seed window side length in pixels.  Must be odd and in [3, 31].
        Default is 7.
    max_pixels : int
        Maximum size of the adaptive neighborhood.  Larger values allow
        more smoothing in homogeneous regions at the cost of computation.
        Default is 50.
    similarity_threshold : float
        Half-width of the intensity tolerance band as a fraction of the
        seed-window mean.  Smaller values grow tighter, more selective
        neighborhoods.  Default is 0.5.
    enl : float
        Equivalent Number of Looks.  Controls the noise variance
        σ² = 1/ENL used in the MMSE coefficient.  Set to 0 for
        automatic estimation from the image.  Default is 0.0 (auto).

    References
    ----------
    Vasile, G., Trovè, E., Lee, J.-S., and Buzuloiu, V. (2006).
        "Intensity-Driven Adaptive-Neighborhood Technique for Polarimetric
        and Interferometric SAR Parameters Estimation," IEEE Transactions
        on Geoscience and Remote Sensing, 44(6), 1609–1621.
        DOI: 10.1109/TGRS.2006.873742

    Examples
    --------
    Scalar (intensity) filtering:

    >>> from grdl.image_processing.filters import IDANFilter
    >>> f = IDANFilter(kernel_size=7, max_pixels=50)
    >>> intensity_filtered = f.apply(intensity_image)  # (rows, cols)

    Polarimetric matrix filtering:

    >>> c3_filtered = f.apply(c3, input_kind='matrix', matrix_type='C3')
    >>> channels = np.stack([shh, shv, svh, svv])
    >>> c3_filt = f.apply(channels, input_kind='channels', matrix_type='C3')
    """

    # The adaptive neighborhood has no fixed extent, so tiles cannot
    # reproduce a full-image result exactly.
    tile_halo = None
    _supports_images = True

    max_pixels: Annotated[
        int,
        Range(min=5, max=500),
        Desc('Maximum adaptive neighborhood size'),
    ] = 50

    similarity_threshold: Annotated[
        float,
        Range(min=0.0, max=1.0),
        Desc('Intensity tolerance band as fraction of seed mean'),
    ] = 0.5

    def __init__(
        self,
        kernel_size: int = 7,
        max_pixels: int = 50,
        similarity_threshold: float = 0.5,
        enl: float = 0.0,
    ) -> None:
        self.kernel_size = kernel_size
        self.enl = enl
        self.max_pixels = max_pixels
        self.similarity_threshold = similarity_threshold
        self._resolve_params({})

    # ------------------------------------------------------------------
    # Scalar (per-band) path via BandwiseTransformMixin
    # ------------------------------------------------------------------

    def _apply_2d(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply IDAN to a single 2D real or complex image.

        Called by ``BandwiseTransformMixin.apply()`` for each band.
        Wraps the image in a pseudo 1×1×rows×cols polarimetric matrix
        to reuse the core IDAN loop, then unwraps the result.
        """
        params = self._resolve_params(kwargs)
        rows, cols = image.shape

        # Auto-estimate ENL if needed
        if params['enl'] == 0.0:
            amp = np.abs(image).astype(np.float64)
            mean2 = (np.mean(amp)) ** 2
            var = np.var(amp)
            ci2 = var / (mean2 + 1e-20)
            enl = self._estimate_enl(ci2 * np.ones(1))
        else:
            enl = float(params['enl'])

        # Wrap in pseudo-matrix (1, 1, rows, cols)
        mat = image.reshape(1, 1, rows, cols)
        filt = _idan_filter_matrix_impl(
            mat, params['kernel_size'], params['similarity_threshold'],
            params['max_pixels'], enl
        )
        result = filt[0, 0]

        if np.iscomplexobj(image):
            return result.astype(image.dtype)
        return np.real(result).astype(image.dtype)

    # ------------------------------------------------------------------
    # Polarimetric matrix paths
    # ------------------------------------------------------------------

    def _filter_matrix(self, matrix: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        """Filter a polarimetric covariance or coherency matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Shape (N, N, rows, cols) where N ∈ {2, 3, 4}.
            Validated real or complex polarimetric tensor.
        params : dict[str, Any]
            Validated kernel_size, max_pixels, similarity_threshold, and enl.

        Returns
        -------
        np.ndarray
            Filtered matrix with the same shape and dtype as input.
        """
        n = matrix.shape[0]

        # Auto-estimate ENL from span if needed
        if params['enl'] == 0.0:
            span = np.zeros(matrix.shape[2:], dtype=np.float64)
            for i in range(n):
                span += np.real(matrix[i, i])
            enl = self._estimate_enl(
                np.var(span) / (np.mean(span) ** 2 + 1e-20) * np.ones(1)
            )
            enl = max(enl, 1.0)
        else:
            enl = float(params['enl'])

        logger.debug(
            'IDANFilter: kernel_size=%d, max_pixels=%d, '
            'similarity_threshold=%.3f, enl=%.2f',
            params['kernel_size'], params['max_pixels'],
            params['similarity_threshold'], enl,
        )

        return _idan_filter_matrix_impl(
            matrix.astype(np.complex128),
            params['kernel_size'],
            params['similarity_threshold'],
            params['max_pixels'],
            enl,
        ).astype(matrix.dtype)

