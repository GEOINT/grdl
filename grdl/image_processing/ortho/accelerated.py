# -*- coding: utf-8 -*-
"""
Accelerated Resampling Backends for Orthorectification.

Provides a unified ``resample()`` function that auto-dispatches to
the fastest available backend:

1. **numba** — JIT-compiled kernels with ``prange`` across rows.
   All CPU cores, no framework dependency (5-15x over scipy).
   Preferred default when installed.
2. **torch GPU** — ``torch.nn.functional.grid_sample`` on CUDA/MPS.
   Fastest for large images (10-50x over scipy).
3. **torch CPU** — Same kernel on CPU. PyTorch's internal thread pool
   beats single-threaded scipy (3-8x).
4. **scipy_parallel** — ``map_coordinates`` chunked across a
   ``ThreadPoolExecutor``. Scipy releases the GIL so threads
   scale well (2-6x).
5. **scipy** — Single-threaded fallback.

The numba backend supports order 0 (nearest), 1 (bilinear), 3 (Keys
cubic, 4x4), and 5 (Lanczos-3, 6x6). Torch supports 0 and 1.

Dependencies
------------
scipy
numba (optional — JIT parallel acceleration, preferred)
torch (optional — GPU/CPU acceleration)

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
2026-03-08

Modified
--------
2026-03-08
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Backend availability ───────────────────────────────────────────

_TORCH_AVAILABLE = False
_NUMBA_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import numba
    _NUMBA_AVAILABLE = True
except ImportError:
    pass


def detect_backend(prefer: str = 'auto') -> str:
    """Return the best available backend name.

    Parameters
    ----------
    prefer : str
        Preferred backend.  ``'auto'`` picks the fastest available.
        Valid values: ``'auto'``, ``'torch_gpu'``, ``'torch'``,
        ``'numba'``, ``'scipy_parallel'``, ``'scipy'``.

    Returns
    -------
    str
        Backend name string.
    """
    if prefer != 'auto':
        return prefer

    if _NUMBA_AVAILABLE:
        return 'numba'

    if _TORCH_AVAILABLE:
        if torch.cuda.is_available():
            return 'torch_gpu'
        if (hasattr(torch.backends, 'mps')
                and torch.backends.mps.is_available()):
            return 'torch_gpu'
        return 'torch'

    return 'scipy_parallel'


def _get_torch_device() -> 'torch.device':
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if (hasattr(torch.backends, 'mps')
            and torch.backends.mps.is_available()):
        return torch.device('mps')
    return torch.device('cpu')


# ── Main entry point ───────────────────────────────────────────────

def resample(
    image: np.ndarray,
    row_map: np.ndarray,
    col_map: np.ndarray,
    valid_mask: np.ndarray,
    order: int = 1,
    nodata: float = 0.0,
    backend: str = 'auto',
    num_workers: Optional[int] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """Resample *image* using coordinate maps.

    This is the main acceleration entry point used by
    ``Orthorectifier``.  It dispatches to the fastest available
    backend while preserving the input dtype.

    Parameters
    ----------
    image : np.ndarray
        Source image, shape ``(H, W)`` or ``(B, H, W)``.
        Can be real or complex.
    row_map : np.ndarray
        Source row coordinates for every output pixel,
        shape ``(OH, OW)``.
    col_map : np.ndarray
        Source column coordinates, shape ``(OH, OW)``.
    valid_mask : np.ndarray
        Boolean mask, shape ``(OH, OW)``.  Only valid pixels
        are sampled; others get *nodata*.
    order : int
        Interpolation order (0=nearest, 1=bilinear, 3=cubic,
        5=lanczos-3).
    nodata : float
        Fill value for invalid pixels.
    backend : str
        ``'auto'``, ``'torch_gpu'``, ``'torch'``, ``'numba'``,
        ``'scipy_parallel'``, ``'scipy'``.
    num_workers : int, optional
        Thread count for ``scipy_parallel``.
        Defaults to ``os.cpu_count() - 1``.
    device : str, optional
        Torch device override (``'cuda'``, ``'mps'``, ``'cpu'``).

    Returns
    -------
    np.ndarray
        Resampled image with shape ``(OH, OW)`` or
        ``(B, OH, OW)``, same dtype as *image*.
    """
    chosen = detect_backend(backend)

    # Torch only supports order 0/1; fall back for higher
    if order > 1 and chosen in ('torch', 'torch_gpu'):
        if _NUMBA_AVAILABLE:
            chosen = 'numba'
        else:
            chosen = 'scipy_parallel'

    # Numba supports 0, 1, 3, 5; other orders fall back
    if chosen == 'numba' and order not in (0, 1, 3, 5):
        chosen = 'scipy_parallel'

    logger.info("resample: backend=%s, order=%d", chosen, order)

    if chosen == 'torch_gpu':
        dev = device or str(_get_torch_device())
        return _resample_torch(
            image, row_map, col_map, valid_mask,
            order=order, nodata=nodata, device=dev,
        )

    if chosen == 'torch':
        return _resample_torch(
            image, row_map, col_map, valid_mask,
            order=order, nodata=nodata, device='cpu',
        )

    if chosen == 'numba':
        return _resample_numba(
            image, row_map, col_map, valid_mask,
            order=order, nodata=nodata,
        )

    if chosen == 'scipy_parallel':
        return _resample_scipy_parallel(
            image, row_map, col_map, valid_mask,
            order=order, nodata=nodata,
            num_workers=num_workers,
        )

    return _resample_scipy(
        image, row_map, col_map, valid_mask,
        order=order, nodata=nodata,
    )


# ── PyTorch backend ───────────────────────────────────────────────

def _resample_torch(
    image: np.ndarray,
    row_map: np.ndarray,
    col_map: np.ndarray,
    valid_mask: np.ndarray,
    order: int = 1,
    nodata: float = 0.0,
    device: str = 'cpu',
) -> np.ndarray:
    """Resample using ``torch.nn.functional.grid_sample``.

    Handles 2D/3D (bands-first), real/complex.
    """
    in_dtype = image.dtype
    if np.isnan(nodata) and np.issubdtype(in_dtype, np.integer):
        in_dtype = np.float32
    is_complex = np.iscomplexobj(image)
    is_multiband = image.ndim == 3

    if is_multiband:
        H, W = image.shape[1], image.shape[2]
    else:
        H, W = image.shape
    OH, OW = row_map.shape

    mode = 'nearest' if order == 0 else 'bilinear'
    dev = torch.device(device)

    # Normalised sample grid: coords in [-1, 1]
    norm_row = np.where(
        valid_mask,
        2.0 * row_map / (H - 1) - 1.0,
        -2.0,
    ).astype(np.float32)
    norm_col = np.where(
        valid_mask,
        2.0 * col_map / (W - 1) - 1.0,
        -2.0,
    ).astype(np.float32)

    # grid_sample wants (x, y) = (col, row), shape (1, OH, OW, 2)
    grid = torch.from_numpy(
        np.stack([norm_col, norm_row], axis=-1)[np.newaxis]
    ).to(dev)

    # Prepare input tensor
    if is_complex:
        planes = np.stack(
            [image.real.astype(np.float32),
             image.imag.astype(np.float32)],
            axis=0,
        )
        inp = torch.from_numpy(planes[np.newaxis]).to(dev)
    elif is_multiband:
        # Already (B, H, W) → (1, B, H, W)
        inp = torch.from_numpy(
            image.astype(np.float32)[np.newaxis]
        ).to(dev)
    else:
        inp = torch.from_numpy(
            image.astype(np.float32)[np.newaxis, np.newaxis]
        ).to(dev)

    with torch.no_grad():
        out_t = F.grid_sample(
            inp, grid, mode=mode,
            padding_mode='zeros', align_corners=True,
        )

    out_np = out_t.cpu().numpy()  # (1, C, OH, OW)

    if is_complex:
        result = np.empty((OH, OW), dtype=in_dtype)
        result.real = out_np[0, 0]
        result.imag = out_np[0, 1]
        result[~valid_mask] = complex(nodata, 0)
    elif is_multiband:
        result = out_np[0].astype(in_dtype)  # (B, OH, OW)
        result[:, ~valid_mask] = nodata
    else:
        result = out_np[0, 0].astype(in_dtype)
        result[~valid_mask] = nodata

    return result


# ── Numba backend ──────────────────────────────────────────────────

if _NUMBA_AVAILABLE:
    @numba.njit(parallel=True, cache=True)
    def _numba_bilinear_2d(
        image: np.ndarray,
        row_map: np.ndarray,
        col_map: np.ndarray,
        valid_mask: np.ndarray,
        nodata: float,
    ) -> np.ndarray:
        """Bilinear resample a single 2D plane."""
        H, W = image.shape
        OH, OW = row_map.shape
        out = np.full((OH, OW), nodata, dtype=image.dtype)
        for i in numba.prange(OH):
            for j in range(OW):
                if not valid_mask[i, j]:
                    continue
                r = row_map[i, j]
                c = col_map[i, j]
                r0 = int(np.floor(r))
                c0 = int(np.floor(c))
                r1 = r0 + 1
                c1 = c0 + 1
                if r0 < 0 or r1 >= H or c0 < 0 or c1 >= W:
                    continue
                dr = r - r0
                dc = c - c0
                out[i, j] = (
                    image[r0, c0] * (1.0 - dr) * (1.0 - dc)
                    + image[r0, c1] * (1.0 - dr) * dc
                    + image[r1, c0] * dr * (1.0 - dc)
                    + image[r1, c1] * dr * dc
                )
        return out

    @numba.njit(cache=True)
    def _cubic_weight(t: float) -> float:
        """Keys cubic weight (a=-0.5, Catmull-Rom)."""
        at = abs(t)
        if at <= 1.0:
            return (1.5 * at - 2.5) * at * at + 1.0
        if at <= 2.0:
            return ((-0.5 * at + 2.5) * at - 4.0) * at + 2.0
        return 0.0

    @numba.njit(parallel=True, cache=True)
    def _numba_cubic_2d(
        image: np.ndarray,
        row_map: np.ndarray,
        col_map: np.ndarray,
        valid_mask: np.ndarray,
        nodata: float,
    ) -> np.ndarray:
        """Cubic resample (4x4 Keys kernel) with prange."""
        H, W = image.shape
        OH, OW = row_map.shape
        out = np.full((OH, OW), nodata, dtype=image.dtype)
        for i in numba.prange(OH):
            for j in range(OW):
                if not valid_mask[i, j]:
                    continue
                r = row_map[i, j]
                c = col_map[i, j]
                ri = int(np.floor(r))
                ci = int(np.floor(c))
                if (ri - 1 < 0 or ri + 2 >= H
                        or ci - 1 < 0 or ci + 2 >= W):
                    continue
                val = 0.0
                for kr in range(4):
                    wr = _cubic_weight(r - (ri - 1 + kr))
                    for kc in range(4):
                        wc = _cubic_weight(c - (ci - 1 + kc))
                        val += (
                            image[ri - 1 + kr, ci - 1 + kc]
                            * wr * wc
                        )
                out[i, j] = val
        return out

    @numba.njit(cache=True)
    def _lanczos3_weight(t: float) -> float:
        """Lanczos-3 kernel weight (6-tap sinc window)."""
        at = abs(t)
        if at < 1.0e-8:
            return 1.0
        if at >= 3.0:
            return 0.0
        pt = np.pi * t
        return (
            np.sin(pt) / pt
            * np.sin(pt / 3.0) / (pt / 3.0)
        )

    @numba.njit(parallel=True, cache=True)
    def _numba_lanczos3_2d(
        image: np.ndarray,
        row_map: np.ndarray,
        col_map: np.ndarray,
        valid_mask: np.ndarray,
        nodata: float,
    ) -> np.ndarray:
        """Lanczos-3 resample (6x6 kernel) with prange."""
        H, W = image.shape
        OH, OW = row_map.shape
        out = np.full((OH, OW), nodata, dtype=image.dtype)
        for i in numba.prange(OH):
            for j in range(OW):
                if not valid_mask[i, j]:
                    continue
                r = row_map[i, j]
                c = col_map[i, j]
                ri = int(np.floor(r))
                ci = int(np.floor(c))
                if (ri - 2 < 0 or ri + 3 >= H
                        or ci - 2 < 0 or ci + 3 >= W):
                    continue
                val = 0.0
                wsum = 0.0
                for kr in range(6):
                    wr = _lanczos3_weight(r - (ri - 2 + kr))
                    for kc in range(6):
                        wc = _lanczos3_weight(c - (ci - 2 + kc))
                        w = wr * wc
                        val += (
                            image[ri - 2 + kr, ci - 2 + kc] * w
                        )
                        wsum += w
                if abs(wsum) > 1.0e-12:
                    out[i, j] = val / wsum
                else:
                    out[i, j] = val
        return out

    _NUMBA_KERNELS = {
        1: _numba_bilinear_2d,
        3: _numba_cubic_2d,
        5: _numba_lanczos3_2d,
    }


def _resample_numba(
    image: np.ndarray,
    row_map: np.ndarray,
    col_map: np.ndarray,
    valid_mask: np.ndarray,
    order: int = 1,
    nodata: float = 0.0,
) -> np.ndarray:
    """Resample using numba-compiled kernels with prange."""
    in_dtype = image.dtype
    # Promote integer dtypes to float32 when nodata is NaN, since
    # integer types cannot represent NaN.
    if np.isnan(nodata) and np.issubdtype(in_dtype, np.integer):
        in_dtype = np.float32
    is_complex = np.iscomplexobj(image)
    is_multiband = image.ndim == 3

    effective_order = order if order in (1, 3, 5) else 1
    kernel = _NUMBA_KERNELS[effective_order]

    rm = row_map.astype(np.float64)
    cm = col_map.astype(np.float64)

    if order == 0:
        rm = np.floor(rm + 0.5)
        cm = np.floor(cm + 0.5)

    def _apply(plane: np.ndarray) -> np.ndarray:
        return kernel(
            plane.astype(np.float64),
            rm, cm, valid_mask, float(nodata),
        )

    if is_complex:
        real_out = _apply(image.real)
        imag_out = kernel(
            image.imag.astype(np.float64),
            rm, cm, valid_mask, 0.0,
        )
        result = np.empty(row_map.shape, dtype=in_dtype)
        result.real = real_out
        result.imag = imag_out
        return result

    if is_multiband:
        bands = []
        for b in range(image.shape[0]):
            bands.append(_apply(image[b]).astype(in_dtype))
        return np.stack(bands, axis=0)

    return _apply(image).astype(in_dtype)


# ── Scipy parallel backend ────────────────────────────────────────

def _resample_scipy_parallel(
    image: np.ndarray,
    row_map: np.ndarray,
    col_map: np.ndarray,
    valid_mask: np.ndarray,
    order: int = 1,
    nodata: float = 0.0,
    num_workers: Optional[int] = None,
) -> np.ndarray:
    """Parallel ``map_coordinates`` using ThreadPoolExecutor.

    Scipy releases the GIL during ``map_coordinates``, so threads
    scale linearly on multi-core machines.
    """
    from concurrent.futures import ThreadPoolExecutor
    from scipy.ndimage import map_coordinates

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 1) - 1)

    in_dtype = image.dtype
    if np.isnan(nodata) and np.issubdtype(in_dtype, np.integer):
        in_dtype = np.float32
    is_complex = np.iscomplexobj(image)
    is_multiband = image.ndim == 3
    OH, OW = row_map.shape

    r = np.where(valid_mask, row_map, 0.0)
    c = np.where(valid_mask, col_map, 0.0)

    # Collect planes to resample
    planes = []
    if is_complex:
        planes.append((image.real.astype(np.float64), order))
        planes.append((image.imag.astype(np.float64), order))
    elif is_multiband:
        for b in range(image.shape[0]):
            planes.append((image[b].astype(np.float64), order))
    else:
        planes.append((image.astype(np.float64), order))

    # Chunk rows for parallelism
    chunk_h = max(1, OH // num_workers)
    row_slices = []
    for s in range(0, OH, chunk_h):
        row_slices.append(slice(s, min(s + chunk_h, OH)))

    def _process_chunk(
        plane: np.ndarray, interp_order: int, rs: slice,
    ) -> Tuple[slice, np.ndarray]:
        coords = np.array([r[rs], c[rs]])
        vals = map_coordinates(
            plane, coords, order=interp_order,
            mode='constant', cval=float(nodata),
        )
        return rs, vals

    results_per_plane = [
        np.empty((OH, OW), dtype=np.float64) for _ in planes
    ]

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = []
        for p_idx, (plane, interp_order) in enumerate(planes):
            for rs in row_slices:
                fut = pool.submit(
                    _process_chunk, plane, interp_order, rs,
                )
                futures.append((p_idx, fut))

        for p_idx, fut in futures:
            rs, vals = fut.result()
            results_per_plane[p_idx][rs] = vals

    for arr in results_per_plane:
        arr[~valid_mask] = nodata

    if is_complex:
        result = np.empty((OH, OW), dtype=in_dtype)
        result.real = results_per_plane[0]
        result.imag = results_per_plane[1]
        return result

    if is_multiband:
        return np.stack(
            [a.astype(in_dtype) for a in results_per_plane], axis=0,
        )

    return results_per_plane[0].astype(in_dtype)


# ── Scipy sequential backend ──────────────────────────────────────

def _resample_scipy(
    image: np.ndarray,
    row_map: np.ndarray,
    col_map: np.ndarray,
    valid_mask: np.ndarray,
    order: int = 1,
    nodata: float = 0.0,
) -> np.ndarray:
    """Single-threaded ``map_coordinates`` fallback."""
    from scipy.ndimage import map_coordinates

    in_dtype = image.dtype
    if np.isnan(nodata) and np.issubdtype(in_dtype, np.integer):
        in_dtype = np.float32
    is_complex = np.iscomplexobj(image)
    is_multiband = image.ndim == 3
    OH, OW = row_map.shape

    r = np.where(valid_mask, row_map, 0.0)
    c = np.where(valid_mask, col_map, 0.0)
    coords = np.array([r, c])

    if is_complex:
        real_out = map_coordinates(
            image.real.astype(np.float64), coords,
            order=order, mode='constant', cval=0.0,
        )
        imag_out = map_coordinates(
            image.imag.astype(np.float64), coords,
            order=order, mode='constant', cval=0.0,
        )
        result = np.empty((OH, OW), dtype=in_dtype)
        result.real = real_out
        result.imag = imag_out
        result[~valid_mask] = complex(nodata, 0)
        return result

    if is_multiband:
        bands = []
        for b in range(image.shape[0]):
            vals = map_coordinates(
                image[b].astype(np.float64), coords,
                order=order, mode='constant', cval=float(nodata),
            )
            vals[~valid_mask] = nodata
            bands.append(vals.astype(in_dtype))
        return np.stack(bands, axis=0)

    vals = map_coordinates(
        image.astype(np.float64), coords,
        order=order, mode='constant', cval=float(nodata),
    )
    vals[~valid_mask] = nodata
    return vals.astype(in_dtype)
