# -*- coding: utf-8 -*-
"""
Compute backend hierarchy for grdl.shapes.

Dispatches every hot path to the fastest available backend at runtime:

1. GPU array backend (CuPy, NumPy drop-in) for large-N coordinate math.
2. Numba JIT for inherently sequential tight loops (adaptive refinement,
   geodesic-ellipse bisection).
3. Multiprocess / multithread executors for independent per-shape work.
4. NumPy vectorization as the baseline.

Every promotion is guarded and falls back to NumPy with a UserWarning
when unavailable. Public APIs accept and return numpy.ndarray only --
GPU arrays are an internal implementation detail.

Dependencies
------------
numpy (core)
cupy (optional, GPU)
numba (optional, JIT)

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
2026-04-18

Modified
--------
2026-04-18
"""

# Standard library
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Iterable, List, Optional

# Third-party
import numpy as np


logger = logging.getLogger(__name__)


# ── Backend availability probes (performed once at import) ────────────

_CUPY_AVAILABLE = False
_NUMBA_AVAILABLE = False
_CUCIM_AVAILABLE = False

try:
    import cupy as _cupy  # noqa: F401
    # Probe a real device; installed cupy without CUDA still raises here.
    _cupy.asarray([0.0])
    _CUPY_AVAILABLE = True
except Exception:
    _cupy = None

try:
    import numba as _numba  # noqa: F401
    _NUMBA_AVAILABLE = True
except ImportError:
    _numba = None

try:
    import cucim  # noqa: F401
    _CUCIM_AVAILABLE = True
except ImportError:
    pass


# ── Backend configuration ─────────────────────────────────────────────

@dataclass
class ComputeBackend:
    """Runtime configuration for the shapes compute backend."""

    array: str = 'auto'           # 'cupy' | 'numpy' | 'auto'
    jit: str = 'auto'             # 'numba' | 'none' | 'auto'
    parallel: str = 'auto'        # 'process' | 'thread' | 'none' | 'auto'
    gpu_threshold: int = 10_000   # min N before GPU promotion is worthwhile
    max_workers: Optional[int] = None  # default: os.cpu_count()

    # Cached resolved values (populated by resolve())
    _array_module: Optional[ModuleType] = field(default=None, repr=False)
    _jit_decorator: Optional[Callable] = field(default=None, repr=False)

    def resolve(self) -> None:
        """Populate cached values based on availability and env overrides."""
        # ── Array backend ────────────────────────────────────────────
        req = self.array
        if req == 'auto':
            req = 'cupy' if _CUPY_AVAILABLE else 'numpy'
        if req == 'cupy':
            if _CUPY_AVAILABLE:
                self._array_module = _cupy
                self.array = 'cupy'
            else:
                warnings.warn(
                    "cupy requested but not available; falling back to numpy.",
                    UserWarning, stacklevel=2,
                )
                self._array_module = np
                self.array = 'numpy'
        elif req == 'numpy':
            self._array_module = np
            self.array = 'numpy'
        else:
            raise ValueError(f"Unknown array backend: {req!r}")

        # ── JIT ──────────────────────────────────────────────────────
        req = self.jit
        if req == 'auto':
            req = 'numba' if _NUMBA_AVAILABLE else 'none'
        if req == 'numba':
            if _NUMBA_AVAILABLE:
                self._jit_decorator = _numba.njit(cache=True, fastmath=False)
                self.jit = 'numba'
            else:
                warnings.warn(
                    "numba requested but not available; JIT disabled.",
                    UserWarning, stacklevel=2,
                )
                self._jit_decorator = _identity_decorator
                self.jit = 'none'
        elif req == 'none':
            self._jit_decorator = _identity_decorator
            self.jit = 'none'
        else:
            raise ValueError(f"Unknown jit backend: {req!r}")

        # ── Parallel ─────────────────────────────────────────────────
        if self.parallel == 'auto':
            # Threads are safer by default (skimage / pyproj release the GIL)
            # and avoid pickling cost. Process pool is opt-in.
            self.parallel = 'thread'
        if self.parallel not in ('process', 'thread', 'none'):
            raise ValueError(f"Unknown parallel backend: {self.parallel!r}")


def _identity_decorator(func):
    return func


# ── Singleton state ───────────────────────────────────────────────────

_BACKEND: Optional[ComputeBackend] = None


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def detect_backend() -> ComputeBackend:
    """Return the current backend, probing once and caching the result.

    Respects environment variables:

    - ``GRDL_SHAPES_BACKEND``  : ``cupy`` | ``numpy`` | ``auto`` (default)
    - ``GRDL_SHAPES_JIT``      : ``numba`` | ``none`` | ``auto`` (default)
    - ``GRDL_SHAPES_PARALLEL`` : ``process`` | ``thread`` | ``none`` | ``auto``
    - ``GRDL_SHAPES_GPU_THRESHOLD`` : integer, default 10000
    - ``GRDL_SHAPES_MAX_WORKERS`` : integer, default os.cpu_count()

    Returns
    -------
    ComputeBackend
        The resolved, cached backend. Call :func:`set_backend` to reset.
    """
    global _BACKEND
    if _BACKEND is None:
        try:
            threshold = int(_env('GRDL_SHAPES_GPU_THRESHOLD', '10000'))
        except ValueError:
            threshold = 10_000
        mw_env = os.environ.get('GRDL_SHAPES_MAX_WORKERS')
        max_workers = int(mw_env) if mw_env else None

        _BACKEND = ComputeBackend(
            array=_env('GRDL_SHAPES_BACKEND', 'auto'),
            jit=_env('GRDL_SHAPES_JIT', 'auto'),
            parallel=_env('GRDL_SHAPES_PARALLEL', 'auto'),
            gpu_threshold=threshold,
            max_workers=max_workers,
        )
        _BACKEND.resolve()
        logger.debug(
            "grdl.shapes backend resolved: array=%s jit=%s parallel=%s "
            "gpu_threshold=%d",
            _BACKEND.array, _BACKEND.jit, _BACKEND.parallel,
            _BACKEND.gpu_threshold,
        )
    return _BACKEND


def set_backend(
    array: Optional[str] = None,
    jit: Optional[str] = None,
    parallel: Optional[str] = None,
    gpu_threshold: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> ComputeBackend:
    """Override the cached backend. Useful for tests and explicit control.

    Any argument left as ``None`` keeps its current value. Resolving
    happens eagerly so fallback warnings fire at the call site.

    Returns
    -------
    ComputeBackend
        The newly resolved backend.
    """
    global _BACKEND
    current = _BACKEND if _BACKEND is not None else detect_backend()
    _BACKEND = ComputeBackend(
        array=array if array is not None else current.array,
        jit=jit if jit is not None else current.jit,
        parallel=parallel if parallel is not None else current.parallel,
        gpu_threshold=(
            gpu_threshold if gpu_threshold is not None
            else current.gpu_threshold
        ),
        max_workers=(
            max_workers if max_workers is not None else current.max_workers
        ),
    )
    _BACKEND.resolve()
    return _BACKEND


def get_array_module(n: int) -> ModuleType:
    """Return the array module to use for a problem of size ``n``.

    Returns cupy when the configured backend is cupy AND ``n`` meets the
    GPU threshold; otherwise numpy. Small problems stay on CPU because
    GPU dispatch overhead dominates below threshold.
    """
    backend = detect_backend()
    if (
        backend._array_module is _cupy
        and _CUPY_AVAILABLE
        and n >= backend.gpu_threshold
    ):
        return _cupy
    return np


def asnumpy(a: Any) -> np.ndarray:
    """Return ``a`` as a ``numpy.ndarray``, converting from cupy if needed."""
    if _CUPY_AVAILABLE and isinstance(a, _cupy.ndarray):
        return _cupy.asnumpy(a)
    return np.asarray(a)


def maybe_jit(func: Callable) -> Callable:
    """Decorator: JIT-compile ``func`` with numba if available, else passthrough.

    Applied at function definition time. The decorator is stable per
    backend configuration; if the backend is changed at runtime, the
    already-applied decorator sticks with the function. For tests that
    need to re-exercise JIT vs non-JIT paths, wrap calls in functions
    that check ``detect_backend().jit`` explicitly.
    """
    backend = detect_backend()
    try:
        return backend._jit_decorator(func)
    except Exception as exc:  # compile failure: fall back with a warning
        warnings.warn(
            f"numba failed to compile {func.__name__}: {exc}. "
            "Falling back to pure Python.",
            UserWarning, stacklevel=2,
        )
        return func


def batch_map(
    fn: Callable,
    items: Iterable[Any],
    parallel: Optional[str] = None,
) -> List[Any]:
    """Map ``fn`` over ``items`` using the configured parallel backend.

    Parameters
    ----------
    fn : Callable
        Function applied to each item. Must be picklable when the
        process pool is selected.
    items : Iterable
        Work items.
    parallel : str, optional
        ``'process'`` | ``'thread'`` | ``'none'``. Overrides the backend
        default for this call.

    Returns
    -------
    list
        Results in input order. Worker exceptions are re-raised on the
        main thread.
    """
    backend = detect_backend()
    mode = parallel if parallel is not None else backend.parallel
    items = list(items)

    if mode == 'none' or len(items) <= 1:
        return [fn(x) for x in items]

    max_workers = backend.max_workers or os.cpu_count() or 1
    n_workers = min(max_workers, len(items))

    if mode == 'thread':
        executor_cls = ThreadPoolExecutor
    elif mode == 'process':
        executor_cls = ProcessPoolExecutor
    else:
        raise ValueError(f"Unknown parallel mode: {mode!r}")

    with executor_cls(max_workers=n_workers) as executor:
        return list(executor.map(fn, items))


# ── Convenience re-exports for consumers ──────────────────────────────

def cupy_available() -> bool:
    """True when cupy is importable and a device was detected."""
    return _CUPY_AVAILABLE


def numba_available() -> bool:
    """True when numba is importable."""
    return _NUMBA_AVAILABLE


def cucim_available() -> bool:
    """True when cucim is importable (GPU rasterization)."""
    return _CUCIM_AVAILABLE


# Forward decls for batch ops implemented in shapes.base; importing here
# would create circular imports. The __init__ module re-binds these.
to_pixels_batch: Optional[Callable] = None
rasterize_batch: Optional[Callable] = None
