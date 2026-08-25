# -*- coding: utf-8 -*-
"""
Quick-look Generator - Batch PNG generation from SICD NITF files.

Scans a directory for SICD NITF files, forms a mean-power multilook
image, applies Mangis density remap for improved SAR contrast (matching
GRDK display), and saves a PNG with the same file stem.

Two multilook modes are supported (selectable at runtime):

**Spatial-domain (default)**
  Computes per-pixel intensity (|z|²), then averages over non-overlapping
  ``looks_rg × looks_az`` blocks.  No FFT, no GPU required, very low memory
  footprint. The output image is smaller by the multilook factor in each
  dimension.

**Frequency-domain** (``--freq-multilook`` / ``spatial_multilook=False``)
  Uses ``MultilookDecomposition`` (sub-aperture spectral splitting via 2-D
  FFT). Preserves coherence information across looks; useful if you need
  sub-aperture fidelity (e.g., for subsequent CSI detection). GPU-accelerated
  when torch/CUDA or CuPy is available. Output image is same size as input.

GPU acceleration (frequency-domain mode only):

- **PyTorch / CUDA**: if ``torch`` is installed and a CUDA device is
  present, the image is moved to the GPU as a complex tensor before
  calling ``MultilookDecomposition.decompose()``.
- **CuPy**: if ``cupy`` is installed and CUDA is available, the image
  is wrapped in a ``cupy.ndarray`` before decomposition.
- **CPU fallback**: used when neither GPU back-end is available.

CUDA out-of-memory handling (frequency-domain mode only):

1. After every file the PyTorch CUDA cache is flushed with
   ``torch.cuda.empty_cache()`` so cached-but-unused memory is returned
   to the allocator before the next file starts.
2. If a CUDA out-of-memory error is raised during decomposition, the GPU
   cache is flushed and the file is automatically retried on CPU.

Usage (CLI)
-----------
::

    python -m grdl.discovery.gen_quicklooks /data/sar --looks 5
    python -m grdl.discovery.gen_quicklooks /data/sar --freq-multilook   # GPU
    python -m grdl.discovery.gen_quicklooks /data/sar --output /tmp/quicklooks

Usage (API)
-----------
::

    from grdl.discovery.gen_quicklooks import QuicklookGenerator

    # Spatial-domain (default, low memory)
    gen = QuicklookGenerator(looks_rg=10, looks_az=10)
    gen.run('/data/sar', output_dir='/tmp/ql')

    # Frequency-domain (GPU-accelerated, sub-aperture fidelity)
    gen = QuicklookGenerator(looks_rg=10, looks_az=10, spatial_multilook=False)
    gen.run('/data/sar')

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
import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

# Third-party
import numpy as np

# GRDL internal — GPU optional imports mirror the pattern in multilook.py
from grdl._torch_optional import torch, HAS_TORCH as _HAS_TORCH

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None  # type: ignore[assignment]
    _HAS_CUPY = False

logger = logging.getLogger(__name__)

# NITF extensions recognised as potential SICD files
_SICD_EXTENSIONS: Tuple[str, ...] = ('.nitf', '.ntf', '.nsf')

# Detect the OOM exception type at import time (None if torch absent/no CUDA).
_CudaOOMError: Optional[type] = None
if _HAS_TORCH:
    try:
        _CudaOOMError = torch.cuda.OutOfMemoryError  # type: ignore[union-attr]
    except AttributeError:
        # Older torch versions expose OOM only as RuntimeError
        _CudaOOMError = RuntimeError


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _detect_gpu() -> str:
    """Return the best available GPU back-end name, or ``'cpu'``.

    Priority: ``'torch'`` (CUDA) > ``'cupy'`` > ``'cpu'``.
    """
    if _HAS_TORCH:
        try:
            if torch.cuda.is_available():  # type: ignore[union-attr]
                return 'torch'
        except Exception:
            pass
    if _HAS_CUPY:
        try:
            if cp.cuda.is_available():
                return 'cupy'
        except Exception:
            pass
    return 'cpu'


def _to_device(image: np.ndarray, backend: str) -> object:
    """Move *image* to the requested compute device.

    Ensures native byte order before any GPU transfer — ``torch.from_numpy``
    and CuPy both require native endianness and will raise otherwise (e.g.
    when the SICD reader returns big-endian complex data on a little-endian
    host).

    Parameters
    ----------
    image : np.ndarray
        Complex 2-D SAR chip (CPU numpy array).
    backend : str
        One of ``'torch'``, ``'cupy'``, or ``'cpu'``.

    Returns
    -------
    np.ndarray | cupy.ndarray | torch.Tensor
        Array on the requested device; unchanged for ``'cpu'``.
    """
    # Normalise to native byte order so GPU back-ends don't reject the array.
    # np.dtype.isnative is the canonical check; copy=False avoids a copy when
    # the data is already native-endian.
    if not image.dtype.isnative:
        image = image.astype(image.dtype.newbyteorder('='), copy=False)
    # Ensure C-contiguous layout (required by torch.from_numpy)
    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)

    if backend == 'torch':
        return torch.from_numpy(image).to('cuda')  # type: ignore[union-attr]
    if backend == 'cupy':
        return cp.asarray(image)
    return image


def _to_numpy(result: object) -> np.ndarray:
    """Return *result* as a CPU numpy array regardless of origin device."""
    if _HAS_TORCH and isinstance(result, torch.Tensor):  # type: ignore[arg-type]
        return result.cpu().numpy()
    if _HAS_CUPY and isinstance(result, cp.ndarray):
        return cp.asnumpy(result)
    return np.asarray(result)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _mean_power(grid: np.ndarray) -> np.ndarray:
    """Average intensity (|z|^2) across all sub-looks.

    Parameters
    ----------
    grid : np.ndarray
        Complex multilook grid, shape ``(looks_rg, looks_az, rows, cols)``.

    Returns
    -------
    np.ndarray
        2-D float32 mean-power image, shape ``(rows, cols)``.
    """
    power = np.abs(grid) ** 2            # (M, N, rows, cols) float
    return power.mean(axis=(0, 1)).astype(np.float32)


def _save_png(image_u8: np.ndarray, path: Path) -> None:
    """Write a uint8 greyscale PNG to *path*.

    Uses ``Pillow`` when available, otherwise falls back to ``imageio``
    or a raw PNG written with only the standard library.

    Parameters
    ----------
    image_u8 : np.ndarray
        uint8 greyscale image, shape ``(rows, cols)``.
    path : Path
        Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image  # type: ignore[import-untyped]
        Image.fromarray(image_u8, mode='L').save(path)
        return
    except ImportError:
        pass
    try:
        import imageio  # type: ignore[import-untyped]
        imageio.imwrite(str(path), image_u8)
        return
    except ImportError:
        pass
    # Pure stdlib fallback via matplotlib (always available in GRDL envs)
    import matplotlib  # noqa: PLC0415
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt  # noqa: PLC0415
    fig, ax = plt.subplots(figsize=(image_u8.shape[1] / 100,
                                    image_u8.shape[0] / 100), dpi=100)
    ax.imshow(image_u8, cmap='gray', vmin=0, vmax=255)
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _clear_gpu_cache() -> None:
    """Release unused memory held by the PyTorch CUDA allocator.

    No-op when torch or CUDA is unavailable.  Should be called after every
    file is processed so the caching allocator returns pages to the OS before
    the next (potentially larger) file is loaded.
    """
    if _HAS_TORCH:
        try:
            torch.cuda.empty_cache()  # type: ignore[union-attr]
        except Exception:
            pass


def _is_cuda_oom(exc: BaseException) -> bool:
    """Return True if *exc* is a CUDA out-of-memory error."""
    if _CudaOOMError is not None and isinstance(exc, _CudaOOMError):
        # For older torch that maps OOM to plain RuntimeError, require the
        # canonical message text so we don't swallow unrelated RuntimeErrors.
        if _CudaOOMError is RuntimeError:
            return 'out of memory' in str(exc).lower()
        return True
    return False


def _spatial_mean_power(
    image: np.ndarray,
    looks_rg: int,
    looks_az: int,
) -> np.ndarray:
    """Compute mean intensity via spatial-domain block averaging.

    Computes per-pixel power (|z|²), then averages over non-overlapping
    ``looks_rg × looks_az`` pixel blocks.  The output image is reduced in
    size by the multilook factors::

        out_rows = rows // looks_rg
        out_cols = cols // looks_az

    Any trailing rows/cols that don't fill a complete block are discarded.
    This path requires no FFT and no GPU memory, making it suitable for
    large files or memory-constrained environments.

    Parameters
    ----------
    image : np.ndarray
        Complex 2-D SAR chip, shape ``(rows, cols)``.
    looks_rg : int
        Range block size (rows to average together).
    looks_az : int
        Azimuth block size (columns to average together).

    Returns
    -------
    np.ndarray
        Float32 mean-power image, shape
        ``(rows // looks_rg, cols // looks_az)``.
    """
    rows, cols = image.shape
    # Trim to an integer multiple of the block size
    r_trim = (rows // looks_rg) * looks_rg
    c_trim = (cols // looks_az) * looks_az
    power = (np.abs(image[:r_trim, :c_trim]) ** 2).astype(np.float32)
    # Reshape to (out_rows, looks_rg, out_cols, looks_az) then average
    out_rows = r_trim // looks_rg
    out_cols = c_trim // looks_az
    return (
        power
        .reshape(out_rows, looks_rg, out_cols, looks_az)
        .mean(axis=(1, 3))
    )


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

class QuicklookGenerator:
    """Generate PNG quick-looks from SICD NITF files.

    Applies Mangis density remap (logarithmic amplitude-to-density mapping)
    for improved SAR image contrast, matching the display used in GRDK.

    Parameters
    ----------
    looks_rg : int
        Range (row) multilook factor.  Default 10.
    looks_az : int
        Azimuth (column) multilook factor.  Default 10.
    overlap : float
        Sub-band overlap fraction, ``[0.0, 1.0)``.  Default 0.0.
        Only used in frequency-domain mode.
    deweight : bool
        Remove apodization weighting before multilook.  Default True.
        Only used in frequency-domain mode.
    spatial_multilook : bool
        If True (default), use spatial-domain block averaging.  Much lower
        memory footprint; no GPU required; output image is smaller by the
        look factors.  If False, use frequency-domain ``MultilookDecomposition``
        with GPU acceleration and automatic CPU fallback on out-of-memory.
    backend : str or None
        Force a specific compute back-end: ``'torch'``, ``'cupy'``, or
        ``'cpu'``.  ``None`` (default) auto-detects the best available.
        Ignored when ``spatial_multilook=True``.
    workers : int
        Number of parallel worker processes for batch scanning.  Default 1
        (serial, avoids GPU context forking issues).
    """

    def __init__(
        self,
        looks_rg: int = 10,
        looks_az: int = 10,
        overlap: float = 0.0,
        deweight: bool = True,
        spatial_multilook: bool = True,
        backend: Optional[str] = None,
        workers: int = 1,
    ) -> None:
        self.looks_rg = looks_rg
        self.looks_az = looks_az
        self.overlap = overlap
        self.deweight = deweight
        self.spatial_multilook = spatial_multilook
        self.backend: str = 'cpu' if spatial_multilook else (
            backend if backend is not None else _detect_gpu()
        )
        self.workers = max(1, workers)

    # ------------------------------------------------------------------
    # Directory scanning
    # ------------------------------------------------------------------

    def find_sicd_files(self, directory: Path) -> List[Path]:
        """Recursively find NITF files that appear to be SICD images.

        A file is accepted if:

        1. Its extension is in ``_SICD_EXTENSIONS``, and
        2. ``SICDReader`` can successfully open it (metadata check only).

        Parameters
        ----------
        directory : Path
            Root directory to search.

        Returns
        -------
        list of Path
            Sorted list of confirmed SICD file paths.
        """
        from grdl.IO.sar.sicd import SICDReader

        candidates = sorted(
            p for p in directory.rglob('*')
            if p.suffix.lower() in _SICD_EXTENSIONS
        )
        confirmed: List[Path] = []
        for path in candidates:
            try:
                with SICDReader(path) as reader:
                    # Minimal check: grid metadata must be present
                    if (
                        reader.metadata.grid is not None
                        and reader.metadata.grid.row is not None
                        and reader.metadata.grid.col is not None
                        and reader.metadata.grid.row.imp_resp_bw is not None
                        and reader.metadata.grid.col.imp_resp_bw is not None
                    ):
                        confirmed.append(path)
            except Exception as exc:
                logger.debug('Skipping %s: %s', path.name, exc)
        return confirmed

    # ------------------------------------------------------------------
    # Single-file processing
    # ------------------------------------------------------------------

    def process_file(
        self,
        nitf_path: Path,
        output_path: Path,
    ) -> bool:
        """Form a quick-look PNG for a single SICD NITF file.

        In frequency-domain mode the GPU cache is flushed before returning
        (Option 2), and a CUDA out-of-memory error triggers an automatic
        CPU retry (Option 1).  In spatial-domain mode (Option 3) no GPU is
        used at all.

        Parameters
        ----------
        nitf_path : Path
            Input SICD NITF file.
        output_path : Path
            Destination PNG file path.

        Returns
        -------
        bool
            ``True`` on success, ``False`` if the file was skipped or
            an error occurred.
        """
        from grdl.IO.sar.sicd import SICDReader

        try:
            with SICDReader(nitf_path) as reader:
                rows = reader.metadata.rows
                cols = reader.metadata.cols
                logger.info(
                    'Reading %s  (%d×%d)  mode=%s  backend=%s',
                    nitf_path.name, rows, cols,
                    'spatial' if self.spatial_multilook else 'freq',
                    'n/a' if self.spatial_multilook else self.backend,
                )
                image_cpu: np.ndarray = reader.read_full()
                sicd_metadata = reader.metadata

            if self.spatial_multilook:
                mean_pwr = _spatial_mean_power(
                    image_cpu, self.looks_rg, self.looks_az
                )
            else:
                mean_pwr = self._freq_domain_mean_power(
                    image_cpu, sicd_metadata, self.backend
                )

            # Apply Mangis density remap (amplitude → [0,1])
            from grdl.contrast import MangisDensity
            amplitude = np.sqrt(mean_pwr)
            density_remap = MangisDensity()
            normalized = density_remap.apply(amplitude)
            image_u8 = (normalized * 255.0).astype(np.uint8)
            _save_png(image_u8, output_path)
            logger.info('Wrote %s', output_path)
            return True

        except Exception as exc:
            logger.warning('Failed to process %s: %s', nitf_path.name, exc)
            return False

    def _freq_domain_mean_power(
        self,
        image_cpu: np.ndarray,
        sicd_metadata: object,
        backend: str,
    ) -> np.ndarray:
        """Run frequency-domain multilook with OOM-safe GPU → CPU fallback.

        Attempts decomposition on *backend*.  If a CUDA out-of-memory error
        is raised the GPU cache is flushed (Option 2) and the operation is
        retried on CPU (Option 1).

        Parameters
        ----------
        image_cpu : np.ndarray
            Complex 2-D image (CPU array, any endianness).
        sicd_metadata : SICDMetadata
            SICD metadata from the reader.
        backend : str
            Requested compute back-end (``'torch'``, ``'cupy'``, or
            ``'cpu'``).

        Returns
        -------
        np.ndarray
            Float32 mean-power image, same spatial shape as *image_cpu*.
        """
        from grdl.image_processing.sar.multilook import MultilookDecomposition

        ml = MultilookDecomposition(
            metadata=sicd_metadata,
            looks_rg=self.looks_rg,
            looks_az=self.looks_az,
            overlap=self.overlap,
            deweight=self.deweight,
        )

        def _run(bk: str) -> np.ndarray:
            img = _to_device(image_cpu, bk)
            grid = ml.decompose(img)
            return _mean_power(_to_numpy(grid))

        try:
            return _run(backend)
        except BaseException as exc:
            if not _is_cuda_oom(exc):
                raise
            # Option 2: flush GPU cache, then Option 1: retry on CPU
            logger.warning(
                'CUDA OOM for %s — flushing GPU cache and retrying on CPU',
                getattr(image_cpu, 'shape', '?'),
            )
            _clear_gpu_cache()
            return _run('cpu')

    # ------------------------------------------------------------------
    # Batch runner
    # ------------------------------------------------------------------

    def run(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
    ) -> Tuple[int, int]:
        """Scan *input_dir* and generate quick-looks for all SICD files.

        Parameters
        ----------
        input_dir : Path
            Directory to scan (recursively) for SICD NITF files.
        output_dir : Path or None
            Directory to write PNGs.  If ``None``, PNGs are written
            alongside the source NITF files.

        Returns
        -------
        tuple[int, int]
            ``(n_ok, n_failed)`` counts.
        """
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f'Input path is not a directory: {input_dir}')

        logger.info(
            'Scanning %s for SICD NITF files  (looks %d×%d, backend=%s)',
            input_dir, self.looks_rg, self.looks_az, self.backend,
        )

        sicd_files = self.find_sicd_files(input_dir)
        if not sicd_files:
            logger.warning('No SICD NITF files found under %s', input_dir)
            return 0, 0

        logger.info('Found %d SICD file(s)', len(sicd_files))

        # Build output paths
        pairs: List[Tuple[Path, Path]] = []
        for src in sicd_files:
            if output_dir is not None:
                # Preserve relative directory structure under output_dir
                rel = src.relative_to(input_dir)
                dst = Path(output_dir) / rel.with_suffix('.png')
            else:
                dst = src.with_suffix('.png')
            pairs.append((src, dst))

        # Serial processing is the default to avoid GPU context forking
        if self.workers == 1:
            ok = failed = 0
            for src, dst in pairs:
                if self.process_file(src, dst):
                    ok += 1
                else:
                    failed += 1
                # Option 2: release PyTorch's cached (unused) VRAM between
                # files so the next file has maximum GPU memory available.
                if not self.spatial_multilook:
                    _clear_gpu_cache()
            return ok, failed

        # Parallel (CPU-only recommended; GPU contexts don't fork safely)
        ok = failed = 0
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self.process_file, src, dst): src
                for src, dst in pairs
            }
            for future in as_completed(futures):
                src_path = futures[future]
                try:
                    if future.result():
                        ok += 1
                    else:
                        failed += 1
                except Exception as exc:
                    logger.warning('Worker exception for %s: %s', src_path.name, exc)
                    failed += 1

        return ok, failed


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='gen_quicklooks',
        description=(
            'Scan a directory for SICD NITF files and write PNG quick-looks.\n\n'
            'Two multilook modes are available:\n'
            '  spatial-domain (default) — block-average intensity,\n'
            '    no FFT, no GPU, lower memory, smaller output image;\n'
            '  frequency-domain (--freq-multilook) — sub-aperture FFT via MultilookDecomposition,\n'
            '    GPU-accelerated when available, with automatic CPU retry on OOM.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Directory to scan recursively for SICD NITF files.',
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        metavar='OUTPUT_DIR',
        help=(
            'Directory to write PNG files. '
            'Subdirectory structure is preserved relative to INPUT_DIR. '
            'Default: write PNGs alongside each source NITF.'
        ),
    )

    parser.add_argument(
        '--looks', '-l',
        type=int,
        default=None,
        metavar='N',
        help='Square multilook factor (sets both --looks-rg and --looks-az). Default: 10.',
    )
    parser.add_argument(
        '--looks-rg',
        type=int,
        default=10,
        metavar='N',
        help='Range multilook factor. Default: 10. Ignored if --looks is specified.',
    )
    parser.add_argument(
        '--looks-az',
        type=int,
        default=10,
        metavar='N',
        help='Azimuth multilook factor. Default: 10. Ignored if --looks is specified.',
    )

    parser.add_argument(
        '--overlap',
        type=float,
        default=0.0,
        metavar='FRAC',
        help='Sub-band overlap fraction [0, 1). Default: 0.0.',
    )
    parser.add_argument(
        '--no-deweight',
        action='store_true',
        help='Disable apodization removal before multilook.',
    )
    parser.add_argument(
        '--freq-multilook',
        action='store_true',
        help=(
            'Use frequency-domain sub-aperture decomposition (FFT-based) instead of '
            'the default spatial-domain block averaging. '
            'Enables GPU acceleration via --backend (or auto-detect). '
            'Output image preserves input dimensions but requires more memory.'
        ),
    )

    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        '--backend',
        choices=['torch', 'cupy', 'cpu'],
        default=None,
        help='Force a compute back-end (frequency-domain mode only). Default: auto-detect.',
    )
    gpu_group.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration (equivalent to --backend cpu).',
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        metavar='N',
        help=(
            'Number of parallel worker processes. '
            'Use 1 (default) when GPU back-end is active.'
        ),
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose (DEBUG) logging.',
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for ``python -m grdl.discovery.gen_quicklooks``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        stream=sys.stderr,
    )

    # Resolve looks
    if args.looks is not None:
        looks_rg = looks_az = args.looks
    else:
        looks_rg = args.looks_rg
        looks_az = args.looks_az

    # Resolve backend
    backend: Optional[str] = None
    if args.no_gpu:
        backend = 'cpu'
    elif args.backend is not None:
        backend = args.backend

    gen = QuicklookGenerator(
        looks_rg=looks_rg,
        looks_az=looks_az,
        overlap=args.overlap,
        deweight=not args.no_deweight,
        spatial_multilook=not args.freq_multilook,
        backend=backend,
        workers=args.workers,
    )

    ok, failed = gen.run(args.input_dir, output_dir=args.output)

    print(f'Done: {ok} succeeded, {failed} failed.', file=sys.stderr)
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
