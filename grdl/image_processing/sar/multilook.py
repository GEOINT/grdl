# -*- coding: utf-8 -*-
"""
2D Multilook Decomposition - Multi-dimensional sub-aperture spectral
splitting of complex SAR imagery.

Splits a complex SAR image into an M x N grid of sub-aperture looks by
dividing the 2D frequency support in both range and azimuth simultaneously.
Uses ``Tiler`` from ``grdl.data_prep`` to partition the signal support
region into rectangular frequency bins. Each sub-look is produced by
isolating a 2D sub-band in the spatial frequency domain and transforming
back to the image domain via 2D IFFT.

Accounts for oversampling in both dimensions (signal bandwidth occupying
only a fraction of the full DFT extent) and supports configurable overlap
between adjacent sub-bands. Optionally removes the original apodization
window before splitting using separable 2D deweighting.

Dependencies
------------
torch (optional, for GPU-accelerated FFT path)

Author
------
Duane Smalley, PhD
duane.d.smalley@gmail.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-17

Modified
--------
2026-02-17
"""

# Standard library
import dataclasses
from typing import Annotated, Any, Optional, Tuple, TYPE_CHECKING, Union

# Third-party
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# GRDL internal
from grdl.image_processing.base import ImageProcessor
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality, ProcessorCategory
from grdl.IO.models import SICDMetadata
from grdl.data_prep import Tiler

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata


# ===================================================================
# Helpers
# ===================================================================

def _validate_complex_2d(image: np.ndarray) -> None:
    """Validate that *image* is a 2D complex numpy array."""
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"image must be a numpy ndarray, got {type(image).__name__}"
        )
    if not np.iscomplexobj(image):
        raise TypeError(
            f"image must be complex-valued (complex64 or complex128), "
            f"got {image.dtype}"
        )
    if image.ndim != 2:
        raise ValueError(
            f"image must be 2D (rows, cols), got {image.ndim}D "
            f"with shape {image.shape}"
        )


def _validate_complex_2d_torch(tensor: 'torch.Tensor') -> None:
    """Validate that *tensor* is a 2D complex torch tensor."""
    if not tensor.is_complex():
        raise TypeError(
            f"tensor must be complex-valued, got {tensor.dtype}"
        )
    if tensor.ndim != 2:
        raise ValueError(
            f"tensor must be 2D (rows, cols), got {tensor.ndim}D "
            f"with shape {tuple(tensor.shape)}"
        )


# ===================================================================
# MultilookDecomposition
# ===================================================================

@processor_version('0.1.0')
@processor_tags(
    modalities=[ImageModality.SAR],
    category=ProcessorCategory.STACKS,
    description='2D sub-aperture spectral decomposition into M x N grid',
)
class MultilookDecomposition(ImageProcessor):
    """Split a complex SAR image into a 2D grid of sub-aperture looks.

    Divides the 2D frequency support of a complex SAR image into an
    ``looks_rg x looks_az`` grid of sub-bands. Each sub-look is produced
    by isolating a rectangular region in the 2D spatial frequency domain
    and transforming back to the image domain via 2D IFFT.

    Uses ``Tiler`` from ``grdl.data_prep`` to partition the frequency
    support into rectangular bin regions, ensuring consistent edge
    handling with the rest of the library.

    Oversampling is handled automatically in both dimensions: the signal
    bandwidth (``imp_resp_bw``) typically occupies only a fraction of the
    full DFT extent (``1 / ss``). Sub-band splitting operates only within
    the signal support; frequencies outside the support remain zeroed.

    Parameters
    ----------
    metadata : SICDMetadata
        SICD metadata from the reader.  Must contain populated ``grid``
        with both ``row`` (range) and ``col`` (azimuth) ``SICDDirParam``.
    looks_rg : int
        Number of range (row) sub-aperture looks.  Default 3.
    looks_az : int
        Number of azimuth (column) sub-aperture looks.  Default 3.
    overlap : float
        Fractional overlap between adjacent sub-bands in both dimensions,
        in [0.0, 1.0).  Default 0.0 (no overlap).
    deweight : bool
        If True and weight functions are available in the metadata,
        remove the original apodization before splitting.  Default True.

    Raises
    ------
    ValueError
        If *metadata* lacks the required grid fields,
        ``looks_rg * looks_az < 2``, or *overlap* is out of range.

    Examples
    --------
    >>> from grdl.IO.sar import SICDReader
    >>> from grdl.image_processing.sar import MultilookDecomposition
    >>>
    >>> with SICDReader('image.nitf') as reader:
    ...     image = reader.read_full()
    ...     ml = MultilookDecomposition(reader.metadata,
    ...                                looks_rg=3, looks_az=3)
    ...     grid = ml.decompose(image)      # (3, 3, rows, cols) complex
    ...     db = ml.to_db(grid)             # (3, 3, rows, cols) float
    ...     flat = ml.to_flat_stack(grid)    # (9, rows, cols) complex
    """

    __gpu_compatible__ = True

    # -- Annotated tunable params --
    looks_rg: Annotated[int, Range(min=1, max=16),
                        Desc('Range sub-aperture looks')] = 3
    looks_az: Annotated[int, Range(min=1, max=16),
                        Desc('Azimuth sub-aperture looks')] = 3
    overlap: Annotated[float, Range(min=0.0, max=0.99),
                       Desc('Fractional overlap between sub-bands')] = 0.0
    deweight: Annotated[bool,
                        Desc('Remove apodization before splitting')] = True

    def __init__(
        self,
        metadata: SICDMetadata,
        looks_rg: int = 3,
        looks_az: int = 3,
        overlap: float = 0.0,
        deweight: bool = True,
    ) -> None:
        # ---- validate total looks ----
        if looks_rg * looks_az < 2:
            raise ValueError(
                f"Total looks (looks_rg * looks_az = {looks_rg * looks_az}) "
                f"must be >= 2"
            )

        # ---- validate overlap ----
        if not 0.0 <= overlap < 1.0:
            raise ValueError(
                f"overlap must be in [0.0, 1.0), got {overlap}"
            )

        # ---- validate grid ----
        if metadata.grid is None:
            raise ValueError("metadata.grid is None; SICD grid is required")

        # -- Range (row) params --
        row_param = metadata.grid.row
        if row_param is None:
            raise ValueError(
                "metadata.grid.row is None; range direction required"
            )
        if row_param.imp_resp_bw is None:
            raise ValueError(
                "row imp_resp_bw is required for range direction"
            )
        if row_param.ss is None:
            raise ValueError(
                "row ss (sample spacing) is required for range direction"
            )

        # -- Azimuth (col) params --
        col_param = metadata.grid.col
        if col_param is None:
            raise ValueError(
                "metadata.grid.col is None; azimuth direction required"
            )
        if col_param.imp_resp_bw is None:
            raise ValueError(
                "col imp_resp_bw is required for azimuth direction"
            )
        if col_param.ss is None:
            raise ValueError(
                "col ss (sample spacing) is required for azimuth direction"
            )

        # Store parameters
        self._looks_rg = looks_rg
        self._looks_az = looks_az
        self._overlap = overlap
        self._deweight = deweight

        # Range (row) frequency domain parameters
        self._rg_imp_resp_bw: float = row_param.imp_resp_bw
        self._rg_ss: float = row_param.ss
        self._rg_delta_k1: Optional[float] = row_param.delta_k1
        self._rg_delta_k2: Optional[float] = row_param.delta_k2
        self._rg_wgt_funct: Optional[np.ndarray] = row_param.wgt_funct

        # Azimuth (col) frequency domain parameters
        self._az_imp_resp_bw: float = col_param.imp_resp_bw
        self._az_ss: float = col_param.ss
        self._az_delta_k1: Optional[float] = col_param.delta_k1
        self._az_delta_k2: Optional[float] = col_param.delta_k2
        self._az_wgt_funct: Optional[np.ndarray] = col_param.wgt_funct

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def looks_rg(self) -> int:
        """Number of range sub-aperture looks."""
        return self._looks_rg

    @property
    def looks_az(self) -> int:
        """Number of azimuth sub-aperture looks."""
        return self._looks_az

    @property
    def overlap(self) -> float:
        """Fractional overlap between adjacent sub-bands."""
        return self._overlap

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """Sub-look grid dimensions as ``(looks_rg, looks_az)``."""
        return (self._looks_rg, self._looks_az)

    # ------------------------------------------------------------------
    # Support geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_support_geometry(
        n_samples: int,
        imp_resp_bw: float,
        ss: float,
        delta_k1: Optional[float],
        delta_k2: Optional[float],
    ) -> Tuple[int, int, float]:
        """Compute support bin boundaries in fftshift coordinates.

        Parameters
        ----------
        n_samples : int
            Number of samples along this dimension.
        imp_resp_bw : float
            Impulse response bandwidth (cycles/meter).
        ss : float
            Sample spacing (meters).
        delta_k1 : float or None
            Minimum frequency offset (cycles/meter).
        delta_k2 : float or None
            Maximum frequency offset (cycles/meter).

        Returns
        -------
        support_start : int
            Start bin of signal support (fftshift coordinates).
        support_stop : int
            Stop bin (exclusive) of signal support.
        osr : float
            Oversampling ratio.
        """
        k_full = 1.0 / ss
        osr = k_full / imp_resp_bw
        n_support = n_samples / osr

        if delta_k1 is not None and delta_k2 is not None:
            bin_per_k = n_samples * ss
            support_start_f = delta_k1 * bin_per_k
            support_stop_f = delta_k2 * bin_per_k
        else:
            support_start_f = -n_support / 2.0
            support_stop_f = n_support / 2.0

        support_start = int(round(support_start_f + n_samples / 2.0))
        support_stop = int(round(support_stop_f + n_samples / 2.0))

        support_start = max(0, min(support_start, n_samples))
        support_stop = max(0, min(support_stop, n_samples))

        return support_start, support_stop, osr

    # ------------------------------------------------------------------
    # Tiler construction
    # ------------------------------------------------------------------

    def _build_tiler(
        self, n_support_rg: int, n_support_az: int
    ) -> Tiler:
        """Build a Tiler that partitions the 2D frequency support.

        Uses the same sub-band width formula as ``SublookDecomposition``:

            bw = n_support / (1 + (N - 1) * (1 - overlap))
            stride = bw * (1 - overlap)

        Parameters
        ----------
        n_support_rg : int
            Number of bins in range support.
        n_support_az : int
            Number of bins in azimuth support.

        Returns
        -------
        Tiler
            Configured for the frequency bin grid.
        """
        tile_rg, stride_rg = self._tile_stride(
            n_support_rg, self._looks_rg
        )
        tile_az, stride_az = self._tile_stride(
            n_support_az, self._looks_az
        )
        return Tiler(
            nrows=n_support_rg,
            ncols=n_support_az,
            tile_size=(tile_rg, tile_az),
            stride=(stride_rg, stride_az),
        )

    def _tile_stride(
        self, n_support: int, n_looks: int
    ) -> Tuple[int, int]:
        """Compute integer tile size and stride for one dimension.

        Computes tile width from the sub-band bandwidth formula, then
        derives the stride so that the ``Tiler`` produces exactly
        *n_looks* tiles (accounting for integer rounding and the Tiler's
        edge-snapping coverage algorithm).

        Parameters
        ----------
        n_support : int
            Signal support bins in this dimension.
        n_looks : int
            Number of looks along this dimension.

        Returns
        -------
        tile : int
            Tile width in bins (>= 1).
        stride : int
            Stride in bins (>= 1, <= tile).
        """
        if n_looks == 1:
            return n_support, n_support

        # Sub-band bandwidth from the standard formula
        bw = n_support / (1.0 + (n_looks - 1) * (1.0 - self._overlap))
        tile = max(1, int(round(bw)))

        # Derive stride so Tiler yields exactly n_looks tiles:
        #   n_looks = 1 + ceil((n_support - tile) / stride)
        #   => stride = ceil((n_support - tile) / (n_looks - 1))
        remaining = n_support - tile
        if remaining <= 0:
            return min(tile, n_support), min(tile, n_support)

        stride = max(1, int(np.ceil(remaining / (n_looks - 1))))

        # Tiler requires stride <= tile_size
        tile = max(tile, stride)

        return tile, stride

    # ------------------------------------------------------------------
    # Deweighting
    # ------------------------------------------------------------------

    def _build_deweight_2d(
        self, n_support_rg: int, n_support_az: int
    ) -> Optional[np.ndarray]:
        """Build 2D inverse-weight as outer product of 1D deweights.

        Returns None if deweighting is disabled or no weight functions
        are available in either dimension.

        Parameters
        ----------
        n_support_rg : int
            Range support bins.
        n_support_az : int
            Azimuth support bins.

        Returns
        -------
        np.ndarray or None
            Shape ``(n_support_rg, n_support_az)``, or None.
        """
        if not self._deweight:
            return None

        dw_rg = self._build_deweight_1d(self._rg_wgt_funct, n_support_rg)
        dw_az = self._build_deweight_1d(self._az_wgt_funct, n_support_az)

        if dw_rg is None and dw_az is None:
            return None

        if dw_rg is None:
            dw_rg = np.ones(n_support_rg)
        if dw_az is None:
            dw_az = np.ones(n_support_az)

        return dw_rg[:, np.newaxis] * dw_az[np.newaxis, :]

    @staticmethod
    def _build_deweight_1d(
        wgt_funct: Optional[np.ndarray], n_bins: int
    ) -> Optional[np.ndarray]:
        """Interpolate a 1D weight function to *n_bins* and return inverse.

        Parameters
        ----------
        wgt_funct : np.ndarray or None
            Original weight function samples.
        n_bins : int
            Number of frequency bins in the support.

        Returns
        -------
        np.ndarray or None
            Inverse weight ``1/w``, shape ``(n_bins,)``, or None.
        """
        if wgt_funct is None:
            return None

        wgt = wgt_funct.ravel().astype(np.float64)
        if wgt.size == n_bins:
            interp_wgt = wgt.copy()
        else:
            x_orig = np.linspace(0, 1, wgt.size)
            x_new = np.linspace(0, 1, n_bins)
            interp_wgt = np.interp(x_new, x_orig, wgt)

        floor = np.max(interp_wgt) * 1e-6
        np.maximum(interp_wgt, floor, out=interp_wgt)
        return 1.0 / interp_wgt

    # ------------------------------------------------------------------
    # execute() protocol
    # ------------------------------------------------------------------

    def execute(
        self,
        metadata: 'ImageMetadata',
        source: np.ndarray,
        **kwargs: Any,
    ) -> tuple:
        """Execute multi-look decomposition via the universal protocol.

        Parameters
        ----------
        metadata : ImageMetadata
            Input image metadata.
        source : np.ndarray
            Complex 2-D SAR image.

        Returns
        -------
        tuple[np.ndarray, ImageMetadata]
            ``(multilook_grid, updated_metadata)`` where the grid has
            shape ``(looks_rg, looks_az, rows, cols)`` and metadata
            reflects the new band count.
        """
        self._metadata = metadata
        result = self.decompose(source)
        updated = dataclasses.replace(
            metadata, bands=result.shape[0] * result.shape[1]
        )
        return result, updated

    # ------------------------------------------------------------------
    # Core decomposition
    # ------------------------------------------------------------------

    def decompose(
        self, image: Union[np.ndarray, 'torch.Tensor']
    ) -> np.ndarray:
        """Decompose a complex SAR image into a 2D grid of sub-looks.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            2D complex array, shape ``(rows, cols)``.  If a torch tensor
            is provided, the FFT is computed on the tensor's device
            (CPU or CUDA) and the result is returned as a numpy array.

        Returns
        -------
        np.ndarray
            Complex sub-look grid, shape
            ``(looks_rg, looks_az, rows, cols)``.
            Always a numpy array regardless of input type.

        Raises
        ------
        TypeError
            If *image* is not complex-valued.
        ValueError
            If *image* is not 2D.
        """
        if _HAS_TORCH and isinstance(image, torch.Tensor):
            return self._decompose_torch(image)
        return self._decompose_numpy(image)

    def _decompose_numpy(self, image: np.ndarray) -> np.ndarray:
        """Numpy 2D FFT path.

        Parameters
        ----------
        image : np.ndarray
            2D complex array.

        Returns
        -------
        np.ndarray
            Shape ``(looks_rg, looks_az, rows, cols)``.
        """
        _validate_complex_2d(image)
        rows, cols = image.shape

        # 1. Compute support geometry for both dimensions
        #    Rows (axis 0) = range, Cols (axis 1) = azimuth
        rg_start, rg_stop, _ = self._compute_support_geometry(
            rows, self._rg_imp_resp_bw, self._rg_ss,
            self._rg_delta_k1, self._rg_delta_k2,
        )
        az_start, az_stop, _ = self._compute_support_geometry(
            cols, self._az_imp_resp_bw, self._az_ss,
            self._az_delta_k1, self._az_delta_k2,
        )
        n_support_rg = rg_stop - rg_start
        n_support_az = az_stop - az_start

        # 2. 2D FFT + fftshift
        spectrum = np.fft.fftshift(np.fft.fft2(image))

        # 3. Deweight the 2D support region
        dw_2d = self._build_deweight_2d(n_support_rg, n_support_az)
        if dw_2d is not None:
            spectrum[rg_start:rg_stop, az_start:az_stop] *= dw_2d

        # 4. Build Tiler over the support region
        #    Tiler rows = range bins, Tiler cols = azimuth bins
        tiler = self._build_tiler(n_support_rg, n_support_az)
        regions = tiler.tile_positions()

        # 5. Extract sub-looks
        result = np.zeros(
            (self._looks_rg, self._looks_az, rows, cols),
            dtype=image.dtype,
        )

        for idx, region in enumerate(regions):
            i_rg = idx // self._looks_az
            i_az = idx % self._looks_az

            # Offset ChipRegion to absolute spectrum coordinates
            r0 = region.row_start + rg_start
            r1 = region.row_end + rg_start
            c0 = region.col_start + az_start
            c1 = region.col_end + az_start

            sub_spectrum = np.zeros_like(spectrum)
            sub_spectrum[r0:r1, c0:c1] = spectrum[r0:r1, c0:c1]

            result[i_rg, i_az] = np.fft.ifft2(
                np.fft.ifftshift(sub_spectrum)
            )

        return result

    def _decompose_torch(self, tensor: 'torch.Tensor') -> np.ndarray:
        """PyTorch 2D FFT path (CPU or CUDA).

        Parameters
        ----------
        tensor : torch.Tensor
            2D complex tensor.

        Returns
        -------
        np.ndarray
            Shape ``(looks_rg, looks_az, rows, cols)``.
        """
        if not _HAS_TORCH:
            raise TypeError(
                "torch is not installed. Install PyTorch to use "
                "tensor inputs."
            )

        _validate_complex_2d_torch(tensor)
        rows, cols = tensor.shape

        rg_start, rg_stop, _ = self._compute_support_geometry(
            rows, self._rg_imp_resp_bw, self._rg_ss,
            self._rg_delta_k1, self._rg_delta_k2,
        )
        az_start, az_stop, _ = self._compute_support_geometry(
            cols, self._az_imp_resp_bw, self._az_ss,
            self._az_delta_k1, self._az_delta_k2,
        )
        n_support_rg = rg_stop - rg_start
        n_support_az = az_stop - az_start

        spectrum = torch.fft.fftshift(torch.fft.fft2(tensor))

        # Deweight
        dw_2d = self._build_deweight_2d(n_support_rg, n_support_az)
        if dw_2d is not None:
            dw_tensor = torch.from_numpy(dw_2d).to(
                device=tensor.device, dtype=tensor.real.dtype
            )
            spectrum[rg_start:rg_stop, az_start:az_stop] *= dw_tensor

        tiler = self._build_tiler(n_support_rg, n_support_az)
        regions = tiler.tile_positions()

        results = []
        for region in regions:
            r0 = region.row_start + rg_start
            r1 = region.row_end + rg_start
            c0 = region.col_start + az_start
            c1 = region.col_end + az_start

            sub_spectrum = torch.zeros_like(spectrum)
            sub_spectrum[r0:r1, c0:c1] = spectrum[r0:r1, c0:c1]

            look = torch.fft.ifft2(torch.fft.ifftshift(sub_spectrum))
            results.append(look)

        stack = torch.stack(results, dim=0)
        grid = stack.reshape(
            self._looks_rg, self._looks_az, rows, cols
        )
        return grid.cpu().numpy()

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_magnitude(self, grid: np.ndarray) -> np.ndarray:
        """Convert complex grid to magnitude (|z|).

        Parameters
        ----------
        grid : np.ndarray
            Complex grid from ``decompose()``.

        Returns
        -------
        np.ndarray
            Real-valued magnitude array, same shape.
        """
        return np.abs(grid)

    def to_power(self, grid: np.ndarray) -> np.ndarray:
        """Convert complex grid to power (|z|^2).

        Parameters
        ----------
        grid : np.ndarray
            Complex grid from ``decompose()``.

        Returns
        -------
        np.ndarray
            Real-valued power array, same shape.
        """
        return np.abs(grid) ** 2

    def to_db(
        self, grid: np.ndarray, floor: float = -50.0
    ) -> np.ndarray:
        """Convert complex grid to magnitude in dB.

        Computes ``20 * log10(|z|)``, clamped to a floor value.

        Parameters
        ----------
        grid : np.ndarray
            Complex grid from ``decompose()``.
        floor : float
            Minimum dB value.  Default -50.0.

        Returns
        -------
        np.ndarray
            Real-valued dB array, same shape.
        """
        mag = np.abs(grid)
        db = 20.0 * np.log10(mag + np.finfo(mag.dtype).tiny)
        np.maximum(db, floor, out=db)
        return db

    def to_flat_stack(self, grid: np.ndarray) -> np.ndarray:
        """Reshape ``(M, N, rows, cols)`` grid to ``(M*N, rows, cols)``.

        Parameters
        ----------
        grid : np.ndarray
            Multi-look grid from ``decompose()``, shape
            ``(looks_rg, looks_az, rows, cols)``.

        Returns
        -------
        np.ndarray
            Flat stack, shape ``(looks_rg * looks_az, rows, cols)``.
        """
        m, n, rows, cols = grid.shape
        return grid.reshape(m * n, rows, cols)
