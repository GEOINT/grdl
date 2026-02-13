# -*- coding: utf-8 -*-
"""
Sublook Decomposition - Sub-aperture spectral splitting of complex SAR imagery.

Splits a complex SAR image into N sub-aperture looks by dividing the
frequency support in azimuth or range. Accounts for oversampling (signal
bandwidth occupying only a fraction of the DFT extent) and supports
configurable overlap between adjacent sub-bands. Optionally removes the
original apodization window before splitting so sub-looks are not shaped
by the collection window.

Accepts both numpy arrays and PyTorch tensors as input. Always returns
numpy arrays. The torch path enables GPU-accelerated FFTs for large images.

Dependencies
------------
torch (optional, for GPU-accelerated FFT path)

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
2026-02-10

Modified
--------
2026-02-10
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
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality
from grdl.IO.models import SICDMetadata

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata


# ===================================================================
# Helpers
# ===================================================================

def _validate_complex_2d(image: np.ndarray) -> None:
    """Validate that *image* is a 2D complex numpy array.

    Parameters
    ----------
    image : np.ndarray
        Array to validate.

    Raises
    ------
    TypeError
        If *image* is not a numpy array or not complex-valued.
    ValueError
        If *image* is not 2D.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"image must be a numpy ndarray, got {type(image).__name__}"
        )
    if not np.iscomplexobj(image):
        raise TypeError(
            f"image must be complex-valued (complex64 or complex128), "
            f"got {image.dtype}. Pass the complex image from the SAR reader."
        )
    if image.ndim != 2:
        raise ValueError(
            f"image must be 2D (rows, cols), got {image.ndim}D "
            f"with shape {image.shape}"
        )


def _validate_complex_2d_torch(tensor: 'torch.Tensor') -> None:
    """Validate that *tensor* is a 2D complex torch tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to validate.

    Raises
    ------
    TypeError
        If *tensor* is not complex-valued.
    ValueError
        If *tensor* is not 2D.
    """
    if not tensor.is_complex():
        raise TypeError(
            f"tensor must be complex-valued (complex64 or complex128), "
            f"got {tensor.dtype}. Pass the complex image from the SAR reader."
        )
    if tensor.ndim != 2:
        raise ValueError(
            f"tensor must be 2D (rows, cols), got {tensor.ndim}D "
            f"with shape {tuple(tensor.shape)}"
        )


# ===================================================================
# SublookDecomposition
# ===================================================================

@processor_version('0.1.0')
@processor_tags(modalities=[ImageModality.SAR])
class SublookDecomposition(ImageProcessor):
    """Split a complex SAR image into sub-aperture looks.

    Divides the frequency support of a complex SAR image into *num_looks*
    sub-bands along the azimuth or range dimension. Each sub-look is
    produced by isolating a sub-band in the spatial frequency domain and
    transforming back to the image domain.

    Oversampling is handled automatically: the signal bandwidth
    (``imp_resp_bw``) typically occupies only a fraction of the full DFT
    extent (``1 / ss``). Sub-band splitting operates only within the
    signal support; frequencies outside the support remain zeroed.

    Adjacent sub-bands may overlap by a configurable fraction. For
    *num_looks=3* and *overlap=0.1*, each pair of neighbouring sub-bands
    shares 10 % of their bandwidth.

    Parameters
    ----------
    metadata : SICDMetadata
        SICD metadata from the reader. Must contain populated ``grid``
        with ``row`` and ``col`` ``SICDDirParam`` fields.
    num_looks : int
        Number of sub-aperture looks to produce. Must be >= 2.
        Default is 2.
    dimension : str
        Axis along which to split the aperture.
        ``'azimuth'`` splits along columns (axis 1, ``grid.col``).
        ``'range'`` splits along rows (axis 0, ``grid.row``).
        Default is ``'azimuth'``.
    overlap : float
        Fractional overlap between adjacent sub-bands, in [0.0, 1.0).
        Default is 0.0 (no overlap).
    deweight : bool
        If True and a weight function is available in the metadata,
        divide the spectrum by the original apodization window before
        splitting. This ensures sub-looks are not shaped by the
        collection window (e.g., Taylor, Hamming). Default is True.

    Raises
    ------
    ValueError
        If *metadata* lacks the required grid fields, *num_looks* < 2,
        *overlap* is out of range, or *dimension* is invalid.

    Examples
    --------
    >>> from grdl.IO.sar import SICDReader
    >>> from grdl.image_processing.sar import SublookDecomposition
    >>>
    >>> with SICDReader('image.nitf') as reader:
    ...     image = reader.read()           # complex64 array
    ...     sublook = SublookDecomposition(reader.metadata, num_looks=3,
    ...                                    overlap=0.1)
    ...     looks = sublook.decompose(image)  # (3, rows, cols) complex
    ...     mag = sublook.to_magnitude(looks) # (3, rows, cols) float
    """

    __gpu_compatible__ = True

    # -- Annotated scalar fields for GUI introspection (__param_specs__) --
    num_looks: Annotated[int, Range(min=2), Desc('Number of sublooks')] = 2
    dimension: Annotated[str, Options('azimuth', 'range'), Desc('Split dimension (azimuth or range)')] = 'azimuth'
    overlap: Annotated[float, Range(min=0.0, max=0.99), Desc('Fractional overlap between sublooks')] = 0.0
    deweight: Annotated[bool, Desc('Apply deweighting before decomposition')] = True

    def __init__(
        self,
        metadata: SICDMetadata,
        num_looks: int = 2,
        dimension: str = 'azimuth',
        overlap: float = 0.0,
        deweight: bool = True,
    ) -> None:
        # ---- validate dimension ----
        if dimension not in ('azimuth', 'range'):
            raise ValueError(
                f"dimension must be 'azimuth' or 'range', got {dimension!r}"
            )

        # ---- validate num_looks ----
        if num_looks < 2:
            raise ValueError(
                f"num_looks must be >= 2, got {num_looks}"
            )

        # ---- validate overlap ----
        if not 0.0 <= overlap < 1.0:
            raise ValueError(
                f"overlap must be in [0.0, 1.0), got {overlap}"
            )

        # ---- extract grid parameters ----
        if metadata.grid is None:
            raise ValueError("metadata.grid is None; SICD grid is required")

        dir_param = (
            metadata.grid.col if dimension == 'azimuth'
            else metadata.grid.row
        )
        if dir_param is None:
            raise ValueError(
                f"metadata.grid.{'col' if dimension == 'azimuth' else 'row'} "
                f"is None; direction parameters are required"
            )

        if dir_param.imp_resp_bw is None:
            raise ValueError(
                "imp_resp_bw is required in the grid direction parameters"
            )
        if dir_param.ss is None:
            raise ValueError(
                "ss (sample spacing) is required in the grid direction "
                "parameters"
            )

        self._num_looks = num_looks
        self._dimension = dimension
        self._overlap = overlap
        self._deweight = deweight
        self._axis = 1 if dimension == 'azimuth' else 0

        # Frequency domain parameters
        self._imp_resp_bw: float = dir_param.imp_resp_bw
        self._ss: float = dir_param.ss
        self._k_ctr: Optional[float] = dir_param.k_ctr
        self._delta_k1: Optional[float] = dir_param.delta_k1
        self._delta_k2: Optional[float] = dir_param.delta_k2
        self._wgt_funct: Optional[np.ndarray] = dir_param.wgt_funct

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_looks(self) -> int:
        """Number of sub-aperture looks."""
        return self._num_looks

    @property
    def dimension(self) -> str:
        """Decomposition dimension (``'azimuth'`` or ``'range'``)."""
        return self._dimension

    @property
    def overlap(self) -> float:
        """Fractional overlap between adjacent sub-bands."""
        return self._overlap

    # ------------------------------------------------------------------
    # Sub-band geometry (pre-computed on first call)
    # ------------------------------------------------------------------

    def _compute_subband_bins(
        self, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute sub-band start/stop bin indices for *n_samples* FFT.

        Parameters
        ----------
        n_samples : int
            Number of samples along the decomposition axis.

        Returns
        -------
        starts : np.ndarray
            Start bin index for each sub-band, shape ``(num_looks,)``.
        stops : np.ndarray
            Stop bin index (exclusive) for each sub-band.
        osr : float
            Oversampling ratio.
        """
        # Full DFT extent in cycles/meter
        k_full = 1.0 / self._ss

        # Oversampling ratio
        osr = k_full / self._imp_resp_bw

        # Signal support in bins (centered in DFT)
        n_support = n_samples / osr

        # Support boundaries in bin indices (centered at DC = bin 0 after
        # fftshift, but we work in fftshift coordinates then convert back)
        if self._delta_k1 is not None and self._delta_k2 is not None:
            # Exact support from metadata (cycles/meter → bins)
            bin_per_k = n_samples * self._ss  # bins per (cycles/meter)
            support_start_bin = self._delta_k1 * bin_per_k
            support_stop_bin = self._delta_k2 * bin_per_k
            n_support = support_stop_bin - support_start_bin
        else:
            # Symmetric support from bandwidth
            support_start_bin = -n_support / 2.0
            support_stop_bin = n_support / 2.0

        # Sub-band width and stride in bins
        sub_bw_bins = n_support / (
            1.0 + (self._num_looks - 1) * (1.0 - self._overlap)
        )
        stride_bins = sub_bw_bins * (1.0 - self._overlap)

        # Sub-band boundaries (in shifted coordinates, relative to DC)
        starts_shifted = np.array([
            support_start_bin + i * stride_bins
            for i in range(self._num_looks)
        ])
        stops_shifted = starts_shifted + sub_bw_bins

        # Convert from DC-centered (shifted) to FFT bin indices
        # In fftshift coordinates: bin 0 is -N/2, bin N/2 is DC
        # fft output: bin 0 is DC, bin N/2 is Nyquist
        # shifted_coord → fft_bin: (shifted + N/2) mod N ... but simpler
        # to work in fftshift space and apply ifftshift at the end.
        # We'll keep these as float bin indices in shifted space.
        # The actual masking will use fftshift'd spectrum.

        # Round to integer bins
        starts_int = np.round(starts_shifted + n_samples / 2.0).astype(int)
        stops_int = np.round(stops_shifted + n_samples / 2.0).astype(int)

        # Clamp to valid range
        np.clip(starts_int, 0, n_samples, out=starts_int)
        np.clip(stops_int, 0, n_samples, out=stops_int)

        return starts_int, stops_int, osr

    def _build_deweight_array(
        self, n_support_bins: int
    ) -> Optional[np.ndarray]:
        """Interpolate the weight function to *n_support_bins* samples.

        Returns None if deweighting is disabled or no weight function
        is available.

        Parameters
        ----------
        n_support_bins : int
            Number of frequency bins in the signal support region.

        Returns
        -------
        np.ndarray or None
            Inverse weight function (1 / w), shape ``(n_support_bins,)``,
            or None if deweighting is not applicable.
        """
        if not self._deweight or self._wgt_funct is None:
            return None

        wgt = self._wgt_funct.ravel().astype(np.float64)
        if wgt.size == n_support_bins:
            interp_wgt = wgt
        else:
            x_orig = np.linspace(0, 1, wgt.size)
            x_new = np.linspace(0, 1, n_support_bins)
            interp_wgt = np.interp(x_new, x_orig, wgt)

        # Avoid division by near-zero; clamp to a small positive floor
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
        """Execute sub-look decomposition via the universal protocol.

        Parameters
        ----------
        metadata : ImageMetadata
            Input image metadata.
        source : np.ndarray
            Complex 2-D SAR image.

        Returns
        -------
        tuple[np.ndarray, ImageMetadata]
            ``(sub_look_stack, updated_metadata)`` where the stack has
            shape ``(num_looks, rows, cols)`` and metadata reflects the
            new band count.
        """
        self._metadata = metadata
        result = self.decompose(source)
        updated = dataclasses.replace(metadata, bands=result.shape[0])
        return result, updated

    # ------------------------------------------------------------------
    # Core decomposition
    # ------------------------------------------------------------------

    def decompose(
        self, image: Union[np.ndarray, 'torch.Tensor']
    ) -> np.ndarray:
        """Decompose a complex SAR image into sub-aperture looks.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            2D complex array, shape ``(rows, cols)``. If a torch tensor
            is provided, the FFT is computed on the tensor's device
            (CPU or CUDA) and the result is returned as a numpy array.

        Returns
        -------
        np.ndarray
            Complex sub-look stack, shape ``(num_looks, rows, cols)``.
            Always a numpy array regardless of input type.

        Raises
        ------
        TypeError
            If *image* is not complex-valued, or is a torch tensor
            when torch is not installed.
        ValueError
            If *image* is not 2D.
        """
        if _HAS_TORCH and isinstance(image, torch.Tensor):
            return self._decompose_torch(image)
        return self._decompose_numpy(image)

    def _decompose_numpy(self, image: np.ndarray) -> np.ndarray:
        """Numpy FFT path.

        Parameters
        ----------
        image : np.ndarray
            2D complex array.

        Returns
        -------
        np.ndarray
            Complex sub-look stack, shape ``(num_looks, rows, cols)``.
        """
        _validate_complex_2d(image)
        axis = self._axis
        n_samples = image.shape[axis]

        starts, stops, osr = self._compute_subband_bins(n_samples)

        # FFT along decomposition axis, then shift DC to center
        spectrum = np.fft.fftshift(np.fft.fft(image, axis=axis), axes=axis)

        # Deweight within signal support
        support_start = starts[0]
        support_stop = stops[-1]
        n_support_bins = support_stop - support_start
        deweight_arr = self._build_deweight_array(n_support_bins)
        if deweight_arr is not None:
            # Build broadcastable shape
            shape = [1, 1]
            shape[axis] = n_support_bins
            dw = deweight_arr.reshape(shape)
            slices = [slice(None), slice(None)]
            slices[axis] = slice(support_start, support_stop)
            spectrum[tuple(slices)] = spectrum[tuple(slices)] * dw

        # Extract sub-looks
        rows, cols = image.shape
        result = np.zeros(
            (self._num_looks, rows, cols), dtype=image.dtype
        )

        for i in range(self._num_looks):
            sub_spectrum = np.zeros_like(spectrum)
            slices = [slice(None), slice(None)]
            slices[axis] = slice(starts[i], stops[i])
            sub_spectrum[tuple(slices)] = spectrum[tuple(slices)]

            # Shift back and inverse FFT
            result[i] = np.fft.ifft(
                np.fft.ifftshift(sub_spectrum, axes=axis), axis=axis
            )

        return result

    def _decompose_torch(self, tensor: 'torch.Tensor') -> np.ndarray:
        """PyTorch FFT path (CPU or CUDA).

        Parameters
        ----------
        tensor : torch.Tensor
            2D complex tensor.

        Returns
        -------
        np.ndarray
            Complex sub-look stack, shape ``(num_looks, rows, cols)``.
        """
        if not _HAS_TORCH:
            raise TypeError(
                "torch is not installed. Install PyTorch to use tensor inputs."
            )

        _validate_complex_2d_torch(tensor)
        dim = self._axis
        n_samples = tensor.shape[dim]

        starts, stops, osr = self._compute_subband_bins(n_samples)

        # FFT along decomposition axis, then shift DC to center
        spectrum = torch.fft.fftshift(torch.fft.fft(tensor, dim=dim), dim=dim)

        # Deweight within signal support
        support_start = int(starts[0])
        support_stop = int(stops[-1])
        n_support_bins = support_stop - support_start
        deweight_arr = self._build_deweight_array(n_support_bins)
        if deweight_arr is not None:
            dw_tensor = torch.from_numpy(deweight_arr).to(
                device=tensor.device, dtype=tensor.real.dtype
            )
            shape = [1, 1]
            shape[dim] = n_support_bins
            dw_tensor = dw_tensor.reshape(shape)
            slices = [slice(None), slice(None)]
            slices[dim] = slice(support_start, support_stop)
            spectrum[tuple(slices)] = spectrum[tuple(slices)] * dw_tensor

        # Extract sub-looks
        rows, cols = tensor.shape
        results = []

        for i in range(self._num_looks):
            sub_spectrum = torch.zeros_like(spectrum)
            slices = [slice(None), slice(None)]
            slices[dim] = slice(int(starts[i]), int(stops[i]))
            sub_spectrum[tuple(slices)] = spectrum[tuple(slices)]

            # Shift back and inverse FFT
            look = torch.fft.ifft(
                torch.fft.ifftshift(sub_spectrum, dim=dim), dim=dim
            )
            results.append(look)

        stack = torch.stack(results, dim=0)
        return stack.cpu().numpy()

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_magnitude(self, stack: np.ndarray) -> np.ndarray:
        """Convert complex sub-look stack to magnitude (|z|).

        Parameters
        ----------
        stack : np.ndarray
            Complex sub-look stack from ``decompose()``.
            Shape ``(num_looks, rows, cols)``.

        Returns
        -------
        np.ndarray
            Real-valued magnitude array, same shape.
        """
        return np.abs(stack)

    def to_power(self, stack: np.ndarray) -> np.ndarray:
        """Convert complex sub-look stack to power (|z|^2).

        Parameters
        ----------
        stack : np.ndarray
            Complex sub-look stack from ``decompose()``.
            Shape ``(num_looks, rows, cols)``.

        Returns
        -------
        np.ndarray
            Real-valued power array, same shape.
        """
        return np.abs(stack) ** 2

    def to_db(
        self, stack: np.ndarray, floor: float = -50.0
    ) -> np.ndarray:
        """Convert complex sub-look stack to magnitude in dB.

        Computes ``20 * log10(|z|)``, clamped to a floor value.

        Parameters
        ----------
        stack : np.ndarray
            Complex sub-look stack from ``decompose()``.
            Shape ``(num_looks, rows, cols)``.
        floor : float
            Minimum dB value. Pixels below this are clamped.
            Default is -50.0.

        Returns
        -------
        np.ndarray
            Real-valued dB array, same shape.
        """
        mag = np.abs(stack)
        db = 20.0 * np.log10(mag + np.finfo(mag.dtype).tiny)
        np.maximum(db, floor, out=db)
        return db
