# -*- coding: utf-8 -*-
"""
Sublook Decomposition - Sub-aperture spectral splitting of complex SAR imagery.

Splits a complex SAR image into N sub-aperture looks by dividing the
frequency support in azimuth or range. Accounts for oversampling (signal
bandwidth occupying only a fraction of the DFT extent) and supports
configurable overlap between adjacent sub-bands. Optionally removes the
original apodization window before splitting so sub-looks are not shaped
by the collection window.

For stripmap and ScanSAR/TOPS collections the center-of-aperture (COA)
spatial frequency varies across the image, so a global spectral cut would
slice a different sub-aperture for every pixel. The decomposition therefore
deskews (centers) the phase history using the SICD ``DeltaKCOAPoly`` before
cutting the sub-bands and re-skews each look afterward. This is automatically
mode-adaptive: ``DeltaKCOAPoly`` is ~0 for spotlight (deskew is a no-op) and
non-trivial for stripmap/ScanSAR, so a single code path covers all modes.

Accepts numpy arrays, CuPy arrays, and PyTorch tensors as input. Returns
the same array type as the input for numpy and CuPy; the PyTorch path
always returns a numpy array. Both the CuPy and torch paths support
GPU-accelerated FFTs for large images.

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
2026-06-02
"""

# Standard library
import dataclasses
import logging
from typing import Annotated, Any, Optional, Tuple, TYPE_CHECKING, Union

# Third-party
import numpy as np

from grdl._torch_optional import torch, HAS_TORCH as _HAS_TORCH

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

# GRDL internal
from grdl.image_processing.base import ImageProcessor, ImageTransform
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality
from grdl.IO.models import SICDMetadata
from grdl.IO.models.common import Poly2D

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata

logger = logging.getLogger(__name__)


# ===================================================================
# Helpers
# ===================================================================

def _validate_complex_2d(image: np.ndarray) -> None:
    """Validate that *image* is a 2D complex numpy or cupy array.

    Parameters
    ----------
    image : np.ndarray
        Array to validate.  Also accepts ``cupy.ndarray`` when CuPy is
        installed.

    Raises
    ------
    TypeError
        If *image* is not a numpy/cupy array or not complex-valued.
    ValueError
        If *image* is not 2D.
    """
    is_cupy = _HAS_CUPY and isinstance(image, cp.ndarray)
    if not isinstance(image, np.ndarray) and not is_cupy:
        raise TypeError(
            f"image must be a numpy or cupy ndarray, got {type(image).__name__}"
        )
    xp = cp if is_cupy else np
    if not xp.iscomplexobj(image):
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
    deskew : bool
        If True and ``DeltaKCOAPoly`` is available in the metadata,
        center (deskew) the phase history before cutting the sub-bands
        and re-skew each look afterward. Required for correct sub-aperture
        cuts in stripmap and ScanSAR/TOPS where the center-of-aperture
        frequency varies across the image. A near no-op for spotlight
        (``DeltaKCOAPoly`` ~ 0). Default is True.

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
    deskew: Annotated[bool, Desc('Center the phase history (DeltaKCOA) before cutting')] = True

    def __init__(
        self,
        metadata: SICDMetadata,
        num_looks: int = 2,
        dimension: str = 'azimuth',
        overlap: float = 0.0,
        deweight: bool = True,
        deskew: bool = True,
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
        self._deskew = deskew
        self._axis = 1 if dimension == 'azimuth' else 0

        # Frequency domain parameters
        self._imp_resp_bw: float = dir_param.imp_resp_bw
        self._ss: float = dir_param.ss
        self._k_ctr: Optional[float] = dir_param.k_ctr
        self._delta_k1: Optional[float] = dir_param.delta_k1
        self._delta_k2: Optional[float] = dir_param.delta_k2
        self._wgt_funct: Optional[np.ndarray] = dir_param.wgt_funct

        # ---- deskew (phase-history centering) parameters ----
        # Collection mode is informational only; the DeltaKCOAPoly drives
        # the math (~0 for spotlight, non-trivial for stripmap / ScanSAR).
        self._mode_type: Optional[str] = None
        if (metadata.collection_info is not None
                and metadata.collection_info.radar_mode is not None):
            self._mode_type = metadata.collection_info.radar_mode.mode_type

        self._sgn: int = dir_param.sgn if dir_param.sgn is not None else -1
        self._delta_k_coa_poly = dir_param.delta_k_coa_poly

        # Pixel-to-meters mapping relative to the SCP. Needed to evaluate
        # the DeltaKCOAPoly (which is a polynomial in row/col meters).
        self._row_ss: Optional[float] = (
            metadata.grid.row.ss if metadata.grid.row is not None else None
        )
        self._col_ss: Optional[float] = (
            metadata.grid.col.ss if metadata.grid.col is not None else None
        )
        self._scp_row: Optional[float] = None
        self._scp_col: Optional[float] = None
        self._first_row: int = 0
        self._first_col: int = 0
        image_data = metadata.image_data
        if image_data is not None:
            self._first_row = int(image_data.first_row)
            self._first_col = int(image_data.first_col)
            if image_data.scp_pixel is not None:
                self._scp_row = float(image_data.scp_pixel.row)
                self._scp_col = float(image_data.scp_pixel.col)

        # Whether a usable deskew is actually possible with this metadata.
        self._can_deskew: bool = (
            self._deskew
            and self._delta_k_coa_poly is not None
            and self._delta_k_coa_poly.coefs is not None
            and self._scp_row is not None
            and self._scp_col is not None
            and self._row_ss is not None
            and self._col_ss is not None
        )
        if self._deskew and not self._can_deskew:
            logger.info(
                "Sublook deskew requested but DeltaKCOAPoly / SCP geometry "
                "is unavailable (mode=%s); falling back to global spectral "
                "cut. This is correct for spotlight but may be inaccurate "
                "for stripmap / ScanSAR.",
                self._mode_type,
            )

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

    @property
    def deskew(self) -> bool:
        """Whether phase-history centering (DeltaKCOA deskew) is enabled."""
        return self._deskew

    # ------------------------------------------------------------------
    # Sub-band geometry (pre-computed on first call)
    # ------------------------------------------------------------------

    def _compute_subband_bins(
        self, n_samples: int, centered: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute sub-band start/stop bin indices for *n_samples* FFT.

        Parameters
        ----------
        n_samples : int
            Number of samples along the decomposition axis.
        centered : bool
            If True, the phase history has been deskewed so the signal
            support is symmetric about DC; ignore the metadata
            ``delta_k1``/``delta_k2`` (which describe the *skewed* support)
            and place the support symmetrically. Default is False.

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
        if (not centered
                and self._delta_k1 is not None
                and self._delta_k2 is not None):
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

    @staticmethod
    def _polyint_coefs(coefs: np.ndarray, axis: int) -> np.ndarray:
        """Integrate a 2-D polynomial's coefficients along *axis*.

        For ``coefs[i, j]`` (coefficient of ``x**i * y**j``), integration
        with respect to ``x`` (axis 0) maps ``coefs[i, j]`` to
        ``coefs[i, j] / (i + 1)`` at power ``i + 1``; integration with
        respect to ``y`` (axis 1) maps it to ``coefs[i, j] / (j + 1)`` at
        power ``j + 1``. The integration constant is zero.

        Parameters
        ----------
        coefs : np.ndarray
            2-D coefficient array, shape ``(order_x + 1, order_y + 1)``.
        axis : int
            ``0`` to integrate in the row (x) coordinate, ``1`` for col (y).

        Returns
        -------
        np.ndarray
            Integrated coefficient array with one extra order on *axis*.
        """
        c = np.asarray(coefs, dtype=np.float64)
        if axis == 0:
            out = np.zeros((c.shape[0] + 1, c.shape[1]), dtype=np.float64)
            divisors = np.arange(1, c.shape[0] + 1).reshape(-1, 1)
            out[1:, :] = c / divisors
        else:
            out = np.zeros((c.shape[0], c.shape[1] + 1), dtype=np.float64)
            divisors = np.arange(1, c.shape[1] + 1).reshape(1, -1)
            out[:, 1:] = c / divisors
        return out

    def _compute_deskew_phase(
        self, rows: int, cols: int
    ) -> Optional[np.ndarray]:
        """Compute the deskew phase ramp that centers the phase history.

        Integrates ``DeltaKCOAPoly`` along the decomposition axis and
        evaluates it over the image grid (coordinates in meters relative to
        the SCP). The returned array ``phase`` is applied as
        ``image * exp(1j * sgn * 2*pi * phase)`` to drag every pixel's
        center-of-aperture frequency to DC, and its conjugate re-skews the
        looks afterward.

        Returns ``None`` when deskewing is disabled or the required
        metadata (``DeltaKCOAPoly``, SCP pixel, sample spacings) is missing,
        in which case the global spectral cut is used unchanged.

        Notes
        -----
        The integral deramp handles the smooth (linear / low-order)
        center-of-aperture variation that covers spotlight, stripmap, and
        normal ScanSAR. For extreme TOPS azimuth steering the per-pixel COA
        frequency can exceed the sampling rate and alias; the SICD standard
        (NGA.STND.0024-1, Sec. 4.4) notes that ``KCOA`` computed from the
        polynomial "may need to be adjusted by an integer multiple of
        ``1 / SS``" to recover the effective COA. This wrap is *not* undone
        here -- it is a burst-domain concern resolved at focusing / mosaic
        time, not in sub-aperture splitting. Inputs that already alias will
        not deskew correctly.

        Parameters
        ----------
        rows : int
            Number of image rows.
        cols : int
            Number of image columns.

        Returns
        -------
        np.ndarray or None
            Deskew phase, shape ``(rows, cols)``, float64, or ``None``.
        """
        if not self._can_deskew:
            return None

        # Pixel indices -> meters relative to the SCP (full-image indices
        # account for chip offsets via first_row / first_col).
        row_m = (
            (np.arange(rows, dtype=np.float64) + self._first_row - self._scp_row)
            * self._row_ss
        ).reshape(-1, 1)
        col_m = (
            (np.arange(cols, dtype=np.float64) + self._first_col - self._scp_col)
            * self._col_ss
        ).reshape(1, -1)

        # Integrate DeltaKCOAPoly along the decomposition axis:
        # axis 0 (range) -> integrate in row (x); axis 1 (azimuth) -> col (y).
        int_coefs = self._polyint_coefs(self._delta_k_coa_poly.coefs, self._axis)
        phase = Poly2D(coefs=int_coefs)(row_m, col_m)
        return np.broadcast_to(phase, (rows, cols)).astype(np.float64)

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

        base_name = 'channel0'
        if getattr(metadata, 'channel_metadata', None):
            base_name = metadata.channel_metadata[0].name

        channel_metadata = ImageTransform._make_derived_channels(
            names=[f'{base_name}_sublook_{i}' for i in range(result.shape[0])],
            source_indices=[[0] for _ in range(result.shape[0])],
            role='look',
        )
        updated = dataclasses.replace(
            metadata,
            bands=result.shape[0],
            axis_order='CYX',
            channel_metadata=channel_metadata,
        )
        return result, updated

    # ------------------------------------------------------------------
    # Core decomposition
    # ------------------------------------------------------------------

    def decompose(
        self, image: Union[np.ndarray, 'torch.Tensor']
    ) -> np.ndarray:
        """Decompose a complex SAR image into sub-aperture looks.

        Dispatches to the appropriate backend based on input type:

        - ``torch.Tensor`` → ``_decompose_torch`` (PyTorch CUDA / CPU)
        - ``cupy.ndarray`` → ``_decompose_numpy`` with ``xp = cupy``
        - ``numpy.ndarray`` → ``_decompose_numpy`` with ``xp = numpy``

        Parameters
        ----------
        image : np.ndarray, cupy.ndarray, or torch.Tensor
            2D complex array, shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray or cupy.ndarray
            Complex sub-look stack, shape ``(num_looks, rows, cols)``.
            Returns ``np.ndarray`` for numpy or torch input;
            ``cupy.ndarray`` for CuPy input.

        Raises
        ------
        TypeError
            If *image* is not complex-valued, or is a torch tensor
            when torch is not installed.
        ValueError
            If *image* is not 2D.
        """
        if _HAS_TORCH and isinstance(image, torch.Tensor):
            logger.info(
                "Sublook decomposition: backend=torch, num_looks=%d",
                self._num_looks,
            )
            return self._decompose_torch(image)
        logger.info(
            "Sublook decomposition: backend=numpy, num_looks=%d",
            self._num_looks,
        )
        return self._decompose_numpy(image)

    def _decompose_numpy(self, image: np.ndarray) -> np.ndarray:
        """Numpy / CuPy FFT path.

        Uses the xp pattern so the same implementation runs on either
        backend: ``xp = cupy`` when *image* is a ``cupy.ndarray``,
        otherwise ``xp = numpy``.  Helper methods (``_compute_subband_bins``
        and ``_build_deweight_array``) always run on CPU and return numpy
        arrays; the deweight array is uploaded to the GPU via
        ``xp.asarray()`` when needed.

        Parameters
        ----------
        image : np.ndarray or cupy.ndarray
            2D complex array.

        Returns
        -------
        np.ndarray or cupy.ndarray
            Complex sub-look stack, shape ``(num_looks, rows, cols)``.
            Same array type as *image*.
        """
        xp = cp if (_HAS_CUPY and isinstance(image, cp.ndarray)) else np
        _validate_complex_2d(image)
        axis = self._axis
        rows, cols = image.shape
        n_samples = image.shape[axis]

        # Center the phase history (deskew) so the COA frequency lands at DC
        # uniformly across the image. A no-op for spotlight; required for
        # stripmap / ScanSAR where DeltaKCOA varies across the scene.
        deskew_phase = self._compute_deskew_phase(rows, cols)
        if deskew_phase is not None:
            ramp = xp.exp(
                1j * self._sgn * 2.0 * np.pi * xp.asarray(deskew_phase)
            ).astype(image.dtype)
            image = image * ramp

        starts, stops, osr = self._compute_subband_bins(
            n_samples, centered=deskew_phase is not None
        )

        # FFT along decomposition axis, then shift DC to center
        spectrum = xp.fft.fftshift(xp.fft.fft(image, axis=axis), axes=axis)

        # Deweight within signal support
        support_start = starts[0]
        support_stop = stops[-1]
        n_support_bins = support_stop - support_start
        deweight_arr = self._build_deweight_array(n_support_bins)
        if deweight_arr is not None:
            # Build broadcastable shape; upload numpy deweight to device if needed
            shape = [1, 1]
            shape[axis] = n_support_bins
            dw = xp.asarray(deweight_arr.reshape(shape))
            slices = [slice(None), slice(None)]
            slices[axis] = slice(support_start, support_stop)
            spectrum[tuple(slices)] = spectrum[tuple(slices)] * dw

        # Re-skew ramp restores each look to the original image geometry.
        reskew = None
        if deskew_phase is not None:
            reskew = xp.exp(
                -1j * self._sgn * 2.0 * np.pi * xp.asarray(deskew_phase)
            ).astype(image.dtype)

        # Extract sub-looks
        result = xp.zeros(
            (self._num_looks, rows, cols), dtype=image.dtype
        )

        for i in range(self._num_looks):
            sub_spectrum = xp.zeros_like(spectrum)
            slices = [slice(None), slice(None)]
            slices[axis] = slice(starts[i], stops[i])
            sub_spectrum[tuple(slices)] = spectrum[tuple(slices)]

            # Shift back and inverse FFT
            look = xp.fft.ifft(
                xp.fft.ifftshift(sub_spectrum, axes=axis), axis=axis
            )
            if reskew is not None:
                look = look * reskew
            result[i] = look

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
        rows, cols = tensor.shape
        n_samples = tensor.shape[dim]

        # Center the phase history (deskew) before cutting; no-op for
        # spotlight, required for stripmap / ScanSAR.
        deskew_phase = self._compute_deskew_phase(rows, cols)
        reskew = None
        if deskew_phase is not None:
            phase_t = torch.from_numpy(deskew_phase).to(
                device=tensor.device, dtype=tensor.real.dtype
            )
            two_pi_sgn = self._sgn * 2.0 * np.pi
            ramp = torch.exp(1j * two_pi_sgn * phase_t).to(tensor.dtype)
            reskew = torch.exp(-1j * two_pi_sgn * phase_t).to(tensor.dtype)
            tensor = tensor * ramp

        starts, stops, osr = self._compute_subband_bins(
            n_samples, centered=deskew_phase is not None
        )

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
            if reskew is not None:
                look = look * reskew
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
