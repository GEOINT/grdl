# -*- coding: utf-8 -*-
"""
Polarimetric Matrix Products - Spatially-averaged covariance and coherency
matrices for multi-polarization SAR data.

Two classes are provided:

``CovarianceMatrix``
    Computes the N×N covariance matrix **[C]** using the lexicographic
    target vector.

    For **reciprocal quad-pol** (S_HV = S_VH assumed) the 3×3 [C3] uses:

        k = [ S_HH, sqrt(2) * S_HV, S_VV ]^T

    For **dual-pol** the 2×2 [C2] uses:

        k = [ S_co, S_cross ]^T

    In both cases: C = < k * k^H > (spatial ensemble average via boxcar).

``CoherencyMatrix``
    Computes the N×N coherency matrix **[T]** using the Pauli target vector.

    For **reciprocal quad-pol** the 3×3 [T3] uses:

        k_P = [ S_HH + S_VV, S_HH - S_VV, 2 * S_HV ]^T / sqrt(2)

    For **dual-pol** the 2×2 [T2] uses:

        k_P = [ S_co + S_cross, S_co - S_cross ]^T / sqrt(2)

    T = < k_P * k_P^H > (spatial ensemble average).

Output format
-------------
Both classes return a complex NumPy array of shape
``(N, N, rows, cols)`` where N is the matrix dimension (2 or 3).
Metadata is updated with ``axis_order='CCYX'`` and per-element
channel descriptors in ``channel_metadata`` (flattened row-major
ordering of matrix elements).

The diagonal elements C[i, i, :, :] are always real-valued (returned
as complex with zero imaginary part), and the off-diagonal elements
satisfy the Hermitian property: C[i, j] = conj(C[j, i]).

Dependencies
------------
scipy (for uniform_filter spatial averaging)

Author
------
Duane Smalley, PhD / Viplob Banerjee
geoint.org

License
-------
MIT License
Copyright (c) 2024 geoint.org

Created
-------
2026-04-15
"""

# Standard library
import dataclasses
from typing import Annotated, Any, Dict, Optional, Tuple, TYPE_CHECKING

# Third-party
import numpy as np
from scipy.ndimage import uniform_filter

try:
    import cupy as cp
    import cupyx.scipy.ndimage as _cupyx_ndimage
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False
    cp = None
    _cupyx_ndimage = None

# GRDL internal
from grdl.image_processing.base import ImageProcessor, ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality, ProcessorCategory

if TYPE_CHECKING:
    from grdl.IO.models.base import ChannelMetadata, ImageMetadata


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _smooth(arr: np.ndarray, ws: int, is_gpu: bool) -> np.ndarray:
    """Boxcar spatial average over a real or complex 2-D array."""
    if not is_gpu:
        if np.iscomplexobj(arr):
            return (
                uniform_filter(arr.real, size=ws).astype(arr.real.dtype)
                + 1j * uniform_filter(arr.imag, size=ws).astype(arr.imag.dtype)
            )
        return uniform_filter(arr, size=ws)
    ndi = _cupyx_ndimage.uniform_filter
    if cp.iscomplexobj(arr):
        return (
            ndi(arr.real, size=ws).astype(arr.real.dtype)
            + 1j * ndi(arr.imag, size=ws).astype(arr.imag.dtype)
        )
    return ndi(arr, size=ws)


def _build_target_vector_c(
    shh: np.ndarray,
    shv: np.ndarray,
    svh: Optional[np.ndarray],
    svv: Optional[np.ndarray],
    dual: bool,
) -> list:
    """Lexicographic target vector components.

    Dual-pol:  [s_co, s_cross]
    Quad-pol:  [shh, sqrt(2)*mean(shv,svh), svv]  (reciprocal)
    """
    if dual:
        return [shh, shv]
    xp = cp if (_HAS_CUPY and isinstance(shh, cp.ndarray)) else np
    cross = shv if svh is None else (shv + svh) * xp.float32(0.5)
    norm = xp.float32(np.sqrt(2.0))
    return [shh, norm * cross, svv]


def _build_target_vector_t(
    shh: np.ndarray,
    shv: np.ndarray,
    svh: Optional[np.ndarray],
    svv: Optional[np.ndarray],
    dual: bool,
) -> list:
    """Pauli target vector components (1/sqrt(2) factor included).

    Dual-pol:  [(s_co + s_cross), (s_co - s_cross)] / sqrt(2)
    Quad-pol:  [(shh + svv), (shh - svv), 2*mean(shv,svh)] / sqrt(2)
    """
    xp = cp if (_HAS_CUPY and isinstance(shh, cp.ndarray)) else np
    norm = xp.float32(1.0 / np.sqrt(2.0))
    if dual:
        return [
            (shh + shv) * norm,
            (shh - shv) * norm,
        ]
    cross = shv if svh is None else (shv + svh) * xp.float32(0.5)
    two_cross = cross * xp.float32(2.0)
    return [
        (shh + svv) * norm,
        (shh - svv) * norm,
        two_cross * norm,
    ]


def _compute_matrix(
    components: list,
    window_size: int,
    is_gpu: bool,
) -> np.ndarray:
    """Compute spatially-averaged Hermitian matrix from target vector.

    Parameters
    ----------
    components : list of 2-D complex arrays
        Target vector channels, length N.
    window_size : int
        Boxcar averaging window (pixels, square).
    is_gpu : bool
        Whether to use CuPy paths.

    Returns
    -------
    np.ndarray
        Complex array, shape ``(N, N, rows, cols)``.
    """
    n = len(components)
    rows, cols = components[0].shape
    xp = cp if is_gpu else np
    mat = xp.zeros((n, n, rows, cols), dtype=np.complex64)

    for i in range(n):
        for j in range(i, n):
            elem = _smooth(components[i] * xp.conj(components[j]),
                           window_size, is_gpu)
            mat[i, j] = elem
            if i != j:
                mat[j, i] = xp.conj(elem)

    return mat


def _extract_channels(
    source: np.ndarray,
    metadata: 'ImageMetadata',
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract HH, HV, VH, VV in that order from a CYX source."""
    if source.ndim != 3:
        return None, None, None, None

    axis_order = getattr(metadata, 'axis_order', None)
    channel_metadata = getattr(metadata, 'channel_metadata', None)
    bands = getattr(metadata, 'bands', source.shape[0])

    # Determine if channels-first
    if axis_order == 'CYX':
        chs_first = True
    elif axis_order == 'YXC':
        chs_first = False
    else:
        chs_first = (source.shape[0] == bands and source.shape[-1] != bands)

    n = source.shape[0] if chs_first else source.shape[-1]
    if n < 2:
        return None, None, None, None

    def ch(i):
        return source[i] if chs_first else source[..., i]

    if n >= 4:
        return ch(0), ch(1), ch(2), ch(3)
    if n == 2:
        return ch(0), ch(1), None, None
    return None, None, None, None


def _make_matrix_channel_metadata(n: int, kind: str) -> list:
    """Flat channel descriptors for the N×N matrix (row-major)."""
    from grdl.IO.models.base import ChannelMetadata

    descriptors = []
    idx = 0
    for i in range(n):
        for j in range(n):
            name = f'{kind}[{i},{j}]'
            descriptors.append(
                ChannelMetadata(
                    index=idx,
                    name=name,
                    role='polarimetric_matrix',
                    source_indices=[i, j],
                )
            )
            idx += 1
    return descriptors


# ---------------------------------------------------------------------------
# CovarianceMatrix
# ---------------------------------------------------------------------------

@processor_version('0.1.0')
@processor_tags(
    modalities=[ImageModality.SAR],
    category=ProcessorCategory.ENHANCE,
    description='Spatially-averaged polarimetric covariance matrix [C]',
)
class CovarianceMatrix(ImageProcessor):
    """Polarimetric covariance matrix [C2] or [C3].

    Computes the spatially-averaged Hermitian covariance matrix from the
    lexicographic target vector for dual-pol or reciprocal quad-pol SAR data.

    **Dual-pol [C2]** (2×2):
        k = [S_co, S_cross]^T
        C2 = <k * k^H>

    **Quad-pol [C3]** (3×3, reciprocal):
        k = [S_HH, sqrt(2) * S_HV, S_VV]^T
        C3 = <k * k^H>

    The averaging operator ``<.>`` is a square boxcar filter of
    ``window_size × window_size`` pixels applied independently to the
    real and imaginary parts of each matrix element.

    Parameters
    ----------
    window_size : int
        Spatial averaging window side length (pixels, odd, >= 3).
        Default 7.

    Returns
    -------
    np.ndarray
        Complex array, shape ``(N, N, rows, cols)`` (``axis_order='CCYX'``),
        dtype complex64. N=2 for dual-pol, N=3 for quad-pol.

    Examples
    --------
    >>> from grdl.IO.sar import BIOMASSReader
    >>> from grdl.image_processing.decomposition import CovarianceMatrix
    >>>
    >>> with BIOMASSReader('scene/') as reader:
    ...     cube = reader.read_full()      # (4, rows, cols) CYX complex
    ...     cmat = CovarianceMatrix(window_size=9)
    ...     C3, meta = cmat.execute(reader.metadata, cube)
    ...     # C3 shape: (3, 3, rows, cols), diagonal is real-valued power
    """

    __gpu_compatible__ = True

    window_size: Annotated[
        int, Range(min=3, max=63), Desc('Boxcar averaging window size')
    ] = 7

    def __init__(self, window_size: int = 7) -> None:
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be an odd integer >= 3, got {window_size}"
            )
        self.window_size = window_size

    def compute(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: Optional[np.ndarray] = None,
        svv: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the covariance matrix directly from channel arrays.

        Parameters
        ----------
        shh : np.ndarray
            HH channel (or co-pol for dual-pol). Shape ``(rows, cols)``.
        shv : np.ndarray
            HV channel (or cross-pol for dual-pol). Shape ``(rows, cols)``.
        svh : np.ndarray, optional
            VH channel. If None, HV is used as the cross-pol (reciprocal).
        svv : np.ndarray, optional
            VV channel. If None, dual-pol [C2] is computed.

        Returns
        -------
        np.ndarray
            Complex array, shape ``(N, N, rows, cols)``.
        """
        is_gpu = _HAS_CUPY and isinstance(shh, cp.ndarray)
        dual = svv is None
        components = _build_target_vector_c(shh, shv, svh, svv, dual)
        return _compute_matrix(components, self.window_size, is_gpu)

    def execute(
        self,
        metadata: 'ImageMetadata',
        source: np.ndarray,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        """Execute via the universal protocol.

        Channels may be passed as kwargs (``shh``, ``shv``, ``svh``,
        ``svv``) or extracted automatically from a CYX/YXC source cube.
        With only 2 channels present or extracted, [C2] is computed;
        with 4 channels, [C3] (reciprocal) is computed.

        Returns
        -------
        tuple[np.ndarray, ImageMetadata]
            ``(C_matrix, updated_metadata)`` where ``C_matrix`` has
            shape ``(N, N, rows, cols)`` and metadata carries
            ``axis_order='CCYX'``.
        """
        self._metadata = metadata

        shh = kwargs.pop('shh', None)
        shv = kwargs.pop('shv', None)
        svh = kwargs.pop('svh', None)
        svv = kwargs.pop('svv', None)

        if shh is None:
            shh, shv, svh, svv = _extract_channels(source, metadata)

        result = self.compute(shh, shv, svh, svv)
        n = result.shape[0]

        updated = dataclasses.replace(
            metadata,
            bands=n * n,
            axis_order='CCYX',
            channel_metadata=_make_matrix_channel_metadata(n, 'C'),
            dtype=str(result.dtype),
        )
        return result, updated

    def __repr__(self) -> str:
        return f"CovarianceMatrix(window_size={self.window_size})"


# ---------------------------------------------------------------------------
# CoherencyMatrix
# ---------------------------------------------------------------------------

@processor_version('0.1.0')
@processor_tags(
    modalities=[ImageModality.SAR],
    category=ProcessorCategory.ENHANCE,
    description='Spatially-averaged polarimetric coherency matrix [T]',
)
class CoherencyMatrix(ImageProcessor):
    """Polarimetric coherency matrix [T2] or [T3].

    Computes the spatially-averaged Hermitian coherency matrix from the
    Pauli target vector for dual-pol or reciprocal quad-pol SAR data.

    **Dual-pol [T2]** (2×2):
        k_P = [(S_co + S_cross), (S_co - S_cross)]^T / sqrt(2)
        T2  = <k_P * k_P^H>

    **Quad-pol [T3]** (3×3, reciprocal):
        k_P = [(S_HH + S_VV), (S_HH - S_VV), 2 * S_HV]^T / sqrt(2)
        T3  = <k_P * k_P^H>

    The Pauli basis is unitary: under monostatic reciprocity ``trace(T3)
    = trace(C3) = span``.  [T3] is the preferred matrix for scattering
    mechanism parameter estimation (entropy, alpha, etc.).

    Parameters
    ----------
    window_size : int
        Spatial averaging window side length (pixels, odd, >= 3).
        Default 7.

    Returns
    -------
    np.ndarray
        Complex array, shape ``(N, N, rows, cols)`` (``axis_order='CCYX'``),
        dtype complex64. N=2 for dual-pol, N=3 for quad-pol.

    Examples
    --------
    >>> from grdl.IO.sar import BIOMASSReader
    >>> from grdl.image_processing.decomposition import CoherencyMatrix
    >>>
    >>> with BIOMASSReader('scene/') as reader:
    ...     cube = reader.read_full()
    ...     tmat = CoherencyMatrix(window_size=9)
    ...     T3, meta = tmat.execute(reader.metadata, cube)
    ...     # T3.diagonal: T3[0,0] = <|shh+svv|^2>/2 (odd-bounce power)
    ...     #              T3[1,1] = <|shh-svv|^2>/2 (even-bounce power)
    ...     #              T3[2,2] = 2*<|shv|^2>     (volume power)
    """

    __gpu_compatible__ = True

    window_size: Annotated[
        int, Range(min=3, max=63), Desc('Boxcar averaging window size')
    ] = 7

    def __init__(self, window_size: int = 7) -> None:
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be an odd integer >= 3, got {window_size}"
            )
        self.window_size = window_size

    def compute(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: Optional[np.ndarray] = None,
        svv: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the coherency matrix directly from channel arrays.

        Parameters
        ----------
        shh : np.ndarray
            HH channel (or co-pol). Shape ``(rows, cols)``.
        shv : np.ndarray
            HV channel (or cross-pol). Shape ``(rows, cols)``.
        svh : np.ndarray, optional
            VH channel. If None, HV is used (reciprocity assumed).
        svv : np.ndarray, optional
            VV channel. If None, dual-pol [T2] is computed.

        Returns
        -------
        np.ndarray
            Complex array, shape ``(N, N, rows, cols)``.
        """
        is_gpu = _HAS_CUPY and isinstance(shh, cp.ndarray)
        dual = svv is None
        components = _build_target_vector_t(shh, shv, svh, svv, dual)
        return _compute_matrix(components, self.window_size, is_gpu)

    def execute(
        self,
        metadata: 'ImageMetadata',
        source: np.ndarray,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        """Execute via the universal protocol.

        Channels may be passed as kwargs (``shh``, ``shv``, ``svh``,
        ``svv``) or extracted automatically from a CYX/YXC source cube.
        With only 2 channels, [T2] is computed; with 4 channels, [T3].

        Returns
        -------
        tuple[np.ndarray, ImageMetadata]
            ``(T_matrix, updated_metadata)`` where ``T_matrix`` has
            shape ``(N, N, rows, cols)`` and metadata carries
            ``axis_order='CCYX'``.
        """
        self._metadata = metadata

        shh = kwargs.pop('shh', None)
        shv = kwargs.pop('shv', None)
        svh = kwargs.pop('svh', None)
        svv = kwargs.pop('svv', None)

        if shh is None:
            shh, shv, svh, svv = _extract_channels(source, metadata)

        result = self.compute(shh, shv, svh, svv)
        n = result.shape[0]

        updated = dataclasses.replace(
            metadata,
            bands=n * n,
            axis_order='CCYX',
            channel_metadata=_make_matrix_channel_metadata(n, 'T'),
            dtype=str(result.dtype),
        )
        return result, updated

    def __repr__(self) -> str:
        return f"CoherencyMatrix(window_size={self.window_size})"


# ---------------------------------------------------------------------------
# Mueller basis transformation matrix (module-level constants)
# ---------------------------------------------------------------------------
# A maps the lexicographic 4-scattering-vector basis to the Stokes basis.
# A * A^H = I  (unitary), so A^{-1} = A^H.
_A_MUELLER: np.ndarray = (1.0 / np.sqrt(2.0)) * np.array(
    [
        [1,   0,  0,  1],
        [1,   0,  0, -1],
        [0,   1,  1,  0],
        [0, -1j, 1j,  0],
    ],
    dtype=np.complex128,
)
_A_MUELLER_INV: np.ndarray = np.conj(_A_MUELLER).T.copy()  # = A^H


# ---------------------------------------------------------------------------
# StokesVector
# ---------------------------------------------------------------------------


@processor_version('0.1.0')
@processor_tags(
    modalities=[ImageModality.SAR],
    category=ProcessorCategory.ENHANCE,
    description='Spatially-averaged Stokes parameters for dual-pol or co-pol SAR channels',
)
class StokesVector(ImageProcessor):
    """Spatially-averaged Stokes vector parameters from SAR polarimetric data.

    Computes the four Stokes parameters that describe the polarization state
    of the backscattered wave. For a complex field pair (E_H, E_V)::

        S0 = <|E_H|² + |E_V|²>      total intensity              (>= 0)
        S1 = <|E_H|² - |E_V|²>      H vs V linear polarization
        S2 =  2 * <Re(E_H · E_V*)>  ±45° linear polarization
        S3 = -2 * <Im(E_H · E_V*)>  right vs left circular

    where ``<·>`` denotes a spatial boxcar average over ``window_size``×``window_size``
    pixels.  The degree of polarization can be obtained via
    :meth:`degree_of_polarization`.

    Channel selection
    -----------------
    * **Dual-pol** (2 input channels): channel 0 → E_H, channel 1 → E_V.
    * **Quad-pol** (4 input channels, CYX cube): SHH (ch 0) → E_H,
      SVV (ch 3) → E_V  (co-pol pair convention).

    Channel arrays may also be supplied directly as ``e_h`` / ``e_v`` kwargs
    to :meth:`execute`.

    Parameters
    ----------
    window_size : int
        Boxcar averaging window (pixels, odd, >= 3).  Default ``7``.

    Returns
    -------
    np.ndarray
        Float32 array, shape ``(4, rows, cols)`` (``axis_order='CYX'``).
        Channels ordered [S0, S1, S2, S3].

    Examples
    --------
    >>> sv = StokesVector(window_size=9)
    >>> stokes = sv.compute(shh, svv)
    >>> dop = sv.degree_of_polarization(stokes)
    """

    __gpu_compatible__ = True

    window_size: Annotated[
        int, Range(min=3, max=63), Desc('Boxcar averaging window size (pixels, odd)')
    ] = 7

    def __init__(self, window_size: int = 7) -> None:
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be an odd integer >= 3, got {window_size}"
            )
        self.window_size = window_size

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(self, e_h: np.ndarray, e_v: np.ndarray) -> np.ndarray:
        """Compute spatially-averaged Stokes parameters.

        Parameters
        ----------
        e_h : np.ndarray
            Horizontal complex field component, shape ``(rows, cols)``.
        e_v : np.ndarray
            Vertical complex field component, shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Float32 array, shape ``(4, rows, cols)``.
        """
        is_gpu = _HAS_CUPY and isinstance(e_h, cp.ndarray)
        xp = cp if is_gpu else np
        ws = self.window_size

        cross = e_h * xp.conj(e_v)
        S0 = _smooth(xp.abs(e_h) ** 2 + xp.abs(e_v) ** 2,      ws, is_gpu)
        S1 = _smooth(xp.abs(e_h) ** 2 - xp.abs(e_v) ** 2,      ws, is_gpu)
        S2 = _smooth(2.0 * xp.real(cross),                       ws, is_gpu)
        S3 = _smooth(-2.0 * xp.imag(cross),                      ws, is_gpu)

        return xp.stack([S0, S1, S2, S3], axis=0).astype(xp.float32)

    def degree_of_polarization(self, stokes: np.ndarray) -> np.ndarray:
        """Per-pixel degree of polarization in ``[0, 1]``.

        Parameters
        ----------
        stokes : np.ndarray
            Output of :meth:`compute`, shape ``(4, rows, cols)``.

        Returns
        -------
        np.ndarray
            Float32, shape ``(rows, cols)``.  0 = unpolarized, 1 = fully polarized.
        """
        xp = cp if (_HAS_CUPY and isinstance(stokes, cp.ndarray)) else np
        S0, S1, S2, S3 = stokes[0], stokes[1], stokes[2], stokes[3]
        eps = xp.finfo(xp.float32).tiny
        dop = xp.sqrt(S1 ** 2 + S2 ** 2 + S3 ** 2) / (S0 + eps)
        return xp.clip(dop, 0.0, 1.0).astype(xp.float32)

    # ------------------------------------------------------------------
    # ImageProcessor protocol
    # ------------------------------------------------------------------

    def execute(
        self,
        metadata: 'ImageMetadata',
        source: np.ndarray,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        """Execute Stokes vector computation via the universal protocol.

        Parameters
        ----------
        metadata : ImageMetadata
            Source image metadata (used for channel extraction).
        source : np.ndarray
            Source image cube.
        e_h, e_v : np.ndarray, optional kwargs
            If provided, used directly (bypasses cube extraction).

        Returns
        -------
        tuple[np.ndarray, ImageMetadata]
            ``(stokes_stack, updated_metadata)``, shape ``(4, rows, cols)``,
            ``axis_order='CYX'``.
        """
        self._metadata = metadata

        e_h = kwargs.pop('e_h', None)
        e_v = kwargs.pop('e_v', None)

        if e_h is None:
            shh, shv, svh, svv = _extract_channels(source, metadata)
            if svv is not None:
                e_h, e_v = shh, svv   # quad-pol: co-pol pair
            else:
                e_h, e_v = shh, shv   # dual-pol

        result = self.compute(e_h, e_v)  # (4, rows, cols)
        rows, cols = result.shape[1], result.shape[2]

        from grdl.IO.models.base import ChannelMetadata as _CM
        source_idx = [0, 1, 2, 3]
        channel_meta = [
            _CM(index=0, name='S0', role='stokes', source_indices=source_idx),
            _CM(index=1, name='S1', role='stokes', source_indices=source_idx),
            _CM(index=2, name='S2', role='stokes', source_indices=source_idx),
            _CM(index=3, name='S3', role='stokes', source_indices=source_idx),
        ]

        updated = dataclasses.replace(
            metadata,
            rows=rows,
            cols=cols,
            bands=4,
            axis_order='CYX',
            channel_metadata=channel_meta,
            dtype='float32',
        )
        return result, updated

    def __repr__(self) -> str:
        return f"StokesVector(window_size={self.window_size})"


# ---------------------------------------------------------------------------
# KennaughMatrix
# ---------------------------------------------------------------------------


@processor_version('0.1.0')
@processor_tags(
    modalities=[ImageModality.SAR],
    category=ProcessorCategory.ENHANCE,
    description='Spatially-averaged polarimetric Kennaugh matrix [K] (4×4 real)',
)
class KennaughMatrix(ImageProcessor):
    """Polarimetric Kennaugh matrix [K] (4×4 real symmetric).

    The Kennaugh matrix is a 4×4 real matrix that fully characterises the
    polarimetric scattering behaviour of a spatially-averaged target and is
    the Mueller-calculus analogue of the coherency matrix.

    Derivation::

        C4  = < [SHH, SHV, SVH, SVV]^T [...]^H >   4×4 lex. covariance
        M   =  A · C4 · A^H                          Mueller matrix
        K   =  (1/2) · Re(M)                         Kennaugh matrix

    where::

        A = (1/√2) · [[1,  0,  0,  1],
                      [1,  0,  0, -1],
                      [0,  1,  1,  0],
                      [0, -j,  j,  0]]

    and A is unitary (A · A^H = I).

    Under monostatic reciprocity (SHV ≈ SVH) the 4th row/column of K is
    effectively zero, reducing the active degrees of freedom to the upper-left
    3×3 sub-matrix.

    Physical interpretation
    -----------------------
    * ``K[0, 0]`` — co-pol power contribution (surface+double-bounce)
    * ``K[1, 1]`` — Huynen "a" parameter (double-bounce dominance)
    * ``K[0, 1]`` — co-pol coherence (real part)
    * ``K[2, 2]``, ``K[3, 3]`` — cross-pol and circular contributions

    Parameters
    ----------
    window_size : int
        Boxcar averaging window (pixels, odd, >= 3).  Default ``7``.

    Returns
    -------
    np.ndarray
        Float32 array, shape ``(4, 4, rows, cols)`` (``axis_order='KKYX'``).

    Examples
    --------
    >>> km = KennaughMatrix(window_size=9)
    >>> K4, meta = km.execute(reader.metadata, cube)
    >>> # Pure surface scatterer → K4[1,1] ≈ K4[2,2] ≈ K4[3,3] ≈ 0
    """

    __gpu_compatible__ = False   # np.einsum complex path; cupy not yet verified

    window_size: Annotated[
        int, Range(min=3, max=63), Desc('Boxcar averaging window size (pixels, odd)')
    ] = 7

    def __init__(self, window_size: int = 7) -> None:
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be an odd integer >= 3, got {window_size}"
            )
        self.window_size = window_size

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: Optional[np.ndarray] = None,
        svv: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the spatially-averaged Kennaugh matrix.

        Parameters
        ----------
        shh : np.ndarray
            Complex HH channel, shape ``(rows, cols)``.
        shv : np.ndarray
            Complex HV channel.
        svh : np.ndarray, optional
            VH channel.  If ``None``, SHV is reused (reciprocity assumed).
        svv : np.ndarray
            Complex VV channel.  Required.

        Returns
        -------
        np.ndarray
            Float32, shape ``(4, 4, rows, cols)``.
        """
        if svh is None:
            svh = shv
        if svv is None:
            raise ValueError(
                "svv is required for KennaughMatrix; it cannot be inferred."
            )

        is_gpu = _HAS_CUPY and isinstance(shh, cp.ndarray)

        # Build 4×4 lexicographic covariance C4 (from all 4 channels, no √2 rescaling)
        C4 = _compute_matrix([shh, shv, svh, svv], self.window_size, is_gpu)
        # C4: (4, 4, rows, cols)  complex64

        # Promote to complex128 for numerical precision in the basis change
        C4_d = C4.astype(np.complex128)

        # Mueller matrix: M = A · C4 · A^H
        # einsum 'ik,klyx,lj->ijyx' computes per-pixel M[i,j] = Σ_k Σ_l A[i,k]*C4[k,l]*A^H[l,j]
        M = np.einsum('ik,klyx,lj->ijyx', _A_MUELLER, C4_d, _A_MUELLER_INV)

        # Kennaugh = (1/2) Re(M)
        return (0.5 * np.real(M)).astype(np.float32)

    # ------------------------------------------------------------------
    # ImageProcessor protocol
    # ------------------------------------------------------------------

    def execute(
        self,
        metadata: 'ImageMetadata',
        source: np.ndarray,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        """Execute Kennaugh matrix computation via the universal protocol.

        Requires quad-pol input.  Channels may be supplied as explicit
        ``shh`` / ``shv`` / ``svh`` / ``svv`` kwargs or extracted from a
        CYX source cube.

        Returns
        -------
        tuple[np.ndarray, ImageMetadata]
            ``(K_matrix, updated_metadata)``, shape ``(4, 4, rows, cols)``,
            ``axis_order='KKYX'``.
        """
        self._metadata = metadata

        shh = kwargs.pop('shh', None)
        shv = kwargs.pop('shv', None)
        svh = kwargs.pop('svh', None)
        svv = kwargs.pop('svv', None)

        if shh is None:
            shh, shv, svh, svv = _extract_channels(source, metadata)

        result = self.compute(shh, shv, svh, svv)  # (4, 4, rows, cols)
        rows, cols = result.shape[2], result.shape[3]

        updated = dataclasses.replace(
            metadata,
            rows=rows,
            cols=cols,
            bands=16,
            axis_order='KKYX',
            channel_metadata=_make_matrix_channel_metadata(4, 'K'),
            dtype='float32',
        )
        return result, updated

    def __repr__(self) -> str:
        return f"KennaughMatrix(window_size={self.window_size})"
