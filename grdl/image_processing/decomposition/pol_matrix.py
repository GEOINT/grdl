"""
Polarimetric Matrix Products - Spatially-averaged covariance and coherency
matrices for multi-polarization SAR data.

The implementation is basis-agnostic for orthogonal polarization pairs
(for example H/V, X/Y, or R/L) while preserving co-pol vs cross-pol role.

Covariance and coherency construction are provided by:

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

StokesVector and KennaughMatrix provide additional polarization products.
Shared filter-boundary helpers validate matrix descriptors, prepare formed
matrices or raw channels, and pack/unpack real upper-triangle storage. They
do not implement covariance/coherency basis conversion or C4/T4 formation.

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

References
----------
[1] Lee, J. S., and Pottier, E. (2009), Polarimetric Radar Imaging:
    From Basics to Applications, CRC Press.
[2] Cloude, S. R., and Pottier, E. (1997), "An entropy based classification
    scheme for land applications of polarimetric SAR," IEEE TGRS.
[3] Boerner, W.-M. et al. (1998), "Polarimetry in radar remote sensing:
    basic and applied concepts," in Polarimetric Radar Imaging.
[4] Huynen, J. R. (1970), Phenomenological Theory of Radar Targets,
    PhD thesis, TU Delft.

Future Work
-----------
Hybrid/compact-pol transmit configurations (for example pi/4 and circular
transmit modes) are intentionally deferred and will be added in future
algorithms that build on these matrix products.

Author
------
Duane Smalley, PhD / Viplob Banerjee / Jason Fritz, PhD
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
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Annotated, Any, cast

# Third-party
import numpy as np

try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False
    cp = None

# GRDL internal
from grdl.exceptions import ValidationError
from grdl.image_processing.base import ImageProcessor
from grdl.image_processing.filters.linear import boxcar_mean
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.tiling import WindowedTileHalo
from grdl.image_processing.versioning import processor_tags, processor_version
from grdl.IO.models.polarization import channel_axis, resolve_raw_indices
from grdl.vocabulary import ImageModality, ProcessorCategory

if TYPE_CHECKING:
    from grdl.IO.models.base import ChannelMetadata, ImageMetadata


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

MATRIX_TYPES = ("C2", "T2", "C3", "T3", "C4", "T4")


def validate_matrix_type(matrix_type: str | None) -> str | None:
    """Normalize a basis descriptor without inferring covariance versus coherency.

    Parameters
    ----------
    matrix_type : str or None
        C2/T2/C3/T3/C4/T4, case-insensitive. None and 'auto' mean unspecified.

    Returns
    -------
    str or None
        Uppercase descriptor, or None for an unspecified basis.

    Raises
    ------
    ValidationError
        If the descriptor is not supported. This does not inspect matrix data.
    """
    if matrix_type in (None, "auto"):
        return None
    if not isinstance(matrix_type, str) or matrix_type.upper() not in MATRIX_TYPES:
        raise ValidationError(f"Unsupported matrix_type {matrix_type!r}; expected {MATRIX_TYPES}")
    return matrix_type.upper()


def unpack_matrix(packed: np.ndarray) -> np.ndarray:
    """Decode real CYX upper-triangle bands into a Hermitian CCYX tensor.

    Parameters
    ----------
    packed : np.ndarray
        Real cube with 4, 9, or 16 bands. For each matrix row, store its
        diagonal followed by real/imaginary pairs for the upper triangle.
        C2/T2 order is [M11, Re(M12), Im(M12), M22].

    Returns
    -------
    np.ndarray
        Complex (N, N, rows, cols) array with a conjugate lower triangle.
        Precision follows np.result_type(packed.dtype, np.complex64).

    Raises
    ------
    ValidationError
        If rank, band count, or real-valued storage requirements are violated.
        Use prepare_polarimetric_matrix() for finite-value and basis checks.
    """
    if packed.ndim != 3 or packed.shape[0] not in (4, 9, 16) or np.iscomplexobj(packed):
        raise ValidationError("Packed matrices require 4, 9, or 16 real CYX bands")
    dimension = int(np.sqrt(packed.shape[0]))
    matrix = np.zeros(
        (dimension, dimension, *packed.shape[1:]),
        dtype=np.result_type(packed.dtype, np.complex64),
    )
    index = 0
    for row in range(dimension):
        matrix[row, row] = packed[index]
        index += 1
        for col in range(row + 1, dimension):
            matrix[row, col] = packed[index] + 1j * packed[index + 1]
            matrix[col, row] = matrix[row, col].conj()
            index += 2
    return matrix


def pack_matrix(matrix: np.ndarray) -> np.ndarray:
    """Encode a CCYX tensor as real CYX upper-triangle bands.

    Parameters
    ----------
    matrix : np.ndarray
        Validated (N, N, rows, cols) Hermitian matrix tensor. The upper
        triangle is authoritative; this helper does not validate Hermiticity.

    Returns
    -------
    np.ndarray
        N*N real bands: each diagonal followed by real/imaginary pairs for
        that row's upper triangle. Diagonal imaginary parts are not stored.
        The result uses the input's real-component dtype.
    """
    bands = []
    for row in range(matrix.shape[0]):
        bands.append(matrix[row, row].real)
        for col in range(row + 1, matrix.shape[0]):
            bands.extend((matrix[row, col].real, matrix[row, col].imag))
    return np.stack(bands)


def prepare_polarimetric_matrix(
    source: np.ndarray,
    *,
    input_kind: str,
    matrix_type: str | None = None,
    channel_metadata: list["ChannelMetadata"] | None = None,
) -> np.ndarray:
    """Validate CCYX matrices or form them once from canonical CYX channels.

    Formed matrices are never converted between C and T bases. Raw channels
    support C2/T2 and reciprocal C3/T3 only, in co/cross or HH,HV,[VH],VV order.
    Channel descriptors, when supplied, take precedence over positional order.

    Parameters
    ----------
    source : np.ndarray
        Numeric finite CCYX tensor, complex CYX SLC cube, or real packed CYX
        bands, as selected by input_kind.
    input_kind : {'matrix', 'channels', 'packed'}
        Input representation. Shape or dtype alone never selects C versus T.
    matrix_type : str, optional
        Required formation basis for channels; optional dimension descriptor
        for formed or packed matrices. C4/T4 must be preformed or packed.
    channel_metadata : list of ChannelMetadata, optional
        Names/roles for ordering raw channels; ignored for formed matrices.

    Returns
    -------
    np.ndarray
        CCYX matrix. Formed input is returned without dtype or basis change.
        Channels use CovarianceMatrix/CoherencyMatrix with window_size=1
        (preserving input precision), averaging HV/VH when four channels are supplied.
        Packed input is decoded with unpack_matrix().

    Raises
    ------
    ValidationError
        If input representation, finite values, matrix dimension, descriptor,
        or channel-count requirements are violated. Structural checks do not
        certify Hermiticity, positive definiteness, or physical matrix basis.
    """
    kind = validate_matrix_type(matrix_type)
    if source.dtype.kind not in "fciub" or not np.isfinite(source).all():
        raise ValidationError("Polarimetric input must be numeric and contain no non-finite values")
    if input_kind == "channels":
        if source.ndim != 3 or not np.iscomplexobj(source):
            raise ValidationError("SLC channels require a complex CYX cube")
        if kind is None:
            raise ValidationError("SLC channels require explicit matrix_type C2/T2/C3/T3")
        expected = (2,) if kind.endswith("2") else (3, 4)
        if kind.endswith("4") or source.shape[0] not in expected:
            raise ValidationError(
                f'{kind} requires {"2" if kind.endswith("2") else "4 or 3 reciprocal"} '
                f"SLC channels, got {source.shape[0]}; C4/T4 formation is unsupported"
            )
        if channel_metadata is not None:
            from grdl.IO.models.base import ImageMetadata

            metadata = ImageMetadata(
                format="array",
                rows=source.shape[1],
                cols=source.shape[2],
                bands=source.shape[0],
                dtype=str(source.dtype),
                axis_order="CYX",
                channel_metadata=channel_metadata,
            )
            source = extract_channels(source, metadata)
            if source.shape[0] not in expected:
                raise ValidationError(
                    f"{kind} channel descriptors resolve incompatible raw count {source.shape[0]}"
                )
        if kind.startswith("C"):
            return CovarianceMatrix(window_size=1).compute(source)
        return CoherencyMatrix(window_size=1).compute(source)
    if input_kind == "packed":
        source = unpack_matrix(source)
    elif input_kind != "matrix":
        raise ValidationError(f"Unsupported polarimetric input_kind {input_kind!r}")
    if source.ndim != 4:
        raise ValidationError(f"Expected 4D matrix (N, N, rows, cols), got shape {source.shape}")
    dimension = source.shape[0]
    if source.shape[1] != dimension:
        raise ValidationError(f"Matrix must be square in first two dims, got {source.shape[:2]}")
    if dimension not in (2, 3, 4):
        raise ValidationError(f"Matrix dimension must be 2, 3, or 4, got {dimension}")
    if kind is not None and dimension != int(kind[1]):
        raise ValidationError(
            f"matrix_type={kind} expects {kind[1]}x{kind[1]}, got {dimension}x{dimension}"
        )
    if any(size == 0 for size in source.shape):
        raise ValidationError("Polarimetric matrices must have nonempty spatial dimensions")
    return source


def smooth(arr: np.ndarray, ws: int, is_gpu: bool) -> np.ndarray:
    """Delegate reflected boxcar averaging; the array selects its backend."""
    return boxcar_mean(arr, ws, mode="reflect")


def target_components(channels: np.ndarray | list[np.ndarray]) -> list[np.ndarray]:
    """Promote component views to one complex precision without stacking them."""
    dtype = np.result_type(np.complex64, *(channel.dtype for channel in channels))
    return [channel.astype(dtype, copy=False) for channel in channels]


def build_target_vector_c(channels: np.ndarray | list[np.ndarray]) -> list:
    """Lexicographic target vector components.

    Dual-pol:  [s_co, s_cross]
    Quad-pol:  [sxx, sqrt(2)*mean(sxy,syx), syy]  (reciprocal)
    """
    channels = target_components(channels)
    if len(channels) == 2:
        return channels
    scalar = np.finfo(channels[0].dtype).dtype.type
    norm = scalar(np.sqrt(2.0))
    if len(channels) == 3:
        cross = channels[1]
        return [channels[0], norm * cross, channels[2]]
    cross = (channels[1] + channels[2]) * scalar(0.5)
    return [channels[0], norm * cross, channels[3]]


def build_target_vector_t(channels: np.ndarray | list[np.ndarray]) -> list:
    """Pauli target vector components (1/sqrt(2) factor included).

    Dual-pol:  [(s_co + s_cross), (s_co - s_cross)] / sqrt(2)
    Quad-pol:  [(sxx + syy), (sxx - syy), 2*mean(sxy,syx)] / sqrt(2)
    """
    channels = target_components(channels)
    scalar = np.finfo(channels[0].dtype).dtype.type
    norm = scalar(1.0 / np.sqrt(2.0))
    if len(channels) == 2:
        return [
            (channels[0] + channels[1]) * norm,
            (channels[0] - channels[1]) * norm,
        ]
    if len(channels) == 3:
        cross = channels[1]
        two_cross = cross * scalar(2.0)
        return [
            (channels[0] + channels[2]) * norm,
            (channels[0] - channels[2]) * norm,
            two_cross * norm,
        ]
    cross = (channels[1] + channels[2]) * scalar(0.5)
    two_cross = cross * scalar(2.0)
    return [
        (channels[0] + channels[3]) * norm,
        (channels[0] - channels[3]) * norm,
        two_cross * norm,
    ]


def iter_matrix_elements(
    components: list[np.ndarray],
    window_size: int,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield averaged upper-triangle elements without a full matrix temporary.

    Diagonals are real power arrays. Off-diagonal products are formed once
    and averaged after multiplication, not by averaging scattering fields.
    """
    components = target_components(components)
    xp: Any = np
    if cp is not None and isinstance(components[0], cp.ndarray):
        xp = cp
    for row, component in enumerate(components):
        for col in range(row, len(components)):
            product = xp.abs(component) ** 2 if row == col else component * components[col].conj()
            yield row, col, boxcar_mean(product, window_size, mode="reflect")


def compute_dual_elements(
    components: list[np.ndarray],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return real diagonals and the complex upper off-diagonal of C2/T2."""
    first, cross, second = (
        element for _, _, element in iter_matrix_elements(components, window_size)
    )
    return first, cross, second


def compute_matrix(
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
        Complex array, shape ``(N, N, rows, cols)``, preserving component
        precision with exactly real diagonals.
    """
    n = len(components)
    rows, cols = components[0].shape
    xp: Any = cp if is_gpu else np
    dtype = np.result_type(np.complex64, *(component.dtype for component in components))
    mat = xp.empty((n, n, rows, cols), dtype=dtype)

    for row, col, element in iter_matrix_elements(components, window_size):
        mat[row, col] = element
        if row != col:
            mat[col, row] = element.conj()

    return cast(np.ndarray, mat)


def stokes_from_c2(
    c11: np.ndarray,
    c12: np.ndarray,
    c22: np.ndarray,
    c21: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Canonical Stokes with S3 = -2 Im(C12) for Hermitian C2.

    When C21 is supplied, use both off-diagonals independently, preserving
    the compact-pol interface for inputs that are not exactly Hermitian.
    """
    elements = [c11, c12, c22] if c21 is None else [c11, c12, c22, c21]
    dtype = np.finfo(np.result_type(np.complex64, *(value.dtype for value in elements))).dtype
    first, second = c11.real.astype(dtype, copy=False), c22.real.astype(dtype, copy=False)
    if c21 is None:
        s2 = 2.0 * c12.real.astype(dtype, copy=False)
        s3 = -2.0 * c12.imag.astype(dtype, copy=False)
    else:
        s2 = c12.real.astype(dtype, copy=False) + c21.real.astype(dtype, copy=False)
        s3 = c21.imag.astype(dtype, copy=False) - c12.imag.astype(dtype, copy=False)
    return first + second, first - second, s2, s3


def divide_by_power(value: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Normalize without a low-power cutoff; exactly zero power yields zero."""
    dtype = np.result_type(np.float32, value.dtype, power.dtype)
    if cp is not None and isinstance(value, cp.ndarray):
        nonzero = power != 0
        denominator = cp.where(nonzero, power, 1)
        return cast(np.ndarray, cp.where(nonzero, value / denominator, 0).astype(dtype, copy=False))
    return cast(
        np.ndarray, np.divide(value, power, out=np.zeros_like(value, dtype=dtype), where=power != 0)
    )


def stokes_degree_of_polarization(
    s0: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    s3: np.ndarray,
) -> np.ndarray:
    """Stokes norm normalized before squaring, preserving real precision.

    A zero vector returns zero: a numerical undefined-power policy, not a
    physical assertion of unpolarization. Negative-power and NaN inputs
    retain the existing clipped-ratio behavior.
    """
    xp: Any = np
    if cp is not None and isinstance(s0, cp.ndarray):
        xp = cp
    first = divide_by_power(s1, s0)
    second = divide_by_power(s2, s0)
    third = divide_by_power(s3, s0)
    dop = xp.hypot(xp.hypot(first, second), third)
    zero_power_limit = xp.sign(xp.hypot(xp.hypot(s1, s2), s3))
    dop = xp.where(s0 == 0, zero_power_limit, dop)
    return cast(np.ndarray, xp.clip(xp.where(s0 < 0, 0, dop), 0.0, 1.0))


def c2_degree_of_polarization(
    c11: np.ndarray,
    c12: np.ndarray,
    c22: np.ndarray,
    c21: np.ndarray | None = None,
) -> np.ndarray:
    """Barakat degree of polarization for covariance matrix C2.
    C2 purity with stable power normalization and zero-vector DoP = 0.

    Normalize each element by the trace before forming the determinant,
    avoiding squared-power underflow/overflow. An explicit C21 is retained
    rather than replaced by conj(C12). No nonzero-power threshold is used.
    """
    xp: Any = np
    if cp is not None and isinstance(c11, cp.ndarray):
        xp = cp
    if c21 is None:
        c21 = c12.conj()
    dtype = np.result_type(np.float32, c11.dtype, c12.dtype, c21.dtype, c22.dtype)
    c11, c12, c21, c22 = (value.astype(dtype, copy=False) for value in (c11, c12, c21, c22))
    trace = c11 + c22
    first, cross, lower, second = (divide_by_power(value, trace) for value in (c11, c12, c21, c22))
    radicand = (1.0 - 4.0 * (first * second - cross * lower)).real
    zero_matrix = (c11 == 0) & (c12 == 0) & (c21 == 0) & (c22 == 0)
    invalid_zero_trace = (trace == 0) & ~zero_matrix
    if xp.any(invalid_zero_trace):
        with xp.errstate(divide="ignore", invalid="ignore", over="ignore"):
            determinant = (
                c11[invalid_zero_trace] * c22[invalid_zero_trace]
                - c12[invalid_zero_trace] * c21[invalid_zero_trace]
            )
            radicand[invalid_zero_trace] = (
                1.0 - 4.0 * determinant / np.finfo(np.float64).tiny
            ).real
    return cast(np.ndarray, xp.where(zero_matrix, 0.0, xp.sqrt(xp.clip(radicand, 0.0, 1.0))))


def stack_channels(*channels: np.ndarray | None) -> np.ndarray:
    """Stack non-None channels into shape (n, rows, cols)."""
    valid = [c for c in channels if c is not None]
    if not valid:
        return np.array([])
    xp = cp if (_HAS_CUPY and isinstance(valid[0], cp.ndarray)) else np
    return cast(np.ndarray, xp.stack(valid, axis=0))


def canonicalize_channel_cube(
    source: np.ndarray,
    metadata: "ImageMetadata",
) -> tuple[np.ndarray, "ImageMetadata"]:
    """Normalize a labelled channel cube to CYX before matrix construction."""
    axis = channel_axis(source.shape, metadata)
    if axis == 2:
        source = source.transpose(2, 0, 1)
    return source, dataclasses.replace(metadata, axis_order="CYX")


def extract_channels(
    source: np.ndarray,
    metadata: "ImageMetadata",
    indices: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Stack resolved raw channels in canonical co/cross order as CYX."""
    axis = channel_axis(source.shape, metadata)
    cube = source if axis == 0 else source.transpose(2, 0, 1)
    if indices is None:
        indices = resolve_raw_indices(cube.shape[0], metadata)
    return stack_channels(*(cube[index] for index in indices))


def pop_channel_stack(kwargs: dict[str, Any]) -> np.ndarray | None:
    """Pop explicit channel kwargs and return a stacked channel ndarray."""
    explicit = any(
        name in kwargs
        for name in (
            "sxx",
            "shh",
            "co_pol_a",
            "co_pol",
            "sxy",
            "shv",
            "cross_pol",
            "syx",
            "svh",
            "cross_pol_2",
            "syy",
            "svv",
            "co_pol_b",
        )
    )
    sxx = kwargs.pop("sxx", kwargs.pop("shh", kwargs.pop("co_pol_a", kwargs.pop("co_pol", None))))
    sxy = kwargs.pop("sxy", kwargs.pop("shv", kwargs.pop("cross_pol", None)))
    syx = kwargs.pop("syx", kwargs.pop("svh", kwargs.pop("cross_pol_2", None)))
    syy = kwargs.pop("syy", kwargs.pop("svv", kwargs.pop("co_pol_b", None)))
    if not explicit:
        return None
    if sxx is None or sxy is None or (syx is not None and syy is None):
        raise ValueError(
            "Explicit channels require co-pol and cross-pol; a second cross requires co_pol_b."
        )
    if _log.isEnabledFor(logging.INFO):
        _log.info(
            "pol_matrix explicit channels: resolved=%s",
            [
                "co_pol_a<-provided",
                "cross_pol<-provided",
                "cross_pol_2<-provided" if syx is not None else "cross_pol_2<-None",
                "co_pol_b<-provided" if syy is not None else "co_pol_b<-None",
            ],
        )
    return stack_channels(sxx, sxy, syx, syy)


def make_matrix_channel_metadata(
    n: int,
    kind: str,
    source_indices: tuple[int, ...] | None = None,
) -> list:
    """Flat channel descriptors for the N×N matrix (row-major)."""
    from grdl.IO.models.base import ChannelMetadata

    dependencies: list[list[int]] = [[] for _ in range(n)]
    if source_indices is not None:
        indices = list(source_indices)
        if kind == "C":
            dependencies = (
                [[indices[0]], [indices[1]]]
                if n == 2
                else [[indices[0]], indices[1:-1], [indices[-1]]]
            )
        elif kind == "T":
            dependencies = (
                [indices, indices]
                if n == 2
                else [[indices[0], indices[-1]], [indices[0], indices[-1]], indices[1:-1]]
            )
        else:
            dependencies = [indices for _ in range(n)]
    descriptors = []
    idx = 0
    for i in range(n):
        for j in range(n):
            name = f"{kind}[{i},{j}]"
            descriptors.append(
                ChannelMetadata(
                    index=idx,
                    name=name,
                    role="polarimetric_matrix",
                    source_indices=list(dict.fromkeys(dependencies[i] + dependencies[j])),
                )
            )
            idx += 1
    return descriptors


# ---------------------------------------------------------------------------
# CovarianceMatrix
# ---------------------------------------------------------------------------


@processor_version("0.2.0")
@processor_tags(
    modalities=[ImageModality.SAR],
    category=ProcessorCategory.ENHANCE,
    description="Spatially-averaged polarimetric covariance matrix [C]",
)
class CovarianceMatrix(WindowedTileHalo, ImageProcessor):
    """Polarimetric covariance matrix [C2] or [C3].

    Computes the spatially-averaged Hermitian covariance matrix from the
    lexicographic target vector for dual-pol or reciprocal quad-pol SAR data.

    **Dual-pol [C2]** (2×2):
        k = [S_co, S_cross]^T
        C2 = <k * k^H>

    **Quad-pol [C3]** (3×3, reciprocal):
        k = [S_xx, sqrt(2) * S_xy, S_yy]^T
        C3 = <k * k^H>

    The averaging operator ``<.>`` is a square boxcar filter of
    ``window_size × window_size`` pixels applied independently to the
    real and imaginary parts of each matrix element.

    Parameters
    ----------
    window_size : int
        Spatial averaging window side length (pixels, odd, >= 1).
        Default 1 (no spatial averaging).

    Returns
    -------
    np.ndarray
        Complex array, shape ``(N, N, rows, cols)`` (``axis_order='CCYX'``),
        dtype complex64 or complex128 following input precision.
        N=2 for dual-pol, N=3 for quad-pol.

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

    window_size: Annotated[int, Range(min=1, max=63), Desc("Boxcar averaging window size")] = 1

    def __init__(self, window_size: int = 1) -> None:
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(f"window_size must be an odd integer >= 1, got {window_size}")
        self.window_size = window_size

    def compute(self, channels: np.ndarray) -> np.ndarray:
        """Compute the covariance matrix directly from channel arrays.

        Parameters
        ----------
        channels : np.ndarray
            Complex stacked channels, shape ``(n, rows, cols)``, where n is
            2 (dual-pol), 3 (reciprocal with single cross-pol), or 4 (full).

        Returns
        -------
        np.ndarray
            Complex array, shape ``(N, N, rows, cols)``.
        """
        if channels.ndim != 3 or channels.shape[0] not in (2, 3, 4):
            raise ValueError("channels must have shape (n, rows, cols) with n in {2,3,4}.")
        is_gpu = _HAS_CUPY and isinstance(channels, cp.ndarray)
        components = build_target_vector_c(channels)
        return compute_matrix(components, self.window_size, is_gpu)

    def execute(
        self,
        metadata: "ImageMetadata",
        source: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, "ImageMetadata"]:
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

        channels = pop_channel_stack(kwargs)
        indices = None
        if channels is None:
            source, metadata = canonicalize_channel_cube(source, metadata)
            indices = resolve_raw_indices(source.shape[0], metadata)
            channels = extract_channels(source, metadata, indices)
        if channels.size == 0:
            raise ValueError("Channel extraction failed; provide explicit channels")

        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(
                "%s execute: input_channels=%d window_size=%d",
                self.__class__.__name__,
                int(channels.shape[0]),
                self.window_size,
            )

        result = self.compute(channels)
        n = result.shape[0]

        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(
                "%s execute: output_matrix=%dx%d axis_order=CCYX", self.__class__.__name__, n, n
            )

        updated = dataclasses.replace(
            metadata,
            bands=n * n,
            rows=result.shape[2],
            cols=result.shape[3],
            axis_order="CCYX",
            channel_metadata=make_matrix_channel_metadata(n, "C", indices),
            dtype=str(result.dtype),
        )
        return result, updated

    def __repr__(self) -> str:
        return f"CovarianceMatrix(window_size={self.window_size})"


# ---------------------------------------------------------------------------
# CoherencyMatrix
# ---------------------------------------------------------------------------


@processor_version("0.2.0")
@processor_tags(
    modalities=[ImageModality.SAR],
    category=ProcessorCategory.ENHANCE,
    description="Spatially-averaged polarimetric coherency matrix [T]",
)
class CoherencyMatrix(WindowedTileHalo, ImageProcessor):
    """Polarimetric coherency matrix [T2] or [T3].

    Computes the spatially-averaged Hermitian coherency matrix from the
    Pauli target vector for dual-pol or reciprocal quad-pol SAR data.

    **Dual-pol [T2]** (2×2):
        k_P = [(S_co + S_cross), (S_co - S_cross)]^T / sqrt(2)
        T2  = <k_P * k_P^H>

    **Quad-pol [T3]** (3×3, reciprocal):
        k_P = [(S_xx + S_yy), (S_xx - S_yy), 2 * S_xy]^T / sqrt(2)
        T3  = <k_P * k_P^H>

    The Pauli basis is unitary: under monostatic reciprocity ``trace(T3)
    = trace(C3) = span``.  [T3] is the preferred matrix for scattering
    mechanism parameter estimation (entropy, alpha, etc.).

    Parameters
    ----------
    window_size : int
        Spatial averaging window side length (pixels, odd, >= 1).
        Default 1 (no spatial averaging).

    Returns
    -------
    np.ndarray
        Complex array, shape ``(N, N, rows, cols)`` (``axis_order='CCYX'``),
        dtype complex64 or complex128 following input precision.
        N=2 for dual-pol, N=3 for quad-pol.

    Examples
    --------
    >>> from grdl.IO.sar import BIOMASSReader
    >>> from grdl.image_processing.decomposition import CoherencyMatrix
    >>>
    >>> with BIOMASSReader('scene/') as reader:
    ...     cube = reader.read_full()
    ...     tmat = CoherencyMatrix(window_size=9)
    ...     T3, meta = tmat.execute(reader.metadata, cube)
    ...     # T3.diagonal: T3[0,0] = <|sxx+syy|^2>/2 (odd-bounce power)
    ...     #              T3[1,1] = <|sxx-syy|^2>/2 (even-bounce power)
    ...     #              T3[2,2] = 2*<|sxy|^2>     (volume power)
    """

    __gpu_compatible__ = True

    window_size: Annotated[int, Range(min=1, max=63), Desc("Boxcar averaging window size")] = 1

    def __init__(self, window_size: int = 1) -> None:
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(f"window_size must be an odd integer >= 1, got {window_size}")
        self.window_size = window_size

    def compute(self, channels: np.ndarray) -> np.ndarray:
        """Compute the coherency matrix directly from channel arrays.

        Parameters
        ----------
        channels : np.ndarray
            Complex stacked channels, shape ``(n, rows, cols)``, where n is
            2 (dual-pol), 3 (reciprocal with single cross-pol), or 4 (full).

        Returns
        -------
        np.ndarray
            Complex array, shape ``(N, N, rows, cols)``.
        """
        if channels.ndim != 3 or channels.shape[0] not in (2, 3, 4):
            raise ValueError("channels must have shape (n, rows, cols) with n in {2,3,4}.")
        is_gpu = _HAS_CUPY and isinstance(channels, cp.ndarray)
        components = build_target_vector_t(channels)
        return compute_matrix(components, self.window_size, is_gpu)

    def execute(
        self,
        metadata: "ImageMetadata",
        source: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, "ImageMetadata"]:
        """Execute via the universal protocol.

        Channels may be passed as kwargs (``sxx``, ``sxy``, ``syx``,
        ``syy``) or extracted automatically from a CYX/YXC source cube.
        With only 2 channels, [T2] is computed; with 4 channels, [T3].

        Returns
        -------
        tuple[np.ndarray, ImageMetadata]
            ``(T_matrix, updated_metadata)`` where ``T_matrix`` has
            shape ``(N, N, rows, cols)`` and metadata carries
            ``axis_order='CCYX'``.
        """
        self._metadata = metadata

        channels = pop_channel_stack(kwargs)
        indices = None
        if channels is None:
            source, metadata = canonicalize_channel_cube(source, metadata)
            indices = resolve_raw_indices(source.shape[0], metadata)
            channels = extract_channels(source, metadata, indices)
        if channels.size == 0:
            raise ValueError("Channel extraction failed; provide explicit channels")

        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(
                "%s execute: input_channels=%d window_size=%d",
                self.__class__.__name__,
                int(channels.shape[0]),
                self.window_size,
            )

        result = self.compute(channels)
        n = result.shape[0]

        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(
                "%s execute: output_matrix=%dx%d axis_order=CCYX", self.__class__.__name__, n, n
            )

        updated = dataclasses.replace(
            metadata,
            bands=n * n,
            rows=result.shape[2],
            cols=result.shape[3],
            axis_order="CCYX",
            channel_metadata=make_matrix_channel_metadata(n, "T", indices),
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
        [1, 0, 0, 1],
        [1, 0, 0, -1],
        [0, 1, 1, 0],
        [0, -1j, 1j, 0],
    ],
    dtype=np.complex128,
)
_A_MUELLER_INV: np.ndarray = np.conj(_A_MUELLER).T.copy()  # = A^H


# ---------------------------------------------------------------------------
# StokesVector
# ---------------------------------------------------------------------------


@processor_version("0.2.0")
@processor_tags(
    modalities=[ImageModality.SAR],
    category=ProcessorCategory.ENHANCE,
    description="Spatially-averaged Stokes parameters for dual-pol or co-pol SAR channels",
)
class StokesVector(WindowedTileHalo, ImageProcessor):
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
        Boxcar averaging window (pixels, odd, >= 1).  Default ``1``.

    Returns
    -------
    np.ndarray
        Float32/float64 array matching input precision, shape ``(4, rows, cols)``
        (``axis_order='CYX'``).
        Channels ordered [S0, S1, S2, S3].

    Examples
    --------
    >>> sv = StokesVector(window_size=9)
    >>> stokes = sv.compute(sxx, syy)
    >>> dop = sv.degree_of_polarization(stokes)
    """

    __gpu_compatible__ = True

    window_size: Annotated[
        int, Range(min=1, max=63), Desc("Boxcar averaging window size (pixels, odd)")
    ] = 1

    def __init__(self, window_size: int = 1) -> None:
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(f"window_size must be an odd integer >= 1, got {window_size}")
        self.window_size = window_size

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(self, e_x: np.ndarray, e_y: np.ndarray) -> np.ndarray:
        """Compute spatially-averaged Stokes parameters.

        Parameters
        ----------
        e_x : np.ndarray
            Co-pol (or first component) complex field, shape ``(rows, cols)``.
        e_y : np.ndarray
            Cross-pol (or second component) complex field, shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Float32/float64 array matching input precision, shape ``(4, rows, cols)``.
        """
        xp: Any = np
        if cp is not None and isinstance(e_x, cp.ndarray):
            xp = cp
        c11, c12, c22 = compute_dual_elements([e_x, e_y], self.window_size)
        return cast(np.ndarray, xp.stack(stokes_from_c2(c11, c12, c22), axis=0))

    def degree_of_polarization(self, stokes: np.ndarray) -> np.ndarray:
        """Per-pixel degree of polarization in ``[0, 1]``.

        Parameters
        ----------
        stokes : np.ndarray
            Output of :meth:`compute`, shape ``(4, rows, cols)``.

        Returns
        -------
        np.ndarray
            Float32/float64 matching input precision, shape ``(rows, cols)``.
            Zero power returns 0 as a numerical policy for undefined DoP,
            not a physical assertion that an absent signal is unpolarized.
        """
        return stokes_degree_of_polarization(stokes[0], stokes[1], stokes[2], stokes[3])

    # ------------------------------------------------------------------
    # ImageProcessor protocol
    # ------------------------------------------------------------------

    def execute(
        self,
        metadata: "ImageMetadata",
        source: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, "ImageMetadata"]:
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

        explicit = any(name in kwargs for name in ("e_x", "e_y", "e_h", "e_v"))
        e_x = kwargs.pop("e_x", kwargs.pop("e_h", None))
        e_y = kwargs.pop("e_y", kwargs.pop("e_v", None))

        source_idx = []
        if not explicit:
            axis = channel_axis(source.shape, metadata)
            cube = source if axis == 0 else source.transpose(2, 0, 1)
            source_idx = list(resolve_raw_indices(cube.shape[0], metadata, "stokes"))
            e_x, e_y = (cube[index] for index in source_idx)
            if _log.isEnabledFor(logging.INFO):
                _log.info(
                    "%s execute: extracted_channels=%d window_size=%d",
                    self.__class__.__name__,
                    int(cube.shape[0]),
                    self.window_size,
                )
        else:
            if e_x is None or e_y is None:
                raise ValueError("Both e_x and e_y must be provided when bypassing extraction.")
            if _log.isEnabledFor(logging.INFO):
                _log.info(
                    "%s execute: using explicit e_x/e_y window_size=%d",
                    self.__class__.__name__,
                    self.window_size,
                )

        result = self.compute(e_x, e_y)  # (4, rows, cols)
        rows, cols = result.shape[1], result.shape[2]

        from grdl.IO.models.base import ChannelMetadata as _CM

        channel_meta = [
            _CM(index=0, name="S0", role="stokes", source_indices=source_idx),
            _CM(index=1, name="S1", role="stokes", source_indices=source_idx),
            _CM(index=2, name="S2", role="stokes", source_indices=source_idx),
            _CM(index=3, name="S3", role="stokes", source_indices=source_idx),
        ]

        updated = dataclasses.replace(
            metadata,
            rows=rows,
            cols=cols,
            bands=4,
            axis_order="CYX",
            channel_metadata=channel_meta,
            dtype=str(result.dtype),
        )
        return result, updated

    def __repr__(self) -> str:
        return f"StokesVector(window_size={self.window_size})"


# ---------------------------------------------------------------------------
# KennaughMatrix
# ---------------------------------------------------------------------------


@processor_version("0.1.1")
@processor_tags(
    modalities=[ImageModality.SAR],
    category=ProcessorCategory.ENHANCE,
    description="Spatially-averaged polarimetric Kennaugh matrix [K] (4×4 real)",
)
class KennaughMatrix(WindowedTileHalo, ImageProcessor):
    """Polarimetric Kennaugh matrix [K] (4×4 real symmetric).

    The Kennaugh matrix is a 4×4 real matrix that fully characterises the
    polarimetric scattering behaviour of a spatially-averaged target and is
    the Mueller-calculus analogue of the coherency matrix.

    Derivation::

        C4  = < [S_xx, S_xy, S_yx, S_yy]^T [...]^H >   4×4 lex. covariance
        M   =  A · C4 · A^H                          Mueller matrix
        K   =  (1/2) · Re(M)                         Kennaugh matrix

    where::

        A = (1/√2) · [[1,  0,  0,  1],
                      [1,  0,  0, -1],
                      [0,  1,  1,  0],
                      [0, -j,  j,  0]]

    and A is unitary (A · A^H = I).

    Under monostatic reciprocity (S_xy ≈ S_yx) the 4th row/column of K is
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
        Boxcar averaging window (pixels, odd, >= 1).  Default ``1``.

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

    __gpu_compatible__ = False  # np.einsum complex path; cupy not yet verified

    window_size: Annotated[
        int, Range(min=1, max=63), Desc("Boxcar averaging window size (pixels, odd)")
    ] = 1

    def __init__(self, window_size: int = 1) -> None:
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(f"window_size must be an odd integer >= 1, got {window_size}")
        self.window_size = window_size

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(self, channels: np.ndarray) -> np.ndarray:
        """Compute the spatially-averaged Kennaugh matrix.

        Parameters
        ----------
        channels : np.ndarray
            Complex stacked channels, shape ``(n, rows, cols)`` with n in
            {3,4}. For n=3, channels are interpreted as [co, cross, co]
            and reciprocity is assumed.

        Returns
        -------
        np.ndarray
            Float32, shape ``(4, 4, rows, cols)``.
        """
        if channels.ndim != 3 or channels.shape[0] not in (3, 4):
            raise ValueError("channels must have shape (n, rows, cols) with n in {3,4}.")

        is_gpu = _HAS_CUPY and isinstance(channels, cp.ndarray)
        if channels.shape[0] == 3:
            c4_channels = [channels[0], channels[1], channels[1], channels[2]]
        else:
            c4_channels = [channels[0], channels[1], channels[2], channels[3]]

        # Build 4×4 lexicographic covariance C4 (from 4 channels, no √2 rescaling)
        C4 = compute_matrix(c4_channels, self.window_size, is_gpu)

        # Promote to complex128 for numerical precision in the basis change
        C4_d = C4.astype(np.complex128)

        # Mueller matrix: M = A · C4 · A^H
        # einsum 'ik,klyx,lj->ijyx' computes per-pixel M[i,j] = Σ_k Σ_l A[i,k]*C4[k,l]*A^H[l,j]
        M = np.einsum("ik,klyx,lj->ijyx", _A_MUELLER, C4_d, _A_MUELLER_INV)

        # Kennaugh = (1/2) Re(M)
        return cast(np.ndarray, (0.5 * np.real(M)).astype(np.float32))

    # ------------------------------------------------------------------
    # ImageProcessor protocol
    # ------------------------------------------------------------------

    def execute(
        self,
        metadata: "ImageMetadata",
        source: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, "ImageMetadata"]:
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

        channels = pop_channel_stack(kwargs)
        indices = None
        if channels is None:
            source, metadata = canonicalize_channel_cube(source, metadata)
            indices = resolve_raw_indices(source.shape[0], metadata)
            channels = extract_channels(source, metadata, indices)
        if channels.size == 0:
            raise ValueError("Channel extraction failed; provide explicit channels")

        if _log.isEnabledFor(logging.INFO):
            _log.info(
                "%s execute: input_channels=%d window_size=%d",
                self.__class__.__name__,
                int(channels.shape[0]),
                self.window_size,
            )

        result = self.compute(channels)  # (4, 4, rows, cols)
        rows, cols = result.shape[2], result.shape[3]

        if _log.isEnabledFor(logging.INFO):
            _log.info("%s execute: output_matrix=4x4 axis_order=KKYX", self.__class__.__name__)

        updated = dataclasses.replace(
            metadata,
            rows=rows,
            cols=cols,
            bands=16,
            axis_order="KKYX",
            channel_metadata=make_matrix_channel_metadata(4, "K", indices),
            dtype="float32",
        )
        return result, updated

    def __repr__(self) -> str:
        return f"KennaughMatrix(window_size={self.window_size})"
