# -*- coding: utf-8 -*-
"""
Polarimetric Matrix Products - Spatially-averaged covariance and coherency
matrices for multi-polarization SAR data.

The implementation is basis-agnostic for orthogonal polarization pairs
(for example H/V, X/Y, or R/L) while preserving co-pol vs cross-pol role.

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


_log = logging.getLogger(__name__)


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


def _build_target_vector_c(channels: np.ndarray) -> list:
    """Lexicographic target vector components.

    Dual-pol:  [s_co, s_cross]
    Quad-pol:  [sxx, sqrt(2)*mean(sxy,syx), syy]  (reciprocal)
    """
    n = channels.shape[0]
    if n == 2:
        return [channels[0], channels[1]]
    xp = cp if (_HAS_CUPY and isinstance(channels, cp.ndarray)) else np
    if n == 3:
        cross = channels[1]
        return [channels[0], xp.float32(np.sqrt(2.0)) * cross, channels[2]]
    cross = (channels[1] + channels[2]) * xp.float32(0.5)
    norm = xp.float32(np.sqrt(2.0))
    return [channels[0], norm * cross, channels[3]]


def _build_target_vector_t(channels: np.ndarray) -> list:
    """Pauli target vector components (1/sqrt(2) factor included).

    Dual-pol:  [(s_co + s_cross), (s_co - s_cross)] / sqrt(2)
    Quad-pol:  [(sxx + syy), (sxx - syy), 2*mean(sxy,syx)] / sqrt(2)
    """
    xp = cp if (_HAS_CUPY and isinstance(channels, cp.ndarray)) else np
    norm = xp.float32(1.0 / np.sqrt(2.0))
    n = channels.shape[0]
    if n == 2:
        return [
            (channels[0] + channels[1]) * norm,
            (channels[0] - channels[1]) * norm,
        ]
    if n == 3:
        cross = channels[1]
        two_cross = cross * xp.float32(2.0)
        return [
            (channels[0] + channels[2]) * norm,
            (channels[0] - channels[2]) * norm,
            two_cross * norm,
        ]
    cross = (channels[1] + channels[2]) * xp.float32(0.5)
    two_cross = cross * xp.float32(2.0)
    return [
        (channels[0] + channels[3]) * norm,
        (channels[0] - channels[3]) * norm,
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


def _stack_channels(*channels: Optional[np.ndarray]) -> np.ndarray:
    """Stack non-None channels into shape (n, rows, cols)."""
    valid = [c for c in channels if c is not None]
    if not valid:
        return np.array([])
    xp = cp if (_HAS_CUPY and isinstance(valid[0], cp.ndarray)) else np
    return xp.stack(valid, axis=0)


def _extract_channels(
    source: np.ndarray,
    metadata: 'ImageMetadata',
) -> np.ndarray:
    """Extract polarimetric channels as a stacked ndarray.

    Returns a stacked ndarray of shape (n, rows, cols) where n is the number
    of channels extracted (2, 3, or 4 for dual, reciprocal, or full quad).
    Channel order: [co_pol_a, cross_pol, cross_pol_2, co_pol_b].

    Preferred extraction uses channel metadata roles/names so the method remains
    agnostic to specific orthogonal bases (H/V, X/Y, R/L). If channel names
    are unavailable or ambiguous, index-based fallback is used.
    """
    if source.ndim != 3:
        return np.array([])  # Empty array signals extraction failure

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
        return np.array([])  # Empty array signals extraction failure

    def _input_descriptors() -> list:
        desc: list = []
        for i in range(n):
            if channel_metadata and len(channel_metadata) > i:
                cm = channel_metadata[i]
                name = getattr(cm, 'name', None)
                pol = getattr(cm, 'polarization', None)
                tx_pol = getattr(cm, 'tx_polarization', None)
                rx_pol = getattr(cm, 'rcv_polarization', None)
                role = getattr(cm, 'role', None)
                parts = [f'idx={i}']
                if name:
                    parts.append(f'name={name}')
                if pol:
                    parts.append(f'pol={pol}')
                elif tx_pol and rx_pol:
                    parts.append(f'pol={tx_pol}{rx_pol}')
                if role:
                    parts.append(f'role={role}')
                desc.append(' '.join(parts))
            else:
                desc.append(f'idx={i}')
        return desc

    def _log_resolved(strategy: str, resolved: list) -> None:
        if not _log.isEnabledFor(logging.INFO):
            return
        _log.info(
            'pol_matrix channel extraction: strategy=%s axis_order=%s n=%d input=%s resolved=%s',
            strategy,
            axis_order,
            n,
            _input_descriptors(),
            resolved,
        )

    def ch(i):
        return source[i] if chs_first else source[..., i]

    def _norm_role(role: Optional[str]) -> str:
        if not role:
            return ''
        return ''.join(c for c in role.lower() if c.isalnum() or c == '_')

    def _role_kind(role: Optional[str]) -> Optional[str]:
        r = _norm_role(role)
        if not r:
            return None
        if r in {'copol', 'co_pol', 'parallelpol', 'parallel_pol'}:
            return 'co'
        if r in {'crosspol', 'cross_pol', 'xpol', 'x_pol'}:
            return 'cross'
        return None

    def _norm_token(name: str) -> Optional[str]:
        if not name:
            return None
        up = ''.join(c for c in name.upper() if c.isalnum())
        for tok in (
            'HH', 'HV', 'VH', 'VV',
            'XX', 'XY', 'YX', 'YY',
            'RR', 'RL', 'LR', 'LL',
        ):
            if tok in up:
                return tok
        return None

    def _is_co(token: str) -> bool:
        return len(token) == 2 and token[0] == token[1]

    def _is_cross(token: str) -> bool:
        return len(token) == 2 and token[0] != token[1]

    if channel_metadata and len(channel_metadata) >= n:
        role_co: list = []
        role_cross: list = []

        for i in range(n):
            cm = channel_metadata[i]
            extras = getattr(cm, 'extras', None) or {}
            role_candidates = [
                getattr(cm, 'role', None),
                extras.get('polarimetric_role'),
                extras.get('pol_role'),
                extras.get('role'),
            ]
            kind = next((_role_kind(r) for r in role_candidates if _role_kind(r)), None)
            if kind == 'co':
                role_co.append((i, ch(i)))
            elif kind == 'cross':
                role_cross.append((i, ch(i)))

        if len(role_co) >= 2 and len(role_cross) >= 1:
            c0_idx, c0 = role_co[0]
            c1_idx, c1 = role_co[1]
            x0_idx, x0 = role_cross[0]
            x1_idx, x1 = role_cross[1] if len(role_cross) > 1 else (None, None)
            _log_resolved(
                'metadata_role',
                [
                    f'co_pol_a<-idx={c0_idx}',
                    f'cross_pol<-idx={x0_idx}',
                    f'cross_pol_2<-idx={x1_idx}' if x1 is not None else 'cross_pol_2<-None',
                    f'co_pol_b<-idx={c1_idx}',
                ],
            )
            return _stack_channels(c0, x0, x1, c1)
        if len(role_co) >= 1 and len(role_cross) >= 1:
            c0_idx, c0 = role_co[0]
            x0_idx, x0 = role_cross[0]
            x1_idx, x1 = role_cross[1] if len(role_cross) > 1 else (None, None)
            _log_resolved(
                'metadata_role',
                [
                    f'co_pol_a<-idx={c0_idx}',
                    f'cross_pol<-idx={x0_idx}',
                    f'cross_pol_2<-idx={x1_idx}' if x1 is not None else 'cross_pol_2<-None',
                ],
            )
            return _stack_channels(c0, x0, x1)

        by_token: Dict[str, Tuple[int, np.ndarray]] = {}
        for i in range(n):
            cm = channel_metadata[i]
            name = getattr(cm, 'name', None)
            pol = getattr(cm, 'polarization', None)
            tx_pol = getattr(cm, 'tx_polarization', None)
            rx_pol = getattr(cm, 'rcv_polarization', None)
            tok = (
                _norm_token(name or '')
                or _norm_token(pol or '')
                or _norm_token(f'{tx_pol}{rx_pol}' if tx_pol and rx_pol else '')
            )
            if tok and tok not in by_token:
                by_token[tok] = (i, ch(i))

        # Prefer canonical orthogonal bases with explicit co/cross roles.
        for co_a, co_b, cross_pref in (
            ('HH', 'VV', ('HV', 'VH')),
            ('XX', 'YY', ('XY', 'YX')),
            ('RR', 'LL', ('RL', 'LR')),
        ):
            if co_a in by_token and co_b in by_token:
                x1 = next((by_token[t] for t in cross_pref if t in by_token), None)
                x2 = next((by_token[t] for t in cross_pref[::-1] if t in by_token), None)
                if x1 is not None:
                    if x2 is x1:
                        x2 = None
                    c0_idx, c0 = by_token[co_a]
                    c1_idx, c1 = by_token[co_b]
                    x0_idx, x0 = x1
                    if x2 is not None:
                        x1_idx, x1_arr = x2
                    else:
                        x1_idx, x1_arr = (None, None)
                    _log_resolved(
                        'token_canonical',
                        [
                            f'co_pol_a({co_a})<-idx={c0_idx}',
                            f'cross_pol<-idx={x0_idx}',
                            f'cross_pol_2<-idx={x1_idx}' if x1_arr is not None else 'cross_pol_2<-None',
                            f'co_pol_b({co_b})<-idx={c1_idx}',
                        ],
                    )
                    return _stack_channels(c0, x0, x1_arr, c1)

        co_tokens = [t for t in by_token if _is_co(t)]
        cross_tokens = [t for t in by_token if _is_cross(t)]
        co_tokens.sort()
        cross_tokens.sort()
        if len(co_tokens) >= 2 and len(cross_tokens) >= 1:
            c0_idx, c0 = by_token[co_tokens[0]]
            c1_idx, c1 = by_token[co_tokens[1]]
            x0_idx, x0 = by_token[cross_tokens[0]]
            if len(cross_tokens) > 1:
                x1_idx, x1 = by_token[cross_tokens[1]]
            else:
                x1_idx, x1 = (None, None)
            _log_resolved(
                'token_generic',
                [
                    f'co_pol_a({co_tokens[0]})<-idx={c0_idx}',
                    f'cross_pol({cross_tokens[0]})<-idx={x0_idx}',
                    f'cross_pol_2({cross_tokens[1]})<-idx={x1_idx}' if x1 is not None else 'cross_pol_2<-None',
                    f'co_pol_b({co_tokens[1]})<-idx={c1_idx}',
                ],
            )
            return _stack_channels(c0, x0, x1, c1)
        if len(co_tokens) >= 1 and len(cross_tokens) >= 1:
            c0_idx, c0 = by_token[co_tokens[0]]
            x0_idx, x0 = by_token[cross_tokens[0]]
            if len(cross_tokens) > 1:
                x1_idx, x1 = by_token[cross_tokens[1]]
            else:
                x1_idx, x1 = (None, None)
            _log_resolved(
                'token_generic',
                [
                    f'co_pol_a({co_tokens[0]})<-idx={c0_idx}',
                    f'cross_pol({cross_tokens[0]})<-idx={x0_idx}',
                    f'cross_pol_2({cross_tokens[1]})<-idx={x1_idx}' if x1 is not None else 'cross_pol_2<-None',
                ],
            )
            return _stack_channels(c0, x0, x1)

    # Index-based fallback
    if n >= 4:
        _log_resolved(
            'index_fallback',
            ['co_pol_a<-idx=0', 'cross_pol<-idx=1', 'cross_pol_2<-idx=2', 'co_pol_b<-idx=3'],
        )
        return _stack_channels(ch(0), ch(1), ch(2), ch(3))
    if n == 3:
        # Common cube layout for reciprocal quad-pol products:
        # [co_pol_a, cross_pol, co_pol_b] with one cross-pol channel.
        _log_resolved(
            'index_fallback',
            ['co_pol_a<-idx=0', 'cross_pol<-idx=1', 'co_pol_b<-idx=2'],
        )
        return _stack_channels(ch(0), ch(1), ch(2))
    if n == 2:
        _log_resolved(
            'index_fallback',
            ['co_pol_a<-idx=0', 'cross_pol<-idx=1'],
        )
        return _stack_channels(ch(0), ch(1))
    return np.array([])


def _pop_channel_stack(kwargs: Dict[str, Any]) -> Optional[np.ndarray]:
    """Pop explicit channel kwargs and return a stacked channel ndarray."""
    sxx = kwargs.pop('sxx', kwargs.pop('shh', kwargs.pop('co_pol_a', kwargs.pop('co_pol', None))))
    if sxx is None:
        return None
    sxy = kwargs.pop('sxy', kwargs.pop('shv', kwargs.pop('cross_pol', None)))
    syx = kwargs.pop('syx', kwargs.pop('svh', kwargs.pop('cross_pol_2', None)))
    syy = kwargs.pop('syy', kwargs.pop('svv', kwargs.pop('co_pol_b', None)))
    if sxy is None:
        raise ValueError('At least one co-pol and one cross-pol channel are required.')
    if _log.isEnabledFor(logging.INFO):
        _log.info(
            'pol_matrix explicit channels: resolved=%s',
            [
                'co_pol_a<-provided',
                'cross_pol<-provided',
                'cross_pol_2<-provided' if syx is not None else 'cross_pol_2<-None',
                'co_pol_b<-provided' if syy is not None else 'co_pol_b<-None',
            ],
        )
    return _stack_channels(sxx, sxy, syx, syy)


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
        int, Range(min=1, max=63), Desc('Boxcar averaging window size')
    ] = 1

    def __init__(self, window_size: int = 1) -> None:
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be an odd integer >= 1, got {window_size}"
            )
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
            raise ValueError(
                'channels must have shape (n, rows, cols) with n in {2,3,4}.'
            )
        is_gpu = _HAS_CUPY and isinstance(channels, cp.ndarray)
        components = _build_target_vector_c(channels)
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

        channels = _pop_channel_stack(kwargs)
        if channels is None:
            channels = _extract_channels(source, metadata)
        if channels.size == 0:
            raise ValueError('Channel extraction failed; provide explicit channels')

        if _log.isEnabledFor(logging.INFO):
            _log.info(
                '%s execute: input_channels=%d window_size=%d',
                self.__class__.__name__,
                int(channels.shape[0]),
                self.window_size,
            )

        result = self.compute(channels)
        n = result.shape[0]

        if _log.isEnabledFor(logging.INFO):
            _log.info('%s execute: output_matrix=%dx%d axis_order=CCYX', self.__class__.__name__, n, n)

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
    ...     # T3.diagonal: T3[0,0] = <|sxx+syy|^2>/2 (odd-bounce power)
    ...     #              T3[1,1] = <|sxx-syy|^2>/2 (even-bounce power)
    ...     #              T3[2,2] = 2*<|sxy|^2>     (volume power)
    """

    __gpu_compatible__ = True

    window_size: Annotated[
        int, Range(min=1, max=63), Desc('Boxcar averaging window size')
    ] = 1

    def __init__(self, window_size: int = 1) -> None:
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be an odd integer >= 1, got {window_size}"
            )
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
            raise ValueError(
                'channels must have shape (n, rows, cols) with n in {2,3,4}.'
            )
        is_gpu = _HAS_CUPY and isinstance(channels, cp.ndarray)
        components = _build_target_vector_t(channels)
        return _compute_matrix(components, self.window_size, is_gpu)

    def execute(
        self,
        metadata: 'ImageMetadata',
        source: np.ndarray,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
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

        channels = _pop_channel_stack(kwargs)
        if channels is None:
            channels = _extract_channels(source, metadata)
        if channels.size == 0:
            raise ValueError('Channel extraction failed; provide explicit channels')

        if _log.isEnabledFor(logging.INFO):
            _log.info(
                '%s execute: input_channels=%d window_size=%d',
                self.__class__.__name__,
                int(channels.shape[0]),
                self.window_size,
            )

        result = self.compute(channels)
        n = result.shape[0]

        if _log.isEnabledFor(logging.INFO):
            _log.info('%s execute: output_matrix=%dx%d axis_order=CCYX', self.__class__.__name__, n, n)

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
        Boxcar averaging window (pixels, odd, >= 1).  Default ``1``.

    Returns
    -------
    np.ndarray
        Float32 array, shape ``(4, rows, cols)`` (``axis_order='CYX'``).
        Channels ordered [S0, S1, S2, S3].

    Examples
    --------
    >>> sv = StokesVector(window_size=9)
    >>> stokes = sv.compute(sxx, syy)
    >>> dop = sv.degree_of_polarization(stokes)
    """

    __gpu_compatible__ = True

    window_size: Annotated[
        int, Range(min=1, max=63), Desc('Boxcar averaging window size (pixels, odd)')
    ] = 1

    def __init__(self, window_size: int = 1) -> None:
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be an odd integer >= 1, got {window_size}"
            )
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
            Float32 array, shape ``(4, rows, cols)``.
        """
        is_gpu = _HAS_CUPY and isinstance(e_x, cp.ndarray)
        xp = cp if is_gpu else np
        ws = self.window_size

        cross = e_x * xp.conj(e_y)
        S0 = _smooth(xp.abs(e_x) ** 2 + xp.abs(e_y) ** 2,      ws, is_gpu)
        S1 = _smooth(xp.abs(e_x) ** 2 - xp.abs(e_y) ** 2,      ws, is_gpu)
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

        e_x = kwargs.pop('e_x', kwargs.pop('e_h', None))
        e_y = kwargs.pop('e_y', kwargs.pop('e_v', None))

        if e_x is None:
            extracted = _extract_channels(source, metadata)
            if extracted.size == 0:
                raise ValueError('Channel extraction failed; provide explicit e_x/e_y')
            # Unstack: extracted has shape (n, rows, cols)
            n = extracted.shape[0]
            if n >= 4:
                e_x, e_y = extracted[0], extracted[3]  # quad-pol: co-pol pair
            elif n == 3:
                e_x, e_y = extracted[0], extracted[2]  # reciprocal: co-pol pair
            elif n == 2:
                e_x, e_y = extracted[0], extracted[1]  # dual-pol
            else:
                raise ValueError(f'Invalid number of channels extracted: {n}')
            if _log.isEnabledFor(logging.INFO):
                _log.info(
                    '%s execute: extracted_channels=%d window_size=%d',
                    self.__class__.__name__,
                    int(n),
                    self.window_size,
                )
        else:
            e_x = kwargs.pop('e_x', kwargs.pop('e_h', None))
            e_y = kwargs.pop('e_y', kwargs.pop('e_v', None))
            if _log.isEnabledFor(logging.INFO):
                _log.info('%s execute: using explicit e_x/e_y window_size=%d', self.__class__.__name__, self.window_size)

        result = self.compute(e_x, e_y)  # (4, rows, cols)
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

    __gpu_compatible__ = False   # np.einsum complex path; cupy not yet verified

    window_size: Annotated[
        int, Range(min=1, max=63), Desc('Boxcar averaging window size (pixels, odd)')
    ] = 1

    def __init__(self, window_size: int = 1) -> None:
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be an odd integer >= 1, got {window_size}"
            )
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
            raise ValueError(
                'channels must have shape (n, rows, cols) with n in {3,4}.'
            )

        is_gpu = _HAS_CUPY and isinstance(channels, cp.ndarray)
        if channels.shape[0] == 3:
            c4_channels = [channels[0], channels[1], channels[1], channels[2]]
        else:
            c4_channels = [channels[0], channels[1], channels[2], channels[3]]

        # Build 4×4 lexicographic covariance C4 (from 4 channels, no √2 rescaling)
        C4 = _compute_matrix(c4_channels, self.window_size, is_gpu)
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

        channels = _pop_channel_stack(kwargs)
        if channels is None:
            channels = _extract_channels(source, metadata)
        if channels.size == 0:
            raise ValueError('Channel extraction failed; provide explicit channels')

        if _log.isEnabledFor(logging.INFO):
            _log.info(
                '%s execute: input_channels=%d window_size=%d',
                self.__class__.__name__,
                int(channels.shape[0]),
                self.window_size,
            )

        result = self.compute(channels)  # (4, 4, rows, cols)
        rows, cols = result.shape[2], result.shape[3]

        if _log.isEnabledFor(logging.INFO):
            _log.info('%s execute: output_matrix=4x4 axis_order=KKYX', self.__class__.__name__)

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
