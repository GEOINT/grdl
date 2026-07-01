# -*- coding: utf-8 -*-
"""
Full-Pol H/A/Alpha Decomposition - Cloude-Pottier eigenvector/eigenvalue
decomposition of the 3x3 coherency matrix [T3] for quad-pol SAR data.

Given quad-pol (HH, HV, VH, VV) complex SLC channels, builds the spatially
averaged 3x3 coherency matrix [T3] and performs eigendecomposition to
extract entropy (H), mean alpha angle, anisotropy (A), span, and the three
eigenvalues.

References
----------
Cloude, S.R. and Pottier, E. (1997), "An entropy based classification scheme
for land applications of polarimetric SAR," IEEE Trans. Geoscience and Remote
Sensing, 35(1), pp.68-78.

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
2026-06-25

Modified
--------
2026-06-25
"""

# Standard library
from typing import Annotated, Any, Dict, Tuple, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.decomposition.h_a_alpha_base import HAalphaBase
from grdl.image_processing.decomposition.pol_matrix import CoherencyMatrix
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.image_processing.params import Range, Desc
from grdl.vocabulary import ImageModality

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata


@processor_version('1.0.0')
@processor_tags(modalities=[ImageModality.SAR])
class FullPolHAalpha(HAalphaBase):
    """Full-pol H/A/Alpha (Cloude-Pottier) eigenvalue decomposition.

    Decomposes quad-pol SAR data into physically meaningful parameters
    via eigenvalue analysis of the spatially-averaged 3x3 coherency
    matrix [T3]:

    - **entropy** (H): Randomness of the scattering process [0, 1].
      0 = single deterministic mechanism, 1 = fully random.

    - **alpha**: Mean scattering angle [0, 90] degrees.
      0 = surface, 45 = dipole/volume, 90 = dihedral.

    - **anisotropy** (A): Relative strength of the second vs third
      eigenvalues [0, 1].  1 = two mechanisms dominate equally,
      0 = second and third contribute equally.

    - **span**: Total backscattered power (sum of eigenvalues).

    - **lambda_1, lambda_2, lambda_3**: Individual eigenvalues
      (descending order).

    Parameters
    ----------
    window_size : int
        Side length of the square boxcar averaging window used to
        estimate the coherency matrix [T3].  Must be odd and >= 3.
        Default 7.  Ignored when using ``decompose_from_t3()``.

    Examples
    --------
    From scattering matrix channels:

    >>> from grdl.image_processing.decomposition import FullPolHAalpha
    >>> decomp = FullPolHAalpha(window_size=9)
    >>> components = decomp.decompose(shh, shv, svh, svv)
    >>> print(components['entropy'].shape)  # same as input

    From a pre-computed (possibly filtered) coherency matrix:

    >>> from grdl.image_processing.decomposition import CoherencyMatrix
    >>> t3 = CoherencyMatrix(window_size=1).compute(channels)
    >>> # ... apply speckle filter to t3 ...
    >>> components = decomp.decompose_from_t3(t3)
    >>> rgb, meta = decomp.to_rgb(components)
    """

    window_size: Annotated[int, Range(min=1, max=31),
                           Desc('Boxcar averaging window size')] = 7

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def component_names(self) -> Tuple[str, ...]:
        """Names of decomposition output components."""
        return ('entropy', 'alpha', 'anisotropy', 'span',
                'lambda_1', 'lambda_2', 'lambda_3')

    def decompose(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: np.ndarray,
        svv: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Decompose quad-pol data into H/A/Alpha parameters.

        Parameters
        ----------
        shh : np.ndarray
            Complex S_HH channel. Shape ``(rows, cols)``.
        shv : np.ndarray
            Complex S_HV channel. Shape ``(rows, cols)``.
        svh : np.ndarray
            Complex S_VH channel. Shape ``(rows, cols)``.
        svv : np.ndarray
            Complex S_VV channel. Shape ``(rows, cols)``.

        Returns
        -------
        Dict[str, np.ndarray]
            Keys: ``'entropy'``, ``'alpha'``, ``'anisotropy'``,
            ``'span'``, ``'lambda_1'``, ``'lambda_2'``, ``'lambda_3'``.
            All real-valued float64 arrays with same spatial shape.
        """
        self._validate_scattering_matrix(shh, shv, svh, svv)
        self._validate_internal_matrix_window_size('decompose_from_t3')

        # -- 1. Build spatially-averaged [T3] coherency matrix --
        # CoherencyMatrix expects CYX (4, rows, cols) input
        from grdl.IO.models.base import ImageMetadata, ChannelMetadata
        rows, cols = shh.shape
        cube = np.stack([shh, shv, svh, svv], axis=0)
        meta = ImageMetadata(
            format='INTERNAL',
            rows=rows,
            cols=cols,
            bands=4,
            dtype='complex64',
            axis_order='CYX',
            channel_metadata=[
                ChannelMetadata(index=0, name='HH', polarization='HH'),
                ChannelMetadata(index=1, name='HV', polarization='HV'),
                ChannelMetadata(index=2, name='VH', polarization='VH'),
                ChannelMetadata(index=3, name='VV', polarization='VV'),
            ],
        )
        coh = CoherencyMatrix(window_size=self.window_size)
        # Returns (3, 3, rows, cols) complex
        t3_ccyx, _ = coh.execute(meta, cube)

        return self.decompose_from_t3(t3_ccyx)

    def decompose_from_t3(
        self,
        t3: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Decompose a pre-computed coherency matrix [T3].

        Use this when you want to apply a different speckle filter
        (e.g. Refined Lee, NL-SAR) to [T3] before decomposition.

        Parameters
        ----------
        t3 : np.ndarray
            Coherency matrix, shape ``(3, 3, rows, cols)``, complex.
            Must be Hermitian (``t3[i,j] == conj(t3[j,i])``).

        Returns
        -------
        Dict[str, np.ndarray]
            Keys: ``'entropy'``, ``'alpha'``, ``'anisotropy'``,
            ``'span'``, ``'lambda_1'``, ``'lambda_2'``, ``'lambda_3'``.
            All real-valued float64 arrays with same spatial shape.
        """
        if t3.ndim != 4 or t3.shape[0] != 3 or t3.shape[1] != 3:
            raise ValueError(
                f"Expected t3 shape (3, 3, rows, cols), got {t3.shape}"
            )

        rows, cols = t3.shape[2], t3.shape[3]

        # -- 2. Reshape to (rows*cols, 3, 3) for batch eigendecomposition --
        t3_yx = np.transpose(t3, (2, 3, 0, 1))  # (rows, cols, 3, 3)
        t3_flat = t3_yx.reshape(-1, 3, 3)

        # -- 3. Hermitian eigendecomposition (eigh guarantees real sorted) --
        eigenvalues, eigenvectors = np.linalg.eigh(t3_flat)
        # eigh returns ascending order; reverse to descending
        eigenvalues = eigenvalues[..., ::-1]
        eigenvectors = eigenvectors[..., ::-1]

        # Clamp any tiny negative values from numerical noise
        np.maximum(eigenvalues, 0.0, out=eigenvalues)

        # -- 4. Reshape back to spatial --
        eigenvalues = eigenvalues.reshape(rows, cols, 3)
        eigenvectors = eigenvectors.reshape(rows, cols, 3, 3)

        # -- 5. Compute H/A/Alpha --
        p = self._normalize_eigenvalues(eigenvalues)
        entropy = self._entropy(p)

        # Alpha: cos(alpha_i) = |e_i[0]| (first component of each eigenvector)
        # eigenvectors shape: (rows, cols, 3, 3) — last axis is eigenvector
        # components, second-to-last is which eigenvector
        cos_alpha = np.abs(eigenvectors[..., 0])  # (rows, cols, 3)
        alpha = self._alpha_mean(p, cos_alpha)

        # Anisotropy: A = (lambda_2 - lambda_3) / (lambda_2 + lambda_3)
        lam2 = eigenvalues[..., 1]
        lam3 = eigenvalues[..., 2]
        denom = lam2 + lam3
        with np.errstate(divide='ignore', invalid='ignore'):
            anisotropy = np.where(denom > 0.0, (lam2 - lam3) / denom, 0.0)
        np.clip(anisotropy, 0.0, 1.0, out=anisotropy)

        # Span = sum of eigenvalues
        span = eigenvalues.sum(axis=-1)

        return {
            'entropy': entropy,
            'alpha': alpha,
            'anisotropy': anisotropy,
            'span': span,
            'lambda_1': eigenvalues[..., 0],
            'lambda_2': eigenvalues[..., 1],
            'lambda_3': eigenvalues[..., 2],
        }

    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        """Create an RGB composite from full-pol H/A/Alpha decomposition.

        Channel mapping:

        - **Red**: Entropy [0, 1]
        - **Green**: Alpha / 90 [0, 1]
        - **Blue**: Anisotropy [0, 1]

        Parameters
        ----------
        components : Dict[str, np.ndarray]
            Output of ``decompose()`` or ``decompose_from_t3()``.
        representation : str
            Ignored (components are real-valued).
        percentile_low : float
            Unused for full-pol (all channels already in [0, 1]).
        percentile_high : float
            Unused for full-pol (all channels already in [0, 1]).

        Returns
        -------
        tuple[np.ndarray, ImageMetadata]
            ``(rgb, metadata)`` — rgb is shape ``(3, rows, cols)``, dtype
            float32, values in [0, 1].
        """
        from grdl.IO.models.base import ImageMetadata, ChannelMetadata

        required = {'entropy', 'alpha', 'anisotropy'}
        missing = required - set(components.keys())
        if missing:
            raise ValueError(
                f"Missing component keys: {missing}. "
                f"Expected keys from decompose(): {required}"
            )

        # Red: Entropy (already [0, 1])
        r = np.clip(components['entropy'], 0.0, 1.0).astype(np.float32)

        # Green: Alpha / 90 (normalized to [0, 1])
        g = np.clip(components['alpha'] / 90.0, 0.0, 1.0).astype(np.float32)

        # Blue: Anisotropy (already [0, 1])
        b = np.clip(components['anisotropy'], 0.0, 1.0).astype(np.float32)

        rgb = np.stack([r, g, b], axis=0)
        metadata = ImageMetadata(
            format='HAalphaRGB',
            rows=int(rgb.shape[1]),
            cols=int(rgb.shape[2]),
            dtype=str(rgb.dtype),
            bands=3,
            axis_order='CYX',
            channel_metadata=[
                ChannelMetadata(
                    index=0, name='entropy', role='decomposition',
                    extras={'halpha_component': 'entropy',
                            'formula': 'H in [0, 1]',
                            'display': 'Red'},
                ),
                ChannelMetadata(
                    index=1, name='alpha_norm', role='decomposition',
                    extras={'halpha_component': 'alpha',
                            'formula': 'alpha / 90 in [0, 1]',
                            'display': 'Green'},
                ),
                ChannelMetadata(
                    index=2, name='anisotropy', role='decomposition',
                    extras={'halpha_component': 'anisotropy',
                            'formula': 'A in [0, 1]',
                            'display': 'Blue'},
                ),
            ],
        )
        return rgb, metadata

    def __repr__(self) -> str:
        return f"FullPolHAalpha(window_size={self.window_size})"
