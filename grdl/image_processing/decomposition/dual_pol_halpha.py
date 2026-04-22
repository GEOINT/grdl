# -*- coding: utf-8 -*-
"""
Dual-Pol H/Alpha Decomposition - Eigenvalue decomposition of the 2x2
coherency matrix for dual-polarization SAR data.

Given co-pol (e.g. S_VV) and cross-pol (e.g. S_VH) complex SLC channels,
forms the spatially averaged 2x2 coherency matrix and decomposes it via
closed-form eigenvalue analysis.  Outputs are the Cloude-Pottier entropy
(H), mean alpha angle, anisotropy, and total span.

The 2x2 coherency matrix is::

    C2 = [[<|S_co|^2>,     <S_co * S_cross^*>],
          [<S_cross * S_co^*>, <|S_cross|^2>  ]]

where <.> denotes spatial averaging (boxcar of size ``window_size``).

Dependencies
------------
scipy

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
2026-02-16

Modified
--------
2026-03-10
"""

# Standard library
import dataclasses
from typing import Annotated, Any, Dict, Tuple, TYPE_CHECKING

# Third-party
import numpy as np
from scipy.ndimage import uniform_filter

# GRDL internal
from grdl.image_processing.decomposition.base import PolarimetricDecomposition
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.image_processing.params import Range, Desc
from grdl.vocabulary import ImageModality

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata


@processor_version('1.0.0')
@processor_tags(modalities=[ImageModality.SAR])
class DualPolHAlpha(PolarimetricDecomposition):
    """Dual-pol H/Alpha eigenvalue decomposition.

    Decomposes dual-polarization SAR data (one co-pol + one cross-pol
    channel) into four physically meaningful parameters via eigenvalue
    analysis of the spatially-averaged 2x2 coherency matrix:

    - **entropy** (H): Randomness of the scattering process [0, 1].
      0 = single deterministic mechanism, 1 = fully random.

    - **alpha**: Mean scattering angle [0, 90] degrees.
      0 = surface, 45 = dipole/volume, 90 = dihedral.

    - **anisotropy** (A): Relative strength of the dominant mechanism
      [0, 1].  1 = single dominant, 0 = two equal mechanisms.

    - **span**: Total backscattered power (trace of C2).

    This is a dual-pol specialization of ``PolarimetricDecomposition``.
    Use ``decompose_dual()`` for the natural 2-channel interface, or
    ``decompose()`` for the standard 4-channel quad-pol interface
    (uses ``shh`` as co-pol and ``shv`` as cross-pol; ``svh`` and
    ``svv`` are ignored).

    Parameters
    ----------
    window_size : int
        Side length of the square boxcar averaging window used to
        estimate the coherency matrix.  Must be odd and >= 3.
        Default 7.

    Examples
    --------
    >>> import numpy as np
    >>> from grdl.image_processing.decomposition import DualPolHAlpha
    >>>
    >>> halpha = DualPolHAlpha(window_size=9)
    >>> components = halpha.decompose_dual(s_vv, s_vh)
    >>> print(components['entropy'].shape)  # same as input
    >>> rgb = halpha.to_rgb(components)     # (3, rows, cols) float32
    """

    window_size: Annotated[int, Range(min=3, max=31),
                           Desc('Boxcar averaging window size')] = 7

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def component_names(self) -> Tuple[str, str, str, str]:
        """Names of decomposition output components.

        Returns
        -------
        Tuple[str, str, str, str]
            ``('entropy', 'alpha', 'anisotropy', 'span')``.
        """
        return ('entropy', 'alpha', 'anisotropy', 'span')

    def decompose(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: np.ndarray,
        svv: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Decompose via the standard quad-pol interface.

        For dual-pol data, ``shh`` is used as the co-pol channel and
        ``shv`` as the cross-pol channel.  ``svh`` and ``svv`` are
        ignored.  Prefer ``decompose_dual()`` for clarity when working
        with dual-pol data directly.

        Parameters
        ----------
        shh : np.ndarray
            Co-pol channel (e.g. S_VV or S_HH).  Complex, shape
            ``(rows, cols)``.
        shv : np.ndarray
            Cross-pol channel (e.g. S_VH or S_HV).  Complex, shape
            ``(rows, cols)``.
        svh : np.ndarray
            Ignored for dual-pol decomposition.
        svv : np.ndarray
            Ignored for dual-pol decomposition.

        Returns
        -------
        Dict[str, np.ndarray]
            Keys ``'entropy'``, ``'alpha'``, ``'anisotropy'``, ``'span'``.
        """
        return self.decompose_dual(shh, shv)

    def decompose_dual(
        self,
        s_co: np.ndarray,
        s_cross: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Decompose dual-pol data into H/Alpha parameters.

        Parameters
        ----------
        s_co : np.ndarray
            Co-pol channel (e.g. S_VV).  Complex, shape ``(rows, cols)``.
        s_cross : np.ndarray
            Cross-pol channel (e.g. S_VH).  Complex, shape ``(rows, cols)``.

        Returns
        -------
        Dict[str, np.ndarray]
            Keys ``'entropy'``, ``'alpha'``, ``'anisotropy'``, ``'span'``.
            All real-valued float64 arrays with the same shape as inputs.

        Raises
        ------
        TypeError
            If inputs are not complex-valued numpy arrays.
        ValueError
            If inputs are not 2-D or have mismatched shapes.
        """
        self._validate_dual_inputs(s_co, s_cross)
        ws = self.window_size

        # -- 1. Coherency matrix elements (spatial averaging) --
        c11 = uniform_filter(np.abs(s_co) ** 2, size=ws)
        c22 = uniform_filter(np.abs(s_cross) ** 2, size=ws)
        c12_real = uniform_filter(np.real(s_co * np.conj(s_cross)), size=ws)
        c12_imag = uniform_filter(np.imag(s_co * np.conj(s_cross)), size=ws)
        c12_mag2 = c12_real ** 2 + c12_imag ** 2

        # -- 2. Closed-form 2x2 eigenvalues --
        trace = c11 + c22
        det = c11 * c22 - c12_mag2
        disc = np.sqrt(np.maximum(trace ** 2 - 4.0 * det, 0.0))
        lam1 = (trace + disc) * 0.5
        lam2 = (trace - disc) * 0.5

        # Ensure non-negative (numerical noise)
        np.maximum(lam1, 0.0, out=lam1)
        np.maximum(lam2, 0.0, out=lam2)

        # -- 3. Span --
        span = lam1 + lam2

        # -- 4. Pseudo-probabilities --
        safe_span = np.where(span > 0.0, span, 1.0)
        p1 = lam1 / safe_span
        p2 = lam2 / safe_span

        # -- 5. Entropy: H = -sum(p_i * log2(p_i)) --
        entropy = np.zeros_like(span)
        mask1 = p1 > 0.0
        mask2 = p2 > 0.0
        entropy[mask1] -= p1[mask1] * np.log2(p1[mask1])
        entropy[mask2] -= p2[mask2] * np.log2(p2[mask2])
        np.clip(entropy, 0.0, 1.0, out=entropy)

        # -- 6. Alpha angles from eigenvectors --
        # Eigenvector for lambda_i: [C12, lambda_i - C11] (unnormalized)
        # alpha_i = arccos(|C12| / norm)
        c12_abs = np.sqrt(c12_mag2)

        norm1 = np.sqrt(c12_mag2 + (lam1 - c11) ** 2)
        norm2 = np.sqrt(c12_mag2 + (lam2 - c11) ** 2)

        # Avoid division by zero where eigenvectors are degenerate
        safe_norm1 = np.where(norm1 > 0.0, norm1, 1.0)
        safe_norm2 = np.where(norm2 > 0.0, norm2, 1.0)

        alpha1 = np.arccos(np.clip(c12_abs / safe_norm1, 0.0, 1.0))
        alpha2 = np.arccos(np.clip(c12_abs / safe_norm2, 0.0, 1.0))

        # Where norm was zero, set alpha to 0
        alpha1 = np.where(norm1 > 0.0, alpha1, 0.0)
        alpha2 = np.where(norm2 > 0.0, alpha2, 0.0)

        # Mean alpha (probability-weighted), in degrees
        alpha = np.degrees(p1 * alpha1 + p2 * alpha2)

        # -- 7. Anisotropy --
        anisotropy = np.where(span > 0.0, disc / safe_span, 0.0)
        np.clip(anisotropy, 0.0, 1.0, out=anisotropy)

        return {
            'entropy': entropy,
            'alpha': alpha,
            'anisotropy': anisotropy,
            'span': span,
        }

    def execute(
        self,
        metadata: 'ImageMetadata',
        source: np.ndarray,
        **kwargs: Any,
    ) -> tuple:
        """Execute the decomposition via the universal protocol.

        Dual-pol channels can be provided as keyword arguments
        (``s_co``, ``s_cross``) or extracted from a 2-band source
        array (band 0 = co-pol, band 1 = cross-pol).

        Parameters
        ----------
        metadata : ImageMetadata
            Input image metadata.
        source : np.ndarray
            Input array — 2-D single-channel or 3-D with 2+ bands
            containing co-pol and cross-pol channels.

        Returns
        -------
        tuple[Dict[str, np.ndarray], ImageMetadata]
        """
        self._metadata = metadata
        s_co = kwargs.pop('s_co', None)
        s_cross = kwargs.pop('s_cross', None)
        if s_co is None and source.ndim == 3:
            axis_order = getattr(metadata, 'axis_order', None)
            if axis_order == 'CYX' and source.shape[0] >= 2:
                s_co = source[0]
                s_cross = source[1]
            elif axis_order == 'YXC' and source.shape[-1] >= 2:
                s_co = source[..., 0]
                s_cross = source[..., 1]
            else:
                channel_metadata = getattr(metadata, 'channel_metadata', None)
                bands = getattr(metadata, 'bands', None)
                n_channels = len(channel_metadata) if channel_metadata else bands
                if n_channels == 2 and source.shape[0] == 2 and source.shape[-1] != 2:
                    s_co = source[0]
                    s_cross = source[1]
                elif source.shape[-1] >= 2:
                    s_co = source[..., 0]
                    s_cross = source[..., 1]
        components = self.decompose_dual(s_co, s_cross)
        updated = dataclasses.replace(metadata, bands=len(components))
        return components, updated

    @classmethod
    def rgb_channel_metadata(cls) -> list:
        """Canonical ChannelMetadata descriptors for the 3-band H/Alpha RGB output.

        Returns
        -------
        list[ChannelMetadata]
            Three entries in R/G/B band order:
            ``[span_db, entropy, alpha_norm]``.
        """
        from grdl.IO.models.base import ChannelMetadata

        return [
            ChannelMetadata(
                index=0, name='span_db', role='decomposition',
                extras={'halpha_component': 'span',
                        'formula': '10\u00b7log10(span)',
                        'display': 'Red'},
            ),
            ChannelMetadata(
                index=1, name='entropy', role='decomposition',
                extras={'halpha_component': 'entropy',
                        'formula': 'H \u2208 [0, 1]',
                        'display': 'Green'},
            ),
            ChannelMetadata(
                index=2, name='alpha_norm', role='decomposition',
                extras={'halpha_component': 'alpha',
                        'formula': '\u03b1 / 90 \u2208 [0, 1]',
                        'display': 'Blue'},
            ),
        ]

    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        """Create an RGB composite from H/Alpha decomposition.

        Channel mapping:

        - **Red**: Span in dB (total power)
        - **Green**: Entropy (scattering randomness)
        - **Blue**: Alpha / 90 (scattering mechanism angle)

        Parameters
        ----------
        components : Dict[str, np.ndarray]
            Output of ``decompose()`` or ``decompose_dual()``.
        representation : str
            Ignored for dual-pol (components are already real-valued).
            Accepted for interface compatibility.
        percentile_low : float
            Lower percentile for Span contrast stretch.  Default 2.0.
        percentile_high : float
            Upper percentile for Span contrast stretch.  Default 98.0.

        Returns
        -------
        tuple[np.ndarray, ImageMetadata]
            ``(rgb, metadata)`` — rgb is shape ``(3, rows, cols)``, dtype
            float32, values in [0, 1]; metadata carries H/Alpha channel
            descriptors and spatial dimensions.
        """
        from grdl.IO.models.base import ImageMetadata

        required = {'entropy', 'alpha', 'span'}
        missing = required - set(components.keys())
        if missing:
            raise ValueError(
                f"Missing component keys: {missing}. "
                f"Expected keys from decompose(): {required}"
            )

        # Red: Span in dB, percentile-stretched
        span = components['span']
        span_db = 10.0 * np.log10(np.maximum(span, np.finfo(np.float64).tiny))
        r = self._percentile_stretch(span_db, percentile_low, percentile_high)

        # Green: Entropy already in [0, 1]
        g = np.clip(components['entropy'], 0.0, 1.0).astype(np.float32)

        # Blue: Alpha normalised to [0, 1] (0-90 degrees)
        b = np.clip(components['alpha'] / 90.0, 0.0, 1.0).astype(np.float32)

        rgb = np.stack([r, g, b], axis=0)  # (3, rows, cols) float32
        metadata = ImageMetadata(
            format='HAlphaRGB',
            rows=int(rgb.shape[1]),
            cols=int(rgb.shape[2]),
            dtype=str(rgb.dtype),
            bands=3,
            axis_order='CYX',
            channel_metadata=self.rgb_channel_metadata(),
        )
        return rgb, metadata

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_dual_inputs(
        self,
        s_co: np.ndarray,
        s_cross: np.ndarray,
    ) -> None:
        """Validate dual-pol inputs."""
        for name, arr in [('s_co', s_co), ('s_cross', s_cross)]:
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"{name} must be a numpy ndarray, got {type(arr).__name__}"
                )
            if not np.iscomplexobj(arr):
                raise TypeError(
                    f"{name} must be complex-valued, got {arr.dtype}"
                )
            if arr.ndim != 2:
                raise ValueError(
                    f"{name} must be 2D (rows, cols), got {arr.ndim}D"
                )
        if s_co.shape != s_cross.shape:
            raise ValueError(
                f"Shape mismatch: s_co {s_co.shape} vs s_cross {s_cross.shape}"
            )

    def __repr__(self) -> str:
        return f"DualPolHAlpha(window_size={self.window_size})"
