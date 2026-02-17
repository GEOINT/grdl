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
duane.d.smalley@gmail.com

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
2026-02-16
"""

# Standard library
from typing import Annotated, Dict, Tuple

# Third-party
import numpy as np
from scipy.ndimage import uniform_filter

# GRDL internal
from grdl.image_processing.base import ImageProcessor
from grdl.image_processing.versioning import processor_version
from grdl.image_processing.params import Range, Desc


@processor_version('1.0.0')
class DualPolHAlpha(ImageProcessor):
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
    >>> components = halpha.decompose(s_vv, s_vh)
    >>> print(components['entropy'].shape)  # same as input
    >>> rgb = halpha.to_rgb(components)     # (rows, cols, 3) float32
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
        self._validate_inputs(s_co, s_cross)
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

    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> np.ndarray:
        """Create an RGB composite from H/Alpha decomposition.

        Channel mapping:

        - **Red**: Span in dB (total power)
        - **Green**: Entropy (scattering randomness)
        - **Blue**: Alpha / 90 (scattering mechanism angle)

        Parameters
        ----------
        components : Dict[str, np.ndarray]
            Output of ``decompose()``.
        percentile_low : float
            Lower percentile for Span contrast stretch.  Default 2.0.
        percentile_high : float
            Upper percentile for Span contrast stretch.  Default 98.0.

        Returns
        -------
        np.ndarray
            RGB image, shape ``(rows, cols, 3)``, dtype float32,
            values in [0, 1].
        """
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

        return np.dstack([r, g, b])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_inputs(
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

    @staticmethod
    def _percentile_stretch(
        arr: np.ndarray,
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> np.ndarray:
        """Percentile-stretch a real array to [0, 1]."""
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        vmin = np.percentile(finite, percentile_low)
        vmax = np.percentile(finite, percentile_high)
        span = vmax - vmin
        if span < np.finfo(np.float32).eps:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr - vmin) / span, 0.0, 1.0).astype(np.float32)

    def __repr__(self) -> str:
        return f"DualPolHAlpha(window_size={self.window_size})"
