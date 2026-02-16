# -*- coding: utf-8 -*-
"""
Polar Format Algorithm - SAR image formation via polar-to-rectangular
interpolation and 2D Fourier transform.

Implements the three-stage PFA pipeline:

1. **Range interpolation** — resample each pulse from polar kv to
   uniform kv grid (keystone formatting).
2. **Azimuth interpolation** — resample each range bin from
   ``kv * tan(phi)`` to uniform ku grid.
3. **Compress** — 2D Fourier transform from spatial frequency to
   image domain (``fft2`` for SGN=+1, ``ifft2`` for SGN=-1).

Uses ``scipy.interpolate.interp1d`` as the default interpolation
backend. The interpolator is injectable via constructor for future
bandlimited wideband interpolators.

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
2026-02-12

Modified
--------
2026-02-12
"""

# Standard library
from typing import Any, Callable, Dict, Optional, Union

# Third-party
import numpy as np
from scipy import fft
from scipy.interpolate import interp1d
from scipy.signal.windows import taylor as _taylor_window

# GRDL internal
from grdl.image_processing.sar.image_formation.base import (
    ImageFormationAlgorithm,
)
from grdl.image_processing.sar.image_formation.geometry import (
    CollectionGeometry,
)
from grdl.image_processing.sar.image_formation.polar_grid import PolarGrid


def _scipy_interp1d(
    x_old: np.ndarray,
    y_old: np.ndarray,
    x_new: np.ndarray,
) -> np.ndarray:
    """Default interpolation using scipy interp1d (linear).

    Parameters
    ----------
    x_old : np.ndarray
        Original sample coordinates.
    y_old : np.ndarray
        Original sample values (complex).
    x_new : np.ndarray
        Target sample coordinates.

    Returns
    -------
    np.ndarray
        Interpolated values at ``x_new``.
    """
    func = interp1d(
        x_old, y_old,
        kind='linear',
        bounds_error=False,
        fill_value=0.0,
    )
    return func(x_new)


_WINDOW_FUNCTIONS = {
    'uniform': None,
    'taylor': lambda n: _taylor_window(n, nbar=4, sll=35, norm=True),
    'hamming': np.hamming,
    'hanning': np.hanning,
}


class PolarFormatAlgorithm(ImageFormationAlgorithm):
    """Polar Format Algorithm for spotlight SAR image formation.

    Three-stage pipeline: range interpolation → azimuth interpolation
    → 2D IFFT. Each stage is exposed as a public method for
    intermediate inspection (tapout points).

    Parameters
    ----------
    grid : PolarGrid
        Computed polar grid with k-space bounds and grid dimensions.
    interpolator : callable, optional
        Interpolation function with signature
        ``(x_old, y_old, x_new) -> y_new``.
        Defaults to ``scipy.interpolate.interp1d`` (linear).
    weighting : str or callable, optional
        Window function applied to k-space data before the transform.
        Tapers the edges to reduce sidelobe artifacts (Gibbs
        phenomenon). Built-in options:

        - ``'uniform'`` — no weighting (default)
        - ``'taylor'`` — Taylor window (4-bar, -35 dB sidelobes)
        - ``'hamming'`` — Hamming window (-42 dB first sidelobe)
        - ``'hanning'`` — Hanning window (-31 dB first sidelobe)

        Pass a callable ``f(n) -> ndarray`` for custom windows.
    phase_sgn : int
        CPHD PhaseSGN convention (``+1`` or ``-1``).  Controls
        whether ``compress()`` uses ``fft2`` (SGN = +1) or
        ``ifft2`` (SGN = -1).  Default ``-1``.

    Examples
    --------
    Full pipeline with default (linear) interpolator:

    >>> pfa = PolarFormatAlgorithm(grid=polar_grid)
    >>> image = pfa.form_image(signal, geometry)

    Bandwidth-preserving windowed sinc interpolator:

    >>> from grdl.interpolation import windowed_sinc_interpolator
    >>> pfa = PolarFormatAlgorithm(
    ...     grid=polar_grid,
    ...     interpolator=windowed_sinc_interpolator(kernel_length=8),
    ... )
    >>> image = pfa.form_image(signal, geometry)

    Taylor-weighted compression (reduces azimuth sidelobes):

    >>> pfa = PolarFormatAlgorithm(grid=polar_grid, weighting='taylor')
    >>> image = pfa.form_image(signal, geometry)

    Stage-by-stage with tapout:

    >>> range_interp = pfa.interpolate_range(signal, geometry)
    >>> az_interp = pfa.interpolate_azimuth(range_interp, geometry)
    >>> image = pfa.compress(az_interp)
    """

    def __init__(
        self,
        grid: PolarGrid,
        interpolator: Optional[Callable] = None,
        weighting: Union[str, Callable, None] = None,
        phase_sgn: int = -1,
    ) -> None:
        self.grid = grid
        self._interp = interpolator or _scipy_interp1d
        self._weight_func = self._resolve_weighting(weighting)
        self._pad_factor = 1.0
        self._phase_sgn = phase_sgn

    @staticmethod
    def _resolve_weighting(
        weighting: Union[str, Callable, None],
    ) -> Optional[Callable]:
        """Resolve weighting parameter to a window function.

        Parameters
        ----------
        weighting : str, callable, or None
            Window specification.

        Returns
        -------
        callable or None
            Window function ``f(n) -> ndarray``, or None for uniform.
        """
        if weighting is None or weighting == 'uniform':
            return None
        if callable(weighting):
            return weighting
        if isinstance(weighting, str):
            key = weighting.lower()
            if key not in _WINDOW_FUNCTIONS:
                raise ValueError(
                    f"Unknown weighting '{weighting}'. "
                    f"Options: {list(_WINDOW_FUNCTIONS.keys())}"
                )
            return _WINDOW_FUNCTIONS[key]
        raise TypeError(
            f"weighting must be str, callable, or None, got {type(weighting)}"
        )

    # ------------------------------------------------------------------
    # Stage 1: Range interpolation
    # ------------------------------------------------------------------

    def interpolate_range(
        self,
        signal: np.ndarray,
        geometry: CollectionGeometry,
    ) -> np.ndarray:
        """Resample each pulse from polar kv to uniform kv grid.

        Parameters
        ----------
        signal : np.ndarray
            Phase history data, shape ``(npulses, nsamples)``.
        geometry : CollectionGeometry
            Collection geometry (provides phi, k_sf, frequency params).

        Returns
        -------
        np.ndarray
            Range-interpolated data, shape
            ``(npulses, rec_n_samples)``.
        """
        pg = self.grid
        npulses = signal.shape[0]

        # Uniform kv grid
        kv_min, kv_max = pg.kv_bounds
        sampling = (kv_max - kv_min) / pg.rec_n_samples
        kv_uniform = np.arange(pg.rec_n_samples) * sampling + kv_min

        result = np.zeros(
            (npulses, pg.rec_n_samples), dtype=signal.dtype,
        )

        for i in range(npulses):
            kv_polar = pg.get_kv_for_pulse(i)
            result[i, :] = self._interp(kv_polar, signal[i, :], kv_uniform)

        return result

    # ------------------------------------------------------------------
    # Stage 2: Azimuth interpolation
    # ------------------------------------------------------------------

    def interpolate_azimuth(
        self,
        range_interpolated: np.ndarray,
        geometry: CollectionGeometry,
    ) -> np.ndarray:
        """Resample each range bin from keystone ku to uniform ku grid.

        Parameters
        ----------
        range_interpolated : np.ndarray
            Range-interpolated data, shape
            ``(npulses, rec_n_samples)``.
        geometry : CollectionGeometry
            Collection geometry (provides phi for tan projection).

        Returns
        -------
        np.ndarray
            Azimuth-interpolated data, shape
            ``(rec_n_pulses, rec_n_samples)``.
        """
        pg = self.grid

        # Uniform output ku grid
        ku_min, ku_max = pg.ku_bounds
        sampling = (ku_max - ku_min) / pg.rec_n_pulses
        ku_uniform = np.arange(pg.rec_n_pulses) * sampling + ku_min

        # Keystone projection: ku = kv * tan(phi)
        proj = np.tan(geometry.phi)
        kv_ks = np.linspace(
            pg.kv_bounds[0], pg.kv_bounds[1], pg.rec_n_samples,
        )

        # Ground-plane projection can reverse the phi ordering
        # (phi decreasing → ku_ks descending). Interpolators require
        # monotonically increasing x, so detect and flip once.
        ascending = proj[-1] >= proj[0]

        result = np.zeros(
            (pg.rec_n_pulses, pg.rec_n_samples),
            dtype=range_interpolated.dtype,
        )

        for i in range(pg.rec_n_samples):
            ku_ks = kv_ks[i] * proj
            col = range_interpolated[:, i]
            if not ascending:
                ku_ks = ku_ks[::-1]
                col = col[::-1]
            result[:, i] = self._interp(ku_ks, col, ku_uniform)

        return result

    # ------------------------------------------------------------------
    # Stage 3: Compress (2D Fourier transform)
    # ------------------------------------------------------------------

    def compress(
        self,
        interpolated: np.ndarray,
        pad_factor: float = 1.25,
    ) -> np.ndarray:
        """Transform from spatial frequency to image domain.

        Applies optional weighting (apodization), zero-pads to avoid
        circular aliasing (fold-over), then a 2D Fourier transform to
        convert the rectangular k-space data into a complex SAR image.

        The transform direction depends on the CPHD PhaseSGN:

        - ``SGN = -1``: k-space has ``exp(-j k R)`` → ``ifft2`` focuses
        - ``SGN = +1``: k-space has ``exp(+j k R)`` → ``fft2`` focuses

        The DFT is inherently circular: without zero-padding the image
        PSF wraps from one edge to the other.  Padding by at least 2x
        in each dimension eliminates this fold-over artifact.

        Parameters
        ----------
        interpolated : np.ndarray
            Rectangular k-space data, shape
            ``(rec_n_pulses, rec_n_samples)``.
        pad_factor : float
            Zero-pad multiplier for each dimension.  The transform size
            is ``ceil(dim * pad_factor)``.  Values > 1.0 oversample the
            output image (finer pixel spacing, same scene extent).
            Set to 1.0 to disable padding.

        Returns
        -------
        np.ndarray
            Complex SAR image, shape
            ``(ceil(rec_n_pulses * pad_factor),
            ceil(rec_n_samples * pad_factor))``.
            Rows = azimuth, cols = range.
        """
        self._pad_factor = pad_factor

        data = interpolated
        naz, nrg = data.shape

        if self._weight_func is not None:
            w_az = self._weight_func(naz).astype(data.real.dtype)
            w_rg = self._weight_func(nrg).astype(data.real.dtype)
            data = data * w_az[:, np.newaxis] * w_rg[np.newaxis, :]

        # Zero-pad to suppress circular aliasing (fold-over)
        if pad_factor > 1.0:
            naz_pad = int(np.ceil(naz * pad_factor))
            nrg_pad = int(np.ceil(nrg * pad_factor))
            padded = np.zeros((naz_pad, nrg_pad), dtype=data.dtype)
            # Center the k-space data in the padded array
            az_start = (naz_pad - naz) // 2
            rg_start = (nrg_pad - nrg) // 2
            padded[az_start:az_start + naz, rg_start:rg_start + nrg] = data
            data = padded

        shifted = np.fft.ifftshift(data)
        if self._phase_sgn >= 0:
            image = np.fft.fftshift(fft.fft2(shifted))
        else:
            image = np.fft.fftshift(fft.ifft2(shifted))

        # Output: rows = azimuth, cols = range
        # Transpose to SICD (rows = range, cols = azimuth) at write time
        return image

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def form_image(
        self,
        signal: np.ndarray,
        geometry: Any,
    ) -> np.ndarray:
        """Run the full PFA pipeline: range → azimuth → compress.

        Parameters
        ----------
        signal : np.ndarray
            Phase history data, shape ``(npulses, nsamples)``.
        geometry : CollectionGeometry
            Collection geometry.

        Returns
        -------
        np.ndarray
            Complex SAR image.
        """
        range_interp = self.interpolate_range(signal, geometry)
        az_interp = self.interpolate_azimuth(range_interp, geometry)
        return self.compress(az_interp)

    def get_output_grid(self) -> Dict[str, Any]:
        """Return output grid parameters for SICD metadata.

        Adjusts sample spacing and output dimensions to reflect the
        zero-pad factor applied during compression.  Bandwidth and
        resolution are unaffected by oversampling.

        Returns
        -------
        Dict[str, Any]
            Grid parameters from the PolarGrid, adjusted for
            ``pad_factor``.
        """
        grid = self.grid.get_output_grid()
        pf = self._pad_factor

        # Sample spacing gets finer by pad_factor
        grid['rg_ss'] /= pf
        grid['az_ss'] /= pf

        # Output dimensions grow by pad_factor
        grid['rec_n_samples'] = int(np.ceil(
            grid['rec_n_samples'] * pf,
        ))
        grid['rec_n_pulses'] = int(np.ceil(
            grid['rec_n_pulses'] * pf,
        ))

        grid['pad_factor'] = pf

        return grid
