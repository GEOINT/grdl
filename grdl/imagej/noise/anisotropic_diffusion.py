# -*- coding: utf-8 -*-
"""
Anisotropic Diffusion - Port of Fiji's Anisotropic Diffusion 2D plugin.

Implements Perona-Malik anisotropic diffusion for edge-preserving
smoothing. Unlike isotropic Gaussian blurring which smooths uniformly
in all directions, anisotropic diffusion reduces noise in homogeneous
regions while preserving (and even enhancing) edges. The diffusion
rate is controlled by a conductance function that depends on the local
image gradient.

Particularly useful for:
- SAR speckle reduction while preserving target boundaries
- Noise smoothing in PAN/EO imagery without blurring edges
- Pre-processing thermal imagery for segmentation
- Reducing sensor noise in HSI bands before spectral analysis
- Smoothing MSI classification inputs without degrading boundaries
- Preparing noisy imagery for edge-based feature detection

Attribution
-----------
Algorithm: Perona & Malik, "Scale-Space and Edge Detection Using
Anisotropic Diffusion", IEEE Trans. PAMI 12(7), 1990, pp. 629-639.

Fiji implementation: Thomas Broicher.
Source: ``fiji/process/Anisotropic_Diffusion_2D.java`` (Fiji, GPL-2).
This is an independent NumPy reimplementation following the published
algorithm, not a derivative of the GPL source.

Author
------
Jason Fritz

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-09

Modified
--------
2026-02-09
"""

# Standard library
from typing import Any

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version, processor_tags


def _conductance_exp(gradient_sq: np.ndarray, kappa_sq: float) -> np.ndarray:
    """Exponential conductance function (Perona-Malik option 1).

    ``g(|nabla I|) = exp(-(|nabla I| / kappa)^2)``

    Favors high-contrast edges. Smoothing decreases rapidly as the
    gradient exceeds kappa.
    """
    return np.exp(-gradient_sq / kappa_sq)


def _conductance_inv(gradient_sq: np.ndarray, kappa_sq: float) -> np.ndarray:
    """Inverse quadratic conductance (Perona-Malik option 2).

    ``g(|nabla I|) = 1 / (1 + (|nabla I| / kappa)^2)``

    Favors wide regions over small ones. Smoothing decreases more
    gradually than the exponential form.
    """
    return 1.0 / (1.0 + gradient_sq / kappa_sq)


@processor_tags(modalities=['SAR', 'PAN', 'EO', 'MSI', 'HSI', 'thermal'],
                category='noise')
@processor_version('2.0.0')
class AnisotropicDiffusion(ImageTransform):
    """Perona-Malik anisotropic diffusion, ported from Fiji.

    Iteratively smooths an image while preserving edges. At each
    step, the diffusion flux is modulated by a conductance function
    that reduces diffusion across strong gradients.

    Supports complex-valued input (SLC/SICD SAR data). For complex
    images, the conductance (edge strength) is computed from the
    gradient magnitude ``|nabla I|^2``, while the diffusion update
    is applied to the full complex signal, preserving phase.

    Parameters
    ----------
    n_iterations : int
        Number of diffusion iterations. More iterations produce
        stronger smoothing. Typical range: 5-50. Default is 20.
    kappa : float
        Conductance parameter (gradient threshold). Gradients larger
        than kappa are treated as edges and preserved. Gradients
        smaller than kappa are smoothed. Typical range: 10-100 for
        8-bit imagery; scale proportionally for wider dynamic ranges.
        Default is 20.0.
    gamma : float
        Integration constant (time step). Controls the rate of
        diffusion per iteration. Must be <= 0.25 for numerical
        stability with 4-connected diffusion. Default is 0.1.
    conductance : str
        Conductance function. One of:

        - ``'exponential'`` (default): ``exp(-(grad/kappa)^2)``.
          Strong edge preservation; good for images with clear edges.
        - ``'quadratic'``: ``1 / (1 + (grad/kappa)^2)``.
          Smoother transition; better for images with gradual edges.

    Notes
    -----
    Independent reimplementation of ``fiji/process/Anisotropic_Diffusion_2D.java``
    by Thomas Broicher (Fiji, GPL-2). The algorithm follows the
    Perona-Malik (1990) formulation.

    The diffusion equation is:

    ``dI/dt = div(g(|nabla I|) * nabla I)``

    discretized using 4-connected finite differences (north, south,
    east, west neighbors).

    For SAR imagery, ``kappa`` should be set relative to the speckle
    standard deviation. A common heuristic is ``kappa ≈ 2 * sigma_noise``.

    For complex-valued input (e.g. SAR SLC or NGA SICD), the
    conductance function uses ``|delta|^2 = Re(delta)^2 + Im(delta)^2``
    (the squared magnitude of the complex gradient) so that edge
    detection operates on signal amplitude regardless of phase. The
    diffusion update ``result += gamma * sum(c * delta)`` applies the
    real-valued conductance coefficient to the complex-valued delta,
    preserving the interferometric phase relationship.

    Examples
    --------
    SAR speckle reduction:

    >>> from grdl.imagej import AnisotropicDiffusion
    >>> ad = AnisotropicDiffusion(n_iterations=30, kappa=15.0)
    >>> denoised = ad.apply(sar_amplitude)

    Complex SLC speckle reduction (preserves phase):

    >>> ad = AnisotropicDiffusion(n_iterations=20, kappa=50.0)
    >>> denoised_slc = ad.apply(sicd_complex)

    Strong smoothing with quadratic conductance:

    >>> ad = AnisotropicDiffusion(n_iterations=50, kappa=25.0,
    ...                           conductance='quadratic')
    >>> heavily_smoothed = ad.apply(noisy_thermal)
    """

    __imagej_source__ = 'fiji/process/Anisotropic_Diffusion_2D.java'
    __imagej_version__ = '2.0.0'
    __gpu_compatible__ = True

    CONDUCTANCE_FUNCS = ('exponential', 'quadratic')

    def __init__(
        self,
        n_iterations: int = 20,
        kappa: float = 20.0,
        gamma: float = 0.1,
        conductance: str = 'exponential',
    ) -> None:
        if n_iterations < 1:
            raise ValueError(
                f"n_iterations must be >= 1, got {n_iterations}"
            )
        if kappa <= 0:
            raise ValueError(f"kappa must be > 0, got {kappa}")
        if not (0 < gamma <= 0.25):
            raise ValueError(
                f"gamma must be in (0, 0.25], got {gamma}"
            )
        cond_lower = conductance.lower()
        if cond_lower not in self.CONDUCTANCE_FUNCS:
            raise ValueError(
                f"Unknown conductance '{conductance}'. "
                f"Must be one of {self.CONDUCTANCE_FUNCS}"
            )

        self.n_iterations = n_iterations
        self.kappa = kappa
        self.gamma = gamma
        self.conductance = cond_lower

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply anisotropic diffusion to a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``. Supports both
            real-valued and complex-valued (SLC/SICD) input.

        Returns
        -------
        np.ndarray
            Diffused image, same shape as input. dtype is complex128
            for complex input, float64 otherwise.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(
                f"Expected 2D image, got shape {source.shape}"
            )

        is_complex = np.iscomplexobj(source)
        if is_complex:
            image = source.astype(np.complex128)
        else:
            image = source.astype(np.float64)

        kappa_sq = self.kappa * self.kappa

        # Select conductance function
        if self.conductance == 'exponential':
            g_func = _conductance_exp
        else:
            g_func = _conductance_inv

        result = image.copy()

        for _ in range(self.n_iterations):
            # Compute gradients in 4 directions using finite differences
            # Pad with replicate boundary to avoid edge effects
            padded = np.pad(result, 1, mode='edge')

            # Directional differences (from center to neighbor)
            delta_n = padded[:-2, 1:-1] - result   # north
            delta_s = padded[2:, 1:-1] - result     # south
            delta_e = padded[1:-1, 2:] - result     # east
            delta_w = padded[1:-1, :-2] - result    # west

            # Conductance coefficients — for complex data, use the
            # squared magnitude |delta|^2 = Re^2 + Im^2 so that edge
            # detection operates on gradient amplitude, not the complex
            # value itself. The conductance is always real-valued.
            if is_complex:
                grad_sq_n = np.real(delta_n * np.conj(delta_n))
                grad_sq_s = np.real(delta_s * np.conj(delta_s))
                grad_sq_e = np.real(delta_e * np.conj(delta_e))
                grad_sq_w = np.real(delta_w * np.conj(delta_w))
            else:
                grad_sq_n = delta_n * delta_n
                grad_sq_s = delta_s * delta_s
                grad_sq_e = delta_e * delta_e
                grad_sq_w = delta_w * delta_w

            c_n = g_func(grad_sq_n, kappa_sq)
            c_s = g_func(grad_sq_s, kappa_sq)
            c_e = g_func(grad_sq_e, kappa_sq)
            c_w = g_func(grad_sq_w, kappa_sq)

            # Update: accumulate weighted flux from all directions.
            # For complex data, real-valued conductance * complex delta
            # preserves the phase of the diffusion flux.
            result += self.gamma * (
                c_n * delta_n + c_s * delta_s +
                c_e * delta_e + c_w * delta_w
            )

        return result
