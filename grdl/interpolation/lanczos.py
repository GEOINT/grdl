# -*- coding: utf-8 -*-
"""
Lanczos Interpolation - Lanczos-windowed sinc for bandlimited
reconstruction.

The Lanczos kernel uses a sinc window to truncate the ideal sinc,
parameterized by ``a`` (number of lobes). Widely used in image
resampling and signal processing.

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

# Third-party
import numpy as np

# GRDL internal
from grdl.interpolation.base import KernelInterpolator


class LanczosInterpolator(KernelInterpolator):
    """Lanczos-windowed sinc interpolator.

    Kernel: ``sinc(x) * sinc(x / a)`` for ``|x| < a``, zero otherwise.

    Parameters
    ----------
    a : int
        Number of lobes (kernel half-width in samples). The kernel
        uses ``2 * a`` input samples per output point. Common values:

        - 2 — fast, 4 taps (good for image resampling)
        - 3 — standard, 6 taps (good general-purpose default)
        - 4 — high quality, 8 taps
        - 5 — very high quality, 10 taps

        Default is 3.

    Examples
    --------
    >>> interp = LanczosInterpolator(a=3)
    >>> y_new = interp(x_old, y_old, x_new)
    """

    def __init__(self, a: int = 3) -> None:
        if a < 1:
            raise ValueError(f"a must be >= 1, got {a}")
        self.a = a
        super().__init__(kernel_length=2 * a)

    def _compute_weights(self, dx: np.ndarray) -> np.ndarray:
        """Compute Lanczos kernel weights."""
        weights = np.sinc(dx) * np.sinc(dx / self.a)
        weights[np.abs(dx) >= self.a] = 0.0
        return weights


def lanczos_interpolator(
    a: int = 3,
) -> LanczosInterpolator:
    """Create a Lanczos interpolator.

    Convenience factory function. See :class:`LanczosInterpolator`
    for full documentation.

    Parameters
    ----------
    a : int
        Number of lobes. Default is 3.

    Returns
    -------
    LanczosInterpolator
        Callable interpolator.
    """
    return LanczosInterpolator(a=a)
