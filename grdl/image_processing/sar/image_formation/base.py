# -*- coding: utf-8 -*-
"""
Image Formation Algorithm ABC - Algorithm-agnostic interface.

Defines the abstract contract for SAR image formation algorithms.
The interface is intentionally minimal: signal + geometry in, complex
image out. Algorithm-specific configuration (polar grid, range-Doppler
parameters, etc.) belongs in each concrete class's ``__init__``.

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
from abc import ABC, abstractmethod
from typing import Any, Dict

# Third-party
import numpy as np


class ImageFormationAlgorithm(ABC):
    """ABC for SAR image formation algorithms.

    The interface is intentionally minimal: signal + geometry in,
    complex image out. Algorithm-specific configuration (polar grid,
    range-Doppler parameters, subaperture partitioning) goes in each
    concrete class's ``__init__``.

    Subclasses
    ----------
    - ``PolarFormatAlgorithm`` — spotlight PFA (range interp → azimuth
      interp → IFFT)
    - Future: ``RangeDopplerAlgorithm``, ``ChirpScalingAlgorithm``
    """

    @abstractmethod
    def form_image(
        self,
        signal: np.ndarray,
        geometry: Any,
    ) -> np.ndarray:
        """Transform phase history data into a complex SAR image.

        Parameters
        ----------
        signal : np.ndarray
            Phase history, shape ``(num_pulses, num_samples)``.
        geometry : Any
            Collection geometry providing per-pulse state vectors
            and coordinate system parameters. Typically a
            ``CollectionGeometry`` instance for PFA.

        Returns
        -------
        np.ndarray
            Complex SAR image.
        """

    @abstractmethod
    def get_output_grid(self) -> Dict[str, Any]:
        """Return output grid parameters for SICD metadata population.

        Returns
        -------
        Dict[str, Any]
            Algorithm-specific grid parameters. For PFA this includes
            sample spacing, impulse response bandwidth, k-space bounds,
            and unit vectors. Keys match SICD Grid fields.
        """
