# -*- coding: utf-8 -*-
"""
H/A/Alpha Base - Shared abstract base for entropy/alpha decompositions.

Provides ``HAalphaBase``, an abstract base class containing the common
entropy, alpha-mean, and eigenvalue normalization math shared by both the
dual-pol (``DualPolHalpha``) and full-pol (``FullPolHAalpha``)
eigenvalue-based decompositions.

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
from abc import abstractmethod
from typing import Dict, Tuple, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.decomposition.base import PolarimetricDecomposition

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata


class HAalphaBase(PolarimetricDecomposition):
    """Abstract base for H/A/Alpha eigenvalue decompositions.

    Provides shared computation helpers used by both dual-pol and
    full-pol specializations:

    - ``_normalize_eigenvalues``: safe normalization to pseudo-probabilities
    - ``_entropy``: Shannon entropy over N eigenvalues (base-N logarithm)
    - ``_alpha_mean``: probability-weighted mean alpha angle

    Subclasses must implement ``decompose()``, ``to_rgb()``, and
    ``component_names``.
    """

    # ------------------------------------------------------------------
    # Shared computation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
        """Normalize eigenvalues to pseudo-probabilities.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Array of eigenvalues, shape ``(..., N)`` where the last axis
            contains the N eigenvalues per pixel.

        Returns
        -------
        np.ndarray
            Pseudo-probabilities ``p_i = lambda_i / sum(lambda)``,
            same shape as input.  Where total is zero, returns uniform
            distribution (1/N).
        """
        total = eigenvalues.sum(axis=-1, keepdims=True)
        safe_total = np.where(total > 0.0, total, 1.0)
        p = eigenvalues / safe_total
        # Where total was zero, assign uniform
        p = np.where(total > 0.0, p, 1.0 / eigenvalues.shape[-1])
        return p

    @staticmethod
    def _entropy(p: np.ndarray) -> np.ndarray:
        """Compute Shannon entropy over pseudo-probabilities.

        Uses base-N logarithm (where N is the number of eigenvalues)
        so that entropy is normalized to [0, 1].

        Parameters
        ----------
        p : np.ndarray
            Pseudo-probabilities, shape ``(..., N)``.

        Returns
        -------
        np.ndarray
            Entropy values in [0, 1], shape ``(...)``.
        """
        n = p.shape[-1]
        # Safe log: where p <= 0, contribution is 0
        with np.errstate(divide='ignore', invalid='ignore'):
            log_p = np.where(p > 0.0, np.log(p) / np.log(n), 0.0)
        h = -np.sum(p * log_p, axis=-1)
        return np.clip(h, 0.0, 1.0)

    @staticmethod
    def _alpha_mean(p: np.ndarray, cos_alpha: np.ndarray) -> np.ndarray:
        """Compute probability-weighted mean alpha angle.

        Parameters
        ----------
        p : np.ndarray
            Pseudo-probabilities, shape ``(..., N)``.
        cos_alpha : np.ndarray
            Cosine of the alpha angle for each eigenvector, shape
            ``(..., N)``.  Typically ``|e_i[0]|`` (first component of
            the eigenvector).

        Returns
        -------
        np.ndarray
            Mean alpha angle in degrees, shape ``(...)``.
        """
        alpha_i = np.arccos(np.clip(cos_alpha, 0.0, 1.0))
        alpha_mean = np.sum(p * alpha_i, axis=-1)
        return np.degrees(alpha_mean)
