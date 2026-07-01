# -*- coding: utf-8 -*-
"""
Degree of Polarization (Full-Pol) — Barakat generalised DoP from [T3].

Computes the Barakat degree of polarization from the spatially-averaged
3x3 coherency matrix [T3]:

    m = sqrt(max(1 - 27 * det([T3]) / trace([T3])^3,  0))  ∈ [0, 1]

m = 0: completely depolarised (random volume scattering);
m = 1: fully polarised (single deterministic mechanism).

References
----------
Barakat, R. (1977). "Degree of polarization and the principal
    idempotents," Optica Acta, 24(9), pp. 1093–1096.

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
2026-06-30

Modified
--------
2026-06-30
"""

# Standard library
import logging
from typing import Annotated, Dict, Tuple, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.decomposition.base import PolarimetricDecomposition
from grdl.image_processing.decomposition.pol_matrix import CoherencyMatrix
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.image_processing.params import Range, Desc
from grdl.vocabulary import ImageModality

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata

logger = logging.getLogger(__name__)


@processor_version('1.0.0')
@processor_tags(modalities=[ImageModality.SAR])
class DegreeOfPolarization(PolarimetricDecomposition):
    """Barakat generalised degree of polarization (full-pol).

    Decomposes quad-pol SAR data into a single scalar parameter
    representing the degree of polarization of the scattered wave,
    derived from the spatially-averaged 3x3 coherency matrix [T3]:

        m = sqrt(max(1 - 27 * det([T3]) / trace([T3])^3,  0))

    m = 0: completely depolarised (random scattering, e.g. volume);
    m = 1: fully polarised (single deterministic mechanism).

    Parameters
    ----------
    window_size : int
        Side length of the square boxcar averaging window used to
        estimate [T3].  Must be odd and >= 3.  Default 7.

    Examples
    --------
    >>> from grdl.image_processing.decomposition import DegreeOfPolarization
    >>> dop = DegreeOfPolarization(window_size=7)
    >>> comp = dop.decompose(shh, shv, svh, svv)
    >>> print(comp['dop'].shape)

    From a pre-computed coherency matrix:

    >>> from grdl.image_processing.decomposition import CoherencyMatrix
    >>> t3 = CoherencyMatrix(window_size=7).compute(channels)
    >>> comp = dop.decompose_from_t3(t3)

    References
    ----------
    Barakat, R. (1977). "Degree of polarization and the principal
        idempotents," Optica Acta, 24(9), pp. 1093–1096.
    """

    __gpu_compatible__ = False

    window_size: Annotated[int, Range(min=1, max=31),
                           Desc('Boxcar averaging window size')] = 7

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def component_names(self) -> Tuple[str, ...]:
        return ('dop',)

    def decompose(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: np.ndarray,
        svv: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Decompose quad-pol data into degree of polarization.

        Parameters
        ----------
        shh, shv, svh, svv : np.ndarray
            Complex scattering matrix channels, each shape ``(rows, cols)``.

        Returns
        -------
        Dict[str, np.ndarray]
            Key: ``'dop'``.  Real float64 array in [0, 1].
        """
        self._validate_scattering_matrix(shh, shv, svh, svv)
        self._validate_internal_matrix_window_size('decompose_from_t3')
        channels = np.stack([shh, shv, svh, svv], axis=0)
        t3 = CoherencyMatrix(window_size=self.window_size).compute(channels)
        return self.decompose_from_t3(t3)

    def decompose_from_t3(self, t3: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute degree of polarization from a pre-computed [T3].

        Parameters
        ----------
        t3 : np.ndarray
            Coherency matrix, shape ``(3, 3, rows, cols)``, complex.

        Returns
        -------
        Dict[str, np.ndarray]
            Key: ``'dop'``.  Real float64 array in [0, 1].
        """
        if t3.ndim != 4 or t3.shape[:2] != (3, 3):
            raise ValueError(
                f"Expected t3 shape (3, 3, rows, cols), got {t3.shape}"
            )
        rows, cols = t3.shape[2], t3.shape[3]

        # Batch (N, 3, 3) determinant and trace
        t3_flat = t3.transpose(2, 3, 0, 1).reshape(-1, 3, 3)
        det_v = np.linalg.det(t3_flat).real
        trace_v = (
            t3_flat[:, 0, 0].real
            + t3_flat[:, 1, 1].real
            + t3_flat[:, 2, 2].real
        )

        eps = np.finfo(np.float64).tiny
        safe_trace3 = np.where(np.abs(trace_v) > eps, trace_v ** 3, eps)
        m = np.sqrt(np.maximum(1.0 - 27.0 * det_v / safe_trace3, 0.0))
        dop = np.clip(m, 0.0, 1.0).reshape(rows, cols)

        logger.debug("DegreeOfPolarization: mean DoP = %.3f", float(np.nanmean(dop)))
        return {'dop': dop}

    def to_rgb(
        self,
        components: Dict[str, np.ndarray],
        representation: str = 'db',
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> Tuple[np.ndarray, 'ImageMetadata']:
        """Create a grayscale RGB image from the degree of polarization.

        Returns
        -------
        tuple[np.ndarray, ImageMetadata]
            ``(rgb, metadata)`` — rgb shape ``(3, rows, cols)``, float32 [0, 1].
        """
        from grdl.IO.models.base import ImageMetadata, ChannelMetadata

        dop = np.clip(components['dop'], 0.0, 1.0).astype(np.float32)
        rgb = np.stack([dop, dop, dop], axis=0)  # grayscale

        meta = ImageMetadata(
            format='DoPRGB',
            rows=dop.shape[0],
            cols=dop.shape[1],
            bands=3,
            dtype='float32',
            axis_order='CYX',
            channel_metadata=[
                ChannelMetadata(index=0, name='dop', role='rgb_red'),
                ChannelMetadata(index=1, name='dop', role='rgb_green'),
                ChannelMetadata(index=2, name='dop', role='rgb_blue'),
            ],
        )
        return rgb, meta
