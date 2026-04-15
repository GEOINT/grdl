# -*- coding: utf-8 -*-
"""
Pauli Decomposition - Quad-pol Pauli basis scattering matrix decomposition.

Decomposes the 2x2 complex scattering matrix [S] into three Pauli basis
components that separate physical scattering mechanisms:

    surface (alpha)       = (S_HH + S_VV) / sqrt(2)  -- odd-bounce
    double_bounce (beta)  = (S_HH - S_VV) / sqrt(2)  -- even-bounce
    volume (gamma)        = (S_HV + S_VH) / sqrt(2)  -- cross-pol

All arithmetic is performed in the complex domain. Phase relationships
between co-pol channels (HH, VV) drive the constructive/destructive
interference that separates surface from double-bounce scattering.
Cross-pol mixing (HV + VH) captures the full volume scattering
contribution from both cross-polarized channels.

The 1/sqrt(2) normalization ensures the Pauli basis is unitary: under
monostatic reciprocity (S_HV = S_VH), total component power equals the
span of the scattering matrix.

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
2026-01-30

Modified
--------
2026-03-10
"""

# Standard library
import logging
from typing import Dict, Tuple, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.decomposition.base import PolarimetricDecomposition
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from grdl.IO.models.base import ChannelMetadata, ImageMetadata


@processor_version('0.1.0')
@processor_tags(modalities=[ImageModality.SAR])
class PauliDecomposition(PolarimetricDecomposition):
    """
    Quad-pol Pauli basis decomposition.

    Decomposes the 2x2 scattering matrix [S] into three orthogonal
    Pauli basis components representing distinct physical scattering
    mechanisms:

    - **surface** (alpha): ``(S_HH + S_VV) / sqrt(2)`` --
      odd-bounce / surface scattering. Constructive interference between
      co-pol channels indicates single-reflection geometry.

    - **double_bounce** (beta): ``(S_HH - S_VV) / sqrt(2)`` --
      even-bounce / dihedral scattering. Destructive interference
      (180 deg phase shift between HH and VV) indicates double-reflection
      geometry (buildings, tree trunks over ground).

    - **volume** (gamma): ``(S_HV + S_VH) / sqrt(2)`` --
      cross-pol / volume scattering. Depolarization from randomly
      oriented scatterers (forest canopy, rough surfaces). Uses both
      cross-pol channels for the full bistatic contribution.

    The output components are **complex-valued**, preserving the phase
    information from the scattering matrix interference. Convert to
    magnitude, power, or dB with the provided convenience methods
    when ready to discard phase.

    Parameters
    ----------
    None. The Pauli decomposition has no tunable parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from grdl.image_processing import PauliDecomposition
    >>>
    >>> pauli = PauliDecomposition()
    >>>
    >>> # Decompose complex scattering matrix channels
    >>> components = pauli.decompose(shh, shv, svh, svv)
    >>> surface = components['surface']            # complex
    >>> dbl_bounce = components['double_bounce']   # complex
    >>> volume = components['volume']              # complex
    >>>
    >>> # Convert to display representations
    >>> db_components = pauli.to_db(components)
    >>> rgb = pauli.to_rgb(components, representation='db')
    """

    @property
    def component_names(self) -> Tuple[str, str, str]:
        """
        Names of the Pauli decomposition components.

        Returns
        -------
        Tuple[str, str, str]
            ``('surface', 'double_bounce', 'volume')``.
        """
        return ('surface', 'double_bounce', 'volume')

    def _build_component_metadata(
        self,
        metadata: 'ImageMetadata',
    ) -> list['ChannelMetadata']:
        """Return Pauli-specific output lineage metadata."""
        from grdl.IO.models.base import ChannelMetadata

        return [
            ChannelMetadata(
                index=0,
                name='surface',
                role='decomposition',
                source_indices=[0, 3],
            ),
            ChannelMetadata(
                index=1,
                name='double_bounce',
                role='decomposition',
                source_indices=[0, 3],
            ),
            ChannelMetadata(
                index=2,
                name='volume',
                role='decomposition',
                source_indices=[1, 2],
            ),
        ]

    def decompose(
        self,
        shh: np.ndarray,
        shv: np.ndarray,
        svh: np.ndarray,
        svv: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Decompose the scattering matrix into Pauli basis components.

        Performs the complex arithmetic::

            surface       = (S_HH + S_VV) / sqrt(2)
            double_bounce = (S_HH - S_VV) / sqrt(2)
            volume        = (S_HV + S_VH) / sqrt(2)

        Phase information from channel interference is fully preserved.
        The addition ``S_HH + S_VV`` produces constructive interference
        when HH and VV are in phase (surface scattering) and destructive
        interference when they are 180 deg out of phase. The subtraction
        ``S_HH - S_VV`` captures the complementary double-bounce signature.

        Parameters
        ----------
        shh : np.ndarray
            Complex S_HH channel. Shape (rows, cols).
        shv : np.ndarray
            Complex S_HV channel. Shape (rows, cols).
        svh : np.ndarray
            Complex S_VH channel. Shape (rows, cols).
        svv : np.ndarray
            Complex S_VV channel. Shape (rows, cols).

        Returns
        -------
        Dict[str, np.ndarray]
            Complex-valued components with keys ``'surface'``,
            ``'double_bounce'``, ``'volume'``. All arrays have the
            same shape and dtype as the inputs.

        Raises
        ------
        TypeError
            If any input is not complex-valued.
        ValueError
            If inputs are not 2D or have mismatched shapes.
        """
        self._validate_scattering_matrix(shh, shv, svh, svv)
        logger.info(
            "Pauli decomposition: shape %s, 4 channels, dtype %s",
            shh.shape, shh.dtype,
        )

        # Normalization constant in matching precision to avoid
        # silent upcast from complex64 to complex128.
        if shh.dtype == np.complex64:
            norm = np.float32(1.0 / np.sqrt(2.0))
        else:
            norm = 1.0 / np.sqrt(2.0)

        result = {
            'surface': (shh + svv) * norm,
            'double_bounce': (shh - svv) * norm,
            'volume': (shv + svh) * norm,
        }
        logger.debug(
            "Pauli magnitudes: surface=[%.4f, %.4f], "
            "double_bounce=[%.4f, %.4f], volume=[%.4f, %.4f]",
            float(np.min(np.abs(result['surface']))),
            float(np.max(np.abs(result['surface']))),
            float(np.min(np.abs(result['double_bounce']))),
            float(np.max(np.abs(result['double_bounce']))),
            float(np.min(np.abs(result['volume']))),
            float(np.max(np.abs(result['volume']))),
        )
        return result

    @classmethod
    def rgb_channel_metadata(cls) -> list:
        """Canonical ChannelMetadata descriptors for the 3-band Pauli RGB output.

        Returns
        -------
        list[ChannelMetadata]
            Three entries in R/G/B band order:
            ``[double_bounce, volume, surface]``.
        """
        from grdl.IO.models.base import ChannelMetadata

        return [
            ChannelMetadata(
                index=0, name='double_bounce', role='decomposition',
                source_indices=[0, 3],
                extras={'pauli_component': 'double_bounce',
                        'formula': 'T3[1,1] = <|S_HH-S_VV|\u00b2>/2',
                        'display': 'Red'},
            ),
            ChannelMetadata(
                index=1, name='volume', role='decomposition',
                source_indices=[1, 2],
                extras={'pauli_component': 'volume',
                        'formula': 'T3[2,2] = 2\u00b7<|S_HV|\u00b2>',
                        'display': 'Green'},
            ),
            ChannelMetadata(
                index=2, name='surface', role='decomposition',
                source_indices=[0, 3],
                extras={'pauli_component': 'surface',
                        'formula': 'T3[0,0] = <|S_HH+S_VV|\u00b2>/2',
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
        """
        Create Pauli RGB composite.

        Channel mapping:

        - **Red**: double_bounce (even-bounce / dihedral)
        - **Green**: volume (cross-pol / depolarization)
        - **Blue**: surface (odd-bounce / single reflection)

        Parameters
        ----------
        components : Dict[str, np.ndarray]
            Output of ``decompose()``. Must contain keys ``'surface'``,
            ``'double_bounce'``, and ``'volume'``.
        representation : str
            How to convert complex components before stretching.
            One of ``'db'`` (20*log10(|z|)), ``'magnitude'`` (|z|),
            or ``'power'`` (|z|^2). Default ``'db'``.
        percentile_low : float
            Lower percentile for contrast stretch. Default 2.0.
        percentile_high : float
            Upper percentile for contrast stretch. Default 98.0.

        Returns
        -------
        tuple[np.ndarray, ImageMetadata]
            ``(rgb, metadata)`` — rgb is shape (3, rows, cols), dtype
            float32, values in [0, 1]; metadata carries Pauli channel
            descriptors and spatial dimensions.

        Raises
        ------
        ValueError
            If ``representation`` is not one of the supported values,
            or if required component keys are missing.
        """
        from grdl.IO.models.base import ImageMetadata

        required = {'surface', 'double_bounce', 'volume'}
        missing = required - set(components.keys())
        if missing:
            raise ValueError(
                f"Missing component keys: {missing}. "
                f"Expected keys from decompose(): {required}"
            )

        converters = {
            'db': self.to_db,
            'magnitude': self.to_magnitude,
            'power': self.to_power,
        }
        if representation not in converters:
            raise ValueError(
                f"representation must be one of {list(converters.keys())}, "
                f"got '{representation}'"
            )

        real_components = converters[representation](components)

        r = self._percentile_stretch(
            real_components['double_bounce'], percentile_low, percentile_high
        )
        g = self._percentile_stretch(
            real_components['volume'], percentile_low, percentile_high
        )
        b = self._percentile_stretch(
            real_components['surface'], percentile_low, percentile_high
        )

        rgb = np.stack([r, g, b], axis=0)  # (3, rows, cols) float32
        metadata = ImageMetadata(
            format='PauliRGB',
            rows=int(rgb.shape[1]),
            cols=int(rgb.shape[2]),
            dtype=str(rgb.dtype),
            bands=3,
            axis_order='CYX',
            channel_metadata=self.rgb_channel_metadata(),
        )
        return rgb, metadata

    def __repr__(self) -> str:
        return "PauliDecomposition()"
