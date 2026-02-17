# -*- coding: utf-8 -*-
"""
Coherent Shape Index - Color sub-aperture imaging for SAR target discrimination.

Produces an RGB composite from complex SAR imagery where man-made targets
with stable scattering across the aperture appear brightly coloured and
natural clutter appears greyscale.

The processor delegates sub-aperture splitting to
:class:`~grdl.image_processing.sar.SublookDecomposition`, reusing its
oversampling-aware FFT splitting, deweighting, and overlap logic.  On top
of the sub-look magnitudes it applies sarpy-style intensity preservation
(scaling each pixel's RGB so that ``max(R, G, B) = |original|``) and
platform direction correction (reversing RGB order for right-looking
platforms based on ``SCPCOA.SideOfTrack`` from the SICD metadata).

Optional display normalisation uses grdl's
:class:`~grdl.image_processing.intensity.ToDecibels` and
:class:`~grdl.image_processing.intensity.PercentileStretch` processors.

Dependencies
------------
numpy

Author
------
Ava Courtney

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-17

Modified
--------
2026-02-17
"""

# Standard library
import dataclasses
from typing import Annotated, Any, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageProcessor
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.image_processing.intensity import ToDecibels, PercentileStretch
from grdl.image_processing.sar.sublook import SublookDecomposition
from grdl.vocabulary import ImageModality, ProcessorCategory
from grdl.IO.models import SICDMetadata

if TYPE_CHECKING:
    from grdl.IO.models.base import ImageMetadata


# ===================================================================
# Constants
# ===================================================================

_NUM_LOOKS = 3
"""CSI always produces exactly 3 sub-aperture looks mapped to R, G, B."""


# ===================================================================
# CSIProcessor
# ===================================================================

@processor_version('0.2.0')
@processor_tags(
    modalities=[ImageModality.SAR],
    category=ProcessorCategory.ENHANCE,
    description='Coherent Shape Index (color sub-aperture) RGB composite',
)
class CSIProcessor(ImageProcessor):
    """Coherent Shape Index (Color Sub-aperture Imaging) processor.

    Produces an RGB composite from complex SAR imagery where the colour
    of each pixel encodes coherence across sub-apertures:

    - **Brightly coloured** pixels indicate persistent, frequency-selective
      scatterers (man-made structures, corner reflectors).
    - **Greyscale** pixels indicate distributed or decorrelated scatterers
      (vegetation, water, rough ground).

    Processing stages:

    1. **Sub-look generation** -- Split the complex image into 3
       sub-apertures via :class:`SublookDecomposition` (oversampling-aware,
       optional deweighting, configurable overlap).
    2. **Detection** -- Compute magnitude of each sub-look.
    3. **Intensity preservation** -- Scale each pixel's RGB magnitudes
       so that ``max(R, G, B) = |original|``, preserving full spatial
       resolution in intensity while encoding sub-aperture coherence
       as colour.
    4. **Platform direction** -- Reverse RGB order for right-looking
       platforms (from ``SCPCOA.SideOfTrack``) so colour mapping is
       consistent regardless of look direction.
    5. **Normalization** (optional) -- Apply dB conversion + percentile
       stretch (or direct percentile stretch) via grdl's
       :class:`ToDecibels` and :class:`PercentileStretch` processors.

    When ``normalization='none'`` (default), the output has the same
    dynamic range as the original magnitude.  Use ``normalization='log'``
    or ``'percentile'`` for display-ready [0, 1] output.

    Parameters
    ----------
    metadata : SICDMetadata
        SICD metadata from the reader.  Must contain populated ``grid``
        with ``row`` and ``col`` direction parameters.
    dimension : str
        Frequency-domain axis to split: ``'azimuth'`` or ``'range'``.
        Default is ``'azimuth'``.
    overlap : float
        Fractional overlap between adjacent sub-bands, in [0.0, 1.0).
        Default is 0.5 (50 %% overlap for smooth colour transitions).
    deweight : bool
        Remove the collection apodization window before splitting.
        Default is True.
    normalization : str
        Normalisation method applied after compositing:
        ``'none'`` (raw magnitude, default), ``'log'`` (dB + percentile
        stretch), or ``'percentile'`` (direct percentile stretch).
    plow : float
        Lower percentile for contrast stretch.  Default 2.0.
    phigh : float
        Upper percentile for contrast stretch.  Default 98.0.
    floor_db : float
        Minimum dB floor when ``normalization='log'``.  Default -60.0.

    Raises
    ------
    ValueError
        If *metadata* lacks required grid fields, *overlap* is out of
        range, or *dimension* is invalid.

    Examples
    --------
    >>> from grdl.IO.sar import SICDReader
    >>> from grdl.image_processing.sar import CSIProcessor
    >>>
    >>> with SICDReader('image.nitf') as reader:
    ...     image = reader.read()
    ...     csi = CSIProcessor(reader.metadata)
    ...     rgb = csi.apply(image)  # (rows, cols, 3) float64
    """

    # -- Annotated scalar fields for GUI introspection (__param_specs__) --
    dimension: Annotated[str, Options('azimuth', 'range'), Desc('Frequency axis to split')] = 'azimuth'
    overlap: Annotated[float, Range(min=0.0, max=0.99), Desc('Fractional overlap between sub-bands')] = 0.5
    deweight: Annotated[bool, Desc('Remove collection apodization before splitting')] = True
    normalization: Annotated[str, Options('none', 'log', 'percentile'), Desc('Normalization method per channel')] = 'none'
    plow: Annotated[float, Range(min=0.0, max=100.0), Desc('Lower percentile for contrast stretch')] = 2.0
    phigh: Annotated[float, Range(min=0.0, max=100.0), Desc('Upper percentile for contrast stretch')] = 98.0
    floor_db: Annotated[float, Range(max=0.0), Desc('Minimum dB floor (log mode)')] = -60.0

    def __init__(
        self,
        metadata: SICDMetadata,
        dimension: str = 'azimuth',
        overlap: float = 0.5,
        deweight: bool = True,
        normalization: str = 'none',
        plow: float = 2.0,
        phigh: float = 98.0,
        floor_db: float = -60.0,
    ) -> None:
        # ---- validate dimension ----
        if dimension not in ('azimuth', 'range'):
            raise ValueError(
                f"dimension must be 'azimuth' or 'range', got {dimension!r}"
            )

        # ---- validate overlap ----
        if not 0.0 <= overlap < 1.0:
            raise ValueError(
                f"overlap must be in [0.0, 1.0), got {overlap}"
            )

        # ---- validate normalization ----
        if normalization not in ('none', 'log', 'percentile'):
            raise ValueError(
                f"normalization must be 'none', 'log', or 'percentile', "
                f"got {normalization!r}"
            )

        # ---- validate percentile ordering ----
        if plow >= phigh:
            raise ValueError(
                f"plow ({plow}) must be less than phigh ({phigh})"
            )

        self.dimension = dimension
        self.overlap = overlap
        self.deweight = deweight
        self.normalization = normalization
        self.plow = plow
        self.phigh = phigh
        self.floor_db = floor_db

        # Sub-aperture decomposition engine (validates metadata grid fields)
        self._sublook = SublookDecomposition(
            metadata=metadata,
            num_looks=_NUM_LOOKS,
            dimension=dimension,
            overlap=overlap,
            deweight=deweight,
        )

        # Platform look direction from SCPCOA (for RGB order correction)
        self._platform_direction: str = 'L'  # default left-looking
        if (metadata.scpcoa is not None
                and metadata.scpcoa.side_of_track is not None):
            self._platform_direction = metadata.scpcoa.side_of_track.upper()[0]

        # Normalization processors (grdl intensity primitives)
        self._to_db = ToDecibels(floor_db=floor_db)
        self._stretch = PercentileStretch(plow=plow, phigh=phigh)

    # ------------------------------------------------------------------
    # execute() protocol
    # ------------------------------------------------------------------

    def execute(
        self,
        metadata: 'ImageMetadata',
        source: np.ndarray,
        **kwargs: Any,
    ) -> tuple:
        """Execute CSI processing via the universal protocol.

        Parameters
        ----------
        metadata : ImageMetadata
            Input image metadata.
        source : np.ndarray
            Complex 2-D SAR image.

        Returns
        -------
        tuple[np.ndarray, ImageMetadata]
            ``(rgb_composite, updated_metadata)`` where the composite
            has shape ``(rows, cols, 3)``.
        """
        self._metadata = metadata
        result = self.apply(source, **kwargs)
        updated = dataclasses.replace(
            metadata,
            rows=result.shape[0],
            cols=result.shape[1],
            bands=3,
            dtype=str(result.dtype),
        )
        return result, updated

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Produce a CSI RGB composite from complex SAR imagery.

        Parameters
        ----------
        source : np.ndarray
            2D complex array, shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            RGB image, shape ``(rows, cols, 3)``.  When
            ``normalization='none'``, dtype is float64 with values in
            the same range as the original magnitude.  When
            ``normalization='log'`` or ``'percentile'``, dtype is
            float32 with values in [0, 1].

        Raises
        ------
        TypeError
            If *source* is not complex-valued.
        ValueError
            If *source* is not 2D.
        """
        params = self._resolve_params(kwargs)
        norm_mode = params['normalization']

        # 1. Sub-look decomposition (grdl SublookDecomposition)
        looks = self._sublook.decompose(source)  # (3, rows, cols) complex

        # 2. Detection -- magnitude of each sub-look
        mag = np.abs(looks)  # (3, rows, cols)

        # 3. Intensity preservation (sarpy approach)
        #    Scale RGB so max(R,G,B) = |original| at each pixel,
        #    preserving full spatial resolution in intensity.
        original_mag = np.abs(source)
        rgb_max = mag.max(axis=0)
        np.maximum(rgb_max, np.finfo(np.float64).tiny, out=rgb_max)
        scale = original_mag / rgb_max
        mag *= scale

        # 4. Composite -- stack as (rows, cols, 3) RGB
        rgb = np.dstack([mag[0], mag[1], mag[2]])

        # 5. Platform direction correction
        if self._platform_direction == 'R':
            rgb = rgb[:, :, ::-1]

        # 6. Optional normalization (grdl intensity processors)
        if norm_mode == 'none':
            return rgb

        if norm_mode == 'log':
            rgb = self._to_db.apply(
                rgb, floor_db=params['floor_db'],
            )

        return self._stretch.apply(
            rgb, plow=params['plow'], phigh=params['phigh'],
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def decompose_looks(self, source: np.ndarray) -> np.ndarray:
        """Return the raw sub-aperture looks without compositing.

        Useful for inspection or custom compositing logic.

        Parameters
        ----------
        source : np.ndarray
            2D complex SAR image.

        Returns
        -------
        np.ndarray
            Complex sub-look stack, shape ``(3, rows, cols)``.
        """
        return self._sublook.decompose(source)

    def __repr__(self) -> str:
        return (
            f"CSIProcessor(dimension={self.dimension!r}, "
            f"overlap={self.overlap}, "
            f"normalization={self.normalization!r}, "
            f"platform_direction={self._platform_direction!r})"
        )
