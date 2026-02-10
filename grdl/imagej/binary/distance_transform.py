# -*- coding: utf-8 -*-
"""
Distance Transform (EDM) - Port of ImageJ's Process > Binary > Distance Map.

Computes the Euclidean Distance Map (EDM) of a binary image. Each
foreground pixel is assigned a value equal to its Euclidean distance
to the nearest background pixel. This is a fundamental building block
for watershed segmentation, Voronoi tessellation, and shape analysis.

Particularly useful for:
- Pre-processing for watershed segmentation of touching objects in SAR
- Shape-based feature extraction from classified PAN/EO masks
- Computing buffer zones around detected targets in any modality
- Thinning and skeletonization preprocessing
- Voronoi partitioning of labeled regions in MSI/HSI classification
- Proximity analysis for thermal hotspot clustering

Attribution
-----------
ImageJ implementation: Wayne Rasband (NIH).
Source: ``ij/plugin/filter/EDM.java`` in ImageJ 1.54j.
ImageJ 1.x source is in the public domain.

The ImageJ EDM uses a two-pass raster scanning algorithm. This port
delegates to ``scipy.ndimage.distance_transform_edt`` which implements
the exact Euclidean distance transform via the Saito & Toriwaki
algorithm, producing identical results.

Dependencies
------------
scipy

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
from typing import Any, Optional, Tuple

# Third-party
import numpy as np
from scipy.ndimage import distance_transform_edt

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.BINARY)
@processor_version('1.54j')
class DistanceTransform(ImageTransform):
    """Euclidean Distance Map, ported from ImageJ 1.54j.

    Computes the Euclidean distance from each foreground pixel to
    the nearest background pixel. Background pixels receive a value
    of 0.

    Parameters
    ----------
    normalize : bool
        If True, normalize the output to [0, 1] by dividing by the
        maximum distance. Default is False (raw pixel distances),
        matching ImageJ's behavior.
    pixel_size : tuple of float, optional
        Physical pixel dimensions ``(row_spacing, col_spacing)`` for
        anisotropic images. Default is ``(1.0, 1.0)`` (isotropic).
        Useful when SAR images have different azimuth and range
        pixel spacings.

    Notes
    -----
    Port of ``ij/plugin/filter/EDM.java`` from ImageJ 1.54j
    (public domain). Original by Wayne Rasband.

    The input should be a binary image (0 = background, nonzero =
    foreground). Non-binary images are thresholded at > 0.

    Examples
    --------
    Basic distance map:

    >>> from grdl.imagej import DistanceTransform
    >>> dt = DistanceTransform()
    >>> edm = dt.apply(binary_mask)

    Normalized distance for shape analysis:

    >>> dt = DistanceTransform(normalize=True)
    >>> edm_norm = dt.apply(building_mask)

    Anisotropic SAR (5m azimuth, 2m range):

    >>> dt = DistanceTransform(pixel_size=(5.0, 2.0))
    >>> edm = dt.apply(ship_mask)
    """

    __imagej_source__ = 'ij/plugin/filter/EDM.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = True

    def __init__(
        self,
        normalize: bool = False,
        pixel_size: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.normalize = normalize
        if pixel_size is not None:
            if len(pixel_size) != 2:
                raise ValueError(
                    f"pixel_size must have 2 elements, got {len(pixel_size)}"
                )
            if any(p <= 0 for p in pixel_size):
                raise ValueError(
                    f"pixel_size values must be > 0, got {pixel_size}"
                )
            self.pixel_size = tuple(float(p) for p in pixel_size)
        else:
            self.pixel_size = (1.0, 1.0)

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the Euclidean Distance Map of a binary image.

        Parameters
        ----------
        source : np.ndarray
            2D binary image. Shape ``(rows, cols)``. Non-zero pixels
            are foreground.

        Returns
        -------
        np.ndarray
            Distance map, dtype float64, same shape as input.
            Foreground pixels have positive values; background = 0.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(
                f"Expected 2D image, got shape {source.shape}"
            )

        mask = source.astype(np.float64) > 0

        if not mask.any():
            return np.zeros_like(source, dtype=np.float64)

        edt = distance_transform_edt(mask, sampling=self.pixel_size)

        if self.normalize:
            edt_max = edt.max()
            if edt_max > 0:
                edt = edt / edt_max

        return edt
