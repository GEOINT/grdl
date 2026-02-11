# -*- coding: utf-8 -*-
"""
Image Detector Base Class - Abstract interface for sparse vector detectors.

Defines the ``ImageDetector`` ABC for image processors that produce sparse
geo-registered vector detections (points, bounding boxes, polygons) rather
than dense raster arrays. Inherits from ``ImageProcessor`` for version
checking and detection input flow.

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
2026-02-06

Modified
--------
2026-02-11
"""

# Standard library
from abc import abstractmethod
from typing import Any, Optional, Tuple, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageProcessor
from grdl.image_processing.detection.models import DetectionSet

if TYPE_CHECKING:
    from grdl.geolocation.base import Geolocation


class ImageDetector(ImageProcessor):
    """
    Abstract base class for image detectors producing sparse vector outputs.

    Unlike ``ImageTransform`` (ndarray in, ndarray out), an ``ImageDetector``
    produces a ``DetectionSet`` of sparse geo-registered features: points,
    bounding boxes, or polygons in both pixel and geographic coordinates.

    Input is a numpy array (image data) plus an optional ``Geolocation``
    for pixel-to-geographic coordinate transforms. If no ``Geolocation``
    is provided, detections contain only pixel coordinates.

    Detection inputs from upstream detectors can be passed through
    ``**kwargs`` -- see ``detection_input_specs`` on ``ImageProcessor``.

    Subclasses must implement:

    - ``detect()`` -- run detection on image data
    - ``output_fields`` (property) -- declare the output field names

    Examples
    --------
    >>> detector = SomeDetector(threshold=0.5)
    >>> detections = detector.detect(image, geolocation=geo)
    >>> len(detections)
    42
    >>> detections.output_fields
    ('sar.change_magnitude', 'identity.label')
    """

    @abstractmethod
    def detect(
        self,
        source: np.ndarray,
        geolocation: Optional['Geolocation'] = None,
        **kwargs: Any,
    ) -> DetectionSet:
        """
        Run detection on source imagery.

        Parameters
        ----------
        source : np.ndarray
            Input image. Shape depends on the specific detector:
            ``(rows, cols)`` for single-band, ``(bands, rows, cols)`` or
            ``(rows, cols, bands)`` for multi-band.
        geolocation : Geolocation, optional
            Geolocation object for pixel-to-geographic transforms.
            If None, detections will contain pixel coordinates only.

        Returns
        -------
        DetectionSet
            Collection of sparse detections with metadata.
        """
        ...

    @property
    @abstractmethod
    def output_fields(self) -> Tuple[str, ...]:
        """
        Declare the output field names for this detector.

        Returns field names that appear in each detection's ``properties``
        dictionary. Names from the GRDL data dictionary are strongly
        encouraged (e.g., ``'sar.change_magnitude'``). Custom names are
        permitted but will generate a ``UserWarning`` when the
        ``DetectionSet`` is created.

        Returns
        -------
        Tuple[str, ...]
            Field name strings.
        """
        ...

