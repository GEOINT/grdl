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
2026-02-06
"""

# Standard library
from abc import abstractmethod
from typing import Any, Optional, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageProcessor
from grdl.image_processing.detection.models import DetectionSet, OutputSchema

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
    - ``output_schema`` (property) -- declare the output format

    The base class provides:

    - ``_geo_register_detections()`` -- batch-transform pixel coordinates
      to geographic coordinates using a ``Geolocation`` object

    Examples
    --------
    >>> detector = SomeDetector(threshold=0.5)
    >>> detections = detector.detect(image, geolocation=geo)
    >>> len(detections)
    42
    >>> detections.output_schema.field_names
    ('label', 'label_confidence')
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
    def output_schema(self) -> OutputSchema:
        """
        Declare the output schema for this detector.

        The schema describes what fields appear in each detection's
        ``properties`` dictionary. This enables downstream consumers
        to inspect the output format without running the detector.

        Returns
        -------
        OutputSchema
            Schema describing detection properties.
        """
        ...

    def _geo_register_detections(
        self,
        detections: DetectionSet,
        geolocation: 'Geolocation',
    ) -> None:
        """
        Transform pixel coordinates to geographic coordinates in-place.

        Collects all pixel coordinates from the detection geometries,
        batch-transforms them via ``geolocation.pixel_to_latlon()``,
        and populates each detection's geographic coordinates.

        Parameters
        ----------
        detections : DetectionSet
            Detections with pixel coordinates to geo-register.
        geolocation : Geolocation
            Coordinate transform to apply.
        """
        if len(detections) == 0:
            return

        for detection in detections:
            geom = detection.geometry

            if geom.geometry_type == 'Point':
                row, col = float(geom.pixel_coordinates[0]), float(geom.pixel_coordinates[1])
                lat, lon, _ = geolocation.pixel_to_latlon(row, col)
                geom.geographic_coordinates = np.array(
                    [lat, lon], dtype=np.float64
                )

            elif geom.geometry_type == 'BoundingBox':
                rows = np.array([
                    geom.pixel_coordinates[0],
                    geom.pixel_coordinates[2],
                ], dtype=np.float64)
                cols = np.array([
                    geom.pixel_coordinates[1],
                    geom.pixel_coordinates[3],
                ], dtype=np.float64)
                lats, lons, _ = geolocation.pixel_to_latlon(rows, cols)
                geom.geographic_coordinates = np.array([
                    float(np.min(lats)), float(np.min(lons)),
                    float(np.max(lats)), float(np.max(lons)),
                ], dtype=np.float64)

            elif geom.geometry_type == 'Polygon':
                rows = geom.pixel_coordinates[:, 0]
                cols = geom.pixel_coordinates[:, 1]
                lats, lons, _ = geolocation.pixel_to_latlon(rows, cols)
                geom.geographic_coordinates = np.column_stack([lats, lons])
