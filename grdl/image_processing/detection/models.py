# -*- coding: utf-8 -*-
"""
Detection Data Models - Sparse geo-registered detection output types.

Provides data models for image detector output: ``Detection`` for individual
detections carrying shapely geometries in pixel and geographic space, and
``DetectionSet`` for collections of detections from a single detector run.

Geometry is represented using ``shapely.geometry`` objects directly.
Detectors construct shapely geometries (``Point``, ``Polygon``,
``box()``, etc.) and pass them to ``Detection``.

Coordinate Conventions
----------------------
- **Pixel space**: shapely ``(x, y)`` = ``(col, row)``.
  Origin at top-left of image.
- **Geographic space**: shapely ``(x, y)`` = ``(longitude, latitude)``
  in WGS84.  This matches the GeoJSON coordinate order.

Detection properties should use hierarchical field names from the GRDL
data dictionary (see ``grdl.image_processing.detection.fields``).

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
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Third-party
from shapely.geometry import mapping as shapely_mapping

# GRDL internal
from grdl.image_processing.detection.fields import is_dictionary_field


class Detection:
    """
    A single sparse geo-registered detection.

    Represents a discrete feature detected in imagery: a point, bounding
    box, or polygon with associated properties.  Geometry is stored as
    shapely objects in both pixel and geographic coordinate spaces.

    Parameters
    ----------
    pixel_geometry : shapely.geometry.base.BaseGeometry
        Geometry in pixel space.  Coordinate convention:
        ``(x, y)`` = ``(col, row)`` with origin at top-left.
    properties : Dict[str, Any]
        Detection attributes.  Keys should use names from the GRDL
        data dictionary (e.g., ``'sar.change_magnitude'``).
    confidence : float, optional
        Overall detection confidence in [0, 1].  None if not applicable.
    geo_geometry : shapely.geometry.base.BaseGeometry, optional
        Geometry in geographic space.  Convention:
        ``(x, y)`` = ``(longitude, latitude)`` in WGS84.
        Populated by geo-registration; None before registration.
    """

    def __init__(
        self,
        pixel_geometry: Any,
        properties: Dict[str, Any],
        confidence: Optional[float] = None,
        geo_geometry: Any = None,
    ) -> None:
        self.pixel_geometry = pixel_geometry
        self.geo_geometry = geo_geometry
        self.properties = properties
        self.confidence = confidence

    def to_geojson_feature(self) -> Dict[str, Any]:
        """
        Convert to a GeoJSON Feature dictionary.

        Uses geographic geometry if available, otherwise falls back
        to pixel geometry.  Geometry serialization is handled by
        ``shapely.geometry.mapping()``.

        Returns
        -------
        Dict[str, Any]
            GeoJSON Feature with ``'type'``, ``'geometry'``, and
            ``'properties'`` keys.  Confidence is included in
            properties if set.
        """
        geom = (
            self.geo_geometry
            if self.geo_geometry is not None
            else self.pixel_geometry
        )
        props = dict(self.properties)
        if self.confidence is not None:
            props['confidence'] = self.confidence
        return {
            'type': 'Feature',
            'geometry': shapely_mapping(geom),
            'properties': props,
        }

    def __repr__(self) -> str:
        geom_type = self.pixel_geometry.geom_type
        return (
            f"Detection(geom_type={geom_type!r}, "
            f"confidence={self.confidence!r})"
        )


class DetectionSet:
    """
    Collection of detections from a single detector run.

    Bundles the detection results with metadata about the detector that
    produced them: its name, version, and declared output fields.
    Supports iteration, indexing, and export to GeoJSON FeatureCollection.

    Parameters
    ----------
    detections : List[Detection]
        Individual detections.
    detector_name : str
        Name of the detector class that produced these results.
    detector_version : str
        Processor version of the detector (from ``@processor_version``).
    output_fields : Tuple[str, ...], optional
        Field names declared by the detector.  Names from the GRDL data
        dictionary are strongly encouraged; non-dictionary names will
        emit a ``UserWarning``.
    metadata : Dict[str, Any], optional
        Additional run metadata (e.g., processing time, input shape).
    """

    def __init__(
        self,
        detections: List[Detection],
        detector_name: str,
        detector_version: str,
        output_fields: Tuple[str, ...] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.detections = detections
        self.detector_name = detector_name
        self.detector_version = detector_version
        self.output_fields = output_fields
        self.metadata = metadata or {}

        # Warn on non-dictionary field names
        for name in self.output_fields:
            if not is_dictionary_field(name):
                warnings.warn(
                    f"Field '{name}' is not in the GRDL data dictionary. "
                    f"Consider using a standardized field name.",
                    UserWarning,
                    stacklevel=2,
                )

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self):
        return iter(self.detections)

    def __getitem__(self, index: int) -> Detection:
        return self.detections[index]

    def to_geojson(self) -> Dict[str, Any]:
        """
        Convert to GeoJSON FeatureCollection.

        Includes detector metadata in the top-level ``'properties'`` key.

        Returns
        -------
        Dict[str, Any]
            GeoJSON FeatureCollection with ``'type'``, ``'features'``,
            and ``'properties'`` keys.
        """
        return {
            'type': 'FeatureCollection',
            'features': [d.to_geojson_feature() for d in self.detections],
            'properties': {
                'detector_name': self.detector_name,
                'detector_version': self.detector_version,
                'output_fields': self.output_fields,
                **self.metadata,
            },
        }

    def filter_by_confidence(
        self, min_confidence: float
    ) -> 'DetectionSet':
        """
        Return a new DetectionSet with detections above the threshold.

        Detections without a confidence value are excluded.

        Parameters
        ----------
        min_confidence : float
            Minimum confidence threshold in [0, 1].

        Returns
        -------
        DetectionSet
            Filtered detection set with the same metadata.
        """
        filtered = [
            d for d in self.detections
            if d.confidence is not None and d.confidence >= min_confidence
        ]
        return DetectionSet(
            detections=filtered,
            detector_name=self.detector_name,
            detector_version=self.detector_version,
            output_fields=self.output_fields,
            metadata=self.metadata,
        )

    def __repr__(self) -> str:
        return (
            f"DetectionSet(detector={self.detector_name!r}, "
            f"version={self.detector_version!r}, "
            f"count={len(self.detections)})"
        )
