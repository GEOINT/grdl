# -*- coding: utf-8 -*-
"""
Detection Data Models - Sparse geo-registered detection output types.

Provides data models for image detector output: ``Geometry`` for spatial
locations in pixel and geographic space, ``OutputField`` and ``OutputSchema``
for self-declared output formats, ``Detection`` for individual detections,
and ``DetectionSet`` for collections of detections from a single detector run.

All geometry coordinates are stored as numpy arrays. Geographic coordinates
follow GeoJSON conventions: (longitude, latitude) ordering.

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
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import numpy as np


# Valid geometry types
_VALID_GEOMETRY_TYPES = {'Point', 'BoundingBox', 'Polygon'}

# Valid output field dtypes
_VALID_DTYPES = {'float', 'int', 'str', 'bool'}


class Geometry:
    """
    Spatial location of a detection in pixel and geographic space.

    Stores both pixel-space coordinates (row/col) and optionally
    geographic-space coordinates (lat/lon). Geographic coordinates
    are populated by geo-registration after detection.

    Coordinate Conventions
    ----------------------
    - **Pixel coordinates**: (row, col) with (0, 0) at top-left.
      Stored as ndarray with columns [row, col].
    - **Geographic coordinates**: (latitude, longitude) in WGS84.
      Stored as ndarray with columns [lat, lon].
      GeoJSON export uses (lon, lat) ordering per the GeoJSON spec.

    Parameters
    ----------
    geometry_type : str
        One of ``'Point'``, ``'BoundingBox'``, or ``'Polygon'``.
    pixel_coordinates : np.ndarray
        Pixel-space coordinates. Shape depends on geometry type:

        - Point: ``(2,)`` as ``[row, col]``
        - BoundingBox: ``(4,)`` as ``[row_min, col_min, row_max, col_max]``
        - Polygon: ``(N, 2)`` as ``[[row, col], ...]`` vertex array
    geographic_coordinates : np.ndarray, optional
        Geographic coordinates matching the pixel coordinates.
        Same shape conventions but with ``[lat, lon]`` instead of
        ``[row, col]``. Populated by geo-registration.
    """

    def __init__(
        self,
        geometry_type: str,
        pixel_coordinates: np.ndarray,
        geographic_coordinates: Optional[np.ndarray] = None,
    ) -> None:
        if geometry_type not in _VALID_GEOMETRY_TYPES:
            raise ValueError(
                f"geometry_type must be one of {_VALID_GEOMETRY_TYPES}, "
                f"got {geometry_type!r}"
            )
        if not isinstance(pixel_coordinates, np.ndarray):
            raise TypeError(
                f"pixel_coordinates must be a numpy ndarray, "
                f"got {type(pixel_coordinates).__name__}"
            )

        self.geometry_type = geometry_type
        self.pixel_coordinates = pixel_coordinates
        self.geographic_coordinates = geographic_coordinates

    @classmethod
    def point(
        cls,
        row: float,
        col: float,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> 'Geometry':
        """
        Create a point geometry.

        Parameters
        ----------
        row : float
            Pixel row coordinate.
        col : float
            Pixel column coordinate.
        lat : float, optional
            Latitude in degrees (WGS84).
        lon : float, optional
            Longitude in degrees (WGS84).

        Returns
        -------
        Geometry
            Point geometry.
        """
        pixel = np.array([row, col], dtype=np.float64)
        geo = None
        if lat is not None and lon is not None:
            geo = np.array([lat, lon], dtype=np.float64)
        return cls('Point', pixel, geo)

    @classmethod
    def bounding_box(
        cls,
        row_min: float,
        col_min: float,
        row_max: float,
        col_max: float,
        corners_latlon: Optional[np.ndarray] = None,
    ) -> 'Geometry':
        """
        Create a bounding box geometry.

        Parameters
        ----------
        row_min : float
            Top edge row.
        col_min : float
            Left edge column.
        row_max : float
            Bottom edge row.
        col_max : float
            Right edge column.
        corners_latlon : np.ndarray, optional
            Geographic corners as ``(4,)`` array
            ``[lat_min, lon_min, lat_max, lon_max]``.

        Returns
        -------
        Geometry
            Bounding box geometry.
        """
        pixel = np.array(
            [row_min, col_min, row_max, col_max], dtype=np.float64
        )
        return cls('BoundingBox', pixel, corners_latlon)

    @classmethod
    def polygon(
        cls,
        pixel_vertices: np.ndarray,
        geo_vertices: Optional[np.ndarray] = None,
    ) -> 'Geometry':
        """
        Create a polygon geometry.

        Parameters
        ----------
        pixel_vertices : np.ndarray
            Vertex array, shape ``(N, 2)`` as ``[[row, col], ...]``.
            The polygon is implicitly closed (first vertex need not
            be repeated).
        geo_vertices : np.ndarray, optional
            Geographic vertex array, shape ``(N, 2)`` as
            ``[[lat, lon], ...]``.

        Returns
        -------
        Geometry
            Polygon geometry.
        """
        vertices = np.asarray(pixel_vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError(
                f"pixel_vertices must have shape (N, 2), "
                f"got {vertices.shape}"
            )
        geo = None
        if geo_vertices is not None:
            geo = np.asarray(geo_vertices, dtype=np.float64)
        return cls('Polygon', vertices, geo)

    def to_geojson(self) -> Dict[str, Any]:
        """
        Convert to GeoJSON geometry dictionary.

        Uses geographic coordinates if available, otherwise falls back
        to pixel coordinates. GeoJSON uses (longitude, latitude) ordering.

        Returns
        -------
        Dict[str, Any]
            GeoJSON-compatible geometry dictionary with ``'type'`` and
            ``'coordinates'`` keys.
        """
        coords = self.geographic_coordinates
        use_geo = coords is not None

        if self.geometry_type == 'Point':
            if use_geo:
                # GeoJSON: [lon, lat]
                return {
                    'type': 'Point',
                    'coordinates': [float(coords[1]), float(coords[0])],
                }
            pixel = self.pixel_coordinates
            return {
                'type': 'Point',
                'coordinates': [float(pixel[1]), float(pixel[0])],
            }

        elif self.geometry_type == 'BoundingBox':
            # GeoJSON represents bounding boxes as Polygons
            if use_geo:
                lat_min, lon_min, lat_max, lon_max = (
                    float(coords[0]), float(coords[1]),
                    float(coords[2]), float(coords[3]),
                )
            else:
                pixel = self.pixel_coordinates
                lat_min, lon_min = float(pixel[0]), float(pixel[1])
                lat_max, lon_max = float(pixel[2]), float(pixel[3])

            ring = [
                [lon_min, lat_min],
                [lon_max, lat_min],
                [lon_max, lat_max],
                [lon_min, lat_max],
                [lon_min, lat_min],
            ]
            return {'type': 'Polygon', 'coordinates': [ring]}

        else:  # Polygon
            if use_geo:
                verts = coords
            else:
                verts = self.pixel_coordinates

            # GeoJSON: list of [lon, lat] with closure
            ring = [[float(v[1]), float(v[0])] for v in verts]
            if ring and ring[0] != ring[-1]:
                ring.append(ring[0])
            return {'type': 'Polygon', 'coordinates': [ring]}

    def __repr__(self) -> str:
        return (
            f"Geometry(type={self.geometry_type!r}, "
            f"pixel={self.pixel_coordinates!r}, "
            f"geo={'set' if self.geographic_coordinates is not None else 'None'})"
        )


class OutputField:
    """
    Declaration of a single output field in a detector's schema.

    Each detector declares what properties its detections will carry.
    An ``OutputField`` describes one such property: its name, data type,
    human-readable description, and optional physical units.

    Parameters
    ----------
    name : str
        Field name (e.g., ``'change_magnitude'``, ``'label'``).
    dtype : str
        Data type: ``'float'``, ``'int'``, ``'str'``, or ``'bool'``.
    description : str
        Human-readable description of the field.
    units : str, optional
        Physical units (e.g., ``'dB'``, ``'meters'``). None for
        dimensionless or non-numeric fields.

    Raises
    ------
    ValueError
        If ``dtype`` is not one of the valid types.
    """

    def __init__(
        self,
        name: str,
        dtype: str,
        description: str,
        units: Optional[str] = None,
    ) -> None:
        if dtype not in _VALID_DTYPES:
            raise ValueError(
                f"dtype must be one of {_VALID_DTYPES}, got {dtype!r}"
            )
        self.name = name
        self.dtype = dtype
        self.description = description
        self.units = units

    def __repr__(self) -> str:
        parts = f"OutputField(name={self.name!r}, dtype={self.dtype!r}"
        if self.units is not None:
            parts += f", units={self.units!r}"
        return parts + ")"


class OutputSchema:
    """
    Self-declared output schema for a detector.

    A detector declares its output schema as a tuple of ``OutputField``
    instances. This enables downstream consumers to inspect what a detector
    produces without running it. The schema version is the processor's
    ``@processor_version`` -- there is no separate schema version.

    Parameters
    ----------
    fields : Tuple[OutputField, ...]
        Ordered tuple of field declarations.

    Examples
    --------
    Phenomenological detector schema:

    >>> schema = OutputSchema(fields=(
    ...     OutputField('change_magnitude', 'float', 'Change in dB', units='dB'),
    ...     OutputField('coherence_loss', 'float', 'Coherence loss ratio'),
    ... ))

    Classification detector schema:

    >>> schema = OutputSchema(fields=(
    ...     OutputField('label', 'str', 'Semantic class label'),
    ...     OutputField('label_confidence', 'float', 'Classification confidence'),
    ... ))
    """

    def __init__(self, fields: Tuple[OutputField, ...]) -> None:
        self.fields = fields

    @property
    def field_names(self) -> Tuple[str, ...]:
        """
        Ordered tuple of field names in this schema.

        Returns
        -------
        Tuple[str, ...]
            Field names.
        """
        return tuple(f.name for f in self.fields)

    def validate_properties(self, properties: Dict[str, Any]) -> None:
        """
        Validate a properties dictionary against this schema.

        Checks that all declared fields are present and that values
        match the declared types.

        Parameters
        ----------
        properties : Dict[str, Any]
            Properties dictionary from a ``Detection``.

        Raises
        ------
        ValueError
            If required fields are missing or values have wrong types.
        """
        _dtype_to_types = {
            'float': (int, float, np.integer, np.floating),
            'int': (int, np.integer),
            'str': (str,),
            'bool': (bool, np.bool_),
        }

        for field in self.fields:
            if field.name not in properties:
                raise ValueError(
                    f"Missing required field '{field.name}' "
                    f"(expected by output schema)"
                )
            value = properties[field.name]
            expected_types = _dtype_to_types[field.dtype]
            if not isinstance(value, expected_types):
                raise ValueError(
                    f"Field '{field.name}' has type "
                    f"{type(value).__name__}, expected {field.dtype}"
                )

    def __repr__(self) -> str:
        return f"OutputSchema(fields={self.field_names!r})"


class Detection:
    """
    A single sparse geo-registered detection.

    Represents a discrete feature detected in imagery: a point, bounding
    box, or polygon with associated properties declared by the detector's
    output schema.

    Parameters
    ----------
    geometry : Geometry
        Spatial location (pixel coordinates, optionally geo-registered).
    properties : Dict[str, Any]
        Detection properties. Keys must match the detector's
        ``OutputSchema`` field names.
    confidence : float, optional
        Overall detection confidence in [0, 1]. None if not applicable.
    """

    def __init__(
        self,
        geometry: Geometry,
        properties: Dict[str, Any],
        confidence: Optional[float] = None,
    ) -> None:
        self.geometry = geometry
        self.properties = properties
        self.confidence = confidence

    def to_geojson_feature(self) -> Dict[str, Any]:
        """
        Convert to a GeoJSON Feature dictionary.

        Returns
        -------
        Dict[str, Any]
            GeoJSON Feature with ``'type'``, ``'geometry'``, and
            ``'properties'`` keys. Confidence is included in properties
            if set.
        """
        props = dict(self.properties)
        if self.confidence is not None:
            props['confidence'] = self.confidence
        return {
            'type': 'Feature',
            'geometry': self.geometry.to_geojson(),
            'properties': props,
        }

    def __repr__(self) -> str:
        return (
            f"Detection(geometry={self.geometry.geometry_type!r}, "
            f"confidence={self.confidence!r})"
        )


class DetectionSet:
    """
    Collection of detections from a single detector run.

    Bundles the detection results with metadata about the detector that
    produced them: its name, version, and output schema. Supports
    iteration, indexing, and export to GeoJSON FeatureCollection.

    Parameters
    ----------
    detections : List[Detection]
        Individual detections.
    detector_name : str
        Name of the detector class that produced these results.
    detector_version : str
        Processor version of the detector (from ``@processor_version``).
    output_schema : OutputSchema
        Schema describing the properties of each detection.
    metadata : Dict[str, Any], optional
        Additional run metadata (e.g., processing time, input shape).
    """

    def __init__(
        self,
        detections: List[Detection],
        detector_name: str,
        detector_version: str,
        output_schema: OutputSchema,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.detections = detections
        self.detector_name = detector_name
        self.detector_version = detector_version
        self.output_schema = output_schema
        self.metadata = metadata or {}

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
                'schema_fields': self.output_schema.field_names,
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
            output_schema=self.output_schema,
            metadata=self.metadata,
        )

    def __repr__(self) -> str:
        return (
            f"DetectionSet(detector={self.detector_name!r}, "
            f"version={self.detector_version!r}, "
            f"count={len(self.detections)})"
        )
