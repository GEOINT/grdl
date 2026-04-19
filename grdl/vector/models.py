# -*- coding: utf-8 -*-
"""
Vector Data Models - Generic geo-registered feature representations.

Provides data models for vector feature data: ``Feature`` for individual
features carrying shapely geometries with properties, ``FieldSchema`` for
describing feature attribute schemas, and ``FeatureSet`` for collections
of features with CRS and schema metadata.

Bridges to/from ``DetectionSet`` and optionally ``GeoDataFrame``.

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-03-25
"""

# Standard library
import json
import uuid
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

# Third-party
from shapely.geometry import (
    box as shapely_box,
    mapping as shapely_mapping,
    shape as shapely_shape,
)
from shapely.geometry.base import BaseGeometry
import shapely


class Feature:
    """
    A single geo-registered feature.

    Represents a discrete geographic feature: a point, line, or polygon
    with associated properties.  Geometry is stored as a shapely object.

    Parameters
    ----------
    geometry : shapely.geometry.base.BaseGeometry
        Feature geometry in the coordinate reference system of the
        containing ``FeatureSet``.
    properties : Dict[str, Any]
        Feature attributes as key-value pairs.
    feature_id : str, optional
        Unique feature identifier.  Auto-generated UUID if not provided.
    """

    def __init__(
        self,
        geometry: BaseGeometry,
        properties: Optional[Dict[str, Any]] = None,
        feature_id: Optional[str] = None,
    ) -> None:
        self.geometry = geometry
        self.properties = properties if properties is not None else {}
        self.id = feature_id if feature_id is not None else str(uuid.uuid4())

    def to_geojson_feature(self) -> Dict[str, Any]:
        """
        Convert to a GeoJSON Feature dictionary.

        Returns
        -------
        Dict[str, Any]
            GeoJSON Feature with ``'type'``, ``'geometry'``,
            ``'properties'``, and ``'id'`` keys.
        """
        return {
            'type': 'Feature',
            'id': self.id,
            'geometry': shapely_mapping(self.geometry),
            'properties': dict(self.properties),
        }

    @classmethod
    def from_geojson_feature(cls, feature_dict: Dict[str, Any]) -> 'Feature':
        """
        Create a Feature from a GeoJSON Feature dictionary.

        Parameters
        ----------
        feature_dict : Dict[str, Any]
            GeoJSON Feature dictionary.

        Returns
        -------
        Feature
        """
        geometry = shapely_shape(feature_dict['geometry'])
        properties = dict(feature_dict.get('properties', {}))
        feature_id = feature_dict.get('id')
        return cls(
            geometry=geometry,
            properties=properties,
            feature_id=str(feature_id) if feature_id is not None else None,
        )

    def __repr__(self) -> str:
        geom_type = self.geometry.geom_type
        return (
            f"Feature(id={self.id!r}, geom_type={geom_type!r}, "
            f"properties={len(self.properties)} fields)"
        )


class FieldSchema:
    """
    Schema definition for a single feature attribute field.

    Parameters
    ----------
    name : str
        Field name.
    dtype : str
        Data type string (e.g., ``'str'``, ``'float'``, ``'int'``,
        ``'bool'``).
    description : str, optional
        Human-readable field description.
    nullable : bool, optional
        Whether the field allows None values.  Default True.
    """

    def __init__(
        self,
        name: str,
        dtype: str,
        description: str = '',
        nullable: bool = True,
    ) -> None:
        self.name = name
        self.dtype = dtype
        self.description = description
        self.nullable = nullable

    def __repr__(self) -> str:
        return (
            f"FieldSchema(name={self.name!r}, dtype={self.dtype!r}, "
            f"nullable={self.nullable!r})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            'name': self.name,
            'dtype': self.dtype,
            'description': self.description,
            'nullable': self.nullable,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FieldSchema':
        """Deserialize from a plain dictionary."""
        return cls(
            name=d['name'],
            dtype=d['dtype'],
            description=d.get('description', ''),
            nullable=d.get('nullable', True),
        )


class FeatureSet:
    """
    Collection of features with CRS, schema, and metadata.

    Bundles a list of ``Feature`` objects with coordinate reference system
    information and optional field schema declarations.  Supports iteration,
    indexing, filtering, GeoJSON serialization, and bridging to/from
    ``DetectionSet`` and ``GeoDataFrame``.

    Parameters
    ----------
    features : List[Feature]
        Individual features.
    crs : str, optional
        Coordinate reference system identifier (e.g., ``'EPSG:4326'``).
        Default ``'EPSG:4326'``.
    schema : List[FieldSchema], optional
        Schema definitions for feature properties.
    metadata : Dict[str, Any], optional
        Additional metadata.
    """

    def __init__(
        self,
        features: List[Feature],
        crs: str = 'EPSG:4326',
        schema: Optional[List[FieldSchema]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.features = features
        self.crs = crs
        self.schema = schema if schema is not None else []
        self.metadata = metadata if metadata is not None else {}

    # -----------------------------------------------------------------
    # Collection protocol
    # -----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.features)

    def __iter__(self):
        return iter(self.features)

    def __getitem__(self, index: int) -> Feature:
        return self.features[index]

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------
    @property
    def count(self) -> int:
        """Number of features in the set."""
        return len(self.features)

    @property
    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Combined bounding box of all features as (minx, miny, maxx, maxy).

        Returns None if the set is empty.
        """
        if not self.features:
            return None
        all_bounds = [f.geometry.bounds for f in self.features]
        return (
            min(b[0] for b in all_bounds),
            min(b[1] for b in all_bounds),
            max(b[2] for b in all_bounds),
            max(b[3] for b in all_bounds),
        )

    @property
    def geometry_types(self) -> Set[str]:
        """Set of distinct geometry type names."""
        return {f.geometry.geom_type for f in self.features}

    # -----------------------------------------------------------------
    # Accessors
    # -----------------------------------------------------------------
    def get_geometries(self) -> List[BaseGeometry]:
        """
        Return a list of all feature geometries.

        Returns
        -------
        List[BaseGeometry]
        """
        return [f.geometry for f in self.features]

    def get_property_array(self, name: str) -> List[Any]:
        """
        Return values of a named property across all features.

        Parameters
        ----------
        name : str
            Property key name.

        Returns
        -------
        List[Any]
            Property values in feature order.  None where the key
            is absent from a feature's properties.
        """
        return [f.properties.get(name) for f in self.features]

    # -----------------------------------------------------------------
    # Filtering
    # -----------------------------------------------------------------
    def filter_by_bbox(
        self,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
    ) -> 'FeatureSet':
        """
        Return features whose geometry intersects the bounding box.

        Parameters
        ----------
        minx, miny, maxx, maxy : float
            Bounding box coordinates.

        Returns
        -------
        FeatureSet
            Filtered feature set with same CRS and schema.
        """
        bbox = shapely_box(minx, miny, maxx, maxy)
        filtered = [f for f in self.features if f.geometry.intersects(bbox)]
        return FeatureSet(
            features=filtered,
            crs=self.crs,
            schema=self.schema,
            metadata=self.metadata,
        )

    def filter_by_property(
        self,
        name: str,
        value: Any,
    ) -> 'FeatureSet':
        """
        Return features where a property matches the given value.

        Parameters
        ----------
        name : str
            Property key.
        value : Any
            Value to match.

        Returns
        -------
        FeatureSet
        """
        filtered = [
            f for f in self.features
            if f.properties.get(name) == value
        ]
        return FeatureSet(
            features=filtered,
            crs=self.crs,
            schema=self.schema,
            metadata=self.metadata,
        )

    def filter_by_geometry(
        self,
        geometry: BaseGeometry,
        predicate: str = 'intersects',
    ) -> 'FeatureSet':
        """
        Return features satisfying a spatial predicate against a geometry.

        Parameters
        ----------
        geometry : BaseGeometry
            Reference geometry.
        predicate : str
            Spatial predicate: ``'intersects'``, ``'contains'``,
            ``'within'``.  Default ``'intersects'``.

        Returns
        -------
        FeatureSet
        """
        pred_fn = getattr(geometry, predicate, None)
        if pred_fn is None:
            raise ValueError(f"Unknown spatial predicate: {predicate!r}")
        # For 'contains' and 'within', swap the logic correctly
        if predicate == 'contains':
            filtered = [f for f in self.features if geometry.contains(f.geometry)]
        elif predicate == 'within':
            filtered = [f for f in self.features if f.geometry.within(geometry)]
        else:
            filtered = [f for f in self.features if getattr(f.geometry, predicate)(geometry)]
        return FeatureSet(
            features=filtered,
            crs=self.crs,
            schema=self.schema,
            metadata=self.metadata,
        )

    # -----------------------------------------------------------------
    # GeoJSON serialization
    # -----------------------------------------------------------------
    def to_geojson(self) -> Dict[str, Any]:
        """
        Convert to a GeoJSON FeatureCollection dictionary.

        Includes CRS and schema in top-level properties.

        Returns
        -------
        Dict[str, Any]
            GeoJSON FeatureCollection.
        """
        result = {
            'type': 'FeatureCollection',
            'features': [f.to_geojson_feature() for f in self.features],
        }
        # Store metadata in top-level properties
        props: Dict[str, Any] = {}
        if self.crs:
            props['crs'] = self.crs
        if self.schema:
            props['schema'] = [s.to_dict() for s in self.schema]
        if self.metadata:
            props.update(self.metadata)
        if props:
            result['properties'] = props
        return result

    @classmethod
    def from_geojson(cls, geojson: Dict[str, Any]) -> 'FeatureSet':
        """
        Create a FeatureSet from a GeoJSON FeatureCollection dictionary.

        Parameters
        ----------
        geojson : Dict[str, Any]
            GeoJSON FeatureCollection.

        Returns
        -------
        FeatureSet
        """
        features = [
            Feature.from_geojson_feature(f)
            for f in geojson.get('features', [])
        ]
        props = geojson.get('properties', {})
        crs = props.pop('crs', 'EPSG:4326')
        schema_dicts = props.pop('schema', [])
        schema = [FieldSchema.from_dict(s) for s in schema_dicts]
        metadata = dict(props)
        return cls(
            features=features,
            crs=crs,
            schema=schema,
            metadata=metadata,
        )

    # -----------------------------------------------------------------
    # DetectionSet bridge
    # -----------------------------------------------------------------
    @classmethod
    def from_detection_set(
        cls,
        detection_set: 'DetectionSet',
        use_geo_geometry: bool = True,
        crs: str = 'EPSG:4326',
    ) -> 'FeatureSet':
        """
        Create a FeatureSet from a DetectionSet.

        Parameters
        ----------
        detection_set : DetectionSet
            Source detection set.
        use_geo_geometry : bool
            If True, use geographic geometry when available; otherwise
            use pixel geometry.  Default True.
        crs : str
            CRS for the resulting FeatureSet.  Default ``'EPSG:4326'``.

        Returns
        -------
        FeatureSet
        """
        from grdl.image_processing.detection.models import DetectionSet as DS

        features = []
        for i, det in enumerate(detection_set):
            if use_geo_geometry and det.geo_geometry is not None:
                geom = det.geo_geometry
            else:
                geom = det.pixel_geometry
            props = dict(det.properties)
            if det.confidence is not None:
                props['confidence'] = det.confidence
            features.append(Feature(
                geometry=geom,
                properties=props,
                feature_id=str(i),
            ))

        metadata = {
            'source_detector': detection_set.detector_name,
            'source_detector_version': detection_set.detector_version,
        }
        metadata.update(detection_set.metadata)

        return cls(
            features=features,
            crs=crs,
            metadata=metadata,
        )

    def to_detection_set(
        self,
        detector_name: str = 'FeatureSet',
        detector_version: str = '1.0.0',
        output_fields: Optional[Tuple[str, ...]] = None,
    ) -> 'DetectionSet':
        """
        Convert to a DetectionSet.

        Feature geometries become pixel geometries on the Detection.
        All feature properties are preserved.

        Parameters
        ----------
        detector_name : str
            Name for the detector on the DetectionSet.
        detector_version : str
            Version string for the DetectionSet.
        output_fields : Tuple[str, ...], optional
            Field names for the DetectionSet.  If None, inferred from
            the union of all feature property keys.

        Returns
        -------
        DetectionSet
        """
        from grdl.image_processing.detection.models import (
            Detection,
            DetectionSet,
        )

        if output_fields is None:
            all_keys: set = set()
            for f in self.features:
                all_keys.update(f.properties.keys())
            all_keys.discard('confidence')
            output_fields = tuple(sorted(all_keys))

        detections = []
        for feat in self.features:
            props = dict(feat.properties)
            confidence = props.pop('confidence', None)
            detections.append(Detection(
                pixel_geometry=feat.geometry,
                properties=props,
                confidence=confidence,
            ))

        return DetectionSet(
            detections=detections,
            detector_name=detector_name,
            detector_version=detector_version,
            output_fields=output_fields,
            metadata=dict(self.metadata),
        )

    # -----------------------------------------------------------------
    # GeoDataFrame bridge (optional geopandas)
    # -----------------------------------------------------------------
    def to_geodataframe(self):
        """
        Convert to a geopandas GeoDataFrame.

        Returns
        -------
        geopandas.GeoDataFrame

        Raises
        ------
        ImportError
            If geopandas is not installed.
        """
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "to_geodataframe() requires geopandas. "
                "Install with: pip install geopandas"
            )

        records = []
        geometries = []
        for f in self.features:
            records.append({**f.properties, '_feature_id': f.id})
            geometries.append(f.geometry)

        gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=self.crs)
        return gdf

    @classmethod
    def from_geodataframe(cls, gdf) -> 'FeatureSet':
        """
        Create a FeatureSet from a geopandas GeoDataFrame.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Source GeoDataFrame.

        Returns
        -------
        FeatureSet
        """
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "from_geodataframe() requires geopandas. "
                "Install with: pip install geopandas"
            )

        features = []
        prop_cols = [c for c in gdf.columns if c != gdf.geometry.name and c != '_feature_id']
        for idx, row in gdf.iterrows():
            geom = row.geometry
            props = {col: row[col] for col in prop_cols}
            fid = row.get('_feature_id', str(idx))
            features.append(Feature(
                geometry=geom,
                properties=props,
                feature_id=str(fid),
            ))

        crs_str = str(gdf.crs) if gdf.crs is not None else 'EPSG:4326'
        return cls(features=features, crs=crs_str)

    def __repr__(self) -> str:
        return (
            f"FeatureSet(count={len(self.features)}, "
            f"crs={self.crs!r}, "
            f"geometry_types={self.geometry_types!r})"
        )
