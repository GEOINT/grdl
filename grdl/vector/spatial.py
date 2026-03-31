# -*- coding: utf-8 -*-
"""
Spatial Operators - Core vector spatial processing operators.

Provides ``VectorProcessor`` implementations for common spatial
operations: buffer, intersection, union, dissolve, spatial join,
clip, centroid, and convex hull.

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
2026-03-25
"""

# Standard library
from typing import Annotated, Any, Optional

# Third-party
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

# GRDL internal
from grdl.image_processing.params import Desc, Range, Options
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ProcessorCategory, ImageModality
from grdl.vector.base import VectorProcessor
from grdl.vector.models import Feature, FeatureSet


@processor_version('1.0.0')
@processor_tags(
    category=ProcessorCategory.ANALYZE,
    description='Buffer feature geometries by a fixed distance.',
)
class BufferOperator(VectorProcessor):
    """Buffer all feature geometries by a fixed distance.

    Parameters
    ----------
    distance : float
        Buffer distance in CRS units.
    resolution : int
        Number of segments per quarter circle.  Default 16.
    """

    distance: Annotated[float, Range(min=0.0), Desc('Buffer distance in CRS units')] = 1.0
    resolution: Annotated[int, Range(min=1, max=128), Desc('Segments per quarter circle')] = 16

    def process(self, features: FeatureSet, **kwargs: Any) -> FeatureSet:
        params = self._resolve_params(kwargs)
        dist = params['distance']
        res = params['resolution']
        new_features = []
        for f in features:
            buffered = f.geometry.buffer(dist, resolution=res)
            new_features.append(Feature(
                geometry=buffered,
                properties=dict(f.properties),
                feature_id=f.id,
            ))
        return FeatureSet(
            features=new_features,
            crs=features.crs,
            schema=features.schema,
            metadata=features.metadata,
        )


@processor_version('1.0.0')
@processor_tags(
    category=ProcessorCategory.ANALYZE,
    description='Compute intersection of two feature sets.',
)
class IntersectionOperator(VectorProcessor):
    """Compute pairwise intersection of features with an overlay set.

    The overlay FeatureSet is passed via ``kwargs['overlay']``.

    For each feature in the input, computes its intersection with each
    feature in the overlay.  Only non-empty intersections are kept.
    Properties from both features are merged (input takes precedence).
    """

    def process(self, features: FeatureSet, **kwargs: Any) -> FeatureSet:
        overlay = kwargs.get('overlay')
        if overlay is None:
            raise ValueError(
                "IntersectionOperator requires 'overlay' keyword argument "
                "containing a FeatureSet."
            )
        new_features = []
        idx = 0
        for f in features:
            for o in overlay:
                if f.geometry.intersects(o.geometry):
                    isect = f.geometry.intersection(o.geometry)
                    if not isect.is_empty:
                        props = {**o.properties, **f.properties}
                        new_features.append(Feature(
                            geometry=isect,
                            properties=props,
                            feature_id=str(idx),
                        ))
                        idx += 1
        return FeatureSet(
            features=new_features,
            crs=features.crs,
            metadata=features.metadata,
        )


@processor_version('1.0.0')
@processor_tags(
    category=ProcessorCategory.ANALYZE,
    description='Compute geometric union of all features.',
)
class UnionOperator(VectorProcessor):
    """Compute the geometric union of all features in the set.

    Produces a single feature whose geometry is the union of all
    input geometries.  Properties from the first feature are preserved.
    """

    def process(self, features: FeatureSet, **kwargs: Any) -> FeatureSet:
        if not features.features:
            return FeatureSet(features=[], crs=features.crs)
        geoms = [f.geometry for f in features]
        merged = unary_union(geoms)
        props = dict(features.features[0].properties) if features.features else {}
        result_feature = Feature(
            geometry=merged,
            properties=props,
            feature_id='union',
        )
        return FeatureSet(
            features=[result_feature],
            crs=features.crs,
            metadata=features.metadata,
        )


@processor_version('1.0.0')
@processor_tags(
    category=ProcessorCategory.ANALYZE,
    description='Dissolve features by a property field.',
)
class DissolveOperator(VectorProcessor):
    """Dissolve (merge) features that share the same value for a field.

    Parameters
    ----------
    by : str
        Property field name to group by before dissolving.
    """

    def __init__(self, by: str = '') -> None:
        self.by = by

    def process(self, features: FeatureSet, **kwargs: Any) -> FeatureSet:
        by_field = kwargs.get('by', self.by)
        if not by_field:
            raise ValueError(
                "DissolveOperator requires a 'by' field name."
            )
        groups: dict = {}
        for f in features:
            key = f.properties.get(by_field)
            groups.setdefault(key, []).append(f)

        new_features = []
        for key, group in groups.items():
            merged = unary_union([f.geometry for f in group])
            props = dict(group[0].properties)
            new_features.append(Feature(
                geometry=merged,
                properties=props,
                feature_id=str(key),
            ))
        return FeatureSet(
            features=new_features,
            crs=features.crs,
            metadata=features.metadata,
        )


@processor_version('1.0.0')
@processor_tags(
    category=ProcessorCategory.ANALYZE,
    description='Spatial join of two feature sets.',
)
class SpatialJoinOperator(VectorProcessor):
    """Join features from two sets based on spatial relationship.

    The right FeatureSet is passed via ``kwargs['overlay']``.

    Parameters
    ----------
    predicate : str
        Spatial predicate: ``'intersects'``, ``'contains'``,
        ``'within'``.  Default ``'intersects'``.
    how : str
        Join type: ``'inner'``, ``'left'``.  Default ``'inner'``.
    """

    def __init__(
        self,
        predicate: str = 'intersects',
        how: str = 'inner',
    ) -> None:
        self.predicate = predicate
        self.how = how

    def process(self, features: FeatureSet, **kwargs: Any) -> FeatureSet:
        overlay = kwargs.get('overlay')
        if overlay is None:
            raise ValueError(
                "SpatialJoinOperator requires 'overlay' keyword argument."
            )
        predicate = kwargs.get('predicate', self.predicate)
        how = kwargs.get('how', self.how)

        new_features = []
        idx = 0
        for f in features:
            matched = False
            for o in overlay:
                if getattr(f.geometry, predicate)(o.geometry):
                    props = {**o.properties, **f.properties}
                    new_features.append(Feature(
                        geometry=f.geometry,
                        properties=props,
                        feature_id=str(idx),
                    ))
                    idx += 1
                    matched = True
            if not matched and how == 'left':
                new_features.append(Feature(
                    geometry=f.geometry,
                    properties=dict(f.properties),
                    feature_id=str(idx),
                ))
                idx += 1

        return FeatureSet(
            features=new_features,
            crs=features.crs,
            metadata=features.metadata,
        )


@processor_version('1.0.0')
@processor_tags(
    category=ProcessorCategory.ANALYZE,
    description='Clip features to a clipping geometry.',
)
class ClipOperator(VectorProcessor):
    """Clip all features to a clipping geometry.

    The clip geometry is passed via ``kwargs['clip_geometry']``.
    """

    def process(self, features: FeatureSet, **kwargs: Any) -> FeatureSet:
        clip_geom = kwargs.get('clip_geometry')
        if clip_geom is None:
            raise ValueError(
                "ClipOperator requires 'clip_geometry' keyword argument."
            )
        new_features = []
        for f in features:
            if f.geometry.intersects(clip_geom):
                clipped = f.geometry.intersection(clip_geom)
                if not clipped.is_empty:
                    new_features.append(Feature(
                        geometry=clipped,
                        properties=dict(f.properties),
                        feature_id=f.id,
                    ))
        return FeatureSet(
            features=new_features,
            crs=features.crs,
            schema=features.schema,
            metadata=features.metadata,
        )


@processor_version('1.0.0')
@processor_tags(
    category=ProcessorCategory.ANALYZE,
    description='Compute centroid of each feature geometry.',
)
class CentroidOperator(VectorProcessor):
    """Replace each feature geometry with its centroid."""

    def process(self, features: FeatureSet, **kwargs: Any) -> FeatureSet:
        new_features = []
        for f in features:
            new_features.append(Feature(
                geometry=f.geometry.centroid,
                properties=dict(f.properties),
                feature_id=f.id,
            ))
        return FeatureSet(
            features=new_features,
            crs=features.crs,
            schema=features.schema,
            metadata=features.metadata,
        )


@processor_version('1.0.0')
@processor_tags(
    category=ProcessorCategory.ANALYZE,
    description='Compute convex hull of each feature geometry.',
)
class ConvexHullOperator(VectorProcessor):
    """Replace each feature geometry with its convex hull."""

    def process(self, features: FeatureSet, **kwargs: Any) -> FeatureSet:
        new_features = []
        for f in features:
            new_features.append(Feature(
                geometry=f.geometry.convex_hull,
                properties=dict(f.properties),
                feature_id=f.id,
            ))
        return FeatureSet(
            features=new_features,
            crs=features.crs,
            schema=features.schema,
            metadata=features.metadata,
        )
