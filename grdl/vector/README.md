# `grdl.vector` — Geo-Registered Feature Data and Spatial Operators

Generic vector feature data model and spatial operators for GRDL.
Provides `Feature` / `FeatureSet` (parallel to `Detection` /
`DetectionSet` in `image_processing`), spatial operations (buffer,
intersection, union, dissolve, clip, centroid, convex hull), and
raster-vector conversion.

`grdl.vector` is the home for non-detection geo-registered geometry —
ground-truth labels, mask polygons, AOI boundaries, classification
regions, feature exports. For detector outputs, use
`grdl.image_processing.Detection` / `DetectionSet` instead.

---

## Data Model

```python
from grdl.vector import Feature, FeatureSet, FieldSchema

schema = FieldSchema([
    ('label',      'str'),
    ('confidence', 'float'),
])

feat = Feature(
    geometry=...,                # Point, LineString, Polygon
    properties={'label': 'building', 'confidence': 0.92},
)
features = FeatureSet(schema=schema, features=[feat])
```

`FeatureSet` is iterable, length-aware, and serializes to GeoJSON
`FeatureCollection` via its writer.

---

## Spatial Operators

All inherit `VectorProcessor` (the vector-domain ABC). Each operates
on a `FeatureSet` and returns a new `FeatureSet`.

| Operator | Purpose |
|----------|---------|
| `BufferOperator` | Geometric buffer at a configurable radius |
| `IntersectionOperator` | Set-intersection of overlapping features |
| `UnionOperator` | Set-union (dissolves overlaps) |
| `DissolveOperator` | Merge features by attribute key |
| `SpatialJoinOperator` | Attach attributes from a second FeatureSet |
| `ClipOperator` | Clip to a bounding polygon |
| `CentroidOperator` | Replace each geometry with its centroid |
| `ConvexHullOperator` | Per-feature convex hull |

```python
from grdl.vector import BufferOperator, DissolveOperator

buffered = BufferOperator(radius_m=50.0).apply(features)
merged   = DissolveOperator(by='label').apply(buffered)
```

---

## Raster ↔ Vector Conversion

```python
from grdl.vector import RasterToPoints, Rasterize

# Boolean mask → point FeatureSet (one feature per True pixel)
points = RasterToPoints(geolocation=geo).apply(mask)

# FeatureSet → boolean raster
raster = Rasterize(shape=(rows, cols), geolocation=geo).apply(features)
```

---

## I/O

```python
from grdl.vector import VectorReader, VectorWriter

# Read GeoJSON / Shapefile / KML
features = VectorReader('aoi.geojson').read()

# Write to GeoJSON
VectorWriter('out.geojson').write(features)
```

---

## Module Layout

```
grdl/vector/
├── models.py        Feature, FieldSchema, FeatureSet
├── base.py          VectorProcessor (ABC)
├── spatial.py       BufferOperator, IntersectionOperator, UnionOperator,
│                    DissolveOperator, SpatialJoinOperator, ClipOperator,
│                    CentroidOperator, ConvexHullOperator
├── conversion.py    RasterToPoints, Rasterize
└── io.py            VectorReader, VectorWriter
```

---

## When to Use What

| Task | Use this |
|------|----------|
| Detector output (geo-registered) | `grdl.image_processing.DetectionSet` |
| Ground-truth labels / AOI polygons | `grdl.vector.FeatureSet` |
| Apply a coregistration to detections | `grdl.transforms.transform_detection_set` |
| Export to GeoJSON | `DetectionSet.to_geojson()` or `VectorWriter` |
| Spatial join / overlay analysis | `grdl.vector` operators |

---

## Dependencies

`shapely>=2.0` (geometry kernel) and `pyproj>=3.4` (CRS handling). Some
operators delegate to `rasterio` for raster ↔ vector conversion. All
covered by the `[geolocation]` and `[detection]` extras combined.
