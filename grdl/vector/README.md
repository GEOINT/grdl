# `grdl.vector` — Geo-Registered Feature Data and Spatial Operators

Generic vector feature data model and spatial operators for GRDL. It
provides `Feature` / `FeatureSet` (parallel to `Detection` /
`DetectionSet` in `image_processing`), a `VectorProcessor` ABC, eight
shapely-backed spatial operators (buffer, intersection, union, dissolve,
spatial join, clip, centroid, convex hull), raster ↔ vector conversion,
GeoJSON-native I/O, and a coordinate-only fast path for large GeoJSON.

`grdl.vector` is the home for non-detection geo-registered geometry —
ground-truth labels, mask polygons, AOI boundaries, classification
regions, feature exports. For detector outputs, use
`grdl.image_processing.Detection` / `DetectionSet`; for geometric
analysis primitives that project through a geolocation, use
`grdl.shapes` (see *vector vs shapes vs detection* below).

---

## Data Model

Three classes in [models.py](models.py): `Feature`, `FieldSchema`, and
`FeatureSet`. Geometry is stored as native shapely objects.

### Feature

```python
from shapely.geometry import Point, Polygon
from grdl.vector import Feature

feat = Feature(
    geometry=Point(-118.15, 34.05),          # any shapely geometry
    properties={'label': 'building', 'confidence': 0.92},
    feature_id=None,                          # auto-generated UUID if None
)

feat.id          # 'a3f1...' (str)
feat.geometry    # shapely geometry
feat.properties  # dict

gj = feat.to_geojson_feature()                # -> GeoJSON Feature dict
feat2 = Feature.from_geojson_feature(gj)      # round-trips id + properties
```

`Feature(geometry, properties=None, feature_id=None)` — `properties`
defaults to an empty dict and `feature_id` to a fresh `uuid4()` string.

### FieldSchema

A `FieldSchema` describes **one** attribute field. A `FeatureSet`'s
`schema` is a *list* of these.

```python
from grdl.vector import FieldSchema

label_field = FieldSchema(
    name='label',
    dtype='str',              # type tag string: 'str', 'int', 'float', 'bool', ...
    description='Object class label',
    nullable=True,
)

label_field.to_dict()                 # {'name', 'dtype', 'description', 'nullable'}
FieldSchema.from_dict(label_field.to_dict())
```

`FieldSchema(name, dtype, description='', nullable=True)`. The
`dtype` is a free-form type tag (a string such as `'str'`, `'int'`,
`'float'`, `'bool'`) used for documentation and round-tripping schema
through GeoJSON — it is not enforced against property values.

### FeatureSet

```python
from grdl.vector import Feature, FieldSchema, FeatureSet

schema = [
    FieldSchema('label', 'str'),
    FieldSchema('confidence', 'float'),
]

fs = FeatureSet(
    features=[feat],            # List[Feature] — required, first positional arg
    crs='EPSG:4326',           # default
    schema=schema,             # optional List[FieldSchema]
    metadata={'source': 'analyst-1'},
)
```

`FeatureSet(features, crs='EPSG:4326', schema=None, metadata=None)`.
The collection is iterable, indexable, and length-aware:

```python
len(fs)              # feature count
fs.count             # same as len(fs)
fs[0]                # first Feature
for f in fs: ...     # iterate Features

fs.bounds            # (minx, miny, maxx, maxy) or None if empty
fs.geometry_types    # {'Point', 'Polygon', ...}
fs.get_geometries()  # List[shapely geometry]
fs.get_property_array('confidence')   # values in feature order (None where absent)
```

Filtering returns a new `FeatureSet` (CRS, schema, metadata preserved):

```python
fs.filter_by_bbox(minx, miny, maxx, maxy)        # geometries intersecting bbox
fs.filter_by_property('label', 'building')        # exact property match
fs.filter_by_geometry(some_geom, predicate='intersects')  # 'intersects'|'contains'|'within'
```

GeoJSON `FeatureCollection` export / import (CRS and schema travel in
the top-level `properties`):

```python
gj = fs.to_geojson()                  # -> FeatureCollection dict
fs2 = FeatureSet.from_geojson(gj)
```

#### Bridges

`FeatureSet` bridges to/from `DetectionSet` and (optionally) geopandas:

```python
# DetectionSet -> FeatureSet (uses geo_geometry when available)
fs = FeatureSet.from_detection_set(det_set, use_geo_geometry=True, crs='EPSG:4326')

# FeatureSet -> DetectionSet (geometries become pixel_geometry; a
# 'confidence' property maps to Detection.confidence)
ds = fs.to_detection_set(detector_name='FeatureSet', detector_version='1.0.0')

# geopandas (requires optional geopandas)
gdf = fs.to_geodataframe()
fs  = FeatureSet.from_geodataframe(gdf)
```

---

## Spatial Operators

All eight operators in [spatial.py](spatial.py) inherit
`VectorProcessor` and delegate the geometry math to shapely. Each takes
a `FeatureSet` and returns a new `FeatureSet`. Call them via
`process(features, **kwargs)`:

| Operator | Construction / call | Purpose |
|----------|--------------------|---------|
| `BufferOperator` | `BufferOperator(distance=, resolution=)` | Buffer every geometry by a fixed distance (CRS units) |
| `IntersectionOperator` | `.process(features, overlay=other)` | Pairwise intersection of features with an overlay set |
| `UnionOperator` | `UnionOperator()` | Union of all geometries into one feature |
| `DissolveOperator` | `DissolveOperator(by='field')` | Merge features sharing a property value |
| `SpatialJoinOperator` | `SpatialJoinOperator(predicate=, how=)` + `.process(features, overlay=other)` | Attach attributes from a second set by spatial relationship |
| `ClipOperator` | `.process(features, clip_geometry=geom)` | Clip every geometry to a clipping geometry |
| `CentroidOperator` | `CentroidOperator()` | Replace each geometry with its centroid |
| `ConvexHullOperator` | `ConvexHullOperator()` | Replace each geometry with its convex hull |

Note that several operators receive their *second* operand (the overlay,
clip geometry, or dissolve field) as a **keyword argument to
`process()`**, not as a constructor argument:

```python
from shapely.geometry import box
from grdl.vector import (
    BufferOperator, DissolveOperator, ClipOperator,
    IntersectionOperator, SpatialJoinOperator,
)

# Buffer: distance / resolution are tunable Annotated parameters
buffered = BufferOperator(distance=50.0, resolution=16).process(features)

# Dissolve by attribute (field name on the constructor or per-call)
merged = DissolveOperator(by='label').process(buffered)

# Clip needs the clip geometry at call time
aoi = box(-118.30, 34.00, -118.00, 34.20)
clipped = ClipOperator().process(features, clip_geometry=aoi)

# Intersection / spatial join take the second set via overlay=
overlapped = IntersectionOperator().process(features, overlay=other)
joined = SpatialJoinOperator(predicate='intersects', how='left').process(
    features, overlay=labels,
)
```

`BufferOperator.distance` (default `1.0`) and `.resolution` (default
`16`) are declared as `Annotated` tunable parameters with `Range`
constraints, so they are discoverable by grdl-runtime and overridable
per call: `BufferOperator().process(features, distance=100.0)`.
`SpatialJoinOperator(predicate='intersects', how='inner')` supports
`how='inner'` (matches only) or `how='left'` (keep unmatched input
features). `predicate` is any shapely binary predicate name
(`'intersects'`, `'contains'`, `'within'`, ...).

### Chaining operators

Operators compose by feeding one result into the next:

```python
from shapely.geometry import box
from grdl.vector import BufferOperator, DissolveOperator, ClipOperator

aoi = box(-118.30, 34.00, -118.00, 34.20)

result = ClipOperator().process(
    DissolveOperator(by='label').process(
        BufferOperator(distance=50.0).process(features)
    ),
    clip_geometry=aoi,
)
```

### The `VectorProcessor` ABC

[base.py](base.py) defines `VectorProcessor`, which inherits
`ImageProcessor` for version stamping and parameter flow but operates on
`FeatureSet` objects rather than ndarrays. Subclasses implement the
single abstract method:

```python
def process(self, features: FeatureSet, **kwargs) -> FeatureSet:
    ...
```

`execute(metadata, source, **kwargs)` is the pipeline entry point. It
dispatches to `process()` **only** when `source` is a `FeatureSet`,
returning `(result_feature_set, metadata)`. It deliberately fails fast
on the wrong input type:

- passing an `np.ndarray` raises `TypeError` ("operates on FeatureSet
  objects, not numpy arrays — use an ImageTransform for raster data");
- passing anything that is not a `FeatureSet` raises `TypeError`.

This is the key contract distinction from `ImageTransform` (ndarray in,
ndarray out): a `VectorProcessor` is FeatureSet in, FeatureSet out, and
will not silently accept raster data.

---

## Raster ↔ Vector Conversion

[conversion.py](conversion.py) provides `RasterToPoints` and
`Rasterize`. These are plain converter classes (not `VectorProcessor`
subclasses) with a `.convert()` method. They work in **pixel
coordinates** — they do not take a geolocation. To attach geographic
coordinates, project the result through a `grdl.geolocation.Geolocation`
afterward (or use `CoordSet.to_image` / a geolocation directly).

```python
import numpy as np
from grdl.vector import RasterToPoints, Rasterize

# Raster -> point FeatureSet: one point feature per pixel >= threshold.
# Each feature carries 'value', 'row', 'col' properties; geometry is
# Point(col, row) in pixel space.
mask = np.random.rand(256, 256) > 0.99
points = RasterToPoints(threshold=0.5, band=0, sample_step=1).convert(
    mask.astype(float), crs='EPSG:4326',
)

# FeatureSet -> raster: burn geometries into a numpy array. Pixels whose
# center falls inside a geometry get burn_value (or the value_field
# property); everything else gets fill_value.
arr = Rasterize(value_field=None, fill_value=0.0, burn_value=1.0).convert(
    features, shape=(256, 256),
)
```

- `RasterToPoints(threshold=0.0, band=0, sample_step=1)` — `band`
  selects the layer of a 3D `(bands, rows, cols)` array; `sample_step`
  sub-samples the grid. `.convert(source, crs='EPSG:4326')`.
- `Rasterize(value_field=None, fill_value=0.0, burn_value=1.0)` —
  `.convert(features, shape)` where `shape` is `(rows, cols)`. Uses a
  pixel-center-in-geometry test bounded by each geometry's bbox; for
  large arrays prefer `rasterio.features`.

---

## I/O

[io.py](io.py) provides `VectorReader` and `VectorWriter`. Both expose
**static methods** — there is no instance state to construct. GeoJSON is
supported natively with no third-party dependency; all other formats are
delegated to optional geopandas/fiona.

```python
from grdl.vector import VectorReader, VectorWriter

# Read (GeoJSON native; .shp/.gpkg/.kml/.gml/.fgb/.parquet via geopandas)
VectorReader.can_read('aoi.geojson')           # True
features = VectorReader.read('aoi.geojson', crs=None)   # -> FeatureSet

# Write (driver inferred from extension; override with driver=)
VectorWriter.write(features, 'out.geojson')
VectorWriter.write(features, 'out.gpkg', driver='GPKG')   # needs geopandas
```

- `VectorReader.read(path, crs=None)` reads the file (raising
  `FileNotFoundError` if missing, `ValueError` for unknown extensions,
  `ImportError` if a non-GeoJSON format is requested without geopandas).
  `crs` overrides the CRS on the returned `FeatureSet`.
- `VectorWriter.write(features, path, driver=None)` creates parent
  directories as needed and writes GeoJSON with 2-space indentation.

Native extensions (no deps): `.geojson`, `.json`. Geopandas-backed
extensions: `.shp`, `.gpkg`, `.kml`, `.gml`, `.fgb`, `.parquet`.

---

## Coordinate-Only Fast Path

[coords.py](coords.py) is a low-overhead alternative to the full
`Feature`/`FeatureSet` model for the common case where only vertex
**coordinates** are needed — e.g. loading a large GeoJSON of lat/lon
points and polygon rings, projecting them into image space through a
`Geolocation`, and overlaying them on imagery to inspect detections.

It skips shapely geometry construction, per-feature `Feature` objects,
property copies, and UUID generation, building flat numpy arrays
directly. Every vertex of every geometry is stacked into one `(N, 2)`
array so a whole file can be projected with a single vectorized
`Geolocation.latlon_to_image` call.

```python
import numpy as np
from grdl.vector import read_coords, coords_from_geojson, CoordSet

# Read a GeoJSON file straight into flat coordinate arrays
cs = read_coords('detections.geojson', holes=True)   # -> CoordSet

cs.xy            # (N, 2) float64 — [lon, lat] (GeoJSON axis order)
cs.latlon        # (N, 2) — [lat, lon] view (axis order for latlon_to_image)
cs.n_features    # source feature count
cs.n_parts       # number of points/lines/rings
cs.n_vertices    # total vertices

# Iterate parts (a part is a point, a line, or a polygon ring)
for feat_idx, part_type, verts in cs.iter_parts():
    # part_type in {'point', 'line', 'exterior', 'hole'}; verts is (M, 2)
    ...

# Project all vertices into image pixel space in one batched call.
# geo.elevation (if attached) handles terrain internally; otherwise the
# height fallback is used.
pix = cs.to_image(geolocation=geo, height=0.0)   # CoordSet, space='pixel'
pix.xy           # (N, 2) — [row, col]
```

Also available: `coords_from_geojson(geojson_dict, holes=True)` for an
already-loaded GeoJSON dict (FeatureCollection, Feature, or bare
geometry). Supported geometry types: `Point`, `MultiPoint`,
`LineString`, `MultiLineString`, `Polygon`, `MultiPolygon`,
`GeometryCollection`. Set `holes=False` to drop polygon interior rings.

A `CoordSet` records its coordinate space in `.space` (`'lonlat'` or
`'pixel'`); `.latlon` and `.to_image` require `'lonlat'` and raise
`ValueError` otherwise.

---

## Module Layout

```
grdl/vector/
├── models.py        Feature, FieldSchema, FeatureSet (+ DetectionSet / geopandas bridges)
├── base.py          VectorProcessor (ABC)
├── spatial.py       BufferOperator, IntersectionOperator, UnionOperator,
│                    DissolveOperator, SpatialJoinOperator, ClipOperator,
│                    CentroidOperator, ConvexHullOperator
├── conversion.py    RasterToPoints, Rasterize (pixel-space converters)
├── io.py            VectorReader, VectorWriter (GeoJSON native; geopandas optional)
└── coords.py        CoordSet, read_coords, coords_from_geojson (fast path)
```

---

## vector vs shapes vs detection

Three GRDL domains carry geometry; pick by intent:

| Use this | For | Key type |
|----------|-----|----------|
| `grdl.vector` | Generic geo-registered vector features — ground-truth labels, AOI boundaries, mask polygons, classification regions, GeoJSON/shapefile exports, spatial overlay analysis | `FeatureSet` (shapely geometry + properties) |
| `grdl.shapes` | Geometric analysis primitives defined in lat/lon that project through a `Geolocation` (with per-vertex DEM) to pixel masks/overlays — ROIs, CEP/uncertainty ellipses, range rings, detector cueing | `GeographicShape` subclasses (`Circle`, `Ellipse`, `GeoPolygon`, `Arc`, ...) |
| `grdl.image_processing.detection` | Detector output — features produced by a CFAR/ML detector, with confidence and a detector/version stamp | `DetectionSet` (`Detection` objects) |

Rules of thumb:

- It came out of a detector → `DetectionSet`.
- It is an analyst-drawn or ground-truth label / boundary / mask you
  want to store, filter, overlay, or export → `FeatureSet`.
- It is a parametric geographic shape (circle, ellipse, arc) you want to
  rasterize against an image or use to cue a detector → `grdl.shapes`.

The domains interoperate: `FeatureSet.from_detection_set` /
`to_detection_set` bridge to detections, and `FeatureSet` serializes to
the same GeoJSON `FeatureCollection` that `DetectionSet.to_geojson()`
produces.

---

## Dependencies

| Package | Required by |
|---------|-------------|
| `shapely>=2.0` | geometry kernel — all of `models`, `spatial`, `conversion` |
| `pyproj>=3.4` | CRS handling |
| `numpy` | `conversion`, `coords` |
| `geopandas` / `fiona` | *optional* — non-GeoJSON I/O (`.shp`, `.gpkg`, `.kml`, ...) and the `GeoDataFrame` bridge |
| `rasterio` | *optional* — recommended for large-array rasterization |

GeoJSON read/write and all eight spatial operators work with only
shapely + pyproj installed; geopandas is needed solely for other vector
formats and the `to_geodataframe` / `from_geodataframe` bridge.
