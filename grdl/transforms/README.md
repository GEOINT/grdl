# Transforms Module

Applies co-registration spatial transforms to detection geometries (points, bounding boxes, polygons) in pixel space. This enables a detect-then-transform workflow: run detection on the original image to preserve fidelity, then map the resulting coordinates to a reference image without raster interpolation.

---

## Quick Start

```python
from grdl.coregistration import AffineCoRegistration
from grdl.transforms.detection import transform_detection_set

# Estimate registration
coreg = AffineCoRegistration(fixed_pts, moving_pts)
result = coreg.estimate(fixed_image, moving_image)

# Run detection on the moving image (original geometry)
detections = detector.detect(moving_image)

# Transform detection coordinates to the fixed image's pixel space
aligned_detections = transform_detection_set(detections, result)
```

---

All three functions share the same trailing arguments:

- `inverse` (`bool`, default `False`): `False` applies the forward transform
  (moving -> fixed); `True` applies the inverse (fixed -> moving).
- `bbox_mode` (`str`, default `'refit'`): box-handling strategy, see below. Any other value
  raises `ValueError`.

They are also importable from the package root: `from grdl.transforms import
transform_pixel_geometry, transform_detection, transform_detection_set`.

## Functions

### transform_detection_set

Transform all detections in a `DetectionSet`. Returns a **new** `DetectionSet` (the input is not mutated) carrying the same detector name/version and output fields, with transformed pixel coordinates. Two metadata keys are added to record what happened: `coordinate_transform_applied=True` and `transform_direction='forward'` or `'inverse'`.

```python
from grdl.transforms.detection import transform_detection_set

aligned = transform_detection_set(
    detection_set=detections,
    result=registration_result,
    inverse=False,          # forward: moving -> fixed (default)
    bbox_mode='refit',      # 'refit' or 'polygon'
)
```

### transform_detection

Transform a single `Detection` object. Returns a new `Detection` with transformed pixel geometry; all other attributes (`properties`, `confidence`, `geo_geometry`) are copied through unchanged.

```python
from grdl.transforms.detection import transform_detection

aligned_det = transform_detection(detection, registration_result)
```

### transform_pixel_geometry

Transform a raw shapely geometry using a `RegistrationResult`. Supports `Point` and `Polygon` geometries only -- any other `geom_type` raises `ValueError`. Returns a new geometry of the same kind (or a `box`/`Polygon` for polygon input, depending on `bbox_mode`).

```python
from grdl.transforms.detection import transform_pixel_geometry
from shapely.geometry import Point

point = Point(100, 50)  # (col, row) = (x, y) shapely convention
transformed = transform_pixel_geometry(point, registration_result)
```

---

## Bounding Box Handling

The `bbox_mode` parameter controls how axis-aligned bounding boxes behave under non-trivial transforms:

| Mode | Behavior |
|------|----------|
| `'refit'` (default) | If the polygon is a 4-corner axis-aligned rectangle, transform all corners and return the minimal axis-aligned `box` enclosing them. Non-rectangular polygons fall through to exact-shape transformation. |
| `'polygon'` | Transform every vertex and return a general `Polygon` (preserves the exact transformed shape, including any shear/rotation the transform introduces) |

Under a rotating or perspective transform, an axis-aligned box does not stay axis-aligned.
`'refit'` keeps the result a clean bounding box (the common case for detection boxes);
`'polygon'` keeps the geometry faithful at the cost of producing a rotated/sheared polygon.
`Point` geometries are unaffected by `bbox_mode`.

---

## Coordinate Conventions

- Detection pixel geometries use shapely convention: `(x, y) = (col, row)`.
- The transform functions handle the conversion to `(row, col)` internally.
- Forward transform (`inverse=False`): moving image coordinates to fixed image coordinates.
- Inverse transform (`inverse=True`): fixed image coordinates to moving image coordinates.

---

## Dependencies

- `shapely` -- required for geometry operations
- `grdl.coregistration` -- provides `RegistrationResult` and `apply_transform_to_points`
- `grdl.image_processing.detection.models` -- provides `Detection` and `DetectionSet`
