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

## Functions

### transform_detection_set

Transform all detections in a `DetectionSet`. Returns a new `DetectionSet` with transformed pixel coordinates and preserved metadata.

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

Transform a single `Detection` object. Returns a new `Detection` with transformed pixel geometry; all other attributes (confidence, properties, geo_geometry) are preserved.

```python
from grdl.transforms.detection import transform_detection

aligned_det = transform_detection(detection, registration_result)
```

### transform_pixel_geometry

Transform a raw shapely geometry (Point or Polygon) using a `RegistrationResult`.

```python
from grdl.transforms.detection import transform_pixel_geometry
from shapely.geometry import Point

point = Point(100, 50)  # (col, row) shapely convention
transformed = transform_pixel_geometry(point, registration_result)
```

---

## Bounding Box Handling

The `bbox_mode` parameter controls how axis-aligned bounding boxes behave under non-trivial transforms:

| Mode | Behavior |
|------|----------|
| `'refit'` (default) | Transform all corners, then compute the minimal axis-aligned bounding box enclosing them |
| `'polygon'` | Transform all corners and return as a general Polygon (preserves exact transformed shape) |

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
