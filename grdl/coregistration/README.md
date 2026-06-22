# Co-Registration Module

Aligns a moving image to a fixed (reference) image by estimating a spatial transform. The two-step interface separates estimation (`estimate`) from application (`apply`), allowing the same transform to be reused across multiple images or bands. Supports control-point-based affine and projective transforms, and automated feature matching with RANSAC.

---

## Quick Start

```python
import numpy as np
from grdl.coregistration import AffineCoRegistration

# Define matched control points: (row, col) in each image
fixed_pts = np.array([[100, 200], [100, 800], [900, 200], [900, 800]])
moving_pts = np.array([[102, 203], [101, 802], [901, 201], [902, 803]])

# Estimate and apply
coreg = AffineCoRegistration(fixed_pts, moving_pts)
result = coreg.estimate(fixed_image, moving_image)
aligned = coreg.apply(moving_image, result)

# Inspect quality
print(result.residual_rms)    # RMS error in pixels
print(result.num_matches)     # number of control points used
```

---

## The CoRegistration Interface

All algorithms implement the `CoRegistration` ABC:

```python
result = coreg.estimate(fixed, moving, fixed_geo=None, moving_geo=None)
aligned = coreg.apply(moving, result, output_shape=None)
```

- **`estimate(fixed, moving, fixed_geo=None, moving_geo=None)`** -> `RegistrationResult`.
  `fixed`/`moving` are `(rows, cols)` or `(rows, cols, bands)` arrays. The optional
  `fixed_geo`/`moving_geo` are `grdl.geolocation` objects for algorithms that can use a
  geographic prior; the bundled algorithms accept them for interface compatibility but do
  not currently use them. Coordinate convention throughout is **`(row, col)`**.
- **`apply(moving, result, output_shape=None)`** -> warped array. `output_shape` is
  `(rows, cols)`; when `None`, the moving image shape is used. Multi-band input keeps its
  band count; pixels mapped outside the source are filled with `0`.

The two-step split lets one estimated transform be reused across multiple images or bands.

---

## Available Methods

### AffineCoRegistration

Least-squares affine transform (6 DOF: translation, rotation, scale, shear) from control point correspondences. Requires >= 3 non-collinear point pairs. Estimation uses only the control points supplied at construction -- the `fixed`/`moving` image arrays passed to `estimate()` are accepted for ABC compliance but not read. `inlier_ratio` is always `1.0` (no outlier rejection).

```python
from grdl.coregistration import AffineCoRegistration

coreg = AffineCoRegistration(
    control_points_fixed=fixed_pts,     # shape (N, 2), (row, col), N >= 3
    control_points_moving=moving_pts,   # shape (N, 2), (row, col), same N
    interpolation_order=1,              # 0=nearest, 1=bilinear, 3=bicubic
)
result = coreg.estimate(fixed_image, moving_image)
aligned = coreg.apply(moving_image, result)
```

Construction raises `ValueError` if fewer than 3 pairs are given or the two control-point arrays differ in shape.

### ProjectiveCoRegistration

Projective (homography) transform (8 DOF) via Direct Linear Transform (DLT) with Hartley point normalization for numerical stability. Requires >= 4 non-collinear point pairs. Use when perspective distortion is present. Like the affine variant, estimation is purely control-point driven and `inlier_ratio` is `1.0`.

```python
from grdl.coregistration import ProjectiveCoRegistration

coreg = ProjectiveCoRegistration(
    control_points_fixed=fixed_pts,     # shape (N, 2), (row, col), N >= 4
    control_points_moving=moving_pts,
    interpolation_order=1,
)
result = coreg.estimate(fixed_image, moving_image)
aligned = coreg.apply(moving_image, result)
```

Construction raises `ValueError` if fewer than 4 pairs are given or the shapes differ.

### FeatureMatchCoRegistration

Automated feature detection and matching with RANSAC outlier rejection. **Requires `opencv-python-headless`** -- importing it without OpenCV raises `DependencyError`, and it is added to the package namespace only when OpenCV is present.

```python
from grdl.coregistration import FeatureMatchCoRegistration

coreg = FeatureMatchCoRegistration(
    method='orb',              # 'orb' or 'sift' feature detector
    max_features=5000,         # max keypoints to detect
    transform_type='affine',   # 'affine' or 'homography'
    ransac_threshold=5.0,      # RANSAC inlier distance, pixels
    match_ratio=0.75,          # Lowe ratio test threshold
    interpolation_order=1,     # warp interpolation order
)
result = coreg.estimate(fixed_image, moving_image)
aligned = coreg.apply(moving_image, result)

print(result.inlier_ratio)  # fraction of matches classified as inliers
```

Internally it detects keypoints in both images, matches descriptors with a brute-force
matcher + Lowe's ratio test, and fits the transform with RANSAC. OpenCV works in `(x, y)`
= `(col, row)`; this class converts all matched points and the fitted matrix back to GRDL's
`(row, col)` convention, so `result.transform_matrix` is directly compatible with the
control-point methods. Raises `ProcessorError` if feature detection yields no descriptors
or too few matches survive for the chosen `transform_type` (needs >= 3 affine / >= 4
homography), or if RANSAC fails to find a valid model. `transform_type='homography'`
produces a `(3, 3)` matrix; `'affine'` a `(2, 3)` matrix.

---

## RegistrationResult

All `estimate()` calls return a `RegistrationResult` containing the transform matrix and quality metrics.

```python
result.transform_matrix        # (2, 3) affine or (3, 3) projective; moving -> fixed
result.residual_rms            # RMS error in pixels on the matched points
result.num_matches             # number of point pairs used (inliers for RANSAC)
result.inlier_ratio            # inlier fraction (1.0 for the least-squares methods)
result.metadata                # dict: method name, interpolation_order, max_residual, ...

result.is_affine               # True if (2, 3) matrix
result.is_projective           # True if (3, 3) matrix
result.inverse_transform_matrix  # inverse (fixed -> moving), always (3, 3)

# Transform arbitrary points (row, col)
pts = np.array([[50, 100], [200, 300]])
transformed = result.transform_points(pts)                # forward: moving -> fixed
inverse_pts = result.transform_points(pts, inverse=True)  # inverse: fixed -> moving
```

`transform_matrix` maps **moving -> fixed**. `inverse_transform_matrix` expands an affine
`(2, 3)` matrix to homogeneous `(3, 3)` before inverting, so the inverse is always `(3, 3)`
(raises `np.linalg.LinAlgError` if the matrix is singular). `transform_points(points,
inverse=False)` applies the forward or inverse matrix to an `(N, 2)` `(row, col)` array and
returns an `(N, 2)` array.

---

## Utility Functions

Helpers in `grdl.coregistration.utils` back the algorithms and are useful directly when
working with transform matrices and quality metrics. Import them from the submodule:

```python
from grdl.coregistration.utils import (
    apply_transform_to_points,
    compute_residuals,
    compute_rms,
    warp_image,
    estimate_overlap_fraction,
)
```

| Function | Signature | Purpose |
|----------|-----------|---------|
| `apply_transform_to_points` | `(points, transform_matrix)` | Apply a `(2, 3)` affine or `(3, 3)` projective matrix to an `(N, 2)` `(row, col)` array (homography divides by `w`). |
| `compute_residuals` | `(fixed_points, moving_points, transform_matrix)` | Per-point Euclidean residuals (pixels) after transforming the moving points onto the fixed points. Returns `(N,)`. |
| `compute_rms` | `(residuals)` | RMS of a residual vector. |
| `warp_image` | `(image, transform_matrix, output_shape=None, order=1, fill_value=0.0)` | Inverse-map warp of a `(rows, cols[, bands])` image via `scipy.ndimage.map_coordinates`. The matrix is forward (moving -> fixed); the function inverts it internally. This is what `apply()` calls. |
| `estimate_overlap_fraction` | `(fixed_shape, moving_shape, transform_matrix)` | Fraction (0..1) of the fixed image covered by the transformed moving footprint -- a quick overlap-sufficiency check before registering. |

These functions are **not** re-exported from the package root; import them from
`grdl.coregistration.utils`.

---

## Applying Transforms to Detections

Use `grdl.transforms` to apply a registration result to detection geometries without raster interpolation:

```python
from grdl.transforms.detection import transform_detection_set

transformed_detections = transform_detection_set(detections, result)
```

See the [transforms README](../transforms/README.md) for details.

---

## When to Use Which Method

| Scenario | Method |
|----------|--------|
| Known GCPs, rigid/affine distortion | `AffineCoRegistration` |
| Known GCPs, perspective distortion | `ProjectiveCoRegistration` |
| No GCPs, automated alignment | `FeatureMatchCoRegistration` |
