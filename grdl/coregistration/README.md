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

## Available Methods

### AffineCoRegistration

Least-squares affine transform (6 DOF: translation, rotation, scale, shear) from control point correspondences. Requires >= 3 non-collinear point pairs.

```python
from grdl.coregistration import AffineCoRegistration

coreg = AffineCoRegistration(
    control_points_fixed=fixed_pts,     # shape (N, 2), (row, col)
    control_points_moving=moving_pts,   # shape (N, 2), (row, col)
    interpolation_order=1,              # 0=nearest, 1=bilinear, 3=bicubic
)
result = coreg.estimate(fixed_image, moving_image)
aligned = coreg.apply(moving_image, result)
```

### ProjectiveCoRegistration

Projective (homography) transform (8 DOF) via Direct Linear Transform with point normalization. Requires >= 4 non-collinear point pairs. Use when perspective distortion is present.

```python
from grdl.coregistration import ProjectiveCoRegistration

coreg = ProjectiveCoRegistration(
    control_points_fixed=fixed_pts,
    control_points_moving=moving_pts,
)
result = coreg.estimate(fixed_image, moving_image)
aligned = coreg.apply(moving_image, result)
```

### FeatureMatchCoRegistration

Automated feature detection and matching with RANSAC outlier rejection. Requires `opencv-python-headless`.

```python
from grdl.coregistration import FeatureMatchCoRegistration

coreg = FeatureMatchCoRegistration(method='orb', max_features=1000)
result = coreg.estimate(fixed_image, moving_image)
aligned = coreg.apply(moving_image, result)

print(result.inlier_ratio)  # fraction of matches classified as inliers
```

---

## RegistrationResult

All `estimate()` calls return a `RegistrationResult` containing the transform matrix and quality metrics.

```python
result.transform_matrix        # (2, 3) affine or (3, 3) projective
result.residual_rms            # RMS error in pixels
result.num_matches             # number of point pairs used
result.inlier_ratio            # inlier fraction (meaningful for RANSAC)
result.is_affine               # True if (2, 3) matrix
result.is_projective           # True if (3, 3) matrix
result.inverse_transform_matrix  # inverse (fixed -> moving)

# Transform arbitrary points (row, col)
pts = np.array([[50, 100], [200, 300]])
transformed = result.transform_points(pts)
inverse_pts = result.transform_points(pts, inverse=True)
```

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
