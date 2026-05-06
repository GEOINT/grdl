# Shapes Module

Geographic shapes (circles, ellipses, polygons, arcs) as first-class analysis primitives. Every shape is defined in WGS-84 lat/lon and projects accurately into any GRDL geolocation (SICD R/Rdot, SIDD, RPC/RSM, Affine basemap) with per-vertex DEM sampling. Shapes produce pixel masks, overlay on imagery, cue detectors to a region of interest, and combine via covariance arithmetic or set operations. Perimeter generation uses Karney's geodesic algorithm via `pyproj.Geod`; adaptive refinement guarantees sub-pixel fidelity across SAR foreshortening and sloped terrain.

---

## Quick Start

```python
import numpy as np
from grdl.shapes import Circle, Ellipse, overlay_shape

# Circular ROI (1 km radius) on an image
roi = Circle(center_lat=34.05, center_lon=-118.15, radius_m=1000.0)
mask = roi.rasterize(geolocation=geo, image_shape=image.shape)

# Uncertainty ellipse, 1 km major x 0.25 km minor, N-aligned
err = Ellipse(
    center_lat=34.05, center_lon=-118.15,
    semi_major_m=1000.0, semi_minor_m=250.0,
    rotation_deg=0.0,
)

# Overlay on a matplotlib axes
ax.imshow(image, cmap='gray')
overlay_shape(ax, err, geo, color='lime', linewidth=2.0)
```

---

## Architecture

### The `GeographicShape` contract

Every shape inherits from the `GeographicShape` ABC ([base.py](base.py)). Subclasses implement exactly one method:

```python
def _perimeter_latlon(self, n: int) -> np.ndarray:
    """Return (N, 2) perimeter vertices as [lat_deg, lon_deg]."""
```

The base class adds everything else: DEM sampling, projection, adaptive refinement, rasterization, containment testing.

### Pipeline

```
_perimeter_latlon(n)            (subclass)
        │  (N, 2) lat/lon
        ▼
perimeter_latlon(n, geo,        (base.py)
                  height=None)
        │  (N, 3) lat/lon/HAE — DEM-sampled, or constant
        ▼
to_pixels(geo, pixel_tol, ...)  (base.py + refine.py)
        │  adaptive midpoint subdivision against pixel tolerance
        │  (M, 2) fractional [row, col]
        ▼
rasterize(image_shape)          (rasterize.py, skimage)
        │  boolean mask in pixel space
        ▼
  overlay / burn / cue / mask
```

Every stage is vectorized. `to_pixels` issues one batched `latlon_to_image` call per refinement iteration; `rasterize` runs through `skimage.draw.polygon` in C; adaptive refinement typically converges in 2–6 iterations.

### Height handling (important for SAR)

`perimeter_latlon`, `to_pixels`, and `overlay_shape` accept an optional `height` parameter:

- `height=None` (default): sample the DEM attached to the geolocation at every vertex.
- `height=<float>`: use a constant HAE (metres) for every vertex; DEM sampling skipped.

For a ground shape rendered on slant-range SAR imagery over steep terrain, per-vertex DEM variation causes layover-driven self-intersection of the projected perimeter. Pass `height=<shape-center HAE>` to render a clean oval. On orthorectified imagery (flat affine grid), heights are irrelevant — either mode works.

---

## Concrete Shapes

| Class | Accuracy | Use when |
|-------|----------|----------|
| `Circle` | geodesic, exact | CEP, ROI, uniform radius |
| `Ellipse` | tangent-plane, <1 mm up to ~10 km; grows as `(R/R_earth)²` | Uncertainty / CEP / standard error ellipses |
| `GeodesicEllipse` | two-foci exact on WGS-84 | Large shapes >50 km, polar regions, or when tangent-plane error is unacceptable |
| `GeoPolygon` | user-defined | Arbitrary regions; supports geodesic, rhumb, or straight edges |
| `Arc` | geodesic | Open curves, bearing-limited arcs, range rings |

### Ellipse ↔ Covariance

`Ellipse` exposes the 2×2 covariance matrix (local ENU frame at the centre) via `.covariance`, and `Ellipse.from_covariance(center, cov)` inverts the mapping. The error-propagation routines in [combine.py](combine.py) use this directly — no shape rasterization needed for Gaussian arithmetic.

### Edge modes (`GeoPolygon`)

```python
GeoPolygon(vertices, edge_mode='geodesic')  # Karney-exact, default
GeoPolygon(vertices, edge_mode='rhumb')     # constant bearing, loxodrome
GeoPolygon(vertices, edge_mode='straight')  # linear in lat/lon (fast, small-shape only)
```

---

## Combination & Error Propagation

[combine.py](combine.py) provides four operations:

| Function | Math | Result |
|----------|------|--------|
| `convolve_ellipses(ellipses)` | Σ Cov_i (Gaussian sum) | `Ellipse` — total 1-σ uncertainty |
| `combine_evidence(ellipses)` | inv(Σ inv(Cov_i)) (Fisher information) | `Ellipse` — tighter than any input |
| `minkowski_sum(a, b)` | Convex hull of vertex sums in ENU | `GeoPolygon` — set-valued worst case |
| `union_shapes([...])` / `intersect_shapes([...])` | shapely unary_union / intersection | `GeoPolygon` |

Covariance combination parallel-transports each input into a shared ENU frame via a 2D rotation for meridian convergence; `convolve_ellipses` warns when centres are more than 50 km apart (tangent-plane assumption no longer holds — use `GeodesicEllipse` or the Minkowski sum instead).

Example — three independent 1-σ error sources:

```python
from grdl.shapes import Ellipse, convolve_ellipses

pixel_loc = Ellipse(lat, lon, 8.0, 8.0, rotation_deg=0.0)
pointing  = Ellipse(lat, lon, 40.0, 20.0, rotation_deg=30.0)
georeg    = Ellipse(lat, lon, 25.0, 15.0, rotation_deg=0.0)

total_1s = convolve_ellipses([pixel_loc, pointing, georeg])
# a, b, rotation_deg reflect the combined 1-sigma uncertainty
```

See [grdl/example/shapes/error_budget_overlay.py](../example/shapes/error_budget_overlay.py) for a complete demo.

---

## Detection Cueing

[cueing.py](cueing.py) wraps any `grdl.image_processing.detection.ImageDetector` so it only fires inside a geographic region. Two modes:

- **Full-image cueing** (default): run detector on the whole image, post-filter detections against the shape mask. Preserves CFAR statistics across the scene.
- **Chip-mode cueing**: crop to the shape's pixel bounding box, detect locally, offset detections back. Faster on a small ROI.

```python
from grdl.shapes import Circle, cued_detect
from grdl.image_processing.detection.cfar import CACFARDetector

roi = Circle(center_lat=34.1, center_lon=-118.2, radius_m=15_000)
detector = CACFARDetector(guard_cells=2, training_cells=4, pfa=1e-3)

detections = cued_detect(detector, image, roi, geo)
```

See [grdl/example/shapes/cued_detection_overlay.py](../example/shapes/cued_detection_overlay.py).

---

## Display

[display.py](display.py) has two complementary entry points:

- `overlay_shape(ax, shape, geo, ...)` — attach a matplotlib `Polygon` patch to an existing `Axes`. Supports edge-only and fill+alpha. Lazy-imports matplotlib.
- `burn_shape(image, shape, geo, ...)` — rasterise the shape (outline + optional fill) into an RGB `uint8` copy. Headless / file-output use.

Both accept `height` for constant-altitude projection (see *Height handling* above).

---

## Rasterization

[rasterize.py](rasterize.py) exposes `rasterize_polygon(pixels, image_shape, fill, outline, outline_thickness, closed)` — a thin wrapper around `skimage.draw.polygon` / `polygon_perimeter`, plus a `scipy.ndimage.binary_dilation` thickening pass when `outline_thickness > 1`. Most users call it indirectly via `shape.rasterize(geo, image_shape)`.

---

## Adaptive Refinement

[refine.py](refine.py) subdivides each projected edge whose pixel-space midpoint deviates from the projected geodesic midpoint by more than `pixel_tolerance`. Fully vectorized — every iteration evaluates all candidate midpoints in one batched `latlon_to_image` call. Default tolerance of 0.5 pixel guarantees sub-pixel faithfulness under steep SAR foreshortening or sloped terrain.

Configurable per call:

```python
shape.to_pixels(geo, n_initial=128, pixel_tolerance=0.5, max_subdivisions=6)
shape.to_pixels(geo, refine=False)   # skip refinement — faster, less accurate
```

---

## Backend & Acceleration

[backend.py](backend.py) provides a singleton `ComputeBackend` dispatcher with four tiers: CuPy (GPU arrays), Numba (JIT), thread-parallel, and NumPy (baseline). All public shape APIs accept and return `numpy.ndarray` — GPU arrays are internal only. The dispatcher is informational today: `detect_backend()` reports what is available; `set_backend()` lets callers pin a choice. `cupy_available()`, `numba_available()`, and `cucim_available()` probe individual backends.

---

## Dependencies

| Package | Required by |
|---------|-------------|
| `numpy` | all |
| `pyproj` | Karney geodesics (circle, arc, ellipse, polygon, combine, refine) |
| `scipy` | rasterize (binary_dilation) |
| `scikit-image` | rasterize (polygon, polygon_perimeter) |
| `shapely` | union_shapes, intersect_shapes |
| `matplotlib` | overlay_shape (lazy import) |

All optional accelerators (`cupy`, `numba`, `cucim`) are probed at import and never required.

---

## File Layout

```
shapes/
  __init__.py         Public re-exports and the advertised API surface.
  base.py             GeographicShape ABC, perimeter_latlon, to_pixels,
                      rasterize, contains, _sample_dem_heights helper.
  circle.py           Circle — geodesic forward from centre on n bearings.
  arc.py              Arc — open perimeter, bearing-range-limited.
  ellipse.py          Ellipse (tangent-plane) + GeodesicEllipse (two-foci,
                      per-bearing bisection on WGS-84).
  polygon.py          GeoPolygon — geodesic/rhumb/straight edge modes.
  combine.py          convolve_ellipses, combine_evidence, minkowski_sum,
                      union_shapes, intersect_shapes.
  cueing.py           cued_detect — full-image or chip-mode detector cueing.
  display.py          overlay_shape, overlay_shapes, burn_shape.
  rasterize.py        rasterize_polygon primitive (skimage + scipy).
  refine.py           adaptive_refine — pixel-tolerance perimeter subdivision.
  backend.py          ComputeBackend dispatcher; availability probes.
```

---

## Design Principles

- **Accuracy first.** Karney geodesic maths everywhere a geodesic makes sense; no local-flat-earth shortcuts unless explicitly opted in via `GeoPolygon(edge_mode='straight')` or `Ellipse` within its ~50 km validity window.
- **One concept per file.** Circles, ellipses, polygons, arcs, display, rasterization, refinement, combination, and backend dispatch are all separately importable.
- **Vectorized.** Every public method accepts and emits `numpy.ndarray`. Projection and refinement issue batched calls to the underlying geolocation — no per-vertex Python loops in the hot path.
- **Pluggable geolocation.** Shapes do not know about SAR vs EO vs basemap. Any GRDL `Geolocation` that implements `latlon_to_image` works — including `ChipGeolocation` wrappers.
- **Explicit height semantics.** Terrain-correct projection (DEM) is the default; caller-supplied constant height is available for slant-plane rendering over steep terrain.
