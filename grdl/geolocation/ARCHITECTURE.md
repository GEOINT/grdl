# Geolocation Module — Architecture

*Modified: 2026-03-22*

## Overview

The `geolocation` module transforms between image pixel coordinates and
geographic coordinates (latitude/longitude/height) for any supported
imagery type. It handles SAR slant-range geometry, EO rational polynomial
models, geocoded raster affine transforms, and terrain elevation lookup.

Every geolocation class inherits from `Geolocation`, which provides
scalar/stacked-array dispatch, DEM-iterative refinement, and footprint
computation. Subclasses implement only the vectorized
`_image_to_latlon_array` and `_latlon_to_image_array` methods.

All public methods use the **(N, M) stacked ndarray** convention: inputs
are (N, 2) or (N, 3) arrays where each row is a point, and outputs are
(N, M) arrays. Scalar inputs return (M,) arrays compatible with tuple
unpacking.

---

## Directory Layout

```
geolocation/
├── __init__.py              # Top-level re-exports
├── base.py                  # Geolocation ABC, NoGeolocation, DEM builder
├── projection.py            # R/Rdot engine (COAProjection, formation projectors)
├── coordinates.py           # geodetic ↔ ECEF ↔ ENU conversions (WGS-84)
├── utils.py                 # Footprint, bounds, geographic distance helpers
│
├── sar/                     # SAR-specific geolocation
│   ├── __init__.py          #   Re-exports SAR classes
│   ├── _backend.py          #   sarpy/sarkit availability probing
│   ├── sicd.py              #   SICDGeolocation (native R/Rdot, sarpy, sarkit)
│   ├── sidd.py              #   SIDDGeolocation (Plane/Geographic/Cylindrical)
│   ├── gcp.py               #   GCPGeolocation (Delaunay on GCPs)
│   ├── nisar.py             #   NISARGeolocation (RSLC grid spline)
│   └── sentinel1_slc.py     #   Sentinel1SLCGeolocation (annotation grid spline)
│
├── eo/                      # EO-specific geolocation
│   ├── __init__.py          #   Re-exports EO classes
│   ├── _backend.py          #   rasterio/pyproj availability probing
│   ├── affine.py            #   AffineGeolocation (geocoded rasters)
│   ├── rpc.py               #   RPCGeolocation (RPC00B rational polynomials)
│   └── rsm.py               #   RSMGeolocation (RSM variable-order polynomials)
│
└── elevation/               # DEM / terrain elevation models
    ├── __init__.py          #   Re-exports, conditional imports
    ├── _backend.py          #   rasterio availability probing
    ├── base.py              #   ElevationModel ABC
    ├── constant.py          #   ConstantElevation (fixed-height fallback)
    ├── dted.py              #   DTEDElevation (DTED Level 0/1/2 tiles)
    ├── geotiff_dem.py       #   GeoTIFFDEM (single GeoTIFF DEM)
    ├── tiled_geotiff_dem.py #   TiledGeoTIFFDEM (multi-tile: FABDEM, SRTM, Copernicus)
    ├── geoid.py             #   GeoidCorrection (EGM96 undulation lookup)
    └── open_elevation.py    #   open_elevation() factory (auto-detect format)
```

---

## Class Hierarchy

```
Geolocation (ABC)                       ← base.py
│  image_to_latlon()                      scalar/stacked dispatch → (N, 3)
│  latlon_to_image()                      scalar/stacked dispatch → (N, 2)
│  _image_to_latlon_with_dem()            iterative DEM refinement loop
│  get_footprint()                        perimeter sampling → polygon
│  get_bounds()                           bounding box from footprint
│
├── NoGeolocation                       ← base.py
│     Raises NotImplementedError for all operations
│
├── SICDGeolocation                     ← sar/sicd.py
│     Native R/Rdot via COAProjection, sarpy, or sarkit backends
│
├── SIDDGeolocation                     ← sar/sidd.py
│     PlaneProjection, GeographicProjection, CylindricalProjection
│     Optional R/Rdot refinement when TimeCOAPoly + ARPPoly available
│
├── GCPGeolocation                      ← sar/gcp.py
│     Delaunay triangulation on ground control points
│
├── NISARGeolocation                    ← sar/nisar.py
│     RectBivariateSpline on RSLC geolocation grid
│     Dispatches to AffineGeolocation for GSLC products
│
├── Sentinel1SLCGeolocation             ← sar/sentinel1_slc.py
│     RectBivariateSpline on annotation grid
│
├── AffineGeolocation                   ← eo/affine.py
│     Affine transform + pyproj CRS reprojection
│
├── RPCGeolocation                      ← eo/rpc.py
│     RPC00B rational polynomials (forward: direct, inverse: Newton)
│
└── RSMGeolocation                      ← eo/rsm.py
      RSM variable-order polynomials (forward: direct, inverse: Newton)


ElevationModel (ABC)                    ← elevation/base.py
│  get_elevation()                        scalar/array dispatch + geoid
│
├── ConstantElevation                   ← elevation/constant.py
│     Returns a fixed height for all queries
│
├── DTEDElevation                       ← elevation/dted.py
│     DTED Level 0/1/2 tiles via rasterio, spatial tile indexing
│
├── GeoTIFFDEM                          ← elevation/geotiff_dem.py
│     Single GeoTIFF DEM via rasterio, bilinear/bicubic/quintic
│
└── TiledGeoTIFFDEM                     ← elevation/tiled_geotiff_dem.py
      Multi-tile GeoTIFF (FABDEM, SRTM, Copernicus) with cross-tile
      interpolation and LRU tile cache


GeoidCorrection                         ← elevation/geoid.py
  EGM96 undulation lookup (PGM grid or GeoTIFF)
  Composed into ElevationModel instances via constructor injection


COAProjection                           ← projection.py
  Image pixel → (R, Rdot, time_coa, arp, varp)
  Dispatches to formation-specific projectors:
    _pfa_projector        PFA (Polar Format)
    _inca_projector       INCA (spotlight/sliding spotlight)
    _rgazcomp_projector   RgAzComp (range-azimuth compression)
    _plane_projector      PLANE (general fallback)
```

---

## When to Use What

### Sensor-Specific Geolocation

| Imagery type | Class | Projection method |
|-------------|-------|-------------------|
| SICD (complex SAR) | `SICDGeolocation` | R/Rdot contour intersection |
| SIDD (detected SAR) | `SIDDGeolocation` | Plane/Geographic/Cylindrical grid |
| BIOMASS L1 | `GCPGeolocation` | Delaunay interpolation on GCPs |
| NISAR RSLC | `NISARGeolocation` | Bivariate spline on geolocation grid |
| NISAR GSLC | `NISARGeolocation` → `AffineGeolocation` | Affine (auto-dispatched) |
| Sentinel-1 SLC | `Sentinel1SLCGeolocation` | Bivariate spline on annotation grid |
| Geocoded raster (GeoTIFF, etc.) | `AffineGeolocation` | Affine + CRS reprojection |
| EO NITF with RPC | `RPCGeolocation` | Rational polynomial coefficients |
| EO NITF with RSM | `RSMGeolocation` | Replacement sensor model polynomials |
| Unknown / no metadata | `NoGeolocation` | Raises NotImplementedError |

### Elevation Models

| DEM source | Class | Notes |
|-----------|-------|-------|
| DTED folder (Level 0/1/2) | `DTEDElevation` | Auto-indexes `.dt0`/`.dt1`/`.dt2` tiles |
| Single GeoTIFF DEM | `GeoTIFFDEM` | Any CRS, bilinear/bicubic/quintic interpolation |
| Multi-tile GeoTIFF folder | `TiledGeoTIFFDEM` | FABDEM, SRTM, Copernicus; cross-tile interpolation |
| No DEM available | `ConstantElevation` | Fixed height fallback |
| Auto-detect format | `open_elevation()` | Factory: inspects path, returns best model |

### Coordinate Conversions

| Conversion | Function | Input | Output |
|-----------|----------|-------|--------|
| Geodetic → ECEF | `geodetic_to_ecef(pts_Nx3)` | (N, 3) `[lat, lon, alt]` | (N, 3) `[x, y, z]` |
| ECEF → Geodetic | `ecef_to_geodetic(pts_Nx3)` | (N, 3) `[x, y, z]` | (N, 3) `[lat, lon, alt]` |
| Geodetic → Local ENU | `geodetic_to_enu(pts_Nx3, ref_3)` | (N, 3) + (3,) ref | (N, 3) `[e, n, u]` |
| ENU → Geodetic | `enu_to_geodetic(pts_Nx3, ref_3)` | (N, 3) + (3,) ref | (N, 3) `[lat, lon, alt]` |
| Meters per degree | `meters_per_degree(lat)` | scalar | (2,) `[m_lat, m_lon]` |

### Low-Level R/Rdot Projection (Advanced)

| Need | Function |
|------|----------|
| Pixel → ground at constant HAE | `image_to_ground_hae(coa, row, col, hae)` |
| Pixel → ground with DEM | `image_to_ground_dem(coa, row, col, elevation_model)` |
| Pixel → ground on arbitrary plane | `image_to_ground_plane(coa, row, col, gref, uZ)` |
| Ground → pixel (inverse) | `ground_to_image(coa, ecf_point)` |
| WGS-84 surface normal | `wgs84_norm(ecf)` |

---

## Key Design Patterns

### 1. Stacked (N, M) Array Convention

All public methods use stacked ndarrays where each row is a point:

```
# Scalar → (M,) result (tuple-unpackable)
lat, lon, h = geo.image_to_latlon(500, 1000)         → (3,)
row, col = geo.latlon_to_image(34.0, -118.0)         → (2,)

# (N, 2) input → (N, 3) or (N, 2) output
geo.image_to_latlon(pixels_Nx2)                       → (N, 3)
geo.latlon_to_image(coords_Nx2)                       → (N, 2)
geo.latlon_to_image(coords_Nx3)                       → (N, 2)  # height in col 3
```

Implemented via `_is_scalar()` and `_to_stacked()` helpers in `base.py`.
Subclasses only implement the vectorized `_*_array()` methods — the
ABC handles all dispatch logic.

**Indexing conventions:**
- `result[i]` — the i-th point (row vector)
- `result[:, 0]` — first coordinate across all points (e.g., all latitudes)

### 2. Iterative DEM Refinement

When a DEM is configured, `_image_to_latlon_with_dem()` iterates
**per point**:

```
for each point independently:
  project at initial height
    → DEM lookup at projected (lat, lon)
    → re-project at DEM height
    → repeat until height converges (< 0.5 m tolerance, max 5 iterations)
```

This handles height-dependent projections (RPC, RSM) where the ground
position shifts with elevation. For affine geolocation (height-
independent), a single iteration suffices.

**`_handles_dem_internally` bypass:** Subclasses whose sensor model
already performs per-point DEM iteration inside `_image_to_latlon_array`
set `_handles_dem_internally = True`. The base class wrapper then passes
through to the subclass directly, avoiding a redundant outer DEM loop.

| Class | `_handles_dem_internally` | Why |
|-------|--------------------------|-----|
| `SICDGeolocation` (native) | `True` | R/Rdot engine passes `self.elevation` to `image_to_ground_hae()` which iterates per-point |
| `SIDDGeolocation` (R/Rdot) | `True` | Same R/Rdot engine with DEM integration |
| `SIDDGeolocation` (grid-only) | `False` | Grid projection is height-independent; base class provides DEM heights |
| `RPCGeolocation` | `False` | Newton-Raphson at fixed HAE; base class iterates with DEM |
| `RSMGeolocation` | `False` | Same pattern as RPC |
| `AffineGeolocation` | `False` | Affine is 2D; base class provides DEM heights in output |
| Grid-based (NISAR, S1) | `False` | Grid positions are fixed; base class provides DEM heights |

The inverse path uses `_latlon_to_image_with_dem()` which does a
single DEM lookup (no iteration needed since lat/lon are already known).
The same `_handles_dem_internally` flag controls bypass.

### 3. Backend Selection (SAR)

`SICDGeolocation` supports three projection backends:

| Backend | Dependency | Preference |
|---------|------------|------------|
| `native` | None (pure GRDL) | Default when no sarpy/sarkit |
| `sarpy` | `sarpy` | Used when available |
| `sarkit` | `sarkit` | Used when available, no sarpy |

Selection is automatic via `sar/_backend.py` probing. The `native`
backend uses `COAProjection` from `projection.py` — a standalone
implementation of NGA.STND.0024-1 Volume 3 with no external
dependencies beyond numpy/scipy.

### 4. Grid Interpolation (Sentinel-1, NISAR)

Sensors that provide geolocation grids (regular in image space, warped
in geographic space) use a common pattern:

- **Forward** (image → geo): `RectBivariateSpline` on the regular
  image-space grid for fast O(1) lookup.
- **Inverse** (geo → image): `LinearNDInterpolator` on Delaunay
  triangulation of the scattered geographic points.

### 5. Formation-Specific Projectors (R/Rdot Engine)

`COAProjection` dispatches to formation-specific projectors based on
the SICD image formation algorithm:

```
COAProjection.__init__(metadata)
  → inspects metadata.image_formation.image_form_algo
  → selects _pfa_projector, _inca_projector, _rgazcomp_projector,
    or _plane_projector
  → projector converts (row, col) → (time_coa, R, Rdot)
```

Each projector handles the formation-specific relationship between
pixel indices and physical range/range-rate. The `COAProjection.projection()`
method then resolves ARP position and velocity at the center of aperture
time to complete the (R, Rdot, ARP, VARP) tuple needed for ground
intersection.

### 6. Geoid Correction (Composition)

`GeoidCorrection` is not an `ElevationModel` — it is composed into
elevation models via constructor injection:

```python
geoid = GeoidCorrection('/path/to/egm96.pgm')
dem = GeoTIFFDEM('/path/to/dem.tif', geoid=geoid)

# dem.get_elevation() internally adds geoid undulation to convert
# MSL heights (from the DEM file) to HAE (WGS-84 ellipsoid heights)
```

---

## Data Flow

### Image → Geographic (Forward)

```
User: geo.image_to_latlon(row, col)           → (3,)  [lat, lon, h]
User: geo.image_to_latlon(pixels_Nx2)         → (N, 3) [lat, lon, h]
  │
  ├─ Scalar? → pack into (1, 2) array
  ├─ Already (N, 2)? → use directly
  │
  └─ _image_to_latlon_with_dem(pixels_Nx2)
       │
       ├─ No DEM → _image_to_latlon_array(pixels_Nx2)
       │            └─ Subclass-specific projection → (N, 3)
       │
       ├─ _handles_dem_internally? → _image_to_latlon_array()
       │   directly (subclass does its own DEM iteration)
       │
       └─ Base-class DEM loop → iterate per point:
            project → DEM lookup → update height → converge
            → (N, 3) [lat, lon, dem_h]
```

### Geographic → Image (Inverse)

```
User: geo.latlon_to_image(lat, lon)            → (2,)  [row, col]
User: geo.latlon_to_image(lat, lon, h)         → (2,)  [row, col]
User: geo.latlon_to_image(coords_Nx2)          → (N, 2) [row, col]
User: geo.latlon_to_image(coords_Nx3)          → (N, 2) [row, col]  (h in col 3)
  │
  ├─ Scalar? → pack into (1, 2) or (1, 3) array
  ├─ Already (N, 2) or (N, 3)? → use directly
  │
  └─ _latlon_to_image_with_dem(coords)
       │
       ├─ No DEM → _latlon_to_image_array(coords)
       │
       ├─ _handles_dem_internally? → _latlon_to_image_array()
       │   directly (subclass queries DEM itself)
       │
       └─ Base-class DEM → single lookup at (lat, lon)
            → pass per-point DEM heights to subclass
       │
       └─ _latlon_to_image_array(coords_with_dem_h)
            │                                        → (N, 2)
            ├─ SAR (SICD): ground_to_image() via COAProjection
            ├─ SAR (grid): LinearNDInterpolator lookup
            ├─ EO (affine): inverse affine + CRS transform
            └─ EO (RPC/RSM): direct polynomial evaluation
```

### Round-Trip Pattern

```
pixels (N, 2) [row, col]
  → geo.image_to_latlon(pixels)        → geo_pts (N, 3) [lat, lon, h]
  → geo.latlon_to_image(geo_pts)       → px_back (N, 2) [row, col]
  → np.linalg.norm(px_back - pixels, axis=1)  → per-point error
```

---

## API Summary

### Geolocation Methods

| Method | Input | Output |
|--------|-------|--------|
| `image_to_latlon(row, col)` | two scalars | (3,) `[lat, lon, h]` |
| `image_to_latlon(pixels_Nx2)` | (N, 2) `[row, col]` | (N, 3) `[lat, lon, h]` |
| `latlon_to_image(lat, lon)` | two scalars | (2,) `[row, col]` |
| `latlon_to_image(lat, lon, h)` | three scalars | (2,) `[row, col]` |
| `latlon_to_image(coords_Nx2)` | (N, 2) `[lat, lon]` | (N, 2) `[row, col]` |
| `latlon_to_image(coords_Nx3)` | (N, 3) `[lat, lon, h]` | (N, 2) `[row, col]` |

### Coordinate Functions

| Function | Input | Output |
|----------|-------|--------|
| `geodetic_to_ecef(pts)` | (N, 3) or (3,) `[lat, lon, alt]` | (N, 3) or (3,) `[x, y, z]` |
| `ecef_to_geodetic(pts)` | (N, 3) or (3,) `[x, y, z]` | (N, 3) or (3,) `[lat, lon, alt]` |
| `geodetic_to_enu(pts, ref)` | (N, 3) + (3,) `[lat, lon, alt]` | (N, 3) `[e, n, u]` |
| `enu_to_geodetic(pts, ref)` | (N, 3) + (3,) `[lat, lon, alt]` | (N, 3) `[lat, lon, alt]` |
| `meters_per_degree(lat)` | scalar | (2,) `[m_lat, m_lon]` |

---

## Dependencies

| Submodule | Required | Optional |
|-----------|----------|----------|
| `base`, `coordinates`, `utils` | numpy | — |
| `projection` | numpy, scipy | — |
| `sar/sicd` | numpy | sarpy, sarkit |
| `sar/sidd`, `sar/gcp` | numpy, scipy | — |
| `sar/nisar`, `sar/sentinel1_slc` | numpy, scipy | — |
| `eo/affine` | numpy, rasterio, pyproj | — |
| `eo/rpc`, `eo/rsm` | numpy | — |
| `elevation/dted`, `elevation/geotiff_dem` | numpy, rasterio | pyproj |
| `elevation/geoid` | numpy | rasterio |

All optional dependencies are probed at import time via `_backend.py`
modules. Missing dependencies produce clear `ImportError` messages at
construction time, not silent degradation.
