# Geolocation Module — Implementation Patterns

Reference for recurring patterns in `grdl/geolocation/`. Follow these when adding new geolocation subclasses, elevation models, or coordinate transforms.

**Key dependency:** Geolocation classes consume metadata dataclasses from `grdl/IO/models/` via `from_reader()` factories. The metadata flow is always:

```
Reader (IO module)
  -> metadata dataclass (IO/models)
    -> from_reader() factory (geolocation)
      -> Geolocation subclass instance
```

---

## 1. Geolocation ABC — Template Method on 1D Arrays

**File:** `base.py`

All geolocation subclasses implement two protected methods that operate on 1D arrays. The base class handles scalar/batch dispatch, stacked array formatting, and DEM iteration.

```
Geolocation (ABC)
├── _image_to_latlon_array(rows_1d, cols_1d, height)  # ABSTRACT -> (lats, lons, heights)
├── _latlon_to_image_array(lats_1d, lons_1d, height)  # ABSTRACT -> (rows, cols)
├── image_to_latlon(...)     # PUBLIC — dispatches scalar/stacked/separate, returns (N,3) or (3,)
├── latlon_to_image(...)     # PUBLIC — dispatches scalar/stacked/separate, returns (N,2) or (2,)
├── _resolve_height(height)  # Height fallback: explicit > default_hae
├── _fill_nan_heights(dem_h, fallback)  # NaN fill for DEM gaps
└── default_hae (property)   # Override per subclass (SICD: SCP HAE, RPC: 0.0)
```

**Rules:**
- Subclasses **only** implement `_image_to_latlon_array` and `_latlon_to_image_array`
- Internal arrays are always 1D `(N,)` — never touch stacked `(N, M)` arrays
- Public methods handle three input forms:
  - Scalar: `geo.image_to_latlon(500, 1000)` -> `(3,)` array (tuple-unpackable)
  - Stacked: `geo.image_to_latlon(points_Nx2)` -> `(N, 3)` array
  - Separate arrays: `geo.image_to_latlon(rows_arr, cols_arr)` -> tuple of arrays
- Height parameter: scalar broadcasts to all points; array provides per-point heights

---

## 2. Stacked ndarray Convention

All geolocation and coordinate APIs use `(N, M)` stacked arrays where `result[i]` gives the i-th point.

| Method | Input | Output |
|--------|-------|--------|
| `image_to_latlon(row, col)` | scalar | `(3,)` — `[lat, lon, h]` |
| `image_to_latlon(pixels_Nx2)` | `(N, 2)` — `[row, col]` | `(N, 3)` — `[lat, lon, h]` |
| `latlon_to_image(lat, lon, h)` | scalar | `(2,)` — `[row, col]` |
| `latlon_to_image(points_Nx3)` | `(N, 3)` — `[lat, lon, h]` | `(N, 2)` — `[row, col]` |
| `geodetic_to_ecef(point)` | `(3,)` — `[lat, lon, h]` | `(3,)` — `[X, Y, Z]` |
| `geodetic_to_ecef(points)` | `(N, 3)` | `(N, 3)` |

**Rules:**
- Do NOT provide separate `*_batch()` methods — one method handles both
- Scalar tuple unpacking works: `lat, lon, h = geo.image_to_latlon(r, c)`
- Batch input uses stacked columns, not separate row/col arrays

---

## 3. Height Resolution and NaN Fill

**File:** `base.py`

Two base-class methods ensure consistent DEM behavior across all subclasses.

**`_resolve_height(height)`** — "What height when the caller passes 0.0?"
- Returns explicit height if non-zero
- Falls back to `default_hae` (which each subclass overrides)

**`_fill_nan_heights(dem_h, fallback_height)`** — "What when DEM has no coverage?"
- Fills NaN gaps in DEM-queried heights
- Uses `fallback_height` if non-zero, otherwise `default_hae`

```python
# CORRECT: use base class methods
hae = self._resolve_height(height)
self._fill_nan_heights(dem_h, height)

# WRONG: inline reimplementation
hae = float(height) if height != 0.0 else self._get_scp_hae()  # don't do this
```

**`default_hae` by subclass:**

| Geolocation | `default_hae` | Source |
|-------------|---------------|--------|
| `SICDGeolocation` | SCP HAE | `geo_data.scp` (LLH HAE, or derived from SCP ECF) |
| `SIDDGeolocation` | measurement reference point HAE | plane-projection reference point / SRP |
| `CornerGeolocation` | construction `height` (default 0.0) | the `height=` constructor arg |
| `RPCGeolocation` | 0.0 | base-class default (not overridden) |
| `RSMGeolocation` | 0.0 | base-class default (not overridden) |
| `AffineGeolocation` | 0.0 | base-class default (not overridden) |
| `NISARGeolocation`, `Sentinel1SLCGeolocation`, `GCPGeolocation` | 0.0 | grid/GCP heights are intrinsic; base-class default |

**Rules:**
- **Never** reimplement height resolution inline
- Override `default_hae` property in your subclass when the sensor model carries a meaningful scene reference height (SCP, reference point); grid- and polynomial-based models that already encode height leave it at the base default of 0.0

---

## 4. DEM Ownership

**The geolocation object owns the DEM. The orthorectifier does not.**

DEM is attached to `geo.elevation` and used internally by R/Rdot iteration or the base class DEM loop. The orthorectifier maps coordinates *through* the geolocation — it never accepts an `elevation` parameter.

```python
# CORRECT
geo = SICDGeolocation.from_reader(reader, backend='native')
geo.elevation = open_elevation(dted_path, geoid_path=geoid_path)
result = orthorectify(geolocation=geo, reader=reader, output_grid=grid)

# WRONG — orthorectifier has no elevation parameter
result = orthorectify(geolocation=geo, elevation=dem, ...)  # does not exist
```

**Base class DEM iteration** (`_image_to_latlon_with_dem`):
- Projects pixel -> ground at initial height
- Queries DEM at projected lat/lon
- Re-projects at DEM height
- Converges in up to 5 iterations
- Subclasses that handle DEM internally set `_handles_dem_internally = True` to bypass

---

## 5. Newton-Raphson Inversion (Image -> Ground)

**Files:** `eo/rpc.py`, `eo/rsm.py`

When the sensor model maps ground -> image (direct polynomial), inverting it requires Newton-Raphson iteration.

```python
def _newton_raphson_inverse(self, rows, cols, h_arr, coefficients):
    n = len(rows)

    # Initial guess from normalization center
    lats = np.full(n, center_lat)
    lons = np.full(n, center_lon)

    # Finite-difference step sizes from coefficient scale factors
    dlat = lat_scale * 1e-6
    dlon = lon_scale * 1e-6

    for _ in range(20):  # max iterations
        r0, c0 = evaluate(lats, lons, h_arr, coefficients)
        dr, dc = rows - r0, cols - c0

        if np.max(np.sqrt(dr**2 + dc**2)) < 1e-8:  # pixel tolerance
            break

        # 2x2 Jacobian via finite differences
        r_dlat, c_dlat = evaluate(lats + dlat, lons, h_arr, coefficients)
        r_dlon, c_dlon = evaluate(lats, lons + dlon, h_arr, coefficients)

        dr_dlat = (r_dlat - r0) / dlat
        dc_dlat = (c_dlat - c0) / dlat
        dr_dlon = (r_dlon - r0) / dlon
        dc_dlon = (c_dlon - c0) / dlon

        det = dr_dlat * dc_dlon - dr_dlon * dc_dlat
        det = np.where(np.abs(det) < 1e-30, 1e-30, det)  # singular guard

        lats += (dc_dlon * dr - dr_dlon * dc) / det
        lons += (-dc_dlat * dr + dr_dlat * dc) / det

    return lats, lons, h_arr.copy()
```

**Rules:**
- Max 20 iterations, 1e-8 pixel tolerance
- Initial guess from normalization center (RPC: `lat_off/long_off`, RSM: center of norm window)
- Finite-difference step size derived from coefficient scale factors (not hard-coded)
- Singular Jacobian guard: clamp determinant to 1e-30 minimum
- Fully vectorized — operates on all N points simultaneously

---

## 6. ICHIPB Chip Transform Integration

**Files:** `eo/rpc.py`, `eo/rsm.py`

When ICHIPB metadata is present, chip pixel coordinates must be transformed to full-image coordinates before polynomial evaluation.

```python
# In _image_to_latlon_array (image -> ground):
if self.ichipb is not None:
    rows, cols = _apply_ichipb_forward(rows, cols, self.ichipb)  # chip -> full
# ... then evaluate polynomial with full-image coords ...

# In _latlon_to_image_array (ground -> image):
rows, cols = polynomial_evaluate(lats, lons, h, coefficients)
if self.ichipb is not None:
    rows, cols = _apply_ichipb_inverse(rows, cols, self.ichipb)  # full -> chip
```

**Transform:**
```python
full_row = fi_row_off + fi_row_scale * chip_row
full_col = fi_col_off + fi_col_scale * chip_col
```

**Rules:**
- Forward (chip -> full) is applied **before** polynomial evaluation in `_image_to_latlon_array`
- Inverse (full -> chip) is applied **after** polynomial evaluation in `_latlon_to_image_array`
- Pass `ichipb` through `from_reader()` factory — extract from `reader.metadata.ichipb`

---

## 7. Multi-Segment RSM Dispatch

**File:** `eo/rsm.py`

When RSM is partitioned into a grid of sections, select the correct segment per pixel.

```
Constructor
├── Accept RSMSegmentGrid (dict of (rsn, csn) -> RSMCoefficients)
├── _precompute_segment_bounds() — pixel bounding box per segment from norm center +/- norm_sf
└── Store single-segment fallback (self.rsm) for backward compat

Ground -> Image (_latlon_to_image_array)
├── Evaluate ALL segments (vectorized across all points)
├── Per-point: pick segment whose output pixel falls in its valid bounds
└── Fallback to nearest segment center if outside all bounds

Image -> Ground (_image_to_latlon_array)
├── _select_segments_for_pixels(rows, cols) — assign each pixel to a segment
├── Group point indices by segment key
├── Run Newton-Raphson per group with that segment's coefficients
└── Reassemble results in original order
```

**Rules:**
- If only one segment, bypass all multi-segment logic (stored as `self._segments = None`)
- Segment bounds: `[row_off - |row_norm_sf|, row_off + |row_norm_sf|]`
- Keep backward-compatible `self.rsm` for single-segment access

---

## 8. from_reader() Factory

Every geolocation subclass provides a `from_reader()` classmethod that extracts metadata from a reader. The DEM-related parameters are declared as **explicit named keyword arguments** (not `**kwargs`) and forwarded to the constructor by name — so editors and type checkers see them and callers get clear errors.

```python
@classmethod
def from_reader(
    cls,
    reader: object,
    dem_path: Optional[str] = None,
    geoid_path: Optional[str] = None,
    interpolation: int = 3,
) -> 'MyGeolocation':
    meta = reader.metadata
    if meta.<required_field> is None:
        raise ValueError("Reader metadata has no <required_field>.")
    shape = reader.get_shape()
    return cls(
        coefficients=meta.<coefficients>,
        ichipb=getattr(meta, 'ichipb', None),
        shape=shape,
        dem_path=dem_path,
        geoid_path=geoid_path,
        interpolation=interpolation,
    )
```

**The `dem_path` / `geoid_path` / `interpolation` contract (recent change):**
Every `from_reader()` factory now accepts these three explicitly and forwards
them to the constructor, which calls `open_elevation()` to populate
`geo.elevation`. This makes the one-line `Class.from_reader(reader,
dem_path=..., interpolation=...)` form equivalent to constructing and then
assigning `geo.elevation`.

| Geolocation | `dem_path`/`geoid_path`/`interpolation` in `from_reader`? | Extra params |
|-------------|-----------------------------------------------------------|--------------|
| `SICDGeolocation` | Yes | `backend`, `delta_arp`, `delta_varp`, `range_bias`, `per_point_normal` |
| `SIDDGeolocation` | Yes | `refine`, `per_point_normal` |
| `NISARGeolocation` | Yes | — (GSLC dispatches to `AffineGeolocation`) |
| `Sentinel1SLCGeolocation` | Yes | — |
| `AffineGeolocation` | Yes | — |
| `RPCGeolocation` | Yes | (ichipb pulled from metadata) |
| `RSMGeolocation` | Yes | (ichipb, rsm_id, rsm_segments pulled from metadata) |
| `CornerGeolocation` | Yes | `height` |
| `GCPGeolocation` | **No** — only `crs` | heights come from the GCPs themselves |

> Gotcha: `SICDGeolocation.from_reader` defaults `per_point_normal=False`,
> while the `SICDGeolocation` constructor defaults it to `True`.

**Rules:**
- Validate required metadata fields; raise `ValueError` with clear message
- Use `getattr(meta, 'field', None)` for optional fields (not all metadata types have all fields)
- Declare `dem_path`/`geoid_path`/`interpolation` as explicit named parameters and forward them by name (do not hide them behind `**kwargs`)
- Extract shape from `reader.get_shape()`, not from metadata directly

---

## 9. create_geolocation() Global Factory

**File:** `__init__.py`

Dispatches on reader metadata type to select the correct geolocation subclass.

```python
def create_geolocation(reader, **kwargs) -> Geolocation:
    meta = reader.metadata
    # Lazy imports to avoid circular dependencies
    try:
        from grdl.IO.models.eo_nitf import EONITFMetadata
        if isinstance(meta, EONITFMetadata):
            # RSM preferred over RPC per STDI-0002 (higher-fidelity geometry)
            if getattr(meta, 'rsm', None) is not None:
                return RSMGeolocation.from_reader(reader, **kwargs)
            if getattr(meta, 'rpc', None) is not None:
                return RPCGeolocation.from_reader(reader, **kwargs)
            # Corner fallback; ValueError/TypeError fall through
            try:
                return CornerGeolocation.from_reader(reader, **kwargs)
            except (ValueError, TypeError):
                pass
    except ImportError:
        pass
    # ... more isinstance checks ...
    raise TypeError(f"Cannot determine geolocation for {type(meta).__name__}")
```

**Dispatch priority (first match wins):**
1. SICD -> SICDGeolocation
2. SIDD -> SIDDGeolocation
3. NISAR -> NISARGeolocation (GSLC products -> AffineGeolocation)
4. Sentinel-1 SLC -> Sentinel1SLCGeolocation
5. EO NITF with RSM -> RSMGeolocation (**RSM preferred over RPC** per STDI-0002)
6. EO NITF with RPC -> RPCGeolocation
7. EO NITF without RPC/RSM -> CornerGeolocation (corner-coordinate fallback)
8. BIOMASS with GCPs -> GCPGeolocation
9. Any reader with affine transform + CRS -> AffineGeolocation
10. Any metadata with GCPs -> GCPGeolocation

**Rules:**
- Use lazy imports (`try/except ImportError`) to avoid circular deps and allow partial install
- Forward all `**kwargs` to `from_reader()`
- RSM is chosen over RPC whenever both TREs are present
- The corner fallback is wrapped in `try/except (ValueError, TypeError)` so a missing corner source falls through to the affine/GCP fallbacks
- When adding a new sensor: add an `isinstance` check in priority order

---

## 10. ChipGeolocation Offset Wrapper

**File:** `chip.py`

Wraps a parent geolocation with a row/col offset for sub-region processing.

```python
class ChipGeolocation(Geolocation):
    def __init__(self, geolocation, row_offset, col_offset, shape):
        self._parent = geolocation
        self._row_off = row_offset
        self._col_off = col_offset
        self.elevation = geolocation.elevation  # inherit DEM
        super().__init__(shape, ...)

    def _image_to_latlon_array(self, rows, cols, height):
        return self._parent._image_to_latlon_array(
            rows + self._row_off, cols + self._col_off, height)

    def _latlon_to_image_array(self, lats, lons, height):
        full_rows, full_cols = self._parent._latlon_to_image_array(lats, lons, height)
        return full_rows - self._row_off, full_cols - self._col_off
```

**Rules:**
- Inherits parent's elevation model — no separate DEM
- Inherits `_handles_dem_internally` from parent
- Offset is added before delegation, subtracted from result

---

## 11. Elevation Model Hierarchy

**File:** `elevation/base.py`, `elevation/dted.py`, `elevation/geotiff_dem.py`

All DEM implementations inherit `ElevationModel` and implement `_get_elevation_array()`.

```
ElevationModel (ABC)
├── _get_elevation_array(lats_1d, lons_1d) -> heights_1d  # ABSTRACT
├── get_elevation(...)  # PUBLIC — handles scalar/stacked/separate dispatch
└── Geoid correction applied automatically if geoid_path provided
```

**Subclass contract:**
- Return MSL heights (geoid correction to HAE is done by base class)
- Return `np.nan` for locations with no coverage (base class fills with fallback)
- Accept 1D arrays, return 1D array of same length

**DTED implementation specifics:**
- Scans directory for `.dt0/.dt1/.dt2` tiles at init
- Builds spatial index: `(lon_floor, lat_floor)` -> file path
- Groups query points by tile, batch-reads per tile
- Configurable interpolation: 1 (bilinear), 3 (bicubic, default), 5 (quintic)
- Cross-tile boundary handling loads adjacent tiles as needed
- `.dt2` > `.dt1` > `.dt0` priority; discovery covers the standard
  `<root>/<lon>/<lat>.dt?` layout and nested archive layouts
  `<root>/dted/dted{2,1,0}/...`; supports `bbox`-bounded discovery

**Tiled backends (recommended for large coverage):**

| Class | File | Use when |
|-------|------|----------|
| `TiledGeoTIFFDEM` | `tiled_geotiff_dem.py` | Directory of GeoTIFF tiles (FABDEM, SRTM, Copernicus). LRU handle cache (`max_open_tiles`), cross-tile interpolation, tile-name auto-parse (`N43E004`, `Copernicus_DSM_...`, etc.). |
| `TiledGeoDTED` | `tiled_geotiff_dted.py` | DTED archive. bbox-aware, pickle-light path-only index, LRU handles — preferred by `open_elevation()` over `DTEDElevation`. |

Both expose a `tile_count` attribute and the same `(N,)` interpolation-kernel
margin logic (bilinear 1 px, bicubic 2 px, quintic 3 px) for cross-tile reads.

---

## 12. Coordinate Conversion Utilities

**File:** `coordinates.py`

Stateless vectorized functions for geodetic/ECEF/ENU conversions. All use the stacked `(N, 3)` convention.

```python
# All accept (3,) scalar or (N, 3) batch.
# The ENU reference is a single (3,) [lat, lon, alt] array — NOT three
# separate scalars.
ecef = geodetic_to_ecef(np.array([lat_deg, lon_deg, h_meters]))
geo  = ecef_to_geodetic(np.array([X, Y, Z]))

ref  = np.array([ref_lat, ref_lon, ref_h])     # (3,)
enu  = geodetic_to_enu(points_Nx3, ref)         # (N, 3) [east, north, up]
geo  = enu_to_geodetic(enu_Nx3, ref)            # (N, 3) [lat, lon, alt]

m_lat, m_lon = meters_per_degree(34.05)         # (2,) ellipsoidal m/deg
```

**Exact signatures:**

| Function | Signature |
|----------|-----------|
| `geodetic_to_ecef` | `geodetic_to_ecef(points)` |
| `ecef_to_geodetic` | `ecef_to_geodetic(points, max_iter=10, tol=1e-12)` |
| `geodetic_to_enu` | `geodetic_to_enu(points, ref)` |
| `enu_to_geodetic` | `enu_to_geodetic(points, ref)` |
| `meters_per_degree` | `meters_per_degree(lat)` |

**Rules:**
- WGS-84 constants defined at module level (`WGS84_A`, `WGS84_B`, `WGS84_F`, `WGS84_E1_SQ`, `WGS84_E2_SQ`)
- `ecef_to_geodetic` uses iterative Bowring's method (1e-12 tolerance, 10 max iterations)
- `meters_per_degree` uses the meridional (M) and prime-vertical (N) radii of curvature — not the spherical 111320 m/deg approximation
- The ENU `ref` is a single `(3,)` array; do not pass separate `ref_lat`/`ref_lon`/`ref_h` scalars
- Shape-preserving: scalar input -> scalar output, batch -> batch
- All angles in degrees at the public API; radians only inside implementations
- `wgs84_norm(ecf)` (the WGS-84 surface normal) lives in `projection.py`, not `coordinates.py`

---

## 13. SAR Backend Selection (native/sarpy/sarkit)

**File:** `sar/sicd.py`

SICD geolocation supports multiple projection backends with automatic selection.

```python
# Priority: explicit > sarpy > native > sarkit
def _select_backend(self, requested, metadata):
    if requested == 'native':
        return self._build_native_projection(metadata)
    if requested == 'sarpy' and _HAS_SARPY:
        return self._build_sarpy_projection(metadata)
    # Auto-select: try sarpy first (if available), fall back to native
    if _HAS_SARPY:
        return self._build_sarpy_projection(metadata)
    return self._build_native_projection(metadata)
```

**Native R/Rdot backend:**
- Constructs `COAProjection` from SICD metadata
- Formation-specific projectors: PFA, INCA/RMA, RgAzComp, PLANE
- Sets `_handles_dem_internally = True` when COAProjection handles DEM iteration

**Rules:**
- Backend is selected once at init; no runtime switching
- `from_reader()` tries to build `COAProjection` first; falls back to sarpy if metadata is incomplete
- `delta_arp`, `delta_varp`, `range_bias` corrections passed through to projection engine

---

## Adding a New Geolocation Subclass — Checklist

1. Create `grdl/geolocation/<modality>/<sensor>.py`
2. Import and subclass `Geolocation` from `base.py`
3. Implement `_image_to_latlon_array(rows, cols, height)` and `_latlon_to_image_array(lats, lons, height)`
4. Override `default_hae` property if the sensor has a reference height
5. Set `_handles_dem_internally = True` if your projection does its own DEM iteration
6. Add `from_reader(cls, reader, **kwargs)` classmethod
7. Add `isinstance` check in `create_geolocation()` factory (`__init__.py`)
8. Add to `__all__` in `__init__.py`
9. Write tests with synthetic coefficients (no real data required)
10. If the sensor uses ICHIPB or similar chip transforms, integrate them in the forward/inverse methods
