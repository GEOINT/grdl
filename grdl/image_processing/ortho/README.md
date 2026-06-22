# Orthorectification Module

Geometric reprojection from native acquisition geometry to ground-referenced geographic grids. Supports WGS-84 degree grids, local East-North-Up (ENU) meter grids, DEM terrain correction, tiled processing for large images, and multi-backend accelerated resampling.

## Quick Start

```python
from grdl.image_processing.ortho import orthorectify

result = orthorectify(
    geolocation=geo,
    source_array=image,
    enu_grid=dict(pixel_size_m=1.0, ref_lat=36.0, ref_lon=-75.5),
)

ortho = result.data          # ndarray, shape (rows, cols)
grid  = result.output_grid   # ENUGrid with coordinate transforms
result.save_geotiff('ortho.tif')
```

## Two Ways to Orthorectify

### orthorectify() (recommended)

Keyword-argument function that auto-resolves resolution, elevation, and grid bounds. Provide either ``reader`` or ``source_array``. Full signature:

```python
orthorectify(
    geolocation,                  # required; set geolocation.elevation for terrain
    *,
    reader=None,                  # OR source_array=None (mutually exclusive)
    source_array=None,
    metadata=None,                # for auto-resolution (SICDMetadata, etc.)
    output_grid=None,             # explicit grid overrides auto-computation
    resolution=None,              # (pixel_size_lat, pixel_size_lon) in degrees
    interpolation='bilinear',     # 'nearest'|'bilinear'|'bicubic'|'lanczos'
    bands=None,                   # band indices (reader mode)
    nodata=0.0,
    margin=0.0,                   # footprint margin in degrees
    scale_factor=1.0,             # multiplier on auto-computed resolution
    roi=None,                     # (min_lat, max_lat, min_lon, max_lon)
    tile_size=None,               # int or (int, int) → tiled processing
    enu_grid=None,                # dict → ENU output (see ENUGrid below)
    batch_size=2_000_000,
) -> OrthoResult
```

```python
from grdl.image_processing.ortho import orthorectify

# From a reader (handles large files, reads only needed pixels)
geo.elevation = dem
result = orthorectify(
    geolocation=geo,
    reader=reader,
    interpolation='bicubic',
    nodata=np.nan,
)

# With explicit WGS-84 grid
result = orthorectify(
    geolocation=geo,
    source_array=image,
    output_grid=grid,
)

# Tiled processing (constant peak memory)
result = orthorectify(
    geolocation=geo,
    reader=reader,
    tile_size=1024,
)

# Geographic sub-region with DEM
geo.elevation = dem
result = orthorectify(
    geolocation=geo,
    reader=reader,
    metadata=reader.metadata,
    roi=(36.0, 36.1, -75.8, -75.7),
    tile_size=2048,
)
```

> **The DEM lives on the geolocation, never on the orthorectifier.**
> Attach terrain to `geo.elevation` and the geolocation uses it
> internally during its R/Rdot inverse. `orthorectify()` and
> `Orthorectifier` have **no** `elevation` parameter. If `geo.elevation`
> is unset, projection falls back to the WGS-84 ellipsoid (height 0) and
> output is terrain-uncorrected. See the GRDL `CLAUDE.md` "DEM /
> Elevation Ownership" section.

### Orthorectifier (direct control)

Low-level class for custom workflows. Compute the mapping once, then resample multiple images or bands.

```python
from grdl.image_processing.ortho import Orthorectifier, GeographicGrid

grid = GeographicGrid.from_geolocation(geo, pixel_size_lat=0.001, pixel_size_lon=0.001)
geo.elevation = dem
ortho = Orthorectifier(geo, grid, interpolation='bilinear')
ortho.compute_mapping()

band1 = ortho.apply(image_band1, nodata=np.nan)
band2 = ortho.apply(image_band2, nodata=np.nan)
```

**`Orthorectifier` API** (`output_grid` accepts any `OutputGridProtocol`):

| Method | Purpose |
|--------|---------|
| `Orthorectifier(geolocation, output_grid, interpolation='bilinear')` | Construct; `interpolation` is `'nearest'`/`'bilinear'`/`'bicubic'`/`'lanczos'`. |
| `compute_mapping()` | Inverse-map every output pixel → source pixel; caches `source_rows`, `source_cols`, `valid_mask`. DEM sampled here (once). |
| `apply(source, nodata=0.0, backend='auto', source_origin=None)` | Resample a pre-loaded array through the cached mapping. |
| `apply_from_reader(reader, ...)` | Read only the source chip the mapping needs, then resample (large files). |
| `get_output_geolocation_metadata()` | CRS, affine transform, bounds, pixel sizes for the output grid. |

The mapping is computed once and reused across `apply()` calls, so
multiple bands or images on the same grid share the work.

### orthorectify_point_roi() (point-centered chip)

Single-call helper for a fixed-size ground area centered on a geographic
or pixel point. Handles complex SAR magnitude conversion and DEM lookup.

```python
from grdl.image_processing.ortho import orthorectify_point_roi

result = orthorectify_point_roi(
    reader=reader,
    lat=34.05, lon=-118.25,       # OR row=..., col=...
    width_m=500, height_m=500,
    pixel_size_m=0.25,            # None → auto from metadata
    interpolation='lanczos',
    band=0,
    complex_mode='magnitude',     # complex SAR → magnitude before resample
    elevation=dem,                # optional ElevationModel
    nodata=float('nan'),
)
# Returns PointRoiResult (data + ENU grid + geolocation metadata)
```

## Output Grids

### GeographicGrid (WGS-84 degrees)

Geographic grid with lat/lon bounds and degree-per-pixel spacing.

```python
from grdl.image_processing.ortho import GeographicGrid

# From geolocation footprint
grid = GeographicGrid.from_geolocation(geo, pixel_size_lat=0.001, pixel_size_lon=0.001)

# Explicit bounds
grid = GeographicGrid(
    min_lat=36.0, max_lat=37.0,
    min_lon=-75.5, max_lon=-74.5,
    pixel_size_lat=0.001, pixel_size_lon=0.001,
)

# Coordinate transforms
lat, lon = grid.image_to_latlon(row, col)
row, col = grid.latlon_to_image(lat, lon)

# Tile extraction
tile = grid.sub_grid(row_start=0, col_start=0, row_end=512, col_end=512)
```

### ENUGrid (local meters)

Local East-North-Up grid centered on a WGS-84 reference point. Bounds and pixel sizes in meters. Drop-in replacement for GeographicGrid.

```python
from grdl.image_processing.ortho import ENUGrid

# From geolocation footprint
grid = ENUGrid.from_geolocation(geo, pixel_size_m=1.0)

# Explicit bounds
grid = ENUGrid(
    ref_lat=36.0, ref_lon=-75.5, ref_alt=0.0,
    min_east=-500, max_east=500,
    min_north=-500, max_north=500,
    pixel_size_east=1.0, pixel_size_north=1.0,
)

# Same interface as GeographicGrid
lat, lon = grid.image_to_latlon(row, col)
row, col = grid.latlon_to_image(lat, lon)
```

### UTMGrid (UTM projection)

UTM projection grid with automatic zone detection. Bounds and pixel sizes in meters within a UTM zone.

```python
from grdl.image_processing.ortho import UTMGrid

# From geolocation footprint (auto-detects UTM zone)
grid = UTMGrid.from_geolocation(geo, pixel_size_m=1.0)

# Explicit bounds in UTM coordinates
grid = UTMGrid(
    zone=18, north=True,
    min_easting=300000, max_easting=310000,
    min_northing=3990000, max_northing=4000000,
    pixel_size=1.0,
)

# Same interface as GeographicGrid
lat, lon = grid.image_to_latlon(row, col)
row, col = grid.latlon_to_image(lat, lon)
```

### WebMercatorGrid (EPSG:3857)

Web Mercator grid compatible with web mapping tile systems. Bounds and pixel sizes in Web Mercator meters.

```python
from grdl.image_processing.ortho import WebMercatorGrid

# From geolocation footprint
grid = WebMercatorGrid.from_geolocation(geo, pixel_size_m=1.0)

# Same interface as GeographicGrid
lat, lon = grid.image_to_latlon(row, col)
row, col = grid.latlon_to_image(lat, lon)
```

**Grid convention (all grid types):** Row 0 = north edge (top), increases southward. Column 0 = west edge (left), increases eastward.

### OutputGridProtocol

`GeographicGrid`, `ENUGrid`, `UTMGrid`, and `WebMercatorGrid` all satisfy `OutputGridProtocol`, a `@runtime_checkable` `Protocol` that formalises the grid contract. Custom grids are accepted by `Orthorectifier` as long as they implement:

```python
from grdl.image_processing.ortho import OutputGridProtocol

class MyGrid:
    rows: int
    cols: int
    def image_to_latlon(self, row, col): ...
    def latlon_to_image(self, lat, lon): ...
    def sub_grid(self, row_start, col_start, row_end, col_end): ...

assert isinstance(MyGrid(...), OutputGridProtocol)  # runtime check
```

## Elevation / DEM Integration

Pass an `ElevationModel` to inject terrain heights into the geolocation inverse. Without a DEM, all points project to a constant height surface.

Both `DTEDElevation` and `GeoTIFFDEM` default to **bicubic** (order=3) interpolation, producing C1-continuous height fields. This eliminates the derivative kinks at DEM cell boundaries that cause visible line distortion in sub-meter orthorectified imagery. `DTEDElevation` also performs cross-tile boundary stitching so interpolation kernels are seamless across 1-degree tile edges. Use `interpolation=1` for faster bilinear, or `interpolation=5` for quintic.

```python
from grdl.geolocation.elevation import open_elevation

dem = open_elevation('/data/srtm/', geoid_path='/data/egm96.tif',
                     location=(36.0, -75.5))

geo.elevation = dem
result = orthorectify(
    geolocation=geo,
    source_array=image,
    enu_grid=dict(pixel_size_m=1.0),
)
```

During mapping, each output grid pixel is:
1. Converted to lat/lon via the grid
2. DEM height queried at that lat/lon
3. `geolocation.latlon_to_image(lat, lon, height=dem_h)` gives the source pixel

## Resampling Backends

The `accelerated` module dispatches to the fastest available backend:

| Backend | Priority | Orders | Notes |
|---------|----------|--------|-------|
| `numba` | 1 | 0, 1, 3, 5 | JIT-compiled, row-parallel via `prange`. Fastest CPU option. |
| `torch_gpu` | 2 | 0, 1 | CUDA or MPS GPU. 10-50x over scipy. |
| `torch` | 3 | 0, 1 | CPU PyTorch. ~2x over scipy. |
| `scipy_parallel` | 4 | any | `ThreadPoolExecutor` + `map_coordinates`. 2-6x over scipy. |
| `scipy` | 5 | any | Single-threaded fallback. Always available. |

Interpolation orders: 0 = nearest, 1 = bilinear, 3 = cubic (Catmull-Rom), 5 = Lanczos-3.

Force a specific backend:
```python
ortho.apply(image, backend='numba')
```

## Auto-Resolution

When no explicit grid or resolution is provided, `orthorectify()` auto-computes pixel spacing from metadata:

| Format | Source of Ground Spacing |
|--------|------------------------|
| SICD | `grid.row.ss` / `grid.col.ss`, projected to ground via graze angle |
| Sentinel-1 SLC | Range/azimuth pixel spacing, projected via incidence angle |
| NISAR | `scene_center_ground_range_spacing` or grid coordinate spacing |
| GeoTIFF | Affine transform pixel sizes (degrees or meters depending on CRS) |
| BIOMASS | `range_pixel_spacing` / `azimuth_pixel_spacing` |

## OrthoResult

Returned by `orthorectify()`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `np.ndarray` | Orthorectified image |
| `output_grid` | `GeographicGrid`, `ENUGrid`, `UTMGrid`, or `WebMercatorGrid` | The grid used |
| `geolocation_metadata` | `dict` | CRS, affine transform, bounds |
| `orthorectifier` | `Orthorectifier` | Cached mapping for reuse |

Always available: `shape` (`result.data.shape`). ENU-only properties
(return `None` for non-ENU grids): `is_enu`, `enu_reference_point`
(`(lat, lon, alt)`), `pixel_size_meters` (`(east_m, north_m)`),
`bounds_meters` (`(min_east, min_north, max_east, max_north)`).

```python
result.save_geotiff('output.tif')  # georeferenced GeoTIFF
```

## Contracts

### OutputGridProtocol (required for Orthorectifier)

Must implement (see `OutputGridProtocol`):
- `rows: int`, `cols: int` — grid dimensions
- `image_to_latlon(row, col) -> (lat, lon)` — pixel to geographic
- `latlon_to_image(lat, lon) -> (row, col)` — geographic to pixel
- `sub_grid(row_start, col_start, row_end, col_end) -> grid` — tile extraction

All grid types (`GeographicGrid`, `ENUGrid`, `UTMGrid`, `WebMercatorGrid`) satisfy this protocol.

### Geolocation (required)

Must implement:
- `latlon_to_image(lat, lon, height=0.0) -> (row, col)` — inverse projection
- `shape -> (rows, cols)` — source image dimensions
- `get_bounds() -> (min_lon, min_lat, max_lon, max_lat)` — for auto grid

### ElevationModel (optional)

Must implement:
- `get_elevation(lat, lon) -> height` — scalar or array, returns NaN for missing

### ImageReader (for `apply_from_reader` / builder with reader)

Must implement:
- `read_chip(row_start, row_end, col_start, col_end) -> ndarray`

## Examples

See `grdl/example/ortho/` for working scripts:

- `chip_ortho.py` — Ground-extent chip extraction + ENU ortho (CLI)
- `compare_sidd_ortho.py` — Dual-SIDD ortho comparison with PCA, NCC, coregistration
- `ortho_biomass.py` — BIOMASS ortho with Pauli RGB
- `ortho_combined.py` — Auto-detect SICD/SIDD, WGS-84 + ENU output
- `ortho_sicd.py` — SICD ortho with DEM and ENU
- `ortho_sidd.py` — SIDD ortho with DEM and ENU