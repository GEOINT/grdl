# Orthorectification Module

Geometric reprojection from native acquisition geometry to ground-referenced geographic grids. Supports WGS-84 degree grids, local East-North-Up (ENU) meter grids, DEM terrain correction, tiled processing for large images, and multi-backend accelerated resampling.

## Quick Start

```python
from grdl.image_processing.ortho import OrthoBuilder

result = (
    OrthoBuilder()
    .with_source_array(image)
    .with_geolocation(geo)
    .with_enu_grid(pixel_size_m=1.0, ref_lat=36.0, ref_lon=-75.5)
    .run()
)

ortho = result.data          # ndarray, shape (rows, cols)
grid  = result.output_grid   # ENUGrid with coordinate transforms
result.save_geotiff('ortho.tif')
```

## Two Ways to Orthorectify

### OrthoBuilder (recommended)

Builder-pattern API that auto-resolves resolution, elevation, and grid bounds. Supports tiled processing for memory-constrained scenarios.

```python
# From a reader (handles large files, reads only needed pixels)
result = (
    OrthoBuilder()
    .with_reader(reader)
    .with_geolocation(geo)
    .with_elevation(dem)
    .with_interpolation('bicubic')
    .with_nodata(np.nan)
    .run()
)

# With explicit WGS-84 grid
result = (
    OrthoBuilder()
    .with_source_array(image)
    .with_geolocation(geo)
    .with_output_grid(grid)
    .run()
)

# Tiled processing (constant peak memory)
result = (
    OrthoBuilder()
    .with_reader(reader)
    .with_geolocation(geo)
    .with_tile_size(1024)
    .run()
)
```

### Orthorectifier (direct control)

Low-level class for custom workflows. Compute the mapping once, then resample multiple images or bands.

```python
from grdl.image_processing.ortho import Orthorectifier, OutputGrid

grid = OutputGrid.from_geolocation(geo, pixel_size_lat=0.001, pixel_size_lon=0.001)
ortho = Orthorectifier(geo, grid, interpolation='bilinear', elevation=dem)
ortho.compute_mapping()

band1 = ortho.apply(image_band1, nodata=np.nan)
band2 = ortho.apply(image_band2, nodata=np.nan)
```

## Output Grids

### OutputGrid (WGS-84 degrees)

Geographic grid with lat/lon bounds and degree-per-pixel spacing.

```python
from grdl.image_processing.ortho import OutputGrid

# From geolocation footprint
grid = OutputGrid.from_geolocation(geo, pixel_size_lat=0.001, pixel_size_lon=0.001)

# Explicit bounds
grid = OutputGrid(
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

Local East-North-Up grid centered on a WGS-84 reference point. Bounds and pixel sizes in meters. Drop-in replacement for OutputGrid.

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

# Same interface as OutputGrid
lat, lon = grid.image_to_latlon(row, col)
row, col = grid.latlon_to_image(lat, lon)
```

**Grid convention (both types):** Row 0 = north edge (top), increases southward. Column 0 = west edge (left), increases eastward.

### OutputGridProtocol

Both `OutputGrid` and `ENUGrid` satisfy `OutputGridProtocol`, a `@runtime_checkable` `Protocol` that formalises the grid contract. Custom grids are accepted by `Orthorectifier` as long as they implement:

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

```python
from grdl.geolocation.elevation import open_elevation

dem = open_elevation('/data/srtm/', geoid_path='/data/egm96.tif',
                     location=(36.0, -75.5))

result = (
    OrthoBuilder()
    .with_source_array(image)
    .with_geolocation(geo)
    .with_elevation(dem)
    .with_enu_grid(pixel_size_m=1.0)
    .run()
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

When no explicit grid or resolution is provided, `OrthoBuilder` auto-computes pixel spacing from metadata:

| Format | Source of Ground Spacing |
|--------|------------------------|
| SICD | `grid.row.ss` / `grid.col.ss`, projected to ground via graze angle |
| Sentinel-1 SLC | Range/azimuth pixel spacing, projected via incidence angle |
| NISAR | `scene_center_ground_range_spacing` or grid coordinate spacing |
| GeoTIFF | Affine transform pixel sizes (degrees or meters depending on CRS) |
| BIOMASS | `range_pixel_spacing` / `azimuth_pixel_spacing` |

## OrthoResult

Returned by `OrthoBuilder.run()`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `np.ndarray` | Orthorectified image |
| `output_grid` | `OutputGrid` or `ENUGrid` | The grid used |
| `geolocation_metadata` | `dict` | CRS, affine transform, bounds |
| `orthorectifier` | `Orthorectifier` | Cached mapping for reuse |

Properties for ENU grids: `is_enu`, `enu_reference_point`, `pixel_size_meters`, `bounds_meters`.

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

Both `OutputGrid` and `ENUGrid` satisfy this protocol.

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