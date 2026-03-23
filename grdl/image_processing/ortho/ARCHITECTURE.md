# Ortho Module — Architecture

## Module Map

```
ortho/
  __init__.py          Public API re-exports
  ortho.py             OutputGridProtocol, validate_sub_grid_indices,
                       GeographicGrid (alias: OutputGrid), Orthorectifier (core mapping + resampling)
  enu_grid.py          ENUGrid (local meters grid, satisfies OutputGridProtocol)
  utm_grid.py          UTMGrid (UTM projection grid, satisfies OutputGridProtocol)
  web_mercator_grid.py WebMercatorGrid (Web Mercator projection grid, satisfies OutputGridProtocol)
  ortho_builder.py     OrthoBuilder (builder), OrthoResult (output container)
  accelerated.py       Multi-backend resampling dispatch (numba/torch/scipy)
  resolution.py        Auto-compute output pixel spacing from metadata
```

## Class Hierarchy

```
OutputGridProtocol (Protocol, runtime_checkable)
  ├── rows, cols
  ├── image_to_latlon(row, col)
  ├── latlon_to_image(lat, lon)
  └── sub_grid(row_start, col_start, row_end, col_end)

GeographicGrid                 WGS-84 degree grid (satisfies OutputGridProtocol)
  ├── from_geolocation()       Factory from geolocation footprint
  ├── image_to_latlon()        Grid pixel → lat/lon
  ├── latlon_to_image()        lat/lon → grid pixel
  └── sub_grid()               Tile extraction (via validate_sub_grid_indices)
  (OutputGrid is a backwards-compatible alias for GeographicGrid)

UTMGrid                        UTM projection grid (satisfies OutputGridProtocol)
  ├── from_geolocation()       Factory from geolocation footprint
  ├── image_to_latlon()        Grid pixel → UTM → geodetic
  ├── latlon_to_image()        Geodetic → UTM → grid pixel
  └── sub_grid()               Tile extraction (via validate_sub_grid_indices)

WebMercatorGrid                Web Mercator grid (satisfies OutputGridProtocol)
  ├── from_geolocation()       Factory from geolocation footprint
  ├── image_to_latlon()        Grid pixel → Web Mercator → geodetic
  ├── latlon_to_image()        Geodetic → Web Mercator → grid pixel
  └── sub_grid()               Tile extraction (via validate_sub_grid_indices)

ENUGrid                        ENU meters grid (satisfies OutputGridProtocol)
  ├── from_geolocation()       Factory from geolocation footprint
  ├── image_to_latlon()        Grid pixel → ENU → geodetic
  ├── latlon_to_image()        Geodetic → ENU → grid pixel
  └── sub_grid()               Tile extraction (via validate_sub_grid_indices)

ImageTransform (ABC, grdl.image_processing.base)
  └── Orthorectifier          @processor_version('0.1.0')
        ├── compute_mapping()    Orchestrates sequential/parallel mapping
        │     ├── _compute_strip()     Single strip: grid→latlon→DEM→source
        │     ├── _compute_mapping_parallel()   ThreadPool of _compute_strip
        │     └── _finalize_mapping()  Validate bounds, cache, log coverage
        ├── apply()              Resample from array
        └── apply_from_reader()  Resample from reader (large files)

orthorectify()                Keyword-argument function (recommended entry point)
                              Wraps OrthoBuilder with all params as kwargs

OrthoBuilder                  Builder-pattern orchestrator (advanced use)
  ├── with_*()                 15 fluent builder methods
  ├── run()                    Execute → OrthoResult
  └── _run_tiled()             Tiled execution for large grids

OrthoResult                    Output container
  ├── data                     Orthorectified ndarray
  ├── output_grid              Grid used (GeographicGrid, ENUGrid, UTMGrid, or WebMercatorGrid)
  └── save_geotiff()           Write georeferenced GeoTIFF

validate_sub_grid_indices()    Shared bounds validation for sub_grid()
```

## Data Flow

### Standard Flow (`OrthoBuilder.run`)

```
OrthoBuilder.run()
  │
  ├── 1. Resolve output grid (any OutputGridProtocol)
  │     ├── Explicit grid? → use it
  │     ├── ENU params? → construct ENUGrid
  │     ├── Explicit resolution? → GeographicGrid.from_geolocation(geo, res)
  │     └── Auto-resolve → compute_output_resolution(metadata) → GeographicGrid
  │
  ├── 2. Create Orthorectifier(geo, grid, interp, elevation)
  │
  ├── 3. compute_mapping()
  │     │
  │     │  Core: _compute_strip(row_start, row_end, out_cols)
  │     │    For each output pixel in the strip:
  │     │      lat, lon = grid.image_to_latlon(i, j)
  │     │      if DEM: h = elevation.get_elevation(lat, lon)
  │     │      src_row, src_col = geo.latlon_to_image(lat, lon, height=h)
  │     │
  │     ├── < 1M pixels → single _compute_strip(0, rows)
  │     └── >= 1M pixels → _compute_mapping_parallel (ThreadPool of strips)
  │
  │     _finalize_mapping(): validate bounds, cache, log coverage
  │     Result: source_rows, source_cols, valid_mask  (cached for reuse)
  │
  ├── 4. Resample
  │     ├── Source from array (with_source_array) → apply(array)
  │     └── Source from reader (with_reader) → apply_from_reader(reader)
  │           └── Computes bounding box of needed source pixels,
  │               reads only that chip, adjusts mapping to chip-relative
  │
  └── 5. Return OrthoResult(data, grid, metadata, orthorectifier)
```

### Tiled Pipeline (`OrthoBuilder._run_tiled`)

```
_run_tiled()
  │
  ├── 1. Resolve full output grid
  ├── 2. Plan tiles via grdl.data_prep.Tiler
  ├── 3. For each tile:
  │     ├── sub_grid = grid.sub_grid(tile bounds)
  │     ├── Create Orthorectifier for sub_grid
  │     ├── compute_mapping()
  │     ├── Read source chip (only needed rows/cols)
  │     ├── Resample into output array slice
  │     └── (mapping discarded, memory freed)
  └── 4. Return OrthoResult with assembled output
```

Peak memory per tile = O(tile_rows x tile_cols) instead of O(full_grid).

## Resampling Backend Dispatch

```
resample(image, row_map, col_map, valid_mask, order, backend)
  │
  ├── detect_backend(prefer)
  │     Priority: numba → torch_gpu → torch_cpu → scipy_parallel → scipy
  │
  ├── Complex data? → split real/imag, resample each, recombine
  ├── Multi-band (B,H,W)? → stack, single dispatch call
  │
  └── Dispatch:
        ├── _resample_numba()        JIT prange kernels
        │     ├── order 1: _numba_bilinear_2d  (2x2 tap)
        │     ├── order 3: _numba_cubic_2d     (4x4 Catmull-Rom)
        │     └── order 5: _numba_lanczos3_2d  (6x6 sinc window)
        │
        ├── _resample_torch()        F.grid_sample (normalized coords)
        │     ├── order 0: mode='nearest'
        │     └── order 1: mode='bilinear'
        │
        ├── _resample_scipy_parallel()   ThreadPoolExecutor + map_coordinates
        │     └── Row-chunk partitioning, GIL released by NumPy
        │
        └── _resample_scipy()        Single-threaded map_coordinates
```

Fallback rules:
- torch requested but order > 1 → numba or scipy_parallel
- numba requested but order not in {0,1,3,5} → scipy_parallel

## Grid Interface Contract

``GeographicGrid``, ``ENUGrid``, ``UTMGrid``, and ``WebMercatorGrid`` all satisfy ``OutputGridProtocol``
(a ``@runtime_checkable`` ``Protocol``).  ``Orthorectifier`` type-hints
its ``output_grid`` parameter as ``OutputGridProtocol``, so any custom
grid that implements the protocol is accepted.

```python
@runtime_checkable
class OutputGridProtocol(Protocol):
    rows: int
    cols: int
    def image_to_latlon(self, row, col) -> (lat, lon): ...
    def latlon_to_image(self, lat, lon) -> (row, col): ...
    def sub_grid(self, row_start, col_start, row_end, col_end) -> OutputGridProtocol: ...
```

Both grids delegate ``sub_grid`` validation to the shared
``validate_sub_grid_indices()`` helper (bounds checks, empty-region
checks) to avoid duplicated logic.

**GeographicGrid** stores bounds in degrees and uses linear interpolation.
**ENUGrid** stores bounds in meters and converts via `geodetic_to_enu` / `enu_to_geodetic`.
**UTMGrid** stores bounds in UTM meters and converts via UTM projection.
**WebMercatorGrid** stores bounds in Web Mercator meters and converts via EPSG:3857 projection.

## External Dependencies

```
ortho.py
  ├── grdl.image_processing.base.ImageTransform (ABC)
  ├── grdl.image_processing.versioning (@processor_version)
  ├── grdl.image_processing.params (Annotated, Options, Desc)
  └── typing.Protocol, runtime_checkable (OutputGridProtocol)

enu_grid.py
  ├── grdl.geolocation.coordinates (geodetic_to_enu, enu_to_geodetic)
  └── grdl.image_processing.ortho.ortho.validate_sub_grid_indices

ortho_builder.py
  ├── grdl.data_prep.Tiler (tiled processing)
  ├── grdl.IO.geotiff.GeoTIFFWriter (save_geotiff)
  └── resolution.compute_output_resolution

resolution.py
  ├── grdl.IO.models.sicd.SICDMetadata
  ├── grdl.IO.models.sentinel1_slc.Sentinel1SLCMetadata (optional)
  └── grdl.IO.models.nisar.NISARMetadata (optional)

accelerated.py
  ├── torch (optional)
  ├── numba (optional)
  └── scipy.ndimage.map_coordinates (always available)
```

## Auto-Resolution Dispatch

`compute_output_resolution(metadata)` in `resolution.py`:

```
metadata type          → resolution source
─────────────────────────────────────────────────
SICDMetadata           → grid.row.ss / grid.col.ss, graze angle projection
Sentinel1SLCMetadata   → range/azimuth spacing, incidence angle projection
NISARMetadata          → scene_center_ground_range_spacing (RSLC) or grid spacing (GSLC)
dict with 'transform'  → GeoTIFF affine pixel sizes
dict with 'range_pixel_spacing' → BIOMASS range/azimuth spacing
```

All paths convert meters to degrees via `spacing_m / 111320` (lat) and
`spacing_m / (111320 * cos(center_lat))` (lon).

## Key Design Decisions

1. **Inverse mapping, not forward.** Every output pixel asks "where do I come from in the source?" via `latlon_to_image`. This guarantees no holes in the output and enables standard resampling kernels.

2. **Mapping is cached.** `compute_mapping()` stores `source_rows`, `source_cols`, `valid_mask` on the Orthorectifier. Multiple calls to `apply()` reuse the same mapping (e.g., different bands, different images on the same grid).

3. **Grid objects are value types.** GeographicGrid, ENUGrid, UTMGrid, and WebMercatorGrid carry no mutable state. They define a coordinate system and can be shared freely.

4. **DEM integrates at mapping time, not resampling time.** Terrain heights are looked up once during `compute_mapping()` and baked into the source coordinates. The resampler sees only 2D fractional pixel coordinates.

5. **Backend dispatch is transparent.** `resample()` auto-detects the best backend. User code is identical regardless of whether numba, torch, or scipy runs underneath.

6. **Grid contract is a Protocol, not an ABC.** `OutputGridProtocol` formalises the shared interface (`rows`, `cols`, `image_to_latlon`, `latlon_to_image`, `sub_grid`) without requiring inheritance. `GeographicGrid`, `ENUGrid`, `UTMGrid`, and `WebMercatorGrid` all satisfy it structurally. Custom grids are accepted by `Orthorectifier` as long as they implement the protocol.

7. **Single mapping core.** `_compute_strip()` is the only place that implements grid→latlon→DEM→source-pixel logic. Sequential mapping calls it once for the full grid; parallel mapping calls it per row-chunk in a thread pool. `_finalize_mapping()` handles validation and caching. Changes to the mapping algorithm (e.g., DEM handling) need to be made in one place only.

8. **Shared validation.** `validate_sub_grid_indices()` centralises the bounds-checking logic used by `GeographicGrid.sub_grid()`, `ENUGrid.sub_grid()`, `UTMGrid.sub_grid()`, and `WebMercatorGrid.sub_grid()`, preventing drift between implementations.

## Examples

See `grdl/example/ortho/` for working scripts:

| Script | Description |
|--------|-------------|
| `chip_ortho.py` | Ground-extent chip extraction + ENU ortho. CLI with lat/lon or pixel center. |
| `compare_sidd_ortho.py` | Dual-SIDD comparison: ortho to shared ENU grid, PCA decomposition, NCC alignment, feature matching, red/blue difference overlay. |
| `ortho_biomass.py` | BIOMASS L1A ortho with Pauli RGB composite (quad-pol). |
| `ortho_combined.py` | Auto-detects SICD/SIDD, orthos to WGS-84 and ENU grids. |
| `ortho_sicd.py` | SICD complex SAR ortho with DEM and ENU output. |
| `ortho_sidd.py` | SIDD derived product ortho with DEM and ENU output. |

All examples use `orthorectify()` as the recommended entry point. `OrthoBuilder` is available for advanced cases requiring partial configuration or builder reuse.