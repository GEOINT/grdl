# grdl.discovery

Fast metadata scanning, in-memory cataloging, spatial/temporal querying, beam footprint computation, and extensible data discovery for GRDL.

The discovery module is a **pure data layer** with no visualization dependencies.  It scans imagery files and product directories via GRDL's existing readers, extracts typed metadata and geospatial features, and provides filtering, statistics, GeoJSON export, spatial overlap and proximity queries, and radar beam ground footprint computation.

## Quick Start

```python
from grdl.discovery import MetadataScanner, LocalCatalog

scanner = MetadataScanner()
results = scanner.scan_directory('/data/imagery', recursive=True)

catalog = LocalCatalog()
catalog.add_batch(results)

# Filter
sar = catalog.filter(modality='SAR')

# GeoJSON (footprints, SCPs, orbit tracks, beam footprints, GCPs)
geojson = catalog.to_geojson(sar)

# Spatial queries — all vectorized via numpy
overlaps = catalog.find_overlapping(sar[0])
nearby = catalog.find_nearby(lat=32.13, lon=-81.14, radius_km=100)
pairs = catalog.find_pairs(max_distance_km=10, same_modality=True)

# Beam footprint — works on ScanResult or metadata directly
from grdl.discovery import compute_beam_footprint
beam = compute_beam_footprint(sar[0])                    # from ScanResult
beam = compute_beam_footprint(reader.metadata)            # from metadata
beam = compute_beam_footprint(meta, threshold_db=-6.0)    # custom threshold
```

## Supported Formats

| Format | Reader | Metadata Type | Geospatial Features |
|--------|--------|---------------|---------------------|
| SICD | `SICDReader` | `SICDMetadata` | SCP, image corners, ARP trajectory, geometry angles, grid vectors, slant/ground range, **beam footprint** |
| SIDD | `SIDDReader` | `SIDDMetadata` | Reference point, image corners, ARP trajectory, collection geometry, phenomenology, product plane |
| CPHD | `CPHDReader` | `CPHDMetadata` | Footprint |
| Sentinel-1 SLC | `Sentinel1SLCReader` | `Sentinel1SLCMetadata` | 210-point geolocation grid, orbit track, burst boundaries, swath parameters |
| BIOMASS L1 | `BIOMASSL1Reader` | `BIOMASSMetadata` | Corner coordinates, 60 GCPs per product |
| TerraSAR-X | `TerraSARReader` | `TerraSARMetadata` | Scene center, corners, geolocation grid, orbit track |
| NISAR | `NISARReader` | `NISARMetadata` | Geolocation grid, orbit track |
| EO NITF | `EONITFReader` | `EONITFMetadata` | Footprint (via RPC/RSM geolocation) |
| Sentinel-2 | `Sentinel2Reader` | `Sentinel2Metadata` | Footprint (via affine geolocation) |
| GeoTIFF | `GeoTIFFReader` | `ImageMetadata` | Footprint (via affine geolocation) |

Product directories (`.SAFE`, BIOMASS, TerraSAR) are automatically detected and passed to the appropriate reader as a unit.

## Architecture

```
grdl/discovery/
    __init__.py          # Re-exports: MetadataScanner, ScanResult, LocalCatalog,
    |                    #   DiscoveryPlugin, PluginRegistry, DataSynthesizer,
    |                    #   GRDLCatalogPlugin, compute_beam_footprint
    base.py              # DiscoveryPlugin ABC + PluginRegistry
    scanner.py           # MetadataScanner, ScanResult, compute_beam_footprint
    catalog.py           # LocalCatalog (filtering, spatial/temporal queries, GeoJSON, SQLite)
    synthesizer.py       # DataSynthesizer (test data generation)
    plugins/
        __init__.py
        grdl_catalog.py  # GRDLCatalogPlugin bridge
```

No module in this package imports matplotlib, Leaflet, or any visualization library.  Downstream tools (like the Metadata Explorer web UI) consume the data layer for rendering.

## Classes

### MetadataScanner

Parallel metadata-only scanner.  Uses GRDL's `open_any()` and `open_sar()` to detect formats, extracts metadata and geospatial features, then closes the reader immediately.

```python
scanner = MetadataScanner(compute_footprints=True)

# Single file
result = scanner.scan_file('/data/scene.nitf')

# Directory (parallel, 4 threads)
results = scanner.scan_directory(
    '/data/imagery',
    recursive=True,
    extensions={'.nitf', '.tif', '.h5'},  # optional filter
    max_workers=4,
    progress_callback=lambda done, total, r: print(f'{done}/{total}'),
)
```

The scanner automatically detects product directories:
- `.SAFE` directories (Sentinel-1) -- routes to `Sentinel1SLCReader`
- Directories with `measurement/` + `annotation/` (BIOMASS) -- routes to `BIOMASSL1Reader`
- Directories with `imageData/` + `ANNOTATION/` (TerraSAR) -- routes to `TerraSARReader`

Files inside product directories are skipped (the reader handles the internal structure).

Default recognized file extensions: `.nitf`, `.ntf`, `.nsf`, `.tif`, `.tiff`, `.geotiff`, `.h5`, `.he5`, `.hdf5`, `.hdf`, `.jp2`, `.j2k`, `.cphd`.

### ScanResult

Dataclass containing everything extracted from a single file:

```python
result.filepath        # Path — absolute path
result.format          # str — 'SICD', 'GeoTIFF', 'Sentinel-1_IW_SLC', ...
result.rows            # int
result.cols            # int
result.dtype           # str — 'complex64', 'float32', ...
result.bands           # int or None
result.crs             # str or None
result.modality        # str or None — 'SAR', 'EO', 'MSI', 'IR', 'HSI'
result.sensor          # str or None — 'Umbra-05', 'S1A', 'BIOMASS', ...
result.datetime        # datetime or None — acquisition time
result.footprint       # dict or None — GeoJSON polygon (image extent)
result.bounds          # (west, south, east, north) or None
result.metadata_ref    # ImageMetadata — typed object with callable polynomials
result.metadata_dict   # dict — JSON-safe serialization for display
result.geospatial      # dict — extracted geospatial features (see below)
result.scan_time_ms    # float
result.error           # str or None

result.to_json()       # -> dict (all fields except metadata_ref)
```

**`metadata_ref`** preserves the actual typed metadata object (`SICDMetadata`, `Sentinel1SLCMetadata`, etc.) with callable `Poly1D`, `Poly2D`, and `XYZPoly` polynomials for downstream evaluation and plotting.

**`metadata_dict`** is a recursively serialized JSON-safe dict where numpy arrays become lists, polynomials become `{'_type': 'Poly2D', 'shape': [3, 3], 'coefs': [...]}` stubs, and `XYZ`/`LatLon` become simple dicts.

### ScanResult.geospatial

Per-format geospatial feature extraction.  All coordinates are WGS-84 degrees.

**SICD:**
```python
{
    'scp':             {'lat': -33.389, 'lon': -70.793, 'hae': 507.4},
    'scp_ecef':        {'x': 1753884.8, 'y': -5034490.2, 'z': -3490387.0},
    'image_corners':   [{'lat': ..., 'lon': ...}, ...],       # 4 corners (if populated)
    'arp_at_scp':      {'lat': -34.941, 'lon': -71.488, 'alt': 531571.6},
    'orbit_track':     [{'lat': ..., 'lon': ..., 'alt': ...}, ...],  # 30 points from ARP poly
    'geometry_angles': {
        'graze_ang': 69.36, 'incidence_ang': 20.64, 'azim_ang': 200.22,
        'twist_ang': -55.79, 'slope_ang': 78.57, 'layover_ang': 257.76,
        'doppler_cone_ang': 73.98,
    },
    'slant_range_m':   564422.5,
    'ground_range_m':  185109.2,
    'side_of_track':   'R',
    'grid': {
        'image_plane': 'SLANT', 'type': 'RGAZIM',
        'row_ss': 0.265, 'col_ss': 0.508,
        'row_imp_resp_wid': 0.302, 'col_imp_resp_wid': 0.572,
        'row_uvect': {'x': ..., 'y': ..., 'z': ...},
        'col_uvect': {'x': ..., 'y': ..., 'z': ...},
    },
    'beam_footprint_3db':  {'type': 'Polygon', 'coordinates': [[[lon, lat], ...]]},
    'beam_footprint_10db': {'type': 'Polygon', 'coordinates': [[[lon, lat], ...]]},
}
```

**SIDD:**
```python
{
    'image_corners':   [{'lat': ..., 'lon': ...}, ...],
    'reference_point': {'lat': -33.389, 'lon': -70.793, 'alt': 507.4},
    'sample_spacing':  {'row': 0.625, 'col': 0.625},
    'product_plane':   {'row_uvect': {...}, 'col_uvect': {...}},
    'orbit_track':     [{'lat': ..., 'lon': ..., 'alt': ...}, ...],
    'geometry_angles': {'azimuth': ..., 'graze': ..., ...},
    'phenomenology':   {'shadow_angle': ..., 'layover_angle': ..., ...},
}
```

**Sentinel-1 SLC:**
```python
{
    'geolocation_grid': [                          # 210 tie points
        {'lat': 45.65, 'lon': 38.84, 'height': 13.0,
         'line': 0, 'pixel': 0, 'incidence_angle': 30.26,
         'elevation_angle': 26.99},
        ...
    ],
    'orbit_track': [                               # 16-17 state vectors
        {'lat': 50.42, 'lon': 45.50, 'alt': 703173.5,
         'time': '2025-11-07T03:39:05'},
        ...
    ],
    'bursts': [                                    # 9 bursts (IW mode)
        {'index': 0, 'azimuth_time': '2025-11-07T03:40:38',
         'first_line': 0, 'last_line': 1502},
        ...
    ],
    'geometry_angles': {'incidence_angle_mid': 33.63},
    'swath': {
        'name': 'IW1', 'polarization': 'VV',
        'range_pixel_spacing': 2.33, 'azimuth_pixel_spacing': 13.94,
        'radar_frequency': 5.405e9,
    },
}
```

**BIOMASS:**
```python
{
    'image_corners': [{'name': 'corner1', 'lon': 26.06, 'lat': 54.08}, ...],
    'gcps': [{'lat': 54.40, 'lon': 24.83, 'height': -33.19, 'row': 0.0, 'col': 0.0}, ...],
}
```

**TerraSAR-X:**
```python
{
    'scene_center':     {'lat': ..., 'lon': ...},
    'image_corners':    [{'lat': ..., 'lon': ..., 'hae': ...}, ...],
    'geolocation_grid': [{'lat': ..., 'lon': ..., 'height': ..., 'incidence_angle': ...}, ...],
    'orbit_track':      [{'lat': ..., 'lon': ..., 'alt': ..., 'time': ...}, ...],
}
```

**NISAR:**
```python
{
    'geolocation_grid': [{'lon': ..., 'lat': ...}, ...],
    'orbit_track':      [{'lat': ..., 'lon': ..., 'alt': ...}, ...],
}
```

### compute_beam_footprint

Computes the radar beam ground illumination contour from SICD antenna metadata.  Evaluates the antenna gain polynomial in directional cosine space, finds the contour at the specified dB threshold, projects through the collection geometry onto the WGS-84 ellipsoid, and returns a GeoJSON polygon.

No files are read or written.  Accepts either a `ScanResult` or a metadata object directly.

```python
from grdl.discovery import compute_beam_footprint

# From a ScanResult (scanner workflow)
beam = compute_beam_footprint(result)

# From a metadata object directly (no scanner, no file I/O)
from grdl.IO.sar import SICDReader
reader = SICDReader('scene.nitf')
beam = compute_beam_footprint(reader.metadata)
reader.close()

# Custom threshold (-6dB, -10dB, etc.)
beam_6db = compute_beam_footprint(result.metadata_ref, threshold_db=-6.0)
beam_10db = compute_beam_footprint(result.metadata_ref, threshold_db=-10.0)
```

**Returns:** GeoJSON Polygon dict or `None` (non-SICD formats, missing antenna data).

```python
{
    'type': 'Polygon',
    'coordinates': [[[lon, lat], [lon, lat], ...]]  # closed ring, 72 points
}
```

**How it works:**
1. Extracts the two-way antenna gain `Poly2D` (or sums tx + rcv if no two-way)
2. Evaluates gain over directional cosine space `(dcx, dcy)`
3. Traces the contour at `threshold_db` below peak via radial sweep
4. Converts `(dcx, dcy)` to ECF look vectors using the antenna coordinate frame (ACF) orientation polynomials at SCP time
5. Intersects each look vector with the WGS-84 ellipsoid (ray-ellipsoid intersection)
6. Converts ECF intersection points to geodetic `(lat, lon)` via `grdl.geolocation.coordinates.ecef_to_geodetic`

**Requires:** SICD metadata with `antenna.tx` or `antenna.two_way` (gain Poly2D + ACF orientation XYZPoly) and `scpcoa` (ARP position, SCP ECF).

The `-3dB` and `-10dB` beam footprints are also auto-extracted during scanning and stored in `result.geospatial['beam_footprint_3db']` and `result.geospatial['beam_footprint_10db']`.

### LocalCatalog

In-memory index with filtering, spatial/temporal queries, statistics, GeoJSON export, and optional SQLite persistence.

```python
catalog = LocalCatalog(db_path='~/.cache/grdl/catalog.db')  # persistent
catalog = LocalCatalog()                                      # in-memory only

catalog.add(result)
catalog.add_batch(results)

# Filter (all parameters optional, combined with AND)
items = catalog.filter(
    modality='SAR',
    sensor='Umbra',           # case-insensitive substring
    date_start=datetime(2024, 1, 1),
    date_end=datetime(2025, 12, 31),
    bbox=(-80, 30, -70, 40),  # (west, south, east, north)
    format='SICD',
    text='ocean',             # searches filepath and sensor
)

# Statistics
stats = catalog.get_statistics()
# {
#   'total': 67,
#   'by_modality': {'SAR': 53, 'Unknown': 14},
#   'by_sensor': {'Umbra-05': 8, 'BIOMASS': 14, ...},
#   'by_format': {'SICD': 19, 'SIDD': 13, ...},
#   'date_range': {'earliest': '2021-01-31T...', 'latest': '2025-08-27T...'},
#   'errors': 0,
# }

# GeoJSON FeatureCollection with all geospatial layers
geojson = catalog.to_geojson()
catalog.to_geojson(items)  # filtered subset
```

#### GeoJSON Feature Types

`to_geojson()` emits multiple feature types per scanned file:

| `feature_type` | Geometry | Source | Description |
|----------------|----------|--------|-------------|
| `footprint` | Polygon | All formats | Image coverage polygon |
| `scp` | Point | SICD | Scene Center Point |
| `reference_point` | Point | SIDD | Plane projection reference |
| `scene_center` | Point | TerraSAR | Scene center |
| `orbit_track` | LineString | SICD, SIDD, S1, TerraSAR, NISAR | Satellite ground track |
| `gcps` | MultiPoint | BIOMASS | Ground control points |
| `geolocation_grid` | MultiPoint | S1, TerraSAR, NISAR | Image-to-ground tie points |
| `beam_footprint_3db` | Polygon | SICD | -3dB beam ground illumination |
| `beam_footprint_10db` | Polygon | SICD | -10dB beam ground illumination |

Each feature carries `filepath` and `filename` in its properties for cross-referencing.

#### Spatial and Temporal Queries

All query methods are **vectorized** using `grdl.geolocation.utils.geographic_distance_batch` and numpy boolean indexing.  All 8 queries across a 67-item catalog complete in under 2ms total.

All methods accept a target as a filepath string, `Path`, or `ScanResult`.  Results always exclude the target itself.

```python
# What overlaps this file?  Returns (ScanResult, overlap_fraction) pairs.
overlaps = catalog.find_overlapping('scene.nitf', min_overlap_fraction=0.5)
# [(<ScanResult SIDD>, 1.0), (<ScanResult SICD>, 0.87), ...]

# What's near this lat/lon?  Returns (ScanResult, distance_km) pairs.
nearby = catalog.find_nearby(lat=32.13, lon=-81.14, radius_km=200)
# [(<ScanResult>, 0.0), (<ScanResult>, 12.3), ...]

# Which images contain this point?
containing = catalog.find_containing(lat=32.13, lon=-81.14)
# [<ScanResult>, <ScanResult>, ...]

# What covers the same region as this file?
same_region = catalog.find_same_region('scene.nitf', radius_km=50)
# [(<ScanResult>, 0.02), (<ScanResult>, 15.4), ...]

# What was collected around the same time?
temporal = catalog.find_temporal_neighbors('scene.nitf', window_days=90)
# [(<ScanResult>, 8.7), (<ScanResult>, 25.1), ...]  days offset

# Same region AND same time window (multi-temporal / multi-sensor fusion)
colocated = catalog.find_colocated('scene.nitf', radius_km=50, window_days=90)
# [{'result': <ScanResult>, 'distance_km': 0.02, 'time_offset_days': 3.5,
#   'same_sensor': False, 'same_modality': True}, ...]

# All pairs suitable for change detection / interferometry
pairs = catalog.find_pairs(
    max_distance_km=10,
    max_time_delta_days=365,
    same_modality=True,     # SAR-SAR only
    cross_sensor=False,     # any sensor combination
)
# [{'a': <ScanResult>, 'b': <ScanResult>,
#   'distance_km': 0.0, 'time_delta_days': 104.2,
#   'sensor_a': 'Umbra-05', 'sensor_b': 'Umbra-09'}, ...]

# Group everything into spatial clusters
clusters = catalog.group_by_region(cluster_radius_km=25)
# [[<6 items over Savannah>, <5 items over Santiago>, ...]]
for cluster in clusters:
    sensors = {r.sensor for r in cluster}
    print(f'{len(cluster)} items, sensors: {sensors}')
```

| Method | Returns | Use Case |
|--------|---------|----------|
| `find_overlapping` | `[(ScanResult, fraction)]` | Coverage analysis, mosaic planning |
| `find_nearby` | `[(ScanResult, km)]` | Point-based search, AOI queries |
| `find_containing` | `[ScanResult]` | "What images cover this coordinate?" |
| `find_same_region` | `[(ScanResult, km)]` | Multi-pass stacking, regional analysis |
| `find_temporal_neighbors` | `[(ScanResult, days)]` | Time-series, change detection candidates |
| `find_colocated` | `[{result, km, days, ...}]` | Multi-sensor fusion, cross-calibration |
| `find_pairs` | `[{a, b, km, days, ...}]` | InSAR baseline selection, bi-temporal change |
| `group_by_region` | `[[ScanResult]]` | Collection planning, site inventories |

### DiscoveryPlugin and PluginRegistry

Extensible plugin system for connecting external data sources.

```python
from grdl.discovery import DiscoveryPlugin, PluginRegistry

class MySTACPlugin(DiscoveryPlugin):
    @property
    def name(self): return 'My STAC'

    @property
    def description(self): return 'Query my STAC catalog'

    def discover(self, bbox=None, **kwargs):
        # query STAC, download, return local paths
        return [Path('/data/downloaded/scene.tif')]

    def get_config_schema(self):
        return {'bbox': {'type': 'array', 'description': 'Bounding box'}}

registry = PluginRegistry()
registry.register(MySTACPlugin())
paths = registry.get('My STAC').discover(bbox=[-80, 30, -70, 40])
```

#### GRDLCatalogPlugin

Built-in plugin that bridges GRDL's existing catalog classes:

```python
from grdl.discovery import GRDLCatalogPlugin
from grdl.IO import Sentinel1SLCCatalog

plugin = GRDLCatalogPlugin(
    catalog_cls=Sentinel1SLCCatalog,
    search_path='/data/sentinel1',
    sensor_name='Sentinel-1 SLC',
)
paths = plugin.discover()  # calls Sentinel1SLCCatalog.discover_local()
```

### DataSynthesizer

Generate geocoded GeoTIFF test imagery with known metadata.

```python
from grdl.discovery import DataSynthesizer

synth = DataSynthesizer()

# Complex SAR with point targets
synth.synthesize_sar(
    'test_sar.tif',
    rows=1024, cols=1024,
    center_lat=38.0, center_lon=-77.0,
    pixel_spacing=1.0, noise_sigma=0.1,
)

# RGB EO with spatial pattern
synth.synthesize_eo(
    'test_eo.tif',
    rows=512, cols=512, bands=3,
    center_lat=38.0, center_lon=-77.0,
    gsd=0.5,
    pattern='checkerboard',  # or 'gradient', 'noise', 'resolution_target'
)

# 13-band multispectral
synth.synthesize_multispectral(
    'test_msi.tif',
    rows=512, cols=512, bands=13,
    center_lat=38.0, center_lon=-77.0, gsd=10.0,
)
```

All synthesized files are geocoded GeoTIFF (EPSG:4326) that `open_any()` reads with full affine geolocation.

## Performance

Tested on `~/sar_data` (67 imagery files across SICD, SIDD, CPHD, Sentinel-1 SLC, BIOMASS, and GeoTIFF):

**Scanning:**

| Metric | Value |
|--------|-------|
| Total scan time (67 files) | ~1 second |
| Per-file average | ~15 ms |
| SICD (complex SAR NITF) | ~50 ms |
| SIDD (derived SAR NITF) | ~25 ms |
| Sentinel-1 SAFE directory | ~120 ms |
| BIOMASS product directory | ~80 ms |
| GeoTIFF | ~5 ms |

Scanning is parallelized via `ThreadPoolExecutor` (default 4 workers).

**Queries (67-item catalog):**

| Method | Time |
|--------|------|
| `find_overlapping` | 0.1 ms |
| `find_nearby` | 1.1 ms |
| `find_containing` | < 0.1 ms |
| `find_same_region` | < 0.1 ms |
| `find_temporal_neighbors` | < 0.1 ms |
| `find_colocated` | 0.1 ms |
| `find_pairs` | 0.2 ms |
| `group_by_region` | 0.2 ms |
| **All 8 queries** | **< 2 ms** |

All spatial queries use vectorized numpy operations and `grdl.geolocation.utils.geographic_distance_batch`.

## Dependencies

**Required:** `numpy`, `scipy` (GRDL core dependencies)

**Used via GRDL (no additional install):** `grdl.IO` readers, `grdl.geolocation` factories, `grdl.geolocation.coordinates` for ECF-to-geodetic conversion, `grdl.geolocation.utils` for Haversine distance.

**Optional (synthesizer only):** `rasterio` for GeoTIFF writing.
