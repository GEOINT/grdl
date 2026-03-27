# Geolocation Module

Transforms between image pixel coordinates and geographic coordinates
(latitude, longitude, height) for any 2D correlated imagery. Works with
SAR complex data, SAR detected products, EO NITF with rational polynomial
models, geocoded rasters, and grid-based products.

See [ARCHITECTURE.md](ARCHITECTURE.md) for class hierarchy and internal
design decisions.

---

## Quick Start

Every geolocation object follows the same pattern:

```python
# 1. Load imagery
reader = SICDReader('image.nitf')

# 2. Create geolocation from metadata
geo = SICDGeolocation.from_reader(reader)

# 3. Optional: attach DEM for terrain correction
geo.elevation = open_elevation('/data/srtm/')

# 4. Transform coordinates
result = geo.image_to_latlon(500, 1000)        # → (3,) [lat, lon, h]
lat, lon, h = result                           # tuple unpacking works
px = geo.latlon_to_image(lat, lon)             # → (2,) [row, col]
```

All public methods accept **two input forms** and return stacked ndarrays:

```python
# Scalar (returns shape (3,) — tuple unpacking works)
lat, lon, h = geo.image_to_latlon(500, 1000)

# Stacked (N, 2) array of points (returns shape (N, 3))
pixels = np.array([[100, 400],
                    [200, 500],
                    [300, 600]])
geo_pts = geo.image_to_latlon(pixels)          # → (3, 3) array
# geo_pts[0] → [lat, lon, h] for first point
# geo_pts[:, 0] → all latitudes
```

---

## SAR Geolocation

### SICD (Complex SAR)

SICD uses the R/Rdot (Range-Doppler) sensor model. The native backend
solves the range/range-rate contour intersection for all SICD formation
types (PFA, INCA, RgAzComp, PLANE).

```python
from grdl.IO.sar import SICDReader
from grdl.geolocation import SICDGeolocation, open_elevation

with SICDReader('complex.nitf') as reader:
    # From reader (auto-selects best backend)
    geo = SICDGeolocation.from_reader(reader)

    # Or from metadata directly (always uses native backend)
    geo = SICDGeolocation(reader.metadata)

    # Single pixel → (3,) array [lat, lon, h]
    lat, lon, h = geo.image_to_latlon(500, 1000)

    # Batch of pixels (vectorized — no loops needed)
    pixels = np.array([[100, 500],
                        [200, 600],
                        [300, 700],
                        [400, 800]])           # (4, 2) [row, col]
    geo_pts = geo.image_to_latlon(pixels)      # → (4, 3) [lat, lon, h]

    # Round-trip
    px_back = geo.latlon_to_image(geo_pts)     # → (4, 2) [row, col]
    # px_back ≈ pixels  (sub-pixel accuracy)
```

**With DEM (terrain-corrected projection):**

```python
geo = SICDGeolocation.from_reader(reader)
geo.elevation = open_elevation('/data/dted/', geoid_path='/data/egm96.pgm')

# Heights now come from the DEM, not a constant surface
lat, lon, terrain_h = geo.image_to_latlon(500, 1000)
# terrain_h is the DEM height at (lat, lon) in meters HAE
```

The native R/Rdot engine handles DEM iteration internally — the DEM is
passed directly into the range/range-rate convergence loop for maximum
accuracy. No redundant outer iteration.

**With adjustable parameters:**

```python
geo = SICDGeolocation(
    reader.metadata,
    delta_arp=np.array([0.1, -0.2, 0.05]),   # ARP correction (meters ECF)
    delta_varp=np.array([0.01, 0.0, -0.01]),  # velocity correction (m/s)
    range_bias=2.5,                           # range bias (meters)
)
```

**Backend selection:**

```python
# Force native backend (pure numpy, no external dependencies)
geo = SICDGeolocation(reader.metadata, backend='native')

# Force sarpy backend (delegates to sarpy.geometry.point_projection)
geo = SICDGeolocation(reader.metadata, raw_meta=sarpy_sicd, backend='sarpy')
```

**Properties:**

```python
geo.default_hae      # SCP height above WGS-84 (meters)
geo.projection_type  # 'R/Rdot'
geo.has_rdot         # True when native or sarpy backend is available
geo.backend          # 'native', 'sarpy', or 'sarkit'
```

---

### SIDD (Detected SAR Products)

SIDD stores imagery in a map-projected grid (Plane, Geographic, or
Cylindrical). When R/Rdot metadata is available (TimeCOAPoly + ARPPoly),
the grid projection is refined through the R/Rdot engine for sub-meter
accuracy.

```python
from grdl.IO.sar import SIDDReader
from grdl.geolocation import SIDDGeolocation, open_elevation

with SIDDReader('detected.nitf') as reader:
    # Default: enables R/Rdot refinement when metadata supports it
    geo = SIDDGeolocation.from_reader(reader)

    # Or disable R/Rdot refinement (grid-only, faster but less accurate)
    geo = SIDDGeolocation(reader.metadata, refine=False)

    lat, lon, h = geo.image_to_latlon(1000, 2000)
```

**With DEM:**

```python
geo = SIDDGeolocation.from_reader(reader)
geo.elevation = open_elevation('/data/srtm_30m.tif')

# R/Rdot mode: DEM is fed into the R/Rdot iteration (handles internally)
# Grid-only mode: base class provides DEM heights in output
lat, lon, h = geo.image_to_latlon(1000, 2000)
```

**Inverse (geographic → pixel):**

```python
# SIDD inverse also uses DEM when available
row, col = geo.latlon_to_image(34.05, -118.25)

# Or with explicit per-point heights embedded in input
coords = np.array([[34.0, -118.2, 450.0],
                    [34.1, -118.3, 520.0]])   # (N, 3) [lat, lon, h]
pixels = geo.latlon_to_image(coords)          # → (N, 2) [row, col]
```

**Properties:**

```python
geo.projection_type  # 'PlaneProjection', 'GeographicProjection', or 'CylindricalProjection'
geo.has_rdot         # True when R/Rdot refinement is active
geo.default_hae      # Reference point height
```

---

### BIOMASS L1 (Ground Control Points)

BIOMASS L1 provides geolocation via Ground Control Points. The
GCPGeolocation class builds a Delaunay triangulation for interpolation.

```python
from grdl.IO.sar import BIOMASSL1Reader
from grdl.geolocation import GCPGeolocation

with BIOMASSL1Reader('biomass_l1.tif') as reader:
    geo = GCPGeolocation.from_dict(
        reader.metadata.get('geo_info', {}),
        reader.metadata,
    )

    lat, lon, h = geo.image_to_latlon(500, 1000)

    # Interpolation quality check
    errors = geo.get_interpolation_error()
    print(f"RMS error: {errors['rms_error_m']:.1f} m")
```

Heights come from the GCP data itself. External DEM is not used.

---

### NISAR RSLC

NISAR RSLC provides a 3-D geolocation grid embedded in the HDF5 product.
The class builds bivariate spline interpolators from this grid.

```python
from grdl.IO.sar import NISARReader
from grdl.geolocation import NISARGeolocation

with NISARReader('nisar_rslc.h5') as reader:
    # from_reader auto-dispatches: RSLC → NISARGeolocation, GSLC → AffineGeolocation
    geo = NISARGeolocation.from_reader(reader)

    lat, lon, h = geo.image_to_latlon(500, 1000)
```

---

### Sentinel-1 IW SLC

Sentinel-1 SLC provides an annotation geolocation grid. Forward and
inverse transforms use spline interpolation on this grid.

```python
from grdl.IO.sar import Sentinel1SLCReader
from grdl.geolocation import Sentinel1SLCGeolocation

with Sentinel1SLCReader('s1_slc.safe') as reader:
    geo = Sentinel1SLCGeolocation.from_reader(reader)

    pixels = np.column_stack([
        np.arange(0, 1000, 100),   # rows
        np.arange(0, 1000, 100),   # cols
    ])                             # (10, 2)
    geo_pts = geo.image_to_latlon(pixels)  # → (10, 3) [lat, lon, h]
```

---

## EO Geolocation

### Geocoded Rasters (GeoTIFF, COG, GRD)

For imagery with an affine transform and CRS (GeoTIFFs, Cloud-Optimized
GeoTIFFs, orthorectified products, SAR GRD). Handles any CRS —
automatically reprojects to WGS-84.

```python
from grdl.IO import GeoTIFFReader
from grdl.geolocation import AffineGeolocation

with GeoTIFFReader('ortho.tif') as reader:
    geo = AffineGeolocation.from_reader(reader)

    # Pixel to geographic (always returns WGS-84 lat/lon)
    lat, lon, h = geo.image_to_latlon(0, 0)

    # Geographic to pixel
    row, col = geo.latlon_to_image(lat, lon)

    # Works with any CRS (UTM, state plane, etc.)
    # Reprojection to WGS-84 is handled internally
```

**Direct construction (without reader):**

```python
from rasterio.transform import Affine

transform = Affine(0.5, 0.0, 500000.0,
                   0.0, -0.5, 4500000.0)  # UTM Zone 17N

geo = AffineGeolocation(
    transform=transform,
    shape=(10000, 10000),
    crs='EPSG:32617',
)
```

---

### EO NITF with RPC (Rational Polynomial Coefficients)

RPC00B models map between geodetic (lat, lon, height) and image pixels
via 20-term cubic rational polynomials. Common in commercial satellite
imagery (WorldView, GeoEye, RapidEye).

```python
from grdl.IO.eo import EONITFReader
from grdl.geolocation import RPCGeolocation, open_elevation

with EONITFReader('satellite.ntf') as reader:
    geo = RPCGeolocation.from_reader(reader)

    # Forward (image → ground): Newton-Raphson at height=0
    lat, lon, h = geo.image_to_latlon(2048, 2048)

    # Inverse (ground → image): direct polynomial evaluation
    row, col = geo.latlon_to_image(lat, lon, h)
```

**With DEM for terrain-corrected projection:**

```python
geo = RPCGeolocation.from_reader(reader)
geo.elevation = open_elevation(
    '/data/fabdem/',
    geoid_path='/data/egm96.pgm',
)

# Iterates per-point: project → DEM lookup → re-project → converge
lat, lon, terrain_h = geo.image_to_latlon(2048, 2048)

# Flat vs terrain-corrected comparison
geo_flat = RPCGeolocation.from_reader(reader)
flat_lat, flat_lon, _ = geo_flat.image_to_latlon(2048, 2048)
# lat != flat_lat when terrain height differs from 0
```

**Batch processing (10,000+ points):**

```python
pixels = np.column_stack([
    np.random.uniform(0, 4096, 10000),   # rows
    np.random.uniform(0, 4096, 10000),   # cols
])                                       # (10000, 2)

# Fully vectorized — no Python loops
geo_pts = geo.image_to_latlon(pixels)    # → (10000, 3) [lat, lon, h]

# Round-trip validation
px_back = geo.latlon_to_image(geo_pts)   # → (10000, 2) [row, col]
max_error = np.max(np.linalg.norm(px_back - pixels, axis=1))
# max_error < 0.01 pixels
```

**Direct construction from coefficients:**

```python
from grdl.IO.models.eo_nitf import RPCCoefficients

rpc = RPCCoefficients(
    line_off=2048.0, samp_off=2048.0,
    lat_off=38.0, long_off=-77.0, height_off=250.0,
    line_scale=2048.0, samp_scale=2048.0,
    lat_scale=0.5, long_scale=0.5, height_scale=500.0,
    line_num_coef=np.array([...]),  # 20 coefficients
    line_den_coef=np.array([...]),
    samp_num_coef=np.array([...]),
    samp_den_coef=np.array([...]),
)

geo = RPCGeolocation(rpc, shape=(4096, 4096))
```

---

### EO NITF with RSM (Replacement Sensor Model)

RSM provides higher-fidelity rational polynomials with variable-order
terms. Supports geodetic (G), cartographic (C), and rectangular ECEF (R)
ground domain types.

```python
from grdl.IO.eo import EONITFReader
from grdl.geolocation import RSMGeolocation

with EONITFReader('image_rsm.ntf') as reader:
    geo = RSMGeolocation.from_reader(reader)

    lat, lon, h = geo.image_to_latlon(2048, 2048)
    row, col = geo.latlon_to_image(lat, lon, h)
```

Usage is identical to RPCGeolocation. DEM attachment, batch processing,
and stacked array inputs all work the same way.

---

## Elevation / DEM

### Attaching a DEM

The DEM is **always** attached to the geolocation object. It is never
passed separately to downstream consumers (orthorectifier, etc.).

```python
from grdl.geolocation import open_elevation

# Auto-detect format from path
dem = open_elevation('/data/srtm/')                          # DTED directory
dem = open_elevation('/data/copernicus_30m.tif')             # single GeoTIFF
dem = open_elevation('/data/fabdem/')                         # tiled GeoTIFF directory
dem = open_elevation('/data/dted/', geoid_path='/data/egm96.pgm')  # with geoid

# Attach to geolocation
geo.elevation = dem

# All subsequent calls use terrain heights automatically
lat, lon, h = geo.image_to_latlon(500, 1000)
row, col = geo.latlon_to_image(lat, lon)
```

### Elevation Models

**ConstantElevation** — fixed height fallback:

```python
from grdl.geolocation.elevation import ConstantElevation
dem = ConstantElevation(height=500.0)
h = dem.get_elevation(34.05, -118.25)  # always 500.0
```

**DTEDElevation** — DTED Level 0/1/2 with configurable interpolation:

```python
from grdl.geolocation.elevation import DTEDElevation
dem = DTEDElevation('/data/dted/', geoid_path='/data/egm96.pgm')
# Default is bicubic interpolation (order=3), recommended for ortho.
# Cross-tile boundary stitching and nodata void handling are automatic.
h = dem.get_elevation(34.05, -118.25)       # scalar query
hs = dem.get_elevation(lats_arr, lons_arr)  # vectorized
```

**GeoTIFFDEM** — single GeoTIFF with configurable interpolation:

```python
from grdl.geolocation.elevation import GeoTIFFDEM
dem = GeoTIFFDEM('/data/srtm_30m.tif', geoid_path='/data/egm96.pgm')
# Default is bicubic interpolation (order=3), recommended for ortho
```

**TiledGeoTIFFDEM** — multi-tile coverage (FABDEM, Copernicus):

```python
from grdl.geolocation.elevation import TiledGeoTIFFDEM
dem = TiledGeoTIFFDEM(
    '/data/fabdem/',
    geoid_path='/data/egm96.pgm',
    interpolation=3,    # bicubic (default)
    max_open_tiles=8,   # LRU cache size
)
# Seamless cross-tile interpolation
h = dem.get_elevation(34.05, -118.25)
```

**open_elevation()** — auto-detect factory:

```python
from grdl.geolocation.elevation import open_elevation

dem = open_elevation(
    '/data/terrain/',
    geoid_path='/data/egm96.pgm',
    location=(34.05, -118.25),  # optional coverage check
    fallback_height=0.0,        # if nothing works
)
# Returns DTEDElevation, GeoTIFFDEM, TiledGeoTIFFDEM, or ConstantElevation
# Never returns None
```

### Geoid Correction

When you provide `geoid_path`, the elevation model automatically converts
MSL heights (as stored in most DEM files) to HAE (Height Above Ellipsoid,
which geolocation models require):

```
height_hae = height_msl + geoid_undulation
```

Supported formats: EGM96 PGM (`.pgm`), any geoid GeoTIFF (`.tif`).
Scale and offset are read from file metadata when available (GeoTIFF
band scale/offset tags, PGM `# Scale` / `# Offset` comment lines),
falling back to EGM96 defaults for standard `.pgm` files.

---

## Coordinate Utilities

Standalone functions for coordinate system conversions. All accept stacked
(N, 3) arrays and return (N, 3) arrays. Scalar inputs return (3,) arrays
with tuple unpacking support.

```python
from grdl.geolocation import (
    geodetic_to_ecef,
    ecef_to_geodetic,
    geodetic_to_enu,
    enu_to_geodetic,
    meters_per_degree,
)

# Geodetic ↔ ECEF (scalar — tuple unpacking)
x, y, z = geodetic_to_ecef(np.array([34.05, -118.25, 100.0]))
lat, lon, h = ecef_to_geodetic(np.array([x, y, z]))

# Geodetic ↔ ECEF (batch — stacked arrays)
pts = np.array([[34.05, -118.25, 100.0],
                [34.06, -118.24, 120.0]])          # (N, 3)
ecef_pts = geodetic_to_ecef(pts)                   # → (N, 3)
geo_pts = ecef_to_geodetic(ecef_pts)               # → (N, 3)

# Geodetic ↔ Local ENU (meters relative to a reference point)
ref = np.array([34.05, -118.25, 100.0])            # (3,) reference
pt = np.array([34.06, -118.24, 120.0])
e, n, u = geodetic_to_enu(pt, ref)
lat, lon, h = enu_to_geodetic(np.array([e, n, u]), ref)

# Batch ENU
pts = np.array([[34.06, -118.24, 120.0],
                [34.07, -118.23, 130.0]])           # (N, 3)
enu_pts = geodetic_to_enu(pts, ref)                 # → (N, 3)
geo_back = enu_to_geodetic(enu_pts, ref)            # → (N, 3)

# Meters per degree at a latitude
m_lat, m_lon = meters_per_degree(34.05)             # → (2,)
# m_lat ≈ 110,900 m/deg, m_lon ≈ 91,900 m/deg at 34°N
```

---

## Footprint and Bounds

Every geolocation object can compute its geographic footprint:

```python
# Polygon perimeter + bounding box
footprint = geo.get_footprint()
# footprint['type']        → 'Polygon'
# footprint['coordinates'] → [(lon, lat), (lon, lat), ...]
# footprint['bounds']      → (min_lon, min_lat, max_lon, max_lat)

# Shortcut for just the bounding box
min_lon, min_lat, max_lon, max_lat = geo.get_bounds()
```

---

## Integration with Orthorectification

The geolocation object is the single source of truth for coordinate
transforms in the orthorectification pipeline. The DEM lives on the
geolocation — the orthorectifier maps coordinates *through* the
geolocation object.

```python
from grdl.IO.sar import SICDReader
from grdl.geolocation import SICDGeolocation, open_elevation
from grdl.image_processing.ortho import orthorectify, GeographicGrid

with SICDReader('complex.nitf') as reader:
    geo = SICDGeolocation.from_reader(reader)
    geo.elevation = open_elevation('/data/dted/', geoid_path='/data/egm96.pgm')

    result = orthorectify(
        geolocation=geo,       # geo.elevation handles terrain internally
        reader=reader,
        output_grid=GeographicGrid.from_geolocation(geo, resolution=0.00005),
    )
```

**Common mistake -- forgetting to set the DEM on the geolocation:**

```python
# WRONG: no DEM attached — R/Rdot iteration uses height=0
geo = SICDGeolocation.from_reader(reader)
# geo.elevation is None — terrain is ignored!

result = orthorectify(
    geolocation=geo,       # no terrain correction in the coordinate transform
    reader=reader,
    output_grid=GeographicGrid.from_geolocation(geo, resolution=0.00005),
)
```

Always attach the DEM to the geolocation object *before* orthorectification.
The orthorectifier has no ``elevation=`` parameter -- terrain correction
happens inside the geolocation's R/Rdot iteration via ``geo.elevation``.

---

## Common Patterns

### Stacked Array Convention

All geolocation methods use the **(N, M) stacked ndarray** convention:

- **Inputs** are (N, 2) or (N, 3) arrays where each row is one point
- **Outputs** are (N, M) arrays where each row is one result
- Scalar inputs return (M,) arrays compatible with tuple unpacking

```python
# Scalar shorthand — tuple unpacking
lat, lon, h = geo.image_to_latlon(500, 1000)
row, col = geo.latlon_to_image(lat, lon)

# Batch — stacked arrays
pixels = np.array([[100, 500],
                    [200, 600],
                    [300, 700]])              # (3, 2) [row, col]
geo_pts = geo.image_to_latlon(pixels)        # (3, 3) [lat, lon, h]
px_back = geo.latlon_to_image(geo_pts)       # (3, 2) [row, col]

# Indexing: result[i] is the i-th point, result[:, j] is the j-th coordinate
geo_pts[0]      # [lat, lon, h] of first point
geo_pts[:, 0]   # all latitudes
geo_pts[:, 2]   # all heights
```

### Height in Inverse Calls

For `latlon_to_image`, height is embedded as the 3rd column:

```python
# Scalar: pass height as 3rd argument
row, col = geo.latlon_to_image(34.05, -118.25, 500.0)

# Batch: include height as 3rd column in (N, 3) input
coords = np.array([[34.0, -118.2, 450.0],
                    [34.1, -118.3, 520.0]])   # (N, 3) [lat, lon, h]
pixels = geo.latlon_to_image(coords)          # → (N, 2) [row, col]

# Without height: (N, 2) input uses default height (DEM or 0)
coords_2d = np.array([[34.0, -118.2],
                        [34.1, -118.3]])      # (N, 2) [lat, lon]
pixels = geo.latlon_to_image(coords_2d)       # → (N, 2) [row, col]
```

### Round-Trip Validation

```python
# Forward then inverse — should recover original pixels
pixels = np.array([[500, 1000], [600, 1200]])
geo_pts = geo.image_to_latlon(pixels)          # (2, 3)
px_back = geo.latlon_to_image(geo_pts)         # (2, 2) — height from col 3 is used
error = np.max(np.linalg.norm(px_back - pixels, axis=1))
# error < 0.01 pixels for well-conditioned models
```

### Chip Offset Wrapper

When working with a sub-region (chip) of a larger image, use the
library-provided ``ChipGeolocation`` class to offset chip-local
coordinates to full-image coordinates automatically:

```python
from grdl.geolocation import ChipGeolocation

# region is a ChipRegion from data_prep (has .row_start, .col_start, .nrows, .ncols)
chip_geo = ChipGeolocation(
    geo,
    row_offset=region.row_start,
    col_offset=region.col_start,
    shape=(region.nrows, region.ncols),
)

# Chip-local (0, 0) maps to (row_start, col_start) in the full image
lat, lon, h = chip_geo.image_to_latlon(0, 0)

# Batch works the same way
pixels = np.array([[0, 0], [10, 20], [50, 100]])   # chip-local
geo_pts = chip_geo.image_to_latlon(pixels)          # (3, 3) [lat, lon, h]
```

### Scene Properties

```python
geo.shape          # (rows, cols) image dimensions
geo.crs            # coordinate reference system (usually 'WGS84')
geo.elevation      # attached ElevationModel (or None)
geo.default_hae    # scene reference height (SCP, ref point, etc.)
```
