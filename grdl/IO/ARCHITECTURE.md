# IO Module Architecture

Technical design documentation for the IO module.

## Design Philosophy

The IO module is built on three core principles:

1. **Abstraction** - Hide format-specific complexity behind clean interfaces
2. **Composability** - Readers work seamlessly with other GRDL modules
3. **Performance** - Lazy loading and memory-efficient operations by default

## Module Structure

```
grdl/
├── exceptions.py            # Custom exception hierarchy (GrdlError, ValidationError, etc.)
├── py.typed                 # PEP 561 type stub marker
├── IO/
│   ├── __init__.py          # Public API exports + open_image()
│   ├── base.py              # Abstract base classes (ABCs)
│   ├── models/              # Typed metadata dataclasses
│   │   ├── __init__.py      # Re-exports all metadata classes
│   │   ├── base.py          # ImageMetadata (base dataclass)
│   │   ├── common.py        # Shared primitives (XYZ, LatLonHAE, RowCol, Poly1D, Poly2D, XYZPoly)
│   │   ├── sicd.py          # SICDMetadata + ~35 nested section dataclasses
│   │   ├── sidd.py          # SIDDMetadata + ~25 nested section dataclasses
│   │   ├── biomass.py       # BIOMASSMetadata (flat typed fields)
│   │   ├── viirs.py         # VIIRSMetadata (flat typed fields)
│   │   └── aster.py         # ASTERMetadata (flat typed fields)
│   ├── geotiff.py           # GeoTIFFReader (base data format, rasterio)
│   ├── hdf5.py              # HDF5Reader (base data format, h5py)
│   ├── jpeg2000.py          # JP2Reader (base data format, glymur)
│   ├── nitf.py              # NITFReader (base data format, rasterio/GDAL)
│   ├── eo/                  # EO (visible/panchromatic) readers — scaffold
│   │   ├── __init__.py      # EO exports + open_eo()
│   │   └── _backend.py      # rasterio/glymur availability detection
│   ├── ir/                  # IR/thermal readers
│   │   ├── __init__.py      # IR exports + open_ir()
│   │   ├── _backend.py      # rasterio/h5py availability detection
│   │   └── aster.py         # ASTERReader (wraps GeoTIFFReader)
│   ├── multispectral/       # Multispectral/hyperspectral readers
│   │   ├── __init__.py      # MS exports + open_multispectral()
│   │   ├── _backend.py      # h5py/xarray/spectral availability detection
│   │   └── viirs.py         # VIIRSReader (wraps h5py)
│   ├── sar/                 # SAR-specific format readers
│   │   ├── __init__.py      # SAR exports + open_sar()
│   │   ├── _backend.py      # sarkit/sarpy availability detection
│   │   ├── sicd.py          # SICDReader (sarkit primary, sarpy fallback)
│   │   ├── cphd.py          # CPHDReader (sarkit primary, sarpy fallback)
│   │   ├── crsd.py          # CRSDReader (sarkit-only)
│   │   ├── sidd.py          # SIDDReader (sarkit-only)
│   │   ├── biomass.py       # BIOMASSL1Reader (rasterio)
│   │   └── biomass_catalog.py  # BIOMASSCatalog + load_credentials
│   ├── README.md            # User documentation
│   ├── ARCHITECTURE.md      # This file
│   └── TODO.md              # Roadmap and planned features
├── image_processing/
│   ├── base.py              # ImageProcessor, ImageTransform, BandwiseTransformMixin
│   ├── versioning.py        # @processor_version, @processor_tags, TunableParameterSpec
│   ├── pipeline.py          # Pipeline (sequential transform composition)
│   └── ...                  # ortho/, decomposition/, detection/ subdomains
├── imagej/                  # 12 ImageJ/Fiji ports (10 subdirs matching ImageJ menu hierarchy)
├── data_prep/               # ChipBase ABC, ChipExtractor, Tiler, Normalizer
└── coregistration/          # Affine, projective, feature-matching alignment
```

## Base Class Hierarchy

### ImageReader (ABC)

The root abstract class for all imagery readers.

**Design Rationale:**

- **Lazy Loading**: Constructors load only metadata, not pixel data
  - Allows quick inspection of large datasets
  - Reduces memory footprint for batch processing
  - Enables streaming workflows

- **Spatial Subsetting**: `read_chip()` is the primary interface
  - Most workflows process small regions of large images
  - Avoids loading entire images into memory
  - Enables parallel processing of image tiles

- **Context Managers**: `__enter__`/`__exit__` for automatic cleanup
  - Ensures file handles are properly closed
  - Prevents resource leaks in long-running processes
  - Pythonic interface users expect

**Key Methods:**

| Method | Purpose | When to Use |
|--------|---------|------------|
| `_load_metadata()` | Abstract - load format metadata | Constructor calls this |
| `read_chip()` | Read spatial subset | Default - memory efficient |
| `read_full()` | Read entire image | Small images or full processing |
| `get_shape()` | Image dimensions | Planning memory allocation |
| `get_dtype()` | Pixel data type | Buffer allocation |
| `get_geolocation()` | Georeferencing | Coordinate transforms |

**Metadata Contract:**

All `ImageReader` subclasses must populate `self.metadata` with an `ImageMetadata` dataclass (or a format-specific subclass). Required fields on every metadata instance:

- `format`: Format name (e.g., `'SICD'`, `'GRD'`, `'BIOMASS_L1_SCS'`)
- `rows`: Image height in pixels
- `cols`: Image width in pixels
- `dtype`: NumPy dtype string (e.g., `'complex64'`, `'float32'`)

Format-specific metadata is provided via typed subclasses:

| Format | Metadata Class | Notes |
|--------|---------------|-------|
| GeoTIFF, NITF, HDF5 | `ImageMetadata` | Base class; `bands`, `crs` on base |
| SICD | `SICDMetadata(ImageMetadata)` | 17 nested section dataclasses (collection_info, geo_data, grid, etc.) |
| SIDD | `SIDDMetadata(ImageMetadata)` | 13 nested section dataclasses (product_creation, display, measurement, etc.) |
| BIOMASS | `BIOMASSMetadata(ImageMetadata)` | Flat typed fields (mission, swath, polarizations, etc.) |
| VIIRS | `VIIRSMetadata(ImageMetadata)` | Flat typed fields (satellite, product, calibration, etc.) |
| ASTER | `ASTERMetadata(ImageMetadata)` | Flat typed fields (acquisition, orbital, solar geometry, etc.) |

All metadata classes support dict-like `[]` access for backward compatibility (`meta['format']`, `'rows' in meta`, `meta.keys()`) alongside native attribute access (`meta.format`, `meta.collection_info.radar_mode.mode_type`)

### ImageWriter (ABC)

Abstract class for writing imagery.

**Design Rationale:**

- **Incremental Writes**: `write_chip()` enables streaming
  - Write large images piece-by-piece
  - Update existing files without rewriting
  - Support for tiled output formats

- **Metadata Preservation**: Constructor accepts metadata dict
  - Ensures geolocation survives read-write cycles
  - Maintains sensor-specific annotations
  - Format conversion without information loss

**Not Yet Implemented** - writers are lower priority than readers.

### CatalogInterface (ABC)

Abstract class for image discovery and spatial queries.

**Design Rationale:**

- **Separation of Concerns**: Cataloging is distinct from reading
  - Can index thousands of files without opening all
  - Spatial queries don't require loading pixel data
  - Enables database-backed implementations

- **Format Agnostic**: Works across SAR, EO, MSI
  - Unified interface for heterogeneous collections
  - Enables multi-sensor workflows
  - Simplifies collection management

**Implemented**: `BIOMASSCatalog` in `sar/biomass_catalog.py` provides local
discovery, ESA MAAP STAC search, OAuth2-authenticated download, and SQLite tracking.

### Integration with Other GRDL Modules

IO readers are designed to compose with other GRDL modules. **Always use the purpose-built GRDL module for each task** — IO for loading, `data_prep` for chip planning, `image_processing` for transforms:

**Data Preparation (chip/tile planning):**
- `ChipExtractor` and `Tiler` from `grdl.data_prep` compute chip and tile index bounds (index-only, no pixel data). Use their `ChipRegion` output to drive `reader.read_chip()` calls instead of hand-rolling `for r in range(0, rows, chunk):` loops.
- `Normalizer` handles intensity normalization (`minmax`, `zscore`, `percentile`, `unit_norm`).

**Full integration example** (see `grdl/example/image_processing/sar/sublook_compare.py`):

```python
from grdl.IO import SICDReader                           # IO: load imagery
from grdl.data_prep import ChipExtractor                 # data_prep: plan chips
from grdl.image_processing.sar import SublookDecomposition  # processing: transform

with SICDReader('image.nitf') as reader:
    rows, cols = reader.get_shape()

    # data_prep plans the chip (boundary-snapped, index-only)
    extractor = ChipExtractor(nrows=rows, ncols=cols)
    region = extractor.chip_at_point(rows // 2, cols // 2,
                                     row_width=2048, col_width=2048)

    # IO reads only the planned region
    chip = reader.read_chip(region.row_start, region.row_end,
                            region.col_start, region.col_end)

    # image_processing transforms the data
    sublook = SublookDecomposition(reader.metadata, num_looks=3)
    looks = sublook.decompose(chip)
```

**Other integration points:**
- **Geolocation**: `Geolocation.from_reader(reader)` constructs coordinate transforms
- **Orthorectification**: `Orthorectifier.apply_from_reader()` reads chips directly from a reader
- **Decomposition**: Complex data from `BIOMASSL1Reader` feeds directly into `PauliDecomposition`
- **Detection**: `ImageDetector._geo_register_detections()` uses Geolocation for pixel-to-latlon transforms
- **Pipeline**: `Pipeline` chains multiple `ImageTransform` instances into a single callable with progress
  callback rescaling. Pipelines are themselves `ImageTransform` instances and can be nested.
- **BandwiseTransformMixin**: Mixin that auto-applies 2D transforms across 3D `(bands, rows, cols)` stacks,
  enabling all single-band processors to work on multi-band imagery without manual band looping.
- **Progress Callbacks**: Long-running processors (SRM, CLAHE, RollingBall) accept an optional
  `progress_callback` keyword argument for real-time progress reporting.
- **ImageJ Ports**: 12 classic ImageJ/Fiji algorithms (`grdl.imagej`) inherit from `ImageTransform`, enabling
  direct use in processing pipelines alongside orthorectification and decomposition.

## SAR Readers Implementation

### Technology Choices

| Format | Backend Library | Rationale |
|--------|----------------|-----------|
| SICD | sarkit (primary), sarpy (fallback) | sarkit is the modern NGA library; sarpy fallback for compatibility |
| CPHD | sarkit (primary), sarpy (fallback) | sarkit is the modern NGA library; sarpy fallback for compatibility |
| CRSD | sarkit (only) | sarpy does not fully support CRSD |
| SIDD | sarkit (only) | sarpy does not fully support SIDD |
| GeoTIFF | rasterio | Industry standard for GeoTIFF, handles COGs |
| NITF | rasterio (GDAL) | Generic NITF via GDAL driver; SAR NITF uses sarkit/sarpy |
| BIOMASS | rasterio | Magnitude/phase GeoTIFFs + XML annotation |

### Backend Fallback Pattern (`_backend.py`)

SAR readers use a shared backend detection module:
- `_HAS_SARKIT` / `_HAS_SARPY` flags set at import time
- `require_sar_backend()` returns `'sarkit'` or `'sarpy'` (prefers sarkit)
- `require_sarkit()` raises `ImportError` when sarkit is required (CRSD, SIDD)
- Key API difference: sarkit takes `BinaryIO`, sarpy takes filepath strings

### SICDReader

**Format Background:**
- SICD = Sensor Independent Complex Data (NGA standard)
- Complex-valued SAR imagery (I+jQ channels)
- Typically in NITF container format
- Rich metadata including collection geometry, sensor params

**Implementation Details:**

```python
class SICDReader(ImageReader):
    def __init__(self, filepath):
        self.backend = require_sar_backend('SICD')  # 'sarkit' or 'sarpy'
        super().__init__(filepath)

    # sarkit path: file opened as binary, sarkit.sicd.NitfReader(file)
    # sarpy path: sarpy.io.complex.converter.open_complex(str(filepath))
```

**Key Decisions:**

1. **Dual backend with sarkit preferred**: sarkit is the modern NGA library
   - sarkit: `NitfReader(BinaryIO)` with `read_sub_image()` for windowed reads
   - sarpy: `open_complex(filepath)` with slice indexing `reader[r1:r2, c1:c2]`
   - Raw XML tree (`_xmltree`) available for advanced users

2. **Complex data type**: Always returns `complex64`
   - SICD spec defines complex data
   - Users convert to magnitude with `np.abs()` as needed
   - Preserves phase information for interferometry

3. **Geolocation format**: Returns lat/lon/height
   - SICD stores Scene Center Point (SCP) and corner coords
   - Native geometry, not projected coordinates
   - Users transform to desired projection separately

### CPHDReader

**Format Background:**
- CPHD = Compensated Phase History Data (NGA standard)
- Raw phase history (unfocused SAR)
- Used for custom SAR focusing algorithms
- Multi-channel support (multiple apertures/polarizations)

**Implementation Details:**

Unlike SICD (formed imagery), CPHD doesn't have fixed rows/cols. Instead:
- Multiple channels, each with vectors × samples
- `get_shape()` returns dimensions of first channel
- `read_chip()` accepts channel index in `bands` parameter

**Key Decisions:**

1. **Channel-based reading**: `bands` parameter selects channel
   - CPHD is fundamentally multi-channel
   - Different from multi-band imagery (different concept)
   - Consistent with ImageReader API despite different semantics

2. **No automatic focusing**: Returns raw phase history
   - Focusing algorithms are complex and use-case specific
   - Belongs in processing module, not I/O
   - Users apply SARPY's focusing or custom algorithms

### GeoTIFFReader

**Format Background:**
- GeoTIFF is the foundational raster format for geospatial imagery
- Covers SAR GRD products, EO imagery, MSI, Cloud-Optimized GeoTIFFs (COG)
- Lives at IO base level (`IO/geotiff.py`) — not in a modality submodule
- Previously named `GRDReader`; renamed to reflect general-purpose nature

**Implementation Details:**

Uses Rasterio for GeoTIFF reading:
- Native support for Cloud-Optimized GeoTIFFs (COG)
- Handles coordinate reference systems (CRS)
- Affine transform for pixel ↔ coordinate mapping

**Key Decisions:**

1. **Real-valued data**: Returns magnitude, not complex
   - GRD products are already detected (|I+jQ|²)
   - Standard EO/MSI imagery is real-valued
   - Simpler data type (float32 instead of complex64)

2. **Geolocation includes CRS**: Unlike SICD
   - GeoTIFF is geocoded with embedded CRS and affine transform
   - Users can directly map pixels to lat/lon
   - Supports any projection (UTM, Geographic, etc.)

3. **Band indexing**: Follows rasterio convention
   - Users provide 0-based indices
   - Internally converts to rasterio's 1-based indexing
   - Consistent with NumPy/Python conventions

### open_sar() Auto-Detection

Convenience function that tries readers in order:
1. SICDReader (NITF containers often SICD)
2. CPHDReader
3. CRSDReader
4. SIDDReader
5. GeoTIFFReader (fallback for GRD products)

**Trade-offs:**
- **Pro**: User-friendly for unknown files
- **Pro**: Enables format-agnostic batch processing
- **Con**: Slower than direct reader (multiple open attempts)
- **Con**: May misidentify ambiguous formats

**Recommended Use:**
- Interactive exploration of unknown files
- Scripts processing mixed-format collections
- Prototyping before production code

**Not Recommended:**
- High-performance pipelines (use specific reader)
- Large batch jobs (overhead adds up)

## Metadata Architecture

### Package Structure (`models/`)

Metadata is organized as a Python package at `grdl/IO/models/` with the following modules:

| Module | Contents |
|--------|----------|
| `base.py` | `ImageMetadata` — base dataclass with `format`, `rows`, `cols`, `dtype`, `bands`, `crs`; dict-like `__getitem__`/`__contains__`/`keys()` |
| `common.py` | Shared primitive types: `XYZ`, `LatLon`, `LatLonHAE`, `RowCol`, `Poly1D`, `Poly2D`, `XYZPoly` |
| `sicd.py` | `SICDMetadata` + ~35 nested dataclasses covering all 17 SICD sections |
| `sidd.py` | `SIDDMetadata` + ~25 nested dataclasses covering all 13 SIDD sections |
| `biomass.py` | `BIOMASSMetadata` — flat typed fields for BIOMASS annotation XML |
| `viirs.py` | `VIIRSMetadata` — flat typed fields for VIIRS HDF5 attributes |
| `aster.py` | `ASTERMetadata` — flat typed fields for ASTER GeoTIFF + XML |
| `__init__.py` | Re-exports everything; preserves `from grdl.IO.models import ...` paths |

### Design Decisions

1. **Dataclasses over dicts**: Provides IDE autocomplete, type checking, and self-documenting field names while remaining lightweight (no runtime overhead vs. plain attributes).

2. **Nested composition**: Complex formats (SICD, SIDD) use deeply nested dataclasses mirroring the source specification structure. Example: `meta.geo_data.scp.llh.lat` mirrors the SICD XML path `GeoData/SCP/LLH/Lat`.

3. **Optional sections**: All format-specific nested sections default to `None`, so metadata is always constructible even when the source file has incomplete metadata.

4. **Backward-compatible dict access**: `ImageMetadata.__getitem__` and `__contains__` use `dataclasses.fields()` introspection, so existing code using `meta['format']` continues to work.

5. **Shared primitives**: Types like `XYZ`, `LatLonHAE`, `Poly2D` are reused across SICD and SIDD, avoiding duplication.

## Mission-Specific Readers

### BIOMASS Implementation

**Format Background:**
- BIOMASS is ESA's P-band SAR satellite mission (435 MHz)
- L1 SCS (Single-look Complex Slant) products store complex data
- Unlike other SAR formats, uses magnitude + phase GeoTIFFs (not complex-valued TIFFs)
- SAFE-like directory structure with XML annotation

**Implementation Details:**

```python
class BIOMASSL1Reader(ImageReader):
    """Reader for BIOMASS L1 SCS products."""

    def __init__(self, filepath):
        # filepath is a directory (SAFE-like structure)
        # annotation/: XML metadata
        # measurement/: magnitude and phase TIFFs
```

**Key Decisions:**

1. **Mission-Specific Module**: Created `biomass.py` instead of adding to generic `sar.py`
   - **Rationale**: BIOMASS has mission-specific processing levels (L1a/b/c, L2a/b, L3)
   - Users expect mission-specific interfaces, not generic format readers
   - Sets precedent for future satellite readers (Sentinel-1, RADARSAT, etc.)
   - Decision deferred on whether to create readers for other missions

2. **Complex Data Reconstruction**: Magnitude and phase stored separately
   - Phase TIFFs store phase in radians
   - Reconstruct complex: `complex = magnitude * exp(1j * phase)`
   - Handled transparently in `read_chip()` and `read_full()`

3. **Multi-Polarization**: All 4 polarizations in same TIFF files
   - Band 1: HH, Band 2: HV, Band 3: VH, Band 4: VV
   - `bands` parameter selects polarizations (0-based indexing)
   - Consistent with multi-band imagery convention

4. **Metadata Format**: Mission-specific fields
   - Standard fields: `rows`, `cols`, `polarizations`, `orbit_number`
   - BIOMASS-specific: `swath`, `product_type`, pixel spacings
   - Enables multi-sensor fusion with consistent metadata keys

5. **Geolocation**: Slant range geometry with GCPs
   - BIOMASS L1 is not geocoded (unlike GRD products)
   - GCPs provided for geolocation transforms
   - Compatible with SICD geolocation format

**Format vs. Mission Trade-off:**

| Approach | Pros | Cons |
|----------|------|------|
| Format-based (sar.py) | Consistent with current code | Loses mission context and processing levels |
| Mission-specific (biomass.py) | Captures mission structure | Different pattern, may proliferate files |

**Decision**: Mission-specific for BIOMASS
- Mission processing levels (L1a/b/c, L2a/b, L3) are important to users
- Rich mission metadata doesn't fit generic readers
- Pattern may be adopted for other missions in future

### BIOMASS Catalog System

**Implementation Details:**

```python
class BIOMASSCatalog(CatalogInterface):
    """Catalog and download manager for BIOMASS products.

    Uses ESA MAAP STAC API for search and OAuth2 for downloads.
    """
```

**Key Features:**

1. **Local Discovery**: Scans file system for BIOMASS product directories
   - Pattern matching on directory names (BIO_S*_SCS*)
   - Indexes products in SQLite database
   - Extracts metadata without loading pixel data

2. **ESA MAAP STAC API**: Queries the operational MAAP catalog
   - POST-based STAC search at `catalog.maap.eo.esa.int/catalogue`
   - CQL2 filtering by product type, orbit, bounding box, date range
   - Collection IDs: `BiomassLevel1a`, `BiomassLevel1b`, `BiomassLevel2a`
   - Stores results in local database for offline access

3. **OAuth2 Authentication**: Token-based access to ESA data
   - Offline token stored in `~/.config/geoint/credentials.json` (repo-agnostic)
   - Exchanged for short-lived access tokens via MAAP IAM endpoint
   - Public client secret shared by all MAAP users (published in ESA docs)
   - Fallback to environment variables (`ESA_MAAP_OFFLINE_TOKEN`)
   - `load_credentials()` utility exported from `grdl.IO`

4. **Download Management**: Streaming product download with extraction
   - Bearer token authentication on download requests
   - Streaming to disk with progress reporting
   - Automatic ZIP extraction to product directory
   - Database updated with local path on completion

5. **SQLite Database**: Unified tracking of local and remote products
   - Schema supports all BIOMASS product levels
   - Indexed on common query fields (date, orbit, processing level)
   - Tracks download URL, corner coordinates, full STAC metadata JSON
   - Enables fast queries without re-scanning file system or re-querying API

**Design Decisions:**

1. **Mission-Specific Catalog**: `BIOMASSCatalog` instead of generic `ImageCatalog`
   - BIOMASS has specific product naming conventions
   - ESA MAAP API endpoints are mission-specific
   - Allows optimized schema for BIOMASS metadata
   - Generic catalog planned for future (multi-mission support)

2. **SQLite vs. Other Databases**: Chose SQLite for simplicity
   - No server setup required
   - File-based, portable
   - Sufficient for most use cases (thousands of products)
   - Can migrate to PostgreSQL/PostGIS for large-scale deployments

3. **Dual Index**: Products tracked both locally and remotely
   - Enables "download missing products" workflows
   - Tracks download status
   - Supports offline work (query database without ESA access)

4. **Repo-Agnostic Credentials**: `~/.config/geoint/credentials.json`
   - Shared across all projects using ESA data (not tied to GRDL)
   - Never committed to version control
   - Follows XDG Base Directory convention

## EO / IR / Multispectral Submodules

### Modality-Based Organization

Following the `sar/` pattern, new modality submodules wrap base-level format readers
(GeoTIFFReader, HDF5Reader) with sensor-specific metadata extraction:

| Submodule | Modality | Sensors | Backend |
|-----------|----------|---------|---------|
| `eo/` | Electro-Optical (VIS/NIR) | _(scaffold — planned: Landsat OLI, Sentinel-2, HLS)_ | rasterio, glymur |
| `ir/` | Thermal / Infrared | ASTER (L1T, GDEM) | rasterio |
| `multispectral/` | Multispectral / Hyperspectral | VIIRS (VNP46A1, VNP13A1) | h5py |

### ASTERReader (`ir/aster.py`)

**Format Background:**
- ASTER = Advanced Spaceborne Thermal Emission and Reflection Radiometer (Terra satellite)
- L1T products: Registered Radiance at Sensor, GeoTIFF format
- ASTGTM products: Global Digital Elevation Model v3, GeoTIFF format
- TIR subsystem (bands 10-14, 90 m) is the only active subsystem
- SWIR detector failed April 2008; VNIR still operational

**Implementation:**
- Wraps `rasterio` directly for pixel reads (same as GeoTIFFReader)
- Extracts ASTER-specific metadata from GeoTIFF tags and companion XML
- Detects product type (L1T vs GDEM) from filename patterns
- Detects band availability (VNIR/SWIR/TIR) from filename
- Populates `ASTERMetadata` dataclass with acquisition, orbital, solar geometry fields

### VIIRSReader (`multispectral/viirs.py`)

**Format Background:**
- VIIRS = Visible Infrared Imaging Radiometer Suite (Suomi NPP, NOAA-20, NOAA-21)
- 22 bands from visible (412 nm) through thermal (12 µm)
- HDF5 format with rich file-level and dataset-level attributes
- Products: VNP46A1 (nighttime lights), VNP13A1 (vegetation index), surface reflectance

**Implementation:**
- Wraps `h5py` directly for pixel reads (same as HDF5Reader)
- Extracts VIIRS-specific metadata from HDF5 file-level attrs (satellite, product, temporal)
- Extracts dataset-level calibration attrs (scale_factor, add_offset, fill_value)
- Auto-detects first suitable 2D+ numeric dataset or accepts explicit path
- Populates `VIIRSMetadata` dataclass with satellite, temporal, spatial, and calibration fields

## Memory Management

### Lazy Loading Strategy

```python
# Constructor: Load only metadata (~KB)
reader = SICDReader('large_image.nitf')  # Fast, ~MB of memory

# Chip read: Load only requested region
chip = reader.read_chip(0, 1024, 0, 1024)  # ~8MB for 1024×1024 complex

# Full read: Load entire image
full = reader.read_full()  # Could be GB - use with caution
```

### Complex Data Memory Footprint

| Data Type | Bytes/Pixel | 1K×1K | 10K×10K | 50K×50K |
|-----------|-------------|-------|---------|---------|
| complex64 | 8 | 8 MB | 800 MB | 20 GB |
| float32 (magnitude) | 4 | 4 MB | 400 MB | 10 GB |
| uint8 (quantized) | 1 | 1 MB | 100 MB | 2.5 GB |

**Implication**: Convert to magnitude early if phase not needed.

### Chunked Processing Pattern

For images that don't fit in memory, use `ChipExtractor` from `grdl.data_prep` to plan chip regions instead of hand-rolling index arithmetic. `ChipExtractor` handles boundary snapping and uniform chip sizing:

```python
from grdl.data_prep import ChipExtractor

def process_large_image(reader, chunk_size=1024):
    rows, cols = reader.get_shape()
    extractor = ChipExtractor(nrows=rows, ncols=cols)

    for region in extractor.chip_positions(row_width=chunk_size,
                                           col_width=chunk_size):
        chip = reader.read_chip(region.row_start, region.row_end,
                                region.col_start, region.col_end)
        result = process(chip)
        writer.write_chip(result, region.row_start, region.col_start)
```

For overlapping tiles (e.g., ML inference with context padding), use `Tiler`:

```python
from grdl.data_prep import Tiler

tiler = Tiler(nrows=rows, ncols=cols, tile_size=256, stride=128)
for region in tiler.tile_positions():
    chip = reader.read_chip(region.row_start, region.row_end,
                            region.col_start, region.col_end)
```

This pattern is enabled by:
- `ChipExtractor` / `Tiler` - index-only chip planning with boundary snapping
- `read_chip()` - spatial subsetting driven by `ChipRegion` bounds
- `write_chip()` - incremental writes
- Standardized indexing across all readers

## Error Handling Strategy

### Exception Hierarchy

GRDL provides a custom exception hierarchy in `grdl.exceptions` that subclasses both
`GrdlError` and the appropriate built-in exception for backward compatibility:

```
GrdlError (base)
├── ValidationError(GrdlError, ValueError)    # Bad inputs (shape, dtype, parameter range)
├── ProcessorError(GrdlError, RuntimeError)   # Algorithm failures in apply()/detect()
├── DependencyError(GrdlError, ImportError)   # Missing optional dependency
└── GeolocationError(GrdlError, RuntimeError) # Coordinate transform failures

Built-in exceptions (still raised by IO module):
├── FileNotFoundError       # File doesn't exist
├── NotADirectoryError      # Expected directory, got file
└── IOError                 # Disk read/write failure
```

Downstream consumers (e.g., GRDK's `WorkflowExecutor`) can catch `GrdlError` to handle
all GRDL-specific errors distinctly from Python builtins, or catch the built-in base
class (`ValueError`, `ImportError`, etc.) for backward-compatible handling.

### Graceful Degradation

Readers check for dependencies and fail fast:

```python
if not SARPY_AVAILABLE:
    raise ImportError("SARPY required. Install: pip install sarpy")
```

**Design Rationale:**
- Fails at import time, not deep in execution
- Clear error message with installation command
- Allows partial installation (e.g., only rasterio, no sarpy)

### Validation Strategy

**File Existence**: Checked in `ImageReader.__init__`
- Fails fast before attempting format-specific parsing
- Prevents confusing backend errors

**Index Bounds**: Checked in `read_chip()`
- Prevents backend crashes with cryptic messages
- Provides clear "index out of bounds" errors

**Format Validation**: Delegated to backend (SARPY, rasterio)
- They have comprehensive format validators
- Don't reimplement complex format checking
- Trust authoritative libraries

## Type System

All public APIs have full type hints:

```python
def read_chip(
    self,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    bands: Optional[List[int]] = None
) -> np.ndarray:
```

**Benefits:**
- IDE autocomplete and inline documentation
- Static type checking with mypy
- Runtime validation with typeguard (optional)
- Self-documenting code

**Conventions:**
- `Union[str, Path]` for file paths (accept both)
- `Optional[X]` for nullable parameters
- Typed dataclasses for metadata (`ImageMetadata`, `SICDMetadata`, etc.) with ~60 nested types in `grdl.IO.models`
- `np.ndarray` without shape hints (too complex, varies by reader)

## Testing Strategy

### Unit Tests

Each reader/module has a dedicated test suite. Shared fixtures live in `tests/conftest.py`:
- `tests/test_io_biomass.py` - BIOMASS reader, metadata, chip I/O, geolocation
- `tests/test_geolocation_biomass.py` - GCP geolocation, round-trip accuracy
- `tests/test_image_processing_ortho.py` - Orthorectification with synthetic data
- `tests/test_image_processing_decomposition.py` - Pauli decomposition
- `tests/test_image_processing_detection.py` - Detection models, geo-registration, GeoJSON
- `tests/test_image_processing_versioning.py` - Processor versioning decorator
- `tests/test_image_processing_tunable.py` - Tunable parameter validation
- `tests/test_imagej.py` - ImageJ/Fiji port tests (12 components)
- `tests/test_coregistration.py` - Coregistration tests
- `tests/test_benchmarks.py` - Performance benchmarks (pytest-benchmark)

### Test Data

Small synthetic files in `tests/data/`:
- Avoid large real-world data in repo
- Generate minimal valid files for format testing
- Use SARPY's test data generators for SICD/CPHD

### Mock Strategy

For testing without dependencies:
```python
@unittest.mock.patch('grdl.IO.sar.SARPY_AVAILABLE', False)
def test_sicd_import_error():
    with pytest.raises(ImportError):
        SICDReader('dummy.nitf')
```

## Future Extensions

### Planned Readers

1. **CRSD** (Compensated Received Signal Data)
   - NGA standard for raw signal data
   - Use SARPY backend (supports CRSD 1.0)
   - Similar to CPHD reader structure

2. **SLC** (Single Look Complex)
   - Focused complex SAR (like SICD but less metadata)
   - Often in GeoTIFF format
   - Use rasterio backend with SAR-specific metadata parsing

3. **EO GeoTIFF**
   - Standard optical imagery
   - Already supported by GeoTIFFReader at IO base level
   - Handle multi-band (RGB, multispectral)

### Planned Writers

Implement `ImageWriter` subclasses:
- `GeoTIFFWriter` - Generic raster output
- `SICDWriter` - SICD format (via SARPY)
- `GeoJSONWriter` - Vector output

### Cloud Support

Add support for cloud-native formats:
- COG (Cloud-Optimized GeoTIFF) via rasterio
- Zarr for chunked array storage
- S3/GCS readers with fsspec backend

### Catalog Extension

`BIOMASSCatalog` is fully implemented with local discovery, ESA MAAP STAC search,
OAuth2 download, and SQLite tracking. Future catalog work:
- Generic `ImageCatalog` supporting multiple missions and formats
- PostGIS backend for large-scale deployments
- Spatial indexing (R-tree, quad-tree)
- Metadata caching for fast queries

## Dependencies Rationale

### SARPY
- **Pros**: Official NGA implementation, comprehensive SICD/CPHD/CRSD support, actively maintained
- **Cons**: Heavy dependency chain, GDAL dependency can be tricky
- **Alternative**: Roll our own NITF parser (rejected - massive complexity)

### Rasterio
- **Pros**: Industry standard, excellent GDAL wrapper, handles all raster formats
- **Cons**: GDAL dependency (system-level install complexity)
- **Alternative**: PIL/Pillow (rejected - no georeferencing support)

### NumPy
- **Pros**: Universal array library, GRDL already depends on it
- **Cons**: None
- **Alternative**: None viable

## Performance Considerations

### Benchmark Targets

| Operation | Target Latency | Notes |
|-----------|----------------|-------|
| Constructor (metadata only) | <100ms | Even for large files |
| 1K×1K chip read | <50ms | From local SSD |
| 10K×10K chip read | <500ms | Complex data, includes decode |
| Full metadata extraction | <200ms | All available fields |

### Optimization Opportunities

1. **Metadata Caching**: Serialize metadata to sidecar files
   - Avoid re-parsing NITF headers
   - Especially valuable for SICD (rich metadata)

2. **Chunk Size Tuning**: Align with format internal blocking
   - NITF often uses 256×256 or 512×512 blocks
   - GeoTIFF tiles commonly 256×256 or 512×512
   - Reading aligned regions is faster

3. **Parallel Reading**: Multi-threaded chip extraction
   - GIL released in NumPy/rasterio C code
   - Can parallelize across spatial regions
   - Especially effective for large full-image reads

## API Stability

**Semantic Versioning:**
- Breaking changes to base.py increment major version
- New readers/features increment minor version
- Bug fixes increment patch version

**Processor Versioning:**
To support both formal, long-term development and rapid prototyping, processor versioning is designed to be flexible. All image processors can be decorated with `@processor_version` to explicitly set their version.

If the `@processor_version` decorator is used without an argument, or not at all, the processor automatically inherits the project-wide version from the package metadata (defined in `pyproject.toml`). This provides a graceful fallback, ensuring that every processor has a version string. This is critical for downstream tools like GRDK, which rely on this metadata for display and caching.

- **Explicit Versioning**: For stable, production-ready processors, explicitly define the version to decouple it from the main project's release cycle. This is crucial for scientific users who need:
    - **Reproducibility**: To ensure that scientific results obtained with a specific processor version can be replicated consistently over time, independent of subsequent library updates.
    - **Stability for Research**: To provide control over when changes to the processor are adopted, preventing unexpected alterations to research workflows that rely on a specific algorithm's behavior.
    - **Independence from Library Release Cycles**: To guarantee a processor's behavior remains fixed even if the main GRDL library undergoes minor or major updates, which might introduce changes to other components.
  ```python
  @processor_version('1.2.0')
  class MyStableProcessor(ImageTransform):
      ...
  ```

- **Implicit Versioning**: For rapid development or processors that version in lock-step with the library, omit the version argument.
  ```python
  @processor_version
  class MyRapidProcessor(ImageTransform):
      ...
  ```
This dual approach allows developers to move fast while maintaining a consistent and reliable versioning scheme across the entire system.

**Deprecation Policy:**
- 2 minor version warning period
- Use Python warnings module
- Suggest migration path in warning message

**Backwards Compatibility:**
- Abstract base classes are the stability contract
- Format-specific extensions may change more freely
- Document which APIs are "stable" vs "experimental"

## Related Documentation

- [README.md](README.md) - User-facing documentation and examples
- [TODO.md](TODO.md) - Planned features and roadmap
- [/CLAUDE.md](/CLAUDE.md) - Project-wide coding standards
- [/LIBRARIES.md](/LIBRARIES.md) - Integration with other GRDL modules
