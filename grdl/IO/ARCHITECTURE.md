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
│   ├── __init__.py          # Public API exports
│   ├── base.py              # Abstract base classes (ABCs)
│   ├── sar.py               # SAR format readers (SICD, CPHD, GRD)
│   ├── biomass.py           # BIOMASS mission readers (L1 SCS)
│   ├── catalog.py           # BIOMASS catalog and download manager
│   ├── eo.py                # EO imagery readers (planned)
│   ├── geospatial.py        # Vector data readers/writers (planned)
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

All `ImageReader` subclasses must populate `self.metadata` dict with:
- `'format'`: Format name (e.g., 'SICD', 'GRD')
- `'rows'`: Image height in pixels
- `'cols'`: Image width in pixels
- `'dtype'`: NumPy dtype string
- Format-specific fields as needed

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

**Implemented**: `BIOMASSCatalog` in `catalog.py` provides local discovery,
ESA MAAP STAC search, OAuth2-authenticated download, and SQLite tracking.

### Integration with Image Processing

IO readers integrate with the image processing module through composable patterns:

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
- **Data Preparation**: `ChipExtractor` and `Tiler` (both inheriting `ChipBase`) in `grdl.data_prep` compute
  chip and tile index bounds within bounded images. `Normalizer` handles intensity normalization.
- **ImageJ Ports**: 12 classic ImageJ/Fiji algorithms (`grdl.imagej`) inherit from `ImageTransform`, enabling
  direct use in processing pipelines alongside orthorectification and decomposition. Includes spatial filters
  (RollingBallBackground, UnsharpMask, RankFilters, MorphologicalFilter), contrast enhancement (CLAHE,
  GammaCorrection), thresholding/segmentation (AutoLocalThreshold, StatisticalRegionMerging), edge/feature
  detection (EdgeDetector, FindMaxima), frequency-domain filtering (FFTBandpassFilter), and stack operations
  (ZProjection). Each port carries `__imagej_source__`, `__imagej_version__`, `__gpu_compatible__`, and
  `@processor_tags` metadata for provenance tracking and capability discovery.

## SAR Readers Implementation

### Technology Choices

| Format | Backend Library | Rationale |
|--------|----------------|-----------|
| SICD/CPHD | SARPY | NGA official implementation, handles NITF complexity |
| GRD | Rasterio | Industry standard for GeoTIFF, handles COGs |
| CRSD | SARPY (planned) | Consistent with SICD/CPHD |
| SLC | Rasterio (planned) | Often stored as GeoTIFF |

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
        # SARPY's open_complex() handles:
        # - NITF parsing
        # - SICD metadata extraction
        # - Lazy dataset setup
        self.reader = open_complex(filepath)
        self.sicd_meta = self.reader.sicd_meta
```

**Key Decisions:**

1. **Expose SARPY metadata object**: `self.sicd_meta` available for advanced users
   - SICD metadata is extremely rich (100+ fields)
   - Extracting all to dict would be overwhelming
   - Direct access for power users, simplified dict for common fields

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

### GRDReader

**Format Background:**
- GRD = Ground Range Detected (industry term, not a standard)
- Geocoded, magnitude-detected SAR products
- Common format for Sentinel-1, RADARSAT-2, etc.
- Usually GeoTIFF with embedded GCP/geotransform

**Implementation Details:**

Uses Rasterio for GeoTIFF reading:
- Native support for Cloud-Optimized GeoTIFFs (COG)
- Handles coordinate reference systems (CRS)
- Affine transform for pixel ↔ coordinate mapping

**Key Decisions:**

1. **Real-valued data**: Returns magnitude, not complex
   - GRD products are already detected (|I+jQ|²)
   - No phase information available
   - Simpler data type (float32 instead of complex64)

2. **Geolocation includes CRS**: Unlike SICD
   - GRD is geocoded, SICD is in native geometry
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
3. GRDReader (GeoTIFF)

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

For images that don't fit in memory:

```python
def process_large_image(reader, chunk_size=1024):
    rows, cols = reader.get_shape()

    for r in range(0, rows, chunk_size):
        for c in range(0, cols, chunk_size):
            r_end = min(r + chunk_size, rows)
            c_end = min(c + chunk_size, cols)

            chip = reader.read_chip(r, r_end, c, c_end)
            # Process chip (e.g., filter, detect, classify)
            result = process(chip)
            # Write result
            writer.write_chip(result, r, c)
```

This pattern is enabled by:
- `read_chip()` - spatial subsetting
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
- `Dict[str, Any]` for heterogeneous metadata
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
   - Reuse GRDReader structure (also rasterio)
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
