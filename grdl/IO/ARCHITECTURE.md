# IO Module Architecture

Technical design documentation for the IO module.

Modified: 2026-03-10

## Design Philosophy

The IO module is built on three core principles:

1. **Abstraction** - Hide format-specific complexity behind clean interfaces
2. **Composability** - Readers work seamlessly with other GRDL modules
3. **Performance** - Lazy loading and memory-efficient operations by default

## Module Structure

```
grdl/
├── exceptions.py            # Custom exception hierarchy (GrdlError, DependencyError, etc.)
├── py.typed                 # PEP 561 type stub marker
├── IO/
│   ├── __init__.py          # Public API exports + open_image(), get_writer(), write()
│   ├── base.py              # Abstract base classes (ABCs)
│   ├── models/              # Typed metadata dataclasses
│   │   ├── __init__.py      # Re-exports all metadata classes
│   │   ├── base.py          # ImageMetadata (base dataclass)
│   │   ├── common.py        # Shared primitives (XYZ, LatLonHAE, RowCol, Poly1D, Poly2D, XYZPoly)
│   │   ├── sicd.py          # SICDMetadata + ~35 nested section dataclasses
│   │   ├── sidd.py          # SIDDMetadata + ~25 nested section dataclasses
│   │   ├── cphd.py          # CPHDMetadata + channel/PVP/waveform dataclasses
│   │   ├── biomass.py       # BIOMASSMetadata (flat typed fields)
│   │   ├── viirs.py         # VIIRSMetadata (flat typed fields)
│   │   ├── aster.py         # ASTERMetadata (flat typed fields)
│   │   ├── sentinel2.py     # Sentinel2Metadata (flat typed fields)
│   │   ├── sentinel1_slc.py # Sentinel1SLCMetadata + burst/orbit/calibration dataclasses
│   │   ├── terrasar.py      # TerraSARMetadata + product/scene/radar dataclasses
│   │   └── nisar.py         # NISARMetadata + identification/orbit/swath dataclasses
│   ├── geotiff.py           # GeoTIFFReader, GeoTIFFWriter (rasterio)
│   ├── hdf5.py              # HDF5Reader, HDF5Writer (h5py)
│   ├── jpeg2000.py          # JP2Reader (glymur/rasterio)
│   ├── nitf.py              # NITFReader, NITFWriter (rasterio/GDAL)
│   ├── png.py               # PngWriter
│   ├── numpy_io.py          # NumpyWriter
│   ├── generic.py           # GDALFallbackReader, open_any()
│   ├── probe.py             # InvasiveProbeReader
│   ├── eo/                  # EO (visible/panchromatic) readers
│   │   ├── __init__.py      # EO exports + open_eo()
│   │   ├── _backend.py      # rasterio/glymur availability detection
│   │   └── sentinel2.py     # Sentinel2Reader (wraps JP2Reader)
│   ├── ir/                  # IR/thermal readers
│   │   ├── __init__.py      # IR exports + open_ir()
│   │   ├── _backend.py      # rasterio/h5py availability detection
│   │   └── aster.py         # ASTERReader (wraps rasterio)
│   ├── multispectral/       # Multispectral/hyperspectral readers
│   │   ├── __init__.py      # MS exports + open_multispectral()
│   │   ├── _backend.py      # h5py/xarray/spectral availability detection
│   │   └── viirs.py         # VIIRSReader (wraps h5py)
│   ├── sar/                 # SAR-specific format readers
│   │   ├── __init__.py      # SAR exports + open_sar()
│   │   ├── _backend.py      # sarkit/sarpy availability detection
│   │   ├── sicd.py          # SICDReader (sarkit primary, sarpy fallback)
│   │   ├── sicd_writer.py   # SICDWriter
│   │   ├── cphd.py          # CPHDReader (sarkit primary, sarpy fallback)
│   │   ├── crsd.py          # CRSDReader (sarkit-only)
│   │   ├── sidd.py          # SIDDReader (sarkit-only)
│   │   ├── sidd_writer.py   # SIDDWriter
│   │   ├── biomass.py       # BIOMASSL1Reader (rasterio) + open_biomass()
│   │   ├── biomass_catalog.py  # BIOMASSCatalog + load_credentials
│   │   ├── sentinel1_slc.py # Sentinel1SLCReader (rasterio)
│   │   ├── terrasar.py      # TerraSARReader + open_terrasar()
│   │   └── nisar.py         # NISARReader + open_nisar()
│   ├── README.md            # User documentation
│   ├── ARCHITECTURE.md      # This file
│   └── TODO.md              # Roadmap and planned features
├── image_processing/
│   ├── base.py              # ImageProcessor, ImageTransform, BandwiseTransformMixin
│   ├── versioning.py        # @processor_version, @processor_tags, TunableParameterSpec
│   ├── pipeline.py          # Pipeline (sequential transform composition)
│   └── ...                  # ortho/, decomposition/, detection/ subdomains
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
| CPHD | `CPHDMetadata(ImageMetadata)` | Channel, PVP, waveform, antenna, scene coordinates |
| BIOMASS | `BIOMASSMetadata(ImageMetadata)` | Flat typed fields (mission, swath, polarizations, etc.) |
| VIIRS | `VIIRSMetadata(ImageMetadata)` | Flat typed fields (satellite, product, calibration, etc.) |
| ASTER | `ASTERMetadata(ImageMetadata)` | Flat typed fields (acquisition, orbital, solar geometry, etc.) |
| Sentinel-2 | `Sentinel2Metadata(ImageMetadata)` | Flat typed fields (tile, processing level, bands, etc.) |
| Sentinel-1 SLC | `Sentinel1SLCMetadata(ImageMetadata)` | Burst, orbit, Doppler, calibration/noise vectors |
| TerraSAR-X | `TerraSARMetadata(ImageMetadata)` | Product, scene, radar params, orbit, calibration |
| NISAR | `NISARMetadata(ImageMetadata)` | Identification, orbit, attitude, swath, geolocation grid |

All metadata classes support dict-like `[]` access for backward compatibility (`meta['format']`, `'rows' in meta`, `meta.keys()`) alongside native attribute access (`meta.format`, `meta.collection_info.radar_mode.mode_type`)

### ImageWriter (ABC)

Abstract class for writing imagery.

**Design Rationale:**

- **Incremental Writes**: `write_chip()` enables streaming
  - Write large images piece-by-piece
  - Update existing files without rewriting
  - Support for tiled output formats

- **Metadata Preservation**: Constructor accepts metadata
  - Ensures geolocation survives read-write cycles
  - Maintains sensor-specific annotations
  - Format conversion without information loss

**Implementations:** `GeoTIFFWriter`, `HDF5Writer`, `NITFWriter`, `SICDWriter`, `SIDDWriter`, `PngWriter`, `NumpyWriter`. Factory access via `get_writer(format, path)` or the convenience `write(data, path)` function.

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

**Implemented**: `BIOMASSCatalog` in `sar/biomass_catalog.py` provides local
discovery, ESA MAAP STAC search, OAuth2-authenticated download, and SQLite tracking.

### Integration with Other GRDL Modules

IO readers compose with other GRDL modules. **Use the purpose-built module for each task** -- IO for loading, `data_prep` for chip planning, `image_processing` for transforms:

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
- **Pipeline**: `Pipeline` chains multiple `ImageTransform` instances; pipelines are themselves `ImageTransform` instances and can be nested
- **BandwiseTransformMixin**: Auto-applies 2D transforms across 3D `(bands, rows, cols)` stacks

## SAR Readers Implementation

### Technology Choices

| Format | Backend Library | Rationale |
|--------|----------------|-----------|
| SICD | sarkit (primary), sarpy (fallback) | sarkit is the modern NGA library; sarpy fallback for compatibility |
| CPHD | sarkit (primary), sarpy (fallback) | sarkit is the modern NGA library; sarpy fallback for compatibility |
| CRSD | sarkit (only) | sarpy does not fully support CRSD |
| SIDD | sarkit (only) | sarpy does not fully support SIDD |
| BIOMASS | rasterio | Magnitude/phase GeoTIFFs + XML annotation |
| Sentinel-1 SLC | rasterio | SAFE archive with TIFF measurements + XML annotation |
| TerraSAR-X / TanDEM-X | rasterio | CoSSC GeoTIFFs + XML annotation |
| NISAR | h5py | HDF5 with SLC/RSLC datasets |
| GeoTIFF | rasterio | Industry standard for GeoTIFF, handles COGs |
| NITF | rasterio (GDAL) | Generic NITF via GDAL driver; SAR NITF uses sarkit/sarpy |

### Backend Fallback Pattern (`_backend.py`)

SAR readers use a shared backend detection module:
- `_HAS_SARKIT` / `_HAS_SARPY` flags set at import time
- `require_sar_backend()` returns `'sarkit'` or `'sarpy'` (prefers sarkit)
- `require_sarkit()` raises `DependencyError` when sarkit is required (CRSD, SIDD)
- Key API difference: sarkit takes `BinaryIO`, sarpy takes filepath strings

### SICDReader

**Format Background:**
- SICD = Sensor Independent Complex Data (NGA standard)
- Complex-valued SAR imagery (I+jQ channels)
- Typically in NITF container format
- Rich metadata including collection geometry, sensor params

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

### CPHDReader

**Format Background:**
- CPHD = Compensated Phase History Data (NGA standard)
- Raw phase history (unfocused SAR)
- Multi-channel support (multiple apertures/polarizations)

Unlike SICD (formed imagery), CPHD doesn't have fixed rows/cols. Instead:
- Multiple channels, each with vectors x samples
- `get_shape()` returns dimensions of first channel
- `read_chip()` accepts channel index in `bands` parameter

**Key Decisions:**

1. **Channel-based reading**: `bands` parameter selects channel
   - CPHD is fundamentally multi-channel
   - Different from multi-band imagery (different concept)
   - Consistent with ImageReader API despite different semantics

2. **No automatic focusing**: Returns raw phase history
   - Focusing algorithms are use-case specific
   - Belongs in processing module, not I/O

### GeoTIFFReader

**Format Background:**
- Foundational raster format for geospatial imagery
- Covers SAR GRD products, EO imagery, MSI, Cloud-Optimized GeoTIFFs (COG)
- Lives at IO base level (`IO/geotiff.py`) -- not in a modality submodule

**Key Decisions:**

1. **Real-valued data**: Returns magnitude, not complex
   - GRD products are already detected (|I+jQ|^2)
   - Standard EO/MSI imagery is real-valued

2. **Geolocation includes CRS**: Unlike SICD
   - GeoTIFF is geocoded with embedded CRS and affine transform
   - Supports any projection (UTM, Geographic, etc.)

3. **Band indexing**: Users provide 0-based indices; internally converts to rasterio's 1-based indexing

### open_sar() Auto-Detection

Convenience function that tries readers in order:
1. SICDReader (NITF containers)
2. CPHDReader
3. CRSDReader (sarkit-only)
4. SIDDReader
5. Sentinel1SLCReader (SAFE directories)
6. TerraSARReader (TSX/TDX directories)
7. NISARReader (HDF5)
8. GeoTIFFReader (fallback for GRD products)

**Trade-offs:**
- **Pro**: User-friendly for unknown files; enables format-agnostic batch processing
- **Con**: Slower than direct reader (multiple open attempts); may misidentify ambiguous formats

Best for interactive exploration and prototyping. Use specific readers in performance-critical pipelines.

## Metadata Architecture

### Package Structure (`models/`)

Metadata is organized as a Python package at `grdl/IO/models/`:

| Module | Contents |
|--------|----------|
| `base.py` | `ImageMetadata` -- base dataclass with `format`, `rows`, `cols`, `dtype`, `bands`, `crs`; dict-like access |
| `common.py` | Shared primitive types: `XYZ`, `LatLon`, `LatLonHAE`, `RowCol`, `Poly1D`, `Poly2D`, `XYZPoly` |
| `sicd.py` | `SICDMetadata` + ~35 nested dataclasses covering all 17 SICD sections |
| `sidd.py` | `SIDDMetadata` + ~25 nested dataclasses covering all 13 SIDD sections |
| `cphd.py` | `CPHDMetadata` + channel, PVP, waveform, antenna, scene coordinate dataclasses |
| `biomass.py` | `BIOMASSMetadata` -- flat typed fields for BIOMASS annotation XML |
| `viirs.py` | `VIIRSMetadata` -- flat typed fields for VIIRS HDF5 attributes |
| `aster.py` | `ASTERMetadata` -- flat typed fields for ASTER GeoTIFF + XML |
| `sentinel2.py` | `Sentinel2Metadata` -- flat typed fields for Sentinel-2 JP2 products |
| `sentinel1_slc.py` | `Sentinel1SLCMetadata` + burst, orbit, Doppler, calibration/noise dataclasses |
| `terrasar.py` | `TerraSARMetadata` + product, scene, radar, orbit, calibration dataclasses |
| `nisar.py` | `NISARMetadata` + identification, orbit, attitude, swath, geolocation dataclasses |
| `__init__.py` | Re-exports everything; preserves `from grdl.IO.models import ...` paths |

### Design Decisions

1. **Dataclasses over dicts**: IDE autocomplete, type checking, and self-documenting field names with no runtime overhead.

2. **Nested composition**: Complex formats (SICD, SIDD) use deeply nested dataclasses mirroring the source specification structure. Example: `meta.geo_data.scp.llh.lat` mirrors the SICD XML path `GeoData/SCP/LLH/Lat`.

3. **Optional sections**: All format-specific nested sections default to `None`, so metadata is always constructible even from incomplete source files.

4. **Backward-compatible dict access**: `ImageMetadata.__getitem__` and `__contains__` use `dataclasses.fields()` introspection, so `meta['format']` continues to work alongside `meta.format`.

5. **Shared primitives**: Types like `XYZ`, `LatLonHAE`, `Poly2D` are reused across SICD, SIDD, and CPHD.

## Mission-Specific Readers

### BIOMASS Implementation

**Format Background:**
- BIOMASS is ESA's P-band SAR satellite mission (435 MHz)
- L1 SCS products store complex data as magnitude + phase GeoTIFFs (not complex-valued TIFFs)
- SAFE-like directory structure with XML annotation

**Key Decisions:**

1. **Mission-Specific Module**: Created `biomass.py` instead of adding to generic `sar.py`
   - BIOMASS has mission-specific processing levels (L1a/b/c, L2a/b, L3)
   - Sets precedent adopted by Sentinel-1, TerraSAR-X, and NISAR readers

2. **Complex Data Reconstruction**: `complex = magnitude * exp(1j * phase)`, handled transparently in `read_chip()` and `read_full()`

3. **Multi-Polarization**: All 4 polarizations in same TIFF files (HH/HV/VH/VV as bands 1-4); `bands` parameter selects polarizations (0-based)

4. **Geolocation**: Slant range geometry with GCPs (not geocoded like GRD products)

### BIOMASS Catalog System

`BIOMASSCatalog(CatalogInterface)` provides:

1. **Local Discovery**: File system scan with pattern matching, SQLite indexing
2. **ESA MAAP STAC API**: POST-based search with CQL2 filtering (product type, orbit, bbox, date)
3. **OAuth2 Authentication**: Offline token from `~/.config/geoint/credentials.json` or `ESA_MAAP_OFFLINE_TOKEN` env var
4. **Download Management**: Streaming download with progress, automatic ZIP extraction, database tracking
5. **SQLite Database**: Unified local/remote product tracking with indexed query fields

Design chose mission-specific catalog over generic `ImageCatalog` because BIOMASS has unique naming conventions, API endpoints, and metadata schema. SQLite chosen for zero-config portability. Credentials follow XDG convention and are repo-agnostic.

## EO / IR / Multispectral Submodules

Following the `sar/` pattern, modality submodules wrap base-level format readers
(GeoTIFFReader, HDF5Reader, JP2Reader) with sensor-specific metadata extraction:

| Submodule | Modality | Implemented Sensors | Backend |
|-----------|----------|---------------------|---------|
| `eo/` | Electro-Optical (VIS/NIR) | Sentinel-2 (L1C, L2A) | glymur, rasterio |
| `ir/` | Thermal / Infrared | ASTER (L1T, GDEM) | rasterio |
| `multispectral/` | Multispectral / Hyperspectral | VIIRS (VNP46A1, VNP13A1) | h5py |

### Sentinel2Reader (`eo/sentinel2.py`)

- Wraps `JP2Reader` for pixel access on JPEG2000 band files
- Supports Level-1C (TOA Reflectance) and Level-2A (Surface Reflectance) products
- Handles both standalone JP2 files and SAFE archive directory structure
- Detects Sentinel-2 products from filename patterns (T*_*_B*.jp2, S2A/S2B/S2C prefixes)
- Populates `Sentinel2Metadata` dataclass

### ASTERReader (`ir/aster.py`)

- Wraps `rasterio` directly for pixel reads
- Extracts ASTER-specific metadata from GeoTIFF tags and companion XML
- Detects product type (L1T vs GDEM) and band availability (VNIR/SWIR/TIR) from filename
- Populates `ASTERMetadata` dataclass with acquisition, orbital, solar geometry fields

### VIIRSReader (`multispectral/viirs.py`)

- Wraps `h5py` directly for pixel reads
- Extracts VIIRS-specific metadata from HDF5 file-level and dataset-level attributes
- Auto-detects first suitable 2D+ numeric dataset or accepts explicit path
- Populates `VIIRSMetadata` dataclass with satellite, temporal, spatial, and calibration fields

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
all GRDL-specific errors, or catch the built-in base class (`ValueError`, `ImportError`,
etc.) for backward-compatible handling.

### Graceful Degradation

Readers check for dependencies and fail fast:

```python
from grdl.exceptions import DependencyError

if not SARPY_AVAILABLE:
    raise DependencyError("sarpy required. Install: pip install sarpy")
```

- Fails at construction time, not deep in execution
- Clear error message with installation command
- Allows partial installation (e.g., only rasterio, no sarpy)

### Validation Strategy

- **File Existence**: Checked in `ImageReader.__init__` -- fails fast before format-specific parsing
- **Index Bounds**: Checked in `read_chip()` -- prevents backend crashes with cryptic messages
- **Format Validation**: Delegated to backend libraries (sarkit, sarpy, rasterio) -- trust authoritative implementations

## Memory Management

### Lazy Loading Strategy

```python
reader = SICDReader('large_image.nitf')  # Constructor: metadata only (~KB)
chip = reader.read_chip(0, 1024, 0, 1024)  # ~8MB for 1024x1024 complex
full = reader.read_full()  # Could be GB - use with caution
```

### Complex Data Memory Footprint

| Data Type | Bytes/Pixel | 1Kx1K | 10Kx10K | 50Kx50K |
|-----------|-------------|-------|---------|---------|
| complex64 | 8 | 8 MB | 800 MB | 20 GB |
| float32 (magnitude) | 4 | 4 MB | 400 MB | 10 GB |
| uint8 (quantized) | 1 | 1 MB | 100 MB | 2.5 GB |

Convert to magnitude early if phase is not needed.

### Chunked Processing

Use `ChipExtractor` or `Tiler` from `grdl.data_prep` for chip/tile planning instead of hand-rolling index arithmetic:

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

## Future Extensions

### Planned Readers

- **Landsat OLI** -- EO GeoTIFF with band-specific metadata
- **HLS** (Harmonized Landsat Sentinel) -- Cloud-optimized GeoTIFFs
- **MODIS** -- HDF-EOS2 multispectral products
- **EMIT / PRISMA / EnMAP** -- Hyperspectral HDF5 / NetCDF products

### Cloud Support

- COG (Cloud-Optimized GeoTIFF) via rasterio (partially supported via GeoTIFFReader)
- Zarr for chunked array storage
- S3/GCS readers with fsspec backend

### Catalog Extension

- Generic `ImageCatalog` supporting multiple missions and formats
- PostGIS backend for large-scale deployments
- Spatial indexing (R-tree, quad-tree)

## API Stability

- Breaking changes to `base.py` increment major version
- New readers/features increment minor version
- Bug fixes increment patch version
- Abstract base classes are the stability contract; format-specific extensions may change more freely

## Related Documentation

- [README.md](README.md) - User-facing documentation and examples
- [TODO.md](TODO.md) - Planned features and roadmap
- [/CLAUDE.md](/CLAUDE.md) - Project-wide coding standards
- [/LIBRARIES.md](/LIBRARIES.md) - Integration with other GRDL modules
