# IO Module Architecture

Technical design documentation for the IO module.

Modified: 2026-06-16

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
│   ├── __init__.py          # Public API + reader/writer factories: get_reader(),
│   │                        #   open_reader(), get_writer(), write(),
│   │                        #   register_reader(), register_writer(),
│   │                        #   list_reader_formats(), open_image() (deprecated)
│   ├── base.py              # Abstract base classes (ABCs)
│   ├── performance.py       # ReadConfig + parallel read helpers (chunked/band)
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
│   │   ├── nisar.py         # NISARMetadata + identification/orbit/swath dataclasses
│   │   ├── eo_nitf.py       # EONITFMetadata + RPCCoefficients + RSMCoefficients
│   │   └── stanag4607.py    # STANAG4607Metadata + dwell/target dataclasses
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
│   │   ├── sentinel2.py     # Sentinel2Reader (wraps JP2Reader)
│   │   ├── nitf.py          # EONITFReader (multi-segment unification, RPC/RSM,
│   │   │                    #   decimation, read_mask/get_lut/normalize_abpp)
│   │   ├── nitf_writer.py   # write_chip() (geolocation-preserving NITF chip-out)
│   │   ├── _tre_xml.py      # xml:TRE / xml:DES parser
│   │   ├── _tre_rsm_error.py # RSM error TRE parsers + summarize_accuracy()
│   │   ├── _tre_band.py     # BANDSB / BANDSA parsers
│   │   └── _tre_airborne.py # SENSRB / MENSRB / MENSRA / ACFTB parsers
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
│   │   ├── sicd.py          # SICDReader (sarkit primary, sarpy fallback) — _enforce_2d=True
│   │   ├── sicd_writer.py   # SICDWriter
│   │   ├── sicd_collection.py # SICDCollectionReader + open_sicd_collection()
│   │   ├── cphd.py          # CPHDReader (sarkit primary, sarpy fallback) — _enforce_2d=True
│   │   ├── crsd.py          # CRSDReader (sarkit-only) — _enforce_2d=True
│   │   ├── sidd.py          # SIDDReader (sarkit-only)
│   │   ├── sidd_writer.py   # SIDDWriter
│   │   ├── biomass.py       # BIOMASSL1Reader (rasterio) + open_biomass()
│   │   ├── sentinel1_slc.py # Sentinel1SLCReader (rasterio)
│   │   ├── sentinel1_l0/    # Sentinel1L0Reader (raw L0), open_safe_product,
│   │   │                    #   ReaderConfig, CRSD conversion + verification
│   │   ├── terrasar.py      # TerraSARReader + open_terrasar()
│   │   └── nisar.py         # NISARReader + open_nisar()
│   ├── gmti/                # STANAG 4607 GMTI
│   │   ├── __init__.py      # GMTI exports + open_gmti()
│   │   ├── stanag4607.py    # STANAG4607Reader (editions 2/3/4)
│   │   ├── stanag4607_writer.py # STANAG4607Writer
│   │   ├── helpers.py       # dwell_footprint_polygon, ground_relative_velocity,
│   │   │                    #   filter_target_reports, summarize
│   │   └── cphd_steering.py # build_steering_matrix_from_cphd_metadata
│   ├── catalog/             # Remote query, download & SQLite cataloging
│   │   ├── __init__.py      # Catalog exports + credential helpers
│   │   ├── remote_utils.py  # load_credentials, get_cdse_token,
│   │   │                    #   get_earthdata_token, download_file
│   │   ├── biomass_catalog.py     # BIOMASSCatalog (ESA MAAP STAC)
│   │   ├── sentinel1_catalog.py   # Sentinel1SLCCatalog (CDSE OData)
│   │   ├── sentinel2_catalog.py   # Sentinel2Catalog (CDSE OData)
│   │   ├── nisar_catalog.py       # NISARCatalog (NASA Earthdata)
│   │   ├── terrasar_catalog.py    # TerraSARCatalog (local)
│   │   ├── aster_catalog.py       # ASTERCatalog (local)
│   │   └── viirs_catalog.py       # VIIRSCatalog (local)
│   ├── README.md            # User documentation
│   ├── ARCHITECTURE.md      # This file
│   ├── PATTERNS.md          # Recurring implementation patterns
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
| `read_chip()` | Abstract - read spatial subset | Default - memory efficient |
| `get_shape()` | Abstract - image dimensions | Planning memory allocation |
| `get_dtype()` | Abstract - pixel data type | Buffer allocation |
| `read_full()` | Concrete - delegates to `read_chip` over full extent | Small images or full processing |
| `read_band(index)` | Concrete - single band, always 2-D | Per-band processing regardless of axis order |
| `close()` | Concrete (no-op default) | Override when handles need cleanup |

Static shape-contract helpers callable from concrete readers: `_ensure_2d(arr)`
(squeeze singleton band axis), `_assert_2d(data, context, strict)` (strict
single-channel validation; see §"Strict 2-D Shape Policy"), and
`_validate_single_pol(arr, context)` (raise unless strictly 2-D).

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
| EO NITF | `EONITFMetadata(ImageMetadata)` | RPC/RSM coefficients, sensor ID, target geometry |
| Sentinel-1 L0 | `Sentinel1L0Metadata(ImageMetadata)` | Burst index, swath/polarization tables, orbit/attitude |
| STANAG 4607 (GMTI) | `STANAG4607Metadata` | Edition + dwell/target dataclasses (not an `ImageReader`) |

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

**Implementations:** `GeoTIFFWriter`, `HDF5Writer`, `NITFWriter`, `SICDWriter`, `SIDDWriter`, `PngWriter`, `NumpyWriter`, plus the EO NITF chip-out `write_chip()` and `STANAG4607Writer`. Registry-based factory access (`get_writer(format, path)` / convenience `write(data, path)`) covers `geotiff`, `numpy`, `png`, `hdf5`, `nitf`; `SICDWriter`, `SIDDWriter`, and `STANAG4607Writer` are used directly because they take typed metadata containers rather than the `write(ndarray)` contract.

**Reader factory:** `get_reader(format, path)` mirrors `get_writer` for the read side — same registry pattern, lazy imports, and `ImportError` with install guidance. `register_reader()` / `register_writer()` extend the registries at runtime; `list_reader_formats()` enumerates all registered reader keys.

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

**Implemented**: `BIOMASSCatalog`, `Sentinel1SLCCatalog`, `Sentinel2Catalog`, `NISARCatalog`, `TerraSARCatalog`, `ASTERCatalog`, `VIIRSCatalog` — all inherit from `CatalogInterface` with local discovery, SQLite tracking, and (where applicable) remote search and download.

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
| SIDD | sarkit (primary), sarpy (fallback) | sarkit preferred; sarpy fallback for compatibility |
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

### Reader / Writer Factory

The headline IO feature is a symmetric, registry-backed factory for both
readers and writers. Entry points for opening imagery:

| Function | When to use |
|----------|-------------|
| `get_reader(format, path)` | You know the format. Fast, unambiguous, best for pipelines. |
| `open_reader(path)` | Unknown or mixed-format file set. Auto-detects via extension map then `open_any()`. **Primary entry point.** |
| `open_any(path)` | Ambiguous files with uninformative extensions — runs the full cascade directly. |
| `open_sar(path)` | SAR-only auto-detection (modality-specific cascade). |
| `open_image(path)` | **Deprecated** alias for `open_reader()`; emits `DeprecationWarning`. |

**Registry design.** `get_reader` and `get_writer` are symmetric: both use a
module-level registry (`_READER_REGISTRY` with 20 entries / `_WRITER_REGISTRY`
with 5 entries) mapping a lowercase format key to a `(module_path, class_name)`
tuple. `importlib.import_module` loads the module lazily on first use, so an
optional dependency (`sarpy`, `rasterio`, `h5py`, ...) is only imported when its
reader is actually requested — `import grdl.IO` itself never pulls them in.

- Reader keys are normalized via `_normalize_reader_key()` (lowercase, `_`→`-`),
  so `'sentinel1-slc'` and `'sentinel1_slc'` are equivalent.
- `get_reader` raises `ValueError` for an unknown key and `ImportError` (naming
  the missing package, referencing `requirements-optional.txt`) when the
  optional dependency is absent.
- `list_reader_formats()` returns all registered keys.
- `register_reader(format, module_path, class_name, overwrite=False)` and
  `register_writer(...)` extend the registries at runtime (e.g. third-party
  readers); `overwrite=False` guards against clobbering an existing key.

The 20 registered reader keys: `geotiff`, `nitf`, `hdf5`, `jpeg2000`, `sicd`,
`cphd`, `crsd`, `sidd`, `biomass`, `sentinel1-slc`, `sentinel1-l0`, `terrasar`,
`nisar`, `sentinel2`, `eo-nitf`, `aster`, `viirs`, `stanag4607`, `gdal`,
`probe`. Writer keys: `geotiff`, `numpy`, `png`, `hdf5`, `nitf`.

**`open_reader()` cascade.** It extends the writer-side factory pattern to the
reader side:
1. Looks up the file extension in `_READER_EXTENSION_MAP` → calls `get_reader(fmt, path)`
2. If an `ImportError` fires (missing library), saves the message and falls through
3. If a `ValueError` fires (format mismatch from a specialized reader), falls through
4. Delegates to `open_any()` for NITF sniffing, modality cascades, GDAL fallback, and invasive probing
5. Emits a `UserWarning` naming the missing package when a fallback reader is used (richer metadata / stricter validation would have been available with the package installed)

### open_sar() Auto-Detection

Convenience function that tries readers in order:
1. SICDReader (NITF containers)
2. CPHDReader
3. CRSDReader (sarkit-only)
4. SIDDReader
5. Sentinel-1 SAFE directories — dispatches by product identifier in the
   directory name: `Sentinel1L0Reader` when the name contains `RAW`, otherwise
   `Sentinel1SLCReader`
6. TerraSARReader (TSX1_/TDX1_ directories or any dir with a TSX/TDX annotation XML)
7. NISARReader (`.h5` / `.hdf5`)
8. GeoTIFFReader (`.tif` / `.tiff` fallback for GRD products)

(`SICDCollectionReader` is opened explicitly via `open_sicd_collection()`, not
through this cascade.)

**Trade-offs:**
- **Pro**: User-friendly for unknown files; enables format-agnostic batch processing
- **Con**: Slower than direct reader (multiple open attempts); may misidentify ambiguous formats

Best for interactive exploration and prototyping. Use `get_reader(format, path)` in performance-critical pipelines.

### Strict 2-D Shape Policy (`_enforce_2d`)

Single-channel SAR readers (SICD, CPHD, CRSD) set the `_enforce_2d = True` class attribute. `ImageReader.read_chip()` and `read_full()` wrap their return arrays in `_assert_2d(data, context, strict)`:

- Returns the array unchanged if already `(rows, cols)`
- Raises `ValueError` in strict mode if a singleton band axis is present (e.g., `(1, R, C)` from a backend version change) — catches reader implementation defects immediately at read time rather than silently propagating bad shapes into processing code
- Non-strict (default on all other readers): silently squeezes `(1, R, C)` → `(R, C)`

To apply the policy to a new single-channel reader, set `_enforce_2d = True` on the class and call `self._assert_2d(data, context=f'{type(self).__name__}.read_chip', strict=self._enforce_2d)` before returning.

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
| `eo_nitf.py` | `EONITFMetadata` -- RPC/RSM coefficients + full commercial TRE dataclasses |
| `stanag4607.py` | `STANAG4607Metadata` + dwell/target-report dataclasses (GMTI) |
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

### Catalog Family & Credential Handling

The same `CatalogInterface` ABC is implemented by seven catalogs:
`BIOMASSCatalog` (ESA MAAP STAC), `Sentinel1SLCCatalog` and `Sentinel2Catalog`
(CDSE OData), `NISARCatalog` (NASA Earthdata), and `TerraSARCatalog`,
`ASTERCatalog`, `VIIRSCatalog` (local discovery only). Remote credential and
download logic is consolidated in `catalog/remote_utils.py`:

- `load_credentials(provider)` reads `~/.config/geoint/credentials.json` with
  env-var fallbacks (`COPERNICUS_*`, `EARTHDATA_*`, `ESA_MAAP_OFFLINE_TOKEN`).
- `get_cdse_token()` / `get_earthdata_token()` perform the provider OAuth2 flow.
- `download_file()` is a shared streaming downloader with resume support.
- SQLite catalog databases default to `~/.config/geoint/catalogs/`.

Centralizing auth in `remote_utils` keeps each catalog focused on its
provider-specific query schema and naming conventions.

## EO NITF Reader (`eo/nitf.py`)

`EONITFReader` is the most capable reader in the module and warrants its own
design notes.

**Multi-segment unification.** Commercial NITFs frequently store one logical
image across many image segments, plus overviews and masks. The reader groups
heterogeneous segments and auto-selects a *primary group*; `read_chip()` routes
a full-image bbox request across whichever primary segments overlap it, filling
inter-segment gaps with `nodata` (else 0). Overviews/masks never break loading
and are reachable by reopening with `image_index=N` to pin a single segment.

**TRE suite.** Image subheaders and TRE_OVERFLOW DES are scanned for the full
commercial TRE set — geolocation (RPC00B, RSMPCA/RSMIDA/RSMGGA), RSM error
model (RSMPIA, RSMDCA/B, RSMECA/B, RSMAPA/B), exploitation (CSEXRA, CSCRNA,
CSEPHA, USE00A, ICHIPB, BLOCKA, AIMIDB, STDIDC, PIAIMC), band characterization
(BANDSB/BANDSA → `band_names`/`wavelengths`), and airborne (SENSRB, MENSRB,
MENSRA, ACFTB). Parsers live in `_tre_*.py` modules. Multi-section RSMPCA is
collected into an `RSMSegmentGrid` (see PATTERNS §6).

**Pixel domain.** `read_chip(decimation=)` exploits embedded overviews and
JPEG2000 reduced-resolution levels (GDAL `out_shape`) rather than reading at
full resolution then slicing. `read_mask()`, `get_lut()`, and `normalize_abpp()`
(stretch to actual bits-per-pixel) round out the pixel API. Remote `https://`,
`s3://`, and `/vsi*` URIs are passed straight to GDAL.

**Chip-out writer.** `eo/nitf_writer.write_chip()` writes a new NITF that
geolocates identically to the parent: ICHIPB (composed with the parent's own
ICHIPB when present) plus RPC00B and serialized RSMIDA/RSMPCA. Multi-section RSM
grids collapse to the section covering the chip center (GDAL cannot repeat a TRE
name).

## GMTI Reader (`gmti/`)

`STANAG4607Reader` parses STANAG 4607 packets into typed segment dataclasses
(editions 2, 3, and 4 auto-detected from the 2-char ASCII version field). It is
a pure-Python (`struct`) reader and does *not* inherit `ImageReader` — GMTI is a
moving-target report stream, not raster imagery.

- Iteration: `iter_packets()`, `iter_dwells()`, `iter_target_reports()` yield
  `(dwell, target)` tuples.
- `to_detection_set(confidence_field='gmti.snr_db', snr_normalization=40.0)`
  bridges target reports into the GRDL detection ecosystem: each target becomes
  a `Detection` with a WGS84 `shapely.Point`, `gmti.*` properties, and a
  confidence derived from SNR (dB) / `snr_normalization`, clamped to `[0, 1]`.
- Geometry/analysis helpers (`helpers.py`): `dwell_footprint_polygon`,
  `ground_relative_velocity`, `filter_target_reports`, `summarize`.
- `cphd_steering.build_steering_matrix_from_cphd_metadata()` derives a scene-
  projected per-channel steering matrix from a CPHD `Antenna` section (returns
  `CPHDMetadataSteering`), for metadata-driven STAP detection.

`STANAG4607Writer` is deliberately kept out of `_WRITER_REGISTRY`: it serializes
typed segment dataclasses, not the `write(ndarray)` contract that `get_writer`
assumes.

## Sentinel-1 Level 0 Reader (`sar/sentinel1_l0/`)

`Sentinel1L0Reader` reads raw (unfocused) Sentinel-1 L0 SAFE products. Unlike
the focused readers, its native API is burst-centric: `read_burst()`,
`iter_bursts()`, `read_swath()` (with a `read_chip()` shim over the currently
selected burst). `ReaderConfig` tunes SAFE validation, annotation parsing, POE
orbit loading, and burst-boundary detection. `open_safe_product()` is the
convenience factory.

The FDBAQ decoder backend (`sentinel1decoder>=2.0`) is an optional dependency
behind the `grdl[s1_l0]` extra. Optional CRSD conversion
(`Sentinel1L0ToCRSD`, `convert_s1_l0_to_crsd`) and verification
(`verify_crsd_split_gates`) are guarded by a `try/except ImportError` in the
subpackage `__init__` that swaps in stub callables raising a clear `ImportError`
when the dependency is absent.

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
