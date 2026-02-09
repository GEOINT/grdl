# IO Module TODO

Roadmap and planned features for the IO module.

## Current Status

### ✅ Completed

- [x] Base class architecture (`base.py`)
  - [x] `ImageReader` ABC with lazy loading
  - [x] `ImageWriter` ABC with incremental writes
  - [x] `CatalogInterface` ABC for discovery
- [x] SAR readers (`sar.py`)
  - [x] `SICDReader` - SICD format via SARPY
  - [x] `CPHDReader` - CPHD format via SARPY
  - [x] `GRDReader` - GRD GeoTIFF via rasterio
  - [x] `open_sar()` - Auto-detection utility
- [x] BIOMASS readers (`biomass.py`)
  - [x] `BIOMASSL1Reader` - BIOMASS L1 SCS format (magnitude/phase GeoTIFFs)
  - [x] `open_biomass()` - Auto-detection utility
  - [x] Full quad-pol support (HH, HV, VH, VV)
  - [x] XML annotation metadata parsing
  - [x] Complex data reconstruction from magnitude/phase
- [x] Catalog system (`catalog.py`)
  - [x] `BIOMASSCatalog` - BIOMASS-specific catalog and download manager
  - [x] Local file system discovery
  - [x] ESA MAAP STAC API search (`query_esa()`)
  - [x] CQL2 filtering (product type, orbit, bbox, date)
  - [x] OAuth2 authentication (offline token exchange)
  - [x] Streaming product download with ZIP extraction
  - [x] SQLite database for product tracking
  - [x] Metadata extraction and indexing
  - [x] Spatial overlap detection
  - [x] `load_credentials()` utility (repo-agnostic `~/.config/geoint/`)
  - [x] Operational collection support (`BiomassLevel1a/1b/2a`)
- [x] Geolocation integration
  - [x] GCP-based coordinate transforms via `grdl.geolocation`
  - [x] Vectorized batch transforms
  - [x] `Geolocation.from_reader()` factory
- [x] Image processing module (`image_processing/`)
  - [x] `ImageProcessor` / `ImageTransform` ABCs
  - [x] `@processor_version()` decorator and versioning system
  - [x] `TunableParameterSpec` for runtime-adjustable parameters
  - [x] `DetectionInputSpec` for detection input chaining
  - [x] `Orthorectifier` / `OutputGrid` - slant-to-ground projection
  - [x] `PauliDecomposition` - quad-pol Pauli basis decomposition
  - [x] `ImageDetector` ABC with geo-registration helpers
  - [x] `Detection`, `DetectionSet`, `Geometry`, `OutputSchema`, `OutputField` data models
  - [x] GeoJSON export and optional DataFrame conversion
- [x] ImageJ/Fiji algorithm ports (`imagej/`)
  - [x] 12 classic algorithms ported from ImageJ/Fiji (all inherit `ImageTransform`)
  - [x] Spatial filters: RollingBallBackground, UnsharpMask, RankFilters, MorphologicalFilter
  - [x] Contrast & enhancement: CLAHE (vectorized), GammaCorrection
  - [x] Thresholding & segmentation: AutoLocalThreshold, StatisticalRegionMerging (vectorized edges)
  - [x] Edge & feature detection: EdgeDetector, FindMaxima
  - [x] Frequency domain: FFTBandpassFilter
  - [x] Stack operations: ZProjection
  - [x] Full attribution to original ImageJ/Fiji authors
  - [x] Version strings mirroring original source versions
  - [x] `@processor_tags` metadata on all 12 components (modality, category)
  - [x] `__gpu_compatible__` flag on all 12 components
  - [x] Tests covering all 12 components
- [x] Data preparation module (`data_prep/`)
  - [x] ChipBase ABC -- image dimension management and coordinate snapping
  - [x] ChipExtractor -- point-centered and whole-image chip region computation
  - [x] Tiler -- stride-based tile region computation
  - [x] Normalizer -- minmax, zscore, percentile, unit_norm with fit/transform
- [x] Coregistration module (`coregistration/`)
  - [x] Affine transform alignment
  - [x] Projective transform alignment
  - [x] Feature-based matching (OpenCV)
- [x] Pipeline composition (`image_processing/pipeline.py`)
  - [x] Sequential transform chaining with progress callback rescaling
- [x] BandwiseTransformMixin for automatic 3D stack support
- [x] Custom exception hierarchy (`grdl/exceptions.py`)
  - [x] GrdlError, ValidationError, ProcessorError, DependencyError, GeolocationError
- [x] PEP 561 type marker (`grdl/py.typed`)
- [x] `pyproject.toml` with setuptools backend and optional dependency extras
- [x] Shared test fixtures (`tests/conftest.py`)
- [x] Performance benchmarks (`tests/test_benchmarks.py`, pytest-benchmark)
- [x] Documentation
  - [x] README.md with examples (including BIOMASS and catalog)
  - [x] ARCHITECTURE.md with design decisions
  - [x] This TODO.md
- [x] Example scripts (`example/`)
  - [x] `catalog/discover_and_download.py` - MAAP catalog search and download
  - [x] `catalog/view_product.py` - Pauli RGB and HH dB viewer with interactive markers
  - [x] `ortho/ortho_biomass.py` - Orthorectification with Pauli RGB
- [x] Ground truth data (`ground_truth/`)
  - [x] `biomass_calibration_targets.geojson` - BIOMASS cal/val sites
- [x] Testing
  - [x] Test suite structure (`tests/`)
  - [x] BIOMASS L1 reader tests
  - [x] Geolocation tests with interactive marker plotting
  - [x] Orthorectification tests
  - [x] Pauli decomposition tests
  - [x] Detection model and geo-registration tests
  - [x] Processor versioning tests
  - [x] Tunable parameter tests
  - [x] ImageJ/Fiji ports tests (124 tests across 12 components)

## High Priority

### SAR Readers (sar.py)

- [ ] **CRSDReader** - CRSD format support
  - Use SARPY's CRSD reader
  - Similar structure to CPHDReader
  - Test with CRSD 1.0 sample files

- [ ] **SLCReader** - Single Look Complex
  - Distinguish from GRD (also GeoTIFF)
  - Parse SAR-specific metadata tags
  - Handle complex data in GeoTIFF

- [ ] **SIDD Support** - Sensor Independent Derived Data
  - Detected/processed SAR imagery
  - Use SARPY's SIDD reader
  - Coordinate with SICDReader (similar structure)

### EO Readers (eo.py) - NEW MODULE

- [ ] **GeoTIFFReader** - General raster imagery
  - Reuse GRDReader implementation (also rasterio)
  - Handle RGB, multispectral, hyperspectral
  - Support for COG (Cloud-Optimized GeoTIFF)

- [ ] **NITFReader** - NITF 2.1 imagery
  - Use SARPY or rasterio for NITF parsing
  - Handle TREs (Tagged Record Extensions)
  - Support multi-image NITF files

- [ ] **JP2Reader** - JPEG2000 imagery
  - Via rasterio or glymur
  - Common in satellite imagery (Sentinel-2)
  - Handle tiled/pyramidal structure

- [ ] **HDF5Reader** - HDF5/NetCDF formats
  - Use h5py or xarray
  - Common in climate/weather data
  - Handle multi-dimensional arrays

### Geospatial Readers (geospatial.py) - NEW MODULE

- [ ] **GeoJSONReader** - GeoJSON vector data
  - Use geopandas or fiona
  - Return as GeoDataFrame or dict
  - Handle large files (streaming)

- [ ] **GeoJSONWriter** - Write GeoJSON
  - From GeoDataFrame or dict
  - Pretty-print option for readability
  - Coordinate precision control

- [ ] **ShapefileReader** - ESRI Shapefile
  - Via geopandas/fiona
  - Handle .shp, .shx, .dbf bundle
  - Support for .prj (projection info)

- [ ] **ShapefileWriter** - Write shapefiles
  - From GeoDataFrame
  - Automatic .prj generation
  - Handle attribute types

- [ ] **KMLReader** - Google Earth KML/KMZ
  - Parse placemark, polygon, linestring
  - Extract style information
  - Handle KMZ (zipped KML)

### Catalog (catalog.py)

- [x] **BIOMASSCatalog** - BIOMASS data management (COMPLETED)
  - Local discovery and indexing
  - ESA MAAP STAC API search with CQL2 filtering
  - OAuth2 authentication (offline token exchange)
  - Streaming download with ZIP extraction and progress reporting
  - SQLite database tracking
  - `load_credentials()` utility with env var fallback

- [ ] **Generic ImageCatalog** - Multi-format catalog
  - Support SAR, EO, and other formats
  - Unified database schema
  - Format-agnostic queries

- [ ] **Parallel Metadata Extraction**
  - Thread/process pool for batch operations
  - Progress tracking for large collections
  - Query by sensor, date, location

- [ ] **SpatialCatalog** - Geospatial queries
  - Build spatial index (R-tree)
  - Find overlapping images
  - Query by bounding box or polygon

## Medium Priority

### Writers (sar.py, eo.py)

- [ ] **GeoTIFFWriter** - Generic raster output
  - Via rasterio
  - COG output option
  - Compression options (LZW, DEFLATE, etc.)

- [ ] **SICDWriter** - SICD format output
  - Via SARPY
  - Preserve metadata in conversions
  - Handle complex data properly

- [ ] **NITFWriter** - NITF 2.1 output
  - Via SARPY
  - Custom TRE injection
  - Multi-image NITF support

### Advanced Features

- [ ] **Cloud Storage Support**
  - S3/GCS readers via fsspec
  - Streaming large files from cloud
  - Credential management

- [ ] **Lazy Metadata Loading**
  - Option to defer metadata parsing
  - Faster instantiation for batch operations
  - Trade-off: metadata access may raise exceptions

- [ ] **Parallel Chip Reading**
  - Thread pool for concurrent chip reads
  - Useful for tiled processing
  - Configurable number of workers

- [ ] **Metadata Caching**
  - Sidecar .json files with pre-parsed metadata
  - Dramatically faster re-opening
  - Automatic invalidation on file change

- [ ] **Format Conversion Utilities**
  - SICD → GeoTIFF converter
  - GRD → COG converter
  - Batch conversion scripts

## Low Priority / Research

### Experimental Formats

- [ ] **SAFE Format** - Sentinel-1/2 native
  - Parse manifest.safe
  - Handle multi-file structure
  - Extract orbit/calibration data

- [ ] **ENVI Format** - Hyperspectral imagery
  - .hdr + .bil/.bip/.bsq
  - Header parsing
  - Wavelength/band metadata

- [ ] **Zarr Arrays** - Cloud-native chunked arrays
  - Read/write Zarr stores
  - S3/GCS backend support
  - Chunk-aligned operations

- [ ] **COG Streaming** - Optimize COG access
  - Range request optimization
  - Overview pyramid usage
  - Minimize HTTP requests

### Performance Optimizations

- [ ] **Memory-Mapped I/O**
  - Option for mmap-backed arrays
  - Faster for local files
  - OS handles caching

- [ ] **Dask Integration**
  - Return Dask arrays instead of NumPy
  - Lazy evaluation
  - Distributed processing

- [ ] **GPU Direct I/O**
  - Load directly to GPU memory
  - Avoid CPU ↔ GPU copy
  - Requires CUDA/ROCm support

### Quality of Life

- [ ] **Progress Bars**
  - TQDM integration for long reads
  - Optional (don't clutter non-interactive use)
  - Configurable via environment variable

- [ ] **Logging**
  - Structured logging with Python logging module
  - Debug mode for troubleshooting
  - Performance metrics logging

- [ ] **Validation Utilities**
  - Check format compliance (e.g., valid SICD?)
  - Metadata completeness checks
  - Corruption detection

## Testing TODO

### Unit Tests

- [ ] SAR readers
  - [x] SICDReader basic functionality
  - [ ] SICDReader edge cases (multi-segment NITF)
  - [x] CPHDReader basic functionality
  - [ ] CPHDReader multi-channel handling
  - [x] GRDReader basic functionality
  - [ ] GRDReader multi-band imagery
  - [ ] open_sar() format detection logic

- [ ] EO readers (once implemented)
  - [ ] GeoTIFFReader RGB imagery
  - [ ] GeoTIFFReader multispectral bands
  - [ ] NITFReader with TREs

- [ ] Geospatial readers (once implemented)
  - [ ] GeoJSONReader valid GeoJSON
  - [ ] ShapefileReader .prj handling
  - [ ] KMLReader placemarks and polygons

- [ ] Catalog (once implemented)
  - [ ] ImageCatalog recursive discovery
  - [ ] SpatialCatalog overlap detection
  - [ ] MetadataCatalog query performance

### Integration Tests

- [ ] End-to-end workflows
  - [ ] Read SICD → process → write GeoTIFF
  - [ ] Catalog images → query overlaps → process subset
  - [ ] Multi-format pipeline (SAR + EO fusion)

### Performance Tests

- [ ] Benchmarks
  - [ ] Chip read latency vs. size
  - [ ] Metadata extraction time
  - [ ] Memory usage for large files
  - [ ] Parallel reading speedup

### Test Data

- [ ] Synthetic test files
  - [ ] Minimal valid SICD (via SARPY generators)
  - [ ] Minimal valid CPHD
  - [ ] Small GeoTIFF with georeferencing
  - [ ] Valid GeoJSON fixtures

- [ ] Real-world data
  - [ ] Not in repo (too large)
  - [ ] Document sources for public datasets
  - [ ] Test suite can download if needed

## Documentation TODO

### User Documentation

- [ ] **Tutorials**
  - [ ] "Reading Your First SAR Image"
  - [ ] "Working with SICD Metadata"
  - [ ] "Processing Large Images in Chunks"
  - [ ] "Building an Image Catalog"

- [ ] **How-To Guides**
  - [ ] "Converting SICD to GeoTIFF"
  - [ ] "Extracting Geolocation from SAR"
  - [ ] "Finding Overlapping Images"
  - [ ] "Handling Missing Dependencies"

- [ ] **API Reference**
  - [ ] Auto-generated from docstrings (Sphinx)
  - [ ] Cross-references to related functions
  - [ ] Usage examples in every public function

### Developer Documentation

- [ ] **Contributing Guide**
  - [ ] How to add a new reader
  - [ ] Testing requirements
  - [ ] Code review process

- [ ] **Architecture Decisions**
  - [ ] Why SARPY over custom NITF parser
  - [ ] Why rasterio over GDAL directly
  - [ ] Memory management strategy

## Dependencies TODO

- [x] **Package Setup**
  - [x] `pyproject.toml` with setuptools backend
  - [x] Declare optional dependencies
    - `pip install grdl[sar]` → sarpy
    - `pip install grdl[eo]` → rasterio
    - `pip install grdl[biomass]` → rasterio + requests
    - `pip install grdl[coregistration]` → opencv-python-headless
    - `pip install grdl[all]` → all optional deps
    - `pip install grdl[dev]` → pytest, pytest-benchmark, ruff, mypy, black
  - [ ] Continuous integration (GitHub Actions)

- [x] **Dependency Pinning**
  - [x] Broad version ranges in pyproject.toml (don't over-constrain)

- [ ] **Compatibility Testing**
  - [ ] Test with multiple NumPy versions
  - [ ] Test with multiple SARPY versions
  - [ ] Test with multiple rasterio versions

## Breaking Changes / API Evolution

### Potential Breaking Changes

Track here for next major version:

- [ ] **Standardize band indexing**: All 0-based (currently GRD converts internally)
- [ ] **Geolocation format**: Unified dict structure across all readers
- [ ] **Metadata standardization**: Common keys across formats (e.g., 'acquisition_time')
- [x] **Error handling**: Custom exception hierarchy (`grdl.exceptions`) -- implemented

### API Additions (Non-Breaking)

Can add in minor versions:

- [ ] **`read_bands()` method**: Explicit multi-band reading
- [ ] **`get_band_names()` method**: Semantic band labels (e.g., 'VV', 'VH')
- [ ] **`to_dask()` method**: Return Dask array for lazy evaluation
- [ ] **`get_footprint()` method**: Return geographic footprint as shapely geometry

## Questions / Decisions Needed

- [ ] **Metadata format**: Dict vs. custom classes vs. dataclasses?
  - Dict: Flexible but no type safety
  - Custom classes: Type safe but rigid
  - Dataclasses: Middle ground, good for Python 3.7+

- [ ] **Geolocation standard**: Always convert to WGS84 lat/lon or preserve native?
  - WGS84: Simple, consistent
  - Native: Preserves precision, matches source

- [ ] **Complex data return type**: NumPy complex64 or separate I/Q arrays?
  - complex64: Standard, compact
  - Separate: Explicit, some algorithms prefer

- [ ] **Async I/O**: Support asyncio for concurrent reads?
  - Pro: Better for networked storage
  - Con: Complexity, limited backend support

## Community Requests

Track user-requested features here:

- [ ] _No requests yet - library not yet released_

## Version Milestones

### v0.1.0 (Current)
- [x] Base classes (ImageReader, ImageWriter, CatalogInterface ABCs)
- [x] SAR readers (SICD, CPHD, GRD)
- [x] BIOMASS L1 SCS reader (complex P-band SAR)
- [x] BIOMASS catalog (MAAP STAC search, OAuth2 download, SQLite tracking)
- [x] Geolocation module (GCP interpolation, batch transforms)
- [x] Image processing module
  - [x] ImageProcessor / ImageTransform / BandwiseTransformMixin ABCs
  - [x] Orthorectification (Orthorectifier, OutputGrid)
  - [x] Polarimetric decomposition (PauliDecomposition)
  - [x] Detection data models (Geometry, Detection, DetectionSet, OutputSchema)
  - [x] ImageDetector ABC with geo-registration helpers
  - [x] Processor versioning (`@processor_version` decorator)
  - [x] Processor capability tags (`@processor_tags` decorator)
  - [x] Tunable parameter system (TunableParameterSpec)
  - [x] Detection input chaining (DetectionInputSpec)
  - [x] Pipeline composition (sequential transform chaining)
  - [x] Progress callback protocol (`_report_progress()`)
- [x] ImageJ/Fiji algorithm ports (12 components, all with tags and GPU flags)
  - [x] Spatial filters, contrast enhancement, thresholding, segmentation
  - [x] Edge detection, peak detection, frequency-domain filtering, stack projection
  - [x] Vectorized CLAHE and SRM edge construction
- [x] Data preparation module (ChipBase, ChipExtractor, Tiler, Normalizer)
- [x] Coregistration module (affine, projective, feature-matching)
- [x] Custom exception hierarchy (GrdlError, ValidationError, ProcessorError, etc.)
- [x] PEP 561 type marker (py.typed)
- [x] pyproject.toml with optional dependency extras
- [x] Shared test fixtures (conftest.py)
- [x] Performance benchmarks (pytest-benchmark)
- [x] Example scripts (catalog discovery, Pauli viewer, ortho workflow)
- [x] Ground truth data (BIOMASS cal/val targets GeoJSON)
- [x] Documentation framework

### v0.2.0 (Next)
- [ ] Additional SAR readers (CRSD, SLC, SIDD)
- [ ] Basic EO readers (GeoTIFF, NITF)
- [ ] Concrete ImageDetector implementations
- [ ] Test coverage >80%

### v0.3.0
- [ ] Geospatial readers/writers (GeoJSON, Shapefile)
- [ ] Generic multi-format catalog
- [ ] Performance optimizations

### v1.0.0 (Stable)
- [ ] Full test coverage (>90%)
- [ ] Complete documentation
- [ ] API stability guarantees
- [ ] PyPI release

## Notes

- Prioritize real-world use cases over feature completeness
- Get user feedback early on API design
- Don't implement writers until reader API is stable
- Cloud support is important but requires careful design
- Performance matters - benchmark before optimizing
