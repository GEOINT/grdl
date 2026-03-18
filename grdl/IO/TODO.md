# IO Module TODO

Roadmap and planned features for the IO module.

## Current Status (v0.2.0)

### Completed

- Base class architecture (`base.py`): `ImageReader`, `ImageWriter`, `CatalogInterface` ABCs
- SAR readers (`sar/`): `SICDReader`, `CPHDReader`, `CRSDReader`, `SIDDReader`, sarkit/sarpy dual backend, `open_sar()`
- SAR writers (`sar/`): `SICDWriter` (sarpy), `SIDDWriter` (sarpy)
- Base format readers: `GeoTIFFReader`, `HDF5Reader`, `NITFReader`, `JP2Reader`, `open_image()`
- Base format writers: `GeoTIFFWriter` (rasterio/COG), `HDF5Writer` (h5py), `NITFWriter` (rasterio), `PNGWriter` (Pillow)
- BIOMASS: `BIOMASSL1Reader` (quad-pol SCS), `BIOMASSCatalog` (MAAP STAC, OAuth2, download, SQLite)
- Sensor-specific readers: `Sentinel1SLCReader`, `Sentinel2Reader`, `TerraSARReader`, `NISARReader`
- EO/IR/Multispectral submodules: `ASTERReader`, `VIIRSReader`, `open_ir()`, `open_multispectral()`
- Typed metadata (`models/`): `SICDMetadata`, `SIDDMetadata`, `BIOMASSMetadata`, `VIIRSMetadata`, `ASTERMetadata`
- Generic fallback: `GDALReader` for rasterio-supported formats
- Geolocation integration: `Geolocation.from_reader()` factory, GCP-based transforms
- Custom exceptions: `DependencyError` adopted throughout IO

## High Priority

### SAR

- [ ] **SLCReader** — Generic Single Look Complex reader
  - Distinguish SLC from GRD (both GeoTIFF)
  - Parse SAR-specific metadata tags
  - Handle complex data in GeoTIFF

### Catalog

- [ ] **Generic ImageCatalog** — Multi-format, multi-sensor catalog
  - Unified database schema across SAR/EO/MSI
  - Format-agnostic spatial and temporal queries
  - Build on existing `BIOMASSCatalog` patterns

## Medium Priority

### Writers

- [ ] **COG output** — Cloud-Optimized GeoTIFF via GeoTIFFWriter
  - Tiled layout with overview pyramids
  - Compression options (LZW, DEFLATE, ZSTD)

### Advanced Features

- [ ] **Cloud Storage Support** — S3/GCS readers via fsspec
- [ ] **Parallel Chip Reading** — Thread pool for concurrent chip reads
- [ ] **Metadata Caching** — Sidecar .json files for faster re-opening

## Low Priority / Research

- [ ] **SAFE Format** — Sentinel-1/2 native manifest.safe parsing
- [ ] **ENVI Format** — Hyperspectral .hdr + .bil/.bip/.bsq
- [ ] **Zarr Arrays** — Cloud-native chunked array support
- [ ] **Dask Integration** — Lazy evaluation with Dask arrays

## API Evolution

### Potential Breaking Changes (next major version)

- [ ] Standardize band indexing: all 0-based
- [ ] Unified geolocation dict structure across all readers

### Non-Breaking Additions

- [ ] `read_bands()` — Explicit multi-band reading
- [ ] `get_band_names()` — Semantic band labels (e.g., 'VV', 'VH')
- [ ] `get_footprint()` — Geographic footprint as shapely geometry

## Open Questions

- **Geolocation standard**: Always WGS84 lat/lon or preserve native CRS?
- **Complex data return type**: NumPy complex64 vs. separate I/Q arrays?

## Version Milestones

### v0.3.0 (Next)

- [ ] Generic multi-format catalog
- [ ] SLCReader
- [ ] Concrete `ImageDetector` implementations
- [ ] Test coverage >80%

### v1.0.0 (Stable)

- [ ] Full test coverage (>90%)
- [ ] API stability guarantees
- [ ] PyPI release

## Notes

- Prioritize real-world use cases over feature completeness
- Get user feedback early on API design
- Cloud support is important but requires careful design
- Performance matters — benchmark before optimizing
