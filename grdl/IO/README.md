# IO Module

Input/Output operations for geospatial imagery and vector data.

## Overview

The IO module provides a unified interface for reading geospatial imagery across SAR, IR, multispectral, and EO formats. **Always use an IO reader to load imagery** -- don't call `rasterio.open()` or `h5py.File()` directly. The readers handle lazy loading, band indexing, resource cleanup, and typed metadata extraction.

All readers inherit from `ImageReader` (defined in `base.py`), ensuring a consistent API across formats. Readers are organized into modality-based submodules (`sar/`, `ir/`, `multispectral/`, `eo/`) mirroring how sensors are used in practice. Metadata is returned as typed dataclasses (`SICDMetadata`, `SIDDMetadata`, `BIOMASSMetadata`, `VIIRSMetadata`, `ASTERMetadata`) with nested attribute access, IDE autocomplete, and backward-compatible dict-like `[]` access.

## Supported Formats

### Base Data Formats (IO level)

| Format | Reader Class | Backend | Status |
|--------|-------------|---------|--------|
| GeoTIFF/COG | `GeoTIFFReader` | rasterio | ✅ Implemented |
| HDF5/HDF-EOS5 | `HDF5Reader` | h5py | ✅ Implemented |
| NITF | `NITFReader` | rasterio/GDAL | ✅ Implemented |

### SAR (Synthetic Aperture Radar) — `sar/` submodule

| Format | Reader Class | Backend | Status |
|--------|-------------|---------|--------|
| SICD | `SICDReader` | sarkit (primary), sarpy (fallback) | ✅ Implemented |
| CPHD | `CPHDReader` | sarkit (primary), sarpy (fallback) | ✅ Implemented |
| CRSD | `CRSDReader` | sarkit | ✅ Implemented |
| SIDD | `SIDDReader` | sarkit | ✅ Implemented |
| BIOMASS L1 SCS | `BIOMASSL1Reader` | rasterio | ✅ Implemented |
| Sentinel-1 SLC | `Sentinel1SLCReader` | rasterio | ✅ Implemented |
| TerraSAR-X / TanDEM-X | `TerraSARReader` | numpy (SSC), rasterio (detected) | ✅ Implemented |

### IR (Infrared / Thermal) — `ir/` submodule

| Format | Reader Class | Backend | Status |
|--------|-------------|---------|--------|
| ASTER L1T | `ASTERReader` | rasterio | ✅ Implemented |
| ASTER GDEM | `ASTERReader` | rasterio | ✅ Implemented |
| ECOSTRESS | - | - | 🔄 Planned |
| Landsat TIRS | - | - | 🔄 Planned |

### Multispectral / Hyperspectral — `multispectral/` submodule

| Format | Reader Class | Backend | Status |
|--------|-------------|---------|--------|
| VIIRS HDF5 | `VIIRSReader` | h5py | ✅ Implemented |
| MODIS | - | - | 🔄 Planned |
| EMIT | - | - | 🔄 Planned |
| PRISMA | - | - | 🔄 Planned |

### EO (Electro-Optical) — `eo/` submodule

| Format | Reader Class | Backend | Status |
|--------|-------------|---------|--------|
| Landsat OLI | - | - | 🔄 Planned |
| Sentinel-2 | - | - | 🔄 Planned |
| WorldView | - | - | 🔄 Planned |

### Geospatial Vector

| Format | Type | Support | Status |
|--------|------|---------|--------|
| GeoJSON | Vector | Reader/Writer | 🔄 Planned |
| Shapefile | Vector | Reader/Writer | 🔄 Planned |
| KML/KMZ | Vector | Reader | 🔄 Planned |

### Catalog & Discovery

| Feature | Status |
|---------|--------|
| BIOMASS catalog & download (MAAP STAC API) | ✅ Implemented |
| OAuth2 credential management | ✅ Implemented |
| File discovery by extension | ✅ Implemented |
| Metadata extraction | ✅ Implemented |
| Spatial overlap detection | ✅ Implemented |
| SQLite product tracking | ✅ Implemented |
| Multi-format catalog | 🔄 Planned |

## Installation

Install GRDL with the desired optional dependency extras via `pyproject.toml`:

```bash
pip install -e .                     # Core (numpy + scipy)
pip install -e ".[sar]"              # SAR readers (+ sarpy)
pip install -e ".[eo]"               # EO readers (+ rasterio)
pip install -e ".[biomass]"          # BIOMASS catalog (+ rasterio, requests)
pip install -e ".[coregistration]"   # Image alignment (+ opencv-python-headless)
pip install -e ".[all]"              # All optional dependencies
pip install -e ".[dev]"              # Development tools (pytest, ruff, mypy, etc.)
```

## Quick Start

### Reading SAR Data

#### Auto-detect Format

```python
from grdl.IO import open_sar

# Automatically detect and open SAR file
with open_sar('image.nitf') as reader:
    meta = reader.metadata
    print(f"Format: {meta.format}")
    print(f"Shape: {reader.get_shape()}")

    # Read a spatial chip
    chip = reader.read_chip(0, 1024, 0, 1024)
    print(f"Chip shape: {chip.shape}")
```

#### SICD - Complex SAR Imagery

```python
from grdl.IO import SICDReader
import numpy as np

with SICDReader('sicd_image.nitf') as reader:
    meta = reader.metadata  # SICDMetadata with typed nested access

    # Collection info (nested dataclass)
    ci = meta.collection_info
    if ci:
        print(f"Collector: {ci.collector_name}")
        print(f"Classification: {ci.classification}")

    # Image formation parameters
    if meta.radar_collection and meta.radar_collection.tx_frequency:
        tx = meta.radar_collection.tx_frequency
        print(f"Tx Freq: {tx.min} - {tx.max} Hz")

    # Read complex data
    chip = reader.read_chip(1000, 2000, 1000, 2000)

    # Convert to magnitude in dB
    magnitude_db = 20 * np.log10(np.abs(chip) + 1e-10)

    # Scene center (nested: geo_data → scp → llh → lat/lon/hae)
    if meta.geo_data and meta.geo_data.scp:
        scp = meta.geo_data.scp.llh
        print(f"Scene Center: {scp.lat:.4f}, {scp.lon:.4f}, {scp.hae:.1f} m")
```

#### CPHD - Phase History Data

```python
from grdl.IO import CPHDReader

with CPHDReader('cphd_data.cphd') as reader:
    meta = reader.metadata
    print(f"Channels: {meta.bands}")
    print(f"Format: {meta.format}")

    # Read phase history data
    ph_data = reader.read_full(bands=[0])  # First channel
```

#### HDF5 - NASA, JAXA, Hyperspectral Products

```python
from grdl.IO import HDF5Reader

# Browse datasets in an HDF5 file
datasets = HDF5Reader.list_datasets('MOD09GA.h5')
for path, shape, dtype in datasets:
    print(f"{path}: {shape} ({dtype})")

# Open with explicit dataset path
with HDF5Reader('MOD09GA.h5', dataset_path='/MODIS_Grid/sur_refl_b01') as reader:
    print(f"Shape: {reader.get_shape()}")
    chip = reader.read_chip(0, 512, 0, 512)

# Auto-detect first suitable dataset
with HDF5Reader('product.h5') as reader:
    print(f"Selected: {reader.dataset_path}")
    full = reader.read_full()
```

#### GeoTIFF - Any Raster Imagery (SAR GRD, EO, MSI)

```python
from grdl.IO import GeoTIFFReader, open_image

# Auto-detect any supported raster format
with open_image('scene.tif') as reader:
    print(f"Format: {reader.metadata.format}")
    chip = reader.read_chip(0, 1024, 0, 1024)

# Or use the GeoTIFFReader directly
with GeoTIFFReader('sentinel1_grd.tif') as reader:
    meta = reader.metadata  # ImageMetadata
    print(f"CRS: {meta.crs}")
    print(f"Dimensions: {meta.rows} x {meta.cols}")

    # Read data (real-valued, not complex)
    chip = reader.read_chip(0, 1000, 0, 1000)

    # Multi-band support
    if meta.bands > 1:
        # Read specific bands (0-based indexing)
        vv_vh = reader.read_chip(0, 1000, 0, 1000, bands=[0, 1])
```

#### CRSD - Compensated Radar Signal Data

```python
from grdl.IO import CRSDReader

with CRSDReader('data.crsd') as reader:
    print(f"Channels: {reader.metadata.bands}")
    shape = reader.get_shape()
    signal = reader.read_chip(0, 100, 0, 200)
```

#### SIDD - Sensor Independent Derived Data

```python
from grdl.IO import SIDDReader

with SIDDReader('derived.nitf', image_index=0) as reader:
    meta = reader.metadata  # SIDDMetadata
    print(f"Num images: {meta.num_images}")
    print(f"Pixel type: {meta.display.pixel_type if meta.display else '?'}")
    chip = reader.read_chip(0, 512, 0, 512)
```

### BIOMASS ESA Satellite Data

#### BIOMASS L1 SCS - Complex P-band SAR

```python
from grdl.IO import BIOMASSL1Reader, open_biomass
import numpy as np

# Auto-detect and open BIOMASS product
with open_biomass('BIO_S1_SCS__1S_...') as reader:
    meta = reader.metadata  # BIOMASSMetadata with typed fields

    print(f"Mission: {meta.mission}")
    print(f"Swath: {meta.swath}")
    print(f"Polarizations: {meta.polarizations}")  # [HH, HV, VH, VV]
    print(f"Orbit: {meta.orbit_number} ({meta.orbit_pass})")
    print(f"Dimensions: {meta.rows} x {meta.cols}")

    # Read HH polarization chip (band 0)
    hh_chip = reader.read_chip(0, 1024, 0, 1024, bands=[0])

    # Convert to magnitude in dB
    hh_mag_db = 20 * np.log10(np.abs(hh_chip) + 1e-10)

    # Read all polarizations
    all_pols = reader.read_chip(0, 512, 0, 512)  # Shape: (4, 512, 512)

    for i, pol in enumerate(reader.polarizations):
        mag = np.abs(all_pols[i])
        print(f"{pol}: magnitude range [{mag.min():.2f}, {mag.max():.2f}]")

    # Geolocation info (slant range geometry)
    print(f"Projection: {meta.projection}")  # Slant Range
    print(f"Range spacing: {meta.range_pixel_spacing:.2f} m")
    print(f"Azimuth spacing: {meta.azimuth_pixel_spacing:.2f} m")
```

### TerraSAR-X / TanDEM-X

#### SSC (Complex) and Detected Products

```python
from grdl.IO import TerraSARReader, open_terrasar
import numpy as np

# Open from product directory, XML file, or .cos data file
with TerraSARReader('/path/to/TSX1_SAR__SSC.../', polarization='HH') as reader:
    meta = reader.metadata  # TerraSARMetadata with typed fields

    # Product info
    pi = meta.product_info
    print(f"Satellite: {pi.satellite}")       # "TSX-1" or "TDX-1"
    print(f"Product: {pi.product_type}")       # "SSC", "MGD", "GEC", "EEC"
    print(f"Mode: {pi.imaging_mode}")          # "SM", "HS", "SL", "SC", "ST"
    print(f"Polarizations: {pi.polarization_list}")  # ["HH", "VV"]
    print(f"Orbit: {pi.absolute_orbit} ({pi.orbit_direction})")

    # Radar parameters
    rp = meta.radar_params
    print(f"Center freq: {rp.center_frequency/1e9:.2f} GHz")
    print(f"PRF: {rp.prf:.1f} Hz")

    # Scene geometry
    si = meta.scene_info
    print(f"Center: {si.center_lat:.4f}, {si.center_lon:.4f}")
    print(f"Incidence: {si.incidence_angle_center:.1f} deg")

    # Geolocation grid (from GEOREF.xml)
    print(f"Geo grid points: {len(meta.geolocation_grid)}")

    # Read complex SSC chip
    chip = reader.read_chip(0, 1024, 0, 1024)  # complex64
    magnitude_db = 20 * np.log10(np.abs(chip) + 1e-10)

# Convenience factory function
with open_terrasar('/path/to/product/', polarization='VV') as reader:
    full = reader.read_full()

# Auto-detect via open_sar (works with TSX1_/TDX1_ dirs or any dir with TSX annotation XML)
from grdl.IO import open_sar
with open_sar('/path/to/product/') as reader:
    chip = reader.read_chip(0, 512, 0, 512)
```

### IR / Thermal Imagery

#### ASTER - Thermal Infrared and DEM Products

```python
from grdl.IO import ASTERReader, open_ir

# Auto-detect ASTER product type
with open_ir('AST_L1T_00305042006.tif') as reader:
    meta = reader.metadata  # ASTERMetadata with typed fields
    print(f"Product: {meta.processing_level}")  # "L1T"
    print(f"Date: {meta.acquisition_date}")
    print(f"Cloud: {meta.cloud_cover}%")
    chip = reader.read_chip(0, 512, 0, 512)

# Or use the reader directly
with ASTERReader('ASTGTM_N34W119_dem.tif') as reader:
    meta = reader.metadata
    print(f"Product: {meta.processing_level}")  # "GDEM"
    print(f"CRS: {meta.crs}")
    print(f"TIR available: {meta.tir_available}")
    elevation = reader.read_full()
```

### Multispectral Imagery

#### VIIRS - Nighttime Lights, Vegetation Index, and More

```python
from grdl.IO import VIIRSReader, open_multispectral

# Auto-detect VIIRS product
with open_multispectral('VNP46A1.A2024001.h09v05.002.h5') as reader:
    meta = reader.metadata  # VIIRSMetadata with typed fields
    print(f"Satellite: {meta.satellite_name}")   # "Suomi NPP"
    print(f"Product: {meta.product_short_name}")  # "VNP46A1"
    print(f"Day/Night: {meta.day_night_flag}")
    chip = reader.read_chip(0, 256, 0, 256)

# Browse datasets and select one
datasets = VIIRSReader.list_datasets('VNP13A1.h5')
for path, shape, dtype in datasets:
    print(f"{path}: {shape} ({dtype})")

# Open with explicit dataset path
with VIIRSReader('VNP13A1.h5', dataset_path='/HDFEOS/GRIDS/NDVI') as reader:
    meta = reader.metadata
    print(f"Scale: {meta.scale_factor}")
    print(f"Fill: {meta.fill_value}")
    print(f"Units: {meta.dataset_units}")
    ndvi = reader.read_full()
```

### BIOMASS ESA Satellite Data

#### BIOMASS Data Catalog & Download

Requires an ESA MAAP offline token stored in `~/.config/geoint/credentials.json`.
See the top-level [README](../../README.md#credentials) for setup instructions.

```python
from grdl.IO import BIOMASSCatalog

# Initialize catalog (SQLite DB created automatically)
catalog = BIOMASSCatalog('/data/biomass')

# Discover local products on disk
local_products = catalog.discover_local(update_db=True)
print(f"Found {len(local_products)} local products")

# Search ESA MAAP STAC catalog
products = catalog.query_esa(
    bbox=(115.5, -31.5, 116.8, -30.5),  # (min_lon, min_lat, max_lon, max_lat)
    product_type="S3_SCS__1S",           # Single-pol processing
    max_results=20,
)
print(f"Found {len(products)} products")

# Filter and inspect results
for p in products:
    props = p.get("properties", {})
    print(f"  {p['id']}")
    print(f"    Date:  {props.get('datetime', '?')}")
    print(f"    Orbit: {props.get('sat:absolute_orbit', '?')}")
    print(f"    Pols:  {props.get('sar:polarizations', '?')}")

# Download a product (OAuth2 Bearer token, auto-extracted)
product_path = catalog.download_product(
    products[0]['id'],
    destination='/data/biomass',
    extract=True,
)
print(f"Downloaded: {product_path}")

# Find overlapping products
bbox = (115.5, -31.5, 116.8, -30.5)
overlapping = catalog.find_overlapping(bbox, local_products)
print(f"Found {len(overlapping)} products overlapping bbox")

# Clean up
catalog.close()
```

## Architecture

### Base Classes

All readers inherit from abstract base classes in `base.py`:

- **`ImageReader`** - Base for all imagery readers
  - `read_chip()` - Read spatial subset (memory efficient)
  - `read_full()` - Read entire image
  - `get_shape()` - Get image dimensions
  - `get_dtype()` - Get data type
  - Context manager support (`with` statements)

- **`ImageWriter`** - Base for all imagery writers
  - `write()` - Write full image
  - `write_chip()` - Write spatial subset
  - Context manager support

- **`CatalogInterface`** - Base for image discovery
  - `discover_images()` - Find images by extension
  - `get_metadata_summary()` - Batch metadata extraction
  - `find_overlapping()` - Spatial overlap queries

### Typed Metadata

All readers populate `self.metadata` with a typed dataclass. Format-specific readers return specialized subclasses with nested attribute access:

| Reader | Metadata Class | Key Nested Types |
|--------|---------------|-----------------|
| `GeoTIFFReader` | `ImageMetadata` | (flat fields) |
| `HDF5Reader` | `ImageMetadata` | (flat fields) |
| `NITFReader` | `ImageMetadata` | (flat fields) |
| `SICDReader` | `SICDMetadata` | 17 sections: `collection_info`, `image_data`, `geo_data`, `grid`, `timeline`, `position`, `radar_collection`, `image_formation`, `scpcoa`, `radiometric`, `antenna`, `error_statistics`, `match_info`, `rg_az_comp`, `pfa`, `rma` |
| `SIDDReader` | `SIDDMetadata` | `product_creation`, `display`, `geo_data`, `measurement`, `exploitation_features`, `downstream_reprocessing`, `compression`, `digital_elevation_data`, `product_processing`, `annotations` |
| `BIOMASSL1Reader` | `BIOMASSMetadata` | `mission`, `swath`, `polarizations`, `orbit_number`, `range_pixel_spacing`, `azimuth_pixel_spacing`, `prf`, `corner_coords`, `gcps` |
| `Sentinel1SLCReader` | `Sentinel1SLCMetadata` | `product_info`, `swath_timing`, `bursts`, `orbit_state_vectors`, `geolocation_grid`, `doppler_centroid`, `calibration_vectors` |
| `TerraSARReader` | `TerraSARMetadata` | `product_info`, `scene_info`, `image_info`, `radar_params`, `orbit_state_vectors`, `geolocation_grid`, `calibration`, `doppler_info`, `processing_info` |
| `VIIRSReader` | `VIIRSMetadata` | `satellite_name`, `product_short_name`, `day_night_flag`, `geospatial_bounds`, `scale_factor`, `add_offset`, `fill_value`, `dataset_path` |
| `ASTERReader` | `ASTERMetadata` | `processing_level`, `acquisition_date`, `sun_azimuth`, `sun_elevation`, `cloud_cover`, `vnir_available`, `swir_available`, `tir_available` |

```python
from grdl.IO.models import SICDMetadata, LatLonHAE, XYZ

# Typed nested access with IDE autocomplete
meta: SICDMetadata = reader.metadata
meta.geo_data.scp.llh.lat          # float — Scene Center Point latitude
meta.grid.row.ss                   # float — row sample spacing
meta.scpcoa.graze_ang              # float — grazing angle
meta.collection_info.radar_mode.mode_type  # str — 3 levels deep

# Backward-compatible dict-like access still works
meta['format']                     # str
meta['rows']                       # int
'scpcoa' in meta                   # True
list(meta.keys())                  # all field names
```

The `models/` package provides ~70 dataclasses organized in `common.py` (shared primitives like `XYZ`, `LatLonHAE`, `RowCol`, `Poly1D`, `Poly2D`, `XYZPoly`), `sicd.py`, `sidd.py`, `biomass.py`, `sentinel1_slc.py`, `terrasar.py`, `viirs.py`, and `aster.py`.

### Design Principles

1. **Lazy Loading** - Data only loaded when explicitly requested
2. **Consistent API** - All readers share common interface
3. **Context Managers** - Automatic resource cleanup with `with` statements
4. **Type Safety** - Full type hints for all public APIs; typed metadata dataclasses with nested attribute access
5. **Format Agnostic** - Abstract interfaces hide format-specific details
6. **Graceful Degradation** - Works with subset of dependencies installed

## Memory Considerations

### Large Images

For large SAR or EO images that don't fit in memory, use `ChipExtractor` or `Tiler` from `grdl.data_prep` to plan chip regions. **Do not write your own chunking loops** -- `ChipExtractor` handles boundary snapping and uniform chip sizing:

```python
from grdl.data_prep import ChipExtractor

# BAD - Loads entire image into memory
with SICDReader('large_image.nitf') as reader:
    full_image = reader.read_full()  # May cause OOM

# BAD - Hand-rolled chunking (misses boundary edge cases)
for r in range(0, rows, chip_size):
    for c in range(0, cols, chip_size):
        chip = reader.read_chip(r, min(r + chip_size, rows), ...)

# GOOD - Use ChipExtractor for chip planning
with SICDReader('large_image.nitf') as reader:
    rows, cols = reader.get_shape()
    extractor = ChipExtractor(nrows=rows, ncols=cols)

    for region in extractor.chip_positions(row_width=1024, col_width=1024):
        chip = reader.read_chip(region.row_start, region.row_end,
                                region.col_start, region.col_end)
        # Process chip...
```

### Complex vs. Magnitude

SICD data is complex-valued (I+jQ), requiring 2x memory:

```python
# Complex data: 8 bytes per pixel (float32 real + float32 imag)
complex_chip = sicd_reader.read_chip(0, 1000, 0, 1000)  # ~8 MB

# Magnitude: 4 bytes per pixel (float32)
magnitude_chip = np.abs(complex_chip)  # ~4 MB

# dB magnitude (in-place to save memory)
db_chip = 20 * np.log10(np.abs(complex_chip) + 1e-10)
```

## Error Handling

GRDL provides a custom exception hierarchy in `grdl.exceptions`. All custom exceptions
subclass both `GrdlError` and the appropriate built-in exception for backward compatibility:

```python
from grdl.IO import open_sar
from grdl.exceptions import GrdlError, DependencyError, ValidationError

try:
    reader = open_sar('image.dat')
except FileNotFoundError:
    print("File does not exist")
except DependencyError as e:
    print(f"Missing dependency: {e}")   # subclass of both GrdlError and ImportError
except ValidationError as e:
    print(f"Invalid input: {e}")        # subclass of both GrdlError and ValueError
except GrdlError as e:
    print(f"GRDL error: {e}")           # catch-all for GRDL-specific errors
```

## API Reference

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design decisions.

See [TODO.md](TODO.md) for planned features and roadmap.

## Examples

See `grdl/example/` for working workflows:
- `catalog/discover_and_download.py` - Search ESA MAAP catalog, download products
- `catalog/view_product.py` - Load BIOMASS L1A, display HH dB and Pauli RGB with interactive markers
- `ortho/ortho_biomass.py` - Orthorectification with Pauli RGB composite output
- `sar/view_sicd.py` - SICD magnitude viewer (linear, CLI-driven)
- `image_processing/sar/sublook_compare.py` - **Full GRDL integration**: IO + data_prep + image_processing

## Data Preparation

The `grdl.data_prep` module provides index-only chip/tile planning and normalization. `ChipExtractor` and `Tiler` compute `ChipRegion` bounds -- they never touch pixel data. **Use them with IO readers instead of writing ad-hoc chunking loops:**

```python
from grdl.IO import GeoTIFFReader
from grdl.data_prep import Tiler, ChipExtractor, Normalizer

with GeoTIFFReader('scene.tif') as reader:
    rows, cols = reader.get_shape()
    extractor = ChipExtractor(nrows=rows, ncols=cols)

    # Point-centered chip (boundary-snapped)
    region = extractor.chip_at_point(500, 1000, row_width=64, col_width=64)
    chip = reader.read_chip(region.row_start, region.row_end,
                            region.col_start, region.col_end)

    # Whole-image partitioning
    for region in extractor.chip_positions(row_width=256, col_width=256):
        chip = reader.read_chip(region.row_start, region.row_end,
                                region.col_start, region.col_end)
        # Process each chip...

# Overlapping tiles with stride
tiler = Tiler(nrows=1000, ncols=2000, tile_size=256, stride=128)
tile_regions = tiler.tile_positions()

# Normalize intensity values
norm = Normalizer(method='minmax')
normalized = norm.normalize(image)
```

## Contributing

When adding new readers:

1. Inherit from `ImageReader` or `ImageWriter` base class
2. Implement all abstract methods
3. Follow file header standard from `/CLAUDE.md`
4. Add full type hints and NumPy-style docstrings
5. Include usage examples in docstrings
6. Update this README and `__init__.py`'s `__all__` export

## License

MIT License - See [LICENSE](../../LICENSE) for details.