# IO Module

Input/Output operations for geospatial imagery and vector data.

## Overview

The IO module provides a unified interface for reading and writing various geospatial data formats, with a focus on remote sensing imagery. All readers inherit from abstract base classes defined in `base.py`, ensuring consistent APIs across different formats.

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
| SLC | - | - | 🔄 Planned |

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
    print(f"Format: {reader.metadata['format']}")
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
    # Access metadata
    print(f"Collector: {reader.metadata['collector_name']}")
    print(f"Center Freq: {reader.metadata['center_frequency']} Hz")
    print(f"Bandwidth: {reader.metadata['bandwidth']} Hz")

    # Read complex data
    chip = reader.read_chip(1000, 2000, 1000, 2000)

    # Convert to magnitude image
    magnitude = np.abs(chip)

    # Convert to dB
    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    # Scene center from metadata
    scp = reader.metadata.get('scp_llh')
    if scp:
        print(f"Scene Center: {scp}")  # [lat, lon, height]
```

#### CPHD - Phase History Data

```python
from grdl.IO import CPHDReader

with CPHDReader('cphd_data.cphd') as reader:
    print(f"Channels: {reader.metadata['num_channels']}")
    print(f"Collector: {reader.metadata['collector_name']}")

    # Access channel information
    for channel_id, info in reader.metadata['channels'].items():
        print(f"Channel {channel_id}:")
        print(f"  Vectors: {info['num_vectors']}")
        print(f"  Samples: {info['num_samples']}")

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
    print(f"Format: {reader.metadata['format']}")
    chip = reader.read_chip(0, 1024, 0, 1024)

# Or use the GeoTIFFReader directly
with GeoTIFFReader('sentinel1_grd.tif') as reader:
    # Access geolocation from metadata
    print(f"CRS: {reader.metadata['crs']}")
    print(f"Bounds: {reader.metadata['bounds']}")
    print(f"Resolution: {reader.metadata['resolution']}")

    # Read data (real-valued, not complex)
    chip = reader.read_chip(0, 1000, 0, 1000)

    # Multi-band support
    if reader.metadata['bands'] > 1:
        # Read specific bands (0-based indexing)
        vv_vh = reader.read_chip(0, 1000, 0, 1000, bands=[0, 1])
```

#### CRSD - Compensated Radar Signal Data

```python
from grdl.IO import CRSDReader

with CRSDReader('data.crsd') as reader:
    print(f"Channels: {reader.metadata['num_channels']}")
    shape = reader.get_shape()
    signal = reader.read_chip(0, 100, 0, 200)
```

#### SIDD - Sensor Independent Derived Data

```python
from grdl.IO import SIDDReader

with SIDDReader('derived.nitf', image_index=0) as reader:
    print(f"Num images: {reader.metadata['num_images']}")
    chip = reader.read_chip(0, 512, 0, 512)
```

### BIOMASS ESA Satellite Data

#### BIOMASS L1 SCS - Complex P-band SAR

```python
from grdl.IO import BIOMASSL1Reader, open_biomass
import numpy as np

# Auto-detect and open BIOMASS product
with open_biomass('BIO_S1_SCS__1S_...') as reader:
    print(f"Mission: {reader.metadata['mission']}")
    print(f"Swath: {reader.metadata['swath']}")
    print(f"Polarizations: {reader.metadata['polarizations']}")  # [HH, HV, VH, VV]
    print(f"Orbit: {reader.metadata['orbit_number']} ({reader.metadata['orbit_pass']})")
    print(f"Dimensions: {reader.metadata['rows']} x {reader.metadata['cols']}")

    # Read HH polarization chip (band 0)
    hh_chip = reader.read_chip(0, 1024, 0, 1024, bands=[0])

    # Convert to magnitude in dB
    hh_mag_db = 20 * np.log10(np.abs(hh_chip) + 1e-10)

    # Read all polarizations
    all_pols = reader.read_chip(0, 512, 0, 512)  # Shape: (4, 512, 512)

    for i, pol in enumerate(reader.polarizations):
        mag = np.abs(all_pols[i])
        print(f"{pol}: magnitude range [{mag.min():.2f}, {mag.max():.2f}]")

    # Geolocation info from metadata (slant range geometry)
    print(f"Projection: {reader.metadata['projection']}")  # Slant Range
    print(f"Range spacing: {reader.metadata['range_pixel_spacing']:.2f} m")
    print(f"Azimuth spacing: {reader.metadata['azimuth_pixel_spacing']:.2f} m")
```

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

### Design Principles

1. **Lazy Loading** - Data only loaded when explicitly requested
2. **Consistent API** - All readers share common interface
3. **Context Managers** - Automatic resource cleanup with `with` statements
4. **Type Safety** - Full type hints for all public APIs
5. **Format Agnostic** - Abstract interfaces hide format-specific details
6. **Graceful Degradation** - Works with subset of dependencies installed

## Memory Considerations

### Large Images

For large SAR or EO images that don't fit in memory:

```python
# BAD - Loads entire image into memory
with SICDReader('large_image.nitf') as reader:
    full_image = reader.read_full()  # May cause OOM

# GOOD - Process in chunks
with SICDReader('large_image.nitf') as reader:
    rows, cols = reader.get_shape()
    chip_size = 1024

    for r in range(0, rows, chip_size):
        for c in range(0, cols, chip_size):
            chip = reader.read_chip(
                r, min(r + chip_size, rows),
                c, min(c + chip_size, cols)
            )
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

## ImageJ/Fiji Algorithm Ports

GRDL includes 12 classic image processing algorithms ported from ImageJ/Fiji under `grdl.imagej`, selected for relevance to remotely sensed imagery (PAN, MSI, HSI, SAR, thermal). All inherit from `ImageTransform`, carry `@processor_tags` metadata for capability discovery, and declare `__gpu_compatible__` for downstream GPU dispatch.

| Subdirectory | ImageJ Menu | Components | GPU | Use Cases |
|-------------|------------|-----------|-----|-----------|
| `filters/` | Process > Filters | RankFilters, UnsharpMask | No | Noise removal, sharpening |
| `background/` | Process > Subtract Background | RollingBallBackground | No | Background subtraction |
| `binary/` | Process > Binary | MorphologicalFilter | No | Morphological operations |
| `enhance/` | Process > Enhance Contrast | CLAHE, GammaCorrection | Yes | Dynamic range, local contrast |
| `edges/` | Process > Find Edges | EdgeDetector | No | Boundary detection |
| `fft/` | Process > FFT | FFTBandpassFilter | Yes | Bandpass filtering, stripe removal |
| `find_maxima/` | Process > Find Maxima | FindMaxima | No | Target/peak detection |
| `threshold/` | Image > Adjust > Threshold | AutoLocalThreshold | No | Local thresholding, OBIA |
| `segmentation/` | Plugins > Segmentation | StatisticalRegionMerging | No | Land cover segmentation |
| `stacks/` | Image > Stacks | ZProjection | Yes | Multi-temporal composites |

```python
from grdl.imagej import CLAHE, FindMaxima, StatisticalRegionMerging
from grdl.image_processing import Pipeline

# Enhance local contrast for thermal imagery
enhanced = CLAHE(block_size=127, max_slope=3.0).apply(thermal_band)

# Detect bright targets in SAR amplitude
targets = FindMaxima(prominence=20.0).find_peaks(sar_amplitude)

# Segment MSI band into land cover regions
labels = StatisticalRegionMerging(Q=50).apply(msi_band)

# Compose a multi-step pipeline
pipe = Pipeline([CLAHE(block_size=63), FindMaxima(prominence=10.0)])
```

Each ported component carries `__imagej_source__`, `__imagej_version__`, `__gpu_compatible__`, and `@processor_tags` metadata. ImageJ 1.x ports are public domain; Fiji plugin ports (CLAHE, AutoLocalThreshold, SRM) are independent NumPy reimplementations of published algorithms.

## Data Preparation

The `grdl.data_prep` module provides chip/tile index computation and normalization utilities:

```python
from grdl.data_prep import Tiler, ChipExtractor, Normalizer

# Compute chip region centered at a point (snaps to stay inside image)
extractor = ChipExtractor(nrows=rows, ncols=cols)
region = extractor.chip_at_point(500, 1000, row_width=64, col_width=64)
chip = image[region.row_start:region.row_end, region.col_start:region.col_end]

# Partition image into uniform chip regions
regions = extractor.chip_positions(row_width=256, col_width=256)

# Compute overlapping tile positions with stride
tiler = Tiler(nrows=rows, ncols=cols, tile_size=256, stride=128)
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