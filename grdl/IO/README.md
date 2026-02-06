# IO Module

Input/Output operations for geospatial imagery and vector data.

## Overview

The IO module provides a unified interface for reading and writing various geospatial data formats, with a focus on remote sensing imagery. All readers inherit from abstract base classes defined in `base.py`, ensuring consistent APIs across different formats.

## Supported Formats

### SAR (Synthetic Aperture Radar)

| Format | Type | Reader Class | Status |
|--------|------|-------------|--------|
| SICD | NGA Standard Complex | `SICDReader` | ✅ Implemented |
| CPHD | NGA Phase History | `CPHDReader` | ✅ Implemented |
| CRSD | NGA Received Signal | - | 🔄 Planned |
| GRD | Ground Range Detected | `GRDReader` | ✅ Implemented |
| SLC | Single Look Complex | - | 🔄 Planned |

### Mission-Specific Readers

| Mission | Type | Reader Class | Status |
|---------|------|-------------|--------|
| BIOMASS L1 SCS | ESA P-band SAR | `BIOMASSL1Reader` | ✅ Implemented |

### EO (Electro-Optical)

| Format | Type | Reader Class | Status |
|--------|------|-------------|--------|
| GeoTIFF | Raster | - | 🔄 Planned |
| NITF | Container | - | 🔄 Planned |
| COG | Cloud-Optimized GeoTIFF | - | 🔄 Planned |

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

### Core Dependencies

```bash
pip install numpy
```

### SAR Support

```bash
# For SICD/CPHD (NGA standards)
pip install sarpy

# For GRD products (GeoTIFF)
pip install rasterio

# For BIOMASS ESA satellite products
pip install rasterio  # For L1 SCS magnitude/phase TIFFs
pip install requests  # For ESA MAAP STAC queries and downloads
```

### EO Support (Planned)

```bash
pip install rasterio gdal pillow
```

### Geospatial Support (Planned)

```bash
pip install geopandas shapely fiona
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

    # Get geolocation
    geo = reader.get_geolocation()
    print(f"Scene Center: {geo['scp_llh']}")  # [lat, lon, height]
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

#### GRD - Geocoded SAR Products

```python
from grdl.IO import GRDReader

with GRDReader('sentinel1_grd.tif') as reader:
    # Access geolocation
    geo = reader.get_geolocation()
    print(f"CRS: {geo['crs']}")
    print(f"Bounds: {geo['bounds']}")
    print(f"Resolution: {geo['resolution']}")

    # Read data (real-valued, not complex)
    chip = reader.read_chip(0, 1000, 0, 1000)

    # Multi-band support
    if reader.metadata['bands'] > 1:
        # Read specific bands (0-based indexing)
        vv_vh = reader.read_chip(0, 1000, 0, 1000, bands=[0, 1])
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

    # Get geolocation (slant range geometry)
    geo = reader.get_geolocation()
    print(f"Projection: {geo['projection']}")  # Slant Range
    print(f"Range spacing: {geo['range_pixel_spacing']:.2f} m")
    print(f"Azimuth spacing: {geo['azimuth_pixel_spacing']:.2f} m")
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
  - `get_geolocation()` - Get georeferencing info
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

All readers raise specific exceptions:

```python
from grdl.IO import open_sar

try:
    reader = open_sar('image.dat')
except FileNotFoundError:
    print("File does not exist")
except ImportError as e:
    print(f"Missing dependency: {e}")
except ValueError as e:
    print(f"Invalid format: {e}")
```

## API Reference

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design decisions.

See [TODO.md](TODO.md) for planned features and roadmap.

## Examples

See `example/` for working BIOMASS workflows:
- `catalog/discover_and_download.py` - Search ESA MAAP catalog, download products
- `catalog/view_product.py` - Load BIOMASS L1A, display HH dB and Pauli RGB with interactive markers
- `ortho/ortho_biomass.py` - Orthorectification with Pauli RGB composite output

## ImageJ/Fiji Algorithm Ports

GRDL includes 12 classic image processing algorithms ported from ImageJ/Fiji under `grdl.imagej`, selected for relevance to remotely sensed imagery (PAN, MSI, HSI, SAR, thermal). All inherit from `ImageTransform` and can be used directly in processing pipelines.

| Category | Components | Use Cases |
|----------|-----------|-----------|
| Spatial Filters | RollingBallBackground, UnsharpMask, RankFilters, MorphologicalFilter | Background subtraction, sharpening, noise removal, morphological analysis |
| Contrast & Enhancement | CLAHE, GammaCorrection | Dynamic range adjustment, local contrast enhancement |
| Thresholding & Segmentation | AutoLocalThreshold, StatisticalRegionMerging | Land cover segmentation, OBIA, target region extraction |
| Edge & Feature Detection | EdgeDetector, FindMaxima | Boundary detection, target/peak detection in SAR/thermal |
| Frequency Domain | FFTBandpassFilter | Bandpass filtering, pushbroom stripe removal |
| Stack Operations | ZProjection | Multi-temporal composites (max, mean, median, etc.) |

```python
from grdl.imagej import CLAHE, FindMaxima, StatisticalRegionMerging

# Enhance local contrast for thermal imagery
enhanced = CLAHE(block_size=127, max_slope=3.0).apply(thermal_band)

# Detect bright targets in SAR amplitude
targets = FindMaxima(prominence=20.0).find_peaks(sar_amplitude)

# Segment MSI band into land cover regions
labels = StatisticalRegionMerging(Q=50).apply(msi_band)
```

Each ported component carries `__imagej_source__` and `__imagej_version__` attributes for provenance tracking. ImageJ 1.x ports are public domain; Fiji plugin ports (CLAHE, AutoLocalThreshold, SRM) are independent NumPy reimplementations of published algorithms.

See `tests/test_imagej.py` for 124 tests covering all 12 components.

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