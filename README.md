# GRDL - GEOINT Rapid Development Library

A modular Python library of fundamental building blocks for deriving geospatial intelligence from any 2D correlated imagery -- SAR, electro-optical, multispectral, hyperspectral, space-based, or terrestrial. GRDL provides composable, well-documented components that developers combine into workflows for image processing, machine learning, geospatial transforms, and sensor exploitation.

## What is GRDL?

GEOINT (Geospatial Intelligence) development requires stitching together domain-specific operations -- reading sensor formats, applying signal processing, transforming coordinate systems, chunking large images, preparing data for ML pipelines. Each project tends to re-implement these operations from scratch.

GRDL solves this by providing a library of **small, focused modules** that each do one thing well. If it's 2D correlated data, GRDL is built to work with it. Developers pick the pieces they need and compose them into workflows without inheriting a monolithic framework.

### Design Principles

- **Modular** -- Each module is self-contained with minimal cross-dependencies. Use only what you need.
- **Verbose** -- Functions and classes document their inputs, outputs, assumptions, and units. No guessing what a parameter expects.
- **Composable** -- Modules are designed to chain together. The output of one operation is a natural input to the next.
- **Sensor-agnostic** -- Built around 2D correlated data as the common abstraction. The same tools work whether the source is a SAR sensor, an EO camera, a multispectral imager, or a terrestrial scanner.
- **Not a framework** -- GRDL is a library, not an application. It doesn't impose structure on your project.

## Module Areas

| Domain | Description | Status |
|--------|-------------|--------|
| **I/O** | Readers and writers for geospatial imagery formats (SICD, CPHD, GRD, BIOMASS) | Implemented |
| **Geolocation** | Pixel-to-geographic coordinate transforms (GCP interpolation, affine, SICD) | Implemented |
| **Image Processing** | Orthorectification, polarimetric decomposition, filtering, enhancement | Implemented |
| **Data Preparation** | Chunking, tiling, resampling, and formatting for ML/AI pipelines | Planned |
| **Sensor Processing** | Sensor-specific operations (SAR phase history, EO radiometry, MSI band math) | Planned |
| **ML/AI Utilities** | Feature extraction, annotation tools, dataset builders, and model integration helpers | Planned |

## Project Structure

```
GRDL/
├── grdl/                            # Library source
│   ├── IO/                          # Input/Output module
│   │   ├── base.py                  #   ImageReader / ImageWriter / CatalogInterface ABCs
│   │   ├── sar.py                   #   SICDReader, CPHDReader, GRDReader, open_sar()
│   │   ├── biomass.py               #   BIOMASSL1Reader, open_biomass()
│   │   └── catalog.py               #   BIOMASSCatalog, load_credentials()
│   ├── geolocation/                 # Coordinate transform module
│   │   ├── base.py                  #   Geolocation ABC, NoGeolocation
│   │   ├── utils.py                 #   Footprint, bounds, distance helpers
│   │   ├── sar/
│   │   │   └── gcp.py               #   GCPGeolocation (Delaunay interpolation)
│   │   └── eo/                      #   EO geolocation (planned)
│   └── image_processing/            # Image transforms module
│       ├── base.py                  #   ImageTransform ABC
│       ├── ortho/
│       │   └── ortho.py             #   Orthorectifier, OutputGrid
│       └── decomposition/
│           ├── base.py              #   PolarimetricDecomposition ABC
│           └── pauli.py             #   PauliDecomposition (quad-pol)
├── example/                         # Example scripts
│   ├── catalog/
│   │   ├── discover_and_download.py #   BIOMASS MAAP catalog search & download
│   │   └── view_product.py          #   BIOMASS viewer with Pauli decomposition
│   └── ortho/
│       └── ortho_biomass.py         #   Orthorectification with Pauli RGB
├── ground_truth/                    # Reference data for calibration & validation
│   └── biomass_calibration_targets.geojson
├── tests/                           # Test suite
│   ├── test_io_biomass.py           #   BIOMASS reader tests
│   ├── test_geolocation_biomass.py  #   Geolocation tests with interactive markers
│   ├── test_image_processing_ortho.py        #   Orthorectification tests
│   └── test_image_processing_decomposition.py #  Pauli decomposition tests
├── example_images/                  # Small sample data for tests and demos
├── requirements.txt                 # Core: numpy, scipy
├── requirements-dev.txt             # Dev: pytest, black, flake8, mypy
├── requirements-optional.txt        # Optional: sarpy, rasterio, requests
└── CLAUDE.md                        # Development standards and coding guide
```

## Quick Start

### Reading BIOMASS SAR Data

```python
from grdl.IO import BIOMASSL1Reader
from grdl.geolocation import Geolocation
import numpy as np

with BIOMASSL1Reader('path/to/BIO_S3_SCS__1S_...') as reader:
    print(f"Polarizations: {reader.metadata['polarizations']}")
    rows, cols = reader.get_shape()

    # Read HH channel (complex-valued)
    hh = reader.read_chip(0, rows, 0, cols, bands=[0])
    hh_db = 20 * np.log10(np.abs(hh) + 1e-10)

    # Pixel-to-geographic coordinate transform
    geo = Geolocation.from_reader(reader)
    lat, lon, _ = geo.pixel_to_latlon(rows // 2, cols // 2)
    print(f"Center: ({lat:.6f}, {lon:.6f})")
```

### Searching & Downloading from ESA MAAP

```python
from grdl.IO import BIOMASSCatalog

catalog = BIOMASSCatalog('/data/biomass')

# Search the MAAP STAC catalog
products = catalog.query_esa(
    bbox=(115.5, -31.5, 116.8, -30.5),
    product_type="S3_SCS__1S",
    max_results=10,
)

# Download a product (uses OAuth2 via ~/.config/geoint/credentials.json)
catalog.download_product(products[0]['id'], extract=True)
catalog.close()
```

### Geolocation Transforms

```python
from grdl.IO import open_biomass
from grdl.geolocation import Geolocation
import numpy as np

with open_biomass('path/to/product') as reader:
    geo = Geolocation.from_reader(reader)

    # Single pixel (returns scalars)
    lat, lon, height = geo.pixel_to_latlon(500, 1000)

    # Array of pixels (returns arrays, vectorized)
    rows = np.array([100, 200, 300])
    cols = np.array([400, 500, 600])
    lats, lons, heights = geo.pixel_to_latlon(rows, cols)

    # Inverse: geographic to pixel (also accepts scalar or array)
    row, col = geo.latlon_to_pixel(-31.05, 116.19)
```

### Pauli Decomposition (Quad-Pol SAR)

```python
from grdl.IO import BIOMASSL1Reader
from grdl.image_processing import PauliDecomposition

with BIOMASSL1Reader('path/to/product') as reader:
    rows, cols = reader.get_shape()
    shh = reader.read_chip(0, rows, 0, cols, bands=[0])
    shv = reader.read_chip(0, rows, 0, cols, bands=[1])
    svh = reader.read_chip(0, rows, 0, cols, bands=[2])
    svv = reader.read_chip(0, rows, 0, cols, bands=[3])

pauli = PauliDecomposition()
components = pauli.decompose(shh, shv, svh, svv)

# Complex-valued components (phase preserved)
surface = components['surface']            # (S_HH + S_VV) / sqrt(2)
double_bounce = components['double_bounce']  # (S_HH - S_VV) / sqrt(2)
volume = components['volume']              # (S_HV + S_VH) / sqrt(2)

# Convert to display representations
db = pauli.to_db(components)          # Dict of dB arrays
rgb = pauli.to_rgb(components)        # (rows, cols, 3) float32 [0, 1]
```

### Orthorectification

```python
from grdl.image_processing import Orthorectifier, OutputGrid

geo = Geolocation.from_reader(reader)
grid = OutputGrid.from_geolocation(geo, pixel_size_lat=0.001,
                                   pixel_size_lon=0.001)
ortho = Orthorectifier(geo, grid, interpolation='nearest')
ortho.compute_mapping()
result = ortho.apply(hh_db, nodata=np.nan)
```

## Installation

```bash
git clone https://github.com/geoint-org/GRDL.git
```

Add the library to your Python path:

```python
import sys
sys.path.insert(0, "/path/to/GRDL")
```

### Dependencies

Core dependencies vary by module. Each module documents its own requirements.

- `numpy` -- Used across all modules
- `scipy` -- Geolocation interpolation
- `rasterio` -- GeoTIFF / raster I/O (optional)
- `sarpy` -- SICD / CPHD format support (optional)
- `requests` -- ESA MAAP catalog & download (optional)

## Credentials

BIOMASS catalog access requires an ESA MAAP offline token. Store credentials in `~/.config/geoint/credentials.json`:

```json
{
    "esa_maap": {
        "offline_token": "<your-offline-token>"
    }
}
```

Generate a token at [https://portal.maap.eo.esa.int/](https://portal.maap.eo.esa.int/) after creating an [EO Sign In](https://eoiam-idp.eo.esa.int/) account.

## Contributing

GRDL grows one well-built module at a time. When adding a new module:

1. Keep it focused -- one concept per module.
2. Make it sensor-agnostic where possible; sensor-specific logic goes in dedicated submodules.
3. Document inputs, outputs, and units in docstrings.
4. Include example usage.
5. List module-specific dependencies.
6. Add sample data to `example_images/` if needed for demos.

See [CLAUDE.md](CLAUDE.md) for full development standards, file header format, and coding conventions.

## License

MIT License -- see [LICENSE](LICENSE) for details.
