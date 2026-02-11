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

### Use the Right Module

GRDL modules are purpose-built. Each one owns a specific responsibility. **Use them for that purpose instead of writing ad-hoc code:**

| Task | Module | Not this |
|------|--------|----------|
| Load imagery from any format | `grdl.IO` (`SICDReader`, `VIIRSReader`, `ASTERReader`, ...) | Raw `rasterio.open()` / `h5py.File()` calls |
| Plan chip regions or tile an image | `grdl.data_prep` (`ChipExtractor`, `Tiler`) | Hand-rolled `for r in range(0, rows, chunk):` loops |
| Normalize pixel values for ML | `grdl.data_prep.Normalizer` | Inline `(x - x.min()) / (x.max() - x.min())` |
| Transform pixel to lat/lon | `grdl.geolocation` (`GCPGeolocation`, etc.) | Manual interpolation of GCPs |
| Decompose polarimetric SAR | `grdl.image_processing` (`PauliDecomposition`) | Manual `(shh + svv) / sqrt(2)` arithmetic |
| Align two images | `grdl.coregistration` (`AffineCoRegistration`, ...) | Custom OpenCV `findHomography` wrappers |

This matters because the library modules handle edge cases (boundary snapping, band indexing, lazy loading, resource cleanup) that ad-hoc code typically misses. Compose them at the application level:

```python
from grdl.IO import SICDReader
from grdl.data_prep import ChipExtractor
from grdl.image_processing.sar import SublookDecomposition

# IO: load the image and metadata
with SICDReader('image.nitf') as reader:
    rows, cols = reader.get_shape()

    # data_prep: plan a center chip (handles boundary snapping)
    extractor = ChipExtractor(nrows=rows, ncols=cols)
    region = extractor.chip_at_point(rows // 2, cols // 2,
                                     row_width=2048, col_width=2048)

    # IO: read only the planned region
    chip = reader.read_chip(region.row_start, region.row_end,
                            region.col_start, region.col_end)

    # image_processing: decompose into sub-aperture looks
    sublook = SublookDecomposition(reader.metadata, num_looks=3)
    looks = sublook.decompose(chip)
```

## Module Areas

| Domain | Description | Status |
|--------|-------------|--------|
| **I/O** | Readers and writers for geospatial imagery formats | Implemented |
| | Base formats: GeoTIFF, HDF5, NITF, JPEG2000 | |
| | SAR: SICD, CPHD, CRSD, SIDD, BIOMASS | |
| | IR: ASTER (L1T, GDEM) | |
| | Multispectral: VIIRS (nightlights, vegetation, surface reflectance) | |
| | EO: scaffold (Landsat, Sentinel-2, WorldView planned) | |
| **Geolocation** | Pixel-to-geographic coordinate transforms (GCP interpolation, affine, SICD) | Implemented |
| **Image Processing** | Orthorectification, polarimetric decomposition, SAR sublook, detection models, processor versioning | Implemented |
| **Data Preparation** | Chip extraction, tiling, and normalization for ML/AI pipelines | Implemented |
| **Coregistration** | Affine, projective, and feature-matching image alignment | Implemented |
| **Sensor Processing** | Sensor-specific operations (SAR phase history, EO radiometry, MSI band math) | Planned |
| **ML/AI Utilities** | Feature extraction, annotation tools, dataset builders, and model integration helpers | Planned |

## Project Structure

```
GRDL/
├── grdl/                            # Library source
│   ├── exceptions.py                # Custom exception hierarchy (GrdlError, ValidationError, etc.)
│   ├── py.typed                     # PEP 561 type stub marker
│   ├── IO/                          # Input/Output module
│   │   ├── base.py                  #   ImageReader / ImageWriter / CatalogInterface ABCs
│   │   ├── geotiff.py               #   GeoTIFFReader (rasterio), open_image()
│   │   ├── hdf5.py                  #   HDF5Reader (h5py)
│   │   ├── jpeg2000.py              #   JP2Reader (glymur)
│   │   ├── nitf.py                  #   NITFReader (rasterio/GDAL)
│   │   ├── models/                  #   Typed metadata dataclasses
│   │   │   ├── base.py              #     ImageMetadata base class
│   │   │   ├── common.py            #     Shared primitives (XYZ, LatLonHAE, Poly2D, ...)
│   │   │   ├── sicd.py              #     SICDMetadata (~35 nested dataclasses)
│   │   │   ├── sidd.py              #     SIDDMetadata (~25 nested dataclasses)
│   │   │   ├── biomass.py           #     BIOMASSMetadata (flat typed fields)
│   │   │   ├── viirs.py             #     VIIRSMetadata (flat typed fields)
│   │   │   └── aster.py             #     ASTERMetadata (flat typed fields)
│   │   ├── sar/                     #   SAR-specific formats
│   │   │   ├── _backend.py          #     sarkit/sarpy availability detection
│   │   │   ├── sicd.py              #     SICDReader (sarkit primary, sarpy fallback)
│   │   │   ├── cphd.py              #     CPHDReader (sarkit primary, sarpy fallback)
│   │   │   ├── crsd.py              #     CRSDReader (sarkit only)
│   │   │   ├── sidd.py              #     SIDDReader (sarkit only)
│   │   │   ├── biomass.py           #     BIOMASSL1Reader, open_biomass()
│   │   │   └── biomass_catalog.py   #     BIOMASSCatalog, load_credentials()
│   │   ├── ir/                      #   IR/thermal readers
│   │   │   ├── _backend.py          #     rasterio/h5py availability detection
│   │   │   └── aster.py             #     ASTERReader (L1T, GDEM)
│   │   ├── multispectral/           #   Multispectral/hyperspectral readers
│   │   │   ├── _backend.py          #     h5py/xarray/spectral availability detection
│   │   │   └── viirs.py             #     VIIRSReader (nightlights, vegetation, reflectance)
│   │   └── eo/                      #   EO readers (scaffold)
│   │       └── _backend.py          #     rasterio/glymur availability detection
│   ├── geolocation/                 # Coordinate transform module
│   │   ├── base.py                  #   Geolocation ABC, NoGeolocation
│   │   ├── utils.py                 #   Footprint, bounds, distance helpers
│   │   ├── sar/
│   │   │   └── gcp.py               #   GCPGeolocation (Delaunay interpolation)
│   │   └── eo/                      #   EO geolocation (planned)
│   ├── image_processing/            # Image transforms module
│   │   ├── base.py                  #   ImageProcessor, ImageTransform, BandwiseTransformMixin ABCs
│   │   ├── versioning.py            #   @processor_version, @processor_tags, TunableParameterSpec
│   │   ├── pipeline.py              #   Pipeline (sequential transform composition)
│   │   ├── ortho/
│   │   │   └── ortho.py             #   Orthorectifier, OutputGrid
│   │   ├── decomposition/
│   │   │   ├── base.py              #   PolarimetricDecomposition ABC
│   │   │   └── pauli.py             #   PauliDecomposition (quad-pol)
│   │   ├── detection/
│   │   │   ├── base.py              #   ImageDetector ABC
│   │   │   └── models.py            #   Detection, DetectionSet, Geometry, OutputSchema
│   │   └── sar/                     #   SAR-specific transforms (metadata-dependent)
│   │       └── sublook.py           #   SublookDecomposition (sub-aperture splitting)
│   ├── data_prep/                   # Data preparation for ML/AI pipelines
│   │   ├── __init__.py              #   Module exports (ChipBase, ChipRegion, ChipExtractor, Tiler, Normalizer)
│   │   ├── base.py                  #   ChipBase ABC, ChipRegion NamedTuple, shared helpers
│   │   ├── tiler.py                 #   Tiler (stride-based tile region computation)
│   │   ├── chip_extractor.py        #   ChipExtractor (point-centered and whole-image chip regions)
│   │   └── normalizer.py            #   Normalizer (minmax, zscore, percentile, unit_norm)
│   └── coregistration/              # Image alignment and registration
│       ├── __init__.py              #   Module exports
│       ├── base.py                  #   Base coregistration classes
│       ├── affine.py                #   Affine transform alignment
│       ├── projective.py            #   Projective transform alignment
│       ├── feature_match.py         #   Feature-based matching (OpenCV)
│       └── utils.py                 #   Coregistration utilities
│   └── example/                     # Example scripts
│       ├── catalog/
│       │   ├── discover_and_download.py #   BIOMASS MAAP catalog search & download
│       │   └── view_product.py          #   BIOMASS viewer with Pauli decomposition
│       ├── ortho/
│       │   └── ortho_biomass.py         #   Orthorectification with Pauli RGB
│       ├── sar/
│       │   └── view_sicd.py             #   SICD magnitude viewer (linear)
│       └── image_processing/
│           └── sar/
│               └── sublook_compare.py   #   Full GRDL integration: IO + data_prep + image_processing
├── ground_truth/                    # Reference data for calibration & validation
│   └── biomass_calibration_targets.geojson
├── tests/                           # Test suite
│   ├── conftest.py                              #   Shared pytest fixtures (synthetic images)
│   ├── test_io_imports.py                       #   IO module import verification
│   ├── test_io_biomass.py                       #   BIOMASS reader tests
│   ├── test_io_geotiff.py                       #   GeoTIFF reader tests
│   ├── test_io_hdf5.py                          #   HDF5 reader tests
│   ├── test_io_ir_readers.py                    #   ASTER reader tests
│   ├── test_io_multispectral_readers.py         #   VIIRS reader tests
│   ├── test_io_models.py                        #   Metadata model tests
│   ├── test_geolocation_biomass.py              #   Geolocation tests with interactive markers
│   ├── test_image_processing_ortho.py           #   Orthorectification tests
│   ├── test_image_processing_decomposition.py   #   Pauli decomposition tests
│   ├── test_image_processing_detection.py       #   Detection models & geo-registration tests
│   ├── test_image_processing_versioning.py      #   Processor versioning tests
│   ├── test_image_processing_tunable.py         #   Tunable parameter tests
│   ├── test_image_processing_sar_sublook.py     #   SAR sublook decomposition tests
│   ├── test_coregistration.py                   #   Coregistration tests
│   └── test_benchmarks.py                       #   Performance benchmarks (pytest-benchmark)
├── example_images/                  # Small sample data for tests and demos
├── pyproject.toml                   # Package config, dependencies, and tool settings
└── CLAUDE.md                        # Development standards and coding guide
```

## Quick Start

### Reading BIOMASS SAR Data

```python
from grdl.IO import BIOMASSL1Reader
from grdl.geolocation.sar.gcp import GCPGeolocation
import numpy as np

with BIOMASSL1Reader('path/to/BIO_S3_SCS__1S_...') as reader:
    print(f"Polarizations: {reader.metadata['polarizations']}")
    rows, cols = reader.get_shape()

    # Read HH channel (complex-valued)
    hh = reader.read_chip(0, rows, 0, cols, bands=[0])
    hh_db = 20 * np.log10(np.abs(hh) + 1e-10)

    # Pixel-to-geographic coordinate transform
    geo = GCPGeolocation(
        reader.metadata['gcps'], (rows, cols),
    )
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
from grdl.geolocation.sar.gcp import GCPGeolocation
import numpy as np

with open_biomass('path/to/product') as reader:
    geo = GCPGeolocation(
        reader.metadata['gcps'],
        (reader.metadata['rows'], reader.metadata['cols']),
    )

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

### Sub-Aperture (Sublook) Decomposition

Split a complex SAR image into sub-aperture looks for coherent change detection, speckle analysis, or interferometry. Handles oversampled imagery and supports configurable overlap between sub-bands. Accepts numpy arrays or PyTorch tensors (GPU); always returns numpy.

```python
from grdl.IO.sar import SICDReader
from grdl.image_processing import SublookDecomposition

with SICDReader('image.nitf') as reader:
    image = reader.read_full()                        # complex64

    # 3 azimuth sub-looks, 10% overlap, deweight original apodization
    sublook = SublookDecomposition(
        reader.metadata, num_looks=3, overlap=0.1, deweight=True,
    )
    looks = sublook.decompose(image)                  # (3, rows, cols) complex

    # Convert to display representations
    mag = sublook.to_magnitude(looks)                 # |z|
    db = sublook.to_db(looks)                         # 20*log10(|z|)
    pwr = sublook.to_power(looks)                     # |z|^2

    # GPU-accelerated path (optional, requires torch)
    import torch
    image_gpu = torch.from_numpy(image).cuda()
    looks_gpu = sublook.decompose(image_gpu)          # returns numpy
```

### Orthorectification

```python
from grdl.image_processing import Orthorectifier, OutputGrid

geo = GCPGeolocation(reader.metadata['gcps'], (rows, cols))
grid = OutputGrid.from_geolocation(geo, pixel_size_lat=0.001,
                                   pixel_size_lon=0.001)
ortho = Orthorectifier(geo, grid, interpolation='nearest')
ortho.compute_mapping()
result = ortho.apply(hh_db, nodata=np.nan)
```

### Detection Data Models

```python
from grdl.image_processing import Detection, DetectionSet, Geometry, OutputSchema, OutputField

# Define what your detector outputs
schema = OutputSchema([
    OutputField('confidence', 'float', 'Detection confidence [0, 1]'),
    OutputField('label', 'str', 'Target class label'),
])

# Create geo-registered detections
det = Detection(
    geometry=Geometry.point(row=500, col=300, lat=-31.05, lon=116.19),
    properties={'confidence': 0.92, 'label': 'vehicle'},
)
results = DetectionSet(output_schema=schema)
results.append(det)

# Export to GeoJSON FeatureCollection
geojson = results.to_geojson()
```

### Processor Versioning & Tunable Parameters

```python
from grdl.image_processing import processor_version, TunableParameterSpec
from grdl.image_processing import ImageDetector, DetectionSet, OutputSchema

@processor_version('1.0.0')
class MyDetector(ImageDetector):
    """Versioned detector with tunable parameters."""

    @property
    def tunable_parameter_specs(self):
        return [
            TunableParameterSpec('threshold', float, default=0.5,
                                 min_value=0.0, max_value=1.0,
                                 description='Detection confidence threshold'),
        ]

    @property
    def output_schema(self):
        return OutputSchema([...])

    def detect(self, source, geolocation=None, **kwargs):
        self._validate_tunable_parameters(kwargs)
        threshold = self._get_tunable_parameter('threshold', kwargs)
        ...
```

### Data Preparation

Chip/tile index computation and normalization for ML/AI pipelines. `ChipExtractor` and `Tiler` compute index bounds only -- they never touch pixel data. Pair them with an IO reader for the actual read:

```python
from grdl.IO import GeoTIFFReader
from grdl.data_prep import Tiler, ChipExtractor, Normalizer

# -- Point-centered chip extraction with IO reader --
with GeoTIFFReader('scene.tif') as reader:
    rows, cols = reader.get_shape()
    extractor = ChipExtractor(nrows=rows, ncols=cols)

    # Plan a 256x256 chip around a target (boundary-snapped)
    region = extractor.chip_at_point(500, 1000, row_width=256, col_width=256)

    # Read only the planned region from disk
    chip = reader.read_chip(region.row_start, region.row_end,
                            region.col_start, region.col_end)

# -- Whole-image tiling for batch processing --
with GeoTIFFReader('large_scene.tif') as reader:
    rows, cols = reader.get_shape()
    extractor = ChipExtractor(nrows=rows, ncols=cols)

    for region in extractor.chip_positions(row_width=512, col_width=512):
        chip = reader.read_chip(region.row_start, region.row_end,
                                region.col_start, region.col_end)
        # Process each chip...

# -- Stride-based overlapping tiles --
tiler = Tiler(nrows=1000, ncols=2000, tile_size=256, stride=128)
tile_regions = tiler.tile_positions()

# -- Normalize to [0, 1] range --
norm = Normalizer(method='minmax')
normalized = norm.normalize(image)
```

### Processor Tags & GPU Compatibility

Processors declare their capabilities for downstream discovery and dispatch:

```python
from grdl.image_processing import processor_tags

# Query processor capabilities
print(PauliDecomposition.__processor_tags__)
# {'modalities': ('SAR',), 'category': 'decomposition', ...}

print(PauliDecomposition.__gpu_compatible__)
```

### Custom Exceptions

GRDL-specific exceptions for clean error handling:

```python
from grdl.exceptions import GrdlError, ValidationError, ProcessorError, DependencyError

try:
    result = processor.apply(bad_input)
except ValidationError as e:
    print(f"Invalid input: {e}")
except ProcessorError as e:
    print(f"Processing failed: {e}")
```

## Installation

```bash
git clone https://github.com/geoint-org/GRDL.git
cd GRDL

# Install core library
pip install -e .

# Install with optional dependencies
pip install -e ".[sar]"         # SAR format readers (sarpy)
pip install -e ".[eo]"          # EO format readers (rasterio)
pip install -e ".[biomass]"     # BIOMASS catalog (rasterio + requests)
pip install -e ".[coregistration]"  # Image alignment (opencv-python-headless)
pip install -e ".[all]"         # Everything
pip install -e ".[dev]"         # Development tools (pytest, ruff, mypy, etc.)
```

### Dependencies

Core dependencies (`numpy`, `scipy`) are installed automatically. Optional extras by module:

- `numpy` -- Used across all modules (core)
- `scipy` -- Geolocation interpolation (core)
- `sarkit` -- SICD / CPHD / CRSD / SIDD format support (primary SAR backend, `[sar]` extra)
- `sarpy` -- SICD / CPHD fallback backend (`[sar]` extra)
- `rasterio` -- GeoTIFF / raster I/O (`[eo]` / `[biomass]` extra)
- `requests` -- ESA MAAP catalog & download (`[biomass]` extra)
- `opencv-python-headless` -- Feature-matching coregistration (`[coregistration]` extra)

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
