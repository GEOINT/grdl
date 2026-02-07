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
| **Image Processing** | Orthorectification, polarimetric decomposition, detection models, processor versioning | Implemented |
| **ImageJ/Fiji Ports** | 12 classic image processing algorithms ported from ImageJ/Fiji for remote sensing | Implemented |
| **Data Preparation** | Tiling, chip extraction, normalization for ML/AI pipelines | Implemented |
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
│   │   ├── sar.py                   #   SICDReader, CPHDReader, GRDReader, open_sar()
│   │   ├── biomass.py               #   BIOMASSL1Reader, open_biomass()
│   │   └── catalog.py               #   BIOMASSCatalog, load_credentials()
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
│   │   └── detection/
│   │       ├── base.py              #   ImageDetector ABC
│   │       └── models.py            #   Detection, DetectionSet, Geometry, OutputSchema
│   ├── imagej/                      # ImageJ/Fiji algorithm ports (organized by ImageJ menu)
│   │   ├── __init__.py              #   Barrel re-exports all 12 components
│   │   ├── _taxonomy.py             #   Category constants shared with GRDK
│   │   ├── filters/                 #   Process > Filters
│   │   │   ├── rank_filters.py      #     RankFilters (median, min, max, mean, variance, despeckle)
│   │   │   └── unsharp_mask.py      #     UnsharpMask (Gaussian-based sharpening)
│   │   ├── background/              #   Process > Subtract Background
│   │   │   └── rolling_ball.py      #     RollingBallBackground (Sternberg background subtraction)
│   │   ├── binary/                  #   Process > Binary
│   │   │   └── morphology.py        #     MorphologicalFilter (erode, dilate, open, close, tophat)
│   │   ├── enhance/                 #   Process > Enhance Contrast
│   │   │   ├── clahe.py             #     CLAHE (vectorized CLAHE + reference implementation)
│   │   │   └── gamma.py             #     GammaCorrection (power-law intensity transform)
│   │   ├── edges/                   #   Process > Find Edges
│   │   │   └── edge_detection.py    #     EdgeDetector (Sobel, Prewitt, Roberts, LoG, Scharr)
│   │   ├── fft/                     #   Process > FFT
│   │   │   └── fft_bandpass.py      #     FFTBandpassFilter (frequency-domain bandpass + stripe suppression)
│   │   ├── find_maxima/             #   Process > Find Maxima
│   │   │   └── find_maxima.py       #     FindMaxima (prominence-based peak/target detection)
│   │   ├── threshold/               #   Image > Adjust > Threshold
│   │   │   └── auto_local_threshold.py  #  AutoLocalThreshold (8 local thresholding methods)
│   │   ├── segmentation/            #   Plugins > Segmentation
│   │   │   └── statistical_region_merging.py  #  StatisticalRegionMerging (vectorized SRM)
│   │   └── stacks/                  #   Image > Stacks
│   │       └── z_projection.py      #     ZProjection (stack projection: max, mean, median, etc.)
│   ├── data_prep/                   # Data preparation for ML/AI pipelines
│   │   ├── __init__.py              #   Module exports (Tiler, ChipExtractor, Normalizer)
│   │   ├── base.py                  #   Base utilities
│   │   ├── tiler.py                 #   Tiler (overlapping tile extraction and reconstruction)
│   │   ├── chip_extractor.py        #   ChipExtractor (point/polygon chip extraction)
│   │   └── normalizer.py            #   Normalizer (minmax, zscore, percentile, unit_norm)
│   └── coregistration/              # Image alignment and registration
│       ├── __init__.py              #   Module exports
│       ├── base.py                  #   Base coregistration classes
│       ├── affine.py                #   Affine transform alignment
│       ├── projective.py            #   Projective transform alignment
│       ├── feature_match.py         #   Feature-based matching (OpenCV)
│       └── utils.py                 #   Coregistration utilities
├── example/                         # Example scripts
│   ├── catalog/
│   │   ├── discover_and_download.py #   BIOMASS MAAP catalog search & download
│   │   └── view_product.py          #   BIOMASS viewer with Pauli decomposition
│   └── ortho/
│       └── ortho_biomass.py         #   Orthorectification with Pauli RGB
├── ground_truth/                    # Reference data for calibration & validation
│   └── biomass_calibration_targets.geojson
├── tests/                           # Test suite
│   ├── conftest.py                              #   Shared pytest fixtures (synthetic images)
│   ├── test_io_biomass.py                       #   BIOMASS reader tests
│   ├── test_geolocation_biomass.py              #   Geolocation tests with interactive markers
│   ├── test_image_processing_ortho.py           #   Orthorectification tests
│   ├── test_image_processing_decomposition.py   #   Pauli decomposition tests
│   ├── test_image_processing_detection.py       #   Detection models & geo-registration tests
│   ├── test_image_processing_versioning.py      #   Processor versioning tests
│   ├── test_image_processing_tunable.py         #   Tunable parameter tests
│   ├── test_imagej.py                           #   ImageJ/Fiji ports (12 components)
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

### ImageJ/Fiji Algorithm Ports

12 classic image processing algorithms ported from ImageJ/Fiji, selected for relevance to remotely sensed imagery. All inherit from `ImageTransform`, carry `@processor_tags` metadata for capability discovery, and declare `__gpu_compatible__` for downstream GPU dispatch.

```python
from grdl.imagej import (
    RollingBallBackground, CLAHE, UnsharpMask, FFTBandpassFilter,
    RankFilters, MorphologicalFilter, EdgeDetector, GammaCorrection,
    FindMaxima, AutoLocalThreshold, StatisticalRegionMerging, ZProjection,
)

# Background subtraction for SAR amplitude
rb = RollingBallBackground(radius=50)
corrected = rb.apply(sar_amplitude)

# Local contrast enhancement for thermal imagery
clahe = CLAHE(block_size=127, max_slope=3.0)
enhanced = clahe.apply(thermal_band)

# Edge detection in PAN imagery
edges = EdgeDetector(method='sobel').apply(pan_image)

# Frequency-domain stripe removal from pushbroom sensor artifacts
bp = FFTBandpassFilter(suppress_stripes='horizontal', stripe_tolerance=5.0)
cleaned = bp.apply(msi_band)

# Peak/target detection in SAR amplitude
fm = FindMaxima(prominence=20.0)
targets = fm.find_peaks(sar_amplitude)  # (N, 2) array of [row, col]

# Region segmentation for land cover analysis
srm = StatisticalRegionMerging(Q=50)
labels = srm.apply(msi_band)

# Stack projection for multi-temporal composites
zp = ZProjection(method='median')
composite = zp.apply(image_stack)  # (slices, rows, cols) -> (rows, cols)
```

### Pipeline Composition

Chain multiple transforms into a single callable pipeline:

```python
from grdl.imagej import GammaCorrection, UnsharpMask, EdgeDetector
from grdl.image_processing import Pipeline

pipe = Pipeline([
    GammaCorrection(gamma=0.5),
    UnsharpMask(sigma=2.0, weight=0.6),
    EdgeDetector(method='sobel'),
])
result = pipe.apply(image)

# With progress reporting
result = pipe.apply(image, progress_callback=lambda f: print(f"{f:.0%}"))
```

### Data Preparation

Tiling, chip extraction, and normalization for ML/AI pipelines:

```python
from grdl.data_prep import Tiler, ChipExtractor, Normalizer

# Split a large image into overlapping tiles
tiler = Tiler(tile_size=256, stride=128)
tiles = tiler.tile(image)
reconstructed = tiler.untile(tiles, image.shape)

# Extract chips at point locations
extractor = ChipExtractor(chip_size=64)
chips = extractor.extract_at_points(image, points)

# Normalize to [0, 1] range
norm = Normalizer(method='minmax')
normalized = norm.normalize(image)
```

### Processor Tags & GPU Compatibility

Processors declare their capabilities for downstream discovery and dispatch:

```python
from grdl.image_processing import processor_tags

# Query processor capabilities
print(CLAHE.__processor_tags__)
# {'modalities': ('SAR', 'PAN', 'EO', 'MSI', 'HSI', 'thermal'), 'category': 'enhance', ...}

print(CLAHE.__gpu_compatible__)   # True  (pure numpy, CuPy-friendly)
print(RollingBallBackground.__gpu_compatible__)  # False (scipy dependency)
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
- `scipy` -- Geolocation interpolation, ImageJ port filters (core)
- `sarpy` -- SICD / CPHD format support (`[sar]` extra)
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
