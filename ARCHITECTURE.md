# GRDL — Library Architecture

*Modified: 2026-04-26*

## Overview

GRDL (GEOINT Rapid Development Library) is a modular Python library for
geospatial intelligence workflows. It operates on any 2D correlated
imagery — SAR, EO, MSI, hyperspectral, space-based, or terrestrial.

**This is a library, not a framework.** Every module is independently
usable. Modules compose at the application level — each does its job,
the application wires them together.

**Scale:** ~63,000 lines of library code across 165 Python files, plus
~29,000 lines of tests. Nine primary domains.

---

## Module Map

```
grdl/                            ~60k lines, 152 files
│
├── IO/                          Input/Output — format readers and writers
│   ├── base.py                    ImageReader, ImageWriter, CatalogInterface ABCs
│   ├── models/                    Typed metadata dataclasses (SICD, SIDD, CPHD, ...)
│   ├── sar/                       SAR readers (SICD, CPHD, CRSD, SIDD, Sentinel-1, BIOMASS, ...)
│   ├── eo/                        EO readers (Sentinel-2, NITF RPC/RSM)
│   ├── ir/                        IR/thermal readers (ASTER)
│   ├── multispectral/             MSI readers (VIIRS)
│   ├── catalog/                   Remote query, download, SQLite cataloging
│   ├── geotiff.py                 GeoTIFF reader/writer
│   ├── hdf5.py                    HDF5 reader/writer
│   ├── generic.py                 GDAL fallback reader, open_any()
│   └── probe.py                   Format sniffing
│
├── geolocation/                 Pixel ↔ geographic coordinate transforms
│   ├── base.py                    Geolocation ABC, NoGeolocation, DEM refinement
│   ├── projection.py              R/Rdot engine (COAProjection, formation projectors)
│   ├── coordinates.py             geodetic ↔ ECEF ↔ ENU conversions
│   ├── utils.py                   Footprint, bounds, distance helpers
│   ├── sar/                       SAR geolocation (SICD, SIDD, Sentinel-1, NISAR, GCP)
│   ├── eo/                        EO geolocation (Affine, RPC, RSM)
│   └── elevation/                 Terrain models (DTED, GeoTIFF DEM, geoid, constant)
│
├── image_processing/            Image transforms, detection, formation
│   ├── base.py                    ImageProcessor, ImageTransform, BandwiseTransformMixin
│   ├── params.py                  Range, Options, Desc, ParamSpec (tunable parameters)
│   ├── versioning.py              @processor_version, @processor_tags
│   ├── pipeline.py                Pipeline (sequential composition)
│   ├── intensity.py               ToDecibels, PercentileStretch
│   ├── ortho/                     Orthorectification (orthorectify(), GeographicGrid, ENUGrid, UTMGrid, WebMercatorGrid)
│   ├── filters/                   Spatial filters (mean, gaussian, median, Lee, phase)
│   ├── decomposition/             Polarimetric (Pauli quad-pol, DualPolHAlpha)
│   ├── detection/                 CFAR detectors (CA, GO, SO, OS)
│   └── sar/                       SAR-specific (sublook, CSI, dominance, image formation)
│
├── interpolation/               1D bandwidth-preserving interpolation kernels
│   └── [Lanczos, KaiserSinc, Lagrange, Farrow, Polyphase, Thiran]
│
├── data_prep/                   Index-only chip/tile planning + normalization
│   └── [ChipExtractor, Tiler, Normalizer]
│
├── coregistration/              Image-to-image alignment
│   └── [Affine, Projective, FeatureMatch]
│
├── contrast/                    Display-time dynamic range adjustment
│   ├── auto.py                    auto_select(metadata) modality dispatcher
│   ├── density.py                 MangisDensity, Brighter, Darker, HighContrast, GDM, PEDF
│   ├── nrl.py                     NRLStretch (linear→log knee remap)
│   ├── linear.py                  LinearStretch
│   ├── logarithmic.py             LogStretch (bounded log2)
│   ├── gamma.py                   GammaCorrection
│   ├── sigmoid.py                 SigmoidStretch
│   ├── histogram.py               HistogramEqualization, CLAHE (skimage)
│   └── [percentile.py, decibel.py re-exports]
│
├── vector/                      Geo-registered feature data, spatial operators
│   └── [Feature, FeatureSet, BufferOperator, IntersectionOperator, ...]
│
├── transforms/                  Detection geometry transforms
│
├── exceptions.py                GrdlError → ValidationError, ProcessorError, ...
└── vocabulary.py                ImageModality, ProcessorCategory, DetectionType, ...
```

---

## ABC Hierarchy

Every domain defines contracts via abstract base classes. Concrete
implementations inherit and specialize. Sensor-specific logic goes in
modality submodules (`sar/`, `eo/`, `ir/`, `multispectral/`).

```
ImageReader (ABC)                              IO/base.py
├── GeoTIFFReader, HDF5Reader, NITFReader        base format readers
├── SICDReader, CPHDReader, SIDDReader, ...       IO/sar/
├── Sentinel2Reader, EONITFReader                 IO/eo/
├── ASTERReader                                   IO/ir/
└── VIIRSReader                                   IO/multispectral/

ImageWriter (ABC)                              IO/base.py
├── GeoTIFFWriter, HDF5Writer, NITFWriter
├── SICDWriter, SIDDWriter
├── NumpyWriter, PngWriter
└── write()                                     convenience function

Geolocation (ABC)                              geolocation/base.py
├── SICDGeolocation                               R/Rdot (native/sarpy/sarkit)
├── SIDDGeolocation                               Plane/Geographic/Cylindrical
├── GCPGeolocation                                Delaunay on GCPs
├── NISARGeolocation                              Bivariate spline
├── Sentinel1SLCGeolocation                       Annotation grid spline
├── AffineGeolocation                             Affine + CRS reprojection
├── RPCGeolocation                                RPC00B rational polynomials
├── RSMGeolocation                                RSM variable-order polynomials
└── NoGeolocation                                 Fallback (raises)

ElevationModel (ABC)                           geolocation/elevation/base.py
├── ConstantElevation                             Fixed height
├── DTEDElevation                                 DTED tiles (bicubic default, cross-tile stitching)
└── GeoTIFFDEM                                    GeoTIFF DEM (bicubic default)

ImageProcessor (ABC)                           image_processing/base.py
├── ImageTransform (ABC)                          dense raster → raster
│   ├── BandwiseTransformMixin                      auto band iteration
│   │   ├── MeanFilter, GaussianFilter, MedianFilter, ...
│   │   ├── LeeFilter, ComplexLeeFilter             SAR speckle
│   │   └── PhaseGradientFilter
│   ├── ToDecibels, PercentileStretch
│   ├── Orthorectifier                              @processor_version('0.1.0')
│   └── Pipeline                                    sequential composition
├── ImageDetector (ABC)                           raster → DetectionSet
│   └── CFARDetector (ABC, template method)
│       ├── CACFARDetector, GOCFARDetector
│       ├── SOCFARDetector, OSCFARDetector
├── PolarimetricDecomposition (ABC)
│   ├── PauliDecomposition (quad-pol)
│   └── DualPolHAlpha (dual-pol)
├── SublookDecomposition, MultilookDecomposition
├── CSIProcessor
└── ImageFormationAlgorithm (ABC)
    ├── PolarFormatAlgorithm, StripmapPFA
    ├── RangeDopplerAlgorithm
    └── FastBackProjection

Interpolator (ABC)                             interpolation/base.py
├── LanczosInterpolator
├── KaiserSincInterpolator
├── LagrangeInterpolator
├── FarrowInterpolator
├── PolyphaseInterpolator
└── ThiranDelayFilter

CoRegistration (ABC)                           coregistration/
├── AffineCoRegistration
├── ProjectiveCoRegistration
└── FeatureMatchCoRegistration (OpenCV)

ImageTransform (ABC, contrast operators)      contrast/
├── MangisDensity, Brighter, Darker, HighContrast      sarpy Density family
├── GDM (Generalized Density Mapping)
├── PEDF (Piecewise Extended Density Format)
├── NRLStretch                                          sarpy NRL port
├── LinearStretch, LogStretch                           sarpy Linear/Log ports
├── GammaCorrection, SigmoidStretch
├── HistogramEqualization, CLAHE                        global + adaptive HE
└── PercentileStretch, ToDecibels                       re-exports from intensity.py

VectorProcessor (ABC)                          vector/base.py
├── BufferOperator, IntersectionOperator
├── UnionOperator, DissolveOperator
├── SpatialJoinOperator, ClipOperator
├── CentroidOperator, ConvexHullOperator
└── RasterToPoints, Rasterize                  raster ↔ vector conversion
```

---

## Cross-Module Dependency Graph

```
exceptions.py, vocabulary.py              ← no deps (foundation layer)
        │
    IO/models/                            ← metadata dataclasses
        │
    IO/base, IO/readers, IO/writers       ← depend on models
        │
    geolocation/base, coordinates         ← depend on IO/models
        │
    geolocation/sar, eo, elevation        ← depend on base
        │
    image_processing/base, params         ← depend on vocabulary
        │
    image_processing/filters, detection   ← depend on base
    image_processing/decomposition
        │
    image_processing/ortho                ← depends on geolocation + base
    image_processing/sar/image_formation  ← depends on geolocation + IO/models
        │
    data_prep                             ← orthogonal (used by ortho, multilook)
    interpolation                         ← orthogonal (used by image formation)
    coregistration                        ← orthogonal (uses OpenCV)
    transforms                            ← depends on coregistration + detection
```

**Key integration points:**
- **Orthorectification** requires `Geolocation` + optional `ElevationModel` + optional `ImageReader`
- **SAR image formation** requires `SICDMetadata`/`CPHDMetadata` + `Interpolator`
- **Detection pipeline** requires `Geolocation` for geo-registration of detections
- **Data prep** (`Tiler`) used by `OrthoBuilder` tiling and `MultilookDecomposition`

---

## Key Design Patterns

### 1. Stacked ndarray API (Scalar/Array Dispatch)

All geolocation and coordinate methods return stacked ndarrays.
Subclasses implement only the vectorized `_*_array()` method; the
base class handles dispatch and stacking.

```python
# Scalar — returns (3,) ndarray [lat, lon, h]; tuple unpacking works
lat, lon, h = geo.image_to_latlon(500, 1000)

# Stacked (N, 2) input — returns (N, 3) ndarray
result = geo.image_to_latlon(pixels_Nx2)

# Inverse: scalar → (2,) [row, col]; stacked (N, 3) → (N, 2)
row, col = geo.latlon_to_image(lat, lon)
pixels = geo.latlon_to_image(coords_Nx3)

# Coordinate functions: (N, 3) in → (N, 3) out
ecef = geodetic_to_ecef(pts_Nx3)
geo_pts = ecef_to_geodetic(ecef_Nx3)
```

### 2. Processor Metadata System

Every `ImageProcessor` subclass declares version, capabilities, and
tunable parameters via decorators and `Annotated` type hints:

```python
@processor_version('1.0.0')
@processor_tags(modalities=[IM.SAR], category=PC.FILTERS,
                description='Adaptive speckle filter')
class LeeFilter(BandwiseTransformMixin, ImageTransform):
    kernel_size: Annotated[int, Range(min=3, max=101),
                           Desc('Kernel size')] = 7
```

- `__init_subclass__` collects specs into `__param_specs__`
- Auto-generates `__init__` if none defined
- `_resolve_params(kwargs)` merges defaults with runtime overrides
- grdl-runtime reads `__param_specs__` for catalog; grdk for UI controls

### 3. Backend Probing

Optional dependencies (`sarpy`, `rasterio`, `h5py`, `pyproj`, `torch`,
`numba`, `opencv`) are probed at import time via `_backend.py` modules.
Missing packages produce clear `ImportError` at construction, not silent
degradation.

### 4. Functions over Fluent Chaining

Public APIs prefer **keyword-argument functions** over fluent builder
patterns, following NumPy/SciPy conventions:

```python
geo.elevation = dem
result = orthorectify(
    geolocation=geo,
    reader=reader,
    interpolation='bilinear',
)
```

`OrthoBuilder` exists for advanced cases (partial config, reuse) but
`orthorectify()` is the recommended entry point.

### 5. Composition at the Application Level

Each module does one job. The application wires them together:

```python
# IO reads → geolocation transforms → image_processing orthorectifies
with SICDReader(path) as reader:
    geo = SICDGeolocation.from_reader(reader)
    geo.elevation = open_elevation('/data/dted/')
    result = orthorectify(
        geolocation=geo, reader=reader,
    )
```

### 6. Fail Fast

- All imports at file top — missing dependency fails on import
- Input validation in constructors — clear exceptions before work
- No silent type coercion or error swallowing

### 7. No Global State

No singletons, module-level mutable state, or side effects on import.

---

## Metadata Model

Typed dataclasses in `IO/models/` capture sensor-specific metadata
parsed from file headers. Common primitives are shared:

| Type | Purpose |
|------|---------|
| `XYZ` | 3D coordinate with vector math (`__add__`, `__sub__`, `dot`, `norm`) |
| `Poly1D`, `Poly2D` | Callable polynomial evaluation |
| `XYZPoly` | 3D polynomial (evaluates to `XYZ`) |
| `LatLon`, `LatLonHAE`, `RowCol` | Typed coordinate pairs |

Sensor models:

| Model | Sensor | Key contents |
|-------|--------|-------------|
| `SICDMetadata` | Complex SAR | ~35 nested dataclasses (grid, scpcoa, timeline, ...) |
| `SIDDMetadata` | Detected SAR | Measurement, display, geographic data |
| `CPHDMetadata` | Phase history | Channel, dwell, antenna patterns |
| `BIOMASSMetadata` | ESA BIOMASS | GCPs, orbit, polarizations |
| `EONITFMetadata` | EO NITF | `RPCCoefficients`, `RSMCoefficients` |
| `Sentinel1SLCMetadata` | Sentinel-1 | Annotation, calibration, orbit state |
| `NISARMetadata` | NISAR | RSLC/GSLC geolocation grids |

---

## Optional Dependencies

| Extra | Packages | Modules enabled |
|-------|----------|----------------|
| `sar` | sarpy, sarkit | IO/sar (SICD, CPHD, CRSD, SIDD) |
| `eo` | rasterio, glymur | IO/geotiff, IO/jpeg2000, IO/eo |
| `hdf5` | h5py | IO/hdf5 |
| `multispectral` | h5py, xarray, spectral | IO/multispectral |
| `geolocation` | pyproj | geolocation/eo, geolocation/elevation |
| `coregistration` | opencv-python-headless | coregistration/feature_match |
| `contrast` | scikit-image | contrast/histogram (CLAHE) |
| `all` | everything above | full installation |

Core (always required): `numpy>=1.20.0`, `scipy>=1.7.0`

---

## Sub-Module Architecture Documents

Each major module has its own detailed architecture document:

| Document | Covers |
|----------|--------|
| [IO/ARCHITECTURE.md](grdl/IO/ARCHITECTURE.md) | Readers, writers, metadata models, format detection |
| [geolocation/ARCHITECTURE.md](grdl/geolocation/ARCHITECTURE.md) | Coordinate transforms, R/Rdot engine, DEM integration |
| [image_processing/ARCHITECTURE.md](grdl/image_processing/ARCHITECTURE.md) | Filters, detection, decomposition, SAR formation, processor metadata |
| [image_processing/ortho/ARCHITECTURE.md](grdl/image_processing/ortho/ARCHITECTURE.md) | Orthorectifier, output grids, resampling backends, tiled pipeline |
| [contrast/ARCHITECTURE.md](grdl/contrast/ARCHITECTURE.md) | Display contrast operators, sarpy port, stats threading, modality dispatch |

---

## Testing

Tests live in `tests/` (~29,000 lines). Conventions:

- Pattern: `test_<domain>_<module>.py`
- Framework: `pytest` (no `unittest.TestCase`)
- Data: small synthetic arrays, not real imagery
- Benchmarks: `pytest-benchmark` (skip with `--benchmark-disable`)

```bash
pytest tests/ -v --benchmark-disable       # full suite
pytest tests/test_ortho_builder.py -v      # single module
```

---

## Packaging

- **Source of truth:** `pyproject.toml` (version, deps, extras)
- **PyPI:** published via GitHub Release → `publish.yml` (OIDC trusted publishing)
- **Package data:** `.md` files included via `[tool.setuptools.package-data]` + `MANIFEST.in`
- **Current version:** 0.5.0
