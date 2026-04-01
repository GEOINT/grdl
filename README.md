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
| Load imagery from any format | `grdl.IO` (`SICDReader`, `NISARReader`, `VIIRSReader`, `Sentinel2Reader`, `EONITFReader`, ...) | Raw `rasterio.open()` / `h5py.File()` calls |
| Write imagery to disk | `grdl.IO` (`GeoTIFFWriter`, `SICDWriter`, `SIDDWriter`, `NumpyWriter`, `PngWriter`) | Raw `rasterio` / `h5py` write calls |
| Open any supported format | `grdl.IO.generic.open_any()` | Manual format detection |
| Plan chip regions or tile an image | `grdl.data_prep` (`ChipExtractor`, `Tiler`) | Hand-rolled `for r in range(0, rows, chunk):` loops |
| Normalize pixel values for ML | `grdl.data_prep.Normalizer` | Inline `(x - x.min()) / (x.max() - x.min())` |
| Transform pixel to lat/lon | `grdl.geolocation` (`AffineGeolocation`, `SICDGeolocation`, `RPCGeolocation`, `RSMGeolocation`, ...) | Manual interpolation of GCPs or affine math |
| Coordinate system conversion | `grdl.geolocation.coordinates` (geodetic/ECEF/ENU) | Manual WGS84 math |
| Terrain elevation lookup | `grdl.geolocation.elevation` (`DTEDElevation`, `GeoTIFFDEM`) | Raw `rasterio.open()` on DEM tiles |
| Decompose polarimetric SAR | `grdl.image_processing` (`PauliDecomposition`, `DualPolHAlpha`) | Manual `(shh + svv) / sqrt(2)` arithmetic |
| Sub-aperture dominance detection | `grdl.image_processing.sar.dominance` (`compute_dominance`) | Manual sublook power ratios |
| CSI RGB composite | `grdl.image_processing.sar.CSIProcessor` | Ad-hoc HSV mapping |
| CFAR target detection | `grdl.image_processing.detection.cfar` (CA, GO, SO, OS variants) | Manual threshold loops |
| SAR image formation | `grdl.image_processing.sar.image_formation` (PFA, RDA, FFBP) | Custom FFT-based pipelines |
| Interpolation / resampling | `grdl.interpolation` (`PolyphaseInterpolator`, `LanczosInterpolator`, ...) | Manual sinc convolution |
| Align two images | `grdl.coregistration` (`AffineCoRegistration`, ...) | Custom OpenCV `findHomography` wrappers |
| Transform detection geometries | `grdl.transforms` (`transform_detection_set`) | Manual coordinate mapping |

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
| | Base formats: GeoTIFF, HDF5, NITF, JPEG2000, NumPy, PNG | |
| | SAR: SICD, CPHD, CRSD, SIDD, BIOMASS, Sentinel-1 SLC, TerraSAR-X, NISAR | |
| | IR: ASTER (L1T, GDEM) | |
| | Multispectral: VIIRS (nightlights, vegetation, surface reflectance) | |
| | EO: Sentinel-2, EO NITF (WorldView, GeoEye, Pleiades, aerial) | |
| | Writers: SICD, SIDD, GeoTIFF, HDF5, NITF, NumPy, PNG | |
| | Generic: `GDALFallbackReader` (`open_any()`), `InvasiveProbeReader` | |
| **Geolocation** | Image-to-geographic coordinate transforms with DEM integration | Implemented |
| | EO: `AffineGeolocation` (geocoded rasters via affine + pyproj) | |
| | SAR: `SICDGeolocation` (native R/Rdot + sarpy), `SIDDGeolocation`, `GCPGeolocation`, `NISARGeolocation`, `Sentinel1SLCGeolocation` | |
| | EO: `AffineGeolocation`, `RPCGeolocation` (RPC00B), `RSMGeolocation` (RSMPCA) | |
| | Projection: `COAProjection` (native R/Rdot engine), `image_to_ground_hae`, `image_to_ground_dem`, `ground_to_image` | |
| | Elevation: `ElevationModel` ABC, `DTEDElevation`, `GeoTIFFDEM`, `ConstantElevation`, `GeoidCorrection` | |
| | Coordinates: `geodetic_to_ecef`, `ecef_to_geodetic`, `geodetic_to_enu`, `enu_to_geodetic` | |
| **Image Processing** | Orthorectification, polarimetric decomposition, SAR sublook, CSI, dominance features, detection models, CFAR detectors, image formation, processor versioning & metadata | Implemented |
| | Ortho: `Orthorectifier`, `GeographicGrid`, `ENUGrid`, `UTMGrid`, `WebMercatorGrid`, accelerated resampling | |
| | SAR: `SublookDecomposition`, `CSIProcessor`, `DominanceFeatures`, `compute_dominance`, `compute_sublook_entropy` | |
| | Image Formation: `PolarFormatAlgorithm`, `RangeDopplerAlgorithm`, `FastBackProjection`, `StripmapPFA` | |
| | Detection: `Detection`, `DetectionSet`, CFAR variants (CA, GO, SO, OS) | |
| | Decomposition: `PauliDecomposition`, `DualPolHAlpha` | |
| **Interpolation** | 1D bandwidth-preserving interpolation kernels for SAR image formation | Implemented |
| | `LanczosInterpolator`, `KaiserSincInterpolator`, `LagrangeInterpolator`, `FarrowInterpolator`, `PolyphaseInterpolator`, `ThiranDelayFilter` | |
| **Data Preparation** | Chip extraction, tiling, and normalization for ML/AI pipelines | Implemented |
| **Coregistration** | Affine, projective, and feature-matching image alignment | Implemented |
| **Transforms** | Detection geometry transforms (apply coregistration to vector detections) | Implemented |
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
│   │   ├── geotiff.py               #   GeoTIFFReader, GeoTIFFWriter (rasterio)
│   │   ├── hdf5.py                  #   HDF5Reader, HDF5Writer (h5py)
│   │   ├── jpeg2000.py              #   JP2Reader (glymur)
│   │   ├── nitf.py                  #   NITFReader, NITFWriter (rasterio/GDAL)
│   │   ├── numpy_io.py              #   NumpyWriter (.npy / .npz)
│   │   ├── png.py                   #   PngWriter
│   │   ├── generic.py               #   GDALFallbackReader, open_any()
│   │   ├── probe.py                 #   InvasiveProbeReader (format sniffing)
│   │   ├── models/                  #   Typed metadata dataclasses
│   │   │   ├── base.py              #     ImageMetadata base class
│   │   │   ├── common.py            #     Shared primitives (XYZ, LatLonHAE, Poly2D, ...)
│   │   │   ├── sicd.py              #     SICDMetadata (~35 nested dataclasses)
│   │   │   ├── sidd.py              #     SIDDMetadata (~25 nested dataclasses)
│   │   │   ├── cphd.py              #     CPHDMetadata
│   │   │   ├── biomass.py           #     BIOMASSMetadata (flat typed fields)
│   │   │   ├── viirs.py             #     VIIRSMetadata (flat typed fields)
│   │   │   ├── aster.py             #     ASTERMetadata (flat typed fields)
│   │   │   ├── sentinel1_slc.py     #     Sentinel1SLCMetadata
│   │   │   ├── sentinel2.py         #     Sentinel2Metadata
│   │   │   ├── terrasar.py          #     TerraSARMetadata
│   │   │   ├── nisar.py             #     NISARMetadata
│   │   │   └── eo_nitf.py           #     EONITFMetadata, RPCCoefficients, RSMCoefficients
│   │   ├── sar/                     #   SAR-specific formats
│   │   │   ├── _backend.py          #     sarkit/sarpy availability detection
│   │   │   ├── sicd.py              #     SICDReader (sarkit primary, sarpy fallback)
│   │   │   ├── sicd_writer.py       #     SICDWriter
│   │   │   ├── cphd.py              #     CPHDReader (sarkit primary, sarpy fallback)
│   │   │   ├── crsd.py              #     CRSDReader (sarkit only)
│   │   │   ├── sidd.py              #     SIDDReader (sarkit only)
│   │   │   ├── sidd_writer.py       #     SIDDWriter
│   │   │   ├── biomass.py           #     BIOMASSL1Reader, open_biomass()
│   │   │   ├── biomass_catalog.py   #     BIOMASSCatalog, load_credentials()
│   │   │   ├── sentinel1_slc.py     #     Sentinel1SLCReader
│   │   │   ├── terrasar.py          #     TerraSARReader, open_terrasar()
│   │   │   └── nisar.py             #     NISARReader, open_nisar()
│   │   ├── ir/                      #   IR/thermal readers
│   │   │   ├── _backend.py          #     rasterio/h5py availability detection
│   │   │   └── aster.py             #     ASTERReader (L1T, GDEM)
│   │   ├── multispectral/           #   Multispectral/hyperspectral readers
│   │   │   ├── _backend.py          #     h5py/xarray/spectral availability detection
│   │   │   └── viirs.py             #     VIIRSReader (nightlights, vegetation, reflectance)
│   │   └── eo/                      #   EO readers
│   │       ├── _backend.py          #     rasterio/glymur availability detection
│   │       ├── sentinel2.py         #     Sentinel2Reader
│   │       └── nitf.py              #     EONITFReader (RPC/RSM extraction)
│   │   └── catalog/                 #   Remote query, download & cataloging
│   │       ├── remote_utils.py      #     Shared credentials, token auth
│   │       ├── biomass_catalog.py   #     BIOMASSCatalog (ESA MAAP STAC)
│   │       └── sentinel1_catalog.py #     Sentinel1SLCCatalog (CDSE OData)
│   ├── geolocation/                 # Coordinate transform module
│   │   ├── base.py                  #   Geolocation ABC, NoGeolocation
│   │   ├── utils.py                 #   Footprint, bounds, distance helpers
│   │   ├── coordinates.py           #   geodetic_to_ecef, ecef_to_geodetic, geodetic_to_enu, enu_to_geodetic
│   │   ├── projection.py            #   COAProjection, image_to_ground_hae/dem, ground_to_image
│   │   ├── sar/
│   │   │   ├── gcp.py               #   GCPGeolocation (Delaunay interpolation)
│   │   │   ├── sicd.py              #   SICDGeolocation (SICD imagery via sarpy/sarkit)
│   │   │   ├── sidd.py              #   SIDDGeolocation (SIDD imagery)
│   │   │   ├── nisar.py             #   NISARGeolocation
│   │   │   └── sentinel1_slc.py     #   Sentinel1SLCGeolocation
│   │   ├── eo/
│   │   │   ├── affine.py            #   AffineGeolocation (geocoded rasters, affine + pyproj)
│   │   │   ├── rpc.py               #   RPCGeolocation (RPC00B rational polynomials)
│   │   │   └── rsm.py               #   RSMGeolocation (RSMPCA replacement sensor model)
│   │   └── elevation/               #   Terrain elevation models
│   │       ├── base.py              #   ElevationModel ABC
│   │       ├── constant.py          #   ConstantElevation (fixed-height fallback)
│   │       ├── dted.py              #   DTEDElevation (DTED tiles, bicubic, cross-tile stitching)
│   │       ├── geotiff_dem.py       #   GeoTIFFDEM (GeoTIFF DEM via rasterio)
│   │       └── geoid.py             #   GeoidCorrection (EGM96 geoid undulation)
│   ├── image_processing/            # Image transforms module
│   │   ├── base.py                  #   ImageProcessor, ImageTransform, BandwiseTransformMixin ABCs
│   │   ├── params.py                #   Range, Options, Desc, ParamSpec (Annotated constraint markers)
│   │   ├── versioning.py            #   @processor_version, @processor_tags, DetectionInputSpec
│   │   ├── pipeline.py              #   Pipeline (sequential transform composition)
│   │   ├── ortho/
│   │   │   ├── ortho.py             #   OutputGridProtocol, GeographicGrid, Orthorectifier
│   │   │   ├── ortho_builder.py     #   OrthoBuilder, OrthoResult
│   │   │   ├── enu_grid.py          #   ENUGrid (local East-North-Up grid)
│   │   │   ├── utm_grid.py          #   UTMGrid (UTM projection grid)
│   │   │   ├── web_mercator_grid.py #   WebMercatorGrid (Web Mercator projection grid)
│   │   │   ├── accelerated.py       #   resample(), detect_backend() (accelerated resampling)
│   │   │   └── resolution.py        #   compute_output_resolution (auto pixel spacing)
│   │   ├── decomposition/
│   │   │   ├── base.py              #   PolarimetricDecomposition ABC
│   │   │   ├── pauli.py             #   PauliDecomposition (quad-pol)
│   │   │   └── dual_pol.py          #   DualPolHAlpha (dual-pol H/Alpha)
│   │   ├── detection/
│   │   │   ├── base.py              #   ImageDetector ABC
│   │   │   ├── models.py            #   Detection, DetectionSet, Geometry, OutputSchema
│   │   │   ├── fields.py            #   Data dictionary (Fields.sar, Fields.physical, etc.)
│   │   │   └── cfar/                #   CFAR detector variants
│   │   │       ├── _base.py         #     CFARDetector ABC
│   │   │       ├── ca.py            #     CACFARDetector (Cell-Averaging)
│   │   │       ├── go.py            #     GOCFARDetector (Greatest-Of)
│   │   │       ├── so.py            #     SOCFARDetector (Smallest-Of)
│   │   │       └── os.py            #     OSCFARDetector (Ordered-Statistics)
│   │   └── sar/                     #   SAR-specific transforms (metadata-dependent)
│   │       ├── sublook.py           #   SublookDecomposition (sub-aperture splitting)
│   │       ├── csi.py               #   CSIProcessor (Coherent Shape Index RGB composite)
│   │       ├── dominance.py         #   DominanceFeatures, compute_dominance, compute_sublook_entropy
│   │       └── image_formation/     #   SAR image formation algorithms
│   │           ├── pfa.py           #     PolarFormatAlgorithm
│   │           ├── rda.py           #     RangeDopplerAlgorithm
│   │           ├── ffbp.py          #     FastBackProjection
│   │           └── stripmap_pfa.py  #     StripmapPFA
│   ├── interpolation/               # 1D bandwidth-preserving interpolation kernels
│   │   ├── base.py                  #   Interpolator, KernelInterpolator ABCs
│   │   ├── lanczos.py               #   LanczosInterpolator
│   │   ├── windowed_sinc.py         #   KaiserSincInterpolator
│   │   ├── lagrange.py              #   LagrangeInterpolator
│   │   ├── farrow.py                #   FarrowInterpolator
│   │   ├── polyphase.py             #   PolyphaseInterpolator
│   │   └── thiran.py                #   ThiranDelayFilter (IIR allpass)
│   ├── transforms/                  # Detection geometry transforms
│   │   └── detection.py             #   transform_pixel_geometry, transform_detection, transform_detection_set
│   ├── data_prep/                   # Data preparation for ML/AI pipelines
│   │   ├── base.py                  #   ChipBase ABC, ChipRegion NamedTuple, shared helpers
│   │   ├── tiler.py                 #   Tiler (stride-based tile region computation)
│   │   ├── chip_extractor.py        #   ChipExtractor (point-centered and whole-image chip regions)
│   │   └── normalizer.py            #   Normalizer (minmax, zscore, percentile, unit_norm)
│   ├── coregistration/              # Image alignment and registration
│   │   ├── base.py                  #   Base coregistration classes
│   │   ├── affine.py                #   Affine transform alignment
│   │   ├── projective.py            #   Projective transform alignment
│   │   ├── feature_match.py         #   Feature-based matching (OpenCV)
│   │   └── utils.py                 #   Coregistration utilities
│   └── example/                     # Example scripts
│       ├── catalog/
│       │   ├── discover_and_download.py #   BIOMASS MAAP catalog search & download
│       │   └── view_product.py          #   BIOMASS viewer with Pauli decomposition
│       ├── geolocation/
│       │   └── geolocation_overlay.py   #   Geolocation overlay visualization
│       ├── interpolation/
│       │   ├── polyphaseinterpolation.py #   Polyphase interpolation demo
│       │   └── lfm_polyphase.py         #   LFM chirp polyphase resampling
│       ├── IO/
│       │   ├── sar/
│       │   │   └── view_sicd.py         #   SICD magnitude viewer (linear)
│       │   ├── eo/
│       │   │   └── view_sentinel2.py    #   Sentinel-2 viewer
│       │   ├── HDF5/
│       │   │   └── load_earthdata.py    #   HDF5 EarthData loader
│       │   └── test_file_loading.py     #   Generic file loading test
│       ├── ortho/
│       │   ├── chip_ortho.py            #   Ground-extent chip + ENU ortho
│       │   ├── compare_sidd_ortho.py    #   Dual-SIDD ortho comparison + coregistration
│       │   ├── ortho_biomass.py         #   Orthorectification with Pauli RGB
│       │   ├── ortho_combined.py        #   Combined SICD/SIDD auto-detect ortho
│       │   ├── ortho_sicd.py            #   SICD orthorectification
│       │   └── ortho_sidd.py            #   SIDD orthorectification
│       └── image_processing/
│           ├── filtering/
│           │   └── phase_gradient.py    #   Phase gradient filter demo
│           └── sar/
│               ├── sublook_compare.py       #   IO + data_prep + sublook integration
│               ├── csi_detection_overlay.py #   CSI + dominance detection overlay
│               ├── ifp_example.py           #   Image formation (PFA) example
│               ├── ffbp_stripmap_example.py #   FFBP stripmap formation
│               ├── rda_stripmap_example.py  #   RDA stripmap formation
│               ├── dump_cphd_metadata.py    #   CPHD metadata inspector
│               └── detection_workflow/
│                   └── csi_detect_workflow.py #   Full CSI detection workflow
├── ground_truth/                    # Reference data for calibration & validation
│   └── biomass_calibration_targets.geojson
├── tests/                           # Test suite (70 test files)
│   ├── conftest.py                              #   Shared pytest fixtures
│   ├── test_io_*.py                             #   IO reader/writer tests
│   ├── test_geolocation_*.py                    #   Geolocation tests
│   ├── test_image_processing_*.py               #   Image processing tests
│   ├── test_interpolation*.py                   #   Interpolation tests
│   ├── test_transforms_detection.py             #   Transform tests
│   ├── test_coregistration.py                   #   Coregistration tests
│   ├── test_integration.py                      #   Full integration tests
│   └── test_benchmarks.py                       #   Performance benchmarks (pytest-benchmark)
├── example_images/                  # Small sample data for tests and demos
├── pyproject.toml                   # Package config, dependencies, and tool settings
├── environment.yml                  # Conda environment specification (conda-forge)
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
    lat, lon, _ = geo.image_to_latlon(rows // 2, cols // 2)
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

All geolocation classes share the same `Geolocation` ABC and return stacked ndarrays: scalar calls return a 1-D array (e.g., `(3,)` for `[lat, lon, h]`) that supports tuple unpacking, and batch calls accept an `(N, 2)` stacked ndarray and return `(N, 3)`. The ABC constructor accepts optional `dem_path`, `geoid_path`, and `interpolation` (DEM spline order: 1=bilinear, 3=bicubic, 5=quintic) parameters for DEM integration. All subclasses use the same base-class methods for height resolution and NaN fill, ensuring consistent terrain-corrected behavior across SICD, SIDD, RPC, RSM, and geocoded rasters.

```python
from grdl.geolocation.sar.gcp import GCPGeolocation
from grdl.geolocation.sar.sicd import SICDGeolocation
from grdl.geolocation.eo.affine import AffineGeolocation
import numpy as np

# --- GCPGeolocation (BIOMASS SAR via Delaunay interpolation) ---
from grdl.IO import open_biomass

with open_biomass('path/to/product') as reader:
    geo = GCPGeolocation(
        reader.metadata['gcps'],
        (reader.metadata['rows'], reader.metadata['cols']),
    )

    # Single pixel — returns (3,) ndarray [lat, lon, h]; tuple unpacking works
    lat, lon, height = geo.image_to_latlon(500, 1000)

    # Batch of pixels — pass (N, 2) stacked array, returns (N, 3)
    pixels = np.array([[100, 400], [200, 500], [300, 600]])
    coords = geo.image_to_latlon(pixels)   # shape (3, 3) — each row is [lat, lon, h]

    # Inverse: geographic to pixel — scalar returns (2,) [row, col]
    row, col = geo.latlon_to_image(-31.05, 116.19)

# --- AffineGeolocation (geocoded rasters: GeoTIFF, SAR GRD, etc.) ---
geo = AffineGeolocation('scene.tif')
lat, lon, height = geo.image_to_latlon(100, 200)

# --- SICDGeolocation (SICD complex SAR imagery) ---
from grdl.IO.sar import SICDReader

with SICDReader('image.nitf') as reader:
    geo = SICDGeolocation(reader.metadata)
    lat, lon, height = geo.image_to_latlon(500, 1000)
```

#### Factory and Chip Adapter

`create_geolocation()` auto-detects the reader type and returns the appropriate `Geolocation` subclass. `ChipGeolocation` wraps any geolocation to offset coordinates for a sub-region chip:

```python
from grdl.geolocation import create_geolocation, ChipGeolocation

# Auto-detect geolocation from any supported reader
geo = create_geolocation(reader)

# Wrap for a chip starting at (row_offset, col_offset)
chip_geo = ChipGeolocation(geo, row_offset=1000, col_offset=2000)
lat, lon, h = chip_geo.image_to_latlon(0, 0)  # maps to (1000, 2000) in full image
```

#### Elevation Models

Plug a DEM into any geolocation class for terrain-corrected transforms:

```python
from grdl.geolocation.elevation import (
    DTEDElevation, GeoTIFFDEM, ConstantElevation, GeoidCorrection,
)

# DTED tiles (directory of .dt1/.dt2 files)
# Default is bicubic interpolation with cross-tile boundary stitching
dem = DTEDElevation('/data/dted/')

# GeoTIFF DEM (single file, bicubic default)
dem = GeoTIFFDEM('/data/srtm_30m.tif')

# Fixed-height fallback (e.g., sea-level for ocean scenes)
dem = ConstantElevation(height=0.0)

# EGM96 geoid undulation correction
geoid = GeoidCorrection('/data/egm96.tif')

# Pass DEM path to any Geolocation subclass (interpolation= sets DEM spline order)
geo = AffineGeolocation('scene.tif', dem_path='/data/srtm_30m.tif')
geo = SICDGeolocation(metadata, dem_path='/data/dted/', interpolation=1)  # bilinear
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

### CSI & Dominance Detection

Coherent Shape Index (CSI) maps sub-aperture scattering behavior to an RGB composite. Dominance features detect pixels where a single sub-look dominates the aperture energy -- a signature of coherent man-made targets.

```python
from grdl.IO.sar import SICDReader
from grdl.data_prep import ChipExtractor
from grdl.image_processing.sar import SublookDecomposition, CSIProcessor
from grdl.image_processing.sar.dominance import compute_dominance
import numpy as np

with SICDReader('image.nitf') as reader:
    meta = reader.metadata
    rows, cols = reader.get_shape()
    extractor = ChipExtractor(rows, cols)
    region = extractor.chip_at_point(rows // 2, cols // 2, 5000, 5000)
    chip = reader.read_chip(region.row_start, region.row_end,
                            region.col_start, region.col_end)

# Sub-aperture decomposition
sublook = SublookDecomposition(meta, num_looks=7, dimension='azimuth')
looks = sublook.decompose(chip)                  # (7, rows, cols) complex

# Dominance ratio: high values = single look dominates
dominance = compute_dominance(looks, window_size=7, dom_window=3)

# Threshold detections
det_mask = dominance > (np.mean(dominance) + 3.0 * np.std(dominance))

# CSI RGB composite
csi = CSIProcessor(meta, dimension='azimuth', normalization='log')
csi_rgb = csi.apply(chip)                        # (rows, cols, 3) float32 [0, 1]
```

See `grdl/example/image_processing/sar/csi_detection_overlay.py` for a full CLI example that overlays detection polygons on a CSI composite.

### CFAR Detection

Constant False Alarm Rate (CFAR) detectors for automatic target detection in SAR imagery:

```python
from grdl.image_processing.detection.cfar import (
    CACFARDetector,   # Cell-Averaging
    GOCFARDetector,   # Greatest-Of
    SOCFARDetector,   # Smallest-Of
    OSCFARDetector,   # Ordered-Statistics
)

detector = CACFARDetector(
    guard_cells=2, background_cells=4, pfa=1e-6,
)
detection_set = detector.detect(image)

for det in detection_set:
    print(f"  {det.pixel_geometry.centroid}: conf={det.confidence:.2f}")
```

### Interpolation

Bandwidth-preserving 1D interpolation kernels for SAR image formation and signal processing:

```python
from grdl.interpolation import (
    LanczosInterpolator,      # Windowed sinc (Lanczos kernel)
    KaiserSincInterpolator,   # Kaiser-windowed sinc
    LagrangeInterpolator,     # Lagrange polynomial
    FarrowInterpolator,       # Farrow structure (variable delay)
    PolyphaseInterpolator,    # Polyphase FIR (efficient resampling)
    ThiranDelayFilter,        # Thiran IIR allpass (fractional delay)
)

# Resample a signal by rational factor P/Q
interp = PolyphaseInterpolator(up=4, down=3, num_taps=64)
resampled = interp.resample(signal)

# Fractional-sample delay
delay = ThiranDelayFilter(delay=0.3, order=4)
delayed = delay.apply(signal)
```

### SAR Image Formation

Full image formation algorithms from phase history to focused imagery:

```python
from grdl.image_processing.sar.image_formation import (
    PolarFormatAlgorithm,    # Spotlight PFA
    RangeDopplerAlgorithm,   # Stripmap RDA
    FastBackProjection,      # Time-domain backprojection (FFBP)
    StripmapPFA,             # Stripmap PFA (subaperture)
)
```

See `grdl/example/image_processing/sar/ifp_example.py` for a complete image formation example.

### EO NITF Geolocation (RPC & RSM)

Read EO NITF files (WorldView, GeoEye, Pleiades, aerial) and geolocate using RPC or RSM coefficients:

```python
from grdl.IO.eo.nitf import EONITFReader
from grdl.geolocation.eo.rpc import RPCGeolocation
from grdl.geolocation.eo.rsm import RSMGeolocation

with EONITFReader('worldview.ntf') as reader:
    print(f"RPC: {reader.has_rpc}, RSM: {reader.has_rsm}")

    if reader.has_rpc:
        geo = RPCGeolocation.from_reader(reader)
        lat, lon, h = geo.image_to_latlon(2048, 2048, height=100.0)
        row, col = geo.latlon_to_image(lat, lon, h)

    if reader.has_rsm:
        geo = RSMGeolocation.from_reader(reader)
        lat, lon, h = geo.image_to_latlon(2048, 2048)
```

### Native R/Rdot Projection Engine

Pure-numpy SICD Volume 3 projection — no sarpy required:

```python
from grdl.IO.sar import SICDReader
from grdl.geolocation.sar.sicd import SICDGeolocation

with SICDReader('image.nitf') as reader:
    # Native backend (no sarpy dependency)
    geo = SICDGeolocation(reader.metadata, backend='native')
    lat, lon, h = geo.image_to_latlon(500, 1000)

    # With calibration adjustments
    import numpy as np
    geo = SICDGeolocation(
        reader.metadata,
        backend='native',
        range_bias=2.5,
        delta_arp=np.array([0.1, -0.2, 0.05]),
    )
```

### Orthorectification

```python
from grdl.image_processing.ortho import orthorectify, GeographicGrid

# DEM belongs on the geolocation object — the orthorectifier
# maps coordinates *through* the geolocation, never queries DEM directly
geo.elevation = dem

# Recommended: orthorectify() function (keyword arguments, Pythonic)
result = orthorectify(
    geolocation=geo,
    reader=reader,
    interpolation='bilinear',
)
ortho = result.data                         # ndarray
result.save_geotiff('ortho.tif')            # georeferenced output

# With explicit grid and ENU coordinates
result = orthorectify(
    geolocation=geo,
    reader=reader,
    enu_grid=dict(pixel_size_m=1.0),
)

# Direct control: Orthorectifier for custom workflows
from grdl.image_processing.ortho import Orthorectifier

grid = GeographicGrid.from_geolocation(geo, pixel_size_lat=0.001,
                                       pixel_size_lon=0.001)
ortho = Orthorectifier(geo, grid, interpolation='nearest')
ortho.compute_mapping()
result = ortho.apply(image, nodata=np.nan)
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

### Processor Metadata: Versioning, Tags & Tunable Parameters

Every `ImageProcessor` subclass should declare its metadata. This is optional but **highly recommended** -- it enables downstream systems (grdl-runtime catalog discovery, grdk GUI controls) to introspect, filter, and configure processors automatically.

#### Version & Tags (decorators)

```python
from grdl.image_processing import (
    ImageTransform, processor_version, processor_tags,
    ImageModality, ProcessorCategory,
)

@processor_version('1.0.0')
@processor_tags(
    modalities=[ImageModality.SAR, ImageModality.PAN],
    category=ProcessorCategory.FILTERS,
    description='Adaptive edge-preserving smoothing filter',
)
class MyFilter(ImageTransform):
    ...
```

- `@processor_version` stamps `__processor_version__` on the class. A runtime warning is issued at first instantiation if missing.
- `@processor_tags` stamps `__processor_tags__` with modality, category, and description metadata for catalog and UI filtering.

#### Tunable Parameters (Annotated constraints)

Declare tunable parameters as class-body annotations using `typing.Annotated` with constraint markers from `grdl.image_processing.params`:

```python
from typing import Annotated
from grdl.image_processing import (
    ImageTransform, processor_version, processor_tags,
    Range, Options, Desc,
    ImageModality, ProcessorCategory,
)

@processor_version('1.0.0')
@processor_tags(modalities=[ImageModality.PAN], category=ProcessorCategory.FILTERS)
class AdaptiveFilter(ImageTransform):
    """Edge-preserving smoothing with tunable parameters."""

    sigma: Annotated[float, Range(min=0.1, max=100.0),
                     Desc('Gaussian kernel sigma')] = 2.0
    method: Annotated[str, Options('bilateral', 'guided'),
                      Desc('Filter algorithm')] = 'bilateral'
    iterations: Annotated[int, Range(min=1, max=20),
                          Desc('Number of filter passes')] = 1

    def apply(self, source, **kwargs):
        params = self._resolve_params(kwargs)
        sigma = params['sigma']
        method = params['method']
        iterations = params['iterations']
        # ... implementation
```

**How it works:**

| Marker | Purpose |
|--------|---------|
| `Range(min=, max=)` | Inclusive numeric bounds -- validated at init and runtime |
| `Options('a', 'b')` | Discrete allowed values -- validated at init and runtime |
| `Desc('...')` | Human-readable label for GUIs and docs |

- Parameters are collected into `cls.__param_specs__` at class definition time.
- An `__init__` is auto-generated (keyword-only args with defaults) unless the class defines its own.
- `_resolve_params(kwargs)` merges instance defaults with runtime overrides and validates all values.
- `Range` and `Options` are mutually exclusive on the same parameter.
- grdk reads `__param_specs__` to build dynamic parameter controls in its widget UI.

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

### Querying Processor Metadata at Runtime

Downstream tools (grdl-runtime, grdk) read the metadata stamps directly:

```python
from grdl.image_processing import PauliDecomposition, SublookDecomposition

# Version
print(PauliDecomposition.__processor_version__)   # '0.1.0'

# Tags (modalities, category, description)
print(SublookDecomposition.__processor_tags__)
# {'modalities': (<ImageModality.SAR>,), 'category': None, ...}

# Tunable parameters (ParamSpec introspection)
for spec in SublookDecomposition.__param_specs__:
    print(f"  {spec.name}: {spec.param_type.__name__}, default={spec.default}")

# GPU compatibility flag
print(SublookDecomposition.__gpu_compatible__)     # False
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

### Conda Environment (recommended)

```bash
git clone https://github.com/geoint-org/GRDL.git
cd GRDL

# Create the full environment from environment.yml
conda env create -f environment.yml
conda activate grdl
pip install -e .
```

### Pip Install (minimal)

```bash
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
7. **For ImageProcessor subclasses**: add `@processor_version`, `@processor_tags`, and `Annotated` parameter declarations. This metadata is what makes your processor discoverable and configurable in grdl-runtime and grdk.

See [CLAUDE.md](CLAUDE.md) for full development standards, file header format, and coding conventions.

## Publishing to PyPI

### Dependency Management

All dependencies are defined in `pyproject.toml`. Keep these files synchronized:

- **`pyproject.toml`** — source of truth for versions and dependencies
- **`requirements.txt`** — regenerate with `pip freeze > requirements.txt` after updating `pyproject.toml`
- **`.github/workflows/publish.yml`** — automated PyPI publication (do not edit manually)

### Releasing a New Version

The publish workflow triggers on **GitHub Release creation**, not on tag push alone.

```bash
# 1. Bump version in pyproject.toml
#    Edit: version = "X.Y.Z"

# 2. Update requirements.txt if dependencies changed
pip install -e ".[all,dev]" && pip freeze > requirements.txt

# 3. Commit and push
git add pyproject.toml requirements.txt
git commit -m "Bump version to X.Y.Z"
git push origin main

# 4. Create and push a git tag
git tag vX.Y.Z
git push origin vX.Y.Z

# 5. Create a GitHub Release (triggers the publish workflow)
gh release create vX.Y.Z --title "vX.Y.Z" --notes "Release notes here"

# 6. Verify the workflow succeeded
gh run list --limit 1
```

The workflow (`.github/workflows/publish.yml`):
- Builds wheels and source distribution using `python -m build`
- Publishes to PyPI with OIDC trusted publishing (no API keys)
- Artifacts are available at [pypi.org/p/grdl](https://pypi.org/p/grdl)

See [CLAUDE.md](CLAUDE.md#dependency-management) for detailed dependency management guidelines.

## License

MIT License -- see [LICENSE](LICENSE) for details.
