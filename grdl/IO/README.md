# IO Module

Input/Output operations for geospatial imagery and vector data.

## Overview

The IO module provides a unified interface for reading geospatial imagery across SAR, IR, multispectral, and EO formats. **Always use an IO reader to load imagery** -- don't call `rasterio.open()` or `h5py.File()` directly. The readers handle lazy loading, band indexing, resource cleanup, and typed metadata extraction.

All readers inherit from `ImageReader` (defined in `base.py`), ensuring a consistent API across formats. Readers are organized into modality-based submodules (`sar/`, `ir/`, `multispectral/`, `eo/`) mirroring how sensors are used in practice. Metadata is returned as typed dataclasses (`SICDMetadata`, `SIDDMetadata`, `CPHDMetadata`, `BIOMASSMetadata`, `NISARMetadata`, `Sentinel2Metadata`, `VIIRSMetadata`, `ASTERMetadata`, etc.) with nested attribute access, IDE autocomplete, and backward-compatible dict-like `[]` access.

## Supported Formats

### Base Data Formats (IO level)

| Format | Reader Class | Backend | Status |
|--------|-------------|---------|--------|
| GeoTIFF/COG | `GeoTIFFReader` | rasterio | ✅ Implemented |
| HDF5/HDF-EOS5 | `HDF5Reader` | h5py | ✅ Implemented |
| NITF | `NITFReader` | rasterio/GDAL | ✅ Implemented |
| JPEG2000 | `JP2Reader` | rasterio (primary), glymur (fallback) | ✅ Implemented |

### SAR (Synthetic Aperture Radar) — `sar/` submodule

| Format | Reader Class | Backend | Status |
|--------|-------------|---------|--------|
| SICD | `SICDReader` | sarkit (primary), sarpy (fallback) | ✅ Implemented |
| CPHD | `CPHDReader` | sarkit (primary), sarpy (fallback) | ✅ Implemented |
| CRSD | `CRSDReader` | sarkit | ✅ Implemented |
| SIDD | `SIDDReader` | sarkit (primary), sarpy (fallback) | ✅ Implemented |
| BIOMASS L1 SCS | `BIOMASSL1Reader` | rasterio | ✅ Implemented |
| Sentinel-1 SLC | `Sentinel1SLCReader` | rasterio | ✅ Implemented |
| Sentinel-1 L0 (raw) | `Sentinel1L0Reader` | sentinel1decoder (`grdl[s1_l0]`) | ✅ Implemented |
| TerraSAR-X / TanDEM-X | `TerraSARReader` | numpy (SSC), rasterio (detected) | ✅ Implemented |
| NISAR RSLC/GSLC | `NISARReader` | h5py | ✅ Implemented |
| SICD collection | `SICDCollectionReader` | sarkit / sarpy | ✅ Implemented |

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
| Sentinel-2 MSI | `Sentinel2Reader` | JP2Reader (rasterio/glymur) | ✅ Implemented |
| EO NITF (RPC/RSM) | `EONITFReader` | rasterio | ✅ Implemented |
| | | Parses: RPC00B, RSMPCA (multi-segment), RSMIDA, RSMGGA, RSMPIA, RSMDCA/B, RSMECA/B, RSMAPA/B, CSEXRA, CSCRNA, CSEPHA, USE00A, ICHIPB, BLOCKA, AIMIDB, STDIDC, PIAIMC, BANDSB, BANDSA, SENSRB, MENSRB, MENSRA, ACFTB — from image subheaders and TRE_OVERFLOW DES | |
| | | Multi-image: heterogeneous segment grouping + primary auto-selection (overviews/masks never fail loading); placement via ICHIPB → ILOC/IALVL → stacking; `image_index` pinning | |
| | | Pixel domain: `read_chip(decimation=)` (overview-group / `out_shape` served), `read_mask()`, `get_lut()`, `normalize_abpp()`; remote `https://`/`s3://`/`/vsi*` URIs | |
| Landsat OLI | - | - | 🔄 Planned |
| WorldView | - | - | 🔄 Planned |

### GMTI (Ground Moving Target Indicator) — `gmti/` submodule

| Format | Reader / Writer | Backend | Status |
|--------|-----------------|---------|--------|
| STANAG 4607 (editions 2/3/4) | `STANAG4607Reader` / `STANAG4607Writer` | pure Python (`struct`) | ✅ Implemented |
| | | `to_detection_set()` bridges target reports → GRDL `DetectionSet`; helpers for dwell footprint, ground-relative velocity, target filtering, summary | |
| | | `build_steering_matrix_from_cphd_metadata()` → `CPHDMetadataSteering` (metadata-driven STAP column dictionary) | |

### Writers

| Format | Writer Class | Backend | Status |
|--------|-------------|---------|--------|
| GeoTIFF | `GeoTIFFWriter` | rasterio | ✅ Implemented |
| HDF5 | `HDF5Writer` | h5py | ✅ Implemented |
| NITF | `NITFWriter` | rasterio/GDAL | ✅ Implemented |
| NumPy (.npy/.npz) | `NumpyWriter` | numpy | ✅ Implemented |
| PNG | `PngWriter` | Pillow | ✅ Implemented |
| SICD (NITF) | `SICDWriter` | sarpy | ✅ Implemented |
| SIDD (NITF) | `SIDDWriter` | sarpy | ✅ Implemented |
| EO NITF chip-out | `write_chip()` | rasterio/GDAL | ✅ Implemented |
| | | Writes ICHIPB (composed with parent) + RPC00B + serialized RSMIDA/RSMPCA so chips geolocate identically to the parent | |

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
| Sentinel-1 SLC catalog & download (CDSE OData) | ✅ Implemented |
| Sentinel-2 catalog & download (CDSE OData) | ✅ Implemented |
| ASTER catalog & local discovery | ✅ Implemented |
| VIIRS catalog & local discovery | ✅ Implemented |
| NISAR catalog & download (NASA Earthdata) | ✅ Implemented |
| TerraSAR-X catalog & local discovery | ✅ Implemented |
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

## Reader Factory & Auto-Detection

The IO module exposes a single, registry-backed factory for opening imagery.
**`open_reader()` is the recommended entry point** for unknown or mixed-format
file sets; `get_reader()` is the fast path when you already know the format.

```python
from grdl.IO import (
    open_reader,       # PRIMARY: auto-detect any supported file
    get_reader,        # explicit format, fast and unambiguous
    list_reader_formats,
)

# 1. Auto-detect (extension fast-path → get_reader → open_any cascade)
with open_reader('scene.tif') as reader:
    print(reader.metadata.format)
    chip = reader.read_chip(0, 1024, 0, 1024)

# 2. Explicit format — best for pipelines (no detection cost)
with get_reader('sicd', 'image.nitf') as reader:
    chip = reader.read_chip(0, 512, 0, 512)

# 3. Enumerate every registered format key
print(list_reader_formats())
# ['aster', 'biomass', 'cphd', 'crsd', 'eo-nitf', 'gdal', 'geotiff',
#  'hdf5', 'jpeg2000', 'nisar', 'nitf', 'probe', 'sentinel1-l0',
#  'sentinel1-slc', 'sentinel2', 'sicd', 'sidd', 'stanag4607',
#  'terrasar', 'viirs']
```

### The 20 registered reader formats

`get_reader(format, filepath)` accepts any of these keys (case-insensitive;
`_` and `-` are interchangeable, so `'sentinel1-slc'` == `'sentinel1_slc'`):

| Key | Reader class | Module |
|-----|--------------|--------|
| `geotiff` | `GeoTIFFReader` | `grdl.IO.geotiff` |
| `nitf` | `NITFReader` | `grdl.IO.nitf` |
| `hdf5` | `HDF5Reader` | `grdl.IO.hdf5` |
| `jpeg2000` | `JP2Reader` | `grdl.IO.jpeg2000` |
| `sicd` | `SICDReader` | `grdl.IO.sar.sicd` |
| `cphd` | `CPHDReader` | `grdl.IO.sar.cphd` |
| `crsd` | `CRSDReader` | `grdl.IO.sar.crsd` |
| `sidd` | `SIDDReader` | `grdl.IO.sar.sidd` |
| `biomass` | `BIOMASSL1Reader` | `grdl.IO.sar.biomass` |
| `sentinel1-slc` | `Sentinel1SLCReader` | `grdl.IO.sar.sentinel1_slc` |
| `sentinel1-l0` | `Sentinel1L0Reader` | `grdl.IO.sar.sentinel1_l0` |
| `terrasar` | `TerraSARReader` | `grdl.IO.sar.terrasar` |
| `nisar` | `NISARReader` | `grdl.IO.sar.nisar` |
| `sentinel2` | `Sentinel2Reader` | `grdl.IO.eo.sentinel2` |
| `eo-nitf` | `EONITFReader` | `grdl.IO.eo.nitf` |
| `aster` | `ASTERReader` | `grdl.IO.ir.aster` |
| `viirs` | `VIIRSReader` | `grdl.IO.multispectral.viirs` |
| `stanag4607` | `STANAG4607Reader` | `grdl.IO.gmti.stanag4607` |
| `gdal` | `GDALFallbackReader` | `grdl.IO.generic` |
| `probe` | `InvasiveProbeReader` | `grdl.IO.probe` |

The registry stores `(module_path, class_name)` tuples and imports the module
lazily via `importlib` on first use — optional dependencies (`sarpy`,
`rasterio`, `h5py`, ...) are only required when their reader is actually
requested. A missing dependency raises `ImportError` naming the package and
pointing at `requirements-optional.txt`.

### `open_reader()` vs `open_any()`

| Function | Strategy |
|----------|----------|
| `open_reader(path)` | Extension fast-path (`_READER_EXTENSION_MAP`) → `get_reader(fmt, path)`; on `ImportError` (missing library) it records the message and falls through to `open_any()`. Emits a `UserWarning` naming the missing package when a fallback reader is used. **Primary auto-detect entry point.** |
| `open_any(path)` | Full ambiguity cascade: NITF sniffing (SICD/SIDD/EO discrimination) → modality openers (`open_sar`, `open_eo`, ...) → base readers → `GDALFallbackReader` → `InvasiveProbeReader`. Called internally by `open_reader()`; use directly for ambiguous files with no informative extension. |
| `open_sar(path)` | SAR-only modality cascade (see below). |

```python
from grdl.IO import open_reader, open_any

# open_reader prefers the registry, falls back to open_any automatically
with open_reader('mystery.dat') as reader:
    chip = reader.read_chip(0, 256, 0, 256)

# open_any directly when extension is missing or misleading
with open_any('/data/some_nitf_without_extension') as reader:
    print(type(reader).__name__)
```

> **Deprecated:** `open_image()` is a backward-compatible alias for
> `open_reader()` that emits a `DeprecationWarning`. **Use `open_reader()`
> in all new code.**

### Runtime registration

Register custom readers/writers without editing the library. The new key is
immediately available to `get_reader`/`get_writer` and `list_reader_formats`.

```python
from grdl.IO import register_reader, register_writer, get_reader

register_reader('myformat', 'mypkg.io.myreader', 'MyFormatReader')
register_writer('myformat', 'mypkg.io.mywriter', 'MyFormatWriter')

with get_reader('myformat', 'data.myf') as reader:
    chip = reader.read_chip(0, 100, 0, 100)

# Replace an existing entry (e.g. to swap in a faster backend)
register_reader('geotiff', 'mypkg.io.fast_gtiff', 'FastGeoTIFFReader',
                overwrite=True)
```

### Writer factory & `write()` convenience

`get_writer(format, path, metadata=None)` mirrors the reader factory.
`write(data, path, metadata=None, format=None, geolocation=None)` handles the
full open/write/close lifecycle and auto-detects the writer from the file
extension when `format` is omitted.

```python
import numpy as np
from grdl.IO import write, get_writer

arr = np.random.rand(512, 512).astype('float32')

# One-shot convenience (extension → format)
write(arr, '/tmp/out.tif')

# Explicit writer with metadata
with get_writer('geotiff', '/tmp/out2.tif') as writer:
    writer.write(arr)
```

Registered writer keys: `geotiff`, `numpy`, `png`, `hdf5`, `nitf`.
(`SICDWriter`, `SIDDWriter`, and `STANAG4607Writer` are *not* in the writer
registry because they take typed metadata containers rather than the
`write(ndarray)` contract — use those classes directly.)

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
from grdl.IO import GeoTIFFReader, open_reader, get_reader

# Auto-detect any supported raster format (replaces deprecated open_image)
with open_reader('scene.tif') as reader:
    print(f"Format: {reader.metadata.format}")
    chip = reader.read_chip(0, 1024, 0, 1024)

# Explicit format via the reader factory — faster and unambiguous
with get_reader('geotiff', 'scene.tif') as reader:
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
    from grdl.image_processing.intensity import ToDecibels
    to_db = ToDecibels()
    hh_mag_db = to_db.apply(hh_chip)

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

### NISAR - NASA L-band/S-band SAR

```python
from grdl.IO import NISARReader, open_nisar
import numpy as np

# Auto-detect and open NISAR product (RSLC or GSLC)
with open_nisar('/path/to/NISAR_L1_PR_RSLC_001.h5') as reader:
    meta = reader.metadata  # NISARMetadata with typed fields

    # Product identification
    print(f"Product: {meta.product_type}")     # "RSLC" or "GSLC"
    print(f"Radar band: {meta.radar_band}")    # "LSAR" or "SSAR"
    print(f"Frequency: {meta.frequency}")      # "A" or "B"
    print(f"Polarization: {meta.polarization}")  # "HH", "HV", etc.

    # Available sub-bands and polarizations
    print(f"Frequencies: {meta.available_frequencies}")    # ["A", "B"]
    print(f"Polarizations: {meta.available_polarizations}")  # ["HH", "HV"]

    # Read complex chip
    chip = reader.read_chip(0, 1024, 0, 1024)  # complex64
    magnitude_db = 20 * np.log10(np.abs(chip) + 1e-10)

# Open specific frequency and polarization
with NISARReader('/path/to/product.h5', frequency='A', polarization='HV') as reader:
    full = reader.read_full()

# GSLC products include a validity mask
with NISARReader('/path/to/NISAR_GSLC.h5') as reader:
    chip = reader.read_chip(0, 512, 0, 512)
    mask = reader.read_mask(0, 512, 0, 512)  # uint8 validity mask

# Auto-detect via open_sar (works with .h5/.hdf5 NISAR files)
from grdl.IO import open_sar
with open_sar('/path/to/NISAR_product.h5') as reader:
    chip = reader.read_chip(0, 512, 0, 512)
```

### Sentinel-2 MSI - Multispectral Imagery

```python
from grdl.IO.eo import Sentinel2Reader, open_eo

# Open a Sentinel-2 band (JP2 format)
with Sentinel2Reader('T10SEG_20240101T183901_B04.jp2') as reader:
    meta = reader.metadata  # Sentinel2Metadata with typed fields

    print(f"Satellite: {meta.satellite}")        # "S2A", "S2B", "S2C"
    print(f"Level: {meta.processing_level}")      # "L1C" or "L2A"
    print(f"Band: {meta.band_id}")               # "B04"
    print(f"MGRS tile: {meta.mgrs_tile_id}")     # "T10SEG"
    print(f"Resolution: {meta.resolution_tier}m") # 10, 20, or 60

    chip = reader.read_chip(0, 1024, 0, 1024)

# Auto-detect via open_eo (detects Sentinel-2 from filename)
with open_eo('T10SEG_20240101T183901_B02.jp2') as reader:
    chip = reader.read_chip(0, 512, 0, 512)
```

#### EO NITF (WorldView, GeoEye, commercial) - RPC/RSM Geolocation

`EONITFReader` is the richest reader in the module. It unifies heterogeneous
multi-segment NITFs, parses the full commercial TRE suite, supports decimated
and remote reads, and exposes mask/LUT/radiometric helpers.

```python
from grdl.IO import EONITFReader
from grdl.IO.eo.nitf import write_chip

with EONITFReader('worldview.ntf') as reader:
    meta = reader.metadata  # EONITFMetadata

    # Geolocation models (use with grdl.geolocation.eo.rpc / .rsm)
    print(meta.rpc)            # RPCCoefficients (RPC00B)
    print(meta.rsm)            # RSMCoefficients (RSMPCA, first segment)
    print(meta.rsm_segments)   # RSMSegmentGrid (all RSMPCA sections)
    print(meta.rsm_id)         # RSMIdentification (RSMIDA)

    # Band characterization (BANDSB/BANDSA)
    print(meta.bandsb.band_names, meta.bandsb.wavelengths)

    # Multi-segment grouping is automatic; pin a specific segment to read
    # overviews or masks the unified path skips:
    print(meta.image_segments)  # ImageSegmentInfo per physical segment
    print(meta.image_groups)    # ImageGroupInfo (primary + overviews/masks)

    # Full-resolution chip from the primary group
    chip = reader.read_chip(0, 4096, 0, 4096)

    # Decimated read — exploits embedded overviews / JPEG2000 reduced
    # resolution levels (out_shape) rather than reading then slicing
    thumb = reader.read_chip(0, 40000, 0, 40000, decimation=8)

    # Validity mask and lookup table
    mask = reader.read_mask(0, 4096, 0, 4096)   # None if no mask segment
    lut = reader.get_lut(band=0)                # None if no LUT

    # Stretch to the actual bit depth (ABPP) for display
    display = reader.normalize_abpp(chip)

# Pin a non-primary segment (e.g. an overview) by index
with EONITFReader('worldview.ntf', image_index=2) as reader:
    overview = reader.read_full()

# Remote URIs work transparently (GDAL /vsicurl, /vsis3)
with EONITFReader('https://bucket.example.com/scene.ntf') as reader:
    chip = reader.read_chip(0, 1024, 0, 1024)

# Chip-out to a new NITF that geolocates identically to the parent
# (writes ICHIPB composed with the parent + RPC00B + serialized RSM)
out_path = write_chip(reader, 0, 1024, 0, 1024, 'chip.ntf')
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

### GMTI - STANAG 4607 Ground Moving Target Indicator

The `grdl.IO.gmti` submodule reads and writes STANAG 4607 GMTI files
(editions 2, 3, and 4) and bridges target reports into the GRDL detection
ecosystem.

```python
from grdl.IO import (
    open_gmti, STANAG4607Reader, STANAG4607Writer,
    dwell_footprint_polygon, ground_relative_velocity,
    filter_target_reports, gmti_summarize,
)

with open_gmti('mission.4607') as reader:   # or STANAG4607Reader('mission.4607')
    print(f"Edition: {reader.edition}")      # 2, 3, or 4 (auto-detected)

    # Iterate the parsed segment hierarchy
    for packet in reader.iter_packets():
        ...
    for dwell in reader.iter_dwells():
        ...
    for dwell, target in reader.iter_target_reports():
        print(target.latitude, target.longitude)

    # Bridge to GRDL Detections (shapely Point geometries in WGS84)
    detections = reader.to_detection_set(
        confidence_field='gmti.snr_db',  # field used to derive confidence
        snr_normalization=40.0,          # SNR(dB) / 40 → confidence in [0, 1]
    )

    # Quick-look statistics
    stats = gmti_summarize(reader)
```

**Helper functions** operate on the typed dwell/target dataclasses:

```python
# Dwell sensor footprint as a shapely Polygon
poly = dwell_footprint_polygon(dwell)

# Target ground-relative velocity (m/s) from radial velocity + geometry
v_ground = ground_relative_velocity(dwell, target)

# Filter target reports by SNR, area, classification, etc.
strong = filter_target_reports(dwell.target_reports, min_snr_db=12.0)
```

**Writing** uses typed segment dataclasses (not ndarrays), so the
`STANAG4607Writer` class is used directly rather than through `get_writer`.

**CPHD-driven steering** — derive a per-channel, scene-projected steering
matrix from a CPHD `Antenna` section for metadata-driven STAP detectors:

```python
from grdl.IO.gmti import build_steering_matrix_from_cphd_metadata
from grdl.IO import CPHDReader
import numpy as np

with CPHDReader('collect.cphd') as cphd:
    steering = build_steering_matrix_from_cphd_metadata(
        cphd.metadata,
        rcv_pos_ecf=rcv_pos_ecf,    # (Nc, 3) per-channel APC ECF positions
        srp_ecf=srp_ecf,            # (3,) stabilization reference point ECF
        xr_offsets_m=xr_offsets_m,  # (n_xr,) scene cross-range offsets from SRP
        eval_time=0.0,
        ref_channel=0,
    )
    # steering is a CPHDMetadataSteering whose matrix columns align with the
    # image cross-range bins — use as the STAP column dictionary.
```

### Sentinel-1 Level 0 (raw) and CRSD conversion

The `grdl.IO.sar.sentinel1_l0` subpackage reads raw Sentinel-1 L0 SAFE
products (burst-indexed packet parsing, optional FDBAQ decompression). The
decoder backend requires the optional `grdl[s1_l0]` extra
(`sentinel1decoder>=2.0`).

```python
from grdl.IO.sar import Sentinel1L0Reader, open_safe_product, ReaderConfig

# open_safe_product is the convenience factory; also reachable via open_sar
with open_safe_product('/data/S1A_..._RAW__0SDV_....SAFE') as reader:
    # Burst-aware native API
    iq = reader.read_burst(0, polarization='VV')   # (num_lines, num_samples) complex
    for burst_iq in reader.iter_bursts(polarization='VV'):
        ...
    swath_iq = reader.read_swath(swath=1, polarization='VV')

# Tune burst detection / orbit loading via ReaderConfig
config = ReaderConfig(validate_safe=True, load_poe=True,
                      burst_line_filter_ratio=0.9)
reader = Sentinel1L0Reader('/data/product.SAFE', config=config)
```

Optional CRSD conversion and verification (also gated by the `s1_l0` extra):

```python
from grdl.IO.sar.sentinel1_l0 import (
    Sentinel1L0ToCRSD, convert_s1_l0_to_crsd, verify_crsd_split_gates,
)

crsd_path = convert_s1_l0_to_crsd('/data/product.SAFE', '/data/out.crsd')
result = verify_crsd_split_gates(crsd_path)
```

## Parallel Reading

Rasterio-based readers support opt-in multi-threaded reads via `ReadConfig`. The `EONITFReader`, `GeoTIFFReader`, and `BIOMASSL1Reader` default to `parallel=True`.

```python
from grdl.IO import EONITFReader, GeoTIFFReader
from grdl.IO.performance import ReadConfig

# Default: parallel reading enabled for rasterio-based readers
with EONITFReader('worldview.ntf') as reader:
    chip = reader.read_chip(0, 10000, 0, 10000)  # uses parallel chunked read

# Explicit configuration
config = ReadConfig(
    parallel=True,           # enable multi-threaded reads
    max_workers=8,           # thread pool size (default: cpu_count - 1)
    gdal_num_threads=4,      # GDAL internal decompression threads
    chunk_threshold=4_000_000,  # min pixels for chunked parallel read
)
with GeoTIFFReader('large_scene.tif', read_config=config) as reader:
    chip = reader.read_chip(0, 5000, 0, 5000)

# Disable parallel reading if needed
with EONITFReader('small.ntf', read_config=ReadConfig(parallel=False)) as reader:
    chip = reader.read_chip(0, 512, 0, 512)
```

**Thread safety by backend:**

| Backend | Readers | Parallel? |
|---------|---------|-----------|
| rasterio (GDAL) | EONITFReader, GeoTIFFReader, BIOMASSL1Reader, GDALFallbackReader, Sentinel1SLCReader, TerraSARReader, ASTERReader | Yes (releases GIL) |
| h5py | HDF5Reader, NISARReader, VIIRSReader | No (holds GIL) |
| glymur | JP2Reader, Sentinel2Reader | No (not thread-safe) |

**Parallel strategy cascade** (automatic, based on read size):
1. **Chunked parallel** — large windows (>4M pixels): split into tile-aligned sub-windows, read concurrently
2. **Parallel bands** — multi-band imagery: read each band in a separate thread
3. **Serial** — single band, small window: standard single-threaded read

## Architecture

### Base Classes

All readers inherit from abstract base classes in `base.py`:

- **`ImageReader`** - Base for all imagery readers
  - **Abstract:** `_load_metadata()`, `read_chip()`, `get_shape()`, `get_dtype()`
  - **Concrete:** `read_full()` (delegates to `read_chip`), `read_band(index)` (always 2-D), `close()`, context manager support
  - **Class attribute:** `_enforce_2d` (default `False`; single-channel SAR readers set `True`)
  - **Shape helpers:** `_ensure_2d(arr)`, `_assert_2d(data, context, strict)`, `_validate_single_pol(arr, context)`

- **`ImageWriter`** - Base for all imagery writers
  - **Abstract:** `write(data, geolocation=None)`, `write_chip(data, row_start, col_start, geolocation=None)`
  - `close()` and context manager support

- **`CatalogInterface`** - Base for image discovery
  - **Abstract:** `discover_images(extensions=None, recursive=True)`, `get_metadata_summary(image_paths)`, `find_overlapping(reference_bounds, image_paths)`

### Typed Metadata

All readers populate `self.metadata` with a typed dataclass. Format-specific readers return specialized subclasses with nested attribute access:

| Reader | Metadata Class | Key Nested Types |
|--------|---------------|-----------------|
| `GeoTIFFReader` | `ImageMetadata` | (flat fields) |
| `HDF5Reader` | `ImageMetadata` | (flat fields) |
| `NITFReader` | `ImageMetadata` | (flat fields) |
| `JP2Reader` | `ImageMetadata` | (flat fields) |
| `SICDReader` | `SICDMetadata` | 17 sections: `collection_info`, `image_data`, `geo_data`, `grid`, `timeline`, `position`, `radar_collection`, `image_formation`, `scpcoa`, `radiometric`, `antenna`, `error_statistics`, `match_info`, `rg_az_comp`, `pfa`, `rma` |
| `CPHDReader` | `CPHDMetadata` | `channels`, `pvp`, `global_params`, `collection_info`, `tx_waveform`, `rcv_parameters`, `antenna_pattern`, `scene_coordinates`, `reference_geometry`, `dwell_polynomial` |
| `CRSDReader` | `ImageMetadata` | (flat fields + extras) |
| `SIDDReader` | `SIDDMetadata` | `product_creation`, `display`, `geo_data`, `measurement`, `exploitation_features`, `downstream_reprocessing`, `compression`, `digital_elevation_data`, `product_processing`, `annotations` |
| `BIOMASSL1Reader` | `BIOMASSMetadata` | `mission`, `swath`, `polarizations`, `orbit_number`, `range_pixel_spacing`, `azimuth_pixel_spacing`, `prf`, `corner_coords`, `gcps` |
| `Sentinel1SLCReader` | `Sentinel1SLCMetadata` | `product_info`, `swath_timing`, `bursts`, `orbit_state_vectors`, `geolocation_grid`, `doppler_centroid`, `calibration_vectors` |
| `Sentinel1L0Reader` | `Sentinel1L0Metadata` | burst index, swath/polarization tables, orbit & attitude (raw L0) |
| `TerraSARReader` | `TerraSARMetadata` | `product_info`, `scene_info`, `image_info`, `radar_params`, `orbit_state_vectors`, `geolocation_grid`, `calibration`, `doppler_info`, `processing_info` |
| `NISARReader` | `NISARMetadata` | `identification`, `orbit`, `attitude`, `swath_parameters`, `grid_parameters`, `geolocation_grid`, `calibration`, `processing_info` |
| `Sentinel2Reader` | `Sentinel2Metadata` | `satellite`, `processing_level`, `product_type`, `band_id`, `mgrs_tile_id`, `resolution_tier`, `sensing_datetime` |
| `VIIRSReader` | `VIIRSMetadata` | `satellite_name`, `product_short_name`, `day_night_flag`, `geospatial_bounds`, `scale_factor`, `add_offset`, `fill_value`, `dataset_path` |
| `ASTERReader` | `ASTERMetadata` | `processing_level`, `acquisition_date`, `sun_azimuth`, `sun_elevation`, `cloud_cover`, `vnir_available`, `swir_available`, `tir_available` |
| `EONITFReader` | `EONITFMetadata` | `rpc` (RPCCoefficients), `rsm` (RSMCoefficients), `rsm_segments` (RSMSegmentGrid), `rsm_id` (RSMIdentification), `rsm_pia`/`rsm_dca`/`rsm_eca`/`rsm_apa` (RSM error model), `ichipb` (ICHIPBMetadata), `csexra` (CSEXRAMetadata), `cscrna` (CSCRNAMetadata), `use00a` (USE00AMetadata), `blocka` (BLOCKAMetadata), `bandsb`/`bandsa` (band characterization → `band_names`/`wavelengths`), `sensrb`/`mensrb`/`mensra`/`acftb` (airborne), `image_segments` (ImageSegmentInfo), `image_groups` (ImageGroupInfo), `collection_info` (CollectionInfo), `accuracy` (AccuracyInfo), `idatim`, `tgtid`, `isource`, `igeolo` |

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

The `models/` package provides ~90 dataclasses organized in `common.py` (shared primitives like `XYZ`, `LatLonHAE`, `RowCol`, `Poly1D`, `Poly2D`, `XYZPoly`), `sicd.py`, `sidd.py`, `cphd.py`, `biomass.py`, `sentinel1_slc.py`, `terrasar.py`, `nisar.py`, `sentinel2.py`, `viirs.py`, `aster.py`, and `eo_nitf.py`.

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

## Catalogs & Credentials

All catalog classes implement the `CatalogInterface` ABC (`discover_images`,
`get_metadata_summary`, `find_overlapping`) and add local SQLite tracking plus,
where applicable, remote search and download:

| Catalog | Remote API | Auth |
|---------|-----------|------|
| `BIOMASSCatalog` | ESA MAAP STAC | `ESA_MAAP_OFFLINE_TOKEN` (OAuth2 offline token) |
| `Sentinel1SLCCatalog` | CDSE OData | `COPERNICUS_*` (CDSE token) |
| `Sentinel2Catalog` | CDSE OData | `COPERNICUS_*` (CDSE token) |
| `NISARCatalog` | NASA Earthdata | `EARTHDATA_*` |
| `ASTERCatalog` | local discovery | — |
| `VIIRSCatalog` | local discovery | — |
| `TerraSARCatalog` | local discovery | — |

### Credential convention

Remote helpers in `grdl.IO.catalog.remote_utils` read credentials from
`~/.config/geoint/credentials.json` (XDG-style, repo-agnostic), with
environment-variable fallbacks:

```python
from grdl.IO.catalog import (
    load_credentials, get_cdse_token, get_earthdata_token, download_file,
)

creds = load_credentials('copernicus')        # provider block from JSON / env
token = get_cdse_token()                       # CDSE OAuth2 access token
edl = get_earthdata_token()                    # NASA Earthdata token
download_file(url, '/data/dest.zip', token=token)  # streaming download w/ resume
```

`credentials.json` layout (env-var fallbacks shown):

```json
{
  "copernicus": {"username": "...", "password": "..."},   // COPERNICUS_USERNAME / _PASSWORD
  "earthdata":  {"username": "...", "password": "..."},   // EARTHDATA_USERNAME / _PASSWORD
  "esa_maap":   {"offline_token": "..."}                  // ESA_MAAP_OFFLINE_TOKEN
}
```

SQLite catalog databases default to `~/.config/geoint/catalogs/`.

## API Reference

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design decisions.

See [PATTERNS.md](PATTERNS.md) for recurring implementation patterns (TRE parsers, ReadConfig integration, multi-segment TRE collection, etc.).

See [TODO.md](TODO.md) for planned features and roadmap.

## Examples

See `grdl/example/` for working workflows:
- `catalog/discover_and_download.py` - Search ESA MAAP catalog, download products
- `catalog/view_product.py` - Load BIOMASS L1A, display HH dB and Pauli RGB with interactive markers
- `ortho/chip_ortho.py` - Ground-extent chip extraction and ENU orthorectification
- `ortho/compare_sidd_ortho.py` - Dual-SIDD ortho comparison with PCA, NCC, and coregistration
- `ortho/ortho_biomass.py` - Orthorectification with Pauli RGB composite output
- `ortho/ortho_combined.py` - Combined SICD/SIDD auto-detect orthorectification
- `ortho/ortho_sicd.py` - SICD orthorectification with DEM and ENU grids
- `ortho/ortho_sidd.py` - SIDD orthorectification with DEM and ENU grids
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