# IO Package Migration Plan (COMPLETED 2026-02-09)

## Context

The current IO module has all readers in flat files (`sar.py`, `biomass.py`, `catalog.py`) and uses only sarpy for SAR formats. The goal is to:

1. **Keep base data formats at IO/ level** — GeoTIFF and NITF readers live in `IO/` so modality submodules don't depend on each other
2. **Reorganize** modality-specific readers into submodules (`sar/`)
3. **Migrate** SICD and CPHD readers to use **sarkit as primary backend** with sarpy fallback
4. **Add** CRSD and SIDD readers (sarkit-only, NGA standards)
5. **Rename** GRDReader → `GeoTIFFReader` at `IO/geotiff.py` (it's a general GeoTIFF reader, not SAR-specific)
6. **Move** BIOMASS reader and catalog under `sar/` (it is SAR data)

MSI and EO submodules deferred — GeoTIFF at IO base level covers most EO/MSI formats already.

## New Directory Structure

```
grdl/IO/
├── __init__.py              # Top-level API + open_image() convenience
├── base.py                  # ImageReader, ImageWriter, CatalogInterface ABCs (kept as-is)
├── geotiff.py               # GeoTIFFReader (rasterio) — base data format
├── nitf.py                  # NITFReader (rasterio/GDAL) — base data format
├── sar/                     # SAR-specific formats
│   ├── __init__.py          # SAR public API + open_sar()
│   ├── _backend.py          # sarkit/sarpy availability detection
│   ├── sicd.py              # SICDReader (sarkit primary, sarpy fallback)
│   ├── cphd.py              # CPHDReader (sarkit primary, sarpy fallback)
│   ├── crsd.py              # CRSDReader (sarkit only, new)
│   ├── sidd.py              # SIDDReader (sarkit only, new)
│   ├── biomass.py           # BIOMASSL1Reader (rasterio, moved from IO/biomass.py)
│   └── biomass_catalog.py   # BIOMASSCatalog + load_credentials (moved from IO/catalog.py)
├── ARCHITECTURE.md
├── README.md
└── TODO.md
```

Base data formats (GeoTIFF, NITF) live at the IO level so modality submodules
can import them without cross-submodule dependencies. SAR readers that use NITF
containers (SICD, SIDD) go through sarkit/sarpy, not a generic NITF reader.

## Backend Fallback Pattern

### `sar/_backend.py`

Detects which SAR libraries are available at import time:

```python
_HAS_SARKIT = False
_HAS_SARPY = False

try:
    import sarkit
    _HAS_SARKIT = True
except ImportError:
    pass

try:
    import sarpy
    _HAS_SARPY = True
except ImportError:
    pass

def require_sar_backend(format_name: str) -> str:
    """Return 'sarkit' or 'sarpy', or raise ImportError."""
    if _HAS_SARKIT:
        return 'sarkit'
    if _HAS_SARPY:
        return 'sarpy'
    raise ImportError(
        f"Reading {format_name} requires sarkit or sarpy. "
        f"Install with: pip install sarkit"
    )

def require_sarkit(format_name: str) -> None:
    """Require sarkit specifically (for CRSD/SIDD with no sarpy fallback)."""
    if not _HAS_SARKIT:
        raise ImportError(
            f"Reading {format_name} requires sarkit. "
            f"Install with: pip install sarkit"
        )
```

## SAR Readers

### `sar/sicd.py` — SICDReader(ImageReader)

Refactored from current `sar.py` SICDReader. Uses sarkit primary, sarpy fallback.

**sarkit path:**
- Open file as binary, pass to `sarkit.sicd.NitfReader(file)`
- `read_chip()` → `reader.read_sub_image(start_row, start_col, stop_row, stop_col)`
- Metadata from `reader.metadata.xmltree` (XPath: `{*}ImageData/{*}NumRows`, etc.)

**sarpy path (fallback):**
- `sarpy.io.complex.converter.open_complex(filepath)` (current implementation)
- `read_chip()` → `reader[row_start:row_end, col_start:col_end]`
- Metadata from `reader.get_sicds_as_tuple()`

**Key difference:** sarkit takes `BinaryIO`, sarpy takes filepath string. The wrapper manages this.

### `sar/cphd.py` — CPHDReader(ImageReader)

Refactored from current `sar.py` CPHDReader. Uses sarkit primary, sarpy fallback.

**sarkit path:**
- `sarkit.cphd.Reader(file)` — reads channels by identifier string
- `read_chip()` → `reader.read_signal(channel_id, start_vector, stop_vector)`
- Channel info from `reader.metadata.xmltree`

**sarpy path (fallback):**
- `sarpy.io.phase_history.converter.open_phase_history(filepath)` (current implementation)

### `sar/crsd.py` — CRSDReader(ImageReader) — NEW

sarkit-only. Reads Compensated Radar Signal Data.

- `sarkit.crsd.Reader(file)` — similar API to CPHD
- `read_signal()`, `read_pvps()`, `read_ppps()`, `read_support_array()`
- Metadata via `reader.metadata.xmltree`

### `sar/sidd.py` — SIDDReader(ImageReader) — NEW

sarkit-only. Reads Sensor Independent Derived Data (processed SAR products in NITF).

- `sarkit.sidd.NitfReader(file)` — similar to SICD NitfReader
- `read_image(image_index)`, `read_image_sub_image(image_index, ...)`
- Multiple product images per file (accessed by index)
- Metadata from `reader.metadata.images[i].xmltree`

### `sar/biomass.py` — BIOMASSL1Reader(ImageReader)

Direct move from current `IO/biomass.py`. No API changes — uses rasterio for magnitude/phase GeoTIFFs and XML for annotation metadata.

### `sar/biomass_catalog.py` — BIOMASSCatalog, load_credentials

Direct move from current `IO/catalog.py`. No API changes. Contains MAAP STAC search, OAuth2, SQLite tracking.

### `sar/__init__.py` — open_sar() auto-detection

Updated from current `sar.py`:
```python
def open_sar(filepath):
    """Auto-detect SAR format and return reader."""
    # 1. Try SICD (NITF with complex data)
    # 2. Try CPHD
    # 3. Try CRSD
    # 4. Try SIDD
    # 5. Try GeoTIFFReader as fallback (SAR GRD products)
```

## Base Format Readers (IO level)

### `IO/geotiff.py` — GeoTIFFReader(ImageReader)

Refactored from current `GRDReader` in `sar.py`. Lives at IO base level because GeoTIFF
is a foundational format used across modalities (SAR GRD, EO, MSI).

- Backend: rasterio (unchanged)
- Reads any GeoTIFF including SAR GRD products, COGs, standard EO imagery
- `metadata['format']` changes from `'GRD'` to `'GeoTIFF'`
- Otherwise identical API to current GRDReader
- SAR `open_sar()` falls back to this for GRD products

### `IO/nitf.py` — NITFReader(ImageReader)

Generic NITF (National Imagery Transmission Format) reader for non-SAR NITF files.

- Backend: rasterio (GDAL's NITF driver)
- Reads standard NITF imagery segments — multi-band, various dtypes
- Metadata from NITF file/image subheaders via rasterio/GDAL
- `metadata['format']` = `'NITF'`
- SAR-specific NITF (SICD, SIDD) should use the SAR readers instead — those
  extract SAR XML metadata that this generic reader does not parse
- Future base formats (JP2, HDF5) would also live at IO level

## Top-Level `IO/__init__.py`

Re-exports from base-level readers and submodules. Includes `open_image()` convenience:

```python
# Base classes
from grdl.IO.base import ImageReader, ImageWriter, CatalogInterface

# Base format readers (IO level)
from grdl.IO.geotiff import GeoTIFFReader
from grdl.IO.nitf import NITFReader

# SAR
from grdl.IO.sar import (
    SICDReader, CPHDReader, CRSDReader, SIDDReader,
    BIOMASSL1Reader, BIOMASSCatalog,
    open_sar, open_biomass, load_credentials,
)


def open_image(filepath):
    """Open any supported raster image file. Tries GeoTIFF, then NITF."""
    # Try GeoTIFF first (most common), then NITF
    ...
```

## Files to Create

| File | Source | Notes |
|------|--------|-------|
| `grdl/IO/geotiff.py` | from `sar.py` GRDReader | Rename to GeoTIFFReader, lives at IO base level |
| `grdl/IO/nitf.py` | new | NITFReader using rasterio/GDAL NITF driver, IO base level |
| `grdl/IO/sar/__init__.py` | new | SAR exports, `open_sar()`, `open_biomass()` |
| `grdl/IO/sar/_backend.py` | new | sarkit/sarpy detection |
| `grdl/IO/sar/sicd.py` | from `sar.py` SICDReader | Add sarkit backend, keep sarpy fallback |
| `grdl/IO/sar/cphd.py` | from `sar.py` CPHDReader | Add sarkit backend, keep sarpy fallback |
| `grdl/IO/sar/crsd.py` | new | sarkit-only |
| `grdl/IO/sar/sidd.py` | new | sarkit-only |
| `grdl/IO/sar/biomass.py` | from `IO/biomass.py` | Move, update imports |
| `grdl/IO/sar/biomass_catalog.py` | from `IO/catalog.py` | Move, update imports |

## Files to Modify

| File | Changes |
|------|---------|
| `grdl/IO/__init__.py` | Replace flat imports with base-level + submodule re-exports, add `open_image()` |
| `grdl/IO/base.py` | No changes needed (ABCs are clean) |
| `grdl/IO/README.md` | Update examples for new import paths |
| `grdl/IO/ARCHITECTURE.md` | Update directory tree and design patterns |
| `grdl/IO/TODO.md` | Mark completed items, update roadmap |
| `README.md` | Update IO section directory tree and examples |

## Files to Delete

| File | Replacement |
|------|-------------|
| `grdl/IO/sar.py` | Replaced by `sar/` package |
| `grdl/IO/biomass.py` | Moved to `sar/biomass.py` |
| `grdl/IO/catalog.py` | Moved to `sar/biomass_catalog.py` |

## Tests

| Test File | Covers |
|-----------|--------|
| `tests/test_io_sar_sicd.py` | SICDReader: backend detection, API contract, import validation |
| `tests/test_io_sar_cphd.py` | CPHDReader: backend detection, API contract |
| `tests/test_io_sar_crsd.py` | CRSDReader: sarkit-only, API contract |
| `tests/test_io_sar_sidd.py` | SIDDReader: sarkit-only, API contract |
| `tests/test_io_sar_backend.py` | _backend.py: detection logic, fallback behavior |
| `tests/test_io_geotiff.py` | GeoTIFFReader: rasterio reads, metadata |
| `tests/test_io_nitf.py` | NITFReader: rasterio/GDAL NITF reads, metadata |
| `tests/test_io_imports.py` | Top-level imports work, submodule imports work |

Tests use synthetic data or mock backends — no real SAR files in repo.

## Implementation Order

1. **Create `IO/geotiff.py`** — move GRDReader to IO base level, rename to GeoTIFFReader
2. **Create `IO/nitf.py`** — generic NITF reader using rasterio/GDAL
3. **Create `sar/_backend.py`** — sarkit/sarpy detection, foundation for SAR readers
4. **Create `sar/sicd.py`** — migrate SICDReader, add sarkit backend
5. **Create `sar/cphd.py`** — migrate CPHDReader, add sarkit backend
6. **Create `sar/crsd.py`** — new CRSDReader (sarkit-only)
7. **Create `sar/sidd.py`** — new SIDDReader (sarkit-only)
8. **Move `sar/biomass.py`** — from IO/biomass.py, update imports
9. **Move `sar/biomass_catalog.py`** — from IO/catalog.py, update imports
10. **Create `sar/__init__.py`** — SAR exports, open_sar(), open_biomass()
11. **Update `IO/__init__.py`** — re-exports from base-level + submodules, add open_image()
12. **Delete old flat files** — sar.py, biomass.py, catalog.py
13. **Write tests**
14. **Update documentation** — README.md, ARCHITECTURE.md, TODO.md

## Verification

```bash
source /Users/duanesmalley/opt/anaconda3/etc/profile.d/conda.sh && conda activate starlight

# Import checks
python -c "from grdl.IO import SICDReader, CPHDReader, CRSDReader, SIDDReader; print('SAR imports OK')"
python -c "from grdl.IO import GeoTIFFReader, NITFReader, open_image; print('base format imports OK')"
python -c "from grdl.IO.geotiff import GeoTIFFReader; from grdl.IO.nitf import NITFReader; print('direct base format imports OK')"
python -c "from grdl.IO import BIOMASSL1Reader, BIOMASSCatalog; print('BIOMASS imports OK')"
python -c "from grdl.IO.sar import open_sar; print('submodule imports OK')"
python -c "from grdl.IO.sar._backend import _HAS_SARKIT, _HAS_SARPY; print(f'sarkit={_HAS_SARKIT}, sarpy={_HAS_SARPY}')"

# Run tests
pytest tests/test_io_*.py -v
```
