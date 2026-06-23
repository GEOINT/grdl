# GRDL Example Scripts → Notebook Conversion Plan

**Date**: 2026-06-22  
**Branch**: `jupyter-examples`  
**Total Scripts**: 34 legacy `.py` examples  
**Converted**: 3 notebooks ✅  
**Remaining**: 31 scripts

---

## Status Dashboard

| Tier | Priority | Scripts | Converted | Remaining |
|------|----------|---------|-----------|-----------|
| 0 | Complete | 3 | ✅ 3 | 0 |
| 1 | Foundational SAR (High) | 4 | ✅ 3 | 1 (skipped) |
| 2 | Image Formation (Medium) | 5 | ✅ 5 | 0 |
| 3 | Orthorectification (Medium) | 8 | ✅ 7 | 1 |
| 4 | Shapes & Geo (Low) | 3 | ✅ 3 | 0 |
| 5 | IO & Catalog (Low) | 8 | ✅ 7 | 1 (skipped) |
| 6 | Interpolation (Low) | 2 | ✅ 2 | 0 |
| **TOTAL** | | **34** | **30** | **4** |

---

## Tier 0: Complete ✅

**Status**: Migration + validation complete

1. ✅ **`image_processing/filtering/phase_gradient.py`**
   - → `notebooks/image_processing/phase_gradient_filtering.ipynb`
   - Complex Lee filter, phase gradient, texture analysis
   
2. ✅ **`image_processing/sar/sublook_compare.py`**
   - → `notebooks/image_processing/sublook_decomposition.ipynb`
   - Azimuth sub-aperture decomposition
   
3. ✅ **`image_processing/sar/dump_cphd_metadata.py`**
   - → `notebooks/image_processing/cphd_metadata_inspection.ipynb`
   - CPHD metadata extraction, PVP derivation

**Validation**: All 3 notebooks validated against merged main (39 commits), IO factory pattern applied.

---

## Tier 1: Foundational Image Processing ⭐⭐⭐

**Priority**: HIGH — Core SAR algorithms, flagship workflows  
**Status**: 3/4 complete ✅

### 1.1 CSI Detection Overlay ⭐⭐⭐ ✅

**Script**: `grdl/example/image_processing/sar/csi_detection_overlay.py`  
**Target Notebook**: `notebooks/image_processing/csi_detection_overlay.ipynb`

**What it demonstrates**:
- CSI (Coherent Shape Index) RGB composite from sub-apertures
- CFAR detection on CSI magnitude
- Overlay detections as shapes on imagery
- End-to-end: sublook → CSI → detect → visualize

**Modules used**:
- `grdl.IO.get_reader('sicd', ...)` (factory pattern)
- `grdl.image_processing.sar.CSIProcessor`
- `grdl.image_processing.detection.cfar.CACFARDetector`
- `grdl.shapes.overlay_shapes()`
- `grdl.contrast.*` for display

**Why high priority**: Flagship SAR detection workflow, integrates 4+ major modules.

**Output directory**: `notebooks/image_processing/`

---

### 1.2 NISAR Pauli Decomposition ⭐⭐⭐ ✅

**Script**: `grdl/example/image_processing/sar/nisar_pauli_decomposition.py`  
**Target Notebook**: `notebooks/image_processing/pauli_decomposition.ipynb` ✅

**What it demonstrates**:
- Quad-pol SAR decomposition (HH, HV, VH, VV → surface/double-bounce/volume)
- Pauli RGB: `|HH - VV|` (blue), `|HV + VH|` (red), `|HH + VV|` (green)
- Physical scattering interpretation

**Modules used**:
- `grdl.IO.get_reader('nisar', ...)` or `grdl.IO.NISARReader`
- `grdl.image_processing.decomposition.PauliDecomposition`
- `grdl.contrast.*` for RGB stretch

**Why high priority**: Multi-polarimetric foundation, visually striking output, demonstrates typed metadata propagation.

**Output directory**: `notebooks/image_processing/`

---

### 1.3 Sentinel-1 H/Alpha Decomposition ⭐⭐ ✅

**Script**: `grdl/example/image_processing/sar/sentinel1_halpha_decomposition.py`  
**Target Notebook**: `notebooks/image_processing/halpha_decomposition.ipynb` ✅

**What it demonstrates**:
- Dual-pol H/Alpha decomposition (entropy/alpha scattering angle)
- Cloude-Pottier classification zones
- Sentinel-1 SLC dual-pol handling

**Modules used**:
- `grdl.IO.get_reader('sentinel1-slc', ...)`
- `grdl.image_processing.decomposition.DualPolHAlpha`
- `grdl.contrast.*`

**Why priority**: Complements Pauli for dual-pol sensors (Sentinel-1 is most common open SAR).

**Output directory**: `notebooks/image_processing/`

---

## Tier 2: Image Formation Processors (IFP) ⭐⭐

**Priority**: MEDIUM — Complex algorithms, debugging visibility benefits from notebooks  
**Status**: 5/5 complete ✅

### 2.1 PFA (Polar Format Algorithm) ⭐⭐⭐ ✅

**Script**: `grdl/example/image_processing/sar/ifp_example.py`  
**Target Notebook**: `notebooks/image_processing/pfa_image_formation.ipynb` ✅

**What it demonstrates**:
- CPHD → SICD image formation via PFA
- Scene sizing (`'full'` vs `'minimal'`)
- Polar grid construction, kernel application

**Modules used**:
- `grdl.IO.get_reader('cphd', ...)`
- `grdl.image_processing.sar.image_formation.PolarFormatAlgorithm`
- `grdl.IO.sar.SICDWriter`

**Why priority**: Core IFP algorithm. Note: existing IFP notebooks may already cover this — merge/consolidate.

**Output directory**: `notebooks/image_processing/`

---

### 2.2 Stripmap PFA ⭐⭐ ✅

**Script**: `grdl/example/image_processing/sar/stripmap_ifp_example.py`  
**Target Notebook**: `notebooks/image_processing/stripmap_pfa.ipynb` ✅

**What it demonstrates**:
- PFA variant for stripmap collections (SRP drift handling)
- Dynamic stripmap mode detection

**Modules used**:
- `grdl.image_processing.sar.image_formation.StripmapPFA`

**Output directory**: `notebooks/image_processing/`

---

### 2.3 RDA (Range-Doppler Algorithm) ⭐⭐ ✅

**Script**: `grdl/example/image_processing/sar/rda_stripmap_example.py`  
**Target Notebook**: `notebooks/image_processing/rda_stripmap.ipynb` ✅

**What it demonstrates**:
- Alternative IFP for stripmap: range compression → azimuth compression
- RDA vs PFA trade-offs

**Modules used**:
- `grdl.image_processing.sar.image_formation.RangeDopplerAlgorithm`

**Output directory**: `notebooks/image_processing/`

---

### 2.4 FFBP (Fast Factorized Back-Projection) ⭐ ✅

**Script**: `grdl/example/image_processing/sar/ffbp_stripmap_example.py`  
**Target Notebook**: `notebooks/image_processing/ffbp_stripmap.ipynb` ✅

**What it demonstrates**:
- Time-domain back-projection (handles arbitrary flight paths)
- Factorized speedup (sub-aperture recursion)

**Modules used**:
- `grdl.image_processing.sar.image_formation.FastBackProjection`

**Output directory**: `notebooks/image_processing/`

---

### 2.5 CPHD → SICD Conversion ⭐ ✅

**Script**: `grdl/example/image_processing/sar/cphd_to_sicd_example.py`  
**Target Notebook**: `notebooks/image_processing/cphd_to_sicd.ipynb` ✅

**What it demonstrates**:
- Format conversion utility (phase history → focused image)
- Metadata mapping (CPHD → SICD)

**Modules used**:
- `grdl.IO.sar.cphd_to_sicd`

**Output directory**: `notebooks/image_processing/`

---

## Tier 3: Orthorectification & Projection ⭐

**Priority**: MEDIUM — Projection/resampling, interactive tuning benefits from notebooks  
**Status**: 7/8 complete ✅ (sicd_ortho_demo merged with ortho_sicd)

### 3.1 SICD Orthorectification ⭐⭐ ✅

**Script**: `grdl/example/ortho/ortho_sicd.py`  
**Target Notebook**: `notebooks/ortho/ortho_sicd.ipynb` ✅

**What it demonstrates**:
- Slant-plane → geographic grid projection
- DEM terrain correction (if DEM available, else ellipsoid)
- UTM/WebMercator output grids

**Modules used**:
- `grdl.IO.get_reader('sicd', ...)`
- `grdl.geolocation.create_geolocation()` (auto-detect)
- `grdl.image_processing.ortho.orthorectify()`
- `grdl.image_processing.ortho.UTMGrid`, `WebMercatorGrid`

**Output directory**: `notebooks/ortho/`

---

### 3.2 SIDD Orthorectification ⭐⭐ ✅

**Script**: `grdl/example/ortho/ortho_sidd.py`  
**Target Notebook**: `notebooks/ortho/ortho_sidd.ipynb` ✅

**What it demonstrates**:
- SIDD (pre-geolocated product) orthorectification
- SIDD vs SICD geolocation differences

**Modules used**:
- `grdl.IO.get_reader('sidd', ...)`
- `grdl.geolocation.SIDDGeolocation.from_reader()`

**Output directory**: `notebooks/ortho/`

---

### 3.3 BIOMASS Orthorectification ⭐⭐ ✅

**Script**: `grdl/example/ortho/ortho_biomass.py`  
**Target Notebook**: `notebooks/ortho/ortho_biomass.ipynb` ✅

**What it demonstrates**:
- ESA BIOMASS L1 GCP-based geolocation
- Delaunay interpolation for GCP grids

**Modules used**:
- `grdl.IO.get_reader('biomass', ...)`
- `grdl.geolocation.GCPGeolocation.from_reader()`

**Output directory**: `notebooks/ortho/`

---

### 3.4 Combined Orthorectification ⭐ ✅

**Script**: `grdl/example/ortho/ortho_combined.py`  
**Target Notebook**: `notebooks/ortho/ortho_combined.ipynb` ✅

**What it demonstrates**:
- Multi-sensor ortho comparison (SICD vs SIDD vs BIOMASS)
- Side-by-side projected output

**Output directory**: `notebooks/ortho/`

---

### 3.5 Chip Orthorectification ⭐ ✅

**Script**: `grdl/example/ortho/chip_ortho.py`  
**Target Notebook**: `notebooks/ortho/chip_ortho.ipynb` ✅

**What it demonstrates**:
- Orthorectifying a sub-region chip (not full image)
- `ChipGeolocation` offset handling

**Modules used**:
- `grdl.geolocation.ChipGeolocation`

**Output directory**: `notebooks/ortho/`

---

### 3.6 Point ROI Orthorectification ⭐ ✅

**Script**: `grdl/example/ortho/point_roi_ortho.py`  
**Target Notebook**: `notebooks/ortho/point_roi_ortho.ipynb` ✅

**What it demonstrates**:
- Orthorectifying a small region around a lat/lon point
- `ortho.roi` module for ROI-based grids

**Modules used**:
- `grdl.image_processing.ortho.roi`

**Output directory**: `notebooks/ortho/`

---

### 3.7 Compare SIDD Orthorectification ⭐ ✅

**Script**: `grdl/example/ortho/compare_sidd_ortho.py`  
**Target Notebook**: `notebooks/ortho/compare_sidd_ortho.ipynb` ✅

**What it demonstrates**:
- SIDD pre-projected vs re-orthorectified comparison

**Output directory**: `notebooks/ortho/`

---

### 3.8 SICD Orthorectification Demo ⭐ ✅

**Script**: `grdl/example/ortho/sicd_ortho_demo.py`  
**Target Notebook**: ✅ Merged with `ortho_sicd.ipynb` (similar content)

**Output directory**: `notebooks/ortho/`

---

## Tier 4: Shapes & Geolocation Utilities ⭐

**Priority**: LOW — Utility demonstrations, simpler workflows  
**Status**: 3/3 complete ✅ (2026-06-23)

### 4.1 Cued Detection Overlay ⭐ ✅

**Script**: `grdl/example/shapes/cued_detection_overlay.py`  
**Target Notebook**: `notebooks/shapes/cued_detection.ipynb` ✅

**What it demonstrates**:
- ROI-cued CFAR detection (only run detector inside geographic shapes)
- `Circle`, `Ellipse`, `GeoPolygon` ROI definition
- `grdl.shapes.cued_detect()`

**Modules used**:
- `grdl.shapes.*`
- `grdl.image_processing.detection.cfar.*`

**Output directory**: `notebooks/shapes/`

---

### 4.2 Error Budget Overlay ⭐ ✅

**Script**: `grdl/example/shapes/error_budget_overlay.py`  
**Target Notebook**: `notebooks/shapes/error_ellipse_overlay.ipynb` ✅

**What it demonstrates**:
- Error ellipse combination (`convolve_ellipses`)
- Uncertainty propagation, covariance arithmetic
- Overlay ellipses on imagery

**Modules used**:
- `grdl.shapes.Ellipse`, `convolve_ellipses`, `overlay_shape()`

**Output directory**: `notebooks/shapes/`

---

### 4.3 Geolocation Overlay ⭐ ✅

**Script**: `grdl/example/geolocation/geolocation_overlay.py`  
**Target Notebook**: `notebooks/geolocation/footprint_overlay.ipynb` ✅

**What it demonstrates**:
- Image footprint computation
- Overlay on slippy maps (web tiles)

**Modules used**:
- `grdl.geolocation.*`
- `grdl.geolocation.utils.footprint()`

**Output directory**: `notebooks/geolocation/`

---

## Tier 5: IO & Catalog ⭐

**Priority**: LOW — Data access workflows, scripts adequate  
**Status**: 7/8 complete ✅ (test_file_loading.py skipped, view_product.py merged)

### 5.1 View SICD ⭐ ✅

**Script**: `grdl/example/IO/sar/view_sicd.py`  
**Target Notebook**: `notebooks/IO/view_sicd.ipynb` ✅

**What it demonstrates**:
- Quick SICD viewer (metadata + chip display)

**Output directory**: `notebooks/IO/`

---

### 5.2 Compare CRSD ⭐ ✅

**Script**: `grdl/example/IO/sar/compare_crsd.py`  
**Target Notebook**: `notebooks/IO/compare_crsd.ipynb` ✅

**What it demonstrates**:
- CRSD file inspection, comparison across products

**Output directory**: `notebooks/IO/`

---

### 5.3 View Sentinel-2 ⭐ ✅

**Script**: `grdl/example/IO/eo/view_sentinel2.py`  
**Target Notebook**: `notebooks/IO/view_sentinel2.ipynb` ✅

**What it demonstrates**:
- Sentinel-2 MSI band access, RGB composite

**Modules used**:
- `grdl.IO.get_reader('sentinel2', ...)`

**Output directory**: `notebooks/IO/`

---

### 5.4 RapidEye NITF RPC ⭐ ✅

**Script**: `grdl/example/IO/eo/rapideye_nitf_rpc.py`  
**Target Notebook**: `notebooks/IO/rapideye_nitf_rpc.ipynb` ✅

**What it demonstrates**:
- EO NITF with RPC geolocation
- `grdl.geolocation.RPCGeolocation`

**Output directory**: `notebooks/IO/`

---

### 5.5 Load Earthdata (HDF5) ⭐ ✅

**Script**: `grdl/example/IO/HDF5/load_earthdata.py`  
**Target Notebook**: `notebooks/IO/load_earthdata.ipynb` ✅

**What it demonstrates**:
- Generic HDF5 reader for NASA Earthdata products

**Output directory**: `notebooks/IO/`

---

### 5.6 Test File Loading ⭐ ⏭️ SKIPPED

**Script**: `grdl/example/IO/test_file_loading.py`  
**Target Notebook**: ⏭️ Skipped (test script, not tutorial)

**Output directory**: N/A

---

### 5.7 Catalog Discovery ⭐ ✅

**Script**: `grdl/example/catalog/catalog_discovery.py`  
**Target Notebook**: `notebooks/catalog/catalog_discovery.ipynb` ✅

**What it demonstrates**:
- Query remote STAC/OData catalogs (BIOMASS, Sentinel-1/2, NISAR)
- Browse available products

**Modules used**:
- `grdl.IO.catalog.*`

**Output directory**: `notebooks/catalog/`

---

### 5.8 Discover and Download ⭐ ✅

**Script**: `grdl/example/catalog/discover_and_download.py`  
**Target Notebook**: `notebooks/catalog/discover_and_download.ipynb` ✅

**What it demonstrates**:
- Full workflow: query → select → download

**Output directory**: `notebooks/catalog/`

---

### 5.9 View Product ⭐ ✅

**Script**: `grdl/example/catalog/view_product.py`  
**Target Notebook**: ✅ Merged with `discover_and_download.ipynb`

**Output directory**: `notebooks/catalog/`

---

## Tier 6: Interpolation ⭐

**Priority**: LOW — Niche signal processing, scripts adequate  
**Status**: 2/2 complete ✅ (2026-06-23)

### 6.1 Polyphase Interpolation ⭐ ✅

**Script**: `grdl/example/interpolation/polyphaseinterpolation.py`  
**Target Notebook**: `notebooks/interpolation/polyphase_interpolation.ipynb` ✅

**What it demonstrates**:
- 1D polyphase resampling (bandwidth-preserving)
- Kernel comparison (Lanczos, Kaiser sinc, Lagrange)

**Output directory**: `notebooks/interpolation/`

---

### 6.2 LFM Polyphase ⭐ ✅

**Script**: `grdl/example/interpolation/lfm_polyphase.py`  
**Target Notebook**: `notebooks/interpolation/lfm_polyphase.ipynb` ✅

**What it demonstrates**:
- Linear Frequency Modulated (LFM) signal resampling
- Signal processing example

**Output directory**: `notebooks/interpolation/`

---

## Conversion Template

Use this template for all new notebooks:

```python
# Cell 1 (Markdown): Title + Description
# - Source script reference: `grdl/example/path/to/script.py`
# - Learning objectives (bulleted list)
# - Data setup instructions (generic paths, no hardcoded values)

# Cell 2 (Python): Preamble
"""GRDL Notebook Preamble — Environment validation and autoreload."""
try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except (NameError, AttributeError):
    pass

try:
    import grdl
except ImportError as exc:
    raise ImportError(
        "\n"
        "═══════════════════════════════════════════════════════════════\n"
        "  grdl is NOT installed in this Python environment.\n"
        "  Install with:  pip install -e '.[all]'\n"
        "  from the grdl repository root.\n"
        "═══════════════════════════════════════════════════════════════\n"
    ) from exc

print(f"grdl {grdl.__version__} — ready")

# Cell 3 (Python): Imports + Config
from pathlib import Path
from grdl.IO import get_reader  # Factory pattern (recommended)
# ... all other imports

# USER CONFIGURATION — Set your file path here
DATA_PATH = Path("/path/to/your/file.ext")  # ← CHANGE THIS
assert DATA_PATH.exists(), f"File not found: {DATA_PATH}"

# Cell 4+ (Python/Markdown): Processing steps
with get_reader('format', DATA_PATH) as reader:
    # ... processing
```

### Mandatory Patterns

1. **IO Factory Pattern**: Use `get_reader('format', path)` instead of direct reader imports
2. **Context Managers**: All readers must use `with` statements
3. **Generic Paths**: No hardcoded personal paths (use `/path/to/your/file.ext` placeholders)
4. **Preamble**: Standard preamble with grdl version check and autoreload
5. **Source Reference**: First markdown cell must reference source script
6. **No sys.path Hacks**: All imports via editable install

---

## Output Directory Structure

```
notebooks/
  validate_notebooks.sh        # Automated validation
  CONVERSION_PLAN.md           # This file
  image_processing/
    phase_gradient_filtering.ipynb          ✅ Complete
    sublook_decomposition.ipynb             ✅ Complete
    cphd_metadata_inspection.ipynb          ✅ Complete
    csi_detection_overlay.ipynb             ✅ Tier 1.1 (2026-06-23)
    pauli_decomposition.ipynb               ✅ Tier 1.2 (2026-06-23)
    halpha_decomposition.ipynb              ✅ Tier 1.3 (2026-06-23)
    csi_detection_workflow.ipynb            ⬜ Tier 1.4
    pfa_image_formation.ipynb               ✅ Tier 2.1 (2026-06-23)
    stripmap_pfa.ipynb                      ✅ Tier 2.2 (2026-06-23)
    rda_stripmap.ipynb                      ✅ Tier 2.3 (2026-06-23)
    ffbp_stripmap.ipynb                     ✅ Tier 2.4 (2026-06-23)
    cphd_to_sicd.ipynb                      ✅ Tier 2.5 (2026-06-23)
  ortho/
    ortho_sicd.ipynb                        ✅ Tier 3.1 (2026-06-23)
    ortho_sidd.ipynb                        ✅ Tier 3.2 (2026-06-23)
    ortho_biomass.ipynb                     ✅ Tier 3.3 (2026-06-23)
    ortho_combined.ipynb                    ✅ Tier 3.4 (2026-06-23)
    chip_ortho.ipynb                        ✅ Tier 3.5 (2026-06-23)
    point_roi_ortho.ipynb                   ✅ Tier 3.6 (2026-06-23)
    compare_sidd_ortho.ipynb                ✅ Tier 3.7 (2026-06-23)
  shapes/
    cued_detection.ipynb                    ✅ Tier 4.1 (2026-06-23)
    error_ellipse_overlay.ipynb             ✅ Tier 4.2 (2026-06-23)
  geolocation/
    footprint_overlay.ipynb                 ✅ Tier 4.3 (2026-06-23)
  IO/
    view_sicd.ipynb                         ✅ Tier 5.1 (2026-06-23)
    compare_crsd.ipynb                      ✅ Tier 5.2 (2026-06-23)
    view_sentinel2.ipynb                    ✅ Tier 5.3 (2026-06-23)
    rapideye_nitf_rpc.ipynb                 ✅ Tier 5.4 (2026-06-23)
    load_earthdata.ipynb                    ✅ Tier 5.5 (2026-06-23)
  catalog/
    catalog_discovery.ipynb                 ✅ Tier 5.7 (2026-06-23)
    discover_and_download.ipynb             ✅ Tier 5.8 (2026-06-23)
  interpolation/
    polyphase_interpolation.ipynb           ✅ Tier 6.1 (2026-06-23)
    lfm_polyphase.ipynb                     ✅ Tier 6.2 (2026-06-23)
```

---

## Validation Checklist (Per Notebook)

Before marking a notebook complete:

- [ ] Preamble executes (Cell 1-2)
- [ ] All imports resolve (no `ImportError`)
- [ ] Generic placeholder paths (no personal data paths)
- [ ] Context managers used (`with get_reader(...)`)
- [ ] Factory pattern used (`get_reader('format', ...)`)
- [ ] Markdown cells reference source script
- [ ] Summary cell documents key patterns
- [ ] Runs `bash notebooks/validate_notebooks.sh` → PASS

---

## Next Action

**Status**: ✅ ALL TIERS COMPLETE (30/34 notebooks, 88%)

**Completion summary**:
- Tier 0 (baseline): ✅ 3/3 complete
- Tier 1 (foundational SAR): ✅ 3/4 complete (Tier 1.4 skipped per user)
- Tier 2 (image formation): ✅ 5/5 complete (2026-06-23)
- Tier 3 (orthorectification): ✅ 7/8 complete (2026-06-23, sicd_ortho_demo merged)
- Tier 4 (shapes & geo): ✅ 3/3 complete (2026-06-23)
- Tier 5 (IO & catalog): ✅ 7/8 complete (2026-06-23, test_file_loading skipped, view_product merged)
- Tier 6 (interpolation): ✅ 2/2 complete (2026-06-23)

**Skipped scripts** (not converted to notebooks):
- Tier 1.4: `csi_detect_workflow.py` (user requested skip)
- Tier 5.6: `test_file_loading.py` (test script, not tutorial)
- Tier 3.8: `sicd_ortho_demo.py` (merged with ortho_sicd.ipynb)
- Tier 5.9: `view_product.py` (merged with discover_and_download.ipynb)
- Tiers 4-6 (14 notebooks): As needed / optional
