# Conflicts Between Documentation and Implementation

## 1. Missing Concrete Implementations (Documented as Architecture, Not Implemented)

| Feature | .md Says | .py Reality |
|---|---|---|
| **ImageWriter** | Full ABC with `write()`, `write_chip()`, streaming writes | ABC only — **zero concrete writers**. No GeoTIFF, NITF, or any output format. |
| **ImageDetector** | Full ABC + usage examples showing concrete detectors | ABC only — **zero production detectors**. Only `DummyDetector` and `_BiasedDetector` in tests. |
| **CRSD Reader** | Listed as "Planned" in TODO, but `sar.py` docstring says *"Provides readers for SICD, CPHD, CRSD"* | **Does not exist.** The docstring in `sar.py` is lying — no `CRSDReader` class anywhere. |
| **SLC Reader** | Listed as planned for v0.2.0 | **Does not exist.** |
| **SICD Geolocation** | Architecture describes factory `Geolocation.from_reader()` dispatching to SICD | **Raises `NotImplementedError`** with message "SICD geolocation is not yet implemented" |
| **Affine Geolocation** | Implied by factory method and GRD reader (geocoded imagery uses affine transforms) | **Raises `NotImplementedError`** with message "Affine transform geolocation is not yet implemented" |
| **EO Geolocation** | Mentioned in architecture (generic EO support) | **Empty placeholder directory** — `grdl/geolocation/eo/` has no implementation |

## 2. API Signature Mismatch

**`DetectionSet` — documented API does not match implementation:**

The .md documentation shows a **builder pattern**:
```python
results = DetectionSet(output_schema=schema)
results.append(det)        # <-- method doesn't exist
```

The actual implementation requires **all arguments upfront**:
```python
results = DetectionSet(
    detections=[det1, det2],       # required list
    detector_name="MyDetector",    # required, not in docs
    detector_version="1.0.0",     # required, not in docs
    output_schema=schema
)
```

Two problems:
- `append()` method **does not exist** on `DetectionSet`
- `detector_name` and `detector_version` are **required** constructor params but not shown in docs

## 3. Docstring / Comment Lies

| Location | Claim | Reality |
|---|---|---|
| `grdl/IO/sar.py` module docstring | "Provides readers for various SAR formats including NGA standards (SICD, CPHD, **CRSD**)" | CRSD reader does not exist |
| `Geolocation.from_reader()` factory | Implies it dispatches to multiple geolocation backends | Only dispatches to GCPGeolocation; SICD and Affine branches raise `NotImplementedError` |

## 4. Aspirational v0.2.0+ Features Conflated with v0.1.0

The `.md` files present these as part of the architecture/design, but they're **unimplemented roadmap items** that could confuse a developer reading the docs alongside the code:

- **Concrete ImageDetector implementations** — v0.2.0 target
- **SIDD reader** — v0.2.0 target
- **Basic EO readers (GeoTIFF, NITF)** — v0.2.0 target
- **Generic multi-format catalog** — v0.3.0 target
- **Geospatial writers (GeoJSON, Shapefile, KML)** — v0.3.0 target

## 5. What Actually Matches (No Conflicts)

These all match between docs and implementation:
- `Tiler.tile_positions()` (index-only, returns `List[ChipRegion]`)
- `ChipExtractor.chip_at_point()` / `chip_positions()` (index-only, returns `ChipRegion`)
- `Normalizer.normalize()` / `fit()` / `transform()`
- `Pipeline` constructor and `apply()`
- `OutputGrid.from_geolocation()` with `pixel_size_lat`/`pixel_size_lon`
- `PauliDecomposition.to_rgb()` with `representation` param
- `DetectionSet.to_geojson()` and `filter_by_confidence()`
- All 12 ImageJ ports (implemented and exported)
- All 3 co-registration strategies (Affine, Projective, FeatureMatch)
- BIOMASS L1 reader + catalog + OAuth2
- SICD, CPHD, GRD readers
- GCPGeolocation
- Exception hierarchy

## Summary

**Critical conflicts (code says one thing, reality is another):**
1. `sar.py` docstring claims CRSD support — it doesn't exist
2. `DetectionSet` documented API (`append()`, simple constructor) doesn't match implementation
3. `Geolocation.from_reader()` factory has dead branches that raise `NotImplementedError`

**Missing implementations (docs describe the vision, code has only ABCs):**
4. No concrete `ImageWriter` — can read imagery but can't write any format
5. No concrete `ImageDetector` — detection framework exists but no actual detectors
6. No SICD/Affine geolocation — only GCP-based works
7. No CRSD or SLC readers despite being referenced

The library is **read-heavy and write-absent** — strong on ingestion (4 readers, catalog, OAuth2) but has no output path for processed imagery.
