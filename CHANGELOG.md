# Changelog

All notable changes to GRDL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **Robust MAD normalization** (`grdl/data_prep`): `Normalizer` gains a
  ``'mad'`` method -- robust z-score ``(x - median) / (1.4826 * MAD)`` where
  ``MAD = median(|x - median(x)|)``. Outlier-resistant (50%% breakdown point);
  the ``1.4826`` consistency constant is exported as
  `grdl.data_prep.MAD_TO_STD`.
- `compute_image_statistics(..., mad=True)` and `Normalizer.fit_streaming`
  (for ``method='mad'``) compute the streaming median and MAD over a full
  image. The MAD is taken about the median, so it adds one deviation pass
  (non-negative single-pass float32 histogram) on top of the median pass.
- `StatsResult` gains `median` and `mad` fields and a `mad_std` property
  (``1.4826 * mad``). Both stay ``nan`` unless ``mad=True`` was requested; the
  internal median percentile is not leaked into ``percentiles``.

---

## [0.6.1] — 2026-06-17

### Added

- **Reader factory** (`grdl/IO/__init__.py`): `_READER_REGISTRY` — a 20-entry
  module-level dict mapping case-insensitive format keys (e.g. `'sicd'`,
  `'geotiff'`, `'sentinel1-slc'`) to `(module_path, ClassName)` tuples for
  lazy loading, mirroring the existing `_WRITER_REGISTRY`.
- `get_reader(format, filepath)` — symmetric counterpart to `get_writer`.
  Normalizes the key, lazy-imports the module, and raises a clean `ImportError`
  pointing to `requirements-optional.txt` when an optional dependency is absent.
- `list_reader_formats()` — returns `sorted(_READER_REGISTRY.keys())` for
  runtime enumeration of all registered reader formats.
- `register_reader(format, module_path, class_name, overwrite=False)` and
  `register_writer(...)` — runtime registration of custom readers/writers
  into the factory (keys normalized to lowercase-hyphen; `ValueError` on
  conflict unless `overwrite=True`).
- `_READER_EXTENSION_MAP` — 14-extension map (`.tif`, `.nitf`, `.h5`, `.jp2`,
  etc.) to registry keys, used by `open_reader()`.
- `open_reader(filepath)` — new primary auto-detect entry point, replacing
  `open_image()`. Extension → `get_reader()` fast path; captures `ImportError`
  into `_missing_lib_msg` and falls through to `open_any()`. Emits a
  `UserWarning` naming the missing package when a GDAL or
  `InvasiveProbeReader` fallback is used.
- `ImageReader._enforce_2d: bool = False` — class attribute; SAR readers that
  return single-channel data override this to `True`.
- `ImageReader._assert_2d(data, context, strict)` — static method that
  validates or squeezes an array to `(rows, cols)`. In strict mode, raises
  `ValueError` naming the reader class and method when a singleton band axis
  is present (e.g. `(1, R, C)`), catching backend shape regressions at the
  earliest possible point.

### Changed

- `open_any()` (`grdl/IO/generic.py`): accumulates `ImportError` and
  `DependencyError` messages from every skipped specialized reader into
  `_import_failures: List[str]`. Emits a consolidated `UserWarning` listing
  all missing libraries with bullet points before degrading to
  `GDALFallbackReader` (Phase 3) and again before `InvasiveProbeReader`
  (Phase 5), giving users an actionable path to resolve environment issues.
- `SICDReader`, `CPHDReader`, `CRSDReader`: set `_enforce_2d = True` and
  wrap the return paths of `read_chip()` and `read_full()` in
  `self._assert_2d(data, context, strict=self._enforce_2d)`, guaranteeing
  `(rows, cols)` output for all single-pol reads.
- `PolarFormatAlgorithm` / `PolarGrid`: default `scene_sizing` changed from
  `'toa'` to `'full'` — output grids are now sized to the data-unambiguous
  full-scene extent (FX swath in range, pulse-Nyquist in azimuth) instead of
  the receive-window grazing-angle heuristic, fixing ~50% cross-range
  cropping on space-based spotlight CPHDs. `'toa'` remains available.
- `IO.gmti`: consolidated `cphd_steering` onto the validated GMTI
  steering-matrix algorithm (`build_steering_matrix_from_cphd_metadata`),
  SRP-relative polynomial convention.

### Deprecated

- `open_image(filepath)` — replaced by `open_reader(filepath)`. `open_image`
  now emits a `DeprecationWarning` and delegates to `open_reader`. Will be
  removed in a future major release.

---

## [0.6.0] — 2026-06-09

### Added

- **Heterogeneous multi-image NITF support** in `EONITFReader`. Segments
  are grouped by compatible imaging characteristics (bands, dtype,
  ICHIPB `SCALE_FACTOR`, IMAG magnification, ICAT); the primary imagery
  group is auto-discovered (full resolution → geolocated → largest) and
  unified into one full-image grid. Mixed-segment files (overviews,
  cloud masks, support images) no longer raise `ValueError` at load.
  - New placement authority chain per group: ICHIPB chip→full affine →
    ILOC/IALVL attachment chain into the common coordinate system
    (MIL-STD-2500C) → sequential row stacking fallback (warns).
  - TRE aggregation, RPC extraction, and full-image dims are scoped to
    the primary group so `RSMGeolocation.from_reader()` /
    `RPCGeolocation.from_reader()` and `read_chip()` share the primary
    full-image coordinate space.
  - New `ImageGroupInfo` model and `EONITFMetadata.image_groups`;
    `ImageSegmentInfo` gains `group_id`, `is_primary`, `bands`,
    `dtype`, `iid1`, `icat`, `irep`, `imag`, `idlvl`, `ialvl`,
    `iloc_row`, `iloc_col`, and `placement`. Non-primary segments stay
    readable via `image_index` pinning.
- **EO NITF reader build-out** — coverage of the full STDI-0002 /
  MIL-STD-2500C permutation space plus read-path accelerations:
  - **TRE-overflow DES**: `load_xml_tres` walks `xml:DES` so TREs
    overflowed into a `TRE_OVERFLOW` Data Extension Segment (large RSM
    payloads) parse identically to subheader TREs.
  - **RSM error model**: RSMPIA, RSMDCA/RSMDCB, RSMECA/RSMECB,
    RSMAPA/RSMAPB parsing (`grdl.IO.models.rsm_error`,
    `grdl.IO.eo._tre_rsm_error`); `AccuracyInfo` now prefers rigorous
    CE90/LE90 derived from RSM error covariance over CSEXRA/USE00A.
  - **Corner-fallback geolocation**: `CornerGeolocation`
    (`grdl.geolocation.eo.corner`) from CSCRNA → BLOCKA → IGEOLO
    (ICORDS G/C/D/N/S/U incl. full MGRS decode);
    `create_geolocation()` now always returns a model for a NITF with
    any corner source. New `CSCRNAMetadata` + CSCRNA parsers.
  - **Band + airborne TREs**: BANDSB/BANDSA (`grdl.IO.models.eo_band`)
    feed `band_names`/`wavelengths`; SENSRB, MENSRB, MENSRA, ACFTB
    (`grdl.IO.models.eo_airborne`) for airborne EO.
  - **Pixel-domain support**: `read_mask()` (NITF pad-pixel / block
    masks, gap-aware in unified mode), `get_lut()` (IREP=LU color
    tables), `normalize_abpp()` (true ABPP bit-depth scaling).
  - **Chip-out writer**: `grdl.IO.eo.write_chip()` writes a NITF chip
    carrying composed ICHIPB + RPC00B + serialized RSMIDA/RSMPCA so
    chips geolocate identically to the parent.
  - **NITF dispatch in `open_any()`**: sniffs SICD/SIDD XML DES and
    routes SAR NITFs to `SICDReader`/`SIDDReader`, EO NITFs to
    `EONITFReader`.
  - **Remote NITFs**: `EONITFReader` accepts `https://`, `s3://`, and
    `/vsi*` URIs (GDAL range reads); `ReadConfig.gdal_env` +
    `apply_gdal_env()` for GDAL tuning.

### Performance

- **Decimated reads**: `read_chip(..., decimation=N)` served from
  matching IMAG overview segment groups, GDAL `out_shape` (embedded
  overviews / J2K resolution levels), or full-resolution read + slice
  fallback.
- **Parallel segment stitching**: unified multi-segment chips fan
  per-segment windows across the `ReadConfig` thread pool.
- **Parallel segment opening**: multi-image discovery opens + parses
  all subdatasets concurrently; unreadable segments are skipped with a
  warning instead of failing the file.
- **Nested DTED archive discovery**: `DTEDElevation` finds tiles in
  nested layouts (resolution subdirectories, per-longitude folders)
  via bbox-driven candidate paths; without a bbox a recursive scan
  discovers tiles at any depth. `open_elevation` routes both layouts.
- **DEM query deduplication**: repeat quantized coordinates within one
  DEM-cache query collapse to a single sample, bounding query count by
  distinct DEM cells during the R/Rdot vertical-step search.
- **Explicit DEM kwargs on `from_reader`**: every Geolocation factory
  (SICD, SIDD, GCP, NISAR, Sentinel-1, Affine, RPC, RSM) now takes
  `dem_path`, `geoid_path`, and `interpolation` explicitly instead of
  `**kwargs` pass-through.

### Changed

- **Single-pass percentiles in streaming statistics.** `StreamingStats`
  gains a `'float32'` histogram spacing (now the default) that bins values
  by the top bits of their IEEE-754 total-order key — a fixed,
  data-independent geometry with ~0.55% relative bin width at 65536 bins,
  needing no data-range pass. `compute_image_statistics` /
  `Normalizer.fit_streaming` (`hist_spacing='auto'`) now compute
  percentiles in the same pass as the exact moments: the image is read
  (and any process pool spun up) once instead of twice. Explicit
  `'log'`/`'linear'` spacing keeps the previous two-pass behavior.
- **Tile-local valid-data masks.** `'metadata'`/`'both'` masking
  rasterizes the valid-data polygon per tile instead of materializing a
  full-image boolean mask per worker (rows × cols bytes each on large
  imagery), and tiles wholly outside the polygon's bounding box are
  skipped without reading. Pixels exactly on the polygon outline may
  differ from the full-image rasterization by sub-pixel rounding where
  edges cross tile windows (~1e-5 of valid pixels on a 43-vertex
  Umbra polygon; interior fill is exact).

### Fixed

- **`'nonzero_finite'`/`'both'` masking now tests raw pixel values**,
  before the value transform. Previously zero-fill pixels passed the
  nonzero test under `transform='decibel'` (floored to -200 dB instead of
  0) and skewed the statistics.
- **Histogram binning no longer promotes float32 tiles to float64.** The
  log-spacing floor used `np.finfo(np.float64).tiny`, silently promoting
  the whole tile (~3x slower `log10`, double the memory bandwidth); the
  floor now matches the tile's dtype. `_decibel` likewise stays in the
  magnitude's native float dtype.

## [0.5.0] — 2026-04-26

### Added

- **New top-level `grdl.contrast` module** for display-time dynamic range
  adjustment across SAR / EO / MSI / PAN / HSI imagery. Full port of
  `sarpy.visualization.remap` plus generic stretches.
  - SAR (sarpy ports): `MangisDensity`, `Brighter`, `Darker`,
    `HighContrast`, `GDM`, `PEDF`, `NRLStretch`, `LogStretch`
  - Generic: `LinearStretch`, `GammaCorrection`, `SigmoidStretch`,
    `HistogramEqualization`, `CLAHE`
  - Re-exports from `image_processing.intensity`: `PercentileStretch`,
    `ToDecibels`
  - `auto_select(metadata)` modality-driven dispatcher (SAR → `Brighter`,
    EO NITF → `gamma`, MSI/unknown → `percentile`)
  - Helper utilities: `clip_cast`, `linear_map`, `nan_safe_stats`
  - All operators are `ImageTransform` subclasses with
    `@processor_version` / `@processor_tags` metadata; output is
    `float32` in `[0, 1]`; cross-tile consistency via per-call kwargs
    (`data_mean=`, `min_value=`/`max_value=`, `stats=`)
- **`orthorectify_point_roi()` + `PointRoiResult`** in
  `grdl.image_processing.ortho` — single-call helper for flat-plane ENU
  orthorectification centered on a lat/lon or pixel point. Auto-detects
  geolocation from any GRDL reader; samples a DEM at the center for
  flat-plane terrain; handles complex SAR via `complex_mode='magnitude'`
  or `'complex'`.
- **Lanczos-3 interpolation** in `Orthorectifier`. Use
  `interpolation='lanczos'` (maps to numba backend order 5). Keeps
  detail under both up-sampling and down-sampling.
- **SIDD dispatch** in `compute_output_resolution`. Resolves native
  pixel spacing from `measurement.plane_projection`,
  `cylindrical_projection`, exploitation-features product resolution,
  or `geographic_projection`.
- **Contrast-aware example**: `grdl/example/ortho/point_roi_ortho.py` +
  YAML config. Runs on any GRDL-supported format; auto-selects a
  display contrast from the reader's metadata.
- **Doc pages**: `grdl/contrast/README.md`, `grdl/contrast/ARCHITECTURE.md`,
  `grdl/image_processing/README.md`, `grdl/vector/README.md`.

### Changed

- `_build_elevation_model` / `open_elevation` now correctly **fall back
  to `ConstantElevation`** when a DEM directory contains no usable
  tiles. Previously returned an empty `DTEDElevation` (silent failure).
- Expanded `SIDDReader` and metadata coverage (parsing breadth +
  stacked typed metadata).
- Tiled GeoTIFF DEM and geoid backend refactor for clearer geoid
  handling.

### Test surface

- 65 new contrast tests + updated elevation tests
- Total: **2,212 passing** (+~280 over v0.4.3)
- Numerical regression tests against `sarpy.visualization.remap` for
  `LogStretch`, `MangisDensity`, `NRLStretch`

### Install

```bash
pip install grdl==0.5.0
pip install grdl[contrast]==0.5.0    # for CLAHE (scikit-image)
pip install grdl[all]==0.5.0
```

---

## [0.4.3] — 2026-04-19

- Bump version

## [0.4.2] — 2026-04-19

- Scrub personal email from `pyproject.toml` authors

## Earlier

See `git log` for the full history. Earlier releases predate this
changelog and are documented in commit messages.
