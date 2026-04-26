# Changelog

All notable changes to GRDL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

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
