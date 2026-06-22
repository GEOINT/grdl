# IO Module — Implementation Patterns

Reference for recurring patterns in `grdl/IO/`. Follow these when adding new readers, writers, TRE parsers, or metadata models.

---

## 1. Reader ABC Contract

**File:** `base.py`

Every reader inherits `ImageReader` and implements four abstract methods. The constructor calls `_load_metadata()` automatically — pixel data is never loaded until `read_chip()`.

```
__init__(filepath)          # validates path, calls _load_metadata()
├── _load_metadata()        # ABSTRACT — populate self.metadata
├── read_chip(r0, r1, c0, c1, bands)   # ABSTRACT — windowed pixel access
├── get_shape()             # ABSTRACT — (rows, cols) or (rows, cols, bands)
├── get_dtype()             # ABSTRACT — np.dtype
├── read_full(bands)        # CONCRETE — delegates to read_chip with full extent
└── close()                 # CONCRETE (empty default) — override if resources need cleanup
```

**Rules:**
- Indices are 0-based, half-open: `[row_start, row_end)`, `[col_start, col_end)`
- `bands` parameter is 0-based; `None` means all bands
- Validate bounds in `read_chip()` before backend call; raise `ValueError` for OOB
- All readers support context managers (`with Reader(...) as r:`)
- Single-band output is squeezed to 2D `(rows, cols)`; multi-band is `(bands, rows, cols)`
- Single-channel SAR readers (SICD, CPHD, CRSD) set `_enforce_2d = True` — call `self._assert_2d(data, context=..., strict=self._enforce_2d)` before returning to guarantee the contract and catch backend shape regressions early

---

## 2. Backend Availability Guard

**Files:** `eo/_backend.py`, `sar/_backend.py`, `ir/_backend.py`, `multispectral/_backend.py`

Guard optional dependencies at module level. Fail fast at construction, not at read time.

```python
# Module level — always succeeds
try:
    import rasterio
    from rasterio.windows import Window
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

# Constructor — fails before parent __init__
class MyReader(ImageReader):
    def __init__(self, filepath):
        if not _HAS_RASTERIO:
            raise DependencyError(
                "rasterio is required for <format> reading. "
                "Install with: conda install -c conda-forge rasterio"
            )
        super().__init__(filepath)
```

**Rules:**
- Flag naming: `_HAS_<LIBRARY>` (e.g., `_HAS_RASTERIO`, `_HAS_H5PY`, `_HAS_SARKIT`)
- Check **before** `super().__init__()` so the reader is never half-initialized
- Error message includes exact install command (conda preferred, pip fallback)

---

## 3. Typed Metadata Dataclass

**Files:** `models/base.py`, `models/eo_nitf.py`, `models/sicd.py`, etc.

Metadata is always a `@dataclass` inheriting from `ImageMetadata`. Sensor-specific fields are typed; unknown fields go to `extras: Dict[str, Any]`.

```python
@dataclass
class MyFormatMetadata(ImageMetadata):
    """Metadata for <format> imagery."""
    sensor_specific_field: Optional[float] = None
    coefficients: Optional[MyCoefficients] = None
```

**Rules:**
- All sensor-specific fields are `Optional` with `None` default
- `ImageMetadata` implements dict-like access (`metadata['rows']`) for backward compat
- NumPy arrays in dataclasses use `field(default_factory=lambda: np.zeros(N))`
- Keep `from datetime import datetime` imports when adding datetime fields
- Nested dataclasses (e.g., `RSMCoefficients`, `ICHIPBMetadata`) are defined in the same models file

---

## 4. TRE Parser

**File:** `eo/nitf.py`

NITF Tagged Record Extensions are fixed-width CEDATA strings parsed positionally. Every TRE gets its own parser function.

```python
def _parse_<TRE_NAME>_tre(tre_value: str) -> Optional[<TypedDataclass>]:
    """Parse a <TRE_NAME> TRE CEDATA string.

    Field layout per STDI-0002 <Volume/Appendix/Table>.
    """
    try:
        v = tre_value.strip()
        if len(v) < <MIN_BYTES>:
            return None

        pos = 0

        def read_str(n: int) -> str:
            nonlocal pos
            s = v[pos:pos + n].strip()
            pos += n
            return s

        def read_float(n: int = 21) -> float:
            nonlocal pos
            raw = v[pos:pos + n].strip()
            pos += n
            return float(raw)

        def read_int(n: int) -> int:
            nonlocal pos
            raw = v[pos:pos + n].strip()
            pos += n
            return int(raw)

        # Extract fields in spec order...
        field_a = read_str(80)
        field_b = read_float()

        return <TypedDataclass>(field_a=field_a, field_b=field_b)
    except (ValueError, IndexError):
        return None
```

**Rules:**
- Parser **never raises** — returns `None` on any format anomaly
- Outer `try/except (ValueError, IndexError)` catches all parse failures
- Field widths are per STDI-0002 specification; comment the spec reference
- Use `read_opt_float(n)` variant for fields that can be empty/missing (returns `None`)
- Position counter (`pos`) is mutated via `nonlocal` in closures
- Add `_parse_<TRE>_tre` to the reader's namespace loop in `_load_metadata()`

**Integrating a new TRE into `_load_metadata()`:**

```python
# In the TRE scanning loop:
for key, val in tags.items():
    key_upper = key.upper()
    if '<TRE_NAME>' in key_upper and my_var is None:
        my_var = _parse_<TRE_NAME>_tre(val)
```

---

## 5. Multi-TRE Aggregation

**File:** `eo/nitf.py`

When multiple TREs provide overlapping information, aggregate with explicit priority.

```python
def _build_accuracy_info(
    csexra: Optional[CSEXRAMetadata],
    use00a: Optional[USE00AMetadata],
    rpc: Optional[RPCCoefficients],
) -> Optional[AccuracyInfo]:
    """Priority: CSEXRA > USE00A > RPC err_bias."""
    if csexra and csexra.ce90 is not None:
        return AccuracyInfo(ce90=csexra.ce90, source='CSEXRA')
    if use00a and use00a.mean_gsd is not None:
        return AccuracyInfo(mean_gsd=use00a.mean_gsd, source='USE00A')
    if rpc and rpc.err_bias is not None:
        return AccuracyInfo(ce90=rpc.err_bias, source='RPC00B')
    return None
```

**Rules:**
- Document priority order in the docstring
- Return `None` when no source provides the field
- Store the `source` name so consumers know which TRE provided the value

---

## 6. Multi-Segment TRE Collection

**File:** `eo/nitf.py`

Some TREs (e.g., RSMPCA) appear multiple times in a single NITF. Collect all instances.

```python
# In _load_metadata():
rsm_segments_list: List[RSMCoefficients] = []

for key, val in tags.items():
    if 'RSMPCA' in key_upper:
        seg = _parse_rsmpca_tre(val)
        if seg is not None:
            rsm_segments_list.append(seg)

# After loop: build grid
if rsm_segments_list:
    rsm = rsm_segments_list[0]  # backward compat
    seg_dict = {(s.rsn or 1, s.csn or 1): s for s in rsm_segments_list}
    rsm_segments = RSMSegmentGrid(
        num_row_sections=nrg, num_col_sections=ncg, segments=seg_dict)
```

**Rules:**
- Do NOT use `and var is None` guard for multi-instance TREs
- Index by the TRE's section identifiers (e.g., `(rsn, csn)`)
- Keep backward-compatible single-instance field (`rsm`) set to first segment

---

## 7. ReadConfig Parallel Integration

**Files:** `performance.py`, `geotiff.py`, `eo/nitf.py`

Opt-in parallel I/O for rasterio-based readers. Thread safety requires GDAL backend.

```python
class MyReader(ImageReader):
    def __init__(self, filepath, read_config=None):
        self.read_config = read_config or ReadConfig(parallel=True)
        super().__init__(filepath)

    def read_chip(self, row_start, row_end, col_start, col_end, bands=None):
        # ... bounds validation ...
        window = Window(col_start, row_start, width, height)

        cfg = self.read_config
        if cfg.parallel:
            _ensure_gdal_threads(cfg)
            workers = _resolve_workers(cfg)
            n_pixels = width * height

            if n_pixels >= cfg.chunk_threshold:
                data = chunked_parallel_read(dataset, window, band_indices, workers)
            elif len(band_indices) > 1:
                data = parallel_band_read(dataset, window, band_indices, workers)
            else:
                data = dataset.read(band_indices, window=window)
        else:
            data = dataset.read(window=window)
        # ... squeeze single band ...
```

**Rules:**
- Default `parallel=True` for rasterio readers (GeoTIFF, EO NITF, BIOMASS)
- Default `parallel=False` for fallback readers and mock-heavy test targets
- **Never** use for h5py backends (not thread-safe with shared handle)
- **Never** use for glymur backends (not thread-safe)
- `_ensure_gdal_threads()` is called once per process (global flag)
- Decision cascade: chunked (large window) > parallel bands (multi-band) > serial (single band)

**Thread safety by backend:**

| Backend | Thread-safe reads? | Use parallel? |
|---------|-------------------|---------------|
| rasterio (GDAL) | Yes (releases GIL) | Yes |
| h5py | No (GIL held) | No |
| glymur | No | No |
| sarpy/sarkit | Unknown | No (default) |

---

## 8. Multi-File Reader (Concurrent Datasets)

**File:** `sar/biomass.py`

When a product stores data across multiple files (e.g., magnitude + phase), read them concurrently.

```python
cfg = self.read_config
if cfg.parallel:
    _ensure_gdal_threads(cfg)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        mag_future = pool.submit(self.magnitude_dataset.read, bands, window=window)
        phase_future = pool.submit(self.phase_dataset.read, bands, window=window)
        mag_data = mag_future.result()
        phase_data = phase_future.result()
else:
    mag_data = self.magnitude_dataset.read(bands, window=window)
    phase_data = self.phase_dataset.read(bands, window=window)
```

**Rules:**
- Cap workers to number of independent datasets (2 for mag+phase)
- Each dataset has its own file handle — no shared state
- Reconstruct complex data **after** both reads complete

---

## 9. SAFE Directory Resolver

**Files:** `sar/sentinel1_slc.py`, `eo/sentinel2.py`

Products in directory structures (SAFE, TerraSAR-X) need flexible path resolution.

```python
def _resolve_safe_dir(filepath: Path) -> Path:
    """Accept SAFE dir, measurement file, or any file inside the tree."""
    if filepath.suffix == '.SAFE' and filepath.is_dir():
        return filepath
    # Walk up looking for manifest.safe
    current = filepath
    while current != current.parent:
        if (current / 'manifest.safe').exists():
            return current
        current = current.parent
    raise ValueError(f"No SAFE root found for: {filepath}")
```

**Rules:**
- Accept the product directory, any file inside it, or a specific data file
- Walk upward to find the sentinel file (`manifest.safe`, main XML, etc.)
- Discover available sub-products (swaths, polarizations, bands) eagerly at init
- Provide helpful error messages listing what was found vs. requested

---

## 10. XML Annotation Parser

**Files:** `sar/sentinel1_slc.py`, `sar/terrasar.py`

Sensor products with XML metadata use stateless extractor functions.

```python
def _xml_text(elem: ET.Element, path: str) -> Optional[str]:
    """Safe XPath text extraction."""
    node = elem.find(path)
    return node.text.strip() if node is not None and node.text else None

def _xml_float(elem: ET.Element, path: str) -> Optional[float]:
    t = _xml_text(elem, path)
    return float(t) if t else None

def _extract_product_info(root: ET.Element) -> ProductInfo:
    return ProductInfo(
        mission=_xml_text(root, 'adsHeader/missionId'),
        mode=_xml_text(root, 'adsHeader/mode'),
    )
```

**Rules:**
- Helpers return `None` for missing elements — never raise
- Strip XML namespaces if vendor includes them: `_strip_namespace(root)`
- One extractor function per logical XML section
- Extractors return typed dataclasses, not dicts
- Parse all XMLs (annotation, calibration, noise) independently in `_load_metadata()`

---

## 11. Reader / Writer Factory Registry

**File:** `__init__.py`

Both readers and writers use symmetric module-level registries for lazy loading. Registering a new reader or writer in the appropriate dict (or via `register_reader()` / `register_writer()` at runtime) is sufficient — no other code changes are needed. The reader registry has 20 entries; the writer registry has 5.

```python
# Reader registry (20 entries)
_READER_REGISTRY: Dict[str, tuple] = {
    'sicd':    ('grdl.IO.sar.sicd',   'SICDReader'),
    'geotiff': ('grdl.IO.geotiff',    'GeoTIFFReader'),
    # ... geotiff, nitf, hdf5, jpeg2000, sicd, cphd, crsd, sidd, biomass,
    #     sentinel1-slc, sentinel1-l0, terrasar, nisar, sentinel2, eo-nitf,
    #     aster, viirs, stanag4607, gdal, probe
}

# Writer registry
_WRITER_REGISTRY: Dict[str, tuple] = {
    'geotiff': ('grdl.IO.geotiff',    'GeoTIFFWriter'),
    'numpy':   ('grdl.IO.numpy_io',   'NumpyWriter'),
    # ...
}

def get_reader(format: str, filepath) -> ImageReader:
    key = _normalize_reader_key(format)   # lowercase, strip, '_' → '-'
    if key not in _READER_REGISTRY:
        raise ValueError(f"Unknown reader format: {format!r}. "
                         f"Supported formats: {sorted(_READER_REGISTRY.keys())}.")
    module_path, class_name = _READER_REGISTRY[key]
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Reader '{format}' requires optional dependencies that are "
            f"not installed: {exc}. See requirements-optional.txt."
        ) from exc
    return getattr(module, class_name)(filepath)

def get_writer(format: str, filepath, metadata=None) -> ImageWriter:
    key = format.strip().lower()
    if key not in _WRITER_REGISTRY:
        raise ValueError(f"Unknown writer format: {format!r}. "
                         f"Supported formats: {sorted(_WRITER_REGISTRY.keys())}")
    module_path, class_name = _WRITER_REGISTRY[key]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(filepath, metadata=metadata)
```

**Runtime registration** — extend either registry without editing the library:

```python
def register_reader(format, module_path, class_name, overwrite=False):
    key = _normalize_reader_key(format)
    if not overwrite and key in _READER_REGISTRY:
        raise ValueError(f"Reader format {format!r} is already registered. "
                         "Use overwrite=True to replace it.")
    _READER_REGISTRY[key] = (module_path, class_name)
# register_writer() is identical against _WRITER_REGISTRY
```

**Rules:**
- Both registries use `(module_path, ClassName)` tuples — imports are deferred until the factory is called
- Reader keys are normalized (`lowercase`, `_` → `-`) so `'sentinel1-slc'` == `'sentinel1_slc'`; writer keys are only lowercased
- `get_reader` raises `ImportError` with a `requirements-optional.txt` reference for missing optional libraries
- Reader extension auto-detection via `_READER_EXTENSION_MAP`; writer extension via `_EXTENSION_MAP`
- `register_reader()` / `register_writer()` add or (with `overwrite=True`) replace entries at runtime; default `overwrite=False` guards against clobbering
- Use `list_reader_formats()` to enumerate registered reader keys at runtime
- Convenience `write()` handles the writer lifecycle (open, write, close)
- `open_reader()` wraps `get_reader` with extension-based dispatch and `open_any()` fallback; on `ImportError`/`ValueError` from a specialized reader it falls through and emits a `UserWarning` if a fallback reader handles the file
- `open_image()` is a deprecated alias for `open_reader()` (emits `DeprecationWarning`) — use `open_reader()` in new code
- Writers that take a typed metadata container instead of an ndarray (`SICDWriter`, `SIDDWriter`, `STANAG4607Writer`) are **not** registered — call them directly

## 12. Strict 2-D Shape Enforcement (`_enforce_2d`)

**Files:** `base.py`, `sar/sicd.py`, `sar/cphd.py`, `sar/crsd.py`

Single-channel SAR readers guarantee a `(rows, cols)` return shape and catch backend regressions at read time.

```python
class MySinglePolReader(ImageReader):
    _enforce_2d: bool = True   # guarantees (rows, cols) output

    def read_chip(self, row_start, row_end, col_start, col_end, bands=None):
        # ... backend call ...
        data = backend.read(...)          # may return (1, R, C) depending on version
        return self._assert_2d(
            data,
            context=f'{type(self).__name__}.read_chip',
            strict=self._enforce_2d,      # True → ValueError on singleton axis
        )
```

`ImageReader._assert_2d(data, context, strict)` behavior:

| Input shape | `strict=False` | `strict=True` |
|-------------|----------------|---------------|
| `(R, C)` | returned as-is | returned as-is |
| `(1, R, C)` | squeezed → `(R, C)` | `ValueError` with context |
| `(R, C, 1)` | squeezed → `(R, C)` | `ValueError` with context |
| `(N, R, C)` N>1 | `ValueError` | `ValueError` |

**Rules:**
- Set `_enforce_2d = True` on every new single-pol/single-channel reader
- Always pass `context=f'{type(self).__name__}.read_chip'` so error messages identify the exact reader
- Do **not** set `_enforce_2d = True` on multi-band readers (GeoTIFFReader, VIIRSReader, etc.)
- `_enforce_2d = False` on the base class — existing readers are unaffected by default

## 13. Optional-Feature Stub Fallback

**File:** `sar/sentinel1_l0/__init__.py`

When a *subset* of a subpackage depends on a heavier optional dependency, guard
only that subset at import time and swap in stub callables that raise a clear
`ImportError` when actually invoked. The core reader stays importable.

```python
from grdl.IO.sar.sentinel1_l0.reader import (
    ReaderConfig, Sentinel1L0Reader, open_safe_product,  # always available
)

try:
    from grdl.IO.sar.sentinel1_l0.crsd_converter import (
        Sentinel1L0ToCRSD, convert_s1_l0_to_crsd,
    )
except ImportError as _err:
    _MSG = ("CRSD conversion dependencies are not installed. "
            "Install the optional dependencies to use this feature.")

    def convert_s1_l0_to_crsd(*args, **kwargs):
        raise ImportError(_MSG) from _err
    # ... same stub for Sentinel1L0ToCRSD, verify_crsd_split_gates ...
```

**Rules:**
- Guard only the optional-feature imports, never the core reader
- Each stub re-raises the original `ImportError` via `from _err` so the root cause is preserved
- Keep the public name in `__all__` so `import` succeeds and the failure surfaces only on call
- Differs from §2 (whole-module backend guard) — here the module imports fine and only the optional capability is degraded

## 14. Non-Raster Reader (Report Streams)

**Files:** `gmti/stanag4607.py`

Some formats are not raster imagery and should **not** inherit `ImageReader`.
STANAG 4607 GMTI is a stream of dwell/target reports, so `STANAG4607Reader` is a
plain class with iterator methods and a bridge into the GRDL detection model.

```python
class STANAG4607Reader:
    def iter_packets(self): ...
    def iter_dwells(self): ...
    def iter_target_reports(self):     # yields (dwell, target)
        for dwell in self.metadata.dwells:
            for target in dwell.target_reports:
                yield dwell, target

    def to_detection_set(self, confidence_field='gmti.snr_db',
                         snr_normalization=40.0) -> 'DetectionSet':
        # each target → Detection(geo_geometry=Point(lon, lat), ...)
        ...
```

**Rules:**
- Do not force the `read_chip`/`get_shape` raster contract onto non-raster data
- Still register it (`stanag4607` key) so `get_reader` / `open_reader` can dispatch by extension
- Provide a typed-metadata writer used directly (not via `get_writer`, which assumes the `write(ndarray)` contract)
- Bridge into the shared GRDL model (`DetectionSet`) rather than inventing a parallel one
