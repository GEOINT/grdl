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

## 11. Writer Factory Registry

**File:** `__init__.py`

Writers are registered in a module-level dict for lazy loading.

```python
_WRITER_REGISTRY = {
    'geotiff': ('grdl.IO.geotiff', 'GeoTIFFWriter'),
    'numpy':   ('grdl.IO.numpy_io', 'NumpyWriter'),
}

def get_writer(format: str, filepath, metadata=None) -> ImageWriter:
    if format not in _WRITER_REGISTRY:
        raise ValueError(f"Unknown format '{format}'. Supported: {list(_WRITER_REGISTRY)}")
    module_path, class_name = _WRITER_REGISTRY[format]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(filepath, metadata)
```

**Rules:**
- Writers are not imported until requested (lazy loading)
- Extension auto-detection via `_EXTENSION_MAP` (case-insensitive)
- Convenience `write()` function handles lifecycle (open, write, close)
