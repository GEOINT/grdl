# Sentinel-1 Level 0 Reader

`grdl.IO.sar.Sentinel1L0Reader` reads raw Sentinel-1 L0 SAFE
products — unfocused ISP packet streams with FDBAQ compression.
This page covers:

1. [Product file layout](#1-product-file-layout)
2. [Auxiliary files you need](#2-auxiliary-files-you-need)
3. [Installing the optional decoder](#3-installing-the-optional-decoder)
4. [Reading a product — quick start](#4-reading-a-product--quick-start)
5. [Working with bursts](#5-working-with-bursts)
6. [Orbit and geometry](#6-orbit-and-geometry)
7. [Reader configuration](#7-reader-configuration)
8. [Limitations and gotchas](#8-limitations-and-gotchas)

---

## 1. Product file layout

A Sentinel-1 L0 SAFE directory looks like this **once fully
unpacked** from the `.zip` ESA ships:

```
S1A_IW_RAW__0SDV_<start>_<stop>_<orbit>_<datatake>_<id>.SAFE/
├── manifest.safe                                       [REQUIRED]
├── s1a-iw-raw-s-vv-<times>-<ids>.dat                   [REQUIRED]
├── s1a-iw-raw-s-vv-<times>-<ids>-index.dat             [recommended]
├── s1a-iw-raw-s-vv-<times>-<ids>-annot.dat             [recommended]
├── s1a-iw-raw-s-vh-<times>-<ids>.dat                   [per pol]
├── s1a-iw-raw-s-vh-<times>-<ids>-index.dat
├── s1a-iw-raw-s-vh-<times>-<ids>-annot.dat
├── S1A_OPER_AUX_POEORB_OPOD_<...>.EOF                  [OPTIONAL]
├── support/
│   ├── s1-level-0.xsd
│   ├── s1-level-0-annot.xsd
│   └── s1-level-0-index.xsd
└── <product-name>.SAFE-report-<timestamp>.pdf          [QA report]
```

| File                                       | Purpose                                                                 | Reader role                                                                 |
| ------------------------------------------ | ----------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `manifest.safe`                            | SAFE XFDU manifest (XML). Mission, mode, orbit, polarizations, footprint, acquisition period. | **Required.** `Sentinel1L0Reader._load_metadata()` parses it up front.      |
| `*.dat` (one per pol)                      | Raw ISP packet stream — the actual radar payload with FDBAQ-encoded I/Q. | **Required** for any packet access. One per transmit-receive polarization.  |
| `*-index.dat`                              | Burst-index file (binary, 36-byte records per [L0FMT] Table 3-9).       | Used for fast burst seeking when the decoder is unavailable.                |
| `*-annot.dat`                              | Per-packet annotation (binary, 26-byte records per [L0FMT] Table 3-8).  | Parsed on demand via `binary_parser.parse_packet_annotation_file()`.        |
| `*.EOF` (co-located)                       | ESA Precise Orbit Ephemerides (POE) — orbit state vectors at 10 s spacing, position accuracy < 5 cm. | Auto-detected if present in the SAFE dir. Populates `metadata.orbit_state_vectors`. |
| `support/*.xsd`                            | Schemas for the three file formats.                                     | Not read by the reader; kept for validation tools.                          |
| `*.SAFE-report-*.pdf`                      | Human-readable acquisition quality report.                              | Not read by the reader.                                                     |

**There is no `annotation/` subdirectory in L0 products.** The
`/annotation/*.xml` files you find in L1 (SLC, GRD) products do
not exist at L0. All metadata the reader exposes comes from
`manifest.safe` and from decoded ISP packet headers — plus the
external POE file for precise orbit state.

---

## 2. Auxiliary files you need

Three classes of file matter:

### 2a. Required to open the product

- `manifest.safe`
- At least one `*.dat` measurement file

Without these the reader raises `InvalidSAFEProductError`.

### 2b. Required to decode I/Q samples

- The optional `sentinel1decoder` package (see [§3](#3-installing-the-optional-decoder))
- The `*.dat` file of interest

Without the package, the reader still opens and exposes
manifest-derived metadata, but every `read_burst()` /
`read_swath()` / `read_chip()` call raises `DependencyError`.

### 2c. Recommended for precise orbit / geolocation

- An ESA **POE** file (`S1[ABCD]_OPER_AUX_POEORB_OPOD_*.EOF`)
  whose validity period covers your acquisition. Download from
  <https://step.esa.int/auxdata/orbits/Sentinel-1/POEORB/>.

The reader looks for POE files in this order:

1. An explicit directory passed via
   `ReaderConfig(poe_directory=...)`
2. The SAFE directory itself (e.g. the one ESA often ships
   alongside `manifest.safe`)
3. A sibling `POEORB/` directory next to the SAFE

So the easy way to organise a working tree is:

```
<your-data-root>/
├── POEORB/
│   ├── S1A_OPER_AUX_POEORB_OPOD_20251122T070400_V20251101T225942_20251103T005942.EOF
│   ├── S1A_OPER_AUX_POEORB_OPOD_20251123T070409_V20251102T225942_20251104T005942.EOF
│   └── ...
├── S1A_IW_RAW__0SDV_20251103T151953_..._35FC.SAFE/
└── S1A_IW_RAW__0SDV_20251204T165155_..._9476.SAFE/
```

The reader will pick the POE file whose validity window covers
the acquisition's start time.

### 2d. About attitude data

**There is no public Sentinel-1 auxiliary product that carries
attitude.** The S1 Copernicus Auxiliary Data Service distributes
only orbit (`POEORB`, `RESORB`), instrument (`AUX_INS`),
calibration (`AUX_CAL`), and processing-parameter (`AUX_PP1`,
`AUX_PP2`) files. S1 POE `.EOF` files contain **position and
velocity only** — not quaternions or Euler angles.

ESA embeds attitude data in only two places:

1. The `<attitudeList>` element of the **annotation XML inside
   any L1 product** (SLC, GRD). The IPF computes these records
   from onboard star-tracker and gyro telemetry during
   focusing.
2. Internal ESA Quality Control feeds that are not publicly
   distributed.

For an L0 workflow you therefore have two practical options:

- **Use the yaw-steering fallback.** `GeometryCalculator`
  computes ACF vectors from orbit + nominal TOPS steering when
  attitude is absent. For IW this is usually accurate enough
  because the commanded steering dominates the attitude
  motion.
- **Borrow attitude from a co-temporal L1 SLC product.** If you
  have (or can download) a Sentinel-1 L1 product with the same
  orbit + datatake IDs as your L0 (matching
  `<absolute_orbit>_<datatake>` in the filename), parse its
  annotation XML with
  `grdl.IO.sar.sentinel1_l0.annotation_parser.parse_annotation_file()`
  and assign the parsed `attitude_records` onto
  `reader.metadata.attitude_records` before building a
  `GeometryCalculator`.

---

## 3. Installing the optional decoder

Packet decoding is done by an external package,
`sentinel1decoder`, which ships with `pandas`, `numpy`, and the
FDBAQ decompression tables. It is exposed as an optional GRDL
extra:

```bash
pip install "grdl[s1_l0]"
```

or with the full dependency set:

```bash
pip install "grdl[all]"
```

The reader is safe to import without this package installed —
only the actual packet-decode calls error out with a
`grdl.exceptions.DependencyError`.

---

## 4. Reading a product — quick start

### 4a. Factory function (tersest)

```python
from grdl.IO.sar import open_safe_product

with open_safe_product("path/to/S1A_IW_RAW__0SDV_....SAFE") as r:
    print(r.metadata.summary())
    first_burst = r.bursts[0]
    iq = r.read_burst(first_burst)   # numpy complex64, (pulses, samples)
```

### 4b. Explicit class + context manager

```python
from grdl.IO.sar import Sentinel1L0Reader

with Sentinel1L0Reader("path/to/product.SAFE") as r:
    # r.metadata is a Sentinel1L0Metadata (inherits ImageMetadata)
    meta = r.metadata
    print(meta.product_id, meta.mission, meta.mode)
    print("Polarizations:", meta.polarizations)
    print("Shape (rows, cols):", r.get_shape())
    print("dtype:", r.get_dtype())
```

### 4c. Auto-detection via `open_sar()`

```python
from grdl.IO.sar import open_sar

reader = open_sar("path/to/product.SAFE")
# Returns Sentinel1L0Reader because the product ID contains "RAW".
```

---

## 5. Working with bursts

A Sentinel-1 IW L0 product contains interleaved bursts across
three sub-swaths (`IW1`, `IW2`, `IW3`), each acquired with a
beam-steering cycle. The reader surfaces:

```python
with Sentinel1L0Reader(path) as r:
    # All bursts for the currently selected polarization.
    all_bursts = r.bursts

    # Or query explicitly:
    vh_bursts = r.get_burst_info(polarization="VH")

    # Group by swath:
    swath_info = r.get_swath_info(polarization="VH")
    for swath_number, info in swath_info.items():
        print(f"swath {swath_number}: {info.num_bursts} bursts")

    # Decode a specific burst.  ``burst`` is either an index
    # into the returned list or a BurstInfo instance.
    burst = vh_bursts[10]
    iq = r.read_burst(burst, polarization="VH")
    print(iq.shape, iq.dtype)   # e.g. (1548, 24110) complex64

    # Iterate over every burst + decoded data, optionally
    # filtered by polarization and swath.
    for info, iq in r.iter_bursts(polarization="VV", swath=11):
        process(iq)

    # Parallel decode for a specific (swath, burst) pair:
    iq = r.read_burst_parallel(swath=11, burst=17,
                               polarization="VV",
                               num_workers=8)

    # Per-packet SWST array for fine timing (one float per pulse):
    swst = r.get_burst_swst_array(swath=11, burst_index=17,
                                  polarization="VV")
```

### 5a. `read_chip` semantics

The `ImageReader` ABC requires a `read_chip(row_start, row_end,
col_start, col_end)` method. For L0, the reader maps this onto
**the currently selected burst**:

```python
with Sentinel1L0Reader(path) as r:
    r.set_current_burst(burst_index=17, polarization="VV")
    # Reads pulses 0..500, range samples 0..4000 of burst 17.
    chip = r.read_chip(0, 500, 0, 4000)
```

For burst-scoped work, prefer `read_burst()` directly — it is
clearer and avoids hidden state.

---

## 6. Orbit and geometry

Once a POE file is loaded (automatic if co-located), the
metadata carries full orbit state vectors:

```python
with Sentinel1L0Reader(path) as r:
    # 10-second-spaced OSVs from the POE file.
    osvs = r.orbit_state_vectors
    print(f"Loaded {len(osvs)} orbit state vectors")

    # Build interpolators for position/velocity at arbitrary times.
    geom = r.get_geometry_calculator()
    if geom is not None:
        import numpy as np
        times = np.array([0.0, 1.0, 2.0])  # seconds from start_time
        positions, velocities = (
            geom.interpolate_position_velocity(times)
        )
        print(positions.shape)   # (3, 3) — (N, [X Y Z])

    # Timing calculator for relative-seconds math and pulse trains.
    timing = r.get_timing_calculator()
    if timing is not None:
        print("PRF:", timing.prf_hz, "Hz")
```

If the reader can't find a POE file, `orbit_state_vectors` will
be empty and `get_geometry_calculator()` returns `None`. Either
provide `ReaderConfig(poe_directory=...)` or disable POE loading.

---

## 7. Reader configuration

```python
from grdl.IO.sar import Sentinel1L0Reader, S1L0ReaderConfig
from pathlib import Path

cfg = S1L0ReaderConfig(
    validate_safe=True,                             # default
    parse_annotations=True,                         # no-op for L0
    load_poe=True,                                  # default
    poe_directory=Path("~/SAR_DATA/Level0/POEORB").expanduser(),
    burst_gap_threshold_us=1_000_000,               # 1 s
    burst_line_filter_ratio=0.9,                    # drop edge bursts
)

with Sentinel1L0Reader(path, config=cfg) as r:
    ...
```

All `ReaderConfig` fields are optional; defaults match typical
ESA IW products. Set `load_poe=False` to skip the POE search
entirely if you know you don't need precise orbit data.

---

## 8. Limitations and gotchas

- **No annotation XML.** L0 products don't have an
  `annotation/` directory. Radar parameters come from the
  decoded packet headers (per-swath, per-pulse), so
  `metadata.radar_parameters` is filled from mode-level
  defaults first and refined from packet timing when the
  decoder is available. If you opened a product without the
  optional decoder, `radar_parameters` will reflect nominal IW
  values only.

- **Attitude is normally empty.** POE files don't carry
  attitude. The ACF path that uses measured attitude will fall
  back to yaw-steering geometry.

- **Raw swath numbering vs. logical names.** Packet headers
  carry raw swath numbers (10, 11, 12 for IW1, IW2, IW3). The
  `BurstInfo.swath` field holds the raw number; use
  `constants.raw_swath_to_name()` / `raw_swath_to_logical()` to
  translate.

- **`sentinel1decoder` returns interleaved I+Q samples.** The
  shape returned by `read_burst()` is
  `(num_pulses, 2 × num_quads)` because the decoder emits the
  in-phase and quadrature components side-by-side. For
  magnitude/phase work, treat them as a single complex array
  of shape `(num_pulses, num_samples_complex)`.

- **Large memory footprint.** A full IW swath can be multi-GB
  once decoded. Use `iter_bursts()` for streaming work, or
  `read_burst()` on single bursts.

- **Not all polarizations are addressable by name when keys
  collide.** The reader maps each measurement file to a key of
  the form `"VV"` / `"VH"` / `"HH"` / `"HV"`. If two files
  share a polarization for a product (shouldn't happen in
  stock ESA products), the later file wins.

---

## Quick reference

| Task                                  | Call                                            |
| ------------------------------------- | ----------------------------------------------- |
| Open a product                        | `Sentinel1L0Reader(path)`                       |
| Get metadata                          | `reader.metadata` (a `Sentinel1L0Metadata`)     |
| All bursts (current polarization)     | `reader.bursts`                                 |
| All bursts (specific polarization)    | `reader.get_burst_info(polarization="VV")`      |
| Per-swath burst grouping              | `reader.get_swath_info(polarization="VV")`      |
| Decode one burst                      | `reader.read_burst(info_or_index)`              |
| Decode one burst in parallel          | `reader.read_burst_parallel(swath, burst)`      |
| Decode a whole swath                  | `reader.read_swath(swath)`                      |
| Iterate bursts and I/Q                | `reader.iter_bursts(polarization=..., swath=...)` |
| Per-packet SWST                       | `reader.get_burst_swst_array(swath, burst)`     |
| Interpolated positions/velocities     | `reader.get_geometry_calculator()`              |
| Time-relative math (`t_ref`)          | `reader.get_timing_calculator()`                |
| Close / release resources             | `reader.close()` or use `with ... as`           |
