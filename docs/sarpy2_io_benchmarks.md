# sarpy 2.0 Migration — IO Benchmark & Validation Summary

Benchmark and validation report for migrating GRDL's sarpy requirement from
`>=1.3.0` to `>=2.0.0` (issue #45). Covers compatibility validation, reader IO
throughput, geolocation-load cost, a memory-vs-chip-size sweep, a multi-file
sweep, and the latent reader-backend bugs found and fixed along the way.

All measurements use real Umbra and Capella imagery; harness and raw logs live in
`grdl_script/2026_06_01_sarpy2_validation/` (out of tree).

## 1. Environment & method

| | |
|---|---|
| sarpy (old) | 1.3.59 (conda-forge, the prior requirement) |
| sarpy (new) | 2.0.1 (PyPI — not yet on conda-forge) |
| sarkit | 1.4.1 | 
| numpy / python | 2.3.5 / 3.12.8 |
| platform | macOS arm64 |

The two sarpy versions cannot share an interpreter, so each benchmark runs in two
cloned conda envs and merges the results. GRDL's SAR readers prefer sarkit and
expose no backend flag, so the sarpy/sarkit backend is forced by toggling the
availability flags in `grdl.IO.sar._backend`. Chip reads are materialized
(`np.ascontiguousarray`) so lazy/mmap views actually transfer all bytes. Timings
are the median of N warm repeats; memory is reported via tracemalloc peak (Python
allocation) and a background-sampled peak RSS delta.

## 2. Compatibility (the actual #45 change)

The migration itself is light — sarpy 2.0 kept the legacy API GRDL uses:

- All 23 sarpy import paths GRDL uses resolve under 2.0.1.
- Every SICD/SIDD element-tree constructor and every projection/geometry function
  (`image_to_ground_geo`, `ground_to_image_geo`, `wgs_84_norm`, `ecf_to_geodetic`,
  `open_complex`, `open_phase_history`) is unchanged and not deprecated.
- numpy ≥ 1.19 is sufficient (env already on 2.3.5).
- sarpy 2.0 now depends on sarkit (which GRDL already prefers).

The only behavioral change GRDL is exposed to: sarpy 2.0 marks its *legacy SICD/SIDD
reader+writer* implementations `@deprecated("… use sarkit")`. GRDL's exposed
deprecated paths are the SICD/SIDD writers (tracked for a future sarkit migration).

**Real-data geolocation parity:** on Umbra and Capella SICD, the sarpy 2.0 projection
backend agrees with GRDL's native R/Rdot to **~0.00 m**.

**Test suite (sarpy 2.0.1):** `2372 passed, 59 skipped, 2 failed`. Both failures
pre-exist on sarpy 1.3.59 (a DEM-cache amortization test and the known-unresolved
StripmapPFA image-formation test) — **zero regressions from the migration**. One test
(`test_contrast_nrl::test_matches_sarpy`) is skip-guarded because sarpy 2.0.1 ships a
broken `visualization.remap` (`NameError: stats_calculation`, upstream — GRDL's own
`NRLStretch` is unaffected).

## 3. Reader IO throughput — sarpy 1.3.59 vs 2.0.1 vs sarkit

Median chip read of a centered 4096² chip (Umbra), open / chip / close:

**SICD** (1.6 GB, 14835×14570; 134 MB complex chip)

| backend | open | chip | throughput | chip alloc (tracemalloc) |
|---|--:|--:|--:|--:|
| sarpy 1.3.59 | 17.2 ms | 122.6 ms | 1.09 GB/s | 335 MB |
| sarkit 1.4.1 | 17.2 ms | 25.5 ms | 5.27 GB/s | 134 MB |
| sarpy 2.0.1 | 16.7 ms | **25.0 ms** | **5.38 GB/s** | 134 MB |

**SIDD** (498 MB, 22859×22859; 17 MB uint8 chip)

| backend | open | chip | throughput |
|---|--:|--:|--:|
| sarpy 1.3.59 | 9.0 ms | 4.13 ms | 4.06 GB/s |
| sarkit 1.4.1 | 22.1 ms | 92.5 ms | 0.18 GB/s |
| sarpy 2.0.1 | 8.4 ms | **3.87 ms** | **4.34 GB/s** |

**CPHD** (3.0 GB, 7888 vec × 51030 samp; 134 MB complex chip)

| backend | open | chip | throughput |
|---|--:|--:|--:|
| sarpy 1.3.59 | 493 ms | 128 ms | 1.05 GB/s |
| sarkit 1.4.1 | **12 ms** | 179 ms | 0.75 GB/s |
| sarpy 2.0.1 | 492 ms | **25.6 ms** | **5.23 GB/s** |

Notes: sarpy 2.0's reader is sarkit-backed, so chip reads are ~5× faster than 1.3.59
for SICD/CPHD. GRDL's sarkit SIDD reader decodes+caches the *whole* image on first
chip; its sarkit CPHD reader reads full-width signal then slices (1.67 GB) — both
inefficient for partial reads. sarpy's CPHD *open* is slow (~490 ms vs sarkit's 12 ms).

## 4. Geolocation-load (file → ready geolocation object)

`Geolocation.from_reader` is ~0.1 ms (it only references already-loaded metadata), so
cost is the reader's metadata parse. Total (open + build):

| backend | SICD | SIDD |
|---|--:|--:|
| native | 17.0 ms | — |
| sarkit 1.4.1 | 17.6 ms | 22.5 ms |
| sarpy 1.3.59 | 17.7 ms | 8.6 ms |
| sarpy 2.0.1 | 17.1 ms | 8.6 ms |

SICD geolocation setup is ~17 ms regardless of backend; SIDD metadata loads ~2.6×
faster via sarpy than sarkit (sarkit's SIDD NITF-open + XML parse is the slow part).
SIDD geolocation is always native.

## 5. Memory vs chip size (SICD, materialized center chip)

`tm/chip` = tracemalloc peak ÷ ideal chip bytes (allocation bloat).

| backend | tm/chip (all sizes) | 4096² time | 8192² time |
|---|--:|--:|--:|
| sarpy 1.3.59 | **2.50×** | 126 ms | 507 ms |
| sarkit 1.4.1 | 1.00× | 30 ms | 76 ms |
| sarpy 2.0.1 | **1.00×** | **18 ms** | **67 ms** |

**sarpy 1.3.59 allocates 2.5× the chip bytes at every size and is ~7× slower; sarpy
2.0.1 has zero allocation bloat and is the fastest backend.**

## 6. Multi-file sweep — chip read GB/s (4096²)

| format | file | sarpy 1.3.59 | sarkit 1.4.1 | sarpy 2.0.1 |
|---|---|--:|--:|--:|
| SICD | Umbra-04 | 1.05 | 5.03 | **5.33** |
| SICD | Capella | 1.37 | 1.21 | **3.93** |
| SIDD | Umbra-09 | 4.56 | 0.19 | **4.51** |
| SIDD | Umbra-08 | 5.15 | 0.38 | **5.13** |
| CPHD | Umbra-08 | 1.03 | 0.69 | **4.92** |
| CPHD | Capella | 1.20 | 0.88 | **2.73** |

sarpy 2.0.1 wins or ties on every file across both sensors and all three formats.

## 7. Latent reader-backend bugs found & fixed

Exercising the sarpy reader backend (never used in normal operation — sarkit is
preferred) surfaced **7 pre-existing GRDL bugs**, all failing identically on sarpy
1.3.59 and 2.0.1 (so none are migration regressions). Each was fixed by mirroring
SICD's working patterns (notably the `reader[r0:r1, c0:c1[, idx]]` __getitem__ form):

| # | file | bug |
|---|---|---|
| 1 | `IO/sar/sicd.py` | `_sarpy_latlon` received ndarray image corners (`getattr(ic,'FRFC')`) |
| 2 | `IO/sar/sidd.py` | `_extract_measurement_sarpy` called undefined `_sarpy_xyzpoly` / `_sarpy_poly1d` |
| 3 | `IO/sar/sidd.py` | `read_chip`/`read_full` passed `np.s_[...]` to sarpy `read_chip` |
| 4 | `IO/sar/cphd.py` | `_collection_info_sarpy` iterated dict-like `ParametersCollection` as a list |
| 5 | `IO/sar/cphd.py` | `_global_sarpy` called `.isoformat()` on a `numpy.datetime64` |
| 6 | `IO/sar/cphd.py` | CPHD `read_chip` used nonexistent `dim1_range=`/`dim2_range=` kwargs |
| 7 | `IO/sar/sidd.py` | SIDD `ReferencePoint` read `ECF`; sarpy field is `ECEF` (broke native SIDD geolocation via the sarpy reader) |

138 IO + geolocation tests pass after the fixes. These are pre-existing defects unrelated
to the sarpy version.

## 8. Conclusions

- Migrating to `sarpy>=2.0` is **safe** (zero GRDL test regressions, geolocation parity to
  ~0 m) and a **performance win**: chip reads are ~5–7× faster than sarpy 1.3.59 for
  SICD/CPHD and the 2.5× memory over-allocation of sarpy 1.3.x is eliminated.
- sarpy 2.0.1 is the fastest reader backend on every file tested, edging out sarkit.
- Operational note: sarpy 2.0 is PyPI-only for now, so `environment.yml` installs it from
  the pip section until conda-forge packages it.
- Follow-ups (separate from #45): migrate GRDL's SICD/SIDD *writers* off the
  sarpy-deprecated APIs to sarkit; consider making GRDL's sarkit SIDD/CPHD readers do
  true partial reads instead of decode-all / full-width.
