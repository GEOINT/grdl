# Sentinel-1 L0 to CRSD Conversion — Validation Report

This document describes the GRDL Sentinel-1 Level 0 to CRSD converter
and compares its output against three reference CRSD files. The
comparison records where GRDL output agrees with the references and
where it differs, and characterizes each difference.
It is a structural and metric comparison, not a certification of
equivalence: no downstream image-formation result has been compared
between the two products.

## 1. Conversion Overview

The GRDL converter (`grdl.IO.sar.sentinel1_l0.crsd_converter`) transforms
a Sentinel-1 IW-mode Level 0 SAFE product into a CRSD 1.0 file. The
pipeline:

1. Opens the SAFE product via `Sentinel1L0Reader`
2. Segments raw ISP packets into bursts by swath (IW1/IW2/IW3)
3. Loads precise orbit data (POEORB `.EOF` file) — auto-downloaded
   from [ASF](https://s1-orbits.asf.alaska.edu) or
   [ESA](https://step.esa.int/auxdata/orbits/Sentinel-1/POEORB/)
   if not found locally
4. For each burst: decodes FDBAQ-compressed I/Q, interpolates the
   satellite state vector, and computes reference geometry
5. Quantizes complex64 signal to CI2 (int8 I/Q) with per-vector
   amplitude scale factors (`AmpSF`)
6. Builds the CRSD XML metadata tree and writes the file via
   `sarkit.crsd.Writer`

Output is one CRSD file per polarization, with one channel per burst.

## 2. Test Datasets

Three Sentinel-1A IW-mode Level 0 products were converted and compared.
Datasets 1 and 3 are from the same orbit (062240) but different time
windows within that pass.

| # | Scene ID | Date | Orbit | Rel. Orbit |
|---|----------|------|-------|------------|
| 1 | `S1A_IW_RAW__0SDV_20251209T151925_…_1F07` | 2025-12-09 | 062240 | 043 |
| 2 | `S1A_IW_RAW__0SDV_20251108T152805_…_9365` | 2025-11-08 | 061788 | 116 |
| 3 | `S1A_IW_RAW__0SDV_20251209T151950_…_959C` | 2025-12-09 | 062240 | 043 |

A reference CRSD file was provided for each dataset. The provenance
of these reference files is not documented here and is not relied
upon; they serve only as a point of comparison.

## 3. Summary Comparison

| Metric | Dataset 1 | Dataset 2 | Dataset 3 |
|--------|-----------|-----------|-----------|
| **Channels** | 35 / 35 | 34 / 34 | 34 / 34 |
| **File size (GRDL / Reference)** | 2.328 / 2.326 GB | 2.251 / 2.249 GB | 2.251 / 2.249 GB |
| **File size delta** | 2.0 MB | 1.6 MB | 1.6 MB |
| **RcvPos diff (mean)** | 0.1 m | 0.0 m | 0.046 m |
| **Orbit radius** | 7069.9 km ✓ | 7069.7 km ✓ | 7069.7 km ✓ |
| **PVP vectors (GRDL / Reference)** | +9 per burst | +9 per burst | +9 per burst |
| **IW1 range samples** | Match | Match | Match |
| **IW2 range samples** | Match | Match | Match |
| **IW3 range samples** | 19784 vs 19970 | 19784 vs 19970 | 19784 vs 19970 |
| **RcvStart offset** | 0.82 s | 0.01 s | 0.82 s |
| **Power ratio (GRDL / Reference)** | 1.54× | 1.72× | 1.07–1.45× |

### 3a. Per-Swath Detail — Dataset 3

| Swath | Channels | Vectors (GRDL / Reference) | Range Samples | Status |
|-------|----------|---------------------|---------------|--------|
| IW1 | 11 / 11 | 1409 / 1400 | 23868 / 23868 | ✓ Match |
| IW2 | 11 / 11 | 1548 / 1540 | 24140 / 24140 | ✓ Match |
| IW3 | 12 / 12 | 1410 / 1400 | 19784 / 19970 | △ 186 samples |

The per-swath pattern is identical across all three datasets.

### 3b. PVP Detail — Dataset 3 (First 3 Channels)

| Channel | RcvPos Δ (mean) | RcvPos Δ (max) | Orbit Radius | RcvStart Δ | Power Ratio |
|---------|----------------|----------------|--------------|------------|-------------|
| 1 (IW1) | 0.046 m | 0.047 m | 7069.7 km | 0.819 s | 1.453 |
| 2 (IW1) | 0.072 m | 0.072 m | 7069.7 km | 0.819 s | 1.139 |
| 3 (IW1) | 0.048 m | 0.048 m | 7069.6 km | 0.819 s | 1.066 |

## 4. Known Differences — Explained

### 4a. Channel Identifiers (Cosmetic)

GRDL and the reference files use different burst-counter schemes in
their channel IDs:

- **GRDL**: `043_532449967_IW1` — uses the raw downlink **Space Packet
  Count** from the ISP packet headers. This is a monotonic satellite
  lifecycle counter that never resets.
- **Reference**: `043_090475_IW1` — uses an **orbit-relative burst
  index** that resets each orbit.

Both encodings produce the same relative ordering (orbit number +
monotonic counter + swath). The CRSD spec places no constraint on
the channel identifier format beyond uniqueness. Any downstream
processor matches channels by swath and ordering, not by the
numeric portion of the ID.

**Assessed impact:** the CRSD spec constrains channel identifiers
only by uniqueness, so both encodings are conformant. A consumer
that matches channels by swath and ordering is unaffected; one
that parses the numeric portion, or that joins against the reference
channel IDs, will not match.

### 4b. Extra PVP Vectors (+9 Per Burst)

GRDL consistently includes ~9 more pulse vectors per burst than the
reference files (e.g. 1409 vs 1400 for IW1, 1548 vs 1540 for IW2,
1410 vs 1400 for IW3).

**Cause:** GRDL includes all decoded ISP packets belonging to the
burst segment, including calibration/noise packets at the burst
edges. The reference files trim these edge packets to match the
nominal burst length specified in the Sentinel-1 instrument timing.

**Assessed impact:** the extra vectors carry PVP timing and
positions consistent with the rest of the burst, and add < 1% to
the burst length, accounting for the file size difference
(~1.6–2.0 MB). Image formation processors that handle
variable-length bursts should tolerate them. Whether the trailing
calibration/noise packets the reference files trim are benign as
*echo* data was not established — a processor that assumes every
vector in a channel is a science echo may fold them into the aperture.

### 4c. IW3 Range Sample Count (~186 Samples)

IW1 and IW2 range sample counts match exactly. IW3 differs by ~186
samples (GRDL: 19784, reference: 19970; or GRDL: 19756, reference:
19942).

**Cause:** GRDL computes the sample count directly from the L0
packet field **Number of Quads** (`n_samples = mode_quads × 2`),
which reflects the actual number of digitised I/Q pairs in the
downlinked echo. The reference files appear to use a slightly wider
range window — likely derived from the SWST (Sampling Window
Start Time) and SWEC (Sampling Window Echo Count) timing fields,
zero-padding to a fixed window length.

At the IW3 range sampling rate of 64.345 MHz, 186 samples ≈ 2.89 μs
of additional range extent (approximately 434 m in slant range).
This falls within the guard interval at the edge of the sampling
window where no target echo energy is expected.

**Assessed impact:** the 186 extra reference samples fall outside
the echo extent defined by the PVP timing parameters (FRCV1, FRCV2,
SC0), in the guard interval where no target echo energy is
expected. A processor that derives its range window from those PVP
fields should be unaffected. This has not been confirmed against
the reference samples themselves — their contents were not inspected,
and a processor that instead trusts the declared sample count would
see a shorter range window from GRDL output.

### 4d. RcvStart Timing Offset

Datasets 1 and 3 (orbit 062240) show a consistent 0.819 s offset
between GRDL and the reference RcvStart values. Dataset 2 (orbit
061788) shows only a 0.011 s offset.

**Cause:** The CRSD `RcvStart` field is relative to the collection
reference epoch. GRDL and the reference file derive slightly
different epochs from the L0 timing metadata. The 0.82 s offset for orbit
062240 is consistent across all 34–35 channels in both datasets
from that orbit, confirming it is a constant epoch offset rather
than a per-pulse timing error.

The key evidence:
- The offset is **identical** for every channel within each dataset
- Intra-burst pulse spacing (PRI) matches exactly
- Orbit positions at corresponding pulses agree to < 0.1 m

**Assessed impact:** the offset is constant across all channels in
a file, so computations using **relative** timing (pulse-to-pulse
intervals, range delay differences, Doppler estimation) are
unaffected — the constant cancels. Absolute timing does *not*
cancel: geolocation, multi-file coherent processing, and any
comparison against an external time reference will be offset by
0.82 s unless the consumer re-derives the epoch from the orbit
state vectors and GPS coarse/fine time fields. Which epoch
convention is correct per the CRSD spec was not determined; the
0.82 s discrepancy is unexplained, not shown to be benign.

### 4e. Signal Power Ratio (1.1–1.7×)

GRDL signal power is consistently higher than the reference files by
a factor of 1.1–1.7×, varying by channel and dataset.

**Cause:** Both converters quantize the FDBAQ-decoded complex64
signal to CI2 (int8 I/Q pairs), but they use different amplitude
scaling strategies:

- **GRDL** scales each pulse so the peak |I| or |Q| maps to ±127,
  recording the scale factor in the PVP `AmpSF` field. This
  maximizes dynamic range per pulse but preserves the raw FDBAQ
  amplitude envelope.
- **Reference files** apply a different normalization — likely a
  per-burst or per-channel scale rather than per-pulse — resulting
  in lower mean CI2 amplitudes.

The `AmpSF` field exists precisely to recover the true amplitude:
`true_value = ci2_value × AmpSF`. Any CRSD consumer that applies
the scale factor will reconstruct the correct signal power regardless
of the quantization scheme.

The ratio varies across channels because it depends on the scene
content (clutter level, target presence) and how the FDBAQ
decompressor distributes dynamic range across the echo.

**Assessed impact:** CI2 is a lossy format and the `AmpSF` PVP
field carries the per-pulse scale needed to recover the
pre-quantization amplitude. A consumer that applies `AmpSF` should
recover the true signal power from either product, to within CI2
quantization error. The two quantization strategies differ in how
much of the int8 range they use, so their quantization error is not
identical; no radiometric comparison of reconstructed amplitudes
was performed.

## 5. Validation Summary

| Check | Result |
|-------|--------|
| Channel count matches | ✓ All 3 datasets |
| Swath structure (IW1/IW2/IW3 ordering) | ✓ Identical |
| Orbit position agreement (< 0.1 m) | ✓ All channels |
| Orbit radius (LEO sanity check) | ✓ 7069.6–7069.9 km |
| PVP field names and dtypes | ✓ Identical (16 fields) |
| IW1/IW2 range sample count | ✓ Exact match |
| sarkit round-trip (write → read) | ✓ 67 unit tests pass |
| Sarkit consistency (default full-file checks) | ✗ Fails on generated files |
| Sarkit split-gate consistency (non-schema checks) | ✓ Passes after converter/metadata fixes |
| Sarkit split-gate schema validation | ✗ Fails (cause not established) |

The GRDL converter produces CRSD files that parse via
`sarkit.crsd`, carry channel and swath structure matching the
reference files, and agree with them on orbit positions to < 0.1 m.
Sarkit schema validation currently fails (§6), so these files are
**not** validated as interchangeable with the reference files, and
no such claim is made here. The five differences
documented in §4 are characterized individually; several rest on
assumptions about downstream processor behavior that have not been
tested against an independent image-formation implementation.

## 6. Sarkit CRSD Consistency Findings

The following findings summarize `sarkit.verification.CrsdConsistency`
run against the three generated VV files in
`/data/sar/raw_crsd/sentinel-1/test` (full-file mode).

### 6a. Current status

**Generated files do not pass `CrsdConsistency` in its default
full-file mode.** That is the operative result.

To isolate which checks fail, the suite was split and run in two
parts. This is a diagnostic decomposition, not an alternative pass
criterion — a file that fails the default gate has not been
validated, regardless of how the subsets score:

- Non-schema consistency checks (with `check_against_schema`
  excluded): pass on regenerated test output
  (`consistency_fail_count=0`).
- XML schema validation, run separately: fails, on
  metadata-versus-schema constraint mismatches.

So the failure is localized to the schema layer rather than to
geometry, timing, or packing. Whether the mismatch originates in
the metadata GRDL emits, in the schema revision being validated
against, or in the verifier has not been determined, and no
conclusion should be drawn about which until it is. Aligning the
metadata with the schema remains open work.

## 7. Comparison Tool

The CLI wrapper that used to front this comparison lived in the
examples directory, removed in `1ffe87c`.  Compare two CRSD files
through the reader instead — `CRSDReader` carries the metadata the
sections above are written against:

```python
from grdl.IO.sar import CRSDReader

with CRSDReader('grdl_output.crsd') as a, CRSDReader('reference.crsd') as b:
    print(a.metadata.extras['num_channels'],
          b.metadata.extras['num_channels'])
    print(a.get_shape(), b.get_shape())
    print(a.metadata.reference_geometry, b.metadata.reference_geometry)
```

## 8. Scope and Attribution

This work is independent format-conversion research, intended for
personal, research, and experimental use. It is not an official
product, carries no NGA authority, and is not offered as an
authoritative CRSD product. The developers have not established
that GRDL output conforms to the CRSD standard: schema validation
currently fails (§6) and the cause is not established.

The scope of what was tested is stated in §2–§5: three Sentinel-1A
IW-mode L0 products, structural and metric comparison only, with
the §6 consistency findings run against the VV files. No downstream
image-formation or radiometric comparison was made. The reference
files were not themselves independently verified, so they serve as
the comparison baseline, not a normative one.

CRSD (version 1.0) is a standard of the U.S. National
Geospatial-Intelligence Agency, registered in the NSG Standards
Registry (nsgreg.nga.mil). GRDL is not affiliated with or endorsed
by NGA. The CRSD files described here are produced by GRDL, and
they do not currently pass CRSD schema validation (§6). GRDL emits
the `api.nsgreg.nga.mil` schema namespace for format identification
only; this implies no NGA registration, review, or endorsement.
