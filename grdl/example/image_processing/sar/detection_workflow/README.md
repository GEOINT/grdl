# SAR Detection Workflow — GRDL Ecosystem Integration Example

This example is a deliberate conglomeration of the GRDL ecosystem. It pulls from
**grdl** (the core library), **grdl-runtime** (the workflow execution framework),
and **grdl-te** (the testing and evaluation suite) to build a complete, end-to-end
SAR ship detection pipeline. It is meant to show how the three repositories fit
together in practice.

---

## What This Example Demonstrates

### 1. Using GRDL Library Modules

The example uses several `grdl` modules directly:

| Module | Role in this example |
|--------|---------------------|
| `grdl.IO.sar.SICDReader` | Open an Umbra SICD NITF file and extract metadata |
| `grdl.data_prep.ChipExtractor` | Extract a centered chip via the grdl-runtime `.chip()` builder call |
| `grdl.image_processing.sar.SublookDecomposition` | Split the complex SAR chip into N sub-aperture looks in the frequency domain |
| `grdl.image_processing.sar.CSIProcessor` | Produce a Coherent Shape Index RGB composite from the same chip |

### 2. Creating a Custom Processing Component

`DominanceDetection` is a custom class written in this example — it does not exist
in `grdl`. It implements **aperture dominance detection**: for each pixel it
computes the fraction of total sub-look power concentrated in a contiguous block
of looks. Pixels whose dominance ratio exceeds mean + N·σ are candidate targets.
Morphological opening and closing clean the binary mask, and `skimage.measure.label`
segments it into individual detections.

The class exposes an `apply()` method, which is the convention recognised by
grdl-runtime's `Workflow.step()` for callable steps. This lets a hand-rolled
class participate in a managed workflow without inheriting from any grdl base class.

```python
class DominanceDetection:
    def apply(self, looks: np.ndarray) -> tuple:
        ...  # returns (dominance_map, labeled_regions)
```

### 3. Building Workflows with grdl-runtime

The example builds two independent `Workflow` objects using the fluent builder API:

**Chip extraction** (framework mode — executes from a file path):

```python
chip_wf = (
    Workflow("Chip", version="1.0.0", modalities=["SAR"])
    .reader(SICDReader)
    .chip("center", size=10000)
)
chip_wf_result = chip_wf.execute(SICD_PATH)
```

**Detection pipeline** (array mode — executes from an in-memory array):

```python
det_wf = (
    Workflow("Detection", version="1.0.0", modalities=["SAR"])
    .step(SublookDecomposition, num_looks=7, dimension="azimuth")
    .step(DominanceDetection, smooth_win=7, dom_window=3,
          dom_sigma=3.0, morph_size=3)
)
det_result = det_wf.execute(chip, metadata=chip_metadata)
```

**CSI pipeline** (array mode):

```python
csi_wf = (
    Workflow("CSI", version="1.0.0", modalities=["SAR"])
    .step(CSIProcessor, dimension="azimuth", normalization="log")
)
csi_result = csi_wf.execute(chip, metadata=chip_metadata)
```

grdl-runtime handles metadata injection automatically: because
`SublookDecomposition.__init__` declares a `metadata` parameter, the framework
injects `chip_metadata` without any extra wiring. `DominanceDetection` has no
`metadata` parameter, so the framework passes only the array output from the
previous step.

### 4. Evaluating Workflows with grdl-te

When `benchmarking = True`, the example uses grdl-te's `ActiveBenchmarkRunner`
to time each workflow over multiple iterations with optional warmup:

```python
det_runner = ActiveBenchmarkRunner(
    det_wf,
    bench_source,
    iterations=bench_iterations,
    warmup=bench_warmup,
    tags={"workflow": "Detection"},
)
det_rec = det_runner.run(metadata=chip_metadata)
print_report([det_rec, csi_rec])
```

`BenchmarkSource` wraps the input data. It can be constructed from the real chip
(`BenchmarkSource.from_array(chip)`) or from a synthetic array of a named size
preset (`BenchmarkSource.synthetic("small", dtype=np.complex64)`), which lets you
run the performance evaluation without any real imagery on hand.

`print_report` prints an aggregated table of mean wall time, standard deviation,
and peak RSS per workflow.

---

## Files

```
detection_workflow/
├── README.md                 ← This file
├── config.yaml               ← All tunable parameters
└── csi_detect_workflow.py    ← The example script
```

---

## Prerequisites

```bash
conda activate grdl
```

The following packages must be importable in the `grdl` environment:

| Package | Purpose |
|---------|---------|
| `grdl` | Core library (IO, image processing, data prep) |
| `grdl-runtime` (`grdl_rt`) | Workflow execution framework |
| `grdl-te` (`grdl_te`) | Benchmarking infrastructure |
| `sarkit` or `sarpy` | SICD NITF reading backend used by `SICDReader` |
| `scikit-image` | Morphological operations and connected-component labeling |
| `scipy` | `uniform_filter` for power smoothing |
| `matplotlib` | Result visualization |
| `pyyaml` | Config file parsing |

---

## Configuration

Edit [`config.yaml`](config.yaml) to point at your data and tune the pipeline:

```yaml
input:
  sicd_path: /path/to/your/scene_SICD.nitf
  chip_size: 10000        # pixels on each side of the center chip

sublook:
  num_looks: 7            # sub-aperture count (detection); CSI always uses 3
  dimension: azimuth      # frequency-domain axis to split

detection:
  smooth_win: 7           # uniform filter window for power smoothing
  dom_window: 3           # contiguous look window for dominance ratio
  sigma: 3                # detection threshold (mean + N*sigma)
  morph_size: 3           # morphological opening/closing kernel size

benchmark:
  source: file            # "file" = use the real chip, "synthetic" = generate
  synthetic_size: small   # preset size when source = "synthetic"
  iterations: 3
  warmup: 1

display:
  vmax_percentile: 98     # amplitude saturation point for the detection overlay
```

---

## Running the Example

```bash
conda activate grdl
python csi_detect_workflow.py
```

Two flags near the top of `main()` control optional output:

```python
plotting     = True   # show matplotlib figures
benchmarking = False  # run ActiveBenchmarkRunner and print timing report
```

---

## Output

**When `plotting = True`** (default), three matplotlib figures appear:

1. **Aperture Dominance map** — per-pixel dominance ratio coloured on an
   `inferno` scale. Man-made targets with coherent scattering across sub-looks
   appear as bright spots; distributed clutter is uniformly dim.

2. **CSI composite** — three-channel RGB image where colour encodes sub-aperture
   coherence. Brightly coloured pixels are frequency-selective (corners, facets);
   grey pixels are distributed scatterers.

3. **Detection overlay** — SAR amplitude image in greyscale with red contours
   drawn around detected regions. The title reports the object count and
   detection threshold used.

**When `benchmarking = True`**, a summary table is printed:

```
Workflow: Detection
  Mean wall time : 1.234 s  (std: 0.012 s)
  Peak RSS       : 1.8 GB

Workflow: CSI
  Mean wall time : 0.843 s  (std: 0.008 s)
  Peak RSS       : 1.4 GB
```

---

## Ecosystem Dependency Map

```
grdl (SICDReader, SublookDecomposition, CSIProcessor, ChipExtractor)
  ↓
grdl-runtime (Workflow builder, metadata injection, deferred step construction)
  ↓
csi_detect_workflow.py
  ↑
grdl-te (ActiveBenchmarkRunner, BenchmarkSource, print_report)
```

`DominanceDetection` is defined locally in this example. It demonstrates that
any object with an `apply()` method can participate in a grdl-runtime workflow
without inheriting from a grdl base class or registering with the framework.
