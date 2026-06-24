# GRDL Jupyter Notebooks

**Interactive examples and tutorials for the GRDL library.**

## What's Here

These notebooks demonstrate GRDL workflows across all domains:

- **`image_processing/`** — SAR processing, decomposition, detection, image formation
- **`ortho/`** — Orthorectification workflows (SICD, SIDD, BIOMASS)
- **`IO/`** — Reading SAR, EO, MSI, and HDF5 formats
- **`geolocation/`** — Coordinate transforms, DEM integration
- **`interpolation/`** — Bandwidth-preserving resampling kernels
- **`shapes/`** — Geographic ROI analysis, error ellipses
- **`catalog/`** — Remote query and download (Sentinel-1/2, NISAR, BIOMASS)

## Getting Started

### 1. Install GRDL + Notebook Dependencies

```bash
# From the GRDL repository root
pip install -e ".[all]"
pip install -r notebooks/requirements.txt
```

### 2. Launch Jupyter Lab

```bash
cd notebooks/
jupyter lab
```

### 3. Configure Data Paths

Each notebook has a **configuration cell** (usually Cell 3) where you set file paths:

```python
SICD_PATH = Path("/path/to/your/sicd_file.nitf")  # ← CHANGE THIS
DEM_PATH = Path("/path/to/dem/directory")         # ← CHANGE THIS
```

Update these paths to point to your local data before running cells.

## Notebook Structure

All notebooks follow a consistent pattern:

1. **Preamble cell** — Environment validation, autoreload magic
2. **Imports cell** — GRDL modules and dependencies
3. **Configuration cell** — User-editable paths and parameters
4. **Processing cells** — Step-by-step workflow with explanations
5. **Visualization cell** — Interactive napari viewer (where applicable)

## Learning Path

**New to GRDL?** Start here:

1. [IO/view_sicd.ipynb](IO/view_sicd.ipynb) — Open and inspect SAR imagery
2. [image_processing/sublook_decomposition.ipynb](image_processing/sublook_decomposition.ipynb) — Sub-aperture splitting
3. [ortho/sicd_ortho_demo.ipynb](ortho/sicd_ortho_demo.ipynb) — Geometric correction
4. [image_processing/csi_detection_overlay.ipynb](image_processing/csi_detection_overlay.ipynb) — Full detection pipeline

**Advanced workflows:**

- [image_processing/cphd_to_sicd.ipynb](image_processing/cphd_to_sicd.ipynb) — Phase history → image formation
- [ortho/ortho_combined.ipynb](ortho/ortho_combined.ipynb) — Multi-format orthorectification
- [catalog/discover_and_download.ipynb](catalog/discover_and_download.ipynb) — Remote data acquisition

## Requirements

- **Python:** 3.11+
- **GRDL:** Latest version (`pip install -e ".[all]"`)
- **napari:** Interactive visualization (`pip install "napari[all]"`)
- **ipykernel:** Jupyter kernel support

See [requirements.txt](requirements.txt) for full dependencies.

## Data Requirements

Notebooks require **real SAR/EO imagery** to run. Example data sources:

- **SICD/SIDD:** [UMBRA Open Data](https://umbra.space/open-data)
- **CPHD:** Contact your SAR data provider
- **Sentinel-1:** [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
- **Sentinel-2:** [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
- **NISAR:** [NASA Earthdata](https://www.earthdata.nasa.gov/)
- **DEM:** [FABDEM](https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn) or [DTED](https://www.nga.mil/ProductsServices/TopographicalTerritorial/Pages/DTED.aspx)

## Understanding Notebook Outputs

When you execute notebooks, two types of outputs are generated:

### 1. **Embedded Outputs** (stored INSIDE the `.ipynb` file)

When you run cells with matplotlib plots or napari screenshots, Jupyter **embeds the images as base64-encoded PNG data inside the notebook JSON**. This bloats the `.ipynb` file from ~10 KB to several MB.

**Example:** A cell with `plt.imshow()` stores the plot image inside the notebook itself.

**Solution:** Clear these before committing:
```bash
jupyter nbconvert --clear-output --inplace notebooks/**/*.ipynb
```

This strips embedded images/outputs while keeping your code and markdown.

### 2. **Files Written to Disk** (created by GRDL workflows)

Some notebooks demonstrate **writing processed data to files**.

| File Type | Created By | Example |
|-----------|-----------|---------|
| `.png`, `.jpg` | `matplotlib.savefig()`, napari screenshots | Saving plots for reports |
| `.tif`, `.tiff` | `GeoTIFFWriter`, orthorectification outputs | Orthorectified imagery |
| `.nitf` | `SICDWriter`, `SIDDWriter` | Formed SAR images |
| `.h5`, `.hdf5` | Intermediate SAR processing results | CPHD data, cached sublooks |
| `.npy`, `.npz` | `NumpyWriter`, cached arrays | Detection masks, features |

**These are gitignored** via `.gitignore` patterns to prevent accidentally committing large data files:
```gitignore
**/notebooks/**/*.png
**/notebooks/**/*.tif
**/notebooks/**/*.h5
# ... etc
```


### Best Practice: Clear Outputs Before Committing

```bash
# Strip embedded outputs (keeps code, removes images/results)
jupyter nbconvert --clear-output --inplace notebooks/**/*.ipynb

# Then commit
git add notebooks/
git commit -m "Update notebooks (outputs cleared)"
```

**This keeps your `.ipynb` files small (~10-20 KB each) in the repository.**

## Memory Management

**SAR processing is memory-intensive.** A 5000×5000 complex chip = 400 MB. Sub-aperture decomposition (7 looks) = 2.8 GB.

### Best Practices

1. **Delete intermediate arrays:** `del chip; import gc; gc.collect()`
2. **Monitor memory usage:** Check `array.nbytes / 1e6` (MB)
3. **Process in chunks:** Use smaller `CHIP_SIZE` or tile-based processing

**See [MEMORY_MANAGEMENT.md](MEMORY_MANAGEMENT.md) for detailed guidance.**

### Common Issues

**Kernel crashes or freezes:**
- Reduce `CHIP_SIZE` from 5000 to 2048 (87% less RAM)
- Delete arrays after processing: `del looks; gc.collect()`
- Restart kernel: `Kernel → Restart & Clear Output`

**High memory usage (>8 GB):**
- Check what's in memory: Run `%whos` in a cell
- Delete unused variables before visualization
- Use `.astype(np.float32)` instead of `float64` (50% less RAM)

**Expected RAM usage:**
- Small chip (2048×2048): ~500 MB - 1 GB
- Medium chip (5000×5000): 2 GB - 4 GB  
- Large chip (8192×8192): 8 GB - 16 GB

## Troubleshooting

**"Module not found" errors:**
```bash
pip install -e ".[all]"  # Install all GRDL dependencies
```

**napari doesn't open:**
```bash
pip install "napari[all]">=0.4.19
```

**"MemoryError" or kernel crash:**
- See Memory Management section above
- Reduce `CHIP_SIZE` in configuration cells
- Delete intermediate arrays: `del large_array; gc.collect()`
- Restart kernel if memory doesn't free

## License

MIT License — same as GRDL library.  
See [LICENSE](../LICENSE) for full text.

## Contributing

Found a bug in a notebook? Have a new example workflow?  
Open an issue or pull request at [github.com/geoint-org/GRDL](https://github.com/geoint-org/GRDL).
