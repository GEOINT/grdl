# Synthetic Data Generation Capabilities

This document summarizes the current synthetic data generation capabilities available within the `grdx` repository workspace, primarily focusing on creating mock datasets for testing, discovery workflows, and data augmentation.

## 1. `grdl.discovery.synthesizer.DataSynthesizer`

The primary utility for generating synthetic imagery from scratch is the `DataSynthesizer` class located in `grdl/grdl/discovery/synthesizer.py`. It produces geocoded GeoTIFF files with known affine transforms, allowing testing of pipelines with controllable spatial patterns and geolocation metadata.

It supports the following types of imagery:

### Panchromatic / Electro-Optical (EO)

The `synthesize_eo` method creates EO images with customizable ground sample distance (GSD), number of bands, and spatial patterns. 

Supported patterns: `'checkerboard'`, `'gradient'`, `'noise'`, `'resolution_target'` (Siemens star).

**Example:**
```python
from pathlib import Path
from grdl.discovery.synthesizer import DataSynthesizer

synth = DataSynthesizer()

# Generate a 512x512 EO chip with a resolution target pattern
chip_path = synth.synthesize_eo(
    output_path=Path("mock_eo_chip_resolution.tif"),
    rows=512,
    cols=512,
    bands=1,  # 1 band for Panchromatic
    center_lat=35.0,
    center_lon=-118.0,
    gsd=0.5,  # 0.5 meter GSD
    pattern="resolution_target"
)
```

### Synthetic Aperture Radar (SAR)

The `synthesize_sar` method generates a synthetic complex Gaussian SAR image. It injects a few simulated point targets at known pixel fractions with configurable amplitudes.

**Example:**
```python
from pathlib import Path
from grdl.discovery.synthesizer import DataSynthesizer

synth = DataSynthesizer()

# Generate a 1024x1024 SAR chip with complex noise and point targets
chip_path = synth.synthesize_sar(
    output_path=Path("mock_sar_chip.tif"),
    rows=1024,
    cols=1024,
    center_lat=35.0,
    center_lon=-118.0,
    pixel_spacing=1.0,
    noise_sigma=0.1
)
```

### Multispectral

The `synthesize_multispectral` method generates a multispectral cube where each band gets a slightly different gradient phase mixed with noise.

**Example:**
```python
from pathlib import Path
from grdl.discovery.synthesizer import DataSynthesizer

synth = DataSynthesizer()

# Generate a 13-band multispectral chip
chip_path = synth.synthesize_multispectral(
    output_path=Path("mock_multispectral_chip.tif"),
    rows=512,
    cols=512,
    bands=13,
    center_lat=35.0,
    center_lon=-118.0,
    gsd=10.0
)
```

---

## 2. Image Augmentation & Manipulation

In addition to generating base images from scratch, there are a few utilities capable of augmenting existing mock images.

### Noise Generation (`grdl-imagej`)

The `NoiseGenerator` class located in `grdl-imagej/grdl_imagej/noise/noise_generator.py` can add various types of synthetic noise to images. This is useful for data augmentation or testing denoising algorithms on cleanly generated synthetic chips.

Supported noise models: `'gaussian'`, `'poisson'` (shot noise), `'salt_pepper'`, `'speckle'`.

**Example:**
```python
from grdl_imagej.noise.noise_generator import NoiseGenerator
import numpy as np

# Assuming `clean_chip` is a loaded numpy array of a generated image
clean_chip = np.ones((512, 512)) * 128.0 

# Add Gaussian noise
ng_gaussian = NoiseGenerator(noise_type='gaussian', sigma=25.0, seed=42)
noisy_gaussian_chip = ng_gaussian.apply(clean_chip)

# Add SAR Speckle noise
ng_speckle = NoiseGenerator(noise_type='speckle', sigma=10.0)
noisy_speckle_chip = ng_speckle.apply(clean_chip)
```

### Compact Polarimetry Synthesis

The module `grdl.IO.sar.compact_pol_synthesis` provides functionalities for synthesizing compact polarimetric SAR data from fully polarimetric data (e.g., converting quad-pol to compact pol). While not a from-scratch generator, it allows for creating specialized polarimetric test cases from existing SAR images.
