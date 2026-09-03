# Complex Wishart Filter for Polarimetric SAR Data

## Overview

The **WishartFilter** is a multi-dimensional speckle filter specifically designed for polarimetric Synthetic Aperture Radar (SAR) covariance and coherency matrix images. Unlike traditional scalar filters that operate on single-channel intensity images, the Wishart filter operates on the full covariance matrix at each pixel, preserving polarimetric information while reducing speckle noise.

## Theory

### Complex Wishart Distribution

For polarimetric SAR data, each pixel contains a covariance matrix **C** (or coherency matrix **T**) that follows a scaled complex Wishart distribution:

```
C ~ W_c(L, Σ)
```

where:
- **L** is the Equivalent Number of Looks (ENL)
- **Σ** is the expected covariance matrix

The complex Wishart distribution is the multivariate generalization of the gamma distribution for Hermitian positive-definite matrices.

### Wishart Distance

The filter uses the Wishart distance to measure statistical similarity between covariance matrices:

```
d(C₁, C₂) = ln(det(C₁)) + ln(det(C₂)) + trace(C₁⁻¹C₂ + C₂⁻¹C₁) - 2p
```

where **p** is the matrix dimension (2, 3, or 4). Lower distances indicate more similar matrices.

### Adaptive Filtering

The filter performs weighted averaging of neighboring covariance matrices:

```
Ĉ(x) = Σ w(x,y) · C(y) / Σ w(x,y)
```

Weights are computed using exponential decay based on Wishart distance:

```
w(x,y) = exp(-d(C(x), C(y)) / σ)    if d < threshold
       = 0                           otherwise
```

The adaptive threshold preserves edges while smoothing homogeneous regions.

## Supported Matrix Formats

### C2 (Dual-Pol)
- **Dimensions**: 2×2 covariance matrix
- **Bands**: 4 [C11, C12_real, C12_imag, C22]
- **Example**: RCM compact-pol, Sentinel-1 dual-pol

### C3 (Compact-Pol)
- **Dimensions**: 3×3 covariance matrix
- **Bands**: 9 (diagonal + upper triangle)
- **Format**: [C11, C12_r, C12_i, C13_r, C13_i, C22, C23_r, C23_i, C33]
- **Example**: RADARSAT-2 compact-pol

### C4 (Full Quad-Pol)
- **Dimensions**: 4×4 covariance matrix
- **Bands**: 16 (full Hermitian matrix)
- **Example**: ALOS PALSAR-2, Radarsat-2 full-pol

### T3 and T4 (Coherency Matrices)
- Similar structure to C3 and C4
- Represent data in Pauli basis rather than linear basis

## Usage

### Basic Example

```python
from grdl.image_processing.filters import WishartFilter
import numpy as np

# Load C2 covariance matrix image (4 bands)
covmat = np.load('rcm_c2_covmat.npy')  # Shape: (4, rows, cols)

# Create filter with default parameters
wishart_filter = WishartFilter(
    kernel_size=7,      # 7×7 smoothing window
    enl=0.0,            # Auto-estimate ENL from data
    matrix_type='auto'  # Auto-detect from band count
)

# Apply filter
filtered = wishart_filter.apply(covmat)
```

### Advanced Usage

```python
# Explicit matrix type and custom parameters
wishart_filter = WishartFilter(
    kernel_size=9,           # Larger window for more smoothing
    enl=4.0,                 # Explicit ENL (e.g., from metadata)
    matrix_type='C3',        # Explicit matrix type
    sigma_range=2.5,         # Tighter edge preservation
    min_weight_sum=0.2       # Stricter weight threshold
)

# Apply to C3 data
filtered_c3 = wishart_filter.apply(covmat_c3)
```

### Processing Pipeline

```python
from grdl.IO.geotiff import GeoTIFFReader, GeoTIFFWriter
from grdl.image_processing.filters import WishartFilter

# Read input covariance matrix
with GeoTIFFReader('input_c2.tif') as reader:
    covmat = reader.read_full()
    metadata = reader.metadata

# Apply Wishart filter
filt = WishartFilter(kernel_size=7, enl=0.0)
filtered = filt.apply(covmat)

# Write output
with GeoTIFFWriter('output_c2_filtered.tif', metadata=metadata) as writer:
    writer.write(filtered)
```

## Parameters

### kernel_size
- **Type**: int
- **Range**: Odd integers, 3 ≤ kernel_size ≤ 31
- **Default**: 7
- **Description**: Square window size for neighborhood analysis. Larger values provide more speckle reduction but may blur edges and increase computation time.
- **Recommendations**:
  - Small (3-5): Fast, minimal smoothing, good edge preservation
  - Medium (7-9): Balanced smoothing and speed (recommended)
  - Large (11+): Strong smoothing, slower, may blur edges

### enl
- **Type**: float
- **Range**: ≥ 0.0
- **Default**: 0.0 (auto-estimate)
- **Description**: Equivalent Number of Looks. Controls the noise threshold. Set to 0.0 for automatic estimation from image statistics.
- **Recommendations**:
  - 0.0: Auto-estimate (recommended for unknown data)
  - 1-2: Single-look complex (SLC) data
  - 4-10: Multi-looked data
  - Check SAR product metadata for nominal ENL value

### matrix_type
- **Type**: str
- **Options**: 'auto', 'C2', 'C3', 'C4', 'T3', 'T4'
- **Default**: 'auto'
- **Description**: Covariance/coherency matrix type. Use 'auto' to detect from band count, or specify explicitly for validation.
- **Band count mapping**:
  - C2: 4 bands
  - C3/T3: 9 bands
  - C4/T4: 16 bands

### sigma_range
- **Type**: float
- **Range**: 0.5 ≤ sigma_range ≤ 10.0
- **Default**: 3.0
- **Description**: Adaptive threshold for similarity weighting in standard deviations. Controls edge preservation.
- **Recommendations**:
  - Low (1.0-2.0): Strong edge preservation, less smoothing
  - Medium (2.5-4.0): Balanced (recommended)
  - High (5.0+): More smoothing, softer edges

### min_weight_sum
- **Type**: float
- **Range**: 0.0 ≤ min_weight_sum ≤ 1.0
- **Default**: 0.1
- **Description**: Minimum weight sum threshold. Pixels with insufficient similar neighbors retain original values.
- **Recommendations**:
  - 0.0: Always filter (not recommended)
  - 0.1-0.2: Standard (recommended)
  - 0.5+: Conservative, less smoothing in heterogeneous areas

## Performance Considerations

### Computational Complexity
- **Time**: O(N × K² × D²), where:
  - N = number of pixels
  - K = kernel_size
  - D = matrix dimension (2, 3, or 4)
- **Space**: O(N × B), where B = number of bands

### Processing Time Estimates (approximate)
For a 1000×1000 pixel C2 image on a modern CPU:
- kernel_size=5: ~30 seconds
- kernel_size=7: ~60 seconds
- kernel_size=9: ~120 seconds

**Recommendations**:
1. Use smaller kernel sizes (5-7) for initial testing
2. Process in tiles for large images
3. Consider downsampling for preview/testing
4. GPU acceleration (future enhancement)

## Validation

The filter preserves important matrix properties:
- **Hermitian structure**: C† = C (conjugate transpose equals original)
- **Positive semi-definite**: All eigenvalues ≥ 0
- **Physical validity**: Diagonal elements (intensities) remain non-negative

## Examples

See the following example scripts:
- `grdl/example/image_processing/sar/wishart_filter_example.py` - Comprehensive usage examples
- `grdl/image_processing/filters/test_wishart.py` - Test suite with synthetic data

## References

1. Lee, J.S., Grunes, M.R., Ainsworth, T.L., Du, L.J., Schuler, D.L., and Cloude, S.R. (1999). 
   "Unsupervised classification using polarimetric decomposition and the complex Wishart classifier." 
   *IEEE Transactions on Geoscience and Remote Sensing*, 37(5), 2249-2258.

2. Vasile, G., Trouvé, E., Lee, J.S., and Buzuloiu, V. (2006). 
   "Intensity-driven adaptive-neighborhood technique for polarimetric and interferometric SAR parameters estimation." 
   *IEEE Transactions on Geoscience and Remote Sensing*, 44(6), 1609-1621.

3. Conradsen, K., Nielsen, A.A., Skriver, H., and Schou, J. (2003). 
   "A test statistic in the complex Wishart distribution and its application to change detection in polarimetric SAR data." 
   *IEEE Transactions on Geoscience and Remote Sensing*, 41(1), 4-19.

4. Lopez-Martinez, C. and Fabregas, X. (2003). 
   "Polarimetric SAR speckle noise model." 
   *IEEE Transactions on Geoscience and Remote Sensing*, 41(10), 2232-2242.

## Author

Jason Fritz, PhD  
43161141+stryder-vtx@users.noreply.github.com

## License

MIT License  
Copyright (c) 2024 geoint.org

See LICENSE file for full text.

## Version History

- **1.0.0** (2026-08-25): Initial implementation
  - Support for C2, C3, C4, T3, T4 matrices
  - Auto-detection of matrix type
  - Adaptive edge-preserving filtering
  - Wishart distance-based weighting
