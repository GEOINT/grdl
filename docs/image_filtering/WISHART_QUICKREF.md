# Quick Reference: WishartFilter

## Import
```python
from grdl.image_processing.filters import WishartFilter
```

## Basic Usage
```python
# Auto-detect matrix type and ENL
filt = WishartFilter(kernel_size=7)
filtered = filt.apply(covmat_image)
```

## Common Configurations

### Fast Processing (Small Window)
```python
filt = WishartFilter(
    kernel_size=5,      # Smaller, faster
    enl=4.0,            # Known ENL
    sigma_range=2.5     # Good edge preservation
)
```

### Strong Smoothing (Large Window)
```python
filt = WishartFilter(
    kernel_size=11,     # Larger, more smoothing
    enl=0.0,            # Auto-estimate
    sigma_range=4.0     # More smoothing
)
```

### Edge-Preserving (Tight Threshold)
```python
filt = WishartFilter(
    kernel_size=7,
    enl=0.0,
    sigma_range=1.5,    # Very tight threshold
    min_weight_sum=0.2  # Conservative filtering
)
```

## Matrix Type Mapping

| Type  | Bands | Description        | Example Data           |
|-------|-------|--------------------|------------------------|
| C2    | 4     | Dual-pol           | Sentinel-1, RCM        |
| C3    | 9     | Compact quad-pol   | RADARSAT-2 compact     |
| C4    | 16    | Full quad-pol      | ALOS PALSAR-2, UAVSAR  |
| T3    | 9     | Coherency 3×3      | Decomposition output   |
| T4    | 16    | Coherency 4×4      | Full-pol coherency     |

## Band Format

### C2 (4 bands)
```
[C11, C12_real, C12_imag, C22]
```

### C3 (9 bands)
```
[C11, C12_r, C12_i, C13_r, C13_i, C22, C23_r, C23_i, C33]
```

### C4 (16 bands)
```
[C11, C12_r, C12_i, C13_r, C13_i, C14_r, C14_i,
 C22, C23_r, C23_i, C24_r, C24_i,
 C33, C34_r, C34_i,
 C44]
```

## Performance Tips

1. **Start small**: Test with `kernel_size=5` before trying larger windows
2. **Tile large images**: Process in chunks to manage memory
3. **Use known ENL**: If available from metadata, specify explicitly
4. **Consider downsampling**: For preview/testing, downsample first
5. **Profile first**: Time your workflow before optimizing

## Typical Processing Times

For 1000×1000 pixel C2 image (single-threaded CPU):
- `kernel_size=5`: ~30 seconds
- `kernel_size=7`: ~60 seconds  
- `kernel_size=9`: ~120 seconds

## Troubleshooting

### ValueError: Cannot auto-detect matrix dimension
**Cause**: Band count doesn't match C2/C3/C4 (4/9/16 bands)  
**Fix**: Specify `matrix_type` explicitly or check input data

### Output has NaN or Inf values
**Cause**: Singular or near-singular covariance matrices  
**Fix**: Check input data quality; filter adds regularization

### Excessive smoothing / blurred edges
**Cause**: `sigma_range` too high or `kernel_size` too large  
**Fix**: Reduce `sigma_range` to 1.5-2.5 or use smaller `kernel_size`

### Insufficient smoothing
**Cause**: `sigma_range` too low or `min_weight_sum` too high  
**Fix**: Increase `sigma_range` to 3.5-5.0 or lower `min_weight_sum`

## See Also
- Full documentation: `WISHART_README.md`
- Examples: `wishart_filter_example.py`
- Tests: `test_wishart.py`
