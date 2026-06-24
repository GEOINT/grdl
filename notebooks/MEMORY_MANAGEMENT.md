# Jupyter Notebook Memory Management — Best Practices for GRDL

## Current Issues (Identified in Audit)

### **Issue 1: Intermediate Arrays Never Deleted**
- **Only 2 of 31 notebooks** use `del` statements
- Large SAR arrays (400 MB - 3 GB) accumulate in memory
- Users hit "kernel crashed" without understanding why

### **Issue 2: No Memory Usage Reporting**
- Users have no feedback on RAM consumption
- No warnings before hitting system limits

### **Issue 3: Duplicate Display Data**
- Converting `chip` → `chip_display` doesn't free `chip`
- Both arrays sit in memory simultaneously

---

## Memory Management Patterns

### **Pattern 1: Delete After Processing**

**Bad (accumulates RAM):**
```python
chip = reader.read_chip(...)           # 400 MB
looks = sublook.decompose(chip)        # 2.8 GB
dominance = compute_dominance(looks)   # 200 MB
csi_rgb = csi_proc.apply(chip)         # 600 MB
# Total: 4 GB in memory!
```

**Good (minimal RAM):**
```python
chip = reader.read_chip(...)           # 400 MB
looks = sublook.decompose(chip)        # 2.8 GB
del chip  # ← Free 400 MB

dominance = compute_dominance(looks)   # 200 MB
del looks  # ← Free 2.8 GB (if not needed for CSI)

# OR: Keep looks if needed for CSI
csi_rgb = csi_proc.apply(chip)
# Total: 800 MB (vs 4 GB)
```

### **Pattern 2: Memory Usage Reporting**

Add after each major allocation:

```python
import gc

chip = reader.read_chip(...)
print(f"Memory: {chip.nbytes / 1e6:.1f} MB")

looks = sublook.decompose(chip)
print(f"Memory: {looks.nbytes / 1e6:.1f} MB")
print(f"Total allocated: {(chip.nbytes + looks.nbytes) / 1e9:.2f} GB")

# After deletion
del chip
gc.collect()  # Force garbage collection
print(f"After cleanup: {looks.nbytes / 1e9:.2f} GB")
```

### **Pattern 3: Check What's in Memory**

Add a diagnostic cell:

```python
# Optional: Check current memory usage
import sys

def get_size_mb(var):
    """Get variable size in MB."""
    return sys.getsizeof(var) / 1e6

print("Current variables in memory:")
for name in dir():
    if not name.startswith('_') and name not in ['In', 'Out', 'get_ipython']:
        obj = eval(name)
        if hasattr(obj, 'nbytes'):  # NumPy arrays
            print(f"  {name:20s}: {obj.nbytes / 1e6:8.1f} MB")
```

### **Pattern 4: Process in Chunks**

**Bad (loads entire image):**
```python
full_image = reader.read_full()  # 50 GB!
result = process(full_image)
```

**Good (processes chips):**
```python
from grdl.data_prep import Tiler

tiler = Tiler(rows, cols, tile_size=2048, overlap=256)
results = []

for region in tiler.tile_positions():
    chip = reader.read_chip(region.row_start, region.row_end,
                            region.col_start, region.col_end)
    result = process(chip)
    results.append(result)
    del chip  # Free immediately after processing
```

---

## Checklist for All Notebooks

### **Before Each Cell:**
- [ ] Will this create a large array? (>100 MB)
- [ ] Do I still need previous large arrays?
- [ ] Should I delete something before proceeding?

### **After Processing:**
- [ ] Delete intermediate complex arrays
- [ ] Delete magnitude arrays after display conversion
- [ ] Delete sublook stacks after feature extraction
- [ ] Keep only what napari/plotting needs

### **Common Deletions:**

| Variable | When to Delete | Saves |
|----------|----------------|-------|
| `complex_chip` | After `magnitude = np.abs(chip)` | 50% (real vs complex) |
| `chip` | After sublook decomposition | 100% of chip size |
| `looks` | After dominance/CSI extraction | 700% of chip size (for 7 looks) |
| `cube` (multi-pol) | After channel extraction | 400% of single-pol (for 4 pols) |
| `magnitude` | After display conversion or ortho | 100% of magnitude size |
| `dominance` | After detection mask creation | 100% of feature size |

---

## Specific Notebook Fixes

### **1. CSI Detection Overlay**

**Add after Cell 5 (read chip):**
```python
print(f"✓ Chip loaded: {chip.nbytes / 1e6:.1f} MB")
```

**Replace Cell 7 with:**
```python
sublook = SublookDecomposition(meta, num_looks=NUM_LOOKS, dimension=DIMENSION)
looks = sublook.decompose(chip)
print(f"Sublook stack: {looks.shape}, dtype: {looks.dtype}")
print(f"Memory: {looks.nbytes / 1e6:.1f} MB")

# Free original chip (no longer needed)
del chip
import gc; gc.collect()
print(f"✓ Freed chip, {looks.nbytes / 1e6:.1f} MB remaining")
```

**Add after Cell 9 (dominance):**
```python
# Note: Keep looks for CSI processing later
print(f"✓ Dominance computed: {dominance.nbytes / 1e6:.1f} MB")
```

**Add after Cell 11 (detection mask):**
```python
# Free dominance (no longer needed)
del dominance
gc.collect()
print(f"✓ Freed dominance array")
```

**Replace Cell 18 (CSI) with:**
```python
print("Computing CSI composite...")
csi_proc = CSIProcessor(meta, dimension=DIMENSION, normalization=CSI_NORM)
csi_rgb = csi_proc.apply_sublook_stack(looks)  # Pass pre-computed sublooks

print(f"CSI RGB shape: {csi_rgb.shape}, dtype: {csi_rgb.dtype}")
print(f"CSI range: [{csi_rgb.min()}, {csi_rgb.max()}]")

# Free sublooks (no longer needed)
del looks
gc.collect()
print(f"✓ Freed sublook stack, {csi_rgb.nbytes / 1e6:.1f} MB remaining")
```

**Final memory footprint:**
- Before: `chip` + `looks` + `dominance` + `csi_rgb` + `labeled` = ~4.2 GB
- After: `csi_rgb` + `labeled` = ~800 MB
- **Savings: 81% reduction**

---

### **2. Sublook Decomposition**

**Add after Cell 7 (sublooks):**
```python
# Free original chip (no longer needed for display)
del chip
import gc; gc.collect()
print(f"✓ Freed chip array")
```

**Replace Cell 9 (display prep) with:**
```python
to_db = ToDecibels()
stretch = PercentileStretch(plow=PLOW, phigh=PHIGH)

# Process and immediately assign (overwrites looks)
looks_display = stretch.apply(to_db.apply(looks))
print(f"Sublook display: {looks_display.shape}, range [{looks_display.min():.3f}, {looks_display.max():.3f}]")

# Note: Original complex 'looks' no longer needed
print(f"Memory: {looks_display.nbytes / 1e6:.1f} MB")
```

---

### **3. Pauli Decomposition**

**Add after Cell 6 (extract pols):**
```python
# Free full cube (no longer needed)
del cube
import gc; gc.collect()
print(f"✓ Freed multi-pol cube")
```

**Add after Cell 8 (RGB composite):**
```python
# Free individual polarization channels
del shh, shv, svh, svv, components
gc.collect()
print(f"✓ Freed polarization channels and components")
print(f"RGB composite: {rgb_float.nbytes / 1e6:.1f} MB")
```

---

### **4. Ortho Notebooks**

**Already good!** `sicd_ortho_demo` includes:
```python
magnitude = np.abs(complex_chip).astype(np.float32)
del complex_chip  # free memory 
```

**But add after orthorectification:**
```python
# Free original magnitude array
del magnitude
gc.collect()
print(f"✓ Freed slant-range magnitude, ortho: {ortho_magnitude.nbytes / 1e6:.1f} MB")
```

---

## When to Restart Kernel

Even with proper deletions, Python's garbage collector may not free memory immediately.

**Signs you need to restart:**
- Kernel using >8 GB RAM after deletions
- Slow cell execution (thrashing)
- "MemoryError" or kernel crashes

**Solution:**
```
Kernel → Restart & Clear Output
```

Then re-run cells from the top.

---

## Advanced: Memory Profiling

For debugging memory issues:

```python
# Install: pip install memory_profiler
%load_ext memory_profiler

def my_function():
    chip = reader.read_chip(...)
    result = process(chip)
    return result

%memit my_function()
# Output: peak memory: 2048.5 MiB, increment: 1500.2 MiB
```

---

## Summary: 3 Simple Rules

1. **Delete after each major processing step**  
   `del array; import gc; gc.collect()`

2. **Report memory usage after allocations**  
   `print(f"Memory: {arr.nbytes / 1e6:.1f} MB")`

3. **Keep only what napari needs**  
   Final visualization layers + labeled regions

**Target:** <1 GB RAM for most notebooks (down from 4-10 GB currently)
