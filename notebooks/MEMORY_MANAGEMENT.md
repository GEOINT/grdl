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


### **Pattern 2: Process in Chunks**

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

### **Pattern 3: napari Blocking + Automatic Cleanup**

**For interactive napari visualizations**, use `napari.run()` to block execution until the user closes the viewer, then automatically clean up memory.

**Benefits:**
- Works with "Run All" — pauses at visualization, continues after window closes
- Automatic memory cleanup — no manual intervention needed
- No accidental early cleanup — only happens after viewer is closed
- Clean workflow — one action (closing window) triggers cleanup

**Standard napari pattern:**
```python
import napari

# Close any existing viewer to avoid conflicts
try:
    viewer.close()
except (NameError, AttributeError):
    pass

viewer = napari.Viewer(title="Image Analysis")

# Add all layers
viewer.add_image(rgb_composite, name="RGB Composite", rgb=True)
viewer.add_image(component_1, name="Component 1", visible=False)
viewer.add_image(component_2, name="Component 2", visible=False)

print(f"napari viewer opened with {len(viewer.layers)} layers")
print("\nExecution paused — close the napari window to continue and free memory...")

# Block until viewer is closed
napari.run()

# Automatic cleanup after viewer closes
print("\n✓ Viewer closed — cleaning up memory...")
del rgb_composite, component_1, component_2, viewer
import gc
gc.collect()
print("✓ Memory cleanup complete")
```

**Execution flow:**
1. Cell creates viewer and displays layers
2. Execution **pauses** at `napari.run()` — user can explore data
3. User closes napari window (click X)
4. Execution **resumes** automatically
5. Cleanup code runs immediately, freeing memory
6. Next cell (or "Run All" continuation) proceeds

**What NOT to do:**
```python
# BAD: No blocking — viewer opens but cell completes immediately
viewer = napari.Viewer()
viewer.add_image(data)
# Execution continues, data still in memory, viewer may conflict with next run

# BAD: Manual cleanup cell — user forgets to run it
del data  # Separate cleanup cell that may never execute
```

**Memory footprint example (Pauli decomposition):**
- Before viewer cell: `components` (100 MB) + `rgb_float` (25 MB) + `rgb_uint8` (12 MB) = 137 MB
- During viewing: Same 137 MB (viewer displays, doesn't copy)
- After closing: 0 MB (cleanup runs automatically)

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


