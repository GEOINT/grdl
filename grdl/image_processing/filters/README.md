# Spatial Filters

All filters expose `apply(source, **kwargs)`. Use independent image filtering
for scalar bands and explicit joint filtering for polarimetric matrices.
Import concrete filters from `grdl.image_processing.filters`.

## Array Conventions

| Representation | Shape | Layout |
|---|---|---|
| Single image | `(rows, cols)` | `YX` |
| Independent bands or SLC channels | `(channels, rows, cols)` | `CYX` |
| Formed polarimetric matrix | `(N, N, rows, cols)` | `CCYX` |
| Packed Hermitian matrix | `(N*N, rows, cols)` real | `CYX` |

Direct `apply()` calls use canonical layouts. They do not accept YXC cubes as
channel-first arrays. A labelled external YXC cube can be normalized at the
`execute(metadata, source)` boundary. The shared polarimetric adapter returns
canonical CYX/CCYX output; generic bandwise transform execution restores the
external YXC layout when applicable. Convert to channel-last only for an
external consumer that requires it, for example `cube.transpose(1, 2, 0)`.

GeoTIFF, GDAL fallback, and HDF5 readers label full multi-band datasets CYX;
single selected bands are returned as 2D arrays. HDF5 assumes stored CYX for
3D datasets. Reader metadata describes the full dataset, so adjust it when
reading a subset of bands or a spatial chip. These readers' `get_shape()`
methods still report legacy `(rows, cols, bands)` tuples for multi-band data;
use the returned array's `.shape` for its actual layout.

## Independent Images

```python
import numpy as np
from grdl.image_processing.filters import MeanFilter, ComplexLeeFilter

rng = np.random.default_rng(42)
image = rng.random((15, 17)).astype(np.float32)
cube = rng.random((3, 15, 17)).astype(np.float32)

mean = MeanFilter(kernel_size=3)
smoothed = mean.apply(image)
bands = mean.apply(cube, kernel_size=5)  # Independent bands; instance stays at 3.

slc = cube.astype(np.complex64) * np.exp(1j * cube)
phase_preserved = ComplexLeeFilter(kernel_size=3, enl=2.0).apply(slc)
```

**Bandwise filtering does not preserve inter-channel covariance.** In particular,
applying a scalar complex filter to each polarization is not equivalent to
joint polarimetric matrix filtering. Do not send a 4D tensor to a scalar filter.

## Joint Polarimetric Filtering

Wishart, Refined Lee, and IDAN inherit the same interface from
`PolarimetricFilter`:

```text
apply(source, *, input_kind=None, matrix_type=None, channel_metadata=None, **kwargs)
```

| `input_kind` | Input | Result |
|---|---|---|
| `matrix` | Formed square CCYX tensor, N = 2, 3, or 4 | Same shape and dtype |
| `channels` | Complex CYX SLC cube, explicit C2/T2/C3/T3 basis | Formed and filtered CCYX matrix |
| `packed` | Real CYX matrix-element storage, 4/9/16 bands | Filtered real packed CYX cube |
| `image` | IDAN only: YX image or CYX independent bands | Same shape and dtype |

Four-dimensional input defaults to `matrix`. IDAN defaults to independent
`image` filtering for other ranks, including complex 3D cubes. Wishart also
recognizes real 3D packed input without `input_kind`; prefer explicit arguments
in new code. Refined Lee and Wishart require explicit channel selection in
standalone `apply()` calls.

### Form a Matrix From Channels

```python
import numpy as np
from grdl.image_processing.filters import WishartFilter, RefinedLeeFilter, IDANFilter

rng = np.random.default_rng(7)
channels = (
    rng.normal(size=(4, 9, 11)) + 1j * rng.normal(size=(4, 9, 11))
).astype(np.complex64)  # HH, HV, VH, VV

wishart = WishartFilter(kernel_size=3)
t3 = wishart.apply(channels, input_kind='channels', matrix_type='T3')
c3 = RefinedLeeFilter(kernel_size=3, nlook=1.0).apply(
    channels, input_kind='channels', matrix_type='C3',
)
joint = IDANFilter(kernel_size=3, max_pixels=8, enl=2.0).apply(
    channels, input_kind='channels', matrix_type='T3',
)
```

Formation uses the existing `CovarianceMatrix` or `CoherencyMatrix` with
`window_size=1`, once before filtering. The builders currently return complex64.
Dual-pol C2/T2 needs two channels in co/cross order. Reciprocal C3/T3 accepts
three channels `[HH, HV, VV]` or four `[HH, HV, VH, VV]`; four-channel formation
averages HV and VH, i.e., the monostatic assumption. Optional `ChannelMetadata` 
names/roles resolve channel order for other supported orthogonal polarization 
labels. Layout and polarization order are different concepts: descriptors do not
make a YXC array CYX.

### Filter a Preformed Matrix

```python
from grdl.image_processing.decomposition.pol_matrix import CoherencyMatrix
from grdl.image_processing.filters import RefinedLeeFilter

t3 = CoherencyMatrix(window_size=1).compute(channels)
filtering = RefinedLeeFilter(kernel_size=3, nlook=1.0)
filtered = filtering.apply(t3, input_kind='matrix', matrix_type='T3', nlook=2.0)
```

All three joint filters accept formed C2/T2/C3/T3/C4/T4 matrices. `matrix_type`
checks the dimension; it does not convert C to T, verify the physical basis,
or certify positive definiteness/Hermiticity. No C4/T4 construction from raw
SLC channels is provided. If a matrix basis is unknown, it may be left
unspecified for formed input; never infer C versus T from shape or dtype.

### Packed Storage

`pack_matrix()` and `unpack_matrix()` in `decomposition.pol_matrix` own storage
conversion. Each matrix row contributes its diagonal followed by real/imaginary
pairs for upper-triangle elements. For C2/T2 the order is
`[M11, Re(M12), Im(M12), M22]`. Lower-triangle elements are reconstructed by
conjugation, and diagonal imaginary parts are not stored. Packed bands are
matrix elements, not raw scattering channels. Filter kernels operate on CCYX
matrices, not packed bands.

## Metadata-Aware Execution

```python
from grdl.IO.models.base import ImageMetadata

metadata = ImageMetadata(
    format='array', rows=9, cols=11, bands=4, dtype='complex64', axis_order='CYX',
)
result, updated = wishart.execute(
    metadata, channels, input_kind='channels', matrix_type='T3',
)
assert updated.axis_order == 'CCYX'
assert updated.bands == 9
```

The shared adapter carries channel descriptors to `apply()`. Matrix results
receive rows, cols, dtype, N*N bands, and flattened row-major element descriptors.
Existing matrix descriptors are retained when their count matches. Wishart and
Refined Lee infer `channels` for complex 3D execution input; IDAN requires it
explicitly to distinguish joint processing from independent bands.

Keep representation options local to the relevant pipeline step. A channel-to-
matrix step followed by a matrix filter cannot be described by forwarding
`input_kind='channels'` unchanged to every step.

## Filter Support

GPU means the class advertises CuPy compatibility, not that every dtype or
boundary combination has been validated. Joint filters are currently CPU-only.

| Filter | Input domain | Output dtype | GPU | Declared halo/global behavior |
|---|---|---|---|---|
| Mean | Real images | float32, or float64 for float64 input | Yes | Global/undeclared |
| Gaussian | Real images | float32, or float64 for float64 input | Yes | Global/undeclared |
| Median / Min / Max | Real images | Input dtype | Yes | Global/undeclared |
| StdDev | Real images | float64 | Yes | Global/undeclared |
| Lee | Real amplitude/intensity | float64 | Yes | Half kernel with fixed ENL; global with auto ENL |
| ComplexLee | Complex images | complex128 | Yes | Half kernel with fixed ENL; global with auto ENL |
| EnhancedLee | Real or complex images | float64 / complex128 | Yes | Half kernel with fixed ENL; global with auto ENL |
| LeeSigma | Real or complex images | float64 / complex128 | No | Half kernel with fixed ENL; global with auto ENL |
| PhaseGradient | Complex images | float64 | Yes | Global/undeclared |
| RefinedLee | Joint matrix/channels/packed | Prepared matrix dtype; packed real dtype | No | Half kernel |
| IDAN | Independent images or joint matrix/channels/packed | Input/prepared matrix dtype | No | Global |
| Wishart | Joint matrix/channels/packed | Prepared matrix dtype; packed real dtype | No | Half kernel; no ENL estimate |

Do not pass complex arrays to the real-image general filters: complex support
has not been defined, and some existing kernels cast to real rather than reject
them. Dtype preservation is algorithm-specific, not guaranteed by the bandwise
mixin. Joint input validation rejects non-finite numeric data; do not assume all
scalar filters currently share that policy.

## Parameters and Tiling

Use constructor defaults for reuse and keyword overrides on `apply()` for
individual calls. Joint filters validate construction and invocation through
the same Annotated specifications, with finite numeric values, odd kernels,
and additional algorithm constraints. Refined Lee exposes strictly positive
`nlook`, not an independent `enl` control. IDAN exposes `enl`, `max_pixels`, and
`similarity_threshold`. Wishart exposes `sigma_range` and `min_weight_sum`;
its deprecated `enl` argument is informational and does not change weighting.
Representation descriptors are separate from discovered algorithm tunables.

`tile_halo` is an instance-level declaration. If call-time settings differ
from constructor settings, the caller must plan tiling with those same effective
settings; the property does not receive runtime overrides. IDAN is always
global. Scalar auto-ENL estimation is scene-global. General-purpose and phase
filters remain conservatively global; no finite-support declaration is implied
by a kernel size alone.

## Migration and Ownership

Prefer `apply(matrix, input_kind='matrix')` over `filter_matrix(matrix)`.
Prefer `apply(cube, input_kind='channels', matrix_type='T3')` over
`filter_channels(...)`. Both old names remain warning adapters shared by the
three joint filters, not separate implementations. They use canonical validation,
including support for three reciprocal channels; they do not enforce the old
four-channel-only rule. This alpha API does not promise legacy validation text.

`filters/base.py` owns dispatch, effective parameter validation, deprecated
adapters, and matrix result metadata. `decomposition/pol_matrix.py` owns channel
formation, descriptor normalization, and packing. Concrete filters own algorithm
kernels and capability declarations; Wishart has no concrete `execute()` or
private packing/dimension-inference implementation.