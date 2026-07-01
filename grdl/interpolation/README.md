# Interpolation Module

Bandwidth-preserving 1D interpolation kernels for signal resampling. All interpolators share a uniform callable signature `(x_old, y_old, x_new) -> y_new`, handle real and complex data, support mildly non-uniform input grids, and fill out-of-bounds points with zero.

`x_old` must be monotonic (ascending or descending -- a descending grid, such as azimuth k-space, is flipped internally). Every interpolator is a callable object: construct it once, then call it like a function. Where [numba](https://numba.pydata.org/) is installed, the FIR kernels and the Thiran IIR recursion JIT-compile and parallelize their inner loops (`numba.prange`) for a large speedup; without numba they fall back to vectorized numpy and produce identical results.

---

## Quick Start

```python
import numpy as np
from grdl.interpolation import LanczosInterpolator

# Create sample data
x_old = np.linspace(0, 10, 100)
y_old = np.sin(2 * np.pi * x_old / 5)

# Interpolate to finer grid
x_new = np.linspace(0, 10, 500)
interp = LanczosInterpolator(a=3)
y_new = interp(x_old, y_old, x_new)
```

---

## Available Interpolators

### FIR Kernel Interpolators

All FIR interpolators inherit from `KernelInterpolator` and accept non-uniform input grids.

| Class | Factory function | Method | Best for |
|-------|-----------------|--------|----------|
| `LanczosInterpolator` | `lanczos_interpolator(a=3)` | Lanczos-windowed sinc | General-purpose resampling |
| `KaiserSincInterpolator` | `windowed_sinc_interpolator(kernel_length=8, beta=5.0, oversample=1.0)` | Kaiser-windowed sinc | Adjustable sidelobe control |
| `LagrangeInterpolator` | `lagrange_interpolator(order=3)` | Maximally flat FIR delay | Low-order fractional delay |
| `FarrowInterpolator` | `farrow_interpolator(filter_order=3, poly_order=3)` | Farrow polynomial structure | Real-time variable delay |
| `PolyphaseInterpolator` | `polyphase_interpolator(kernel_length=8, num_phases=128, beta=5.0, prototype='kaiser', transition_width=0.5)` | Pre-computed filter bank | PFA k-space resampling (fastest FIR) |

Every factory simply constructs the matching class with the same keyword arguments; use whichever form you prefer. The class and factory are interchangeable.

### IIR Interpolator

| Class | Factory function | Method | Best for |
|-------|-----------------|--------|----------|
| `ThiranDelayFilter` | `thiran_delay(signal, delay, order=3)` | IIR allpass, maximally flat group delay | Uniform-grid fractional delay shift |

`ThiranDelayFilter` is the one exception to the `(x_old, y_old, x_new)` contract: it is a fixed fractional-delay filter, so the object is constructed with `(delay, order=3)` and then **called on a uniformly-sampled signal** -- `filt(signal) -> delayed_signal`. It does not resample to a new grid.

---

## Interpolator Details

### LanczosInterpolator

Lanczos-windowed sinc. Parameter `a` controls the number of lobes (kernel uses `2a` taps).

```python
from grdl.interpolation import LanczosInterpolator

interp = LanczosInterpolator(a=2)   # 4 taps, fast
interp = LanczosInterpolator(a=3)   # 6 taps, standard default
interp = LanczosInterpolator(a=5)   # 10 taps, high quality

y_new = interp(x_old, y_old, x_new)
```

### KaiserSincInterpolator

Kaiser-windowed sinc with adjustable sidelobe attenuation via `beta`. `kernel_length` is the number of taps (must be even); higher `beta` deepens sidelobe suppression at the cost of a wider mainlobe. `oversample > 1.0` narrows the passband to `Nyquist / oversample` to suppress aliasing when downsampling -- pair it with a larger `kernel_length` (16-64) so the narrower filter still has enough taps.

```python
from grdl.interpolation import KaiserSincInterpolator

interp = KaiserSincInterpolator(kernel_length=8, beta=5.0)
# Anti-aliased downsample: narrow the passband, widen the kernel
aa = KaiserSincInterpolator(kernel_length=64, beta=5.0, oversample=1.25)
y_new = interp(x_old, y_old, x_new)
```

### LagrangeInterpolator

Maximally flat FIR fractional-delay filter (Laakso et al. Eq. 42). `order` sets the polynomial order; the kernel uses `order + 1` taps. Low order is cheap and smooth but rolls off in the upper passband -- best for low-order fractional delays rather than aggressive resampling.

```python
from grdl.interpolation import LagrangeInterpolator

interp = LagrangeInterpolator(order=3)   # 4 taps, cubic (default)
y_new = interp(x_old, y_old, x_new)
```

### FarrowInterpolator

Farrow-structure polynomial approximation (Laakso et al. Eqs. 59-63). `filter_order` sets the kernel length (`filter_order + 1` taps) and `poly_order` sets the degree of the per-tap polynomial fit to the prototype Lagrange filters. The polynomial fit is exact when `poly_order >= filter_order`; lower `poly_order` trades accuracy for cheaper evaluation -- the classic structure for real-time variable delay where the delay changes per sample.

```python
from grdl.interpolation import FarrowInterpolator

interp = FarrowInterpolator(filter_order=3, poly_order=3)   # exact cubic
fast   = FarrowInterpolator(filter_order=5, poly_order=4)   # cheaper approximation
y_new = interp(x_old, y_old, x_new)
```

### PolyphaseInterpolator

Pre-computed filter bank with table lookup plus linear interpolation between phases. The **fastest FIR method** when many output points each need a different fractional delay -- the typical PFA k-space resampling pattern. The continuous delay axis is quantized into `num_phases` branches (finer = less phase-quantization error) and each branch has `kernel_length` taps (must be even).

Two prototype designs are available:

- `prototype='kaiser'` (default): Kaiser-windowed sinc branches, sidelobe control via `beta`. Simple and well-characterized.
- `prototype='remez'`: an optimal equiripple (Parks-McClellan / Remez) lowpass prototype of length `kernel_length * num_phases`, decomposed into the polyphase branches. `transition_width` sets the transition-band fraction (e.g. `0.5` -> passband 0.4 of Nyquist, `oversample = 1.25`). Better stopband rejection; requires `scipy`.

```python
from grdl.interpolation import PolyphaseInterpolator

# Kaiser prototype (default) -- the standard PFA choice
interp = PolyphaseInterpolator(kernel_length=8, num_phases=128, beta=5.0)

# Remez prototype for stronger stopband rejection
sharp = PolyphaseInterpolator(
    kernel_length=32, num_phases=256, prototype='remez', transition_width=0.4,
)
y_new = interp(x_old, y_old, x_new)
```

### ThiranDelayFilter

IIR allpass fractional delay for **uniformly-sampled** signals. Maximally flat group delay (Thiran 1971; Laakso et al. Eq. 86). It shifts the entire signal by a fractional number of samples rather than resampling to a new grid -- construct it with the delay, then call it on the signal. The requested `delay` must satisfy `delay >= order - 0.5` (the allpass needs that much group-delay headroom to stay stable). For repeated use at one delay, build the filter object once instead of calling the convenience function each time.

```python
from grdl.interpolation import ThiranDelayFilter, thiran_delay

signal = np.sin(2 * np.pi * np.arange(256) / 32)

# Convenience function (builds the filter internally)
delayed = thiran_delay(signal, delay=3.7, order=3)

# Reusable object -- coefficients computed once
filt = ThiranDelayFilter(delay=3.7, order=3)
delayed = filt(signal)
```

---

## Choosing an Interpolator

| Scenario | Recommended |
|----------|-------------|
| General image/signal resampling | `LanczosInterpolator(a=3)` |
| SAR PFA k-space resampling | `PolyphaseInterpolator` (fastest) |
| Need precise sidelobe control | `KaiserSincInterpolator` |
| Low-order fractional delay | `LagrangeInterpolator` |
| Real-time variable delay | `FarrowInterpolator` |
| Shift uniform signal by fractional samples | `ThiranDelayFilter` |

---

## Base Classes

- `Interpolator` -- ABC defining the callable `(x_old, y_old, x_new) -> y_new` interface. `x_old`/`y_old` are `(N,)`, `x_new` is `(M,)`, and the result is `(M,)`. `x_old` must be monotonic.
- `KernelInterpolator` -- Template for FIR kernel methods. Handles neighbor gathering via `np.searchsorted`, per-output-point local spacing, weight normalization (rows sum to 1 to preserve the DC level), and out-of-bounds zero fill. Subclasses implement only `_compute_weights(dx)`, where `dx` is the `(M, kernel_length)` matrix of normalized neighbor distances. `kernel_length` must be `>= 2`.

## Performance / numba

When numba is installed, `KaiserSincInterpolator`, `LagrangeInterpolator`, `FarrowInterpolator`, `PolyphaseInterpolator`, and `ThiranDelayFilter` JIT-compile and parallelize their inner loops (`numba.prange` over output points; the Thiran IIR recursion is JIT-compiled). It is an optional dependency -- without it the same algorithms run through vectorized numpy with identical numerical output, just slower. For the heaviest workloads (PFA k-space resampling), `PolyphaseInterpolator` is the recommended FIR method.
