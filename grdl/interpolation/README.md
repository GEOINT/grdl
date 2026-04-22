# Interpolation Module

Bandwidth-preserving 1D interpolation kernels for signal resampling. All interpolators share a uniform callable signature `(x_old, y_old, x_new) -> y_new`, handle real and complex data, support mildly non-uniform input grids, and fill out-of-bounds points with zero.

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
| `KaiserSincInterpolator` | `windowed_sinc_interpolator(...)` | Kaiser-windowed sinc | Adjustable sidelobe control |
| `LagrangeInterpolator` | `lagrange_interpolator(order=5)` | Maximally flat FIR delay | Low-order fractional delay |
| `FarrowInterpolator` | `farrow_interpolator(order=3)` | Farrow polynomial structure | Real-time variable delay |
| `PolyphaseInterpolator` | `polyphase_interpolator(...)` | Pre-computed filter bank | PFA k-space resampling (fastest FIR) |

### IIR Interpolator

| Class | Factory function | Method | Best for |
|-------|-----------------|--------|----------|
| `ThiranDelayFilter` | `thiran_delay(signal, delay, order)` | IIR allpass, maximally flat group delay | Uniform-grid fractional delay shift |

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

Kaiser-windowed sinc with adjustable sidelobe attenuation via `beta`.

```python
from grdl.interpolation import KaiserSincInterpolator

interp = KaiserSincInterpolator(kernel_length=8, beta=6.0)
y_new = interp(x_old, y_old, x_new)
```

### PolyphaseInterpolator

Pre-computed filter bank with table lookup. Fastest FIR method when many output points each need a different fractional delay (typical in PFA). Supports Kaiser or Remez prototype filters.

```python
from grdl.interpolation import PolyphaseInterpolator

interp = PolyphaseInterpolator(
    kernel_length=8, num_phases=64, prototype='kaiser', beta=6.0
)
y_new = interp(x_old, y_old, x_new)
```

### ThiranDelayFilter

IIR allpass fractional delay for uniformly-sampled signals. Shifts the entire signal by a fractional number of samples.

```python
from grdl.interpolation import thiran_delay

signal = np.sin(2 * np.pi * np.arange(256) / 32)
delayed = thiran_delay(signal, delay=3.7, order=4)
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

- `Interpolator` -- ABC defining the callable `(x_old, y_old, x_new) -> y_new` interface.
- `KernelInterpolator` -- Template for FIR kernel methods. Handles neighbor gathering via `np.searchsorted`, local spacing computation, weight normalization, and OOB zero fill. Subclasses implement `_compute_weights(dx)`.
